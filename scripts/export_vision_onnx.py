#!/usr/bin/env python3
"""MiniCPM-OCR vision encoder → 部署用静态 ONNX（塔 + 2×2 merge + merger）。

产物：
    vision_448_merged.onnx
      input : pixel_values  float32 [1, 3, 14, 14336]   # 448×448 → 32×32=1024 patch，NaViT 打包
      output: image_embeds  float32 [1, 256, 1024]      # 1024 token → 2×2 空间合并 → 256 token

用法：
    python3 export_vision_onnx.py --model-dir /path/to/origin --out-dir ./out \
        [--image /path/to/page.jpeg]

    --image 只用于打印「官方 processor 对 448×448 究竟产生多少 image token」的核对；
    不传也能导出（导出只看形状，不看像素值）。

和 `vision_448x448.onnx`（塔单独，输出 [1,1024,1152]）的区别：
    合并图把塔 + 2×2 merge + merger 打进一张图，输出直接是喂给 LLM 的 image embeds，
    板端 axllm 的 vision encoder 只接受这一张图（单位置输出）。

已知坑（全部踩过，见 docs/conversion.md §3）：
  1. `target_sizes` 会被常量折叠 —— 图里只剩 pixel_values 一个输入，换分辨率必须重导。
  2. merger 里 int32 target_sizes 会与常量 int64 在 Reshape/Concat 处冲突 → 用 int64。
  3. `unsqueeze(0)` 会在 ONNX 里留下符号维 → 改成静态 `reshape(1, 256, 1024)`。
  4. config 里 dtype=bfloat16，`from_pretrained` 默认按 bf16 加载 → 显式 dtype=float32。

依赖：torch / transformers(>=5.22, 原生 minicpm_ocr) / onnx / onnxruntime / numpy / Pillow。
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

SIDE = 448
PATCH = 14
GRID = SIDE // PATCH          # 32
NUM_PATCHES = GRID * GRID      # 1024
PACK_W = NUM_PATCHES * PATCH   # 14336


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def build_pixel_values(processor, image: Path | None, side: int = SIDE) -> torch.Tensor:
    """PIL 图 → NaViT 打包的 pixel_values；(x/255-0.5)/0.5 归一化 + patch 重排。

    没有图时用「全白 448×448」——导出不依赖像素值，只是为了拿到正确形状的输入。
    """
    ip = processor.image_processor
    if image is None:
        arr = np.full((side, side, 3), 255, dtype=np.uint8)
    else:
        arr = np.asarray(Image.open(image).convert("RGB").resize((side, side), Image.BICUBIC), dtype=np.uint8)
    x = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    x = (x - 0.5) / 0.5
    return ip.reshape_by_patch(x, ip.patch_size).unsqueeze(0)


class VisionMerged(torch.nn.Module):
    """塔 + (2×2 merge) + merger；target_sizes 固定为常量，图里只暴露 pixel_values。"""

    def __init__(self, vision, merger, target_sizes: torch.Tensor, use_vit_merger: bool = False):
        super().__init__()
        self.vision = vision
        self.merger = merger
        self.register_buffer("target_sizes", target_sizes)
        self.use_vit_merger = use_vit_merger
        g = int(target_sizes[0, 0])
        merge_h, merge_w = merger.merge_kernel_size
        self.out_tokens = (g // merge_h) ** int(merger.merger_times) * (g // merge_w) ** int(merger.merger_times)
        self.llm_dim = merger.mlp[-1].linear_2.out_features

    def forward(self, pixel_values):
        out = self.vision(pixel_values, target_sizes=self.target_sizes, use_vit_merger=self.use_vit_merger)
        embeds = self.merger(out.last_hidden_state, self.target_sizes)[0]
        # 静态 reshape 替掉 unsqueeze(0)：NPU 编译器不接受符号维
        return embeds.reshape(1, self.out_tokens, self.llm_dim)


def dump_graph(path: str) -> dict:
    m = onnx.load(path, load_external_data=False)
    ops = collections.Counter(n.op_type for n in m.graph.node)
    init = {i.name for i in m.graph.initializer}
    return {
        "inputs": [
            {"name": i.name, "dtype": onnx.TensorProto.DataType.Name(i.type.tensor_type.elem_type),
             "shape": [d.dim_value if d.HasField("dim_value") else str(d.dim_param) for d in i.type.tensor_type.shape.dim]}
            for i in m.graph.input if i.name not in init
        ],
        "outputs": [
            {"name": o.name, "dtype": onnx.TensorProto.DataType.Name(o.type.tensor_type.elem_type),
             "shape": [d.dim_value if d.HasField("dim_value") else str(d.dim_param) for d in o.type.tensor_type.shape.dim]}
            for o in m.graph.output
        ],
        "nodes": len(m.graph.node),
        "op_types": ops.most_common(12),
        "mb": round(os.path.getsize(path) / 1024**2, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="HF 原始权重目录（openbmb/MiniCPM-OCR-1.0）")
    ap.add_argument("--out-dir", default=".", help="ONNX 与 json 输出目录")
    ap.add_argument("--image", default=None, help="可选：用于核对 448×448 的 image token 数")
    ap.add_argument("--side", type=int, default=SIDE, help="固定边长（默认 448；改这个必须重导）")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    torch.set_num_threads(min(12, os.cpu_count() or 4))

    print(f"[load] {args.model_dir} (fp32)", flush=True)
    processor = AutoProcessor.from_pretrained(args.model_dir)
    model = AutoModelForImageTextToText.from_pretrained(args.model_dir, dtype=torch.float32).eval()
    vision = model.model.vision_tower.eval()
    merger = model.model.merger.eval()
    cfg = model.config
    print(f"[cfg] downsample_mode={cfg.downsample_mode} merger_times={cfg.merger_times} "
          f"merge_kernel={cfg.merge_kernel_size} insert_layer_id={cfg.insert_layer_id}", flush=True)

    grid = args.side // PATCH
    # int64：merger 内部 view(merged_h, 2, merged_w, 2, C)，int32 会与常量 int64 拼 shape 触发 ORT 类型冲突
    target_sizes = torch.tensor([[grid, grid]], dtype=torch.int64)
    pixel_values = build_pixel_values(processor, Path(args.image) if args.image else None, args.side)
    print(f"[input] side={args.side} grid={grid}x{grid} pixel_values={tuple(pixel_values.shape)}",
          flush=True)

    # --- 核对：官方 processor 对 448×448 产生多少 image token（需要图） ---
    token_check = {}
    if args.image:
        img = Image.open(args.image).convert("RGB").resize((args.side, args.side), Image.BICUBIC)
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Table Recognition:"}]}]
        prompt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                               enable_thinking=False)
        pin = processor(images=[img], text=[prompt], return_tensors="pt")
        token_check = {
            "target_sizes": pin["target_sizes"].tolist(),
            "patches": int(pin["target_sizes"].prod(-1).sum()),
            "image_tokens": int((pin["input_ids"] == model.config.image_token_id).sum()),
            "seq_len": int(pin["input_ids"].shape[-1]),
            "pixel_values": list(pin["pixel_values"].shape),
        }
        print(f"[token-check] {token_check}", flush=True)

    # --- torch 参考 ---
    wrapped = VisionMerged(vision, merger, target_sizes, use_vit_merger=False).eval()
    t0 = time.perf_counter()
    ref = wrapped(pixel_values)
    print(f"[torch] merged out={tuple(ref.shape)} ({time.perf_counter() - t0:.2f}s)", flush=True)

    # --- 导出 ---
    merged_path = out_dir / f"vision_{args.side}_merged.onnx"
    t0 = time.perf_counter()
    torch.onnx.export(
        wrapped,
        (pixel_values,),
        str(merged_path),
        input_names=["pixel_values"],
        output_names=["image_embeds"],
        opset_version=18,
        dynamo=False,
        do_constant_folding=True,
    )
    print(f"[export] {merged_path} {merged_path.stat().st_size / 1024**2:.1f}MB "
          f"in {time.perf_counter() - t0:.1f}s", flush=True)

    # --- ORT 对分 ---
    sess = ort.InferenceSession(str(merged_path), providers=["CPUExecutionProvider"])
    print(f"[onnx] inputs={[i.name for i in sess.get_inputs()]} "
          f"outputs={[o.name for o in sess.get_outputs()]}", flush=True)
    t0 = time.perf_counter()
    ont = sess.run(None, {"pixel_values": pixel_values.numpy()})[0]
    cos = cosine(ref.numpy(), ont)
    print(f"[merged] out={ont.shape} cosine={cos:.10f} "
          f"max_abs={float(np.abs(ref.numpy() - ont).max()):.2e} ort={time.perf_counter() - t0:.2f}s",
          flush=True)

    result = {
        "side": args.side, "grid": f"{grid}x{grid}", "patch_size": PATCH,
        "token_check": token_check,
        "onnx": merged_path.name,
        "output_shape": list(ont.shape),
        "cosine": cos,
        "max_abs_diff": float(np.abs(ref.numpy() - ont).max()),
        "graph": dump_graph(str(merged_path)),
        "notes": {
            "target_sizes_folded": "torch.onnx.export(dynamo=False) 把 target_sizes 当常量"
                                   "（position_ids / view 需要 Python int），图里只有 pixel_values；"
                                   "换分辨率必须重新导出。",
            "merger_input_order": "merger 输入是塔输出 (1,1024,1152)，token 顺序 h-major/w-minor；"
                                  "内部 view(h/2,2,w/2,2,C).permute(0,2,1,3,4) 完成 2×2 合并。",
        },
    }
    (out_dir / "vision-merged.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
    print(f"[ok] cosine={cos:.10f}（导出验收线 0.999）；json → {out_dir / 'vision-merged.json'}")
    return 0 if cos >= 0.999 else 1


if __name__ == "__main__":
    raise SystemExit(main())
