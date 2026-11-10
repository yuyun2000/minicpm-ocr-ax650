#!/usr/bin/env python3
"""组装 MiniCPM-OCR 的 axllm 运行时目录（AX650 板端可直接 `axllm serve .`）。

目录布局（最终模型包，≈2.5 GB）：
    config.json                                   axllm 模型配置（vlm_type=MiniCPMV46VL）
    qwen3_5_text_p128_l{0..23}_together.axmodel   24 层文本 LLM（INT8）
    qwen3_5_text_post.axmodel                     post（BF16，≈1017 MB）
    model.embed_tokens.weight.bfloat16.bin        bf16 embedding（≈508 MB）
    minicpm_ocr_tokenizer.txt                     axllm 本地分词器（magic 430 / HUGGINGFACE BPE）
    post_config.json                              采样配置
    minicpm_ocr_vision_448.axmodel                vision encoder（U16！见 docs/accuracy.md）
    bin/axllm                                     AX650 aarch64 运行时（取自 AXERA 官方包）

用法：
  python3 assemble_package.py \
      --work-dir /path/to/work/minicpm-ocr      # 含 llm_v2/（或 llm_final/）与 vision axmodel
      --dest /path/to/axllm_pkg \
      --tokenizer /path/to/minicpm_ocr_tokenizer.txt \
      --axllm-bin /path/to/axera/bin/axllm \
      --axllm-version /path/to/axera/bin/axllm.version.json \
      --ctx 2048

说明：
  - 层 axmodel 的文件名模板从实际文件名里解析（`*_p128_l%d_together.axmodel`），
    不硬编码 p 数，避免 prefill 桶改了名字对不上。
  - tokenizer.txt 由 ax-llm 仓库的 `tokenizer.axera/tests/convert_tokenizer.py` 生成
    （magic 430 / tokenizer_type 3 / HUGGINGFACE BPE）；本脚本只搬运，不生成。
  - 24 层的 layer_types = [linear, linear, linear, full] × 6，与 openbmb 原始
    text_config.layer_types 一致。
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

LAYER_TYPES = ["linear_attention"] * 3 + ["full_attention"]
LAYER_TYPES = (LAYER_TYPES * 6)[:24]          # 24 层：18 linear + 6 full
EMBED_NAME = "model.embed_tokens.weight.bfloat16.bin"
VISION_NAME = "minicpm_ocr_vision_448.axmodel"
POST_NAME = "qwen3_5_text_post.axmodel"


def find_layer_dir(work: Path, layers_hint: int = 24) -> Path:
    for name in ("llm_v2", "llm_final", "llm_par", "llm_out", "llm_model_dir"):
        d = work / name
        if d.is_dir() and len(list(d.glob("*_l*_together.axmodel"))) == layers_hint:
            return d
    raise SystemExit(f"[ERR] 在 {work} 下找不到含 {layers_hint} 层 axmodel 的目录"
                     f"（找过 llm_v2/llm_final/llm_par/llm_out/llm_model_dir）")


def find_vision(work: Path) -> Path | None:
    for cand in (work / VISION_NAME,
                 work / "vision_task2" / "compile" / "model.axmodel",
                 work / "vision_task" / "compile" / "model.axmodel"):
        if cand.is_file():
            return cand
    return None


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True, help="转换工作目录（含 LLM/vision 产物）")
    ap.add_argument("--dest", required=True, help="输出模型包目录")
    ap.add_argument("--tokenizer", required=True, help="minicpm_ocr_tokenizer.txt")
    ap.add_argument("--post-config", default=None, help="post_config.json（不给就写空对象）")
    ap.add_argument("--axllm-bin", default=None, help="AXERA 官方 axllm 可执行文件（AX650 aarch64）")
    ap.add_argument("--axllm-version", default=None, help="axllm.version.json（可选）")
    ap.add_argument("--ctx", type=int, default=2048, help="decode 上下文（与 llm_build2 --max_context 一致）")
    ap.add_argument("--prefill-len", type=int, default=1152,
                    help="llm_build2 --prefill_len（只用于 model_name 的 P 段；桶大小由编译产物决定）")
    ap.add_argument("--model-name", default=None,
                    help="默认 MiniCPM-OCR-1.0-AX650-C<ctx>-P<prefill_len>（与板端 /v1/models 实测一致）")
    args = ap.parse_args()

    work, dest = Path(args.work_dir), Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    # ---- 1) 24 层 ----
    layer_dir = find_layer_dir(work)
    layers = sorted(layer_dir.glob("*_l*_together.axmodel"))
    m = re.match(r"(.*_p)(\d+)(_l)", layers[0].name)
    if not m:
        raise SystemExit(f"[ERR] 无法解析层名: {layers[0].name}")
    p_tokens = m.group(2)
    template = f"{m.group(1)}{p_tokens}_l%d_together.axmodel"
    for f in layers:
        link_or_copy(f, dest / f.name)
    print(f"[layers] {len(layers)} 层 ← {layer_dir}（template={template}，桶步长 p{p_tokens}）")

    # ---- 2) post / embed ----
    post_src = layer_dir / POST_NAME
    if not post_src.is_file():
        raise SystemExit(f"[ERR] 缺 {post_src}")
    link_or_copy(post_src, dest / POST_NAME)
    embed_src = layer_dir / EMBED_NAME
    if not embed_src.is_file():
        embed_src = work / "embed" / EMBED_NAME
    if not embed_src.is_file():
        raise SystemExit(f"[ERR] 缺 {EMBED_NAME}（找过 {layer_dir} 与 {work / 'embed'}）")
    link_or_copy(embed_src, dest / EMBED_NAME)
    print(f"[post ] {POST_NAME} {post_src.stat().st_size / 1e6:.0f} MB")
    print(f"[embed] {EMBED_NAME} {embed_src.stat().st_size / 1e6:.0f} MB")

    # ---- 3) vision ----
    vision_src = find_vision(work)
    if vision_src is None:
        print("[warn] 没找到 vision axmodel —— 包只能跑文本，图像路径不可用", file=sys.stderr)
    else:
        shutil.copy2(vision_src, dest / VISION_NAME)   # 注意：vision 459 MB（INT8）/ U16 会更大
        print(f"[vision] {VISION_NAME} {vision_src.stat().st_size / 1e6:.0f} MB ← {vision_src}")

    # ---- 4) tokenizer / post_config ----
    tok = Path(args.tokenizer)
    if not tok.is_file():
        raise SystemExit(f"[ERR] 缺 tokenizer: {tok}")
    shutil.copy2(tok, dest / "minicpm_ocr_tokenizer.txt")
    print(f"[tok  ] minicpm_ocr_tokenizer.txt {tok.stat().st_size / 1e6:.1f} MB")
    pc_src = Path(args.post_config) if args.post_config else None
    if pc_src and pc_src.is_file():
        shutil.copy2(pc_src, dest / "post_config.json")
    else:
        (dest / "post_config.json").write_text("{}\n", encoding="utf-8")
        print("[post_config] 未提供，写空对象（serve 默认 greedy）", file=sys.stderr)

    # ---- 5) axllm 运行时 ----
    if args.axllm_bin:
        bin_dir = dest / "bin"
        bin_dir.mkdir(exist_ok=True)
        shutil.copy2(args.axllm_bin, bin_dir / "axllm")
        (bin_dir / "axllm").chmod(0o755)
        if args.axllm_version and Path(args.axllm_version).is_file():
            shutil.copy2(args.axllm_version, bin_dir / "axllm.version.json")
        print(f"[bin  ] bin/axllm ← {args.axllm_bin}")
    else:
        print("[warn] 未提供 --axllm-bin；模型包不含运行时（从 AXERA 官方 AX650 包取 bin/axllm）",
              file=sys.stderr)

    # ---- 6) config.json ----
    # 注意：层文件名里的 p128 是 prefill *步长*，真正的预填充桶上限是 prefill_len（1152）。
    # 板端实测 /v1/models 返回的 model_name 是 MiniCPM-OCR-1.0-AX650-C2048-P1152。
    model_name = args.model_name or \
        f"MiniCPM-OCR-1.0-AX650-C{args.ctx}-P{args.prefill_len}"
    cfg = {
        "system_prompt": "",
        "model_name": model_name,
        "url_tokenizer_model": "minicpm_ocr_tokenizer.txt",
        "tokenizer_type": "MiniCPMV46VL",
        "post_config_path": "post_config.json",
        "template_filename_axmodel": template,
        "axmodel_num": 24,
        "full_attention_interval": 4,
        "layer_types": LAYER_TYPES,
        "filename_post_axmodel": POST_NAME,
        "filename_tokens_embed": EMBED_NAME,
        "tokens_embed_num": 248100,
        "tokens_embed_size": 1024,
        "vlm_type": "MiniCPMV46VL",
        "filename_image_encoder_axmodel": VISION_NAME,
        "vision_cache_dir": "vision_cache",
        "vision_width": 448,
        "vision_height": 448,
        "vision_patch_size": 14,
        "b_use_mmap_load_embed": True,
        "b_use_mmap_load_layer": True,
        "devices": [0],
    }
    (dest / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[config] {dest / 'config.json'}（model_name={model_name}）")

    # ---- 7) 校验 ----
    problems = []
    for key in ("url_tokenizer_model", "post_config_path", "filename_post_axmodel",
                "filename_tokens_embed", "filename_image_encoder_axmodel"):
        if not (dest / cfg[key]).is_file():
            problems.append(f"缺 {key}: {cfg[key]}")
    for i in range(cfg["axmodel_num"]):
        if not (dest / (template % i)).is_file():
            problems.append(f"缺层 {i}: {template % i}")
    axm_mb = sum(f.stat().st_size for f in dest.glob("*.axmodel")) / 1e6
    embed_mb = (dest / EMBED_NAME).stat().st_size / 1e6
    summary = {
        "dest": str(dest),
        "bucket_step": int(p_tokens),
        "prefill_len": args.prefill_len,
        "ctx": args.ctx,
        "layers": len(layers),
        "vision": vision_src is not None,
        "total_axmodel_mb": round(axm_mb, 1),
        "embed_mb": round(embed_mb, 1),
        "total_pkg_gb": round((axm_mb + embed_mb) / 1000, 2),
        "problems": problems,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if problems:
        print("[ERR] 包不完整，上面 problems 逐条修", file=sys.stderr)
        return 1
    print("[ok] 模型包就绪；下一步：scripts/deploy_to_board.sh 推板启动")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
