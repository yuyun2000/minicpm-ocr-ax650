#!/usr/bin/env python3
"""导出 MiniCPM-OCR 视觉塔的前 N 层为静态 ONNX 探针（单输入，便于在 NPU 上逐层定位精度损失）。

用法：
  .venv/bin/python demo/minicpm-ocr/vision_probe_export.py --layers 2 --out /tmp/probe2.onnx
  # 会同时生成 /tmp/probe2_ref.npy（torch 参考输出），供板端对拍
"""
import argparse

import numpy as np
import torch
from transformers import AutoModelForImageTextToText

MODEL_DIR = "demo/minicpm-ocr/origin"
INPUT_NPY = "/tmp/vision_in.npy"  # (1,3,14,14336) NaViT 打包 + [-1,1] 归一化


class Probe(torch.nn.Module):
    def __init__(self, vt, n: int):
        super().__init__()
        self.vt = vt
        self.n = n
        self.register_buffer("ts", torch.tensor([[32, 32]], dtype=torch.int64))

    def forward(self, pixel_values):
        import torch.nn.functional as F

        ts = self.ts
        hs = self.vt.embeddings(pixel_values, target_sizes=ts)
        cu = F.pad(torch.cumsum(ts[:, 0] * ts[:, 1], dim=0, dtype=torch.int32), (1, 0))
        max_seq = int(cu[-1].item())
        for layer in self.vt.encoder.layers[: self.n]:
            hs = layer(hs, attention_mask=None, cu_seqlens=cu, max_seqlen=max_seq)
        return hs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ref", default=None)
    args = ap.parse_args()

    out = args.out or f"/tmp/probe{args.layers}.onnx"
    ref_path = args.ref or f"/tmp/probe{args.layers}_ref.npy"

    model = AutoModelForImageTextToText.from_pretrained(MODEL_DIR, dtype=torch.float32).eval()
    probe = Probe(model.model.vision_tower, args.layers).eval()
    x = torch.from_numpy(np.load(INPUT_NPY)).float()
    with torch.no_grad():
        ref = probe(x)
    np.save(ref_path, ref.numpy().astype(np.float32))
    torch.onnx.export(probe, (x,), out, input_names=["pixel_values"], output_names=["hidden"],
                      opset_version=17, dynamo=False)
    print(f"[probe] layers={args.layers} out={out} ref={ref_path} shape={tuple(ref.shape)}")


if __name__ == "__main__":
    main()
