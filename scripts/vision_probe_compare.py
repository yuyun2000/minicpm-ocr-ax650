#!/usr/bin/env python3
"""对比板端 ax_run_model 输出与 torch/ONNX 参考的 cosine（视觉精度定位用）。

用法：
  .venv/bin/python demo/minicpm-ocr/vision_probe_compare.py \
      --ref /tmp/probe2_ref.npy --npu /tmp/probe2_npu.bin
"""
import argparse

import numpy as np


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="torch/ONNX 参考 .npy")
    ap.add_argument("--npu", required=True, help="板端 ax_run_model 输出 .bin（float32）")
    args = ap.parse_args()

    ref = np.load(args.ref).astype(np.float32)
    npu = np.fromfile(args.npu, dtype=np.float32)
    print(f"ref shape={ref.shape} npu elems={npu.size} (expect {ref.size})")
    n = min(ref.size, npu.size)
    print(f"cosine = {cosine(npu[:n], ref.ravel()[:n]):.6f}")
    print(f"npu  mean={npu.mean():.4f} std={npu.std():.4f} absmax={np.abs(npu).max():.2f}")
    print(f"ref  mean={ref.mean():.4f} std={ref.std():.4f} absmax={np.abs(ref).max():.2f}")


if __name__ == "__main__":
    main()
