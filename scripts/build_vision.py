#!/usr/bin/env python3
"""用 pulsar2 build 把 MiniCPM-OCR 的 vision encoder 编成 AX650 axmodel。

⚠ 核心结论（2026-11-10 实测，详见 docs/accuracy.md）：**vision 必须用 U16 激活**。
   INT8/U8 全量化会把 27 层视觉塔打崩：

     2 层探针 INT8            cosine 0.8606
     27 层生产模型 INT8       cosine **0.311**（板端输出与 ONNX 参考完全不相关）
     官方配置 U16（2 层探针） cosine **0.997895** ✅

   本脚本的量化配置逐字段照抄 AXERA 官方转换仓库
   https://github.com/AXERA-TECH/MiniCPM-V-4.6.axera 的
   `model_convert/pulsar2_configs/config_vision_448_npu3.json`：

     calibration_method          MinMax
     conv_bias_data_type         FP32
     enable_smooth_quant         true
     disable_auto_refine_scale   true
     precision_analysis          true / EndToEnd
     layer_configs               DEFAULT → DEFAULT  data_type = U16
     input_processors            DEFAULT src_dtype = FP32

   校准集用 **64 张真实文档切片**（官方 prepare_calibration.py 同款），不要用随机/纯色图。

输入：
  1) scripts/export_vision_onnx.py 产出的 vision_448_merged.onnx
     （pixel_values [1,3,14,14336] → image_embeds [1,256,1024]）
  2) 校准数据 tar.gz（每个样本一个 .npy，形状 (1,3,14,14336)，float32，(x/255-0.5)/0.5）

用法：
  python3 build_vision.py --onnx ./vision_448_merged.onnx --calib ./calib/pixel_values.tar.gz \
      --out-dir ./vision_task --name minicpm_ocr_vision_448
  # 只生成 pulsar2_config.json 不编译（先肉眼核对）：
  python3 build_vision.py ... --dry-run

产物：$OUT_DIR/model.axmodel + compile.log + pulsar2_config.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PACK_W = 14336  # 32×32 patch ×14


def build_config(onnx: Path, out_dir: Path, calib: Path, calib_size: int,
                 input_name: str, chip: str) -> dict:
    """照 AXERA 官方 config_vision_448_npu3.json 落配置。"""
    return {
        "input": str(onnx),
        "output_dir": str(out_dir),
        "output_name": "model.axmodel",
        "work_dir": str(out_dir / "work"),
        "model_type": "ONNX",
        "target_hardware": chip,
        "npu_mode": "NPU1",
        "input_shapes": f"{input_name}:1x3x14x{PACK_W}",
        "onnx_opt": {
            "disable_onnx_optimization": False,
            "enable_onnxsim": False,
            "model_check": True,
        },
        "quant": {
            "input_configs": [{
                "tensor_name": input_name,
                "calibration_dataset": str(calib),
                "calibration_format": "Numpy",
                "calibration_size": calib_size,
                "calibration_mean": [],
                "calibration_std": [],
            }],
            # ↓↓↓ 这六行就是根因所在：官方是 U16 激活，不是 INT8/U8 ↓↓↓
            "calibration_method": "MinMax",
            "conv_bias_data_type": "FP32",
            "enable_smooth_quant": True,
            "disable_auto_refine_scale": True,
            "precision_analysis": True,
            "precision_analysis_method": "EndToEnd",
            "layer_configs": [{
                "start_tensor_names": ["DEFAULT"],
                "end_tensor_names": ["DEFAULT"],
                "data_type": "U16",
            }],
            "highest_mix_precision": False,
        },
        "input_processors": [{
            "tensor_name": "DEFAULT",
            "tensor_format": "RGB",
            "tensor_layout": "NCHW",
            "src_format": "RGB",
            "src_layout": "NCHW",
            "src_dtype": "FP32",
            "mean": [],
            "std": [],
        }],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, help="vision_448_merged.onnx")
    ap.add_argument("--calib", required=True, help="校准数据 tar.gz（每个样本一个 .npy）")
    ap.add_argument("--out-dir", default="./vision_task")
    ap.add_argument("--name", default="minicpm_ocr_vision_448", help="模型名（只写进日志/报告）")
    ap.add_argument("--calib-size", type=int, default=64, help="官方用 64 张真实切片")
    ap.add_argument("--chip", default="AX650")
    ap.add_argument("--input-name", default="pixel_values")
    ap.add_argument("--dry-run", action="store_true", help="只写 pulsar2_config.json，不编译")
    args = ap.parse_args()

    onnx = Path(args.onnx).resolve()
    calib = Path(args.calib).resolve()
    out_dir = Path(args.out_dir).resolve()
    if not onnx.is_file():
        raise SystemExit(f"找不到 ONNX: {onnx}")
    if not calib.is_file():
        raise SystemExit(f"找不到校准数据: {calib}")
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_config(onnx, out_dir, calib, args.calib_size, args.input_name, args.chip)
    cfg_path = out_dir / "pulsar2_config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"[cfg] {cfg_path}（U16 + smooth_quant + FP32 conv_bias + disable_auto_refine_scale）")

    if args.dry_run:
        print(json.dumps(cfg, indent=2))
        return 0

    pulsar2 = os.environ.get("PULSAR2", "pulsar2")
    if not (os.path.sep in pulsar2 and Path(pulsar2).is_file()):
        from shutil import which
        if which(pulsar2) is None:
            raise SystemExit(f"[ERR] 找不到 pulsar2（{pulsar2}）；用 PULSAR2=/abs/path/pulsar2 指定")

    cmd = [pulsar2, "build", "--config", str(cfg_path)]
    print(f"[build] {' '.join(cmd)}")
    log = out_dir / "compile.log"
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        f.write(proc.stdout)
    for line in proc.stdout.splitlines()[-20:]:
        print("   ", line)
    if proc.returncode != 0:
        raise SystemExit(f"[ERR] pulsar2 build 失败（{proc.returncode}），完整日志: {log}")

    ax = out_dir / "model.axmodel"
    if not ax.is_file():
        raise SystemExit(f"[ERR] 没生成 {ax}；看 {log}")
    print(f"[ok] {ax}  {ax.stat().st_size / 1e6:.1f} MB  （{time.time() - t0:.0f}s）")
    print(f"[ok] 产物名: {args.name}（部署时重命名为 minicpm_ocr_vision_448.axmodel 放进模型包）")
    # 板端验收：把 model.axmodel 拉到板上用 ax_run_model 跑一张 448 图，
    # 与 ONNX/torch 参考算 cosine，应 ≥0.99（U16 探针实测 0.9979）。
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ============================================================
# 重要：全量 27 层 + merger 的 U16 图必须用 Pulsar2 **6.0** 编译
# ------------------------------------------------------------
# 已实测：Pulsar2 7.0 / 8.0 编译该 U16 图会在 `build op serially` 阶段原生崩溃
# （tiling 后打印 new_ddr_tensor，进程被杀，exit 255，无 Python 异常）。
# 同配置 2/8 层探针可编过，16 层即崩 —— 规模相关。
#
# 官方 MiniCPM-V-4.6 的 vision axmodel 也是 Pulsar2 6.0 编的（包内 header 可见）。
# 6.0 用法（CLI 风格配置）：
#
#   P60=~/.cache/magnetar/pulsar2/6.0
#   $P60/bin/pulsar2 build \
#     --input vision_sim.onnx \
#     --config cfg60.json \
#     --output_dir ./out --output_name minicpm_ocr_vision_448.axmodel \
#     --target_hardware AX650 --model_type ONNX \
#     --input_shapes "pixel_values:1x3x14x14336"
#
# cfg60.json（关键：layer_configs DEFAULT→DEFAULT = U16，conv bias FP32）：
# {
#   "model_type": "ONNX", "npu_mode": "NPU3",
#   "quant": {
#     "input_configs": [{"tensor_name":"DEFAULT",
#        "calibration_dataset":"<calib.tar.gz>","calibration_size":8,"calibration_format":"Numpy"}],
#     "calibration_method": "MinMax",
#     "conv_bias_data_type": "FP32",
#     "layer_configs": [{"start_tensor_names":["DEFAULT"],"end_tensor_names":["DEFAULT"],"data_type":"U16"}]
#   },
#   "input_processors": [{"tensor_name":"DEFAULT","src_dtype":"FP32"}],
#   "compiler": {"npu_perf": false, "check": 0}
# }
#
# 产物约 511 MB，板端 448×448 单块 615 ms，对 ONNX cosine 0.9953。
# ============================================================
