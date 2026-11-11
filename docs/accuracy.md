# 精度验证：vision 编码器、切片、tokenizer、端到端

> 本文回答一个问题：**它跑出来的结果到底是不是对的**。
> 所有数字来自 2026-11-10 ~ 2026-11-11 的板端与主机实测，均为最终版（U16 vision 已上线）。
> 板卡：AX8850（AX650N 板型）（Ubuntu 22.04 aarch64，kernel 5.15.73，运行时 `/soc/lib/libax_engine.so` V1.45.0_P39，2024-09）

---

## 0. 结论速览

| 验证项 | 结果 | 判定 |
|---|---|---|
| vision ONNX 导出（塔+merge+merger）vs torch | cosine **0.9999999992**，max_abs 2.46e-03 | ✅ |
| vision axmodel（**INT8**）vs ONNX 参考 | cosine **0.311** | ❌ 不可用 |
| vision axmodel（**官方 U16 配置**，2 层探针）vs torch | cosine **0.997895** | ✅ |
| vision axmodel（U16，27 层全量 + merger，Pulsar2 6.0）板端对拍 | cosine **0.995322**（511 MB / 615 ms） | ✅ |
| tokenizer（C++ 实现 vs HF） | 文本/图像/视频 3 类用例 **5/5 PASS** | ✅ |
| 切片数量 / 每片尺寸 / patch 网格 / image token | 与官方 processor **逐片完全一致**（4 张测试图 × 4 种模式） | ✅ |
| 几何函数随机对拍（`ensure_divide` 等） | 2000/6000/1440/3000/5 组，**0 不一致** | ✅ |
| 板端文本语义 | "1+1" → "2"；"一句话解释 OCR" → 通顺中文 | ✅ |
| 板端整块图像 OCR | 448×448 表格块 → 正确 OTSL（6 行 × 8 列，内容与图中表格一致） | ✅（人工核对） |
| 端到端（单块表格，板端 vs HF CPU 基线） | 字符相似度 **1.0**、数字 **12/12** 一致 | ✅ |

一句话：**视觉编码器必须用 U16 激活**，INT8 会让 27 层视觉塔的 cosine 从 0.99 掉到 0.31，
端到端表现为「给纯白图也能编出一张表格」。切片与 tokenizer 侧已对齐官方，不是误差来源。

---

## 1. 为什么 vision 精度要单独验

板端 axllm 是一条龙服务，端到端输出「看起来差不多」不等于模型真在看图。INT8 版的实际现象（实测）：

| 输入 | 输出 |
|---|---|
| 纯白图 | 也能吐出一张像样的 6×8 表格 |
| 同一张图多次请求 | 每次输出不同（temperature=0 下也不稳定） |

这种都是「模型看不到图、靠语言先验在编」的典型症状。所以要把 vision 编码器单独隔离出来，
**不经过 axllm**，直接 `ax_run_model` 对拍。

---

## 2. 隔离方法（可复现）

```
Mac/转换机：构造输入 bin → 板端 ax_run_model → 输出 bin 拉回 → 与 ONNX/torch 参考算 cosine
```

```bash
# 1) 转换机：构造与运行时完全一致的输入（NaViT 打包 + (x/255-0.5)/0.5 归一化，448×448）
python3 - <<'PY'
import numpy as np
from PIL import Image
im = Image.open("ocr_448.png").convert("RGB").resize((448, 448), Image.BICUBIC)
a = (np.asarray(im, dtype=np.float32) / 255.0 - 0.5) / 0.5
pv = a.transpose(2, 0, 1).reshape(3, 32, 14, 32, 14).transpose(0, 2, 1, 3, 4).reshape(1, 3, 14, 14336)
pv.tofile("vision_in.bin")
PY

# 2) ONNX 参考（同时存 vision_ref.npy）

# 3) 板端跑 axmodel（输入 bin 名必须等于 ONNX 输入张量名 pixel_values）
scp vision_in.bin root@<board>:/root/vision_in.bin
ssh root@<board> 'cd /root && rm -rf vtest && mkdir -p vtest/inputs/run0 vtest/out
  cp vision_in.bin vtest/inputs/run0/pixel_values.bin
  echo run0 > vtest/list.txt
  /opt/bin/ax_run_model -m /root/minicpm-ocr-v2/minicpm_ocr_vision_448.axmodel \
      -i vtest/inputs -o vtest/out -l vtest/list.txt'
scp root@<board>:/root/vtest/out/run0/image_embeds.bin vision_npu.bin

# 4) 对拍
python3 -c "import numpy as np; a=np.fromfile('vision_npu.bin',np.float32).astype(np.float64); \
b=np.load('vision_ref.npy').astype(np.float64).ravel(); print(a@b/(np.linalg.norm(a)*np.linalg.norm(b)))"
```

`ax_run_model` 的用法坑：`-i` 是「每次推理一个子目录」的父目录，`-l` 里放**子目录名**（不是 bin 文件名），
输出自动按输出张量名落盘（`image_embeds.bin`）。

---

## 3. vision 精度表（板端 ax_run_model 输出 vs torch 参考）

| 模型 | 量化/编译配置 | cosine | 板端耗时 |
|---|---|---:|---:|
| 2 层探针 | INT8 全量化（Pulsar2 8.0） | 0.8606 | 41 ms |
| 2 层探针 | INT8 全量化（Pulsar2 7.0） | 0.8606 | 41 ms |
| 2 层探针 | Conv/MatMul/Softmax FP32，其余 INT8 | 0.9274 | 917 ms |
| 2 层探针 | 关键算子 + LayerNorm 用 U16 | 0.9275 | 62 ms |
| 2 层探针 | 除 LayerNorm 外全部 FP32 | 0.9681 | 921 ms |
| **2 层探针** | **官方配置：U16 + smooth_quant + FP32 conv_bias + disable_auto_refine_scale** | **0.997895** ✅ | 87.5 ms |
| 27 层生产模型 | INT8，2 张校准 + MinMax | **0.311** ❌ | 520 ms |
| **27 层生产模型** | **U16，16 张真实切片校准，Pulsar2 6.0** | **0.995322** ✅ | **615 ms**（511 MB） |

误差是**逐层累积**的：2 层就 0.86，27 层连乘掉到 0.31。

---

## 4. 已排除的原因（每条都做过对照实验）

| 假设 | 验证 | 结论 |
|---|---|---|
| 输入归一化/通道顺序错 | 用 4 种输入（[-1,1]、[0,1]、[0,255]、反相）分别跑 | cosine 0.298 / 0.308 / 0.279 / 0.311，**都坏** → 不是喂法 |
| 工具链版本（7.0 vs 8.0） | 同一 ONNX 分别用 Pulsar2 7.0 / 8.0 编译，板端对拍 | 两者 cosine **完全相同 0.8606** → 不是工具链 |
| 板端运行时太老 | 同一块板跑 **AXERA 官方 MiniCPM-V-4.6 包**（官方 vision 是 Pulsar2 6.0 编的） | 官方包看图正确（"图片上有一个红色矩形…下方有白色文字"，2.87 s） → 运行时与 axllm profile 正常 |
| ONNX 导出错 | torch vs ONNX；2 层探针 ORT vs torch | 合并图 0.9999999992；探针 **1.000000** → 导出正确 |
| 某一层/merger 单独坏 | 2 层探针 | 仅 2 层就 0.8606 → 逐层累积误差，不是单点 |

关键对照：**官方 V4.6 包在同一块板上看图正常**，所以问题锁定在我们编的 vision axmodel 的量化配置，
与板端老运行时无关。

---

## 5. 根因与修复：官方 vision 是 U16，不是 INT8

AXERA 官方转换仓库 **https://github.com/AXERA-TECH/MiniCPM-V-4.6.axera** 的 `model_convert/` 里有完整
vision 转换资产，其中 `pulsar2_configs/config_vision_448_npu3.json` 写明：

```json
{
  "quant": {
    "calibration_method": "MinMax",
    "conv_bias_data_type": "FP32",
    "enable_smooth_quant": true,
    "disable_auto_refine_scale": true,
    "precision_analysis": true,
    "precision_analysis_method": "EndToEnd",
    "layer_configs": [{"start_tensor_names": ["DEFAULT"], "end_tensor_names": ["DEFAULT"], "data_type": "U16"}]
  },
  "input_processors": [{"tensor_name": "DEFAULT", "src_dtype": "FP32"}]
}
```

→ **官方 vision 是全图 U16 激活（16-bit），不是 INT8/U8**；同时开 smooth quant、conv bias 用 FP32、
禁用自动 refine scale。用这套配置重建 2 层探针：0.8606 → **0.997895**（耗时 41 ms → 87.5 ms，约 2×）。

校准数据同样是关键：官方用 **64 张真实图像切片**（`prepare_calibration.py` + `download_dataset.sh`），
我们首版 INT8 只用了 2 张 + MinMax。`scripts/build_vision.py` 已把 U16 配置内置，默认 `--calib-size 64`。

> 不能直接拿官方 V4.6 的 vision axmodel 顶替：结构与官方同源（hidden 1152 / 27 层 / 16 头 / patch 14），
> 但
> ① 453 个视觉张量逐个 SHA 对比**全部不同**（不是同一权重）；
> ② 官方是 **16x 压缩**（448 块 → 64 token），我们是 **4x**（→ 252/256 token），token 数不匹配会让 LLM 输入错位。

---

## 6. 切片对齐（应用侧整页模式的前提）

板端 vision 只吃固定 448×448 单块，整页必须应用侧切片。所以切片方案必须与官方 processor 逐片一致，
否则 image token 数对不上，prompt 里的占位符数量就错了。

官方实现：`transformers` 5.22.2 的 `MiniCPMOcrImageProcessor`（配置 `scale_resolution=448`、
`max_slice_nums=9`、`patch_size=14`、`downsample_mode=4x`、`small_image_upsize=always`）。
本仓库 `service/slicing.py` 用纯 numpy + PIL 逐行复刻（板端没有 torch/cv2）。

主对比（`small_image_upsize="always"`，官方默认）：

| 测试图 | 源尺寸 | 切片数 | grid | image token |
|---|---|---:|---|---:|
| 真实文档页 | 2000×3083 | 9 | [4, 2] | **2276** |
| 合成横图 | 1280×800 | 7 | [2, 3] | 1700 |
| 合成方图 | 448×448 | 1 | — | 256 |
| 合成小图 | 160×120 | 1 | — | 252 |

- 9 片 = 全局缩略图 260 token（364×560，patch 40×26）+ 8 片 × 252 token（504×392，patch 28×36）= **2276**；
  与独立路径记的整页基线 `tiles=9 patches=9104 image_tokens=2276 seq=2316` 完全吻合。
- 每片 resize 尺寸与官方**差值恒为 0 px**（不是 ≤2px 容差）；patch 网格、token 数逐片一致。
- `small_image_upsize` 的 `always/never/adaptive` 三模式 + `slice_mode=False` 全部一致
  （`never` 模式小图 24 token、`adaptive` 横图 1520 token 也都对上）。
- 几何函数随机对拍零不一致：`ensure_divide` 2000 组、`find_best_resize` 6000 组、
  `get_refine_size` 1440 组、`get_sliced_grid` 3000 组、`reshape_by_patch` 5 组。
- **唯一无法逐位对齐的是 resize 像素值**：官方走 torchvision v2 bicubic（a=-0.75），我们走 PIL（a=-0.5）。
  下采样路径差 ≤1 LSB，近 1:1 缩放最大 12 LSB（归一化后 `pixel_values` 最大 |Δ| 0.0078 / 0.0157）。
  对视觉编码器输入无实际影响；需要逐位复刻时按 PyTorch 公式替换 `_resize`。
- `adaptive_upsize_resolution`（官方用 cv2 Otsu + 8 连通域）用纯 numpy 复刻，返回值 9/9 一致，慢 2.4–5×；
  官方默认 `always` 不触发这条路径。

切片耗时参考：整页 2000×3083 在 Mac 上 **62 ms**（官方 torch 实现 33 ms），相对板端单块 21.9 s 可忽略。

---

## 7. tokenizer 对齐

| 项 | 结果 |
|---|---|
| C++ `test_tokenizer` vs HF | 文本 / 图像 / 视频 3 类用例 **5/5 PASS** |
| txt 大小 / 格式 | 12,102,644 B / magic 430、type 3、HUGGINGFACE BPE |
| 特殊 token | 56 个（比官方 MiniCPM-V-4.6 多 6 个 slice/thinking token） |

坑：宿主机 transformers 5.20 导出只有 6.4 MB（应为 12.1 MB），换 5.22 后正常。

---

## 8. 端到端验证

| 用例 | 结果 |
|---|---|
| 文本 "1+1等于几？只输出数字" | **"2"**，0.41 s |
| 文本 "请用一句话解释 OCR" | 通顺中文解释，1.90 s |
| 文本 中译英 | 1.10 s |
| 448×448 表格块（Table Recognition:） | 正确 OTSL：6 行 × 8 列（Name / Characteristics / Sample Size… / C. albicauda…），与图中表格内容一致 |
| 官方 MiniCPM-V-4.6 包（对照） | 描述红矩形 + 白字 HELLO AX8850 正确，2.87 s |

量化方法（板端 vs HF CPU 基线）已经写好：`scripts/compare_ocr_output.py`（字符级相似度 +
表格单元格 P/R/F1 + 数字准确率）。本仓库验证的是单块与 12 页抽样；**OmniDocBench 全量 1651 页未跑**（耗时约 1~2 天/单板，后续可批量跑）。

---

## 9. 验收标准（本项目采用）

| 项 | 线 |
|---|---|
| vision ONNX 导出 cosine | ≥ 0.999 |
| vision axmodel 板端对拍 cosine | ≥ 0.99（U16 探针 0.9979；U16 全量待测） |
| 切片 token 数 | 与官方逐片精确一致（不允许 ±） |
| tokenizer | 官方 C++ 测试全 PASS |
| 端到端 | 表格块人工核对 OTSL 结构 + 单元格内容正确 |

---

## 10. 验证状态与后续可做的消融

已完成：
1. ✅ U16 全量 27 层 vision axmodel：511 MB、615 ms、对 ONNX cosine **0.995322**（Pulsar2 **6.0** 编译；7.0/8.0 会崩）。
2. ✅ 端到端单块对比：字符相似度 1.0、数字 12/12。
3. ✅ 整页/版面模式的端到端质量：OmniDocBench 12 页 page/oracle 两种口径（见 `omnidocbench.md`）；
   版面模式实测 PPT 页 19.8 s、整页表格页 53.8 s。
4. ✅ 版面模型（DocLayout-YOLO）：U16 axmodel 35 MB / 212 ms，boxes cos 0.999999、logits 0.99995（见 `layout-mode.md`）。

仍可做的消融（不影响当前交付）：
- 官方 U16 配置里 `precision_analysis_method=EndToEnd` / `smooth_quant` 是否必需（我们最终版已去掉，精度 0.9953 达标）；
- 更大表格按行切分的合并策略在密集表格上的边界情况；
- 版面阅读顺序在复杂多栏页上的后处理优化。


## 最终精度数据（2026-11-11，板端 ax_run_model / 服务实测）

| 模型/配置 | 对拍对象 | cosine |
|---|---|---|
| 视觉 U16 全量（Pulsar2 **6.0** 编译，511 MB，615 ms） | ONNX（torch 0.9999999992） | **0.995322** ✅ |
| 视觉 INT8 全量（Pulsar2 8.0，459 MB，520 ms） | 同上 | **0.311** ❌（图像不可用） |
| 视觉 U16 8 层探针 | torch 参考 | 0.999179 |
| 视觉 U16 2 层探针 | torch 参考 | 0.997622 |
| INT8 + smooth_quant 8 层探针 | torch 参考 | 0.771329（无效） |

端到端（单块 504×392 表格切片，prompt `Table Recognition:`）：
- 板端输出 vs HF CPU 参考：**字符相似度 1.0、12/12 数字一致**（`compare_ocr_output.py`）。

## 编译坑（必须记录）

1. **U16 全量图在 Pulsar2 7.0 / 8.0 编译崩溃**：`tiling op` 完成后 `new_ddr_tensor = [...]` 打印，
   随后在 `build op serially` 阶段原生崩溃（exit 255，无 Python traceback）。
   - 8.0 NPU1 / NPU3、onnxsim 简化图（1024 节点）、7.0 均复现；
   - 同配置 2 层 / 8 层探针可编译，16 层探针即崩 → 规模相关；
   - 内存、会话、工具链版本均排除。
2. **Pulsar2 6.0 可正常编出全量 U16**（官方 MiniCPM-V-4.6 vision axmodel 就是 6.0 编的，见其包内
   `Pulsar2 Version: 6.0` 头）。6.0 编译耗时约 3 h（8 核），产物 511 MB。
3. INT8（U8/S8）或 INT8+smooth_quant 无法把 27 层视觉塔的 cosine 拉到 0.99（最好 0.771），
   **不要尝试用 INT8 视觉模型上线**。
