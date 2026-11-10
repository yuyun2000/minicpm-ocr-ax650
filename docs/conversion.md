# 转换流程：HF 原始权重 → AX650 axllm 模型包

> 目标芯片：AX650 / AX650N ｜ 工具链：Pulsar2 8.0（+ Magnetar 做编排）｜ 源模型：`openbmb/MiniCPM-OCR-1.0`（1.30 B，BF16）
> 宿主机：Linux x86_64（实测 Ubuntu 24.04，16C / 62G）—— **Pulsar2 只发 Linux x86_64 独立包，macOS 上最多做到 ONNX 导出**
> 本文所有数字来自 2026-11-10 的本地实测记录，未实测的地方标 `TODO`。

---

## 0. 一句话原理

MiniCPM-OCR 的**文本栈就是 MiniCPM-V-4.6 的 Qwen3.5 hybrid 文本栈**（24 层 = 18 层线性注意力 + 6 层 full attention，
hidden 1024 / head_dim 256 / vocab 248100），AXERA 的 Pulsar2 8.0 已经支持过同架构（官方 `MiniCPM-V-4.6` 可运行包），
所以整条路是：

```
HF 权重 → 拆出纯文本 LLM → pulsar2 llm_build2 逐层 axmodel ┐
        └→ 导出 vision 静态 ONNX → pulsar2 build（U16）   ├→ 组装 axllm 模型包 → 板端 axllm serve
        └→ tokenizer.json → axllm 本地 tokenizer.txt       ┘
```

**必须拆图**：整模型 ONNX 走不通（原因见 §8），这是能落地的唯一路径，不是偏好。

---

## 1. 模型结构（实测参数统计，不是文档抄写）

| 分组 | 参数量 | 说明 |
|---|---:|---|
| vision encoder（27 层 SigLIP 类） | 411.47 M | 每层 15.24 M，hidden 1152，16 heads，patch 14，intermediate 4304 |
| patch + position embedding | 6.32 M | 位置表 70×70×1152（上限 980） |
| `vision_tower.vit_merger` | 104.51 M | **4x 模式下不参与前向，纯死重** |
| LLM 24 层 | 498.11 M | 18×21.56 M（GatedDeltaNet）+ 6×18.35 M（GQA full attn） |
| LLM `embed_tokens` | 254.05 M | `tie_word_embeddings=true`，无独立 `lm_head` |
| merger（4608→4608→1024） | 25.97 M | 1024 token → 256 token 的投影 |
| **合计** | **1300.43 M** | 推理实载 1195.9 M（减掉 vit_merger） |

关键超参（`config.json` / `text_config`）：

| 项 | 值 |
|---|---|
| LLM `layer_types` | `[linear, linear, linear, full] × 6`（`full_attention_interval=4`） |
| LLM hidden / head_dim / heads / KV heads | 1024 / 256 / 8 / 2 |
| LLM vocab / layers / max_position | 248100 / 24 / 262144 |
| vision layers / hidden / heads / patch | 27 / 1152 / 16 / 14 |
| 图像预处理 | `scale_resolution=448`、`max_slice_nums=9`、`patch_size=14`、`downsample_mode=4x` |
| 单块 image token | 448×448 → 1024 patch → **256 token**（4x 即 patch/4） |
| 整页（2000×3083） | 9 片 / 9104 patch / **2276 image token** / seq 2316 |
| 视觉塔输入 | NaViT 打包：`pixel_values (1, 3, 14, 14336)`，无动态维 |

> 视觉塔看到的不是 448×448 的图，而是「高 14、宽 1024×14」的 patch 列：`Conv2d(3,1152,k=14,s=14)`
> 等价于把 patch 序列当图来卷积。所以 ONNX 的输入形状固定为 `[1,3,14,14336]`。

---

## 2. 环境准备

| 项 | 要求 |
|---|---|
| 宿主 | Linux x86_64（Pulsar2 独立包内含 `ld-linux-x86-64.so.2`，macOS 无法执行） |
| Pulsar2 | 8.0（`llm_build2` 需 ≥6.0；本项目实测 8.0） |
| Python | 3.10+；导出 ONNX 需要 `torch / transformers>=5.22（原生 minicpm_ocr）/ onnx / onnxruntime / numpy / Pillow` |
| 磁盘 | 转换工作目录准备 ≥40 GB（24 层 + post + embed ≈2.1 GB，ONNX ≈3.2 GB，Pulsar2 work 目录 13+ GB） |
| 板端 | 从 AXERA 官方 AX650 包里取 `bin/axllm`（aarch64，本项目包内为官方二进制） |

可复现的最小工作目录：

```
work/minicpm-ocr/
├── origin/          # HF 原始权重（2.6 GB）
├── origin_text/     # prep_text_model.py 输出（纯文本 LLM，1.50 GB safetensors，752.2 M 参数）
├── llm_v2/          # llm_build2 输出（24 层 + post + embed）
├── vision_448_merged.onnx
├── vision_task/     # pulsar2 build 的 vision 产物
└── axllm_pkg/       # assemble_package.py 输出的最终模型包
```

---

## 3. Step 1：下载 HF 原始权重

ModelScope 直连可用（实测 14.5 MB/s，2.42 GiB 约 3 min）：

```bash
mkdir -p work/minicpm-ocr/origin && cd work/minicpm-ocr/origin
for f in model.safetensors tokenizer.json chat_template.jinja config.json \
         preprocessor_config.json generation_config.json tokenizer_config.json processor_config.json; do
  curl -sL --retry 3 -o "$f" \
    "https://modelscope.cn/models/OpenBMB/MiniCPM-OCR-1.0/resolve/master/$f"
done
# 校验（HF 侧同一文件的 sha256）
sha256sum model.safetensors   # TODO：把官方 sha256 填进 models/README.md
```

HuggingFace 侧需要镜像（`hf-mirror.com` 或用 `huggingface-cli download`）。

---

## 4. Step 2：把多模态 checkpoint 改造成「纯文本 LLM 目录」

```bash
python3 scripts/prep_text_model.py --src work/minicpm-ocr/origin --out work/minicpm-ocr/origin_text
```

做三件事：

1. **权重键改名**：`model.language_model.*` → `model.*`，视觉张量（`model.vision_tower.*` / `model.merger.*`）剔除。
   AXERA 的 `qwen3_5_text` builder 按纯文本 LLM 布局取 tensor。
2. **config 重标**：顶层 `model_type=qwen3_5`、`text_config.model_type=qwen3_5_text`，
   补 `bos/eos/pad_token_id`、`use_cache`、`dtype=bfloat16`。
3. 输出 `config.json + model.safetensors`（实测 752.2 M 参数 / 1.50 GB）。

**不这么做的原始报错**（两条都实测过）：

| 错误做法 | 报错 |
|---|---|
| 不改键名 | `'Config' object has no attribute 'linear_num_key_heads'` |
| 只改键名、不重标 `text_config.model_type` | `get_norm_attrs AssertionError` |

特殊 token（实测，来自 `generation_config.json` + tokenizer）：`bos=<|im_start|>=248045`、
`eos=[248044, 248046]`、`pad=<|endoftext|>=248044`；图像相关 `image_pad=248056`、`video_pad=248057`。

---

## 5. Step 3：`llm_build2` 编出 24 层文本 LLM

```bash
INPUT=work/minicpm-ocr/origin_text OUT_DIR=work/minicpm-ocr/llm_v2 bash scripts/build_llm.sh

# 脚本实际展开的命令：
pulsar2 llm_build2 \
  --input_path work/minicpm-ocr/origin_text \
  --output_path work/minicpm-ocr/llm_v2 \
  --chip AX650 --weight_type s8 --post_weight_type bf16 \
  --prefill_len 1152 --prefill_step_size 128 --max_context 2048 --parallel 8
```

| 参数 | 为什么是这个值 |
|---|---|
| `--prefill_len 1152` | 预填充桶上限。448 单块就吃 256 image token + prompt；首版桶只有 128 时板上直接报 `input_num_token(276) > prefill_max_token_num(128)` |
| `--prefill_step_size 128` | 桶步长，决定产物名里的 `p128`（`template_filename_axmodel` 必须对应） |
| `--max_context 2048` | 解码上下文；整页多切片时 2276 image token 已超 2048，见 §11 限制 |
| `--weight_type s8` | 层权重 INT8 |
| `--post_weight_type bf16` | post（lm_head/final norm）走 BF16，与 AXERA 官方 MiniCPM-V-4.6 包同策略 |
| `--parallel 8` | 8 并发；实测 24 层约 **60–90 min** |

产物（名字与官方包一致，Pulsar2 把该栈自动识别为 `qwen3_5_dense`）：

| 文件 | 大小 | 说明 |
|---|---:|---|
| `qwen3_5_text_p128_l{0..23}_together.axmodel` | ≈575 MB | 24 层，每层 20.6–26.6 MB |
| `qwen3_5_text_post.axmodel` | 1017 MB | BF16 post |
| `model.embed_tokens.weight.bfloat16.bin` | 508 MB | 248100×1024 bf16，mmap 加载 |

> `FLOAT_MATMUL_USE_CONV_EU=1`：官方 `MiniCPM-V-4.6.axera` 的 `llm_build_ax650.sh` 里设了这个环境变量；
> 本项目 v2 构建记录没落这一项，是否设置 **TODO**（不影响产物命名与加载）。

---

## 6. Step 4：导出 vision 静态 ONNX

```bash
python3 scripts/export_vision_onnx.py --model-dir work/minicpm-ocr/origin \
    --out-dir work/minicpm-ocr --image page.jpeg
```

产物 `vision_448_merged.onnx`（塔 + 2×2 merge + merger 一张图）：

| 项 | 值 |
|---|---|
| 输入 | `pixel_values float32 [1, 3, 14, 14336]` |
| 输出 | `image_embeds float32 [1, 256, 1024]` |
| 大小 / 节点 | 1676.1 MB / 1498 |
| ONNX vs torch cosine | **0.9999999992**（max_abs 2.46e-03） |
| ORT 单次 | 1.77 s |

另外一张「塔单独」的图（`vision_448x448.onnx`，输出 `[1,1024,1152]`，1577 MB，cosine 0.9999999997）只用来分段对分，
部署只用合并图。

### 6.1 四个必踩的坑

| 坑 | 现象 | 处理 |
|---|---|---|
| `target_sizes` 被常量折叠 | ONNX 里只剩 `pixel_values` 一个输入，喂 `target_sizes` 报 `Invalid input name` | 位置编码/`.view()` 需要 Python int，静态导出必然把 grid 固化；**换分辨率必须重导** |
| `target_sizes` dtype | int32 与常量 int64 在 Reshape/Concat 冲突：`Type parameter (T) of Optype (Concat) bound to different types` | merger 侧用 int64 |
| 输出带符号维 | `unsqueeze(0)` 在 ONNX 留 `Unsqueezeimage_embeds_dim_1`，NPU 编译器不接受 | 改成 `reshape(1, 256, 1024)` 静态形状 |
| dtype 不匹配 | config 里 `dtype: bfloat16`，`from_pretrained` 默认按 bf16 加载 → `expected scalar type Float but found BFloat16` | 显式 `torch.float32` 加载 |

图内算子分布（merged）：`Constant 297 / Add 272 / Mul 220 / MatMul 216 / Transpose 137 / Reshape 112 / Split 82 /
LayerNormalization 56 / Concat 30 / Softmax 27 / Tanh 27`。**没有 Gather/window-attention**——4x 模式下
`use_vit_merger=False`，第 6 层后的 `vit_merger` 根本不入图。

---

## 7. Step 5：编译 vision axmodel —— **必须 U16**

```bash
# 1) 准备 64 张真实文档切片的校准集（每个样本一个 (1,3,14,14336) float32 .npy + tar.gz）
# 2) 编译
python3 scripts/build_vision.py \
    --onnx work/minicpm-ocr/vision_448_merged.onnx \
    --calib work/minicpm-ocr/calib/pixel_values.tar.gz \
    --out-dir work/minicpm-ocr/vision_task --name minicpm_ocr_vision_448
```

### 7.1 量化配置（逐字段照抄官方，`build_vision.py` 里已内置）

来源：https://github.com/AXERA-TECH/MiniCPM-V-4.6.axera 的
`model_convert/pulsar2_configs/config_vision_448_npu3.json`：

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

**为什么必须 U16**：vision 是 27 层深度塔，INT8 的逐层误差会连乘。实测（板端 `ax_run_model` 输出 vs torch 参考）：

| 配置 | 2 层探针 cosine | 27 层生产模型 cosine | 板端耗时（2 层） |
|---|---:|---:|---:|
| INT8 全量化（Pulsar2 8.0） | 0.8606 | **0.311** ❌ | 41 ms |
| INT8 全量化（Pulsar2 7.0） | 0.8606 | — | 41 ms |
| Conv/MatMul/Softmax FP32，其余 INT8 | 0.9274 | — | 917 ms |
| 关键算子 + LayerNorm 用 U16 | 0.9275 | — | 62 ms |
| 除 LayerNorm 外全部 FP32 | 0.9681 | — | 921 ms |
| **官方配置（U16 + smooth_quant + FP32 conv_bias）** | **0.997895** ✅ | U16 全量重编中，见 §11 TODO | 87.5 ms |

INT8 板端 0.311 的端到端表现：**给纯白图也能吐出一张像样的表格，同一图多次输出不同** —— 模型实际"看不到图"，靠语言先验在编。
所以「能跑」和「跑对」是两件事，vision 精度必须单独对拍。

校准数据：**64 张真实文档切片**（官方 `prepare_calibration.py` + `download_dataset.sh` 同款），
不要用随机噪声/纯色图。输入归一化在 ONNX 之前就做完了：`(x/255-0.5)/0.5`，因此
`calibration_mean/std` 留空、`src_dtype=FP32`。

产物：`vision_task/model.axmodel`。**INT8 版 459 MB；U16 版体积/耗时待重编完成后补充（TODO）**。

---

## 8. 为什么不能把整模型导成一张 ONNX

5 条路径全失败（`scripts/` 里不再提供这个脚本，失败原因存档在 `docs/troubleshooting.md` §7）：

| 路径 | 关键报错 |
|---|---|
| `torch.onnx.export(dynamo=False)` 直接导 | `received an input of unsupported type: DynamicCache` |
| 张量接口 wrapper + `dynamo=False` | `aten::diff` to ONNX opset 18 is not supported（LLM masking 路径） |
| `torch.onnx.export(dynamo=True)` | `No module named 'onnxscript'`（可装，但见下一条） |
| `torch.export.export` | `GuardOnDataDependentSymNode`（NaViT 变分辨率位置编码要 host int） |
| 仅 LLM backbone | 探针签名冲突（`Qwen3_5TextModel.forward() got multiple values for argument 'use_cache'`） |

卡点三类：① 混合 cache（6 层 KV + 18 层 conv/recurrent 状态）+ `DynamicCache` 入参；② 线性注意力的
`torch.diff` / Python 循环 / chunk 填充在 ONNX 里没有对应算子；③ 变分辨率位置编码依赖 data-dependent SymInt。
→ 正确姿势就是拆图：vision 一张静态图 + LLM 走 `llm_build2` 逐层。

---

## 9. Step 6：tokenizer 转成 axllm 本地格式

axllm 只读本地 txt（`url_tokenizer_model`，字段名带 `url` 是历史遗留，**不支持 HTTP**）。
用 ax-llm 仓库里的转换器：

```bash
# 在 ax-llm 源码仓库里编译出 tokenizer.axera 后：
./tokenizer.axera/tests/convert_tokenizer.py --tokenizer <origin>/tokenizer.json \
    --output minicpm_ocr_tokenizer.txt
# 自检（官方 C++ 测试程序）：
tokenizer.axera/build/test_tokenizer -m 28
```

| 项 | 实测值 |
|---|---|
| 格式 | magic **430** / tokenizer_type **3**（HUGGINGFACE BPE）/ gpt2_byte_bpe |
| 大小 | **12,102,644 B**（12.1 MB） |
| 特殊 token | 56 个（比官方 MiniCPM-V-4.6 多 6 个 slice/thinking 相关 token） |
| 对齐验证 | C++ `test_tokenizer` 文本/图像/视频 3 类用例 **5/5 PASS** |

**踩坑**：宿主机 transformers 5.20 的 BPE 导出分支不同，只生成 6.4 MB（应为 12.1 MB）；
改用 transformers 5.22 生成后正常。转换机和导出机版本要对齐。

---

## 10. Step 7：组装 axllm 模型包

```bash
python3 scripts/assemble_package.py \
  --work-dir work/minicpm-ocr \
  --dest work/minicpm-ocr/axllm_pkg \
  --tokenizer work/minicpm-ocr/minicpm_ocr_tokenizer.txt \
  --axllm-bin /path/to/axera_official/bin/axllm \
  --axllm-version /path/to/axera_official/bin/axllm.version.json \
  --ctx 2048
```

生成的 `config.json` 关键字段（与板端实际生效值一致）：

| 字段 | 值 | 说明 |
|---|---|---|
| `model_name` | `MiniCPM-OCR-1.0-AX650-C2048-P1152` | `/v1/models` 里显示的名字 |
| `tokenizer_type` / `vlm_type` | `MiniCPMV46VL` | 复用官方 profile（同架构 Qwen3.5 hybrid） |
| `url_tokenizer_model` | `minicpm_ocr_tokenizer.txt` | 本地 txt |
| `template_filename_axmodel` | `qwen3_5_text_p128_l%d_together.axmodel` | 对应 prefill 桶步长 128 |
| `filename_post_axmodel` | `qwen3_5_text_post.axmodel` | BF16 post |
| `filename_tokens_embed` | `model.embed_tokens.weight.bfloat16.bin` | bf16 embed |
| `tokens_embed_num` / `tokens_embed_size` | 248100 / 1024 | |
| `axmodel_num` / `full_attention_interval` | 24 / 4 | |
| `layer_types` | 18×linear_attention + 6×full_attention | 与原始 config 逐项一致 |
| `filename_image_encoder_axmodel` | `minicpm_ocr_vision_448.axmodel` | 视觉编码器 |
| `vision_width/height/patch_size` | 448 / 448 / 14 | |
| `b_use_mmap_load_embed` / `b_use_mmap_load_layer` | true / true | 省内存 |
| `devices` | `[0]` | AX650 单卡 |

模型包最终体积 ≈ **2.5 GB**（v2：24 层 575 MB + post 1017 MB + embed 508 MB + vision 459 MB + tokenizer 12.1 MB + axllm 2.4 MB）。

生成的 `config.json` 全文：

```json
{
  "system_prompt": "",
  "model_name": "MiniCPM-OCR-1.0-AX650-C2048-P1152",
  "url_tokenizer_model": "minicpm_ocr_tokenizer.txt",
  "tokenizer_type": "MiniCPMV46VL",
  "post_config_path": "post_config.json",
  "template_filename_axmodel": "qwen3_5_text_p128_l%d_together.axmodel",
  "axmodel_num": 24,
  "full_attention_interval": 4,
  "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention",
                  "... × 6 组（共 24 项）"],
  "filename_post_axmodel": "qwen3_5_text_post.axmodel",
  "filename_tokens_embed": "model.embed_tokens.weight.bfloat16.bin",
  "tokens_embed_num": 248100,
  "tokens_embed_size": 1024,
  "vlm_type": "MiniCPMV46VL",
  "filename_image_encoder_axmodel": "minicpm_ocr_vision_448.axmodel",
  "vision_cache_dir": "vision_cache",
  "vision_width": 448,
  "vision_height": 448,
  "vision_patch_size": 14,
  "b_use_mmap_load_embed": true,
  "b_use_mmap_load_layer": true,
  "devices": [0]
}
```

> 命名提醒：层文件名里的 **`p128` 是 prefill 桶的步长**（`--prefill_step_size 128`），
> 不是桶上限；桶上限是 `--prefill_len 1152`，板端 `/v1/models` 返回的 `model_name` 里 `P1152` 才是它。
> `config.json` 本身没有 prefill 字段——桶大小由编译产物决定（`docs/troubleshooting.md` §2）。

---

## 11. 已知限制与待补充

1. **vision 是固定 448×448 单块**（每块 256 token）。整页多切片要靠应用侧切块（见 `docs/deployment.md` / `service/board_app.py`）。
2. `max_context=2048` 下整页 9 片 ≈2276 image token **超出上下文**，整页模式需限制片数（板端默认每页最多 9 片，实际按 2048 截断，`TODO`：确定建议片数上限）。
3. `vlm_type` 复用官方 `MiniCPMV46VL` profile：预处理/模板与 MiniCPM-OCR 基本一致（chat template 仅 thinking 段差异），
   但**图像插入位置/切片 token 规则需逐样本核对**。
4. DSpark 解码加速（官方称 +55% 效率）未覆盖。
5. 训练/微调链路、PDF 多页批处理、版面模型（官方 SDK 用 PP-DocLayoutV3）未包含。
6. **U16 全量 vision 重编的板端对拍结果未定稿**（本次文档定稿时只拿到 2 层探针 0.997895）；
   重编完成后的 `cosine / axmodel 大小 / 单块耗时` 待补。
7. `FLOAT_MATMUL_USE_CONV_EU=1` 是否影响 v2 构建：未记录。
8. 权重/模型包的 sha256 校验清单：见 `models/README.md`（待填）。
