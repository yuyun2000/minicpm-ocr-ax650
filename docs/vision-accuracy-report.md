# MiniCPM-OCR 视觉编码器在 AX8850 上的精度问题报告（可复现）

> 2026-11-10｜板卡 AX650N（Ubuntu 22.04 aarch64，kernel 5.15.73）｜工具链 Pulsar2 8.0 / 7.0 + Magnetar
> 状态：**已定位到"视觉编码器量化/编译后精度崩塌"**，逐层探针数据见 §4；修复实验进行中（§6）。

## 1. 现象

| 测试 | 结果 |
|---|---|
| 文本链路（LLM axmodel） | ✅ 正常："1+1"→"2"；17 token 生成 2.0 s |
| **视觉链路** | ❌ 板端视觉 axmodel 输出与 ONNX 参考 **cosine = 0.311**（应 ≥0.99） |
| 端到端表现 | 给纯白图也吐出像样表格；同一图多次输出不同 → 模型实际"看不到图"，靠语言先验编 |

## 2. 隔离方法（可复现）

核心思路：**不经过 axllm**，直接拿视觉 axmodel 在板端跑 `ax_run_model`，把输出 bin 拉回 Mac 与 ONNX/ Torch 参考对拍。

```bash
# 1) Mac：构造与运行时完全一致的输入（NaViT 打包 + (x/255-0.5)/0.5 归一化）
python3 - <<'PY'
import numpy as np; from PIL import Image
im = Image.open("/tmp/ocr_448.png").convert("RGB").resize((448,448), Image.BICUBIC)
a = (np.asarray(im, dtype=np.float32)/255.0 - 0.5)/0.5
pv = a.transpose(2,0,1).reshape(3,32,14,32,14).transpose(0,2,1,3,4).reshape(1,3,14,14336)
pv.tofile("/tmp/vision_in.bin")
PY
# 2) ONNX 参考（Mac）
#    sess.run(None, {"pixel_values": pv}) → /tmp/vision_ref.npy
# 3) 板端运行（板子 IP 按实际；输入 bin 名必须等于 ONNX 输入张量名 pixel_values）
scp /tmp/vision_in.bin root@<board>:/root/vision_in.bin
ssh root@<board> 'cd /root && rm -rf vtest && mkdir -p vtest/inputs/run0 vtest/out
  cp vision_in.bin vtest/inputs/run0/pixel_values.bin
  echo run0 > vtest/list.txt
  /opt/bin/ax_run_model -m /root/minicpm-ocr-v2/minicpm_ocr_vision_448.axmodel \
      -i vtest/inputs -o vtest/out -l vtest/list.txt'
scp root@<board>:/root/vtest/out/run0/image_embeds.bin /tmp/vision_npu.bin
# 4) 对拍
python3 -c "import numpy as np; a=np.fromfile('/tmp/vision_npu.bin',np.float32).astype(np.float64); b=np.load('/tmp/vision_ref.npy').astype(np.float64).ravel(); print(a@b/(np.linalg.norm(a)*np.linalg.norm(b)))"
```

> 注意：`ax_run_model` 的 `-i` 是"每次推理一个子目录"的父目录，`-l` 里放子目录名（不是 bin 文件名）；输出自动按输出张量名落盘（`image_embeds.bin`）。

## 3. 不是这些原因（已排除）

| 假设 | 验证 | 结果 |
|---|---|---|
| 输入归一化/通道顺序错 | 用 4 种输入（[-1,1]、[0,1]、[0,255]、反相）分别跑 | cosine 0.298 / 0.308 / 0.279 / 0.311，**都坏** → 不是喂法问题 |
| Pulsar2 版本与板端运行时（V1.45.0_P39.1, 2024-09）不兼容 | 同一 ONNX 分别用 **Pulsar2 7.0 / 8.0** 编译成 axmodel，板端对拍 | 两者 cosine **完全相同 0.8606**（2 层探针）→ 不是工具链/运行时版本问题 |
| ONNX 导出错 | torch vs ONNX 对拍 | 视觉合并图 cosine 0.9999999992；2 层探针 ORT vs torch = **1.000000** → 导出正确 |
| 某个深层/merger 单独坏 | 2 层探针 | 仅 2 层就已 **0.8606** → 误差是**逐层累积**，不是某一层/merger 的偶发问题 |

## 4. 逐层/逐配置数据（全部为板端 ax_run_model 输出 vs Torch 参考）

| 模型 | 量化/编译配置 | cosine | 板端耗时 |
|---|---|---|---|
| 2 层探针 | INT8 全量化（Pulsar2 8.0） | **0.8606** | 41 ms |
| 2 层探针 | INT8 全量化（Pulsar2 7.0） | **0.8606** | 41 ms |
| 2 层探针 | Conv/MatMul/Softmax FP32，其余 INT8 | **0.9274** | 917 ms |
| 27 层视觉塔+merger（生产模型） | INT8，2 张校准样本 + MinMax | **0.311** | 520 ms |

解读：
- 单看 2 层，INT8 就掉到 0.86，**每层平均误差过大**；27 层连乘后掉到 0.31，与端到端"瞎编"完全吻合。
- 把 Conv/MatMul/Softmax 改成 FP32，2 层只回到 0.9274（**仍不达标**），说明还有其它算子在丢精度（嫌疑：LayerNormalization、以及输入张量本身的量化）。
- LayerNormalization 目前无法直接设 FP32（Pulsar2 报 `TileFailException: AxLayerNorm`），只能通过 `ln_scale_data_type` 等间接手段。

## 5. 关键代码/数据位置

| 内容 | 路径 |
|---|---|
| 视觉 ONNX（塔+merge+merger） | `demo/minicpm-ocr/export/vision_448_merged.onnx`（输入 `pixel_values[1,3,14,14336]`，输出 `image_embeds[1,256,1024]`） |
| 视觉 axmodel（生产） | 板端 `/root/minicpm-ocr-v2/minicpm_ocr_vision_448.axmodel`（459 MB） |
| 2 层探针导出脚本 | `demo/minicpm-ocr/vision_probe_export.py` |
| 宿主机编译脚本（可指定 layer_configs） | `demo/minicpm-ocr/host_compile_vision.py` |
| 对拍脚本 | `demo/minicpm-ocr/vision_probe_compare.py` |
| 本报告 | `notes/minicpm-ocr/vision-accuracy-report.md` |

## 6. 修复实验（进行中/本轮已排队）

| 实验 | 配置 | 目的 |
|---|---|---|
| v2 重编 | 16 张真实切片校准 + MSE + transformer_opt_level=2（Pulsar2 8.0） | 验证"校准数据不足"是否是主因 |
| D | 除 LayerNorm 外全部 FP32 | 逼出 LayerNorm 的贡献 |
| E | 关键算子 U16 | 16-bit 激活是否够用、速度代价多大 |

（结果将追加到本报告 §7）

## 7. 实验结果（截至 2026-11-10 15:45 CST）

| 实验 | 配置 | cosine | 板端耗时（2 层探针） |
|---|---|---|---|
| 基线 | INT8 全量化，2 张校准 + MinMax | 0.8606 | 41 ms |
| 7.0 工具链 | INT8 全量化（Pulsar2 7.0） | **0.8606（与 8.0 完全相同）** | 41 ms |
| C | Conv/MatMul/Softmax FP32，其余 INT8 | 0.9274 | 917 ms |
| E | 关键算子 + LayerNorm 用 **U16** | 0.9275 | 62 ms |
| D | 除 LayerNorm 外全部 FP32 | **0.9681** | 921 ms |
| 生产全模型 | INT8，2 张校准 + MinMax | **0.311** | 520 ms |
| v2 全量 | 16 张真实切片 + MSE + transformer_opt_level=2 | 编译中（>65 min） | — |

结论（阶段性）：
1. **不是工具链/运行时版本问题**：7.0 与 8.0 编出的探针在板上表现完全相同。
2. **不是输入归一化/导出问题**：四种输入归一化都坏；ONNX 与 torch 对齐 1.0。
3. 误差**逐层累积**：2 层 0.86 → 27 层 0.31。
4. **量化是主因但不止一处**：把 Conv/MatMul/Softmax 提到 FP32 只到 0.927；U16 同样 0.927；把除 LayerNorm 外全部 FP32 才到 0.968 → **LayerNorm 的量化是最大单项损失源**（Pulsar2 不支持 LayerNorm 直接设 FP32，会报 `TileFailException: AxLayerNorm`）。
5. 代价：FP32 化后 2 层从 41 ms → 917 ms（约 22×），全模型不可接受；U16 是 62 ms（1.5×），但精度提升有限。

## 8. 双向验证：同一块板上跑官方 MiniCPM-V-4.6 包（关键对照）

为了区分"板子/运行时问题"和"我们转换问题"，把 AXERA 官方 `MiniCPM-V-4.6` 包（2.0 GB，含官方 vision axmodel `minicpmv4_6_vision_448.axmodel`，Pulsar2 **6.0** 编译、输出 `image_features [1,64,1024]`）原样放到同一块板上跑：

```
# 板端
cd /root/minicpm-v46 && ./bin/axllm serve . --port 18082
# 提问：图片里有什么？请描述颜色、形状和文字。（测试图：白底 + 红色矩形 + 文字 "HELLO AX650"）
```

**结果：官方包回答正确** —— "这张图片上有一个红色的矩形… 在矩形的下方，有白色的文字"（2.87 s，128 token 内）。

→ **板子运行时（`/soc/lib/libax_engine.so` V1.45.0_P39、2024-09）与 axllm 的 MiniCPMV46VL profile 都是正常的**；
→ 问题锁定在**我们转换出来的 vision axmodel**（量化/编译配置），与板端老运行时无关（用户提出的"运行时太老"假设已被这条对照实验排除）。

## 9. 可复现实验清单

| 实验 | 复现命令/文件 |
|---|---|
| 2 层探针导出 | `demo/minicpm-ocr/vision_probe_export.py --layers 2 --out /tmp/probe2.onnx` |
| 宿主机编译（可指定 layer_configs） | `demo/minicpm-ocr/host_compile_vision.py <onnx> <out_dir> <name>` |
| 板端对拍 | `demo/minicpm-ocr/vision_probe_compare.py --ref /tmp/probe2_ref.npy --npu <hidden.bin>` |
| 官方包对照 | 下载 `AXERA-TECH/MiniCPM-V-4.6` 全包 → 板上 `axllm serve` → 语义提问 |

## 10. 下一步修复选项（按性价比排序）

1. **v2 重编**（进行中）：16 张真实切片 + MSE + transformer_opt_level=2，若 ≥0.99 直接采用（同速度）。
2. **LayerNorm 专项**：试 `ln_scale_data_type`、把 LN 子图单独提精度（U16/FP32 组合），目标在 INT8 主干下把 27 层 cosine 拉到 ≥0.99。
3. **质量模式**：除 LN 外全 FP32（2 层 0.968 是下限值，全模型需实测），性能约 20× 慢，只作精度对照/兜底。
4. **向 AXERA 索取** `MiniCPM-V-4.6` vision axmodel 的编译配置（他们的 448 vision 模型在板上正常），或请他们出 MiniCPM-OCR 的 vision 支持——这是最可能一次到位的外部路径。


## 11. 官方转换配置（找到根因，2026-11-10 16:10 CST）

AXERA 官方转换仓库：**https://github.com/AXERA-TECH/MiniCPM-V-4.6.axera** — `model_convert/` 目录里有完整的 vision 转换资产：

| 文件 | 内容 |
|---|---|
| `pulsar2_configs/config_vision_448_npu3.json` | **vision 编译配置（关键）** |
| `export_onnx.py` | 固定 shape 导出（position embedding / window index 预先烘焙） |
| `prepare_calibration.py` + `download_dataset.sh` | 真实图像校准集（64 张） |
| `llm_build_ax650.sh` | `FLOAT_MATMUL_USE_CONV_EU=1 pulsar2 llm_build --model_type qwen3_5_text --hidden_state_type bf16 ...` |

官方 vision 配置要点（与我们 INT8 配置的差异就是根因）：

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

→ 官方 vision 是**全图 U16 激活（16-bit）**，不是 INT8/U8；并且开了 smooth quant、conv bias 用 FP32。

## 12. 验证：用官方配置重建 2 层探针 → 精度达标

| 配置 | cosine | 板端耗时 |
|---|---|---|
| 我们原配置 INT8（U8/S8） | 0.8606 | 41 ms |
| **官方配置（U16 + smooth quant + FP32 conv bias + disable_auto_refine_scale）** | **0.997895** ✅ | 87.5 ms |

→ 根因确认：**视觉编码器必须用 U16 激活**，INT8 会把 27 层视觉塔打崩。已按官方配置启动全量 vision 重建（`build_vision_uint16.py`，64 张真实切片校准）。

## 13. 官方 V4.6 视觉模块能否直接替换我们的 OCR 视觉模块？

| 项 | 官方 MiniCPM-V-4.6 | 我们的 MiniCPM-OCR 1.0 | 可否替换 |
|---|---|---|---|
| 视觉塔结构 | hidden 1152 / 27 层 / 16 头 / patch 14 / intermediate 4304 | **完全相同** | 结构一致 |
| 视觉权重 | 453 个张量 | 453 个张量 | **逐张量 SHA 对比全部不同** → 不是同一权重 |
| 压缩率 | **16x**（448 块 → 64 tokens） | **4x**（448 块 → 252/256 tokens） | ❌ token 数不匹配，LLM 输入会错位 |

结论：**不能直接替换**。但结构与官方同源，说明可以照它的编译配置（U16）重编我们自己的 vision 权重——这正是当前修复路线。


## 14. U16 全量重编遇到的编译器崩溃（2026-11-10 22:20）

U16 配置在探针上精度达标（见 §12），但**全量 27 层 + merger** 的 U16 编译在 Pulsar2 原生阶段崩溃，现象与已排除项：

| 尝试 | 工具链/模式 | 图 | 结果 |
|---|---|---|---|
| u16f | Pulsar2 8.0 / NPU1 | 原图 1498 节点 | 校准通过，tiling 45917 ops 后崩溃（`new_ddr_tensor = []` 后无输出，native 退出） |
| u16n3 | Pulsar2 8.0 / NPU3 | 原图 | 同上，tiling 15974 ops 后崩溃 |
| u16sim | Pulsar2 8.0 / NPU1 | onnxsim 简化图 1024 节点 | 校准第 7/16 样本崩溃 |
| vision_u16_70 | Pulsar2 7.0 / NPU1 | 简化图 | tiling 47807 ops 后崩溃 |

- 崩溃点：`new_ddr_tensor = []` 之后、`build op serially` 之前，进程被信号杀死（无 Python traceback；`run_pulsar2` 捕获到 returncode≠0）。
- 排除：不是内存（主机 62 GB，峰值 RSS ~15 GB，无 OOM 记录、无 cgroup 限制）、不是会话被杀（用保持连接的 ssh 后台任务复现）、不是输入归一化/工具链版本（7.0/8.0 同样崩）。
- **对照：同一 U16 配置在 8 层探针上编译成功，cosine 0.999179（板端，338.7 ms）**；2 层探针 0.997622。说明 U16 量化本身没问题，是**全量图规模**触发编译器崩溃。

### 下一步（进行中）
1. `compiler.check=1` 诊断构建（Pulsar2 7.0）——希望定位具体算子；
2. 16 层探针 U16 构建——确定可编译规模上限；
3. 若确认为规模限制：按"U16 前 N 层 + INT8 其余"找可编译且精度 ≥0.99 的组合，或向 AXERA 提 issue（用户已授权）。
