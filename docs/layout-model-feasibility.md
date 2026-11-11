# PP-DocLayoutV3（原始仓库的版面模型）能否转 AX8850？—— 可行性调查

> 2026-11-10｜目标：MiniCPM-OCR 原始仓库用「PP-DocLayoutV3 版面检测 + 按区域 OCR」，评估把该版面模型也转成 axmodel 的可行性/收益。

## 1. 原始仓库怎么用版面模型

`MiniCPM-OCR` 仓库（`docs/minicpm_ocr_1dot0_zh.md`、`README.md` Stage 1）：
- Stage 1：**PP-DocLayoutV3**（PaddlePaddle / PaddleX）检测文档区域（text/table/formula/chart/seal），分析阅读顺序，SDK 按区域裁剪；
- Stage 2：MiniCPM-OCR 对每个区域用不同 prompt（Text/Table/Formula/Seal Recognition）识别；
- 我们当前板端是**服务端网格切片（官方 processor 的 9 片）**，没有版面模型——阅读顺序按行优先，多栏/复杂版面会串行。

模型下载地址（仓库文档给出）：
`https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayoutV3_infer.tar`

## 2. 模型本身

| 项 | 值 |
|---|---|
| 格式 | Paddle 3.0 新 IR：`inference.json`（program） + `inference.pdiparams` |
| 参数量大小 | `inference.pdiparams` = **130.8 MB** |
| 架构 | `arch: DETR`（`inference.yml`），`min_subgraph_size: 3` |
| 输入 | `1×3×800×800`（Resize keep_ratio=false + Normalize(mean 0, std 1, norm_type none) + Permute） |
| 输出（社区 ONNX 版实测） | `logits [1,300,25]`、`pred_boxes [1,300,4]`、`out_masks [1,300,200,200]`、`order_logits [1,300,300]` |
| 类别 | 25 类（abstract…vertical_text） |

## 3. 转换尝试记录

### 3.1 Paddle → ONNX（paddle2onnx）

在转换机 192.168.20.38（x86_64）上：

```bash
python3 -m venv .venv-p2o
.venv-p2o/bin/pip install paddle2onnx paddlepaddle onnx packaging
.venv-p2o/bin/paddle2onnx --model_dir PP-DocLayoutV3_infer \
  --model_filename inference.json --params_filename inference.pdiparams \
  --save_file layout.onnx --opset_version 11
```

结果：**失败**。`import paddle` 直接 `Illegal instruction (core dumped)`（EXIT=132），paddlepaddle **3.3.1 与 3.0.0 都如此** → 该机器 CPU 缺少 Paddle 官方 wheel 要求的指令集（疑似 AVX512），不是模型问题。

→ 需要换一台支持 AVX512 的 x86 主机、或 Docker（`paddlepaddle/paddle` 镜像）才能导出。

### 3.2 直接找社区 ONNX（绕过 Paddle）

HF 上已有：
- `Bei0001/PP-DocLayoutV3-ONNX`：`PP-DocLayoutV3.onnx`(5 MB) + `PP-DocLayoutV3.onnx.data`(130 MB)
- `alex-dinh/PP-DocLayoutV3-ONNX`、`PaddlePaddle/PP-DocLayoutV2_onnx`

对 `Bei0001` 版做图分析（onnx，opset 18，1549 nodes）：

```
Conv 130, MatMul 164, Add 289, Mul 157, Reshape 164, Transpose 100,
Softmax 13, LayerNormalization 28, Sigmoid 38, Slice 31, Gather 26,
Resize 8, Erf 1, Sin/Cos 各 2,
⚠ GridSample 18, GatherND 1, GatherElements 1, ScatterElements 4,
⚠ TopK 1, Trilu 6, ReduceMin 3, ReduceMax 5, Where 8, Expand 3, Tile 1
```

**结论：这是 DETR（可变形注意力）结构**，含 `GridSample/GatherND/Trilu/ScatterElements` 等 Pulsar2 对 AX8850 支持很差的算子。要转成 axmodel，大概率需要改图（把 deformable attention 换成普通 attention 或 CPU 回退），工作量大且精度/收益不确定。

## 4. 收益侧：要不要版面模型？

决定"值不值得转"的关键是：**网格切片 vs 理想版面**的精度差。
设计了一个直接对照实验（OmniDocBench 子集，待 vision 精度修复后跑）：

| 模式 | 说明 |
|---|---|
| `page`（现状） | 服务端官方 9 片网格切片，按行优先拼接 |
| `oracle`（上限） | 用 OmniDocBench GT 版面框逐区域裁剪+按类型 prompt 识别，再按阅读顺序拼接 |

指标：文本编辑距离（OmniDocBench v1.6 口径，quick_match）、表格单元格 F1。
- 若 `oracle` 与 `page` 差距小（≤2–3 个点）→ **不值得引入版面模型**，继续网格切片；
- 若差距大 → 再考虑版面模型，且优先选 **DocLayout-YOLO**（纯卷积 YOLO 检测器，1024×1024 输入，~20 MB，Pulsar2 友好），而不是 DETR 的 PP-DocLayoutV3。

## 5. 阶段性结论

1. **PP-DocLayoutV3 转 AX8850 不建议**：DETR+GridSample 结构，Pulsar2 支持差；且本机无法用 paddle2onnx 导出（CPU 指令集问题）。
2. **先量化收益**：用 `demo/minicpm-ocr/eval_omnidoc.py --mode oracle|page` 跑 OmniDocBench 子集对比，再决定是否引入版面模型。
3. 若最终要上版面模型，候选换成 **DocLayout-YOLO（onnx）**，转换路径与 AX8850 兼容性都好得多。

## 6. 复现命令

```bash
# 模型下载
curl -L -o PP-DocLayoutV3_infer.tar \
  https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayoutV3_infer.tar
tar xf PP-DocLayoutV3_infer.tar

# 社区 ONNX 版（免 Paddle）
huggingface-cli download Bei0001/PP-DocLayoutV3-ONNX \
  --local-dir ./layout_onnx

# 收益对照（vision 修复后）
python demo/minicpm-ocr/eval_omnidoc.py --data /tmp/omnidoc --mode page   --limit 20
python demo/minicpm-ocr/eval_omnidoc.py --data /tmp/omnidoc --mode oracle --limit 20
```
