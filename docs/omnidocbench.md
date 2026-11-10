# OmniDocBench 评测：板端 MiniCPM-OCR（AX650）vs 官方仓库指标

> 2026-11-11｜数据：`opendatalab/OmniDocBench`（1651 页标注 + 图片）｜板端：AX650N，U16 vision，axllm 18081
> 评测脚本：`demo/minicpm-ocr/eval_omnidoc.py`（page 模式 / oracle 模式），指标为文本编辑距离（越低越好）

## 1. 评测口径

- 官方仓库（`docs/benchmark_reproduction_en.md`）：OmniDocBench v1.6，`quick_match` 元素匹配后计算
  **TextEdit（文本编辑距离）**、FormulaCDM、TableTEDS；`Overall = ((1-TextEdit)*100 + CDM + TEDS)/3`。
- 本机评测：因为没装 OmniDocBench 的 Docker 评分环境（CDM 需要 LaTeX 环境），先只算 **文本编辑距离**：
  `1 - difflib 相似度`（GT = 按 `order` 排序的 text_block/title/list_group/caption/footnote/reference +
  latex；预测 = 板端服务输出，去 OTSL/Markdown 标记后比较）。
- 因此这里的数字**不能直接和官方 Overall 分数相减**，用于：
  1. 我们的 **无版面模型整页网格切片** 与 **理想版面+按区域 prompt** 的差距（决定要不要引入版面模型）；
  2. 板端模型输出质量是否与 HF 参考处于同一水平（单块对拍已证明一致）。

## 2. 样本

从 1651 页中按 subset 分层抽样 12 页（各 3 页）：
`equation_hard`（公式难）、`layout_hard`（版面难）、`table_hard`（表格难）、`v1.5`（常规）。

## 3. 结果 A：page 模式（我们的现状 = 服务端 9 片网格切片 + 统一 `Text Recognition:` 提示词）

| 页 | subset | TextEdit | 耗时 |
|---|---|---|---|
| 1 | equation_hard | 0.7436 | 129.7 s |
| 2 | equation_hard | 0.8056 | 93.5 s |
| 3 | equation_hard | 0.9016 | 209.3 s |
| 4 | layout_hard | 0.7576 | 80.0 s |
| 5 | layout_hard | 0.4286 | 24.8 s |
| 6 | layout_hard | 0.5185 | 71.3 s |
| 7 | table_hard | 0.9790 | 82.9 s |
| 8 | table_hard | 0.8107 | 103.7 s |
| 9 | table_hard | 0.8814 | 172.1 s |
| 10 | v1.5 | 0.5724 | 50.5 s |
| 11 | v1.5 | 0.7826 | 22.0 s |
| 12 | v1.5 | 0.6336 | 61.6 s |
| **平均** | | **0.7346 (TextEdit 73.46)** | **91.8 s/页** |

按 subset：equation_hard 0.817 / layout_hard 0.568 / table_hard 0.890 / v1.5 0.663。

**结论（A）**：不用版面模型、整页统一 `Text Recognition:` 提示词时，表格页和公式页错误率极高
（表格 0.89、公式 0.82），主要原因是「网格切片 + 单一提示词」把表格/公式当普通文本识别。

## 4. 结果 B：oracle 模式（GT 版面框 + 按类型 prompt，代表模型能力上限）

| 页 | subset | TextEdit | 耗时 | 区域数 |
|---|---|---|---|---|
| 1 | equation_hard | 0.7905 | 161.6 s | 15 |
| 2 | equation_hard | 0.6256 | 128.5 s | 16 |
| 3 | equation_hard | 0.6715 | 234.6 s | 22 |
| 4 | layout_hard | 0.1358 | 96.5 s | 12 |
| 5 | layout_hard | 0.1552 | 26.5 s | 7 |
| 6 | layout_hard | 0.1647 | 40.9 s | 6 |
| 7 | table_hard | 0.9680 | 53.3 s | 4 |
| 8 | table_hard | 0.6868 | 70.1 s | 6 |
| 9 | table_hard | 0.4633 | 131.2 s | 18 |
| 10 | v1.5 | 0.0568 | 37.1 s | 8 |
| 11 | v1.5 | 0.6154 | 8.4 s | 3 |
| 12 | v1.5 | 0.3563 | 70.4 s | 11 |
| **平均** | | **0.4742（TextEdit 47.42）** | **99.2 s/页** | — |

按 subset：equation_hard 0.696 / layout_hard **0.152** / table_hard 0.706 / v1.5 0.343。

## 4.1 A vs B：版面模型能带来多少收益（决定"要不要转 PP-DocLayoutV3"）

| subset | page（网格切片） | oracle（GT 版面） | 提升 |
|---|---|---|---|
| equation_hard | 0.8169 | 0.6959 | 0.121 |
| layout_hard | 0.5682 | **0.1519** | **0.416** |
| table_hard | 0.8904 | 0.7060 | 0.184 |
| v1.5 | 0.6629 | 0.3428 | 0.320 |
| **总体** | **0.7346** | **0.4742** | **0.260** |

**结论**：版面信息带来的总体提升约 **26 个点**（多栏/复杂版面页提升 42 点），收益明确；
但 **PP-DocLayoutV3（DETR + GridSample）不适合 AX650**（见 `layout-model-feasibility.md`）。
若后续要补版面能力，建议换 **DocLayout-YOLO**（纯卷积、AX650 友好、~20 MB）走 Pulsar2 转换。

## 5. 与官方仓库指标的关系

官方 OmniDocBench v1.6 分数是「PP-DocLayoutV3 版面检测 + 按区域 prompt + 全文 Markdown 拼接」的全流程结果，
且 Overall 还包含 TableTEDS / FormulaCDM。我们的 A 结果没有版面模型，**不构成对模型能力的评价**；
B 结果（oracle）才是模型在该 benchmark 上的近似能力值。

## 6. 并发与吞吐（板端 axllm，max concurrency=1）

| 场景 | 并发 | 吞吐 | 延迟 |
|---|---|---|---|
| 文本（`2+3` 类短问答） | 1 | 0.55 req/s | 均值 1.82 s |
| 文本 | 4 | 0.75 req/s | 均值 3.93 s，P95 5.32 s |
| 图像 OCR（504×392 表格块，128 token 输出） | 1 | 0.082 req/s | 均值 12.2 s |
| 图像 OCR | 4 | 0.164 req/s | 请求排队（axllm 单并发） |

整页切片（9 片）：约 **1.5–3.5 min/页**（页面尺寸和内容密度决定；12 页实测平均 91.8 s）。
官方仓库宣称 1.2 pages/s（vLLM/L40S 级别），端侧 AX650 差 2~3 个数量级，属预期。

## 7. 复现

```bash
# 板端服务（192.168.10.144:8080）
curl -s http://192.168.10.144:8080/api/health

# 页面/区域两种评测
BOARD_API=http://192.168.10.144:8080 python demo/minicpm-ocr/eval_omnidoc.py \
  --data /tmp/omnidoc --mode page  --limit 12 --max-tokens 512 --pred-dir /tmp/omnidoc/pred_page
BOARD_API=http://192.168.10.144:8080 python demo/minicpm-ocr/eval_omnidoc.py \
  --data /tmp/omnidoc --mode oracle --limit 12 --max-tokens 512 --pred-dir /tmp/omnidoc/pred_oracle
```
