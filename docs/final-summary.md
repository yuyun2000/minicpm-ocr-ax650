# MiniCPM-OCR on AX8850 —— 最终交付摘要（2026-11-11）

## 交付物

| 类型 | 位置 |
|---|---|
| 转换/部署/服务/评测代码 + 文档 | https://github.com/yuyun2000/minicpm-ocr-ax650 |
| 模型包（U16 vision + U16 版面，2.73 GB，37 文件） | https://huggingface.co/yunyu1258/minicpm-ocr-1.0-ax650 |
| 板端服务（网页 + API） | http://192.168.10.144:8080/ （axllm 18081） |
| 问题报告/笔记 | `notes/minicpm-ocr/`：vision-accuracy-report.md、omnidocbench-eval.md、layout-model-feasibility.md、slicing-verification.md、deployment-report.md |

## 关键数字

- 视觉 U16 cosine **0.9953**（INT8 0.311 → 图像完全不可用）；单块视觉 615 ms
- 单块表格 OCR 与 HF 参考：字符相似度 **1.0**，数字 12/12；耗时 12.1–22 s
- 整页切片：9 片 / 2276 image token，与官方 processor 逐片一致（尺寸差 0 px）；约 147 s/页
- 表格/版面模式：表格整表识别 22.9 s；版面检测 U16 212 ms、PPT 页 19.8 s、整页表格页 53.8 s
- OmniDocBench（12 页分层抽样）：TextEdit page **73.46** vs oracle **47.42**（版面收益 26 点）
- 并发（axllm 单并发）：文本 0.55 req/s；OCR 0.082 req/s（12.2 s/块）
- 20.38 磁盘：65 GB → **8.1 GB**（保留生产产物）

## 关键坑

1. 视觉编码器必须 **U16**（INT8/SQ 都不行），且必须用 **Pulsar2 6.0** 编译全量 27 层——7.0/8.0 在
   `build op serially` 阶段原生崩溃（规模相关，16 层探针即崩）。
2. prefill 桶必须 ≥ 图像 token 数（276）：`--prefill_len 1152 --prefill_step_size 128 --max_context 2048`。
3. 板子双网卡，IP 会在 192.168.10.144/.145 间漂移，服务地址以实际 ping 结果为准。
4. 板端 4 GB CMM 只能同时跑一套模型（官方包与我们的包不能同时 serve）。
