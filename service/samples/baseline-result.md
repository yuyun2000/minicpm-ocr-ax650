# MiniCPM-OCR 1.0 本机 CPU 基线结果

- 生成时间：2026-11-10 14:27:09
- 机器：macOS / Apple M4 Pro / 24 GiB（无 CUDA，torch 2.15.0 CPU）
- 权重：`demo/minicpm-ocr/origin`（BF16 官方权重，加载为 float32）
- 图片：`/tmp/ocr_448.png`，原图 448×448 → 输入 448×448
- 提示词：`Table Recognition:`，贪心解码，max_new_tokens=160

## 关键数字

| 指标 | 数值 |
| :--- | ---: |
| 加载耗时 | 4.0 s |
| 参数量 | 1.300 B |
| 图像预处理 | 0.02 s |
| 视觉块数（patch 总数） | 1024 |
| 图像占位 token 数 | 256 |
| 输入序列长度 | 277 |
| prefill（单次前向） | 5.58 s |
| 缓存（K/V + 线性注意力状态） | 6.5 MB / 12 张量 |
| 生成 token 数 | 102 |
| 生成耗时 | 374.0 s |
| 解码速度 | 0.27 tok/s |
| 端到端 | 383.6 s |
| RSS 峰值 | 5287 MB |

## target_sizes（每块 patch 网格 h×w）

```
[[32, 32]]
```

## 原始模型输出（未后处理，截断到 120 行）

```markdown
<otsl><fcel>Task<fcel>Benchmark<fcel>MiniCPM-V4.7<lcel><nl><fcel>Size<lcel><fcel>1.3B<lcel><nl><fcel>Visual Token Compression Ratio<lcel><fcel>4x<fcel>16x<nl><fcel>General<fcel>MMBench<fcel>83.17<fcel>81.89<nl><ucel><fcel>MMStar<fcel>69.53<fcel>68.80<nl><ucel><fcel>MMT-Bench<fcel>61.69<fcel>61.43<nl></otsl>
```
