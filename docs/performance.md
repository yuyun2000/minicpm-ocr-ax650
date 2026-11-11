# 性能：AX8850 板端实测（AX650N/AX650C 板型）

> 板卡：AX8850（AX650N 板型）aarch64，Ubuntu 22.04，kernel 5.15.73，8 核 / 3.8 GB DDR / CMM 4096 MB
> ⚠️ **本节为历史数据**（首版 INT8 vision 包，2026-11-10）。最终 U16 vision 包的数字见文末「最终实测」一节。
>
> 模型：MiniCPM-OCR 1.0 → AX8850（24 层 INT8 + BF16 post + 首版 INT8 vision），axllm 18081
> 数据来自 2026-11-10 ~ 2026-11-11 板端实测；开头的「历史数据」是首版 INT8 包，最终 U16 包见文末「最终实测」。

---

## 0. 结论速览

| 场景 | 端到端 | decode | 备注 |
|---|---:|---:|---|
| 文本 "1+1"（prompt 21 → 1 token） | **0.41 s** | — | 生成 1 个 token |
| 文本 一句话解释 OCR（22 → 29） | **1.90 s** | ≈15.3 tok/s | |
| 文本 中译英（32 → 13） | **1.10 s** | ≈11.8 tok/s | |
| **图像 OCR**（448×448 表格块，prompt 277 → 237） | **21.94 s**（中位） | **10.8 tok/s** | 两轮 21.925 / 21.952 s |
| 网页测试台端到端（图片 OCR） | 22.1 s / 237 token | 10.7 tok/s | 经 Flask 转发 |
| 网页测试台端到端（纯文本） | 2.0 s / 17 token | — | |

- 图像请求的 **TTFT ≈ 2.09–2.15 s**（其中视觉编码 636–688 ms，文本 prefill 1454–1458 ms）。
- axllm 启动到服务可用 ≈ **3 s**（24 层 + post + vision 全部 mmap 加载）。
- CMM 占用 ≈ **900 MB**（首版 s8/p64 包）。
- 单块图像 ≈ 21.9 s，主要是 237 个输出 token 的逐 token 解码（10.8 tok/s）；**整页 9 块是串行的**，
  按块估算约 3 分钟级（推算，见 §5）。

---

## 1. 板端环境

| 项 | 值 |
|---|---|
| 板卡 | AX8850（AX650N 板型），aarch64 |
| 系统 | Ubuntu 22.04，kernel 5.15.73，Python 3.10.12 |
| 内存 | 3.8 GB DDR，CMM 分区 4096 MB |
| 磁盘 | 24 GB（部署 2.5 GB 模型包后剩 ≈16 GB） |
| NPU 运行时 | `/opt/bin/ax_run_model`；`/soc/lib/libax_engine.so` V1.45.0_P39（2024-09） |
| 模型包 | v2，≈2.5 GB（24 层 575 MB + post 1017 MB + embed 508 MB + vision 459 MB INT8 + tokenizer 12.1 MB + axllm 2.4 MB） |

---

## 2. 文本性能（解码为主）

| 请求 | prompt tokens | completion tokens | 端到端 | decode |
|---|---:|---:|---:|---:|
| "1+1等于几？只输出数字" | 21 | 1 | 0.41 s | — |
| "请用一句话解释 OCR" | 22 | 29 | 1.90 s | ≈15.3 tok/s |
| 中译英 | 32 | 13 | 1.10 s | ≈11.8 tok/s |

短输出时 TTFT 占比高；输出越长、decode 吞吐越接近 10–15 tok/s 区间。
其中 "1+1" 的 0.41 s 可视为**纯 TTFT 下限**（首 token 到出结果）。

---

## 3. 图像性能（448×448 单块）

基准口径：`Table Recognition:` + 一张 448×448 表格块，重复两轮（`notes/minicpm-ocr/perf-image.json`）：

| 指标 | 数值 |
|---|---|
| 端到端（中位） | **21.939 s**（min 21.925 / max 21.952） |
| prompt tokens / image tokens | 277 / 277（1 块 = 256 image token + prompt） |
| completion tokens | 237 |
| decode | **10.8 tok/s**（两轮 10.80 / 10.81） |
| 每 token 平均 | ≈92.6 ms |

板端 `serve.log` 的细粒度拆解：

| 阶段 | 数值 |
|---|---|
| 图像预处理 + 视觉编码（preprocessImage → vision） | 636–688 ms |
| 文本 prefill（277 token，3 个 chunk：128+128+21） | 1454–1458 ms |
| **TTFT（含视觉编码）** | **2090–2145 ms** |
| decode（serve.log 口径） | 11.88–11.94 tok/s |
| 单块端到端（≈240 token 输出） | 21.9 s |

也就是说：**21.9 s 里约 2 s 是"看图 + prefill"，剩下 ≈20 s 全是逐 token 解码**。
想进一步提速，重点在 decode（更小的输出、更短的 prompt、或等官方 DSpark 解码加速）。

网页测试台端到端（多一层 Flask 转发与切片）：

| 场景 | 端到端 | tokens | decode |
|---|---:|---:|---:|
| 图片 OCR（单块） | 22.1 s | 237 | 10.7 tok/s |
| 纯文本 | 2.0 s | 17 | — |

Flask 转发只增加约 0.2 s（22.1 s vs 21.9 s），可以忽略。

---

## 4. 内存与加载

| 项 | 实测值 |
|---|---|
| 加载 24 层后 `remain_cmm` | 3788 MB → **3154 MB** |
| 加载 post 后 `remain_cmm` | **2889 MB** |
| 合计 CMM 占用 | **≈900 MB** |
| 启动到服务可用 | ≈ **3 s**（24 层 + post + vision，mmap） |

> 这是首版 **s8/p64** 包的数字。v2 包（prefill 桶 1152 / ctx 2048）与 U16 vision 包的内存占用
> v2 + U16 包实测：`init post axmodel ok, remain_cmm(2130 MB)`，max_token_len 2047，prefill 桶 1152；
> `/proc/ax_proc/mem_cmm_info` 当前占用约 2.5 GB（CMM 总 4096 MB）。

---

## 5. 整页与吞吐：能算什么、不能算什么

**实测能算的**：单块 448×448 端到端 21.9 s。

**推算（不是实测，标注为估算）**：

| 场景 | 估算 | 算法 |
|---|---|---|
| 整页 9 块（串行） | ≈3.3 min | 9 × 21.9 s + 切片/拼接开销（未计） |
| 单块吞吐 | ≈2.7 块/min | 60 / 21.9 |
| 237 token 输出的有效吞吐 | 10.8 tok/s | 实测 |

整页为什么不能更快：

1. 板端 vision 编码器是**固定 448×448 单块**，9 片必须 9 次前向（应用侧循环）；
2. `max_context=2048`，整页 9 片 ≈2276 image token 已超上下文，整页模式实际要限片数；
3. 没有 batch / 流水线，Flask 是串行逐片调用。

> 并发实测（axllm 最大并发 1，请求排队）：文本并发 1/2/4 → 0.55/0.56/0.75 req/s（均值 1.82/3.13/3.93 s，P95 5.32 s）；
> OCR 块并发 1/2/4 → 0.082/0.164/0.164 req/s（单块 12.2 s）。

---

## 6. 和 CPU 基线的对比（为什么必须上板）

Mac（M4 Pro）上 transformers 原生路径的基线（fp32，无 `causal_conv1d` / `flash-linear-attention` kernel）：

| 指标 | 短边 448 小图（1 块） | 整页 2000×3071（9 块） |
|---|---:|---:|
| prefill（vision + LLM 前向） | 6.98 s | **17.6 s** |
| decode | **0.17–0.20 tok/s**（≈5 s/token） | 未采信（路径异常） |
| RSS 峰值 | 6254 MB | 7343 MB |

CPU 慢的根因不是视觉塔、也不是 1.3B 参数，而是 18 层 GatedDeltaNet 的 depthwise conv 在缺 kernel 时
退化成 `F.conv1d(groups=6144)`：**183.7 ms/层**，18 层占单步 99.8%（≈5 s/token）。这两个 kernel 只发 CUDA/Triton。

→ 同样 237 token：CPU 约 20 分钟量级（5 s/token × 237），板端 **21.9 s**，相差两个数量级。
（CPU 侧是推算：0.17–0.20 tok/s 实测。）

---

## 7. 优化方向（按收益排序）

| # | 方向 | 预期 | 现状 |
|---|---|---|---|
| 1 | 限制输出长度 / 提前停止（表格识别 237 token 偏长） | decode 时间线性下降 | 服务侧 `max_tokens` 可调，未做自适应 |
| 2 | 应用侧按版面切小块、跳过空白片 | 减少无效片 | 已实现空白片跳过（ink_ratio < 0.003） |
| 3 | DSpark 解码加速（官方称 +55% 效率） | decode 提升 | **未覆盖**（官方另有 DSpark 变体） |
| 4 | vision 换 U16 后的耗时代价 | 2 层探针 41 → 87.5 ms（约 2×） | 单块总量 21.9 s 里 vision 只占 0.6 s，影响小 |
| 5 | 多请求并发/流水线 | 吞吐提升 | 未做（单实例串行） |

---

## 8. 待补充（后续优化，不影响交付）

1. v2 包（prefill 1152 / ctx 2048）的实测：启动日志、`max_token_len`、`prefill_max_token_num`、加载耗时。
2. **U16 vision 包** 的单块端到端与 TTFT（预期 vision 段从 ≈650 ms 翻倍到 ≈1.3 s，对 21.9 s 总量影响 <5%）。
3. 整页 9 块真实串行耗时（当前为推算）。
4. 并发 2/4 请求下的 P50/P95 延迟与吞吐。
5. 中文表格 / 密集小字页面的 token 长度分布（决定 `max_tokens` 默认值）。


## 流式（SSE）实测

| 场景 | 首字时间（TTFT） | 总耗时 |
|---|---|---|
| 表格示例（table 模式，233 token） | **2.1 s** | 21.7 s |
| 文本对话（32 token） | **0.56 s** | 2.8 s |
| 版面模式 PPT 页（6 区域） | 第 1 区 ~2.1 s | 19.5 s |

Bash 验证：`curl -N -X POST .../api/ocr_stream` 可看到逐段 `data:` 事件；浏览器 E2E（CDP）断言
流式期间 `#stream-pre` 字数递增、结束后渲染 `<table>` 且含 `83.17`、KPI `TTFT=2,115 ms`。
详见 [streaming.md](streaming.md)。

## 最终实测（2026-11-11，U16 vision 包）

板端：AX8850（8 核 aarch64 / 3.8 GB DDR / CMM 4096 MB），axllm 18081，`max_token_len=2047`，
`current prefill_max_token_num=1152`，`VisionModule init ok: tokens_per_block=256, out_dtype=fp32`。

| 项 | 实测 |
|---|---|
| vision axmodel 体积 / 单块 448×448 耗时 | 511 MB / **615 ms**（ax_run_model，单独测） |
| 视觉对拍（对 ONNX 参考） | cosine **0.995322** |
| 单块 504×392 表格 OCR（277 prompt → 119 token） | **12.1 s**，decode ≈ **9.8–11.8 tok/s** |
| 单块 448×448 表格 OCR（277 → 237 token） | **21.9 s**，decode **11.6 tok/s**（日志 11.10–11.76） |
| 整页切片（9 片，跳过全局缩略片） | **147 s/页**，1515 completion token |
| TTFT（含视觉编码） | 2.09–2.15 s |
| 文本 "1+1" / "一句话解释 OCR" | 0.6 s / 1.9 s（≈15.3 tok/s） |
| CMM（加载完成后） | 剩余 **2130 MB**（`/proc/ax_proc/mem_cmm_info` 当前占用 ≈2.5 GB） |
| 并发（文本，levels 1/2/4） | 0.55 / 0.56 / 0.75 req/s；均值 1.82 / 3.13 / 3.93 s（P95 5.32 s） |
| 并发（OCR 图片块，levels 1/2/4） | 0.082 / 0.164 / 0.164 req/s；单块 12.2 s（axllm 单并发，排队） |

> 结论：瓶颈在 **LLM decode（≈11.6 tok/s）**，vision 只占单块总量约 5%；整页耗时 = 片数 ×（视觉 0.6 s + LLM 解码）。
