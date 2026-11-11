# 流式输出（逐字显示）实现与实测（2026-11-11）

## 背景

用户反馈：示例图 OCR 要等 20 多秒才一次性出结果，感觉慢。axllm 本身支持 OpenAI 兼容的
`"stream": true`（SSE），板端实测文本首 token 0.56 s、26 个 chunk 2.84 s 内输出完。

## 后端（`service/board_app.py`）

新增两个 SSE 端点（不改变原有 `/api/ocr`、`/api/chat` 行为）：

- `POST /api/chat_stream`：`meta` → `delta*` → `done`
- `POST /api/ocr_stream`：`meta`（模式/块数）→ 每块 `part_start` → `delta*` → `part_done` →
  `done`（含完整 content、markdown、metrics）

实现要点：
- `call_axllm_stream()` 解析 axllm SSE，逐段 yield (`delta` / `done`)；
- `resolve_mode()` / `plan_blocks()` 把 auto/table/layout/page/wide/region 统一拆成「识别块」，
  流式与同步共用同一套模式判定，避免两套逻辑漂移；
- `assemble_markdown()` 在 `done` 时组装最终 Markdown（表格模式按行合并 OTSL）；
- Flask `Response(stream_with_context(gen()), mimetype="text/event-stream")` + `X-Accel-Buffering: no`。

## 前端（`service/web/`）

- 图片/对话面板各加「流式输出」开关（默认开）；
- 结果区先渲染 `<pre class="stream-live">` 逐字追加，带光标动画与「首字 X s · N 字」状态；
- 流式中实时更新 KPI（TTFT、总耗时），结束渲染 Markdown 表格 + 完整指标；
- 流式失败（浏览器不支持/网络中断）自动回退到非流式接口；
- 新增「快速示例」按钮（表格示例 / 文档页示例，同源静态图），点一下即载入并开始识别。

## 实测（板端，Chrome headless E2E + 真实请求）

| 场景 | 首字时间（TTFT） | 总耗时 |
|---|---|---|
| 表格示例（1100×986，table 模式，233 token） | **2.1 s** | 21.7 s |
| 文本对话（"用一句话解释 OCR"，32 token） | **0.56 s** | 2.84 s |
| 版面模式 PPT 页（6 区域） | 第 1 区 ~2.1 s，之后逐区域出现 | 19.5 s |

E2E 断言（CDP 驱动真实浏览器）：流式期间 `#stream-pre` 字数从 0 递增到 596；结束后
`markdown-out` 渲染出 1 个 `<table>`、含 `83.17`，KPI `TTFT=2,115 ms`、`总耗时=21.70 s`、状态「完成」。

截图：`docs/images/streaming.png`（流式中）、`docs/images/result-table.png`（完成后的表格）。
