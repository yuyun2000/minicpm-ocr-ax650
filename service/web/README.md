# MiniCPM-OCR 板端测试台 · 前端

AX8850 板端（Ubuntu 22.04 aarch64）自托管的纯静态网页测试台，用于上传文档图片 / 输入纯文本，
调用板端 OCR 推理服务，查看 **Markdown 渲染结果** 与 **TTFT / decode / 切片** 等性能指标。

识别模式（页面「识别模式」单选）：`自动（推荐）` / `表格` / `版面` / `整页` / `单区域`。
- 表格截图选 `自动` 或 `表格`：整表识别后输出 Markdown 表格（整页模式会把表格沿列切开导致乱序）；
- 混合版面文档选 `版面`：DocLayout-YOLO 检测区域后按类型识别、按阅读顺序拼接（需服务端设置 `LAYOUT_MODEL`）；
- `表格`/`版面` 模式会忽略上方 prompt，按区域类型自动选 `Table Recognition:` / `Formula Recognition:` / `Text Recognition:`。

视觉严格按 **M5Stack Web Design** 规范实现（产品密度 · 内部工具页配方 B）。
零依赖、无构建步骤、无 CDN；只用原生 HTML / CSS / JS，由板端 Flask 直接托管。

```
demo/minicpm-ocr/web/
├── index.html    页面结构（M5Stack 组件命名 m5-*，语义化标签 + ARIA）
├── styles.css    设计 token + 组件样式 + 页面布局 + 暗色扩展（单文件，无需构建）
├── app.js        上传/请求/渲染/指标/主题/健康检查（IIFE，无全局污染）
├── README.md     本文件
└── preview.png   页面截图（文档用，运行时不加载）
```

---

## 1. 接口约定

所有请求都走**同源相对路径**，前端不硬编码主机名和端口；页面从哪个地址打开，就请求哪个地址的 `/api/*`。

### 1.1 流式接口（SSE，页面默认使用）

页面「流式输出」开关默认开启时走流式；失败自动回退到下面的非流式接口。

- `POST /api/ocr_stream`（multipart，字段同 `/api/ocr`）→ `text/event-stream`：
  - `{"type":"meta","mode":"table","blocks":1,...}`
  - `{"type":"part_start","index":0,"label":"table","total":1}`
  - `{"type":"delta","index":0,"text":"<"}`
  - `{"type":"part_done","index":0,"content":"<otsl>...","elapsed_s":21.6,"ttft_s":2.1}`
  - `{"type":"done","content":"...","markdown":"| ... |","metrics":{...}}`
  - 出错：`{"type":"error","error":"..."}`
- `POST /api/chat_stream`（JSON `{prompt,max_tokens}`）→ 同样的 `meta/delta/done` 事件。

### 1.2 `POST /api/ocr` — 图片识别（非流式）

`Content-Type: multipart/form-data`

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `image` | file | ✅ | 图片文件（PNG / JPEG / WebP） |
| `prompt` | string | ✅ | 任务词：`Text Recognition:`、`Table Recognition:`、`Formula Recognition:`、`Seal Recognition:`、`Chart to Table:`，或自定义 prompt |
| `mode` | string | ✅ | `auto`（默认，自动选择）/ `table`（表格整表识别）/ `layout`（DocLayout 版面检测）/ `page`（整页切片）/ `wide`（宽条分块）/ `region`（单块） |
| `max_tokens` | int | ✅ | 输出上限，前端限制在 16–4096 |

响应 `200 application/json`：

```json
{
  "content": "<raw model output>",
  "markdown": "<可选，服务端已转好的 markdown>",
  "metrics": {
    "ttft_ms": 420.5,
    "decode_tok_s": 18.43,
    "elapsed_s": 3.421,
    "prompt_tokens": 1024,
    "completion_tokens": 96,
    "slices": 3,
    "slice_times": [512.4, 433.1, 389.7]
  },
  "error": null
}
```

| 字段 | 说明 |
| --- | --- |
| `content` | 模型原始输出（可能是 OTSL、Markdown 或纯文本）。**始终原样展示**在「原始输出」折叠区 |
| `markdown` | 可选。有值就直接渲染成 HTML（表格用 `<table>`），前端不重复实现 OTSL 解析 |
| `metrics.ttft_ms` | 首 token 延迟，毫秒 |
| `metrics.decode_tok_s` | 解码吞吐，token/秒 |
| `metrics.elapsed_s` | 端到端总耗时，秒 |
| `metrics.prompt_tokens` / `completion_tokens` | 提示 / 生成 token 数 |
| `metrics.slices` | 切片数（`region` 模式应为 0 或省略） |
| `metrics.slice_times` | 每片耗时数组，**按毫秒**解释（见 §5 注 1）；顺序与切片顺序一致 |
| `error` | 失败时给字符串；非空则前端按错误态展示 |

### 1.2 `POST /api/chat` — 纯文本对话（无图）

`Content-Type: application/json`

```json
{ "prompt": "用一句话解释什么是 OTSL 表格表示。", "max_tokens": 512 }
```

响应：

```json
{ "content": "...", "metrics": { "ttft_ms": 88.2, "decode_tok_s": 21.7, "elapsed_s": 1.204,
  "prompt_tokens": 12, "completion_tokens": 26 } }
```

### 1.3 `GET /api/health` — 健康检查

```json
{ "ok": true, "board": { "url": "http://192.168.10.145:18081", "model": "minicpm-ocr" } }
```

`board` 的内容前端不假设固定结构，按以下顺序尽量展示：

- 有 `data: [{id|model}]` → 取模型 id 列表（兼容直接透传板端 `/v1/models`）；
- 有 `model` → 直接显示；
- 否则截断 JSON 显示。

页面加载时检查一次，之后由「刷新」按钮手动触发（**不轮询**，避免给板端制造无谓负载）。

---

## 2. 在板端 Flask 里挂载

把 `web/` 作为静态目录即可，不需要模板引擎、不需要构建。

```python
import os
from flask import Flask, jsonify, request, send_from_directory

WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")

# static_url_path="" 是关键：index.html 里用的是 styles.css / app.js 这类相对路径
app = Flask(__name__, static_folder=WEB, static_url_path="")


@app.get("/")
def index():
    return send_from_directory(WEB, "index.html")


@app.post("/api/ocr")
def api_ocr():
    ...  # 读 request.files["image"]、request.form["prompt"|"mode"|"max_tokens"]
    return jsonify({...})  # 结构见 §1.1


@app.post("/api/chat")
def api_chat():
    ...  # request.get_json()["prompt"|"max_tokens"]
    return jsonify({...})


@app.get("/api/health")
def api_health():
    return jsonify({"ok": True, "board": {"model": "minicpm-ocr"}})
```

启动：`python3 web_app.py --host 0.0.0.0 --port 8080`，浏览器访问 `http://<板子 IP>:8080/`。

**注意**

- `static_url_path=""` 不要写成默认的 `/static`，否则 `styles.css` 会 404（页面无样式但仍有内容）。
- 同一个 Flask 实例里 `/api/*` 与静态路由不冲突；静态目录里没有 `api/` 子目录就不会抢占。
- 前后端同源，**不需要 CORS**。如果确实要把页面挪到别的机器，请在同源侧加反向代理，而不是在前端写死跨域地址。
- 大图上传：Flask 默认 `MAX_CONTENT_LENGTH` 为无限制，但板端内存紧张，建议显式设成 20 MB 并在超限时返回 `{"error": "..."}`，前端会把 `error` 原样显示出来。

---

## 3. 页面功能

| 区域 | 功能 |
| --- | --- |
| 上传 | 点击选择 / 拖拽 / `Ctrl+V` 粘贴；缩略图 + 文件名 + 体积 + 像素尺寸；可移除 |
| 任务类型 | 五个官方任务词 + 「自定义 prompt」（选中后显示输入框，并实时回显"将发送：…"） |
| 识别模式 | 整页模式（服务端切片）/ 单区域模式；提示文案随切换变化 |
| 结果 | `markdown` 存在 → 渲染 HTML（表格 → `<table class="m5-table">`）；`content` 原样进「原始输出」折叠区；JSON 输出自动格式化 |
| 指标 | TTFT、Decode tok/s、总耗时、Prompt/Completion tokens、切片数；缺字段显示 `—` 并在右侧说明缺了哪几项 |
| 切片 | 整页模式下显示每片耗时 chip、片数、切片合计与最慢一片；单区域 / 文本对话自动隐藏 |
| 文本对话 | 独立标签页，走 `/api/chat`，`Ctrl/Cmd+Enter` 发送 |
| 状态 | 空态 / 加载态（骨架 + 实时计时）/ 错误态 / 取消（AbortController，长任务可中断） |
| 复制 | 「复制原始输出」，`navigator.clipboard` 不可用时回退 `execCommand` |
| 主题 | 默认跟随系统（`prefers-color-scheme`），头部按钮在 跟随系统 → 浅色 → 深色 间循环，写 `localStorage` |

**渲染边界**（有意为之，避免前端和服务端两套口径）：

1. **OTSL 不在前端解析**。有 `markdown` 字段就渲染；没有则提示「原始 OTSL 在下方」，不做半吊子转换。
2. Markdown 渲染器是**最小子集**：标题、表格、列表、引用、围栏代码块、分隔线、粗体/斜体/行内代码/链接；**不渲染公式**（`$...$` 原样保留），不处理 HTML 内联标签。
3. 模型输出一律先 HTML 转义再套 Markdown，避免模型吐出的 `<script>` 被执行。
4. 段落内保留换行（`<br>`），更贴近 OCR 原文的逐行结构。

---

## 4. M5Stack 设计规范落实

基线：m5stack-web-design skill（m5stack.com 源码快照 `dev@f63985c`，核对 2026-10-22）。

### 4.1 Token（全部取自 skill，未自创）

| 类别 | 取值 | 落地位置 |
| --- | --- | --- |
| 品牌色 | `#0282CB`（M5 Blue）、`#000000`（页脚） | 主色只用于按钮/选中/链接/图标；纯黑只出现在页脚 |
| 主按钮 | 底色 `#0077C8`，hover `#016FAD`，按下 `#0064C2` | skill 03 §3 的可访问性修正：14px 白字用 `#0077C8`（4.70:1）而不是 `#0282CB`（4.15:1）；hover 变暗而不是变亮 |
| 文字蓝 | `#0064C2`（亮）/ `#4DA9DC`（暗） | 链接、小号蓝字用深一档蓝，满足 4.5:1 |
| 正文 / 次级 / 辅助 | `#434343` / `#555555` / `#707070` | 正文、说明、表头 |
| 描边 / 弱底 | `#E5E5E5` / `#F5F7F6` | 卡片 1px 描边、分区底、表头 |
| 字体 | `Montserrat, "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", sans-serif` | 全站；**不加载 web font**（板端离线），靠本机已有字体，中文不会掉到 DejaVu |
| 等宽 | 系统等宽栈 | 原始输出、代码块、接口路径 |
| 圆角 | 4 / 6 / 10 / 16 / 999 px | 输入 4，chip 6，KPI 10，面板 16，按钮胶囊 999 |
| 间距 | 4 / 8 / 12 / 16 / 24 / 32 / 48 / 64 | 全部走 `--m5-space-*`，无 13/18/26 这类散值 |
| 阴影 | 默认只有 1px 描边；hover 才 `0 4px 8px rgba(0,0,0,.1)` | 卡片、面板（描边优先、阴影克制） |
| 动效 | `300ms ease` 为主，100/250/380ms 备用 | hover 只过渡颜色/描边；无滚动浮现、无弹跳 |

### 4.2 组件与布局

- **密度**：product（工具页），正文 16px、辅助 13px、容器 1280px、头部 56px —— 按 05 §4 与配方 B，不是营销页的 22px 正文。
- **头部**：56px sticky、白底、1px 下边线；导航项 hover/激活出现 **3px 蓝色下边线**；logo 用占位方块（品牌红线：非官方站点不复用官网 logo 文件）；页脚黑底并声明「非 m5stack.com 官方站点」。
- **按钮**：胶囊、一屏只有一个主按钮（`开始识别` / `发送对话` 是同一个视觉主操作，随标签页切换），loading 时按钮内旋转环 + `aria-busy`，取消是 ghost 次操作。
- **表单**：填充式输入（`#EEEEEE` 底、4px 圆角），hover 变 `#E9E9E9`，focus 白底 + 1px 蓝边；`label` 用 `for` 绑定，错误集中在结果区 `role="alert"`。
- **表格**：表头浅灰底、无竖线、1px 行分隔、数字 `tabular-nums` 右对齐；窄屏外层 `overflow-x:auto`。
- **指标卡**：`.m5-kpi` 组件，数值 28px + `tabular-nums`，桌面 3×2、平板与手机 2 列。
- **空态**：图形 → 一句「为什么空」→ 一句「下一步」，不是只写"暂无数据"。
- **焦点**：全站 `:focus-visible` 2px 蓝描边 + 2px 偏移（实测 `outline: 2px solid #0064C2`）；Tab 顺序与视觉顺序一致。
- **触摸目标**：窄屏按钮 ≥ 44px，分段控件 ≥ 36px。
- **减少动效**：保留 skill 的 `prefers-reduced-motion` 全局降级。

### 4.3 深浅色

官网本身**没有**暗色模式，skill 03 §4 把暗色定义为「扩展」并要求显式开关。本页按需求做成**跟随系统**，同时满足扩展要求：

- 默认 `auto`：内联脚本在首帧前读 `prefers-color-scheme` 并写 `data-theme`，避免闪白；系统切换时实时跟随。
- 头部有显式开关（跟随系统 / 浅色 / 深色），选择写 `localStorage`。
- 暗色取值即 skill 03 §4 给定值：底 `#1A1A1A`、表面 `#242424`、描边 `#333333`、正文 `#E5E5E5`、次级 `#A0A0A0`、文字蓝提到 `#4DA9DC`。
- 页脚在暗色下仍是纯黑（品牌唯一的大面积黑），与 `#1A1A1A` 页面底可区分。

---

## 5. 已知边界与设计决定

1. **`slice_times` 单位按毫秒处理**。接口约定只给了数组，没有单位；若板端实际返回秒，页面会把数值当 ms 显示（数量级会明显不对）。届时改 `app.js` 的 `sliceLabel()` 一行即可，或在 §1.1 的响应里补 `slice_times_unit` 字段后由前端读取。
2. **不加载 web font**：板端离线，Montserrat 未随包分发；字体栈完整保留，装了则生效，没装则回退到系统无衬线 + 系统中文字体。想完全一致可把 woff2 放进本目录并加 `@font-face`（会增加体积）。
3. **Markdown 是最小子集**，只服务「没有现成 HTML 时的降级阅读」；表格、代码、列表覆盖 OCR 场景，公式不渲染。
4. **OTSL 解析在后端**。前端保留原文 + 提示，避免与 `otsl_to_md.py` 出现两套不一致的实现。
5. **健康检查不轮询**，只有加载时一次 + 手动刷新。
6. **剪贴板**在非 HTTPS 的非 localhost 源上可能不可用，已内置 `execCommand` 回退并在失败时明确提示。
7. **字体/颜色审计**：`audit_tokens.py --all` 会报出暗色扩展色（`#1A1A1A`/`#242424`/`#333333`/`#4DA9DC` 等）、skill 03 的按钮修正色 `#0077C8`、以及来自 `components.css` 的 `#E9E9E9`、shadow 里的 `#818181`——这些都是**有意使用**的 skill 扩展值或官网原值，不是漂移。另有两类误报：`0.8s` 旋转环是 skill 自带的加载动画；`img { max-width: 100% }` 在 `styles.css` 里（审计脚本只在单文件内匹配）。

## 6. 本地预览（不接板端）

```bash
cd demo/minicpm-ocr/web && python3 -m http.server 8000
# 打开 http://127.0.0.1:8000/ —— 布局/主题可用，但 /api/* 会 404：
# 健康标签显示「离线」、提交后结果区给出明确错误。这本身就是错误态的自检。
```

需要联调接口时用 `/tmp` 起的 mock 服务或直接在板端跑 Flask。

## 7. 调试钩子

浏览器控制台可用 `window.__MOC__`（只暴露纯函数，不含运行时状态）：

```js
__MOC__.renderMarkdown('| a | b |\n| --- | ---: |\n| 1 | 2 |')  // 表格 → <table>
__MOC__.normalizeMetrics({ usage: { prompt_tokens: 10 } })       // 旧字段容错
__MOC__.escapeHtml('<script>')                                    // 转义
```

**旧字段容错**：前端还兼容 `metrics.usage.prompt_tokens`、`metrics.usage.completion_tokens`、`metrics.elapsed_s` 缺省等历史写法，
以便板端后端在重构期间不会把页面搞挂；新字段（`ttft_ms`、`slice_times`）优先。

## 8. 浏览器支持

Chrome / Edge / Safari 近两年版本（用到 `fetch`、`AbortController`、`FormData`、CSS 自定义属性、
`prefers-color-scheme`、Flex/Grid）。板端局域网内常见浏览器都能打开，不需要 polyfill。
