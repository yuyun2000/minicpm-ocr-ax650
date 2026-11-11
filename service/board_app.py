#!/usr/bin/env python3
"""MiniCPM-OCR on AX650 —— 板端自托管服务（Flask，局域网可访问）。

- 静态页面：`web/`（M5Stack 风格前端）
- API：
  POST /api/ocr   multipart: image, prompt, mode=page|region, max_tokens
  POST /api/chat  json: {"prompt": str, "max_tokens": int}
  GET  /api/health
  GET  /api/slices?w=&h=   预演切片方案（不推理）
- 推理后端：本机 axllm OpenAI 兼容接口 http://127.0.0.1:18081/v1/chat/completions
- 整页模式：复刻 MiniCPM-V 切片（scale_resolution=448 / max_slice_nums=9 / patch=14 / merge=2），
  逐片 letterbox 到 448×448 送 NPU（板端 vision encoder 是固定 448 输入），再按阅读顺序拼接。
"""
import base64
import io
import json
import math
import os
import re
import time
import urllib.request

from flask import Flask, jsonify, request, send_from_directory
from PIL import Image
import numpy as np

from slicing import slice_image
from otsl_to_md import otsl_to_markdown

AXLLM = os.environ.get("AXLLM_URL", "http://127.0.0.1:18081")
MODEL = os.environ.get("AXLLM_MODEL", "")
WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
SCALE_RESOLUTION = 448
MAX_SLICE_NUMS = 9
PATCH = 14
MERGE = 2

app = Flask(__name__, static_folder=WEB_DIR, static_url_path="")


# ---------- 切片：直接用官方复刻模块 slicing.py（与 transformers processor 逐片对齐）----------
MAX_SLICES_PER_PAGE = int(os.environ.get("MAX_SLICES_PER_PAGE", "9"))


def make_slices(img: Image.Image) -> list[dict]:
    """官方切片（全局缩略 + 网格切片），转成服务端使用的 dict 列表。"""
    plan = slice_image(img, max_slice_nums=MAX_SLICES_PER_PAGE)
    out = []
    for s in plan.slices:
        out.append({"image": s.image, "tokens": s.num_tokens, "grid": (s.grid_h, s.grid_w),
                    "kind": "global" if s.is_global else f"slice-{s.index}"})
    return out


def ink_ratio(im: Image.Image) -> float:
    """非白像素比例，用于跳过空白片。"""
    a = np.asarray(im.convert("L"), dtype=np.uint8)
    return float((a < 240).mean())


def letterbox_448(im: Image.Image) -> tuple[Image.Image, float]:
    """等比缩放并白边填充到 448×448（板端 vision encoder 固定输入）。"""
    w, h = im.size
    s = min(SCALE_RESOLUTION / w, SCALE_RESOLUTION / h)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    canvas = Image.new("RGB", (SCALE_RESOLUTION, SCALE_RESOLUTION), (255, 255, 255))
    canvas.paste(im.resize((nw, nh), Image.BICUBIC), ((SCALE_RESOLUTION - nw) // 2, (SCALE_RESOLUTION - nh) // 2))
    return canvas, (nw * nh) / float(SCALE_RESOLUTION ** 2)


# ---------- 版面检测（DocLayout-YOLO, 1024x1024, 输出 raw [1,14,21504]）----------
LAYOUT_MODEL = os.environ.get("LAYOUT_MODEL", "")
LAYOUT_SIZE = 1024
LAYOUT_NAMES = ["title", "plain text", "abandon", "figure", "figure_caption",
                "table", "table_caption", "table_footnote", "isolate_formula", "formula_caption"]
AX_RUN_MODEL = os.environ.get("AX_RUN_MODEL", "/opt/bin/ax_run_model")


def layout_detect(img: Image.Image, conf_thr: float = 0.35, min_area_ratio: float = 0.002,
                  nms_iou: float = 0.5) -> list[dict]:
    """调用板端 ax_run_model 跑 DocLayout-YOLO（双输出：boxes + score logits），
    返回 [{label, conf, box(原图坐标)}]，按阅读顺序排序。"""
    if not os.path.exists(LAYOUT_MODEL):
        raise RuntimeError(f"版面模型不存在: {LAYOUT_MODEL}")
    w, h = img.size
    r = min(LAYOUT_SIZE / w, LAYOUT_SIZE / h)
    nw, nh = max(1, int(round(w * r))), max(1, int(round(h * r)))
    px, py = (LAYOUT_SIZE - nw) // 2, (LAYOUT_SIZE - nh) // 2
    canvas = Image.new("RGB", (LAYOUT_SIZE, LAYOUT_SIZE), (114, 114, 114))
    canvas.paste(img.resize((nw, nh), Image.BILINEAR), (px, py))
    arr = (np.asarray(canvas, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None]

    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory(prefix="layout_") as td:
        inp = os.path.join(td, "inputs", "run0")
        outd = os.path.join(td, "out")
        os.makedirs(inp); os.makedirs(outd)
        arr.tofile(os.path.join(inp, "images.bin"))
        with open(os.path.join(td, "list.txt"), "w") as f:
            f.write("run0\n")
        subprocess.run([AX_RUN_MODEL, "-m", LAYOUT_MODEL, "-i", os.path.join(td, "inputs"),
                        "-o", outd, "-l", os.path.join(td, "list.txt")],
                       check=True, capture_output=True, timeout=300)
        files = os.listdir(os.path.join(outd, "run0"))
        def pick(keyword):
            cand = [f for f in files if keyword in f]
            return os.path.join(outd, "run0", cand[0]) if cand else None
        box_file = pick("Mul_3") or pick("boxes")
        logit_file = pick("Split_1") or pick("logits")
        if not box_file or not logit_file:
            raise RuntimeError(f"版面模型输出不完整: {files}")
        boxes = np.fromfile(box_file, dtype=np.float32)
        logits = np.fromfile(logit_file, dtype=np.float32)

    n_anchor = boxes.size // 4
    pb = boxes.reshape(4, n_anchor).T
    lg = logits.reshape(10, n_anchor).T
    with np.errstate(over="ignore"):
        sc = 1.0 / (1.0 + np.exp(-lg.astype(np.float64)))
    conf = sc.max(axis=1)
    cls = sc.argmax(axis=1)

    min_area = min_area_ratio * w * h
    dets = []
    for i in np.where(conf >= conf_thr)[0]:
        x1, y1, x2, y2 = [float(v) for v in pb[i]]
        x1 = (x1 - px) / r; x2 = (x2 - px) / r
        y1 = (y1 - py) / r; y2 = (y2 - py) / r
        x1, x2 = max(0.0, min(x1, w)), max(0.0, min(x2, w))
        y1, y2 = max(0.0, min(y1, h)), max(0.0, min(y2, h))
        bw, bh = x2 - x1, y2 - y1
        if bw < 12 or bh < 12 or bw * bh < min_area:
            continue
        label = LAYOUT_NAMES[int(cls[i])] if int(cls[i]) < len(LAYOUT_NAMES) else str(int(cls[i]))
        dets.append({"label": label, "conf": float(conf[i]),
                     "box": [x1, y1, x2, y2]})

    # 同类别 NMS
    def iou(a, b):
        ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / ua if ua > 0 else 0.0

    dets.sort(key=lambda d: -d["conf"])
    kept = []
    for d in dets:
        if all(iou(d["box"], k["box"]) < nms_iou or d["label"] != k["label"] for k in kept):
            kept.append(d)

    for d in kept:
        d["conf"] = round(d["conf"], 4)
        d["box"] = [round(v, 1) for v in d["box"]]

    # 阅读顺序：先按 y 聚类成行（垂直重叠 > 40%），行内按 x
    kept.sort(key=lambda d: (d["box"][1], d["box"][0]))
    ordered, row = [], []
    for d in kept:
        if not row:
            row = [d]; continue
        ry1 = min(x["box"][1] for x in row); ry2 = max(x["box"][3] for x in row)
        y1, y2 = d["box"][1], d["box"][3]
        inter = max(0.0, min(ry2, y2) - max(ry1, y1))
        if inter > 0.4 * max(1e-6, min(ry2 - ry1, y2 - y1)):
            row.append(d)
        else:
            ordered.extend(sorted(row, key=lambda x: x["box"][0])); row = [d]
    if row:
        ordered.extend(sorted(row, key=lambda x: x["box"][0]))
    return ordered


LAYOUT_PROMPT = {
    "table": "Table Recognition:",
    "isolate_formula": "Formula Recognition:",
    "formula_caption": "Text Recognition:",
}


def ocr_layout(img: Image.Image, max_tokens: int) -> dict:
    """版面模式：DocLayout-YOLO 检测区域 → 按类型逐区域识别 → 按阅读顺序组装 markdown。

    单表格裁片场景：若只检出一个 table 区域且覆盖全图 >35%，直接整图按表格识别
    （避免检测框把左右列的表格内容裁掉）。
    """
    dets = layout_detect(img)
    w, h = img.size
    page_area = float(w * h)
    tables = [d for d in dets if d["label"] == "table"]
    if len(dets) == 1 and tables:
        x1, y1, x2, y2 = tables[0]["box"]
        if (x2 - x1) * (y2 - y1) > 0.35 * page_area:
            out = ocr_table(img, max_tokens)
            out["parts"] = [{"label": "table", "conf": round(tables[0]["conf"], 4),
                             "box": [0, 0, w, h],
                             "elapsed_s": sum(p.get("elapsed_s") or 0 for p in out["parts"]),
                             "note": "single-table-page: whole image"}]
            return out
    results, parts = [], []
    for d in dets:
        label = d["label"]
        if label in ("abandon", "figure"):
            continue
        x1, y1, x2, y2 = d["box"]
        pad = 4
        crop = img.crop((int(max(0, x1 - pad)), int(max(0, y1 - pad)),
                         int(min(img.width, x2 + pad)), int(min(img.height, y2 + pad))))
        if crop.width < 8 or crop.height < 8:
            continue
        prompt = LAYOUT_PROMPT.get(label, "Text Recognition:")
        block, _ = letterbox_448(crop)
        r = call_axllm(block, prompt, max_tokens)
        content = r["content"]
        parts.append({"label": label, "conf": d["conf"], "box": d["box"],
                      "elapsed_s": r["elapsed_s"], "tokens": r.get("completion_tokens")})
        if label == "table":
            try:
                md = otsl_to_markdown(content) or content
            except Exception:
                md = content
        else:
            md = content
        results.append(md)
    return {"parts": parts, "markdown": "\n\n".join(results)}


# ---------- 表格模式：整表识别 / 按行切分后合并 OTSL ----------
TABLE_PROMPT = "Table Recognition:"


def rule_score(img: Image.Image) -> tuple[int, int, float]:
    """统计（缩放到 448 后）的水平/垂直长线条数量与墨迹比例，用于表格启发式。

    阈值 235 能覆盖浅灰表格边框（截图里常见 #e5e5e5），同时正文页的竖线统计很低。
    """
    im = img.convert("L")
    g = np.asarray(im.resize((min(im.width, 448), min(im.height, 448)), Image.BILINEAR), dtype=np.uint8)
    dark = g < 235
    h_rules = int((dark.mean(axis=1) > 0.5).sum())
    v_rules = int((dark.mean(axis=0) > 0.5).sum())
    return h_rules, v_rules, float(dark.mean())


def looks_like_table(img: Image.Image) -> bool:
    """表格启发式（实测：表格截图 v_rules 15~90；正文页 v_rules ~10）。

    - 表格：竖线多（列分隔）且至少几条横线；
    - 正文页：横线多但竖线少。
    """
    h_rules, v_rules, ink = rule_score(img)
    if ink > 0.6:  # 整页大图/照片，不按表格走
        return False
    if v_rules >= 15 and h_rules >= 4:
        return True
    if h_rules >= 10 and v_rules >= 6:
        return True
    return False


def _parse_table_rows(text: str):
    """把 OTSL / markdown 表格转成 rows(list[list[str]])，失败返回 None。"""
    md = text
    if "<otsl>" in text:
        try:
            md = otsl_to_markdown(text) or text
        except Exception:
            md = text
    lines = [ln.strip() for ln in md.splitlines() if ln.strip().startswith("|")]
    if not lines:
        return None
    rows = []
    for ln in lines:
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if cells and all(re.fullmatch(r"-{2,}", c) for c in cells if c):
            continue
        rows.append(cells)
    return rows or None


def _rows_to_markdown(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    norm = [r + [""] * (width - len(r)) for r in rows]
    out = ["| " + " | ".join(norm[0]) + " |", "| " + " | ".join(["---"] * width) + " |"]
    for r in norm[1:]:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def _merge_rows(tables: list[list[list[str]]]) -> list[list[str]]:
    """按行切分的多张表垂直合并：去掉重叠造成的连续重复行。"""
    merged: list[list[str]] = []
    for rows in tables:
        for r in rows:
            if merged and r == merged[-1]:
                continue
            merged.append(r)
    return merged


def ocr_table(img: Image.Image, max_tokens: int) -> dict:
    """整表识别：小表一次送；大表按行切（保留全部列）后按行合并，避免把表切碎成列。"""
    w, h = img.size
    parts = []
    contents = []
    if max(w, h) <= 2048:
        block, _ = letterbox_448(img)
        r = call_axllm(block, TABLE_PROMPT, max_tokens)
        contents.append(r["content"])
        parts.append({"kind": "table-full", "size": (w, h), "elapsed_s": r["elapsed_s"],
                      "tokens": r.get("completion_tokens")})
    else:
        # 按行切：每片宽度保持原宽（缩放后仍含全部列），只沿垂直方向切
        scale = 448.0 / w
        strip_h = max(64, int(448.0 / scale)) if scale > 0 else h
        n = max(2, math.ceil(h / max(strip_h, 1)))
        overlap = max(8, int(strip_h * 0.06))
        step = max(1, (h - overlap) // n if n > 1 else h)
        y = 0
        idx = 0
        while y < h:
            y1 = min(h, y + strip_h + overlap)
            crop = img.crop((0, y, w, y1))
            block, _ = letterbox_448(crop)
            r = call_axllm(block, TABLE_PROMPT, max_tokens)
            contents.append(r["content"])
            parts.append({"kind": f"table-row-{idx}", "y": (y, y1), "elapsed_s": r["elapsed_s"],
                          "tokens": r.get("completion_tokens")})
            idx += 1
            if y1 >= h:
                break
            y += step

    if len(contents) == 1:
        try:
            markdown = otsl_to_markdown(contents[0]) or contents[0]
        except Exception:
            markdown = contents[0]
    else:
        tables = [rows for rows in (_parse_table_rows(c) for c in contents) if rows]
        markdown = _rows_to_markdown(_merge_rows(tables)) if tables else "\n\n".join(contents)
    return {"contents": contents, "parts": parts, "markdown": markdown}


# ---------- 宽图（长条文字）分块识别：左右切块后按阅读顺序拼接 ----------
def ocr_wide(img: Image.Image, prompt: str, max_tokens: int) -> dict:
    w, h = img.size
    n = max(2, math.ceil(w / 1000.0))
    step = max(1, w // n)
    overlap = max(20, int(step * 0.06))
    contents, parts = [], []
    x = 0
    idx = 0
    while x < w:
        x1 = min(w, x + step + overlap)
        crop = img.crop((x, 0, x1, h))
        block, _ = letterbox_448(crop)
        r = call_axllm(block, prompt, max_tokens)
        contents.append(r["content"])
        parts.append({"kind": f"wide-{idx}", "x": (x, x1), "elapsed_s": r["elapsed_s"],
                      "tokens": r.get("completion_tokens")})
        idx += 1
        if x1 >= w:
            break
        x += step
    return {"contents": contents, "parts": parts, "markdown": "\n".join(contents)}


# ---------- 推理后端 ----------
def board_model() -> str:
    global MODEL
    if MODEL:
        return MODEL
    try:
        with urllib.request.urlopen(f"{AXLLM}/v1/models", timeout=10) as r:
            MODEL = json.load(r)["data"][0]["id"]
    except Exception:
        MODEL = "minicpm-ocr"
    return MODEL


def call_axllm(image: Image.Image | None, prompt: str, max_tokens: int) -> dict:
    content: list[dict] = [{"type": "text", "text": prompt}]
    if image is not None:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    payload = {"model": board_model(), "messages": [{"role": "user", "content": content if image is not None else prompt}],
               "max_tokens": max_tokens, "temperature": 0.0}
    req = urllib.request.Request(f"{AXLLM}/v1/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as resp:
        data = json.load(resp)
    dt = time.time() - t0
    usage = data.get("usage") or {}
    comp = usage.get("completion_tokens") or 0
    return {"content": (data["choices"][0]["message"]["content"] or ""),
            "elapsed_s": round(dt, 3),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": comp,
            "decode_tok_s": round(comp / dt, 2) if comp and dt else None}


# ---------- 路由 ----------
@app.get("/api/health")
def health():
    try:
        with urllib.request.urlopen(f"{AXLLM}/v1/models", timeout=10) as r:
            return jsonify({"ok": True, "board": json.load(r),
                            "layout_model": os.path.basename(LAYOUT_MODEL) if os.path.exists(LAYOUT_MODEL) else None})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": str(exc)})


@app.post("/api/ocr")
def api_ocr():
    f = request.files.get("image")
    if f is None:
        return jsonify({"error": "缺少 image"})
    prompt = request.form.get("prompt", "")
    mode = request.form.get("mode", "auto")
    max_tokens = int(request.form.get("max_tokens", 768))
    img = Image.open(io.BytesIO(f.read())).convert("RGB")
    t0 = time.time()
    if mode == "auto":
        w, h = img.size
        if looks_like_table(img) and not (h >= 1.2 * w and max(w, h) >= 1800):
            mode = "table"
        elif os.path.exists(LAYOUT_MODEL) and ((h >= 1.2 * w and max(w, h) >= 1200) or max(w, h) >= 2200):
            mode = "layout"
        elif (h >= 1.2 * w and max(w, h) >= 1200) or max(w, h) >= 2200:
            mode = "page"
        elif w >= 2.5 * h and w >= 1200:
            mode = "wide"
        else:
            mode = "region"
    if mode == "layout" and os.path.exists(LAYOUT_MODEL):
        try:
            out = ocr_layout(img, max_tokens)
            elapsed = round(time.time() - t0, 3)
            comp = sum(p.get("tokens") or 0 for p in out["parts"])
            metrics = {"mode": "layout", "slices": len(out["parts"]), "parts": out["parts"],
                       "elapsed_s": elapsed, "completion_tokens": comp,
                       "decode_tok_s": round(comp / elapsed, 2) if comp and elapsed else None,
                       "ttft_ms": int((out["parts"][0]["elapsed_s"]) * 1000) if out["parts"] else None}
            return jsonify({"content": out["markdown"], "markdown": out["markdown"],
                            "metrics": metrics, "error": None})
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": f"{type(exc).__name__}: {exc}"})
    if mode == "table":
        try:
            out = ocr_table(img, max_tokens)
            elapsed = round(time.time() - t0, 3)
            comp = sum(p.get("tokens") or 0 for p in out["parts"])
            metrics = {"mode": "table", "slices": len(out["parts"]), "parts": out["parts"],
                       "elapsed_s": elapsed, "completion_tokens": comp,
                       "decode_tok_s": round(comp / elapsed, 2) if comp and elapsed else None,
                       "ttft_ms": int((out["parts"][0]["elapsed_s"]) * 1000) if out["parts"] else None}
            return jsonify({"content": "\n\n".join(out["contents"]), "markdown": out["markdown"],
                            "metrics": metrics, "error": None})
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": f"{type(exc).__name__}: {exc}"})
    if not prompt:
        prompt = "Text Recognition:"
    try:
        if mode == "wide":
            out = ocr_wide(img, prompt, max_tokens)
            elapsed = round(time.time() - t0, 3)
            comp = sum(p.get("tokens") or 0 for p in out["parts"])
            metrics = {"mode": "wide", "slices": len(out["parts"]), "parts": out["parts"],
                       "elapsed_s": elapsed, "completion_tokens": comp,
                       "decode_tok_s": round(comp / elapsed, 2) if comp and elapsed else None,
                       "ttft_ms": int(out["parts"][0]["elapsed_s"] * 1000) if out["parts"] else None}
            return jsonify({"content": "\n".join(out["contents"]), "markdown": out["markdown"],
                            "metrics": metrics, "error": None})
        if mode == "page":
            slices = make_slices(img)
            skip_global = request.form.get("skip_global", "1") != "0"
            results, slice_times, slice_info = [], [], []
            for s in slices:
                if skip_global and s["kind"] == "global":
                    slice_info.append({"kind": s["kind"], "tokens": s["tokens"], "skipped": "global"})
                    continue
                ratio = ink_ratio(s["image"])
                if ratio < 0.003:  # 空白片跳过，避免模型退化重复
                    slice_info.append({"kind": s["kind"], "tokens": s["tokens"],
                                       "ink": round(ratio, 5), "skipped": True})
                    continue
                boxed, _ = letterbox_448(s["image"])
                r = call_axllm(boxed, prompt, max_tokens)
                r["kind"] = s["kind"]
                r["ink"] = round(ratio, 5)
                r["tokens_planned"] = s["tokens"]
                results.append(r)
                slice_times.append(int(round(r["elapsed_s"] * 1000)))
                slice_info.append({"kind": s["kind"], "tokens": s["tokens"],
                                   "ink": round(ratio, 5), "skipped": False,
                                   "elapsed_s": r["elapsed_s"]})
            content = "\n\n".join(r["content"] for r in results)
            md_parts = []
            for r in results:
                try:
                    md = otsl_to_markdown(r["content"])
                except Exception:
                    md = None
                md_parts.append(md if md else r["content"])
            markdown = "\n\n".join(md_parts)
            comp = sum(r["completion_tokens"] or 0 for r in results)
            elapsed = round(time.time() - t0, 3)
            metrics = {"mode": "page", "slices": len(slices), "used_slices": len(results),
                       "slice_times": slice_times, "slice_time_unit": "ms", "slice_info": slice_info,
                       "elapsed_s": elapsed, "completion_tokens": comp, "prompt_tokens": None,
                       "decode_tok_s": round(comp / elapsed, 2) if comp and elapsed else None,
                       "ttft_ms": int(slice_times[0]) if slice_times else None}
        else:
            boxed, _ = letterbox_448(img)
            r = call_axllm(boxed, prompt, max_tokens)
            content = r["content"]
            try:
                markdown = otsl_to_markdown(content) or content
            except Exception:
                markdown = content
            metrics = {"mode": "region", "slices": 1, "elapsed_s": r["elapsed_s"],
                       "prompt_tokens": r["prompt_tokens"], "completion_tokens": r["completion_tokens"],
                       "decode_tok_s": r["decode_tok_s"], "ttft_ms": int(r["elapsed_s"] * 1000)}
        return jsonify({"content": content, "markdown": markdown, "metrics": metrics, "error": None})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"{type(exc).__name__}: {exc}"})


@app.post("/api/chat")
def api_chat():
    body = request.get_json(silent=True) or {}
    prompt = body.get("prompt", "")
    max_tokens = int(body.get("max_tokens", 256))
    try:
        r = call_axllm(None, prompt, max_tokens)
        return jsonify({"content": r["content"], "metrics": r, "error": None})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"{type(exc).__name__}: {exc}"})


@app.get("/api/slices")
def api_slices():
    """预览整页切片方案（不推理）。"""
    try:
        w = int(request.args.get("w", 2000))
        h = int(request.args.get("h", 3000))
    except ValueError:
        return jsonify({"error": "w/h 必须是整数"})
    img = Image.new("RGB", (max(w, 1), max(h, 1)), (255, 255, 255))
    plan = slice_image(img, max_slice_nums=MAX_SLICES_PER_PAGE)
    return jsonify({"width": w, "height": h, "grid": list(plan.grid) if getattr(plan, "grid", None) else None,
                    "count": len(plan.slices), "total_tokens": plan.total_tokens if hasattr(plan, "total_tokens") else None,
                    "slices": [s.summary() if hasattr(s, "summary") else {"index": s.index,
                               "tokens": s.num_tokens, "size": s.size} for s in plan.slices]})


@app.get("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")), threaded=True)
