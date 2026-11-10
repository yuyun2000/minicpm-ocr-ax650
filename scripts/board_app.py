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
            return jsonify({"ok": True, "board": json.load(r)})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": str(exc)})


@app.post("/api/ocr")
def api_ocr():
    f = request.files.get("image")
    if f is None:
        return jsonify({"error": "缺少 image"})
    prompt = request.form.get("prompt", "Text Recognition:")
    mode = request.form.get("mode", "region")
    max_tokens = int(request.form.get("max_tokens", 768))
    img = Image.open(io.BytesIO(f.read())).convert("RGB")
    t0 = time.time()
    try:
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
