#!/usr/bin/env python3
"""OmniDocBench 子集评测：用板端 MiniCPM-OCR 服务跑页面渲染 OCR，计算文本编辑距离。

两种模式：
- page  : 我们的服务端整页切片（grid 切片，代表"无版面模型"的现状）
- oracle: 用 OmniDocBench 的 GT 版面框逐区域裁剪+识别（代表"有理想版面模型"的模型上限）

指标：归一化文本编辑距离（1 - 字符级相似度，越低越好）+ 表格单元格 F1（可选）。

用法：
  python demo/minicpm-ocr/eval_omnidoc.py --data /tmp/omnidoc --mode page --limit 5
  python demo/minicpm-ocr/eval_omnidoc.py --data /tmp/omnidoc --mode oracle --limit 5
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

import difflib
from PIL import Image

BOARD = os.environ.get("BOARD_API", "http://192.168.10.145:8080")

# OmniDocBench 中参与"正文"GT 的类别（不含 header/footer/page_number/图注等可选类别时通过 --keep-all 调整）
TEXT_CATS = {"title", "text_block", "list_group", "code_txt", "table_caption",
             "equation_caption", "figure_caption", "page_footnote", "table_footnote", "reference"}
TABLE_CATS = {"table"}
EQ_CATS = {"equation_isolated", "equation_inline"}
PROMPT = {
    "table": "Table Recognition:",
    "equation_isolated": "Formula Recognition:",
    "equation_inline": "Formula Recognition:",
}


def post_ocr(image: Image.Image, prompt: str, mode: str = "region", max_tokens: int = 1024) -> dict:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    files = {
        "image": ("page.png", buf.getvalue(), "image/png"),
    }
    data = {"prompt": prompt, "mode": mode, "max_tokens": str(max_tokens)}
    # multipart 手写
    boundary = "----evalboundary"
    parts = []
    for k, v in data.items():
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
    parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"page.png\"\r\n"
                 f"Content-Type: image/png\r\n\r\n".encode() + buf.getvalue() + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)
    req = urllib.request.Request(f"{BOARD}/api/ocr", data=body,
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    d["_elapsed"] = round(time.time() - t0, 2)
    if d.get("error"):
        raise RuntimeError(d["error"])
    return d


def strip_tags(s: str) -> str:
    s = re.sub(r"<otsl>|</otsl>", " ", s)
    s = re.sub(r"</?[a-z]+>", " ", s)
    s = re.sub(r"```[a-zA-Z]*", " ", s)
    s = re.sub(r"[#*`>|]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def edit_distance(pred: str, gt: str) -> float:
    """归一化编辑距离 = 1 - difflib 相似度。"""
    a, b = strip_tags(pred).lower(), strip_tags(gt).lower()
    if not a and not b:
        return 0.0
    return round(1.0 - difflib.SequenceMatcher(None, a, b).ratio(), 4)


def gt_text(page: dict, keep_all: bool = False) -> str:
    dets = sorted([d for d in page["layout_dets"] if not d.get("ignore")],
                  key=lambda d: (d.get("order") is None, d.get("order") or 0))
    out = []
    for d in dets:
        cat = d["category_type"]
        if cat in TEXT_CATS:
            out.append(d.get("text") or "")
        elif cat in EQ_CATS:
            out.append(d.get("latex") or d.get("text") or "")
        elif cat == "table" and keep_all:
            out.append(d.get("html") or "")
    return "\n".join(x for x in out if x.strip())


def order_dets(page: dict) -> list[dict]:
    return sorted([d for d in page["layout_dets"] if not d.get("ignore")],
                  key=lambda d: (d.get("order") is None, d.get("order") or 0))


def run_oracle(page: dict, img_path: Path, keep_all: bool = False, max_tokens: int = 1024) -> tuple[str, list]:
    img = Image.open(img_path).convert("RGB")
    parts, infos = [], []
    for d in order_dets(page):
        cat = d["category_type"]
        if cat in ("abandon", "figure", "chart_mask", "text_mask", "table_mask",
                   "need_mask", "algorithm_mask", "organic_chemical_formula_mask", "unknown_mask"):
            continue
        xs = d["poly"][0::2]
        ys = d["poly"][1::2]
        x0, x1 = max(0, min(xs) - 4), min(img.width, max(xs) + 4)
        y0, y1 = max(0, min(ys) - 4), min(img.height, max(ys) + 4)
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        crop = img.crop((int(x0), int(y0), int(x1), int(y1)))
        prompt = PROMPT.get(cat, "Text Recognition:")
        r = post_ocr(crop, prompt, "region", max_tokens)
        parts.append(r.get("content", ""))
        infos.append({"cat": cat, "elapsed": r["_elapsed"], "chars": len(r.get("content") or "")})
    return "\n".join(parts), infos


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/tmp/omnidoc")
    ap.add_argument("--mode", choices=["page", "oracle"], default="page")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--keep-all", action="store_true", help="GT 是否保留表格 HTML")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--out", default=None)
    ap.add_argument("--pred-dir", default=None, help="保存每页预测输出")
    args = ap.parse_args()

    data = Path(args.data)
    pages = json.load(open(data / "OmniDocBench.json"))
    sel = json.load(open(data / "selected_idx.json"))
    sel = sel[args.start: (args.start + args.limit) if args.limit else None]

    results = []
    for n, idx in enumerate(sel, 1):
        page = pages[idx]
        info = page["page_info"]
        img_path = data / "images" / info["image_path"]
        if not img_path.exists():
            print(f"[{n}/{len(sel)}] skip {info['image_path']} (missing)")
            continue
        gt = gt_text(page, args.keep_all)
        t0 = time.time()
        try:
            if args.mode == "oracle":
                pred, infos = run_oracle(page, img_path, args.keep_all, args.max_tokens)
            else:
                img = Image.open(img_path).convert("RGB")
                r = post_ocr(img, "Text Recognition:", "page", args.max_tokens)
                pred = r.get("content", "")
                infos = [{"metrics": r.get("metrics", {})}]
        except Exception as exc:  # noqa: BLE001
            print(f"[{n}/{len(sel)}] ERROR {info['image_path']}: {exc}")
            results.append({"idx": idx, "image": info["image_path"], "error": str(exc)})
            continue
        dt = round(time.time() - t0, 2)
        ed = edit_distance(pred, gt)
        rec = {"idx": idx, "image": info["image_path"], "subset": info["page_attribute"]["subset"],
               "lang": info["page_attribute"]["language"], "edit": ed,
               "pred_chars": len(strip_tags(pred)), "gt_chars": len(strip_tags(gt)),
               "elapsed_s": dt, "regions": len(infos)}
        results.append(rec)
        print(f"[{n}/{len(sel)}] {info['image_path'][:44]:44s} {rec['subset']:12s} edit={ed:.4f} "
              f"({rec['elapsed_s']}s, {rec['regions']} regions)", flush=True)
        if args.pred_dir:
            pd = Path(args.pred_dir) / f"{idx}.txt"
            pd.parent.mkdir(parents=True, exist_ok=True)
            pd.write_text(pred, encoding="utf-8")

    if results:
        ok = [r for r in results if "edit" in r]
        if ok:
            mean = sum(r["edit"] for r in ok) / len(ok)
            print(f"\n=== {args.mode} 模式：{len(ok)} 页，平均文本编辑距离 = {mean:.4f}"
                  f"（TextEdit={mean*100:.2f}），平均耗时 {sum(r['elapsed_s'] for r in ok)/len(ok):.1f}s/页")
    if args.out:
        json.dump(results, open(args.out, "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
