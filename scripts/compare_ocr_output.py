#!/usr/bin/env python3
"""对比两段 OCR 输出（板端 NPU vs HF 参考）的精度指标。

指标：
- 字符级相似度（difflib ratio，去掉 OTSL 标签）
- 表格 F1：解析 OTSL/markdown 表格的单元格集合，按 (行,列) 对齐后算精确匹配的 P/R/F1
- 数字准确率：抽取所有数字 token，比较集合一致率

用法：
  .venv/bin/python demo/minicpm-ocr/compare_ocr_output.py --ref ref.txt --hyp hyp.txt [--json out.json]
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys

sys.path.insert(0, "demo/minicpm-ocr")
try:
    from otsl_to_md import parse_otsl
except Exception:  # pragma: no cover
    parse_otsl = None


def strip_tags(s: str) -> str:
    s = re.sub(r"<otsl>|</otsl>", " ", s)
    s = re.sub(r"</?[a-z]+>", " ", s)
    s = re.sub(r"```[a-zA-Z]*", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def char_similarity(ref: str, hyp: str) -> float:
    a, b = strip_tags(ref).lower(), strip_tags(hyp).lower()
    return difflib.SequenceMatcher(None, a, b).ratio()


def cell_set(text: str) -> set[tuple[int, int, str]]:
    """把 OTSL/markdown 表格转成 (行, 列, 文本) 单元格集合。"""
    cells: set[tuple[int, int, str]] = set()
    if "<otsl>" in text and parse_otsl is not None:
        try:
            table = parse_otsl(text)
            for r, row in enumerate(table.rows):
                for c, cell in enumerate(row):
                    t = strip_tags(getattr(cell, "text", "") or "")
                    if t:
                        cells.add((r, c, t))
            return cells
        except Exception:
            pass
    rows = [ln for ln in text.splitlines() if ln.strip().startswith("|")]
    for r, ln in enumerate(rows):
        for c, part in enumerate([p for p in ln.strip().strip("|").split("|")]):
            t = part.strip()
            if t and not re.fullmatch(r"-{2,}", t):
                cells.add((r, c, t))
    return cells


def cell_f1(ref: str, hyp: str) -> dict:
    a, b = cell_set(ref), cell_set(hyp)
    inter = len(a & b)
    p = inter / len(b) if b else 0.0
    r = inter / len(a) if a else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"ref_cells": len(a), "hyp_cells": len(b), "matched": inter,
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}


def number_accuracy(ref: str, hyp: str) -> dict:
    a = re.findall(r"\d+(?:\.\d+)?%?", strip_tags(ref))
    b = re.findall(r"\d+(?:\.\d+)?%?", strip_tags(hyp))
    sa, sb = set(a), set(b)
    hit = len(sa & sb)
    return {"ref_numbers": len(sa), "hyp_numbers": len(sb), "matched": hit,
            "accuracy": round(hit / len(sa), 4) if sa else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="参考输出文件（HF 基线的原文）")
    ap.add_argument("--hyp", required=True, help="待测输出文件（板端 NPU）")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    ref = open(args.ref, encoding="utf-8").read()
    hyp = open(args.hyp, encoding="utf-8").read()
    result = {
        "char_similarity": round(char_similarity(ref, hyp), 4),
        "table_cell": cell_f1(ref, hyp),
        "numbers": number_accuracy(ref, hyp),
        "ref_chars": len(strip_tags(ref)),
        "hyp_chars": len(strip_tags(hyp)),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
