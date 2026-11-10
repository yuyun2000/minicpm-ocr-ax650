#!/usr/bin/env python3
"""板端服务并发/吞吐测试：并发发 N 个 OCR 或 chat 请求，统计延迟分布与吞吐。

用法：
  python demo/minicpm-ocr/bench_concurrency.py --mode chat --levels 1,2,4,8 --requests 8
  python demo/minicpm-ocr/bench_concurrency.py --mode ocr --image /tmp/ocr_448.png --levels 1,2,4 --requests 4
"""
from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
import urllib.request

BOARD = None


def post_json(prompt: str, max_tokens: int) -> dict:
    body = json.dumps({"prompt": prompt, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(f"{BOARD}/api/chat", data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    d["_latency"] = time.time() - t0
    return d


def post_ocr(image_path: str, prompt: str, mode: str, max_tokens: int) -> dict:
    data = open(image_path, "rb").read()
    boundary = "----benchboundary"
    parts = []
    for k, v in {"prompt": prompt, "mode": mode, "max_tokens": str(max_tokens)}.items():
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
    parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"a.png\"\r\n"
                 f"Content-Type: image/png\r\n\r\n".encode() + data + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)
    req = urllib.request.Request(f"{BOARD}/api/ocr", data=body,
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    d["_latency"] = time.time() - t0
    return d


def run_level(level: int, total: int, fn) -> dict:
    results: list[float] = []
    errors: list[str] = []
    lock = threading.Lock()
    sem = threading.Semaphore(level)

    def worker(i: int) -> None:
        with sem:
            try:
                d = fn()
                with lock:
                    results.append(d["_latency"])
            except Exception as exc:  # noqa: BLE001
                with lock:
                    errors.append(f"{type(exc).__name__}: {exc}")

    t0 = time.time()
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(total)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0
    r = sorted(results)
    return {
        "concurrency": level,
        "requests": total,
        "ok": len(r),
        "errors": len(errors),
        "wall_s": round(wall, 2),
        "throughput_req_per_s": round(len(r) / wall, 3) if wall else 0,
        "latency_mean_s": round(statistics.mean(r), 2) if r else None,
        "latency_p50_s": round(r[len(r) // 2], 2) if r else None,
        "latency_p95_s": round(r[min(len(r) - 1, int(len(r) * 0.95))], 2) if r else None,
        "latency_min_s": round(r[0], 2) if r else None,
        "latency_max_s": round(r[-1], 2) if r else None,
    }


def main() -> None:
    global BOARD
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default="http://192.168.10.145:8080")
    ap.add_argument("--mode", choices=["chat", "ocr"], default="chat")
    ap.add_argument("--levels", default="1,2,4")
    ap.add_argument("--requests", type=int, default=8)
    ap.add_argument("--prompt", default="用一句话解释什么是 OCR")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--image", default="/tmp/ocr_448.png")
    ap.add_argument("--ocr-mode", default="region")
    ap.add_argument("--ocr-prompt", default="Table Recognition:")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    BOARD = args.board.rstrip("/")
    levels = [int(x) for x in args.levels.split(",") if x.strip()]

    fn = (lambda: post_json(args.prompt, args.max_tokens)) if args.mode == "chat" else \
         (lambda: post_ocr(args.image, args.ocr_prompt, args.ocr_mode, args.max_tokens))
    rows = []
    for lv in levels:
        row = run_level(lv, args.requests, fn)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))
    if args.out:
        json.dump(rows, open(args.out, "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
