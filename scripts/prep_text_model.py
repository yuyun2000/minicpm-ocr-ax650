#!/usr/bin/env python3
"""把 MiniCPM-OCR 1.0 的 HF 权重改造成 `pulsar2 llm_build2` 能吃的「纯文本 LLM」目录。

为什么要这一步（实测，见 docs/conversion.md）：
  MiniCPM-OCR 的 HF checkpoint 是「多模态整体」布局，文本张量前缀是
  `model.language_model.*`；AXERA 的 qwen3_5_text builder 按纯文本 LLM 布局取
  `model.*`，且顶层 `model_type` 必须是 `qwen3_5`。不改会分别报：

    - 不改前缀 / 不改 model_type:  'Config' object has no attribute 'linear_num_key_heads'
    - 只改前缀不改 text_config.model_type:  get_norm_attrs AssertionError

输入目录（HF 原始权重）至少包含：
    config.json, model.safetensors（或分片 + model.safetensors.index.json）
输出目录（llm_build2 --input_path 指向这里）：
    config.json（重标 qwen3_5 / qwen3_5_text）
    model.safetensors（键名 `model.language_model.*` -> `model.*`，视觉张量已剔除）

用法：
  python3 prep_text_model.py --src /path/to/origin --out /path/to/origin_text
  # 可选：--copy-tokenizer 把 tokenizer 相关文件一起拷过去（llm_build2 不需要，
  #       但后面做 tokenizer.txt 转换时省一步）

依赖：torch + safetensors（bf16 权重不能走 numpy 路径）。

注意：本脚本是按 2026-11-10 转换记录的产物布局整理的，**未在本仓库重新执行**；
首次使用请先 diff 一下输出 config.json 与 docs/conversion.md §2 的字段表。
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

# 视觉侧张量前缀（4x 模式下 vit_merger 不参与前向，纯死重，这里一并丢掉）
DROP_PREFIXES = (
    "model.vision_tower.",
    "model.merger.",
    "model.vit_merger.",
    "vision_tower.",
    "merger.",
)

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
    "processor_config.json",
    "preprocessor_config.json",
)


def load_weights(src: Path) -> dict[str, torch.Tensor]:
    """加载单文件或分片 safetensors。"""
    single = src / "model.safetensors"
    if single.is_file():
        print(f"[load] {single} ({single.stat().st_size / 1e9:.2f} GB)")
        return load_file(str(single))

    index = src / "model.safetensors.index.json"
    if not index.is_file():
        raise SystemExit(f"找不到 model.safetensors 或 model.safetensors.index.json: {src}")
    weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
    shards = sorted(set(weight_map.values()))
    state: dict[str, torch.Tensor] = {}
    for shard in shards:
        p = src / shard
        print(f"[load] {p.name} ({p.stat().st_size / 1e9:.2f} GB)")
        state.update(load_file(str(p)))
    return state


def rename_key(key: str) -> str | None:
    """`model.language_model.x` -> `model.x`；视觉张量返回 None（丢弃）。"""
    for pref in DROP_PREFIXES:
        if key.startswith(pref):
            return None
    if key.startswith("model.language_model."):
        return "model." + key[len("model.language_model."):]
    if key.startswith("language_model."):
        return "model." + key[len("language_model."):]
    if key.startswith("model."):
        return key
    # lm_head 之类：tie_word_embeddings=true 时不存在；出现了就原样保留，由 llm_build2 判断
    return key


def resolve_special_ids(src: Path) -> dict:
    """从 generation_config.json / tokenizer_config.json + tokenizer.json 读特殊 token id。

    openbmb/MiniCPM-OCR-1.0 实测：bos=<|im_start|>=248045、eos=[248044, 248046]
    （<|endoftext|> / <|im_end|>）、pad=<|endoftext|>=248044。
    """
    ids: dict = {}
    gen_p = src / "generation_config.json"
    if gen_p.is_file():
        gen = json.loads(gen_p.read_text(encoding="utf-8"))
        for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
            if key in gen:
                ids[key] = gen[key]

    tok_cfg_p = src / "tokenizer_config.json"
    tok_json_p = src / "tokenizer.json"
    if tok_cfg_p.is_file() and tok_json_p.is_file():
        tok_cfg = json.loads(tok_cfg_p.read_text(encoding="utf-8"))
        tok = json.loads(tok_json_p.read_text(encoding="utf-8"))
        str2id = {}
        for tok_id, content in (tok.get("model", {}).get("vocab") or {}).items():
            str2id[tok_id] = content
        for a in tok.get("added_tokens", []):
            str2id[a["content"]] = a["id"]
        for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
            cfg_key = key.replace("_id", "")
            val = tok_cfg.get(cfg_key)
            if key not in ids and isinstance(val, str) and val in str2id:
                ids[key] = str2id[val]
    return ids


def retag_config(cfg: dict, special_ids: dict) -> dict:
    """把 config 重标成 AXERA builder 认识的 qwen3_5 纯文本配置。"""
    text = dict(cfg.get("text_config") or {})
    text["model_type"] = "qwen3_5_text"

    out = {
        "architectures": ["Qwen3_5ForCausalLM"],
        "model_type": "qwen3_5",
        "text_config": text,
        "tie_word_embeddings": bool(text.get("tie_word_embeddings", cfg.get("tie_word_embeddings", True))),
        "dtype": "bfloat16",
        "use_cache": True,
    }
    # 特殊 token id：text_config 里已有就沿用，否则用 generation_config / tokenizer 解析结果
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if key in text:
            out[key] = text[key]
        elif key in special_ids:
            out[key] = special_ids[key]
        elif key in cfg:
            out[key] = cfg[key]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="HF 原始权重目录（含 model.safetensors）")
    ap.add_argument("--out", required=True, help="输出目录（llm_build2 --input_path）")
    ap.add_argument("--copy-tokenizer", action="store_true", help="顺带拷贝 tokenizer/*.json 与 chat template")
    ap.add_argument("--dry-run", action="store_true", help="只统计键名，不落盘")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    if not (src / "config.json").is_file():
        raise SystemExit(f"缺 {src / 'config.json'}")

    state = load_weights(src)
    print(f"[in ] tensors={len(state)} params={sum(t.numel() for t in state.values()) / 1e6:.2f} M")

    renamed: dict[str, torch.Tensor] = {}
    dropped = 0
    for key, tensor in state.items():
        nk = rename_key(key)
        if nk is None:
            dropped += 1
            continue
        if nk in renamed:
            raise SystemExit(f"改名后键冲突: {key} -> {nk}")
        renamed[nk] = tensor.contiguous()

    print(f"[out] tensors={len(renamed)} params={sum(t.numel() for t in renamed.values()) / 1e6:.2f} M "
          f"dropped_vision={dropped}")

    cfg = retag_config(json.loads((src / "config.json").read_text(encoding="utf-8")),
                       resolve_special_ids(src))
    print(f"[cfg] bos={cfg.get('bos_token_id')} eos={cfg.get('eos_token_id')} pad={cfg.get('pad_token_id')}")
    if args.dry_run:
        print(json.dumps(cfg, ensure_ascii=False, indent=2))
        return 0

    out.mkdir(parents=True, exist_ok=True)
    save_file(renamed, str(out / "model.safetensors"), metadata={"format": "pt"})
    (out / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[save] {out / 'model.safetensors'} ({(out / 'model.safetensors').stat().st_size / 1e9:.2f} GB)")

    if args.copy_tokenizer:
        for name in TOKENIZER_FILES:
            p = src / name
            if p.is_file():
                shutil.copy2(p, out / name)
        print("[save] tokenizer/*.json 已拷贝")

    # 自检：文本栈应有的关键张量
    must = [k for k in renamed if k.startswith("model.layers.0.")]
    if not must:
        raise SystemExit("自检失败：改名后没有 model.layers.0.* —— 前缀规则可能与该权重版本不符")
    print(f"[ok] model.layers.0.* 命中 {len(must)} 个张量")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
