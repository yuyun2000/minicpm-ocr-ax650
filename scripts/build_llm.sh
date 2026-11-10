#!/usr/bin/env bash
# =============================================================================
# 用 Pulsar2 8.0 的 llm_build2 把 MiniCPM-OCR 的「纯文本 LLM」编成 AX650 逐层 axmodel
#
# 前置：
#   1) Linux x86_64 宿主机（Pulsar2 只发 Linux x86_64 独立包；macOS 无法执行）
#   2) Pulsar2 8.0 可用（本仓库实测环境：Ubuntu 24.04 / 16C / 62G）
#   3) 已跑过 scripts/prep_text_model.py，得到纯文本目录（--input_path）
#
# 产物（$OUT_DIR）：
#   qwen3_5_text_p128_l{0..23}_together.axmodel   24 层，共 ≈575 MB
#   qwen3_5_text_post.axmodel                     BF16 post，≈1017 MB
#   model.embed_tokens.weight.bfloat16.bin        248100×1024 bf16，≈508 MB
#
# 关键参数（来自实测报告，别随手改）：
#   --prefill_len 1152        预填充桶上限。448×448 单块是 256 image token + prompt，
#                             首版预填充桶只有 128 会直接报
#                             `input_num_token(276) > prefill_max_token_num(128)`
#   --prefill_step_size 128   桶步长（决定产物名里的 p128）
#   --max_context 2048        解码上下文上限
#   --weight_type s8          层权重 INT8
#   --post_weight_type bf16   post 用 BF16（与 AXERA 官方 MiniCPM-V-4.6 包同策略）
#   --parallel 8              8 并发编译
#
# 耗时参考：8 并发下 24 层约 60–90 min。
# =============================================================================
set -euo pipefail

INPUT=${INPUT:-"$PWD/work/origin_text"}    # prep_text_model.py 的输出目录
OUT_DIR=${OUT_DIR:-"$PWD/work/llm_v2"}     # llm_build2 --output_path
CHIP=${CHIP:-AX650}
WEIGHT_TYPE=${WEIGHT_TYPE:-s8}
POST_WEIGHT_TYPE=${POST_WEIGHT_TYPE:-bf16}
PREFILL_LEN=${PREFILL_LEN:-1152}
PREFILL_STEP=${PREFILL_STEP:-128}
MAX_CONTEXT=${MAX_CONTEXT:-2048}
PARALLEL=${PARALLEL:-8}
LOG=${LOG:-"$OUT_DIR/llm_build2.log"}
# 官方 MiniCPM-V-4.6.axera 的 llm_build_ax650.sh 里设了 FLOAT_MATMUL_USE_CONV_EU=1；
# 本项目 v2 构建记录只落了下面的 llm_build2 参数，是否设置该变量未记录（TODO）。
# 想对齐官方就取消下一行注释。
# export FLOAT_MATMUL_USE_CONV_EU=1

command -v pulsar2 >/dev/null 2>&1 || { echo "[ERR] 找不到 pulsar2（先装 Pulsar2 8.0 并加进 PATH）" >&2; exit 1; }
[ -f "$INPUT/config.json" ] || { echo "[ERR] $INPUT/config.json 不存在；先跑 scripts/prep_text_model.py" >&2; exit 1; }

mkdir -p "$OUT_DIR"
echo "[build] input=$INPUT out=$OUT_DIR chip=$CHIP weight=$WEIGHT_TYPE post=$POST_WEIGHT_TYPE"
echo "[build] prefill_len=$PREFILL_LEN step=$PREFILL_STEP ctx=$MAX_CONTEXT parallel=$PARALLEL"
echo "[build] log=$LOG"

# 注意：llm_build2 的 --input_path 必须是「纯文本」目录：
#   - 权重键 model.language_model.* 已改名成 model.*
#   - config.json 顶层 model_type=qwen3_5、text_config.model_type=qwen3_5_text
# 不改会分别报 'Config' object has no attribute 'linear_num_key_heads' 与 get_norm_attrs AssertionError。
set -x
pulsar2 llm_build2 \
  --input_path "$INPUT" \
  --output_path "$OUT_DIR" \
  --chip "$CHIP" \
  --weight_type "$WEIGHT_TYPE" \
  --post_weight_type "$POST_WEIGHT_TYPE" \
  --prefill_len "$PREFILL_LEN" \
  --prefill_step_size "$PREFILL_STEP" \
  --max_context "$MAX_CONTEXT" \
  --parallel "$PARALLEL" \
  2>&1 | tee "$LOG"
set +x

# ---- 产物自检 ----
echo
echo "[check] 层数:"
ls -1 "$OUT_DIR"/*_l*_together.axmodel 2>/dev/null | wc -l
echo "[check] post / embed:"
ls -lh "$OUT_DIR"/qwen3_5_text_post.axmodel "$OUT_DIR"/model.embed_tokens.weight.bfloat16.bin 2>/dev/null || true
echo "[check] 总大小:"
du -sh "$OUT_DIR"
cat <<'EOF'

[ok] 完成。把以下三样交给 scripts/assemble_package.py：
  - $OUT_DIR/qwen3_5_text_p*_l{0..23}_together.axmodel
  - $OUT_DIR/qwen3_5_text_post.axmodel
  - $OUT_DIR/model.embed_tokens.weight.bfloat16.bin
启动时板端 axllm 日志里应出现 prefill_max_token_num: 1152（v2 包）。
EOF
