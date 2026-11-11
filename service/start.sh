#!/usr/bin/env bash
# =============================================================================
# 板端一键启动：axllm 推理服务（18081）+ Flask 网页测试台（8080）
#
# 目录约定（模型包根目录下）：
#   <pkg>/bin/axllm                     AX650 aarch64 运行时（AXERA 官方包）
#   <pkg>/config.json                   axllm 模型配置
#   <pkg>/service/start.sh              本脚本
#   <pkg>/service/board_app.py          Flask 服务
#   <pkg>/service/slicing.py            MiniCPM-V 式切片（纯 numpy+PIL）
#   <pkg>/service/otsl_to_md.py         OTSL → Markdown
#   <pkg>/service/web/                  静态前端（M5Stack 风格）
#
# 用法（板端执行）：
#   ./service/start.sh start      # 起 axllm + Flask（已在跑会复用）
#   ./service/start.sh stop       # 停（先 Flask 再 axllm）
#   ./service/start.sh restart
#   ./service/start.sh status
#   ./service/start.sh logs
#
# 环境变量：
#   AXLLM_PORT=18081  PORT=8080  MAX_SLICES_PER_PAGE=9  AXLLM_MODEL=（默认自动从 /v1/models 取）
#   LAYOUT_MODEL=（可选）DocLayout-YOLO 版面 axmodel 路径；设置后自动/版面模式可用，不设则回退 page
#   LD_LIBRARY_PATH 会追加 /opt/lib:/opt/bin:<pkg>/bin
# =============================================================================
set -uo pipefail

SERVICE_DIR=$(cd "$(dirname "$0")" && pwd)
PKG_DIR=$(cd "$SERVICE_DIR/.." && pwd)
LOG_DIR=${LOG_DIR:-"$PKG_DIR/logs"}
AXLLM_PORT=${AXLLM_PORT:-18081}
WEB_PORT=${PORT:-8080}
AXLLM_PIDFILE="$LOG_DIR/axllm.pid"
WEB_PIDFILE="$LOG_DIR/board_app.pid"
AXLLM_LOG="$LOG_DIR/axllm.log"
WEB_LOG="$LOG_DIR/board_app.log"

mkdir -p "$LOG_DIR"

say() { printf '\033[1;34m[start]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[err]\033[0m %s\n' "$*" >&2; exit 1; }

pid_alive() { [ -f "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null; }

preflight() {
  [ -f "$PKG_DIR/config.json" ] || die "缺 $PKG_DIR/config.json（模型包不完整）"
  [ -x "$PKG_DIR/bin/axllm" ] || [ -f "$PKG_DIR/bin/axllm" ] || die "缺 bin/axllm（从 AXERA 官方 AX650 包取）"
  command -v python3 >/dev/null || die "板端没有 python3"
  python3 - <<'PY' || die "Python 依赖缺失：pip3 install flask pillow numpy"
import flask, PIL, numpy  # noqa: F401
print("[start] python deps ok: flask/PIL/numpy")
PY
  # NPU 运行时/驱动自检（AX650 出厂 rootfs 起就带）
  [ -e /opt/bin/ax_run_model ] || warn "未发现 /opt/bin/ax_run_model（运行时可能不在 /opt）"
  if [ -f "$AXLLM_PIDFILE" ] && pid_alive "$AXLLM_PIDFILE"; then
    warn "axllm 已在跑（pid $(cat "$AXLLM_PIDFILE")），跳过启动"
    return 1
  fi
  # CMM 余量提示（AX650）：低于 ~1.5 GB 时加载 2.5 GB 包可能失败
  if [ -r /proc/ax_proc/mem_cmm_info ]; then
    head -3 /proc/ax_proc/mem_cmm_info || true
  fi
  return 0
}

start_axllm() {
  preflight || return 0
  say "启动 axllm（模型目录 $PKG_DIR，端口 $AXLLM_PORT）"
  # 注意：不要包一层 ( cd ... & echo $! )，否则 pidfile 记的是外层 subshell。
  # 这里先 cd 再直接后台起 setsid，$! 才是 setsid/axllm 的 pid。
  pushd "$PKG_DIR" >/dev/null
  setsid env LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/opt/lib:/opt/bin:$PKG_DIR/bin" \
    ./bin/axllm serve . --port "$AXLLM_PORT" >> "$AXLLM_LOG" 2>&1 < /dev/null &
  echo $! > "$AXLLM_PIDFILE"
  popd >/dev/null
  for _ in $(seq 1 30); do
    sleep 2
    if curl -s --max-time 2 "http://127.0.0.1:$AXLLM_PORT/v1/models" >/dev/null 2>&1; then
      say "axllm ready（日志 $AXLLM_LOG）"
      # 关键自检：prefill 桶必须够放下「448 块 256 image token + prompt」
      if grep -q 'prefill_max_token_num' "$AXLLM_LOG"; then
        grep -o 'prefill_max_token_num : [0-9]*' "$AXLLM_LOG" | tail -1
        if ! grep -Eq 'prefill_max_token_num : (1[0-9]{3}|[2-9][0-9]{3})' "$AXLLM_LOG"; then
          warn "prefill 桶 < 1000：旧版层 axmodel（p128 且 prefill_len<1152），单块图像会报上下文超限"
        fi
      fi
      return 0
    fi
  done
  echo "--- $AXLLM_LOG tail ---" >&2; tail -30 "$AXLLM_LOG" >&2
  die "axllm 30 次轮询仍未就绪"
}

start_web() {
  if [ -f "$WEB_PIDFILE" ] && pid_alive "$WEB_PIDFILE"; then
    warn "Flask 已在跑（pid $(cat "$WEB_PIDFILE")）"
    return 0
  fi
  say "启动 Flask 测试台（端口 $WEB_PORT，后端 http://127.0.0.1:$AXLLM_PORT）"
  pushd "$SERVICE_DIR" >/dev/null
  setsid env AXLLM_URL="http://127.0.0.1:$AXLLM_PORT" PORT="$WEB_PORT" \
    LAYOUT_MODEL="${LAYOUT_MODEL:-}" \
    python3 board_app.py >> "$WEB_LOG" 2>&1 < /dev/null &
  echo $! > "$WEB_PIDFILE"
  popd >/dev/null
  for _ in $(seq 1 15); do
    sleep 1
    if curl -s --max-time 2 "http://127.0.0.1:$WEB_PORT/api/health" >/dev/null 2>&1; then
      say "网页测试台 ready：http://<board-ip>:$WEB_PORT/"
      return 0
    fi
  done
  echo "--- $WEB_LOG tail ---" >&2; tail -30 "$WEB_LOG" >&2
  die "Flask 未就绪（看上面日志；常见原因：缺 flask/pillow/numpy、端口被占）"
}

stop_one() {
  local name=$1 pidfile=$2 exact=${3:-}
  if [ -f "$pidfile" ] && pid_alive "$pidfile"; then
    local pid; pid=$(cat "$pidfile")
    say "停 $name（pid $pid）"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 10); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -9 "$pid" 2>/dev/null || true
  else
    warn "$name 未在跑（无 pidfile 或进程已退）"
  fi
  # 兜底：setsid 在个别环境会 fork，pidfile 可能失效。只对确定的进程名用 pkill -x，
  # 绝不用 pkill -f（会匹配到 SSH 远端 shell 自己的命令行，把当前会话一起杀掉）。
  if [ -n "$exact" ]; then
    pkill -x "$exact" 2>/dev/null && warn "已按进程名清理残留 $exact" || true
  fi
  rm -f "$pidfile"
}

status() {
  printf '%-10s ' axllm;  pid_alive "$AXLLM_PIDFILE" && echo "RUNNING pid=$(cat "$AXLLM_PIDFILE")" || echo "STOPPED"
  printf '%-10s ' flask;  pid_alive "$WEB_PIDFILE" && echo "RUNNING pid=$(cat "$WEB_PIDFILE")" || echo "STOPPED"
  curl -s --max-time 3 "http://127.0.0.1:$AXLLM_PORT/v1/models" 2>/dev/null | head -c 200; echo
  curl -s --max-time 3 "http://127.0.0.1:$WEB_PORT/api/health" 2>/dev/null | head -c 200; echo
}

case "${1:-start}" in
  start)
    start_axllm
    start_web
    echo
    say "全部就绪：axllm :$AXLLM_PORT  flask :$WEB_PORT（日志目录 $LOG_DIR）"
    ;;
  stop)
    # 先停 Flask 再停 axllm，避免页面在推理中被抽走后端
    stop_one flask "$WEB_PIDFILE"
    stop_one axllm "$AXLLM_PIDFILE" axllm
    ;;
  restart|reload)
    "$0" stop
    sleep 2
    "$0" start
    ;;
  status) status ;;
  logs)
    tail -n 40 "$AXLLM_LOG" 2>/dev/null; echo '---'; tail -n 40 "$WEB_LOG" 2>/dev/null
    ;;
  *) die "用法: $0 {start|stop|restart|status|logs}" ;;
esac
