#!/usr/bin/env bash
# =============================================================================
# 把组装好的模型包 + 服务推到 AX650 板并拉起服务（在转换机 / Mac / Linux 上执行均可）
#
# 板端特点（实测）：
#   - **没有 rsync**，rootfs 精简 → 用 `tar cf - | ssh ... tar xf -` 传（2.5 GB ≈ 101 s）
#   - axllm 需要显式 LD_LIBRARY_PATH=/opt/lib:/opt/bin:$PWD/bin，否则启动即退
#   - 只能有一个 axllm 实例占用 NPU/CMM，重启前必须干净地杀掉旧进程
#
# 用法：
#   ./deploy_to_board.sh --pkg ./axllm_pkg --host root@192.168.10.144 \
#       --dst /root/minicpm-ocr-v2 --port 18081 [--password 123456] [--no-start]
#
# 可传多个授权方式：默认用 SSH key；给 --password 则走 SSH_ASKPASS（和原部署脚本一致）。
# =============================================================================
set -euo pipefail

PKG=""
HOST="root@192.168.10.144"
DST="/root/minicpm-ocr-v2"
PORT=18081
SERVICE_PORT=8080
PASSWORD=""
START=1
STRIP=1

usage() { sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-1}"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --pkg) PKG=$2; shift 2 ;;
    --host) HOST=$2; shift 2 ;;
    --dst) DST=$2; shift 2 ;;
    --port) PORT=$2; shift 2 ;;
    --service-port) SERVICE_PORT=$2; shift 2 ;;
    --password) PASSWORD=$2; shift 2 ;;
    --no-start) START=0; shift ;;
    --no-strip) STRIP=0; shift ;;
    -h|--help) usage 0 ;;
    *) echo "未知参数: $1" >&2; usage ;;
  esac
done
[ -n "$PKG" ] || usage
[ -d "$PKG" ] || { echo "[ERR] $PKG 不是目录" >&2; exit 1; }
[ -f "$PKG/config.json" ] || { echo "[ERR] $PKG/config.json 缺失，先用 scripts/assemble_package.py 组装" >&2; exit 1; }

SSH_OPTS=${SSH_OPTS:-"-o StrictHostKeyChecking=accept-new -o ConnectTimeout=20"}
SCP_OPTS=${SCP_OPTS:-"-o StrictHostKeyChecking=accept-new -o ConnectTimeout=20"}

# ---- 可选：密码认证（SSH_ASKPASS，不落盘明文） ----
ASKPASS=""
ssh_cmd() { if [ -n "$ASKPASS" ]; then sshpass_like "$@"; else ssh $SSH_OPTS "$@"; fi; }
sshpass_like() {
  local env_prefix=(env)
  env_prefix+=(SSH_ASKPASS="$ASKPASS" SSH_ASKPASS_REQUIRE=force DISPLAY="${DISPLAY:-:0}" SSH_PASS="$PASSWORD")
  "${env_prefix[@]}" ssh $SSH_OPTS -o PreferredAuthentications=password -o PubkeyAuthentication=no \
    -o NumberOfPasswordPrompts=1 "$@"
}
if [ -n "$PASSWORD" ]; then
  ASKPASS=$(mktemp /tmp/askpass-minicpm.XXXXXX)
  printf '#!/bin/sh\necho "$SSH_PASS"\n' > "$ASKPASS"
  chmod 700 "$ASKPASS"
  trap 'rm -f "$ASKPASS"' EXIT
fi

echo "[1/4] 检查板端环境（磁盘 / 内存 / axllm 是否在跑）"
ssh_cmd "$HOST" "df -h / | tail -1; free -m | head -2; pgrep -x axllm >/dev/null && echo 'axllm: RUNNING' || echo 'axllm: not running'" </dev/null

echo "[2/4] tar over ssh 传输 $PKG → $HOST:$DST"
# tarball 顶层是 package 目录名，板端 --strip-components 去掉它（板端没有 rsync，这是实测可用方式）
if [ "$STRIP" = 1 ]; then
  tar cf - -C "$(dirname "$(cd "$PKG" && pwd)")" "$(basename "$(cd "$PKG" && pwd)")" \
    | ssh_cmd "$HOST" "mkdir -p '$DST' && tar xf - -C '$DST' --strip-components=1 && du -sh '$DST'"
else
  tar cf - -C "$PKG" . | ssh_cmd "$HOST" "mkdir -p '$DST' && tar xf - -C '$DST' && du -sh '$DST'"
fi

if [ "$START" = 0 ]; then
  echo "[3/4] --no-start，跳过启动。手动启动："
  echo "      ssh $HOST 'cd $DST && setsid env LD_LIBRARY_PATH=/opt/lib:/opt/bin:\$PWD/bin ./bin/axllm serve . --port $PORT > serve.log 2>&1 < /dev/null &'"
  exit 0
fi

echo "[3/4] 板端重启 axllm（先杀旧实例，再等 /v1/models 就绪）"
# 用 pkill -x（精确匹配进程名）而不是 pkill -f 'axllm serve'：
# 后者会匹配到这条 SSH 远端 shell 自己的命令行，把当前会话一起杀掉。
ssh_cmd "$HOST" "cd '$DST' || exit 1
  chmod +x bin/axllm 2>/dev/null || true
  pkill -x axllm 2>/dev/null || true
  sleep 3
  setsid env LD_LIBRARY_PATH=/opt/lib:/opt/bin:\$PWD/bin ./bin/axllm serve . --port $PORT > serve.log 2>&1 < /dev/null &
  ok=0
  for i in \$(seq 1 30); do
    sleep 2
    if curl -s --max-time 2 http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1; then ok=1; break; fi
  done
  echo \"--- serve.log tail ---\"; tail -25 serve.log
  [ \$ok = 1 ] && echo 'AXLLM_READY' || { echo 'AXLLM_NOT_READY'; exit 1; }" </dev/null

echo "[4/4] 检查板内 OpenAI 接口"
ssh_cmd "$HOST" "curl -s --max-time 10 http://127.0.0.1:$PORT/v1/models | head -c 300; echo" </dev/null

echo
echo "完成。模型包在 $HOST:$DST"
echo "  文本/单块图像：http://<board-ip>:$PORT/v1/chat/completions"
echo "  网页测试台（把 service/ 一起推上去后）：cd $DST && ./service/start.sh"
echo "  网页：http://<board-ip>:$SERVICE_PORT/"
echo
echo "注意：若日志里 prefill_max_token_num 仍很小（如 128），说明板上的层 axmodel 是旧版（prefill 桶太小），"
echo "      用 scripts/build_llm.sh（PREFILL_LEN=1152）重编后用本脚本重推。"
