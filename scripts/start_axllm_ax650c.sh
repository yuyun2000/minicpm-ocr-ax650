#!/bin/sh
pkill -x axllm 2>/dev/null || true
sleep 2
cd /root/minicpm-ocr-v2 || exit 1
setsid env LD_LIBRARY_PATH=/soc/lib:/opt/lib:/opt/bin:$PWD/bin ./bin/axllm serve . --port 18081 > /root/minicpm-ocr-v2/serve.log 2>&1 < /dev/null &
sleep 20
pgrep -x axllm >/dev/null && echo AXLLM_OK || echo AXLLM_FAIL
curl -s --max-time 5 http://127.0.0.1:18081/v1/models | head -c 120
