#!/bin/sh
cd /root/minicpm-ocr-v2/webapp || exit 1
pkill -f "^python3 board_app.py" 2>/dev/null || true
sleep 1
nohup env PORT=8088 LAYOUT_MODEL=/root/minicpm-ocr-v2/doclayout_bl_u16.axmodel python3 board_app.py > /root/minicpm-ocr-v2/webapp/app.log 2>&1 < /dev/null &
sleep 5
pgrep -f "^python3 board_app.py" >/dev/null && echo FLASK_OK || (echo FLASK_FAIL; tail -5 /root/minicpm-ocr-v2/webapp/app.log)
