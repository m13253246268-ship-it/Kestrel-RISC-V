#!/bin/bash
# Upload yolov8n.pt to the K1 board (host-side).
# Usage: bash _k1_yolo_put_pt.sh
set -eu
: "${SSHPASS:?set SSHPASS to the K1 ssh password}"

PT=/mnt/e/项目/Kestrel_VLLM_RISC-V/modls/Ultralytics_YOLOv8/yolov8n.pt
DST=/root/kv/kestrel_vllm_riscv/tools/yolo/yolov8n.pt

echo "uploading $PT -> $DST ..."
sshpass -e ssh -p 2223 \
  -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  -o LogLevel=ERROR -o ConnectTimeout=20 \
  root@172.26.208.1 "cat > '$DST'" < "$PT"

echo "PUT_OK $DST $(stat -c %s "$PT") bytes"
