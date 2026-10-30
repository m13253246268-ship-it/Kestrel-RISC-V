#!/bin/bash
cd ~/kv/yolo
B=~/kv/kestrel_src/build-riscv/vllm_kestrel
echo "=== 320x320 端到端（8 线程，bus.jpg letterbox 到 320）==="
for r in 1 2 3; do
  VLLM_THREADS=8 OMP_NUM_THREADS=8 VLLM_YOLO_CPU_TIMING=1 \
  $B --yolo-model yolov8n_q8_320.vqf --yolo-image bus.jpg --yolo-conf 0.25 --yolo-iou 0.45 \
  2>&1 | grep -E "forward breakdown|image_total"
done
echo "=== 检测框（确认 320 模型正确）==="
VLLM_THREADS=8 OMP_NUM_THREADS=8 \
$B --yolo-model yolov8n_q8_320.vqf --yolo-image bus.jpg --yolo-conf 0.25 --yolo-iou 0.45 \
  2>/dev/null | tail -1