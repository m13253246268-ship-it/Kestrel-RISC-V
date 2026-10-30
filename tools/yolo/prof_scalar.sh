#!/bin/bash
# V51：标量回退层（分组 depthwise）耗时归属
set -u
cd ~/kv/yolo
B=~/kv/kestrel_src/build-riscv/vllm_kestrel
LOG=/tmp/sp.log
VLLM_THREADS=8 OMP_NUM_THREADS=8 VLLM_YOLO_CPU_TIMING=1 VLLM_YOLO_SCALAR_PROF=1 \
$B --yolo-model yolov8n_q8_320.vqf --yolo-image bus.jpg --yolo-conf 0.25 --yolo-iou 0.45 \
> "$LOG" 2>&1
echo "--- 标量回退层 ---"
grep "SCALAR-CONV" "$LOG"
echo "--- 合计 ---"
grep "SCALAR-CONV" "$LOG" | awk '{s+=$2} END {printf "  scalar total = %.2f ms\n", s}'
echo "--- 总览 ---"
grep "forward breakdown" "$LOG"
grep "image_total" "$LOG"
