#!/bin/bash
# Online K1: thread sweep at 320x320, min/p25 statistic.
set -u
cd /root/kv/kestrel_vllm_riscv/tools/yolo
B=../../build-riscv/vllm_kestrel
N=20

mn() { sort -n | head -1 | awk '{printf "%.1f", $1}'; }
p25() { sort -n > /tmp/_s.txt; local n=$(wc -l < /tmp/_s.txt); sed -n "$(( n/4 + 1 ))p" /tmp/_s.txt | awk '{printf "%.1f", $1}'; }

echo "threads   image_min  image_p25   fwd_min   im2col_min  conv_min"
for T in 1 2 4 8; do
  VLLM_THREADS=$T OMP_NUM_THREADS=$T VLLM_YOLO_CPU_TIMING=1 VLLM_YOLO_FRAMES=$N \
  $B --yolo-model yolov8n_q8_320.vqf --yolo-image test.jpg --yolo-conf 0.25 --yolo-iou 0.45 \
  > /tmp/t$T.log 2>&1
  IMG=$(grep "image_total" /tmp/t$T.log | tail -n +2 | sed -E 's/.*image_total=([0-9.]+)ms.*/\1/' | mn)
  IMGP=$(grep "image_total" /tmp/t$T.log | tail -n +2 | sed -E 's/.*image_total=([0-9.]+)ms.*/\1/' | p25)
  FWD=$(grep "forward breakdown" /tmp/t$T.log | tail -n +2 | sed -E 's/.*total=([0-9.]+)ms.*/\1/' | mn)
  IM2=$(grep "forward breakdown" /tmp/t$T.log | tail -n +2 | sed -E 's/.*im2col=([0-9.]+)ms.*/\1/' | mn)
  CNV=$(grep "forward breakdown" /tmp/t$T.log | tail -n +2 | sed -E 's/.*conv=([0-9.]+)ms.*/\1/' | mn)
  printf "%4d  %10s  %9s  %8s  %10s  %8s\n" "$T" "$IMG" "$IMGP" "$FWD" "$IM2" "$CNV"
done
