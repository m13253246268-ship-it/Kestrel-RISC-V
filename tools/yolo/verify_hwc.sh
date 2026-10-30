#!/bin/bash
# V50 验收：HWC 输入 im2col 的位级对拍 + 320/640 性能
set -u
cd ~/kv/yolo
B=~/kv/kestrel_src/build-riscv/vllm_kestrel

echo "===== 1. 位级对拍（同二进制 HWC=1 vs HWC=0，640 + 320）====="
for M in yolov8n_q8.vqf yolov8n_q8_320.vqf; do
  VLLM_THREADS=8 OMP_NUM_THREADS=8 VLLM_YOLO_HWC=0 $B --yolo-model "$M" \
    --yolo-raw raw.f32 --yolo-raw-out /tmp/hwc0.f32 >/dev/null 2>&1
  VLLM_THREADS=8 OMP_NUM_THREADS=8 VLLM_YOLO_HWC=1 $B --yolo-model "$M" \
    --yolo-raw raw.f32 --yolo-raw-out /tmp/hwc1.f32 >/dev/null 2>&1
  if cmp -s /tmp/hwc0.f32 /tmp/hwc1.f32; then echo "  $M  HWC0 vs HWC1: BIT-EXACT"; else echo "  $M  HWC0 vs HWC1: DIFF"; fi
done

echo
echo "===== 2. 与改造前 CHW 锚点对拍 ====="
for M in yolov8n_q8 yolov8n_q8_320; do
  if [ -f "anchor_${M}.f32" ]; then
    cmp -s "anchor_${M}.f32" /tmp/hwc1.f32 2>/dev/null
    VLLM_THREADS=8 OMP_NUM_THREADS=8 VLLM_YOLO_HWC=1 $B \
      --yolo-model "${M}.vqf" --yolo-raw raw.f32 --yolo-raw-out "/tmp/hwc_${M}.f32" >/dev/null 2>&1
    if cmp -s "anchor_${M}.f32" "/tmp/hwc_${M}.f32"; then
      echo "  $M: ANCHOR_MATCH（与改造前 CHW 逐字节一致）"
    else
      echo "  $M: ANCHOR_DIFF"
      python3 -c "import numpy as np;a=np.fromfile('anchor_${M}.f32',np.float32);b=np.fromfile('/tmp/hwc_${M}.f32',np.float32);print('    diff=%d/%d max=%.3e'%((a!=b).sum(),a.size,np.abs(a-b).max() if a.size==b.size else -1))" 2>/dev/null
    fi
  fi
done

echo
echo "===== 3. 320 性能：HWC=0（基线）====="
for r in 1 2 3; do
  VLLM_THREADS=8 OMP_NUM_THREADS=8 VLLM_YOLO_CPU_TIMING=1 VLLM_YOLO_HWC=0 \
  $B --yolo-model yolov8n_q8_320.vqf --yolo-image bus.jpg --yolo-conf 0.25 --yolo-iou 0.45 \
  2>&1 | grep -E "forward breakdown|image_total"
done

echo
echo "===== 4. 320 性能：HWC=1（新）====="
for r in 1 2 3; do
  VLLM_THREADS=8 OMP_NUM_THREADS=8 VLLM_YOLO_CPU_TIMING=1 VLLM_YOLO_HWC=1 \
  $B --yolo-model yolov8n_q8_320.vqf --yolo-image bus.jpg --yolo-conf 0.25 --yolo-iou 0.45 \
  2>&1 | grep -E "forward breakdown|image_total"
done

echo
echo "===== 5. 640 性能：HWC=0 / HWC=1 ====="
for H in 0 1; do
  for r in 1 2; do
    VLLM_THREADS=8 OMP_NUM_THREADS=8 VLLM_YOLO_CPU_TIMING=1 VLLM_YOLO_HWC=$H \
    $B --yolo-model yolov8n_q8.vqf --yolo-image bus.jpg --yolo-conf 0.25 --yolo-iou 0.45 \
    2>&1 | grep -E "image_total" | sed "s/^/  HWC=$H /"
  done
done
