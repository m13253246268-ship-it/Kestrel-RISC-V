#!/bin/bash
cd ~/kv/yolo
B=~/kv/kestrel_src/build-riscv/vllm_kestrel.new

echo "===== 1. 位级对拍（640 模型，覆盖 3×3 层）====="
VLLM_THREADS=8 OMP_NUM_THREADS=8 VLLM_YOLO_RVV=0 \
  $B --yolo-model yolov8n_q8.vqf --yolo-raw raw.f32 --yolo-raw-out /tmp/rvv0.f32 >/dev/null 2>&1
VLLM_THREADS=8 OMP_NUM_THREADS=8 VLLM_YOLO_RVV=1 \
  $B --yolo-model yolov8n_q8.vqf --yolo-raw raw.f32 --yolo-raw-out /tmp/rvv1.f32 >/dev/null 2>&1
if cmp -s /tmp/rvv0.f32 /tmp/rvv1.f32; then
  echo "  RVV0 vs RVV1: BIT-EXACT（3×3 im2col 向量化位级一致）"
else
  echo "  RVV0 vs RVV1: DIFF!"
  python3 -c "import numpy as np; a=np.fromfile('/tmp/rvv0.f32',np.float32); b=np.fromfile('/tmp/rvv1.f32',np.float32); print('  diff floats=%d/%d'%((a!=b).sum(),a.size))" 2>/dev/null
fi

echo
echo "===== 2. 320 性能（新 3×3 RVV 化）====="
for r in 1 2 3; do
  VLLM_THREADS=8 OMP_NUM_THREADS=8 VLLM_YOLO_RVV=1 VLLM_YOLO_CPU_TIMING=1 \
  $B --yolo-model yolov8n_q8_320.vqf --yolo-image bus.jpg --yolo-conf 0.25 --yolo-iou 0.45 \
  2>&1 | grep -E "forward breakdown|image_total"
done