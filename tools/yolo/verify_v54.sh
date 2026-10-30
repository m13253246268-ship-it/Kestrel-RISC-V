#!/bin/bash
# V54 收尾验收：m0 内核 + t_buf（去 VLA）后的位级与稳态性能
set -u
cd ~/kv/yolo || exit 1
B=~/kv/kestrel_src/build-riscv/vllm_kestrel
[ -x "$B" ] || { echo "FATAL: 缺 $B"; exit 1; }
[ -f raw.f32 ] || { echo "FATAL: 缺 raw.f32"; exit 1; }
export OMP_NUM_THREADS=8 VLLM_THREADS=8
rc=0

echo "===== 1. 与锚点对拍（应 ANCHOR_MATCH）====="
for M in yolov8n_q8 yolov8n_q8_320; do
  A="anchor_${M}.f32"
  [ -f "$A" ] || { echo "  $M: 无锚点，跳过"; continue; }
  $B --yolo-model "${M}.vqf" --yolo-raw raw.f32 --yolo-raw-out "/tmp/f_${M}.f32" >/dev/null 2>&1
  if cmp -s "$A" "/tmp/f_${M}.f32"; then echo "  $M: ANCHOR_MATCH"; else echo "  $M: ANCHOR_DIFF"; rc=1; fi
done

echo
echo "===== 2. 320 稳态（6 帧，取第 2..6 帧平均）====="
for r in 1 2 3; do
  VLLM_YOLO_CPU_TIMING=1 VLLM_YOLO_FRAMES=6 $B --yolo-model yolov8n_q8_320.vqf \
    --yolo-image bus.jpg --yolo-conf 0.25 --yolo-iou 0.45 2>/tmp/f_$r.log >/dev/null
  avg=$(grep -o 'image_total=[0-9.]*ms' /tmp/f_$r.log | sed 's/[^0-9.]//g' | tail -n +2 | \
        awk '{s+=$1;n++} END{if(n>0) printf "%.1f", s/n; else print "NA"}')
  conv=$(grep -o 'conv=[0-9.]*ms' /tmp/f_$r.log | sed 's/[^0-9.]//g' | tail -1)
  echo "  run$r  稳态均价=${avg}ms  conv=${conv}ms"
done

echo
echo "VERDICT: $([ $rc -eq 0 ] && echo ANCHOR_MATCH || echo FAIL)"
exit $rc
