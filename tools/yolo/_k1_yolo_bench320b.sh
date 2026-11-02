#!/bin/bash
# Online K1: robust 320x320 steady-state. Reports min / p25 / median / max of a metric.
set -u
cd /root/kv/kestrel_vllm_riscv/tools/yolo
B=../../build-riscv/vllm_kestrel
N=${1:-20}
ROUNDS=${2:-3}

stats() { # $1=tag  (values on stdin)
  local tag=$1
  local f=/tmp/_vals_$tag.txt
  cat > $f
  local n=$(wc -l < $f)
  if [ "$n" -eq 0 ]; then echo "  $tag: (no data)"; return; fi
  sort -n $f > $f.s
  local mn=$(head -1 $f.s)
  local p25=$(sed -n "$(( n/4 + 1 ))p" $f.s)
  local md=$(sed -n "$(( (n+1)/2 ))p" $f.s)
  local mx=$(tail -1 $f.s)
  printf "  %-8s n=%2d  min=%6.1f  p25=%6.1f  med=%6.1f  max=%6.1f ms\n" "$tag" "$n" "$mn" "$p25" "$md" "$mx"
}

for r in $(seq 1 $ROUNDS); do
  echo "########## round $r (N=$N, threads=8) ##########"
  VLLM_THREADS=8 OMP_NUM_THREADS=8 VLLM_YOLO_CPU_TIMING=1 VLLM_YOLO_FRAMES=$N \
  $B --yolo-model yolov8n_q8_320.vqf --yolo-image test.jpg --yolo-conf 0.25 --yolo-iou 0.45 \
  > /tmp/r_$r.log 2>&1

  grep "image_total" /tmp/r_$r.log | tail -n +2 \
    | sed -E 's/.*image_total=([0-9.]+)ms.*/\1/' | stats image
  grep "forward breakdown" /tmp/r_$r.log | tail -n +2 \
    | sed -E 's/.*total=([0-9.]+)ms.*/\1/' | stats forward
  grep "forward breakdown" /tmp/r_$r.log | tail -n +2 \
    | sed -E 's/.*im2col=([0-9.]+)ms.*/\1/' | stats im2col
  grep "forward breakdown" /tmp/r_$r.log | tail -n +2 \
    | sed -E 's/.*conv=([0-9.]+)ms.*/\1/' | stats conv
  echo "  load: $(cut -d' ' -f1-3 /proc/loadavg)"
done
