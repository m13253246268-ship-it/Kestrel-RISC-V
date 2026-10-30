#!/bin/bash
# V50 Phase 0：固化 CHW 位级锚点（改造前的经典参照）
#   raw = --yolo-raw 输出的原始 float（84*8400=705600 for 640；320 不同）
# 之后全 HWC 改造必须逐字节坍缩回这两个文件。
set -u
cd ~/kv/yolo
B=~/kv/kestrel_src/build-riscv/vllm_kestrel

for M in yolov8n_q8.vqf yolov8n_q8_320.vqf; do
  for R in 1 2; do
    VLLM_THREADS=8 OMP_NUM_THREADS=8 $B --yolo-model "$M" --yolo-raw raw.f32 \
      --yolo-raw-out "/tmp/anchor_${M%.vqf}_r$R.f32" >/dev/null 2>&1
    echo "  $M r$R rc=$? bytes=$(stat -c%s "/tmp/anchor_${M%.vqf}_r$R.f32" 2>/dev/null)"
  done
  if cmp -s "/tmp/anchor_${M%.vqf}_r1.f32" "/tmp/anchor_${M%.vqf}_r2.f32"; then
    echo "  $M: ANCHOR_REPRODUCIBLE  sha256=$(sha256sum /tmp/anchor_${M%.vqf}_r1.f32 | cut -c1-16)"
  else
    echo "  $M: ANCHOR_NOT_REPRODUCIBLE"
  fi
  cp "/tmp/anchor_${M%.vqf}_r1.f32" "$HOME/kv/yolo/anchor_${M%.vqf}.f32"
done
echo "已落盘：$HOME/kv/yolo/anchor_yolov8n_q8.f32  anchor_yolov8n_q8_320.f32"
