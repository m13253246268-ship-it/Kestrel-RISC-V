#!/bin/bash
# L-IME-001 Phase A-0：mode2 lane f32（默认）vs mode3 col int8（T4-1）同机同会话 A/B
set -u
D=/root/kv/models/Qwen2.5-0.5B
DC=/root/kv/models/Qwen2.5-0.5B-col
B=/root/kv/kestrel_vllm_riscv/build-riscv/vllm_kestrel
C=/root/kv/kestrel_vllm_riscv/vqf_convert/vqf_conv
uptime
if [ ! -f "$DC/model.vqf" ]; then
  mkdir -p "$DC"
  for f in "$D"/*; do
    n=$(basename "$f"); [ "$n" = model.vqf ] && continue
    ln -sf "$f" "$DC/$n"
  done
  echo "=== convert col (VLLM_Q4_COL=1) ==="
  VLLM_Q4_COL=1 $C --model "$D" --convert-vqf "$DC/model.vqf" --wmode q4 2>&1 | grep -E 'repack gates|sum=|error|fail' | head
fi
ls -l "$D/model.vqf" "$DC/model.vqf"

run() { # $1 tag  $2 dir  $3 ctx  extra env via $4
  echo "----- $1 ctx=$3 -----"
  env $4 VLLM_THREADS=8 OMP_NUM_THREADS=8 \
    $B --stream-test --model "$2" --stream-ctx $3 --stream-n 8 --threads 8 --warmup 2>&1 \
  | grep -E '^\[PREFILL-(TIMING|KERNELS)\]|^\[STREAM\] (prefill|decoded)' | tail -4
}
for C in 32 128; do
  for i in 1 2; do
    run lane "$D"  $C "VLLM_Q4_RVV=3"
    run col  "$DC" $C "VLLM_Q4_RVV=3 VLLM_Q4_INT=1"
  done
done

echo "=== greedy sanity (col int8) ==="
for M in lane col; do
  if [ $M = lane ]; then DD=$D; EX="VLLM_Q4_RVV=3"; else DD=$DC; EX="VLLM_Q4_RVV=3 VLLM_Q4_INT=1"; fi
  echo "--- $M ---"
  env $EX VLLM_THREADS=8 OMP_NUM_THREADS=8 \
    $B --stream-test --model "$DD" --stream-ctx 32 --stream-n 32 --threads 8 --warmup 2>&1 \
    | grep -E '^\[STREAM\] (text|out|gen)|^\[TOK|TOKIDS' | tail -4
done

echo "=== objdump rv_q4_lane_bt4_worker_m2 ==="
which objdump >/dev/null && objdump -d --no-show-raw-insn "$B" \
  | awk '/<rv_q4_lane_bt4_worker_m2>:/{p=1} p&&/^$/{exit} p' > /tmp/m2.s
wc -l /tmp/m2.s
awk '{print $2}' /tmp/m2.s | sort | uniq -c | sort -rn | head -30
