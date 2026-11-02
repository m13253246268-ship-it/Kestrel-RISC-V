#!/bin/bash
# L-IME-001 Phase A-1：lane f32 / col 原 mode3 / col+INTB（token 分块）同机 A/B
set -u
D=/root/kv/models/Qwen2.5-0.5B
DC=/root/kv/models/Qwen2.5-0.5B-col
B=/root/kv/kestrel_vllm_riscv/build-riscv/vllm_kestrel
uptime

echo "=== (a) Q4CHK bit-compare colbt vs rv_q4_col_tile ==="
env VLLM_Q4_RVV=3 VLLM_Q4_INT=1 VLLM_Q4_INTB=1 VLLM_Q4CHK=1 VLLM_THREADS=8 OMP_NUM_THREADS=8 \
  $B --stream-test --model "$DC" --stream-ctx 32 --stream-n 2 --threads 8 2>&1 \
  | grep -E '^\[Q4CHK-colbt\]' | head -8

run() { # $1 tag  $2 dir  $3 ctx  $4 env
  echo "----- $1 ctx=$3 -----"
  env $4 VLLM_THREADS=8 OMP_NUM_THREADS=8 \
    $B --stream-test --model "$2" --stream-ctx $3 --stream-n 8 --threads 8 --warmup 2>&1 \
  | grep -E '^\[PREFILL-(TIMING|KERNELS)\]|^\[STREAM\] (prefill|decoded)' | tail -4
}
echo "=== (b) A/B ==="
for C in 32 128; do
  for i in 1 2; do
    run lane "$D"  $C "VLLM_Q4_RVV=3"
    run col  "$DC" $C "VLLM_Q4_RVV=3 VLLM_Q4_INT=1"
    run colb "$DC" $C "VLLM_Q4_RVV=3 VLLM_Q4_INT=1 VLLM_Q4_INTB=1"
  done
done

echo "=== (c) greedy text ==="
for M in lane col colb; do
  case $M in
    lane) DD=$D;  EX="VLLM_Q4_RVV=3";;
    col)  DD=$DC; EX="VLLM_Q4_RVV=3 VLLM_Q4_INT=1";;
    colb) DD=$DC; EX="VLLM_Q4_RVV=3 VLLM_Q4_INT=1 VLLM_Q4_INTB=1";;
  esac
  echo "--- $M ---"
  env $EX VLLM_THREADS=8 OMP_NUM_THREADS=8 \
    $B --stream-test --model "$DD" --stream-ctx 32 --stream-n 32 --threads 8 --warmup 2>&1 \
    | grep -E '^\[STREAM\] text' | tail -2
done
echo A1_DONE
