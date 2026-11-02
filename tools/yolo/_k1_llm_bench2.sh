#!/bin/bash
# Online K1: deeper sample for 8 threads + context scaling.
set -u
D=/root/kv/models/Qwen2.5-0.5B
B=/root/kv/kestrel_vllm_riscv/build-riscv/vllm_kestrel

run() { # $1=threads $2=ctx $3=gen
  VLLM_Q4_RVV=3 VLLM_THREADS=$1 OMP_NUM_THREADS=$1 \
    $B --stream-test --model "$D" --stream-ctx $2 --stream-n $3 --threads $1 --warmup 2>&1 \
  | grep -E '^\[STREAM\] (prefill|decoded)' | sed 's/\[STREAM\] //'
}

echo "########## ctx=32 gen=16, t8, 6 rounds ##########"
for r in $(seq 1 6); do
  printf "r%d: " "$r"; run 8 32 16 | tr '\n' ' '; echo
done

echo "########## ctx scaling, t8, gen=16 ##########"
for C in 128 512; do
  printf "ctx=%s: " "$C"; run 8 $C 16 | tr '\n' ' '; echo
done

echo "########## load avg ##########"
cut -d' ' -f1-3 /proc/loadavg
