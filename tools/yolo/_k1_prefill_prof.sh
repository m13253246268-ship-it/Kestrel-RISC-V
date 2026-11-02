#!/bin/bash
# Online K1: prefill 分解（GEMM vs ATTN vs OTHER），Qwen2.5-0.5B Q4 lane
set -u
D=/root/kv/models/Qwen2.5-0.5B
B=/root/kv/kestrel_vllm_riscv/build-riscv/vllm_kestrel
for C in 32 128 512; do
  echo "===== ctx=$C ====="
  VLLM_Q4_RVV=3 VLLM_THREADS=8 OMP_NUM_THREADS=8 \
    $B --stream-test --model "$D" --stream-ctx $C --stream-n 8 --threads 8 --warmup 2>&1 \
  | grep -E '^\[PREFILL-(TIMING|KERNELS)\]|^\[STREAM\] (prefill|decoded)' \
  | tail -4
done
