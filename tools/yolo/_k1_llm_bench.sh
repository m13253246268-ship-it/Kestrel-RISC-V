#!/bin/bash
# Online K1: Qwen2.5-0.5B Q4 lane-major LLM benchmark (stream-test caliber, same as V34).
set -u
D=/root/kv/models/Qwen2.5-0.5B
B=/root/kv/kestrel_vllm_riscv/build-riscv/vllm_kestrel
CTX=${1:-32}
GEN=${2:-16}
ROUNDS=${3:-3}

echo "=== model ==="
ls -l $D/model.vqf

for T in 8 4; do
  for r in $(seq 1 $ROUNDS); do
    OUT=$(VLLM_Q4_RVV=3 VLLM_THREADS=$T OMP_NUM_THREADS=$T \
      $B --stream-test --model "$D" --stream-ctx $CTX --stream-n $GEN --threads $T --warmup 2>&1)
    PF=$(echo "$OUT" | grep -oE 'prefill [0-9]+ tokens in [0-9.]+s \([0-9]+ tok/s\)' | head -1)
    DC=$(echo "$OUT" | grep -oE 'decoded [0-9]+ tokens, [0-9]+ ms/tok \([0-9.]+ tok/s\)' | head -1)
    echo "  t$T r$r  $PF | $DC"
  done
done
