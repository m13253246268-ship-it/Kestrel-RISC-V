#!/bin/bash
set -u
D=/root/kv/models/Qwen2.5-0.5B
B=/root/kv/kestrel_vllm_riscv/build-riscv/vllm_kestrel

echo "=== load ==="
cut -d' ' -f1-3 /proc/loadavg

echo "=== verbose run (ctx=32, gen=32) : text sanity ==="
VLLM_Q4_RVV=3 VLLM_THREADS=8 OMP_NUM_THREADS=8 \
  $B --stream-test --model "$D" --stream-ctx 32 --stream-n 32 --threads 8 --warmup 2>&1 \
  | grep -E '^\[STREAM\]|^\[TOK\]' | tail -12

echo "=== external baseline reachability ==="
for h in pypi.tuna.tsinghua.edu.cn mirrors.aliyun.com github.com; do
  code=$(curl -sS -m 8 -o /dev/null -w '%{http_code}' "https://$h/" 2>/dev/null || echo FAIL)
  echo "  $h -> $code"
done
