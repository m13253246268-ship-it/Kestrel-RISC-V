#!/bin/bash
set -u
cd /root/kv
echo "=== build ==="
gcc -O2 -march=rv64gcv -mabi=lp64d -ffp-contract=off -fopenmp \
    bench_ime_llm.c -o bench_ime_llm -lm 2>&1 | head -30
echo "rc=$?"
ls -l bench_ime_llm 2>/dev/null || exit 1
echo "=== run ==="
./bench_ime_llm 3 2>&1
