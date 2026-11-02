#!/bin/bash
# 复测 V59 的 bench_ime_llm（RVV 臂真实吞吐），并对照 bench_colbt 的 S0/S1
set -u
cd /root/kv/kestrel_vllm_riscv/tools/yolo
echo "===== bench_ime_llm（-O2，同 V59 口径）====="
gcc -O2 -march=rv64gcv -mabi=lp64d -ffp-contract=off -fopenmp \
    bench_ime_llm.c -o /tmp/bench_imellm -lm 2>&1 | grep -E 'error' | head
ls -l /tmp/bench_imellm || exit 1
uptime
timeout 600 /tmp/bench_imellm 2>&1 | tail -40
echo IME_DONE
