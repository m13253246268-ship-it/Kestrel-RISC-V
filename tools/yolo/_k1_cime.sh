#!/bin/bash
# Phase B-0：col 布局直喂 IME 的真实倍数（与 _k1_cb.sh 同编译选项）
set -u
cd /root/kv/kestrel_vllm_riscv/tools/yolo || exit 1
rm -f /tmp/bench_colime
gcc -O3 -march=rv64gcv -mabi=lp64d -ffp-contract=off -fopenmp \
    bench_colime.c -o /tmp/bench_colime -lm > /tmp/bench_colime.cc.log 2>&1 \
    || { grep -E 'error' /tmp/bench_colime.cc.log | head -20; echo BUILD_FAIL; exit 1; }
sha256sum bench_colime.c /tmp/bench_colime
uptime
taskset -c 0-7 timeout 900 /tmp/bench_colime 8
echo CIME_DONE
