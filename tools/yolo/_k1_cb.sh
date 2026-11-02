#!/bin/bash
# 板端构建并运行 bench_colbt（与 _k1_cime.sh 同编译选项）
set -u
cd /root/kv/kestrel_vllm_riscv/tools/yolo || exit 1
rm -f /tmp/bench_colbt
gcc -O3 -march=rv64gcv -mabi=lp64d -ffp-contract=off -fopenmp \
    bench_colbt.c -o /tmp/bench_colbt -lm > /tmp/bench_colbt.cc.log 2>&1 \
    || { grep -E 'error' /tmp/bench_colbt.cc.log | head -20; echo BUILD_FAIL; exit 1; }
sha256sum bench_colbt.c /tmp/bench_colbt
uptime
taskset -c 0-7 timeout 900 /tmp/bench_colbt 8
echo CB_DONE
