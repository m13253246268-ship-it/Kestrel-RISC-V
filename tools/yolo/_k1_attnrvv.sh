#!/bin/bash
# r10 d4：attention RVV 核位级对拍 + 单核微基准（源码已 PUT 到 /tmp/bench_attnrvv.c）
cd /tmp
gcc -O2 -march=rv64gcv -mabi=lp64d -ffp-contract=off bench_attnrvv.c -o bench_attnrvv -lm || exit 1
echo "load=$(cut -d' ' -f1 /proc/loadavg)"
taskset -c 7 ./bench_attnrvv; echo "rc=$?"
echo ATTNRVV_END
