#!/bin/bash
# 板端构建并后台运行 bench_f32pre；再次运行即轮询。
L=/tmp/f32pre.log
if pgrep -f 'f32pre_job' >/dev/null; then echo RUNNING; tail -20 $L; exit 0; fi
if grep -q F32PRE_END $L 2>/dev/null; then cat $L; exit 0; fi
cat > /tmp/f32pre_job.sh <<'EOF'
set -u
cd /root/kv/kestrel_vllm_riscv/tools/yolo || exit 1
rm -f /tmp/bench_f32pre
gcc -O3 -march=rv64gcv -mabi=lp64d -ffp-contract=off -fopenmp \
    bench_f32pre.c -o /tmp/bench_f32pre -lm > /tmp/bench_f32pre.cc.log 2>&1 \
    || { grep -E 'error' /tmp/bench_f32pre.cc.log | head -20; echo BUILD_FAIL; echo F32PRE_END; exit 1; }
sha256sum bench_f32pre.c /tmp/bench_f32pre
uptime
taskset -c 0-7 timeout 1800 /tmp/bench_f32pre 8
echo "rc=$?"
uptime
echo F32PRE_END
EOF
rm -f $L
nohup bash /tmp/f32pre_job.sh > $L 2>&1 &
echo STARTED
