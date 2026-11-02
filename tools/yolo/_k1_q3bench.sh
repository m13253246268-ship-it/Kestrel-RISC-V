#!/bin/bash
# 千问3 四档统一评测（dense 0.6B/1.7B/4B + MoE 30B-A3B）。日志 /tmp/q3bench.log
L=/tmp/q3bench.log
if pgrep -f 'q3bench_job' >/dev/null; then echo RUNNING; tail -10 $L; exit 0; fi
if grep -q Q3BENCH_END $L 2>/dev/null; then cat $L; exit 0; fi
cat > /tmp/q3bench_job.sh <<'EOF'
set -u
B=/root/kv/kestrel_vllm_riscv/build-riscv/vllm_kestrel
run() { # run <tag> <modeldir> <ctx> <n> <extra-env...>
  local TAG=$1 D=$2 CTX=$3 N=$4; shift 4
  printf '%-12s ctx=%-4s load=%s | ' "$TAG" "$CTX" "$(cut -d' ' -f1 /proc/loadavg)"
  env "$@" VLLM_THREADS=8 OMP_NUM_THREADS=8 timeout 3600 \
    $B --stream-test --model "$D" --stream-ctx "$CTX" --stream-n "$N" --threads 8 --warmup 2>&1 \
    | grep -E '^\[PREFILL-TIMING\]|^\[STREAM\] (prefill|decoded)' | sed 's/OTHER=.*//' | tr '\n' ' ' | tr -s ' ' | cut -c1-260
  echo
}
for R in 1 2 3; do
  echo "----- round $R -----"
  for M in Qwen3-0.6B Qwen3-1.7B Qwen3-4B; do
    run $M /root/kv/models/$M 32 16 VLLM_Q4_RVV=3
    run $M /root/kv/models/$M 128 16 VLLM_Q4_RVV=3
  done
  run Qwen3-30B /root/kv/models/Qwen3-30B-A3B 32 4 VLLM_VQF_STREAM=2 VLLM_DN_EC=1 VLLM_Q4_RVV=3
done
uptime
echo Q3BENCH_END
EOF
nohup bash /tmp/q3bench_job.sh > $L 2>&1 &
echo STARTED
