#!/bin/bash
# P0 分桶：colb t8/t4，ctx=32/128，prefill [PREFILL-TIMING/KERNELS]（取最后一组）+ decode [DEC-SINGLE] 均值。
# 后台跑；再次运行即轮询。日志 /tmp/p0prof.log
L=/tmp/p0prof.log
if pgrep -f 'p0prof_job' >/dev/null; then echo RUNNING; tail -8 $L; exit 0; fi
if grep -q P0PROF_END $L 2>/dev/null; then cat $L; exit 0; fi
cat > /tmp/p0prof_job.sh <<'EOF'
set -u
R=/root/kv/models
B=/root/kv/kestrel_vllm_riscv/build-riscv/vllm_kestrel
COLB="VLLM_Q4_RVV=3 VLLM_Q4_INT=1 VLLM_Q4_INTB=1"
sha256sum $B
run() { # model ctx threads
  local O=/tmp/p0_$1_$2_t$3.log
  env $COLB VLLM_DEC_PROF=1 VLLM_THREADS=$3 OMP_NUM_THREADS=$3 timeout 1800 \
    $B --stream-test --model "$R/$1-col" --stream-ctx $2 --stream-n 16 --threads $3 --warmup > $O 2>&1
  echo "== $1 ctx=$2 t$3 rc=$? load=$(cut -d' ' -f1 /proc/loadavg)"
  grep -E '^\[STREAM\] (prefill|decoded)' $O | tr '\n' ' '; echo
  grep '\[PREFILL-TIMING\]' $O | tail -1
  grep '\[PREFILL-KERNELS\]' $O | tail -1
  grep '\[DEC-SINGLE\]' $O | tail -12 | awk '{
    for(i=2;i<=NF;i++){split($i,kv,"="); if(kv[1]!="seq"){s[kv[1]]+=kv[2]; o[i]=kv[1]}} n++}
    END{printf "  DEC avg(last %d):", n; for(i=2;i<=NF;i++) if(o[i]!="") printf " %s=%.2f", o[i], s[o[i]]/n; print ""}'
}
for M in Qwen3-0.6B Qwen3-1.7B Qwen3-4B; do
  for X in 32 128; do run $M $X 8; done
  run $M 128 4
done
uptime
echo P0PROF_END
EOF
nohup bash /tmp/p0prof_job.sh > $L 2>&1 &
echo STARTED
