#!/bin/bash
# IME 方向恢复前置：多位置 needle（--needle-sweep）+ 语料级 NLL（--stream-test + VLLM_STREAM_NLL=1）
# lane / col / colb 三臂。后台跑；再次运行本脚本即轮询。日志 /tmp/qual.log，每臂完整输出 /tmp/qual_<tag>.log
L=/tmp/qual.log
if pgrep -f 'qual_job' >/dev/null; then echo RUNNING; tail -20 $L; exit 0; fi
if grep -q QUAL_END $L 2>/dev/null; then cat $L; exit 0; fi
cat > /tmp/qual_job.sh <<'EOF'
set -u
D=/root/kv/models/Qwen2.5-0.5B
DC=/root/kv/models/Qwen2.5-0.5B-col
B=/root/kv/kestrel_vllm_riscv/build-riscv/vllm_kestrel
uptime; sha256sum "$B"
sw() { # $1 tag $2 dir $3 env
  echo "----- sweep $1 -----"
  env $3 VLLM_THREADS=8 OMP_NUM_THREADS=8 timeout 3600 $B --needle-sweep \
    --longctx-ctx 1024 --longctx-gen 64 --model "$2" --threads 8 > /tmp/qual_sw_$1.log 2>&1
  echo "rc=$?"
  grep -E 'Needle retrieved|Answer PPL|Greedy|pos|WARN|FAIL' /tmp/qual_sw_$1.log | head -60
}
nl() { # $1 tag $2 dir $3 env
  echo "----- nll $1 -----"
  env $3 VLLM_STREAM_NLL=1 VLLM_THREADS=8 OMP_NUM_THREADS=8 timeout 3600 $B --stream-test \
    --model "$2" --threads 8 > /tmp/qual_nl_$1.log 2>&1
  echo "rc=$?"
  grep -E 'NLL|FAIL|WARN' /tmp/qual_nl_$1.log | head -40
}
nl lane "$D"  "VLLM_Q4_RVV=3"
nl col  "$DC" "VLLM_Q4_RVV=3 VLLM_Q4_INT=1"
sw lane "$D"  "VLLM_Q4_RVV=3"
sw col  "$DC" "VLLM_Q4_RVV=3 VLLM_Q4_INT=1"
nl colb "$DC" "VLLM_Q4_RVV=3 VLLM_Q4_INT=1 VLLM_Q4_INTB=1"
echo QUAL_END
EOF
nohup bash /tmp/qual_job.sh > $L 2>&1 &
echo STARTED
