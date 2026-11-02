#!/bin/bash
# rowq4（per-row scale 权重，lane f32 激活）：语料级 NLL + 多位置 needle。后台跑；再次运行即轮询。
L=/tmp/qual2.log
if pgrep -f 'qual2_job' >/dev/null; then echo RUNNING; tail -20 $L; exit 0; fi
if grep -q QUAL2_END $L 2>/dev/null; then cat $L; exit 0; fi
cat > /tmp/qual2_job.sh <<'EOF'
set -u
D=/root/kv/models/Qwen2.5-0.5B-rowq4
B=/root/kv/kestrel_vllm_riscv/build-riscv/vllm_kestrel
uptime; sha256sum "$B"
echo "----- nll rowq4 -----"
env VLLM_Q4_RVV=3 VLLM_STREAM_NLL=1 VLLM_THREADS=8 OMP_NUM_THREADS=8 timeout 3600 $B --stream-test \
  --model "$D" --threads 8 > /tmp/qual_nl_rowq4.log 2>&1
echo "rc=$?"; grep -E 'NLL|FAIL|WARN' /tmp/qual_nl_rowq4.log | head -40
echo "----- sweep rowq4 -----"
env VLLM_Q4_RVV=3 VLLM_THREADS=8 OMP_NUM_THREADS=8 timeout 3600 $B --needle-sweep \
  --longctx-ctx 1024 --longctx-gen 64 --model "$D" --threads 8 > /tmp/qual_sw_rowq4.log 2>&1
echo "rc=$?"; grep -E 'S=1024 pos|Needle retrieved|Answer PPL|Greedy|FAIL' /tmp/qual_sw_rowq4.log | cut -c1-160
echo QUAL2_END
EOF
nohup bash /tmp/qual2_job.sh > $L 2>&1 &
echo STARTED
