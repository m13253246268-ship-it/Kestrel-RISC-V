#!/bin/bash
# r10 d4：attention RVV 重建 + 同二进制 A/B（VLLM_ATTN_RVV=0/1）：位级（NLL/needle 贪心）+ ctx512 计时。
# 后台跑；再次运行即轮询。日志 /tmp/r10c.log
L=/tmp/r10c.log
if pgrep -f 'r10c_job' >/dev/null; then echo RUNNING; tail -8 $L; exit 0; fi
if grep -q R10C_END $L 2>/dev/null; then cat $L; exit 0; fi
cat > /tmp/r10c_job.sh <<'EOF'
set -u
cd /root/kv/kestrel_vllm_riscv
B=build-riscv/vllm_kestrel
sha256sum $B
echo "worker vec insns: $(objdump -d --no-show-raw-insn $B | awk '/<st_attn_x86_head_worker>:$/{p=1;next} p&&/^$/{p=0} p' | grep -cE '\sv[a-z0-9.]+\s')"
D=/root/kv/models/Qwen2.5-0.5B
DC=/root/kv/models/Qwen2.5-0.5B-col
uptime
for R in 0 1; do
  echo "----- nll RVV=$R -----"
  env VLLM_ATTN_RVV=$R VLLM_Q4_RVV=3 VLLM_STREAM_NLL=1 VLLM_THREADS=8 OMP_NUM_THREADS=8 timeout 3600 $B --stream-test \
    --model "$D" --threads 8 2>&1 | grep -E 'NLL_MEAN'
  echo "----- sweep RVV=$R -----"
  env VLLM_ATTN_RVV=$R VLLM_Q4_RVV=3 VLLM_THREADS=8 OMP_NUM_THREADS=8 timeout 3600 $B --needle-sweep \
    --longctx-ctx 1024 --longctx-gen 32 --model "$D" --threads 8 > /tmp/r10c_sw_$R.log 2>&1
  grep -E 'Needle retrieved|Answer PPL|Greedy' /tmp/r10c_sw_$R.log | sed 's/  F32=.*//' | md5sum
  grep -E 'Answer PPL' /tmp/r10c_sw_$R.log | sed 's/  F32=.*//' | tr -s ' ' | tr '\n' ' '; echo
done
for RND in 1 2 3; do
for M in lane0 lane1 colb0 colb1; do
  case $M in
    lane0) DD=$D; R=0;; lane1) DD=$D; R=1;;
    colb0) DD=$DC; R=0;; colb1) DD=$DC; R=1;;
  esac
  printf '%-6s r%d load=%s | ' $M $RND "$(cut -d' ' -f1 /proc/loadavg)"
  env VLLM_ATTN_RVV=$R VLLM_Q4_RVV=3 VLLM_THREADS=8 OMP_NUM_THREADS=8 timeout 900 \
    $B --stream-test --model "$DD" --stream-ctx 512 --stream-n 16 --threads 8 --warmup 2>&1 \
    | grep -E '^\[PREFILL-TIMING\]|^\[STREAM\] (prefill|decoded)' | sed 's/OTHER=.*//' | tr '\n' ' ' | tr -s ' ' | cut -c1-400
  echo
done
done
echo R10C_END
EOF
nohup bash /tmp/r10c_job.sh > $L 2>&1 &
echo STARTED
