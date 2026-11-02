#!/bin/bash
# A-1b 定位：(1) objdump rv_q4_colbt_worker 指令画像 (2) 复测 colb ctx128 decode
set -u
B=/root/kv/kestrel_vllm_riscv/build-riscv/vllm_kestrel
D=/root/kv/models/Qwen2.5-0.5B
DC=/root/kv/models/Qwen2.5-0.5B-col

echo "===== objdump rv_q4_colbt_worker ====="
objdump -d --no-show-raw-insn "$B" | awk '/<rv_q4_colbt_worker>:/{p=1} p&&/^$/{exit} p' > /tmp/cb.s
wc -l /tmp/cb.s
echo "--- 指令频次 top20 ---"
awk 'NF>=2{print $2}' /tmp/cb.s | sort | uniq -c | sort -rn | head -20
echo "--- 关键指令计数 ---"
for m in vle8 vslideup vwmacc vfwcvt vle32 vse32 flh fcvt.s.h vand vsrl vsub; do
  printf '%-12s %s\n' "$m" "$(grep -c "$m" /tmp/cb.s)"
done
echo "--- 栈访问（facc 往返）---"
grep -cE '\b(sd|ld)\b' /tmp/cb.s
grep -E 'addi\s+sp,sp' /tmp/cb.s | head -3

echo "===== 复测 colb ctx128 decode（3 轮，只跑 decode 观察）====="
for i in 1 2 3; do
  uptime | sed 's/^/  load: /'
  for M in col colb; do
    if [ $M = col ]; then EX="VLLM_Q4_RVV=3 VLLM_Q4_INT=1"; else EX="VLLM_Q4_RVV=3 VLLM_Q4_INT=1 VLLM_Q4_INTB=1"; fi
    printf '  %-5s ' "$M"
    env $EX VLLM_THREADS=8 OMP_NUM_THREADS=8 \
      $B --stream-test --model "$DC" --stream-ctx 128 --stream-n 16 --threads 8 --warmup 2>&1 \
      | grep -E '^\[STREAM\] decoded' | tail -1
  done
done
echo A1B_DONE
