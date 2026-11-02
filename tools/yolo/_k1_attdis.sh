#!/bin/bash
# d4：检查 prefill attention 函数在 RISC-V 二进制里是否被自动向量化
B=/root/kv/kestrel_vllm_riscv/build-riscv/vllm_kestrel
[ -s /tmp/vk.dis ] || objdump -d --no-show-raw-insn $B > /tmp/vk.dis
nm $B | grep -iE 'attn' | head -30
grep -nE '<st_attn_batched_packed_neon>:' /tmp/vk.dis | head -2
# 调用目标（找 worker 实际符号）
awk 'index($0,"<st_attn_batched_packed_neon>:"){p=1;next} p&&NF==0{exit} p' /tmp/vk.dis | grep -oE '<[^>]+>' | sort | uniq -c | sort -rn | head -15
for F in st_attn_x86_head_worker; do
  awk -v f="<$F>:" 'index($0,f){p=1;next} p&&NF==0{exit} p' /tmp/vk.dis > /tmp/f.dis
  T=$(wc -l < /tmp/f.dis)
  echo "== $F total=$T"
  awk -F'\t' '{print $2}' /tmp/f.dis | awk '{print $1}' | sort | uniq -c | sort -rn | head -40
  grep -E 'jal|expf' /tmp/f.dis | head
done
