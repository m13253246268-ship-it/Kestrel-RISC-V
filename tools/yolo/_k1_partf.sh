#!/bin/bash
# Part F 质量门：lane f32 基线 vs col(mode3) vs colb(INTB=1)
# 每臂完整输出存 /tmp/partf_<tag>.log，摘要不截断。
set -u
D=/root/kv/models/Qwen2.5-0.5B
DC=/root/kv/models/Qwen2.5-0.5B-col
B=/root/kv/kestrel_vllm_riscv/build-riscv/vllm_kestrel
uptime
sha256sum "$B"
run() { # $1 tag $2 dir $3 env
  echo "----- $1 -----"
  env $3 VLLM_THREADS=8 OMP_NUM_THREADS=8 \
    timeout 1800 $B --longctx-quality --longctx-ctx 1024 --longctx-gen 64 \
       --model "$2" --threads 8 > /tmp/partf_$1.log 2>&1
  echo "rc=$?"
  grep -E 'Needle retrieved|Answer PPL|Greedy|Decode |WARN' /tmp/partf_$1.log
}
run lane "$D"  "VLLM_Q4_RVV=3"
run col  "$DC" "VLLM_Q4_RVV=3 VLLM_Q4_INT=1"
run colb "$DC" "VLLM_Q4_RVV=3 VLLM_Q4_INT=1 VLLM_Q4_INTB=1"
echo PF_DONE
