#!/bin/bash
# C-1a：per-row scale 权重 vs 原 per-block 权重的 Part F 对比（都用 lane 路径，激活 f32）
# 每臂完整输出存 /tmp/pf2_<tag>.log，摘要不截断。
set -u
D=/root/kv/models/Qwen2.5-0.5B
DR=/root/kv/models/Qwen2.5-0.5B-rowq4
B=/root/kv/kestrel_vllm_riscv/build-riscv/vllm_kestrel
uptime
sha256sum "$B" "$DR/model.vqf"
run() { # $1 tag $2 dir
  echo "----- $1 -----"
  env VLLM_Q4_RVV=3 VLLM_THREADS=8 OMP_NUM_THREADS=8 \
    timeout 1800 $B --longctx-quality --longctx-ctx 1024 --longctx-gen 64 \
       --model "$2" --threads 8 > /tmp/pf2_$1.log 2>&1
  echo "rc=$?"
  grep -E 'Needle retrieved|Answer PPL|Greedy|Decode ' /tmp/pf2_$1.log
}
run perblock "$D"
run perrow   "$DR"
echo PF2_DONE
