#!/bin/bash
# d0：decode 分桶实测（VLLM_DEC_PROF=1），0.6B/4B ctx32 n16 t8，各取均值（跳过首步）。
cd /root/kv/kestrel_vllm_riscv
B=build-riscv/vllm_kestrel; R=/root/kv/models
COLB="VLLM_Q4_RVV=3 VLLM_Q4_INT=1 VLLM_Q4_INTB=1"
for M in Qwen3-0.6B Qwen3-4B; do
  L=/tmp/d0_$M.log
  env $COLB VLLM_DEC_PROF=1 VLLM_THREADS=8 OMP_NUM_THREADS=8 \
    timeout 900 $B --stream-test --model $R/$M-col --stream-ctx 32 --stream-n 16 --threads 8 > $L 2>&1
  echo "== $M rc=$? load=$(cut -d' ' -f1 /proc/loadavg)"
  grep -a '\[STREAM\] decoded' $L
  grep -a '\[DEC-SINGLE\]' $L | tail -n +2 | awk '{
    for(i=1;i<=NF;i++){ if(split($i,kv,"=")==2 && kv[1]!="seq"){ s[kv[1]]+=kv[2]; if(!(kv[1] in o)){o[kv[1]]=++n; k[n]=kv[1]} } }
    c++ } END{ if(!c){print "no DEC-SINGLE"; exit}
    t=s["total"]; for(j=1;j<=n;j++){ x=s[k[j]]/c; printf "%s=%.2f(%.1f%%) ", k[j], x, (t>0?100*s[k[j]]/t:0) } printf "\nsteps=%d\n", c }'
  grep -a '\[DEC-SINGLE\]' $L | head -2
done
echo D0_END
