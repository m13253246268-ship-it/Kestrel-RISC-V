#!/bin/bash
# 分工归因：decode 逐 token 分项（VLLM_DEC_PROF=1）
# 口径说明：单 token [DEC-SINGLE] 不作性能结论（M8）；本脚本只取多 token 均值做**归因**，
#           正式数字仍以 [MOEPROF] 分项 + [STREAM] decoded 总值为准。
mkdir -p ~/kv/dkprof ; cd ~/kv/dkprof || exit 1
BN=~/kv/kestrel_src/build-riscv/vllm_kestrel
DNENV="VLLM_DN_LANE=1 VLLM_DN_LANE_PATH=/home/haliu/kv/moe30b/dn_lane.bin"
GLENV="VLLM_GU_LANE=1 VLLM_GU_LANE_PATH=/home/haliu/kv/moe30b/gu_lane.bin"
env OMP_NUM_THREADS=8 VLLM_THREADS=8 VLLM_VQF_STREAM=12 VLLM_DN_EC=1 VLLM_Q4_RVV=3 \
    VLLM_MOE_PROF=1 VLLM_SA_STAT=1 VLLM_SA=1 VLLM_DEC_PROF=1 $GLENV $DNENV \
    timeout 3000 $BN --stream-test --model ~/kv/moe30b --stream-n 96 --threads 8 \
    > prof.log 2> prof.err
echo "rc=$?"
grep -a '\[DEC-SINGLE\]' prof.err prof.log | tail -n +4 > ds.txt
wc -l < ds.txt
awk '{for(i=2;i<=NF;i++){split($i,a,"=");if(a[1]!="seq"&&a[2]!=""){s[a[1]]+=a[2];c[a[1]]++}}}
     END{for(k in s)printf "%s=%.4f ",k,s[k]/c[k];print ""}' ds.txt
grep -ao '\[STREAM\] decoded.*' prof.log
grep -ao '\[MOEPROF\].*' prof.log | tail -n +2 > moe.txt
awk '{for(i=1;i<=NF;i++){split($i,a,"=");if(a[2]!=""&&a[1]~/^(ffn|pf|gk|gs|dk|da|rt|ot)$/){s[a[1]]+=a[2];c[a[1]]++}}}END{for(k in s)printf "%s=%.4f ",k,s[k]/c[k];print ""}' moe.txt
