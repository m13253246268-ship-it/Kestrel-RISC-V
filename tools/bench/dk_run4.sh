#!/bin/bash
# L2c：pf 削减三方向 A/B（n=256）
#   base   = gdn + DN_SAPF（当前最好档，1500 ms/tok）
#   async  = + VLLM_SA_ASYNC=1（madvise 交后台线程）
#   pfdn   = + VLLM_SA_PF=dn（只预取 down，不预取 gate/up）
#   pfnone = + VLLM_SA_PF=none（完全不预取，全靠按需缺页）
#   base2  = base 复跑（首尾各一次，用于控制板端漂移）
mkdir -p ~/kv/dk4 ; cd ~/kv/dk4 || exit 1
BN=~/kv/kestrel_src/build-riscv/vllm_kestrel
BASE="VLLM_GU_LANE=1 VLLM_GU_LANE_PATH=/home/haliu/kv/moe30b/gu_lane.bin VLLM_DN_LANE=1 VLLM_DN_LANE_PATH=/home/haliu/kv/moe30b/dn_lane.bin VLLM_DN_SAPF=1"
runone() {
  P=$1; N=$2; X=$3
  env OMP_NUM_THREADS=8 VLLM_THREADS=8 VLLM_VQF_STREAM=12 VLLM_DN_EC=1 VLLM_Q4_RVV=3 \
      VLLM_MOE_PROF=1 VLLM_SA_STAT=1 VLLM_SA=1 $BASE $X \
      timeout 9000 $BN --stream-test --model ~/kv/moe30b --stream-n $N --threads 8 \
      > run_$P.log 2> err_$P.log
  echo "rc=$?" > rc_$P.txt
  grep -oE '[0-9]+\[MOEPROF\] tok1 ' run_$P.log | sed 's/\[MOEPROF\] tok1 //' > ids_$P.txt
  grep -o '\[MOEPROF\].*' run_$P.log > moe_$P.txt
  {
  echo "== $P n=$N [$X] $(cat rc_$P.txt) =="
  grep -a '\[VQF-SA\] enabled' err_$P.log | head -1
  grep -ao '\[STREAM\] decoded.*' run_$P.log
  echo "ids_md5 $(tr '\n' ' ' < ids_$P.txt | md5sum | cut -d' ' -f1)"
  awk '{for(i=1;i<=NF;i++){split($i,a,"=");if(a[2]!=""&&a[1]~/^(ffn|pf|gk|gs|dk|da|rt|ot)$/){s[a[1]]+=a[2];c[a[1]]++}}}END{for(k in s)printf "%s=%.4f ",k,s[k]/c[k];print ""}' moe_$P.txt
  grep -a '\[SA-STAT\]' err_$P.log | tail -1
  } >> summary.txt
}
: > summary.txt
runone base256   256 ""
runone async256  256 "VLLM_SA_ASYNC=1"
runone pfdn256   256 "VLLM_SA_PF=dn"
runone pfnone256 256 "VLLM_SA_PF=none"
runone base2_256 256 ""
echo ALLDONE > done.flag
