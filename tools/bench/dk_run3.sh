#!/bin/bash
# L2b 复核：base vs DN_SAPF 在 n=128 上交错 A/B（2 轮），排除单次序偏差
mkdir -p ~/kv/dk3 ; cd ~/kv/dk3 || exit 1
BN=~/kv/kestrel_src/build-riscv/vllm_kestrel
BASE="VLLM_GU_LANE=1 VLLM_GU_LANE_PATH=/home/haliu/kv/moe30b/gu_lane.bin VLLM_DN_LANE=1 VLLM_DN_LANE_PATH=/home/haliu/kv/moe30b/dn_lane.bin"
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
  grep -ao '\[STREAM\] decoded.*' run_$P.log
  echo "ids_md5 $(tr '\n' ' ' < ids_$P.txt | md5sum | cut -d' ' -f1)"
  awk '{for(i=1;i<=NF;i++){split($i,a,"=");if(a[2]!=""&&a[1]~/^(ffn|pf|gk|gs|dk|da|rt|ot)$/){s[a[1]]+=a[2];c[a[1]]++}}}END{for(k in s)printf "%s=%.4f ",k,s[k]/c[k];print ""}' moe_$P.txt
  } >> summary.txt
}
: > summary.txt
runone b128r1 128 ""
runone s128r1 128 "VLLM_DN_SAPF=1"
runone s128r2 128 "VLLM_DN_SAPF=1"
runone b128r2 128 ""
echo ALLDONE > done.flag
