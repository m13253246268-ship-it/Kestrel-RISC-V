#!/bin/bash
# L2b：down sidecar 专家级预取（VLLM_DN_SAPF）与 SA 顺序（VLLM_SA_ORDER）A/B
# 基线 = 上一轮 gdn256 = 1627 ms/tok（同 env）
mkdir -p ~/kv/dk2 ; cd ~/kv/dk2 || exit 1
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
  grep -a '^\[STREAM\] text:' run_$P.log > text_$P.txt
  {
  echo "== $P n=$N [$X] $(cat rc_$P.txt) =="
  grep -a '\[VQF-SA\] enabled' err_$P.log | head -1
  grep -ao '\[STREAM\] decoded.*' run_$P.log
  echo "ids_md5 $(tr '\n' ' ' < ids_$P.txt | md5sum | cut -d' ' -f1) cnt=$(wc -l < ids_$P.txt)"
  echo "text_md5 $(md5sum < text_$P.txt | cut -d' ' -f1)"
  awk '{for(i=1;i<=NF;i++){split($i,a,"=");if(a[2]!=""&&a[1]~/^(ffn|pf|gk|gs|dk|da|rt|ot)$/){s[a[1]]+=a[2];c[a[1]]++}}}END{for(k in s)printf "%s=%.4f ",k,s[k]/c[k];print ""}' moe_$P.txt
  grep -a '\[SA-STAT\]' err_$P.log | tail -3
  } >> summary.txt
}
: > summary.txt
runone base256  256 ""
runone dnsapf256 256 "VLLM_DN_SAPF=1"
runone order256 256 "VLLM_SA_ORDER=1"
runone dnsapford256 256 "VLLM_DN_SAPF=1 VLLM_SA_ORDER=1"
echo ALLDONE > done.flag
