#!/bin/bash
# 云端 K1 MoE 最终基准：最优档 n=64/128/256 + def256 首尾基线
mkdir -p /root/kv/final ; cd /root/kv/final || exit 1
BN=/root/kv/kestrel_vllm_riscv/build-riscv/vllm_kestrel
MODEL=/root/kv/models/Qwen3-30B-A3B-col
OPT="VLLM_GU_LANE_PATH=$MODEL/gu_lane.bin VLLM_DN_LANE_PATH=$MODEL/dn_lane.bin VLLM_DN_SAPF=1"
runone() {
  P=$1; N=$2; X=$3
  env OMP_NUM_THREADS=8 VLLM_THREADS=8 VLLM_VQF_STREAM=12 VLLM_DN_EC=1 \
      VLLM_MOE_PROF=1 VLLM_SA_STAT=1 VLLM_SA=1 $X \
      timeout 9000 $BN --stream-test --model $MODEL --stream-n $N --threads 8 \
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
  } >> summary.txt
}
: > summary.txt
runone def256a 256 ""
runone opt64    64  "$OPT"
runone opt128   128 "$OPT"
runone opt256   256 "$OPT"
runone def256b 256 ""
echo ALLDONE > done.flag
