#!/bin/bash
# L2：DN_LANE（down lane-tile sidecar）验收
#   def = 当前最优档（EC + RVV3，down 走 rv_q4_rows_par）
#   dk  = + VLLM_DN_LANE=1（down 走 lane-tile sidecar）
#   gdn = + GU_LANE + DN_LANE（合并档）
mkdir -p ~/kv/dk ; cd ~/kv/dk || exit 1
BN=~/kv/kestrel_src/build-riscv/vllm_kestrel
DNENV="VLLM_DN_LANE=1 VLLM_DN_LANE_PATH=/home/haliu/kv/moe30b/dn_lane.bin"
GLENV="VLLM_GU_LANE=1 VLLM_GU_LANE_PATH=/home/haliu/kv/moe30b/gu_lane.bin"
snap() { { date +%s.%N; grep -E '^(pgpgin|pgmajfault) ' /proc/vmstat; grep ' mmcblk2 ' /proc/diskstats; } > "$1.snap"; }
runone() {
  P=$1; N=$2; X=$3
  snap pre_$P
  env OMP_NUM_THREADS=8 VLLM_THREADS=8 VLLM_VQF_STREAM=12 VLLM_DN_EC=1 VLLM_Q4_RVV=3 \
      VLLM_MOE_PROF=1 VLLM_SA_STAT=1 VLLM_SA=1 $X \
      timeout 9000 $BN --stream-test --model ~/kv/moe30b --stream-n $N --threads 8 \
      > run_$P.log 2> err_$P.log
  echo "rc=$?" > rc_$P.txt
  snap post_$P
  grep -oE '[0-9]+\[MOEPROF\] tok1 ' run_$P.log | sed 's/\[MOEPROF\] tok1 //' > ids_$P.txt
  grep -o '\[MOEPROF\].*' run_$P.log > moe_$P.txt
  grep -a '^\[STREAM\] text:' run_$P.log > text_$P.txt
  {
  echo "== $P n=$N $X $(cat rc_$P.txt) =="
  grep -a '\[DN_LANE\]\|\[GU_LANE\]' run_$P.log err_$P.log | head -3
  grep -ao '\[STREAM\] decoded.*' run_$P.log
  echo "ids_md5 $(tr '\n' ' ' < ids_$P.txt | md5sum | cut -d' ' -f1) cnt=$(wc -l < ids_$P.txt)"
  echo "text_md5 $(md5sum < text_$P.txt | cut -d' ' -f1)"
  awk '{for(i=1;i<=NF;i++){split($i,a,"=");if(a[2]!=""&&a[1]~/^(ffn|pf|gk|gs|dk|da|rt|ot)$/){s[a[1]]+=a[2];c[a[1]]++}}}END{for(k in s)printf "%s=%.4f ",k,s[k]/c[k];print ""}' moe_$P.txt
  } >> summary.txt
}
: > summary.txt
# 第一段：冒烟 + 64 档（md5 门）
runone def8   8   ""
runone dk8    8   "$DNENV"
runone def64  64  ""
runone dk64a  64  "$DNENV"
runone dk64b  64  "$DNENV"
# 第二段：128/256 主判据
runone def128 128 ""
runone dk128  128 "$DNENV"
runone def256 256 ""
runone dk256  256 "$DNENV"
# 第三段：合并档（GU_LANE + DN_LANE）
runone gdn256 256 "$GLENV $DNENV"
echo ALLDONE > done.flag
