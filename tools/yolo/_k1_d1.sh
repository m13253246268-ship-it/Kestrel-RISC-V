#!/bin/bash
# D1：decode 新内核 VLLM_Q4_DEC=1 位级自检 + NLL + decode A/B（def vs dec1，交替 3 轮）。
# 后台跑；再次运行即轮询。日志 /tmp/d1.log
L=/tmp/d1.log
if pgrep -f 'd1_job' >/dev/null; then echo RUNNING; tail -6 $L; exit 0; fi
if grep -q D1_END $L 2>/dev/null; then cat $L; exit 0; fi
cat > /tmp/d1_job.sh <<'EOF'
set -u
cd /root/kv/kestrel_vllm_riscv
[ -f /tmp/vllm_kestrel.pre_d1 ] || cp -f build-riscv/vllm_kestrel /tmp/vllm_kestrel.pre_d1
sha256sum src/model/vllm_safetensors.c
sh tools/build/build_riscv.sh > /tmp/d1_build.log 2>&1; echo "BUILD_EXIT=$?"
grep -iE 'error|warning: .*rv_q4_dec' /tmp/d1_build.log | head -8
B=build-riscv/vllm_kestrel; R=/root/kv/models
sha256sum $B
objdump -d $B | awk '/<rv_q4_dec_tile[^>]*>:$/{p=1;next} p&&/^$/{p=0} p' > /tmp/d1_t.s
echo "dec_tile lines=$(wc -l < /tmp/d1_t.s) vwmacc=$(grep -c vwmacc /tmp/d1_t.s) vmerge=$(grep -c vmerge /tmp/d1_t.s)"
COLB="VLLM_Q4_RVV=3 VLLM_Q4_INT=1 VLLM_Q4_INTB=1"
echo "== 位级 + NLL（ctx32 t8）=="
for M in Qwen3-0.6B Qwen3-4B; do for A in def dec1; do
  X=""; [ $A = dec1 ] && X="VLLM_Q4_DEC=1"
  O=/tmp/d1_chk_${M}_$A.log
  env $COLB $X VLLM_Q4CHK=1 VLLM_STREAM_NLL=1 VLLM_THREADS=8 OMP_NUM_THREADS=8 timeout 1800 \
    $B --stream-test --model $R/$M-col --stream-ctx 32 --stream-n 16 --threads 8 > $O 2>&1
  echo "-- $M $A rc=$? dec=$(grep -c 'Q4CHK-dec' $O)/bad$(grep 'Q4CHK-dec' $O | grep -vc 'bitmismatch=0') txt=$(grep -a '\[STREAM\] text:' $O | md5sum | cut -c1-8)"
  grep 'Q4CHK-dec' $O | sort -u -t' ' -k2,3 | head -4
  grep -E 'SIGILL|Illegal|Segmentation|Q4DEC' $O | head -2
  grep 'CORPUS_NLL_MEAN' $O
done; done
echo "== decode A/B（ctx32 n32 t8，交替 3 轮）=="
for M in Qwen3-0.6B Qwen3-1.7B Qwen3-4B; do
  [ -d $R/$M-col ] || { echo "skip $M"; continue; }
  for r in 1 2 3; do for A in def dec1; do
    X=""; [ $A = dec1 ] && X="VLLM_Q4_DEC=1"
    O=/tmp/d1_perf.log
    env $COLB $X VLLM_THREADS=8 OMP_NUM_THREADS=8 timeout 1800 \
      $B --stream-test --model $R/$M-col --stream-ctx 32 --stream-n 32 --threads 8 > $O 2>&1
    printf '%s r%s %-4s rc=%s load=%s | %s | txt=%s\n' $M $r $A $? "$(cut -d' ' -f1 /proc/loadavg)" \
      "$(grep -a '\[STREAM\] decoded' $O | tail -1)" "$(grep -a '\[STREAM\] text:' $O | md5sum | cut -c1-8)"
  done; done
done
echo "== 4B 分桶 dec1 =="
env $COLB VLLM_Q4_DEC=1 VLLM_DEC_PROF=1 VLLM_THREADS=8 OMP_NUM_THREADS=8 timeout 1800 \
  $B --stream-test --model $R/Qwen3-4B-col --stream-ctx 32 --stream-n 16 --threads 8 > /tmp/d1_prof.log 2>&1
grep -a '\[DEC-SINGLE\]' /tmp/d1_prof.log | tail -n +2 | awk '{
  for(i=1;i<=NF;i++){ if(split($i,kv,"=")==2 && kv[1]!="seq"){ s[kv[1]]+=kv[2]; if(!(kv[1] in o)){o[kv[1]]=++n; k[n]=kv[1]} } }
  c++ } END{ for(j=1;j<=n;j++) printf "%s=%.2f ", k[j], s[k[j]]/c; printf "\nsteps=%d\n", c }'
uptime
echo D1_END
EOF
nohup bash /tmp/d1_job.sh > $L 2>&1 &
echo STARTED
