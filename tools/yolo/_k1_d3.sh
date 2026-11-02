#!/bin/bash
# D3：(e1) o/down 带 resid 位级自检；(e2) VLLM_DEC_SERNRM=1 小算子串行 —— 位级 + NLL + A/B + 分桶。
# 后台跑；再次运行即轮询。日志 /tmp/d3.log
L=/tmp/d3.log
if pgrep -f 'd3_job' >/dev/null; then echo RUNNING; tail -6 $L; exit 0; fi
if grep -q D3_END $L 2>/dev/null; then cat $L; exit 0; fi
cat > /tmp/d3_job.sh <<'EOF'
set -u
cd /root/kv/kestrel_vllm_riscv
[ -f /tmp/vllm_kestrel.pre_d3 ] || cp -f build-riscv/vllm_kestrel /tmp/vllm_kestrel.pre_d3
sha256sum src/model/vllm_safetensors.c
sh tools/build/build_riscv.sh > /tmp/d3_build.log 2>&1; echo "BUILD_EXIT=$?"
grep -iE ' error|warning: .*(rv_q4_dec|dec_small|sernrm)' /tmp/d3_build.log | head -8
B=build-riscv/vllm_kestrel; R=/root/kv/models
sha256sum $B
COLB="VLLM_Q4_RVV=3 VLLM_Q4_INT=1 VLLM_Q4_INTB=1"
echo "== 位级 + NLL（ctx32 t8）=="
for M in Qwen3-0.6B Qwen3-4B; do for A in dec1 dec1ser; do
  X="VLLM_Q4_DEC=1"; [ $A = dec1ser ] && X="VLLM_Q4_DEC=1 VLLM_DEC_SERNRM=1"
  O=/tmp/d3_chk_${M}_$A.log
  env $COLB $X VLLM_Q4CHK=1 VLLM_STREAM_NLL=1 VLLM_THREADS=8 OMP_NUM_THREADS=8 timeout 1800 \
    $B --stream-test --model $R/$M-col --stream-ctx 32 --stream-n 16 --threads 8 > $O 2>&1
  echo "-- $M $A rc=$? dec=$(grep -c 'Q4CHK-dec' $O) resid=$(grep 'Q4CHK-dec' $O | grep -vc 'resid=0') bad=$(grep 'Q4CHK-dec' $O | grep -vc 'bitmismatch=0') txt=$(grep -a '\[STREAM\] text:' $O | md5sum | cut -c1-8)"
  grep 'Q4CHK-dec' $O | grep -v 'resid=0' | sort -u | head -3
  grep -E 'SIGILL|Illegal|Segmentation|Q4DEC' $O | head -2
  grep 'CORPUS_NLL_MEAN' $O
done; done
echo "== decode A/B（ctx32 n32 t8，交替 3 轮）=="
for M in Qwen3-0.6B Qwen3-1.7B Qwen3-4B; do
  [ -d $R/$M-col ] || { echo "skip $M"; continue; }
  for r in 1 2 3; do for A in dec1 dec1ser; do
    X="VLLM_Q4_DEC=1"; [ $A = dec1ser ] && X="VLLM_Q4_DEC=1 VLLM_DEC_SERNRM=1"
    O=/tmp/d3_perf.log
    env $COLB $X VLLM_THREADS=8 OMP_NUM_THREADS=8 timeout 1800 \
      $B --stream-test --model $R/$M-col --stream-ctx 32 --stream-n 32 --threads 8 > $O 2>&1
    printf '%s r%s %-7s rc=%s load=%s | %s | txt=%s\n' $M $r $A $? "$(cut -d' ' -f1 /proc/loadavg)" \
      "$(grep -a '\[STREAM\] decoded' $O | tail -1)" "$(grep -a '\[STREAM\] text:' $O | md5sum | cut -c1-8)"
  done; done
done
for A in dec1 dec1ser; do
  X="VLLM_Q4_DEC=1"; [ $A = dec1ser ] && X="VLLM_Q4_DEC=1 VLLM_DEC_SERNRM=1"
  echo "== 4B 分桶 $A =="
  env $COLB $X VLLM_DEC_PROF=1 VLLM_THREADS=8 OMP_NUM_THREADS=8 timeout 1800 \
    $B --stream-test --model $R/Qwen3-4B-col --stream-ctx 32 --stream-n 16 --threads 8 > /tmp/d3_prof.log 2>&1
  grep -a '\[DEC-SINGLE\]' /tmp/d3_prof.log | tail -n +2 | awk '{
    for(i=1;i<=NF;i++){ if(split($i,kv,"=")==2 && kv[1]!="seq"){ s[kv[1]]+=kv[2]; if(!(kv[1] in o)){o[kv[1]]=++n; k[n]=kv[1]} } }
    c++ } END{ for(j=1;j<=n;j++) printf "%s=%.2f ", k[j], s[k[j]]/c; printf "\nsteps=%d\n", c }'
done
uptime
echo D3_END
EOF
nohup bash /tmp/d3_job.sh > $L 2>&1 &
echo STARTED
