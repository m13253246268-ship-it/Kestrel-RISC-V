#!/bin/bash
# D4：线程池唤醒/同步开销验证 —— VLLM_DEC_SERSILU 串行 silu + TP_SPIN/TP_BIND 扫描。
# 后台跑；再次运行即轮询。日志 /tmp/d4.log
L=/tmp/d4.log
if pgrep -f 'd4_job' >/dev/null; then echo RUNNING; tail -6 $L; exit 0; fi
if grep -q D4_END $L 2>/dev/null; then cat $L; exit 0; fi
cat > /tmp/d4_job.sh <<'EOF'
set -u
cd /root/kv/kestrel_vllm_riscv
[ -f /tmp/vllm_kestrel.pre_d4 ] || cp -f build-riscv/vllm_kestrel /tmp/vllm_kestrel.pre_d4
sha256sum src/model/vllm_safetensors.c
sh tools/build/build_riscv.sh > /tmp/d4_build.log 2>&1; echo "BUILD_EXIT=$?"
grep -iE ' error|warning: .*(dec_silu|dec_small|sersilu)' /tmp/d4_build.log | head -8
B=build-riscv/vllm_kestrel; R=/root/kv/models
sha256sum $B
COLB="VLLM_Q4_RVV=3 VLLM_Q4_INT=1 VLLM_Q4_INTB=1"
arm() { case $1 in
  base) echo "VLLM_Q4_DEC=1";;
  ser)  echo "VLLM_Q4_DEC=1 VLLM_DEC_SERNRM=1";;
  ss)   echo "VLLM_Q4_DEC=1 VLLM_DEC_SERNRM=1 VLLM_DEC_SERSILU=1";;
  ssS)  echo "VLLM_Q4_DEC=1 VLLM_DEC_SERNRM=1 VLLM_DEC_SERSILU=1 VLLM_TP_SPIN=2000000";;
  ssB)  echo "VLLM_Q4_DEC=1 VLLM_DEC_SERNRM=1 VLLM_DEC_SERSILU=1 VLLM_TP_BIND=1";;
  bS)   echo "VLLM_Q4_DEC=1 VLLM_TP_SPIN=2000000";;
  bB)   echo "VLLM_Q4_DEC=1 VLLM_TP_BIND=1";;
esac; }
echo "== NLL + md5（ctx32 t8, ss 臂）=="
for M in Qwen3-0.6B Qwen3-4B; do
  O=/tmp/d4_nll_$M.log
  env $COLB $(arm ss) VLLM_STREAM_NLL=1 VLLM_THREADS=8 OMP_NUM_THREADS=8 timeout 1800 \
    $B --stream-test --model $R/$M-col --stream-ctx 32 --stream-n 16 --threads 8 > $O 2>&1
  echo "-- $M ss rc=$? $(grep 'CORPUS_NLL_MEAN' $O)"
  grep -E 'SIGILL|Illegal|Segmentation' $O | head -2
done
echo "== 4B 分桶（ctx32 n16）=="
for A in base ser ss ssS ssB bS bB; do
  echo "-- $A load=$(cut -d' ' -f1 /proc/loadavg)"
  env $COLB $(arm $A) VLLM_DEC_PROF=1 VLLM_THREADS=8 OMP_NUM_THREADS=8 timeout 1800 \
    $B --stream-test --model $R/Qwen3-4B-col --stream-ctx 32 --stream-n 16 --threads 8 > /tmp/d4_prof.log 2>&1
  grep -a '\[DEC-SINGLE\]' /tmp/d4_prof.log | tail -n +2 | awk '{
    for(i=1;i<=NF;i++){ if(split($i,kv,"=")==2 && kv[1]!="seq"){ s[kv[1]]+=kv[2]; if(!(kv[1] in o)){o[kv[1]]=++n; k[n]=kv[1]} } }
    c++ } END{ for(j=1;j<=n;j++) printf "%s=%.2f ", k[j], s[k[j]]/c; printf "steps=%d\n", c }'
done
echo "== decode A/B（ctx32 n32 t8，交替 3 轮）=="
for M in Qwen3-0.6B Qwen3-1.7B Qwen3-4B; do
  [ -d $R/$M-col ] || { echo "skip $M"; continue; }
  for r in 1 2 3; do for A in base ser ss ssS ssB; do
    O=/tmp/d4_perf.log
    env $COLB $(arm $A) VLLM_THREADS=8 OMP_NUM_THREADS=8 timeout 1800 \
      $B --stream-test --model $R/$M-col --stream-ctx 32 --stream-n 32 --threads 8 > $O 2>&1
    printf '%s r%s %-5s rc=%s load=%s | %s | txt=%s\n' $M $r $A $? "$(cut -d' ' -f1 /proc/loadavg)" \
      "$(grep -a '\[STREAM\] decoded' $O | tail -1)" "$(grep -a '\[STREAM\] text:' $O | md5sum | cut -c1-8)"
  done; done
done
uptime
echo D4_END
EOF
nohup bash /tmp/d4_job.sh > $L 2>&1 &
echo STARTED
