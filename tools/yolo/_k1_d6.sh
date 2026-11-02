#!/bin/bash
# D6：VLLM_Q4_DYN=1 动态 tile 调度（默认关）—— 位级 + NLL + t6/t8 × dyn A/B + 4B 分桶
# + 排查 llama-bench t8 空结果（保留原始输出尾部），补 t6。
# 后台跑；再次运行即轮询。日志 /tmp/d6.log
L=/tmp/d6.log
if pgrep -f 'd6_job' >/dev/null; then echo RUNNING; tail -6 $L; exit 0; fi
if grep -q D6_END $L 2>/dev/null; then cat $L; exit 0; fi
cat > /tmp/d6_job.sh <<'EOF'
set -u
cd /root/kv/kestrel_vllm_riscv
[ -f /tmp/vllm_kestrel.pre_d6 ] || cp -f build-riscv/vllm_kestrel /tmp/vllm_kestrel.pre_d6
sha256sum src/model/vllm_safetensors.c src/core/vllm_tp.c include/core/vllm_tp.h
sh tools/build/build_riscv.sh > /tmp/d6_build.log 2>&1; echo "BUILD_EXIT=$?"
grep -iE ' error|warning: .*(dyn|parfor)' /tmp/d6_build.log | head -8
B=build-riscv/vllm_kestrel; R=/root/kv/models; G=$R/gguf
sha256sum $B
COLB="VLLM_Q4_RVV=3 VLLM_Q4_INT=1 VLLM_Q4_INTB=1 VLLM_Q4_DEC=1"
cfg() { case $1 in base) echo "";; dyn) echo "VLLM_Q4_DYN=1";; dyn1) echo "VLLM_Q4_DYN=1 VLLM_Q4_DYN_GRAIN=1";; esac; }
echo "== 位级 + NLL（ctx32 t8）=="
for M in Qwen3-0.6B Qwen3-4B; do for A in base dyn dyn1; do
  O=/tmp/d6_chk_${M}_$A.log
  env $COLB $(cfg $A) VLLM_Q4CHK=1 VLLM_STREAM_NLL=1 VLLM_THREADS=8 OMP_NUM_THREADS=8 timeout 1800 \
    $B --stream-test --model $R/$M-col --stream-ctx 32 --stream-n 16 --threads 8 > $O 2>&1
  echo "-- $M $A rc=$? dec=$(grep -c 'Q4CHK-dec' $O) bad=$(grep 'Q4CHK-dec' $O | grep -vc 'bitmismatch=0') txt=$(grep -a '\[STREAM\] text:' $O | md5sum | cut -c1-8)"
  grep -E 'SIGILL|Illegal|Segmentation|Q4DEC|\[TP\]' $O | head -2
  grep 'CORPUS_NLL_MEAN' $O
done; done
echo "== decode A/B（ctx32 n32，交替 3 轮）=="
for M in Qwen3-0.6B Qwen3-1.7B Qwen3-4B; do
  for r in 1 2 3; do for T in 6 8; do for A in base dyn dyn1; do
    O=/tmp/d6_perf.log
    env $COLB $(cfg $A) VLLM_THREADS=$T OMP_NUM_THREADS=$T timeout 1800 \
      $B --stream-test --model $R/$M-col --stream-ctx 32 --stream-n 32 --threads $T > $O 2>&1
    printf '%s r%s t%s %-4s rc=%s load=%s | %s | txt=%s\n' $M $r $T $A $? "$(cut -d' ' -f1 /proc/loadavg)" \
      "$(grep -a '\[STREAM\] decoded' $O | tail -1)" "$(grep -a '\[STREAM\] text:' $O | md5sum | cut -c1-8)"
  done; done; done
done
for T in 6 8; do for A in base dyn; do
  echo "== 4B 分桶 t$T $A =="
  env $COLB $(cfg $A) VLLM_DEC_PROF=1 VLLM_THREADS=$T OMP_NUM_THREADS=$T timeout 1800 \
    $B --stream-test --model $R/Qwen3-4B-col --stream-ctx 32 --stream-n 16 --threads $T > /tmp/d6_prof.log 2>&1
  grep -a '\[DEC-SINGLE\]' /tmp/d6_prof.log | tail -n +2 | awk '{
    for(i=1;i<=NF;i++){ if(split($i,kv,"=")==2 && kv[1]!="seq"){ s[kv[1]]+=kv[2]; if(!(kv[1] in o)){o[kv[1]]=++n; k[n]=kv[1]} } }
    c++ } END{ for(j=1;j<=n;j++) printf "%s=%.2f ", k[j], s[k[j]]/c; printf "steps=%d\n", c }'
done; done
echo "== 官方 llama.cpp tg32 t6/t8（-r 3，含原始尾部）=="
for M in Qwen3-0.6B Qwen3-1.7B Qwen3-4B; do for T in 6 8; do
  O=/tmp/d6_lb.log
  env OMP_NUM_THREADS=$T timeout 1800 llama-bench -m $G/$M-Q4_0.gguf -p 0 -n 32 -t $T -r 3 -o md > $O 2>&1
  printf 'LLAMACPP %-11s t%s rc=%s load=%s | ' $M $T $? "$(cut -d' ' -f1 /proc/loadavg)"
  grep -E '\| +tg32 +\|' $O | awk -F'|' '{gsub(/ /,"",$8); printf "tg32=%s t/s", $8}'
  echo
  grep -qE '\| +tg32 +\|' $O || tail -5 $O
done; done
uptime
echo D6_END
EOF
nohup bash /tmp/d6_job.sh > $L 2>&1 &
echo STARTED
