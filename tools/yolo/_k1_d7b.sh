#!/bin/bash
# D7b（per-worker gen，被排除 worker 不唤醒）：DYN 默认开 + VLLM_DEC_THREADS 默认 6（pool=8）—— 位级 + NLL + 默认/逃生舱 A/B
# 后台跑；再次运行即轮询。日志 /tmp/d7b.log
L=/tmp/d7b.log
if pgrep -f 'd7b_job' >/dev/null; then echo RUNNING; tail -6 $L; exit 0; fi
if grep -q D7B_END $L 2>/dev/null; then cat $L; exit 0; fi
cat > /tmp/d7b_job.sh <<'EOF'
set -u
cd /root/kv/kestrel_vllm_riscv
[ -f /tmp/vllm_kestrel.pre_d7b ] || cp -f build-riscv/vllm_kestrel /tmp/vllm_kestrel.pre_d7
sha256sum src/model/vllm_safetensors.c src/core/vllm_tp.c include/core/vllm_tp.h
sh tools/build/build_riscv.sh > /tmp/d7b_build.log 2>&1; echo "BUILD_EXIT=$?"
grep -iE ' error|warning: .*(dyn|parfor|active|forward_impl)' /tmp/d7b_build.log | head -8
B=build-riscv/vllm_kestrel; R=/root/kv/models
sha256sum $B
COLB="VLLM_Q4_RVV=3 VLLM_Q4_INT=1 VLLM_Q4_INTB=1 VLLM_Q4_DEC=1"
cfg() { case $1 in def) echo "";; dt0) echo "VLLM_DEC_THREADS=0";; dyn0) echo "VLLM_Q4_DYN=0 VLLM_DEC_THREADS=0";; esac; }
echo "== 位级 + NLL（ctx32 pool8）=="
for M in Qwen3-0.6B Qwen3-4B; do for A in def dt0 dyn0; do
  O=/tmp/d7b_chk_${M}_$A.log
  env $COLB $(cfg $A) VLLM_Q4CHK=1 VLLM_STREAM_NLL=1 VLLM_THREADS=8 OMP_NUM_THREADS=8 timeout 1800 \
    $B --stream-test --model $R/$M-col --stream-ctx 32 --stream-n 16 --threads 8 > $O 2>&1
  echo "-- $M $A rc=$? dec=$(grep -c 'Q4CHK-dec' $O) bad=$(grep 'Q4CHK-dec' $O | grep -vc 'bitmismatch=0')"
  grep -E 'SIGILL|Illegal|Segmentation|Q4DEC|\[TP\]' $O | head -2
  grep 'CORPUS_NLL_MEAN' $O
done; done
echo "== decode A/B（pool8 ctx32 n32，交替 3 轮；t6p=pool6 默认）=="
for M in Qwen3-0.6B Qwen3-1.7B Qwen3-4B; do
  for r in 1 2 3; do for A in def dt0 dyn0 t6p; do
    O=/tmp/d7b_perf.log; T=8; C=$A; [ $A = t6p ] && { T=6; C=def; }
    env $COLB $(cfg $C) VLLM_THREADS=$T OMP_NUM_THREADS=$T timeout 1800 \
      $B --stream-test --model $R/$M-col --stream-ctx 32 --stream-n 32 --threads $T > $O 2>&1
    printf '%s r%s %-4s rc=%s load=%s | %s | %s | txt=%s\n' $M $r $A $? "$(cut -d' ' -f1 /proc/loadavg)" \
      "$(grep -a '\[STREAM\] prefill' $O | tail -1)" "$(grep -a '\[STREAM\] decoded' $O | tail -1)" \
      "$(grep -a '\[STREAM\] text:' $O | md5sum | cut -c1-8)"
  done; done
done
uptime
echo D7B_END
EOF
nohup bash /tmp/d7b_job.sh > $L 2>&1 &
echo STARTED
