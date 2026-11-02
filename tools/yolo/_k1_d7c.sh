#!/bin/bash
# D7c：VLLM_DEC_THREADS 默认改回 0 后重建 + 0.6B/4B 位级抽检
L=/tmp/d7c.log
if pgrep -f 'd7c_job' >/dev/null; then echo RUNNING; tail -4 $L; exit 0; fi
if grep -q D7C_END $L 2>/dev/null; then cat $L; exit 0; fi
cat > /tmp/d7c_job.sh <<'EOF'
cd /root/kv/kestrel_vllm_riscv
sh tools/build/build_riscv.sh > /tmp/d7c_build.log 2>&1; echo "BUILD_EXIT=$?"
grep -iE ' error|warning: .*(dyn|parfor|active|forward_impl)' /tmp/d7c_build.log | head -4
B=build-riscv/vllm_kestrel; R=/root/kv/models; sha256sum $B
COLB="VLLM_Q4_RVV=3 VLLM_Q4_INT=1 VLLM_Q4_INTB=1 VLLM_Q4_DEC=1"
for M in Qwen3-0.6B Qwen3-4B; do
  O=/tmp/d7c_chk_$M.log
  env $COLB VLLM_Q4CHK=1 VLLM_STREAM_NLL=1 VLLM_THREADS=6 OMP_NUM_THREADS=6 timeout 1800 \
    $B --stream-test --model $R/$M-col --stream-ctx 32 --stream-n 16 --threads 6 > $O 2>&1
  echo "-- $M rc=$? dec=$(grep -c 'Q4CHK-dec' $O) bad=$(grep 'Q4CHK-dec' $O | grep -vc 'bitmismatch=0')"
  grep 'CORPUS_NLL_MEAN' $O
done
echo D7C_END
EOF
nohup bash /tmp/d7c_job.sh > $L 2>&1 &
echo STARTED
