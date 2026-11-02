#!/bin/bash
# D6b：补 vllm_tp_parfor_dyn 原型后重建，确认告警消失 + dyn 位级/NLL 抽检
L=/tmp/d6b.log
if pgrep -f 'd6b_job' >/dev/null; then echo RUNNING; tail -4 $L; exit 0; fi
if grep -q D6B_END $L 2>/dev/null; then cat $L; exit 0; fi
cat > /tmp/d6b_job.sh <<'EOF'
cd /root/kv/kestrel_vllm_riscv
sh tools/build/build_riscv.sh > /tmp/d6b_build.log 2>&1; echo "BUILD_EXIT=$?"
echo "warn_dyn=$(grep -c 'parfor_dyn' /tmp/d6b_build.log)"
B=build-riscv/vllm_kestrel; sha256sum $B
O=/tmp/d6b_chk.log
env VLLM_Q4_RVV=3 VLLM_Q4_INT=1 VLLM_Q4_INTB=1 VLLM_Q4_DEC=1 VLLM_Q4_DYN=1 VLLM_Q4CHK=1 VLLM_STREAM_NLL=1 \
  VLLM_THREADS=8 OMP_NUM_THREADS=8 timeout 1800 $B --stream-test --model /root/kv/models/Qwen3-0.6B-col \
  --stream-ctx 32 --stream-n 16 --threads 8 > $O 2>&1
echo "rc=$? dec=$(grep -c 'Q4CHK-dec' $O) bad=$(grep 'Q4CHK-dec' $O | grep -vc 'bitmismatch=0')"
grep 'CORPUS_NLL_MEAN' $O
echo D6B_END
EOF
nohup bash /tmp/d6b_job.sh > $L 2>&1 &
echo STARTED
