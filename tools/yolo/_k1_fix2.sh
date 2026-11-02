#!/bin/bash
# 转换器截断守卫复验：重建 vqf_conv，正例（完整权重）+ 反例（截断权重）。日志 /tmp/fix2.log
L=/tmp/fix2.log
if pgrep -f 'fix2_job' >/dev/null; then echo RUNNING; tail -8 $L; exit 0; fi
if grep -q FIX2_END $L 2>/dev/null; then cat $L; exit 0; fi
cat > /tmp/fix2_job.sh <<'EOF'
set -u
cd /root/kv/kestrel_vllm_riscv/vqf_convert
./build.sh > /tmp/fix2_build.log 2>&1; echo "CONV_BUILD_EXIT=$?"
grep -iE 'error' /tmp/fix2_build.log | head -5
sha256sum vqf_conv
T=/tmp/q3_06_st; mkdir -p $T
cp /root/kv/models/Qwen3-0.6B/*.json $T/ 2>/dev/null
U=https://www.modelscope.cn/models/Qwen/Qwen3-0.6B/resolve/master/model.safetensors
WANT=1503300328   # fix1 实测完整大小
for k in 1 2 3; do
  [ "$(stat -c %s $T/model.safetensors 2>/dev/null || echo 0)" = "$WANT" ] && break
  rm -f $T/model.safetensors; curl -sL --http1.1 -o $T/model.safetensors $U
done
echo "st_size=$(stat -c %s $T/model.safetensors) want=$WANT"
echo "--- positive ---"
env VLLM_Q4_RVV=3 ./vqf_conv --model $T --convert-vqf /tmp/q3_06_check.vqf --wmode q4 > /tmp/fix2_conv.log 2>&1
echo "  conv_exit=$? size=$(stat -c %s /tmp/q3_06_check.vqf 2>/dev/null || echo 0) (原 model.vqf=$(stat -c %s /root/kv/models/Qwen3-0.6B/model.vqf))"
grep -E 'after-tensor-scan|rope_theta|ERROR' /tmp/fix2_conv.log
rm -f /tmp/q3_06_check.vqf
echo "--- negative (truncate 200MB) ---"
truncate -s 200000000 $T/model.safetensors
env VLLM_Q4_RVV=3 ./vqf_conv --model $T --convert-vqf /tmp/q3_06_bad.vqf --wmode q4 > /tmp/fix2_bad.log 2>&1
echo "  conv_exit=$? left_file=$(ls /tmp/q3_06_bad.vqf 2>/dev/null || echo none)"
tail -3 /tmp/fix2_bad.log
rm -rf $T /tmp/q3_06_bad.vqf
echo FIX2_END
EOF
nohup bash /tmp/fix2_job.sh > $L 2>&1 &
echo STARTED
