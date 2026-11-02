#!/bin/bash
# 上传修复后的源码并重建 引擎 + 转换器；再做一次文本输出与转换器自检。日志 /tmp/fix1.log
L=/tmp/fix1.log
if pgrep -f 'fix1_job' >/dev/null; then echo RUNNING; tail -8 $L; exit 0; fi
if grep -q FIX1_END $L 2>/dev/null; then cat $L; exit 0; fi
cat > /tmp/fix1_job.sh <<'EOF'
set -u
cd /root/kv/kestrel_vllm_riscv
echo "=== rebuild engine ==="
sh tools/build/build_riscv.sh > /tmp/fix1_build.log 2>&1; echo "ENGINE_BUILD_EXIT=$?"
grep -iE 'error|warning' /tmp/fix1_build.log | head -10
sha256sum build-riscv/vllm_kestrel
echo "=== rebuild converter ==="
cd vqf_convert
./build.sh > /tmp/fix1_convbuild.log 2>&1; echo "CONV_BUILD_EXIT=$?"
grep -iE 'error|warning' /tmp/fix1_convbuild.log | head -10
sha256sum vqf_conv
echo "=== stream text check (0.6B) ==="
env VLLM_Q4_RVV=3 VLLM_THREADS=8 OMP_NUM_THREADS=8 timeout 900 \
  ../build-riscv/vllm_kestrel --stream-test --model /root/kv/models/Qwen3-0.6B \
  --stream-ctx 32 --stream-n 16 --threads 8 --warmup > /tmp/fix1_st.log 2>&1
echo "  stream_exit=$?"
grep -E 'STREAM\] (prefill|decoded)' /tmp/fix1_st.log | cut -c1-200
echo "  --- text (full, multi-line) ---"
sed -n '/\[STREAM\] text:/,$p' /tmp/fix1_st.log | head -20
echo "  marker_count(Ċ/Ġ)=$(sed -n '/\[STREAM\] text:/,$p' /tmp/fix1_st.log | grep -c -E 'Ċ|Ġ')"
echo "=== converter check (0.6B safetensors 临时重下，原目录已删) ==="
T=/tmp/q3_06_st; mkdir -p $T
cp /root/kv/models/Qwen3-0.6B/*.json $T/ 2>/dev/null
U=https://www.modelscope.cn/models/Qwen/Qwen3-0.6B/resolve/master/model.safetensors
WANT=$(curl -sIL --http1.1 $U | grep -i '^content-length' | tail -1 | tr -dc '0-9')
for k in 1 2 3; do
  [ "$(stat -c %s $T/model.safetensors 2>/dev/null || echo 0)" = "$WANT" ] && break
  curl -sL --http1.1 -C - -o $T/model.safetensors $U
done
echo "  st_size=$(stat -c %s $T/model.safetensors) want=$WANT"
echo "--- positive: 完整权重应 exit=0 且守卫不误杀 ---"
env VLLM_Q4_RVV=3 ./vqf_conv --model $T --convert-vqf /tmp/q3_06_check.vqf --wmode q4 > /tmp/fix1_conv.log 2>&1
echo "  conv_exit=$? size=$(stat -c %s /tmp/q3_06_check.vqf 2>/dev/null || echo 0)"
grep -E 'after-tensor-scan|ERROR' /tmp/fix1_conv.log; tail -2 /tmp/fix1_conv.log
rm -f /tmp/q3_06_check.vqf
echo "--- negative: 截断权重应 exit!=0 且不留输出文件 ---"
truncate -s 200000000 $T/model.safetensors
env VLLM_Q4_RVV=3 ./vqf_conv --model $T --convert-vqf /tmp/q3_06_bad.vqf --wmode q4 > /tmp/fix1_bad.log 2>&1
echo "  conv_exit=$? left_file=$(ls /tmp/q3_06_bad.vqf 2>/dev/null || echo none)"
tail -3 /tmp/fix1_bad.log
rm -rf $T /tmp/q3_06_bad.vqf
echo FIX1_END
EOF
nohup bash /tmp/fix1_job.sh > $L 2>&1 &
echo STARTED
