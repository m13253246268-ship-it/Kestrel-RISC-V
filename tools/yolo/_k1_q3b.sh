#!/bin/bash
# 千问3-30B-A3B 板端下载+转换+EC 重排+冒烟；后台跑，再次运行即轮询。日志 /tmp/q3b.log
L=/tmp/q3b.log
if pgrep -f 'q3b_job' >/dev/null; then echo RUNNING; tail -8 $L; exit 0; fi
if grep -q Q3B_END $L 2>/dev/null; then cat $L; exit 0; fi
cat > /tmp/q3b_job.sh <<'EOF'
set -u
M=Qwen3-30B-A3B
D=/root/kv/models/$M
C=/root/kv/kestrel_vllm_riscv/vqf_convert/vqf_conv
B=/root/kv/kestrel_vllm_riscv/build-riscv/vllm_kestrel
BASE="https://www.modelscope.cn/models/Qwen/$M/resolve/master"
mkdir -p "$D"; cd "$D"
echo "=== download 30B (16 shards) ==="
for f in config.json generation_config.json tokenizer.json tokenizer_config.json model.safetensors.index.json; do
  [ -s "$f" ] || curl -sSL -m 300 -o "$f" "$BASE/$f"
  echo "  ok $f $(stat -c %s $f 2>/dev/null)"
done
t0=$(date +%s)
for i in $(seq 1 16); do
  F=$(printf 'model-%05d-of-00016.safetensors' $i)
  sz=$(stat -c %s "$F" 2>/dev/null || echo 0)
  if [ "$sz" -lt 1000000 ]; then rm -f "$F"; curl -sSL -m 3600 -o "$F" "$BASE/$F"; fi
  echo "  got $F $(stat -c %s "$F" 2>/dev/null)"
done
echo "  dl_elapsed=$(( $(date +%s)-t0 ))s"
df -h / | tail -1
echo "=== convert (q4 stream) ==="
cp -f /root/kv/models/_shared/vocab.bin /root/kv/models/_shared/merges.bin "$D"/
t0=$(date +%s)
$C --model "$D" --convert-vqf "$D/model.vqf" --wmode q4 --stream 2>&1 | tail -8
echo "  conv_elapsed=$(( $(date +%s)-t0 ))s vqf=$(stat -c %s $D/model.vqf 2>/dev/null || echo 0)"
echo "=== build relayout tool ==="
gcc -O2 -w -o /tmp/vqf_down_relayout /root/kv/kestrel_vllm_riscv/tools/bench/vqf_down_relayout.c 2>&1 | head -5
echo "=== EC relayout dryrun(mode0) ==="
/tmp/vqf_down_relayout "$D/model.vqf" 0 2>&1 | tail -8
echo "=== EC relayout WRITE(mode1) ==="
t0=$(date +%s)
/tmp/vqf_down_relayout "$D/model.vqf" 1 2>&1 | tail -6
echo "  ec_elapsed=$(( $(date +%s)-t0 ))s"
echo "=== smoke (VQF_STREAM=2 DN_EC=1 RVV=3, n=4) ==="
env VLLM_VQF_STREAM=2 VLLM_DN_EC=1 VLLM_Q4_RVV=3 VLLM_THREADS=8 OMP_NUM_THREADS=8 timeout 3000 \
  $B --stream-test --model "$D" --stream-ctx 32 --stream-n 4 --threads 8 2>&1 \
  | grep -E 'PREFILL-TIMING|STREAM\] (prefill|decoded|text)|VQF|FAIL|moe|MoE' | cut -c1-300
echo "=== cleanup shards ==="
rm -f "$D"/model-0000*-of-00016.safetensors
df -h / | tail -1
echo Q3B_END
EOF
nohup bash /tmp/q3b_job.sh > $L 2>&1 &
echo STARTED
