#!/bin/bash
# 千问3 dense 两档（1.7B/4B）板端下载+转换+冒烟；0.6B 已下载，并入本作业转换。后台跑，再次运行即轮询。日志 /tmp/q3a.log
L=/tmp/q3a.log
if pgrep -f 'q3a_job' >/dev/null; then echo RUNNING; tail -8 $L; exit 0; fi
if grep -q Q3A_END $L 2>/dev/null; then cat $L; exit 0; fi
cat > /tmp/q3a_job.sh <<'EOF'
set -u
C=/root/kv/kestrel_vllm_riscv/vqf_convert/vqf_conv
B=/root/kv/kestrel_vllm_riscv/build-riscv/vllm_kestrel
dl() { # dl <model> <file>
  local M=$1 F=$2 D=/root/kv/models/$1
  local BASE="https://www.modelscope.cn/models/Qwen/$M/resolve/master"
  [ -s "$D/$F" ] && { echo "  skip $F"; return 0; }
  curl -sSL -m 3600 -o "$D/$F" "$BASE/$F" || echo "  DL_FAIL $F"
  echo "  got $F $(stat -c %s "$D/$F" 2>/dev/null)"
}
conv() { # conv <model> [extra...]
  local M=$1; shift; local D=/root/kv/models/$M
  cp -f /root/kv/models/_shared/vocab.bin /root/kv/models/_shared/merges.bin "$D"/
  echo "--- convert $M ($*) ---"
  t0=$(date +%s)
  $C --model "$D" --convert-vqf "$D/model.vqf" --wmode q4 "$@" 2>&1 | tail -6
  echo "  conv_elapsed=$(( $(date +%s)-t0 ))s vqf=$(stat -c %s $D/model.vqf 2>/dev/null || echo 0)"
}
smoke() { # smoke <model>
  local M=$1 D=/root/kv/models/$M
  echo "--- smoke $M ---"
  env VLLM_Q4_RVV=3 VLLM_THREADS=8 OMP_NUM_THREADS=8 timeout 1200 $B --stream-test \
    --model "$D" --stream-ctx 32 --stream-n 16 --threads 8 --warmup 2>&1 \
    | grep -E 'PREFILL-TIMING|STREAM\] (prefill|decoded)|^\[STREAM\] (text|OUT)|pt=\[|VQF|FAIL' | cut -c1-300
}
for M in Qwen3-1.7B Qwen3-4B; do
  D=/root/kv/models/$M; mkdir -p "$D"; cd "$D"
  echo "===== download $M ====="; t0=$(date +%s)
  dl $M model.safetensors.index.json
  case $M in
    Qwen3-1.7B) N=2;; Qwen3-4B) N=3;;
  esac
  for i in $(seq -w 1 $N); do dl $M "model-0000$i-of-0000$N.safetensors"; done
  echo "  dl_elapsed=$(( $(date +%s)-t0 ))s"
  conv $M
  smoke $M
  echo "--- cleanup shards $M ---"
  rm -f "$D"/model-0000*-of-0000*.safetensors
  df -h / | tail -1
done
echo "===== Qwen3-0.6B (already downloaded) ====="
conv Qwen3-0.6B
smoke Qwen3-0.6B
rm -f /root/kv/models/Qwen3-0.6B/model.safetensors
df -h / | tail -1
echo Q3A_END
EOF
nohup bash /tmp/q3a_job.sh > $L 2>&1 &
echo STARTED
