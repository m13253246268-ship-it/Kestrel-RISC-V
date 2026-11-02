#!/bin/bash
# 官方 llama.cpp（SpacemiT 包）基线准备：下载 Q8_0 GGUF → 量化到 Q4_0。日志 /tmp/q3gguf.log
L=/tmp/q3gguf.log
if pgrep -f 'q3gguf_job' >/dev/null; then echo RUNNING; tail -6 $L; exit 0; fi
if grep -q Q3GGUF_END $L 2>/dev/null; then cat $L; exit 0; fi
cat > /tmp/q3gguf_job.sh <<'EOF'
set -u
D=/root/kv/models/gguf
mkdir -p "$D"; cd "$D"
for M in Qwen3-0.6B Qwen3-1.7B Qwen3-4B; do
  F=$M-Q8_0.gguf
  if [ ! -s "$F" ]; then
    echo "--- dl $F ---"
    curl -sSL --http1.1 -m 3600 -o "$F" "https://www.modelscope.cn/models/Qwen/$M-GGUF/resolve/master/$F"
  fi
  echo "  $F $(stat -c %s $F)"
done
echo "=== quantize Q4_0 ==="
for M in Qwen3-0.6B Qwen3-1.7B Qwen3-4B; do
  O=$M-Q4_0.gguf
  [ -s "$O" ] && { echo "  skip $O"; continue; }
  t0=$(date +%s)
  llama-quantize "$M-Q8_0.gguf" "$O" Q4_0 2>&1 | tail -3
  echo "  $O $(stat -c %s $O 2>/dev/null || echo 0) elapsed=$(( $(date +%s)-t0 ))s"
done
ls -l
df -h / | tail -1
echo Q3GGUF_END
EOF
nohup bash /tmp/q3gguf_job.sh > $L 2>&1 &
echo STARTED
