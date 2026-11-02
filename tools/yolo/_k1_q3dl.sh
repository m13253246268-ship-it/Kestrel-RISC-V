#!/bin/bash
# 千问3 三档 dense（0.6B/1.7B/4B）板端下载；后台跑，再次运行即轮询。日志 /tmp/q3dl.log
L=/tmp/q3dl.log
if pgrep -f 'q3dl_job' >/dev/null; then echo RUNNING; tail -6 $L; exit 0; fi
if grep -q Q3DL_END $L 2>/dev/null; then cat $L; exit 0; fi
cat > /tmp/q3dl_job.sh <<'EOF'
set -u
for M in Qwen3-0.6B Qwen3-1.7B Qwen3-4B; do
  D=/root/kv/models/$M
  mkdir -p "$D"; cd "$D"
  BASE="https://www.modelscope.cn/models/Qwen/$M/resolve/master"
  echo "===== $M ====="
  for f in config.json generation_config.json tokenizer.json tokenizer_config.json; do
    curl -sSL -m 180 -o "$f" "$BASE/$f" && echo "  ok $f $(stat -c %s $f)"
  done
  echo "--- big file ---"
  t0=$(date +%s)
  curl -sSL -m 3600 -o model.safetensors "$BASE/model.safetensors"
  rc=$?
  t1=$(date +%s)
  sz=$(stat -c %s model.safetensors 2>/dev/null || echo 0)
  echo "  rc=$rc size=$sz elapsed=$((t1-t0))s rate=$((sz/(t1-t0+1)/1048576)) MB/s"
done
df -h / | tail -1
echo Q3DL_END
EOF
nohup bash /tmp/q3dl_job.sh > $L 2>&1 &
echo STARTED
