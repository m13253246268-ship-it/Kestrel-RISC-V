#!/bin/bash
set -u
BASE="https://www.modelscope.cn/models/Qwen/Qwen2.5-0.5B/resolve/master"
D=/root/kv/models/Qwen2.5-0.5B
mkdir -p "$D"
cd "$D"
for f in config.json generation_config.json tokenizer.json tokenizer_config.json vocab.json merges.txt; do
  curl -sSL -m 120 -o "$f" "$BASE/$f" && echo "  ok $f $(stat -c %s $f)"
done
echo "=== big file ==="
t0=$(date +%s)
curl -sSL -m 1800 -o model.safetensors "$BASE/model.safetensors"
rc=$?
t1=$(date +%s)
sz=$(stat -c %s model.safetensors 2>/dev/null || echo 0)
echo "  rc=$rc size=$sz elapsed=$((t1-t0))s rate=$((sz/(t1-t0+1)/1048576)) MB/s"
echo "=== result ==="
ls -l "$D"
