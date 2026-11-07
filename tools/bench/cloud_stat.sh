#!/bin/bash
cd /root/kv/final || exit 1
echo "=== 云端 SA-STAT 末尾 ==="
for f in err_opt256.log err_def256b.log; do
  echo "--- $f ---"
  grep -a 'SA-STAT' "$f" | tail -2
done
echo "=== rss_end ==="
grep -a 'rss_end' run_opt256.log run_def256b.log
echo "=== 全机 buff/cache 与模型大小 ==="
free -m | head -2
ls -l /root/kv/models/Qwen3-30B-A3B-col/model.vqf
