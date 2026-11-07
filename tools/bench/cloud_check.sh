#!/bin/bash
# 云端 K1：文本完整性 + TOKIDS 前 8 核对
cd /root/kv/final || exit 1
echo "=== 完整文本 md5（text 行到 rss_end 之前）==="
for f in run_def256a.log run_opt64.log run_opt128.log run_opt256.log run_def256b.log; do
  n=$(sed -n '/^\[STREAM\] text:/,/^\[STREAM\] rss_end/p' "$f" | sed '$d' | md5sum | cut -d' ' -f1)
  b=$(sed -n '/^\[STREAM\] text:/,/^\[STREAM\] rss_end/p' "$f" | sed '$d' | wc -c)
  echo "$f  md5=$n  bytes=$b"
done
echo
echo "=== TOKIDS 前 8 与全量 md5 ==="
for f in run_def256a.log run_opt64.log run_opt128.log run_opt256.log run_def256b.log; do
  ids=$(grep -oE '[0-9]+\[MOEPROF\] tok1 ' "$f" | sed 's/\[MOEPROF\] tok1 //')
  echo "$f"
  echo "  head8: $(echo "$ids" | head -8 | tr '\n' ' ')"
  echo "  md5:   $(echo "$ids" | tr '\n' ' ' | md5sum | cut -d' ' -f1)  cnt=$(echo "$ids" | wc -l)"
done
