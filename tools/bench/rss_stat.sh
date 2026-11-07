#!/bin/bash
cd /root/kv/final || exit 1
for f in run_opt256.log run_def256b.log; do
  echo "--- $f ---"
  grep -aE 'rss_after_load|rss_after_init|rss_after_prefill|rss_end' "$f"
done
