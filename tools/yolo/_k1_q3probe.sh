#!/bin/bash
set -u
for M in Qwen3-1.7B Qwen3-4B Qwen3-30B-A3B; do
  echo "===== $M ====="
  curl -sS -m 30 "https://www.modelscope.cn/api/v1/models/Qwen/$M/repo/files?Revision=master&Root=" \
    | tr ',' '\n' | grep -E '"Path"|"Size"' | paste - - | grep -vE '\.(md|gitattributes|json)$' | head -25
done
echo "Q3_FILES_END"
