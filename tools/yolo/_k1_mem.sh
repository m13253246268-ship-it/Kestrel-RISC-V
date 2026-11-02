#!/bin/bash
set -u
free -m
echo "--- swap ---"
cat /proc/meminfo | grep -E 'MemTotal|MemAvailable|SwapTotal|SwapFree'
echo "--- model sizes ---"
ls -l /root/kv/models/Qwen2.5-0.5B/model.vqf /root/kv/models/Qwen2.5-0.5B-col/model.vqf
df -h /root | tail -1
echo MEM_DONE
