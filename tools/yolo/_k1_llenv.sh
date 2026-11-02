#!/bin/bash
ls -la /usr/lib/libspine_tcm* /usr/lib/*/libspine_tcm* 2>&1 | head
ls /dev | grep -iE 'tcm|ai' | head
ls /proc/set_ai_thread 2>&1
M=/root/kv/models/gguf/Qwen3-0.6B-Q4_0.gguf
[ -f $M ] || M=$(ls /root/kv/models/gguf/*0.6B*Q4_0*.gguf | head -1)
echo "model=$M"
GGML_LOG_LEVEL=0 llama-bench -m $M -t 4 -p 32 -n 0 -r 1 -v 2>&1 | grep -iE 'SPACEMIT|tcm|ime|repack|cpu_mask|affin|pp32|CPU_REPACK' | head -40
