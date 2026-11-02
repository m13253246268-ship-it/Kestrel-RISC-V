#!/bin/bash
set -u
D=/root/kv/models/Qwen2.5-0.5B
C=/root/kv/kestrel_vllm_riscv/vqf_convert/vqf_conv
ls -l "$D"
echo "=== convert (wmode=q4, lane-major 默认) ==="
t0=$(date +%s)
$C --model "$D" --convert-vqf "$D/model.vqf" --wmode q4 2>&1 | tail -25
t1=$(date +%s)
echo "=== elapsed $((t1-t0))s ==="
ls -l "$D/model.vqf" 2>/dev/null || echo "(no vqf produced)"
