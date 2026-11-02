#!/bin/bash
cd /root/kv/kestrel_vllm_riscv/vqf_convert
echo "=== existing binary ==="
ls -l vqf_conv 2>/dev/null && file vqf_conv 2>/dev/null || echo "(none)"
echo "=== rebuild on device ==="
./build.sh 2>&1 | tail -30
echo "=== help ==="
./vqf_conv --help 2>&1 | head -40
echo "=== free disk ==="
df -h / | tail -1
