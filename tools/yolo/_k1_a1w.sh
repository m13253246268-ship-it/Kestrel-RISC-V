#!/bin/bash
# 查询 A-1 后台构建状态
uptime
pgrep -af 'cc1' | cut -c1-80 | head -3
grep -E 'error|EXIT=|OK:' /tmp/a1_build.log | head -20
ls -l /root/kv/kestrel_vllm_riscv/build-riscv/vllm_kestrel
