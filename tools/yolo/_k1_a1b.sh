#!/bin/bash
# L-IME-001 Phase A-1：板端后台重建（源码已投送），日志 /tmp/a1_build.log
set -u
cd /root/kv/kestrel_vllm_riscv
pgrep -af 'build_riscv|cc1' | head
ls -l build-riscv/vllm_kestrel
[ -f /tmp/vllm_kestrel.pre_a1 ] || cp -f build-riscv/vllm_kestrel /tmp/vllm_kestrel.pre_a1
if ! pgrep -f build_riscv.sh >/dev/null; then
  nohup sh -c 'sh tools/build/build_riscv.sh > /tmp/a1_build.log 2>&1; echo "EXIT=$?" >> /tmp/a1_build.log' >/dev/null 2>&1 &
  echo STARTED
fi
