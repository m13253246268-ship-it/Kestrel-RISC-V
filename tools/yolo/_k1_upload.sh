#!/bin/bash
# Upload kestrel source tree (without build dir) to the online K1.
set -eu
SRC=/mnt/e/项目/Kestrel_VLLM_RISC-V
: "${SSHPASS:?set SSHPASS to the K1 ssh password}"   # 密码走环境变量，不入库
cd "$SRC"
tar czf - --exclude=build-riscv --exclude='*.o' kestrel_vllm_riscv \
  | sshpass -e ssh -p 2223 \
      -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR \
      root@172.26.208.1 'mkdir -p /root/kv && tar xzf - -C /root/kv && echo UPLOAD_OK && du -sh /root/kv/kestrel_vllm_riscv'
