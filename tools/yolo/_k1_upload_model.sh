#!/bin/bash
set -eu
SRC=/mnt/e/项目/Kestrel_VLLM_RISC-V/modls
: "${SSHPASS:?set SSHPASS to the K1 ssh password}"   # 密码走环境变量，不入库
cd "$SRC"
tar cf - Qwen2.5-0.5B \
  | sshpass -e ssh -p 2223 \
      -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR \
      -o Compression=no \
      root@172.26.208.1 'mkdir -p /root/kv/models && tar xf - -C /root/kv/models && echo UPLOAD_OK && du -sh /root/kv/models/Qwen2.5-0.5B && ls -l /root/kv/models/Qwen2.5-0.5B'
