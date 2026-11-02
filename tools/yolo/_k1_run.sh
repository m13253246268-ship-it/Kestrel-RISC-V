#!/bin/bash
# Usage: _k1_run.sh <local-script>  -> pipes the script to the remote bash
set -eu
SCRIPT="$1"
: "${SSHPASS:?set SSHPASS to the K1 ssh password}"   # 密码走环境变量，不入库
sshpass -e ssh -p 2223 \
  -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR \
  root@172.26.208.1 'bash -s' < "$SCRIPT"
