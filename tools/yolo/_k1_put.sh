#!/bin/bash
# Usage: _k1_put.sh <local-file> <remote-path>
# 密码从环境变量 SSHPASS 读取（sshpass -e），不入库。
set -eu
: "${SSHPASS:?set SSHPASS to the K1 ssh password}"
sshpass -e ssh -p 2223 \
  -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR \
  root@172.26.208.1 "cat > '$2'" < "$1"
echo "PUT_OK $2 $(stat -c %s "$1")"
