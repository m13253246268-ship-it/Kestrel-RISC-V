#!/bin/bash
# K1 online device helper: run a remote command over the stunnel-wrapped SSH.
# 密码从环境变量 SSHPASS 读取（sshpass -e），不入库。
set -u
: "${SSHPASS:?set SSHPASS to the K1 ssh password}"
exec sshpass -e ssh -p 2223 \
  -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  -o LogLevel=ERROR -o ConnectTimeout=20 \
  root@172.26.208.1 "$@"
