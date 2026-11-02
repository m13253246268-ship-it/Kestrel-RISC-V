#!/bin/bash
set -u
export DEBIAN_FRONTEND=noninteractive
apt-get install -y llama.cpp-tools-spacemit 2>&1 | tail -8
echo "=== binaries ==="
dpkg -L llama.cpp-tools-spacemit 2>/dev/null | grep -E 'bin/' | head -20
ls -l /usr/bin/llama* 2>/dev/null | head
echo "=== version ==="
llama-cli --version 2>&1 | head -5 || /usr/bin/llama-cli --version 2>&1 | head -5
echo Q3_INSTALL_END
