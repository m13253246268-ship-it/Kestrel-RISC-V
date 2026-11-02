#!/bin/bash
set -u
HOST=${1:-172.26.208.1}
for p in 2222 2223 5901; do
  if timeout 5 bash -c "exec 3<>/dev/tcp/$HOST/$p" 2>/dev/null; then
    echo "OK   $HOST:$p"
  else
    echo "FAIL $HOST:$p"
  fi
done
