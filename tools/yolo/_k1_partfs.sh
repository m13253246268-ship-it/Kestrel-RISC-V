#!/bin/bash
# 后台启动/轮询 /tmp/partf.sh。参数 new = 清旧日志并重跑（需先投送新的 /tmp/partf.sh）。
L=/tmp/partf.log
if pgrep -f 'bash /tmp/partf.sh' >/dev/null; then echo RUNNING; tail -20 $L; exit 0; fi
if [ "${1:-}" = new ]; then rm -f $L /tmp/partf_*.log
elif grep -q PF_DONE $L 2>/dev/null; then cat $L; exit 0; fi
nohup flock -n /tmp/partf.lock timeout 7200 bash /tmp/partf.sh > $L 2>&1 &
echo STARTED
