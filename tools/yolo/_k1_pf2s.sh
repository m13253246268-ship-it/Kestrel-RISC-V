#!/bin/bash
# 后台启动/轮询 /tmp/pf2.sh。参数 new = 清旧日志并重跑（需先投送新的 /tmp/pf2.sh）。
L=/tmp/pf2.log
if pgrep -f 'bash /tmp/pf2.sh' >/dev/null; then echo RUNNING; tail -12 $L; exit 0; fi
if [ "${1:-}" = new ]; then rm -f $L /tmp/pf2_*.log
elif grep -q PF2_DONE $L 2>/dev/null; then cat $L; exit 0; fi
nohup flock -n /tmp/pf2.lock timeout 7200 bash /tmp/pf2.sh > $L 2>&1 &
echo STARTED
