#!/bin/bash
# 后台调度 _k1_a1b2.sh
if pgrep -f 'bash /tmp/a1b2.sh' >/dev/null; then echo RUNNING; uptime; tail -30 /tmp/a1b2.log; exit 0; fi
if grep -q A1B_DONE /tmp/a1b2.log 2>/dev/null; then cat /tmp/a1b2.log; exit 0; fi
nohup bash /tmp/a1b2.sh > /tmp/a1b2.log 2>&1 &
echo STARTED
