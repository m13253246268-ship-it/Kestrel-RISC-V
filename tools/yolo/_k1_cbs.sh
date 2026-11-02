#!/bin/bash
if pgrep -f 'bash /tmp/cb.sh' >/dev/null; then echo RUNNING; tail -20 /tmp/cb.log; exit 0; fi
if grep -q CB_DONE /tmp/cb.log 2>/dev/null; then cat /tmp/cb.log; exit 0; fi
nohup bash /tmp/cb.sh > /tmp/cb.log 2>&1 &
echo STARTED
