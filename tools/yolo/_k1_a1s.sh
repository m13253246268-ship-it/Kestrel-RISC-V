#!/bin/bash
# A-1 A/B 后台调度：运行中 -> 打印尾部；已完成 -> 打印全文；否则启动
if pgrep -f 'bash /tmp/a1.sh' >/dev/null; then echo RUNNING; uptime; tail -4 /tmp/a1.log; exit 0; fi
if grep -q A1_DONE /tmp/a1.log 2>/dev/null; then cat /tmp/a1.log; exit 0; fi
nohup bash /tmp/a1.sh > /tmp/a1.log 2>&1 &
echo STARTED
