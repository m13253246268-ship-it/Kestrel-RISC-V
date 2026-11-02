#!/bin/bash
echo "== d5 =="; grep -aE ' t[468] |LLAMACPP|GB/s' /tmp/d5.log | head -60
echo "== d4 bind =="; grep -aE 'bB|ssB|BIND' /tmp/d4.log | head -20
echo "== cpu_capacity =="; cat /sys/devices/system/cpu/cpu*/cpu_capacity 2>/dev/null | tr '\n' ' '; echo
nproc; uptime
