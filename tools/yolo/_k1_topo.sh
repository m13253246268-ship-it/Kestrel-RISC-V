#!/bin/bash
echo "=== L2 topology ==="
for c in 0 1 2 3 4 5 6 7; do
  printf "cpu%s L2_shared=%s maxfreq=%s\n" "$c" \
    "$(cat /sys/devices/system/cpu/cpu$c/cache/index2/shared_cpu_list 2>/dev/null)" \
    "$(cat /sys/devices/system/cpu/cpu$c/cpufreq/cpuinfo_max_freq 2>/dev/null)"
done
echo "=== L2 size ==="
cat /sys/devices/system/cpu/cpu0/cache/index2/size /sys/devices/system/cpu/cpu4/cache/index2/size 2>/dev/null
echo "=== load ==="
uptime
echo "=== taskset available ==="
command -v taskset || echo "(no taskset)"
