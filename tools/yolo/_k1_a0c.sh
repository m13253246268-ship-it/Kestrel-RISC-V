#!/bin/bash
set -u
B=/root/kv/kestrel_vllm_riscv/build-riscv/vllm_kestrel
for S in rv_q4_lane_bt4_worker rv_q4_lane_tile_m2; do
  objdump -d --no-show-raw-insn "$B" | awk -v s="<$S>:" 'index($0,s){p=1;next} p&&/^$/{exit} p' > /tmp/f.s
  echo "=== $S: $(wc -l < /tmp/f.s) insns ==="
  awk -F'\t' '{split($2,a," "); print a[1]}' /tmp/f.s | sort | uniq -c | sort -rn | head -30
  echo "-- vector-only count: $(awk -F'\t' '{split($2,a," "); if (a[1] ~ /^v/) n++} END{print n}' /tmp/f.s)"
  echo "-- has fcvt.s.h / flh: $(grep -cE 'fcvt\.s\.h|flh|vfwcvt' /tmp/f.s)"
done
