cd /tmp
gcc -O2 -march=rv64gcv_zvfh -mabi=lp64d -ffp-contract=off -fopenmp -o /tmp/bime4 /tmp/bench_ime_q4.c 2>&1 | head -30
echo cc_rc=${PIPESTATUS[0]}
[ -x /tmp/bime4 ] || exit 1
echo "== objdump ime_q4_tile (sp refs / insn count)"
objdump -d /tmp/bime4 | awk '/<ime_q4_tile>:/{f=1} f&&/^$/{exit} f' > /tmp/ime_tile.dis
wc -l < /tmp/ime_tile.dis
grep -cE '\bsp\b' /tmp/ime_tile.dis
grep -E '\bsp\b' /tmp/ime_tile.dis | head
echo "== selftest (tile)"
OMP_WAIT_POLICY=active /tmp/bime4 9 1; echo run_rc=$?
echo "== load $(cat /proc/loadavg)"
echo "== p0 tail"
grep -E '^== |\[STREAM\]|PREFILL-TIMING' /tmp/p0prof.log | tail -6
tail -1 /tmp/p0prof.log
