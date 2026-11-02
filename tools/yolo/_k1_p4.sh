#!/bin/bash
# P4：IME K 分块（VLLM_Q4_IME_KB）位级 + A/B；ctx512 补测；内存峰值（VmHWM/RssAnon/RssFile）。
# 后台跑；再次运行即轮询。日志 /tmp/p4.log
L=/tmp/p4.log
if pgrep -f 'p4_job' >/dev/null; then echo RUNNING; tail -8 $L; exit 0; fi
if grep -q P4_END $L 2>/dev/null; then cat $L; exit 0; fi
cat > /tmp/p4_job.sh <<'EOF'
set -u
cd /root/kv/kestrel_vllm_riscv
[ -f /tmp/vllm_kestrel.pre_p4 ] || cp -f build-riscv/vllm_kestrel /tmp/vllm_kestrel.pre_p4
sha256sum src/model/vllm_safetensors.c
sh tools/build/build_riscv.sh > /tmp/p4_build.log 2>&1; echo "BUILD_EXIT=$?"
grep -iE 'error' /tmp/p4_build.log | head -8
B=build-riscv/vllm_kestrel
sha256sum $B
objdump -d $B | awk '/<rv_ime_q4_tile[^>]*>:$/{p=1;next} p&&/^$/{p=0} p' > /tmp/p4_t.s
echo "tile lines=$(wc -l < /tmp/p4_t.s) insn_2b=$(grep -cE '\s[0-9a-f]{6}2b\s' /tmp/p4_t.s) vle32=$(grep -c vle32 /tmp/p4_t.s)"
R=/root/kv/models
COLB="VLLM_Q4_RVV=3 VLLM_Q4_INT=1 VLLM_Q4_INTB=1"
arm() { case $1 in def) echo "";; ime1) echo "VLLM_Q4_IME=1";;
  k16) echo "VLLM_Q4_IME=1 VLLM_Q4_IME_KB=16";; k32) echo "VLLM_Q4_IME=1 VLLM_Q4_IME_KB=32";;
  k64) echo "VLLM_Q4_IME=1 VLLM_Q4_IME_KB=64";; k7) echo "VLLM_Q4_IME=2 VLLM_Q4_IME_KB=7";; esac; }
# 运行并轮询 /proc 取内存峰值（kB）
run() { # $1=out $2.. = cmd
  local O=$1; shift
  "$@" > $O 2>&1 & local P=$!
  local hwm=0 anon=0 file=0 s a f h
  while kill -0 $P 2>/dev/null; do
    s=$(cat /proc/$P/status 2>/dev/null)
    h=$(echo "$s" | awk '/^VmHWM/{print $2}'); a=$(echo "$s" | awk '/^RssAnon/{print $2}'); f=$(echo "$s" | awk '/^RssFile/{print $2}')
    [ -n "$h" ] && [ "$h" -gt $hwm ] && hwm=$h
    [ -n "$a" ] && [ "$a" -gt $anon ] && anon=$a
    [ -n "$f" ] && [ "$f" -gt $file ] && file=$f
    sleep 0.5
  done
  wait $P; RC=$?
  MEM="hwm=$((hwm/1024))MB anon=$((anon/1024))MB file=$((file/1024))MB"
}
echo "== 位级 + NLL（0.6B ctx32 t8；4B ctx32 覆盖 down cols=9728）=="
for M in Qwen3-0.6B Qwen3-4B; do for A in ime1 k16 k32 k7; do
  [ $M = Qwen3-4B ] && [ $A = k16 ] && continue
  O=/tmp/p4_chk_${M}_$A.log
  run $O env $COLB $(arm $A) VLLM_Q4CHK=1 VLLM_STREAM_NLL=1 VLLM_THREADS=8 OMP_NUM_THREADS=8 timeout 1800 \
    $B --stream-test --model $R/$M-col --stream-ctx 32 --stream-n 16 --threads 8
  echo "-- $M $A rc=$RC colbt=$(grep -c 'Q4CHK-colbt' $O)/bad$(grep 'Q4CHK-colbt' $O | grep -vc 'bitmismatch=0') ime=$(grep -c 'Q4CHK-ime' $O)/bad$(grep 'Q4CHK-ime' $O | grep -vc 'bitmismatch=0')"
  grep 'Q4CHK-ime' $O | grep -E 'cols=(9728|3072)' | head -1
  grep -E 'SIGILL|Illegal|Segmentation' $O | head -2
  grep 'CORPUS_NLL_MEAN' $O
done; done
echo "== 4B ctx128 KB 扫描（每臂 2 轮）=="
for r in 1 2; do for A in def ime1 k16 k32 k64; do
  O=/tmp/p4_perf.log
  run $O env $COLB $(arm $A) VLLM_THREADS=8 OMP_NUM_THREADS=8 timeout 1800 \
    $B --stream-test --model $R/Qwen3-4B-col --stream-ctx 128 --stream-n 16 --threads 8 --warmup
  printf 'Qwen3-4B ctx=128 r%s %-4s rc=%s load=%s | ' $r $A $RC "$(cut -d' ' -f1 /proc/loadavg)"
  grep -a '\[STREAM\] prefill' $O | tail -1 | tr '\n' ' '
  grep -a '\[PREFILL-TIMING\]' $O | tail -1
  echo "   $MEM txt=$(grep -a '\[STREAM\] text:' $O | md5sum | cut -c1-8)"
done; done
echo "== ctx512（每臂 2 轮）=="
for M in Qwen3-0.6B Qwen3-4B; do for r in 1 2; do for A in def ime1 k32; do
  O=/tmp/p4_perf.log
  run $O env $COLB $(arm $A) VLLM_THREADS=8 OMP_NUM_THREADS=8 timeout 3600 \
    $B --stream-test --model $R/$M-col --stream-ctx 512 --stream-n 16 --threads 8 --warmup
  printf '%s ctx=512 r%s %-4s rc=%s load=%s | ' $M $r $A $RC "$(cut -d' ' -f1 /proc/loadavg)"
  grep -a '\[STREAM\] prefill' $O | tail -1 | tr '\n' ' '
  grep -a '\[PREFILL-TIMING\]' $O | tail -1
  echo "   $MEM txt=$(grep -a '\[STREAM\] text:' $O | md5sum | cut -c1-8) $(grep -a 'Q4IME' $O | tail -1)"
done; done; done
free -m | head -2
uptime
echo P4_END
EOF
nohup bash /tmp/p4_job.sh > $L 2>&1 &
echo STARTED
