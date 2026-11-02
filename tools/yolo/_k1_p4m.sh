#!/bin/bash
# P4m：内存峰值（直接采样引擎进程 /proc/PID/status）。4B ctx32 def vs ime1。
cd /root/kv/kestrel_vllm_riscv
B=build-riscv/vllm_kestrel; R=/root/kv/models
COLB="VLLM_Q4_RVV=3 VLLM_Q4_INT=1 VLLM_Q4_INTB=1"
for A in def ime1; do
  X=""; [ $A = ime1 ] && X="VLLM_Q4_IME=1"
  env $COLB $X VLLM_THREADS=8 OMP_NUM_THREADS=8 \
    $B --stream-test --model $R/Qwen3-4B-col --stream-ctx 32 --stream-n 16 --threads 8 > /tmp/p4m.log 2>&1 &
  P=$!; ma=0; mf=0; mh=0
  while kill -0 $P 2>/dev/null; do
    read h a f < <(awk '/^VmHWM/{h=$2}/^RssAnon/{a=$2}/^RssFile/{f=$2}END{print h+0,a+0,f+0}' /proc/$P/status 2>/dev/null)
    [ "${h:-0}" -gt $mh ] && mh=$h; [ "${a:-0}" -gt $ma ] && ma=$a; [ "${f:-0}" -gt $mf ] && mf=$f
    sleep 0.3
  done
  wait $P
  echo "4B ctx32 $A rc=$? VmHWM=$((mh/1024))MB RssAnon_max=$((ma/1024))MB RssFile_max=$((mf/1024))MB txt=$(grep -a '\[STREAM\] text:' /tmp/p4m.log | md5sum | cut -c1-8)"
done
