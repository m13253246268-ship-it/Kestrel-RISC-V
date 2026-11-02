#!/bin/bash
# colb 落地判定：lane f32 默认 vs colb（col 布局 + INT + INTB），端到端 prefill + decode，ctx 32/128/512，3 轮取 min。
# 后台跑；再次运行即轮询。日志 /tmp/colbab.log
L=/tmp/colbab.log
if pgrep -f 'colbab_job' >/dev/null; then echo RUNNING; tail -8 $L; exit 0; fi
if grep -q COLBAB_END $L 2>/dev/null; then cat $L; exit 0; fi
cat > /tmp/colbab_job.sh <<'EOF'
set -u
D=/root/kv/models/Qwen2.5-0.5B
DC=/root/kv/models/Qwen2.5-0.5B-col
B=/root/kv/kestrel_vllm_riscv/build-riscv/vllm_kestrel
uptime; sha256sum "$B"
for C in 32 128 512; do
  for i in 1 2 3; do
    for M in lane colb; do
      if [ $M = lane ]; then DD=$D; EX="VLLM_Q4_RVV=3"; else DD=$DC; EX="VLLM_Q4_RVV=3 VLLM_Q4_INT=1 VLLM_Q4_INTB=1"; fi
      printf 'ctx=%s r=%s %-5s load=%s | ' $C $i $M "$(cut -d' ' -f1 /proc/loadavg)"
      env $EX VLLM_THREADS=8 OMP_NUM_THREADS=8 timeout 900 \
        $B --stream-test --model "$DD" --stream-ctx $C --stream-n 16 --threads 8 --warmup 2>&1 \
        | grep -E '^\[PREFILL-TIMING\]|^\[STREAM\] (prefill|decoded)' | tr '\n' ' ' | cut -c1-400
      echo
    done
  done
done
echo COLBAB_END
EOF
nohup bash /tmp/colbab_job.sh > $L 2>&1 &
echo STARTED
