#!/bin/bash
# r10：colb 默认开后重建 + 验证（col 模型不设 env 应走 colb；lane 模型不受影响）+ attention 反汇编检查。
# 后台跑；再次运行即轮询。日志 /tmp/r10b.log
L=/tmp/r10b.log
if pgrep -f 'r10b_job' >/dev/null; then echo RUNNING; tail -6 $L; exit 0; fi
if grep -q R10B_END $L 2>/dev/null; then cat $L; exit 0; fi
cat > /tmp/r10b_job.sh <<'EOF'
set -u
cd /root/kv/kestrel_vllm_riscv
[ -f /tmp/vllm_kestrel.pre_r10 ] || cp -f build-riscv/vllm_kestrel /tmp/vllm_kestrel.pre_r10
sha256sum src/model/vllm_safetensors.c
sh tools/build/build_riscv.sh > /tmp/r10_build.log 2>&1; echo "BUILD_EXIT=$?"
tail -2 /tmp/r10_build.log
B=build-riscv/vllm_kestrel
sha256sum $B
echo "== ATTN disasm (vector insn count per func) =="
for F in vllm_attn_batched_worker vllm_attn_pv_tile128 hsum_neon4; do
  N=$(objdump -d --no-show-raw-insn $B | awk -v f="<$F" 'index($0,f)&&/>:$/{p=1;next} p&&/^$/{p=0} p' | grep -cE '\sv[a-z0-9.]+\s' )
  T=$(objdump -d --no-show-raw-insn $B | awk -v f="<$F" 'index($0,f)&&/>:$/{p=1;next} p&&/^$/{p=0} p' | wc -l)
  echo "$F total=$T vec=$N"
done
while ! grep -q Q8ROW_END /tmp/q8row.log 2>/dev/null; do sleep 20; done
uptime
D=/root/kv/models/Qwen2.5-0.5B
DC=/root/kv/models/Qwen2.5-0.5B-col
for M in lane col_noenv col_off; do
  case $M in
    lane) DD=$D; EX="VLLM_Q4_RVV=3";;
    col_noenv) DD=$DC; EX="VLLM_Q4_RVV=3";;
    col_off) DD=$DC; EX="VLLM_Q4_RVV=3 VLLM_Q4_INTB=0";;
  esac
  printf '%-9s load=%s | ' $M "$(cut -d' ' -f1 /proc/loadavg)"
  env $EX VLLM_THREADS=8 OMP_NUM_THREADS=8 timeout 900 \
    $B --stream-test --model "$DD" --stream-ctx 128 --stream-n 16 --threads 8 --warmup 2>&1 \
    | grep -E '^\[PREFILL-TIMING\]|^\[STREAM\] (prefill|decoded)|^\[STREAM\] (out|text)' | tr '\n' ' ' | cut -c1-500
  echo
done
echo R10B_END
EOF
nohup bash /tmp/r10b_job.sh > $L 2>&1 &
echo STARTED
