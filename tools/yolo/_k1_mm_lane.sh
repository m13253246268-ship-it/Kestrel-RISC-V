#!/bin/bash
# 附加臂：lane-major 权重（生产默认布局）+ VLLM_Q4_RVV=3，用于看 prefill 上限；
# decode 走 RVV lane 内核（不走 vmadot）。日志 /tmp/lane.log。
L=/tmp/lane.log
if pgrep -f 'lane_job.sh' >/dev/null; then echo RUNNING; tail -6 $L; exit 0; fi
if grep -q LANE_END $L 2>/dev/null; then cat $L; exit 0; fi
cat > /tmp/lane_job.sh <<'EOF'
set -u
B=/root/kv/kestrel_vllm_riscv/build-riscv/vllm_kestrel
M=/root/kv/models
O=/tmp/mm; mkdir -p $O
steady() {
  awk '/ t=[0-9]+ ms=/{
         t=0; ms=0;
         for(i=1;i<=NF;i++){ if($i ~ /^t=/){t=substr($i,3)+0} if($i ~ /^ms=/){ms=substr($i,4)+0} }
         if(t>=2){s+=ms;n++} a+=ms;an++
       }
       END{printf "dec=%.1f steady=%.1f(%.2f t/s)", (an?a/an:0), (n?s/n:0), (n?1000/(s/n):0)}' "$1"
}
lane() { # model threads
  local X=$1 T=$2
  local F=$O/lane-$X-t$T.log CPUS=0-7
  [ "$T" = 4 ] && CPUS=0-3
  taskset -c $CPUS env VLLM_Q4_RVV=3 VLLM_THREADS=$T OMP_NUM_THREADS=$T \
    timeout 1800 $B --stream-test --model $M/Qwen3-$X --stream-ctx 32 --stream-n 32 --threads $T --warmup > $F 2>&1
  local PF=$(grep -oE 'prefill [0-9]+ tokens in [0-9.]+s \([0-9]+ tok/s\)' $F | grep -oE '\([0-9]+ tok/s' | tr -dc '0-9')
  printf 'LANE %-5s t%s load=%s | pp=%s tok/s | %s\n' "$X" "$T" "$(cut -d' ' -f1 /proc/loadavg)" "${PF:-NA}" "$(steady $F)"
}
for R in 1 2 3; do
  echo "----- round $R -----"
  for X in 0.6B 1.7B 4B; do lane $X 4; lane $X 8; done
done
echo LANE_END
EOF
nohup bash /tmp/lane_job.sh > $L 2>&1 &
echo STARTED
