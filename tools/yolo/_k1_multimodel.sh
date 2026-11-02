#!/bin/bash
# 多模型性能测试：本引擎（Q4 decode 走 vmadot）vs 官方 llama.cpp（SpacemiT 包）。
# 模型 Qwen3-0.6B / 1.7B / 4B；每模型 4 臂（官方 t4、本引擎 t4 ctx32、本引擎 t4 ctx128、本引擎 t8）。
# 每臂 3 轮，轮次间反转顺序。日志 /tmp/mm.log；再次运行即轮询。
L=/tmp/mm.log
if pgrep -f 'mm_job.sh' >/dev/null; then echo RUNNING; tail -6 $L; exit 0; fi
if grep -q MM_END $L 2>/dev/null; then cat $L; exit 0; fi
cat > /tmp/mm_job.sh <<'EOF'
set -u
B=/root/kv/kestrel_vllm_riscv/build-riscv/vllm_kestrel
M=/root/kv/models
G=$M/gguf
O=/tmp/mm; mkdir -p $O
echo "BIN md5=$(md5sum $B | cut -d' ' -f1)  size=$(stat -c %s $B)"
echo "LLAMA $(llama-bench --version 2>&1 | head -1)"
echo "START load=$(cut -d' ' -f1-3 /proc/loadavg)"

ids() { # 逐 token id 序列 → 逗号串
  grep -E 'TOKIDS|^ [0-9]+ t=' "$1" | sed 's/\[STREAM\] TOKIDS://' | sed 's/ t=.*//' \
    | tr -s ' ' '\n' | grep -E '^[0-9]+$' | tr '\n' ','
}
steady() { # 每 token ms：全平均 + 剔除 t=1 的稳态平均
  awk '/ t=[0-9]+ ms=/{
         t=0; ms=0;
         for(i=1;i<=NF;i++){ if($i ~ /^t=/){t=substr($i,3)+0} if($i ~ /^ms=/){ms=substr($i,4)+0} }
         if(t>=2){s+=ms;n++} a+=ms;an++
       }
       END{printf "dec=%.1f/t%d steady=%.1f(%.2f t/s)", (an?a/an:0), an, (n?s/n:0), (n?1000/(s/n):0)}' "$1"
}
ours() { # tag dir ctx threads
  local TAG=$1 D=$2 CTX=$3 T=$4
  local F=$O/ours-$TAG-t$T-ctx$CTX.log CPUS=0-7
  [ "$T" = 4 ] && CPUS=0-3
  taskset -c $CPUS env VLLM_Q4_VMADOT=1 VLLM_THREADS=$T OMP_NUM_THREADS=$T \
    timeout 1800 $B --stream-test --model "$D" --stream-ctx "$CTX" --stream-n 32 --threads $T --warmup > $F 2>&1
  local PF=$(grep -oE 'prefill [0-9]+ tokens in [0-9.]+s \([0-9]+ tok/s\)' $F | grep -oE '\([0-9]+ tok/s' | tr -dc '0-9')
  printf 'OURS %-5s t%s ctx=%-3s load=%s | pp=%s tok/s | %s | ids=%s\n' \
    "$TAG" "$T" "$CTX" "$(cut -d' ' -f1 /proc/loadavg)" "${PF:-NA}" "$(steady $F)" "$(ids $F | md5sum | cut -c1-8)"
}
lb() { # model
  local F=$O/ll-$1.log
  taskset -c 0-3 env OMP_NUM_THREADS=4 timeout 1800 \
    llama-bench -m "$G/Qwen3-$1-Q4_0.gguf" -p 32,128 -n 32 -t 4 -r 3 -o md > $F 2>&1
  local R=$(grep -E '\| +(pp32|pp128|tg32) +\|' $F | awk -F'|' '{gsub(/ /,"",$7); gsub(/ /,"",$8); sub(/±.*/,"",$8); printf "%s=%s ", $7, $8}')
  printf 'OFFICIAL %-5s t4 load=%s | %s| tg_ms=%.1f\n' \
    "$1" "$(cut -d' ' -f1 /proc/loadavg)" "$R" "$(echo "$R" | sed -n 's/.*tg32=\([0-9.]*\).*/1000\/\1/p' | bc -l)"
}
run_model() { # model, order
  local X=$1 ORD=$2
  if [ "$ORD" = fwd ]; then
    lb $X
    ours $X $M/Qwen3-$X-col 32 4
    ours $X $M/Qwen3-$X-col 128 4
    ours $X $M/Qwen3-$X-col 32 8
  else
    ours $X $M/Qwen3-$X-col 32 8
    ours $X $M/Qwen3-$X-col 128 4
    ours $X $M/Qwen3-$X-col 32 4
    lb $X
  fi
}
for R in 1 2 3; do
  echo "----- round $R -----"
  ORD=fwd; [ $((R % 2)) = 0 ] && ORD=rev
  for X in 0.6B 1.7B 4B; do run_model $X $ORD; done
  echo "  round $R load=$(cut -d' ' -f1-3 /proc/loadavg)"
done
echo "END load=$(cut -d' ' -f1-3 /proc/loadavg)"
echo MM_END
EOF
nohup bash /tmp/mm_job.sh > $L 2>&1 &
echo STARTED
