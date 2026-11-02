#!/bin/bash
# 4 线程 prefill A/B：纯 IME vmadot（VLLM_Q4_IME=2）vs colbt（=0），Qwen3 0.6B/1.7B/4B。
# 口径：taskset -c 0-3 + VLLM_THREADS=4；两臂都开 VLLM_Q4_VMADOT=1（decode 保持一致）。
# 日志 /tmp/ime4.log，单臂日志 /tmp/ime4/。再次运行脚本即轮询。
L=/tmp/ime4.log
if pgrep -f 'ime4_job.sh' >/dev/null; then echo RUNNING; tail -4 $L; exit 0; fi
if grep -q IME4_END $L 2>/dev/null; then cat $L; exit 0; fi
cat > /tmp/ime4_job.sh <<'EOF'
set -u
B=/root/kv/kestrel_vllm_riscv/build-riscv/vllm_kestrel
M=/root/kv/models
O=/tmp/ime4; mkdir -p $O
echo "BIN md5=$(md5sum $B | cut -d' ' -f1)"
echo "START load=$(cut -d' ' -f1-3 /proc/loadavg)"

ids() { grep -E 'TOKIDS|^ [0-9]+ t=' "$1" | sed 's/\[STREAM\] TOKIDS://' | sed 's/ t=.*//' \
        | tr -s ' ' '\n' | grep -E '^[0-9]+$' | tr '\n' ',' | md5sum | cut -c1-8; }
steady() { awk '/ t=[0-9]+ ms=/{t=0;ms=0;
             for(i=1;i<=NF;i++){if($i~/^t=/){t=substr($i,3)+0} if($i~/^ms=/){ms=substr($i,4)+0}}
             if(t>=2){s+=ms;n++}} END{printf "%.1f", (n?s/n:0)}' "$1"; }
pf() { grep -oE 'prefill [0-9]+ tokens in [0-9.]+s \([0-9]+ tok/s\)' "$1" | grep -oE '\([0-9]+ tok/s' | tr -dc '0-9'; }
tim() { # total/GEMM/ATTN/OTHER
  grep -oE 'PREFILL-TIMING.*' "$1" | sed 's/.*total=\([0-9.]*\)ms.*GEMM=\([0-9.]*\)ms.*ATTN=\([0-9.]*\)ms.*OTHER=\([0-9.]*\)ms.*/\1 \2 \3 \4/' | tail -1; }
kern() {
  grep -oE 'PREFILL-KERNELS.*' "$1" | sed 's/.*QKV=\([0-9.]*\)ms.*O=\([0-9.]*\)ms.*GATEUP=\([0-9.]*\)ms.*DOWN=\([0-9.]*\)ms.*/\1 \2 \3 \4/' | tail -1; }

arm() { # tag ctx ime
  local X=$1 CTX=$2 IME=$3
  local TAG="$X-i$IME-c$CTX"
  local F=$O/$TAG.log
  taskset -c 0-3 env VLLM_Q4_VMADOT=1 VLLM_THREADS=4 OMP_NUM_THREADS=4 VLLM_Q4_IME=$IME \
    timeout 3600 $B --stream-test --model $M/Qwen3-$X-col --stream-ctx $CTX \
    --stream-n 32 --threads 4 --warmup > $F 2>&1
  local T=$(tim $F) K=$(kern $F)
  printf 'A %-5s ctx=%-3s IME=%s load=%s | pp=%s t/s | tot/GEMM/ATTN/OTHER=%s | QKV/O/GU/DN=%s | dec=%s ms | ids=%s\n' \
    "$X" "$CTX" "$IME" "$(cut -d' ' -f1 /proc/loadavg)" "$(pf $F)" "${T:-NA}" "${K:-NA}" "$(steady $F)" "$(ids $F)"
}

for R in 1 2 3; do
  echo "----- round $R -----"
  if [ $((R % 2)) = 1 ]; then ORD="0 2"; else ORD="2 0"; fi
  for X in 0.6B 1.7B 4B; do
    for CTX in 32 128; do
      for IME in $ORD; do arm $X $CTX $IME; done
    done
  done
  echo "  round $R load=$(cut -d' ' -f1-3 /proc/loadavg)"
done
echo "END load=$(cut -d' ' -f1-3 /proc/loadavg)"
echo IME4_END
EOF
nohup bash /tmp/ime4_job.sh > $L 2>&1 &
echo STARTED
