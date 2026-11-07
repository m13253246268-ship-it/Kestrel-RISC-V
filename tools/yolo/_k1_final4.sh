#!/bin/bash
# 最终 4 线程对标：官方 llama-bench(Q4_0) vs 本引擎（VLLM_Q4_IME=2），同会话、同负载、3 轮交替。
# 日志 /tmp/final4.log；单臂日志 /tmp/final4/。再次运行脚本即轮询。
L=/tmp/final4.log
if pgrep -f 'final4_job.sh' >/dev/null; then echo RUNNING; tail -4 $L; exit 0; fi
if grep -q FINAL4_END $L 2>/dev/null; then cat $L; exit 0; fi
cat > /tmp/final4_job.sh <<'EOF'
set -u
B=/root/kv/kestrel_vllm_riscv/build-riscv/vllm_kestrel
M=/root/kv/models
G=$M/gguf
O=/tmp/final4; mkdir -p $O
echo "BIN md5=$(md5sum $B | cut -d' ' -f1)"
echo "GGUF Q4_0 sha=$(sha256sum $G/Qwen3-0.6B-Q4_0.gguf | cut -c1-16) ($(ls -l $G/*.gguf | awk '{print $5}' | tr '\n' ' '))"
echo "VQF(Q4-only) sizes=$(ls -l $M/Qwen3-0.6B-col/model.vqf $M/Qwen3-1.7B-col/model.vqf $M/Qwen3-4B-col/model.vqf | awk '{print $5}' | tr '\n' ' ')"
echo "START load=$(cut -d' ' -f1-3 /proc/loadavg)"

# TOKIDS 是多行块（每行 `<id> t=<step> ms=<x> rss=<y>`），必须把整段 id 序列一起哈希：
# 只取第一行会退化成「首 token 相同」，所有模型都会算出同一个 md5，失去对拍意义。
ids() { grep -E 'TOKIDS|^ *[0-9]+ t=' "$1" \
        | sed -E 's/.*TOKIDS: *//; s/ t=.*//' \
        | tr -s ' \n' '\n' | grep -E '^[0-9]+$' | tr '\n' ',' | md5sum | cut -c1-8; }
idsn() { grep -E 'TOKIDS|^ *[0-9]+ t=' "$1" | sed -E 's/.*TOKIDS: *//; s/ t=.*//' \
        | tr -s ' \n' '\n' | grep -cE '^[0-9]+$'; }
steady() { awk '/ t=[0-9]+ ms=/{t=0;ms=0;
             for(i=1;i<=NF;i++){if($i~/^t=/){t=substr($i,3)+0} if($i~/^ms=/){ms=substr($i,4)+0}}
             if(t>=2){s+=ms;n++}} END{printf "%.1f", (n?s/n:0)}' "$1"; }
# prefill 实际 token 数与按实际 N 反算的 tok/s（引擎只打印整数 tok/s，且 ctx=128 实为 118 token）
pf() { grep -oE 'prefill [0-9]+ tokens in [0-9.]+s' "$1" | tail -1 \
        | awk '{n=$2; s=$5; s=substr(s,1,length(s)-1)+0; if(s>0) printf "%.2f/%d", n/s, n; else printf "0/0"}'; }

ours() { # X ctx
  local X=$1
  local CTX=$2
  local F=$O/ours-$X-c$CTX.log
  taskset -c 0-3 env VLLM_Q4_VMADOT=1 VLLM_THREADS=4 OMP_NUM_THREADS=4 VLLM_Q4_IME=2 \
    timeout 3600 $B --stream-test --model $M/Qwen3-$X-col --stream-ctx $CTX --stream-n 32 \
    --threads 4 --warmup > $F 2>&1
  local P="$(pf $F)"
  printf 'OURS %-5s c%-4s load=%s | pp=%s tok/s (n=%s) | dec=%s ms | ids=%s nid=%s\n' \
    "$X" "$CTX" "$(cut -d' ' -f1 /proc/loadavg)" "${P%/*}" "${P#*/}" "$(steady $F)" "$(ids $F)" "$(idsn $F)"
}
lb() { # X
  local X=$1
  local F=$O/lb-$X.log
  taskset -c 0-3 env OMP_NUM_THREADS=4 timeout 3600 \
    llama-bench -m "$G/Qwen3-$X-Q4_0.gguf" -p 32,128 -n 32 -t 4 -r 3 -o md > $F 2>&1
  local R=$(grep -E '\| +(pp32|pp128|tg32) +\|' $F | awk -F'|' '{gsub(/ /,"",$7); gsub(/ /,"",$8); sub(/±.*/,"",$8); printf "%s=%s ", $7, $8}')
  printf 'OFFICIAL %-5s load=%s | %s| tg_ms=%.1f\n' \
    "$X" "$(cut -d' ' -f1 /proc/loadavg)" "$R" "$(echo "$R" | sed -n 's/.*tg32=\([0-9.]*\).*/1000\/\1/p' | bc -l)"
}

for R in 1 2 3; do
  echo "----- round $R -----"
  for X in 0.6B 1.7B 4B; do
    if [ $((R % 2)) = 1 ]; then lb $X; ours $X 32; ours $X 128
    else ours $X 128; ours $X 32; lb $X; fi
  done
  echo "  round $R load=$(cut -d' ' -f1-3 /proc/loadavg)"
done
echo "END load=$(cut -d' ' -f1-3 /proc/loadavg)"
echo FINAL4_END
EOF
nohup bash /tmp/final4_job.sh > $L 2>&1 &
echo STARTED
