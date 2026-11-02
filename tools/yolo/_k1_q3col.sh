#!/bin/bash
# Qwen3 col(mode3)+colb：重下 safetensors → VLLM_Q4_COL=1 转换 → 质量门(NLL/needle, lane vs colb)
# → 以官方 llama.cpp Q4_0 为基线 3 轮重测。后台跑；再次运行即轮询。日志 /tmp/q3col.log
L=/tmp/q3col.log
if pgrep -f 'q3col_job' >/dev/null; then echo RUNNING; tail -12 $L; exit 0; fi
if grep -q Q3COL_END $L 2>/dev/null; then cat $L; exit 0; fi
cat > /tmp/q3col_job.sh <<'EOF'
set -u
R=/root/kv/models
G=$R/gguf
B=/root/kv/kestrel_vllm_riscv/build-riscv/vllm_kestrel
C=/root/kv/kestrel_vllm_riscv/vqf_convert/vqf_conv
sha256sum $B $C; df -h / | tail -1
MODELS="Qwen3-0.6B Qwen3-1.7B Qwen3-4B"

echo "=== phase1: col convert ==="
for M in $MODELS; do
  D=$R/$M; DC=$R/$M-col
  if [ -s "$DC/model.vqf" ]; then echo "  skip $M (col exists)"; continue; fi
  API="https://www.modelscope.cn/api/v1/models/Qwen/$M/repo/files?Revision=master&Root="
  curl -sL --http1.1 "$API" | python3 -c '
import json,sys
for f in json.load(sys.stdin)["Data"]["Files"]:
    p=f["Path"]
    if p.endswith(".safetensors") or p=="model.safetensors.index.json": print(p, f["Size"])
' > /tmp/q3col_files.txt
  ok=1
  while read -r F SZ; do
    for t in 1 2 3; do
      [ "$(stat -c %s "$D/$F" 2>/dev/null || echo 0)" = "$SZ" ] && break
      rm -f "$D/$F"
      curl -sL --http1.1 -m 3600 -o "$D/$F" "https://www.modelscope.cn/models/Qwen/$M/resolve/master/$F" </dev/null
    done
    have=$(stat -c %s "$D/$F" 2>/dev/null || echo 0)
    echo "  $M/$F $have/$SZ"; [ "$have" = "$SZ" ] || ok=0
  done < /tmp/q3col_files.txt
  if [ $ok != 1 ]; then echo "DL_FAIL $M"; continue; fi
  mkdir -p "$DC"
  VLLM_Q4_COL=1 $C --model "$D" --convert-vqf "$DC/model.vqf" --wmode q4 > /tmp/q3col_conv_$M.log 2>&1
  rc=$?
  echo "  conv $M exit=$rc $(grep -E 'repack gates|after-tensor-scan' /tmp/q3col_conv_$M.log | tr '\n' ' ')"
  if [ $rc != 0 ]; then echo "CONV_FAIL $M (safetensors kept)"; tail -5 /tmp/q3col_conv_$M.log; rm -f "$DC/model.vqf"; continue; fi
  rm -f "$D"/*.safetensors
  for f in "$D"/*; do n=$(basename "$f"); [ "$n" = model.vqf ] && continue; ln -sf "$f" "$DC/$n"; done
  ls -l "$D/model.vqf" "$DC/model.vqf"; df -h / | tail -1
done

echo "=== phase2: quality gate (lane vs colb) ==="
LANE="VLLM_Q4_RVV=3"
COLB="VLLM_Q4_RVV=3 VLLM_Q4_INT=1 VLLM_Q4_INTB=1"
nll() { # tag dir envs
  env $3 VLLM_STREAM_NLL=1 VLLM_THREADS=8 OMP_NUM_THREADS=8 timeout 5400 $B --stream-test \
    --model "$2" --threads 8 > /tmp/q3col_nl_$1.log 2>&1
  echo "  nll $1 rc=$? $(grep -E 'NLL_MEAN' /tmp/q3col_nl_$1.log | tail -1)"
}
swp() { # tag dir envs
  env $3 VLLM_THREADS=8 OMP_NUM_THREADS=8 timeout 7200 $B --needle-sweep \
    --longctx-ctx 1024 --longctx-gen 32 --model "$2" --threads 8 > /tmp/q3col_sw_$1.log 2>&1
  echo "  sweep $1 rc=$? retrieved=$(grep -c 'Needle retrieved  YES' /tmp/q3col_sw_$1.log)/$(grep -c 'Needle retrieved' /tmp/q3col_sw_$1.log)"
  grep -E 'Answer PPL' /tmp/q3col_sw_$1.log | sed 's/  F32=.*//' | tr -s ' ' | tr '\n' ' '; echo
}
for M in $MODELS; do
  [ -s "$R/$M-col/model.vqf" ] || { echo "  no col for $M"; continue; }
  nll ${M}_lane "$R/$M" "$LANE"; nll ${M}_colb "$R/$M-col" "$COLB"
done
for M in Qwen3-0.6B Qwen3-1.7B; do
  [ -s "$R/$M-col/model.vqf" ] || continue
  swp ${M}_lane "$R/$M" "$LANE"; swp ${M}_colb "$R/$M-col" "$COLB"
done

echo "=== phase3: compare vs llama.cpp (3 rounds) ==="
eng() { # tag dir ctx threads envs
  printf 'KESTREL %-16s t%s ctx=%-4s load=%s | ' "$1" "$4" "$3" "$(cut -d' ' -f1 /proc/loadavg)"
  env $5 VLLM_THREADS=$4 OMP_NUM_THREADS=$4 timeout 1800 \
    $B --stream-test --model "$2" --stream-ctx "$3" --stream-n 16 --threads $4 --warmup 2>&1 \
    | grep -E '^\[STREAM\] (prefill|decoded)' | sed 's/\[batch\]//' | tr '\n' ' ' | tr -s ' '
  echo
}
lb() { # tag gguf
  printf 'LLAMACPP %-15s t4 load=%s | ' "$1" "$(cut -d' ' -f1 /proc/loadavg)"
  env OMP_NUM_THREADS=4 timeout 1800 llama-bench -m "$2" -p 32,128 -n 16 -t 4 -r 3 -o md 2>&1 \
    | grep -E '\| +(pp32|pp128|tg16) +\|' | awk -F'|' '{gsub(/ /,"",$7); gsub(/ /,"",$8); sub(/±.*/,"",$8); printf "%s=%s ", $7, $8}'
  echo
}
for RND in 1 2 3; do
  echo "----- round $RND -----"
  for M in $MODELS; do
    lb $M "$G/$M-Q4_0.gguf"
    for X in 32 128; do
      eng ${M}_lane "$R/$M" $X 8 "$LANE"
      [ -s "$R/$M-col/model.vqf" ] && eng ${M}_colb "$R/$M-col" $X 8 "$COLB"
      [ -s "$R/$M-col/model.vqf" ] && eng ${M}_colb "$R/$M-col" $X 4 "$COLB"
    done
  done
done
uptime
echo Q3COL_END
EOF
nohup bash /tmp/q3col_job.sh > $L 2>&1 &
echo STARTED
