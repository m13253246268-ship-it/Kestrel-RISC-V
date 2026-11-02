#!/bin/bash
# 官方基线：unsloth Q4_0 GGUF 下载 + 与自研 VQF q4 同机同口径对比。日志 /tmp/q3cmp.log
L=/tmp/q3cmp.log
if pgrep -f 'q3cmp_job' >/dev/null; then echo RUNNING; tail -10 $L; exit 0; fi
if grep -q Q3CMP_END $L 2>/dev/null; then cat $L; exit 0; fi
cat > /tmp/q3cmp_job.sh <<'EOF'
set -u
G=/root/kv/models/gguf
B=/root/kv/kestrel_vllm_riscv/build-riscv/vllm_kestrel
mkdir -p "$G"; cd "$G"
rm -f Qwen3-*-Q4_0.gguf.* Qwen3-0.6B-Q4_0.gguf Qwen3-1.7B-Q4_0.gguf Qwen3-4B-Q4_0.gguf
echo "=== dl unsloth Q4_0 ==="
for spec in Qwen3-0.6B:382156480 Qwen3-1.7B:1056782912 Qwen3-4B:2375773472; do
  M=${spec%%:*}; SZ=${spec##*:}; F=$M-Q4_0.gguf
  for try in 1 2 3; do
    have=$(stat -c %s "$F" 2>/dev/null || echo 0)
    [ "$have" = "$SZ" ] && break
    echo "  fetch $F try$try (have $have want $SZ)"
    rm -f "$F"
    curl -sSL --http1.1 -m 3600 -o "$F" "https://www.modelscope.cn/models/unsloth/$M-GGUF/resolve/master/$F"
  done
  echo "  ok $F $(stat -c %s "$F" 2>/dev/null || echo 0)"
  [ "$(stat -c %s "$F" 2>/dev/null || echo 0)" = "$SZ" ] || { echo "GGUF_MISMATCH $F"; echo Q3CMP_END; exit 1; }
done
df -h / | tail -1
echo "=== compare (3 rounds) ==="
eng() { # eng <tag> <vqfdir> <ctx>
  printf 'KESTREL %-11s ctx=%-4s load=%s | ' "$1" "$3" "$(cut -d' ' -f1 /proc/loadavg)"
  env VLLM_Q4_RVV=3 VLLM_THREADS=8 OMP_NUM_THREADS=8 timeout 1800 \
    $B --stream-test --model "$2" --stream-ctx "$3" --stream-n 16 --threads 8 --warmup 2>&1 \
    | grep -E '^\[STREAM\] (prefill|decoded)' | sed 's/\[batch\]//' | tr '\n' ' ' | tr -s ' '
  echo
}
lb() { # lb <tag> <gguf>
  printf 'LLAMACPP %-10s load=%s | ' "$1" "$(cut -d' ' -f1 /proc/loadavg)"
  env OMP_NUM_THREADS=4 timeout 1800 llama-bench -m "$2" -p 32,128 -n 16 -t 4 -r 3 -o md 2>&1 \
    | grep -E '\| +(pp32|pp128|tg16) +\|' | awk -F'|' '{gsub(/ /,"",$7); gsub(/ /,"",$8); sub(/±.*/,"",$8); printf "%s=%s ", $7, $8}'
  echo
}
for R in 1 2 3; do
  echo "----- round $R -----"
  for M in Qwen3-0.6B Qwen3-1.7B Qwen3-4B; do
    lb $M "$G/$M-Q4_0.gguf"
    eng $M /root/kv/models/$M 32
    eng $M /root/kv/models/$M 128
  done
done
uptime
echo Q3CMP_END
EOF
nohup bash /tmp/q3cmp_job.sh > $L 2>&1 &
echo STARTED
