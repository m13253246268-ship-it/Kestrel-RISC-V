#!/bin/bash
# 千问3-30B-A3B 补分片 + 重转 + 校验 + EC 重排 + 冒烟。日志 /tmp/q3b2.log
L=/tmp/q3b2.log
if pgrep -f 'q3b2_job' >/dev/null; then echo RUNNING; tail -8 $L; exit 0; fi
if grep -q Q3B2_END $L 2>/dev/null; then cat $L; exit 0; fi
cat > /tmp/q3b2_job.sh <<'EOF'
set -u
M=Qwen3-30B-A3B
D=/root/kv/models/$M
C=/root/kv/kestrel_vllm_riscv/vqf_convert/vqf_conv
B=/root/kv/kestrel_vllm_riscv/build-riscv/vllm_kestrel
BASE="https://www.modelscope.cn/models/Qwen/$M/resolve/master"
cd "$D"
rm -f model.vqf
EXP="3999417504 3999974192 3997360832 3999975056 3999975400 3999975400 3999975472 3997362064 3999975408 3999975400 3999975408 3987400496 3997353632 3999975400 3999975400 1087928584"
i=0
for e in $EXP; do
  i=$((i+1)); F=$(printf 'model-%05d-of-00016.safetensors' $i)
  for try in 1 2 3; do
    sz=$(stat -c %s "$F" 2>/dev/null || echo 0)
    [ "$sz" = "$e" ] && break
    echo "  fetch $F try$try (have $sz want $e)"
    rm -f "$F"
    curl -sSL --http1.1 -m 3600 -o "$F" "$BASE/$F"
  done
  sz=$(stat -c %s "$F" 2>/dev/null || echo 0)
  echo "  ok $F $sz"
  [ "$sz" = "$e" ] || { echo "SHARD_MISMATCH $F"; echo Q3B2_END; exit 1; }
done
df -h / | tail -1
echo "=== convert (q4 stream) ==="
t0=$(date +%s)
$C --model "$D" --convert-vqf "$D/model.vqf" --wmode q4 --stream 2>&1 | tail -12
echo "  conv_elapsed=$(( $(date +%s)-t0 ))s vqf=$(stat -c %s $D/model.vqf 2>/dev/null || echo 0)"
echo "=== header check ==="
python3 - <<'PY'
import struct,sys
p="/root/kv/models/Qwen3-30B-A3B/model.vqf"
h=open(p,"rb").read(4096)
magic,ver,flags,nt=struct.unpack_from("<IIII",h,0)
dim,nl,nh,nkv,hd,ffn,vocab,_=struct.unpack_from("<8I",h,16)
hasq,hasm=struct.unpack_from("<II",h,64)
doff,flen=struct.unpack_from("<QQ",h,200)
ne,mff,topk,sh=struct.unpack_from("<4I",h,180)
print(f"magic={magic:#x} ver={ver} flags={flags:#x} nt={nt} dim={dim} nl={nl} nh={nh} nkv={nkv} hd={hd} ffn={ffn} has_q_norm={hasq} data_off={doff} file_len={flen} ne={ne} moe_ffn={mff} top_k={topk}")
ok = magic==0x57465146 and flen>0 and (flags & 0x2) and (flags & 0x200)
print("HEADER_OK" if ok else "HEADER_BAD")
PY
echo "=== EC relayout dryrun ==="
/tmp/vqf_down_relayout "$D/model.vqf" 0 2>&1 | tail -4
echo "=== EC relayout WRITE ==="
t0=$(date +%s)
/tmp/vqf_down_relayout "$D/model.vqf" 1 2>&1 | tail -4
echo "  ec_elapsed=$(( $(date +%s)-t0 ))s"
echo "=== smoke (VQF_STREAM=2 DN_EC=1 RVV=3, n=4) ==="
env VLLM_VQF_STREAM=2 VLLM_DN_EC=1 VLLM_Q4_RVV=3 VLLM_THREADS=8 OMP_NUM_THREADS=8 timeout 3000 \
  $B --stream-test --model "$D" --stream-ctx 32 --stream-n 4 --threads 8 2>&1 \
  | grep -E 'PREFILL-TIMING|STREAM\] (prefill|decoded|text)|VQF|FAIL|MoE' | cut -c1-300
echo Q3B2_END
EOF
nohup bash /tmp/q3b2_job.sh > $L 2>&1 &
echo STARTED
