#!/bin/bash
# d2 精度门：q8 per-block（基线） vs q8 per-row scale（IME per-channel 入口的精度代价）。
# 格式不变、引擎零改动。后台跑；再次运行即轮询。需在 colbab 计时结束后启动。
L=/tmp/q8row.log
if pgrep -f 'q8row_job' >/dev/null; then echo RUNNING; tail -12 $L; exit 0; fi
if grep -q Q8ROW_END $L 2>/dev/null; then cat $L; exit 0; fi
if pgrep -f 'colbab_job' >/dev/null; then echo WAIT_COLBAB; exit 0; fi
cat > /tmp/q8row_job.sh <<'EOF'
set -u
SRC=/root/kv/kestrel_vllm_riscv/vqf_convert
CB=/tmp/vqf_conv_q8row
cd "$SRC" || exit 1
gcc -O2 -I src src/conv_main.c src/model_cfgio.c src/model_layers.c \
    src/quant_kernels.c src/qk_repack.c src/vqf_vision.c src/vllm_crypto.c \
    -o "$CB" -lm > /tmp/vqf_conv_q8row.cc.log 2>&1 \
    || { grep -E 'error|undefined' /tmp/vqf_conv_q8row.cc.log | head -20; echo BUILD_FAIL; echo Q8ROW_END; exit 1; }
sha256sum src/model_layers.c src/quant_kernels.c "$CB"
D=/root/kv/models/Qwen2.5-0.5B
B=/root/kv/kestrel_vllm_riscv/build-riscv/vllm_kestrel
for V in q8 q8row; do
  DR=/root/kv/models/Qwen2.5-0.5B-$V
  mkdir -p "$DR"; rm -f "$DR/model.vqf"
  for f in "$D"/*; do n=$(basename "$f"); [ "$n" = model.vqf ] && continue; ln -sf "$f" "$DR/$n"; done
  if [ $V = q8row ]; then EX="VLLM_Q8_ROWSCALE=1"; else EX="VLLM_Q8_ROWSCALE=0"; fi
  env $EX "$CB" --model "$D" --convert-vqf "$DR/model.vqf" --wmode q8 > /tmp/$V.conv.log 2>&1 \
    || { tail -20 /tmp/$V.conv.log; echo CONV_FAIL; echo Q8ROW_END; exit 1; }
  echo "$V WARN=$(grep -c 'ROWSCALE8\] WARN' /tmp/$V.conv.log)"; ls -l "$DR/model.vqf"; sha256sum "$DR/model.vqf"
done
uptime; sha256sum "$B"
for V in q8 q8row; do
  DR=/root/kv/models/Qwen2.5-0.5B-$V
  echo "----- nll $V -----"
  env VLLM_STREAM_NLL=1 VLLM_THREADS=8 OMP_NUM_THREADS=8 timeout 3600 $B --stream-test \
    --model "$DR" --threads 8 > /tmp/q8row_nl_$V.log 2>&1
  echo "rc=$?"; grep -E 'NLL|FAIL|WARN|wmode|q8' /tmp/q8row_nl_$V.log | head -20
  echo "----- sweep $V -----"
  env VLLM_THREADS=8 OMP_NUM_THREADS=8 timeout 3600 $B --needle-sweep \
    --longctx-ctx 1024 --longctx-gen 64 --model "$DR" --threads 8 > /tmp/q8row_sw_$V.log 2>&1
  echo "rc=$?"; grep -E 'S=1024 pos|Needle retrieved|Answer PPL|Greedy|FAIL' /tmp/q8row_sw_$V.log | cut -c1-160
done
echo Q8ROW_END
EOF
nohup bash /tmp/q8row_job.sh > $L 2>&1 &
echo STARTED
