#!/bin/bash
# C-1a 精度实验：per-row scale 的 q4 权重（格式不变，引擎零改动）
# 独立构建路径，不覆盖公用 vqf_conv；每次重转，记录 sha256。
set -u
SRC=/root/kv/kestrel_vllm_riscv/vqf_convert
B=/tmp/vqf_conv_rowq4
cd "$SRC" || exit 1
echo "=== 构建 $B ==="
rm -f "$B"
gcc -O2 -I src src/conv_main.c src/model_cfgio.c src/model_layers.c \
    src/quant_kernels.c src/qk_repack.c src/vqf_vision.c src/vllm_crypto.c \
    -o "$B" -lm > /tmp/vqf_conv_rowq4.cc.log 2>&1 \
    || { grep -E 'error|undefined' /tmp/vqf_conv_rowq4.cc.log | head -20; echo BUILD_FAIL; exit 1; }
sha256sum src/model_layers.c src/quant_kernels.c "$B"
D=/root/kv/models/Qwen2.5-0.5B
DR=/root/kv/models/Qwen2.5-0.5B-rowq4
mkdir -p "$DR"
rm -f "$DR/model.vqf"
for f in "$D"/*; do
  n=$(basename "$f"); [ "$n" = model.vqf ] && continue
  ln -sf "$f" "$DR/$n"
done
echo "=== 转 per-row scale q4 ==="
VLLM_Q4_ROWSCALE=1 "$B" --model "$D" --convert-vqf "$DR/model.vqf" --wmode q4 \
  > /tmp/rowq4.conv.log 2>&1 || { tail -20 /tmp/rowq4.conv.log; echo CONV_FAIL; exit 1; }
grep -E 'sum=|error|fail|tensors' /tmp/rowq4.conv.log | head
echo "ROWSCALE WARN 次数: $(grep -c 'ROWSCALE\] WARN' /tmp/rowq4.conv.log)"
ls -l "$D/model.vqf" "$DR/model.vqf"
sha256sum "$DR/model.vqf"
echo ROWQ4_DONE
