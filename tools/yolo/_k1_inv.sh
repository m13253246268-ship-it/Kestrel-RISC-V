#!/bin/bash
echo "==== 1. llsp 源码是否存在 ===="
ls -la /root/kv/llsp 2>/dev/null | head
echo
echo "==== 2. decode Q4 点积函数（ggml/spacemit 源码） ===="
D=/root/kv/llsp
for f in ime_kernels.h rvv_kernels.cpp rvv_kernels.h ime.cpp ime1_kernels.cpp repack.cpp repack.h; do
  [ -f $D/$f ] || continue
  echo "--- $f ---"
  grep -nE 'vec_dot|q4_0|vmadot|ime_mm|ima_q4|kernel' $D/$f | head -20
done
echo
echo "==== 3. 安装的 libggml-cpu 里 vmadot 指令用量 ===="
l=/usr/lib/libggml-cpu.so.0.11.1
ls -la $l 2>/dev/null
objdump -d $l 2>/dev/null | grep -oE 'vmadot[a-z0-9._]*' | sort | uniq -c | sort -rn | head -20
echo
echo "==== 4. 我们自己的 vllm_kestrel 里 vmadot 用量 ===="
objdump -d /root/kv/kestrel_vllm_riscv/build-riscv/vllm_kestrel 2>/dev/null | grep -oE 'vmadot[a-z0-9._]*' | sort | uniq -c | sort -rn | head
echo "(空=我们没有 vmadot)"
echo
echo "==== 5. CPU 支持的指令扩展 ===="
grep -m2 -iE 'isa|model name|processor' /proc/cpuinfo
echo
echo "==== 6. LIBGGML 版本 / 装入的库 ===="
ldconfig -p 2>/dev/null | grep -i ggml
dpkg -l 2>/dev/null | grep -iE 'ggml|llama'