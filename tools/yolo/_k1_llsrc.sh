#!/bin/bash
D=/root/kv/llsp; mkdir -p $D; cd $D
for f in ime.cpp ime.h ime1_kernels.cpp ime2_kernels.cpp ime_kernels.h ime_env.cpp ime_env.h repack.cpp repack.h rvv_kernels.cpp rvv_kernels.h spine_tcm.h; do
  ok=0
  for base in https://raw.githubusercontent.com/ggml-org/llama.cpp/master https://ghfast.top/https://raw.githubusercontent.com/ggml-org/llama.cpp/master https://gh-proxy.com/https://raw.githubusercontent.com/ggml-org/llama.cpp/master; do
    if curl -fsSL -m 40 "$base/ggml/src/ggml-cpu/spacemit/$f" -o $f </dev/null; then ok=1; break; fi
  done
  echo "$f ok=$ok $(wc -c < $f 2>/dev/null)"
done
echo "== installed lib vmadot"
objdump -d /usr/lib/libggml-cpu.so.0.11.1 | grep -c vmadot
objdump -d /usr/lib/libggml-cpu.so.0.11.1 | grep -oE 'vmadot[a-z0-9]*' | sort | uniq -c
cat /proc/cpuinfo | grep -m3 -iE 'isa|model name|uarch'
