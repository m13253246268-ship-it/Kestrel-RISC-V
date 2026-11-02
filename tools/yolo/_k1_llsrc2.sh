#!/bin/bash
D=/root/kv/llsp; cd $D
B=https://gh-proxy.com/https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-cpu/spacemit
for f in ime_env.h repack.cpp repack.h rvv_kernels.cpp rvv_kernels.h spine_tcm.h spine_barrier.h spine_mem_pool.cpp; do
  [ -s $f ] || curl -fsSL -m 60 "$B/$f" -o $f </dev/null
  echo "$f $(wc -c < $f 2>/dev/null)" >&2
done
echo "== installed lib vmadot" >&2
objdump -d /usr/lib/libggml-cpu.so.0.11.1 | grep -oE 'vmadot[a-z0-9]*' | sort | uniq -c >&2
tar czf - ime.cpp ime.h ime1_kernels.cpp ime_kernels.h ime_env.cpp ime_env.h repack.cpp repack.h rvv_kernels.cpp rvv_kernels.h spine_tcm.h spine_barrier.h spine_mem_pool.cpp 2>/dev/null | base64 -w0
