#!/bin/bash
echo "==== A. 我们 vllm_kestrel 里 vmadot 落在哪些函数 ===="
B=/root/kv/kestrel_vllm_riscv/build-riscv/vllm_kestrel
objdump -d $B 2>/dev/null | awk '/^[0-9a-f]+ </{fn=$0} /\tvmadot/{print fn} /^[0-9a-f]+ <\./{fn=$0}' | grep '<' | sort | uniq -c | sort -rn | head -20
echo
echo "==== B. 我们的 decode Q4 内核函数名 ===="
nm -n $B 2>/dev/null | grep -iE 'rv_q4_dec|dec_tile|q4_dec' | head
echo
echo "==== C. llsp 里 decode 路径点积实现（vec_mul / ime1 Q4Q8） ===="
D=/root/kv/llsp
grep -nE 'mul_mat_vec|vec_dot_q4_0|q4_0_q8_0|ime1|gemm_kernel_i8i4|M4 kernel|Q4_0' $D/ime.cpp $D/ime1_kernels.cpp $D/ime_env.cpp 2>/dev/null | head -40