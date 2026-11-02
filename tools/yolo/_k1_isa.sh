#!/bin/bash
grep -m1 -E '^isa' /proc/cpuinfo
gcc --version | head -1
cat > /tmp/zvfh_t.c <<'EOF'
#include <riscv_vector.h>
void f(const _Float16 *a, float *b) {
  size_t vl = __riscv_vsetvl_e16m1(16);
  vfloat16m1_t h = __riscv_vle16_v_f16m1(a, vl);
  __riscv_vse32_v_f32m2(b, __riscv_vfwcvt_f_f_v_f32m2(h, vl), vl);
}
EOF
for A in rv64gcv_zvfh rv64gcv_zvfhmin; do
  gcc -O2 -march=$A -c /tmp/zvfh_t.c -o /tmp/zvfh_t.o 2>&1 | head -3; echo "$A rc=$?"
done
cat > /tmp/zvfh_r.c <<'EOF'
#include <stdio.h>
#include <riscv_vector.h>
int main(void){ _Float16 a[16]; float b[16];
  for(int i=0;i<16;i++) a[i]=(_Float16)(i*0.37f-2.f);
  a[0]=(_Float16)6e-6f; /* subnormal */
  size_t vl = __riscv_vsetvl_e16m1(16);
  __riscv_vse32_v_f32m2(b, __riscv_vfwcvt_f_f_v_f32m2(__riscv_vle16_v_f16m1(a, vl), vl), vl);
  printf("run ok b0=%.9g b5=%.9g\n", b[0], b[5]); return 0; }
EOF
gcc -O2 -march=rv64gcv_zvfh /tmp/zvfh_r.c -o /tmp/zvfh_r 2>&1 | head -3
taskset -c 5 /tmp/zvfh_r; echo "hart5 rc=$?"
