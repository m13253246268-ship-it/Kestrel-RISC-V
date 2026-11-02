cat > /tmp/zvfh_a.c <<'EOF'
#include <stdio.h>
#include <stdint.h>
#include <string.h>
static uint32_t sw(uint16_t h){
  uint32_t s=(uint32_t)(h&0x8000)<<16, e=(h>>10)&0x1f, m=h&0x3ff;
  if(e==0){ if(m==0) return s; e=1; while(!(m&0x400)){m<<=1;e--;} m&=0x3ff; return s|((e+112)<<23)|(m<<13); }
  if(e==31) return s|0x7f800000|(m<<13);
  return s|((e+112)<<23)|(m<<13);
}
int main(){
  static uint16_t a[65536]; static uint32_t b[65536];
  for(int i=0;i<65536;i++) a[i]=(uint16_t)i;
  for(int i=0;i<65536;i+=16){
    size_t vl;
    __asm__ volatile("vsetivli %0,16,e16,m1,ta,ma\n\tvle16.v v1,(%1)\n\tvfwcvt.f.f.v v2,v1\n\tvsetivli zero,16,e32,m2,ta,ma\n\tvse32.v v2,(%2)"
      :"=r"(vl):"r"(a+i),"r"(b+i):"memory","v1","v2","v3");
  }
  int mis=0,nan=0;
  for(int i=0;i<65536;i++){ uint32_t r=sw(a[i]); if(((a[i]>>10)&0x1f)==31 && (a[i]&0x3ff)){nan++; continue;} if(r!=b[i]){ if(mis<5) printf("mis %04x sw=%08x hw=%08x\n",a[i],r,b[i]); mis++; } }
  printf("ZVFH_ASM mismatch=%d (nan skipped=%d)\n",mis,nan);
  return 0;
}
EOF
gcc -O2 -march=rv64gcv_zvfh -o /tmp/zvfh_a /tmp/zvfh_a.c; echo cc_rc=$?
taskset -c 5 /tmp/zvfh_a; echo run_rc=$?
echo "== p0 summary"
grep -E '^== |\[STREAM\]|PREFILL-TIMING|PREFILL-KERNELS|DEC avg' /tmp/p0prof.log
tail -1 /tmp/p0prof.log
