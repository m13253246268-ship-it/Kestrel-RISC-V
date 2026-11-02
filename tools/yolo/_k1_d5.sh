#!/bin/bash
# D5：decode 带宽分析 —— (1) 板端读带宽上限（RVV 流式读，1/2/4/8 线程）
# (2) 本引擎 decode 线程数扫描 t4/t6/t8 + 4B 分桶 (3) 官方 llama.cpp tg 同口径 t4/t8。
# 后台跑；再次运行即轮询。日志 /tmp/d5.log
L=/tmp/d5.log
if pgrep -f 'd5_job' >/dev/null; then echo RUNNING; tail -6 $L; exit 0; fi
if grep -q D5_END $L 2>/dev/null; then cat $L; exit 0; fi
cat > /tmp/d5_bw.c <<'EOF'
#include <riscv_vector.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
int main(int argc,char**argv){
  size_t N=(size_t)(argc>1?atol(argv[1]):512)<<20;           /* MB */
  uint8_t *a=aligned_alloc(4096,N); memset(a,1,N);
  int ths[]={1,2,4,8};
  for(int ti=0;ti<4;ti++){
    int nt=ths[ti]; double best=1e9; uint64_t sink=0;
    for(int r=0;r<5;r++){
      double t0=now();
      #pragma omp parallel num_threads(nt) reduction(+:sink)
      { int id=omp_get_thread_num(); size_t ch=N/nt, s=id*ch, e=s+ch;
        size_t vl=__riscv_vsetvlmax_e64m8();
        vuint64m8_t acc=__riscv_vmv_v_x_u64m8(0,vl);
        for(size_t i=s;i+vl*8<=e;i+=vl*8)
          acc=__riscv_vadd_vv_u64m8(acc,__riscv_vle64_v_u64m8((const uint64_t*)(a+i),vl),vl);
        vuint64m1_t z=__riscv_vmv_v_x_u64m1(0,1);
        sink+=__riscv_vmv_x_s_u64m1_u64(__riscv_vredsum_vs_u64m8_u64m1(acc,z,vl)); }
      double dt=now()-t0; if(dt<best)best=dt;
    }
    printf("BW read t%d  %.2f GB/s (best of 5, %zu MB) sink=%llu\n",nt,N/best/1e9,N>>20,(unsigned long long)(sink&1));
  }
  return 0;
}
EOF
cat > /tmp/d5_job.sh <<'EOF'
set -u
cd /root/kv/kestrel_vllm_riscv
B=build-riscv/vllm_kestrel; R=/root/kv/models; G=$R/gguf
sha256sum $B
COLB="VLLM_Q4_RVV=3 VLLM_Q4_INT=1 VLLM_Q4_INTB=1 VLLM_Q4_DEC=1"
echo "== (1) 读带宽 =="
gcc -O2 -march=rv64gcv -fopenmp /tmp/d5_bw.c -o /tmp/d5_bw && for r in 1 2; do
  echo "-- run $r load=$(cut -d' ' -f1 /proc/loadavg)"; /tmp/d5_bw 512; done
echo "== (2) 本引擎 decode 线程扫描（ctx32 n32，交替 3 轮）=="
for M in Qwen3-0.6B Qwen3-1.7B Qwen3-4B; do
  for r in 1 2 3; do for T in 4 6 8; do
    O=/tmp/d5_perf.log
    env $COLB VLLM_THREADS=$T OMP_NUM_THREADS=$T timeout 1800 \
      $B --stream-test --model $R/$M-col --stream-ctx 32 --stream-n 32 --threads $T > $O 2>&1
    printf '%s r%s t%s rc=%s load=%s | %s | txt=%s\n' $M $r $T $? "$(cut -d' ' -f1 /proc/loadavg)" \
      "$(grep -a '\[STREAM\] decoded' $O | tail -1)" "$(grep -a '\[STREAM\] text:' $O | md5sum | cut -c1-8)"
  done; done
done
for T in 4 8; do
  echo "== 4B 分桶 t$T =="
  env $COLB VLLM_DEC_PROF=1 VLLM_THREADS=$T OMP_NUM_THREADS=$T timeout 1800 \
    $B --stream-test --model $R/Qwen3-4B-col --stream-ctx 32 --stream-n 16 --threads $T > /tmp/d5_prof.log 2>&1
  grep -a '\[DEC-SINGLE\]' /tmp/d5_prof.log | tail -n +2 | awk '{
    for(i=1;i<=NF;i++){ if(split($i,kv,"=")==2 && kv[1]!="seq"){ s[kv[1]]+=kv[2]; if(!(kv[1] in o)){o[kv[1]]=++n; k[n]=kv[1]} } }
    c++ } END{ for(j=1;j<=n;j++) printf "%s=%.2f ", k[j], s[k[j]]/c; printf "steps=%d\n", c }'
done
echo "== (3) 官方 llama.cpp tg32（-r 3）=="
for M in Qwen3-0.6B Qwen3-1.7B Qwen3-4B; do for T in 4 8; do
  printf 'LLAMACPP %-11s t%s load=%s | ' $M $T "$(cut -d' ' -f1 /proc/loadavg)"
  env OMP_NUM_THREADS=$T timeout 1800 llama-bench -m $G/$M-Q4_0.gguf -p 0 -n 32 -t $T -r 3 -o md 2>&1 \
    | grep -E '\| +tg32 +\|' | awk -F'|' '{gsub(/ /,"",$8); printf "tg32=%s t/s", $8}'
  echo
done; done
uptime
echo D5_END
EOF
nohup bash /tmp/d5_job.sh > $L 2>&1 &
echo STARTED
