/* moe_down_layout_probe.c — 隔离"列带 vs 专家连续"访问模式差异（MoE down 取证探针）
 * 同 buffer、同字节数、只改遍历顺序（footprint 相同），排除容量因素。
 * 用法: ./probe [D] [E] [EF] [reps]   D=输出行(2048) E=专家(128) EF=每专家中间维(768)
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec + t.tv_nsec*1e-9; }

int main(int argc, char** argv){
    int D  = argc>1? atoi(argv[1]) : 2048;
    int E  = argc>2? atoi(argv[2]) : 128;
    int EF = argc>3? atoi(argv[3]) : 768;
    int reps = argc>4? atoi(argv[4]) : 3;
    long FFN = (long)E*EF;
    size_t N = (size_t)D*(size_t)FFN;
    uint8_t *buf = (uint8_t*)malloc(N);
    if(!buf){ printf("OOM need %.1f MB\n", N/1e6); return 2; }
    for(size_t i=0;i<N;i++) buf[i]=(uint8_t)(i*131u+7u);

    volatile int64_t sink=0;
    double bestA=1e18, bestB=1e18;

    /* Pattern A: 列带（现状）—— 每专家: for row: 768 连续, 然后跳 FFN */
    for(int rep=0; rep<reps; rep++){
        int64_t s=0; double t0=now();
        for(int e=0;e<E;e++){
            long base=(long)e*EF;
            for(int r=0;r<D;r++){
                const uint8_t *p = buf + (size_t)r*(size_t)FFN + base;
                for(int k=0;k<EF;k++) s += p[k];
            }
        }
        double t=now()-t0; if(t<bestA)bestA=t; sink+=s;
    }

    /* Pattern B: 专家连续 —— 每专家是一块 D*EF 连续内存 */
    for(int rep=0; rep<reps; rep++){
        int64_t s=0; double t0=now();
        for(int e=0;e<E;e++){
            const uint8_t *p = buf + (size_t)e*(size_t)D*(size_t)EF;
            for(int r=0;r<D;r++){
                const uint8_t *q = p + (size_t)r*(size_t)EF;
                for(int k=0;k<EF;k++) s += q[k];
            }
        }
        double t=now()-t0; if(t<bestB)bestB=t; sink+=s;
    }

    double MB = (double)N/1e6;
    printf("D=%d E=%d EF=%d footprint=%.1f MB reps=%d  (single-thread int8 proxy)\n",D,E,EF,MB,reps);
    printf("PatternA column-band : %.3fs  %.1f MB/s\n", bestA, MB/bestA);
    printf("PatternB expert-cont : %.3fs  %.1f MB/s  speedup(B vs A)=%.2fx\n", bestB, MB/bestB, bestA/bestB);
    printf("sink=%lld\n",(long long)sink);
    free(buf);
    return 0;
}
