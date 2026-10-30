/* moe_down_io_probe.c — eMMC 层面"列带分散 vs 顺序"读放大取证
 * 真实文件、同有用字节数、fadvise 强制不走页缓存，对比有效带宽。
 * 用法: ./ioprobe [D] [E] [EF] [reps]
 *   D=2048 行, E=128 专家, EF=768 每专家带宽字节(每行取 EF 字节)
 *   A(列带)   : 每专家, 每行 pread(EF) at r*FFN + e*EF      (分散, 每页只用到 EF/4096)
 *   B(专家连续): 每专家, 每行 pread(EF) at e*D*EF + r*EF    (顺序)
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <string.h>

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec + t.tv_nsec*1e-9; }

int main(int argc, char** argv){
    int D  = argc>1? atoi(argv[1]) : 2048;
    int E  = argc>2? atoi(argv[2]) : 128;
    int EF = argc>3? atoi(argv[3]) : 768;
    int reps = argc>4? atoi(argv[4]) : 2;
    long FFN = (long)E*EF;
    size_t FILESZ = (size_t)D*(size_t)FFN;   /* 约 201 MB */
    const char *path = "/home/haliu/kv/probe/iotest.bin";

    int fd = open(path, O_RDWR|O_CREAT|O_TRUNC, 0644);
    if(fd<0){ perror("open"); return 2; }
    /* 真实写入并分配（避免稀疏洞，否则读洞不产生 eMMC I/O） */
    {
        size_t blk=1<<20; char *w=malloc(blk);
        for(size_t i=0;i<blk;i++) w[i]=(char)(i*131+7);
        size_t done=0; double t0=now();
        while(done<FILESZ){ size_t n=(FILESZ-done<blk)?(FILESZ-done):blk; if(write(fd,w,n)!=(ssize_t)n){perror("write");return 2;} done+=n; }
        fsync(fd);
        printf("file created %.1f MB in %.2fs\n", FILESZ/1e6, now()-t0);
        free(w);
    }

    char *rb = malloc(EF);
    volatile int64_t sink=0;
    double bestA=1e18,bestB=1e18;

    /* A: 列带 —— 分散 pread */
    for(int rep=0; rep<reps; rep++){
        posix_fadvise(fd,0,0,POSIX_FADV_DONTNEED);   /* 强制后续读落盘 */
        int64_t s=0; double t0=now();
        for(int e=0;e<E;e++){
            long base=(long)e*EF;
            for(int r=0;r<D;r++){
                off_t off=(off_t)r*FFN + base;
                ssize_t n=pread(fd,rb,EF,off); if(n>0) s+=rb[0];
            }
        }
        double t=now()-t0; if(t<bestA)bestA=t; sink+=s;
    }
    /* B: 专家连续 —— 顺序 pread */
    for(int rep=0; rep<reps; rep++){
        posix_fadvise(fd,0,0,POSIX_FADV_DONTNEED);
        int64_t s=0; double t0=now();
        for(int e=0;e<E;e++){
            off_t base=(off_t)e*(off_t)D*EF;
            for(int r=0;r<D;r++){
                off_t off=base+(off_t)r*EF;
                ssize_t n=pread(fd,rb,EF,off); if(n>0) s+=rb[0];
            }
        }
        double t=now()-t0; if(t<bestB)bestB=t; sink+=s;
    }

    double MB=(double)FILESZ/1e6;
    printf("D=%d E=%d EF=%d useful=%.1f MB reps=%d (eMMC, fadvise DONTNEED)\n",D,E,EF,MB,reps);
    printf("A column-band(strided)   : %.3fs  %.1f MB/s useful\n", bestA, MB/bestA);
    printf("B expert-contig(seq)     : %.3fs  %.1f MB/s useful  speedup=%.2fx\n", bestB, MB/bestB, bestA/bestB);
    printf("sink=%lld\n",(long long)sink);
    free(rb); close(fd);
    return 0;
}
