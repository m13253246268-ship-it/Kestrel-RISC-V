/* vqf_down_relayout.c — 把 VQF 的 q4_down 张量在"行组-major列带"与"专家连续"之间就地重排
 * 纯排列：只改 72B tile 的存放位置，不改任何字节内容，文件大小不变 ⇒ 引擎点积序不变/位级一致。
 * 用法: ./vqf_down_relayout <model.vqf> <mode>
 *   0 = DRYRUN : 只对第0层做排列+字节和校验，不写盘（自检）
 *   1 = WRITE  : 行带 → 专家连续（48层写回 + 重算尾部 FNV）
 *   2 = REVERT : 专家连续 → 行带（回滚用，48层写回 + 重算 FNV）
 * 本模型参数(Qwen3-30B-A3B)：nl=48 ne=128 d=2048 ef=768
 *   S=(ef>>5)*72=1728  E=S*(d>>2)=884736  per-layer=E*ne=113246208
 *   行带: src[rg*(ne*S)+e*S]   专家连续: dst[e*E+rg*S]   （每次搬 S=1728B）
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

#define DIR_OFF 448
#define TENSOR_SZ 64

static uint64_t fnv_update(uint64_t h, const void *p, size_t n){
    const uint8_t *b=(const uint8_t*)p;
    for(size_t i=0;i<n;i++){ h^=b[i]; h*=0x100000001b3ull; }
    return h;
}
/* inverse=0: 行带->专家连续；inverse=1: 专家连续->行带 */
static void permute_layer(const uint8_t *src, uint8_t *dst, int ne, int d, int ef, int inverse){
    int S=(ef>>5)*72, E=S*(d>>2), rg_n=d>>2;
    for(int rg=0; rg<rg_n; rg++){
        for(int e=0; e<ne; e++){
            if(!inverse) memcpy(dst + (size_t)e*E + (size_t)rg*S, src + (size_t)rg*(size_t)(ne*S) + (size_t)e*S, (size_t)S);
            else         memcpy(dst + (size_t)rg*(size_t)(ne*S) + (size_t)e*S, src + (size_t)e*E + (size_t)rg*S, (size_t)S);
        }
    }
}

int main(int argc, char** argv){
    const char *path = argc>1?argv[1]:"/home/haliu/kv/moe30b/model.vqf";
    char mode = (argc>2)? argv[2][0] : '0';
    int do_write = (mode=='1'||mode=='2');
    int inverse  = (mode=='2');
    int nl=48, ne=128, d=2048, ef=768;

    /* 0) 小型双射往返自检 */
    {
        int td=8, tne=2, tef=1024;
        int tS=(tef>>5)*72, tE=tS*(td>>2), tper=tE*tne;
        uint8_t *a=malloc(tper),*b=malloc(tper),*c=malloc(tper);
        for(int i=0;i<tper;i++) a[i]=(uint8_t)(i*37+11);
        permute_layer(a,b,tne,td,tef,0);
        permute_layer(b,c,tne,td,tef,1);
        int ok=1; for(int i=0;i<tper;i++) if(a[i]!=c[i]){ ok=0; break; }
        uint64_t sa=0,sb=0; for(int i=0;i<tper;i++){ sa+=a[i]; sb+=b[i]; }
        printf("[SELFTEST] roundtrip=%s sum_eq=%s\n", ok?"PASS":"FAIL", (sa==sb)?"PASS":"FAIL");
        free(a);free(b);free(c);
        if(!ok){ printf("selftest failed, abort\n"); return 4; }
    }

    int fd=open(path,O_RDWR); if(fd<0){perror("open");return 2;}
    struct stat st; fstat(fd,&st);
    uint8_t hdr[DIR_OFF];
    if(pread(fd,hdr,sizeof hdr,0)!=(ssize_t)sizeof hdr){perror("hdr");return 2;}
    uint32_t magic=*(uint32_t*)(hdr+0), ver=*(uint32_t*)(hdr+4), flags=*(uint32_t*)(hdr+8), nt=*(uint32_t*)(hdr+12);
    uint64_t data_off=*(uint64_t*)(hdr+200), file_len=*(uint64_t*)(hdr+208);
    printf("magic=%08x ver=%u flags=0x%x nt=%u data_off=%llu file_len=%llu size=%lld\n",
        magic,ver,flags,nt,(unsigned long long)data_off,(unsigned long long)file_len,(long long)st.st_size);
    if((flags & 0x80u)){ printf("encrypted VQF, refuse\n"); return 5; }
    if ((ver != 3 && ver != 5) || !(flags & 0x200u)) {   /* VQF_VERSION_MOE=3 / VQF_VERSION_LANE=5, VQF_FLAG_MOE=1<<9 */
        printf("not a MoE VQF (ver=%u flags=0x%x), refuse\n", ver, flags);
        return 5;
    }
    if(nt==0 || nt>64){ printf("bad nt\n"); return 5; }

    uint8_t dir[64*TENSOR_SZ];
    if(pread(fd,dir,(size_t)nt*TENSOR_SZ,DIR_OFF)!=(ssize_t)((size_t)nt*TENSOR_SZ)){perror("dir");return 2;}
    long doff=-1; uint64_t dbytes=0; uint32_t drows=0,dcols=0;
    for(uint32_t i=0;i<nt;i++){
        uint8_t*e=dir+(size_t)i*TENSOR_SZ; char nm[33]; memcpy(nm,e,32); nm[32]=0;
        uint32_t rows=*(uint32_t*)(e+36),cols=*(uint32_t*)(e+40);
        uint64_t off=*(uint64_t*)(e+48),by=*(uint64_t*)(e+56);
        if(!strcmp(nm,"q4_down")){ doff=(long)off; dbytes=by; drows=rows; dcols=cols; }
    }
    if(doff<0){ printf("q4_down not found\n"); return 3; }
    printf("q4_down off=%ld bytes=%llu rows=%u cols=%u\n",doff,(unsigned long long)dbytes,drows,dcols);

    int S=(ef>>5)*72, E=S*(d>>2);
    if((uint32_t)(nl*d)!=drows || (uint32_t)(ne*ef)!=dcols
       || (uint64_t)nl*(uint64_t)ne*(uint64_t)E!=dbytes){
        printf("PARAM MISMATCH: expect rows=%d cols=%d bytes=%llu got rows=%u cols=%u bytes=%llu\n",
            nl*d, ne*ef, (unsigned long long)((uint64_t)nl*ne*E), drows,dcols,(unsigned long long)dbytes);
        return 6;
    }
    size_t per=(size_t)dbytes/(size_t)nl;
    printf("params OK: S=%d E=%d per-layer=%zu, mode=%c inverse=%d\n",S,E,per,mode,inverse);

    uint8_t *src=malloc(per),*dst=malloc(per);
    if(!src||!dst){ printf("oom\n"); return 7; }

    int nloop = do_write? nl : 1;
    for(int L=0; L<nloop; L++){
        off_t lof=(off_t)doff + (off_t)L*(off_t)per;
        if(pread(fd,src,per,lof)!=(ssize_t)per){ fprintf(stderr,"pread L%d fail\n",L); return 8; }
        permute_layer(src,dst,ne,d,ef,inverse);
        uint64_t ss=0,ds=0; for(size_t i=0;i<per;i++){ ss+=src[i]; ds+=dst[i]; }
        if(ss!=ds){ printf("L%d sum mismatch\n",L); return 9; }
        if(do_write){ if(pwrite(fd,dst,per,lof)!=(ssize_t)per){ fprintf(stderr,"pwrite L%d fail\n",L); return 8; } }
        if((L%8)==0) printf("  layer %d/%d done\n",L,nloop);
    }
    printf("permute %s finished\n", do_write?(inverse?"REVERT":"WRITE"):"DRYRUN");

    if(do_write){
        fsync(fd);
        uint64_t f=0xcbf29ce484222325ull;
        f=fnv_update(f,dir,(size_t)nt*TENSOR_SZ);
        size_t blk=1<<20; uint8_t *buf=malloc(blk);
        uint64_t off=data_off;
        while(off<file_len){ size_t n=(file_len-off<blk)?(size_t)(file_len-off):blk;
            if(pread(fd,buf,n,(off_t)off)!=(ssize_t)n){ fprintf(stderr,"fnv pread fail\n"); return 10; }
            f=fnv_update(f,buf,n); off+=n; }
        free(buf);
        if(pwrite(fd,&f,8,(off_t)file_len)!=8){ perror("pwrite fnv"); return 10; }
        fsync(fd);
        printf("FNV-1a=%016llx written @%llu\n",(unsigned long long)f,(unsigned long long)file_len);
    }
    free(src); free(dst); close(fd);
    return 0;
}
