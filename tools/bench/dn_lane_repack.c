/* dn_lane_repack.c — 从 model.vqf（只读）生成 q4_down 的 lane-tile 布局 sidecar（dn_lane.bin）
 * 纯排列：72B 4x4 块 → 576B lane 块，字节内容不变，引擎点积链序不变 ⇒ 位级一致（与 gu 同族）。
 * 源文件只以 O_RDONLY 打开，绝不写回。
 *
 * **前提（硬约束）**：源 model.vqf 的 q4_down 必须是「专家连续（EC）」布局，
 *   即 src(l,e) = down_off + l*layerB + e*expB 才是一个完整专家块 [d × ef]。
 *   行带（未 EC）布局在同样偏移处是跨专家混合块，会生成**静默错误**的 sidecar。
 *   本工具无法从字节上区分两种布局，故最终判据是引擎侧 TOKIDS/md5 对拍。
 *
 * 用法: ./dn_lane_repack <model.vqf> <dn_lane.bin> <0|1|2>
 *   0 = DRYRUN : 自检 + 第 0 层首/末专家正反往返比对，不写盘
 *   1 = WRITE  : 生成 sidecar（头最后写 complete=1）
 *   2 = VERIFY : 全量反向重排比对源 + 重算数据 FNV
 * sidecar: [0,4096) DNHdr；down 区 @4096；段偏移 l*layerB+e*expB（与源相同）
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include "../../include/model/vqf_format.h"

#define DN_MAGIC   0x4E4C4E44u   /* "DNLN"（小端字节序 D N L N） */
#define DN_VERSION 1u
#define DN_HDR     4096u

typedef struct {
    uint32_t magic, version, complete, rsvd0;
    uint32_t nl, ne, ef, d;
    uint32_t rows, nblk;          /* rows=d（down 输出行）· nblk=ef/32 */
    uint64_t expB, layerB, data_off;
    uint64_t src_size, src_down_off, src_file_len;
    uint64_t src_tail;            /* 源文件 file_len 处的 8B（FNV 尾） */
    uint64_t data_fnv;            /* sidecar [4096, end) 的 FNV-1a */
} DNHdr;

static uint64_t fnv_update(uint64_t h, const void *p, size_t n){
    const uint8_t *b=(const uint8_t*)p;
    for(size_t i=0;i<n;i++){ h^=b[i]; h*=0x100000001b3ull; }
    return h;
}
static uint64_t q4b(uint64_t n){ return (n+31)/32*18; }

/* 一个专家段：rows 行 × nblk 块。inv=0: 4x4→lane；inv=1: lane→4x4 */
static void seg_perm(const uint8_t *src, uint8_t *dst, int rows, int nblk, int inv){
    for(int c=0;c<rows/32;c++) for(int b=0;b<nblk;b++){
        size_t lo=((size_t)c*nblk+b)*576;
        for(int g=0;g<8;g++){
            size_t so=(((size_t)c*8+g)*nblk+b)*72;
            for(int m=0;m<4;m++){
                if(!inv) memcpy(dst+lo+m*16+g*2, src+so+2*m, 2);
                else     memcpy(dst+so+2*m, src+lo+m*16+g*2, 2);
                for(int k=0;k<4;k++){
                    if(!inv) memcpy(dst+lo+64+k*128+m*32+g*4, src+so+8+k*16+m*4, 4);
                    else     memcpy(dst+so+8+k*16+m*4, src+lo+64+k*128+m*32+g*4, 4);
                }
            }
        }
    }
}

static int cand_valid(uint64_t d_off, uint64_t f_len, uint64_t fsz, uint64_t tag, uint64_t dir_end){
    if(!d_off||!f_len) return 0;
    if(d_off&(VQF_ALIGN-1)) return 0;
    if(d_off<dir_end||d_off>=fsz) return 0;
    if(f_len+tag!=fsz) return 0;
    if(f_len<=d_off) return 0;
    return 1;
}

static int pread_all(int fd, void *p, size_t n, uint64_t off){
    uint8_t *b=p; while(n){ ssize_t r=pread(fd,b,n,(off_t)off); if(r<=0) return -1; b+=r; n-=r; off+=r; } return 0;
}
static int pwrite_all(int fd, const void *p, size_t n, uint64_t off){
    const uint8_t *b=p; while(n){ ssize_t r=pwrite(fd,b,n,(off_t)off); if(r<=0) return -1; b+=r; n-=r; off+=r; } return 0;
}
static int selftest(void){
    const int rows=64, nblk=3;
    size_t n=(size_t)(rows/4)*nblk*72;
    uint8_t *a=malloc(n), *b=malloc(n), *c=malloc(n);
    if(!a||!b||!c) return 0;
    uint64_t sa=0, sb=0;
    for(size_t i=0;i<n;i++){ a[i]=(uint8_t)(i*37+11); sa+=a[i]; }
    memset(b,0,n); memset(c,0,n);
    seg_perm(a,b,rows,nblk,0); seg_perm(b,c,rows,nblk,1);
    for(size_t i=0;i<n;i++) sb+=b[i];
    int rt=(memcmp(a,c,n)==0), se=(sa==sb);
    printf("[SELFTEST] roundtrip=%d sum_eq=%d\n", rt, se);
    free(a); free(b); free(c);
    return rt&&se;
}

int main(int argc, char **argv){
    if(argc<4){ fprintf(stderr,"usage: %s <model.vqf> <dn_lane.bin> <0|1|2>\n",argv[0]); return 1; }
    int mode=atoi(argv[3]);
    if(mode<0||mode>2){ fprintf(stderr,"bad mode\n"); return 1; }
    if(!selftest()){ fprintf(stderr,"[SELFTEST] FAIL\n"); return 4; }

    int fd=open(argv[1],O_RDONLY);
    if(fd<0){ perror("open src"); return 2; }
    struct stat st; if(fstat(fd,&st)){ perror("fstat"); return 2; }
    uint64_t fsz=(uint64_t)st.st_size;
    uint8_t hb[sizeof(VQFHeader)];
    if(fsz<sizeof(hb)||pread_all(fd,hb,sizeof(hb),0)){ fprintf(stderr,"read hdr fail\n"); return 2; }
    VQFHeader H; memcpy(&H,hb,sizeof(H));
    if(H.magic!=VQF_MAGIC){ fprintf(stderr,"bad magic\n"); return 3; }
    if(H.flags&VQF_FLAG_ENC){ fprintf(stderr,"encrypted VQF not supported\n"); return 3; }
    if(H.version!=VQF_VERSION_MOE && H.version!=VQF_VERSION_LANE){ fprintf(stderr,"version %u != MOE(%u)/LANE(%u)\n",H.version,VQF_VERSION_MOE,VQF_VERSION_LANE); return 3; }
    if(!(H.flags&VQF_FLAG_MOE)||!(H.flags&VQF_FLAG_Q4_4X4)){ fprintf(stderr,"need MOE|Q4_4X4 flags (0x%x)\n",H.flags); return 3; }
    uint32_t nt=H.n_tensors;
    if(nt<1||nt>64){ fprintf(stderr,"n_tensors %u out of range\n",nt); return 3; }

    uint64_t tag=8, dir_off=(sizeof(VQFHeader)+63)&~(uint64_t)63, dir_end=dir_off+(uint64_t)nt*sizeof(VQFTensor);
    uint64_t nd,nf,ld,lf;
    memcpy(&nd,hb+offsetof(VQFHeader,data_offset),8); memcpy(&nf,hb+offsetof(VQFHeader,file_len),8);
    memcpy(&ld,hb+offsetof(VQFHeaderLegacy,data_offset),8); memcpy(&lf,hb+offsetof(VQFHeaderLegacy,file_len),8);
    int nv=cand_valid(nd,nf,fsz,tag,dir_end), lv=cand_valid(ld,lf,fsz,tag,dir_end);
    if(!(nv&&!lv)){ fprintf(stderr,"header ambiguous/legacy (new=%d legacy=%d), refuse\n",nv,lv); return 3; }
    const VQFArch *A=&H.arch;
    if(A->dim<=0||A->n_layers<=0||A->n_heads<=0||A->head_dim<=0||A->ffn_dim<=0||A->vocab_size<=0){
        fprintf(stderr,"[VQF] arch geometry invalid\n"); return 3; }
    if(H.flags&VQF_FLAG_SIGNED) printf("[WARN] source is SM2-signed; sidecar is NOT covered by signature/FNV\n");

    uint64_t nl=(uint64_t)A->n_layers, ne=(uint64_t)A->n_experts, ef=(uint64_t)A->moe_ffn, d=(uint64_t)A->dim;
    uint64_t ff=(uint64_t)A->ffn_dim, file_len=nf;
    printf("[GEOM] nl=%llu ne=%llu ef=%llu d=%llu ff=%llu (expect 48/128/768/2048/98304)\n",
        (unsigned long long)nl,(unsigned long long)ne,(unsigned long long)ef,(unsigned long long)d,(unsigned long long)ff);
    if(ne*ef!=ff||ef%32||d%32){ fprintf(stderr,"geometry mismatch\n"); return 3; }
    uint64_t expB=q4b(ef*d), layerB=q4b(ff*d);
    if((expB&4095)||layerB!=expB*ne){ fprintf(stderr,"expB=%llu not page aligned / layerB mismatch\n",(unsigned long long)expB); return 3; }

    VQFTensor *T=malloc((size_t)nt*sizeof(VQFTensor));
    if(!T||pread_all(fd,T,(size_t)nt*sizeof(VQFTensor),dir_off)){ fprintf(stderr,"read dir fail\n"); return 2; }
    int f=-1; const char *nm="q4_down";
    for(uint32_t j=0;j<nt;j++) if(strncmp(T[j].name,nm,sizeof(T[j].name))==0){ f=(int)j; break; }
    if(f<0){ fprintf(stderr,"%s not found\n",nm); return 3; }
    /* v3 目录 layout 字段不可信（转换器 MoE 时未写，实测 0）；数据布局由 flag Q4_4X4 保证。 */
    if(T[f].qtype!=VQF_QT_Q4_0||(uint64_t)T[f].rows!=nl*d||T[f].bytes!=layerB*nl
       ||T[f].offset<nd||T[f].offset+T[f].bytes>file_len){
        fprintf(stderr,"%s check fail qt=%u lay=%u rows=%u bytes=%llu off=%llu\n",nm,T[f].qtype,T[f].layout,
            T[f].rows,(unsigned long long)T[f].bytes,(unsigned long long)T[f].offset); return 3; }
    uint64_t D_off=T[f].offset;
    printf("[DIR] %s off=%llu bytes=%llu\n",nm,(unsigned long long)D_off,(unsigned long long)T[f].bytes);

    uint64_t src_tail=0;
    if(pread_all(fd,&src_tail,8,file_len)){ fprintf(stderr,"read tail fail\n"); return 2; }
    printf("[SRC] size=%llu file_len=%llu tail=%016llx expB=%llu layerB=%llu\n",(unsigned long long)fsz,
        (unsigned long long)file_len,(unsigned long long)src_tail,(unsigned long long)expB,(unsigned long long)layerB);
    printf("[PRECOND] 源 q4_down 必须为 EC 专家连续布局（VLLM_DN_EC=1 生效的前提）；\n"
           "          本工具按 src(l,e)=down_off+l*layerB+e*expB 读取；最终以引擎 TOKIDS/md5 对拍为准。\n");

    uint8_t *sb=malloc(expB), *db=malloc(expB), *cb=malloc(expB);
    if(!sb||!db||!cb){ fprintf(stderr,"oom\n"); return 2; }
    int nblk=(int)(ef/32), rows=(int)d;
    printf("[TILE] rows=%d nblk=%d tiles/expert=%d tileB=%d expertB=%llu\n",
        rows,nblk,rows/32,nblk*576,(unsigned long long)expB);
    int rc=0;
    const uint64_t regB=layerB*nl, tot=DN_HDR+regB;
    const uint64_t FNV0=0xcbf29ce484222325ull;

    if(mode==0){
        for(int k=0;k<2&&!rc;k++){
            uint64_t e=k?ne-1:0, so=D_off+e*expB;
            if(pread_all(fd,sb,expB,so)){ fprintf(stderr,"pread fail\n"); rc=2; break; }
            seg_perm(sb,db,rows,nblk,0); seg_perm(db,cb,rows,nblk,1);
            int ok=memcmp(sb,cb,expB)==0;
            printf("[DRYRUN] %s l=0 e=%llu roundtrip=%d\n",nm,(unsigned long long)e,ok);
            if(!ok) rc=5;
        }
        printf("[DRYRUN] sidecar would be %llu bytes (%.2f GiB)\n",
            (unsigned long long)tot,(double)tot/1073741824.0);
    } else if(mode==1){
        int o=open(argv[2],O_WRONLY|O_CREAT|O_EXCL,0644);
        if(o<0){ perror("open sidecar (O_EXCL)"); rc=2; goto out; }
        uint8_t *zh=calloc(1,DN_HDR);
        if(!zh||pwrite_all(o,zh,DN_HDR,0)){ fprintf(stderr,"write hdr0 fail\n"); free(zh); close(o); rc=2; goto out; }
        free(zh);
        uint64_t h=FNV0;
        for(uint64_t l=0;l<nl&&!rc;l++){
            for(uint64_t e=0;e<ne;e++){
                uint64_t rel=l*layerB+e*expB;
                if(pread_all(fd,sb,expB,D_off+rel)){ fprintf(stderr,"pread fail\n"); rc=2; break; }
                seg_perm(sb,db,rows,nblk,0);
                if(pwrite_all(o,db,expB,DN_HDR+rel)){ fprintf(stderr,"pwrite fail\n"); rc=2; break; }
                h=fnv_update(h,db,expB);
            }
            if(!rc){
                fdatasync(o);
                posix_fadvise(o,0,0,POSIX_FADV_DONTNEED);
                posix_fadvise(fd,D_off+l*layerB,layerB,POSIX_FADV_DONTNEED);
                printf("[WRITE] %s layer %llu/%llu\n",nm,(unsigned long long)l+1,(unsigned long long)nl); fflush(stdout);
            }
        }
        if(!rc){
            DNHdr G; memset(&G,0,sizeof(G));
            G.magic=DN_MAGIC; G.version=DN_VERSION; G.complete=1;
            G.nl=(uint32_t)nl; G.ne=(uint32_t)ne; G.ef=(uint32_t)ef; G.d=(uint32_t)d;
            G.rows=(uint32_t)rows; G.nblk=(uint32_t)nblk;
            G.expB=expB; G.layerB=layerB; G.data_off=DN_HDR;
            G.src_size=fsz; G.src_down_off=D_off; G.src_file_len=file_len;
            G.src_tail=src_tail; G.data_fnv=h;
            if(fsync(o)||pwrite_all(o,&G,sizeof(G),0)||fsync(o)){ fprintf(stderr,"final hdr fail\n"); rc=2; }
            else printf("[WRITE] done bytes=%llu data_fnv=%016llx\n",(unsigned long long)tot,(unsigned long long)h);
        }
        close(o);
    } else {
        int o=open(argv[2],O_RDONLY);
        if(o<0){ perror("open sidecar"); rc=2; goto out; }
        struct stat s2; DNHdr G;
        if(fstat(o,&s2)||pread_all(o,&G,sizeof(G),0)){ fprintf(stderr,"read sidecar hdr fail\n"); close(o); rc=2; goto out; }
        if(G.magic!=DN_MAGIC||G.version!=DN_VERSION||G.complete!=1||G.nl!=nl||G.ne!=ne||G.ef!=ef||G.d!=d
           ||G.rows!=(uint32_t)rows||G.nblk!=(uint32_t)nblk
           ||G.expB!=expB||G.layerB!=layerB||G.data_off!=DN_HDR
           ||G.src_size!=fsz||G.src_tail!=src_tail||G.src_down_off!=D_off
           ||G.src_file_len!=file_len||(uint64_t)s2.st_size!=tot){
            fprintf(stderr,"[VERIFY] header mismatch (complete=%u size=%llu)\n",G.complete,(unsigned long long)s2.st_size);
            close(o); rc=6; goto out; }
        uint64_t h=FNV0, bad=0;
        for(uint64_t l=0;l<nl&&!rc;l++){
            for(uint64_t e=0;e<ne;e++){
                uint64_t rel=l*layerB+e*expB;
                if(pread_all(o,db,expB,DN_HDR+rel)||pread_all(fd,sb,expB,D_off+rel)){
                    fprintf(stderr,"pread fail\n"); rc=2; break; }
                h=fnv_update(h,db,expB);
                seg_perm(db,cb,rows,nblk,1);
                if(memcmp(sb,cb,expB)) bad++;
            }
            posix_fadvise(o,DN_HDR+l*layerB,layerB,POSIX_FADV_DONTNEED);
            posix_fadvise(fd,D_off+l*layerB,layerB,POSIX_FADV_DONTNEED);
            printf("[VERIFY] %s layer %llu/%llu bad=%llu\n",nm,(unsigned long long)l+1,(unsigned long long)nl,
                (unsigned long long)bad); fflush(stdout);
        }
        if(!rc){
            int fok=(h==G.data_fnv);
            printf("[VERIFY] segs_bad=%llu fnv=%016llx expect=%016llx fnv_ok=%d\n",(unsigned long long)bad,
                (unsigned long long)h,(unsigned long long)G.data_fnv,fok);
            if(bad||!fok) rc=7; else printf("[VERIFY] PASS\n");
        }
        close(o);
    }
out:
    free(sb); free(db); free(cb); free(T); close(fd);
    return rc;
}
