/* Audio8 固定结构实验入口：独立于 LLM/YOLO 加载器，C11 + 标准数学库。
 * 支持独立 PCM16 WAV 转写，以及冻结 Mel/hidden 回归输入。
 * 清单 ASRLM001/002 绑定 v6 目录+数据 FNV；002 附加固定音频图身份与权重。
 * 仓库根编译：gcc -std=c11 -O2 -ffp-contract=off -D_GNU_SOURCE
 * -Iinclude/core -Iinclude/common src/model/vllm_asr.c src/core/vllm_tp.c -pthread -lm
 */
#ifndef _WIN32
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#endif
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#endif

#define D 512
#define F 1408
#define VOC 151936
#define CAP 512
#define REFS 103
static float *w[REFS];
static float keys[8][CAP][D], vals[8][CAP][D];
static void fail(const char *s) { fprintf(stderr, "ASR: %s\n", s); exit(1); }
static void read_exact(FILE *f, void *p, size_t n) { if (fread(p,1,n,f)!=n) fail("truncated file"); }
static uint32_t u32(const unsigned char *p) { return (uint32_t)p[0]|((uint32_t)p[1]<<8)|((uint32_t)p[2]<<16)|((uint32_t)p[3]<<24); }
static uint64_t u64(const unsigned char *p) { return u32(p)|((uint64_t)u32(p+4)<<32); }
static uint32_t word(FILE *f) { unsigned char b[4]; read_exact(f,b,4); return u32(b); }
static void seek_to(FILE *f, uint64_t off) { if(off>LONG_MAX || fseek(f,(long)off,SEEK_SET)) fail("file offset unsupported"); }
static void *alloc(size_t n) { void *p=malloc(n); if(!p) fail("allocation failed"); return p; }
static FILE *open_file(const char *p,const char *mode) { FILE *f=fopen(p,mode); if(!f) fail("cannot open file"); return f; }
static uint64_t hash(uint64_t h,const unsigned char *p,size_t n) { while(n--) h=(h^*p++)*UINT64_C(1099511628211); return h; }
static void finite_array(const float *p,size_t n) { for(size_t i=0;i<n;i++) if(!isfinite(p[i])) fail("nonfinite tensor"); }
/* 所有桶互斥；encoder 汇总只用于展示，不再次加入 total。 */
enum { P_OTHER,P_LOAD,P_ASSETS,P_FEATURES,P_ENCODER,P_CONV0,P_CONV1,P_CONV2,P_CONVOUT,P_QKV,P_ATTN,P_OUTPROJ,P_FFN,P_TAIL,P_PROJECT,P_PREFILL,P_DECODE,P_TRACE,P_TEXT,P_COUNT };
static int prof_enabled,prof_active;
static double prof_times[P_COUNT],prof_last,prof_start;
static unsigned prof_steps,prof_tokens,prof_selected;
static double prof_now(void) {
    if(!prof_enabled) return 0;
#ifdef _WIN32
    LARGE_INTEGER value,freq;
    if(!QueryPerformanceFrequency(&freq) || !QueryPerformanceCounter(&value)) fail("profile clock failed");
    return (double)value.QuadPart/(double)freq.QuadPart;
#else
    struct timespec ts;
    if(clock_gettime(CLOCK_MONOTONIC,&ts)) fail("profile clock failed");
    return (double)ts.tv_sec+(double)ts.tv_nsec*1e-9;
#endif
}
static int prof_set(int next) {
    int previous=prof_active;
    if(prof_enabled) {double now=prof_now();prof_times[previous]+=now-prof_last;prof_last=now;prof_active=next;}
    return previous;
}
static void prof_report(void) {
    if(!prof_enabled) return;
    prof_set(P_OTHER);
    const char *names[P_COUNT]={"unclassified","load","assets","features","encoder_other","conv0","conv1","conv2","conv_out","norm_qkv","attention","outprojection","ffn","tail_mlp","projector","prefill","decode","trace","text"};
    double encoder=0;
    for(int i=P_ENCODER;i<=P_TAIL;i++) encoder+=prof_times[i];
    for(int i=0;i<P_COUNT;i++) fprintf(stderr,"ASR_PROFILE stage=%s seconds=%.9f\n",names[i],prof_times[i]);
    fprintf(stderr,"ASR_PROFILE stage=encoder seconds=%.9f aggregate=1\n",encoder);
    fprintf(stderr,"ASR_PROFILE stage=total seconds=%.9f decode_steps=%u generated_tokens=%u selected_tokens=%u trace_includes_finite=1\n",prof_last-prof_start,prof_steps,prof_tokens,prof_selected);
}
static void prof_init(void) {
    const char *value=getenv("VLLM_ASR_PROFILE");prof_enabled=value && !strcmp(value,"1");
    if(prof_enabled) {prof_start=prof_last=prof_now();if(atexit(prof_report)) fail("profile atexit failed");}
}
static void dump(FILE *f,const float *p,size_t n) { int previous=prof_set(P_TRACE);finite_array(p,n); if(fwrite(p,4,n,f)!=n) fail("output write failed");prof_set(previous); }

#include "../../include/core/vllm_tp.h"
static int asr_threads;
static void asr_threads_init(void) {
    if(asr_threads) return;
    const char *value=getenv("VLLM_THREADS");
    int requested=1;
    if(value && *value) {
        char *end;long n=strtol(value,&end,10);
        if(*end || n<1 || n>64) fail("VLLM_THREADS must be 1..64");
        requested=(int)n;
    }
    asr_threads=vllm_tp_init(requested);
    fprintf(stderr,"ASR_THREADS requested=%d actual=%d\n",requested,asr_threads);
    if(atexit(vllm_tp_shutdown)) fail("thread atexit failed");
}
typedef struct {
    float *y;
    const float *x,*weight,*bias;
    int in,out;
} asr_matvec;
/* Require at least 32768 multiply-adds per participant and one output block.
 * Only independent output chains are dispatched; scratch stays with the caller. */
static int asr_parallel(int in,int out,int block) {
    asr_threads_init();
    return asr_threads>1 && vllm_tp_worker_id()<0 &&
        out/block>=asr_threads && (size_t)in*out>=(size_t)asr_threads*32768;
}
typedef struct { float *handle; int8_t *q; float *scale; uint32_t rows,cols; int transposed; } asr_qweight;
static asr_qweight qw[512];
static int nq;
static asr_qweight *qfind(const float *p) {
    for(int i=0;i<nq;i++) if(qw[i].handle==p) return &qw[i];
    return NULL;
}
static float qvalue(const asr_qweight *q,size_t index) {
    if(q->transposed) index=(index%q->rows)*q->cols+index/q->rows;
    return (float)q->q[index]*q->scale[index/q->cols];
}
static float *weight_load(FILE *f,const unsigned char *e,int allow_q,int transpose) {
    uint32_t r=u32(e+36),c=u32(e+40),type=u32(e+32);size_t count=(size_t)r*c;
    seek_to(f,u64(e+48));
    if(type==0x100) {
        if(!allow_q || nq>=512) fail("unexpected quantized tensor");
        asr_qweight *q=&qw[nq++];q->rows=r;q->cols=c;q->transposed=transpose;
        q->handle=alloc(4);q->q=alloc(count);q->scale=alloc((size_t)r*4);
        read_exact(f,q->q,count);read_exact(f,q->scale,(size_t)r*4);
        for(uint32_t i=0;i<r;i++) if(!isfinite(q->scale[i]) || q->scale[i]<=0) fail("invalid W8 scale");
        return q->handle;
    }
    if(type!=1) fail("expected floating weight");
    float *p=alloc(count*4);read_exact(f,p,count*4);finite_array(p,count);return p;
}
static void qlinear_row(void *ctx,int j) {
    asr_matvec *a=ctx;asr_qweight *q=qfind(a->weight);double sum=0;
    for(int k=0;k<a->in;k++) sum+=(double)a->x[k]*qvalue(q,(size_t)j*a->in+k);
    a->y[j]=(float)sum;if(a->bias) a->y[j]+=a->bias[j];
}
static void qaline_row(void *ctx,int j) {
    asr_matvec *a=ctx;asr_qweight *q=qfind(a->weight);float sum=0;
    for(int k=0;k<a->in;k++) sum+=a->x[k]*qvalue(q,(size_t)j*a->in+k);
    a->y[j]=sum+(a->bias?a->bias[j]:0.0f);
}
#if defined(__riscv_vector) && defined(VLLM_ASR_W8_RVV) && !defined(VLLM_ASR_SCALAR)
#include "vllm_asr_w8_rvv.h"
#endif
static void quant_linear(float *y,const float *x,const float *weight,int in,int out,const float *bias,int wide) {
    asr_matvec a={y,x,weight,bias,in,out};
#if defined(__riscv_vector) && defined(VLLM_ASR_W8_RVV) && !defined(VLLM_ASR_SCALAR)
    asr_qweight *q=qfind(weight);
    int supported=q->transposed?(in==7680 && out==896):
        ((in==9 && out==480) || (in==4320 && out==480) ||
         (in==896 && (out==896 || out==3584 || out==1024)) ||
         (in==3584 && out==896) || (in==1024 && out==4096) ||
         (in==4096 && out==1024));
    if(!wide && supported) {
        int blocks=(out+W8_LANES-1)/W8_LANES;
        if(asr_parallel(in,out,W8_LANES)) vllm_tp_parfor(0,blocks,w8_encoder_block,&a);
        else for(int j=0;j<blocks;j++) w8_encoder_block(&a,j);
        return;
    }
#endif
    void (*fn)(void *,int)=wide?qlinear_row:qaline_row;
    if(asr_parallel(in,out,1)) vllm_tp_parfor(0,out,fn,&a);
    else for(int j=0;j<out;j++) fn(&a,j);
}
#include "vllm_asr_audio.h"
#include "vllm_asr_wav.h"
static uint64_t model_digest;
static int has_audio;
static void load(const char *path,const char *map) {
    int previous=prof_set(P_LOAD);
    unsigned char h[448], tail[8], mh[20], b[65536];
    uint16_t endian=1;
    if(*(unsigned char *)&endian!=1 || sizeof(float)!=4 || FLT_MANT_DIG!=24) fail("requires IEEE little endian float32");
    FILE *f=open_file(path,"rb"), *m=open_file(map,"rb");
    read_exact(f,h,sizeof h);
    uint32_t n=u32(h+12);
    uint64_t data=u64(h+200), end=u64(h+208);
    if(u32(h)!=0x57465146 || u32(h+4)!=6 || u32(h+8)!=4096 || !n || n>10000 || data%4096 || data<448+(uint64_t)n*64 || end<data || end>LONG_MAX-8) fail("unsupported VQF header");
    if(fseek(f,0,SEEK_END) || ftell(f)!=(long)(end+8)) fail("invalid file length");
    unsigned char *dir=alloc((size_t)n*64);
    seek_to(f,448); read_exact(f,dir,(size_t)n*64);
    uint64_t prev=data;
    for(uint32_t i=0;i<n;i++) {
        unsigned char *e=dir+i*64;
        uint64_t off=u64(e+48), size=u64(e+56);
        uint32_t q=u32(e+32), r=u32(e+36), c=u32(e+40);
        if(!memchr(e,0,32) || (q!=1 && q!=6 && q!=0x100) || u32(e+44)!=(q==0x100?1u:0u) || off%64 || off<prev || off>end || size>end-off || !r || !c || (q==1 && (size%4 || (uint64_t)r*c!=size/4)) || (q==0x100 && (size<(uint64_t)r*4 || (uint64_t)r*c!=size-(uint64_t)r*4))) fail("invalid directory/shape");
        for(uint32_t j=0;j<i;j++) if(!strcmp((char *)e,(char *)dir+j*64)) fail("duplicate tensor");
        prev=off+size;
    }
    uint64_t digest=hash(UINT64_C(14695981039346656037),dir,(size_t)n*64);
    seek_to(f,data);
    for(uint64_t left=end-data;left;) { size_t z=left>sizeof b?sizeof b:(size_t)left; read_exact(f,b,z); digest=hash(digest,b,z); left-=z; }
    read_exact(f,tail,8); read_exact(m,mh,20);
    has_audio=!memcmp(mh,"ASRLM002",8);
    if(digest!=u64(tail) || (!has_audio && memcmp(mh,"ASRLM001",8)) || u64(mh+8)!=digest || u32(mh+16)!=REFS) fail("checksum/model map mismatch");
    for(int i=0;i<REFS;i++) {
        uint32_t index=word(m), rows=word(m), cols=word(m), er=1, ec=D;
        if(i==0) {er=VOC;ec=D;}
        else if(i==2) ec=32;
        else if(i==3 || i==4) ec=1024;
        else if(i==5) {er=D;ec=1024;}
        else if(i>=7) { int slot=(i-7)%12; if(slot>=2 && slot<=8) {er=slot==8?F:D;ec=(slot==6 || slot==7)?F:D;} }
        if(index>=n || rows!=er || cols!=ec) fail("unsupported fixed model shape");
        unsigned char *e=dir+index*64;
        size_t count=(size_t)rows*cols;
        int trans=i>=7 && (i-7)%12>=2 && (i-7)%12<=8;
        if((uint64_t)u32(e+36)*u32(e+40)!=count || (rows>1 && (u32(e+36)!=rows || u32(e+40)!=cols))) fail("map tensor mismatch");
        w[i]=weight_load(f,e,i==0 || i==5 || trans,trans);
        /* 线性层离线目录存 in,out；转换为连续输出行以便标量点积。 */
        if(trans && !qfind(w[i])) {
            float *t=alloc(count*4);
            for(uint32_t r=0;r<rows;r++) for(uint32_t c=0;c<cols;c++) t[(size_t)c*rows+r]=w[i][(size_t)r*cols+c];
            free(w[i]); w[i]=t;
        }
    }
    model_digest=digest;
    if(has_audio) audio_load(f,m,dir,n);
    if(fgetc(m)!=EOF) fail("trailing map bytes");
    free(dir); fclose(m); fclose(f);prof_set(previous);
}
/* 双精度累加控制长点积误差，输出与每个图算子一样舍入到 float32。 */
static void linear_row(void *ctx,int j) {
    asr_matvec *a=ctx;double s=0;const float *row=a->weight+(size_t)j*a->in;
    for(int k=0;k<a->in;k++) s+=(double)a->x[k]*row[k];
    a->y[j]=(float)s;if(a->bias) a->y[j]+=a->bias[j];
}
static void linear(float *y,const float *x,const float *weight,int in,int out,const float *bias) {
    if(qfind(weight)) {quant_linear(y,x,weight,in,out,bias,1);return;}
    if(asr_parallel(in,out,1)) {
        asr_matvec a={y,x,weight,bias,in,out};
        vllm_tp_parfor(0,out,linear_row,&a);return;
    }
    for(int j=0;j<out;j++) { double s=0; const float *row=weight+(size_t)j*in; for(int k=0;k<in;k++) s+=(double)x[k]*row[k]; y[j]=(float)s; if(bias) y[j]+=bias[j]; }
}
static void rms(float *y,const float *x,const float *weight) {
    double s=0; for(int j=0;j<D;j++) { float t=x[j]*x[j]; s+=t; }
    float inv=1.0f/sqrtf((float)(s/D)+1e-6f);
    for(int j=0;j<D;j++) y[j]=(x[j]*inv)*weight[j];
}
static void rope(float *x,int pos) {
    for(int h=0;h<8;h++) for(int j=0;j<32;j++) {
        float a=(float)pos*w[2][j], c=cosf(a), s=sinf(a), x0=x[h*64+j], x1=x[h*64+j+32];
        x[h*64+j]=x0*c-x1*s; x[h*64+j+32]=x1*c+x0*s;
    }
}
static void step(float *x,int pos,float *logits) {
    float z[D],q[D],k[D],v[D],a[D],o[D],gate[F],up[F],scores[CAP];
    if(pos<0 || pos>=CAP) fail("cache capacity exceeded");
    for(int l=0;l<8;l++) {
        float **p=w+7+l*12;
        rms(z,x,p[0]); linear(q,z,p[2],D,D,p[9]); linear(k,z,p[3],D,D,p[10]); linear(v,z,p[4],D,D,p[11]);
        rope(q,pos); rope(k,pos); memcpy(keys[l][pos],k,sizeof k); memcpy(vals[l][pos],v,sizeof v);
        for(int h=0;h<8;h++) {
            float max=-FLT_MAX;
            for(int t=0;t<=pos;t++) { double s=0; for(int j=0;j<64;j++) s+=(double)q[h*64+j]*keys[l][t][h*64+j]; scores[t]=(float)s*0.125f; if(scores[t]>max) max=scores[t]; }
            double sum=0; for(int t=0;t<=pos;t++) {scores[t]=expf(scores[t]-max);sum+=scores[t];}
            for(int t=0;t<=pos;t++) scores[t]/=(float)sum;
            for(int j=0;j<64;j++) {double s=0;for(int t=0;t<=pos;t++) s+=(double)scores[t]*vals[l][t][h*64+j];a[h*64+j]=(float)s;}
        }
        linear(o,a,p[5],D,D,NULL); for(int j=0;j<D;j++) x[j]+=o[j];
        rms(z,x,p[1]); linear(gate,z,p[6],D,F,NULL); linear(up,z,p[7],D,F,NULL);
        for(int j=0;j<F;j++) gate[j]=(gate[j]*(1.0f/(1.0f+expf(-gate[j]))))*up[j];
        linear(o,gate,p[8],F,D,NULL); for(int j=0;j<D;j++) x[j]+=o[j];
    }
    if(logits) {rms(z,x,w[1]);linear(logits,z,w[0],D,VOC,NULL);}
}
static float *project(const float *hidden,int n,int count) {
    int previous=prof_set(P_PROJECT);
    float *out=alloc((size_t)count*D*4), x[1024], z[1024];
    for(int i=0;i<count;i++) {
        int start=i*n/count, end=((i+1)*n+count-1)/count;
        for(int j=0;j<1024;j++) {float s=0;for(int t=start;t<end;t++) s+=hidden[t*1024+j];x[j]=s/(end-start);}
        double mean=0,var=0;for(int j=0;j<1024;j++) mean+=x[j];
        float mu=(float)(mean/1024);
        for(int j=0;j<1024;j++) {x[j]-=mu;float sq=x[j]*x[j];var+=sq;}
        float denom=sqrtf((float)(var/1024)+1e-5f);
        for(int j=0;j<1024;j++) z[j]=(x[j]/denom)*w[3][j]+w[4][j];
        linear(out+i*D,z,w[5],1024,D,w[6]);
    }
    prof_set(previous);return out;
}
static const float *embedding(uint32_t token) {
    static float row[D];asr_qweight *q=qfind(w[0]);
    if(!q) return w[0]+(size_t)token*D;
    for(int k=0;k<D;k++) row[k]=qvalue(q,(size_t)token*D+k);
    return row;
}
static int wav_main(int argc,char **argv) {
    if(argc<6 || argc>8) fail("usage: asr --wav model.vqf model.map assets input.wav [text.txt|-] [trace.bin]");
    load(argv[2],argv[3]);if(!has_audio) fail("WAV requires audio model map");
    wav_assets(argv[4],model_digest);
    uint32_t length,count,mask[390],ids[CAP],generated[CAP],ng=0;
    float *features=wav_features(argv[5],&length,&count);
    int valid;float *hidden=audio_encode(features,(int)length,mask,&valid);free(features);
    float *projected=project(hidden,valid,(int)count);
    prof_set(P_TRACE);
    FILE *trace=argc==8?open_file(argv[7],"wb"):NULL;
    if(trace) {if(fwrite(mask,4,390,trace)!=390) fail("trace write failed");dump(trace,hidden,(size_t)valid*1024);dump(trace,projected,(size_t)count*D);}
    prof_set(P_OTHER);
    free(hidden);length=wav_np+count+wav_ns;if(length>=CAP) fail("WAV prompt exceeds capacity");
    memcpy(ids,wav_prefix,wav_np*4);for(uint32_t i=0;i<count;i++) ids[wav_np+i]=151646;memcpy(ids+wav_np+count,wav_suffix,wav_ns*4);
    float x[D],*logits=alloc(VOC*4);uint32_t audio=0;
    prof_set(P_PREFILL);
    for(uint32_t i=0;i<length;i++) {memcpy(x,ids[i]==151646?projected+D*audio++:embedding(ids[i]),sizeof x);step(x,(int)i,i+1==length?logits:NULL);}
    prof_set(P_DECODE);
    for(uint32_t s=0;s<128;s++) {
        if(trace) {dump(trace,logits,VOC);for(int l=0;l<8;l++) {dump(trace,keys[l][length-1],D);dump(trace,vals[l][length-1],D);}}
        uint32_t token=0;
        for(uint32_t t=0;t<VOC;t++) {if(t>=151646 && t<=151650) continue;if(logits[t]>logits[token]) token=t;}
        prof_selected++;prof_set(P_TRACE);
        if(trace && fwrite(&token,4,1,trace)!=1) fail("trace write failed");
        prof_set(P_DECODE);
        fprintf(stderr,"token %u%s\n",token,(token==151645 || token==151643)?" stop":"");
        if(token==151645 || token==151643) break;
        generated[ng++]=token;prof_tokens++;if(length==CAP || s==127) break;
        memcpy(x,embedding(token),sizeof x);step(x,(int)length,logits);length++;prof_steps++;
    }
    prof_set(P_TEXT);
    char *text=wav_text(generated,ng);FILE *out=argc>=7 && strcmp(argv[6],"-")?open_file(argv[6],"wb"):stdout;
    if(fwrite(text,1,wav_text_size,out)!=wav_text_size || fputc('\n',out)==EOF || fflush(out)) fail("text write failed");
    if(out!=stdout && fclose(out)) fail("text close failed");
    prof_set(P_TRACE);
    if(trace && fclose(trace)) fail("trace close failed");
    prof_set(P_OTHER);
    free(text);free(projected);free(logits);free(wav_asset);for(int i=0;i<REFS;i++) free(w[i]);for(int i=0;i<AR;i++) free(aw[i]);return 0;
}
int main(int argc,char **argv) {
    prof_init();
    if(argc>=2 && !strcmp(argv[1],"--wav")) return wav_main(argc,argv);
    if(argc==5 && !strcmp(argv[1],"--features")) {
        wav_assets(argv[2],0);uint32_t length,count;float *features=wav_features(argv[3],&length,&count);FILE *out=open_file(argv[4],"wb");
        if(fwrite(&length,4,1,out)!=1 || fwrite(&count,4,1,out)!=1) fail("feature header write failed");
        dump(out,features,128*3000);if(fclose(out)) fail("feature close failed");free(features);free(wav_asset);return 0;
    }
    if(argc>=3 && !strcmp(argv[1],"--decode")) {
        wav_assets(argv[2],0);uint32_t ids[CAP];if(argc-3>CAP) fail("too many decode tokens");
        for(int i=3;i<argc;i++) {char *end;unsigned long n=strtoul(argv[i],&end,10);if(!*argv[i] || *end || n>=VOC) fail("invalid decode token");ids[i-3]=(uint32_t)n;}
        char *text=wav_text(ids,(size_t)(argc-3));if(fwrite(text,1,wav_text_size,stdout)!=wav_text_size || putchar('\n')==EOF) fail("decode write failed");free(text);free(wav_asset);return 0;
    }
    if(argc==3 && !strcmp(argv[1],"--mask")) {
        char *end;long value=strtol(argv[2],&end,10);uint32_t mask[390];
        if(*end || value<INT_MIN || value>INT_MAX) fail("invalid mask length");
        audio_mask((int)value,mask);
        for(int i=0;i<390;i++) printf("%u",mask[i]);
        puts("");return 0;
    }
    if(argc!=5) {fprintf(stderr,"usage: asr_lm model.vqf model.map input.bin output.bin\n");return 2;}
    load(argv[1],argv[2]);
    FILE *f=open_file(argv[3],"rb"); unsigned char h[24]; read_exact(f,h,24);
    uint32_t n=u32(h+8), count=u32(h+12), length=u32(h+16), max=u32(h+20);
    int mel=!memcmp(h,"ASRMEL01",8);uint32_t mask[390];
    if((!mel && memcmp(h,"ASRIN001",8)) || !n || n>(mel?3000u:390u) || (mel && !has_audio) || !count || count>390 || !length || length>=CAP || !max || max>CAP) fail("invalid input dimensions/capacity");
    float *hidden;
    if(mel) {
        float *features=alloc(128*3000*4);read_exact(f,features,128*3000*4);finite_array(features,128*3000);
        int valid;hidden=audio_encode(features,(int)n,mask,&valid);n=(uint32_t)valid;free(features);
    } else {hidden=alloc((size_t)n*1024*4);read_exact(f,hidden,(size_t)n*1024*4);finite_array(hidden,(size_t)n*1024);}
    uint32_t ids[CAP], audio=0;
    for(uint32_t i=0;i<length;i++) {ids[i]=word(f);if(ids[i]>=VOC) fail("token out of range");audio+=ids[i]==151646;}
    if(audio!=count || fgetc(f)!=EOF) fail("invalid prompt/trailing input");
    fclose(f); float *projected=project(hidden,(int)n,(int)count);
    prof_set(P_TRACE);
    FILE *out=open_file(argv[4],"wb");
    if(mel) {if(fwrite(mask,4,390,out)!=390) fail("output write failed");dump(out,hidden,(size_t)n*1024);}
    free(hidden);dump(out,projected,(size_t)count*D);
    prof_set(P_OTHER);
    float x[D], *logits=alloc(VOC*4); audio=0;
    prof_set(P_PREFILL);
    for(uint32_t i=0;i<length;i++) {
        memcpy(x,ids[i]==151646?projected+D*audio++:embedding(ids[i]),sizeof x);
        step(x,(int)i,i+1==length?logits:NULL);
    }
    prof_set(P_DECODE);
    for(uint32_t s=0;s<max;s++) {
        dump(out,logits,VOC);
        for(int l=0;l<8;l++) {dump(out,keys[l][length-1],D);dump(out,vals[l][length-1],D);}
        uint32_t token=0;
        for(uint32_t t=0;t<VOC;t++) {
            if(t==151647 || t==151650 || t==151648 || t==151646 || t==151649) continue;
            if(logits[t]>logits[token]) token=t;
        }
        prof_selected++;if(token!=151645 && token!=151643) prof_tokens++;
        prof_set(P_TRACE);
        if(fwrite(&token,4,1,out)!=1) fail("output write failed");
        prof_set(P_DECODE);
        printf("%u%s",token,(token==151645 || token==151643)?" stop\n":"\n");fflush(stdout);
        if(token==151645 || token==151643 || length==CAP || s+1==max) break;
        memcpy(x,embedding(token),sizeof x);step(x,(int)length,logits);length++;prof_steps++;
    }
    prof_set(P_TRACE);
    if(fclose(out)) fail("output close failed");
    prof_set(P_OTHER);
    free(projected);free(logits);for(int i=0;i<REFS;i++) free(w[i]);
    return 0;
}
