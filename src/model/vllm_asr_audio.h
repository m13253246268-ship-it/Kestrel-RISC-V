/* Fixed Audio8 encoder. Included by the standalone ASR harness only. */
#if defined(__riscv_vector) && !defined(VLLM_ASR_SCALAR)
#include <riscv_vector.h>
#endif
#define AR 331
static float *aw[AR];
static const char audio_graph[]="c03f10a6fbb6c5b413b0b4bc08927dee4e59334e91c2db3dca6f095cf9492e16";
static void audio_load(FILE *f,FILE *m,unsigned char *dir,uint32_t n) {
    char identity[64]; read_exact(m,identity,64);
    if(memcmp(identity,audio_graph,64)) fail("audio graph identity mismatch");
    for(int i=0;i<AR;i++) {
        uint32_t r=1,c=1,ix=word(m),mr=word(m),mc=word(m);
        if(i<6) {r=480;c=i%2?1:(i==0?9:4320);}
        else if(i==6) {r=7680;c=896;}
        else if(i==7) {r=1;c=13*896;}
        else if(i>=11 && i<299) {
            int s=(i-11)%16;
            r=s<4?896:s<12?(s%2?896:896):s==12?3584:s==13?3584:896;
            c=s<4?1:s<12?(s%2?1:896):s==12?896:s==14?3584:1;
        } else if(i>=299 && i<305) {int s=i-299;r=s<4?896:1024;c=s==2 || s==4?896:1;}
        else if(i>=305 && i<329) {int s=(i-305)%6;r=s==2 || s==3?4096:1024;c=s==2?1024:s==4?4096:1;}
        else if(i>=329) {r=1024;c=1;}
        if(ix>=n || mr!=r || mc!=c) fail("unsupported audio tensor shape");
        unsigned char *e=dir+(size_t)ix*64;
        if(u32(e+36)!=r || u32(e+40)!=c) fail("audio tensor dtype/shape mismatch");
        aw[i]=weight_load(f,e,r>1 && c>1,i==6);
    }
}
static int audio_mask(int length,uint32_t *mask) {
    int m=length<1?1:length>3000?3000:length,n=0;
    for(int i=0;i<390;i++) {int frames=m-(i/13)*100;frames=frames<0?0:frames>100?100:frames;mask[i]=(uint32_t)(i%13<(frames+7)/8);n+=(int)mask[i];}
    return n;
}
/* Output-lane blocking preserves the ascending input-index accumulation order. */
static void aline_block(void *ctx,int block) {
    asr_matvec *a=ctx;int j=block*8,lanes=a->out-j<8?a->out-j:8;float s[8]={0};
    for(int k=0;k<a->in;k++) for(int b=0;b<lanes;b++) s[b]+=a->x[k]*a->weight[(size_t)(j+b)*a->in+k];
    for(int b=0;b<lanes;b++) a->y[j+b]=s[b]+(a->bias?a->bias[j+b]:0.0f);
}
static void aline(float *y,const float *x,const float *weight,int in,int out,const float *bias) {
    if(qfind(weight)) {quant_linear(y,x,weight,in,out,bias,0);return;}
    if(asr_parallel(in,out,8)) {
        asr_matvec a={y,x,weight,bias,in,out};
        vllm_tp_parfor(0,(out+7)/8,aline_block,&a);return;
    }
    for(int j=0;j<out;j+=8) {
        float s[8]={0};int lanes=out-j<8?out-j:8;
        for(int k=0;k<in;k++) for(int b=0;b<lanes;b++) s[b]+=x[k]*weight[(size_t)(j+b)*in+k];
        for(int b=0;b<lanes;b++) y[j+b]=s[b]+(bias?bias[j+b]:0.0f);
    }
}
static void anorm(float *y,const float *x,int d,const float *weight,const float *bias) {
    float mean=0,var=0;for(int i=0;i<d;i++) mean+=x[i];mean/=d;
    for(int i=0;i<d;i++) {float t=x[i]-mean;var+=t*t;}
    float inv=1.0f/sqrtf(var/d+1e-5f);
    for(int i=0;i<d;i++) y[i]=((x[i]-mean)*inv)*weight[i]+bias[i];
}
static void agelu(float *x,int n) {for(int i=0;i<n;i++) x[i]=(x[i]*(1.0f+erff(x[i]/aw[8][0])))*0.5f;}
/* Explicit opt-in; the independent scalar switch overrides the candidate. */
#if defined(__riscv_vector) && !defined(VLLM_ASR_SCALAR) && defined(VLLM_ASR_CONV_PACK) && !defined(VLLM_ASR_CONV_SCALAR)
#define ASR_CONV_PACK_ACTIVE
static void affn_free(float *p);
static float *affn_pack(const float *weight,int in,int out);
static void affn_down(float *y,const float *x,const float *packed,int in,int out,const float *bias);
#endif
static float *aconv(const float *x,int channels,int h,int width,int layer) {
    int oh=(h+1)/2,ow=(width+1)/2,in=channels*9;
    float *y=alloc((size_t)480*oh*ow*4),*col=alloc((size_t)in*4),v[480];
#ifdef ASR_CONV_PACK_ACTIVE
    float *packed=layer==1 || layer==2?affn_pack(aw[layer*2],in,480):NULL;
#endif
    for(int a=0;a<oh;a++) for(int b=0;b<ow;b++) {
        int k=0;for(int c=0;c<channels;c++) for(int u=0;u<3;u++) for(int t=0;t<3;t++) {
            int r=a*2+u-1,s=b*2+t-1;
            col[k++]=r<0 || r>=h || s<0 || s>=width?0.0f:x[(c*h+r)*width+s];
        }
#ifdef ASR_CONV_PACK_ACTIVE
        if(packed) affn_down(v,col,packed,in,480,aw[layer*2+1]);
        else
#endif
        aline(v,col,aw[layer*2],in,480,aw[layer*2+1]);
        agelu(v,480);
        for(int c=0;c<480;c++) y[(c*oh+a)*ow+b]=v[c];
    }
#ifdef ASR_CONV_PACK_ACTIVE
    affn_free(packed);
#endif
    free(col);return y;
}
/* Independent output lanes; keep each FP32 input-index chain and bias last.
 * Compile with -ffp-contract=off; VLLM_ASR_SCALAR retains the reference path. */
static void aconv_out_range(float *y,const float *x,const float *weight,int in,int out,const float *bias,int start,int end) {
#if defined(__riscv_vector) && !defined(VLLM_ASR_SCALAR)
    for(size_t j=(size_t)start;j<(size_t)end;) {
        size_t vl=__riscv_vsetvl_e32m4((size_t)end-j);
        vfloat32m4_t sum=__riscv_vfmv_v_f_f32m4(0.0f,vl);
        for(int k=0;k<in;k++) {
            vfloat32m4_t value=__riscv_vle32_v_f32m4(weight+(size_t)k*out+j,vl);
            vfloat32m4_t product=__riscv_vfmul_vf_f32m4(value,x[k],vl);
            sum=__riscv_vfadd_vv_f32m4(sum,product,vl);
        }
        sum=__riscv_vfadd_vv_f32m4(sum,__riscv_vle32_v_f32m4(bias+j,vl),vl);
        __riscv_vse32_v_f32m4(y+j,sum,vl);j+=vl;
    }
#else
    for(int j=start;j<end;j++) {float sum=0;for(int k=0;k<in;k++) sum+=x[k]*weight[(size_t)k*out+j];y[j]=sum+bias[j];}
#endif
}
static void aconv_out_block(void *ctx,int block) {
    asr_matvec *a=ctx;int start=block*32,end=start+32<a->out?start+32:a->out;
    aconv_out_range(a->y,a->x,a->weight,a->in,a->out,a->bias,start,end);
}
static void aconv_out(float *y,const float *x,const float *weight,int in,int out,const float *bias) {
    if(qfind(weight)) {quant_linear(y,x,weight,in,out,bias,0);return;}
    if(asr_parallel(in,out,32)) {
        asr_matvec a={y,x,weight,bias,in,out};
        vllm_tp_parfor(0,(out+31)/32,aconv_out_block,&a);return;
    }
    aconv_out_range(y,x,weight,in,out,bias,0,out);
}
/* 编码器 FFN 上下投影按 [tile][k][lane] 打包，每层复用全部 token。
 * 两份打包同时存活的有效载荷峰值为 24.5 MiB，较仅下投影新增 12.25 MiB。
 * 保留输入维递增的 FP32 单链和最后加偏置，严格关闭浮点收缩。 */
#if (defined(__riscv_vector) && !defined(VLLM_ASR_SCALAR) && (!defined(VLLM_ASR_FFN_SCALAR) || !defined(VLLM_ASR_TAIL_SCALAR) || !defined(VLLM_ASR_QKV_SCALAR) || !defined(VLLM_ASR_OUT_SCALAR))) || defined(VLLM_ASR_FFN_TEST) || defined(ASR_CONV_PACK_ACTIVE)
static inline void affn_free(float *p) {if(!qfind(p)) free(p);}
static float *affn_pack(const float *weight,int in,int out) {
    if(qfind(weight)) return (float *)weight;
    float *packed=alloc((size_t)in*out*4);
    for(int j=0;j<out;j+=32) {
        int lanes=out-j<32?out-j:32;
        for(int kb=0;kb<in;kb+=32) {
            int end=kb+32<in?kb+32:in;
            for(int b=0;b<lanes;b++) for(int k=kb;k<end;k++)
                packed[(size_t)j*in+(size_t)k*lanes+b]=weight[(size_t)(j+b)*in+k];
        }
    }
    return packed;
}
static void affn_down_range(float *y,const float *x,const float *packed,int in,int out,const float *bias,int start,int end) {
    for(int j=start;j<end;j+=32) {
        int lanes=out-j<32?out-j:32;
#if defined(__riscv_vector) && !defined(VLLM_ASR_SCALAR)
        for(size_t b=0;b<(size_t)lanes;) {
            size_t vl=__riscv_vsetvl_e32m4((size_t)lanes-b);
            vfloat32m4_t sum=__riscv_vfmv_v_f_f32m4(0.0f,vl);
            for(int k=0;k<in;k++) {
                vfloat32m4_t value=__riscv_vle32_v_f32m4(packed+(size_t)j*in+(size_t)k*lanes+b,vl);
                vfloat32m4_t product=__riscv_vfmul_vf_f32m4(value,x[k],vl);
                sum=__riscv_vfadd_vv_f32m4(sum,product,vl);
            }
            vfloat32m4_t bv=bias?__riscv_vle32_v_f32m4(bias+j+b,vl):__riscv_vfmv_v_f_f32m4(0.0f,vl);
            sum=__riscv_vfadd_vv_f32m4(sum,bv,vl);
            __riscv_vse32_v_f32m4(y+j+b,sum,vl);b+=vl;
        }
#else
        /* 测试入口在非 RVV 平台也直接验证生产打包布局。 */
        for(int b=0;b<lanes;b++) {
            float sum=0;
            for(int k=0;k<in;k++) sum+=x[k]*packed[(size_t)j*in+(size_t)k*lanes+b];
            y[j+b]=sum+(bias?bias[j+b]:0.0f);
        }
#endif
    }
}
static void affn_down_tile(void *ctx,int tile) {
    asr_matvec *a=ctx;
    affn_down_range(a->y,a->x,a->weight,a->in,a->out,a->bias,tile*32,tile*32+32);
}
static void affn_down(float *y,const float *x,const float *packed,int in,int out,const float *bias) {
    if(qfind(packed)) {quant_linear(y,x,packed,in,out,bias,0);return;}
    if(asr_parallel(in,out,32)) {
        asr_matvec a={y,x,packed,bias,in,out};
        vllm_tp_parfor(0,(out+31)/32,affn_down_tile,&a);return;
    }
    affn_down_range(y,x,packed,in,out,bias,0,out);
}
#endif
static float *audio_encode(const float *mel,int length,uint32_t *mask,int *n_out) {
    int previous=prof_set(P_ENCODER);
    int n=audio_mask(length,mask);*n_out=n;
    float *x=alloc((size_t)n*896*4),chunk[128*100],flat[7680];
    int row=0;
    /* Invalid tokens cannot reach valid queries: the exact mask assigns them only
       their own diagonal. All other operators are token-local or chunk-local. */
    for(int c=0;c<30;c++) {
        if(!mask[c*13]) continue;
        for(int f=0;f<128;f++) for(int t=0;t<100;t++) chunk[f*100+t]=mel[f*3000+c*100+t];
        prof_set(P_CONV0);float *a=aconv(chunk,1,128,100,0);
        prof_set(P_CONV1);float *b=aconv(a,480,64,50,1);free(a);
        prof_set(P_CONV2);a=aconv(b,480,32,25,2);free(b);
        prof_set(P_CONVOUT);
        for(int t=0;t<13;t++) if(mask[c*13+t]) {
            for(int f=0;f<480;f++) for(int s=0;s<16;s++) flat[f*16+s]=a[(f*16+s)*13+t];
            aconv_out(x+row*896,flat,aw[6],7680,896,aw[7]+t*896);
            row++;
        }
        prof_set(P_ENCODER);free(a);fprintf(stderr,"audio chunk %d/30\n",c+1);fflush(stderr);
    }
    float *q=alloc((size_t)n*896*4),*k=alloc((size_t)n*896*4),*v=alloc((size_t)n*896*4);
    float z[1024],o[1024],att[896],ff[4096],scores[104];
    for(int l=0;l<18;l++) {
        float **p=aw+11+l*16;
        prof_set(P_QKV);
        /* QKV 9.1875 MiB 先释放，再分配 OUT+FFN 27.5625 MiB；尾部不重叠。 */
#if defined(__riscv_vector) && !defined(VLLM_ASR_SCALAR) && !defined(VLLM_ASR_QKV_SCALAR)
        float *packed_q=affn_pack(p[4],896,896),*packed_k=affn_pack(p[6],896,896),*packed_v=affn_pack(p[8],896,896);
#endif
        for(int t=0;t<n;t++) {
            anorm(z,x+t*896,896,p[0],p[1]);
#if defined(__riscv_vector) && !defined(VLLM_ASR_SCALAR) && !defined(VLLM_ASR_QKV_SCALAR)
            affn_down(q+t*896,z,packed_q,896,896,p[5]);affn_down(k+t*896,z,packed_k,896,896,p[7]);affn_down(v+t*896,z,packed_v,896,896,p[9]);
#else
            aline(q+t*896,z,p[4],896,896,p[5]);aline(k+t*896,z,p[6],896,896,p[7]);aline(v+t*896,z,p[8],896,896,p[9]);
#endif
        }
#if defined(__riscv_vector) && !defined(VLLM_ASR_SCALAR) && !defined(VLLM_ASR_QKV_SCALAR)
        affn_free(packed_q);affn_free(packed_k);affn_free(packed_v);
#endif
#if defined(__riscv_vector) && !defined(VLLM_ASR_SCALAR) && !defined(VLLM_ASR_OUT_SCALAR)
        prof_set(P_OUTPROJ);float *packed_out=affn_pack(p[10],896,896);
#endif
#if defined(__riscv_vector) && !defined(VLLM_ASR_SCALAR) && !defined(VLLM_ASR_FFN_SCALAR)
        prof_set(P_FFN);float *packed_up=affn_pack(p[12],896,3584),*packed_down=affn_pack(p[14],3584,896);
#endif
        for(int t=0;t<n;t++) {
            prof_set(P_ATTN);
            int start=t/104*104,end=start+104;if(end>n) end=n;
            for(int h=0;h<14;h++) {
                float max=-FLT_MAX,sum=0;
                for(int s=start;s<end;s++) {float dot=0;for(int j=0;j<64;j++) dot+=(q[t*896+h*64+j]*aw[9][0])*(k[s*896+h*64+j]*aw[10][0]);scores[s-start]=dot;if(dot>max) max=dot;}
                for(int s=start;s<end;s++) {scores[s-start]=expf(scores[s-start]-max);sum+=scores[s-start];}
                for(int s=start;s<end;s++) scores[s-start]/=sum;
                for(int j=0;j<64;j++) {float dot=0;for(int s=start;s<end;s++) dot+=scores[s-start]*v[s*896+h*64+j];att[h*64+j]=dot;}
            }
            prof_set(P_OUTPROJ);
#if defined(__riscv_vector) && !defined(VLLM_ASR_SCALAR) && !defined(VLLM_ASR_OUT_SCALAR)
            affn_down(o,att,packed_out,896,896,p[11]);
#else
            aline(o,att,p[10],896,896,p[11]);
#endif
            for(int j=0;j<896;j++) x[t*896+j]+=o[j];
            prof_set(P_FFN);anorm(z,x+t*896,896,p[2],p[3]);
#if defined(__riscv_vector) && !defined(VLLM_ASR_SCALAR) && !defined(VLLM_ASR_FFN_SCALAR)
            affn_down(ff,z,packed_up,896,3584,p[13]);
#else
            aline(ff,z,p[12],896,3584,p[13]);
#endif
            agelu(ff,3584);
#if defined(__riscv_vector) && !defined(VLLM_ASR_SCALAR) && !defined(VLLM_ASR_FFN_SCALAR)
            affn_down(o,ff,packed_down,3584,896,p[15]);
#else
            aline(o,ff,p[14],3584,896,p[15]);
#endif
            for(int j=0;j<896;j++) x[t*896+j]+=o[j];
        }
#if defined(__riscv_vector) && !defined(VLLM_ASR_SCALAR) && !defined(VLLM_ASR_FFN_SCALAR)
        affn_free(packed_up);affn_free(packed_down);
#endif
#if defined(__riscv_vector) && !defined(VLLM_ASR_SCALAR) && !defined(VLLM_ASR_OUT_SCALAR)
        prof_set(P_OUTPROJ);affn_free(packed_out);
#endif
        prof_set(P_ENCODER);fprintf(stderr,"audio encoder layer %d/18\n",l+1);fflush(stderr);
    }
    free(q);free(k);free(v);
    float *hidden=alloc((size_t)n*1024*4);
    prof_set(P_TAIL);
    for(int t=0;t<n;t++) {
        anorm(z,x+t*896,896,aw[299],aw[300]);aline(o,z,aw[301],896,896,aw[302]);agelu(o,896);aline(hidden+t*1024,o,aw[303],896,1024,aw[304]);
    }
    /* 各 token 独立；逐层打包复用，层末释放 32 MiB 有效载荷。
     * 编码器打包已释放；复用 z/o/ff，不分配全 token 的 4096 维缓冲。
     * 打包及释放均计入 P_TAIL，TAIL_SCALAR 仅隔离尾部八个线性。 */
    for(int l=0;l<4;l++) {
        float **p=aw+305+l*6;
#if defined(__riscv_vector) && !defined(VLLM_ASR_SCALAR) && !defined(VLLM_ASR_TAIL_SCALAR)
        float *packed_up=affn_pack(p[2],1024,4096),*packed_down=affn_pack(p[4],4096,1024);
#endif
        for(int t=0;t<n;t++) {
            anorm(z,hidden+t*1024,1024,p[0],p[1]);
#if defined(__riscv_vector) && !defined(VLLM_ASR_SCALAR) && !defined(VLLM_ASR_TAIL_SCALAR)
            affn_down(ff,z,packed_up,1024,4096,p[3]);
#else
            aline(ff,z,p[2],1024,4096,p[3]);
#endif
            agelu(ff,4096);
#if defined(__riscv_vector) && !defined(VLLM_ASR_SCALAR) && !defined(VLLM_ASR_TAIL_SCALAR)
            affn_down(o,ff,packed_down,4096,1024,p[5]);
#else
            aline(o,ff,p[4],4096,1024,p[5]);
#endif
            for(int j=0;j<1024;j++) hidden[t*1024+j]+=o[j];
        }
#if defined(__riscv_vector) && !defined(VLLM_ASR_SCALAR) && !defined(VLLM_ASR_TAIL_SCALAR)
        affn_free(packed_up);affn_free(packed_down);
#endif
    }
    for(int t=0;t<n;t++) {
        anorm(z,hidden+t*1024,1024,aw[329],aw[330]);memcpy(hidden+t*1024,z,1024*4);
    }
    prof_set(P_ENCODER);free(x);finite_array(hidden,(size_t)n*1024);prof_set(previous);return hidden;
}
