#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef PF_TEST_LEGACY
#include "../../src/model/vllm_paraformer_linear.h"
#endif

static uint32_t bits(float x) { uint32_t u;memcpy(&u,&x,4);return u; }
static float sample(unsigned i) {
    static const float v[]={0.0f,-0.0f,1.0f,-1.0f,0x1p24f,-0x1p24f,0x1p-20f,-0x1p-20f,0.75f,-0.125f};
    return v[(i*17u+i/7u)%10u];
}
/* Volatile intermediates prevent vectorization/reassociation in this oracle. */
static void reference(float *y,const float *x,const float *w,const float *b,int n,int in,int out) {
    for(int t=0;t<n;t++)for(int o=0;o<out;o++){
        volatile float sum=0.0f;
        for(int j=0;j<in;j++){volatile float product=x[(size_t)t*in+j]*w[(size_t)o*in+j];sum=sum+product;}
        y[(size_t)t*out+o]=sum+(b?b[o]:0.0f);
    }
}
#ifdef PF_TEST_LEGACY
static void pf_linear(float *y,const float *x,const float *w,const float *b,int n,int in,int out) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int t=0;t<n;t++)for(int o=0;o<out;o++){
        float sum=0.0f;const float *a=x+(size_t)t*in,*v=w+(size_t)o*in;
#ifdef _OPENMP
#pragma omp simd reduction(+:sum)
#endif
        for(int j=0;j<in;j++)sum+=a[j]*v[j];
        y[(size_t)t*out+o]=sum+(b?b[o]:0.0f);
    }
}
#endif
static size_t fma_sensitive(size_t *total) {
    const float x[]={-1.0f,0x1.000002p0f,-1.0f,0x1.000002p0f,-1.0f,0x1.000002p0f};
    const float guard=0x1.23456p20f;
    size_t mismatch=0;
    float w[12],b[6]={0},r[18],storage[20];
    for(int o=0;o<6;o++){w[2*o]=1.0f;w[2*o+1]=0x1.fffffcp-1f;}
    reference(r,x,w,NULL,1,2,1);
    if(bits(r[0])!=bits(0.0f)||bits(fmaf(x[1],w[1],-1.0f))!=bits(-0x1p-46f)){
        puts("FMA sensitivity check failed");
        mismatch++;
    }
    for(int out=5;out<=6;out++)for(int mode=0;mode<2;mode++){
        size_t ny=(size_t)3*out;
        for(size_t i=0;i<ny+2;i++)storage[i]=guard;
        const float *bias=mode?b:NULL;
        reference(r,x,w,bias,3,2,out);
        pf_linear(storage+1,x,w,bias,3,2,out);
        if(bits(storage[0])!=bits(guard)||bits(storage[ny+1])!=bits(guard)){
            printf("FMA output guard changed: out=%d mode=%d\n",out,mode);
            mismatch++;
        }
        for(size_t i=0;i<ny;i++)if(bits(storage[i+1])!=bits(r[i])){
            if(mismatch<4)printf("FMA out=%d mode=%d index=%zu got=%08x ref=%08x\n",out,mode,i,bits(storage[i+1]),bits(r[i]));
            mismatch++;
        }
        *total+=ny;
    }
    return mismatch;
}
int main(void) {
    static const int shapes[][3]={{1,4,1},{5,560,1536},{16,512,1536},{7,512,512},{5,512,1024},{5,512,2048},{5,2048,512},{1,512,8404},{3,512,1},{17,13,19},{2,0,3}};
    size_t total=0;
    size_t mismatch=fma_sensitive(&total);
    for(size_t s=0;s<sizeof(shapes)/sizeof(shapes[0]);s++){
        int n=shapes[s][0],in=shapes[s][1],out=shapes[s][2];
        size_t nx=(size_t)n*in,nw=(size_t)out*in,ny=(size_t)n*out;
        float *x=malloc((nx+1)*4),*w=malloc((nw+1)*4),*b=malloc((size_t)out*4),*y=malloc(ny*4),*r=malloc(ny*4);
        if(!x||!w||!b||!y||!r)return 2;
        for(int mode=0;mode<3;mode++){
            for(size_t i=0;i<nx;i++)x[i]=mode==2?(i%2?-0.0f:0.0f):sample((unsigned)i+3);
            for(size_t i=0;i<nw;i++)w[i]=mode==2?(i%2?-1.0f:1.0f):sample((unsigned)i+11);
            for(int o=0;o<out;o++)b[o]=mode==2?-0.0f:sample((unsigned)o+19);
            if(s==0&&mode!=2){x[0]=0x1p24f;x[1]=1.0f;x[2]=-0x1p24f;x[3]=1.0f;for(int j=0;j<4;j++)w[j]=1.0f;b[0]=0.0f;}
            const float *bias=mode==0?NULL:b;
            reference(r,x,w,bias,n,in,out);pf_linear(y,x,w,bias,n,in,out);
            for(size_t i=0;i<ny;i++)if(bits(y[i])!=bits(r[i])){
                if(mismatch<4)printf("shape=%d,%d,%d mode=%d index=%zu got=%08x ref=%08x\n",n,in,out,mode,i,bits(y[i]),bits(r[i]));
                mismatch++;
            }
            total+=ny;
        }
        free(x);free(w);free(b);free(y);free(r);
    }
    printf("linear mismatches=%zu / %zu\n",mismatch,total);
    return mismatch?1:0;
}
