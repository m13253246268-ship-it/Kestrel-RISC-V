#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include "../../src/model/vllm_paraformer_pv.h"

/* Independent original channel-major reference, no vector reductions. */
__attribute__((noinline,optimize("no-tree-vectorize")))
static void reference(float *out,const float *score,float sum,const float *v,
                      int keys,int channels,int stride) {
    for(int c=0;c<channels;c++){
        float a=0;
        for(int j=0;j<keys;j++)a+=(score[j]/sum)*v[j*stride+c];
        out[c]=a;
    }
}
static uint32_t state=12345;
static float sample(void) {
    state=state*1664525u+1013904223u;
    return (float)((int)(state>>8)-8388608)*0x1p-20f;
}
static double now(void) {
    struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);
    return ts.tv_sec+ts.tv_nsec*1e-9;
}
static volatile float sink;
int main(int argc,char **argv) {
    const int widths[]={1,3,7,8,9,15,16,17,31,32,33,127,128,129};
    const int lengths[]={0,1,5,10,15,25,65,128,138};
    float *v=malloc(138*512*sizeof(float)),scores[138],a[131],b[131];
    if(!v)return 2;
    for(int i=0;i<138*512;i++)v[i]=sample();
    for(int i=0;i<138;i++)scores[i]=(i%7)?sample()*sample():0.0f;
    int cases=0;
    for(size_t k=0;k<sizeof(lengths)/sizeof(*lengths);k++)
        for(size_t c=0;c<sizeof(widths)/sizeof(*widths);c++){
            int n=lengths[k],w=widths[c];float sum=1.0f;
            for(int j=0;j<n;j++)sum+=scores[j];
            for(int i=0;i<131;i++)a[i]=b[i]=-1234.0f;
            reference(a+1,scores,sum,v+3,n,w,512);
            pf_pv(b+1,scores,sum,v+3,n,w,512);
            if(memcmp(a,b,sizeof(a))){fprintf(stderr,"FAIL keys=%d channels=%d\n",n,w);free(v);return 1;}
            cases++;
        }
    printf("PASS %d bitwise cases including output guards\n",cases);
    /* Fixed scope: one head PV, 65 keys x 128 channels, stride 512;
     * 10000 calls per sample, five alternating samples; no allocation timed. */
    if(argc==2&&!strcmp(argv[1],"--bench")){
        float sum=0;for(int j=0;j<65;j++)sum+=scores[j];
        for(int r=0;r<5;r++){
            double elapsed[2];
            for(int p=0;p<2;p++){
                int mode=(p+r)%2;double start=now();
                for(int i=0;i<10000;i++){
                    if(mode)pf_pv(b,scores,sum,v,65,128,512);
                    else reference(b,scores,sum,v,65,128,512);
                    sink=b[i%128];
                }
                elapsed[mode]=now()-start;
            }
            printf("sample=%d scalar_ms=%.3f selected_ms=%.3f speedup=%.3f\n",r,elapsed[0]*1000,elapsed[1]*1000,elapsed[0]/elapsed[1]);
        }
    }
    free(v);return 0;
}
