/* Compile twice with the product flags and --gc-sections, toggling VLLM_K1_Q8B_ASM. */
#include "../../src/model/vllm_safetensors.c"
#include <time.h>
int main(int argc, char **argv) {
    enum { rows = 128, cols = 1024, patches = Q8B_PATCH_BLK };
    static uint8_t weights[rows * (cols/32) * 34];
    static float x[patches * cols], out[rows * patches];
    if (argc != 2) return 2;
    uint32_t rng = 42;
    for (size_t i=0; i<sizeof(weights); i+=34) {
        uint16_t scale = 0x2800 + i%4096;
        memcpy(weights+i,&scale,2);
        for (int j=2;j<34;j++) { rng=rng*1664525u+1013904223u; weights[i+j]=rng>>24; }
    }
    for (int i=0;i<patches*cols;i++) { rng=rng*1664525u+1013904223u; x[i]=((int)(rng>>12)-524288)*0.000013f; }
    vllm_q8b_ctx ctx = {.out=out,.q8_w=weights,.x=x,.B=patches,.rows=rows,.cols=cols};
    for (int rep=-1;rep<5;rep++) {
        struct timespec a,b; clock_gettime(CLOCK_MONOTONIC,&a);
        for (int r=0;r<rows;r++) vllm_q8b_worker(&ctx,r);
        clock_gettime(CLOCK_MONOTONIC,&b);
        if (rep>=0) printf("%d x %d x %d: %.3f ms\n",rows,cols,patches,(b.tv_sec-a.tv_sec)*1000.+(b.tv_nsec-a.tv_nsec)*1e-6);
    }
    FILE *f=fopen(argv[1],"wb"); if(!f) return 1;
    return fwrite(out,1,sizeof out,f)!=sizeof out || fclose(f)!=0;
}
