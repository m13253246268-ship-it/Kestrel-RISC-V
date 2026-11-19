/* Build with the ASR strict flags; argv: model.vqf model.map conv_flat.bin.
 * Link src/core/vllm_tp.c with -D_GNU_SOURCE -Iinclude/core -Iinclude/common -pthread.
 * conv_flat.bin is one captured 7680-element FP32 conv_out input. */
#define main asr_entry
#include "../../src/model/vllm_asr.c"
#undef main
static void reference(float *y,const float *x,const float *weight,int in,int out,const float *bias) {
    for(int j=0;j<out;j++) {float sum=0;for(int k=0;k<in;k++) sum+=x[k]*weight[(size_t)k*out+j];y[j]=sum+bias[j];}
}
int main(int argc,char **argv) {
    if(argc!=4) return 2;
    load(argv[1],argv[2]);
    float x[7680],expected[896],actual[896];
    FILE *f=open_file(argv[3],"rb");read_exact(f,x,sizeof x);
    if(fgetc(f)!=EOF) fail("trailing conv input");
    fclose(f);finite_array(x,7680);
    const int widths[]={1,3,7,15,31,32,33,63,65,895,896};
    for(size_t i=0;i<sizeof widths/sizeof widths[0];i++) {
        int out=widths[i];reference(expected,x,aw[6],7680,out,aw[7]);
        aconv_out(actual,x,aw[6],7680,out,aw[7]);
        if(memcmp(expected,actual,(size_t)out*4)) fail("conv_out bit mismatch");
    }
    puts("conv_out real input/weights, 11 shapes including tail VL: bitwise PASS");
    for(int i=0;i<REFS;i++) free(w[i]);
    for(int i=0;i<AR;i++) free(aw[i]);
    return 0;
}
