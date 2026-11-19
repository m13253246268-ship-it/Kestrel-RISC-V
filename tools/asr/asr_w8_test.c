#define main asr_program_main
#include "../../src/model/vllm_asr.c"
#undef main

int main(int argc,char **argv) {
    if(argc==3 && !strcmp(argv[1],"--bad-scale")) {
        FILE *f=tmpfile();unsigned char e[64]={0};int8_t q=1;
        float scale=!strcmp(argv[2],"nan")?NAN:!strcmp(argv[2],"zero")?0:!strcmp(argv[2],"negative")?-1:INFINITY;
        e[33]=1;e[36]=1;e[40]=1;e[44]=1;e[56]=5;
        if(!f || fwrite(&q,1,1,f)!=1 || fwrite(&scale,4,1,f)!=1) return 9;
        weight_load(f,e,1,0);return 10;
    }
    if(argc==3) {load(argv[1],argv[2]);printf("loaded W8 tensors=%d\n",nq);return 0;}
    int8_t data[6]={-127,0,127,2,-3,4};float scales[2]={0.25f,0.5f};
    float handle,x[3]={1,2,3},y[3],expected[3],f32[6];
    qw[0]=(asr_qweight){&handle,data,scales,2,3,0};nq=1;
    for(int i=0;i<6;i++) f32[i]=(float)data[i]*scales[i/3];
    linear(expected,x,f32,3,2,NULL);linear(y,x,&handle,3,2,NULL);
    if(memcmp(y,expected,8)) return 2;
    aline(expected,x,f32,3,2,NULL);aline(y,x,&handle,3,2,NULL);
    if(memcmp(y,expected,8)) return 3;
    qw[0].transposed=1;
    float trans[6];for(int r=0;r<2;r++) for(int c=0;c<3;c++) trans[c*2+r]=f32[r*3+c];
    linear(expected,x,trans,2,3,NULL);linear(y,x,&handle,2,3,NULL);
    if(memcmp(y,expected,12)) return 4;
    aline(expected,x,trans,2,3,NULL);aline(y,x,&handle,2,3,NULL);
    if(memcmp(y,expected,12)) return 5;
    enum {N=512};
    int8_t *large=alloc(N*N);float *ls=alloc(N*4),*lx=alloc(N*4),*ly=alloc(N*4),*ref=alloc(N*4);
    for(int i=0;i<N;i++) {ls[i]=(float)(i+1)/1024;lx[i]=(float)(i%13-6)/8;}
    for(int i=0;i<N*N;i++) large[i]=(int8_t)(i%255-127);
    qw[0]=(asr_qweight){&handle,large,ls,N,N,0};
    for(int mode=0;mode<2;mode++) {
        quant_linear(ly,lx,&handle,N,N,NULL,mode);
        asr_matvec a={ref,lx,&handle,NULL,N,N};
        for(int j=0;j<N;j++) {if(mode) qlinear_row(&a,j);else qaline_row(&a,j);}
        if(memcmp(ly,ref,N*4)) return 6;
    }
    free(large);free(ls);free(lx);free(ly);free(ref);
    const int shapes[][2]={{1,1},{31,17},{33,31},{513,527},{896,896},{896,3584},{3584,896},{1024,4096},{4096,1024},{7680,896}};
    for(size_t shape=0;shape<sizeof shapes/sizeof shapes[0];shape++) {
        int in=shapes[shape][0],out=shapes[shape][1];
        int8_t *q=alloc((size_t)in*out);
        float *s=alloc((size_t)(in>out?in:out)*4),*input=alloc(in*4);
        float *actual=alloc(out*4),*oracle=alloc(out*4),*bias=alloc(out*4);
        for(int i=0;i<in*out;i++) q[i]=(int8_t)(i%256-128);
        for(int i=0;i<(in>out?in:out);i++) s[i]=(i%27+1)*0.001013f;
        for(int i=0;i<in;i++) input[i]=(i%23-11)*0.137f;
        for(int i=0;i<out;i++) bias[i]=(i%7-3)*0.171f;
        for(int transposed=0;transposed<2;transposed++) {
            qw[0]=(asr_qweight){&handle,q,s,(uint32_t)(transposed?in:out),(uint32_t)(transposed?out:in),transposed};
            for(int wide=0;wide<2;wide++) for(int with_bias=0;with_bias<2;with_bias++) {
                const float *b=with_bias?bias:NULL;
                asr_matvec a={oracle,input,&handle,b,in,out};
                for(int j=0;j<out;j++) {if(wide) qlinear_row(&a,j);else qaline_row(&a,j);}
                quant_linear(actual,input,&handle,in,out,b,wide);
                if(memcmp(actual,oracle,(size_t)out*4)) {
                    fprintf(stderr,"W8 dispatch mismatch in=%d out=%d\n",in,out);return 7;
                }
#if defined(__riscv_vector) && defined(VLLM_ASR_W8_RVV) && !defined(VLLM_ASR_SCALAR)
                if(!wide) {
                    a.y=actual;
                    int blocks=(out+W8_LANES-1)/W8_LANES;
                    if(asr_parallel(in,out,W8_LANES)) vllm_tp_parfor(0,blocks,w8_encoder_block,&a);
                    else for(int j=0;j<blocks;j++) w8_encoder_block(&a,j);
                }
#endif
                if(memcmp(actual,oracle,(size_t)out*4)) {
                    fprintf(stderr,"W8 mismatch in=%d out=%d transposed=%d wide=%d bias=%d\n",in,out,transposed,wide,with_bias);return 7;
                }
            }
        }
        free(q);free(s);free(input);free(actual);free(oracle);free(bias);
    }
    puts("W8 production oracle: signed boundaries, non-power scales, cancellation, tails, bias/null, both layouts, FP32/FP64 and parallel bitwise passed");return 0;
}
