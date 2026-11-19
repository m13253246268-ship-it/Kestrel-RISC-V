/* 直接调用生产 FFN 打包和内核，逐字节比较 aline；覆盖尾 VL 和真实 token。
 * 链接 src/core/vllm_tp.c，使用 -D_GNU_SOURCE -Iinclude/core -Iinclude/common -pthread。 */
#define VLLM_ASR_FFN_TEST
#define main asr_entry
#include "../../src/model/vllm_asr.c"
#undef main
static void check_case(const float *x,const float *weight,const float *bias,int in,int out) {
    float *packed=affn_pack(weight,in,out),*a=alloc((size_t)out*4),*b=alloc((size_t)out*4);
    aline(a,x,weight,in,out,bias);affn_down(b,x,packed,in,out,bias);
    if(memcmp(a,b,(size_t)out*4)) fail("production FFN bit mismatch");
    free(packed);free(a);free(b);
}
int main(int argc,char **argv) {
    if(argc!=5) return 2;
    prof_init();load(argv[1],argv[2]);
    const int widths[]={1,3,7,15,16,17,31,32,33,63,64,65,127,128,129,895,896,3583,3584};
    float small_x[17],*small_w=alloc(17*3584*4),small_b[3584];
    const float values[]={0.0f,-0.0f,1.0f,-1.0f,0x1p-120f,-0x1p-120f,0x1p-20f,-0x1p-20f,0.125f,-0.125f};
    for(int mode=0;mode<3;mode++) {
        for(int k=0;k<17;k++) small_x[k]=values[(k+mode)%10];
        for(int j=0;j<3584;j++) {small_b[j]=values[(j+mode)%10];for(int k=0;k<17;k++) small_w[j*17+k]=values[(j+k+mode)%10];}
        for(size_t i=0;i<sizeof widths/sizeof widths[0];i++) {
            check_case(small_x,small_w,small_b,17,widths[i]);
            check_case(small_x,small_w,NULL,17,widths[i]);
        }
    }
    free(small_w);puts("production FFN cancellation/signed-zero/small-values/tail-VL/null-bias bitwise PASS");fflush(stdout);
    for(int direction=0;direction<2;direction++) {
        int in=direction?3584:896,out=direction?896:3584,slot=23+direction*2;
        FILE *f=open_file(argv[3+direction],"rb");if(fseek(f,0,SEEK_END)) fail("input seek");long bytes=ftell(f);
        if(bytes<=0 || bytes%(in*4)) fail("input shape");
        rewind(f);
        size_t n=(size_t)bytes/(in*4);float *x=alloc((size_t)bytes);read_exact(f,x,(size_t)bytes);fclose(f);
        float *a=alloc(n*out*4),*b=alloc(n*out*4);
        for(int repeat=0;repeat<3;repeat++) {
            double start=prof_now();float *packed=affn_pack(aw[slot],in,out);double pack=prof_now()-start;
            start=prof_now();for(size_t t=0;t<n;t++) aline(a+t*out,x+t*in,aw[slot],in,out,aw[slot+1]);double scalar=prof_now()-start;
            start=prof_now();for(size_t t=0;t<n;t++) affn_down(b+t*out,x+t*in,packed,in,out,aw[slot+1]);double vector=prof_now()-start;
            if(memcmp(a,b,n*out*4)) fail("real production FFN bit mismatch");
            printf("production FFN direction=%d tile=32 repeat=%d tokens=%zu original=%.9f kernel=%.9f pack=%.9f speedup=%.6f inclusive=%.6f bytes=%zu bitwise=PASS\n",direction,repeat,n,scalar,vector,pack,scalar/vector,scalar/(vector+pack),(size_t)in*out*4);fflush(stdout);
            free(packed);
        }
        free(x);free(a);free(b);
    }
    for(int i=0;i<REFS;i++) free(w[i]);
    for(int i=0;i<AR;i++) free(aw[i]);
    return 0;
}
