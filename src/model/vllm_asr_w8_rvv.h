#include <riscv_vector.h>
#define W8_LANES 16
/* Independent output lanes retain increasing-k FP32 decode/multiply/add.
 * The 512-byte transpose tile belongs to each task and is never retained. */
static void w8_encoder_block(void *ctx,int block) {
    asr_matvec *a=ctx;
    const asr_qweight *q=qfind(a->weight);
    int j=block*W8_LANES,count=a->out-j;
    if(count>W8_LANES) count=W8_LANES;
    for(int lane=0;lane<count;) {
        size_t vl=__riscv_vsetvl_e32m2((size_t)(count-lane));
        int start=j+lane;
        vfloat32m2_t sum=__riscv_vfmv_v_f_f32m2(0.0f,vl);
        if(q->transposed) {
            for(int k=0;k<a->in;k++) {
                vint8mf2_t b=__riscv_vle8_v_i8mf2(q->q+(size_t)k*q->cols+start,vl);
                vint32m2_t n=__riscv_vsext_vf4_i32m2(b,vl);
                vfloat32m2_t wv=__riscv_vfmul_vf_f32m2(__riscv_vfcvt_f_x_v_f32m2(n,vl),q->scale[k],vl);
                sum=__riscv_vfadd_vv_f32m2(sum,__riscv_vfmul_vf_f32m2(wv,a->x[k],vl),vl);
            }
        } else {
            int8_t tile[32][W8_LANES];
            vfloat32m2_t scale=__riscv_vle32_v_f32m2(q->scale+start,vl);
            for(int kb=0;kb<a->in;kb+=32) {
                int nk=a->in-kb;if(nk>32) nk=32;
                for(size_t l=0;l<vl;l++) for(int k=0;k<nk;k++) tile[k][l]=q->q[(size_t)(start+l)*q->cols+kb+k];
                for(int k=0;k<nk;k++) {
                    vint8mf2_t b=__riscv_vle8_v_i8mf2(tile[k],vl);
                    vint32m2_t n=__riscv_vsext_vf4_i32m2(b,vl);
                    vfloat32m2_t wv=__riscv_vfmul_vv_f32m2(__riscv_vfcvt_f_x_v_f32m2(n,vl),scale,vl);
                    sum=__riscv_vfadd_vv_f32m2(sum,__riscv_vfmul_vf_f32m2(wv,a->x[kb+k],vl),vl);
                }
            }
        }
        if(a->bias) sum=__riscv_vfadd_vv_f32m2(sum,__riscv_vle32_v_f32m2(a->bias+start,vl),vl);
        else sum=__riscv_vfadd_vf_f32m2(sum,0.0f,vl);
        __riscv_vse32_v_f32m2(a->y+start,sum,vl);
        lane+=(int)vl;
    }
}
