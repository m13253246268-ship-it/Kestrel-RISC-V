#ifndef VLLM_PARAFORMER_PV_H
#define VLLM_PARAFORMER_PV_H
#include <stddef.h>
#if defined(__riscv_vector) && !defined(VLLM_PARAFORMER_FORCE_SCALAR)
#include <riscv_vector.h>
#endif

/* Channels are independent lanes; each lane retains ascending key order.
 * Compile with -ffp-contract=off, without fast-math. */
static inline void pf_pv(float *out,const float *score,float sum,
                         const float *v,int keys,int channels,int stride) {
#if defined(__riscv_vector) && !defined(VLLM_PARAFORMER_FORCE_SCALAR)
    for(int c=0;c<channels;){
        size_t vl=__riscv_vsetvl_e32m1((size_t)(channels-c));
        vfloat32m1_t acc=__riscv_vfmv_v_f_f32m1(0.0f,vl);
        for(int j=0;j<keys;j++){
            float p=score[j]/sum;
            vfloat32m1_t value=__riscv_vle32_v_f32m1(v+(size_t)j*stride+c,vl);
            vfloat32m1_t product=__riscv_vfmul_vf_f32m1(value,p,vl);
            acc=__riscv_vfadd_vv_f32m1(acc,product,vl);
        }
        __riscv_vse32_v_f32m1(out+c,acc,vl);
        c+=(int)vl;
    }
#else
    for(int c=0;c<channels;c++){
        float acc=0;
        for(int j=0;j<keys;j++)acc+=(score[j]/sum)*v[(size_t)j*stride+c];
        out[c]=acc;
    }
#endif
}
#endif
