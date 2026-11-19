#ifndef VLLM_PARAFORMER_LINEAR_H
#define VLLM_PARAFORMER_LINEAR_H
#include <stddef.h>
#if defined(__FAST_MATH__)
#error "Paraformer linear requires strict floating point (no fast-math)"
#endif
#if defined(__riscv_vector) && defined(VLLM_PARAFORMER_LINEAR_RVV) && !defined(VLLM_PARAFORMER_FORCE_SCALAR)
#include <riscv_vector.h>
#endif

/* Compile with -ffp-contract=off. Output lanes are independent; each retains
 * ascending j order, positive-zero initialization and a final bias addition.
 * RVV is opt-in pending performance evidence; weights remain row-major. */
static inline void pf_linear(float *y,const float *x,const float *w,const float *b,
                             int n,int in,int out) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int t=0;t<n;t++){
        const float *a=x+(size_t)t*in;
        float *dst=y+(size_t)t*out;
        int o=0;
#if defined(__riscv_vector) && defined(VLLM_PARAFORMER_LINEAR_RVV) && !defined(VLLM_PARAFORMER_FORCE_SCALAR)
        for(;o<out;){
            size_t vl=__riscv_vsetvl_e32m1((size_t)(out-o));
            vfloat32m1_t acc=__riscv_vfmv_v_f_f32m1(0.0f,vl);
            for(int j=0;j<in;j++){
                vfloat32m1_t value=__riscv_vlse32_v_f32m1(w+(size_t)o*in+j,(ptrdiff_t)in*sizeof(float),vl);
                vfloat32m1_t product=__riscv_vfmul_vf_f32m1(value,a[j],vl);
                acc=__riscv_vfadd_vv_f32m1(acc,product,vl);
            }
            vfloat32m1_t bias=b?__riscv_vle32_v_f32m1(b+o,vl):__riscv_vfmv_v_f_f32m1(0.0f,vl);
            acc=__riscv_vfadd_vv_f32m1(acc,bias,vl);
            __riscv_vse32_v_f32m1(dst+o,acc,vl);
            o+=(int)vl;
        }
#elif !defined(VLLM_PARAFORMER_FORCE_SCALAR)
        for(;o+3<out;o+=4){
            const float *v=w+(size_t)o*in;
            float s0=0.0f,s1=0.0f,s2=0.0f,s3=0.0f;
            for(int j=0;j<in;j++){
                float value=a[j];
                s0+=value*v[j];s1+=value*v[(size_t)in+j];
                s2+=value*v[(size_t)2*in+j];s3+=value*v[(size_t)3*in+j];
            }
            dst[o]=s0+(b?b[o]:0.0f);dst[o+1]=s1+(b?b[o+1]:0.0f);
            dst[o+2]=s2+(b?b[o+2]:0.0f);dst[o+3]=s3+(b?b[o+3]:0.0f);
        }
#endif
        for(;o<out;o++){
            const float *v=w+(size_t)o*in;
            float sum=0.0f;
            for(int j=0;j<in;j++)sum+=a[j]*v[j];
            dst[o]=sum+(b?b[o]:0.0f);
        }
    }
}
#endif
