/**
 * test_vexp.c - 校验 yolo_vexp_f32m4 的精度（V51）
 * 目的：区分「多项式 exp 本身不准」与「深网误差放大」。
 * 输出：在 [-40, 40] 上逐点与 libm expf 比较的相对误差统计。
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <riscv_vector.h>

static inline vfloat32m4_t yolo_vexp_f32m4(vfloat32m4_t x, size_t vl) {
    const float LOG2E = 1.4426950408889634f;
    const float LN2HI = 6.93145751953125e-1f;
    const float LN2LO = 1.4286067653301870e-6f;
    x = __riscv_vfmin_vv_f32m4(x, __riscv_vfmv_v_f_f32m4(88.0f, vl), vl);
    x = __riscv_vfmax_vv_f32m4(x, __riscv_vfmv_v_f_f32m4(-87.0f, vl), vl);
    vint32m4_t n = __riscv_vfcvt_x_f_v_i32m4(__riscv_vfmul_vf_f32m4(x, LOG2E, vl), vl);
    vfloat32m4_t nf = __riscv_vfcvt_f_x_v_f32m4(n, vl);
    vfloat32m4_t r = __riscv_vfnmsub_vf_f32m4(nf, LN2HI, x, vl);
    r = __riscv_vfnmsub_vf_f32m4(nf, LN2LO, r, vl);
    vfloat32m4_t p = __riscv_vfmv_v_f_f32m4(1.0f / 5040.0f, vl);
    p = __riscv_vfmadd_vv_f32m4(p, r, __riscv_vfmv_v_f_f32m4(1.0f / 720.0f, vl), vl);
    p = __riscv_vfmadd_vv_f32m4(p, r, __riscv_vfmv_v_f_f32m4(1.0f / 120.0f, vl), vl);
    p = __riscv_vfmadd_vv_f32m4(p, r, __riscv_vfmv_v_f_f32m4(1.0f / 24.0f, vl), vl);
    p = __riscv_vfmadd_vv_f32m4(p, r, __riscv_vfmv_v_f_f32m4(1.0f / 6.0f, vl), vl);
    p = __riscv_vfmadd_vv_f32m4(p, r, __riscv_vfmv_v_f_f32m4(0.5f, vl), vl);
    p = __riscv_vfmadd_vv_f32m4(p, r, __riscv_vfmv_v_f_f32m4(1.0f, vl), vl);
    p = __riscv_vfmadd_vv_f32m4(p, r, __riscv_vfmv_v_f_f32m4(1.0f, vl), vl);
    const vint32m4_t e = __riscv_vsll_vx_i32m4(
                             __riscv_vadd_vx_i32m4(n, 127, vl), 23, vl);
    return __riscv_vfmul_vv_f32m4(p, __riscv_vreinterpret_v_i32m4_f32m4(e), vl);
}

int main(void) {
    const size_t vmax = __riscv_vsetvlmax_e32m4();
    printf("VLEN/32*4 = %d lanes\n", (int)vmax);
    const int N = 200001;
    float *xs = malloc(N * sizeof(float)), *ys = malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) xs[i] = -40.0f + 80.0f * (float)i / (float)(N - 1);
    for (int i = 0; i < N; i += (int)vmax) {
        size_t vl = __riscv_vsetvl_e32m4((size_t)(N - i));
        vfloat32m4_t v = __riscv_vle32_v_f32m4(xs + i, vl);
        __riscv_vse32_v_f32m4(ys + i, yolo_vexp_f32m4(v, vl), vl);
    }
    double maxrel = 0, maxabs = 0;
    double x_at_maxrel = 0;
    for (int i = 0; i < N; i++) {
        double ref = exp((double)xs[i]);
        double d = fabs((double)ys[i] - ref);
        double rel = d / (ref > 1e-30 ? ref : 1e-30);
        if (rel > maxrel) { maxrel = rel; x_at_maxrel = xs[i]; }
        if (d > maxabs) maxabs = d;
    }
    printf("exp: max_rel_err=%.3e (at x=%.2f)  max_abs_err=%.3e\n", maxrel, x_at_maxrel, maxabs);

    /* 端到端 SiLU：t/(1+exp(-t))，只看 |t|<=20 的实际工作区间 */
    double srel = 0; double t_at = 0;
    for (int i = 0; i < N; i++) {
        double t = xs[i];
        if (fabs(t) > 20.0) continue;
        double num = (double)(-t);
        float ex = 0.0f;
        {
            size_t vl = __riscv_vsetvl_e32m4(1);
            vfloat32m4_t v = __riscv_vfmv_v_f_f32m4((float)num, vl);
            float o[4];
            __riscv_vse32_v_f32m4(o, yolo_vexp_f32m4(v, vl), vl);
            ex = o[0];
        }
        float sf = (float)t / (1.0f + ex);
        double sd = (double)t / (1.0 + exp(-(double)t));
        double rel = fabs((double)sf - sd) / (fabs(sd) > 1e-6 ? fabs(sd) : 1e-6);
        if (rel > srel) { srel = rel; t_at = t; }
    }
    printf("silu: max_rel_err=%.3e (at t=%.2f)  |t|<=20\n", srel, t_at);

    /* vfdiv.vv vs 标量 fdiv.s 的逐元素一致性：判断 RVV 向量除是否 1ulp 级
     * （若有偏差，就是「exp 精度提高但端到端误差不降」的另一来源）。 */
    {
        int nd = 100001;
        float *na = malloc(nd * sizeof(float)), *nb = malloc(nd * sizeof(float));
        float *qv = malloc(nd * sizeof(float));
        for (int i = 0; i < nd; i++) {
            na[i] = -20.0f + 40.0f * (float)i / (float)(nd - 1);
            nb[i] = 1.0f + (float)(i * 37 % 1000) * 0.001f + 1e-3f;
        }
        for (int i = 0; i < nd; i += (int)vmax) {
            size_t vl = __riscv_vsetvl_e32m4((size_t)(nd - i));
            vfloat32m4_t a = __riscv_vle32_v_f32m4(na + i, vl);
            vfloat32m4_t bb = __riscv_vle32_v_f32m4(nb + i, vl);
            __riscv_vse32_v_f32m4(qv + i, __riscv_vfdiv_vv_f32m4(a, bb, vl), vl);
        }
        long diff = 0; double mrel = 0;
        for (int i = 0; i < nd; i++) {
            float s = na[i] / nb[i];
            if (s != qv[i]) {
                diff++;
                double rel = fabs((double)qv[i] - (double)s) / fabs((double)s);
                if (rel > mrel) mrel = rel;
            }
        }
        printf("vfdiv.vv vs fdiv.s: diff_elems=%ld/%d  max_rel=%.3e\n", diff, nd, mrel);
        free(na); free(nb); free(qv);
    }
    free(xs); free(ys);
    return 0;
}
