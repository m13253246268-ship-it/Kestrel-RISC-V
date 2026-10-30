/**
 * bench_dotprod.c - YOLO int8 点积内核微基准（K-YOLO-002 V51 Phase 0）
 *
 * 目的：im2col 已从 101ms 降到 48.8ms，点积成为 115.7ms 的最大单项。
 *       在选方向之前先量清楚点积到底受什么限制，避免重犯 V45（分块）/
 *       V47（IME）/V42（列主序）三次「猜错方向」。
 *
 * 12 个变体（同一批真实层形状、同一份权重/激活）：
 *   纯内核（epilogue 只 vse32 存 int32，无归约）：
 *     m0 4 oc / 32 lane 行主序 vlse8（现状）
 *     m1 8 oc / 32 lane 行主序（测「读放大」假设）
 *     m2 4 oc / 64 lane（e8m2→i32m8，测「lane 数」假设）
 *     m3 4 oc / 32 lane k 主序 vle8（无跨步）
 *     m4 8 oc / 32 lane k 主序
 *     m7 8 oc / 16 lane（i32m2 累加器）
 *     m8 16 oc / 8 lane（i32m1 累加器）
 *     m9 4 oc / 16 lane（i32m2 累加器，m0 的 lane 减半对照）
 *   带 epilogue（反量化 + 可选 SiLU）：
 *     m5/m6 = m0 内核 + epilogue
 *     m10/m11 = m7 内核 + epilogue
 *
 * ⚠ V54 教训（务必先读）：m7 纯内核快于 m0（1.08~1.14×），但端到端反而慢
 *    5%~7%——epilogue 在 16 lane 下按「指令条数」计费（vfdiv/栈往返/标量 expf
 *    循环），lane 减半 = 调用数翻倍，把内核收益吃掉。故 **m7/m8/m9 的正收益
 *    不可直接外推到端到端**，只看纯内核口径会误判（详见指导书 V54）。
 *
 * 口径：纯内核口径绝对值低于端到端含 SiLU 的数字，只用于「同一进程内的相对比较」。
 *
 * m3/m4 另单列「行主序→k主序 转置」耗时（若转置吃掉收益则该方向不成立）。
 * V42 曾试列主序判负，但那次把「im2col 两遍法未并行」与「点积被 cache 打穿」
 * 两个问题混在一起；本次把它拆开单独量。
 *
 * 正确性：各变体的 int32 累加结果逐元素必须相同（int32 加法可重排）。
 *
 * 构建（板端）：gcc -O2 -march=rv64gcv -mabi=lp64d -fopenmp -DHAVE_RVV=1 \
 *                   bench_dotprod.c -o bench_dotprod -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif
#if HAVE_RVV
#include <riscv_vector.h>
#endif

typedef struct {
    int C, H, W, KH, KW, sh, sw, ph, pw, OC, OH, OW;
} YLayer;

#include "layers_320.h"

static double now_s(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

/* 确定性 xorshift32（避免线性同余低 8 位短周期导致测试数据规律性过强） */
static uint32_t rnd_state = 0x9E3779B9u;
static int8_t rnd_i8(void) {
    uint32_t x = rnd_state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    rnd_state = x;
    return (int8_t)((int)(x % 255u) - 127);
}

/* ---------- m0：现状（4 oc / 32 lane / 行主序 vlse8） ---------- */
static void dp_m0(const int8_t *W, int OC, int Kp, const int8_t *imc, int P, int32_t *acc) {
    const int nb = OC / 4;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(nb > 2)
#endif
    for (int blk = 0; blk < nb; blk++) {
        const int oc0 = blk * 4;
        const int8_t *w0 = W + (size_t)(oc0 + 0) * Kp;
        const int8_t *w1 = W + (size_t)(oc0 + 1) * Kp;
        const int8_t *w2 = W + (size_t)(oc0 + 2) * Kp;
        const int8_t *w3 = W + (size_t)(oc0 + 3) * Kp;
        int p0 = 0;
#if HAVE_RVV
        const size_t vmax = __riscv_vsetvlmax_e8m1();
        for (; p0 + (int)vmax <= P; p0 += (int)vmax) {
            vint32m4_t v0 = __riscv_vmv_v_x_i32m4(0, vmax);
            vint32m4_t v1 = v0, v2 = v0, v3 = v0;
            const int8_t *base = imc + (size_t)p0 * Kp;
            for (int k = 0; k < Kp; k++) {
                const vint8m1_t xv = __riscv_vlse8_v_i8m1(base + k, (ptrdiff_t)Kp, vmax);
                const vint16m2_t x16 = __riscv_vsext_vf2_i16m2(xv, vmax);
                v0 = __riscv_vwmacc_vx_i32m4(v0, (int16_t)w0[k], x16, vmax);
                v1 = __riscv_vwmacc_vx_i32m4(v1, (int16_t)w1[k], x16, vmax);
                v2 = __riscv_vwmacc_vx_i32m4(v2, (int16_t)w2[k], x16, vmax);
                v3 = __riscv_vwmacc_vx_i32m4(v3, (int16_t)w3[k], x16, vmax);
            }
            __riscv_vse32_v_i32m4(acc + (size_t)(oc0 + 0) * P + p0, v0, vmax);
            __riscv_vse32_v_i32m4(acc + (size_t)(oc0 + 1) * P + p0, v1, vmax);
            __riscv_vse32_v_i32m4(acc + (size_t)(oc0 + 2) * P + p0, v2, vmax);
            __riscv_vse32_v_i32m4(acc + (size_t)(oc0 + 3) * P + p0, v3, vmax);
        }
        for (; p0 < P; p0++) {   /* 尾部与生产同口径：标量 */
            for (int u = 0; u < 4; u++) {
                const int8_t *wr = W + (size_t)(oc0 + u) * Kp;
                const int8_t *ir = imc + (size_t)p0 * Kp;
                int32_t s = 0;
                for (int k = 0; k < Kp; k++) s += (int32_t)wr[k] * (int32_t)ir[k];
                acc[(size_t)(oc0 + u) * P + p0] = s;
            }
        }
#endif
    }
}

/* ---------- m1：8 oc 块 / 32 lane / 行主序 ---------- */
static void dp_m1(const int8_t *W, int OC, int Kp, const int8_t *imc, int P, int32_t *acc) {
    const int nb = OC / 8;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(nb > 2)
#endif
    for (int blk = 0; blk < nb; blk++) {
        const int oc0 = blk * 8;
        const int8_t *w = W + (size_t)oc0 * Kp;
#if HAVE_RVV
        const size_t vmax = __riscv_vsetvlmax_e8m1();
        int p0 = 0;
        for (; p0 + (int)vmax <= P; p0 += (int)vmax) {
            vint32m4_t v0 = __riscv_vmv_v_x_i32m4(0, vmax), v1 = v0, v2 = v0, v3 = v0;
            vint32m4_t v4 = v0, v5 = v0, v6 = v0, v7 = v0;
            const int8_t *base = imc + (size_t)p0 * Kp;
            for (int k = 0; k < Kp; k++) {
                const vint8m1_t xv = __riscv_vlse8_v_i8m1(base + k, (ptrdiff_t)Kp, vmax);
                const vint16m2_t x16 = __riscv_vsext_vf2_i16m2(xv, vmax);
                v0 = __riscv_vwmacc_vx_i32m4(v0, (int16_t)w[0 * Kp + k], x16, vmax);
                v1 = __riscv_vwmacc_vx_i32m4(v1, (int16_t)w[1 * Kp + k], x16, vmax);
                v2 = __riscv_vwmacc_vx_i32m4(v2, (int16_t)w[2 * Kp + k], x16, vmax);
                v3 = __riscv_vwmacc_vx_i32m4(v3, (int16_t)w[3 * Kp + k], x16, vmax);
                v4 = __riscv_vwmacc_vx_i32m4(v4, (int16_t)w[4 * Kp + k], x16, vmax);
                v5 = __riscv_vwmacc_vx_i32m4(v5, (int16_t)w[5 * Kp + k], x16, vmax);
                v6 = __riscv_vwmacc_vx_i32m4(v6, (int16_t)w[6 * Kp + k], x16, vmax);
                v7 = __riscv_vwmacc_vx_i32m4(v7, (int16_t)w[7 * Kp + k], x16, vmax);
            }
            __riscv_vse32_v_i32m4(acc + (size_t)(oc0 + 0) * P + p0, v0, vmax);
            __riscv_vse32_v_i32m4(acc + (size_t)(oc0 + 1) * P + p0, v1, vmax);
            __riscv_vse32_v_i32m4(acc + (size_t)(oc0 + 2) * P + p0, v2, vmax);
            __riscv_vse32_v_i32m4(acc + (size_t)(oc0 + 3) * P + p0, v3, vmax);
            __riscv_vse32_v_i32m4(acc + (size_t)(oc0 + 4) * P + p0, v4, vmax);
            __riscv_vse32_v_i32m4(acc + (size_t)(oc0 + 5) * P + p0, v5, vmax);
            __riscv_vse32_v_i32m4(acc + (size_t)(oc0 + 6) * P + p0, v6, vmax);
            __riscv_vse32_v_i32m4(acc + (size_t)(oc0 + 7) * P + p0, v7, vmax);
        }
        for (; p0 < P; p0++) {
            for (int u = 0; u < 8; u++) {
                const int8_t *wr = W + (size_t)(oc0 + u) * Kp;
                const int8_t *ir = imc + (size_t)p0 * Kp;
                int32_t s = 0;
                for (int k = 0; k < Kp; k++) s += (int32_t)wr[k] * (int32_t)ir[k];
                acc[(size_t)(oc0 + u) * P + p0] = s;
            }
        }
#endif
    }
}

/* ---------- m2：4 oc 块 / 64 lane（e8m2）/ 行主序 ---------- */
static void dp_m2(const int8_t *W, int OC, int Kp, const int8_t *imc, int P, int32_t *acc) {
    const int nb = OC / 4;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(nb > 2)
#endif
    for (int blk = 0; blk < nb; blk++) {
        const int oc0 = blk * 4;
        const int8_t *w0 = W + (size_t)(oc0 + 0) * Kp;
        const int8_t *w1 = W + (size_t)(oc0 + 1) * Kp;
        const int8_t *w2 = W + (size_t)(oc0 + 2) * Kp;
        const int8_t *w3 = W + (size_t)(oc0 + 3) * Kp;
#if HAVE_RVV
        const size_t vmax = __riscv_vsetvlmax_e8m2();
        int p0 = 0;
        for (; p0 + (int)vmax <= P; p0 += (int)vmax) {
            vint32m8_t v0 = __riscv_vmv_v_x_i32m8(0, vmax), v1 = v0, v2 = v0, v3 = v0;
            const int8_t *base = imc + (size_t)p0 * Kp;
            for (int k = 0; k < Kp; k++) {
                const vint8m2_t xv = __riscv_vlse8_v_i8m2(base + k, (ptrdiff_t)Kp, vmax);
                const vint16m4_t x16 = __riscv_vsext_vf2_i16m4(xv, vmax);
                v0 = __riscv_vwmacc_vx_i32m8(v0, (int16_t)w0[k], x16, vmax);
                v1 = __riscv_vwmacc_vx_i32m8(v1, (int16_t)w1[k], x16, vmax);
                v2 = __riscv_vwmacc_vx_i32m8(v2, (int16_t)w2[k], x16, vmax);
                v3 = __riscv_vwmacc_vx_i32m8(v3, (int16_t)w3[k], x16, vmax);
            }
            __riscv_vse32_v_i32m8(acc + (size_t)(oc0 + 0) * P + p0, v0, vmax);
            __riscv_vse32_v_i32m8(acc + (size_t)(oc0 + 1) * P + p0, v1, vmax);
            __riscv_vse32_v_i32m8(acc + (size_t)(oc0 + 2) * P + p0, v2, vmax);
            __riscv_vse32_v_i32m8(acc + (size_t)(oc0 + 3) * P + p0, v3, vmax);
        }
        for (; p0 < P; p0++) {
            for (int u = 0; u < 4; u++) {
                const int8_t *wr = W + (size_t)(oc0 + u) * Kp;
                const int8_t *ir = imc + (size_t)p0 * Kp;
                int32_t s = 0;
                for (int k = 0; k < Kp; k++) s += (int32_t)wr[k] * (int32_t)ir[k];
                acc[(size_t)(oc0 + u) * P + p0] = s;
            }
        }
#endif
    }
}

/* ---------- m3/m4：imc k 主序 imcT[Kp][P]，vle8 连续读 ----------
 * RVV 类型是 sizeless，不能放进数组 ⇒ 4/8 oc 两块各写一份（编译期常量）。 */
static void dp_k4(const int8_t *W, int OC, int Kp, const int8_t *imcT, int P, int32_t *acc) {
    const int nb = OC / 4;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(nb > 2)
#endif
    for (int blk = 0; blk < nb; blk++) {
        const int oc0 = blk * 4;
        const int8_t *w0 = W + (size_t)(oc0 + 0) * Kp;
        const int8_t *w1 = W + (size_t)(oc0 + 1) * Kp;
        const int8_t *w2 = W + (size_t)(oc0 + 2) * Kp;
        const int8_t *w3 = W + (size_t)(oc0 + 3) * Kp;
#if HAVE_RVV
        const size_t vmax = __riscv_vsetvlmax_e8m1();
        int p0 = 0;
        for (; p0 + (int)vmax <= P; p0 += (int)vmax) {
            vint32m4_t v0 = __riscv_vmv_v_x_i32m4(0, vmax), v1 = v0, v2 = v0, v3 = v0;
            for (int k = 0; k < Kp; k++) {
                const vint8m1_t xv = __riscv_vle8_v_i8m1(imcT + (size_t)k * P + p0, vmax);
                const vint16m2_t x16 = __riscv_vsext_vf2_i16m2(xv, vmax);
                v0 = __riscv_vwmacc_vx_i32m4(v0, (int16_t)w0[k], x16, vmax);
                v1 = __riscv_vwmacc_vx_i32m4(v1, (int16_t)w1[k], x16, vmax);
                v2 = __riscv_vwmacc_vx_i32m4(v2, (int16_t)w2[k], x16, vmax);
                v3 = __riscv_vwmacc_vx_i32m4(v3, (int16_t)w3[k], x16, vmax);
            }
            __riscv_vse32_v_i32m4(acc + (size_t)(oc0 + 0) * P + p0, v0, vmax);
            __riscv_vse32_v_i32m4(acc + (size_t)(oc0 + 1) * P + p0, v1, vmax);
            __riscv_vse32_v_i32m4(acc + (size_t)(oc0 + 2) * P + p0, v2, vmax);
            __riscv_vse32_v_i32m4(acc + (size_t)(oc0 + 3) * P + p0, v3, vmax);
        }
        for (; p0 < P; p0++) {
            for (int u = 0; u < 4; u++) {
                const int8_t *wr = W + (size_t)(oc0 + u) * Kp;
                int32_t s = 0;
                for (int k = 0; k < Kp; k++) s += (int32_t)wr[k] * (int32_t)imcT[(size_t)k * P + p0];
                acc[(size_t)(oc0 + u) * P + p0] = s;
            }
        }
#endif
    }
}

static void dp_k8(const int8_t *W, int OC, int Kp, const int8_t *imcT, int P, int32_t *acc) {
    const int nb = OC / 8;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(nb > 2)
#endif
    for (int blk = 0; blk < nb; blk++) {
        const int oc0 = blk * 8;
        const int8_t *w = W + (size_t)oc0 * Kp;
#if HAVE_RVV
        const size_t vmax = __riscv_vsetvlmax_e8m1();
        int p0 = 0;
        for (; p0 + (int)vmax <= P; p0 += (int)vmax) {
            vint32m4_t v0 = __riscv_vmv_v_x_i32m4(0, vmax), v1 = v0, v2 = v0, v3 = v0;
            vint32m4_t v4 = v0, v5 = v0, v6 = v0, v7 = v0;
            for (int k = 0; k < Kp; k++) {
                const vint8m1_t xv = __riscv_vle8_v_i8m1(imcT + (size_t)k * P + p0, vmax);
                const vint16m2_t x16 = __riscv_vsext_vf2_i16m2(xv, vmax);
                v0 = __riscv_vwmacc_vx_i32m4(v0, (int16_t)w[0 * Kp + k], x16, vmax);
                v1 = __riscv_vwmacc_vx_i32m4(v1, (int16_t)w[1 * Kp + k], x16, vmax);
                v2 = __riscv_vwmacc_vx_i32m4(v2, (int16_t)w[2 * Kp + k], x16, vmax);
                v3 = __riscv_vwmacc_vx_i32m4(v3, (int16_t)w[3 * Kp + k], x16, vmax);
                v4 = __riscv_vwmacc_vx_i32m4(v4, (int16_t)w[4 * Kp + k], x16, vmax);
                v5 = __riscv_vwmacc_vx_i32m4(v5, (int16_t)w[5 * Kp + k], x16, vmax);
                v6 = __riscv_vwmacc_vx_i32m4(v6, (int16_t)w[6 * Kp + k], x16, vmax);
                v7 = __riscv_vwmacc_vx_i32m4(v7, (int16_t)w[7 * Kp + k], x16, vmax);
            }
            __riscv_vse32_v_i32m4(acc + (size_t)(oc0 + 0) * P + p0, v0, vmax);
            __riscv_vse32_v_i32m4(acc + (size_t)(oc0 + 1) * P + p0, v1, vmax);
            __riscv_vse32_v_i32m4(acc + (size_t)(oc0 + 2) * P + p0, v2, vmax);
            __riscv_vse32_v_i32m4(acc + (size_t)(oc0 + 3) * P + p0, v3, vmax);
            __riscv_vse32_v_i32m4(acc + (size_t)(oc0 + 4) * P + p0, v4, vmax);
            __riscv_vse32_v_i32m4(acc + (size_t)(oc0 + 5) * P + p0, v5, vmax);
            __riscv_vse32_v_i32m4(acc + (size_t)(oc0 + 6) * P + p0, v6, vmax);
            __riscv_vse32_v_i32m4(acc + (size_t)(oc0 + 7) * P + p0, v7, vmax);
        }
        for (; p0 < P; p0++) {
            for (int u = 0; u < 8; u++) {
                const int8_t *wr = W + (size_t)(oc0 + u) * Kp;
                int32_t s = 0;
                for (int k = 0; k < Kp; k++) s += (int32_t)wr[k] * (int32_t)imcT[(size_t)k * P + p0];
                acc[(size_t)(oc0 + u) * P + p0] = s;
            }
        }
#endif
    }
}

/* ---------- m5：m0 内核 + 生产同款 epilogue（反量化 + 可选 SiLU） ----------
 * 用途：把「点积内核」与「epilogue（vfcvt/2×vfmul/vfadd + 标量 expf SiLU）」
 * 拆开，确认 V50 之后残余时间的真实归属。表达式与 vllm_yolo.c 的
 * YOLO_RVV_EPI 逐字相同（含运算顺序），故位级可对齐。 */
static void dp_m5(const int8_t *W, const float *cs, const float *b,
                  const int8_t *imc, const float *rowsc, int OC, int Kp, int P,
                  int act, float *y) {
    const int nb = OC / 4;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(nb > 2)
#endif
    for (int blk = 0; blk < nb; blk++) {
        const int oc0 = blk * 4;
        const int8_t *w0 = W + (size_t)(oc0 + 0) * Kp;
        const int8_t *w1 = W + (size_t)(oc0 + 1) * Kp;
        const int8_t *w2 = W + (size_t)(oc0 + 2) * Kp;
        const int8_t *w3 = W + (size_t)(oc0 + 3) * Kp;
        const float s0 = cs[oc0 + 0], s1 = cs[oc0 + 1];
        const float s2 = cs[oc0 + 2], s3 = cs[oc0 + 3];
        const float b0 = b ? b[oc0 + 0] : 0.f, b1 = b ? b[oc0 + 1] : 0.f;
        const float b2 = b ? b[oc0 + 2] : 0.f, b3 = b ? b[oc0 + 3] : 0.f;
        float *y0 = y + (size_t)(oc0 + 0) * P;
        float *y1 = y + (size_t)(oc0 + 1) * P;
        float *y2 = y + (size_t)(oc0 + 2) * P;
        float *y3 = y + (size_t)(oc0 + 3) * P;
#if HAVE_RVV
        const size_t vmax = __riscv_vsetvlmax_e8m1();
        int p0 = 0;
        for (; p0 + (int)vmax <= P; p0 += (int)vmax) {
            vint32m4_t a0 = __riscv_vmv_v_x_i32m4(0, vmax);
            vint32m4_t a1 = a0, a2 = a0, a3 = a0;
            const int8_t *base = imc + (size_t)p0 * Kp;
            for (int k = 0; k < Kp; k++) {
                const vint8m1_t xv = __riscv_vlse8_v_i8m1(base + k, (ptrdiff_t)Kp, vmax);
                const vint16m2_t x16 = __riscv_vsext_vf2_i16m2(xv, vmax);
                a0 = __riscv_vwmacc_vx_i32m4(a0, (int16_t)w0[k], x16, vmax);
                a1 = __riscv_vwmacc_vx_i32m4(a1, (int16_t)w1[k], x16, vmax);
                a2 = __riscv_vwmacc_vx_i32m4(a2, (int16_t)w2[k], x16, vmax);
                a3 = __riscv_vwmacc_vx_i32m4(a3, (int16_t)w3[k], x16, vmax);
            }
            const vfloat32m4_t rs = __riscv_vle32_v_f32m4(rowsc + p0, vmax);
#define BENCH_EPI(ACC, DSTP, SV, BV) do {                                     \
                vfloat32m4_t f_ = __riscv_vfmul_vv_f32m4(                     \
                                      __riscv_vfcvt_f_x_v_f32m4(ACC, vmax),   \
                                      __riscv_vfmul_vf_f32m4(rs, SV, vmax),   \
                                      vmax);                                  \
                f_ = __riscv_vfadd_vf_f32m4(f_, BV, vmax);                    \
                if (act) {                                                    \
                    float t_[vmax];                                           \
                    __riscv_vse32_v_f32m4(t_, f_, vmax);                      \
                    for (size_t i_ = 0; i_ < vmax; i_++)                      \
                        t_[i_] = t_[i_] / (1.f + expf(-t_[i_]));               \
                    f_ = __riscv_vle32_v_f32m4(t_, vmax);                      \
                }                                                             \
                __riscv_vse32_v_f32m4(DSTP, f_, vmax);                        \
            } while (0)
            BENCH_EPI(a0, y0 + p0, s0, b0);
            BENCH_EPI(a1, y1 + p0, s1, b1);
            BENCH_EPI(a2, y2 + p0, s2, b2);
            BENCH_EPI(a3, y3 + p0, s3, b3);
#undef BENCH_EPI
        }
        for (; p0 < P; p0++) {
            const int8_t *ir = imc + (size_t)p0 * Kp;
            float *dst[4] = { y0, y1, y2, y3 };
            const int8_t *wp[4] = { w0, w1, w2, w3 };
            const float sv[4] = { s0, s1, s2, s3 };
            const float bv[4] = { b0, b1, b2, b3 };
            for (int u = 0; u < 4; u++) {
                int32_t t = 0;
                for (int k = 0; k < Kp; k++) t += (int32_t)wp[u][k] * (int32_t)ir[k];
                float r = (float)t * (sv[u] * rowsc[p0]) + bv[u];
                if (act) r = r / (1.f + expf(-r));
                dst[u][p0] = r;
            }
        }
#endif
    }
}

/* ---------- m7：8 oc 块 / 16 lane（i32m2 累加器，规避 m1 的 8×i32m4 寄存器溢出） ----------
 * m1（8oc/32lane）慢 40% 是因为 8 个 i32m4 = 32 个向量寄存器顶满预算、再加
 * xv/x16 临时溢出。改用 i32m2（16 lane、2 寄存器/oc），8 oc 仅 16 寄存器。
 * 代价是 p 方向 lane 从 32 降到 16。
 * ⚠ 实测边界（V54）：本形态纯内核略快于 m0，但「收益来自多 oc 共享」这一因果
 *   未被证实（对照 m9 区间横跨 1.00），且端到端因 epilogue 反噬而变慢。 */
static void dp_m7(const int8_t *W, int OC, int Kp, const int8_t *imc, int P, int32_t *acc) {
    const int nb = OC / 8;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(nb > 2)
#endif
    for (int blk = 0; blk < nb; blk++) {
        const int oc0 = blk * 8;
        const int8_t *w = W + (size_t)oc0 * Kp;
#if HAVE_RVV
        const size_t vmax = __riscv_vsetvlmax_e8mf2();   /* 16 lane */
        int p0 = 0;
        for (; p0 + (int)vmax <= P; p0 += (int)vmax) {
            vint32m2_t v0 = __riscv_vmv_v_x_i32m2(0, vmax), v1 = v0, v2 = v0, v3 = v0;
            vint32m2_t v4 = v0, v5 = v0, v6 = v0, v7 = v0;
            const int8_t *base = imc + (size_t)p0 * Kp;
            for (int k = 0; k < Kp; k++) {
                const vint8mf2_t xv = __riscv_vlse8_v_i8mf2(base + k, (ptrdiff_t)Kp, vmax);
                const vint16m1_t x16 = __riscv_vsext_vf2_i16m1(xv, vmax);
                v0 = __riscv_vwmacc_vx_i32m2(v0, (int16_t)w[0 * Kp + k], x16, vmax);
                v1 = __riscv_vwmacc_vx_i32m2(v1, (int16_t)w[1 * Kp + k], x16, vmax);
                v2 = __riscv_vwmacc_vx_i32m2(v2, (int16_t)w[2 * Kp + k], x16, vmax);
                v3 = __riscv_vwmacc_vx_i32m2(v3, (int16_t)w[3 * Kp + k], x16, vmax);
                v4 = __riscv_vwmacc_vx_i32m2(v4, (int16_t)w[4 * Kp + k], x16, vmax);
                v5 = __riscv_vwmacc_vx_i32m2(v5, (int16_t)w[5 * Kp + k], x16, vmax);
                v6 = __riscv_vwmacc_vx_i32m2(v6, (int16_t)w[6 * Kp + k], x16, vmax);
                v7 = __riscv_vwmacc_vx_i32m2(v7, (int16_t)w[7 * Kp + k], x16, vmax);
            }
            __riscv_vse32_v_i32m2(acc + (size_t)(oc0 + 0) * P + p0, v0, vmax);
            __riscv_vse32_v_i32m2(acc + (size_t)(oc0 + 1) * P + p0, v1, vmax);
            __riscv_vse32_v_i32m2(acc + (size_t)(oc0 + 2) * P + p0, v2, vmax);
            __riscv_vse32_v_i32m2(acc + (size_t)(oc0 + 3) * P + p0, v3, vmax);
            __riscv_vse32_v_i32m2(acc + (size_t)(oc0 + 4) * P + p0, v4, vmax);
            __riscv_vse32_v_i32m2(acc + (size_t)(oc0 + 5) * P + p0, v5, vmax);
            __riscv_vse32_v_i32m2(acc + (size_t)(oc0 + 6) * P + p0, v6, vmax);
            __riscv_vse32_v_i32m2(acc + (size_t)(oc0 + 7) * P + p0, v7, vmax);
        }
        for (; p0 < P; p0++) {
            for (int u = 0; u < 8; u++) {
                const int8_t *wr = W + (size_t)(oc0 + u) * Kp;
                const int8_t *ir = imc + (size_t)p0 * Kp;
                int32_t s = 0;
                for (int k = 0; k < Kp; k++) s += (int32_t)wr[k] * (int32_t)ir[k];
                acc[(size_t)(oc0 + u) * P + p0] = s;
            }
        }
#endif
    }
}

/* ---------- m8：16 oc 块 / 8 lane（i32m1 累加器） ----------
 * 更激进的多 oc 共享：16 个 oc 共享一次 vlse8，但 p 方向只有 8 lane。 */
static void dp_m8(const int8_t *W, int OC, int Kp, const int8_t *imc, int P, int32_t *acc) {
    const int nb = OC / 16;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(nb > 2)
#endif
    for (int blk = 0; blk < nb; blk++) {
        const int oc0 = blk * 16;
        const int8_t *w = W + (size_t)oc0 * Kp;
#if HAVE_RVV
        const size_t vmax = __riscv_vsetvlmax_e8mf4();   /* 8 lane */
        int p0 = 0;
        for (; p0 + (int)vmax <= P; p0 += (int)vmax) {
            vint32m1_t v0 = __riscv_vmv_v_x_i32m1(0, vmax), v1 = v0, v2 = v0, v3 = v0;
            vint32m1_t v4 = v0, v5 = v0, v6 = v0, v7 = v0;
            vint32m1_t v8 = v0, v9 = v0, v10 = v0, v11 = v0;
            vint32m1_t v12 = v0, v13 = v0, v14 = v0, v15 = v0;
            const int8_t *base = imc + (size_t)p0 * Kp;
            for (int k = 0; k < Kp; k++) {
                const vint8mf4_t xv = __riscv_vlse8_v_i8mf4(base + k, (ptrdiff_t)Kp, vmax);
                const vint16mf2_t x16 = __riscv_vsext_vf2_i16mf2(xv, vmax);
                v0 = __riscv_vwmacc_vx_i32m1(v0, (int16_t)w[0 * Kp + k], x16, vmax);
                v1 = __riscv_vwmacc_vx_i32m1(v1, (int16_t)w[1 * Kp + k], x16, vmax);
                v2 = __riscv_vwmacc_vx_i32m1(v2, (int16_t)w[2 * Kp + k], x16, vmax);
                v3 = __riscv_vwmacc_vx_i32m1(v3, (int16_t)w[3 * Kp + k], x16, vmax);
                v4 = __riscv_vwmacc_vx_i32m1(v4, (int16_t)w[4 * Kp + k], x16, vmax);
                v5 = __riscv_vwmacc_vx_i32m1(v5, (int16_t)w[5 * Kp + k], x16, vmax);
                v6 = __riscv_vwmacc_vx_i32m1(v6, (int16_t)w[6 * Kp + k], x16, vmax);
                v7 = __riscv_vwmacc_vx_i32m1(v7, (int16_t)w[7 * Kp + k], x16, vmax);
                v8 = __riscv_vwmacc_vx_i32m1(v8, (int16_t)w[8 * Kp + k], x16, vmax);
                v9 = __riscv_vwmacc_vx_i32m1(v9, (int16_t)w[9 * Kp + k], x16, vmax);
                v10 = __riscv_vwmacc_vx_i32m1(v10, (int16_t)w[10 * Kp + k], x16, vmax);
                v11 = __riscv_vwmacc_vx_i32m1(v11, (int16_t)w[11 * Kp + k], x16, vmax);
                v12 = __riscv_vwmacc_vx_i32m1(v12, (int16_t)w[12 * Kp + k], x16, vmax);
                v13 = __riscv_vwmacc_vx_i32m1(v13, (int16_t)w[13 * Kp + k], x16, vmax);
                v14 = __riscv_vwmacc_vx_i32m1(v14, (int16_t)w[14 * Kp + k], x16, vmax);
                v15 = __riscv_vwmacc_vx_i32m1(v15, (int16_t)w[15 * Kp + k], x16, vmax);
            }
            __riscv_vse32_v_i32m1(acc + (size_t)(oc0 + 0) * P + p0, v0, vmax);
            __riscv_vse32_v_i32m1(acc + (size_t)(oc0 + 1) * P + p0, v1, vmax);
            __riscv_vse32_v_i32m1(acc + (size_t)(oc0 + 2) * P + p0, v2, vmax);
            __riscv_vse32_v_i32m1(acc + (size_t)(oc0 + 3) * P + p0, v3, vmax);
            __riscv_vse32_v_i32m1(acc + (size_t)(oc0 + 4) * P + p0, v4, vmax);
            __riscv_vse32_v_i32m1(acc + (size_t)(oc0 + 5) * P + p0, v5, vmax);
            __riscv_vse32_v_i32m1(acc + (size_t)(oc0 + 6) * P + p0, v6, vmax);
            __riscv_vse32_v_i32m1(acc + (size_t)(oc0 + 7) * P + p0, v7, vmax);
            __riscv_vse32_v_i32m1(acc + (size_t)(oc0 + 8) * P + p0, v8, vmax);
            __riscv_vse32_v_i32m1(acc + (size_t)(oc0 + 9) * P + p0, v9, vmax);
            __riscv_vse32_v_i32m1(acc + (size_t)(oc0 + 10) * P + p0, v10, vmax);
            __riscv_vse32_v_i32m1(acc + (size_t)(oc0 + 11) * P + p0, v11, vmax);
            __riscv_vse32_v_i32m1(acc + (size_t)(oc0 + 12) * P + p0, v12, vmax);
            __riscv_vse32_v_i32m1(acc + (size_t)(oc0 + 13) * P + p0, v13, vmax);
            __riscv_vse32_v_i32m1(acc + (size_t)(oc0 + 14) * P + p0, v14, vmax);
            __riscv_vse32_v_i32m1(acc + (size_t)(oc0 + 15) * P + p0, v15, vmax);
        }
        for (; p0 < P; p0++) {
            for (int u = 0; u < 16; u++) {
                const int8_t *wr = W + (size_t)(oc0 + u) * Kp;
                const int8_t *ir = imc + (size_t)p0 * Kp;
                int32_t s = 0;
                for (int k = 0; k < Kp; k++) s += (int32_t)wr[k] * (int32_t)ir[k];
                acc[(size_t)(oc0 + u) * P + p0] = s;
            }
        }
#endif
    }
}

/* ---------- m9：4 oc 块 / 16 lane（i32m2 累加器，m0 的 lane 减半对照） ----------
 * 用途：把「lane 减半」的纯代价单独量出来。
 * ⚠ V54 实测：本对照区间横跨 1.00（0.99~1.11×），故「m7 收益来自多 oc 共享
 *   而非 lane 数」这一因果**不成立**，只能确认 m7 纯内核略快的现象。 */
static void dp_m9(const int8_t *W, int OC, int Kp, const int8_t *imc, int P, int32_t *acc) {
    const int nb = OC / 4;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(nb > 2)
#endif
    for (int blk = 0; blk < nb; blk++) {
        const int oc0 = blk * 4;
        const int8_t *w0 = W + (size_t)(oc0 + 0) * Kp;
        const int8_t *w1 = W + (size_t)(oc0 + 1) * Kp;
        const int8_t *w2 = W + (size_t)(oc0 + 2) * Kp;
        const int8_t *w3 = W + (size_t)(oc0 + 3) * Kp;
#if HAVE_RVV
        const size_t vmax = __riscv_vsetvlmax_e8mf2();   /* 16 lane */
        int p0 = 0;
        for (; p0 + (int)vmax <= P; p0 += (int)vmax) {
            vint32m2_t v0 = __riscv_vmv_v_x_i32m2(0, vmax), v1 = v0, v2 = v0, v3 = v0;
            const int8_t *base = imc + (size_t)p0 * Kp;
            for (int k = 0; k < Kp; k++) {
                const vint8mf2_t xv = __riscv_vlse8_v_i8mf2(base + k, (ptrdiff_t)Kp, vmax);
                const vint16m1_t x16 = __riscv_vsext_vf2_i16m1(xv, vmax);
                v0 = __riscv_vwmacc_vx_i32m2(v0, (int16_t)w0[k], x16, vmax);
                v1 = __riscv_vwmacc_vx_i32m2(v1, (int16_t)w1[k], x16, vmax);
                v2 = __riscv_vwmacc_vx_i32m2(v2, (int16_t)w2[k], x16, vmax);
                v3 = __riscv_vwmacc_vx_i32m2(v3, (int16_t)w3[k], x16, vmax);
            }
            __riscv_vse32_v_i32m2(acc + (size_t)(oc0 + 0) * P + p0, v0, vmax);
            __riscv_vse32_v_i32m2(acc + (size_t)(oc0 + 1) * P + p0, v1, vmax);
            __riscv_vse32_v_i32m2(acc + (size_t)(oc0 + 2) * P + p0, v2, vmax);
            __riscv_vse32_v_i32m2(acc + (size_t)(oc0 + 3) * P + p0, v3, vmax);
        }
        for (; p0 < P; p0++) {
            for (int u = 0; u < 4; u++) {
                const int8_t *wr = W + (size_t)(oc0 + u) * Kp;
                const int8_t *ir = imc + (size_t)p0 * Kp;
                int32_t s = 0;
                for (int k = 0; k < Kp; k++) s += (int32_t)wr[k] * (int32_t)ir[k];
                acc[(size_t)(oc0 + u) * P + p0] = s;
            }
        }
#endif
    }
}

/* ---------- m10/m11：m7 内核 + 生产同款 epilogue ----------
 * 用途：V54 生产接入实测发现「m7 内核快但端到端反而慢」。这里把 m7 的 epilogue
 * 单独量出来，与 m5/m6（m0 的 epilogue）对照，判断 16-lane 是否让 epilogue 的
 * 单位指令开销翻倍。表达式与 V54 实验版 8oc/16lane epilogue 一致
 * （该实验版已从生产删除，现仅存于本基准）。 */
static void dp_m10(const int8_t *W, const float *cs, const float *b,
                   const int8_t *imc, const float *rowsc, int OC, int Kp, int P,
                   int act, float *y) {
    const int nb = OC / 8;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(nb > 2)
#endif
    for (int blk = 0; blk < nb; blk++) {
        const int oc0 = blk * 8;
        const int8_t *w = W + (size_t)oc0 * Kp;
        const float s0 = cs[oc0 + 0], s1 = cs[oc0 + 1], s2 = cs[oc0 + 2], s3 = cs[oc0 + 3];
        const float s4 = cs[oc0 + 4], s5 = cs[oc0 + 5], s6 = cs[oc0 + 6], s7 = cs[oc0 + 7];
        const float b0 = b ? b[oc0 + 0] : 0.f, b1 = b ? b[oc0 + 1] : 0.f;
        const float b2 = b ? b[oc0 + 2] : 0.f, b3 = b ? b[oc0 + 3] : 0.f;
        const float b4 = b ? b[oc0 + 4] : 0.f, b5 = b ? b[oc0 + 5] : 0.f;
        const float b6 = b ? b[oc0 + 6] : 0.f, b7 = b ? b[oc0 + 7] : 0.f;
        float *y0 = y + (size_t)(oc0 + 0) * P;
        float *y1 = y + (size_t)(oc0 + 1) * P;
        float *y2 = y + (size_t)(oc0 + 2) * P;
        float *y3 = y + (size_t)(oc0 + 3) * P;
        float *y4 = y + (size_t)(oc0 + 4) * P;
        float *y5 = y + (size_t)(oc0 + 5) * P;
        float *y6 = y + (size_t)(oc0 + 6) * P;
        float *y7 = y + (size_t)(oc0 + 7) * P;
#if HAVE_RVV
        const size_t vmax = __riscv_vsetvlmax_e8mf2();   /* 16 lane */
        float t_buf[64];
        int p0 = 0;
        for (; p0 + (int)vmax <= P; p0 += (int)vmax) {
            vint32m2_t a0 = __riscv_vmv_v_x_i32m2(0, vmax);
            vint32m2_t a1 = a0, a2 = a0, a3 = a0;
            vint32m2_t a4 = a0, a5 = a0, a6 = a0, a7 = a0;
            const int8_t *base = imc + (size_t)p0 * Kp;
            for (int k = 0; k < Kp; k++) {
                const vint8mf2_t xv = __riscv_vlse8_v_i8mf2(base + k, (ptrdiff_t)Kp, vmax);
                const vint16m1_t x16 = __riscv_vsext_vf2_i16m1(xv, vmax);
                a0 = __riscv_vwmacc_vx_i32m2(a0, (int16_t)w[0 * Kp + k], x16, vmax);
                a1 = __riscv_vwmacc_vx_i32m2(a1, (int16_t)w[1 * Kp + k], x16, vmax);
                a2 = __riscv_vwmacc_vx_i32m2(a2, (int16_t)w[2 * Kp + k], x16, vmax);
                a3 = __riscv_vwmacc_vx_i32m2(a3, (int16_t)w[3 * Kp + k], x16, vmax);
                a4 = __riscv_vwmacc_vx_i32m2(a4, (int16_t)w[4 * Kp + k], x16, vmax);
                a5 = __riscv_vwmacc_vx_i32m2(a5, (int16_t)w[5 * Kp + k], x16, vmax);
                a6 = __riscv_vwmacc_vx_i32m2(a6, (int16_t)w[6 * Kp + k], x16, vmax);
                a7 = __riscv_vwmacc_vx_i32m2(a7, (int16_t)w[7 * Kp + k], x16, vmax);
            }
            const vfloat32m2_t rs = __riscv_vle32_v_f32m2(rowsc + p0, vmax);
#define BENCH_EPI8(ACC, DSTP, SV, BV) do {                                    \
                vfloat32m2_t f_ = __riscv_vfmul_vv_f32m2(                     \
                                      __riscv_vfcvt_f_x_v_f32m2(ACC, vmax),   \
                                      __riscv_vfmul_vf_f32m2(rs, SV, vmax),   \
                                      vmax);                                  \
                f_ = __riscv_vfadd_vf_f32m2(f_, BV, vmax);                    \
                if (act) {                                                    \
                    __riscv_vse32_v_f32m2(t_buf, f_, vmax);                   \
                    for (size_t i_ = 0; i_ < vmax; i_++)                      \
                        t_buf[i_] = t_buf[i_] / (1.f + expf(-t_buf[i_]));     \
                    f_ = __riscv_vle32_v_f32m2(t_buf, vmax);                  \
                }                                                             \
                __riscv_vse32_v_f32m2(DSTP, f_, vmax);                        \
            } while (0)
            BENCH_EPI8(a0, y0 + p0, s0, b0);
            BENCH_EPI8(a1, y1 + p0, s1, b1);
            BENCH_EPI8(a2, y2 + p0, s2, b2);
            BENCH_EPI8(a3, y3 + p0, s3, b3);
            BENCH_EPI8(a4, y4 + p0, s4, b4);
            BENCH_EPI8(a5, y5 + p0, s5, b5);
            BENCH_EPI8(a6, y6 + p0, s6, b6);
            BENCH_EPI8(a7, y7 + p0, s7, b7);
#undef BENCH_EPI8
        }
        for (; p0 < P; p0++) {
            const int8_t *ir = imc + (size_t)p0 * Kp;
            float *dst[8] = { y0, y1, y2, y3, y4, y5, y6, y7 };
            const int8_t *wp[8] = { w + 0 * Kp, w + 1 * Kp, w + 2 * Kp, w + 3 * Kp,
                                    w + 4 * Kp, w + 5 * Kp, w + 6 * Kp, w + 7 * Kp };
            const float sv[8] = { s0, s1, s2, s3, s4, s5, s6, s7 };
            const float bv[8] = { b0, b1, b2, b3, b4, b5, b6, b7 };
            for (int u = 0; u < 8; u++) {
                int32_t t = 0;
                for (int k = 0; k < Kp; k++) t += (int32_t)wp[u][k] * (int32_t)ir[k];
                float r = (float)t * (sv[u] * rowsc[p0]) + bv[u];
                if (act) r = r / (1.f + expf(-r));
                dst[u][p0] = r;
            }
        }
#endif
    }
}

/* 行主序 imc[P][Kp] → k 主序 imcT[Kp][P]，64×64 字节分块 */
static void trans_kmajor(const int8_t *src, int8_t *dst, int P, int Kp) {
    const int TB = 64;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) collapse(2) if((long)P * Kp > 65536)
#endif
    for (int p0 = 0; p0 < P; p0 += TB) {
        for (int k0 = 0; k0 < Kp; k0 += TB) {
            const int p1 = (p0 + TB < P) ? p0 + TB : P;
            const int k1 = (k0 + TB < Kp) ? k0 + TB : Kp;
            for (int p = p0; p < p1; p++)
                for (int k = k0; k < k1; k++)
                    dst[(size_t)k * P + p] = src[(size_t)p * Kp + k];
        }
    }
}

int main(void) {
    int nth = 1;
#ifdef _OPENMP
    nth = omp_get_max_threads();
#endif
    printf("=== YOLO int8 点积内核微基准（yolov8n@320 真实层形状）threads=%d RVV=%d VLEN/8=%d ===\n",
           nth, HAVE_RVV, (int)(HAVE_RVV ? __riscv_vsetvlmax_e8m1() : 32));

    double t[12] = {0}, tt = 0;
    long long macs = 0, macs_m8 = 0;
    double imc_bytes_m0 = 0, imc_bytes_m1 = 0;
    int nbad = 0, nlayer = 0;

    for (int li = 0; li < K_NLAYERS; li++) {
        const YLayer *L = &k_layers[li];
        const int K = L->C * L->KH * L->KW;
        const int Kp = (K + 31) & ~31;
        const int P = L->OH * L->OW;
        const int OC = L->OC;
        if (OC % 8 != 0) continue;

        int8_t *W = (int8_t *)malloc((size_t)OC * Kp);
        int8_t *imc = (int8_t *)malloc((size_t)P * Kp);
        int8_t *imcT = (int8_t *)malloc((size_t)P * Kp);
        int32_t *ref = (int32_t *)malloc((size_t)OC * P * sizeof(int32_t));
        int32_t *out = (int32_t *)malloc((size_t)OC * P * sizeof(int32_t));
        if (!W || !imc || !imcT || !ref || !out) { printf("OOM\n"); return 1; }

        rnd_state = 0x9E3779B9u;
        for (size_t i = 0; i < (size_t)OC * Kp; i++)
            W[i] = rnd_i8();
        rnd_state = 0x7F4A7C15u;
        for (size_t i = 0; i < (size_t)P * Kp; i++)
            imc[i] = rnd_i8();
        trans_kmajor(imc, imcT, P, Kp);

        macs += (long long)OC * P * Kp;
        imc_bytes_m0 += (double)(OC / 4) * P * Kp;
        imc_bytes_m1 += (double)(OC / 8) * P * Kp;
        nlayer++;

        const int R = 3;
        double lo, d;
        const size_t nb_out = (size_t)OC * P * sizeof(int32_t);

        memset(ref, 0x5A, nb_out);
        lo = 1e9;
        for (int r = 0; r < R; r++) { double t0 = now_s(); dp_m0(W, OC, Kp, imc, P, ref); d = now_s() - t0; if (d < lo) lo = d; }
        t[0] += lo;

        memset(out, 0x5A, nb_out);
        lo = 1e9;
        for (int r = 0; r < R; r++) { double t0 = now_s(); dp_m1(W, OC, Kp, imc, P, out); d = now_s() - t0; if (d < lo) lo = d; }
        t[1] += lo;
        if (memcmp(ref, out, nb_out) != 0) { nbad++; printf("  L%02d m1 MISMATCH\n", li); }

        memset(out, 0x5A, nb_out);
        lo = 1e9;
        for (int r = 0; r < R; r++) { double t0 = now_s(); dp_m2(W, OC, Kp, imc, P, out); d = now_s() - t0; if (d < lo) lo = d; }
        t[2] += lo;
        if (memcmp(ref, out, nb_out) != 0) { nbad++; printf("  L%02d m2 MISMATCH\n", li); }

        lo = 1e9;
        for (int r = 0; r < R; r++) { double t0 = now_s(); trans_kmajor(imc, imcT, P, Kp); d = now_s() - t0; if (d < lo) lo = d; }
        tt += lo;

        memset(out, 0x5A, nb_out);
        lo = 1e9;
        for (int r = 0; r < R; r++) { double t0 = now_s(); dp_k4(W, OC, Kp, imcT, P, out); d = now_s() - t0; if (d < lo) lo = d; }
        t[3] += lo;
        if (memcmp(ref, out, nb_out) != 0) { nbad++; printf("  L%02d m3 MISMATCH\n", li); }

        memset(out, 0x5A, nb_out);
        lo = 1e9;
        for (int r = 0; r < R; r++) { double t0 = now_s(); dp_k8(W, OC, Kp, imcT, P, out); d = now_s() - t0; if (d < lo) lo = d; }
        t[4] += lo;
        if (memcmp(ref, out, nb_out) != 0) { nbad++; printf("  L%02d m4 MISMATCH\n", li); }

        /* ---- m5：内核 + epilogue（拆 dequant 与 SiLU） ---- */
        float *csa = (float *)malloc((size_t)OC * sizeof(float));
        float *ba  = (float *)malloc((size_t)OC * sizeof(float));
        float *rsc = (float *)malloc((size_t)P * sizeof(float));
        float *yy  = (float *)malloc((size_t)OC * P * sizeof(float));
        for (int i = 0; i < OC; i++) { csa[i] = 0.001f * (float)(i % 17 + 1); ba[i] = 0.01f * (float)(i % 7); }
        for (int i = 0; i < P; i++) rsc[i] = 0.002f * (float)(i % 13 + 1);
        memset(yy, 0, (size_t)OC * P * sizeof(float));
        lo = 1e9;
        for (int r = 0; r < R; r++) { double t0 = now_s(); dp_m5(W, csa, ba, imc, rsc, OC, Kp, P, 0, yy); d = now_s() - t0; if (d < lo) lo = d; }
        t[5] += lo;
        lo = 1e9;
        for (int r = 0; r < R; r++) { double t0 = now_s(); dp_m5(W, csa, ba, imc, rsc, OC, Kp, P, 1, yy); d = now_s() - t0; if (d < lo) lo = d; }
        t[6] += lo;
        free(csa); free(ba); free(rsc); free(yy);

        memset(out, 0x5A, nb_out);
        lo = 1e9;
        for (int r = 0; r < R; r++) { double t0 = now_s(); dp_m7(W, OC, Kp, imc, P, out); d = now_s() - t0; if (d < lo) lo = d; }
        t[7] += lo;
        if (memcmp(ref, out, nb_out) != 0) { nbad++; printf("  L%02d m7 MISMATCH\n", li); }

        /* m8 需 OC%16==0；yolov8n 全部层 OC 均为 16 倍数，实际不跳过。 */
        if (OC % 16 == 0) {
            memset(out, 0x5A, nb_out);
            lo = 1e9;
            for (int r = 0; r < R; r++) { double t0 = now_s(); dp_m8(W, OC, Kp, imc, P, out); d = now_s() - t0; if (d < lo) lo = d; }
            t[8] += lo;
            macs_m8 += (long long)OC * P * Kp;
            if (memcmp(ref, out, nb_out) != 0) { nbad++; printf("  L%02d m8 MISMATCH\n", li); }
        }

        memset(out, 0x5A, nb_out);
        lo = 1e9;
        for (int r = 0; r < R; r++) { double t0 = now_s(); dp_m9(W, OC, Kp, imc, P, out); d = now_s() - t0; if (d < lo) lo = d; }
        t[9] += lo;
        if (memcmp(ref, out, nb_out) != 0) { nbad++; printf("  L%02d m9 MISMATCH\n", li); }

        /* ---- m10/m11：m7 内核 + epilogue（拆 dequant 与 SiLU），与 m5/m6 对照 ---- */
        if (OC % 8 == 0) {
            float *csa2 = (float *)malloc((size_t)OC * sizeof(float));
            float *ba2  = (float *)malloc((size_t)OC * sizeof(float));
            float *rsc2 = (float *)malloc((size_t)P * sizeof(float));
            float *yy2  = (float *)malloc((size_t)OC * P * sizeof(float));
            for (int i = 0; i < OC; i++) { csa2[i] = 0.001f * (float)(i % 17 + 1); ba2[i] = 0.01f * (float)(i % 7); }
            for (int i = 0; i < P; i++) rsc2[i] = 0.002f * (float)(i % 13 + 1);
            memset(yy2, 0, (size_t)OC * P * sizeof(float));
            lo = 1e9;
            for (int r = 0; r < R; r++) { double t0 = now_s(); dp_m10(W, csa2, ba2, imc, rsc2, OC, Kp, P, 0, yy2); d = now_s() - t0; if (d < lo) lo = d; }
            t[10] += lo;
            lo = 1e9;
            for (int r = 0; r < R; r++) { double t0 = now_s(); dp_m10(W, csa2, ba2, imc, rsc2, OC, Kp, P, 1, yy2); d = now_s() - t0; if (d < lo) lo = d; }
            t[11] += lo;
            free(csa2); free(ba2); free(rsc2); free(yy2);
        }

        free(W); free(imc); free(imcT); free(ref); free(out);
    }

    const char *nm[12] = {
        "m0 4oc/32lane/row-major(vlse8)",
        "m1 8oc/32lane/row-major(vlse8)",
        "m2 4oc/64lane/row-major(vlse8)",
        "m3 4oc/32lane/k-major(vle8)",
        "m4 8oc/32lane/k-major(vle8)",
        "m5 = m0 + epilogue(仅反量化)",
        "m6 = m0 + epilogue(反量化+SiLU)",
        "m7 8oc/16lane/row-major(i32m2)",
        "m8 16oc/8lane/row-major(i32m1)",
        "m9 4oc/16lane/row-major(i32m2 对照)",
        "m10 = m7 + epilogue(仅反量化)",
        "m11 = m7 + epilogue(反量化+SiLU)"
    };
    printf("\n层数=%d  总 MAC=%.3f G  imc 重读字节：m0=%.1f MB  m1=%.1f MB\n",
           nlayer, (double)macs / 1e9, imc_bytes_m0 / 1e6, imc_bytes_m1 / 1e6);
    printf("%-36s %10s %14s %8s\n", "变体", "耗时(ms)", "吞吐(G MAC/s)", "vs m0");
    for (int i = 0; i < 12; i++) {
        /* m8 只覆盖 OC%16==0 的层，吞吐须用 macs_m8 而非全量 macs。 */
        long long denom = (i == 8) ? macs_m8 : macs;
        printf("%-36s %10.2f %14.2f %8.2fx\n", nm[i], t[i] * 1e3,
               (double)denom / t[i] / 1e9, t[0] / t[i]);
    }
    printf("%-36s %10.2f\n", "row->k-major 转置", tt * 1e3);
    printf("%-36s %10.2f  (含转置 vs m0 %.2fx)\n", "m3 + 转置", (t[3] + tt) * 1e3, t[0] / (t[3] + tt));
    printf("%-36s %10.2f  (含转置 vs m0 %.2fx)\n", "m4 + 转置", (t[4] + tt) * 1e3, t[0] / (t[4] + tt));
    printf("\n拆分：内核=%.2fms  反量化=%.2fms  SiLU=%.2fms  （epilogue 合计 %.0f%% of m6）\n",
           t[0] * 1e3, (t[5] - t[0]) * 1e3, (t[6] - t[5]) * 1e3,
           100.0 * (t[6] - t[0]) / t[6]);
    printf("正确性：%s（bad=%d）\n", nbad ? "FAIL" : "ALL MATCH", nbad);
    return nbad ? 2 : 0;
}
