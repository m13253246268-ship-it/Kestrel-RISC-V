/* bench_colbt.c — mode3 int8 GEMM 两种形态吞吐界定（L-IME-001 A-1b）
 *
 * S0 `lane=行`（生产 VLLM_Q4_INTB=1 形态）：col tile 布局，运行时 nibble 解包
 *    + vslideup 拼 32 lane + 8 token 复用 + facc 栈往返。
 * S1 `lane=token`（L-IME-001 §4 路线 A 正式形态）：权重已预解包为行主序 int8，
 *    激活走 vlse8 跨 token 载入，权重走标量 —— 即 V59 bench_ime_llm 能达到
 *    120 G MAC/s 的那条形态，此处补上 f16 scale + f32 epilogue。
 *
 * 数值口径：每 32 列一块 → int 累加 → 转 f32 → 乘 scale → f32 累加。
 * 合成数据：权重 int4 [-8,7]，激活 int8 [-100,99]；32 列内 |acc| ≤ 25600，i16 不溢出。
 * S0 乘法顺序 (s*acc)*xd，S1/S3 为 (xd*s)*acc，二者只差 f32 舍入。
 *
 * 板端构建：
 *   gcc -O3 -march=rv64gcv -mabi=lp64d -ffp-contract=off -fopenmp \
 *       bench_colbt.c -o bench_colbt -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <riscv_vector.h>

/* 与 vllm_safetensors.c:st_f16scale 同式（软件 f16→f32，X60 无 Zvfh） */
static inline uint32_t f16_to_f32_bits(uint16_t h) {
    uint32_t sign = (h >> 15) & 1, exp = (h >> 10) & 0x1F, mant = h & 0x3FF;
    if (exp == 0) {
        if (mant == 0) return sign << 31;
        uint32_t m = mant; int sh = 0;
        while (m < 0x400) { m <<= 1; sh++; }
        return (sign << 31) | ((113u - (uint32_t)sh) << 23) | ((m & 0x3FF) << 13);
    } else if (exp == 0x1F) {
        return (sign << 31) | (0xFFu << 23) | (mant << 13);
    }
    return (sign << 31) | ((exp + 112u) << 23) | (mant << 13);
}
static inline float f16f(const uint8_t *b) {
    uint16_t h; memcpy(&h, b, 2);
    uint32_t fb = f16_to_f32_bits(h); float d; memcpy(&d, &fb, 4); return d;
}
static inline uint16_t f32_to_f16_bits(float v) {
    uint32_t u; memcpy(&u, &v, 4);
    uint32_t sign = (u >> 16) & 0x8000u;
    int32_t e = (int32_t)((u >> 23) & 0xFF) - 127 + 15;
    uint32_t m = u & 0x7FFFFFu;
    if (e <= 0) return (uint16_t)sign;
    if (e >= 31) return (uint16_t)(sign | 0x7C00u);
    return (uint16_t)(sign | ((uint32_t)e << 10) | (m >> 13));
}

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* ---------------- S0：生产形态（lane=行，运行时解包，T=8） ---------------- */
static void s0_gemm(const uint8_t *w, const int8_t *xq_all, const float *xd_all,
                    int rows, int K, int P, float *out, int nt) {
    const int nblk = K / 32, ntiles = rows / 32;
    const int ngrp = (P + 7) / 8;
    const size_t vl = 32, vh = 16;
#pragma omp parallel for schedule(static) num_threads(nt)
    for (int it = 0; it < ntiles * ngrp; it++) {
        const int ti = it / ngrp, g = it % ngrp;
        const int t0 = g * 8;
        const int nt_ = (P - t0 < 8) ? (P - t0) : 8;
        const uint8_t *tile = w + (size_t)ti * (size_t)nblk * 576;
        const int8_t *xq[8]; const float *xd[8];
        for (int i = 0; i < 8; i++) {
            int t = t0 + (i < nt_ ? i : nt_ - 1);
            xq[i] = xq_all + (size_t)t * K;
            xd[i] = xd_all + (size_t)t * nblk;
        }
        float facc[8][32];
        vfloat32m4_t fz = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        for (int i = 0; i < 8; i++) __riscv_vse32_v_f32m4(facc[i], fz, vl);
        for (int b = 0; b < nblk; b++) {
            const uint8_t *sc = tile + (size_t)b * 576;
            const uint8_t *dt = sc + 64;
            float s[32];
            for (int k = 0; k < 16; k++) {
                s[k] = f16f(sc + (size_t)(2 * k) * 2);
                s[16 + k] = f16f(sc + (size_t)(2 * k + 1) * 2);
            }
            vfloat32m4_t vs = __riscv_vle32_v_f32m4(s, vl);
            vint16m2_t z0 = __riscv_vmv_v_x_i16m2(0, vl);
            vint16m2_t a0 = z0, a1 = z0, a2 = z0, a3 = z0,
                       a4 = z0, a5 = z0, a6 = z0, a7 = z0;
            const size_t xo = (size_t)b * 32;
            for (int cc = 0; cc < 32; cc++) {
                vuint8m1_t col = __riscv_vle8_v_u8m1(dt + (size_t)cc * 16, vh);
                vuint8m1_t lo = __riscv_vand_vx_u8m1(col, 0x0F, vh);
                vuint8m1_t hi = __riscv_vsrl_vx_u8m1(col, 4, vh);
                vuint8m1_t wv = __riscv_vslideup_vx_u8m1(lo, hi, 16, vl);
                vint8m1_t ws = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vsub_vx_u8m1(wv, 8, vl));
                a0 = __riscv_vwmacc_vx_i16m2(a0, xq[0][xo + cc], ws, vl);
                a1 = __riscv_vwmacc_vx_i16m2(a1, xq[1][xo + cc], ws, vl);
                a2 = __riscv_vwmacc_vx_i16m2(a2, xq[2][xo + cc], ws, vl);
                a3 = __riscv_vwmacc_vx_i16m2(a3, xq[3][xo + cc], ws, vl);
                a4 = __riscv_vwmacc_vx_i16m2(a4, xq[4][xo + cc], ws, vl);
                a5 = __riscv_vwmacc_vx_i16m2(a5, xq[5][xo + cc], ws, vl);
                a6 = __riscv_vwmacc_vx_i16m2(a6, xq[6][xo + cc], ws, vl);
                a7 = __riscv_vwmacc_vx_i16m2(a7, xq[7][xo + cc], ws, vl);
            }
#define S0_EPI(I, A) do { \
            vfloat32m4_t d_ = __riscv_vfwcvt_f_x_v_f32m4((A), vl); \
            vfloat32m4_t f_ = __riscv_vle32_v_f32m4(facc[I], vl); \
            f_ = __riscv_vfadd_vv_f32m4(f_, __riscv_vfmul_vf_f32m4( \
                     __riscv_vfmul_vv_f32m4(vs, d_, vl), xd[I][b], vl), vl); \
            __riscv_vse32_v_f32m4(facc[I], f_, vl); } while (0)
            S0_EPI(0, a0); S0_EPI(1, a1); S0_EPI(2, a2); S0_EPI(3, a3);
            S0_EPI(4, a4); S0_EPI(5, a5); S0_EPI(6, a6); S0_EPI(7, a7);
#undef S0_EPI
        }
        for (int i = 0; i < nt_; i++) {
            float *o = out + (size_t)(t0 + i) * rows + (size_t)ti * 32;
            for (int k = 0; k < 16; k++) {
                o[2 * k] = facc[i][k];
                o[2 * k + 1] = facc[i][16 + k];
            }
        }
    }
}

/* ---------------- S1：预解包 int8 + lane=token（RB 行/批） ---------------- */
#define S1_R2(M) M(0) M(1)
#define S1_R4(M) S1_R2(M) M(2) M(3)
#define S1_R8(M) S1_R4(M) M(4) M(5) M(6) M(7)

#define S1_DECLF(R) vfloat32m4_t f##R = fz;
#define S1_DECLA(R) vint32m4_t a##R = zi;
#define S1_DECLW(R) const int8_t *W##R = W + (size_t)(r0 + R) * K + (size_t)b * 32;
#define S1_MAC(R)   a##R = __riscv_vwmacc_vx_i32m4(a##R, (int16_t)W##R[cc], x16, vl);
#define S1_EPI(R)   f##R = __riscv_vfadd_vv_f32m4(f##R, __riscv_vfmul_vv_f32m4( \
                        __riscv_vfmul_vf_f32m4(dv, sc32[(size_t)(r0 + R) * nblk + b], vl), \
                        __riscv_vfcvt_f_x_v_f32m4(a##R, vl), vl), vl);
/* RVV strided 访存的步长单位是字节 */
#define S1_STORE(R) __riscv_vsse32_v_f32m4(out + (size_t)p0 * rows + (r0 + (R)), \
                                           (ptrdiff_t)(rows * sizeof(float)), f##R, vl);

#define GEN_S1(NAME, LIST, RB)                                                   \
static void NAME(const int8_t *W, const int8_t *A, const float *xd,              \
                 const float *sc32, int rows, int K, int P, float *out, int nt) {\
    const int nblk = K / 32, np = P / 32;                                        \
    const size_t vl = 32;                                                        \
    _Pragma("omp parallel for schedule(static) num_threads(nt)")                 \
    for (int it = 0; it < (rows / (RB)) * np; it++) {                            \
        const int r0 = (it / np) * (RB), p0 = (it % np) * 32;                    \
        vfloat32m4_t fz = __riscv_vfmv_v_f_f32m4(0.0f, vl);                      \
        LIST(S1_DECLF)                                                           \
        for (int b = 0; b < nblk; b++) {                                         \
            vint32m4_t zi = __riscv_vmv_v_x_i32m4(0, vl);                        \
            LIST(S1_DECLA)                                                       \
            const int8_t *Ab = A + (size_t)p0 * K + (size_t)b * 32;              \
            LIST(S1_DECLW)                                                       \
            for (int cc = 0; cc < 32; cc++) {                                    \
                vint8m1_t x8 = __riscv_vlse8_v_i8m1(Ab + cc, (ptrdiff_t)K, vl);  \
                vint16m2_t x16 = __riscv_vsext_vf2_i16m2(x8, vl);                \
                LIST(S1_MAC)                                                     \
            }                                                                    \
            vfloat32m4_t dv = __riscv_vlse32_v_f32m4(xd + (size_t)p0 * nblk + b,\
                                     (ptrdiff_t)(nblk * sizeof(float)), vl);     \
            LIST(S1_EPI)                                                         \
        }                                                                        \
        LIST(S1_STORE)                                                           \
    }                                                                            \
}

GEN_S1(s1_gemm4, S1_R4, 4)
GEN_S1(s1_gemm8, S1_R8, 8)

/* ---------------- 二分形态（RB=4 固定）----------------
 * 实际实例化只用 BLOCKED=1（!BLOCKED 分支未用；全 K 不归零形态见 s1pure_gemm）。
 * S2：每 32 列归零 + 逐块 f32 epilogue，scale/xd 取常数 1.0，不读内存
 *     → 与 S1pure 的差 = 「逐块 epilogue 断流水」的代价。
 * S3：S2 + 读 f32 scale 与 xd ＝ s1_gemm4 同形，用于校验宏。 */
#define S2_S3_BODY(BLOCKED, SCALED, NAME)                                        \
static void NAME(const int8_t *W, const int8_t *A, const float *xd,              \
                 const float *sc32, int rows, int K, int P, float *out, int nt) {\
    const int nblk = K / 32, np = P / 32;                                        \
    const size_t vl = 32;                                                        \
    _Pragma("omp parallel for schedule(static) num_threads(nt)")                 \
    for (int it = 0; it < (rows / 4) * np; it++) {                               \
        const int r0 = (it / np) * 4, p0 = (it % np) * 32;                        \
        vfloat32m4_t fz = __riscv_vfmv_v_f_f32m4(0.0f, vl);                       \
        vfloat32m4_t f0 = fz, f1 = fz, f2 = fz, f3 = fz;                          \
        const int8_t *Ab0 = A + (size_t)p0 * K;                                   \
        const int8_t *W0 = W + (size_t)(r0 + 0) * K;                              \
        const int8_t *W1 = W + (size_t)(r0 + 1) * K;                              \
        const int8_t *W2 = W + (size_t)(r0 + 2) * K;                              \
        const int8_t *W3 = W + (size_t)(r0 + 3) * K;                              \
        for (int b = 0; b < nblk; b++) {                                          \
            vint32m4_t zi = __riscv_vmv_v_x_i32m4(0, vl);                         \
            vint32m4_t a0 = zi, a1 = zi, a2 = zi, a3 = zi;                        \
            const int8_t *Ab = Ab0 + (size_t)b * 32;                              \
            for (int cc = 0; cc < 32; cc++) {                                     \
                vint8m1_t x8 = __riscv_vlse8_v_i8m1(Ab + cc, (ptrdiff_t)K, vl);   \
                vint16m2_t x16 = __riscv_vsext_vf2_i16m2(x8, vl);                 \
                a0 = __riscv_vwmacc_vx_i32m4(a0, (int16_t)W0[b * 32 + cc], x16, vl);\
                a1 = __riscv_vwmacc_vx_i32m4(a1, (int16_t)W1[b * 32 + cc], x16, vl);\
                a2 = __riscv_vwmacc_vx_i32m4(a2, (int16_t)W2[b * 32 + cc], x16, vl);\
                a3 = __riscv_vwmacc_vx_i32m4(a3, (int16_t)W3[b * 32 + cc], x16, vl);\
            }                                                                     \
            if (!(BLOCKED)) {   /* S2：直接累加 f32，不读 scale/xd */              \
                f0 = __riscv_vfadd_vv_f32m4(f0, __riscv_vfcvt_f_x_v_f32m4(a0, vl), vl);\
                f1 = __riscv_vfadd_vv_f32m4(f1, __riscv_vfcvt_f_x_v_f32m4(a1, vl), vl);\
                f2 = __riscv_vfadd_vv_f32m4(f2, __riscv_vfcvt_f_x_v_f32m4(a2, vl), vl);\
                f3 = __riscv_vfadd_vv_f32m4(f3, __riscv_vfcvt_f_x_v_f32m4(a3, vl), vl);\
            } else {                                                              \
                vfloat32m4_t dv = __riscv_vfmv_v_f_f32m4(1.0f, vl);                \
                vfloat32m4_t c1 = __riscv_vfmv_v_f_f32m4(1.0f, vl);                \
                if (SCALED) {                                                     \
                    dv = __riscv_vlse32_v_f32m4(xd + (size_t)p0 * nblk + b,        \
                                     (ptrdiff_t)(nblk * sizeof(float)), vl);      \
                }                                                                 \
                (void)c1;                                                         \
                float s0 = SCALED ? sc32[(size_t)(r0 + 0) * nblk + b] : 1.0f;      \
                float s1 = SCALED ? sc32[(size_t)(r0 + 1) * nblk + b] : 1.0f;      \
                float s2 = SCALED ? sc32[(size_t)(r0 + 2) * nblk + b] : 1.0f;      \
                float s3 = SCALED ? sc32[(size_t)(r0 + 3) * nblk + b] : 1.0f;      \
                f0 = __riscv_vfadd_vv_f32m4(f0, __riscv_vfmul_vv_f32m4(            \
                         __riscv_vfmul_vf_f32m4(dv, s0, vl),                      \
                         __riscv_vfcvt_f_x_v_f32m4(a0, vl), vl), vl);              \
                f1 = __riscv_vfadd_vv_f32m4(f1, __riscv_vfmul_vv_f32m4(            \
                         __riscv_vfmul_vf_f32m4(dv, s1, vl),                      \
                         __riscv_vfcvt_f_x_v_f32m4(a1, vl), vl), vl);              \
                f2 = __riscv_vfadd_vv_f32m4(f2, __riscv_vfmul_vv_f32m4(            \
                         __riscv_vfmul_vf_f32m4(dv, s2, vl),                      \
                         __riscv_vfcvt_f_x_v_f32m4(a2, vl), vl), vl);              \
                f3 = __riscv_vfadd_vv_f32m4(f3, __riscv_vfmul_vv_f32m4(            \
                         __riscv_vfmul_vf_f32m4(dv, s3, vl),                      \
                         __riscv_vfcvt_f_x_v_f32m4(a3, vl), vl), vl);              \
            }                                                                     \
        }                                                                         \
        const ptrdiff_t ost = (ptrdiff_t)(rows * sizeof(float));                  \
        __riscv_vsse32_v_f32m4(out + (size_t)p0 * rows + (r0 + 0), ost, f0, vl);   \
        __riscv_vsse32_v_f32m4(out + (size_t)p0 * rows + (r0 + 1), ost, f1, vl);   \
        __riscv_vsse32_v_f32m4(out + (size_t)p0 * rows + (r0 + 2), ost, f2, vl);   \
        __riscv_vsse32_v_f32m4(out + (size_t)p0 * rows + (r0 + 3), ost, f3, vl);   \
    }                                                                             \
}
S2_S3_BODY(1, 0, s2_gemm)   /* 逐块 f32 累加，scale/xd 常数 1（无内存读） */
S2_S3_BODY(1, 1, s3_gemm)   /* = s1_gemm4 同形，用于校验宏 */

/* S1pure：i32 累加贯穿全 K，不归零、不逐块 epilogue ＝ bench_ime_llm 的
 * gemm_rvv 同形。用来确认「本机这条形态确实能到 ~175 G/s」。 */
static void s1pure_gemm(const int8_t *W, const int8_t *A, const float *xd,
                        const float *sc32, int rows, int K, int P, float *out, int nt) {
    (void)xd; (void)sc32;
    const int np = P / 32;
    const size_t vl = 32;
#pragma omp parallel for schedule(static) num_threads(nt)
    for (int it = 0; it < (rows / 4) * np; it++) {
        const int r0 = (it / np) * 4, p0 = (it % np) * 32;
        vint32m4_t zi = __riscv_vmv_v_x_i32m4(0, vl);
        vint32m4_t a0 = zi, a1 = zi, a2 = zi, a3 = zi;
        const int8_t *Ab = A + (size_t)p0 * K;
        const int8_t *W0 = W + (size_t)(r0 + 0) * K;
        const int8_t *W1 = W + (size_t)(r0 + 1) * K;
        const int8_t *W2 = W + (size_t)(r0 + 2) * K;
        const int8_t *W3 = W + (size_t)(r0 + 3) * K;
        for (int cc = 0; cc < K; cc++) {
            vint8m1_t x8 = __riscv_vlse8_v_i8m1(Ab + cc, (ptrdiff_t)K, vl);
            vint16m2_t x16 = __riscv_vsext_vf2_i16m2(x8, vl);
            a0 = __riscv_vwmacc_vx_i32m4(a0, (int16_t)W0[cc], x16, vl);
            a1 = __riscv_vwmacc_vx_i32m4(a1, (int16_t)W1[cc], x16, vl);
            a2 = __riscv_vwmacc_vx_i32m4(a2, (int16_t)W2[cc], x16, vl);
            a3 = __riscv_vwmacc_vx_i32m4(a3, (int16_t)W3[cc], x16, vl);
        }
        const ptrdiff_t ost = (ptrdiff_t)(rows * sizeof(float));
        __riscv_vsse32_v_f32m4(out + (size_t)p0 * rows + (r0 + 0), ost,
                               __riscv_vfcvt_f_x_v_f32m4(a0, vl), vl);
        __riscv_vsse32_v_f32m4(out + (size_t)p0 * rows + (r0 + 1), ost,
                               __riscv_vfcvt_f_x_v_f32m4(a1, vl), vl);
        __riscv_vsse32_v_f32m4(out + (size_t)p0 * rows + (r0 + 2), ost,
                               __riscv_vfcvt_f_x_v_f32m4(a2, vl), vl);
        __riscv_vsse32_v_f32m4(out + (size_t)p0 * rows + (r0 + 3), ost,
                               __riscv_vfcvt_f_x_v_f32m4(a3, vl), vl);
    }
}

/* ---------------- 形状与数据 ---------------- */
typedef struct { const char *name; int rows, K; } shape_t;
static const shape_t SH[] = {
    { "qkv     (1152x896)", 1152,  896 },
    { "o       ( 896x896)",  896,  896 },
    { "gate_up (9728x896)", 9728,  896 },
    { "down    ( 896x4864)",  896, 4864 },
};

#define NREP 5

static void nan_fill(float *p, size_t n) {
    for (size_t i = 0; i < n; i++) p[i] = NAN;
}

typedef struct { double mx; long bad; } cmp_t;
static cmp_t cmp_out(const float *a, const float *b, size_t n) {
    cmp_t r = { 0.0, 0 };
    for (size_t i = 0; i < n; i++) {
        if (memcmp(&a[i], &b[i], sizeof(float)) == 0) continue;
        r.bad++;
        double d = fabs((double)a[i] - (double)b[i]);
        if (isnan(d)) d = INFINITY;
        if (d > r.mx) r.mx = d;
    }
    return r;
}

int main(int argc, char **argv) {
    int nt = (argc > 1) ? atoi(argv[1]) : 8;
    printf("[bench_colbt] threads=%d  warmup1 + min-of-%d  G MAC/s\n", nt, NREP);
    printf("%-20s %4s %8s %8s %8s %8s %8s %8s\n",
           "shape", "P", "S0", "S1pure", "S2", "S3", "S1rb4", "S1rb8");
    for (int P = 32; P <= 128; P *= 4) {
        for (size_t si = 0; si < sizeof(SH) / sizeof(SH[0]); si++) {
            const int rows = SH[si].rows, K = SH[si].K, nblk = K / 32;
            int8_t *W = malloc((size_t)rows * K);
            int8_t *A = malloc((size_t)P * K);
            float *xd = malloc((size_t)P * nblk * sizeof(float));
            float *sc32 = malloc((size_t)rows * nblk * sizeof(float));
            uint8_t *col = malloc((size_t)(rows / 32) * nblk * 576);
            float *o0 = malloc((size_t)P * rows * sizeof(float));
            float *o1 = malloc((size_t)P * rows * sizeof(float));
            if (!W || !A || !xd || !sc32 || !col || !o0 || !o1) { printf("OOM\n"); return 1; }
            for (size_t i = 0; i < (size_t)rows * K; i++) W[i] = (int8_t)((int)((i * 37u) % 16u) - 8);
            for (size_t i = 0; i < (size_t)P * K; i++)     A[i] = (int8_t)((int)((i * 91u) % 200u) - 100);
            for (size_t i = 0; i < (size_t)P * nblk; i++)  xd[i] = 0.003f + 1e-5f * (float)(i % 97);
            for (size_t i = 0; i < (size_t)rows * nblk; i++) {
                uint16_t h = f32_to_f16_bits(0.004f + 1e-5f * (float)(i % 89));
                sc32[i] = f16f((const uint8_t *)&h);   /* 统一按 f16 精度，S0/S1 同源 */
            }
            for (int ti = 0; ti < rows / 32; ti++)
                for (int b = 0; b < nblk; b++) {
                    uint8_t *t = col + ((size_t)ti * nblk + b) * 576;
                    for (int r = 0; r < 32; r++) {
                        uint16_t h = f32_to_f16_bits(sc32[(size_t)(ti * 32 + r) * nblk + b]);
                        memcpy(t + (size_t)r * 2, &h, 2);
                    }
                    for (int c = 0; c < 32; c++)
                        for (int r = 0; r < 32; r += 2) {
                            int v0 = W[(size_t)(ti * 32 + r) * K + b * 32 + c] + 8;
                            int v1 = W[(size_t)(ti * 32 + r + 1) * K + b * 32 + c] + 8;
                            t[64 + (size_t)c * 16 + (size_t)r / 2] = (uint8_t)((v1 << 4) | (v0 & 0x0F));
                        }
                }

            const size_t nout = (size_t)P * rows;
            double g0, gp, g2, g3, g4, g8;
#define BENCH(CALL, SINK) do {                                   \
            double best = 1e30;                                   \
            for (int rep = 0; rep <= NREP; rep++) {                \
                double t0 = now_s(); CALL; double t1 = now_s();    \
                if (rep > 0 && t1 - t0 < best) best = t1 - t0;     \
            }                                                     \
            SINK = (double)rows * K * P / best / 1e9;              \
        } while (0)
            BENCH(s0_gemm(col, A, xd, rows, K, P, o0, nt), g0);
            BENCH(s1pure_gemm(W, A, xd, sc32, rows, K, P, o1, nt), gp);
            BENCH(s2_gemm(W, A, xd, sc32, rows, K, P, o1, nt), g2);
            BENCH(s3_gemm(W, A, xd, sc32, rows, K, P, o1, nt), g3);
            BENCH(s1_gemm4(W, A, xd, sc32, rows, K, P, o1, nt), g4);
            BENCH(s1_gemm8(W, A, xd, sc32, rows, K, P, o1, nt), g8);
            printf("%-20s %4d %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f\n",
                   SH[si].name, P, g0, gp, g2, g3, g4, g8);
            /* 正确性：每臂先填 NaN 再单独跑。
             * 整数参考 ref[t][r] = Σ W*A；S1pure/S2(scale=1) 应与 ref 位级相等。
             * S3/S1rb4/S1rb8 与 S0 的乘法顺序不同（(xd*s)*acc vs (s*acc)*xd），
             * 只要求 max|d| 在 f32 舍入量级。 */
            int64_t *ref = malloc(nout * sizeof(int64_t));
            if (!ref) { printf("OOM\n"); return 1; }
            for (int tk = 0; tk < P; tk++)
                for (int r = 0; r < rows; r++) {
                    int64_t s = 0;
                    for (int k = 0; k < K; k++)
                        s += (int64_t)W[(size_t)r * K + k] * A[(size_t)tk * K + k];
                    ref[(size_t)tk * rows + r] = s;
                }
            nan_fill(o0, nout); s0_gemm(col, A, xd, rows, K, P, o0, nt);
            float amax = 0.0f;
            long s0nan = 0;
            for (size_t i = 0; i < nout; i++) {
                if (isnan(o0[i])) s0nan++;
                else if (fabsf(o0[i]) > amax) amax = fabsf(o0[i]);
            }
            long bp = 0, b2 = 0;
            nan_fill(o1, nout); s1pure_gemm(W, A, xd, sc32, rows, K, P, o1, nt);
            for (size_t i = 0; i < nout; i++) if (!(o1[i] == (float)ref[i])) bp++;
            nan_fill(o1, nout); s2_gemm(W, A, xd, sc32, rows, K, P, o1, nt);
            for (size_t i = 0; i < nout; i++) if (!(o1[i] == (float)ref[i])) b2++;
            nan_fill(o1, nout); s3_gemm(W, A, xd, sc32, rows, K, P, o1, nt);
            cmp_t c3 = cmp_out(o0, o1, nout);
            nan_fill(o1, nout); s1_gemm4(W, A, xd, sc32, rows, K, P, o1, nt);
            cmp_t c4 = cmp_out(o0, o1, nout);
            nan_fill(o1, nout); s1_gemm8(W, A, xd, sc32, rows, K, P, o1, nt);
            cmp_t c8 = cmp_out(o0, o1, nout);
            printf("   chk S0nan=%ld max|o|=%.3g | S1pure bad=%ld  S2 bad=%ld (vs int ref) | "
                   "vs S0: S3 %.3g/%ld  rb4 %.3g/%ld  rb8 %.3g/%ld (max|d|/bad)\n",
                   s0nan, (double)amax, bp, b2, c3.mx, c3.bad, c4.mx, c4.bad, c8.mx, c8.bad);
            free(ref);
            free(W); free(A); free(xd); free(sc32); free(col); free(o0); free(o1);
        }
    }
    return 0;
}
