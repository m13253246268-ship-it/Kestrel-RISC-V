/* bench_f32pre.c — lane f32 prefill：q4 运行时解包 vs f32 预解包（RVV 方案 V67）
 *
 * Q  ：复刻生产 rv_q4_lane_bt4_worker_m2（576B tile，vle32+xor+拆半字节+vfcvt，
 *      4 token 复用；外层 token 组、内层 parfor tile，= rv_q4_rows_par_bt 调度）。
 * Qt ：同 Q 内核，改 tile 外层并行、token 组内层（tile 留在缓存里复用）。
 * F  ：权重预解包为 f32（-8..7 精确），scale 预转 f32；内核只剩 vle32 +
 *      vfmul_vf×2 + vfadd×2，链序与 Q 完全相同 → 应与 Q 逐位一致。调度同 Q。
 * Ft ：F 内核 + tile 外层调度。
 *
 * 位级口径：s = s + (lo*xa + hi*xb)，块末 acc = acc + d*s，无 fma。
 * 每臂输出先填 NaN，再分别与标量参考逐位比较。
 *
 * 板端构建：
 *   gcc -O3 -march=rv64gcv -mabi=lp64d -ffp-contract=off -fopenmp \
 *       bench_f32pre.c -o bench_f32pre -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <riscv_vector.h>

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
static inline float f16f(uint16_t h) {
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

/* ---------------- 布局 ----------------
 * tile = 32 行，行 r = g*4 + m（g 0..7，m 0..3）。
 * Q 块 576B：[0,64) f16 scale，索引 m*8+g；[64,576) 数据，字节
 *   64 + (k*4+m)*32 + g*4 + j，低半字节 = 列 k*4+j，高半字节 = 列 16+k*4+j，
 *   存值 = (q+8)^8（内核 xor 0x88 还原 q+8）。
 * F 块 1056 float：[0,32) f32 scale，索引 h*16+l（l<8: m=2h, 否则 m=2h+1，g=l%8）；
 *   [32,1056) 权重 ((h*16 + k*4+j)*2 + {0:lo,1:hi})*16 + l。 */
#define QB 576
#define FB 1056

static inline __attribute__((always_inline))
vfloat32m2_t q_step(vuint32m2_t word, int j, float xa, float xb, vfloat32m2_t s, size_t vl) {
    vuint32m2_t b = __riscv_vsrl_vx_u32m2(word, 8 * j, vl);
    vint32m2_t lo = __riscv_vreinterpret_v_u32m2_i32m2(__riscv_vand_vx_u32m2(b, 0x0Fu, vl));
    vint32m2_t hi = __riscv_vreinterpret_v_u32m2_i32m2(
                        __riscv_vand_vx_u32m2(__riscv_vsrl_vx_u32m2(b, 4, vl), 0x0Fu, vl));
    vfloat32m2_t lof = __riscv_vfcvt_f_x_v_f32m2(__riscv_vsub_vx_i32m2(lo, 8, vl), vl);
    vfloat32m2_t hif = __riscv_vfcvt_f_x_v_f32m2(__riscv_vsub_vx_i32m2(hi, 8, vl), vl);
    vfloat32m2_t t = __riscv_vfadd_vv_f32m2(__riscv_vfmul_vf_f32m2(lof, xa, vl),
                                            __riscv_vfmul_vf_f32m2(hif, xb, vl), vl);
    return __riscv_vfadd_vv_f32m2(s, t, vl);
}

/* 与 rv_q4_scales_m2 同式；合成数据无零/次正规，bad 分支只作兜底 */
static inline __attribute__((always_inline))
vfloat32m2_t q_scales(const uint8_t *base) {
    const size_t v16 = 16;
    vuint16m1_t h = __riscv_vle16_v_u16m1((const uint16_t *)base, v16);
    vuint16m1_t ex16 = __riscv_vand_vx_u16m1(__riscv_vsrl_vx_u16m1(h, 10, v16), 0x1Fu, v16);
    vbool16_t bad = __riscv_vmor_mm_b16(__riscv_vmseq_vx_u16m1_b16(ex16, 0, v16),
                                        __riscv_vmseq_vx_u16m1_b16(ex16, 0x1Fu, v16), v16);
    if (__riscv_vfirst_m_b16(bad, v16) >= 0) {
        float sd[16];
        for (int i = 0; i < 16; i++) { uint16_t hh; __builtin_memcpy(&hh, base + 2 * i, 2); sd[i] = f16f(hh); }
        return __riscv_vle32_v_f32m2(sd, v16);
    }
    vuint32m2_t h32  = __riscv_vzext_vf2_u32m2(h, v16);
    vuint32m2_t sign = __riscv_vsll_vx_u32m2(__riscv_vand_vx_u32m2(h32, 0x8000u, v16), 16, v16);
    vuint32m2_t mant = __riscv_vsll_vx_u32m2(__riscv_vand_vx_u32m2(h32, 0x3FFu, v16), 13, v16);
    vuint32m2_t ex   = __riscv_vand_vx_u32m2(__riscv_vsrl_vx_u32m2(h32, 10, v16), 0x1Fu, v16);
    vuint32m2_t f = __riscv_vor_vv_u32m2(__riscv_vor_vv_u32m2(sign,
        __riscv_vsll_vx_u32m2(__riscv_vadd_vx_u32m2(ex, 112u, v16), 23, v16), v16), mant, v16);
    return __riscv_vreinterpret_v_u32m2_f32m2(f);
}

/* 存一个 m2 累加器：lane 0..7 → 行 g*4+m0，lane 8..15 → 行 g*4+m0+1（字节步长 16） */
static inline __attribute__((always_inline))
void st_acc(float *o, int m0, vfloat32m2_t a) {
    __riscv_vsse32_v_f32m1(o + m0,     16, __riscv_vget_v_f32m2_f32m1(a, 0), 8);
    __riscv_vsse32_v_f32m1(o + m0 + 1, 16, __riscv_vget_v_f32m2_f32m1(a, 1), 8);
}

/* Q：一个 tile × 4 token（= rv_q4_lane_bt4_worker_m2） */
static inline __attribute__((always_inline))
void q_tile4(const uint8_t *wg, int nblk, const float *const x[4], float *const o[4]) {
    const size_t vl = 16;
    vfloat32m2_t z = __riscv_vfmv_v_f_f32m2(0.0f, vl);
    for (int half = 0; half < 2; half++) {
        const int m0 = half * 2;
        vfloat32m2_t a0 = z, a1 = z, a2 = z, a3 = z;
        for (int b = 0; b < nblk; b++) {
            const uint8_t *tb = wg + (size_t)b * QB;
            vfloat32m2_t D = q_scales(tb + (size_t)m0 * 16);
            vfloat32m2_t s0 = z, s1 = z, s2 = z, s3 = z;
            const float *pa = x[0] + (size_t)b * 32, *pb = x[1] + (size_t)b * 32;
            const float *pc = x[2] + (size_t)b * 32, *pd = x[3] + (size_t)b * 32;
            for (int k = 0; k < 4; k++) {
                const uint8_t *base = tb + 64 + (size_t)(k * 4 + m0) * 32;
                vuint32m2_t wv = __riscv_vle32_v_u32m2((const uint32_t *)base, vl);
                wv = __riscv_vxor_vx_u32m2(wv, 0x88888888u, vl);
                for (int j = 0; j < 4; j++) {
                    const int ia = k * 4 + j, ib = 16 + k * 4 + j;
                    s0 = q_step(wv, j, pa[ia], pa[ib], s0, vl);
                    s1 = q_step(wv, j, pb[ia], pb[ib], s1, vl);
                    s2 = q_step(wv, j, pc[ia], pc[ib], s2, vl);
                    s3 = q_step(wv, j, pd[ia], pd[ib], s3, vl);
                }
            }
            a0 = __riscv_vfadd_vv_f32m2(a0, __riscv_vfmul_vv_f32m2(D, s0, vl), vl);
            a1 = __riscv_vfadd_vv_f32m2(a1, __riscv_vfmul_vv_f32m2(D, s1, vl), vl);
            a2 = __riscv_vfadd_vv_f32m2(a2, __riscv_vfmul_vv_f32m2(D, s2, vl), vl);
            a3 = __riscv_vfadd_vv_f32m2(a3, __riscv_vfmul_vv_f32m2(D, s3, vl), vl);
        }
        st_acc(o[0], m0, a0); st_acc(o[1], m0, a1);
        st_acc(o[2], m0, a2); st_acc(o[3], m0, a3);
    }
}

/* F：一个 tile × 4 token，f32 预解包权重 */
static inline __attribute__((always_inline))
void f_tile4(const float *wg, int nblk, const float *const x[4], float *const o[4]) {
    const size_t vl = 16;
    vfloat32m2_t z = __riscv_vfmv_v_f_f32m2(0.0f, vl);
    for (int half = 0; half < 2; half++) {
        const int m0 = half * 2;
        vfloat32m2_t a0 = z, a1 = z, a2 = z, a3 = z;
        for (int b = 0; b < nblk; b++) {
            const float *tb = wg + (size_t)b * FB;
            vfloat32m2_t D = __riscv_vle32_v_f32m2(tb + half * 16, vl);
            const float *wd = tb + 32 + (size_t)half * 16 * 32;
            vfloat32m2_t s0 = z, s1 = z, s2 = z, s3 = z;
            const float *pa = x[0] + (size_t)b * 32, *pb = x[1] + (size_t)b * 32;
            const float *pc = x[2] + (size_t)b * 32, *pd = x[3] + (size_t)b * 32;
            for (int kj = 0; kj < 16; kj++) {
                const int ia = kj, ib = 16 + kj;
                vfloat32m2_t lo = __riscv_vle32_v_f32m2(wd + (size_t)kj * 32, vl);
                vfloat32m2_t hi = __riscv_vle32_v_f32m2(wd + (size_t)kj * 32 + 16, vl);
#define F_STEP(S, P) S = __riscv_vfadd_vv_f32m2(S, __riscv_vfadd_vv_f32m2( \
                    __riscv_vfmul_vf_f32m2(lo, P[ia], vl), __riscv_vfmul_vf_f32m2(hi, P[ib], vl), vl), vl)
                F_STEP(s0, pa); F_STEP(s1, pb); F_STEP(s2, pc); F_STEP(s3, pd);
#undef F_STEP
            }
            a0 = __riscv_vfadd_vv_f32m2(a0, __riscv_vfmul_vv_f32m2(D, s0, vl), vl);
            a1 = __riscv_vfadd_vv_f32m2(a1, __riscv_vfmul_vv_f32m2(D, s1, vl), vl);
            a2 = __riscv_vfadd_vv_f32m2(a2, __riscv_vfmul_vv_f32m2(D, s2, vl), vl);
            a3 = __riscv_vfadd_vv_f32m2(a3, __riscv_vfmul_vv_f32m2(D, s3, vl), vl);
        }
        st_acc(o[0], m0, a0); st_acc(o[1], m0, a1);
        st_acc(o[2], m0, a2); st_acc(o[3], m0, a3);
    }
}

/* 调度。x: [P][K]，out: [P][rows]；P 为 4 的倍数 */
#define SETUP_XO(T0)                                                   \
    const float *xs[4]; float *os[4];                                  \
    for (int i = 0; i < 4; i++) {                                      \
        xs[i] = x + (size_t)((T0) + i) * K;                            \
        os[i] = out + (size_t)((T0) + i) * rows + (size_t)ti * 32;     \
    }

static void run_q(const uint8_t *w, const float *x, float *out, int rows, int K, int P, int nt) {
    const int nblk = K / 32, nti = rows / 32;
    for (int t0 = 0; t0 < P; t0 += 4) {
#pragma omp parallel for schedule(static) num_threads(nt)
        for (int ti = 0; ti < nti; ti++) {
            SETUP_XO(t0)
            q_tile4(w + (size_t)ti * nblk * QB, nblk, xs, os);
        }
    }
}
static void run_qt(const uint8_t *w, const float *x, float *out, int rows, int K, int P, int nt) {
    const int nblk = K / 32, nti = rows / 32;
#pragma omp parallel for schedule(static) num_threads(nt)
    for (int ti = 0; ti < nti; ti++)
        for (int t0 = 0; t0 < P; t0 += 4) {
            SETUP_XO(t0)
            q_tile4(w + (size_t)ti * nblk * QB, nblk, xs, os);
        }
}
static void run_f(const float *w, const float *x, float *out, int rows, int K, int P, int nt) {
    const int nblk = K / 32, nti = rows / 32;
    for (int t0 = 0; t0 < P; t0 += 4) {
#pragma omp parallel for schedule(static) num_threads(nt)
        for (int ti = 0; ti < nti; ti++) {
            SETUP_XO(t0)
            f_tile4(w + (size_t)ti * nblk * FB, nblk, xs, os);
        }
    }
}
static void run_ft(const float *w, const float *x, float *out, int rows, int K, int P, int nt) {
    const int nblk = K / 32, nti = rows / 32;
#pragma omp parallel for schedule(static) num_threads(nt)
    for (int ti = 0; ti < nti; ti++)
        for (int t0 = 0; t0 < P; t0 += 4) {
            SETUP_XO(t0)
            f_tile4(w + (size_t)ti * nblk * FB, nblk, xs, os);
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
#define NSH ((int)(sizeof(SH) / sizeof(SH[0])))
#define NREP 5

static uint32_t rng = 12345u;
static inline uint32_t lcg(void) { rng = rng * 1664525u + 1013904223u; return rng >> 8; }

static void nan_fill(float *p, size_t n) { for (size_t i = 0; i < n; i++) p[i] = NAN; }
static long bitdiff(const float *a, const float *b, size_t n) {
    long bad = 0;
    for (size_t i = 0; i < n; i++) if (memcmp(&a[i], &b[i], 4) != 0) bad++;
    return bad;
}

int main(int argc, char **argv) {
    int nt = (argc > 1) ? atoi(argv[1]) : 8;
    printf("[bench_f32pre] threads=%d  warmup1 + min-of-%d  G MAC/s（括号内为 ms）\n", nt, NREP);
    printf("%-20s %4s %16s %16s %16s %16s %7s\n", "shape", "P", "Q(prod)", "Qt", "F", "Ft", "best/Q");
    for (int P = 32; P <= 128; P *= 4) {
        double tsum[4] = { 0, 0, 0, 0 };
        for (int si = 0; si < NSH; si++) {
            const int rows = SH[si].rows, K = SH[si].K, nblk = K / 32, nti = rows / 32;
            int8_t *q = malloc((size_t)rows * K);
            uint16_t *sh = malloc((size_t)rows * nblk * 2);
            float *x = malloc((size_t)P * K * 4);
            uint8_t *wq = malloc((size_t)nti * nblk * QB);
            float *wf = malloc((size_t)nti * nblk * FB * 4);
            const size_t nout = (size_t)P * rows;
            float *ref = malloc(nout * 4), *o = malloc(nout * 4);
            if (!q || !sh || !x || !wq || !wf || !ref || !o) { printf("OOM\n"); return 1; }
            for (size_t i = 0; i < (size_t)rows * K; i++) q[i] = (int8_t)((int)(lcg() % 16u) - 8);
            for (size_t i = 0; i < (size_t)rows * nblk; i++)
                sh[i] = f32_to_f16_bits(0.002f + 0.01f * (float)(lcg() % 10000u) / 10000.0f);
            for (size_t i = 0; i < (size_t)P * K; i++)
                x[i] = ((float)(lcg() % 2000001u) - 1000000.0f) * 2e-6f;

            /* 打包 */
            for (int ti = 0; ti < nti; ti++)
                for (int b = 0; b < nblk; b++) {
                    uint8_t *tq = wq + ((size_t)ti * nblk + b) * QB;
                    float *tf = wf + ((size_t)ti * nblk + b) * FB;
                    for (int g = 0; g < 8; g++)
                        for (int m = 0; m < 4; m++) {
                            const int r = ti * 32 + g * 4 + m;
                            const int8_t *qr = q + (size_t)r * K + (size_t)b * 32;
                            uint16_t h = sh[(size_t)r * nblk + b];
                            memcpy(tq + (size_t)(m * 8 + g) * 2, &h, 2);
                            for (int k = 0; k < 4; k++)
                                for (int j = 0; j < 4; j++) {
                                    int lo = ((qr[k * 4 + j] + 8) ^ 8) & 0x0F;
                                    int hi = ((qr[16 + k * 4 + j] + 8) ^ 8) & 0x0F;
                                    tq[64 + (size_t)(k * 4 + m) * 32 + g * 4 + j] = (uint8_t)((hi << 4) | lo);
                                }
                            const int hf = m / 2, l = (m % 2) * 8 + g;
                            tf[hf * 16 + l] = f16f(h);
                            for (int kj = 0; kj < 16; kj++) {
                                tf[32 + ((size_t)(hf * 16 + kj) * 2 + 0) * 16 + l] = (float)qr[kj];
                                tf[32 + ((size_t)(hf * 16 + kj) * 2 + 1) * 16 + l] = (float)qr[16 + kj];
                            }
                        }
                }

            /* 标量参考：同链序，-ffp-contract=off */
#pragma omp parallel for schedule(static) num_threads(nt)
            for (int r = 0; r < rows; r++)
                for (int t = 0; t < P; t++) {
                    const float *xt = x + (size_t)t * K;
                    const int8_t *qr = q + (size_t)r * K;
                    float acc = 0.0f;
                    for (int b = 0; b < nblk; b++) {
                        float d = f16f(sh[(size_t)r * nblk + b]);
                        float s = 0.0f;
                        for (int kj = 0; kj < 16; kj++) {
                            const int c = b * 32 + kj;
                            s = s + ((float)qr[c] * xt[c] + (float)qr[c + 16] * xt[c + 16]);
                        }
                        acc = acc + d * s;
                    }
                    ref[(size_t)t * rows + r] = acc;
                }

            double best[4]; long bad[4];
#define BENCH(I, CALL) do {                                             \
            nan_fill(o, nout); CALL; bad[I] = bitdiff(o, ref, nout);     \
            double bt = 1e30;                                           \
            for (int rep = 0; rep <= NREP; rep++) {                      \
                double t0 = now_s(); CALL; double t1 = now_s();          \
                if (rep > 0 && t1 - t0 < bt) bt = t1 - t0;               \
            }                                                           \
            best[I] = bt; } while (0)
            BENCH(0, run_q (wq, x, o, rows, K, P, nt));
            BENCH(1, run_qt(wq, x, o, rows, K, P, nt));
            BENCH(2, run_f (wf, x, o, rows, K, P, nt));
            BENCH(3, run_ft(wf, x, o, rows, K, P, nt));
#undef BENCH
            const double mac = (double)rows * K * P;
            double bf = best[2] < best[3] ? best[2] : best[3];
            printf("%-20s %4d", SH[si].name, P);
            for (int i = 0; i < 4; i++) printf(" %7.2f(%7.2f)", mac / best[i] / 1e9, best[i] * 1e3);
            printf(" %6.2fx\n", best[0] / bf);
            printf("   chk bad(vs scalar ref, bitwise): Q=%ld Qt=%ld F=%ld Ft=%ld / %zu  wbytes Q=%zu F=%zu\n",
                   bad[0], bad[1], bad[2], bad[3], nout,
                   (size_t)nti * nblk * QB, (size_t)nti * nblk * FB * 4);
            for (int i = 0; i < 4; i++) tsum[i] += best[i];
            free(q); free(sh); free(x); free(wq); free(wf); free(ref); free(o);
        }
        double tf = tsum[2] < tsum[3] ? tsum[2] : tsum[3];
        printf("LAYER P=%d  sum ms: Q=%.2f Qt=%.2f F=%.2f Ft=%.2f  bestF/Q=%.2fx  Qt/Q=%.2fx\n",
               P, tsum[0] * 1e3, tsum[1] * 1e3, tsum[2] * 1e3, tsum[3] * 1e3,
               tsum[0] / tf, tsum[0] / tsum[1]);
    }
    return 0;
}
