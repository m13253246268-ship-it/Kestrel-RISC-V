/* bench_colime.c — L-IME-001 Phase B-0
 *
 * 问题：IME 的 45~68 G MAC/s 是「预置 int8、无 scale、无逐块 epilogue」的理想值。
 *       真实的 col 布局权重带 per-32-block f16 scale，epilogue 必须每 32 列做一次，
 *       这逼着 IME 微内核按 Kp=32 分块调用（固定开销占比升高）。
 *       本基准在**同一二进制、同一数据**下量出可落地形态的真实倍数。
 *
 * 三个臂（4 个真实形状，Qwen2.5-0.5B，24 层的单层几何）：
 *   S0      生产 colb（RVV int8，lane=行，T=8，8 线程）        —— 基线
 *   IME-full IME 4 线程，Kp 全量（896/4864），无 scale         —— 上限
 *   IME-blk  IME 4 线程，Kp=32 分块 + 真实 f16 scale + epilogue —— 可落地
 *
 * 数值口径与生产 colb 相同（每 32 列一块 → 整数累加 → (s*acc)*xd → f32 累加），
 * 合成数据：权重 int4 [-8,7]，激活 int8 [-100,99]；32 列内 |acc| ≤ 32*8*100=25600，
 * i16 不溢出。三臂乘法顺序都是 (s*acc)*xd、逐块 f32 累加，理论上位级一致。
 *
 * 板端构建：
 *   gcc -O2 -march=rv64gcv -mabi=lp64d -ffp-contract=off -fopenmp \
 *       bench_colime.c -o bench_colime -lm
 * 运行：./bench_colime [colb_threads]   （IME 固定 4 线程绑 hart0-3）
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>
#include <time.h>
#include <math.h>
#include <sched.h>
#include <omp.h>
#include <riscv_vector.h>

#define YIME_STR(x) #x
#define YIME_VMADOT(vd, vs1, vs2)                                                \
    ".insn 4, " YIME_STR(0x2b | ((vd) << 7) | (3 << 12) | ((vs1) << 15) |        \
                         ((vs2) << 20) | (0x71 << 25)) "\n\t"

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ---------------- f16（X60 无 Zvfh，与生产 st_f16scale 同式） ---------------- */
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

/* ---------------- IME：8(M) × 16(N) × kb_n × 8(K) ---------------- */
static void ime_pin(int h) {
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET(h & 3, &s);
    sched_setaffinity(0, sizeof(s), &s);
}

static void ime_kernel(const int8_t *a, const int8_t *b, int kb_n, int32_t acc[8][16]) {
    uintptr_t pa0 = (uintptr_t)a,          pa1 = (uintptr_t)(a + 32);
    uintptr_t pb0 = (uintptr_t)b,          pb1 = (uintptr_t)(b + 32);
    uintptr_t pb2 = (uintptr_t)(b + 64),   pb3 = (uintptr_t)(b + 96);
    uintptr_t po0 = (uintptr_t)&acc[0][0], po1 = (uintptr_t)&acc[4][0];
    __asm__ volatile(
        "vsetvli  t0, zero, e32, m2, tu, mu \n\t"
        "vmv.v.i  v8, 0                     \n\t"
        "vmv.v.i  v10, 0                    \n\t"
        "vmv.v.i  v12, 0                    \n\t"
        "vmv.v.i  v14, 0                    \n\t"
        "vmv.v.i  v16, 0                    \n\t"
        "vmv.v.i  v18, 0                    \n\t"
        "vmv.v.i  v20, 0                    \n\t"
        "vmv.v.i  v22, 0                    \n\t"
        "mv       t1, %[pa0]                \n\t"
        "mv       t2, %[pa1]                \n\t"
        "mv       t3, %[pb0]                \n\t"
        "mv       t4, %[pb1]                \n\t"
        "mv       t5, %[pb2]                \n\t"
        "mv       t6, %[pb3]                \n\t"
        "1%=:                               \n\t"
        "vsetvli  t0, zero, e8, m1, tu, mu  \n\t"
        "vle8.v   v0, (t1)                  \n\t"
        "vle8.v   v1, (t2)                  \n\t"
        "vle8.v   v2, (t3)                  \n\t"
        "vle8.v   v3, (t4)                  \n\t"
        "vle8.v   v4, (t5)                  \n\t"
        "vle8.v   v5, (t6)                  \n\t"
        YIME_VMADOT(8, 0, 2)
        YIME_VMADOT(10, 0, 3)
        YIME_VMADOT(12, 0, 4)
        YIME_VMADOT(14, 0, 5)
        YIME_VMADOT(16, 1, 2)
        YIME_VMADOT(18, 1, 3)
        YIME_VMADOT(20, 1, 4)
        YIME_VMADOT(22, 1, 5)
        "addi     t1, t1, 64                \n\t"
        "addi     t2, t2, 64                \n\t"
        "addi     t3, t3, 128               \n\t"
        "addi     t4, t4, 128               \n\t"
        "addi     t5, t5, 128               \n\t"
        "addi     t6, t6, 128               \n\t"
        "addi     %[k], %[k], -1            \n\t"
        "bnez     %[k], 1%=b                \n\t"
        "vsetvli  t0, zero, e32, m8, tu, mu \n\t"
        "vse32.v  v8, (%[po0])              \n\t"
        "vse32.v  v16, (%[po1])             \n\t"
        : [k] "+r"(kb_n)
        : [pa0] "r"(pa0), [pa1] "r"(pa1), [pb0] "r"(pb0), [pb1] "r"(pb1),
          [pb2] "r"(pb2), [pb3] "r"(pb3), [po0] "r"(po0), [po1] "r"(po1)
        : "t0", "t1", "t2", "t3", "t4", "t5", "t6",
          "v0", "v1", "v2", "v3", "v4", "v5",
          "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
          "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "memory");
}

/* A: W[OC][K] -> [mt][kb][mv][q][8]；B: A[P][K] -> [nb][kb][t][r][8] */
static int8_t *ime_pack_a(const int8_t *W, int OC, int Kp) {
    const int mt_n = OC / 8, kb_n = Kp / 8;
    int8_t *p = (int8_t *)calloc((size_t)mt_n * kb_n * 64, 1);
    if (!p) return NULL;
    for (int mt = 0; mt < mt_n; mt++)
        for (int kb = 0; kb < kb_n; kb++)
            for (int mv = 0; mv < 2; mv++)
                for (int q = 0; q < 4; q++)
                    memcpy(p + ((size_t)mt * kb_n + kb) * 64 + mv * 32 + q * 8,
                           W + (size_t)(mt * 8 + mv * 4 + q) * Kp + kb * 8, 8);
    return p;
}
static int8_t *ime_pack_b(const int8_t *A, int P, int Kp) {
    const int nb_n = P / 16, kb_n = Kp / 8;
    int8_t *p = (int8_t *)calloc((size_t)nb_n * kb_n * 128, 1);
    if (!p) return NULL;
    for (int nb = 0; nb < nb_n; nb++)
        for (int kb = 0; kb < kb_n; kb++)
            for (int t = 0; t < 4; t++)
                for (int r = 0; r < 4; r++)
                    memcpy(p + ((size_t)nb * kb_n + kb) * 128 + t * 32 + r * 8,
                           A + (size_t)(nb * 16 + t * 4 + r) * Kp + kb * 8, 8);
    return p;
}

/* ---------------- S0：生产 colb 形态（RVV int8，lane=行，T=8） ---------------- */
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

/* ---------------- IME 两臂 ---------------- */
/* IME-full：Kp 全量、无 scale、输出直接转 f32（上界） */
static void ime_full_gemm(const int8_t *apack, const int8_t *bpack,
                          int rows, int K, int P, float *out) {
    const int mt_n = rows / 8, nb_n = P / 16, kb_full = K / 8;
#pragma omp parallel num_threads(4)
    {
        cpu_set_t old;
        const int have_old = (sched_getaffinity(0, sizeof(old), &old) == 0);
        ime_pin(omp_get_thread_num());
#pragma omp for schedule(static)
        for (int it = 0; it < nb_n * mt_n; it++) {
            const int nb = it / mt_n, mt = it % mt_n;
            int32_t acc[8][16];
            ime_kernel(apack + (size_t)mt * kb_full * 64,
                       bpack + (size_t)nb * kb_full * 128, kb_full, acc);
            for (int mv = 0; mv < 2; mv++)
                for (int q = 0; q < 4; q++) {
                    const int oc = mt * 8 + mv * 4 + q;
                    for (int t = 0; t < 4; t++)
                        for (int j = 0; j < 4; j++)
                            out[(size_t)(nb * 16 + t * 4 + j) * rows + oc] =
                                (float)acc[mv * 4 + t][q * 4 + j];
                }
        }
        if (have_old) sched_setaffinity(0, sizeof(old), &old);
    }
}

/* IME-blk：Kp=32 分块调用 + 真实 f16 scale + 逐块 epilogue（可落地形态） */
static void ime_blk_gemm(const int8_t *apack, const int8_t *bpack,
                         const float *sc32, const float *xd,
                         int rows, int K, int P, float *out) {
    const int mt_n = rows / 8, nb_n = P / 16, kb_full = K / 8, nblk = K / 32;
#pragma omp parallel num_threads(4)
    {
        cpu_set_t old;
        const int have_old = (sched_getaffinity(0, sizeof(old), &old) == 0);
        ime_pin(omp_get_thread_num());
#pragma omp for schedule(static)
        for (int it = 0; it < nb_n * mt_n; it++) {
            const int nb = it / mt_n, mt = it % mt_n;
            float facc[16][8];
            memset(facc, 0, sizeof(facc));
            for (int b = 0; b < nblk; b++) {
                int32_t acc[8][16];
                ime_kernel(apack + ((size_t)mt * kb_full + (size_t)b * 4) * 64,
                           bpack + ((size_t)nb * kb_full + (size_t)b * 4) * 128, 4, acc);
                for (int mv = 0; mv < 2; mv++)
                    for (int q = 0; q < 4; q++) {
                        const int oc = mt * 8 + mv * 4 + q;
                        const float s = sc32[(size_t)oc * nblk + b];
                        for (int t = 0; t < 4; t++) {
                            for (int j = 0; j < 4; j++) {
                                const int tok = nb * 16 + t * 4 + j;
                                const float d = xd[(size_t)tok * nblk + b];
                                facc[t * 4 + j][mv * 4 + q] +=
                                    (s * (float)acc[mv * 4 + t][q * 4 + j]) * d;
                            }
                        }
                    }
            }
            for (int mv = 0; mv < 2; mv++)
                for (int q = 0; q < 4; q++) {
                    const int oc = mt * 8 + mv * 4 + q;
                    for (int t = 0; t < 4; t++)
                        for (int j = 0; j < 4; j++)
                            out[(size_t)(nb * 16 + t * 4 + j) * rows + oc] =
                                facc[t * 4 + j][mv * 4 + q];
                }
        }
        if (have_old) sched_setaffinity(0, sizeof(old), &old);
    }
}

/* IME-blk2：同上，但 epilogue 改成「标量转置 128 个 int32 + 8 条 16-lane 向量 FMA」。
 * 用来把「Kp=32 分块的固定开销」与「标量 epilogue 的开销」分开。 */
static void ime_blk2_gemm(const int8_t *apack, const int8_t *bpack,
                          const float *sc32, const float *xd,
                          int rows, int K, int P, float *out) {
    const int mt_n = rows / 8, nb_n = P / 16, kb_full = K / 8, nblk = K / 32;
    const size_t vl = 16;
#pragma omp parallel num_threads(4)
    {
        cpu_set_t old;
        const int have_old = (sched_getaffinity(0, sizeof(old), &old) == 0);
        ime_pin(omp_get_thread_num());
#pragma omp for schedule(static)
        for (int it = 0; it < nb_n * mt_n; it++) {
            const int nb = it / mt_n, mt = it % mt_n;
            vfloat32m2_t F0 = __riscv_vfmv_v_f_f32m2(0.0f, vl), F1 = F0, F2 = F0, F3 = F0,
                          F4 = F0, F5 = F0, F6 = F0, F7 = F0;
            for (int b = 0; b < nblk; b++) {
                int32_t acc[8][16];
                ime_kernel(apack + ((size_t)mt * kb_full + (size_t)b * 4) * 64,
                           bpack + ((size_t)nb * kb_full + (size_t)b * 4) * 128, 4, acc);
                /* acc[mv*4+t][q*4+j] -> tr[t*4+j][mv*4+q]（16 token × 8 oc） */
                int32_t tr[16][8];
                for (int mv = 0; mv < 2; mv++)
                    for (int q = 0; q < 4; q++)
                        for (int t = 0; t < 4; t++)
                            for (int j = 0; j < 4; j++)
                                tr[t * 4 + j][mv * 4 + q] = acc[mv * 4 + t][q * 4 + j];
                /* RVV strided 访存的步长单位是字节 */
                vfloat32m2_t d = __riscv_vlse32_v_f32m2(
                    xd + (size_t)(nb * 16) * nblk + b, (ptrdiff_t)(nblk * sizeof(float)), vl);
#define BLK2_ONE(F, OC) do {                                                  \
                    vint32m2_t iv = __riscv_vlse32_v_i32m2(&tr[0][OC],        \
                                        (ptrdiff_t)(8 * sizeof(int32_t)), vl); \
                    vfloat32m2_t s0 = __riscv_vfmul_vf_f32m2(                 \
                        __riscv_vfcvt_f_x_v_f32m2(iv, vl),                     \
                        sc32[(size_t)(mt * 8 + (OC)) * nblk + b], vl);         \
                    F = __riscv_vfadd_vv_f32m2(F,                             \
                            __riscv_vfmul_vv_f32m2(s0, d, vl), vl);            \
                } while (0);
                BLK2_ONE(F0, 0) BLK2_ONE(F1, 1) BLK2_ONE(F2, 2) BLK2_ONE(F3, 3)
                BLK2_ONE(F4, 4) BLK2_ONE(F5, 5) BLK2_ONE(F6, 6) BLK2_ONE(F7, 7)
#undef BLK2_ONE
            }
            {
                float *ob = out + (size_t)(nb * 16) * rows + (size_t)mt * 8;
                const ptrdiff_t ost = (ptrdiff_t)(rows * sizeof(float));
                __riscv_vsse32_v_f32m2(ob + 0, ost, F0, vl);
                __riscv_vsse32_v_f32m2(ob + 1, ost, F1, vl);
                __riscv_vsse32_v_f32m2(ob + 2, ost, F2, vl);
                __riscv_vsse32_v_f32m2(ob + 3, ost, F3, vl);
                __riscv_vsse32_v_f32m2(ob + 4, ost, F4, vl);
                __riscv_vsse32_v_f32m2(ob + 5, ost, F5, vl);
                __riscv_vsse32_v_f32m2(ob + 6, ost, F6, vl);
                __riscv_vsse32_v_f32m2(ob + 7, ost, F7, vl);
            }
        }
        if (have_old) sched_setaffinity(0, sizeof(old), &old);
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

/* bad = 不位级相等的元素数（NaN 计入）；mx = 有限差值的最大绝对值 */
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
    int nt0 = (argc > 1) ? atoi(argv[1]) : 8;
    printf("[bench_colime] S0(RVV)=%dt  IME=4t(hart0-3)  warmup1 + min-of-%d\n", nt0, NREP);
    printf("%-20s %4s %9s %9s %9s %9s %8s %8s\n",
           "shape", "P", "S0", "IMEfull", "IMEblk", "IMEblk2", "blk2/S0", "full/S0");
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
                sc32[i] = f16f((const uint8_t *)&h);
            }
            /* 与生产一致的 col tile（layout=3） */
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
            /* 模拟「转换期从 col tile 直接打包成 IME 布局」 */
            double tp = now_s();
            int8_t *ap = ime_pack_a(W, rows, K);
            int8_t *bp = ime_pack_b(A, P, K);
            tp = now_s() - tp;
            if (!ap || !bp) { printf("OOM pack\n"); return 1; }

            const size_t nout = (size_t)P * rows;
            const double macs = (double)rows * K * P;
            /* 计时：每臂 warm-up 1 次 + min-of-NREP，并记录 max 看离散 */
            double bmin[4], bmax[4];
            for (int arm = 0; arm < 4; arm++) {
                bmin[arm] = 1e30; bmax[arm] = 0;
                for (int rep = 0; rep <= NREP; rep++) {
                    double t = now_s();
                    if (arm == 0)      s0_gemm(col, A, xd, rows, K, P, o1, nt0);
                    else if (arm == 1) ime_full_gemm(ap, bp, rows, K, P, o1);
                    else if (arm == 2) ime_blk_gemm(ap, bp, sc32, xd, rows, K, P, o1);
                    else               ime_blk2_gemm(ap, bp, sc32, xd, rows, K, P, o1);
                    t = now_s() - t;
                    if (rep == 0) continue;
                    if (t < bmin[arm]) bmin[arm] = t;
                    if (t > bmax[arm]) bmax[arm] = t;
                }
            }
            /* 正确性：每臂先填 NaN 再单独跑，避免残留值掩盖漏写 */
            nan_fill(o0, nout); s0_gemm(col, A, xd, rows, K, P, o0, nt0);
            nan_fill(o1, nout); ime_blk_gemm(ap, bp, sc32, xd, rows, K, P, o1);
            cmp_t cb = cmp_out(o0, o1, nout);
            nan_fill(o1, nout); ime_blk2_gemm(ap, bp, sc32, xd, rows, K, P, o1);
            cmp_t cb2 = cmp_out(o0, o1, nout);
            /* IME-full 无 scale：与整数参考（抽样 oc）比，要求精确相等 */
            nan_fill(o1, nout); ime_full_gemm(ap, bp, rows, K, P, o1);
            long fbad = 0;
            for (int oc = 0; oc < rows; oc += 7)
                for (int tk = 0; tk < P; tk++) {
                    int64_t s = 0;
                    for (int k = 0; k < K; k++)
                        s += (int64_t)W[(size_t)oc * K + k] * A[(size_t)tk * K + k];
                    if (!(o1[(size_t)tk * rows + oc] == (float)s)) fbad++;
                }
            float amax = 0.0f;
            for (size_t i = 0; i < nout; i++) { float v = fabsf(o0[i]); if (v > amax) amax = v; }
            double g[4];
            for (int arm = 0; arm < 4; arm++) g[arm] = macs / bmin[arm] / 1e9;
            printf("%-20s %4d %9.2f %9.2f %9.2f %9.2f %7.2fx %7.2fx\n",
                   SH[si].name, P, g[0], g[1], g[2], g[3], g[3] / g[0], g[1] / g[0]);
            printf("   chk blk: max|d|=%.3g bad=%ld  blk2: max|d|=%.3g bad=%ld  full: bad=%ld  max|o|=%.3g\n",
                   cb.mx, cb.bad, cb2.mx, cb2.bad, fbad, (double)amax);
            printf("   spread(G/s min~max): S0 %.1f~%.1f  full %.1f~%.1f  blk %.1f~%.1f  blk2 %.1f~%.1f  pack=%.1fms\n",
                   macs / bmax[0] / 1e9, g[0], macs / bmax[1] / 1e9, g[1],
                   macs / bmax[2] / 1e9, g[2], macs / bmax[3] / 1e9, g[3], tp * 1e3);
            free(W); free(A); free(xd); free(sc32); free(col); free(o0); free(o1);
            free(ap); free(bp);
        }
    }
    return 0;
}
