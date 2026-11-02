/* bench_ime_q4.c — P1：Q4_0 GEMM，IME(vmadot) 寄存器内 per-block 收尾 vs colb(vwmacc)
 *
 * 口径：
 *   - 两侧输入相同：Q4 nibble 权重 + f16 行 scale（每 32K 一块），激活已量化为
 *     int8 xq + f32 块 scale xd（激活量化两侧相同，不计时）。
 *   - colb = rv_q4_colbt_worker 逐行复刻（32 行 tile × 8 token，scale 软件转换，
 *     facc 走内存），8 线程不绑核 / 4 线程绑 hart0-3 两档。
 *   - IME = 16 行 × 4 token 微内核，整段内联汇编：Q4 解包 → vmadot int32 块内累加 →
 *     块末 vfcvt、(s*d)*xd、vfadd 全在寄存器；scale 用 vle16+vfwcvt（zvfh）。
 *     4 线程绑 hart0-3。激活 4 token 交错打包计入 IME 时间（运行期开销）。
 *   - 浮点顺序与 colb 相同：F=+0；每块 F = F + ((s*d)*xd)；块内整数和精确
 *     （|sum|<=32*127*8=32512，int16/int32 均无溢出）。故要求逐位 0 失配。
 *   - 权重重排为转换期一次性成本，不计时。
 * 编译：gcc -O2 -march=rv64gcv_zvfh -mabi=lp64d -ffp-contract=off -fopenmp
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <sched.h>
#include <omp.h>
#include <riscv_vector.h>

#define YS(x) #x
#define VMADOT(vd, vs1, vs2) \
    ".insn 4, " YS(0x2b | ((vd) << 7) | (3 << 12) | ((vs1) << 15) | ((vs2) << 20) | (0x71 << 25)) "\n\t"

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static void pin(int h) { cpu_set_t s; CPU_ZERO(&s); CPU_SET(h, &s); sched_setaffinity(0, sizeof(s), &s); }

static uint32_t rs = 12345u;
static uint32_t rnd(void) { rs = rs * 1664525u + 1013904223u; return rs >> 8; }

/* 与引擎 f16_to_f32_bits 相同 */
static float f16f(uint16_t h) {
    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1F, m = h & 0x3FF, u;
    if (e == 0) {
        if (m == 0) u = s << 31;
        else { int sh = 0; while (m < 0x400) { m <<= 1; sh++; } u = (s << 31) | ((113u - sh) << 23) | ((m & 0x3FF) << 13); }
    } else if (e == 0x1F) u = (s << 31) | (0xFFu << 23) | (m << 13);
    else u = (s << 31) | ((e + 112) << 23) | (m << 13);
    float f; memcpy(&f, &u, 4); return f;
}

/* ---------------- colb 复刻 ---------------- */
#define CB_T 8
static void colb_worker(const uint8_t *w, const int8_t *xqa, const float *xda, float *ob,
                        int rows, int nblk, int nb, int ngrp, int it) {
    const int ti = it / ngrp, g = it % ngrp, t0 = g * CB_T;
    const int nt = (nb - t0 < CB_T) ? (nb - t0) : CB_T;
    const int cols = nblk * 32;
    const uint8_t *tile = w + (size_t)ti * nblk * 576;
    const size_t vl = 32, vh = 16;
    const int8_t *xq[CB_T]; const float *xd[CB_T];
    for (int i = 0; i < CB_T; i++) { int t = t0 + (i < nt ? i : nt - 1); xq[i] = xqa + (size_t)t * cols; xd[i] = xda + (size_t)t * nblk; }
    float facc[CB_T][32];
    vfloat32m4_t fz = __riscv_vfmv_v_f_f32m4(0.0f, vl);
    for (int i = 0; i < CB_T; i++) __riscv_vse32_v_f32m4(facc[i], fz, vl);
    for (int b = 0; b < nblk; b++) {
        const uint8_t *sc = tile + (size_t)b * 576, *dt = sc + 64;
        float s[32];
        for (int k = 0; k < 16; k++) {
            uint16_t h0, h1; memcpy(&h0, sc + 4 * k, 2); memcpy(&h1, sc + 4 * k + 2, 2);
            s[k] = f16f(h0); s[16 + k] = f16f(h1);
        }
        vfloat32m4_t vs = __riscv_vle32_v_f32m4(s, vl);
        vint16m2_t a0 = __riscv_vmv_v_x_i16m2(0, vl), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
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
#define EPI(I, A) do { vfloat32m4_t d_ = __riscv_vfwcvt_f_x_v_f32m4((A), vl); \
            vfloat32m4_t f_ = __riscv_vle32_v_f32m4(facc[I], vl); \
            f_ = __riscv_vfadd_vv_f32m4(f_, __riscv_vfmul_vf_f32m4(__riscv_vfmul_vv_f32m4(vs, d_, vl), xd[I][b], vl), vl); \
            __riscv_vse32_v_f32m4(facc[I], f_, vl); } while (0)
        EPI(0, a0); EPI(1, a1); EPI(2, a2); EPI(3, a3); EPI(4, a4); EPI(5, a5); EPI(6, a6); EPI(7, a7);
#undef EPI
    }
    for (int i = 0; i < nt; i++) {
        float *o = ob + (size_t)(t0 + i) * rows + (size_t)ti * 32;
        for (int k = 0; k < 16; k++) { o[2 * k] = facc[i][k]; o[2 * k + 1] = facc[i][16 + k]; }
    }
}

static void gemm_colb(const uint8_t *w, const int8_t *xq, const float *xd, float *ob,
                      int rows, int nblk, int nb, int nth, int pinned) {
    const int ngrp = (nb + CB_T - 1) / CB_T, nit = (rows / 32) * ngrp;
#pragma omp parallel num_threads(nth)
    {
        cpu_set_t old; int ho = 0;
        if (pinned) { ho = sched_getaffinity(0, sizeof(old), &old) == 0; pin(omp_get_thread_num() & 3); }
#pragma omp for schedule(static)
        for (int it = 0; it < nit; it++) colb_worker(w, xq, xd, ob, rows, nblk, nb, ngrp, it);
        if (ho) sched_setaffinity(0, sizeof(old), &old);
    }
}

/* ---------------- IME 侧 ----------------
 * 权重 16 行 tile，每块 288B：[0,32) 16 个 f16 scale（行序）；[32,288) 两个 128B 半块 h，
 *   半块内字节 (g,r,kk) 偏移 g*32+r*8+kk：低 nibble = 行 g*4+r 的 K=b*32+h*16+kk，
 *   高 nibble = 同行 K=b*32+h*16+8+kk。（值 0..15，减 8 得 int8）
 * 激活 4 token 组，每块 128B：4 个子块 j，字节 j*32+t*8+kk = xq[t][b*32+j*8+kk]；
 *   xe[b][e] = xd[t=e&3][b]（16 float）。
 * vmadot vd, vs1(W 组 g: 4 行 × 8K), vs2(X 子块: 4 token × 8K) -> vd[r*4+t]（int32）。
 * 收尾：e = r*4+t，行 = g*4+r；F_g = F_g + ((s[行] * float(acc_g)) * xe[t])。
 * 寄存器：v0-v3 gather 索引，v4-v7 X 子块，v8-v15 W 解包，v16-v23 int32 acc(4 组 m2)，
 *   v24-v31 F(4 组 m2)；收尾复用 v8-v15（v10-11 scale f32，v12-13 xe，v14-15 gather）。 */
static const uint16_t IME_IDX[64] = {
    0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,  4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,
    8,8,8,8,9,9,9,9,10,10,10,10,11,11,11,11, 12,12,12,12,13,13,13,13,14,14,14,14,15,15,15,15 };

__attribute__((noinline))
static void ime_q4_tile(const uint8_t *w, const int8_t *x, const float *xe, int nblk, float *out /*[4][16]*/) {
    long k = nblk;
    __asm__ volatile(
        "vsetivli    zero, 16, e16, m1, ta, ma \n\t"
        "vle16.v     v0, (%[idx])              \n\t"
        "addi        t0, %[idx], 32            \n\t"
        "vle16.v     v1, (t0)                  \n\t"
        "addi        t0, %[idx], 64            \n\t"
        "vle16.v     v2, (t0)                  \n\t"
        "addi        t0, %[idx], 96            \n\t"
        "vle16.v     v3, (t0)                  \n\t"
        "vsetvli     t0, zero, e32, m8, ta, ma \n\t"
        "vmv.v.i     v24, 0                    \n\t"   /* F: v24-v31（4 组 e32m2）= +0 */
        "1:                                    \n\t"
        "vsetvli     t0, zero, e32, m8, ta, ma \n\t"
        "vmv.v.i     v16, 0                    \n\t"   /* acc: v16-v23 */
        "vsetvli     t0, zero, e8, m4, ta, ma  \n\t"
        "vle8.v      v4, (%[x])                \n\t"   /* X 4 子块 v4-v7 */
        /* 半块 0：K 0..15 */
        "addi        t1, %[w], 32              \n\t"
        "vle8.v      v12, (t1)                 \n\t"
        "vsrl.vi     v8, v12, 4                \n\t"   /* 高 nibble: K+8  -> v8-v11 */
        "vand.vi     v12, v12, 15              \n\t"   /* 低 nibble: K+0  -> v12-v15 */
        "vsetvli     t0, zero, e8, m8, ta, ma  \n\t"
        "vadd.vi     v8, v8, -8                \n\t"
        "vsetvli     t0, zero, e8, m1, ta, ma  \n\t"
        VMADOT(16, 12, 4) VMADOT(18, 13, 4) VMADOT(20, 14, 4) VMADOT(22, 15, 4)
        VMADOT(16, 8, 5)  VMADOT(18, 9, 5)  VMADOT(20, 10, 5) VMADOT(22, 11, 5)
        /* 半块 1：K 16..31 */
        "vsetvli     t0, zero, e8, m4, ta, ma  \n\t"
        "addi        t1, %[w], 160             \n\t"
        "vle8.v      v12, (t1)                 \n\t"
        "vsrl.vi     v8, v12, 4                \n\t"
        "vand.vi     v12, v12, 15              \n\t"
        "vsetvli     t0, zero, e8, m8, ta, ma  \n\t"
        "vadd.vi     v8, v8, -8                \n\t"
        "vsetvli     t0, zero, e8, m1, ta, ma  \n\t"
        VMADOT(16, 12, 6) VMADOT(18, 13, 6) VMADOT(20, 14, 6) VMADOT(22, 15, 6)
        VMADOT(16, 8, 7)  VMADOT(18, 9, 7)  VMADOT(20, 10, 7) VMADOT(22, 11, 7)
        /* 收尾（寄存器内）：v9 f16 scale -> v10-v11 f32；v12-v13 xe；v14-v15 gather；v8-v9 临时 */
        "vsetivli    zero, 16, e16, m1, ta, ma \n\t"
        "vle16.v     v9, (%[w])                \n\t"
        "vfwcvt.f.f.v v10, v9                  \n\t"
        "vsetivli    zero, 16, e32, m2, ta, ma \n\t"
        "vle32.v     v12, (%[xe])              \n\t"
#define IME_EPI(G, A, F) \
        "vrgatherei16.vv v14, v10, v" #G " \n\t" \
        "vfcvt.f.x.v v8, v" #A "           \n\t" \
        "vfmul.vv    v8, v14, v8           \n\t"   /* s*d */ \
        "vfmul.vv    v8, v8, v12           \n\t"   /* (s*d)*xd */ \
        "vfadd.vv    v" #F ", v" #F ", v8  \n\t"
        IME_EPI(0, 16, 24) IME_EPI(1, 18, 26) IME_EPI(2, 20, 28) IME_EPI(3, 22, 30)
#undef IME_EPI
        "addi        %[w], %[w], 288           \n\t"
        "addi        %[x], %[x], 128           \n\t"
        "addi        %[xe], %[xe], 64          \n\t"
        "addi        %[k], %[k], -1            \n\t"
        "bnez        %[k], 1b                  \n\t"
        "vsetvli     t0, zero, e32, m8, ta, ma \n\t"
        "vse32.v     v24, (%[o])               \n\t"   /* out[g*16 + r*4 + t] */
        : [w] "+r"(w), [x] "+r"(x), [xe] "+r"(xe), [k] "+r"(k)
        : [idx] "r"(IME_IDX), [o] "r"(out)
        : "t0", "t1", "memory",
          "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11",
          "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
          "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
}

/* 激活打包（4 token 组）：xp[(g*nblk+b)*128 + j*32 + t*8 + kk]，xe[(g*nblk+b)*16 + e] */
static void ime_pack_x(const int8_t *xq, const float *xd, int8_t *xp, float *xe, int nblk, int nb, int g) {
    const int cols = nblk * 32;
    int tt[4];
    for (int t = 0; t < 4; t++) { int v = g * 4 + t; tt[t] = v < nb ? v : nb - 1; }
    for (int b = 0; b < nblk; b++) {
        int8_t *d = xp + ((size_t)g * nblk + b) * 128;
        for (int j = 0; j < 4; j++)
            for (int t = 0; t < 4; t++)
                memcpy(d + j * 32 + t * 8, xq + (size_t)tt[t] * cols + b * 32 + j * 8, 8);
        float *e = xe + ((size_t)g * nblk + b) * 16;
        for (int i = 0; i < 16; i++) e[i] = xd[(size_t)tt[i & 3] * nblk + b];
    }
}

static void gemm_ime(const uint8_t *wi, const int8_t *xq, const float *xd, float *ob,
                     int rows, int nblk, int nb, int nth, int8_t *xp, float *xe) {
    const int ng = (nb + 3) / 4, nti = rows / 16, nit = nti * ng;
#pragma omp parallel num_threads(nth)
    {
        cpu_set_t old; int ho = sched_getaffinity(0, sizeof(old), &old) == 0;
        pin(omp_get_thread_num() & 3);
#pragma omp for schedule(static)
        for (int g = 0; g < ng; g++) ime_pack_x(xq, xd, xp, xe, nblk, nb, g);
#pragma omp for schedule(static)
        for (int it = 0; it < nit; it++) {
            const int ti = it / ng, g = it % ng;
            float tmp[64] __attribute__((aligned(64)));
            ime_q4_tile(wi + (size_t)ti * nblk * 288, xp + (size_t)g * nblk * 128,
                        xe + (size_t)g * nblk * 16, nblk, tmp);
            for (int t = 0; t < 4 && g * 4 + t < nb; t++) {
                float *o = ob + (size_t)(g * 4 + t) * rows + (size_t)ti * 16;
                for (int gg = 0; gg < 4; gg++)
                    for (int r = 0; r < 4; r++) o[gg * 4 + r] = tmp[gg * 16 + r * 4 + t];
            }
        }
        if (ho) sched_setaffinity(0, sizeof(old), &old);
    }
}

/* ---------------- 权重生成与两种布局打包 ---------------- */
/* q[row][col] nibble 0..15；sc[row][b] f16 bits */
static void pack_colb(const uint8_t *q, const uint16_t *sc, uint8_t *w, int rows, int nblk) {
    const int cols = nblk * 32;
    for (int ti = 0; ti < rows / 32; ti++)
        for (int b = 0; b < nblk; b++) {
            uint8_t *s = w + ((size_t)ti * nblk + b) * 576, *dt = s + 64;
            for (int r = 0; r < 32; r++) memcpy(s + 2 * r, &sc[(size_t)(ti * 32 + r) * nblk + b], 2);
            for (int cc = 0; cc < 32; cc++)
                for (int k = 0; k < 16; k++) {
                    const size_t c = (size_t)b * 32 + cc;
                    dt[cc * 16 + k] = q[(size_t)(ti * 32 + 2 * k) * cols + c] |
                                      (uint8_t)(q[(size_t)(ti * 32 + 2 * k + 1) * cols + c] << 4);
                }
        }
}

static void pack_ime(const uint8_t *q, const uint16_t *sc, uint8_t *w, int rows, int nblk) {
    const int cols = nblk * 32;
    for (int ti = 0; ti < rows / 16; ti++)
        for (int b = 0; b < nblk; b++) {
            uint8_t *s = w + ((size_t)ti * nblk + b) * 288;
            for (int r = 0; r < 16; r++) memcpy(s + 2 * r, &sc[(size_t)(ti * 16 + r) * nblk + b], 2);
            for (int h = 0; h < 2; h++)
                for (int g = 0; g < 4; g++)
                    for (int r = 0; r < 4; r++)
                        for (int kk = 0; kk < 8; kk++) {
                            const size_t row = (size_t)ti * 16 + g * 4 + r, c = (size_t)b * 32 + h * 16 + kk;
                            s[32 + h * 128 + g * 32 + r * 8 + kk] =
                                q[row * cols + c] | (uint8_t)(q[row * cols + c + 8] << 4);
                        }
        }
}

/* 标量参考（与两侧同浮点顺序）：F=+0；F += (s*sum)*xd */
static void gemm_ref(const uint8_t *q, const uint16_t *sc, const int8_t *xq, const float *xd,
                     float *ob, int rows, int nblk, int nb) {
    const int cols = nblk * 32;
#pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; r++)
        for (int t = 0; t < nb; t++) {
            float F = 0.0f;
            for (int b = 0; b < nblk; b++) {
                int sum = 0;
                for (int c = 0; c < 32; c++)
                    sum += ((int)q[(size_t)r * cols + b * 32 + c] - 8) * xq[(size_t)t * cols + b * 32 + c];
                volatile float sd = f16f(sc[(size_t)r * nblk + b]) * (float)sum;
                volatile float p = sd * xd[(size_t)t * nblk + b];
                F = F + p;
            }
            ob[(size_t)t * rows + r] = F;
        }
}

static void cmp(const char *tag, const float *a, const float *b, size_t n) {
    size_t mis = 0; double md = 0;
    for (size_t i = 0; i < n; i++)
        if (memcmp(&a[i], &b[i], 4)) { mis++; double d = a[i] - b[i]; if (d < 0) d = -d; if (d > md) md = d; }
    printf("  BITCMP %-10s mismatch=%zu/%zu max|d|=%.3g\n", tag, mis, n, md);
}

static uint16_t rnd_f16(void) {
    uint32_t r = rnd();
    uint16_t m = r & 0x3FF, s = (r >> 10) & 1;
    uint16_t e = (r % 97 == 0) ? 0 : (uint16_t)(3 + (r >> 12) % 9);   /* ~1% subnormal，否则 2^-12..2^-4 */
    return (uint16_t)((s << 15) | (e << 10) | m);
}

typedef struct { const char *name; int rows, cols; } Shape;

int main(int argc, char **argv) {
    const int model = argc > 1 ? atoi(argv[1]) : 0;   /* 0=0.6B 1=1.7B 2=4B 9=单 tile 自检 */
    const int reps = argc > 2 ? atoi(argv[2]) : 5;
    static const Shape S[3][4] = {
        { {"qkv", 4096, 1024}, {"o", 1024, 2048}, {"gu", 6144, 1024}, {"down", 1024, 3072} },
        { {"qkv", 4096, 2048}, {"o", 2048, 2048}, {"gu", 12288, 2048}, {"down", 2048, 6144} },
        { {"qkv", 6144, 2560}, {"o", 2560, 4096}, {"gu", 19456, 2560}, {"down", 2560, 9728} } };
    Shape one[1] = { {"tile", 32, 64} };
    const Shape *sh = model == 9 ? one : S[model];
    const int nsh = model == 9 ? 1 : 4;
    const int Ps[2] = {32, 128};
    double tot[2][3] = {{0}};
    printf("P1 model=%d reps=%d\n", model, reps);
    for (int si = 0; si < nsh; si++) {
        const int rows = sh[si].rows, cols = sh[si].cols, nblk = cols / 32;
        uint8_t *q = malloc((size_t)rows * cols);
        uint16_t *sc = malloc((size_t)rows * nblk * 2);
        for (size_t i = 0; i < (size_t)rows * cols; i++) q[i] = rnd() & 15;
        for (size_t i = 0; i < (size_t)rows * nblk; i++) sc[i] = rnd_f16();
        uint8_t *wc = aligned_alloc(64, (size_t)rows * nblk * 576 / 32 + 64);
        uint8_t *wi = aligned_alloc(64, (size_t)rows * nblk * 288 / 16 + 64);
        pack_colb(q, sc, wc, rows, nblk);
        pack_ime(q, sc, wi, rows, nblk);
        for (int pi = 0; pi < 2; pi++) {
            const int P = Ps[pi], ng = (P + 3) / 4;
            int8_t *xq = malloc((size_t)P * cols);
            float *xd = malloc((size_t)P * nblk * 4);
            for (size_t i = 0; i < (size_t)P * cols; i++) xq[i] = (int8_t)((int)(rnd() % 255) - 127);
            for (size_t i = 0; i < (size_t)P * nblk; i++) xd[i] = (float)(rnd() % 100000 + 1) * 1e-7f;
            int8_t *xp = aligned_alloc(64, (size_t)ng * nblk * 128);
            float *xe = aligned_alloc(64, (size_t)ng * nblk * 64);
            float *o0 = malloc((size_t)P * rows * 4), *o1 = malloc((size_t)P * rows * 4),
                  *o2 = malloc((size_t)P * rows * 4), *o3 = malloc((size_t)P * rows * 4);
            gemm_ref(q, sc, xq, xd, o0, rows, nblk, P);
            double best[3] = {1e30, 1e30, 1e30};
            for (int rep = 0; rep < reps + 1; rep++) {          /* 第 0 轮预热 */
                double t;
                t = now_s(); gemm_colb(wc, xq, xd, o1, rows, nblk, P, 8, 0); t = now_s() - t; if (rep && t < best[0]) best[0] = t;
                t = now_s(); gemm_colb(wc, xq, xd, o2, rows, nblk, P, 4, 1); t = now_s() - t; if (rep && t < best[1]) best[1] = t;
                t = now_s(); gemm_ime(wi, xq, xd, o3, rows, nblk, P, 4, xp, xe); t = now_s() - t; if (rep && t < best[2]) best[2] = t;
            }
            const double gop = 2.0 * rows * cols * P * 1e-9;
            printf("%-5s %5dx%-5d P=%-3d colb8 %8.2fms %6.2fGOPS | colb4p %8.2fms %6.2fGOPS | ime4p %8.2fms %6.2fGOPS | ime4/colb8 %.2fx\n",
                   sh[si].name, rows, cols, P, best[0] * 1e3, gop / best[0], best[1] * 1e3, gop / best[1],
                   best[2] * 1e3, gop / best[2], best[0] / best[2]);
            cmp("colb8~ref", o1, o0, (size_t)P * rows);
            cmp("colb4~ref", o2, o0, (size_t)P * rows);
            cmp("ime~ref", o3, o0, (size_t)P * rows);
            cmp("ime~colb8", o3, o1, (size_t)P * rows);
            for (int k = 0; k < 3; k++) tot[pi][k] += best[k];
            free(xq); free(xd); free(xp); free(xe); free(o0); free(o1); free(o2); free(o3);
        }
        free(q); free(sc); free(wc); free(wi);
    }
    for (int pi = 0; pi < 2; pi++)
        printf("SUM P=%-3d colb8 %.2fms colb4p %.2fms ime4p %.2fms  ime4/colb8 %.2fx ime4/colb4 %.2fx\n", Ps[pi],
               tot[pi][0] * 1e3, tot[pi][1] * 1e3, tot[pi][2] * 1e3, tot[pi][0] / tot[pi][2], tot[pi][1] / tot[pi][2]);
    return 0;
}
