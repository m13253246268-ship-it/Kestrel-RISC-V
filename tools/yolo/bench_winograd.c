/**
 * bench_winograd.c - 整数精确 Winograd F(2x2,3x3) 可行性验证（K-YOLO-002 V53）
 *
 * 动机（公理指导）：axiom_registry.json 的 `isomorphic_superposition`
 * （多项式环 → NTT 域，卷积坍缩为逐点乘）在 3x3 卷积上的工程形态就是
 * Winograd。它是唯一剩下的**算法级**（而非内核形态级）减量路线：
 *  3x3 层占 yolov8n@320 全部 MAC 的 82.5%（895M/1085M），F(2x2,3x3) 把
 *  乘法数降到 4/9 = 0.444（2.25x）。
 *
 * 关键问题（此前只是纸面推断，本文件专门实测）：
 *  Q1 整数精确性：把 G 的 1/2 吸收进权重（G'=2G），变换后 Y 恰为 4x 真值，
 *     末尾精确除 4 ⇒ 结果与直接法**逐元素相同**？（若能，位级一致可保住）
 *  Q2 溢出边界：|U|<=9*127=1143、|V|<=4*127=508，M=Σ_ic U*V。int32 够不够？
 *     （此前误算成「必须 int64 ⇒ 车道减半 ⇒ 收益归零」，本文件给出真值）
 *  Q3 性能：Winograd 三趟（输入变换 + 16 个独立 GEMM + 输出变换）在
 *     yolov8n 真实层形状上，是否真的快于现状内核（dp_m0 形态）？
 *     注意 16 个位置是**互相独立的小 GEMM**，可规避寄存器压力——
 *     M 只有 1 个累加器，向量化方向 = tile（与现状的 p 同构）。
 *
 * 构建（板端）：gcc -O2 -march=rv64gcv -mabi=lp64d -fopenmp -DHAVE_RVV=1 \
 *                   bench_winograd.c -o bench_winograd -lm
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

typedef struct { int C, H, W, KH, KW, sh, sw, ph, pw, OC, OH, OW; } YLayer;
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

/* ---------------- Winograd F(2x2,3x3) 变换矩阵 ----------------
 * G' = 2G（把 1/2 吸收进权重，保持整数）
 *   G' = [[2,0,0],[1,1,1],[1,-1,1],[0,0,2]]
 * B^T 元素只有 0/±1（每行 <=2 项）
 *   B^T= [[1,0,-1,0],[0,1,1,0],[0,-1,1,0],[0,1,0,-1]]
 * A^T= [[1,1,1,0],[0,1,-1,-1]]
 * 恒等式：A^T [ (G'W G'^T) ⊙ (B^T d B) ] A = 4 * (直接 3x3 卷积)
 * ------------------------------------------------------------ */

/* 权重变换：w[3][3] (int8) → U[4][4] (int16)，U = G' w G'^T */
static void winograd_weight(const int8_t *w, int wstride, int16_t *U) {
    int32_t t[4][3];
    /* t = G' w : 行 0 = 2*w0, 行1 = w0+w1+w2, 行2 = w0-w1+w2, 行3 = 2*w2 */
    for (int j = 0; j < 3; j++) {
        const int32_t w0 = w[0 * wstride + j], w1 = w[1 * wstride + j], w2 = w[2 * wstride + j];
        t[0][j] = 2 * w0;
        t[1][j] = w0 + w1 + w2;
        t[2][j] = w0 - w1 + w2;
        t[3][j] = 2 * w2;
    }
    /* U = t G'^T : 列 0 = 2*t0, 列1 = t0+t1+t2, 列2 = t0-t1+t2, 列3 = 2*t2 */
    for (int i = 0; i < 4; i++) {
        const int32_t a = t[i][0], b = t[i][1], c = t[i][2];
        U[i * 4 + 0] = (int16_t)(2 * a);
        U[i * 4 + 1] = (int16_t)(a + b + c);
        U[i * 4 + 2] = (int16_t)(a - b + c);
        U[i * 4 + 3] = (int16_t)(2 * c);
    }
}

/* 输入变换：4x4 patch（int8，越界取 0）→ V[4][4] (int16)，V = B^T d B */
static void winograd_input(const int8_t *d, int dstride, int16_t *V) {
    int32_t t[4][4];
    for (int j = 0; j < 4; j++) {
        const int32_t d0 = d[0 * dstride + j], d1 = d[1 * dstride + j];
        const int32_t d2 = d[2 * dstride + j], d3 = d[3 * dstride + j];
        t[0][j] = d0 - d2;
        t[1][j] = d1 + d2;
        t[2][j] = -d1 + d2;
        t[3][j] = d1 - d3;
    }
    for (int i = 0; i < 4; i++) {
        const int32_t a = t[i][0], b = t[i][1], c = t[i][2], e = t[i][3];
        V[i * 4 + 0] = (int16_t)(a - c);
        V[i * 4 + 1] = (int16_t)(b + c);
        V[i * 4 + 2] = (int16_t)(-b + c);
        V[i * 4 + 3] = (int16_t)(b - e);
    }
}

/* 输出变换：M[4][4] (int32) → Y[2][2] (int32)，Y = (A^T M A) / 4（精确除） */
static void winograd_output(const int32_t *M, int32_t *Y, int ystride) {
    int32_t s[2][4];
    for (int j = 0; j < 4; j++) {
        const int32_t m0 = M[0 * 4 + j], m1 = M[1 * 4 + j], m2 = M[2 * 4 + j], m3 = M[3 * 4 + j];
        s[0][j] = m0 + m1 + m2;
        s[1][j] = m1 - m2 - m3;
    }
    for (int i = 0; i < 2; i++) {
        const int32_t a = s[i][0], b = s[i][1], c = s[i][2], e = s[i][3];
        Y[i * ystride + 0] = (a + b + c) / 4;
        Y[i * ystride + 1] = (b - c - e) / 4;
    }
}

/* ================= Part 1：正确性 + 溢出 ================= */
static int part1(void) {
    printf("===== Part 1：整数精确性 / int32 溢出边界 =====\n");
    printf("%5s %7s %7s %14s %14s %10s %s\n",
           "C", "OHxOW", "OC", "max|direct|", "max|M| 上界实测", "int32余量", "结果");
    int nbad = 0;
    const int Cs[] = { 3, 16, 32, 64, 128, 256, 512 };
    for (int ci = 0; ci < (int)(sizeof(Cs) / sizeof(Cs[0])); ci++) {
        const int C = Cs[ci], H = 10, W = 10, OH = 10, OW = 10, OC = 8;
        const int ntiles = (OH / 2) * (OW / 2);
        int8_t *x = (int8_t *)malloc((size_t)C * H * W);
        int8_t *w = (int8_t *)malloc((size_t)OC * C * 9);
        int32_t *yref = (int32_t *)malloc((size_t)OC * OH * OW * sizeof(int32_t));
        int32_t *ywin = (int32_t *)malloc((size_t)OC * OH * OW * sizeof(int32_t));
        int16_t *V = (int16_t *)malloc((size_t)16 * C * ntiles * sizeof(int16_t));
        int16_t *U = (int16_t *)malloc((size_t)16 * OC * C * sizeof(int16_t));
        int32_t *M = (int32_t *)malloc((size_t)16 * OC * ntiles * sizeof(int32_t));
        /* 极值填充：全部 ±127，最大化溢出风险 */
        for (size_t i = 0; i < (size_t)C * H * W; i++) x[i] = (i & 1) ? 127 : -127;
        for (size_t i = 0; i < (size_t)OC * C * 9; i++) w[i] = (i % 3) ? 127 : -127;

        /* --- 直接法参考（int32 累加） --- */
        for (int oc = 0; oc < OC; oc++)
            for (int oy = 0; oy < OH; oy++)
                for (int ox = 0; ox < OW; ox++) {
                    int32_t s = 0;
                    for (int ic = 0; ic < C; ic++)
                        for (int ky = 0; ky < 3; ky++)
                            for (int kx = 0; kx < 3; kx++) {
                                int iy = oy + ky - 1, ix = ox + kx - 1;
                                if (iy < 0 || iy >= H || ix < 0 || ix >= W) continue;
                                s += (int32_t)w[((size_t)oc * C + ic) * 9 + ky * 3 + kx]
                                     * (int32_t)x[((size_t)ic * H + iy) * W + ix];
                            }
                    yref[((size_t)oc * OH + oy) * OW + ox] = s;
                }

        /* --- Winograd：输入变换 → 16 个位置 GEMM → 输出变换 --- */
        for (int ic = 0; ic < C; ic++)
            for (int ty = 0; ty < OH / 2; ty++)
                for (int tx = 0; tx < OW / 2; tx++) {
                    int8_t patch[16] = {0};
                    for (int ky = 0; ky < 4; ky++)
                        for (int kx = 0; kx < 4; kx++) {
                            int iy = 2 * ty + ky - 1, ix = 2 * tx + kx - 1;
                            if (iy >= 0 && iy < H && ix >= 0 && ix < W)
                                patch[ky * 4 + kx] = x[((size_t)ic * H + iy) * W + ix];
                        }
                    int16_t vv[16];
                    winograd_input(patch, 4, vv);
                    const int tile = ty * (OW / 2) + tx;
                    for (int ij = 0; ij < 16; ij++)
                        V[((size_t)ij * C + ic) * ntiles + tile] = vv[ij];
                }
        for (int oc = 0; oc < OC; oc++)
            for (int ic = 0; ic < C; ic++) {
                int16_t uu[16];
                winograd_weight(w + ((size_t)oc * C + ic) * 9, 3, uu);
                for (int ij = 0; ij < 16; ij++)
                    U[((size_t)ij * OC + oc) * C + ic] = uu[ij];
            }
        for (int ij = 0; ij < 16; ij++) {
            const int16_t *Uij = U + (size_t)ij * OC * C;
            const int16_t *Vij = V + (size_t)ij * C * ntiles;
            int32_t *Mij = M + (size_t)ij * OC * ntiles;
            for (int oc = 0; oc < OC; oc++)
                for (int t = 0; t < ntiles; t++) {
                    int32_t s = 0;
                    for (int ic = 0; ic < C; ic++)
                        s += (int32_t)Uij[(size_t)oc * C + ic] * (int32_t)Vij[(size_t)ic * ntiles + t];
                    Mij[(size_t)oc * ntiles + t] = s;
                }
        }
        int64_t maxM = 0;
        for (size_t i = 0; i < (size_t)16 * OC * ntiles; i++) {
            int64_t a = M[i] < 0 ? -(int64_t)M[i] : (int64_t)M[i];
            if (a > maxM) maxM = a;
        }
        for (int oc = 0; oc < OC; oc++)
            for (int ty = 0; ty < OH / 2; ty++)
                for (int tx = 0; tx < OW / 2; tx++) {
                    const int tile = ty * (OW / 2) + tx;
                    int32_t mm[16];
                    for (int ij = 0; ij < 16; ij++)
                        mm[ij] = M[((size_t)ij * OC + oc) * ntiles + tile];
                    int32_t yy[4];
                    winograd_output(mm, yy, 2);
                    for (int ky = 0; ky < 2; ky++)
                        for (int kx = 0; kx < 2; kx++)
                            ywin[((size_t)oc * OH + (2 * ty + ky)) * OW + (2 * tx + kx)] = yy[ky * 2 + kx];
                }

        int64_t maxref = 0;
        for (size_t i = 0; i < (size_t)OC * OH * OW; i++) {
            int64_t a = yref[i] < 0 ? -(int64_t)yref[i] : (int64_t)yref[i];
            if (a > maxref) maxref = a;
        }
        int bad = 0;
        for (size_t i = 0; i < (size_t)OC * OH * OW; i++) if (yref[i] != ywin[i]) bad++;
        nbad += bad;
        printf("%5d %7s %7d %14lld %14lld %10.1fx %s\n",
               C, "10x10", OC, (long long)maxref, (long long)maxM,
               (double)INT32_MAX / (double)(maxM ? maxM : 1),
               bad ? "MISMATCH" : "EXACT");
        free(x); free(w); free(yref); free(ywin); free(V); free(U); free(M);
    }
    printf("结论：%s\n\n", nbad ? "整数 Winograd 与直接法不一致（需查）" : "整数 Winograd 与直接法逐元素完全相同");
    return nbad;
}

/* ================= Part 2：性能 ================= */

/* 现状内核形态（与 bench_dotprod 的 m0 一致）：4 oc / 32 lane / imc 行主序 */
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
#if HAVE_RVV
        const size_t vmax = __riscv_vsetvlmax_e8m1();
        int p0 = 0;
        for (; p0 + (int)vmax <= P; p0 += (int)vmax) {
            vint32m4_t v0 = __riscv_vmv_v_x_i32m4(0, vmax), v1 = v0, v2 = v0, v3 = v0;
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
        for (; p0 < P; p0++) {   /* 尾部标量（与生产同口径） */
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

/* Winograd 的 16 个位置 GEMM：M_ij[oc][t] = Σ_ic U_ij[oc][ic] * V_ij[ic][t]
 * lane = tile（与现状的 p 同构），每 ic 一条连续 vle16 喂 4 个 oc。 */
static void wino_gemm16(const int16_t *U, const int16_t *V, int OC, int C, int ntiles, int32_t *M) {
    for (int ij = 0; ij < 16; ij++) {
        const int16_t *Uij = U + (size_t)ij * OC * C;
        const int16_t *Vij = V + (size_t)ij * C * ntiles;
        int32_t *Mij = M + (size_t)ij * OC * ntiles;
        const int nb = OC / 4;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(nb > 2)
#endif
        for (int blk = 0; blk < nb; blk++) {
            const int oc0 = blk * 4;
            const int16_t *u0 = Uij + (size_t)(oc0 + 0) * C;
            const int16_t *u1 = Uij + (size_t)(oc0 + 1) * C;
            const int16_t *u2 = Uij + (size_t)(oc0 + 2) * C;
            const int16_t *u3 = Uij + (size_t)(oc0 + 3) * C;
#if HAVE_RVV
            const size_t vmax = __riscv_vsetvlmax_e16m2();   /* 与 i32m4 同 lane 数 */
            int t0 = 0;
            for (; t0 + (int)vmax <= ntiles; t0 += (int)vmax) {
                vint32m4_t v0 = __riscv_vmv_v_x_i32m4(0, vmax), v1 = v0, v2 = v0, v3 = v0;
                for (int ic = 0; ic < C; ic++) {
                    const vint16m2_t xv = __riscv_vle16_v_i16m2(Vij + (size_t)ic * ntiles + t0, vmax);
                    v0 = __riscv_vwmacc_vx_i32m4(v0, u0[ic], xv, vmax);
                    v1 = __riscv_vwmacc_vx_i32m4(v1, u1[ic], xv, vmax);
                    v2 = __riscv_vwmacc_vx_i32m4(v2, u2[ic], xv, vmax);
                    v3 = __riscv_vwmacc_vx_i32m4(v3, u3[ic], xv, vmax);
                }
                __riscv_vse32_v_i32m4(Mij + (size_t)(oc0 + 0) * ntiles + t0, v0, vmax);
                __riscv_vse32_v_i32m4(Mij + (size_t)(oc0 + 1) * ntiles + t0, v1, vmax);
                __riscv_vse32_v_i32m4(Mij + (size_t)(oc0 + 2) * ntiles + t0, v2, vmax);
                __riscv_vse32_v_i32m4(Mij + (size_t)(oc0 + 3) * ntiles + t0, v3, vmax);
            }
            for (; t0 < ntiles; t0++) {   /* tile 尾部标量 */
                for (int u = 0; u < 4; u++) {
                    const int16_t *ur = Uij + (size_t)(oc0 + u) * C;
                    int32_t s = 0;
                    for (int ic = 0; ic < C; ic++)
                        s += (int32_t)ur[ic] * (int32_t)Vij[(size_t)ic * ntiles + t0];
                    Mij[(size_t)(oc0 + u) * ntiles + t0] = s;
                }
            }
#endif
        }
    }
}

static int part2(void) {
    printf("===== Part 2：真实 3x3 层形状上的性能对比（8 线程，min-of-3）=====\n");
    double tA = 0, tB = 0, tT = 0, tG = 0;
    long long mac_direct = 0, mac_wino = 0;
    int nl = 0, bad_total = 0;
    const int R = 3;

    for (int li = 0; li < K_NLAYERS; li++) {
        const YLayer *L = &k_layers[li];
        if (L->KH != 3 || L->KW != 3 || L->sh != 1 || L->sw != 1) continue;
        if (L->OH % 2 || L->OW % 2) continue;
        const int C = L->C, H = L->H, W = L->W, OC = L->OC, OH = L->OH, OW = L->OW;
        const int K = C * 9, Kp = (K + 31) & ~31, P = OH * OW;
        const int ntiles = (OH / 2) * (OW / 2);
        if (OC % 4) continue;

        int8_t *x = (int8_t *)malloc((size_t)C * H * W);
        int8_t *Wq = (int8_t *)malloc((size_t)OC * Kp);
        int8_t *imc = (int8_t *)malloc((size_t)P * Kp);
        int32_t *yA = (int32_t *)malloc((size_t)OC * P * sizeof(int32_t));
        int16_t *V = (int16_t *)malloc((size_t)16 * C * ntiles * sizeof(int16_t));
        int16_t *U = (int16_t *)malloc((size_t)16 * OC * C * sizeof(int16_t));
        int32_t *M = (int32_t *)malloc((size_t)16 * OC * ntiles * sizeof(int32_t));
        int32_t *yB = (int32_t *)malloc((size_t)OC * P * sizeof(int32_t));
        if (!x || !Wq || !imc || !yA || !V || !U || !M || !yB) { printf("OOM\n"); return 1; }
        rnd_state = 0x9E3779B9u;
        for (size_t i = 0; i < (size_t)C * H * W; i++) x[i] = rnd_i8();
        rnd_state = 0x7F4A7C15u;
        for (size_t i = 0; i < (size_t)OC * Kp; i++) Wq[i] = rnd_i8();
        /* imc 由 x 直接 gather（im2col 的取数部分，不含量化）。量化（逐 patch 求
         * max + 写回）两臂相同、不在本基准范围，故不重复实现；但 gather 是 direct
         * 侧的真实成本，须计入「direct 完整」口径，与 Winograd 输入变换（同样含
         * 从 x 的 gather）对等。Kp > K 的补齐段必须清零，否则两臂都会读到复用堆
         * 块的残留数据（本基准曾因此误报 C=80 的层 MISMATCH）。 */
        memset(imc, 0, (size_t)P * Kp);
        double t_gather = now_s();
        for (int oy = 0; oy < OH; oy++)
            for (int ox = 0; ox < OW; ox++) {
                int8_t *arow = imc + (size_t)(oy * OW + ox) * Kp;
                int k = 0;
                for (int ic = 0; ic < C; ic++)
                    for (int ky = 0; ky < 3; ky++)
                        for (int kx = 0; kx < 3; kx++) {
                            int iy = oy + ky - 1, ix = ox + kx - 1;
                            arow[k++] = (iy >= 0 && iy < H && ix >= 0 && ix < W)
                                        ? x[((size_t)ic * H + iy) * W + ix] : 0;
                        }
            }
        t_gather = now_s() - t_gather;
        mac_direct += (long long)OC * P * Kp;
        mac_wino += (long long)16 * OC * C * ntiles;

        double loA = 1e9, loB = 1e9, loT = 1e9;
        for (int r = 0; r < R; r++) {
            double t0 = now_s(); dp_m0(Wq, OC, Kp, imc, P, yA); double d = now_s() - t0;
            if (d < loA) loA = d;

            t0 = now_s();
            /* (1) 输入变换（标量，逐 (ic,tile)） */
            for (int ic = 0; ic < C; ic++)
                for (int ty = 0; ty < OH / 2; ty++)
                    for (int tx = 0; tx < OW / 2; tx++) {
                        int8_t patch[16] = {0};
                        for (int ky = 0; ky < 4; ky++)
                            for (int kx = 0; kx < 4; kx++) {
                                int iy = 2 * ty + ky - 1, ix = 2 * tx + kx - 1;
                                if (iy >= 0 && iy < H && ix >= 0 && ix < W)
                                    patch[ky * 4 + kx] = x[((size_t)ic * H + iy) * W + ix];
                            }
                        int16_t vv[16];
                        winograd_input(patch, 4, vv);
                        const int tile = ty * (OW / 2) + tx;
                        for (int ij = 0; ij < 16; ij++)
                            V[((size_t)ij * C + ic) * ntiles + tile] = vv[ij];
                    }
            /* (2) 权重变换（生产上是一次性预处理，这里计入以便两臂都含全部工作） */
            for (int oc = 0; oc < OC; oc++)
                for (int ic = 0; ic < C; ic++) {
                    int16_t uu[16];
                    winograd_weight(Wq + (size_t)oc * Kp + ic * 9, 3, uu);
                    for (int ij = 0; ij < 16; ij++)
                        U[((size_t)ij * OC + oc) * C + ic] = uu[ij];
                }
            /* (3) 16 个位置 GEMM */
            wino_gemm16(U, V, OC, C, ntiles, M);
            /* (4) 输出变换 */
            for (int oc = 0; oc < OC; oc++)
                for (int ty = 0; ty < OH / 2; ty++)
                    for (int tx = 0; tx < OW / 2; tx++) {
                        const int tile = ty * (OW / 2) + tx;
                        int32_t mm[16];
                        for (int ij = 0; ij < 16; ij++)
                            mm[ij] = M[((size_t)ij * OC + oc) * ntiles + tile];
                        int32_t yy[4];
                        winograd_output(mm, yy, 2);
                        for (int ky = 0; ky < 2; ky++)
                            for (int kx = 0; kx < 2; kx++)
                                yB[((size_t)oc * OH + (2 * ty + ky)) * OW + (2 * tx + kx)] = yy[ky * 2 + kx];
                    }
            d = now_s() - t0;
            if (d < loB) loB = d;

            /* 只计 16 个 GEMM 本体（看变换/host 占比） */
            t0 = now_s(); wino_gemm16(U, V, OC, C, ntiles, M); d = now_s() - t0;
            if (d < loT) loT = d;
        }

        /* 正确性：RVV 的 16-GEMM 路径必须与 dp_m0 逐元素相同 */
        long ndiff = 0;
        for (size_t i = 0; i < (size_t)OC * P; i++) if (yA[i] != yB[i]) ndiff++;
        tA += loA; tB += loB; tT += loT; tG += t_gather; nl++;
        bad_total += (int)ndiff;
        printf("L%02d C=%3d %3dx%3d oc=%3d tiles=%5d | direct=%7.2fms  wino=%7.2fms (其中16GEMM=%7.2fms)  %s\n",
               li, C, H, W, OC, ntiles, loA * 1e3, loB * 1e3, loT * 1e3,
               ndiff ? "MISMATCH" : "EXACT");
        if (ndiff) printf("    !! diff=%ld/%d\n", ndiff, OC * P);
        free(x); free(Wq); free(imc); free(yA); free(V); free(U); free(M); free(yB);
    }
    printf("\n汇总（%d 个 3x3 / stride1 层）\n", nl);
    printf("  direct 纯点积（现状内核）    = %8.2f ms   MAC=%.1f M\n", tA * 1e3, mac_direct / 1e6);
    printf("  direct 完整（+im2col gather）= %8.2f ms\n", (tA + tG) * 1e3);
    printf("  winograd（输入+权重变换+16GEMM+输出）= %8.2f ms   MAC=%.1f M（理论 %.2fx）\n",
           tB * 1e3, mac_wino / 1e6, (double)mac_direct / (double)mac_wino);
    printf("  其中 16 个 GEMM 本体        = %8.2f ms\n", tT * 1e3);
    printf("  加速比 direct完整/winograd  = %8.2fx\n", (tA + tG) / tB);
    printf("  加速比 direct/16GEMM        = %8.2fx\n", tA / tT);
    return bad_total ? 1 : 0;
}

int main(void) {
    int nth = 1;
#ifdef _OPENMP
    nth = omp_get_max_threads();
#endif
    printf("=== 整数精确 Winograd F(2x2,3x3) 验证  threads=%d RVV=%d VLEN/32=%d ===\n\n",
           nth, HAVE_RVV, (int)(HAVE_RVV ? __riscv_vsetvlmax_e16m2() : 0));
    int rc = part1();
    rc += part2();
    return rc ? 2 : 0;
}
