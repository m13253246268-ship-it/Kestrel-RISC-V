/**
 * bench_im2col_nhwc.c - im2col 布局微基准（K-YOLO-002 V50 Phase 0）
 *
 * 目的：在投入「全管线 NHWC 改造」之前，用 yolov8n@320 的**真实层形状**
 *       量化对比三条 im2col 实现的耗时，决定是否值得大改：
 *
 *   A = CHW 标量（当前生产实现，逐像素 cross-ic 小段 gather）
 *   B = HWC RVV（激活 NHWC，逐 tap 连续 C 个 float 的 vle32/vse32 + vfmax）
 *   T = CHW→HWC 转置代价（8×8 分块），是 B 方案在「输出仍 CHW」时的额外开销
 *
 * 关键判据：B + T 是否显著快于 A。若否，NHWC 改造不值得。
 *
 * 正确性：A 与 B 的 k 排列不同（A: k=ic*9+tap；B: k=tap*C+ic），但 GEMM 与 k
 *   排列无关（int32 可重排），权重同步重排即可，故两者都是合法布局。本基准
 *   逐元素校验 imcB[tap*C+ic] == imcA[ic*9+tap] 且 rowsc 相同（位级）。
 *
 * 构建（x86 主机无 RVV 时 B 走标量对照；板端 gcc -march=rv64gcv 才有 RVV）：
 *   riscv64-linux-gnu-gcc -O2 -march=rv64gcv -mabi=lp64d -fopenmp \
 *       -ffp-contract=off bench_im2col_nhwc.c -o bench_im2col_nhwc -lm
 *   （板端直接用 gcc，并追加 -DHAVE_RVV=1）
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
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

/* ---- 与生产相同的量化（RVF 与标量位级等价，见 vllm_yolo.c 注释）---- */
static inline void quant_row(const float *row, int K, float inv, int8_t *arow) {
    const float magic = 12582912.0f;   /* 2^23 */
#if HAVE_RVV
    const size_t vmax4 = __riscv_vsetvlmax_e32m4();
    for (int j = 0; j < K; j += (int)vmax4) {
        const size_t rem = (size_t)(K - j);
        const size_t vl = __riscv_vsetvl_e32m4(rem < vmax4 ? rem : vmax4);
        vfloat32m4_t f = __riscv_vle32_v_f32m4(row + j, vl);
        f = __riscv_vfmul_vf_f32m4(f, inv, vl);
        f = __riscv_vfmax_vf_f32m4(f, -127.0f, vl);
        f = __riscv_vfmin_vf_f32m4(f,  127.0f, vl);
        vint32m4_t q = __riscv_vfcvt_x_f_v_i32m4(f, vl);
        vint16m2_t q16 = __riscv_vncvt_x_x_w_i16m2(q, vl);
        vint8m1_t q8 = __riscv_vncvt_x_x_w_i8m1(q16, vl);
        __riscv_vse8_v_i8m1(arow + j, q8, vl);
    }
#else
    for (int j = 0; j < K; j++) {
        int q = (int)(row[j] * inv + magic) - 12582912;
        if (q > 127) q = 127; else if (q < -127) q = -127;
        arow[j] = (int8_t)q;
    }
#endif
}

/* ================= A：CHW 标量（生产实现拷贝） ================= */
static void im2col_chw(const float *x, const YLayer *L, int K, int Kp,
                       int8_t *imc, float *rowsc) {
    const int C = L->C, H = L->H, W = L->W, KH = L->KH, KW = L->KW;
    const int sh = L->sh, sw = L->sw, ph = L->ph, pw = L->pw;
    const int OH = L->OH, OW = L->OW;
#ifdef _OPENMP
    int nth = omp_get_max_threads();
#else
    int nth = 1;
#endif
    float **rb = (float **)calloc((size_t)nth, sizeof(float *));
    for (int i = 0; i < nth; i++) rb[i] = (float *)malloc((size_t)K * sizeof(float));

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int oy = 0; oy < OH; oy++) {
#ifdef _OPENMP
        float *row = rb[omp_get_thread_num()];
#else
        float *row = rb[0];
#endif
        for (int ox = 0; ox < OW; ox++) {
            const int pp = oy * OW + ox;
            int8_t *arow = imc + (size_t)pp * Kp;
            float mx = 0.f;
            int k = 0;
            const int iy0 = oy * sh - ph, ix0 = ox * sw - pw;
            const int interior = (iy0 >= 0 && iy0 + KH - 1 <= H - 1 &&
                                  ix0 >= 0 && ix0 + KW - 1 <= W - 1);
            if (interior) {
                for (int ic = 0; ic < C; ic++) {
                    const float *xc = x + (size_t)ic * H * W;
                    for (int ky = 0; ky < KH; ky++) {
                        const float *xr = xc + (size_t)(iy0 + ky) * W + ix0;
                        for (int kx = 0; kx < KW; kx++, k++) {
                            float v = xr[kx];
                            row[k] = v;
                            float av = fabsf(v); if (av > mx) mx = av;
                        }
                    }
                }
            } else {
                for (int ic = 0; ic < C; ic++) {
                    const float *xc = x + (size_t)ic * H * W;
                    for (int ky = 0; ky < KH; ky++) {
                        const int iy = iy0 + ky;
                        const float *xr = (iy >= 0 && iy < H) ? xc + (size_t)iy * W : NULL;
                        for (int kx = 0; kx < KW; kx++, k++) {
                            float v = 0.f;
                            if (xr) { const int ix = ix0 + kx; if (ix >= 0 && ix < W) v = xr[ix]; }
                            row[k] = v;
                            float av = fabsf(v); if (av > mx) mx = av;
                        }
                    }
                }
            }
            const float s = (mx > 0.f) ? (mx / 127.f) : 1.f;
            if (Kp > K) memset(arow + K, 0, (size_t)(Kp - K));
            quant_row(row, K, 1.f / s, arow);
            rowsc[pp] = s;
        }
    }
    for (int i = 0; i < nth; i++) free(rb[i]);
    free(rb);
}

/* ================= B：HWC RVV（逐 tap 连续 C 个 float） ================= */
static void im2col_hwc(const float *x, const YLayer *L, int K, int Kp,
                       int8_t *imc, float *rowsc) {
    const int C = L->C, H = L->H, W = L->W, KH = L->KH, KW = L->KW;
    const int sh = L->sh, sw = L->sw, ph = L->ph, pw = L->pw;
    const int OH = L->OH, OW = L->OW;
#ifdef _OPENMP
    int nth = omp_get_max_threads();
#else
    int nth = 1;
#endif
    float **rb = (float **)calloc((size_t)nth, sizeof(float *));
    for (int i = 0; i < nth; i++) rb[i] = (float *)malloc((size_t)K * sizeof(float));

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int oy = 0; oy < OH; oy++) {
#ifdef _OPENMP
        float *row = rb[omp_get_thread_num()];
#else
        float *row = rb[0];
#endif
        for (int ox = 0; ox < OW; ox++) {
            const int pp = oy * OW + ox;
            int8_t *arow = imc + (size_t)pp * Kp;
            float mx = 0.f;
            const int iy0 = oy * sh - ph, ix0 = ox * sw - pw;
            for (int ky = 0; ky < KH; ky++) {
                const int iy = iy0 + ky;
                for (int kx = 0; kx < KW; kx++) {
                    const int tap = ky * KW + kx;
                    const int koff = tap * C;
                    float *dst = row + koff;
                    const int ix = ix0 + kx;
                    if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
                        const float *src = x + ((size_t)iy * W + ix) * C;
#if HAVE_RVV
                        const size_t vmaxf = __riscv_vsetvlmax_e32m1();
                        vfloat32m1_t vm1 = __riscv_vfmv_v_f_f32m1(0.0f, vmaxf);
                        size_t lastvl = 1;
                        for (int ic = 0; ic < C; ic += (int)vmaxf) {
                            const size_t rem = (size_t)(C - ic);
                            const size_t vl = __riscv_vsetvl_e32m1(rem < vmaxf ? rem : vmaxf);
                            lastvl = vl;
                            vfloat32m1_t v = __riscv_vle32_v_f32m1(src + ic, vl);
                            __riscv_vse32_v_f32m1(dst + ic, v, vl);
                            vm1 = __riscv_vfmax_vv_f32m1(
                                      vm1, __riscv_vfabs_v_f32m1(v, vl), vl);
                        }
                        float t = __riscv_vfmv_f_s_f32m1_f32(
                                      __riscv_vfredmax_vs_f32m1_f32m1(
                                          vm1, __riscv_vfmv_v_f_f32m1(0.0f, vmaxf), lastvl));
                        if (t > mx) mx = t;
#else
                        for (int ic = 0; ic < C; ic++) {
                            float v = src[ic];
                            dst[ic] = v;
                            float av = fabsf(v); if (av > mx) mx = av;
                        }
#endif
                    } else {
                        memset(dst, 0, (size_t)C * sizeof(float));
                    }
                }
            }
            const float s = (mx > 0.f) ? (mx / 127.f) : 1.f;
            if (Kp > K) memset(arow + K, 0, (size_t)(Kp - K));
            quant_row(row, K, 1.f / s, arow);
            rowsc[pp] = s;
        }
    }
    for (int i = 0; i < nth; i++) free(rb[i]);
    free(rb);
}

/* ================= T：CHW→HWC 转置（8×8 分块） ================= */
static void transpose_chw2hwc(const float *src, float *dst, int C, int H, int W) {
    const int P = H * W;
    const int TB = 8;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) collapse(2)
#endif
    for (int c0 = 0; c0 < C; c0 += TB) {
        for (int p0 = 0; p0 < P; p0 += TB) {
            const int c1 = (c0 + TB < C) ? c0 + TB : C;
            const int p1 = (p0 + TB < P) ? p0 + TB : P;
            for (int p = p0; p < p1; p++)
                for (int c = c0; c < c1; c++)
                    dst[(size_t)p * C + c] = src[(size_t)c * P + p];
        }
    }
}

int main(void) {
    int nth = 1;
#ifdef _OPENMP
    nth = omp_get_max_threads();
#endif
    printf("=== im2col 布局微基准（yolov8n@320 真实层形状）threads=%d RVV=%d ===\n", nth, HAVE_RVV);

    double tA = 0, tB = 0, tT = 0, tA11 = 0, tB11 = 0;
    long long cyc33 = 0, cyc11 = 0;
    int nbad = 0;

    for (int li = 0; li < K_NLAYERS; li++) {
        const YLayer *L = &k_layers[li];
        const int K = L->C * L->KH * L->KW;
        const int Kp = (K + 31) & ~31;
        const int P = L->OH * L->OW;
        const size_t nCHW = (size_t)L->C * L->H * L->W;

        float *xchw = (float *)malloc(nCHW * sizeof(float));
        float *xhwc = (float *)malloc(nCHW * sizeof(float));
        int8_t *imcA = (int8_t *)malloc((size_t)P * Kp);
        int8_t *imcB = (int8_t *)malloc((size_t)P * Kp);
        float *rsA = (float *)malloc((size_t)P * sizeof(float));
        float *rsB = (float *)malloc((size_t)P * sizeof(float));
        if (!xchw || !xhwc || !imcA || !imcB || !rsA || !rsB) { printf("OOM\n"); return 1; }

        /* 确定性填充（无 RNG 依赖，跨平台可复现） */
        for (size_t i = 0; i < nCHW; i++)
            xchw[i] = (float)((int)(i * 2654435761u % 2001) - 1000) / 250.0f;

        memset(imcA, 0, (size_t)P * Kp);
        memset(imcB, 0, (size_t)P * Kp);
        /* 每项跑 R 次取最小，压掉调度抖动 */
        const int R = 3;
        double tA1 = 1e9, tB1 = 1e9, tT1 = 1e9;
        for (int r = 0; r < R; r++) {
            double t0 = now_s();
            transpose_chw2hwc(xchw, xhwc, L->C, L->H, L->W);
            double d = now_s() - t0; if (d < tT1) tT1 = d;
            t0 = now_s(); im2col_chw(xchw, L, K, Kp, imcA, rsA);
            d = now_s() - t0; if (d < tA1) tA1 = d;
            t0 = now_s(); im2col_hwc(xhwc, L, K, Kp, imcB, rsB);
            d = now_s() - t0; if (d < tB1) tB1 = d;
        }

        /* 校验：imcB[tap*C+ic] == imcA[ic*KW*KH+tap]，rowsc 位级相同 */
        int bad = 0;
        if (L->KH == 3 && L->KW == 3) {
            for (int p = 0; p < P && bad < 5; p++) {
                if (rsA[p] != rsB[p]) { bad++; continue; }
                for (int ic = 0; ic < L->C; ic++)
                    for (int tap = 0; tap < 9; tap++)
                        if (imcA[(size_t)p * Kp + ic * 9 + tap] !=
                            imcB[(size_t)p * Kp + tap * L->C + ic]) { bad++; break; }
            }
        } else {
            for (int p = 0; p < P && bad < 5; p++) {
                if (rsA[p] != rsB[p]) { bad++; continue; }
                if (memcmp(imcA + (size_t)p * Kp, imcB + (size_t)p * Kp, (size_t)K) != 0) bad++;
            }
        }
        nbad += bad;

        if (L->KH == 3) { tA += tA1; tB += tB1; tT += tT1; cyc33++; }
        else            { tA11 += tA1; tB11 += tB1; cyc11++; }

        printf("L%02d C=%3d %3dx%3d k%d oc=%3d P=%6d Kp=%4d | CHW=%7.3fms HWC=%7.3fms T=%6.3fms | %s\n",
               li, L->C, L->H, L->W, L->KH, L->OC, P, Kp, tA1 * 1e3, tB1 * 1e3, tT1 * 1e3,
               bad ? "MISMATCH" : "ok");

        free(xchw); free(xhwc); free(imcA); free(imcB); free(rsA); free(rsB);
    }

    printf("\n---- 汇总 ----\n");
    printf("3x3 层（%lld 层）：CHW=%.2fms  HWC=%.2fms  CHW→HWC转置=%.2fms\n",
           cyc33, tA * 1e3, tB * 1e3, tT * 1e3);
    printf("1x1 层（%lld 层）：CHW=%.2fms  HWC=%.2fms（1x1 在 CHW 下已是跨步连续，HWC 无优势预期）\n",
           cyc11, tA11 * 1e3, tB11 * 1e3);
    printf("im2col 合计：CHW=%.2fms  HWC=%.2fms  HWC+转置=%.2fms\n",
           (tA + tA11) * 1e3, (tB + tB11) * 1e3, (tB + tB11 + tT) * 1e3);
    if (tB + tB11 > 0) {
        printf("3x3 加速比(HWC/CHW)=%.2fx  含转置=%.2fx\n",
               tA / tB, tA / (tB + tT));
        printf("全 im2col 加速比：纯 HWC=%.2fx  含转置=%.2fx\n",
               (tA + tA11) / (tB + tB11), (tA + tA11) / (tB + tB11 + tT));
    }
    printf("正确性：%s（bad=%d）\n", nbad ? "FAIL" : "ALL MATCH", nbad);
    return nbad ? 2 : 0;
}
