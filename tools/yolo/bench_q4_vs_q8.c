/* bench_q4_vs_q8.c - 在 K1 板端对比 p 方向 RVV int8 内核的两种权重精度。
 *
 * A 臂 = 当前生产内核（K-YOLO-002 Phase 2，int8 权重，vwmacc.vx 标量广播）
 * B 臂 = 同样结构，但权重是 4bit（Q4_0 风格 nibble），内循环解包后送 vwmacc.vx
 *
 * 只测 GEMM 主体（不含 im2col、不含 SiLU），把「权重精度」这一项单独隔离出来。
 * 两臂的 im2col 读、累加结构、写回完全相同 ⇒ 差值就是 Q4 的净效果。
 *
 * 编译：gcc -O3 -march=rv64gcv -o bench_q4_vs_q8 bench_q4_vs_q8.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <riscv_vector.h>

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

/* ---------- A：int8 权重（现网内核） ---------- */
static void gemm_q8(const int8_t *W, const int8_t *imc, int OC, int Kp, int P,
                    int32_t *y) {
    const size_t vmax = __riscv_vsetvlmax_e8m1();
    for (int oc0 = 0; oc0 + 4 <= OC; oc0 += 4) {
        const int8_t *w0 = W + (size_t)(oc0 + 0) * Kp;
        const int8_t *w1 = W + (size_t)(oc0 + 1) * Kp;
        const int8_t *w2 = W + (size_t)(oc0 + 2) * Kp;
        const int8_t *w3 = W + (size_t)(oc0 + 3) * Kp;
        int32_t *y0 = y + (size_t)(oc0 + 0) * P;
        int32_t *y1 = y + (size_t)(oc0 + 1) * P;
        int32_t *y2 = y + (size_t)(oc0 + 2) * P;
        int32_t *y3 = y + (size_t)(oc0 + 3) * P;
        int p0 = 0;
        for (; p0 + (int)vmax <= P; p0 += (int)vmax) {
            vint32m4_t a0 = __riscv_vmv_v_x_i32m4(0, vmax);
            vint32m4_t a1 = a0, a2 = a0, a3 = a0;
            const int8_t *base = imc + (size_t)p0 * Kp;
            for (int k = 0; k < Kp; k++) {
                const vint8m1_t xv =
                    __riscv_vlse8_v_i8m1(base + k, (ptrdiff_t)Kp, vmax);
                const vint16m2_t x16 = __riscv_vsext_vf2_i16m2(xv, vmax);
                a0 = __riscv_vwmacc_vx_i32m4(a0, (int16_t)w0[k], x16, vmax);
                a1 = __riscv_vwmacc_vx_i32m4(a1, (int16_t)w1[k], x16, vmax);
                a2 = __riscv_vwmacc_vx_i32m4(a2, (int16_t)w2[k], x16, vmax);
                a3 = __riscv_vwmacc_vx_i32m4(a3, (int16_t)w3[k], x16, vmax);
            }
            __riscv_vse32_v_i32m4(y0 + p0, a0, vmax);
            __riscv_vse32_v_i32m4(y1 + p0, a1, vmax);
            __riscv_vse32_v_i32m4(y2 + p0, a2, vmax);
            __riscv_vse32_v_i32m4(y3 + p0, a3, vmax);
        }
    }
}

/* ---------- B：4bit 权重（每字节 2 个有符号 nibble） ---------- */
static inline int8_t q4n(const uint8_t *q, int k) {
    const uint8_t b = q[k >> 1];
    const int v = (k & 1) ? (b >> 4) : (b & 0x0F);
    return (int8_t)((int8_t)(v << 4) >> 4);  /* 先落到 8bit 再做算术右移 ⇒ 符号扩展 */
}

static void gemm_q4(const uint8_t *W, const int8_t *imc, int OC, int Kp, int P,
                    int32_t *y) {
    const size_t vmax = __riscv_vsetvlmax_e8m1();
    const int Kp2 = Kp >> 1;
    for (int oc0 = 0; oc0 + 4 <= OC; oc0 += 4) {
        const uint8_t *w0 = W + (size_t)(oc0 + 0) * Kp2;
        const uint8_t *w1 = W + (size_t)(oc0 + 1) * Kp2;
        const uint8_t *w2 = W + (size_t)(oc0 + 2) * Kp2;
        const uint8_t *w3 = W + (size_t)(oc0 + 3) * Kp2;
        int32_t *y0 = y + (size_t)(oc0 + 0) * P;
        int32_t *y1 = y + (size_t)(oc0 + 1) * P;
        int32_t *y2 = y + (size_t)(oc0 + 2) * P;
        int32_t *y3 = y + (size_t)(oc0 + 3) * P;
        int p0 = 0;
        for (; p0 + (int)vmax <= P; p0 += (int)vmax) {
            vint32m4_t a0 = __riscv_vmv_v_x_i32m4(0, vmax);
            vint32m4_t a1 = a0, a2 = a0, a3 = a0;
            const int8_t *base = imc + (size_t)p0 * Kp;
            for (int k = 0; k < Kp; k++) {
                const vint8m1_t xv =
                    __riscv_vlse8_v_i8m1(base + k, (ptrdiff_t)Kp, vmax);
                const vint16m2_t x16 = __riscv_vsext_vf2_i16m2(xv, vmax);
                a0 = __riscv_vwmacc_vx_i32m4(a0, (int16_t)q4n(w0, k), x16, vmax);
                a1 = __riscv_vwmacc_vx_i32m4(a1, (int16_t)q4n(w1, k), x16, vmax);
                a2 = __riscv_vwmacc_vx_i32m4(a2, (int16_t)q4n(w2, k), x16, vmax);
                a3 = __riscv_vwmacc_vx_i32m4(a3, (int16_t)q4n(w3, k), x16, vmax);
            }
            __riscv_vse32_v_i32m4(y0 + p0, a0, vmax);
            __riscv_vse32_v_i32m4(y1 + p0, a1, vmax);
            __riscv_vse32_v_i32m4(y2 + p0, a2, vmax);
            __riscv_vse32_v_i32m4(y3 + p0, a3, vmax);
        }
    }
}

static int cmpd(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv) {
    /* 真实列形状（OC, Kp, P）：n 走小 Kp/大 P，m/l/x 走大 Kp/小 P。
     * 覆盖 yolov8n..x 的加权分布主导项。 */
    static const int shapes[][3] = {
        { 64,   32, 6400}, { 64,   96, 6400}, { 64,  256, 6400},
        { 64,  576, 6400}, { 64, 1152, 1600},
        {192, 1728, 1600}, {256, 2304, 1600},
        {384, 3456,  400}, {512, 5184,  400},
    };
    const int ns = (int)(sizeof(shapes) / sizeof(shapes[0]));
    const int reps = (argc > 1) ? atoi(argv[1]) : 7;

    printf("VLEN/8 = %zu  reps=%d\n", __riscv_vsetvlmax_e8m1(), reps);
    printf("%5s %6s %6s %12s %12s %8s %14s\n",
           "OC", "Kp", "P", "Q8_ms", "Q4_ms", "Q4/Q8", "checksum_eq");

    for (int si = 0; si < ns; si++) {
        const int OC = shapes[si][0], Kp = shapes[si][1], P = shapes[si][2];
        int8_t  *W8 = (int8_t *)aligned_alloc(64, (size_t)OC * Kp);
        uint8_t *W4 = (uint8_t *)aligned_alloc(64, (size_t)OC * (Kp / 2));
        int8_t  *imc = (int8_t *)aligned_alloc(64, (size_t)P * Kp);
        int32_t *ya = (int32_t *)aligned_alloc(64, (size_t)OC * P * 4);
        int32_t *yb = (int32_t *)aligned_alloc(64, (size_t)OC * P * 4);
        if (!W8 || !W4 || !imc || !ya || !yb) { fprintf(stderr, "oom\n"); return 1; }

        srand(1234 + Kp);
        for (size_t i = 0; i < (size_t)OC * Kp; i++) {
            int8_t v = (int8_t)((rand() % 15) - 7);     /* 落在 4bit 可表示范围 */
            W8[i] = v;
            if ((i & 1) == 0) W4[i >> 1] = (uint8_t)(v & 0x0F);
            else              W4[i >> 1] |= (uint8_t)((v & 0x0F) << 4);
        }
        for (size_t i = 0; i < (size_t)P * Kp; i++)
            imc[i] = (int8_t)((rand() % 255) - 127);

        double ta[32], tb[32];
        for (int r = 0; r < reps; r++) {
            double t0 = now();
            gemm_q8(W8, imc, OC, Kp, P, ya);
            ta[r] = now() - t0;
            t0 = now();
            gemm_q4(W4, imc, OC, Kp, P, yb);
            tb[r] = now() - t0;
        }
        qsort(ta, reps, sizeof(double), cmpd);
        qsort(tb, reps, sizeof(double), cmpd);

        long long d = 0;
        for (size_t i = 0; i < (size_t)OC * P; i++) d += llabs((long long)ya[i] - (long long)yb[i]);

        printf("%5d %6d %6d %12.2f %12.2f %8.3f %14s\n", OC, Kp, P,
               ta[reps / 2] * 1e3, tb[reps / 2] * 1e3,
               tb[reps / 2] / ta[reps / 2], d == 0 ? "EXACT" : "DIFF");
        free(W8); free(W4); free(imc); free(ya); free(yb);
    }
    return 0;
}
