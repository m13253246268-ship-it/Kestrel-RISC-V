/* ================================================================
 * bench_ime_llm.c — LLM prefill GEMM：IME(vmadot) vs RVV(vwmacc) 吞吐界定
 *
 * 目的：在**不改生产代码**的前提下，用 Qwen2.5-0.5B 的真实 GEMM 形状回答
 *   「把 LLM prefill 的 int8 GEMM 换到 IME，能拿到多少吞吐」。
 *
 * 口径（诚实声明）：
 *   - 两侧权重/激活都**已预置为 int8**，两侧都只做「整数 GEMM + 存 int32」，
 *     不含 Q4 nibble 解包、不含反量化、不含 SiLU、不含 lm_head。
 *   - RVV 侧 = 生产同形的 m0 内核：4 oc 共享输入、p 方向 32 lane、
 *     imc 行主序 `imc[p][k]` 用 `vlse8` 跨步取，权重按 oc 行标量广播。
 *   - IME 侧 = YOLO V56 已在生产验证过的 8(M)×16(N)×Kp 微内核，N 方向取 token。
 *   - RVV 跑 8 线程；IME 只能跑 hart0-3（vmadot 在 hart4-7 直接 SIGILL）。
 *   - 打包（A/B 重排）成本**不计入计时**，单独报，因为权重重排应落在
 *     转换期一次性完成（V28 已实测运行时转置慢 9.5×）。
 *
 * 形状（Qwen2.5-0.5B：dim=896 ffn=4864 heads=14 kv=2 hd=64，24 层）：
 *   qkv      OC=1152 (=896+128+128)  K=896
 *   o_proj   OC=896                  K=896
 *   gate_up  OC=9728 (=2*4864)       K=896
 *   down     OC=896                  K=4864
 *
 * 编译（板端）：gcc -O2 -march=rv64gcv -mabi=lp64d -ffp-contract=off -fopenmp
 * 运行：taskset 不设置；IME 内部自行绑 hart0-3，退出后还原。
 * ================================================================ */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>
#include <time.h>
#include <sched.h>
#include <omp.h>
#include <riscv_vector.h>

#define KPAD(k)   (((k) + 7) & ~7)     /* IME 要求 K 为 8 的倍数 */

#define YIME_STR(x) #x
#define YIME_VMADOT(vd, vs1, vs2)                                                \
    ".insn 4, " YIME_STR(0x2b | ((vd) << 7) | (3 << 12) | ((vs1) << 15) |        \
                         ((vs2) << 20) | (0x71 << 25)) "\n\t"

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ---------------- IME 侧 ---------------- */

static void ime_pin(int h) {
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET(h & 3, &s);
    sched_setaffinity(0, sizeof(s), &s);
}

/* 8(M) × 16(N) × kb_n × 8(K)；acc[mv*4+t][q*4+r] */
static void ime_kernel(const int8_t *a, const int8_t *b, int kb_n, int32_t acc[8][16]) {
    uintptr_t pa0 = (uintptr_t)a,      pa1 = (uintptr_t)(a + 32);
    uintptr_t pb0 = (uintptr_t)b,      pb1 = (uintptr_t)(b + 32);
    uintptr_t pb2 = (uintptr_t)(b + 64), pb3 = (uintptr_t)(b + 96);
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
        /* e32/m8 = 256 bit × 8 = 2048 bit = 64 int32，8 行 × 16 列 刚好铺满 */
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

/* A: W[OC][K] -> [mt][kb][mv][32]；B: imc[P][K] -> [nb][kb][t][32] */
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
    const int nb_n = (P + 15) / 16, kb_n = Kp / 8;
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

/* 输出 acc[oc][p]（int32），p 方向 16 一组的顺序与 IME lane 映射一致 */
static void gemm_ime(const int8_t *apack, const int8_t *bpack, int OC, int Kp, int P,
                     int32_t *out, int nt) {
    const int mt_n = OC / 8, kb_n = Kp / 8, nb_n = P / 16;
    double t0 = now_s();
    (void)t0;
#pragma omp parallel num_threads(nt)
    {
        cpu_set_t old;
        const int have_old = (sched_getaffinity(0, sizeof(old), &old) == 0);
        ime_pin(omp_get_thread_num());
#pragma omp for schedule(static)
        for (int nb = 0; nb < nb_n; nb++) {
            const int8_t *bp = bpack + (size_t)nb * kb_n * 128;
            for (int mt = 0; mt < mt_n; mt++) {
                int32_t acc[8][16];
                ime_kernel(apack + (size_t)mt * kb_n * 64, bp, kb_n, acc);
                for (int mv = 0; mv < 2; mv++)
                    for (int q = 0; q < 4; q++) {
                        const int oc = mt * 8 + mv * 4 + q;
                        for (int t = 0; t < 4; t++)
                            memcpy(out + (size_t)oc * P + nb * 16 + t * 4,
                                   acc[mv * 4 + t] + q * 4, 4 * sizeof(int32_t));
                    }
            }
        }
        if (have_old) sched_setaffinity(0, sizeof(old), &old);
    }
}

/* ---------------- RVV 侧（生产 m0 同形） ---------------- */

/* W[oc][Kp] × imc[P][Kp]，p 方向 vlmax(e8m1)=32 lane，4 oc 共享一次取数 */
static void gemm_rvv(const int8_t *W, const int8_t *imc, int OC, int Kp, int P,
                     int32_t *out, int nt) {
#pragma omp parallel num_threads(nt)
    {
#pragma omp for schedule(static)
        for (int oc0 = 0; oc0 < OC; oc0 += 4) {
            for (int p0 = 0; p0 < P; p0 += 32) {
                const int8_t *i0 = imc + (size_t)p0 * Kp;
                const size_t vl = __riscv_vsetvl_e8m1((size_t)(P - p0 < 32 ? P - p0 : 32));
                vint32m4_t a0 = __riscv_vmv_v_x_i32m4(0, vl);
                vint32m4_t a1 = a0, a2 = a0, a3 = a0;
                for (int k = 0; k < Kp; k++) {
                    const int8_t w0 = W[(size_t)(oc0 + 0) * Kp + k];
                    const int8_t w1 = W[(size_t)(oc0 + 1) * Kp + k];
                    const int8_t w2 = W[(size_t)(oc0 + 2) * Kp + k];
                    const int8_t w3 = W[(size_t)(oc0 + 3) * Kp + k];
                    vint8m1_t x = __riscv_vlse8_v_i8m1(i0 + k, (ptrdiff_t)Kp, vl);
                    vint16m2_t x16 = __riscv_vsext_vf2_i16m2(x, vl);
                    a0 = __riscv_vwmacc_vx_i32m4(a0, (int16_t)w0, x16, vl);
                    a1 = __riscv_vwmacc_vx_i32m4(a1, (int16_t)w1, x16, vl);
                    a2 = __riscv_vwmacc_vx_i32m4(a2, (int16_t)w2, x16, vl);
                    a3 = __riscv_vwmacc_vx_i32m4(a3, (int16_t)w3, x16, vl);
                }
                __riscv_vse32_v_i32m4(out + (size_t)(oc0 + 0) * P + p0, a0, vl);
                __riscv_vse32_v_i32m4(out + (size_t)(oc0 + 1) * P + p0, a1, vl);
                __riscv_vse32_v_i32m4(out + (size_t)(oc0 + 2) * P + p0, a2, vl);
                __riscv_vse32_v_i32m4(out + (size_t)(oc0 + 3) * P + p0, a3, vl);
            }
        }
    }
}

/* ---------------- 主程序 ---------------- */

typedef struct { const char *name; int OC, K; } shape_t;
static const shape_t SH[] = {
    { "qkv    (1152x896)", 1152,  896 },
    { "o      ( 896x896)",  896,  896 },
    { "gate_up(9728x896)", 9728,  896 },
    { "down   ( 896x4864)", 896, 4864 },
};
#define NSH (int)(sizeof(SH) / sizeof(SH[0]))
#define NLAYER 24

int main(int argc, char **argv) {
    const int P_list[] = { 32, 128, 512 };
    const int reps = (argc > 1) ? atoi(argv[1]) : 3;
    setvbuf(stdout, NULL, _IONBF, 0);

    printf("=== LLM prefill GEMM: IME(4t hart0-3) vs RVV(8t), int8->int32 ===\n");
    printf("omax=%d, reps=%d, layers=%d\n", omp_get_max_threads(), reps, NLAYER);

    for (int pi = 0; pi < 3; pi++) {
        const int P = P_list[pi];
        double macs_total = 0;
        printf("\n---- P=%d tokens ----\n", P);
        printf("%-18s %10s %12s %12s %9s %9s %8s\n",
               "shape", "MAC(M)", "rvv8 ms", "ime4 ms", "rvv G/s", "ime G/s", "ime/rvv");
        double sum_rvv = 0, sum_ime = 0, sum_macs = 0;

        for (int s = 0; s < NSH; s++) {
            const int OC = SH[s].OC, Kp = KPAD(SH[s].K);
            const size_t wn = (size_t)OC * Kp, an = (size_t)P * Kp;
            int8_t *W = (int8_t *)malloc(wn), *A = (int8_t *)malloc(an);
            int32_t *o1 = (int32_t *)malloc((size_t)OC * P * 4);
            int32_t *o2 = (int32_t *)malloc((size_t)OC * P * 4);
            for (size_t i = 0; i < wn; i++) W[i] = (int8_t)((i * 37u) % 251u - 125);
            for (size_t i = 0; i < an; i++) A[i] = (int8_t)((i * 91u) % 253u - 126);

            int8_t *ap = ime_pack_a(W, OC, Kp);
            int8_t *bp = ime_pack_b(A, P, Kp);
            /* r1 修正：gemm_rvv/gemm_ime 每次只执行 1 层，原 `* NLAYER` 使 G/s 虚高 24×
             * （V59/V60 的绝对值据此错误，相对倍数 ime/rvv 不受影响）。
             * 全模型折算仍按 24 层，见 macs_total。 */
            double macs = (double)OC * Kp * P;
            macs_total += (double)OC * Kp * P * NLAYER;

            double best_r = 1e9, best_i = 1e9, best_r1 = 1e9;
            for (int r = 0; r < reps; r++) {
                double t = now_s();
                gemm_rvv(W, A, OC, Kp, P, o1, 8);
                t = now_s() - t;
                if (t < best_r) best_r = t;
            }
            /* 单线程对照：用来核对「每核 G MAC/s」是否超过单发射上限（32 MAC@1.6GHz=51 G） */
            for (int r = 0; r < reps; r++) {
                double t = now_s();
                gemm_rvv(W, A, OC, Kp, P, o1, 1);
                t = now_s() - t;
                if (t < best_r1) best_r1 = t;
            }
            for (int r = 0; r < reps; r++) {
                double t = now_s();
                gemm_ime(ap, bp, OC, Kp, P, o2, 4);
                t = now_s() - t;
                if (t < best_i) best_i = t;
            }

            /* 正确性：两侧 int8 整数 GEMM 必须逐元素相同 */
            long long bad = 0, maxd = 0;
            for (size_t i = 0; i < (size_t)OC * P; i++) {
                long long d = (long long)o1[i] - (long long)o2[i];
                if (d) { bad++; if (llabs(d) > maxd) maxd = llabs(d); }
            }

            sum_rvv  += best_r;  sum_ime += best_i;  sum_macs += macs;
            printf("%-18s %10.2f %12.2f %12.2f %9.2f %9.2f %8.2fx",
                   SH[s].name, macs / 1e6, best_r * 1e3, best_i * 1e3,
                   macs / best_r / 1e9, macs / best_i / 1e9, best_r / best_i);
            printf("  [rvv1t %.1f G/s = %.1f G/core]\n", macs / best_r1 / 1e9, macs / best_r1 / 1e9);
            if (bad) printf("    !! %lld/%zu elements differ (max|d|=%lld)\n",
                            bad, (size_t)OC * P, maxd);
            free(W); free(A); free(o1); free(o2); free(ap); free(bp);
        }
        printf("%-18s %10.2f %12.2f %12.2f %9.2f %9.2f %8.2fx\n",
               "TOTAL(per layer)", sum_macs / 1e6, sum_rvv * 1e3, sum_ime * 1e3,
               sum_macs / sum_rvv / 1e9, sum_macs / sum_ime / 1e9, sum_rvv / sum_ime);
        printf("  -> P=%d 每 token 摊 %.3f G MAC\n", P, macs_total / P / 1e9);
    }
    return 0;
}
