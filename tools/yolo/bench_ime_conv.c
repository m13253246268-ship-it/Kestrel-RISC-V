/**
 * bench_ime_conv.c - IME vmadot 卷积原型内核（standalone，不进主工程路径）
 *
 * 目的：验证「把生产路径 yconv_dot_rvv_p（vwmacc.vx）换成 IME vmadot 8x16 外积」
 * 是否 (a) 与生产 int32 累加位级一致，(b) 有实质加速。
 *
 * vmadot 语义（本文件经 ime_layout_probe / ime_signed_probe 实测反推）：
 *   funct3=3  ⇒ s8 × s8（有符号；funct3=0 是 u8 × u8）
 *   A 向量 = 32 字节 = 4 组 × 8 个 int8，组号 q，组内 k=0..7
 *   B 向量 = 32 字节 = 4 组 × 8 个 int8，组号 r，组内 k=0..7
 *   输出 vd (m2) = 16 个 int32，lane l = q*4 + r = dot(A[q][0..7], B[r][0..7])
 *   ⇒ 一条指令 = 4(M) × 4(N) × 8(K) = 128 MAC
 *   8 条指令（2 个 A 向量 × 4 个 B 向量）= 8(M) × 16(N) × 8(K) = 1024 MAC
 *   因为有符号直通，**不需要** ORT 那套 u8 零点补偿（行和/列和修正）。
 *
 * 量化口径与生产路径完全相同（K-YOLO-002 Phase 2）：
 *   imc[P][Kp] int8 + rowsc[P]（逐像素行 scale）+ Wq[OC][Kp] int8 + cs[OC]
 *   反量化：y = (float)acc * (cs[oc] * rowsc[p]) + bias[oc]，运算顺序同生产。
 *   acc 为精确 int32 ⇒ 与 vwmacc.vx 路径逐位同值（tiny 自检 + 整层对拍均为 0 diff）。
 *
 * 板端构建：
 *   gcc -O2 -ffp-contract=off -march=rv64gcv -mabi=lp64d -fopenmp bench_ime_conv.c -o bench_ime_conv -lm
 *   ※ -ffp-contract=off 必须加：反量化 epilogue 若被合成 fma，float 结果与生产
 *     的「vfmul + vfadd」两条指令不同（实测 model.0 出现 123012/409600 处 1ulp 差异）。
 * 自检：
 *   ./bench_ime_conv tiny
 *
 * 板端实测（K1-x MUSE-Pi-Pro，2026-10-29，min-of-4，4 个代表层，int32 与 float 全部 0 diff）：
 *   layer                ref1s   ref8s   ime1s   ime4s   ime4 vs ref8
 *   model.0 Kp=32         7.19   11.40    8.17    2.78     4.10x
 *   model.2 Kp=160       24.73   14.04   10.04    3.67     3.83x
 *   model.8 Kp=576       20.29   18.81    3.10    1.41    13.33x
 *   model.9 Kp=576       20.26   20.36    3.12    1.37    14.91x
 *   合计                 72.47   64.61   24.42    9.23     7.0x
 *   packB（未融合、纯 memcpy 循环）25.6ms —— 生产必须把它并进 im2col/量化，否则吃掉收益。
 *   ref8 = OC 方向 8 线程（生产口径）；小 OC 层因线程争用反而比单线程慢，说明
 *   现有 8 线程路径在 OC=16 的层上是亏的。
 */
#define _GNU_SOURCE
#include <math.h>
#include <sched.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <riscv_vector.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define STR(x) #x
/* funct3=3 = s8×s8（与 EP .so 主力编码一致；binutils 的 vmadotu 助记符是 funct3=0 = u8×u8） */
#define VMADOT(vd, vs1, vs2)                                                    \
    ".insn 4, " STR(0x2b | ((vd) << 7) | (3 << 12) | ((vs1) << 15) |            \
                     ((vs2) << 20) | (0x71 << 25)) "\n\t"

#define IME_MAX_T 3   /* 计时取 min-of-reps */

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e3 + (double)t.tv_nsec * 1e-6;
}

static void bind_hart(int h) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(h & 3, &set);   /* IME 仅 hart0-3 */
    sched_setaffinity(0, sizeof(set), &set);
}

/* ---------------- 生产路径等价物：vwmacc.vx，4 OC 一组 ---------------- */
/* 逻辑照抄 vllm_yolo.c:yconv_dot_rvv_p（去掉 SiLU/OMP），可选输出 int32 acc */
static void ref_dot_i8(const int8_t *Wq, const float *cs, const float *b,
                       const int8_t *imc, const float *rowsc,
                       int OC, int Kp, int P, float *y, int32_t *acc32) {
    const size_t vmax = __riscv_vsetvlmax_e8m1();   /* VLEN/8 = 32 */
    for (int oc0 = 0; oc0 + 4 <= OC; oc0 += 4) {
        const int8_t *w0 = Wq + (size_t)(oc0 + 0) * Kp;
        const int8_t *w1 = Wq + (size_t)(oc0 + 1) * Kp;
        const int8_t *w2 = Wq + (size_t)(oc0 + 2) * Kp;
        const int8_t *w3 = Wq + (size_t)(oc0 + 3) * Kp;
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
            if (acc32) {
                __riscv_vse32_v_i32m4(acc32 + (size_t)(oc0 + 0) * P + p0, a0, vmax);
                __riscv_vse32_v_i32m4(acc32 + (size_t)(oc0 + 1) * P + p0, a1, vmax);
                __riscv_vse32_v_i32m4(acc32 + (size_t)(oc0 + 2) * P + p0, a2, vmax);
                __riscv_vse32_v_i32m4(acc32 + (size_t)(oc0 + 3) * P + p0, a3, vmax);
            }
            const vfloat32m4_t rs = __riscv_vle32_v_f32m4(rowsc + p0, vmax);
            for (int u = 0; u < 4; u++) {
                const vint32m4_t acc = (u == 0) ? a0 : (u == 1) ? a1 : (u == 2) ? a2 : a3;
                __riscv_vse32_v_f32m4(
                    y + (size_t)(oc0 + u) * P + p0,
                    __riscv_vfadd_vf_f32m4(
                        __riscv_vfmul_vv_f32m4(
                            __riscv_vfcvt_f_x_v_f32m4(acc, vmax),
                            __riscv_vfmul_vf_f32m4(rs, cs[oc0 + u], vmax), vmax),
                        b ? b[oc0 + u] : 0.f, vmax),
                    vmax);
            }
        }
        /* p 尾部：标量 */
        for (; p0 < P; p0++) {
            const int8_t *ir = imc + (size_t)p0 * Kp;
            for (int u = 0; u < 4; u++) {
                const int8_t *wu = Wq + (size_t)(oc0 + u) * Kp;
                int32_t t = 0;
                for (int k = 0; k < Kp; k++) t += (int32_t)wu[k] * ir[k];
                if (acc32) acc32[(size_t)(oc0 + u) * P + p0] = t;
                y[(size_t)(oc0 + u) * P + p0] =
                    (float)t * (cs[oc0 + u] * rowsc[p0]) + (b ? b[oc0 + u] : 0.f);
            }
        }
    }
}

/* 生产口径的并行参考臂：OC 方向切分 + OpenMP（对应 vllm_yolo.c 的 if(nb>2) 并行） */
static void ref_dot_i8_mt(const int8_t *Wq, const float *cs, const float *b,
                          const int8_t *imc, const float *rowsc,
                          int OC, int Kp, int P, float *y, int nt) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(nt)
#endif
    for (int oc0 = 0; oc0 <= OC - 4; oc0 += 4) {
        const int8_t *w0 = Wq + (size_t)(oc0 + 0) * Kp;
        const int8_t *w1 = Wq + (size_t)(oc0 + 1) * Kp;
        const int8_t *w2 = Wq + (size_t)(oc0 + 2) * Kp;
        const int8_t *w3 = Wq + (size_t)(oc0 + 3) * Kp;
        const size_t vmax = __riscv_vsetvlmax_e8m1();
        for (int p0 = 0; p0 + (int)vmax <= P; p0 += (int)vmax) {
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
            for (int u = 0; u < 4; u++) {
                const vint32m4_t acc = (u == 0) ? a0 : (u == 1) ? a1 : (u == 2) ? a2 : a3;
                __riscv_vse32_v_f32m4(
                    y + (size_t)(oc0 + u) * P + p0,
                    __riscv_vfadd_vf_f32m4(
                        __riscv_vfmul_vv_f32m4(
                            __riscv_vfcvt_f_x_v_f32m4(acc, vmax),
                            __riscv_vfmul_vf_f32m4(rs, cs[oc0 + u], vmax), vmax),
                        b ? b[oc0 + u] : 0.f, vmax),
                    vmax);
            }
        }
    }
}

/* ---------------- IME 打包 ---------------- */
/* apack[mt][kb][mv][32]：byte[q*8+j] = Wq[(mt*8+mv*4+q)][kb*8+j] */
static int8_t *pack_a(const int8_t *Wq, int OC, int Kp) {
    const int mt_n = OC / 8, kb_n = Kp / 8;
    int8_t *p = (int8_t *)aligned_alloc(64, ((size_t)mt_n * kb_n * 64 + 63) & ~(size_t)63);
    for (int mt = 0; mt < mt_n; mt++)
        for (int kb = 0; kb < kb_n; kb++)
            for (int mv = 0; mv < 2; mv++)
                for (int q = 0; q < 4; q++)
                    memcpy(p + ((size_t)mt * kb_n + kb) * 64 + mv * 32 + q * 8,
                           Wq + (size_t)(mt * 8 + mv * 4 + q) * Kp + kb * 8, 8);
    return p;
}

/* bpack[nb][kb][t][32]：byte[r*8+j] = imc[(nb*16+t*4+r)][kb*8+j]，越界补 0 */
static int8_t *pack_b(const int8_t *imc, int P, int Kp) {
    const int nb_n = (P + 15) / 16, kb_n = Kp / 8;
    int8_t *p = (int8_t *)calloc((size_t)nb_n * kb_n * 128, 1);
    for (int nb = 0; nb < nb_n; nb++)
        for (int kb = 0; kb < kb_n; kb++)
            for (int t = 0; t < 4; t++)
                for (int r = 0; r < 4; r++) {
                    const int px = nb * 16 + t * 4 + r;
                    if (px < P)
                        memcpy(p + ((size_t)nb * kb_n + kb) * 128 + t * 32 + r * 8,
                               imc + (size_t)px * Kp + kb * 8, 8);
                }
    return p;
}

/* 8(M) × 16(N) × Kp：acc[8][16]，acc[mv*4+t][q*4+r] = dot(OC mt*8+mv*4+q, px nb*16+t*4+r) */
static void ime_kernel(const int8_t *a0, const int8_t *b0, int kb_n, int32_t acc[8][16]) {
    uintptr_t pa0 = (uintptr_t)a0, pa1 = (uintptr_t)(a0 + 32);
    uintptr_t pb0 = (uintptr_t)b0, pb1 = (uintptr_t)(b0 + 32);
    uintptr_t pb2 = (uintptr_t)(b0 + 64), pb3 = (uintptr_t)(b0 + 96);
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
        VMADOT(8, 0, 2)
        VMADOT(10, 0, 3)
        VMADOT(12, 0, 4)
        VMADOT(14, 0, 5)
        VMADOT(16, 1, 2)
        VMADOT(18, 1, 3)
        VMADOT(20, 1, 4)
        VMADOT(22, 1, 5)
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
        : "t0", "t1", "t2", "t3", "t4", "t5", "t6", "v0", "v1", "v2", "v3",
          "v4", "v5", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
          "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "memory");
}

/* 单线程：整层卷积 */
static void ime_conv_i8(const int8_t *apack, const int8_t *bpack,
                        const float *cs, const float *b, const float *rowsc,
                        int OC, int Kp, int P, float *y, int32_t *acc32) {
    const int mt_n = OC / 8, kb_n = Kp / 8, nb_n = (P + 15) / 16;
    for (int mt = 0; mt < mt_n; mt++) {
        for (int nb = 0; nb < nb_n; nb++) {
            int32_t acc[8][16];
            ime_kernel(apack + (size_t)mt * kb_n * 64, bpack + (size_t)nb * kb_n * 128,
                       kb_n, acc);
            for (int mv = 0; mv < 2; mv++)
                for (int t = 0; t < 4; t++)
                    for (int q = 0; q < 4; q++) {
                        const int oc = mt * 8 + mv * 4 + q;
                        const float s = cs[oc], bb = b ? b[oc] : 0.f;
                        for (int r = 0; r < 4; r++) {
                            const int px = nb * 16 + t * 4 + r;
                            if (px >= P) continue;
                            const int32_t v = acc[mv * 4 + t][q * 4 + r];
                            if (acc32) acc32[(size_t)oc * P + px] = v;
                            y[(size_t)oc * P + px] = (float)v * (s * rowsc[px]) + bb;
                        }
                    }
        }
    }
}

/* 多线程：按 nb 切分，线程绑定 hart0..n-1 */
static void ime_conv_i8_mt(const int8_t *apack, const int8_t *bpack,
                           const float *cs, const float *b, const float *rowsc,
                           int OC, int Kp, int P, float *y, int nt) {
    const int mt_n = OC / 8, kb_n = Kp / 8, nb_n = (P + 15) / 16;
#ifdef _OPENMP
#pragma omp parallel num_threads(nt)
    {
        bind_hart(omp_get_thread_num());
#pragma omp for schedule(static)
#endif
        for (int nb = 0; nb < nb_n; nb++) {
            int32_t acc[8][16];
            const int8_t *bp = bpack + (size_t)nb * kb_n * 128;
            for (int mt = 0; mt < mt_n; mt++) {
                ime_kernel(apack + (size_t)mt * kb_n * 64, bp, kb_n, acc);
                for (int mv = 0; mv < 2; mv++)
                    for (int t = 0; t < 4; t++)
                        for (int q = 0; q < 4; q++) {
                            const int oc = mt * 8 + mv * 4 + q;
                            const float s = cs[oc], bb = b ? b[oc] : 0.f;
                            for (int r = 0; r < 4; r++) {
                                const int px = nb * 16 + t * 4 + r;
                                if (px >= P) continue;
                                y[(size_t)oc * P + px] =
                                    (float)acc[mv * 4 + t][q * 4 + r] * (s * rowsc[px]) + bb;
                            }
                        }
            }
        }
#ifdef _OPENMP
    }
#endif
}

/* ---------------- 微型对拍：隔离 Kp 与符号两个变量 ---------------- */
static int tiny_case(int Kp, int rnd) {
    const int OC = 8, P = 16;
    int8_t Wq[8 * 32], imc[16 * 32];
    float cs[8], bi[8], rowsc[16];
    uint32_t st = 7u;
    for (int i = 0; i < OC * Kp; i++) {
        if (rnd) { st = st * 1664525u + 1013904223u; Wq[i] = (int8_t)(st >> 24); }
        else Wq[i] = (int8_t)((i % 16) + 1);
    }
    for (int i = 0; i < P * Kp; i++) {
        if (rnd) { st = st * 1664525u + 1013904223u; imc[i] = (int8_t)(st >> 24); }
        else imc[i] = (int8_t)((i % 8) + 1);
    }
    for (int i = 0; i < OC; i++) { cs[i] = 1.0f; bi[i] = 0.f; }
    for (int i = 0; i < P; i++) rowsc[i] = 1.0f;

    float y_ref[8 * 16], y_ime[8 * 16];
    int32_t a_ref[8 * 16], a_ime[8 * 16];
    ref_dot_i8(Wq, cs, bi, imc, rowsc, OC, Kp, P, y_ref, a_ref);

    int8_t *ap = pack_a(Wq, OC, Kp);
    int8_t *bp = pack_b(imc, P, Kp);
    ime_conv_i8(ap, bp, cs, bi, rowsc, OC, Kp, P, y_ime, a_ime);

    int bad = 0;
    for (int i = 0; i < OC * P; i++) bad += (a_ref[i] != a_ime[i]);
    printf("[tiny] Kp=%-3d rnd=%d  acc mismatch=%d/%d\n", Kp, rnd, bad, OC * P);
    free(ap); free(bp);
    return bad;
}

/* 稀疏单点探针：确认跨 k-block 的取数与累加 */
static int tiny_probe(void) {
    const int OC = 8, P = 16, Kp = 16;
    int8_t Wq[8 * 16], imc[16 * 16];
    float cs[8], bi[8], rowsc[16];
    float y[8 * 16];
    int32_t acc[8 * 16];
    struct { int wk0, wk8, ak0, ak8, want; } list[] = {
        {1, 2, 3, 4, 11}, {-1, 1, 1, 1, 0}, {127, 0, 127, 0, 16129},
        {-128, 0, -128, 0, 16384}, {-128, 127, 127, -128, -32512}, {-1, -1, -1, -1, 2},
    };
    int bad = 0;
    for (size_t c = 0; c < sizeof(list) / sizeof(list[0]); c++) {
        memset(Wq, 0, sizeof Wq);
        memset(imc, 0, sizeof imc);
        for (int i = 0; i < OC; i++) { cs[i] = 1.f; bi[i] = 0.f; }
        for (int i = 0; i < P; i++) rowsc[i] = 1.f;
        Wq[0 * Kp + 0] = (int8_t)list[c].wk0;
        Wq[0 * Kp + 8] = (int8_t)list[c].wk8;
        imc[0 * Kp + 0] = (int8_t)list[c].ak0;
        imc[0 * Kp + 8] = (int8_t)list[c].ak8;
        int8_t *ap = pack_a(Wq, OC, Kp);
        int8_t *bp = pack_b(imc, P, Kp);
        ime_conv_i8(ap, bp, cs, bi, rowsc, OC, Kp, P, y, acc);
        const int ok = (acc[0] == list[c].want);
        bad += !ok;
        printf("[probe] w(k0=%4d,k8=%4d) a(k0=%4d,k8=%4d)  ime=%-8d want=%-8d %s\n",
               list[c].wk0, list[c].wk8, list[c].ak0, list[c].ak8, acc[0],
               list[c].want, ok ? "OK" : "MISMATCH");
        free(ap); free(bp);
    }
    return bad;
}

/* ---------------- 测试层 ---------------- */
typedef struct { const char *name; int C, OC, K, OH, OW; } Layer;

static uint32_t rng_state = 12345u;
static uint32_t rng(void) { rng_state = rng_state * 1664525u + 1013904223u; return rng_state; }
static float rndf(void) { return ((int32_t)(rng() >> 8) % 20001 - 10000) / 10000.0f; }

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc > 1 && strcmp(argv[1], "tiny") == 0)
        return (tiny_probe() + tiny_case(8, 0) + tiny_case(8, 1) + tiny_case(16, 0) +
                tiny_case(16, 1) + tiny_case(32, 1)) ? 1 : 0;

    const Layer layers[] = {
        {"model.0   C3  OC16 k3s2 320->160", 3, 16, 27, 160, 160},
        {"model.2   C16 OC16 k3s1 160->160", 16, 16, 144, 160, 160},
        {"model.8   C64 OC64 k3s2  80->40 ", 64, 64, 576, 40, 40},
        {"model.9   C64 OC64 k3s1  40->40 ", 64, 64, 576, 40, 40},
    };
    const int nL = (int)(sizeof(layers) / sizeof(layers[0]));
    const int nt = 4;

    printf("=== IME vmadot 卷积原型（min-of-%d，参考臂=单线程 vwmacc.vx）===\n", IME_MAX_T + 1);
    bind_hart(0);

    double tot_ref = 0, tot_ime1 = 0, tot_ime4 = 0, tot_pack = 0;
    long tot_mac = 0;
    long tot_bad = 0;

    for (int li = 0; li < nL; li++) {
        const Layer *L = &layers[li];
        const int P = L->OH * L->OW;
        const int Kp = (L->K + 31) / 32 * 32;

        int8_t *Wq = (int8_t *)aligned_alloc(64, ((size_t)L->OC * Kp + 63) & ~(size_t)63);
        int8_t *imc = (int8_t *)aligned_alloc(64, ((size_t)P * Kp + 63) & ~(size_t)63);
        float *cs = (float *)malloc((size_t)L->OC * sizeof(float));
        float *bi = (float *)malloc((size_t)L->OC * sizeof(float));
        float *rowsc = (float *)malloc((size_t)P * sizeof(float));
        float *y_ref = (float *)malloc((size_t)L->OC * P * sizeof(float));
        float *y_ime = (float *)malloc((size_t)L->OC * P * sizeof(float));
        int32_t *acc_ref = (int32_t *)malloc((size_t)L->OC * P * sizeof(int32_t));
        int32_t *acc_ime = (int32_t *)malloc((size_t)L->OC * P * sizeof(int32_t));

        for (size_t i = 0; i < (size_t)L->OC * Kp; i++) Wq[i] = (int8_t)((rng() >> 16) & 0xFF);
        for (size_t i = 0; i < (size_t)P * Kp; i++) imc[i] = (int8_t)((rng() >> 16) & 0xFF);
        for (int i = 0; i < L->OC; i++) { cs[i] = 0.002f + 0.01f * (rng() & 0xFF) / 255.f; bi[i] = rndf(); }
        for (int i = 0; i < P; i++) rowsc[i] = 0.001f + 0.01f * (rng() & 0xFF) / 255.f;

        int8_t *apack = pack_a(Wq, L->OC, Kp);
        int8_t *bpack = pack_b(imc, P, Kp);

        double t_ref = 1e30, t_ref8 = 1e30, t_ime1 = 1e30, t_ime4 = 1e30, t_pack = 1e30;
        long d1 = -1, d4 = -1;
        for (int r = 0; r <= IME_MAX_T; r++) {
            double t0 = now_ms();
            ref_dot_i8(Wq, cs, bi, imc, rowsc, L->OC, Kp, P, y_ref, acc_ref);
            double d = now_ms() - t0; if (d < t_ref) t_ref = d;

            t0 = now_ms();
            ref_dot_i8_mt(Wq, cs, bi, imc, rowsc, L->OC, Kp, P, y_ime, 8);
            d = now_ms() - t0; if (d < t_ref8) t_ref8 = d;

            t0 = now_ms();
            ime_conv_i8(apack, bpack, cs, bi, rowsc, L->OC, Kp, P, y_ime,
                        r == 0 ? acc_ime : NULL);
            d = now_ms() - t0; if (d < t_ime1) t_ime1 = d;
            if (r == 0) {
                long da = 0;
                for (size_t i = 0; i < (size_t)L->OC * P; i++) da += (acc_ref[i] != acc_ime[i]);
                d1 = 0;
                for (size_t i = 0; i < (size_t)L->OC * P; i++)
                    d1 += (memcmp(&y_ref[i], &y_ime[i], sizeof(float)) != 0);
                if (da || d1) printf("    [dbg] %s acc_diff=%ld y_diff=%ld\n", L->name, da, d1);
            }

            t0 = now_ms();
            ime_conv_i8_mt(apack, bpack, cs, bi, rowsc, L->OC, Kp, P, y_ime, nt);
            d = now_ms() - t0; if (d < t_ime4) t_ime4 = d;
            if (r == 0) {
                d4 = 0;
                for (size_t i = 0; i < (size_t)L->OC * P; i++)
                    d4 += (memcmp(&y_ref[i], &y_ime[i], sizeof(float)) != 0);
            }

            t0 = now_ms();
            free(pack_b(imc, P, Kp));
            d = now_ms() - t0; if (d < t_pack) t_pack = d;
        }

        printf("%s P=%-5d Kp=%-4d ref1=%7.2fms ref8=%6.2fms ime1=%6.2fms ime4=%6.2fms "
               "(vs ref8 %.2fx) packB=%6.2fms diff=%ld/%ld\n",
               L->name, P, Kp, t_ref, t_ref8, t_ime1, t_ime4, t_ref8 / t_ime4,
               t_pack, d1, d4);
        tot_bad += d1 + d4;
        const double mac = (double)L->OC * P * Kp;

        tot_ref += t_ref; tot_ime1 += t_ime1; tot_ime4 += t_ime4; tot_pack += t_pack;
        tot_mac += (long)mac;
        free(Wq); free(imc); free(cs); free(bi); free(rowsc);
        free(y_ref); free(y_ime); free(acc_ref); free(apack); free(bpack);
    }

    printf("\n合计 4 层: ref=%.2fms  ime1=%.2fms  ime4=%.2fms  packB=%.2fms  "
           "MAC=%.3f GMAC  y_diff(total)=%ld\n",
           tot_ref, tot_ime1, tot_ime4, tot_pack, tot_mac / 1e9, tot_bad);
    printf("吞吐: ref=%.1f  ime1=%.1f  ime4=%.1f G MAC/s\n",
           tot_mac / tot_ref / 1e6, tot_mac / tot_ime1 / 1e6, tot_mac / tot_ime4 / 1e6);
    return tot_bad ? 1 : 0;
}
