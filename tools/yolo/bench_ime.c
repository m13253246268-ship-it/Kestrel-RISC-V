/**
 * bench_ime.c - SpacemiT K1 IME (vmadot) 真实吞吐微基准
 *
 * 目的：解决 V46 与 V47 的自家数据矛盾——
 *   V46：带 L1 访存单核 58 G MAC/s（4×vle8 + 4×vmadot）
 *   V47：真实 GEMM 内核仅 3.2~5.9 G MAC/s/核，4 线程微基准最高 21.8 G
 *
 * 分档：
 *   t0  纯 ALU / 单累加器（测 vmadot 延迟）
 *   t1  纯 ALU / 8 独立累加器（测发射吞吐）
 *   t2  4×vle8 + 4×vmadot（V46 口径；地址由编译器算，降低手写汇编风险）
 *   t3  4×vle32 + 4×vmadot（ORT ime1 取数形态）
 *   t4  1/2/4 线程 t1（仅 hart0-3）
 *   t5  RVV 对照（4×vle8 + 4×vwmacc）
 *
 * 指令编码（板端 objdump 实测，见指导书 V55）：
 *   vmadot   = 0xe207382b => opcode 0x2b | vd<<7 | funct3(3)<<12 | vs1<<15 | vs2<<20 | 0x71<<25
 * 用 .insn 4,<32bit> 直接发机器码（binutils 对 custom-1 的 .insn r 形式会误判 funct2）。
 *
 * MAC/vmadot 取 128（V46/V47 共识，m1×m1）。
 * 核绑定：IME 仅 hart0-3 可用（V47：hart4-7 执行 vmadot 直接 SIGILL）。
 *
 * 构建（板端）：gcc -O2 -march=rv64gcv -mabi=lp64d -fopenmp bench_ime.c -o bench_ime
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <sched.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define MAC_PER_VMADOT 128
#define ITER_ALU   2000000L
#define ITER_LOAD   100000L

#define _STR(x) #x
#define _XSTR(x) _STR(x)
#define VMADOT(vd, vs1, vs2)                                                    \
    ".insn 4, " _XSTR(0x2b | ((vd) << 7) | (3 << 12) | ((vs1) << 15) |          \
                        ((vs2) << 20) | (0x71 << 25)) "\n\t"

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static int bind_core(int c) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(c & 3, &set);
    return sched_setaffinity(0, sizeof(set), &set);
}

/* t0：单累加器（测依赖延迟） */
static double ime_dep(long n) {
    double t = now_s();
    __asm__ volatile(
        "vsetvli t0, zero, e8, m1, tu, mu \n\t"
        "vmv.v.i v8, 1                    \n\t"
        "vmv.v.i v9, 1                    \n\t"
        "vmv.v.i v16, 0                   \n\t"
        ".align 4                         \n\t"
        "1:                               \n\t"
        VMADOT(16, 8, 9)
        "addi %[n], %[n], -1              \n\t"
        "bnez %[n], 1b                    \n\t"
        : [n] "+r"(n) : : "t0", "v8", "v9", "v16", "cc");
    return now_s() - t;
}

/* t1：8 个独立累加器（测发射吞吐） */
static double ime_indep(long n) {
    double t = now_s();
    __asm__ volatile(
        "vsetvli t0, zero, e8, m1, tu, mu \n\t"
        "vmv.v.i v8, 1                    \n\t"
        "vmv.v.i v9, 1                    \n\t"
        "vmv.v.i v16, 0                   \n\t"
        "vmv.v.i v18, 0                   \n\t"
        "vmv.v.i v20, 0                   \n\t"
        "vmv.v.i v22, 0                   \n\t"
        "vmv.v.i v24, 0                   \n\t"
        "vmv.v.i v26, 0                   \n\t"
        "vmv.v.i v28, 0                   \n\t"
        "vmv.v.i v30, 0                   \n\t"
        ".align 4                         \n\t"
        "1:                               \n\t"
        VMADOT(16, 8, 9)
        VMADOT(18, 8, 9)
        VMADOT(20, 8, 9)
        VMADOT(22, 8, 9)
        VMADOT(24, 8, 9)
        VMADOT(26, 8, 9)
        VMADOT(28, 8, 9)
        VMADOT(30, 8, 9)
        "addi %[n], %[n], -1              \n\t"
        "bnez %[n], 1b                    \n\t"
        : [n] "+r"(n)
        :
        : "t0", "v8", "v9", "v16", "v18", "v20", "v22",
          "v24", "v26", "v28", "v30", "cc");
    return now_s() - t;
}

/* t2：vle8 + vmadot；地址由 C 计算并掩码在 2KB 内，最内层用 .insn */
static double ime_load8(const int8_t *a, const int8_t *b, long n) {
    double t = now_s();
    for (long i = 0; i < n; i++) {
        const int8_t *pa = a + ((i * 64) & 2047);
        const int8_t *pb = b + ((i * 64) & 2047);
        __asm__ volatile(
            "vsetvli t0, zero, e8, m1, tu, mu \n\t"
            "vle8.v v8,  (%[pa])              \n\t"
            "vle8.v v9,  (%[pa])              \n\t"
            "vle8.v v10, (%[pb])              \n\t"
            "vle8.v v11, (%[pb])              \n\t"
            VMADOT(16, 8, 10)
            VMADOT(18, 8, 11)
            VMADOT(20, 9, 10)
            VMADOT(22, 9, 11)
            :
            : [pa] "r"(pa), [pb] "r"(pb)
            : "t0", "v8", "v9", "v10", "v11",
              "v16", "v18", "v20", "v22", "memory");
    }
    return now_s() - t;
}

/* t3：vle32 + vmadot（ORT 取数形态） */
static double ime_load32(const int32_t *a, const int32_t *b, long n) {
    double t = now_s();
    for (long i = 0; i < n; i++) {
        const int32_t *pa = a + ((i * 16) & 511);
        const int32_t *pb = b + ((i * 16) & 511);
        __asm__ volatile(
            /* vle32 必须配 e32/m1（SEW=32），vmadot 配 e8/m1 —— ORT 源码同样来回切换 */
            "vsetvli t0, zero, e32, m1, tu, mu\n\t"
            "vle32.v v8,  (%[pa])             \n\t"
            "vle32.v v9,  (%[pa])             \n\t"
            "vle32.v v10, (%[pb])             \n\t"
            "vle32.v v11, (%[pb])             \n\t"
            "vsetvli t0, zero, e8, m1, tu, mu \n\t"
            VMADOT(16, 8, 10)
            VMADOT(18, 8, 11)
            VMADOT(20, 9, 10)
            VMADOT(22, 9, 11)
            :
            : [pa] "r"(pa), [pb] "r"(pb)
            : "t0", "v8", "v9", "v10", "v11",
              "v16", "v18", "v20", "v22", "memory");
    }
    return now_s() - t;
}

/* t5：RVV 对照，4×vle8 + 4×vwmacc */
static double rvv_load(const int8_t *a, const int8_t *b, long n) {
    double t = now_s();
    for (long i = 0; i < n; i++) {
        const int8_t *pa = a + ((i * 64) & 2047);
        const int8_t *pb = b + ((i * 64) & 2047);
        __asm__ volatile(
            "vsetvli t0, zero, e8, m1, tu, mu \n\t"
            "vle8.v v8,  (%[pa])              \n\t"
            "vle8.v v9,  (%[pb])              \n\t"
            "vsetvli t0, zero, e16, m2, tu, mu\n\t"
            "vsext.vf2 v16, v8                \n\t"
            "vsext.vf2 v18, v9                \n\t"
            "vsetvli t0, zero, e32, m4, tu, mu\n\t"
            "vwmacc.vv v20, v16, v18          \n\t"
            "vwmacc.vv v24, v16, v18          \n\t"
            :
            : [pa] "r"(pa), [pb] "r"(pb)
            : "t0", "v8", "v9", "v16", "v18", "v20", "v24", "memory");
    }
    return now_s() - t;
}

static double min3_ime_indep(long n) {
    double best = 1e30;
    for (int r = 0; r < 3; r++) {
        double d = ime_indep(n);
        if (d < best) best = d;
    }
    return best;
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("=== IME vmadot 吞吐微基准（K1）MAC/vmadot=%d ===\n", MAC_PER_VMADOT);

    if (bind_core(0) != 0) printf("WARN: bind hart0 failed\n");

    double d0 = ime_dep(ITER_ALU);
    double d1 = min3_ime_indep(ITER_ALU);
    printf("[单核 hart0 / 纯 ALU]\n");
    printf("  t0 单累加器(延迟)  : %8.3f ms  %6.3f G instr/s  %7.2f G MAC/s\n",
           d0 * 1e3, ITER_ALU / d0 / 1e9, ITER_ALU / d0 * MAC_PER_VMADOT / 1e9);
    printf("  t1 8 独立累加器    : %8.3f ms  %6.3f G instr/s  %7.2f G MAC/s\n",
           d1 * 1e3, ITER_ALU * 8 / d1 / 1e9,
           ITER_ALU * 8 / d1 * MAC_PER_VMADOT / 1e9);

    const size_t SZ = 8192;
    int8_t  *a8 = (int8_t *)aligned_alloc(64, SZ);
    int8_t  *b8 = (int8_t *)aligned_alloc(64, SZ);
    int32_t *a32 = (int32_t *)aligned_alloc(64, SZ);
    int32_t *b32 = (int32_t *)aligned_alloc(64, SZ);
    if (!a8 || !b8 || !a32 || !b32) { printf("OOM\n"); return 1; }
    memset(a8, 1, SZ); memset(b8, 1, SZ);
    memset(a32, 1, SZ); memset(b32, 1, SZ);

    printf("[单核 hart0 / 带 L1 访存]\n");
    {
        double d2 = ime_load8(a8, b8, ITER_LOAD);
        printf("  t2 4×vle8  (V46 口径) : %8.3f ms  %6.3f G instr/s  %7.2f G MAC/s\n",
               d2 * 1e3, ITER_LOAD * 4 / d2 / 1e9,
               ITER_LOAD * 4 / d2 * MAC_PER_VMADOT / 1e9);
    }
    {
        double d3 = ime_load32(a32, b32, ITER_LOAD);
        printf("  t3 4×vle32 (ORT 取数) : %8.3f ms  %6.3f G instr/s  %7.2f G MAC/s\n",
               d3 * 1e3, ITER_LOAD * 4 / d3 / 1e9,
               ITER_LOAD * 4 / d3 * MAC_PER_VMADOT / 1e9);
    }
    /* t5 RVV 对照已移除：本基准关注 IME 上限，RVV 基线已有生产实测
     * （8 线程点积 17~25 G MAC/s，见 V51/V54 bench_dotprod）。 */

    printf("[多线程 IME（仅 hart0-3）纯 ALU]\n");
    for (int nt = 1; nt <= 4; nt *= 2) {
        double best = 1e30;
        /* 每线程跑满 ITER_ALU，避免并行启动开销主导；min-of-3 */
        for (int rep = 0; rep < 3; rep++) {
            double t = now_s();
#ifdef _OPENMP
#pragma omp parallel num_threads(nt)
            {
                bind_core(omp_get_thread_num());
                (void)ime_indep(ITER_ALU);
            }
#else
            (void)ime_indep(ITER_ALU);
#endif
            double el = now_s() - t;
            if (el < best) best = el;
        }
        double tot = (double)nt * ITER_ALU * 8;
        printf("  t4 %d 线程            : %8.3f ms  %6.3f G instr/s  %7.2f G MAC/s\n",
               nt, best * 1e3, tot / best / 1e9, tot / best * MAC_PER_VMADOT / 1e9);
    }

    free(a8); free(b8); free(a32); free(b32);
    printf("\n判读：t1 单核若 ≈1.5 G instr/s（≈200 G MAC/s），则 4 核上限远高于 V47 的 21.8 G；\n");
    printf("      t2/t3 若仍保持 >0.4 G instr/s（>50 G MAC/s），则 ORT 路线有真实空间。\n");
    return 0;
}
