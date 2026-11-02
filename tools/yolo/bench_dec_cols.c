/* ================================================================
 * bench_dec_cols.c — 形状扫描：K（列数）对 vmadot 权重流有效带宽的影响
 *
 * 背景（方案指导书 §12.11）：生产 TPROF 显示 o(4.48) / down(4.31) 的有效带宽
 * 比 gu(5.02) / qkv(4.94) 低 10~14%。o 与 down 的共同点是 cols 大（4096/9728），
 * 每 tile 要重读的激活 xr = nblk×128B（16KB/38KB），超出单核 L1。
 *
 * 列（同一权重、同一调度，只改激活读取方式）：
 *   vm     = 2× vm_tile16（生产同款，xr 每 32 行读两遍）
 *   fx     = 同 vm 但 x 指针冻结在第 0 块（128B/块照读，足迹 128B）→ 只测足迹
 *   v32    = vm32_np（32 行融合，xr 每 32 行只读一遍）
 *   v32sp  = vm32_sp（32 行融合 + 软件流水）
 * 配置：
 *   G1 14MB 等字节 K=1024/2560/4096/9728（xr 4/10/16/38 KB）
 *   G2 放大 rows：把「足迹」与「tile 数量/字节数」分开
 *   G3 K=9728 加 resid，排除残差项
 *   BW 同尺寸 4 线程纯读上限
 * 编译：gcc -O2 -march=rv64gcv -mabi=lp64d -ffp-contract=off -fopenmp \
 *          bench_dec_cols.c -o bench_dec_cols
 * ================================================================ */
#define main bench_dec_vmadot_main
#include "bench_dec_vmadot.c"
#undef main

/* ---- 以下宏与 bench_dec_vm2.c 同源：32 行融合内核（xr 每 32 行只读一遍） ---- */
#define VM_HALF(W, OFF, XA, XB)                                            \
    "vsetvli t0, zero, e8, m4, ta, ma\n\t"                                 \
    "addi t1, " W ", " OFF "\n\t"                                          \
    "vle8.v v12, (t1)\n\t"                                                 \
    "vsrl.vi v8, v12, 4\n\t"                                               \
    "vand.vi v12, v12, 15\n\t"                                             \
    "vsetvli t0, zero, e8, m1, ta, ma\n\t"                                 \
    RV_VMADOT(16, XA, 12) RV_VMADOT(18, XA, 13) RV_VMADOT(20, XA, 14) RV_VMADOT(22, XA, 15) \
    RV_VMADOT(16, XB, 8)  RV_VMADOT(18, XB, 9)  RV_VMADOT(20, XB, 10) RV_VMADOT(22, XB, 11)

#define VM_TILE(W, ACC)                                                    \
    "vsetvli t0, zero, e32, m8, ta, ma\n\t"                                \
    "vmv.v.i v16, 0\n\t"                                                   \
    VM_HALF(W, "32", 4, 5)                                                 \
    VM_HALF(W, "160", 6, 7)                                                \
    "vsetivli zero, 12, e32, m2, tu, ma\n\t"                               \
    "vmv.v.v v22, v20\n\t"                                                 \
    "vsetivli zero, 8, e32, m2, tu, ma\n\t"                                \
    "vmv.v.v v22, v18\n\t"                                                 \
    "vsetivli zero, 4, e32, m2, tu, ma\n\t"                                \
    "vmv.v.v v22, v16\n\t"                                                 \
    "vsetivli zero, 16, e32, m2, ta, ma\n\t"                               \
    "vsub.vx v22, v22, t2\n\t"                                             \
    "vfcvt.f.x.v v22, v22\n\t"                                             \
    "vsetivli zero, 16, e16, m1, ta, ma\n\t"                               \
    "vle16.v v26, (" W ")\n\t"                                             \
    "vfwcvt.f.f.v v28, v26\n\t"                                            \
    "vsetivli zero, 16, e32, m2, ta, ma\n\t"                               \
    "vfmul.vv v28, v28, v22\n\t"                                           \
    "vfmul.vf v28, v28, ft0\n\t"                                           \
    "vfadd.vv " ACC ", " ACC ", v28\n\t"

#define VM_CLOB "t0", "t1", "t2", "t3", "ft0", "memory", "vl", "vtype",                       \
    "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11",                  \
    "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",        \
    "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"

#define VM32_FN(NAME)                                                                         \
__attribute__((noinline)) static void NAME(const uint8_t *w, const int8_t *xr, const float *xd, \
        const int32_t *corr, int nblk, float *out, long pfd) {                                \
    long k = nblk; const uint8_t *wb = w + (size_t)nblk * 288; (void)pfd;                      \
    __asm__ volatile(                                                                         \
        ".option push\n\t.option arch, +zvfhmin\n\t"                                          \
        "vsetivli zero, 16, e32, m2, ta, ma\n\t"                                              \
        "vmv.v.i v24, 0\n\t"                                                                  \
        "vmv.v.i v30, 0\n\t"                                                                  \
        "1:\n\t"                                                                              \
        "vsetvli t0, zero, e8, m4, ta, ma\n\t"                                                \
        "vle8.v v4, (%[x])\n\t"                                                               \
        "lw t2, 0(%[c])\n\t"                                                                  \
        "flw ft0, 0(%[d])\n\t"                                                                \
        VM_TILE("%[wa]", "v24")                                                               \
        VM_TILE("%[wb]", "v30")                                                               \
        "addi %[wa], %[wa], 288\n\t"                                                          \
        "addi %[wb], %[wb], 288\n\t"                                                          \
        "addi %[x], %[x], 128\n\t"                                                            \
        "addi %[d], %[d], 4\n\t"                                                              \
        "addi %[c], %[c], 4\n\t"                                                              \
        "addi %[k], %[k], -1\n\t"                                                             \
        "vsetvli t0, zero, e8, m4, ta, ma\n\t"                                                \
        "bnez %[k], 1b\n\t"                                                                   \
        "vsetivli zero, 16, e32, m2, ta, ma\n\t"                                              \
        "vse32.v v24, (%[o])\n\t"                                                             \
        "vse32.v v30, (%[o2])\n\t"                                                            \
        ".option pop\n\t"                                                                     \
        : [wa] "+r"(w), [wb] "+r"(wb), [x] "+r"(xr), [d] "+r"(xd), [c] "+r"(corr), [k] "+r"(k) \
        : [o] "r"(out), [o2] "r"(out + 16)                                                    \
        : VM_CLOB);                                                                           \
}

/* vm_tile16 的可参数化副本：XLOAD 换激活装载，XADV 换 x 指针前进量 */
#define VM16V_FN(NAME, XLOAD, XADV)                                              \
__attribute__((noinline)) static void NAME(const uint8_t *w, const int8_t *xr,   \
        const float *xd, const int32_t *corr, int nblk, float *out) {            \
    long k = nblk;                                                               \
    __asm__ volatile(                                                            \
        ".option push                          \n\t"                              \
        ".option arch, +zvfhmin                \n\t"                              \
        "vsetivli    zero, 16, e32, m2, ta, ma \n\t"                              \
        "vmv.v.i     v24, 0                    \n\t"                              \
        "1:                                    \n\t"                              \
        "vsetvli     t0, zero, e32, m8, ta, ma \n\t"                              \
        "vmv.v.i     v16, 0                    \n\t"                              \
        XLOAD                                                                     \
        "addi        t1, %[w], 32              \n\t"                              \
        "vle8.v      v12, (t1)                 \n\t"                              \
        "vsrl.vi     v8, v12, 4                \n\t"                              \
        "vand.vi     v12, v12, 15              \n\t"                              \
        "vsetvli     t0, zero, e8, m1, ta, ma  \n\t"                              \
        RV_VMADOT(16, 4, 12) RV_VMADOT(18, 4, 13) RV_VMADOT(20, 4, 14) RV_VMADOT(22, 4, 15) \
        RV_VMADOT(16, 5, 8)  RV_VMADOT(18, 5, 9)  RV_VMADOT(20, 5, 10) RV_VMADOT(22, 5, 11) \
        "vsetvli     t0, zero, e8, m4, ta, ma  \n\t"                              \
        "addi        t1, %[w], 160             \n\t"                              \
        "vle8.v      v12, (t1)                 \n\t"                              \
        "vsrl.vi     v8, v12, 4                \n\t"                              \
        "vand.vi     v12, v12, 15              \n\t"                              \
        "vsetvli     t0, zero, e8, m1, ta, ma  \n\t"                              \
        RV_VMADOT(16, 6, 12) RV_VMADOT(18, 6, 13) RV_VMADOT(20, 6, 14) RV_VMADOT(22, 6, 15) \
        RV_VMADOT(16, 7, 8)  RV_VMADOT(18, 7, 9)  RV_VMADOT(20, 7, 10) RV_VMADOT(22, 7, 11) \
        "vsetivli    zero, 12, e32, m2, tu, ma \n\t"                              \
        "vmv.v.v     v22, v20                  \n\t"                              \
        "vsetivli    zero, 8, e32, m2, tu, ma  \n\t"                              \
        "vmv.v.v     v22, v18                  \n\t"                              \
        "vsetivli    zero, 4, e32, m2, tu, ma  \n\t"                              \
        "vmv.v.v     v22, v16                  \n\t"                              \
        "vsetivli    zero, 16, e32, m2, ta, ma \n\t"                              \
        "lw          t2, 0(%[c])               \n\t"                              \
        "vsub.vx     v22, v22, t2              \n\t"                              \
        "vfcvt.f.x.v v22, v22                  \n\t"                              \
        "vsetivli    zero, 16, e16, m1, ta, ma \n\t"                              \
        "vle16.v     v26, (%[w])               \n\t"                              \
        "vfwcvt.f.f.v v28, v26                 \n\t"                              \
        "vsetivli    zero, 16, e32, m2, ta, ma \n\t"                              \
        "flw         ft0, 0(%[d])              \n\t"                              \
        "vfmul.vv    v28, v28, v22             \n\t"                              \
        "vfmul.vf    v28, v28, ft0             \n\t"                              \
        "vfadd.vv    v24, v24, v28             \n\t"                              \
        "addi        %[w], %[w], 288           \n\t"                              \
        XADV                                                                      \
        "addi        %[d], %[d], 4             \n\t"                              \
        "addi        %[c], %[c], 4             \n\t"                              \
        "addi        %[k], %[k], -1            \n\t"                              \
        "bnez        %[k], 1b                  \n\t"                              \
        "vse32.v     v24, (%[o])               \n\t"                              \
        ".option pop                           \n\t"                              \
        : [w] "+r"(w), [x] "+r"(xr), [d] "+r"(xd), [c] "+r"(corr), [k] "+r"(k)    \
        : [o] "r"(out)                                                            \
        : "t0", "t1", "t2", "ft0", "memory", "vl", "vtype",                       \
          "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", \
          "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", \
          "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");                \
}

VM32_FN(vm32_np)

VM16V_FN(vm16_fx,
         "vsetvli     t0, zero, e8, m4, ta, ma  \n\t"
         "vle8.v      v4, (%[x])                \n\t",
         "")

/* 紧凑激活：xq 32B/块（footprint = nblk×32），tile 内 vrgather 展开成 128B
 * idx[i] = i/4（每 8 个元素重复 4 次，与 xr 布局等价，数值完全一致） */
static uint8_t g_idx[128];
__attribute__((noinline))
static void vm16_c(const uint8_t *w, const int8_t *xq, const float *xd, const int32_t *corr,
                   int nblk, float *out) {
    long k = nblk;
    __asm__ volatile(
        ".option push                          \n\t"
        ".option arch, +zvfhmin                \n\t"
        "vsetivli    zero, 16, e32, m2, ta, ma \n\t"
        "vmv.v.i     v24, 0                    \n\t"
        "vsetvli     t0, zero, e8, m4, ta, ma  \n\t"
        "vle8.v      v0, (%[ix])               \n\t"   /* 128B 索引向量 */
        "1:                                    \n\t"
        "vsetvli     t0, zero, e32, m8, ta, ma \n\t"
        "vmv.v.i     v16, 0                    \n\t"
        "li          t0, 32                    \n\t"
        "vsetvli     t0, t0, e8, m1, ta, ma    \n\t"
        "vle8.v      v28, (%[x])               \n\t"   /* 32B 紧凑激活 */
        "vsetvli     t0, zero, e8, m4, ta, ma  \n\t"
        "vrgather.vv v4, v28, v0               \n\t"   /* → 128B，与 xr 同布局 */
        "addi        t1, %[w], 32              \n\t"
        "vle8.v      v12, (t1)                 \n\t"
        "vsrl.vi     v8, v12, 4                \n\t"
        "vand.vi     v12, v12, 15              \n\t"
        "vsetvli     t0, zero, e8, m1, ta, ma  \n\t"
        RV_VMADOT(16, 4, 12) RV_VMADOT(18, 4, 13) RV_VMADOT(20, 4, 14) RV_VMADOT(22, 4, 15)
        RV_VMADOT(16, 5, 8)  RV_VMADOT(18, 5, 9)  RV_VMADOT(20, 5, 10) RV_VMADOT(22, 5, 11)
        "vsetvli     t0, zero, e8, m4, ta, ma  \n\t"
        "addi        t1, %[w], 160             \n\t"
        "vle8.v      v12, (t1)                 \n\t"
        "vsrl.vi     v8, v12, 4                \n\t"
        "vand.vi     v12, v12, 15              \n\t"
        "vsetvli     t0, zero, e8, m1, ta, ma  \n\t"
        RV_VMADOT(16, 6, 12) RV_VMADOT(18, 6, 13) RV_VMADOT(20, 6, 14) RV_VMADOT(22, 6, 15)
        RV_VMADOT(16, 7, 8)  RV_VMADOT(18, 7, 9)  RV_VMADOT(20, 7, 10) RV_VMADOT(22, 7, 11)
        "vsetivli    zero, 12, e32, m2, tu, ma \n\t"
        "vmv.v.v     v22, v20                  \n\t"
        "vsetivli    zero, 8, e32, m2, tu, ma  \n\t"
        "vmv.v.v     v22, v18                  \n\t"
        "vsetivli    zero, 4, e32, m2, tu, ma  \n\t"
        "vmv.v.v     v22, v16                  \n\t"
        "vsetivli    zero, 16, e32, m2, ta, ma \n\t"
        "lw          t2, 0(%[c])               \n\t"
        "vsub.vx     v22, v22, t2              \n\t"
        "vfcvt.f.x.v v22, v22                  \n\t"
        "vsetivli    zero, 16, e16, m1, ta, ma \n\t"
        "vle16.v     v26, (%[w])               \n\t"
        "vfwcvt.f.f.v v28, v26                 \n\t"
        "vsetivli    zero, 16, e32, m2, ta, ma \n\t"
        "flw         ft0, 0(%[d])              \n\t"
        "vfmul.vv    v28, v28, v22             \n\t"
        "vfmul.vf    v28, v28, ft0             \n\t"
        "vfadd.vv    v24, v24, v28             \n\t"
        "addi        %[w], %[w], 288           \n\t"
        "addi        %[x], %[x], 32            \n\t"
        "addi        %[d], %[d], 4             \n\t"
        "addi        %[c], %[c], 4             \n\t"
        "addi        %[k], %[k], -1            \n\t"
        "bnez        %[k], 1b                  \n\t"
        "vse32.v     v24, (%[o])               \n\t"
        ".option pop                           \n\t"
        : [w] "+r"(w), [x] "+r"(xq), [d] "+r"(xd), [c] "+r"(corr), [k] "+r"(k)
        : [o] "r"(out), [ix] "r"(g_idx)
        : "t0", "t1", "t2", "ft0", "memory", "vl", "vtype",
          "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11",
          "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
          "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
}

/* 16 行内核（每次调用算 16 行）；生产同款 = 两次调用 */
typedef void (*tilefn_t)(const uint8_t *, const int8_t *, const float *, const int32_t *, int, float *);
/* 32 行内核（一次调用算 32 行） */
typedef void (*tilefn32_t)(const uint8_t *, const int8_t *, const float *, const int32_t *, int, float *, long);

static void unit16(mat_t *m, int nt, tilefn_t f, const int8_t *xsrc) {
    prep(m, M_VM);
    const size_t ts = (size_t)m->nblk * 288;
    const int nu = m->rows / 32;
    if (nt == 1) {
        for (int u = 0; u < nu; u++) {
            const uint8_t *w = m->w288 + (size_t)(2 * u) * ts;
            f(w, xsrc, m->xd, m->corr, m->nblk, m->out + u * 32);
            f(w + ts, xsrc, m->xd, m->corr, m->nblk, m->out + u * 32 + 16);
        }
        return;
    }
    int next = 0;
    #pragma omp parallel num_threads(nt)
    {
        if (sched_getcpu() != omp_get_thread_num()) __atomic_add_fetch(&g_badcpu, 1, __ATOMIC_RELAXED);
        for (;;) {
            const int u = __atomic_fetch_add(&next, 1, __ATOMIC_RELAXED);
            if (u >= nu) break;
            const uint8_t *w = m->w288 + (size_t)(2 * u) * ts;
            f(w, m->xr, m->xd, m->corr, m->nblk, m->out + u * 32);
            f(w + ts, m->xr, m->xd, m->corr, m->nblk, m->out + u * 32 + 16);
        }
    }
}

static void unit32(mat_t *m, int nt, tilefn32_t f) {
    prep(m, M_VM);
    const size_t ts = (size_t)m->nblk * 288;
    const int nu = m->rows / 32;
    if (nt == 1) {
        for (int u = 0; u < nu; u++)
            f(m->w288 + (size_t)(2 * u) * ts, m->xr, m->xd, m->corr, m->nblk, m->out + u * 32, 0);
        return;
    }
    int next = 0;
    #pragma omp parallel num_threads(nt)
    {
        if (sched_getcpu() != omp_get_thread_num()) __atomic_add_fetch(&g_badcpu, 1, __ATOMIC_RELAXED);
        for (;;) {
            const int u = __atomic_fetch_add(&next, 1, __ATOMIC_RELAXED);
            if (u >= nu) break;
            f(m->w288 + (size_t)(2 * u) * ts, m->xr, m->xd, m->corr, m->nblk, m->out + u * 32, 0);
        }
    }
}

enum { V_VM, V_FX, V_32, V_C, NV };
static const char *VN[NV] = { "vm", "fx", "v32", "vc" };
static void run_v(mat_t *m, int v) {
    switch (v) {
    case V_VM:   unit16(m, 4, vm_tile16, m->xr); break;
    case V_FX:   unit16(m, 4, vm16_fx, m->xr);   break;
    case V_32:   unit32(m, 4, vm32_np);          break;
    case V_C:    unit16(m, 4, vm16_c, m->xq);    break;
    }
}

typedef struct { const char *nm; int rows, K, resid; } cs_t;
static const cs_t CS[] = {
    { "G1 K=1024  24320r", 24320, 1024, 0 },
    { "G1 K=2560   9728r",  9728, 2560, 0 },
    { "G1 K=4096   6080r",  6080, 4096, 0 },
    { "G1 K=9728   2560r",  2560, 9728, 0 },
    { "G2 K=9728   6080r",  6080, 9728, 0 },
    { "G2 K=2560  24320r", 24320, 2560, 0 },
    { "G3 K=9728   2560r+resid", 2560, 9728, 1 },
};
#define NCS ((int)(sizeof CS / sizeof CS[0]))

int main(int argc, char **argv) {
    const int R = argc > 1 ? atoi(argv[1]) : 9;
    cpu_set_t orig; sched_getaffinity(0, sizeof orig, &orig);
    omp_set_dynamic(0);
    { cpu_set_t s; CPU_ZERO(&s); CPU_SET(0, &s); sched_setaffinity(0, sizeof s, &s); }
    pin_team(4);
    /* xr 布局：第 q 组 32B = x[q*8..q*8+8] 重复 4 遍 → idx[i] = 8*(i/32) + (i%8) */
    for (int i = 0; i < 128; i++) g_idx[i] = (uint8_t)(8 * (i / 32) + (i % 8));

    mat_t *M = calloc(NCS, sizeof(mat_t));
    for (int i = 0; i < NCS; i++) M[i] = mk(CS[i].nm, CS[i].rows, CS[i].K, CS[i].resid);

    /* 位级对拍：vc（紧凑激活 + vrgather 展开）应与 vm 完全一致 */
    {
        mat_t *m = &M[3];
        float *ref = malloc(m->rows * 4);
        unit16(m, 1, vm_tile16, m->xr);
        memcpy(ref, m->out, m->rows * 4);
        unit16(m, 1, vm16_c, m->xq);
        int bad = 0;
        for (int r = 0; r < m->rows; r++) if (memcmp(&m->out[r], &ref[r], 4)) bad++;
        printf("[CHK] vc vs vm 位失配=%d / %d\n", bad, m->rows);
        free(ref);
    }

    printf("bench_dec_cols  t4  R=%d  badcpu=%d\n", R, g_badcpu);
    printf("%-24s %7s %6s %5s %6s", "配置", "权重MB", "K", "tiles", "xr_KB");
    for (int v = 0; v < NV; v++) printf(" %7s", VN[v]);
    printf(" %7s\n", "纯读");

    double *ts = malloc(sizeof(double) * NV * NCS * R);
    for (int v = 0; v < NV; v++) for (int k = 0; k < NCS; k++) run_v(&M[k], v);
    for (int r = 0; r < R; r++) for (int j = 0; j < NV * NCS; j++) {
        const int v = (r + j) / NCS % NV, k = (r + j) % NCS;
        double a = now(); run_v(&M[k], v); ts[((size_t)v * NCS + k) * R + r] = now() - a;
    }

    double bw[NCS];
    for (int k = 0; k < NCS; k++) {
        const size_t nb = (size_t)M[k].rows * M[k].nblk * 18;
        double best = 1e9;
        for (int r = 0; r < 5; r++) {
            double a = now();
            size_t seg = nb / 4;
            #pragma omp parallel num_threads(4)
            {
                const uint8_t *p = M[k].w288 + (size_t)omp_get_thread_num() * seg;
                size_t n = seg;
                __asm__ volatile(
                    "vsetvli t0, zero, e8, m8, ta, ma\n\t"
                    "vmv.v.i v8, 0\n\t"
                    "1:\n\t"
                    "vsetvli t0, %[n], e8, m8, tu, ma\n\t"
                    "vle8.v v16, (%[s])\n\t"
                    "vxor.vv v8, v8, v16\n\t"
                    "add %[s], %[s], t0\n\t"
                    "sub %[n], %[n], t0\n\t"
                    "bnez %[n], 1b\n\t"
                    : [s] "+r"(p), [n] "+r"(n) : : "t0", "memory", "vl", "vtype", "v8", "v16");
            }
            double d = now() - a;
            if (d < best) best = d;
        }
        bw[k] = (double)nb / best / 1e9;
    }

    for (int k = 0; k < NCS; k++) {
        printf("%-24s %7.1f %6d %5d %6.1f", CS[k].nm, (double)M[k].rows * M[k].nblk * 18 / 1e6,
               CS[k].K, M[k].rows / 32, (double)M[k].nblk * 128 / 1024.0);
        for (int v = 0; v < NV; v++) {
            double *q = ts + ((size_t)v * NCS + k) * R;
            qsort(q, R, sizeof(double), cmp_dbl);
            printf(" %7.2f", (double)M[k].rows * M[k].nblk * 18 / q[R / 2] / 1e9);
        }
        printf(" %7.2f\n", bw[k]);
    }
    printf("RESULT badcpu=%d\n", g_badcpu);
    sched_setaffinity(0, sizeof orig, &orig);
    return g_badcpu ? 1 : 0;
}
