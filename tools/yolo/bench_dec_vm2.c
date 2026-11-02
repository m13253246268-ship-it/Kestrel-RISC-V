/* ================================================================
 * bench_dec_vm2.c — L-DEC-001 追差距：M=1 vmadot 内核变体 + 带宽上限
 *
 * 复用 bench_dec_vmadot.c（锚 rv_q4_col_tile、rp_tile、vm_tile16、mk 等）。
 * 变体（全部保持与锚相同的逐块浮点顺序：acc = acc + ((s·f32(i))·xd)，块后 + resid）：
 *   VM0     生产同款 vm_tile16（基线）
 *   VM16    宏重写版 16 行 tile（自检用，应与 VM0 同速同位）
 *   VM16pf  + 软件预取（Zicbop prefetch.r，未实现时为 ori x0 = NOP）
 *   VM32    相邻两个 16 行 tile 同一循环，共享激活 xr 加载与标量加载
 *   BW*     纯读 vle8 m8 / memcpy 到线程私有缓冲：同一权重字节数的带宽上限
 * 调度：dyn = 每次领 1 单元（32 行，同生产 DYN_GRAIN=2×16 行）；sta = 每线程连续区段。
 * 编译：gcc -O2 -march=rv64gcv -mabi=lp64d -ffp-contract=off -fopenmp bench_dec_vm2.c -o bench_dec_vm2
 * ================================================================ */
#define main bench_dec_vmadot_main
#include "bench_dec_vmadot.c"
#undef main

/* prefetch.r off(t3)：ORI rd=x0，imm[4:0]=00011，imm[11:5]=off/32；预取当前块 + pfd 字节处的 5 条 64B 行 */
#define PF(imm) ".insn i 0x13, 6, x0, t3, " RV_YS(imm) "\n\t"
#define PF_BLK(W) "add t3, " W ", %[p]\n\t" PF(3) PF(67) PF(131) PF(195) PF(259)

/* 半块：W+OFF 处 128B nibble → v12..15(lo) / v8..11(hi)，与 x 寄存器 XA/XB 做 8 次 vmadot */
#define VM_HALF(W, OFF, XA, XB)                                            \
    "vsetvli t0, zero, e8, m4, ta, ma\n\t"                                 \
    "addi t1, " W ", " OFF "\n\t"                                          \
    "vle8.v v12, (t1)\n\t"                                                 \
    "vsrl.vi v8, v12, 4\n\t"                                               \
    "vand.vi v12, v12, 15\n\t"                                             \
    "vsetvli t0, zero, e8, m1, ta, ma\n\t"                                 \
    RV_VMADOT(16, XA, 12) RV_VMADOT(18, XA, 13) RV_VMADOT(20, XA, 14) RV_VMADOT(22, XA, 15) \
    RV_VMADOT(16, XB, 8)  RV_VMADOT(18, XB, 9)  RV_VMADOT(20, XB, 10) RV_VMADOT(22, XB, 11)

/* 一个 16 行 tile 的一个块：结果累加进 ACC（e32 m2）；需 t2=corr、ft0=xd 已就位 */
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

#define VM16_FN(NAME, PFA)                                                                    \
__attribute__((noinline)) static void NAME(const uint8_t *w, const int8_t *xr, const float *xd, \
        const int32_t *corr, int nblk, float *out, long pfd) {                                \
    long k = nblk;                                                                            \
    __asm__ volatile(                                                                         \
        ".option push\n\t.option arch, +zvfhmin\n\t"                                          \
        "vsetivli zero, 16, e32, m2, ta, ma\n\t"                                              \
        "vmv.v.i v24, 0\n\t"                                                                  \
        "1:\n\t"                                                                              \
        PFA                                                                                   \
        "vsetvli t0, zero, e8, m4, ta, ma\n\t"                                                \
        "vle8.v v4, (%[x])\n\t"                                                               \
        "lw t2, 0(%[c])\n\t"                                                                  \
        "flw ft0, 0(%[d])\n\t"                                                                \
        VM_TILE("%[wa]", "v24")                                                               \
        "addi %[wa], %[wa], 288\n\t"                                                          \
        "addi %[x], %[x], 128\n\t"                                                            \
        "addi %[d], %[d], 4\n\t"                                                              \
        "addi %[c], %[c], 4\n\t"                                                              \
        "addi %[k], %[k], -1\n\t"                                                             \
        "bnez %[k], 1b\n\t"                                                                   \
        "vsetivli zero, 16, e32, m2, ta, ma\n\t"                                              \
        "vse32.v v24, (%[o])\n\t"                                                             \
        ".option pop\n\t"                                                                     \
        : [wa] "+r"(w), [x] "+r"(xr), [d] "+r"(xd), [c] "+r"(corr), [k] "+r"(k)               \
        : [o] "r"(out), [p] "r"(pfd)                                                          \
        : VM_CLOB);                                                                           \
}

/* 32 行：tile A = w，tile B = w + nblk·288（rp_tile 的 2ti / 2ti+1，内存相邻） */
#define VM32_FN(NAME, PFA, PFB)                                                               \
__attribute__((noinline)) static void NAME(const uint8_t *w, const int8_t *xr, const float *xd, \
        const int32_t *corr, int nblk, float *out, long pfd) {                                \
    long k = nblk; const uint8_t *wb = w + (size_t)nblk * 288;                                \
    __asm__ volatile(                                                                         \
        ".option push\n\t.option arch, +zvfhmin\n\t"                                          \
        "vsetivli zero, 16, e32, m2, ta, ma\n\t"                                              \
        "vmv.v.i v24, 0\n\t"                                                                  \
        "vmv.v.i v30, 0\n\t"                                                                  \
        "1:\n\t"                                                                              \
        PFA PFB                                                                               \
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
        "bnez %[k], 1b\n\t"                                                                   \
        "vsetivli zero, 16, e32, m2, ta, ma\n\t"                                              \
        "vse32.v v24, (%[o])\n\t"                                                             \
        "vse32.v v30, (%[o2])\n\t"                                                            \
        ".option pop\n\t"                                                                     \
        : [wa] "+r"(w), [wb] "+r"(wb), [x] "+r"(xr), [d] "+r"(xd), [c] "+r"(corr), [k] "+r"(k) \
        : [o] "r"(out), [o2] "r"(out + 16), [p] "r"(pfd)                                      \
        : VM_CLOB);                                                                           \
}

/* ---- 软件流水：v0 预装下一个半块的 128B 原始 nibble，vle8 与当前 8 次 vmadot 重叠 ---- */
#define VM_HALFP(XA, XB, NEXT)                                             \
    "vsetvli t0, zero, e8, m4, ta, ma\n\t"                                 \
    "vsrl.vi v8, v0, 4\n\t"                                                \
    "vand.vi v12, v0, 15\n\t"                                              \
    NEXT                                                                   \
    "vle8.v v0, (t1)\n\t"                                                  \
    "vsetvli t0, zero, e8, m1, ta, ma\n\t"                                 \
    RV_VMADOT(16, XA, 12) RV_VMADOT(18, XA, 13) RV_VMADOT(20, XA, 14) RV_VMADOT(22, XA, 15) \
    RV_VMADOT(16, XB, 8)  RV_VMADOT(18, XB, 9)  RV_VMADOT(20, XB, 10) RV_VMADOT(22, XB, 11)
#define VM_ZERO "vsetvli t0, zero, e32, m8, ta, ma\n\tvmv.v.i v16, 0\n\t"
#define VM_EPI(W, ACC)                                                     \
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
/* t1 ← 下一块（末块时为本块，避免越界读）的 +32 */
#define NEXT_BLK(W) "addi t4, " W ", 288\n\taddi t1, %[k], -1\n\tbnez t1, 2f\n\tmv t4, " W "\n\t2:\n\taddi t1, t4, 32\n\t"
#define VM_CLOBP VM_CLOB, "t4"

__attribute__((noinline)) static void vm16_sp(const uint8_t *w, const int8_t *xr, const float *xd,
        const int32_t *corr, int nblk, float *out, long pfd) {
    long k = nblk; (void)pfd;
    __asm__ volatile(
        ".option push\n\t.option arch, +zvfhmin\n\t"
        "vsetivli zero, 16, e32, m2, ta, ma\n\t"
        "vmv.v.i v24, 0\n\t"
        "vsetvli t0, zero, e8, m4, ta, ma\n\t"
        "addi t1, %[wa], 32\n\t"
        "vle8.v v0, (t1)\n\t"
        "1:\n\t"
        "vle8.v v4, (%[x])\n\t"
        "lw t2, 0(%[c])\n\t"
        "flw ft0, 0(%[d])\n\t"
        VM_ZERO
        VM_HALFP(4, 5, "addi t1, %[wa], 160\n\t")
        VM_HALFP(6, 7, NEXT_BLK("%[wa]"))
        VM_EPI("%[wa]", "v24")
        "addi %[wa], %[wa], 288\n\t"
        "addi %[x], %[x], 128\n\t"
        "addi %[d], %[d], 4\n\t"
        "addi %[c], %[c], 4\n\t"
        "addi %[k], %[k], -1\n\t"
        "vsetvli t0, zero, e8, m4, ta, ma\n\t"
        "bnez %[k], 1b\n\t"
        "vsetivli zero, 16, e32, m2, ta, ma\n\t"
        "vse32.v v24, (%[o])\n\t"
        ".option pop\n\t"
        : [wa] "+r"(w), [x] "+r"(xr), [d] "+r"(xd), [c] "+r"(corr), [k] "+r"(k)
        : [o] "r"(out)
        : VM_CLOBP);
}

/* 32 行流水：A0 → A1 → B0 → B1 → 下一块 A0 */
__attribute__((noinline)) static void vm32_sp(const uint8_t *w, const int8_t *xr, const float *xd,
        const int32_t *corr, int nblk, float *out, long pfd) {
    long k = nblk; const uint8_t *wb = w + (size_t)nblk * 288; (void)pfd;
    __asm__ volatile(
        ".option push\n\t.option arch, +zvfhmin\n\t"
        "vsetivli zero, 16, e32, m2, ta, ma\n\t"
        "vmv.v.i v24, 0\n\t"
        "vmv.v.i v30, 0\n\t"
        "vsetvli t0, zero, e8, m4, ta, ma\n\t"
        "addi t1, %[wa], 32\n\t"
        "vle8.v v0, (t1)\n\t"
        "1:\n\t"
        "vle8.v v4, (%[x])\n\t"
        "lw t2, 0(%[c])\n\t"
        "flw ft0, 0(%[d])\n\t"
        VM_ZERO
        VM_HALFP(4, 5, "addi t1, %[wa], 160\n\t")
        VM_HALFP(6, 7, "addi t1, %[wb], 32\n\t")
        VM_EPI("%[wa]", "v24")
        VM_ZERO
        VM_HALFP(4, 5, "addi t1, %[wb], 160\n\t")
        VM_HALFP(6, 7, NEXT_BLK("%[wa]"))
        VM_EPI("%[wb]", "v30")
        "addi %[wa], %[wa], 288\n\t"
        "addi %[wb], %[wb], 288\n\t"
        "addi %[x], %[x], 128\n\t"
        "addi %[d], %[d], 4\n\t"
        "addi %[c], %[c], 4\n\t"
        "addi %[k], %[k], -1\n\t"
        "vsetvli t0, zero, e8, m4, ta, ma\n\t"
        "bnez %[k], 1b\n\t"
        "vsetivli zero, 16, e32, m2, ta, ma\n\t"
        "vse32.v v24, (%[o])\n\t"
        "vse32.v v30, (%[o2])\n\t"
        ".option pop\n\t"
        : [wa] "+r"(w), [wb] "+r"(wb), [x] "+r"(xr), [d] "+r"(xd), [c] "+r"(corr), [k] "+r"(k)
        : [o] "r"(out), [o2] "r"(out + 16)
        : VM_CLOBP);
}

VM16_FN(vm16_np, "")
VM16_FN(vm16_pf, PF_BLK("%[wa]"))
VM32_FN(vm32_np, "", "")
VM32_FN(vm32_pf, PF_BLK("%[wa]"), PF_BLK("%[wb]"))

/* 纯读：vle8 m8 流式读 n 字节（可选预取 pfd 处 4 条行），vxor 累积防消除 */
__attribute__((noinline)) static void bw_read(const uint8_t *p, size_t n, long pfd, int pf, uint8_t *sink) {
    if (pf) __asm__ volatile(
        "vsetvli t0, zero, e8, m8, ta, ma\n\t"
        "vmv.v.i v8, 0\n\t"
        "1:\n\t"
        "add t3, %[s], %[p]\n\t" PF(3) PF(67) PF(131) PF(195)
        "vsetvli t0, %[n], e8, m8, tu, ma\n\t"
        "vle8.v v16, (%[s])\n\t"
        "vxor.vv v8, v8, v16\n\t"
        "add %[s], %[s], t0\n\t"
        "sub %[n], %[n], t0\n\t"
        "bnez %[n], 1b\n\t"
        "vsetvli t0, zero, e8, m8, ta, ma\n\t"
        "vse8.v v8, (%[o])\n\t"
        : [s] "+r"(p), [n] "+r"(n) : [p] "r"(pfd), [o] "r"(sink) : VM_CLOB);
    else __asm__ volatile(
        "vsetvli t0, zero, e8, m8, ta, ma\n\t"
        "vmv.v.i v8, 0\n\t"
        "1:\n\t"
        "vsetvli t0, %[n], e8, m8, tu, ma\n\t"
        "vle8.v v16, (%[s])\n\t"
        "vxor.vv v8, v8, v16\n\t"
        "add %[s], %[s], t0\n\t"
        "sub %[n], %[n], t0\n\t"
        "bnez %[n], 1b\n\t"
        "vsetvli t0, zero, e8, m8, ta, ma\n\t"
        "vse8.v v8, (%[o])\n\t"
        : [s] "+r"(p), [n] "+r"(n) : [o] "r"(sink) : VM_CLOB);
}

enum { K_VM0, K_VM16, K_VM16PF, K_VM32, K_VM32PF, K_VM16SP, K_VM32SP, K_BWV, K_BWVPF, K_BWC };
typedef struct { const char *nm; int kind; long pfd; int sta; } cfg_t;
static const cfg_t CF[] = {
    { "VM0 dyn(生产)",      K_VM0,    0,    0 },
    { "VM16pf2304 dyn",     K_VM16PF, 2304, 0 },
    { "VM32 dyn",           K_VM32,   0,    0 },
    { "VM32pf1152 dyn",     K_VM32PF, 1152, 0 },
    { "VM16sp dyn",         K_VM16SP, 0,    0 },
    { "VM32sp dyn",         K_VM32SP, 0,    0 },
    { "VM32sp sta",         K_VM32SP, 0,    1 },
    { "BW vle8 dyn",        K_BWV,    0,    0 },
    { "BW vle8pf1024 dyn",  K_BWVPF,  1024, 0 },
    { "BW memcpy dyn",      K_BWC,    0,    0 },
};
#define NCF ((int)(sizeof CF / sizeof CF[0]))

static uint8_t *g_tmp[8];
static void unit(mat_t *m, const cfg_t *c, int u, int th) {
    const size_t ts = (size_t)m->nblk * 288;
    const uint8_t *w = m->w288 + (size_t)(2 * u) * ts;
    float *o = m->out + u * 32;
    switch (c->kind) {
    case K_VM0:    vm_tile16(w, m->xr, m->xd, m->corr, m->nblk, o);
                   vm_tile16(w + ts, m->xr, m->xd, m->corr, m->nblk, o + 16); break;
    case K_VM16:   vm16_np(w, m->xr, m->xd, m->corr, m->nblk, o, 0);
                   vm16_np(w + ts, m->xr, m->xd, m->corr, m->nblk, o + 16, 0); break;
    case K_VM16PF: vm16_pf(w, m->xr, m->xd, m->corr, m->nblk, o, c->pfd);
                   vm16_pf(w + ts, m->xr, m->xd, m->corr, m->nblk, o + 16, c->pfd); break;
    case K_VM32:   vm32_np(w, m->xr, m->xd, m->corr, m->nblk, o, 0); break;
    case K_VM32PF: vm32_pf(w, m->xr, m->xd, m->corr, m->nblk, o, c->pfd); break;
    case K_VM16SP: vm16_sp(w, m->xr, m->xd, m->corr, m->nblk, o, 0);
                   vm16_sp(w + ts, m->xr, m->xd, m->corr, m->nblk, o + 16, 0); break;
    case K_VM32SP: vm32_sp(w, m->xr, m->xd, m->corr, m->nblk, o, 0); break;
    case K_BWV:    bw_read(w, 2 * ts, 0, 0, g_tmp[th]); return;
    case K_BWVPF:  bw_read(w, 2 * ts, c->pfd, 1, g_tmp[th]); return;
    case K_BWC:    memcpy(g_tmp[th], w, 2 * ts); return;
    }
    if (m->resid) { const float *r = m->resid + u * 32; for (int i = 0; i < 32; i++) o[i] = o[i] + r[i]; }
}

static void run_cfg(mat_t *m, const cfg_t *c, int nt) {
    if (c->kind < K_BWV) prep(m, M_VM);
    const int nu = m->rows / 32;
    if (nt == 1) { for (int u = 0; u < nu; u++) unit(m, c, u, 0); return; }
    int next = 0;
    #pragma omp parallel num_threads(nt)
    {
        const int th = omp_get_thread_num();
        if (sched_getcpu() != th) __atomic_add_fetch(&g_badcpu, 1, __ATOMIC_RELAXED);
        if (c->sta) {
            const int a = (int)((long)nu * th / nt), b = (int)((long)nu * (th + 1) / nt);
            for (int u = a; u < b; u++) unit(m, c, u, th);
        } else for (;;) {
            const int u = __atomic_fetch_add(&next, 1, __ATOMIC_RELAXED);
            if (u >= nu) break;
            unit(m, c, u, th);
        }
    }
}

int main(int argc, char **argv) {
    const int R = argc > 1 ? atoi(argv[1]) : 21;
    cpu_set_t orig; sched_getaffinity(0, sizeof orig, &orig);
    omp_set_dynamic(0);
    { cpu_set_t s; CPU_ZERO(&s); CPU_SET(0, &s); sched_setaffinity(0, sizeof s, &s); }
    pin_team(4);
    for (int i = 0; i < 8; i++) g_tmp[i] = amalloc((size_t)2 * 304 * 288);
    printf("bench_dec_vm2  Qwen3-4B 单层  R=%d  pin badcpu=%d\n", R, g_badcpu);

    mat_t L[7] = { mk("q", 4096, 2560, 0), mk("k", 1024, 2560, 0), mk("v", 1024, 2560, 0),
                   mk("o", 2560, 4096, 1), mk("gate", 9728, 2560, 0), mk("up", 9728, 2560, 0),
                   mk("down", 2560, 9728, 1) };

    printf("== L1：逐元素比位（ref = COL 单线程），所有 VM 变体 × t1/t4 ==\n");
    int l1_fail = 0;
    for (int i = 0; i < 7; i++) {
        mat_t *m = &L[i]; float *ref = amalloc(m->rows * 4);
        run_mat(m, M_COL, 1); memcpy(ref, m->out, m->rows * 4);
        printf("  %-5s rows=%-5d K=%-5d resid=%d :", m->name, m->rows, m->nblk * 32, m->resid != NULL);
        for (int k = 0; k < NCF; k++) {
            if (CF[k].kind >= K_BWV) continue;
            int bad = 0;
            for (int nt = 1; nt <= 4; nt += 3) {
                memset(m->out, 0xA5, m->rows * 4);
                run_cfg(m, &CF[k], nt);
                for (int r = 0; r < m->rows; r++) if (memcmp(&m->out[r], &ref[r], 4)) bad++;
            }
            printf(" %d", bad); l1_fail += bad;
        }
        printf("\n");
        free(ref);
    }

    static const int G_GUD[3] = { 4, 5, 6 }, G_ALL[7] = { 0, 1, 2, 3, 4, 5, 6 };
    const struct { const char *nm; const int *ix; int n; } G[2] = { { "gate+up+down", G_GUD, 3 }, { "全层7矩阵", G_ALL, 7 } };
    for (int g = 0; g < 2; g++) {
        double bytes = 0; for (int i = 0; i < G[g].n; i++) bytes += (double)L[G[g].ix[i]].rows * L[G[g].ix[i]].nblk * 18;
        for (int nt = 1; nt <= 4; nt += 3) {
            printf("== %s  %.1f MB  t%d（配置轮换，中位数） ==\n", G[g].nm, bytes / 1e6, nt);
            double *ts = malloc(sizeof(double) * NCF * R);
            for (int k = 0; k < NCF; k++) for (int i = 0; i < G[g].n; i++) run_cfg(&L[G[g].ix[i]], &CF[k], nt);
            for (int r = 0; r < R; r++) for (int j = 0; j < NCF; j++) {
                const int k = (r + j) % NCF; double a = now();
                for (int i = 0; i < G[g].n; i++) run_cfg(&L[G[g].ix[i]], &CF[k], nt);
                ts[k * R + r] = now() - a;
            }
            double base = 0;
            for (int k = 0; k < NCF; k++) {
                qsort(ts + k * R, R, sizeof(double), cmp_dbl);
                const double md = ts[k * R + R / 2];
                if (k == 0) base = md;
                printf("  %-20s med %8.3f ms  min %8.3f ms  %5.2f GB/s  相对VM0 %+6.1f%%\n",
                       CF[k].nm, md * 1e3, ts[k * R] * 1e3, bytes / md / 1e9, (base / md - 1) * 100);
            }
            free(ts);
        }
    }
    printf("RESULT l1_fail=%d badcpu=%d\n", l1_fail, g_badcpu);
    sched_setaffinity(0, sizeof orig, &orig);
    return (l1_fail || g_badcpu) ? 1 : 0;
}
