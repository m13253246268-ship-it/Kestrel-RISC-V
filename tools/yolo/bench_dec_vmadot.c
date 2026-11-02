/* ================================================================
 * bench_dec_vmadot.c — L-DEC-001 Phase 1：decode(M=1) Q4 GEMV，vmadot vs RVV
 *
 * 锚（原样复刻自 src/model/vllm_safetensors.c，不改生产代码）：
 *   COL = rv_q4_col_tile   位级锚 + 生产默认 decode 路径（VLLM_Q4_DEC 默认关）
 *   DEC = rv_q4_dec_tile   RVV 性能锚（VLLM_Q4_DEC=1，SCV=1）
 * VM  = vmadot 变体：复用 prefill 288B IME 布局（rv_ime_rp_worker 原样）。
 *   - 激活每 8 列一段复制到 4 个槽（xr，128B/块），vs1=激活、vs2=权重：
 *     out[r*4+c] = Σ_i x[8q+i]·w[行 g*4+c][8q+i]，与 r 无关；
 *   - 4 路 vd（g=0..3）用 tu 尾不扰 vmv 合并 → lane l = 行 l（自然行序，无 gather）；
 *   - 无偏 nibble（0..15）× int8 → int32 精确，再减 8·Σx（整数精确）；
 *   - epilogue 与锚同序：acc = acc + ((s·f32(i))·xd)，块循环后 + resid。
 * 形状：Qwen3-4B 单层（dim=2560 ffn=9728 heads=32 kv=8 hd=128）≈56.8MB/格式，远大于 L2。
 * 线程：1 线程绑 hart0；4 线程绑 hart0-3（vmadot 在 hart4-7 SIGILL），sched_getcpu 核验，
 *   结束还原亲和性。三种模式同一轮内轮换（ABC/BCA/CAB），取中位数。
 * 编译：gcc -O2 -march=rv64gcv -mabi=lp64d -ffp-contract=off -fopenmp bench_dec_vmadot.c -o bench_dec_vmadot
 * ================================================================ */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <sched.h>
#include <omp.h>
#include <riscv_vector.h>

static inline uint32_t f16_to_f32_bits(uint16_t h) {
    uint32_t sign = (h >> 15) & 1, exp = (h >> 10) & 0x1F, mant = h & 0x3FF;
    if (exp == 0) {
        if (mant == 0) return sign << 31;
        uint32_t m_norm = mant; int shift = 0;
        while (m_norm < 0x400) { m_norm <<= 1; shift++; }
        return (sign << 31) | ((113 - (uint32_t)shift) << 23) | ((m_norm & 0x3FF) << 13);
    } else if (exp == 0x1F) {
        return (sign << 31) | (0xFF << 23) | (mant << 13);
    }
    return (sign << 31) | ((exp + 112) << 23) | (mant << 13);
}
static float st_f16scale(const uint8_t *b) {
    uint16_t h; memcpy(&h, b, 2);
    uint32_t fb = f16_to_f32_bits(h); float d; memcpy(&d, &fb, 4); return d;
}

static uint64_t g_rs = 0x9E3779B97F4A7C15ull;
static inline uint64_t rnd(void) { g_rs ^= g_rs << 13; g_rs ^= g_rs >> 7; g_rs ^= g_rs << 17; return g_rs; }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

/* ---- COL：rv_q4_col_tile 原样 ---- */
static void rv_q4_col_tile(const uint8_t *tile, const int8_t *xq, const float *xd,
                           float *out, const float *resid, int nblk) {
    const size_t vl16 = 16;
    vfloat32m4_t ze = __riscv_vfmv_v_f_f32m4(0.0f, vl16);
    vfloat32m4_t eacc = ze, oacc = ze;
    for (int b = 0; b < nblk; b++) {
        const uint8_t *sc = tile + (size_t)b * 576;
        const uint8_t *dt = sc + 64;
        vfloat32m4_t ve, vo;
        {
            float scl_e[16], scl_o[16];
            for (int r = 0; r < 16; r++) {
                scl_e[r] = st_f16scale(sc + (size_t)(2 * r)     * 2);
                scl_o[r] = st_f16scale(sc + (size_t)(2 * r + 1) * 2);
            }
            ve = __riscv_vle32_v_f32m4(scl_e, vl16);
            vo = __riscv_vle32_v_f32m4(scl_o, vl16);
        }
        vint16m2_t accE = __riscv_vmv_v_x_i16m2(0, vl16);
        vint16m2_t accO = __riscv_vmv_v_x_i16m2(0, vl16);
        const int8_t *xqB = xq + (size_t)b * 32;
        for (int c = 0; c < 32; c++) {
            vuint8m1_t col = __riscv_vle8_v_u8m1(dt + (size_t)c * 16, vl16);
            vint8m1_t wE = __riscv_vreinterpret_v_u8m1_i8m1(
                               __riscv_vsub_vx_u8m1(__riscv_vand_vx_u8m1(col, 0x0F, vl16), 8, vl16));
            vint8m1_t wO = __riscv_vreinterpret_v_u8m1_i8m1(
                               __riscv_vsub_vx_u8m1(
                                   __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(col, 4, vl16), 0x0F, vl16),
                                   8, vl16));
            int8_t xq_c = xqB[c];
            accE = __riscv_vwmacc_vx_i16m2(accE, xq_c, wE, vl16);
            accO = __riscv_vwmacc_vx_i16m2(accO, xq_c, wO, vl16);
        }
        float xd_b = xd[b];
        vfloat32m4_t de = __riscv_vfwcvt_f_x_v_f32m4(accE, vl16);
        vfloat32m4_t dO = __riscv_vfwcvt_f_x_v_f32m4(accO, vl16);
        eacc = __riscv_vfadd_vv_f32m4(eacc, __riscv_vfmul_vf_f32m4(__riscv_vfmul_vv_f32m4(ve, de, vl16), xd_b, vl16), vl16);
        oacc = __riscv_vfadd_vv_f32m4(oacc, __riscv_vfmul_vf_f32m4(__riscv_vfmul_vv_f32m4(vo, dO, vl16), xd_b, vl16), vl16);
    }
    if (resid) {
        vfloat32m4_t rE = __riscv_vlse32_v_f32m4(resid + 0, 8, vl16);
        vfloat32m4_t rO = __riscv_vlse32_v_f32m4(resid + 1, 8, vl16);
        eacc = __riscv_vfadd_vv_f32m4(eacc, rE, vl16);
        oacc = __riscv_vfadd_vv_f32m4(oacc, rO, vl16);
    }
    __riscv_vsse32_v_f32m4(out + 0, 8, eacc, vl16);
    __riscv_vsse32_v_f32m4(out + 1, 8, oacc, vl16);
}

/* ---- DEC：rv_q4_dec_tile 原样（scv=1 路径） ---- */
static void rv_q4_dec_tile(const uint8_t *tile, const int8_t *xq, const float *xd,
                           const int32_t *sx, float *out, const float *resid, int nblk) {
    const size_t vl = 32, vh = 16;
    vbool8_t hi = __riscv_vmsgeu_vx_u8m1_b8(__riscv_vid_v_u8m1(vl), 16, vl);
    vfloat32m2_t eacc = __riscv_vfmv_v_f_f32m2(0.0f, vh), oacc = eacc;
    for (int b = 0; b < nblk; b++) {
        const uint8_t *sc = tile + (size_t)b * 576;
        const uint8_t *dt = sc + 64;
        float s[32];
        __asm__ volatile(
            ".option push                        \n\t"
            ".option arch, +zvfhmin              \n\t"
            "vsetivli zero, 16, e16, m1, ta, ma \n\t"
            "vlse16.v v8, (%[p]), %[st]          \n\t"
            "vlse16.v v9, (%[q]), %[st]          \n\t"
            "vfwcvt.f.f.v v10, v8                \n\t"
            "vfwcvt.f.f.v v12, v9                \n\t"
            "vsetivli zero, 16, e32, m2, ta, ma \n\t"
            "vse32.v v10, (%[o0])                \n\t"
            "vse32.v v12, (%[o1])                \n\t"
            ".option pop                         \n\t"
            :: [p] "r"(sc), [q] "r"(sc + 2), [st] "r"(4L), [o0] "r"(s), [o1] "r"(s + 16)
            : "memory", "vl", "vtype", "v8", "v9", "v10", "v11", "v12", "v13");
        const int8_t *xb = xq + (size_t)b * 32;
        vint16m2_t aE = __riscv_vmv_v_x_i16m2(0, vl), aO = aE;
        for (int c = 0; c < 32; c += 2) {
            vuint8m1_t col = __riscv_vle8_v_u8m1(dt + (size_t)c * 16, vl);
            vint8m1_t nE = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(col, 0x0F, vl));
            vint8m1_t nO = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vsrl_vx_u8m1(col, 4, vl));
            vint8m1_t xv = __riscv_vmerge_vxm_i8m1(__riscv_vmv_v_x_i8m1(xb[c], vl), xb[c + 1], hi, vl);
            aE = __riscv_vwmacc_vv_i16m2(aE, xv, nE, vl);
            aO = __riscv_vwmacc_vv_i16m2(aO, xv, nO, vl);
        }
        const int32_t corr = 8 * sx[b];
        vint32m2_t iE = __riscv_vsub_vx_i32m2(__riscv_vwadd_vv_i32m2(
                            __riscv_vget_v_i16m2_i16m1(aE, 0), __riscv_vget_v_i16m2_i16m1(aE, 1), vh), corr, vh);
        vint32m2_t iO = __riscv_vsub_vx_i32m2(__riscv_vwadd_vv_i32m2(
                            __riscv_vget_v_i16m2_i16m1(aO, 0), __riscv_vget_v_i16m2_i16m1(aO, 1), vh), corr, vh);
        vfloat32m2_t ve = __riscv_vle32_v_f32m2(s, vh), vo = __riscv_vle32_v_f32m2(s + 16, vh);
        const float xd_b = xd[b];
        eacc = __riscv_vfadd_vv_f32m2(eacc, __riscv_vfmul_vf_f32m2(
                   __riscv_vfmul_vv_f32m2(ve, __riscv_vfcvt_f_x_v_f32m2(iE, vh), vh), xd_b, vh), vh);
        oacc = __riscv_vfadd_vv_f32m2(oacc, __riscv_vfmul_vf_f32m2(
                   __riscv_vfmul_vv_f32m2(vo, __riscv_vfcvt_f_x_v_f32m2(iO, vh), vh), xd_b, vh), vh);
    }
    if (resid) {
        eacc = __riscv_vfadd_vv_f32m2(eacc, __riscv_vlse32_v_f32m2(resid + 0, 8, vh), vh);
        oacc = __riscv_vfadd_vv_f32m2(oacc, __riscv_vlse32_v_f32m2(resid + 1, 8, vh), vh);
    }
    __riscv_vsse32_v_f32m2(out + 0, 8, eacc, vh);
    __riscv_vsse32_v_f32m2(out + 1, 8, oacc, vh);
}

/* ---- 重排：rv_ime_rp_worker 原样（mode3 tile ti → IME tile 2ti/2ti+1） ---- */
static void rp_tile(const uint8_t *src, uint8_t *dst, int nblk, int ti) {
    for (int b = 0; b < nblk; b++) {
        const uint8_t *sc = src + ((size_t)ti * nblk + b) * 576, *dt = sc + 64;
        for (int half = 0; half < 2; half++) {
            uint8_t *s = dst + (((size_t)(2 * ti + half)) * nblk + b) * 288;
            memcpy(s, sc + half * 32, 32);
            for (int h = 0; h < 2; h++)
                for (int g = 0; g < 4; g++)
                    for (int rr = 0; rr < 4; rr++) {
                        const int r = half * 16 + g * 4 + rr, sh = (r & 1) * 4;
                        for (int kk = 0; kk < 8; kk++) {
                            const int col = h * 16 + kk;
                            const uint8_t lo = (dt[col * 16 + r / 2] >> sh) & 15;
                            const uint8_t hi = (dt[(col + 8) * 16 + r / 2] >> sh) & 15;
                            s[32 + h * 128 + g * 32 + rr * 8 + kk] = (uint8_t)(lo | (hi << 4));
                        }
                    }
        }
    }
}

/* ---- VM：M=1 vmadot，16 行/IME tile，out[l] = 行 l（自然序） ----
 * w：288B/块（scale 16×f16 + 2×128B nibble）；xr：128B/块，xr[q*32+r*8+i] = x[8q+i]（r=0..3 同值）；
 * corr[b] = 8·Σx_b；xd[b] 激活 scale。 */
#define RV_YS(x) #x
#define RV_VMADOT(vd, vs1, vs2) \
    ".insn 4, " RV_YS(0x2b | ((vd) << 7) | (3 << 12) | ((vs1) << 15) | ((vs2) << 20) | (0x71 << 25)) "\n\t"
__attribute__((noinline))
static void vm_tile16(const uint8_t *w, const int8_t *xr, const float *xd, const int32_t *corr,
                      int nblk, float *out) {
    long k = nblk;
    __asm__ volatile(
        ".option push                          \n\t"
        ".option arch, +zvfhmin                \n\t"
        "vsetivli    zero, 16, e32, m2, ta, ma \n\t"
        "vmv.v.i     v24, 0                    \n\t"
        "1:                                    \n\t"
        "vsetvli     t0, zero, e32, m8, ta, ma \n\t"
        "vmv.v.i     v16, 0                    \n\t"
        "vsetvli     t0, zero, e8, m4, ta, ma  \n\t"
        "vle8.v      v4, (%[x])                \n\t"
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
        /* 合并：v22 = g3；lane 0..11←g2，0..7←g1，0..3←g0（tu 保尾） */
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
        "vfmul.vv    v28, v28, v22             \n\t"   /* s·f32(i) */
        "vfmul.vf    v28, v28, ft0             \n\t"   /* ·xd */
        "vfadd.vv    v24, v24, v28             \n\t"   /* acc + (...) */
        "addi        %[w], %[w], 288           \n\t"
        "addi        %[x], %[x], 128           \n\t"
        "addi        %[d], %[d], 4             \n\t"
        "addi        %[c], %[c], 4             \n\t"
        "addi        %[k], %[k], -1            \n\t"
        "bnez        %[k], 1b                  \n\t"
        "vse32.v     v24, (%[o])               \n\t"
        ".option pop                           \n\t"
        : [w] "+r"(w), [x] "+r"(xr), [d] "+r"(xd), [c] "+r"(corr), [k] "+r"(k)
        : [o] "r"(out)
        : "t0", "t1", "t2", "ft0", "memory", "vl", "vtype",
          "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11",
          "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
          "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
}

/* ---- 驱动 ---- */
enum { M_COL = 0, M_DEC = 1, M_VM = 2 };
static const char *MN[3] = { "COL(rvv,默认)", "DEC(rvv,Q4_DEC)", "VM(vmadot)" };

typedef struct {
    const char *name; int rows, nblk;
    uint8_t *w576, *w288;
    int8_t *xq; float *xd;                 /* 已量化激活（三模式共用，量化不计时） */
    int32_t *sx, *corr; int8_t *xr;        /* 模式私有前处理（计时内） */
    float *out, *resid;
} mat_t;

static void *amalloc(size_t n) { void *p = aligned_alloc(64, (n + 63) & ~(size_t)63); if (!p) { perror("alloc"); exit(1); } return p; }

static void prep(mat_t *m, int mode) {
    if (mode == M_DEC) {
        for (int b = 0; b < m->nblk; b++) { int32_t s = 0; for (int k = 0; k < 32; k++) s += m->xq[b * 32 + k]; m->sx[b] = s; }
    } else if (mode == M_VM) {
        for (int b = 0; b < m->nblk; b++) {
            const int8_t *x = m->xq + b * 32; int8_t *r = m->xr + (size_t)b * 128; int32_t s = 0;
            for (int q = 0; q < 4; q++) for (int rr = 0; rr < 4; rr++) memcpy(r + q * 32 + rr * 8, x + q * 8, 8);
            for (int k = 0; k < 32; k++) s += x[k];
            m->corr[b] = 8 * s;
        }
    }
}

static void do_tile(const mat_t *m, int mode, int t) {
    if (mode == M_COL)
        rv_q4_col_tile(m->w576 + (size_t)t * m->nblk * 576, m->xq, m->xd, m->out + t * 32,
                       m->resid ? m->resid + t * 32 : NULL, m->nblk);
    else if (mode == M_DEC)
        rv_q4_dec_tile(m->w576 + (size_t)t * m->nblk * 576, m->xq, m->xd, m->sx, m->out + t * 32,
                       m->resid ? m->resid + t * 32 : NULL, m->nblk);
    else {
        float *o = m->out + t * 16;
        vm_tile16(m->w288 + (size_t)t * m->nblk * 288, m->xr, m->xd, m->corr, m->nblk, o);
        if (m->resid) { const float *r = m->resid + t * 16; for (int i = 0; i < 16; i++) o[i] = o[i] + r[i]; }
    }
}

static int g_badcpu = 0;
static void run_mat(mat_t *m, int mode, int nt) {
    prep(m, mode);
    const int ntile = mode == M_VM ? m->rows / 16 : m->rows / 32;
    if (nt == 1) { for (int t = 0; t < ntile; t++) do_tile(m, mode, t); return; }
    int next = 0;
    #pragma omp parallel num_threads(nt)
    {
        if (sched_getcpu() != omp_get_thread_num()) __atomic_add_fetch(&g_badcpu, 1, __ATOMIC_RELAXED);
        for (;;) {
            int t0 = __atomic_fetch_add(&next, 2, __ATOMIC_RELAXED);   /* DYN_GRAIN=2 同生产 */
            if (t0 >= ntile) break;
            for (int t = t0; t < t0 + 2 && t < ntile; t++) do_tile(m, mode, t);
        }
    }
}

static void pin_team(int nt) {
    #pragma omp parallel num_threads(nt)
    {
        cpu_set_t s; CPU_ZERO(&s); CPU_SET(omp_get_thread_num(), &s);
        sched_setaffinity(0, sizeof s, &s);
        if (sched_getcpu() != omp_get_thread_num()) __atomic_add_fetch(&g_badcpu, 1, __ATOMIC_RELAXED);
    }
}

static uint16_t rnd_f16(int emin, int emax) {    /* 正规数 f16，符号随机 */
    int e = emin + (int)(rnd() % (uint64_t)(emax - emin + 1));
    return (uint16_t)(((rnd() & 1) << 15) | (e << 10) | (rnd() & 0x3FF));
}
static void fill_w(uint8_t *w, int rows, int nblk, int pat) {
    for (int ti = 0; ti < rows / 32; ti++) for (int b = 0; b < nblk; b++) {
        uint8_t *sc = w + ((size_t)ti * nblk + b) * 576;
        for (int r = 0; r < 32; r++) { uint16_t h = pat ? 0x3C00 : rnd_f16(5, 20); memcpy(sc + 2 * r, &h, 2); }
        for (int i = 0; i < 512; i++)
            sc[64 + i] = pat == 1 ? 0x00 : pat == 2 ? 0xFF : pat == 3 ? 0x0F : (uint8_t)rnd();
    }
}
static void repack(mat_t *m) { for (int ti = 0; ti < m->rows / 32; ti++) rp_tile(m->w576, m->w288, m->nblk, ti); }

static mat_t mk(const char *name, int rows, int K, int has_resid) {
    mat_t m; memset(&m, 0, sizeof m);
    m.name = name; m.rows = rows; m.nblk = K / 32;
    m.w576 = amalloc((size_t)rows * m.nblk * 18); m.w288 = amalloc((size_t)rows * m.nblk * 18);
    m.xq = amalloc(K); m.xd = amalloc(m.nblk * 4); m.sx = amalloc(m.nblk * 4);
    m.corr = amalloc(m.nblk * 4); m.xr = amalloc((size_t)m.nblk * 128);
    m.out = amalloc(rows * 4); m.resid = has_resid ? amalloc(rows * 4) : NULL;
    fill_w(m.w576, rows, m.nblk, 0); repack(&m);
    for (int i = 0; i < K; i++) m.xq[i] = (int8_t)((int)(rnd() % 255) - 127);
    for (int b = 0; b < m.nblk; b++) { uint16_t h = rnd_f16(5, 15); m.xd[b] = st_f16scale((uint8_t *)&h); if (m.xd[b] < 0) m.xd[b] = -m.xd[b]; }
    if (m.resid) for (int i = 0; i < rows; i++) m.resid[i] = (float)((int)(rnd() % 2001) - 1000) * 0.01f;
    return m;
}

/* 返回位失配数；ref 为单线程 COL */
static int cmp_mode(mat_t *m, int mode, int nt, float *ref, double *mx) {
    run_mat(m, mode, nt);
    int bad = 0;
    for (int i = 0; i < m->rows; i++) {
        if (memcmp(&m->out[i], &ref[i], 4)) bad++;
        double d = (double)m->out[i] - ref[i]; if (d < 0) d = -d; if (d > *mx) *mx = d;
    }
    return bad;
}
static int cmp_dbl(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }

/* 极值探针：scale=1、xd=1 → 输出 = Σ_b f32(int_b)，与 int64 精确参考比较（证 int32 精确累加） */
static int probe_extreme(void) {
    const int rows = 64, K = 9728; int fail = 0;
    const char *wp[4] = { "nib=0", "nib=15", "lo15hi0", "random" };
    const char *xp[3] = { "x=+127", "x=-127", "x=±127交替" };
    for (int p = 1; p <= 4; p++) for (int xv = 0; xv < 3; xv++) {
        mat_t m = mk("ext", rows, K, 0);
        fill_w(m.w576, rows, m.nblk, p); repack(&m);
        for (int i = 0; i < K; i++) m.xq[i] = (int8_t)(xv == 0 ? 127 : xv == 1 ? -127 : ((i & 1) ? -127 : 127));
        for (int b = 0; b < m.nblk; b++) m.xd[b] = 1.0f;
        float *ref = amalloc(rows * 4);
        run_mat(&m, M_COL, 1); memcpy(ref, m.out, rows * 4);
        double mx = 0; int bc = cmp_mode(&m, M_VM, 1, ref, &mx), bi = 0; int64_t amax = 0;
        for (int r = 0; r < rows; r++) {
            int64_t s = 0;
            for (int b = 0; b < m.nblk; b++) {
                const uint8_t *dt = m.w576 + ((size_t)(r / 32) * m.nblk + b) * 576 + 64; int64_t sb = 0;
                for (int c = 0; c < 32; c++) sb += (int64_t)(((dt[c * 16 + (r % 32) / 2] >> ((r & 1) * 4)) & 15) - 8) * m.xq[b * 32 + c];
                if (sb > amax) amax = sb; if (-sb > amax) amax = -sb;
                s += sb;
            }
            float e = (float)s; if (memcmp(&e, &m.out[r], 4)) bi++;
        }
        printf("  [EXT] %-8s %-12s VM-vs-COL bad=%d  VM-vs-int64 bad=%d  max|块和|=%lld\n",
               wp[p - 1], xp[xv], bc, bi, (long long)amax);
        fail += bc + bi;
        free(ref);
    }
    return fail;
}

int main(int argc, char **argv) {
    const int R = argc > 1 ? atoi(argv[1]) : 15;
    cpu_set_t orig; sched_getaffinity(0, sizeof orig, &orig);
    omp_set_dynamic(0);
    { cpu_set_t s; CPU_ZERO(&s); CPU_SET(0, &s); sched_setaffinity(0, sizeof s, &s); }
    pin_team(4);
    printf("bench_dec_vmadot  Qwen3-4B 单层  R=%d  pin badcpu=%d\n", R, g_badcpu);

    printf("== 前置②：int32 精确 / 极值 ==\n");
    int ext_fail = probe_extreme();

    double t0 = now();
    mat_t L[7] = { mk("q", 4096, 2560, 0), mk("k", 1024, 2560, 0), mk("v", 1024, 2560, 0),
                   mk("o", 2560, 4096, 1), mk("gate", 9728, 2560, 0), mk("up", 9728, 2560, 0),
                   mk("down", 2560, 9728, 1) };
    double bytes = 0; for (int i = 0; i < 7; i++) bytes += (double)L[i].rows * L[i].nblk * 18;
    printf("建数据+重排 %.1fs  每格式 %.1f MB/层\n", now() - t0, bytes / 1e6);

    printf("== 前置①③ L1：逐元素比位（ref = COL 单线程） ==\n");
    int l1_fail = 0;
    for (int i = 0; i < 7; i++) {
        mat_t *m = &L[i]; float *ref = amalloc(m->rows * 4);
        run_mat(m, M_COL, 1); memcpy(ref, m->out, m->rows * 4);
        double a = 0, b = 0, c = 0, d = 0;
        int bc4 = cmp_mode(m, M_COL, 4, ref, &a), bd = cmp_mode(m, M_DEC, 4, ref, &b);
        int bv1 = cmp_mode(m, M_VM, 1, ref, &c), bv4 = cmp_mode(m, M_VM, 4, ref, &d);
        printf("  %-5s rows=%-5d K=%-5d resid=%d  COLt4 bad=%d  DECt4 bad=%d  VMt1 bad=%d  VMt4 bad=%d  max|d|=%.3g\n",
               m->name, m->rows, m->nblk * 32, m->resid != NULL, bc4, bd, bv1, bv4, c > d ? c : d);
        l1_fail += bc4 + bd + bv1 + bv4;
        free(ref);
    }

    printf("== 吞吐（单层 7 矩阵，模式轮换，中位数） ==\n");
    const int nts[2] = { 1, 4 };
    for (int ti = 0; ti < 2; ti++) {
        const int nt = nts[ti];
        double *ts[3]; for (int k = 0; k < 3; k++) ts[k] = malloc(R * sizeof(double));
        for (int w = 0; w < 2; w++) for (int k = 0; k < 3; k++) for (int i = 0; i < 7; i++) run_mat(&L[i], k, nt);
        for (int r = 0; r < R; r++) for (int j = 0; j < 3; j++) {
            const int k = (r + j) % 3; double a = now();
            for (int i = 0; i < 7; i++) run_mat(&L[i], k, nt);
            ts[k][r] = now() - a;
        }
        double med[3];
        for (int k = 0; k < 3; k++) {
            qsort(ts[k], R, sizeof(double), cmp_dbl); med[k] = ts[k][R / 2];
            printf("  t%d %-16s med %.3f ms  min %.3f ms  %.2f GB/s\n", nt, MN[k], med[k] * 1e3, ts[k][0] * 1e3, bytes / med[k] / 1e9);
        }
        printf("  t%d VM 相对 COL %+.1f%%  相对 DEC %+.1f%%（正=更快）\n", nt,
               (med[M_COL] / med[M_VM] - 1) * 100, (med[M_DEC] / med[M_VM] - 1) * 100);
        for (int k = 0; k < 3; k++) free(ts[k]);
    }
    printf("RESULT ext_fail=%d l1_fail=%d badcpu=%d\n", ext_fail, l1_fail, g_badcpu);
    sched_setaffinity(0, sizeof orig, &orig);
    return (ext_fail || l1_fail || g_badcpu) ? 1 : 0;
}
