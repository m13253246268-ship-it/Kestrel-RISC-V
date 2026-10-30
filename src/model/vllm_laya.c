/**
 * vllm_laya.c - Laya decision model: ModernBERT-large encoder + decision head.
 *
 * Faithful scalar-F32 port of transformers ModernBertModel + the Laya
 * DecisionModel head (rl_common.py). Correctness-first (no SIMD yet).
 *
 * Pinned semantics (transformers/models/modernbert/modeling_modernbert.py):
 *   embeddings : h = LayerNorm(tok_embed(ids))                 (norm_bias=false)
 *   layer i    : attn_norm(i==0 -> Identity), attention, residual;
 *                mlp_norm, MLP, residual
 *   attention  : bidirectional; sliding layers limit keys to |q-kv| <= local/2 (=64)
 *   RoPE       : inv_freq[j]=1/theta^(2j/hd); emb=cat(freqs,freqs); theta full=160000 sliding=10000
 *   MLP        : input,gate = Wi(x).chunk(2); Wo(gelu(input)*gate)   [exact erf gelu]
 *   head       : h += type_emb[qtype]; 2x TransformerEncoderLayer(norm_first, relu);
 *                scorer = LN -> Linear -> GELU -> Linear(->1) on each marker position
 */
#include "vllm_laya.h"
#include "vllm_platform.h"         /* ST_HAVE_NEON / ST_ARCH_X86 (SIMD 选择, K-LAYA-002) */
#include "vllm_tokenizer_qwen.h"   /* K-LAYA-001: reuse engine BPE (vocab.bin+merges.bin) */
#include "vllm_http.h"             /* vjson: request/response for --laya-decide */

#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* K-LAYA-002 M4：位置级并行的线程号（无 OpenMP 时恒 0 ⇒ 纯串行、逐位不变）。 */
#ifdef _OPENMP
#include <omp.h>
#define OTID omp_get_thread_num()
#else
#define OTID 0
#endif

/* K-LAYA-003 L1：IME（vmadot）路径需要按 hart 绑核（只有 hart0-3 有该扩展）。 */
#if ST_ARCH_RISCV && defined(_OPENMP)
#include <sched.h>
#endif

/* K-LAYA-002 M6（诊断）：env 门控的阶段计时（VLLM_LAYA_TIMING=1）。
 * 仅测量、不改行为；关闭时开销为 0。仅在 laya_forward 外层（串行上下文）累加。 */
enum { TT_EMB, TT_NORM, TT_MM, TT_ROPE, TT_ATTN, TT_ACT, TT_RESID, TT_SCO, TT_N };
static double g_tt[TT_N] = {0};
static double g_tt_total = 0;
static int    g_timing = -1;
/* K-LAYA-003：IME 路径的两个子项单独累计（打包是一次性成本，不能算进稳态） */
static double g_ime_pack_ms = 0, g_ime_gemm_ms = 0;
static double lay_wtime(void) {
#ifdef _OPENMP
    return omp_get_wtime();
#else
    return 0.0;
#endif
}
static inline double tmark(void) { return g_timing == 1 ? lay_wtime() : 0.0; }
static inline void tacc(int i, double t0) { if (g_timing == 1) g_tt[i] += lay_wtime() - t0; }

#define LAYA_D     1024
#define LAYA_H     16
#define LAYA_HD    64
#define LAYA_INTER 2624
#define LAYA_NL    28
#define LAYA_EPS   1e-5f
#define LAYA_HFFN  4096
#define LAYA_SLIDE 64
#define LAYA_NT_MAX 16   /* K-LAYA-002 M4：位置并行的最大线程数（scratch 池上限） */
#ifndef LAYA_QB
#define LAYA_QB    512   /* K-LAYA-002 M5：int8 量化块大小（元素/块）。
                          * M5d：板端微基准 t4r1——块 32→128 提速 ~2.0×（27→55 GF/s，
                          * 每块 scale 开销被摊薄）；256/1024 边际递减。代价：粒度 32→128
                          * （公理 collapse_priority_rule 张力点，须以 J2 决策等价实测守门）。
                          * 注：SCALE 块 = LAYA_QB；vdotq 处理 CHUNK 仍恒 32 字节。
                          * K-LAYA-003 L3：可用 -DLAYA_QB=<n> 覆盖以 A/B 更大的块
                          * （每块一次 vredsum，块越大归约越省；代价是 scale 更粗）。 */
#endif

typedef struct {
    char   name[96];
    int    ndim;
    int    ne[4];
    int    is_f32;
    size_t off;
    size_t nbytes;
    const uint16_t *f16; /* K-LAYA-002 M2：F16 张量指向 blob（不再展开 F32 arena） */
    const float    *f32; /* F32 张量指向 blob（如 temperature） */
    const int8_t   *i8;  /* K-LAYA-002 M5：2D 权重的 int8 表示（per-LAYA_QB 块对称量化） */
    const float    *qsc; /* int8 权重的 per-LAYA_QB 块 scale（行主序，每行 nblk 个） */
    int    rowlen;       /* M5：行宽（2D 为 ne[1]=K，行主序连续维） */
    int    nblk;         /* M5：每行的量化块数 = ceil(rowlen/LAYA_QB) */
    size_t n_elem;
    int8_t *ime_a;       /* K-LAYA-003：IME 打包权重（惰性、仅 VLLM_LAYA_IME=1 时分配） */
} LayaEnt;

struct LayaModel {
    unsigned char *blob;     /* K-LAYA-002 M2：整个 laya_f16.bin 常驻（843 MB），M5 载入后释放 */
    size_t         blob_size;
    float         *f32pool;  /* 1D 张量（norm/bias）展开的 F32 小池（<0.5 MB） */
    int8_t        *i8pool;   /* K-LAYA-002 M5：全部 2D 权重的 int8 arena（~421 MB） */
    float         *qscpool;  /* K-LAYA-002 M5：int8 权重的 per-128 块 scale（~53 MB） */
    LayaEnt       *ent;
    int            n_ent;
    /* K-LAYA-002 P2：RoPE 表缓存。theta 固定（160000 全 / 10000 滑窗）⇒ 表只依赖
     * 序列长度 n，逐请求重算 pow/cosf/sinf 是纯浪费。cosf/sinf/pow 确定性 ⇒
     * 复用与重算**逐位同值**。容量按 LAYA_HD（64）而非 LAYA_D（1024）分配。
     * 线程安全：调用方须串行化同一 model 上的 laya_forward（服务端由 g_laya_lock
     * 保证；lay_dot 档位原本也依赖串行初始化）。 */
    int            rope_cap;
    float         *rope_cosF, *rope_sinF, *rope_cosS, *rope_sinS;
};

/* ---------------- fp16 -> fp32 ---------------- */
static float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t man = h & 0x3FFu;
    uint32_t o;
    if (exp == 0) {
        if (man == 0) { o = sign; }
        else {
            exp = 127 - 15 + 1;
            while (!(man & 0x400u)) { man <<= 1; exp--; }
            man &= 0x3FFu;
            o = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 31) {
        o = sign | 0x7F800000u | (man << 13);
    } else {
        o = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float f;
    memcpy(&f, &o, 4);
    return f;
}

static const LayaEnt *lay_find(const LayaModel *m, const char *name) {
    for (int i = 0; i < m->n_ent; i++)
        if (strcmp(m->ent[i].name, name) == 0) return &m->ent[i];
    fprintf(stderr, "[LAYA] missing tensor: %s\n", name);
    return NULL;
}

static inline float lay_at(const LayaModel *m, const LayaEnt *e, size_t i) {
    (void)m;
    if (e->i8) {   /* M5：int8 反量化（按行主序定位块 scale） */
        size_t r = i / (size_t)e->rowlen, c = i % (size_t)e->rowlen;
        return (float)e->i8[i] * e->qsc[r * (size_t)e->nblk + (c / LAYA_QB)];
    }
    return e->f16 ? f16_to_f32(e->f16[i]) : e->f32[i];
}

/* ------------------------------------------------------------------
 * K-LAYA-002 M1：F32 点积。aarch64 4 累加器 × 16 lane FMA；x86-64 AVX2 8 lane FMA；
 * RISC-V 无 FMA 版（-ffp-contract=off 下走标量单链，与 M10 的 RVV 参考同序）。
 * 公理 superposition_addition_base（lane 规约）+ conservation_law_under_collapse
 * （精度守恒：与标量同语义，仅规约顺序不同 ⇒ 容差门内）。
 * aarch64 → NEON vfmaq_f32；x86-64(-mavx2 -mfma) → AVX2 _mm256_fmadd_ps；
 * 其它 → 标量。尾数走标量补齐（LAYA 主维均 4/8 的倍数，实际无尾）。
 * ------------------------------------------------------------------ */
static inline float lay_dot(const float *a, const float *b, int n) {
#if ST_HAVE_NEON
    /* 4 累加器 × unroll16：隐藏 A76 FMA 延迟（4c），提高发射密度 */
    float32x4_t a0 = vdupq_n_f32(0.f), a1 = vdupq_n_f32(0.f);
    float32x4_t a2 = vdupq_n_f32(0.f), a3 = vdupq_n_f32(0.f);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        a0 = vfmaq_f32(a0, vld1q_f32(a + i),      vld1q_f32(b + i));
        a1 = vfmaq_f32(a1, vld1q_f32(a + i + 4),  vld1q_f32(b + i + 4));
        a2 = vfmaq_f32(a2, vld1q_f32(a + i + 8),  vld1q_f32(b + i + 8));
        a3 = vfmaq_f32(a3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }
    for (; i + 4 <= n; i += 4)
        a0 = vfmaq_f32(a0, vld1q_f32(a + i), vld1q_f32(b + i));
    float s = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
    for (; i < n; i++) s += a[i] * b[i];
    return s;
#elif ST_ARCH_X86 && defined(__AVX2__) && defined(__FMA__)
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),      _mm256_loadu_ps(b + i),      a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8),  _mm256_loadu_ps(b + i + 8),  a1);
        a2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), a2);
        a3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), a3);
    }
    for (; i + 8 <= n; i += 8)
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), a0);
    __m256 s8 = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
    __m128 lo = _mm256_castps256_ps128(s8);
    __m128 hi = _mm256_extractf128_ps(s8, 1);
    __m128 s4 = _mm_add_ps(lo, hi);
    s4 = _mm_hadd_ps(s4, s4);
    s4 = _mm_hadd_ps(s4, s4);
    float s = _mm_cvtss_f32(s4);
    for (; i < n; i++) s += a[i] * b[i];
    return s;
#else
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
#endif
}

/* ------------------------------------------------------------------
 * K-LAYA-002 M2：F16 权重 · F32 激活 点积（F32 累加）。
 * 公理 collapse_to_classic（F16 存储）+ conservation_law_under_collapse
 * （F16→F32 转换无损 ⇒ 与 M1 的 F32 arena 逐位同值，仅累加顺序不同）。
 * aarch64：vcvt_f32_f16(vld1_f16) + vfmaq_f32；x8664：_mm256_cvtph_ps + fmadd。
 * 尾数走标量补齐。
 * ------------------------------------------------------------------ */
static inline float lay_dot_f16(const uint16_t *w, const float *x, int n) {
#if ST_HAVE_NEON
    float32x4_t a0 = vdupq_n_f32(0.f), a1 = vdupq_n_f32(0.f);
    float32x4_t a2 = vdupq_n_f32(0.f), a3 = vdupq_n_f32(0.f);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        a0 = vfmaq_f32(a0, vcvt_f32_f16(vld1_f16((const __fp16 *)(w + i))),      vld1q_f32(x + i));
        a1 = vfmaq_f32(a1, vcvt_f32_f16(vld1_f16((const __fp16 *)(w + i + 4))),  vld1q_f32(x + i + 4));
        a2 = vfmaq_f32(a2, vcvt_f32_f16(vld1_f16((const __fp16 *)(w + i + 8))),  vld1q_f32(x + i + 8));
        a3 = vfmaq_f32(a3, vcvt_f32_f16(vld1_f16((const __fp16 *)(w + i + 12))), vld1q_f32(x + i + 12));
    }
    for (; i + 4 <= n; i += 4)
        a0 = vfmaq_f32(a0, vcvt_f32_f16(vld1_f16((const __fp16 *)(w + i))), vld1q_f32(x + i));
    float s = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
    for (; i < n; i++) s += f16_to_f32(w[i]) * x[i];
    return s;
#elif ST_ARCH_X86 && defined(__AVX2__) && defined(__F16C__) && defined(__FMA__)
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        a0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w + i))),      _mm256_loadu_ps(x + i),      a0);
        a1 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w + i + 8))),  _mm256_loadu_ps(x + i + 8),  a1);
        a2 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w + i + 16))), _mm256_loadu_ps(x + i + 16), a2);
        a3 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w + i + 24))), _mm256_loadu_ps(x + i + 24), a3);
    }
    for (; i + 8 <= n; i += 8)
        a0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w + i))), _mm256_loadu_ps(x + i), a0);
    __m256 s8 = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
    __m128 lo = _mm256_castps256_ps128(s8);
    __m128 hi = _mm256_extractf128_ps(s8, 1);
    __m128 s4 = _mm_add_ps(lo, hi);
    s4 = _mm_hadd_ps(s4, s4);
    s4 = _mm_hadd_ps(s4, s4);
    float s = _mm_cvtss_f32(s4);
    for (; i < n; i++) s += f16_to_f32(w[i]) * x[i];
    return s;
#else
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += f16_to_f32(w[i]) * x[i];
    return s;
#endif
}

/* ------------------------------------------------------------------
 * K-LAYA-002 M5：int8 权重 · int8 激活 点积（per-128=LAYA_QB 块对称量化，int32 累加）。
 * 公理 collapse_to_classic + collapse_priority_rule（细粒度块 scale ⇒ 避免单
 * scale 的不可逆失真）；conservation_law_under_collapse（confidence=0.65 弱约束
 * ⇒ 以 J2 决策输出等价实测验证，不靠理论保证）。
 * aarch64：vcvt_f32_f16 无；用 vdotq_s32（ARMv8.2 dotprod，A76/RK3588 支持）
 * 每指令 16 int8 → 4×int32 累加；逐块 diag(sw)*diag(sx) 缩放后 F32 累加。
 * x86/无 dotprod：标量兜底。
 * ------------------------------------------------------------------ */

#if ST_HAVE_RVV
#include <riscv_vector.h>
#endif

/* ------------------------------------------------------------------
 * K-LAYA-002 M9：RISC-V int8 点积（VLLM_LAYA_RVV 三档，**缺省 = 2**）。
 *
 *   0 = 逐元素 float 链（历史部署基线；s += w*x*sw*sx，j 升序单链，128 次舍入）
 *   1 = 标量 int32 精确块累加（每 128 块用 int32 精确求 Σw·x，块尾一次
 *       s += (float)acc * (sw*sx)；与 ARM vdotq 分支结构完全相同）
 *   2 = RVV int32 精确块累加（= 档 1 的向量实现）
 *
 * 位级口径（perf_axiom_bitwise_gate_v1 / reduction_tree_sensitivity_v1）：
 *   - 档 2 对档 1 **逐位相同**：整数加法精确且可交换/结合 ⇒ 块内 int32 之和
 *     与 lane 顺序无关（唯一值）；块间 float 链仍是 b 升序单链、每块只做一次
 *     s += (float)acc * (sw*sx) ⇒ 舍入点与顺序逐字一致。K%128 的尾块沿用档 0
 *     的表达式 `((w*sw)*x)*sx`（注意与块循环里的 `((w*x)*sw)*sx` 分组不同，
 *     必须原样照抄才能逐位一致）。
 *   - 档 1/2 对档 0 是**改数值路径**（每元素 1 次 float 加法 / 每块 1 次），
 *     非位级等价。档 2/1 采用 int32 精确块累加，**更接近 double 真值**（相对
 *     档 0 提升 ~16.7×，max rel err 3.46e-3 → 7.17e-4）；代价是部署 logits 相对
 *     档 0 位移（n=59 q0 5.992369→5.988916 Δ=3.45e-3；n=480 q5 最大 ~0.16），
 *     该口径差异随默认化一并披露（P0 已披露）。
 *   - 顺序浮点链在数学上无法向量化（s 与 p_j 的串行依赖不可重排），所以
 *     档 2 不可能与档 0 位级相同——差异只能隔离到档 0↔1 这一步。
 *
 * 独立性（审查 P2）：**不复用 VLLM_RVV**。那个变量属于 st_q8 取数内核
 * （0=回标量、1/2/3=不同取数方式、彼此位级一致），两处「0 的含义」不同：
 * 在 st_q8 里 0 是位级安全的关，在这里 0 是数值精度档。共用会让"为 st_q8
 * 设 VLLM_RVV=0 做回归"静默改变 Laya 部署输出，故拆开。
 *
 * 默认档记录：
 *   - P9（2026-10-26）曾回默认 0：因当时的授权依据不足 + 极差 2.2% 超 1% 门。
 *   - P0（2026-10-29，用户显式拍板）默认切回 2。证据缺口已补：
 *       决策等价门 12/12（4 场景×{choice/score/noul}×选项{3,8}，argmax 全一致，
 *       max|Δp|=3.17e-2 < 5e-2 J2 门）；板端配对 8.5–9.7×（mm 11–13×），
 *       L2 极差 1.03–1.20%（板上测量噪声，略超 1% 名义门，如实披露）。
 *       逃生舱 VLLM_LAYA_RVV=0 保留；档 1 作为与档 2 位级全等的标量参考保留。
 * ------------------------------------------------------------------ */
static int lay_dot_level(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("VLLM_LAYA_RVV");
        if (!e || !e[0]) v = 2;                        /* K-LAYA-002 P0：缺省 = 档 2（RVV int32）。
                                                        *   用户 2026-10-29 显式拍板默认化。证据：
                                                        *   决策等价门 12/12（4 场景×类型，argmax 全一致，
                                                        *   max|Δp|=3.17e-2 < 5e-2）；板端配对 8.5–9.7×
                                                        *   （mm 11–13×）；极差 1.03–1.20%（板上测量噪声，
                                                        *   略超 default_enablement 1% 名义门，如实披露）。
                                                        *   逃生舱 VLLM_LAYA_RVV=0 保留。 */
        else if (e[0] == '0') v = 0;
        else if (e[0] == '1') v = 1;
        else if (e[0] == '2') v = 2;
        else v = 2;                                    /* 越界/异常 → 安全侧（保持新默认水平） */
#if ST_ARCH_RISCV
        fprintf(stderr, "[LAYA-RVV] int8 dot level=%d"
                        " (0=float 基线, 1=标量 int32, 2=RVV int32[默认])\n", v);
        fflush(stderr);
#endif
    }
    return v;
}

#if ST_HAVE_RVV
/* e8m1 的 VLMAX = VLEN/8（256 位机 = 32）。每次点积只读一次 cache。 */
static inline size_t lay_rvv_vl8(void) {
    static size_t v = 0;
    if (!v) v = __riscv_vsetvlmax_e8m1();
    return v;
}

/* 一个 128 块内的 int32 精确点积 + RVV 版 lay_dot_i8（见上方口径说明）。
 * 归约长度取 min(vmax, LAYA_QB)：块内真正被写过的 lane 只有这么多，用 vmax 归约
 * 在 VLEN>1024（vmax>128）时会读到未定义尾 lane。VLEN=256 时两者都等于 32。 */
static inline float lay_dot_i8_rvv(const int8_t *w, const float *sw,
                                   const int8_t *x, const float *sx, int K) {
    const size_t vmax = lay_rvv_vl8();
    const size_t vred = vmax < (size_t)LAYA_QB ? vmax : (size_t)LAYA_QB;
    /* K-LAYA-003 L0：两条流合并成 vwmacc（板端微基准 2026-10-30，单线程 K=1024）：
     *   vwmul+vwadd          2.40 GMAC/s
     *   vwmul+vwmacc → i16   3.31 GMAC/s（+38%，本轮最优）
     *   int16 权重 vwmacc    2.25 / 2.12 GMAC/s（m2/m4，取数翻倍 ⇒ 更差）
     *   int8 拆 2/4 条 i32 链 2.72 / 2.76 GMAC/s（非依赖链受限，仅 +11%）
     * 位级：i16 只装 2 个乘积，|Σ| ≤ 2×127×127 = 32258 < 32767 不溢出；
     * 两条流仍并入同一个 i32 精确和，128 块内仍是「精确整数和」⇒ 与逐个
     * vwmul+vwadd 逐位同值（i32 加法精确且可结合）。 */
    const int pair_ok = (2 * (int)vmax <= LAYA_QB) && (LAYA_QB % (2 * (int)vmax) == 0);
    float s = 0.0f;
    int b = 0;
    for (; b + LAYA_QB <= K; b += LAYA_QB) {
        vint32m4_t acc = __riscv_vmv_v_x_i32m4(0, vmax);
        if (pair_ok) {
            for (int c = 0; c < LAYA_QB; c += 2 * (int)vmax) {
                size_t vl = __riscv_vsetvl_e8m1(vmax);
                vint16m2_t p = __riscv_vwmul_vv_i16m2(
                    __riscv_vle8_v_i8m1(w + b + c, vl),
                    __riscv_vle8_v_i8m1(x + b + c, vl), vl);
                p = __riscv_vwmacc_vv_i16m2(
                    p, __riscv_vle8_v_i8m1(w + b + c + (int)vmax, vl),
                       __riscv_vle8_v_i8m1(x + b + c + (int)vmax, vl), vl);
                acc = __riscv_vwadd_wv_i32m4(acc, p, vl);
            }
        } else {
            for (int c = 0; c < LAYA_QB; c += (int)vmax) {
                size_t rem = (size_t)(LAYA_QB - c);
                size_t vl = __riscv_vsetvl_e8m1(rem < vmax ? rem : vmax);
                vint8m1_t wv = __riscv_vle8_v_i8m1(w + b + c, vl);
                vint8m1_t xv = __riscv_vle8_v_i8m1(x + b + c, vl);
                /* |w*x| <= 127*127 = 16129 < 2^15 ⇒ 先 i16 精确乘积，再拓宽累加进 i32 */
                vint16m2_t p = __riscv_vwmul_vv_i16m2(wv, xv, vl);
                acc = __riscv_vwadd_wv_i32m4(acc, p, vl);
            }
        }
        int32_t tot = __riscv_vmv_x_s_i32m1_i32(
            __riscv_vredsum_vs_i32m4_i32m1(acc, __riscv_vmv_v_x_i32m1(0, 1), vred));
        s += (float)tot * (sw[b / LAYA_QB] * sx[b / LAYA_QB]);
    }
    for (; b < K; b++)   /* 尾块：表达式与档 0 逐字相同 */
        s += (float)w[b] * sw[b / LAYA_QB] * (float)x[b] * sx[b / LAYA_QB];
    return s;
}
#endif

static inline float lay_dot_i8(const int8_t *w, const float *sw,
                               const int8_t *x, const float *sx, int K) {
#if ST_HAVE_NEON && defined(__ARM_FEATURE_DOTPROD)
    float32x4_t sumv = vdupq_n_f32(0.f);
    int b = 0;
    for (; b + LAYA_QB <= K; b += LAYA_QB) {
        int32x4_t acc = vdupq_n_s32(0);
        for (int c = 0; c < LAYA_QB; c += 32) {   /* SCALE 块内按 32 字节 CHUNK 做 vdotq */
            acc = vdotq_s32(acc, vld1q_s8(w + b + c),      vld1q_s8(x + b + c));
            acc = vdotq_s32(acc, vld1q_s8(w + b + c + 16), vld1q_s8(x + b + c + 16));
        }
        sumv = vmlaq_n_f32(sumv, vcvtq_f32_s32(acc), sw[b / LAYA_QB] * sx[b / LAYA_QB]);
    }
    float s = vaddvq_f32(sumv);
    for (; b < K; b++) s += (float)w[b] * sw[b / LAYA_QB] * (float)x[b] * sx[b / LAYA_QB];
    return s;
#elif ST_ARCH_RISCV
    /* K-LAYA-002 M9：三档（见 lay_dot_level 上方口径）。level>=1 与档 0 是非位级
     * 等价（顺序浮点链不可向量化），故默认 level=0 保留完整回退。 */
    const int lv = lay_dot_level();
#if ST_HAVE_RVV
    if (lv >= 2) return lay_dot_i8_rvv(w, sw, x, sx, K);
#endif
    if (lv >= 1) {
        float s = 0.0f;
        for (int b = 0; b + LAYA_QB <= K; b += LAYA_QB) {
            int32_t acc = 0;
            for (int j = 0; j < LAYA_QB; j++)
                acc += (int32_t)w[b + j] * (int32_t)x[b + j];
            s += (float)acc * (sw[b / LAYA_QB] * sx[b / LAYA_QB]);
        }
        for (int b = (K / LAYA_QB) * LAYA_QB; b < K; b++)
            s += (float)w[b] * sw[b / LAYA_QB] * (float)x[b] * sx[b / LAYA_QB];
        return s;
    }
    /* level 0：历史基线（逐元素 float 链） */
    {
        float s = 0.0f;
        for (int b = 0; b + LAYA_QB <= K; b += LAYA_QB)
            for (int j = 0; j < LAYA_QB; j++) s += (float)w[b + j] * (float)x[b + j] * sw[b / LAYA_QB] * sx[b / LAYA_QB];
        for (int b = (K / LAYA_QB) * LAYA_QB; b < K; b++) s += (float)w[b] * sw[b / LAYA_QB] * (float)x[b] * sx[b / LAYA_QB];
        return s;
    }
#else
    float s = 0.0f;
    for (int b = 0; b + LAYA_QB <= K; b += LAYA_QB)
        for (int j = 0; j < LAYA_QB; j++) s += (float)w[b + j] * (float)x[b + j] * sw[b / LAYA_QB] * sx[b / LAYA_QB];
    for (int b = (K / LAYA_QB) * LAYA_QB; b < K; b++) s += (float)w[b] * sw[b / LAYA_QB] * (float)x[b] * sx[b / LAYA_QB];
    return s;
#endif
}

/* K-LAYA-002 M5：把一行 F32 激活量化为 int8 + per-128 块对称 scale（写入 xq/xsc）。
 * M5c：NEON 向量化（maxabs=vmax，量化=vdivq+vcvtnq_s32_f32+vqmovn 饱和窄化）。
 * 用真除法 vdivq_f32 + round-to-nearest-even ⇒ 与标量 lrintf(x/d) **位级同值**
 * （|x/d|≤127 ⇒ [-128,127] 饱和与标量 [-127,127] 等效）。 */
static void lay_quant_act(const float *x, int8_t *xq, float *xsc, int K) {
    for (int b = 0; b < K; b += LAYA_QB) {
        int n = (b + LAYA_QB <= K) ? LAYA_QB : (K - b);
        float d;
#if ST_HAVE_NEON
        float32x4_t vmx = vdupq_n_f32(0.f);
        int j = 0;
        for (; j + 4 <= n; j += 4) vmx = vmaxq_f32(vmx, vabsq_f32(vld1q_f32(x + b + j)));
        float mx = vmaxvq_f32(vmx);
        for (; j < n; j++) { float a = fabsf(x[b + j]); if (a > mx) mx = a; }
        d = mx > 0.f ? mx / 127.f : 1.f;
        xsc[b / LAYA_QB] = d;
        float32x4_t vd = vdupq_n_f32(d);
        j = 0;
        for (; j + 16 <= n; j += 16) {
            int32x4_t q0 = vcvtnq_s32_f32(vdivq_f32(vld1q_f32(x + b + j),      vd));
            int32x4_t q1 = vcvtnq_s32_f32(vdivq_f32(vld1q_f32(x + b + j + 4),  vd));
            int32x4_t q2 = vcvtnq_s32_f32(vdivq_f32(vld1q_f32(x + b + j + 8),  vd));
            int32x4_t q3 = vcvtnq_s32_f32(vdivq_f32(vld1q_f32(x + b + j + 12), vd));
            int16x8_t h0 = vcombine_s16(vqmovn_s32(q0), vqmovn_s32(q1));
            int16x8_t h1 = vcombine_s16(vqmovn_s32(q2), vqmovn_s32(q3));
            vst1q_s8(xq + b + j, vcombine_s8(vqmovn_s16(h0), vqmovn_s16(h1)));
        }
        for (; j < n; j++) {
            int q = (int)lrintf(x[b + j] / d);
            if (q > 127) q = 127; else if (q < -127) q = -127;
            xq[b + j] = (int8_t)q;
        }
#else
        float mx = 0.f;
        for (int j = 0; j < n; j++) { float a = fabsf(x[b + j]); if (a > mx) mx = a; }
        d = mx > 0.f ? mx / 127.f : 1.f;
        xsc[b / LAYA_QB] = d;
        for (int j = 0; j < n; j++) {
            int q = (int)lrintf(x[b + j] / d);
            if (q > 127) q = 127; else if (q < -127) q = -127;
            xq[b + j] = (int8_t)q;
        }
#endif
    }
}

/* K-LAYA-002 M5：把一行 F16 权重按 per-128 块对称量化成 int8 + scale（载入期一次性）。 */
static void lay_quant_f16_row(const uint16_t *src, int8_t *dst, float *sc, int K) {
    for (int b = 0; b < K; b += LAYA_QB) {
        int n = (b + LAYA_QB <= K) ? LAYA_QB : (K - b);
        float mx = 0.f;
        for (int j = 0; j < n; j++) { float a = fabsf(f16_to_f32(src[b + j])); if (a > mx) mx = a; }
        float d = mx > 0.f ? mx / 127.f : 1.f;
        sc[b / LAYA_QB] = d;
        for (int j = 0; j < n; j++) {
            int q = (int)lrintf(f16_to_f32(src[b + j]) / d);
            if (q > 127) q = 127; else if (q < -127) q = -127;
            dst[b + j] = (int8_t)q;
        }
    }
}

/* y[o] = sum_i W[o][i]*x[i] (+B[o]); W shape [n_out, n_in]
 * 注（K-LAYA-002 M2b 实测 2026-10-10）：NEON 4 行寄存器分块（16 累加器）在
 * A76 上**退化 27%**（19.3s vs 15.3s，寄存器溢出）⇒ 已回退，逐行点积保留。
 * M5：权重/激活均为 int8（per-128 块 scale），走 lay_dot_i8。 */
static void lay_linear(const LayaModel *m, const LayaEnt *W, const LayaEnt *B,
                       const float *x, float *y, int n_out, int n_in) {
    (void)m;
    if (W->i8) {
        int nblk = (n_in + LAYA_QB - 1) / LAYA_QB;
        int8_t *xq = (int8_t *)malloc((size_t)n_in);
        float  *xs = (float *)malloc(sizeof(float) * (size_t)nblk);
        if (!xq || !xs) { free(xq); free(xs); for (int o = 0; o < n_out; o++) y[o] = 0.f; return; }
        lay_quant_act(x, xq, xs, n_in);
        for (int o = 0; o < n_out; o++) {
            float acc = lay_dot_i8(W->i8 + (size_t)o * n_in, W->qsc + (size_t)o * nblk, xq, xs, n_in);
            if (B) acc += lay_at(m, B, (size_t)o);
            y[o] = acc;
        }
        free(xq); free(xs);
        return;
    }
    const uint16_t *w16 = W->f16;
    const float    *w32 = W->f32;
    for (int o = 0; o < n_out; o++) {
        float acc = w16 ? lay_dot_f16(w16 + (size_t)o * n_in, x, n_in)
                        : lay_dot(w32 + (size_t)o * n_in, x, n_in);
        if (B) acc += lay_at(m, B, (size_t)o);
        y[o] = acc;
    }
}

/* 调试用中间量 dump（默认关）：VLLM_LAYA_DUMP=<前缀> 时把 <前缀>.enc /
 * <前缀>.head 写成裸 float32 行主序，供 tools/laya_gate.py 分段定位偏差。 */
static void lay_dump(const char *suffix, const float *data, size_t n) {
    const char *pfx = getenv("VLLM_LAYA_DUMP");
    if (!pfx || !pfx[0]) return;
    char p[1024];
    snprintf(p, sizeof(p), "%s.%s", pfx, suffix);
    FILE *f = fopen(p, "wb");
    if (f) { fwrite(data, sizeof(float), n, f); fclose(f); }
}

static void lay_ln(const LayaModel *m, const LayaEnt *W, const LayaEnt *B,
                   const float *x, float *y, int n) {
    double mean = 0.0;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= n;
    double var = 0.0;
    for (int i = 0; i < n; i++) { double d = x[i] - mean; var += d * d; }
    var /= n;
    const float invf  = (float)(1.0 / sqrt(var + LAYA_EPS));
    const float meanf = (float)mean;
    const float *w = W->f32;
    const float *bb = B ? B->f32 : NULL;
    /* K-LAYA-002 M1：归一化+缩放+偏置向量化（mean/var 保持 double 累加，与标量同口径） */
#if ST_HAVE_NEON
    float32x4_t vmean = vdupq_n_f32(meanf), vinv = vdupq_n_f32(invf);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vsubq_f32(vld1q_f32(x + i), vmean);
        v = vmulq_f32(vmulq_f32(v, vinv), vld1q_f32(w + i));
        if (bb) v = vaddq_f32(v, vld1q_f32(bb + i));
        vst1q_f32(y + i, v);
    }
    for (; i < n; i++) {
        float v = (x[i] - meanf) * invf * w[i];
        if (bb) v += bb[i];
        y[i] = v;
    }
#elif ST_ARCH_X86 && defined(__AVX2__) && defined(__FMA__)
    __m256 vmean = _mm256_set1_ps(meanf), vinv = _mm256_set1_ps(invf);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_sub_ps(_mm256_loadu_ps(x + i), vmean);
        v = _mm256_mul_ps(_mm256_mul_ps(v, vinv), _mm256_loadu_ps(w + i));
        if (bb) v = _mm256_add_ps(v, _mm256_loadu_ps(bb + i));
        _mm256_storeu_ps(y + i, v);
    }
    for (; i < n; i++) {
        float v = (x[i] - meanf) * invf * w[i];
        if (bb) v += bb[i];
        y[i] = v;
    }
#else
    for (int i = 0; i < n; i++) {
        float v = (float)((x[i] - mean) * invf) * lay_at(m, W, (size_t)i);
        if (B) v += lay_at(m, B, (size_t)i);
        y[i] = v;
    }
#endif
}

static inline float gelu_erf(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.70710678118f));
}
static inline float relu_f(float x) { return x > 0.0f ? x : 0.0f; }

static void rope_table(float *cos_t, float *sin_t, int n, float theta) {
    for (int p = 0; p < n; p++) {
        for (int j = 0; j < LAYA_HD / 2; j++) {
            double inv = 1.0 / pow((double)theta, (2.0 * j) / LAYA_HD);
            float a = (float)(p * inv);
            float c = cosf(a), s = sinf(a);
            cos_t[(size_t)p * LAYA_HD + j] = c;
            sin_t[(size_t)p * LAYA_HD + j] = s;
            cos_t[(size_t)p * LAYA_HD + LAYA_HD / 2 + j] = c;
            sin_t[(size_t)p * LAYA_HD + LAYA_HD / 2 + j] = s;
        }
    }
}

/* in-place rope on one head vector (head_dim) at position pos */
static void rope_head(float *v, const float *cos_t, const float *sin_t, int pos) {
    const float *c = cos_t + (size_t)pos * LAYA_HD;
    const float *s = sin_t + (size_t)pos * LAYA_HD;
    float lo[LAYA_HD / 2], hi[LAYA_HD / 2];
    for (int j = 0; j < LAYA_HD / 2; j++) { lo[j] = v[j]; hi[j] = v[LAYA_HD / 2 + j]; }
    for (int j = 0; j < LAYA_HD / 2; j++) {
        v[j] = v[j] * c[j] + (-hi[j]) * s[j];
        v[LAYA_HD / 2 + j] = v[LAYA_HD / 2 + j] * c[LAYA_HD / 2 + j] + lo[j] * s[LAYA_HD / 2 + j];
    }
}

/* ---------------- load ---------------- */
LayaModel *laya_load(const char *pack_dir) {
    char pbin[1024], pman[1024];
    snprintf(pbin, sizeof(pbin), "%s/laya_f16.bin", pack_dir);
    snprintf(pman, sizeof(pman), "%s/laya_manifest.txt", pack_dir);
    FILE *fb = fopen(pbin, "rb");
    if (!fb) { fprintf(stderr, "[LAYA] cannot open %s\n", pbin); return NULL; }
    fseek(fb, 0, SEEK_END);
    long sz = ftell(fb);
    fseek(fb, 0, SEEK_SET);
    FILE *fm = fopen(pman, "r");
    if (!fm) { fprintf(stderr, "[LAYA] cannot open %s\n", pman); fclose(fb); return NULL; }
    int cap = 512, n = 0;
    LayaEnt *ent = (LayaEnt *)calloc((size_t)cap, sizeof(LayaEnt));
    char line[512];
    while (fgets(line, sizeof(line), fm)) {
        char dtype[24];
        char *tok = strtok(line, " \t\r\n");
        if (!tok) continue;
        LayaEnt e;
        memset(&e, 0, sizeof(e));
        snprintf(e.name, sizeof(e.name), "%s", tok);
        tok = strtok(NULL, " \t\r\n"); e.ndim = atoi(tok);
        for (int i = 0; i < e.ndim; i++) { tok = strtok(NULL, " \t\r\n"); e.ne[i] = atoi(tok); }
        tok = strtok(NULL, " \t\r\n"); snprintf(dtype, sizeof(dtype), "%s", tok);
        tok = strtok(NULL, " \t\r\n"); e.off = (size_t)strtoull(tok, NULL, 10);
        tok = strtok(NULL, " \t\r\n"); e.nbytes = (size_t)strtoull(tok, NULL, 10);
        e.is_f32 = strstr(dtype, "32") ? 1 : 0;
        if (n >= cap) { cap <<= 1; ent = (LayaEnt *)realloc(ent, sizeof(LayaEnt) * (size_t)cap); }
        ent[n++] = e;
    }
    fclose(fm);
    /* K-LAYA-002 M5：载入策略再次收窄——1D（norm/bias）+ F32 小张量（temperature）
     * 展开/拷贝到 F32 池；**全部 2D 权重按行做 per-128 块对称量化，存入 int8 arena
     * + scale 池**，随后释放 843 MB 的 F16 blob ⇒ RSS 843 MB → ~474 MB。
     * 公理 collapse_to_classic + collapse_priority_rule（细粒度块 scale，避免单 scale
     * 不可逆失真）、conservation_law_under_collapse（以 J2 决策输出等价实测守门）。 */
    for (int i = 0; i < n; i++) {
        LayaEnt *e = &ent[i];
        size_t el = 1;
        for (int d = 0; d < e->ndim; d++) el *= (size_t)e->ne[d];
        e->n_elem = el;
        e->rowlen = (e->ndim >= 2) ? e->ne[1] : (e->ndim == 1 ? e->ne[0] : 1);
        if (e->rowlen < 1) e->rowlen = 1;
        e->nblk = (e->rowlen + LAYA_QB - 1) / LAYA_QB;
    }
    unsigned char *blob = (unsigned char *)malloc((size_t)sz);
    if (!blob) { fprintf(stderr, "[LAYA] OOM blob (%.0f MB)\n", sz / 1e6); fclose(fb); free(ent); return NULL; }
    if (fread(blob, 1, (size_t)sz, fb) != (size_t)sz) {
        fprintf(stderr, "[LAYA] blob read failed %s\n", pbin);
        free(blob); fclose(fb); free(ent); return NULL;
    }
    fclose(fb);
    /* VLLM_LAYA_F16=1：逃生舱，跳过 int8 量化、保留 F16 权重（RSS 回到 ~843 MB）。
     * 用途只有一个——与 fp32 golden 做数值/位级对拍。int8 是部署精度（默认），
     * 其固有量化噪声（per-128 块，~1e-2 相对）会让 fp32 门永远不过，
     * 因此不能拿它验证实现正确性。 */
    const int f16_mode = getenv("VLLM_LAYA_F16") != NULL ? 1 : 0;
    size_t f32n = 0, i8n = 0, blkn = 0;
    for (int i = 0; i < n; i++) {
        LayaEnt *e = &ent[i];
        if (e->is_f32 || e->ndim <= 1) f32n += e->n_elem;          /* 展开/拷贝为 F32 */
        else if (f16_mode) { /* 直接引用 blob，不占池 */ }
        else {                                                     /* 2D → int8 + scales */
            size_t rows = e->n_elem / (size_t)e->rowlen;
            i8n  += e->n_elem;
            blkn += rows * (size_t)e->nblk;
        }
    }
    float  *f32pool = f32n ? (float *)malloc(sizeof(float) * f32n) : NULL;
    int8_t *i8pool  = i8n ? (int8_t *)malloc(i8n) : NULL;
    float  *qscpool = blkn ? (float *)malloc(sizeof(float) * blkn) : NULL;
    if ((f32n && !f32pool) || (i8n && !i8pool) || (blkn && !qscpool)) {
        fprintf(stderr, "[LAYA] OOM pools\n");
        free(blob); free(f32pool); free(i8pool); free(qscpool); free(ent); return NULL;
    }
    size_t f32cur = 0, i8cur = 0, sccur = 0;
    for (int i = 0; i < n; i++) {
        LayaEnt *e = &ent[i];
        const unsigned char *src = blob + e->off;
        if (e->is_f32) {                   /* F32 小张量（temperature）：拷贝到 F32 池 */
            float *dst = f32pool + f32cur;
            memcpy(dst, src, sizeof(float) * e->n_elem);
            e->f32 = dst; e->f16 = NULL; e->i8 = NULL;
            f32cur += e->n_elem;
        } else if (e->ndim == 1) {         /* 1D F16 → F32 池（norm/bias） */
            float *dst = f32pool + f32cur;
            const uint16_t *h = (const uint16_t *)src;
            for (size_t j = 0; j < e->n_elem; j++) dst[j] = f16_to_f32(h[j]);
            e->f32 = dst; e->f16 = NULL; e->i8 = NULL;
            f32cur += e->n_elem;
        } else if (f16_mode) {             /* 对拍模式：F16 权重原地引用 blob */
            e->f16 = (const uint16_t *)src; e->i8 = NULL; e->f32 = NULL;
        } else {                           /* 2D F16 → int8 + per-128 块 scale */
            int8_t *q = i8pool + i8cur;
            float  *sc = qscpool + sccur;
            const uint16_t *h = (const uint16_t *)src;
            size_t rows = e->n_elem / (size_t)e->rowlen;
            for (size_t r = 0; r < rows; r++)
                lay_quant_f16_row(h + r * (size_t)e->rowlen, q + r * (size_t)e->rowlen,
                                  sc + r * (size_t)e->nblk, e->rowlen);
            e->i8 = q; e->qsc = sc; e->f16 = NULL; e->f32 = NULL;
            i8cur += e->n_elem;
            sccur += rows * (size_t)e->nblk;
        }
    }
    LayaModel *m = (LayaModel *)calloc(1, sizeof(LayaModel));
    if (f16_mode) {                    /* F16 权重指向 blob，须保留 */
        m->blob = blob; m->blob_size = (size_t)sz;
    } else {
        free(blob); blob = NULL;       /* F16 blob 不再需要 */
        m->blob = NULL; m->blob_size = 0;
    }
    m->ent = ent; m->n_ent = n;
    m->f32pool = f32pool; m->i8pool = i8pool; m->qscpool = qscpool;
    printf("[LAYA] loaded %d tensors (int8 %.0f MB + scales %.1f MB + F32 pool %.2f MB%s)\n",
           n, (double)i8n / 1e6, (double)(sizeof(float) * blkn) / 1e6,
           (double)(sizeof(float) * f32n) / 1e6,
           f16_mode ? ", F16 blob kept for golden check" : "");
    return m;
}

void laya_free(LayaModel *m) {
    if (!m) return;
    if (m->ent)
        for (int i = 0; i < m->n_ent; i++) free(m->ent[i].ime_a);
    free(m->blob); free(m->f32pool); free(m->i8pool); free(m->qscpool); free(m->ent);
    free(m->rope_cosF); free(m->rope_sinF); free(m->rope_cosS); free(m->rope_sinS);
    free(m);
}

/* ------------------------------------------------------------------
 * K-LAYA-002 M10：注意力 RVV 内核（lane 之间互不相关 ⇒ 无跨 lane 归约）。
 *
 * 两个内层循环都选"每个 lane 各自跑完一条完整标量链"的切分，因此与标量
 * 参考路径**逐位相同**，不需要任何容差：
 *   - scores：lane = key（一次 hi-lo 个 key，最多 vsetvl 个），d 升序逐维累加
 *     s_d = fl(0 + fl(q[0]*k[0]) + ... )，每个 lane 就是 lay_dot 的链。
 *     代价：K 需要转置成 [d][n]（lay_transpose_k），否则同一条向量要跨
 *     n*4 字节 stride 取数。
 *   - attn@V：lane = head_dim（m8 = 整个 64 维），j 升序累加
 *     oh[d] += w_j * v[j][d]，每个 lane 就是标量 oh[t] += w*v[t] 的链。
 * 禁 vfmacc：标量路径是两次舍入（乘、加各一次），vfma 只舍入一次。
 *
 * 独立性（审查 P2）：M10 是**位级无损**加速，与 M9 的数值档位无关，故不复用
 * VLLM_LAYA_RVV；用独立开关 VLLM_LAYA_ATTN_RVV=0 可回退标量（默认开）。
 *
 * 证据状态（审查 P10，2026-10-26）：上面是源码层面的同序证明；运行时只对拍过
 * n=59 q0 的单个 logit（5.992369，与标量路径同值）。perf_axiom_bitwise_gate_v1
 * 要求的 out/hidden/完整 logits/KV 哈希四级对拍、以及
 * default_enablement_gate_v1 要求的极差测量都还没做 ⇒ 默认开目前只有源码依据。
 * ------------------------------------------------------------------ */
static inline int lay_attn_rvv_on(void) {
#if ST_HAVE_RVV
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("VLLM_LAYA_ATTN_RVV");
        v = (e && e[0] == '0') ? 0 : 1;              /* 默认开；显式 0 回退标量 */
    }
    return v;
#else
    return 0;
#endif
}

#if ST_HAVE_RVV
/* K[n][D] → Kt[D][n]（8×8 分块，读写都连续；纯搬运，不改数值） */
static void lay_transpose_k(const float *K, float *Kt, int n) {
    for (int p0 = 0; p0 < n; p0 += 8) {
        int pmax = (p0 + 8 < n) ? p0 + 8 : n;
        for (int d0 = 0; d0 < LAYA_D; d0 += 8) {
            int dmax = (d0 + 8 < LAYA_D) ? d0 + 8 : LAYA_D;
            for (int p = p0; p < pmax; p++)
                for (int d = d0; d < dmax; d++)
                    Kt[(size_t)d * (size_t)n + p] = K[(size_t)p * LAYA_D + d];
        }
    }
}

/* scores：lane = key。kt0 = Kt + (h*64)*n + lo，rowstride = n。
 * LMUL 选 m4（16 key/指令）：板端微基准（N=480 全注意力，单线程）
 *   scalar 0.138 / m1 0.297 / m2 0.497 / m4 0.842 / m8 0.272 GMAC/s
 * ⇒ m4 是 m8 的 3.1×（每次取 128 B = 2 条 cache line，m8 的 256 B 反而更慢）。
 *
 * K-LAYA-003 L2（2026-10-30，已拍板，走 J2 决策等价门）：原实现是**单条 64 深
 * 浮点累加链**（acc 依赖上一条 vfadd），实测这条链就是瓶颈——scalar 路径
 * （lay_dot 逐元素同序）与 RVV 路径速度几乎相同（n=59 121 vs 127 ms），说明
 * 两者都被链的延迟而非吞吐限制。改为 4 条独立链（d 按 4 交错）后再两两合并：
 * 链深 64→16。**代价：浮点求和顺序改变，不再与旧值逐位相同**，验收口径从
 * 位级门换成 J2（argmax 一致 + max|Δp| ≤ 5e-2）。 */
/* K-LAYA-003 L2 开关：0（默认）= 原单链（与旧版逐位相同）；1 = 4 条累加链。
 * 审查 B2 处置（2026-10-30）：同批四臂 ×3 轮，L2 单独 82275 ms vs base 78674 ms
 * （更慢），L3 单独 69301 ms，L2+L3 70448 ms ⇒ 收益全部来自 L3，L2 回退为默认关。
 * 同时消除 L2+L3 组合在 q15 的 argmax 翻转。 */
static int lay_attn_split(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("VLLM_LAYA_ATTN_SPLIT");
        v = (e && e[0] == '1') ? 1 : 0;
    }
    return v;
}

static void lay_attn_scores_rvv(const float *q, const float *kt0, size_t rowstride,
                                int lo, int hi, float *att, float kscale) {
    const int split = lay_attn_split();
    int j = lo;
    while (j < hi) {
        size_t vl = __riscv_vsetvl_e32m4((size_t)(hi - j));
        vfloat32m4_t a0 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        vfloat32m4_t a1 = a0, a2 = a0, a3 = a0;
        const float *kt = kt0 + (size_t)(j - lo);
        int d = 0;
        if (!split) {
            for (; d < LAYA_HD; d++) {
                a0 = __riscv_vfadd_vv_f32m4(a0, __riscv_vfmul_vf_f32m4(
                         __riscv_vle32_v_f32m4(kt, vl), q[d], vl), vl);
                kt += rowstride;
            }
            __riscv_vse32_v_f32m4(att + (j - lo), __riscv_vfmul_vf_f32m4(a0, kscale, vl), vl);
            j += (int)vl;
            continue;
        }
        for (; d + 4 <= LAYA_HD; d += 4) {
            a0 = __riscv_vfadd_vv_f32m4(a0, __riscv_vfmul_vf_f32m4(
                     __riscv_vle32_v_f32m4(kt, vl), q[d + 0], vl), vl);
            a1 = __riscv_vfadd_vv_f32m4(a1, __riscv_vfmul_vf_f32m4(
                     __riscv_vle32_v_f32m4(kt + rowstride, vl), q[d + 1], vl), vl);
            a2 = __riscv_vfadd_vv_f32m4(a2, __riscv_vfmul_vf_f32m4(
                     __riscv_vle32_v_f32m4(kt + 2 * rowstride, vl), q[d + 2], vl), vl);
            a3 = __riscv_vfadd_vv_f32m4(a3, __riscv_vfmul_vf_f32m4(
                     __riscv_vle32_v_f32m4(kt + 3 * rowstride, vl), q[d + 3], vl), vl);
            kt += 4 * rowstride;
        }
        for (; d < LAYA_HD; d++) {   /* LAYA_HD 不是 4 的倍数时兜底（当前恒 64 ⇒ 空） */
            a0 = __riscv_vfadd_vv_f32m4(a0, __riscv_vfmul_vf_f32m4(
                     __riscv_vle32_v_f32m4(kt, vl), q[d], vl), vl);
            kt += rowstride;
        }
        vfloat32m4_t s = __riscv_vfadd_vv_f32m4(
            __riscv_vfadd_vv_f32m4(a0, a1, vl), __riscv_vfadd_vv_f32m4(a2, a3, vl), vl);
        __riscv_vse32_v_f32m4(att + (j - lo), __riscv_vfmul_vf_f32m4(s, kscale, vl), vl);
        j += (int)vl;
    }
}

/* attn@V：lane = head_dim。V 的 head 内 64 维连续，无需转置。
 * K-LAYA-003 L2：原实现对每个 t 都是 m 深（滑窗 128 / 全注意力 480）的单链，
 * 同样拆 4 条链 ⇒ 链深降到 m/4。求和顺序改变，走 J2 门。 */
static void lay_attn_av_rvv(const float *V, int lo, int m, int off,
                            const float *att, float inv, float *oh) {
    const int vmax = (int)__riscv_vsetvlmax_e32m8();
    for (int t = 0; t < LAYA_HD; t += vmax) {
        size_t vl = __riscv_vsetvl_e32m8((size_t)(LAYA_HD - t));
        vfloat32m8_t o0 = __riscv_vfmv_v_f_f32m8(0.0f, vl);
        vfloat32m8_t o1 = o0, o2 = o0, o3 = o0;
        int j = 0;
        if (!lay_attn_split()) {
            for (; j < m; j++) {
                float w = att[j] * inv;
                if (w == 0.0f) continue;           /* 与标量一致的 0 权重短路 */
                o0 = __riscv_vfadd_vv_f32m8(o0, __riscv_vfmul_vf_f32m8(
                         __riscv_vle32_v_f32m8(V + (size_t)(lo + j) * LAYA_D + off + t, vl),
                         w, vl), vl);
            }
            __riscv_vse32_v_f32m8(oh + t, o0, vl);
            continue;
        }
        for (; j + 4 <= m; j += 4) {
            const float *v0 = V + (size_t)(lo + j + 0) * LAYA_D + off + t;
            o0 = __riscv_vfadd_vv_f32m8(o0, __riscv_vfmul_vf_f32m8(
                     __riscv_vle32_v_f32m8(v0, vl), att[j + 0] * inv, vl), vl);
            o1 = __riscv_vfadd_vv_f32m8(o1, __riscv_vfmul_vf_f32m8(
                     __riscv_vle32_v_f32m8(v0 + LAYA_D, vl), att[j + 1] * inv, vl), vl);
            o2 = __riscv_vfadd_vv_f32m8(o2, __riscv_vfmul_vf_f32m8(
                     __riscv_vle32_v_f32m8(v0 + 2 * LAYA_D, vl), att[j + 2] * inv, vl), vl);
            o3 = __riscv_vfadd_vv_f32m8(o3, __riscv_vfmul_vf_f32m8(
                     __riscv_vle32_v_f32m8(v0 + 3 * LAYA_D, vl), att[j + 3] * inv, vl), vl);
        }
        for (; j < m; j++)
            o0 = __riscv_vfadd_vv_f32m8(o0, __riscv_vfmul_vf_f32m8(
                     __riscv_vle32_v_f32m8(V + (size_t)(lo + j) * LAYA_D + off + t, vl),
                     att[j] * inv, vl), vl);
        vfloat32m8_t o = __riscv_vfadd_vv_f32m8(
            __riscv_vfadd_vv_f32m8(o0, o1, vl), __riscv_vfadd_vv_f32m8(o2, o3, vl), vl);
        __riscv_vse32_v_f32m8(oh + t, o, vl);
    }
}
#endif

/* run attention for one position over selected keys; returns nothing, writes out_heads[d] */
static void attn_pos(const float *Qp, const float *K, const float *Kt, const float *V,
                     int n, int p, int is_full, float *att, float *out_heads) {
    const float kscale = 1.0f / 8.0f;   /* head_dim^-0.5 */
    /* K-LAYA-002 M7：滑动窗口区间裁剪——滑窗层只遍历有效 key 区间 [lo,hi)，
     * 不再逐 j 判断 abs(p-j)>SLIDE（原实现三个 pass 仍扫全 n，浪费 ~2×）。
     * 数学上逐位等价：被裁剪的 j 原取 att=-1e30 ⇒ exp(-1e30-maxs)=0 ⇒ 对 max/sum/V 零贡献。 */
    int lo = 0, hi = n;
    if (!is_full) {
        lo = p - LAYA_SLIDE; if (lo < 0) lo = 0;
        hi = p + LAYA_SLIDE + 1; if (hi > n) hi = n;
    }
    const int m = hi - lo;
    for (int h = 0; h < LAYA_H; h++) {
        int off = h * LAYA_HD;
        const float *q = Qp + off;
        float maxs = -1e30f;
#if ST_HAVE_RVV
        if (Kt) {   /* K-LAYA-002 M10：lane = key，逐位等价标量 lay_dot 链 */
            lay_attn_scores_rvv(q, Kt + (size_t)off * (size_t)n + lo, (size_t)n,
                                lo, hi, att, kscale);
            for (int j = 0; j < m; j++) if (att[j] > maxs) maxs = att[j];
        } else
#endif
        {
            for (int j = lo; j < hi; j++) {
                const float *k = K + (size_t)j * LAYA_D + off;
                float s = lay_dot(q, k, LAYA_HD) * kscale;   /* K-LAYA-002 M1：SIMD 点积 */
                att[j - lo] = s;
                if (s > maxs) maxs = s;
            }
        }
        float sum = 0.0f;
        for (int j = 0; j < m; j++) {
            float e = expf(att[j] - maxs);
            att[j] = e; sum += e;
        }
        float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
        float *oh = out_heads + off;
#if ST_HAVE_RVV
        if (Kt) {   /* K-LAYA-002 M10：lane = head_dim，逐位等价标量 oh[t] += w*v[t] */
            lay_attn_av_rvv(V, lo, m, off, att, inv, oh);
            continue;
        }
#endif
        memset(oh, 0, sizeof(float) * LAYA_HD);
        for (int j = 0; j < m; j++) {
            float w = att[j] * inv;
            if (w == 0.0f) continue;
            const float *v = V + (size_t)(lo + j) * LAYA_D + off;
            /* K-LAYA-002 M1：加权和向量化（oh += v * w） */
#if ST_HAVE_NEON
            int t = 0;
            for (; t + 4 <= LAYA_HD; t += 4) {
                float32x4_t o = vld1q_f32(oh + t);
                o = vfmaq_n_f32(o, vld1q_f32(v + t), w);
                vst1q_f32(oh + t, o);
            }
            for (; t < LAYA_HD; t++) oh[t] += w * v[t];
#elif ST_ARCH_X86 && defined(__AVX2__) && defined(__FMA__)
            __m256 vw = _mm256_set1_ps(w);
            int t = 0;
            for (; t + 8 <= LAYA_HD; t += 8)
                _mm256_storeu_ps(oh + t, _mm256_fmadd_ps(_mm256_loadu_ps(v + t), vw,
                                                         _mm256_loadu_ps(oh + t)));
            for (; t < LAYA_HD; t++) oh[t] += w * v[t];
#else
            for (int t = 0; t < LAYA_HD; t++) oh[t] += w * v[t];
#endif
        }
    }
}

/* K-LAYA-002 M3：分块 MatMul（Y[T,N] = X[T,K]·W[N,K]^T + B）。
 * 4 token × 2 输出行分块：权重行在 4 个 token 间复用、x 向量在 2 输出行间复用，
 * 内层 load:FMA 由 2:1 降到 0.75:1。板端微基准（真实形状 T=55）：
 * GEMV 8.0 → MatMul 12.1~13.2 GFLOP/s（1.51~1.66×），max|Δ|≈0（同容差）。
 * 按 token 块并行（各块写不同 Y 行，互不重叠 ⇒ 与串行同序、J3 位级门保持）。 */
/* K-LAYA-003 L4：GEMM 分块尺寸（token 块 TB × 输出行块 OB）可扫，默认 8×8。
 * 纯循环重排，每个 (t,o) 仍是一次完整 lay_dot_i8 ⇒ 与 8×8 逐位同值。
 * 板端实测（n=59，QB=512）：OB=8 最优（1283 ms），OB=16/32 退化到 1305/1398；
 * TB=4/8/16/32 在 ±1% 内。故默认 8×8，开关只为复测保留。 */
static int lay_tile_env(int which) {
    static int v[2] = {-1, -1};
    if (v[which] < 0) {
        const char *e = getenv(which ? "VLLM_LAYA_TB" : "VLLM_LAYA_OB");
        int n = e ? atoi(e) : 0;
        if (n < 1) n = 8;
        if (n > 512) n = 512;
        v[which] = n;
    }
    return v[which];
}

/* ==================================================================
 * K-LAYA-003 L1：int8 GEMM 迁 SpacemiT IME（vmadot）
 *
 * 动机（板端实测 2026-10-30，n=480，8 线程）：mm 21.9 s / attn 12.2 s /
 * 其余 1.2 s。mm 的 345 GFLOP 只跑出 15.8 GFLOP/s（≈1 GMAC/s·核），而
 * X60 的 RVV 归约天花板本身就在 dot_rvv 1.85 / dot_noreduce 2.28 GMAC/s
 * ⇒ 8 核上限 ≈18 GMAC/s，172 GMAC 至少 9.6 s，抠循环进不了「可用」区间。
 * IME 的 vmadot = 4(M)×4(N)×8(K) = 128 MAC/指令（单核 ALU 峰值 203 GMAC/s，
 * 见 tools/yolo/bench_ime.c），且本模型权重静态 ⇒ 打包一次即可，没有
 * YOLO 那边摊不掉的每次调用打包成本。
 *
 * 位级口径（不变）：与 lay_dot_i8 完全相同的「每 128 元素一次 int32 精确和
 * + 一次 float 缩放累加」，块序升序、K%128 尾块的四则运算顺序也逐字相同
 * ⇒ 与档 1/档 2 逐位同值（J3 位级门直接验收）。
 *
 * 硬约束：vmadot 只在 hart0-3 存在（hart4-7 执行 SIGILL）⇒ 区内绑核、
 * 区外还原（不还原会把 OMP 池永久钉在 4 核，YOLO §4 踩过）。
 * 开关：VLLM_LAYA_IME=1（默认关）。
 * ================================================================== */
#if ST_ARCH_RISCV && defined(_OPENMP)

#define LIME_STR(x) #x
#define LIME_VMADOT(vd, vs1, vs2)                                                \
    ".insn 4, " LIME_STR(0x2b | ((vd) << 7) | (3 << 12) | ((vs1) << 15) |        \
                         ((vs2) << 20) | (0x71 << 25)) "\n\t"

static int lay_ime_on(void) {
    static int g = -1;
    if (g < 0) {
        const char *e = getenv("VLLM_LAYA_IME");
        g = (e && e[0] == '1') ? 1 : 0;
    }
    return g;
}

/* 8(M=输出行) × 16(N=token) × Kp 微内核：acc[8][16]。
 * 语义：acc[mv*4+j][q*4+r] = 行(mv*4+q) · token(j*4+r) 的 int32 点积。 */
static void lay_ime_kernel(const int8_t *a, const int8_t *b, int kb_n, int32_t acc[8][16]) {
    uintptr_t pa0 = (uintptr_t)a, pa1 = (uintptr_t)(a + 32);
    uintptr_t pb0 = (uintptr_t)b, pb1 = (uintptr_t)(b + 32);
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
        LIME_VMADOT(8, 0, 2)
        LIME_VMADOT(10, 0, 3)
        LIME_VMADOT(12, 0, 4)
        LIME_VMADOT(14, 0, 5)
        LIME_VMADOT(16, 1, 2)
        LIME_VMADOT(18, 1, 3)
        LIME_VMADOT(20, 1, 4)
        LIME_VMADOT(22, 1, 5)
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

/* 权重打包（每张量一次）：[mt][kb] 64 B = 2×32 B，各 32 B = 4 行 × 8 K */
static int8_t *lay_ime_pack_w(const int8_t *Wq, int OC, int K) {
    const int mt_n = OC / 8, kb_n = K / 8;
    double t0 = lay_wtime();
    int8_t *p = (int8_t *)calloc((size_t)mt_n * (size_t)kb_n * 64, 1);
    if (!p) return NULL;
    for (int mt = 0; mt < mt_n; mt++)
        for (int kb = 0; kb < kb_n; kb++)
            for (int mv = 0; mv < 2; mv++)
                for (int q = 0; q < 4; q++)
                    memcpy(p + ((size_t)mt * kb_n + kb) * 64 + mv * 32 + q * 8,
                           Wq + (size_t)(mt * 8 + mv * 4 + q) * K + kb * 8, 8);
    g_ime_pack_ms += (lay_wtime() - t0) * 1e3;
    return p;
}

/* T 为 16 的整数倍。Y[T][N]，逐 (t,o) 与 lay_dot_i8 同值。 */
static void lay_ime_matmul(const LayaEnt *W, const int8_t *Xq, const float *Xs,
                           float *Y, int T, int K, int N) {
    const int kb_tot = K / 8;
    const int nblk = W->nblk;
    const int nfull = K / LAYA_QB;          /* 完整 128 块数（尾块另算） */
    const int8_t *Wa = W->ime_a;
    const float  *Ws = W->qsc;
    const int8_t *Wi = W->i8;
    double tg0 = lay_wtime();
    int nt = 4;
    { int v = omp_get_max_threads(); if (v < nt) nt = v; }
    /* 每线程一块激活打包缓冲：必须在并行区外分配，否则「某个线程 malloc 失败」
     * 会让 omp for 少一个参与者而死锁。 */
    int8_t *bufs[4] = {NULL, NULL, NULL, NULL};
    for (int i = 0; i < nt; i++) bufs[i] = (int8_t *)malloc((size_t)kb_tot * 128);
    int have_all = 1;
    for (int i = 0; i < nt; i++) if (!bufs[i]) have_all = 0;
    if (!have_all) {
        for (int i = 0; i < nt; i++) free(bufs[i]);
        for (size_t i = 0; i < (size_t)T * N; i++) Y[i] = 0.f;
        return;
    }
#pragma omp parallel num_threads(nt)
    {
        const int tid = omp_get_thread_num();
        cpu_set_t old;
        const int have_old = (sched_getaffinity(0, sizeof(old), &old) == 0);
        { cpu_set_t s; CPU_ZERO(&s); CPU_SET(tid & 3, &s);
          sched_setaffinity(0, sizeof(s), &s); }
        {
            int8_t *xp = bufs[tid];
#pragma omp for schedule(static)
            for (int nb = 0; nb < T / 16; nb++) {
                for (int kb = 0; kb < kb_tot; kb++) {
                    int8_t *d = xp + (size_t)kb * 128;
                    for (int t = 0; t < 16; t++)
                        memcpy(d + (t >> 2) * 32 + (t & 3) * 8,
                               Xq + (size_t)(nb * 16 + t) * K + kb * 8, 8);
                }
                for (int mt = 0; mt < N / 8; mt++) {
                    float acc[16][8];
                    for (int t = 0; t < 16; t++)
                        for (int o = 0; o < 8; o++) acc[t][o] = 0.0f;
                    for (int b = 0; b < nfull; b++) {
                        int32_t a8[8][16] __attribute__((aligned(64)));
                        lay_ime_kernel(Wa + ((size_t)mt * kb_tot + (size_t)b * (LAYA_QB / 8)) * 64,
                                       xp + (size_t)b * (LAYA_QB / 8) * 128, LAYA_QB / 8, a8);
                        /* a8 布局：行 = ((ai>>2)<<2)|(bi>>2)，token = ((ai&3)<<2)|(bi&3) */
                        for (int ai = 0; ai < 8; ai++) {
                            const int ro = (ai >> 2) << 2, to = (ai & 3) << 2;
                            for (int bi = 0; bi < 16; bi++) {
                                const int oo = ro | (bi >> 2);
                                const int tt = to | (bi & 3);
                                acc[tt][oo] += (float)a8[ai][bi] *
                                    (Ws[(size_t)(mt * 8 + oo) * nblk + b] *
                                     Xs[(size_t)(nb * 16 + tt) * nblk + b]);
                            }
                        }
                    }
                    for (int o = 0; o < 8; o++) {
                        const int orow = mt * 8 + o;
                        for (int t = 0; t < 16; t++) {
                            float s = acc[t][o];
                            for (int q = nfull * LAYA_QB; q < K; q++)   /* 尾块：与标量逐字同序 */
                                s += (float)Wi[(size_t)orow * K + q] *
                                     Ws[(size_t)orow * nblk + q / LAYA_QB] *
                                     (float)Xq[(size_t)(nb * 16 + t) * K + q] *
                                     Xs[(size_t)(nb * 16 + t) * nblk + q / LAYA_QB];
                            Y[(size_t)(nb * 16 + t) * N + orow] = s;
                        }
                    }
                }
            }
        }
        if (have_old) sched_setaffinity(0, sizeof(old), &old);
    }
    for (int i = 0; i < nt; i++) free(bufs[i]);
    g_ime_gemm_ms += (lay_wtime() - tg0) * 1e3;
}

#endif   /* ST_ARCH_RISCV && _OPENMP */

static void lay_matmul(const LayaModel *m, const LayaEnt *W, const LayaEnt *B,
                       const float *X, float *Y, int T, int K, int N) {
    if (W->i8) {   /* K-LAYA-002 M5：int8 权重 + int8 激活（逐行量化，按 token 并行） */
        int nblk = (K + LAYA_QB - 1) / LAYA_QB;
        int8_t *Xq = (int8_t *)malloc((size_t)T * (size_t)K);
        float  *Xs = (float *)malloc(sizeof(float) * (size_t)T * (size_t)nblk);
        if (!Xq || !Xs) { free(Xq); free(Xs); for (size_t i = 0; i < (size_t)T * N; i++) Y[i] = 0.f; return; }
#pragma omp parallel for schedule(static) if(T > 2)
        for (int t = 0; t < T; t++)
            lay_quant_act(X + (size_t)t * K, Xq + (size_t)t * K, Xs + (size_t)t * nblk, K);
        const int8_t *Wi = W->i8;
        const float  *Ws = W->qsc;
#if ST_ARCH_RISCV && defined(_OPENMP)
        /* K-LAYA-003 L1：IME 路径（默认关，VLLM_LAYA_IME=1 才走）。
         * 权重首次使用时打包一次并挂在 LayaEnt 上；T%16 的尾 token 仍走原 RVV 点积。 */
        if (lay_ime_on() && K % 8 == 0 && N % 8 == 0 && K >= LAYA_QB) {
            if (!W->ime_a) ((LayaEnt *)W)->ime_a = lay_ime_pack_w(Wi, N, K);
            if (W->ime_a) {
                int tb = (T / 16) * 16;
                lay_ime_matmul(W, Xq, Xs, Y, tb, K, N);
                for (int t = tb; t < T; t++)
                    for (int o = 0; o < N; o++)
                        Y[(size_t)t * N + o] =
                            lay_dot_i8(Wi + (size_t)o * K, Ws + (size_t)o * nblk,
                                       Xq + (size_t)t * K, Xs + (size_t)t * nblk, K);
                if (B)
                    for (int t = 0; t < T; t++)
                        for (int o = 0; o < N; o++)
                            Y[(size_t)t * N + o] += lay_at(m, B, (size_t)o);
                free(Xq); free(Xs);
                return;
            }
        }
#endif
#if ST_HAVE_NEON && defined(__ARM_FEATURE_DOTPROD)
        /* K-LAYA-002 M5b：4 token × 1 输出行分块——权重行在 4 个 token 间复用
         * （板端微基准 1.46–1.87× over 逐行；int32 累加精确 ⇒ 与逐行位级同值）。 */
        if (K % 32 == 0) {   /* vdotq CHUNK 需 32 对齐；SCALE 尾块（< LAYA_QB）逐元素补 */
            int Kfull = (K / LAYA_QB) * LAYA_QB;
            int ntile = T / 4;
#pragma omp parallel for schedule(static) if(ntile > 2)
            for (int tb = 0; tb < ntile; tb++) {
                int t = tb * 4;
                const int8_t *x0 = Xq + (size_t)(t + 0) * K, *x1 = Xq + (size_t)(t + 1) * K;
                const int8_t *x2 = Xq + (size_t)(t + 2) * K, *x3 = Xq + (size_t)(t + 3) * K;
                const float  *xs0 = Xs + (size_t)(t + 0) * nblk, *xs1 = Xs + (size_t)(t + 1) * nblk;
                const float  *xs2 = Xs + (size_t)(t + 2) * nblk, *xs3 = Xs + (size_t)(t + 3) * nblk;
                for (int o = 0; o < N; o++) {
                    const int8_t *w  = Wi + (size_t)o * K;
                    const float  *ws = Ws + (size_t)o * nblk;
                    float32x4_t s0 = vdupq_n_f32(0.f), s1 = s0, s2 = s0, s3 = s0;
                    for (int b = 0; b < Kfull; b += LAYA_QB) {
                        /* 注（K-LAYA-002 M8 实测 2026-10-11）：每 token 双交替 int32 累加链
                         * （8 个累加器）在 A76 上**退化 ~4.6%**（寄存器压力，同 M2b/t4r2 教训）
                         * ⇒ 已回退，保留单链。 */
                        int32x4_t a0 = vdupq_n_s32(0), a1 = a0, a2 = a0, a3 = a0;
                        for (int c = 0; c < LAYA_QB; c += 32) {   /* 块内 32 字节 CHUNK */
                            int8x16_t w0 = vld1q_s8(w + b + c), w1 = vld1q_s8(w + b + c + 16);
                            a0 = vdotq_s32(a0, w0, vld1q_s8(x0 + b + c));  a0 = vdotq_s32(a0, w1, vld1q_s8(x0 + b + c + 16));
                            a1 = vdotq_s32(a1, w0, vld1q_s8(x1 + b + c));  a1 = vdotq_s32(a1, w1, vld1q_s8(x1 + b + c + 16));
                            a2 = vdotq_s32(a2, w0, vld1q_s8(x2 + b + c));  a2 = vdotq_s32(a2, w1, vld1q_s8(x2 + b + c + 16));
                            a3 = vdotq_s32(a3, w0, vld1q_s8(x3 + b + c));  a3 = vdotq_s32(a3, w1, vld1q_s8(x3 + b + c + 16));
                        }
                        int bi = b / LAYA_QB;
                        float wsb = ws[bi];
                        s0 = vmlaq_n_f32(s0, vcvtq_f32_s32(a0), wsb * xs0[bi]);
                        s1 = vmlaq_n_f32(s1, vcvtq_f32_s32(a1), wsb * xs1[bi]);
                        s2 = vmlaq_n_f32(s2, vcvtq_f32_s32(a2), wsb * xs2[bi]);
                        s3 = vmlaq_n_f32(s3, vcvtq_f32_s32(a3), wsb * xs3[bi]);
                    }
                    float r0 = vaddvq_f32(s0), r1 = vaddvq_f32(s1);
                    float r2 = vaddvq_f32(s2), r3 = vaddvq_f32(s3);
                    for (int q = Kfull; q < K; q++) {   /* SCALE 尾块补（本模型仅 K=2624 时 64 个） */
                        int bi = q / LAYA_QB;
                        float wq = (float)w[q] * ws[bi];
                        r0 += wq * (float)x0[q] * xs0[bi];
                        r1 += wq * (float)x1[q] * xs1[bi];
                        r2 += wq * (float)x2[q] * xs2[bi];
                        r3 += wq * (float)x3[q] * xs3[bi];
                    }
                    Y[(size_t)(t + 0) * N + o] = r0;
                    Y[(size_t)(t + 1) * N + o] = r1;
                    Y[(size_t)(t + 2) * N + o] = r2;
                    Y[(size_t)(t + 3) * N + o] = r3;
                }
            }
            for (int t = (T / 4) * 4; t < T; t++)   /* token 尾（T%4 ≤ 3） */
                for (int o = 0; o < N; o++)
                    Y[(size_t)t * N + o] = lay_dot_i8(Wi + (size_t)o * K, Ws + (size_t)o * nblk,
                                                      Xq + (size_t)t * K, Xs + (size_t)t * nblk, K);
        } else
#endif
        {
            /* K-LAYA-002 M11：8×8（token × 输出行）分块。
             * 动机（板端微基准 2026-10-26，单线程 K=1024/N=3072）：
             *   dot_scalar 0.51 / dot_rvv 1.85 / dot_noreduce 2.28 GMAC/s
             *   ⇒ 归约只占 18%，瓶颈是取数：原来的 t-outer/o-inner 让每行权重在
             *   每个 token 重读一遍（n=480 时 480 × 421 MB ≈ 202 GB，实测 8 线程
             *   只有 ~10 GB/s ⇒ mm 21.3 s）。改成 8×8 后每块工作集 = 8 KB 权重 +
             *   8 KB 激活 = 16 KB，整块留在 L1，DRAM 流量降约 4×。
             * 位级口径：纯循环重排——每个 (t,o) 仍是一次完整的 lay_dot_i8
             * （同一 int32 链、同一块间 float 链、同一舍入点）⇒ 与 t-outer 逐位相同。
             * 并行按输出行块切分：各块写不同的 Y 列、互不重叠，与线程数无关。
             * K-LAYA-003 L4：OB/TB 改为可扫（VLLM_LAYA_OB / VLLM_LAYA_TB，默认 8）。
             * 纯循环重排 ⇒ 与固定 8×8 逐位同值。 */
            const int OB = lay_tile_env(0), TB = lay_tile_env(1);
#pragma omp parallel for schedule(static) if(T > 2)
            for (int ob = 0; ob < N; ob += OB) {
                int omax = (ob + OB < N) ? ob + OB : N;
                for (int tb = 0; tb < T; tb += TB) {
                    int tmax = (tb + TB < T) ? tb + TB : T;
                    for (int t = tb; t < tmax; t++)
                        for (int o = ob; o < omax; o++)
                            Y[(size_t)t * N + o] =
                                lay_dot_i8(Wi + (size_t)o * K, Ws + (size_t)o * nblk,
                                           Xq + (size_t)t * K, Xs + (size_t)t * nblk, K);
                }
            }
        }
        free(Xq); free(Xs);
        if (B)
            for (int t = 0; t < T; t++)
                for (int o = 0; o < N; o++)
                    Y[(size_t)t * N + o] += lay_at(m, B, (size_t)o);
        return;
    }
    const uint16_t *Wb = W->f16;
    if (Wb) {
        /* t0 = 已由 4-token 向量块写过的行数。NEON 目标由向量块推进；
         * 非 NEON 目标（x86 对拍）向量块被整段编译掉，必须从 0 起全量走标量，
         * 否则只写 T%4 个 token、其余行保持未初始化（曾导致 logits 全部相同）。 */
        int t0 = 0;
#if ST_HAVE_NEON
        int ntile = T / 4;
#pragma omp parallel for schedule(static) if(ntile > 2)
        for (int tb = 0; tb < ntile; tb++) {
            int t = tb * 4;
            const float *x0 = X + (size_t)(t + 0) * K, *x1 = X + (size_t)(t + 1) * K;
            const float *x2 = X + (size_t)(t + 2) * K, *x3 = X + (size_t)(t + 3) * K;
            int o = 0;
            for (; o + 2 <= N; o += 2) {
                const uint16_t *w0 = Wb + (size_t)(o + 0) * K, *w1 = Wb + (size_t)(o + 1) * K;
                float32x4_t a00 = vdupq_n_f32(0), a01 = vdupq_n_f32(0), a02 = vdupq_n_f32(0), a03 = vdupq_n_f32(0);
                float32x4_t a10 = vdupq_n_f32(0), a11 = vdupq_n_f32(0), a12 = vdupq_n_f32(0), a13 = vdupq_n_f32(0);
                int k = 0;
                for (; k + 4 <= K; k += 4) {
                    float32x4_t v0 = vcvt_f32_f16(vld1_f16((const __fp16 *)(w0 + k)));
                    float32x4_t v1 = vcvt_f32_f16(vld1_f16((const __fp16 *)(w1 + k)));
                    float32x4_t u0 = vld1q_f32(x0 + k), u1 = vld1q_f32(x1 + k);
                    float32x4_t u2 = vld1q_f32(x2 + k), u3 = vld1q_f32(x3 + k);
                    a00 = vfmaq_f32(a00, v0, u0); a01 = vfmaq_f32(a01, v0, u1);
                    a02 = vfmaq_f32(a02, v0, u2); a03 = vfmaq_f32(a03, v0, u3);
                    a10 = vfmaq_f32(a10, v1, u0); a11 = vfmaq_f32(a11, v1, u1);
                    a12 = vfmaq_f32(a12, v1, u2); a13 = vfmaq_f32(a13, v1, u3);
                }
                float s0[4] = { vaddvq_f32(a00), vaddvq_f32(a01), vaddvq_f32(a02), vaddvq_f32(a03) };
                float s1[4] = { vaddvq_f32(a10), vaddvq_f32(a11), vaddvq_f32(a12), vaddvq_f32(a13) };
                for (int r = 0; r < 4; r++) {
                    const float *xr = X + (size_t)(t + r) * K;
                    for (int q = k; q < K; q++) {   /* K 尾（本模型 K∈{1024,2624,4096}，均 4 的倍数 ⇒ 空） */
                        float xv = xr[q];
                        s0[r] += f16_to_f32(w0[q]) * xv;
                        s1[r] += f16_to_f32(w1[q]) * xv;
                    }
                    Y[(size_t)(t + r) * N + o]     = s0[r];
                    Y[(size_t)(t + r) * N + o + 1] = s1[r];
                }
            }
            for (; o < N; o++)
                for (int r = 0; r < 4; r++)
                    Y[(size_t)(t + r) * N + o] = lay_dot_f16(Wb + (size_t)o * K, X + (size_t)(t + r) * K, K);
        }
        t0 = ntile * 4;
#endif
        for (int t = t0; t < T; t++)   /* token 尾（T%4 ≤ 3）；非 NEON 目标为全量 */
            for (int o = 0; o < N; o++)
                Y[(size_t)t * N + o] = lay_dot_f16(Wb + (size_t)o * K, X + (size_t)t * K, K);
    } else {
        for (int t = 0; t < T; t++)
            for (int o = 0; o < N; o++)
                Y[(size_t)t * N + o] = lay_dot(W->f32 + (size_t)o * K, X + (size_t)t * K, K);
    }
    if (B) {
        for (int t = 0; t < T; t++)
            for (int o = 0; o < N; o++)
                Y[(size_t)t * N + o] += lay_at(m, B, (size_t)o);
    }
}

int laya_forward(LayaModel *m, const int *ids, int n,
                 const int *markers, int nm, int qtype, float *logits_out) {
    if (n <= 0 || nm <= 0) return -1;
    /* 单线程预热位移档位（lay_dot_i8 / attn_pos 会在 OpenMP 并行区里调用，
     * 静态初始化若放在并行区内会多线程竞争 + 重复打印）。 */
    (void)lay_dot_level();
    (void)lay_attn_rvv_on();
    char nmb[128];
    const LayaEnt *e_tok  = lay_find(m, "encoder.embeddings.tok_embeddings.weight");
    const LayaEnt *e_embn = lay_find(m, "encoder.embeddings.norm.weight");
    const LayaEnt *e_finn = lay_find(m, "encoder.final_norm.weight");
    const LayaEnt *e_te   = lay_find(m, "type_emb.weight");
    if (!e_tok || !e_embn || !e_finn || !e_te) return -1;

    /* K-LAYA-002 P2：RoPE 表按 n 缓存（首次或更长序列时才重算）。
     * 放最前面：此时尚未分配其它 scratch，失败可直接 return，无泄漏。 */
    if (m->rope_cap < n) {
        free(m->rope_cosF); free(m->rope_sinF); free(m->rope_cosS); free(m->rope_sinS);
        m->rope_cosF = m->rope_sinF = m->rope_cosS = m->rope_sinS = NULL;
        m->rope_cap = 0;
        float *cF = (float *)malloc(sizeof(float) * (size_t)n * LAYA_HD);
        float *sF = (float *)malloc(sizeof(float) * (size_t)n * LAYA_HD);
        float *cS = (float *)malloc(sizeof(float) * (size_t)n * LAYA_HD);
        float *sS = (float *)malloc(sizeof(float) * (size_t)n * LAYA_HD);
        if (!cF || !sF || !cS || !sS) { free(cF); free(sF); free(cS); free(sS); return -1; }
        rope_table(cF, sF, n, 160000.0f);
        rope_table(cS, sS, n, 10000.0f);
        m->rope_cosF = cF; m->rope_sinF = sF; m->rope_cosS = cS; m->rope_sinS = sS;
        m->rope_cap = n;
    }
    const float *cosF = m->rope_cosF, *sinF = m->rope_sinF;
    const float *cosS = m->rope_cosS, *sinS = m->rope_sinS;

    size_t nd = (size_t)n * LAYA_D;
    float *x    = malloc(sizeof(float) * nd);
    float *Q    = malloc(sizeof(float) * nd);
    float *K    = malloc(sizeof(float) * nd);
    float *V    = malloc(sizeof(float) * nd);
    float *nrm  = malloc(sizeof(float) * nd);
    float *nn_a = malloc(sizeof(float) * nd);                      /* [n,D] norm 输出 */
    float *qkv_a = malloc(sizeof(float) * nd * 3);                 /* [n,3D] Wqkv / in_proj 输出 */
    float *wi_a = malloc(sizeof(float) * (size_t)n * 2 * LAYA_INTER);
    float *mh_a = malloc(sizeof(float) * (size_t)n * LAYA_INTER);
    float *ff_a = malloc(sizeof(float) * (size_t)n * LAYA_HFFN);
    float *tmp_a = malloc(sizeof(float) * nd);
    float *aou_a = malloc(sizeof(float) * nd);   /* M5c: [n,D] attention 输出，供 Wo/out_proj 分块 MatMul */
    float *sc  = malloc(sizeof(float) * LAYA_D);
    float *sc2 = malloc(sizeof(float) * LAYA_D);
    /* K-LAYA-002 M10：K 的转置副本 Kt[D][n]，供 scores 的 lane=key 内核连续取数。
     * 仅在 RVV 档>=2 时分配（否则 NULL ⇒ attn_pos 走原标量路径）。 */
    float *Kt = lay_attn_rvv_on() ? (float *)malloc(sizeof(float) * LAYA_D * (size_t)n) : NULL;

    /* K-LAYA-002 M4/M5c：位置级并行的线程私有 scratch（只余 attention 的 att）；
     * attention 输出改为 token 维共享缓冲 aou_a，Wo/out_proj 走 lay_matmul 分块路径。
     * 逐 p/逐块计算与串行同序 ⇒ 结果与线程数无关（J3 位级门）。 */
    int nth = 1;
#ifdef _OPENMP
    nth = omp_get_max_threads();
    if (nth < 1) nth = 1;
    if (nth > LAYA_NT_MAX) nth = LAYA_NT_MAX;
#endif
    if (n < 4) nth = 1;
    float *att_t[LAYA_NT_MAX];
    for (int t = 0; t < LAYA_NT_MAX; t++) att_t[t] = NULL;
    for (int t = 0; t < nth; t++) {
        att_t[t] = malloc(sizeof(float) * (size_t)n);
        if (!att_t[t]) { nth = t; break; }
    }
    int ok = x && Q && K && V && nrm && nn_a && qkv_a && wi_a && mh_a && ff_a && tmp_a && aou_a &&
             sc && sc2 && nth >= 1 &&
             (!lay_attn_rvv_on() || Kt);
    if (!ok) goto done;
    if (g_timing < 0) g_timing = getenv("VLLM_LAYA_TIMING") ? 1 : 0;
    double _tw0 = tmark();
    {
    double _t = tmark();
    for (int p = 0; p < n; p++)
        for (int i = 0; i < LAYA_D; i++)
            x[(size_t)p * LAYA_D + i] = lay_at(m, e_tok, (size_t)ids[p] * LAYA_D + i);
    for (int p = 0; p < n; p++)
        lay_ln(m, e_embn, NULL, x + (size_t)p * LAYA_D, x + (size_t)p * LAYA_D, LAYA_D);
    tacc(TT_EMB, _t);
    }

    /* K-LAYA-002 L1b：权重指针一次性预解析（消除每位置/每层循环内的
     * snprintf + lay_find 线性扫描；对结果无影响，纯寻址开销消除）。 */
    const LayaEnt *L_attn_norm[LAYA_NL], *L_wqkv[LAYA_NL], *L_wo_attn[LAYA_NL];
    const LayaEnt *L_mlp_norm[LAYA_NL], *L_wi[LAYA_NL], *L_wo_mlp[LAYA_NL];
    for (int l = 0; l < LAYA_NL; l++) {
        L_attn_norm[l] = NULL;
        if (l > 0) {                       /* 层 0 attn_norm = Identity（无该张量） */
            snprintf(nmb, sizeof(nmb), "encoder.layers.%d.attn_norm.weight", l);
            L_attn_norm[l] = lay_find(m, nmb);
        }
        snprintf(nmb, sizeof(nmb), "encoder.layers.%d.attn.Wqkv.weight", l);
        L_wqkv[l] = lay_find(m, nmb);
        snprintf(nmb, sizeof(nmb), "encoder.layers.%d.attn.Wo.weight", l);
        L_wo_attn[l] = lay_find(m, nmb);
        snprintf(nmb, sizeof(nmb), "encoder.layers.%d.mlp_norm.weight", l);
        L_mlp_norm[l] = lay_find(m, nmb);
        snprintf(nmb, sizeof(nmb), "encoder.layers.%d.mlp.Wi.weight", l);
        L_wi[l] = lay_find(m, nmb);
        snprintf(nmb, sizeof(nmb), "encoder.layers.%d.mlp.Wo.weight", l);
        L_wo_mlp[l] = lay_find(m, nmb);
        if ((l > 0 && !L_attn_norm[l]) || !L_wqkv[l] || !L_wo_attn[l] ||
            !L_mlp_norm[l] || !L_wi[l] || !L_wo_mlp[l]) { ok = 0; goto done; }
    }

    for (int l = 0; l < LAYA_NL; l++) {
        int is_full = (l % 3 == 0);
        const float *cos_t = is_full ? cosF : cosS;
        const float *sin_t = is_full ? sinF : sinS;

        /* A. attn_norm（层 0 = Identity）+ Wqkv MatMul（权重行在 token 间复用） */
        { double _t = tmark();
        if (l == 0) memcpy(nn_a, x, sizeof(float) * nd);
        else
#pragma omp parallel for schedule(static) if(nth > 1)
            for (int p = 0; p < n; p++)
                lay_ln(m, L_attn_norm[l], NULL, x + (size_t)p * LAYA_D, nn_a + (size_t)p * LAYA_D, LAYA_D);
        tacc(TT_NORM, _t); }
        { double _t = tmark();
        lay_matmul(m, L_wqkv[l], NULL, nn_a, qkv_a, n, LAYA_D, 3 * LAYA_D);
        tacc(TT_MM, _t); }
        if (l == 0) {                       /* 调试：首层分步 */
            lay_dump("qkv_in", nn_a, (size_t)n * LAYA_D);
            lay_dump("qkv", qkv_a, (size_t)n * 3 * LAYA_D);
        }
        { double _t = tmark();
#pragma omp parallel for schedule(static) if(nth > 1)
        for (int p = 0; p < n; p++) {
            const float *qkv = qkv_a + (size_t)p * 3 * LAYA_D;
            for (int h = 0; h < LAYA_H; h++) {
                int off = h * LAYA_HD;
                memcpy(Q + (size_t)p * LAYA_D + off, qkv + off, sizeof(float) * LAYA_HD);
                memcpy(K + (size_t)p * LAYA_D + off, qkv + LAYA_D + off, sizeof(float) * LAYA_HD);
                memcpy(V + (size_t)p * LAYA_D + off, qkv + 2 * LAYA_D + off, sizeof(float) * LAYA_HD);
                rope_head(Q + (size_t)p * LAYA_D + off, cos_t, sin_t, p);
                rope_head(K + (size_t)p * LAYA_D + off, cos_t, sin_t, p);
            }
        }
        tacc(TT_ROPE, _t); }
        /* B. attention（位置间混合，逐位置 → aou_a）+ Wo 走分块 MatMul（M5c） */
        { double _t = tmark();
        if (Kt) lay_transpose_k(K, Kt, n);   /* M10：转置一次供 lane=key scores 用 */
#pragma omp parallel for schedule(static) if(nth > 1)
        for (int p = 0; p < n; p++)
            attn_pos(Q + (size_t)p * LAYA_D, K, Kt, V, n, p, is_full, att_t[OTID], aou_a + (size_t)p * LAYA_D);
        tacc(TT_ATTN, _t); }
        { double _t = tmark();
        lay_matmul(m, L_wo_attn[l], NULL, aou_a, tmp_a, n, LAYA_D, LAYA_D);
        tacc(TT_MM, _t); }
        if (l == 0) {
            lay_dump("qr", Q, (size_t)n * LAYA_D);
            lay_dump("kr", K, (size_t)n * LAYA_D);
            lay_dump("ao", aou_a, (size_t)n * LAYA_D);
            lay_dump("wo", tmp_a, (size_t)n * LAYA_D);
        }
        { double _t = tmark();
#pragma omp parallel for schedule(static) if(nth > 1)
        for (int p = 0; p < n; p++)
            for (int i = 0; i < LAYA_D; i++)
                x[(size_t)p * LAYA_D + i] += tmp_a[(size_t)p * LAYA_D + i];
        tacc(TT_RESID, _t); }
        /* C. mlp：norm → Wi MatMul → GELU-gate → Wo MatMul → 残差 */
        { double _t = tmark();
#pragma omp parallel for schedule(static) if(nth > 1)
        for (int p = 0; p < n; p++)
            lay_ln(m, L_mlp_norm[l], NULL, x + (size_t)p * LAYA_D, nn_a + (size_t)p * LAYA_D, LAYA_D);
        tacc(TT_NORM, _t); }
        { double _t = tmark();
        lay_matmul(m, L_wi[l], NULL, nn_a, wi_a, n, LAYA_D, 2 * LAYA_INTER);
        tacc(TT_MM, _t); }
        { double _t = tmark();
#pragma omp parallel for schedule(static) if(nth > 1)
        for (int p = 0; p < n; p++) {
            const float *wi = wi_a + (size_t)p * 2 * LAYA_INTER;
            float *mh = mh_a + (size_t)p * LAYA_INTER;
            for (int i = 0; i < LAYA_INTER; i++) mh[i] = gelu_erf(wi[i]) * wi[LAYA_INTER + i];
        }
        tacc(TT_ACT, _t); }
        { double _t = tmark();
        lay_matmul(m, L_wo_mlp[l], NULL, mh_a, tmp_a, n, LAYA_INTER, LAYA_D);
        tacc(TT_MM, _t); }
        if (l == 0) {
            lay_dump("mh", mh_a, (size_t)n * LAYA_INTER);
            lay_dump("wom", tmp_a, (size_t)n * LAYA_D);
        }
        { double _t = tmark();
#pragma omp parallel for schedule(static) if(nth > 1)
        for (int p = 0; p < n; p++)
            for (int i = 0; i < LAYA_D; i++)
                x[(size_t)p * LAYA_D + i] += tmp_a[(size_t)p * LAYA_D + i];
        tacc(TT_RESID, _t); }
        { char ln[16]; snprintf(ln, sizeof(ln), "l%02d", l);
          lay_dump(ln, x, (size_t)n * LAYA_D); }   /* 调试：逐层末态 */
    }
    { double _t = tmark();
    for (int p = 0; p < n; p++)
        lay_ln(m, e_finn, NULL, x + (size_t)p * LAYA_D, x + (size_t)p * LAYA_D, LAYA_D);

    lay_dump("enc", x, (size_t)n * LAYA_D);   /* 调试：encoder 末态（type_emb 之前） */

    /* type embedding added to every position */
    for (int p = 0; p < n; p++)
        for (int i = 0; i < LAYA_D; i++)
            x[(size_t)p * LAYA_D + i] += lay_at(m, e_te, (size_t)qtype * LAYA_D + i);
    tacc(TT_NORM, _t); }

    /* 2x transformer encoder layer (norm_first, relu) */
    for (int hl = 0; hl < 2; hl++) {
        char p1[96], p2[96], pi[96], po[96], l1[96], l2[96], nb[96];
        snprintf(p1, sizeof(p1), "head.layers.%d.norm1.weight", hl);
        snprintf(nb, sizeof(nb), "head.layers.%d.norm1.bias", hl);
        const LayaEnt *n1w = lay_find(m, p1), *n1b = lay_find(m, nb);
        snprintf(p2, sizeof(p2), "head.layers.%d.norm2.weight", hl);
        snprintf(nb, sizeof(nb), "head.layers.%d.norm2.bias", hl);
        const LayaEnt *n2w = lay_find(m, p2), *n2b = lay_find(m, nb);
        snprintf(pi, sizeof(pi), "head.layers.%d.self_attn.in_proj_weight", hl);
        snprintf(nb, sizeof(nb), "head.layers.%d.self_attn.in_proj_bias", hl);
        const LayaEnt *ipw = lay_find(m, pi), *ipb = lay_find(m, nb);
        snprintf(po, sizeof(po), "head.layers.%d.self_attn.out_proj.weight", hl);
        snprintf(nb, sizeof(nb), "head.layers.%d.self_attn.out_proj.bias", hl);
        const LayaEnt *opw = lay_find(m, po), *opb = lay_find(m, nb);
        snprintf(l1, sizeof(l1), "head.layers.%d.linear1.weight", hl);
        snprintf(nb, sizeof(nb), "head.layers.%d.linear1.bias", hl);
        const LayaEnt *f1w = lay_find(m, l1), *f1b = lay_find(m, nb);
        snprintf(l2, sizeof(l2), "head.layers.%d.linear2.weight", hl);
        snprintf(nb, sizeof(nb), "head.layers.%d.linear2.bias", hl);
        const LayaEnt *f2w = lay_find(m, l2), *f2b = lay_find(m, nb);
        if (!n1w || !n1b || !n2w || !n2b || !ipw || !ipb || !opw || !opb || !f1w || !f1b || !f2w || !f2b)
            { ok = 0; goto done; }

        { double _t = tmark();
        for (int p = 0; p < n; p++)
            lay_ln(m, n1w, n1b, x + (size_t)p * LAYA_D, nrm + (size_t)p * LAYA_D, LAYA_D);
        tacc(TT_NORM, _t); }
        { double _t = tmark();
        lay_matmul(m, ipw, ipb, nrm, qkv_a, n, LAYA_D, 3 * LAYA_D);
        tacc(TT_MM, _t); }
        { double _t = tmark();
#pragma omp parallel for schedule(static) if(nth > 1)
        for (int p = 0; p < n; p++) {
            const float *qkv = qkv_a + (size_t)p * 3 * LAYA_D;
            memcpy(Q + (size_t)p * LAYA_D, qkv, sizeof(float) * LAYA_D);
            memcpy(K + (size_t)p * LAYA_D, qkv + LAYA_D, sizeof(float) * LAYA_D);
            memcpy(V + (size_t)p * LAYA_D, qkv + 2 * LAYA_D, sizeof(float) * LAYA_D);
        }
        tacc(TT_ROPE, _t); }
        { double _t = tmark();
        if (Kt) lay_transpose_k(K, Kt, n);   /* M10：转置一次供 lane=key scores 用 */
#pragma omp parallel for schedule(static) if(nth > 1)
        for (int p = 0; p < n; p++)
            attn_pos(Q + (size_t)p * LAYA_D, K, Kt, V, n, p, 1, att_t[OTID], aou_a + (size_t)p * LAYA_D);
        tacc(TT_ATTN, _t); }
        { double _t = tmark();
        lay_matmul(m, opw, opb, aou_a, tmp_a, n, LAYA_D, LAYA_D);
        tacc(TT_MM, _t); }
        { double _t = tmark();
#pragma omp parallel for schedule(static) if(nth > 1)
        for (int p = 0; p < n; p++)
            for (int i = 0; i < LAYA_D; i++)
                x[(size_t)p * LAYA_D + i] += tmp_a[(size_t)p * LAYA_D + i];
        tacc(TT_RESID, _t); }
        { double _t = tmark();
#pragma omp parallel for schedule(static) if(nth > 1)
        for (int p = 0; p < n; p++)
            lay_ln(m, n2w, n2b, x + (size_t)p * LAYA_D, nn_a + (size_t)p * LAYA_D, LAYA_D);
        tacc(TT_NORM, _t); }
        { double _t = tmark();
        lay_matmul(m, f1w, f1b, nn_a, ff_a, n, LAYA_D, LAYA_HFFN);
        tacc(TT_MM, _t); }
        { double _t = tmark();
#pragma omp parallel for schedule(static) if(nth > 1)
        for (int p = 0; p < n; p++) {
            float *ff1 = ff_a + (size_t)p * LAYA_HFFN;
            for (int i = 0; i < LAYA_HFFN; i++) ff1[i] = relu_f(ff1[i]);
        }
        tacc(TT_ACT, _t); }
        { double _t = tmark();
        lay_matmul(m, f2w, f2b, ff_a, tmp_a, n, LAYA_HFFN, LAYA_D);
        tacc(TT_MM, _t); }
        { double _t = tmark();
#pragma omp parallel for schedule(static) if(nth > 1)
        for (int p = 0; p < n; p++)
            for (int i = 0; i < LAYA_D; i++)
                x[(size_t)p * LAYA_D + i] += tmp_a[(size_t)p * LAYA_D + i];
        tacc(TT_RESID, _t); }
    }

    lay_dump("head", x, (size_t)n * LAYA_D);  /* 调试：决策头末态（scorer 之前） */

    /* scorer */
    const LayaEnt *s0w = lay_find(m, "scorer.0.weight");
    const LayaEnt *s0b = lay_find(m, "scorer.0.bias");
    const LayaEnt *s1w = lay_find(m, "scorer.1.weight");
    const LayaEnt *s1b = lay_find(m, "scorer.1.bias");
    const LayaEnt *s3w = lay_find(m, "scorer.3.weight");
    const LayaEnt *s3b = lay_find(m, "scorer.3.bias");
    if (!s0w || !s0b || !s1w || !s1b || !s3w || !s3b) { ok = 0; goto done; }
    { double _t = tmark();
    for (int k = 0; k < nm; k++) {
        const float *mv = x + (size_t)markers[k] * LAYA_D;
        lay_ln(m, s0w, s0b, mv, sc, LAYA_D);
        lay_linear(m, s1w, s1b, sc, sc2, LAYA_D, LAYA_D);
        for (int i = 0; i < LAYA_D; i++) sc[i] = gelu_erf(sc2[i]);
        lay_linear(m, s3w, s3b, sc, &logits_out[k], 1, LAYA_D);
    }
    tacc(TT_SCO, _t); }

    if (g_timing == 1) {
        double rest = (_tw0 > 0.0) ? (lay_wtime() - _tw0) : 0.0;
        double sum = 0.0; for (int i = 0; i < TT_N; i++) sum += g_tt[i];
        fprintf(stderr, "[LAYA-TIME] n=%d total=%.1fms | emb=%.1f norm=%.1f mm=%.1f rope=%.1f attn=%.1f act=%.1f resid=%.1f sco=%.1f | sum=%.1f (%.0f%%) misc=%.1f",
                n, rest * 1e3, g_tt[TT_EMB]*1e3, g_tt[TT_NORM]*1e3, g_tt[TT_MM]*1e3, g_tt[TT_ROPE]*1e3,
                g_tt[TT_ATTN]*1e3, g_tt[TT_ACT]*1e3, g_tt[TT_RESID]*1e3, g_tt[TT_SCO]*1e3,
                sum * 1e3, rest > 0.0 ? 100.0 * sum / rest : 0.0, (rest - sum) * 1e3);
        if (g_ime_pack_ms > 0.0 || g_ime_gemm_ms > 0.0)
            fprintf(stderr, " | ime_pack=%.1f ime_gemm=%.1f", g_ime_pack_ms, g_ime_gemm_ms);
        fprintf(stderr, "\n");
        g_ime_pack_ms = g_ime_gemm_ms = 0.0;
        for (int i = 0; i < TT_N; i++) g_tt[i] = 0.0;
        g_tt_total += rest;
    }

done:
    free(x); free(Q); free(K); free(V); free(nrm);
    free(nn_a); free(qkv_a); free(wi_a); free(mh_a); free(ff_a); free(tmp_a); free(aou_a);
    free(sc); free(sc2);
    free(Kt);
    for (int t = 0; t < LAYA_NT_MAX; t++) free(att_t[t]);
    return ok ? 0 : -1;
}

/* K-LAYA-001: dump ModernBERT token ids for each line of a file (gate input). */
int laya_tok_dump(const char *dir, const char *path) {
    static QwenTokenizer tok;
    if (qwen_tokenizer_load(&tok, dir) != 0) {
        fprintf(stderr, "[LTOK] tokenizer load failed: %s\n", dir);
        return -1;
    }
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[LTOK] cannot open %s\n", path); return -1; }
    static char line[1 << 20];
    int i = 0;
    while (fgets(line, sizeof(line), f)) {
        int L = (int)strlen(line);
        while (L > 0 && (line[L - 1] == '\n' || line[L - 1] == '\r')) line[--L] = '\0';
        int *ids = (int *)malloc(sizeof(int) * 65536);
        int n = qwen_tokenizer_encode_gpt2(&tok, line, ids, 65536);
        printf("[LTOK] i=%d n=%d:", i, n);
        for (int j = 0; j < n; j++) printf(" %d", ids[j]);
        printf("\n");
        free(ids);
        i++;
    }
    fclose(f);
    return 0;
}

int laya_run_cases(const char *pack_dir, const char *cases_path) {
    LayaModel *m = laya_load(pack_dir);
    if (!m) return -1;
    FILE *f = fopen(cases_path, "r");
    if (!f) { fprintf(stderr, "[LAYA] cannot open %s\n", cases_path); laya_free(m); return -1; }
    char line[1 << 16];
    int qi = 0;
    while (fgets(line, sizeof(line), f)) {
        char *p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '#' || *p == '\n' || *p == '\0') continue;
        char *tok = strtok(p, " \t\r\n");
        if (!tok) continue;
        int qtype = atoi(tok);
        tok = strtok(NULL, " \t\r\n"); int n = atoi(tok);
        int ids[8192];
        for (int i = 0; i < n; i++) { tok = strtok(NULL, " \t\r\n"); ids[i] = atoi(tok); }
        tok = strtok(NULL, " \t\r\n"); int nmm = atoi(tok);
        int mk[256];
        for (int i = 0; i < nmm; i++) { tok = strtok(NULL, " \t\r\n"); mk[i] = atoi(tok); }
        float lg[256];
        if (laya_forward(m, ids, n, mk, nmm, qtype, lg) != 0) {
            printf("[LAYA] q=%d FAILED\n", qi);
        } else {
            /* 位级门：完整输出按 case 编号原样落盘（stdout 只有 6 位小数，不够逐字节比对） */
            { char sfx[32]; snprintf(sfx, sizeof(sfx), "logits%03d", qi);
              lay_dump(sfx, lg, (size_t)nmm); }
            printf("[LAYA] q=%d n=%d nm=%d logits=[", qi, n, nmm);
            for (int i = 0; i < nmm; i++) printf("%s%.6f", i ? ", " : "", lg[i]);
            printf("]\n");
        }
        qi++;
    }
    fclose(f);
    laya_free(m);
    return 0;
}

/* ================= K-LAYA-001: end-to-end decide ================= */
#define LAYA_TOK_CLS  50281
#define LAYA_TOK_SEP  50282
#define LAYA_TOK_MASK 50284
#define LAYA_HEAD_MAXLEN 192
#define LAYA_MAXLEN      512

static char *lay_read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *b = (char *)malloc((size_t)sz + 1);
    if (!b) { fclose(f); return NULL; }
    size_t r = fread(b, 1, (size_t)sz, f);
    b[r] = '\0';
    fclose(f);
    return b;
}

static char *lay_remove_all(const char *src, const char *pat) {
    size_t pl = strlen(pat), sl = strlen(src), o = 0;
    char *out = (char *)malloc(sl + 1);
    const char *p = src;
    while (*p) {
        if (pl && strncmp(p, pat, pl) == 0) { p += pl; continue; }
        out[o++] = *p++;
    }
    out[o] = '\0';
    return out;
}

static int lay_temp_for(VJson *byopt, int qt, int k, double *out) {
    char key[32];
    const char *nm = (qt == 0) ? "choice" : (qt == 1) ? "score" : "noul";
    const char *sz = (k <= 2) ? "2" : (k <= 5) ? "3-5" : (k <= 10) ? "6-10" : "11+";
    snprintf(key, sizeof(key), "%s:%s", nm, sz);
    VJson *v = byopt ? vjson_obj_get(byopt, key) : NULL;
    if (v && v->type == VJ_NUMBER) { *out = vjson_num(v); return 1; }
    return 0;
}

/* render one option i (0-based) into a freshly malloc'd string */
static char *lay_option(const char *tname, VJson *crit, int i, int *know_k, const char **keys) {
    char buf[1024];
    if (strcmp(tname, "choice") == 0 && crit) {
        if (crit->type == VJ_OBJECT) {
            *know_k = (int)crit->u.obj.n;
            const char *key = crit->u.obj.pairs[i].key;
            VJson *v = crit->u.obj.pairs[i].val;
            if (keys) keys[i] = key;
            if (v && v->type == VJ_STRING && vjson_str(v)[0])
                snprintf(buf, sizeof(buf), "%s: %s", key, vjson_str(v));
            else
                snprintf(buf, sizeof(buf), "%s", key);
            return strdup(buf);
        }
        if (crit->type == VJ_ARRAY) {
            *know_k = (int)vjson_array_len(crit);
            const char *key = vjson_str(vjson_array_get(crit, (size_t)i));
            if (keys) keys[i] = key;
            snprintf(buf, sizeof(buf), "%s", key ? key : "");
            return strdup(buf);
        }
    }
    if (strcmp(tname, "score") == 0 && crit && crit->type == VJ_ARRAY) {
        *know_k = (int)vjson_array_len(crit);
        const char *c = vjson_str(vjson_array_get(crit, (size_t)i));
        snprintf(buf, sizeof(buf), "level %d: %s", i, c ? c : "");
        return strdup(buf);
    }
    /* noul */
    *know_k = 2;
    const char *fd = "no, the statement does not hold";
    const char *td = "yes, the statement holds";
    if (crit && crit->type == VJ_OBJECT) {
        VJson *f = vjson_obj_get(crit, "false");
        VJson *t = vjson_obj_get(crit, "true");
        if (f && f->type == VJ_STRING) fd = vjson_str(f);
        if (t && t->type == VJ_STRING) td = vjson_str(t);
    }
    snprintf(buf, sizeof(buf), "%s: %s", i == 0 ? "false" : "true", i == 0 ? fd : td);
    return strdup(buf);
}

/* ================================================================
 * Runner: model + tokenizer + temperature config, loaded once and reused.
 * Backs both the CLI (--laya-decide) and HTTP /v1/systemone (the HTTP server
 * must not reload the ~1.7 GB arena per request).
 * ================================================================ */
struct LayaRunner {
    LayaModel     *m;
    QwenTokenizer  tok;
    VJson         *cfg;      /* owned parsed rl_agent_config.json (may be NULL) */
    char          *cfg_txt;  /* owned raw text */
    VJson         *byopt;    /* borrowed: cfg.temperature_by_options (may be NULL) */
    double         tarr[3];
};

LayaRunner *laya_runner_new(const char *pack_dir) {
    LayaRunner *r = (LayaRunner *)calloc(1, sizeof(*r));
    if (!r) return NULL;
    r->m = laya_load(pack_dir);
    if (!r->m) { free(r); return NULL; }
    if (qwen_tokenizer_load(&r->tok, pack_dir) != 0) {
        fprintf(stderr, "[LAYA] cannot load tokenizer from %s\n", pack_dir);
        laya_free(r->m); free(r); return NULL;
    }
    char cfgp[1024];
    snprintf(cfgp, sizeof(cfgp), "%s/rl_agent_config.json", pack_dir);
    r->cfg_txt = lay_read_file(cfgp);
    r->cfg = r->cfg_txt ? vjson_parse(r->cfg_txt) : NULL;
    r->tarr[0] = r->tarr[1] = r->tarr[2] = 1.0;
    VJson *t3 = r->cfg ? vjson_obj_get(r->cfg, "temperature") : NULL;
    if (t3 && t3->type == VJ_ARRAY)
        for (size_t i = 0; i < 3 && i < vjson_array_len(t3); i++)
            r->tarr[i] = vjson_num(vjson_array_get(t3, i));
    r->byopt = r->cfg ? vjson_obj_get(r->cfg, "temperature_by_options") : NULL;
    return r;
}

void laya_runner_free(LayaRunner *r) {
    if (!r) return;
    if (r->m) laya_free(r->m);
    qwen_tokenizer_free(&r->tok);
    if (r->cfg) vjson_free(r->cfg);
    free(r->cfg_txt);
    free(r);
}

/* tiny bounds-checked output writer (JSON assembled by hand, as before) */
typedef struct { char *b; size_t cap, n; int ovf; } LayaW;
static void w_raw(LayaW *w, const char *s) {
    size_t l = strlen(s);
    if (w->n + l + 1 > w->cap) { w->ovf = 1; return; }
    memcpy(w->b + w->n, s, l);
    w->n += l;
    w->b[w->n] = '\0';
}
static void w_fmt(LayaW *w, const char *fmt, ...) {
    char t[4096];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(t, sizeof(t), fmt, ap);
    va_end(ap);
    w_raw(w, t);
}

int laya_runner_decide(LayaRunner *r, const char *req_text, char *out, size_t cap) {
    if (cap) out[0] = '\0';
    VJson *root = vjson_parse(req_text);
    if (!root) { fprintf(stderr, "[LAYA] bad JSON request\n"); return -1; }
    VJson *questions = vjson_obj_get(root, "questions");
    if (!questions || questions->type != VJ_OBJECT) {
        fprintf(stderr, "[LAYA] request needs a 'questions' object\n");
        vjson_free(root);
        return -1;
    }
    VJson *state = vjson_obj_get(root, "state");

    char *state_str;
    if (state && state->type == VJ_STRING) {
        state_str = strdup(vjson_str(state));
    } else {
        size_t scap = 1 << 20;
        char *b = (char *)malloc(scap);
        if (!state || vjson_serialize(state, b, scap) == 0) b[0] = '\0';
        state_str = b;
    }
    char *state_clean = lay_remove_all(state_str, "[MASK]");

    int *ids = (int *)malloc(sizeof(int) * 8192);
    int *tmp = (int *)malloc(sizeof(int) * 8192);
    int *mk  = (int *)malloc(sizeof(int) * 128);
    float *lg = (float *)malloc(sizeof(float) * 128);

    LayaW w = { out, cap, 0, 0 };
    w_raw(&w, "{\"model\":\"laya\",\"answers\":{");
    for (size_t qi = 0; qi < questions->u.obj.n; qi++) {
        const char *qid = questions->u.obj.pairs[qi].key;
        VJson *q = questions->u.obj.pairs[qi].val;
        VJson *jt = vjson_obj_get(q, "type");
        const char *tname = (jt && jt->type == VJ_STRING) ? vjson_str(jt) : "noul";
        int qt = (strcmp(tname, "choice") == 0) ? 0 : (strcmp(tname, "score") == 0) ? 1 : 2;
        VJson *jins = vjson_obj_get(q, "instructions");
        const char *ins = (jins && jins->type == VJ_STRING) ? vjson_str(jins) : "";
        VJson *crit = vjson_obj_get(q, "criteria");

        /* options */
        char *opts[64];
        const char *keys[64];
        int have_opts = 0;
        for (int i = 0; i < 64; i++) keys[i] = "";
        int k = 0;
        for (int i = 0; i < 64; i++) {
            if (i > 0 && i >= k) break;      /* do not probe past the criteria array */
            int kk = 0;
            char *o = lay_option(tname, crit, i, &kk, keys);
            if (i == 0) { k = kk; if (k <= 0 || k > 64) { free(o); break; } }
            opts[i] = o;
            have_opts = i + 1;
        }
        if (k <= 0 || k > 64) {
            w_fmt(&w, "\"%s\":{\"error\":\"bad criteria\"}", qid);
            continue;
        }

        /* option token ids: [MASK] + tok(" " + opt)[:48] */
        int *optid[64];
        int optn[64], tot = 0, have_optid = 0;
        for (int i = 0; i < k; i++) {
            char *o = lay_remove_all(opts[i], "[MASK]");
            char *sp = (char *)malloc(strlen(o) + 2);
            sp[0] = ' ';
            strcpy(sp + 1, o);
            optid[i] = (int *)malloc(sizeof(int) * 64);
            have_optid = i + 1;
            int n = qwen_tokenizer_encode_gpt2(&r->tok, sp, optid[i] + 0, 48);
            /* prepend MASK: shift */
            for (int j = n; j > 0; j--) optid[i][j] = optid[i][j - 1];
            optid[i][0] = LAYA_TOK_MASK;
            optn[i] = n + 1;
            tot += optn[i];
            free(sp); free(o);
        }
        int budget = LAYA_HEAD_MAXLEN - tot;
        if (budget < 16) {
            int per = (LAYA_HEAD_MAXLEN - 16) / k;
            if (per < 4) per = 4;
            tot = 0;
            for (int i = 0; i < k; i++) { if (optn[i] > per) optn[i] = per; tot += optn[i]; }
            budget = LAYA_HEAD_MAXLEN - tot;
        }
        if (budget < 8) budget = 8;

        char hbuf[4096];
        char *ins_clean = lay_remove_all(ins, "[MASK]");
        snprintf(hbuf, sizeof(hbuf), "%s question: %s", tname, ins_clean);
        int hn = qwen_tokenizer_encode_gpt2(&r->tok, hbuf, tmp, 2048);
        if (hn > budget) hn = budget;

        int n = 0, nmk = 0;
        ids[n++] = LAYA_TOK_CLS;
        for (int i = 0; i < hn && n < 8000; i++) ids[n++] = tmp[i];
        ids[n++] = LAYA_TOK_SEP;
        for (int i = 0; i < k; i++) {
            mk[nmk++] = n;
            for (int j = 0; j < optn[i] && n < 8000; j++) ids[n++] = optid[i][j];
        }
        ids[n++] = LAYA_TOK_SEP;
        int room = LAYA_MAXLEN - n - 1;
        if (room < 0) room = 0;
        int sn = qwen_tokenizer_encode_gpt2(&r->tok, state_clean, tmp, 8192);
        int use = sn < room ? sn : room;
        for (int i = 0; i < use && n < 8191; i++) ids[n++] = tmp[i];
        ids[n++] = LAYA_TOK_SEP;
        if (n > LAYA_MAXLEN) n = LAYA_MAXLEN;
        int nmark = 0;
        for (int i = 0; i < nmk; i++) if (mk[i] < n) nmark++;

        int fwd_ok = (laya_forward(r->m, ids, n, mk, nmark, qt, lg) == 0);

        free(ins_clean);
        for (int i = 0; i < have_optid; i++) free(optid[i]);
        for (int i = 0; i < have_opts; i++) free(opts[i]);

        if (!fwd_ok) {
            w_fmt(&w, "\"%s\":{\"error\":\"forward failed\"}", qid);
            continue;
        }
        double T = r->tarr[qt], tv;
        if (lay_temp_for(r->byopt, qt, k, &tv)) T = tv;
        double mx = lg[0];
        for (int i = 1; i < k; i++) if (lg[i] > mx) mx = lg[i];
        double sum = 0.0, p[64];
        for (int i = 0; i < k; i++) { p[i] = exp(((double)lg[i] - mx) / T); sum += p[i]; }
        for (int i = 0; i < k; i++) p[i] /= sum;

        if (qi) w_raw(&w, ",");
        if (qt == 0) {
            int best = 0;
            for (int i = 1; i < k; i++) if (p[i] > p[best]) best = i;
            double ent = 0.0;
            for (int i = 0; i < k; i++) if (p[i] > 0) ent -= p[i] * log(p[i]);
            double conf = (k > 1) ? 1.0 - ent / log((double)k) : 1.0;
            w_fmt(&w, "\"%s\":{\"type\":\"choice\",\"choice\":\"%s\",\"probabilities\":{",
                  qid, keys[best] ? keys[best] : "");
            for (int i = 0; i < k; i++)
                w_fmt(&w, "%s\"%s\":%.4f", i ? "," : "", keys[i] ? keys[i] : "", p[i]);
            w_fmt(&w, "},\"confidence\":%.4f}", conf);
        } else if (qt == 1) {
            double ev = 0.0;
            for (int i = 0; i < k; i++) ev += i * p[i];
            double ent = 0.0;
            for (int i = 0; i < k; i++) if (p[i] > 0) ent -= p[i] * log(p[i]);
            double conf = (k > 1) ? 1.0 - ent / log((double)k) : 1.0;
            w_fmt(&w, "\"%s\":{\"type\":\"score\",\"score\":%.4f,\"probabilities\":{", qid, ev);
            for (int i = 0; i < k; i++) w_fmt(&w, "%s\"%d\":%.4f", i ? "," : "", i, p[i]);
            w_fmt(&w, "},\"confidence\":%.4f}", conf);
        } else {
            w_fmt(&w, "\"%s\":{\"type\":\"noul\",\"noul\":%.4f}", qid, p[1]);
        }
    }
    w_raw(&w, "}}");

    free(state_str); free(state_clean);
    free(ids); free(tmp); free(mk); free(lg);
    vjson_free(root);
    if (w.ovf) { fprintf(stderr, "[LAYA] output buffer too small\n"); return -1; }
    return 0;
}

int laya_decide(const char *pack_dir, const char *req_path) {
    LayaRunner *r = laya_runner_new(pack_dir);
    if (!r) return -1;
    char *reqtxt = lay_read_file(req_path);
    if (!reqtxt) { fprintf(stderr, "[LAYA] cannot read %s\n", req_path); laya_runner_free(r); return -1; }
    size_t cap = 1 << 20;
    char *out = (char *)malloc(cap);
    int rc = laya_runner_decide(r, reqtxt, out, cap);
    if (rc == 0) { fputs(out, stdout); fputc('\n', stdout); fflush(stdout); }
    else fprintf(stderr, "[LAYA] decide failed\n");
    free(out); free(reqtxt); laya_runner_free(r);
    return rc;
}
