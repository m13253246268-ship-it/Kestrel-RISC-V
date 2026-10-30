/**
 * vllm_vision.c - Vision Encoder Implementation for Qwen3-VL
 *
 * Implements the full ViT-based vision encoder including:
 *   - Patch embedding (3D for video, 2D for images via 3D with T=1)
 *   - 27-layer ViT with GELU activation and DeepStack features
 *   - Bidirectional attention (no causal mask)
 *   - Spatial 2x2 merge
 *   - Merger + DeepStack mergers → LLM embedding space
 */

#include "vllm_vision.h"
#include "vllm_safetensors.h"
#include "vllm_platform.h"
#include "vllm_tp.h"   /* self-contained thread pool (replaces OpenMP) */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

/* 单平台（RK3588/aarch64）：ViT 注意力内核走标量实现（位级语义与
 * 原 AVX2 逐 lane 一致），不再依赖 immintrin / vllm_avx_emul.h。 */
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

/* Q8_0 compact block bytes (34B per 32 floats: f16 scale + 32 int8).
 * Mirrors Q8_BYTES in vllm_safetensors.c; fc2 cols pad 4304 → 4320. */
#define VQ8_BYTES(n)  ((size_t)((n) + 31) / 32 * 34)
#define VF_PAD(vf)    (((vf) + 31) / 32 * 32)

/* ================================================================
 * Internal helpers
 * ================================================================ */

static inline float gelu_approx(float x) {
    /* tanh approximation of GELU: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))) */
    float x3 = x * x * x;
    float inner = 0.7978845608f * (x + 0.044715f * x3); /* sqrt(2/pi) = 0.7978845608 */
    return 0.5f * x * (1.0f + tanhf(inner));
}

static inline void rms_norm_f32(float *__restrict out, const float *__restrict x,
                                 const float *__restrict weight,
                                 int n, float eps) {
    double ss = 0.0;
    for (int i = 0; i < n; i++) {
        float v = x[i];
        ss += (double)v * v;
    }
    float rms = 1.0f / sqrtf((float)(ss / n) + eps);
    for (int i = 0; i < n; i++) {
        out[i] = x[i] * rms * weight[i];
    }
}

/* Layer norm with bias */
static inline void layer_norm_f32(float *__restrict out, const float *__restrict x,
                                   const float *__restrict weight,
                                   const float *__restrict bias,
                                   int n, float eps) {
    double sum = 0.0, ss = 0.0;
    for (int i = 0; i < n; i++) {
        sum += (double)x[i];
        ss  += (double)x[i] * x[i];
    }
    float mean = (float)(sum / n);
    float var  = (float)(ss / n) - mean * mean;
    float inv_std = 1.0f / sqrtf(var + eps);
    for (int i = 0; i < n; i++) {
        float xn = (x[i] - mean) * inv_std;
        out[i] = xn * weight[i] + bias[i];
    }
}

/* Matrix-vector multiply: out = W * x  (no bias)
 * W layout: [out_rows, in_cols] row-major */
typedef struct {
    float *out; const float *W; const float *x;
    int out_rows, in_cols; const float *bias;   /* bias != NULL => +bias */
} vvis_matvec_ctx;

static void vvis_matvec_worker(void *ctx_, int r) {
    vvis_matvec_ctx *c = (vvis_matvec_ctx *)ctx_;
    double sum = 0.0;
    const float *wr = c->W + (size_t)r * c->in_cols;
    for (int cc = 0; cc < c->in_cols; cc++)
        sum += (double)wr[cc] * c->x[cc];
    c->out[r] = (float)sum + (c->bias ? c->bias[r] : 0.0f);
}

static void matvec_f32(float *__restrict out, const float *__restrict W,
                        const float *__restrict x,
                        int out_rows, int in_cols) {
    vvis_matvec_ctx c = { out, W, x, out_rows, in_cols, NULL };
    vllm_tp_parfor(0, out_rows, vvis_matvec_worker, &c);
}

/* Matrix-vector with bias: out = W * x + bias */
static void matvec_f32_bias(float *__restrict out, const float *__restrict W,
                              const float *__restrict x, const float *__restrict bias,
                              int out_rows, int in_cols) {
    vvis_matvec_ctx c = { out, W, x, out_rows, in_cols, bias };
    vllm_tp_parfor(0, out_rows, vvis_matvec_worker, &c);
}

/* Softmax on a vector (NEON max + vector expf + vector normalize; the sum is
 * still accumulated in double like the scalar path, so the normalization
 * scale differs from it only by fp rounding). */
static void softmax_f32(float *x, int n) {
#if defined(__aarch64__)
    float max_val = x[0];
    if (n >= 4) {
        float32x4_t vmax = vld1q_f32(x);
        int i = 4;
        for (; i + 4 <= n; i += 4)
            vmax = vmaxq_f32(vmax, vld1q_f32(x + i));
        max_val = vmaxvq_f32(vmax);
        for (; i < n; i++) if (x[i] > max_val) max_val = x[i];
    } else {
        for (int i = 1; i < n; i++) if (x[i] > max_val) max_val = x[i];
    }
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += (double)x[i];
    }
    float inv_sum = (float)(1.0 / sum);
    int i = 0;
    if (n >= 4) {
        float32x4_t vinv = vdupq_n_f32(inv_sum);
        for (; i + 4 <= n; i += 4)
            vst1q_f32(x + i, vmulq_f32(vld1q_f32(x + i), vinv));
    }
    for (; i < n; i++) x[i] *= inv_sum;
#else
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += (double)x[i];
    }
    float inv_sum = (float)(1.0 / sum);
    for (int i = 0; i < n; i++) {
        x[i] *= inv_sum;
    }
#endif
}

/* Horizontal sum of an 8-wide vector (scalar, mirrors hsum_avx2 tree). */
static inline float hsum8_ps(const float v[8]) {
    float lo[4], hi[4];
    for (int i = 0; i < 4; i++) { lo[i] = v[i]; hi[i] = v[4 + i]; }
    for (int i = 0; i < 4; i++) lo[i] += hi[i];
    float s01 = lo[0] + lo[1], s23 = lo[2] + lo[3];
    return s01 + s23;
}

/* ================================================================
 * Runtime State
 * ================================================================ */

int st_vision_init(STVisionState *vis, const STModelConfig *cfg) {
    memset(vis, 0, sizeof(*vis));
    vis->cfg = cfg;

    int vh = cfg->vis_hidden;
    int vf = cfg->vis_ffn;
    int vo = cfg->vis_out_dim;
    int max_patches = cfg->vis_max_pos;

    vis->hidden_states = calloc((size_t)max_patches * vh, sizeof(float));
    vis->attn_buf      = calloc((size_t)max_patches * vh, sizeof(float));
    vis->mlp_buf       = calloc((size_t)max_patches * vh, sizeof(float));
    vis->qkv_buf       = calloc((size_t)max_patches * 3 * vh, sizeof(float));
    vis->fc1_buf       = calloc((size_t)max_patches * VF_PAD(vf), sizeof(float));

    /* Patch features buffer */
    vis->patch_features = calloc((size_t)max_patches * vh, sizeof(float));

    /* Vision RoPE cos/sin tables ([max_patches, head_dim]) */
    vis->rot_cos = calloc((size_t)max_patches * (vh / cfg->vis_heads), sizeof(float));
    vis->rot_sin = calloc((size_t)max_patches * (vh / cfg->vis_heads), sizeof(float));

    /* DeepStack features: post-merger output [max_grid_tokens, vis_out_dim].
     * Each deepstack merger (full-norm) runs over the spatially-merged grid,
     * so the row dimension is the merged token count (max_patches/4). */
    if (cfg->vis_ds_count > 0) {
        vis->ds_features = calloc(cfg->vis_ds_count, sizeof(float*));
        int max_ds_tokens = max_patches / (cfg->vis_merge * cfg->vis_merge);
        for (int i = 0; i < cfg->vis_ds_count; i++) {
            vis->ds_features[i] = calloc((size_t)max_ds_tokens * cfg->vis_out_dim, sizeof(float));
        }
    }

    /* Visual tokens output: main grid only (deepstack features live in
     * ds_features, injected into the LLM — not part of the token sequence). */
    int max_grid_tokens = max_patches / (cfg->vis_merge * cfg->vis_merge);
    vis->visual_tokens = calloc((size_t)max_grid_tokens * vo, sizeof(float));
    vis->n_vis_tokens = 0;

    vis->pixel_values = NULL;
    vis->is_allocated = 1;
    return 0;
}

void st_vision_free(STVisionState *vis) {
    free(vis->hidden_states);
    free(vis->attn_buf);
    free(vis->mlp_buf);
    free(vis->qkv_buf);
    free(vis->fc1_buf);
    free(vis->patch_features);
    free(vis->rot_cos);
    free(vis->rot_sin);
    free(vis->visual_tokens);
    free(vis->pixel_values);
    if (vis->ds_features) {
        for (int i = 0; i < vis->cfg->vis_ds_count; i++) {
            free(vis->ds_features[i]);
        }
        free(vis->ds_features);
    }
    memset(vis, 0, sizeof(*vis));
}

/* ================================================================
 * RISC-V：视觉 Q8 权重 8x8 转换（VLLM_VIS_RVV=1）
 *
 * 视觉权重在 VQF 里是 mode-0（34B/块，行步长 (cols/32)*34），RISC-V 上只能
 * 走 st_q8_matvec_batch 的通用标量 worker。转成 8x8（272B/组）后，批量
 * matvec 会分派到 LLM prefill 的 RVV 内核（一次取数喂 4 个 patch）。
 * 板端微基准（4096x1024, 256 patches）：0.277 → 1.394 GMAC/s（5.03x，
 * 逐位一致）。代价是多一份 ~320 MB 常驻，故默认关。
 * ================================================================ */
static uint8_t *g_vis_repack_hold[4];

int st_vision_rvv_prepare(STVisionState *vis) {
    const char *e = getenv("VLLM_VIS_RVV");
    if (!e || e[0] != '1') return 0;                  /* 默认关 */
    if (g_vis_repack_hold[0]) return 1;               /* 已转换过 */
    if (!vis || !vis->cfg || !vis->w.is_allocated) return -1;

    const STModelConfig *cfg = vis->cfg;
    const int vd = cfg->vis_depth, vh = cfg->vis_hidden;
    const int vf = cfg->vis_ffn, vf_pad = VF_PAD(vf);
    /* qkv 走 st_q8_matvec_batch_qkv（head-contiguous 输出），其余三个走
     * st_q8_matvec_batch；两者在 g_vis_q8_mode2=1 时都分派到 mode-2 内核。 */
    const int rows4[4] = { 3 * vh, vh, vf, vh };
    const int cols4[4] = { vh, vh, vh, vf_pad };
    uint8_t *base4[4] = { vis->w.q8_attn_qkv_weight, vis->w.q8_attn_proj_weight,
                          vis->w.q8_mlp_fc1_weight,  vis->w.q8_mlp_fc2_weight };
    double mb = 0.0;
    for (int k = 0; k < 4; k++) {
        if (!base4[k]) goto fail;
        const size_t layer_bytes = VQ8_BYTES((size_t)rows4[k] * (size_t)cols4[k]);
        uint8_t *buf = (uint8_t *)malloc(layer_bytes * (size_t)vd);
        if (!buf) goto fail;
        for (int l = 0; l < vd; l++) {
            uint8_t *tmp = st_q8_repack_0to2(
                base4[k] + (size_t)l * layer_bytes, rows4[k], cols4[k]);
            if (!tmp) { free(buf); goto fail; }
            memcpy(buf + (size_t)l * layer_bytes, tmp, layer_bytes);
            free(tmp);
        }
        g_vis_repack_hold[k] = buf;
        mb += (double)(layer_bytes * (size_t)vd) / 1048576.0;
    }
    vis->w.q8_attn_qkv_weight  = g_vis_repack_hold[0];
    vis->w.q8_attn_proj_weight = g_vis_repack_hold[1];
    vis->w.q8_mlp_fc1_weight   = g_vis_repack_hold[2];
    vis->w.q8_mlp_fc2_weight   = g_vis_repack_hold[3];
    st_q8_vis_set_mode2(1);
    fprintf(stderr, "[VIS-RVV] Q8 权重已转 8x8（qkv/proj/fc1/fc2）：%d 层，额外常驻 %.0f MB\n",
            vd, mb);
    fflush(stderr);
    return 1;

fail:
    for (int k = 0; k < 4; k++) {
        free(g_vis_repack_hold[k]);
        g_vis_repack_hold[k] = NULL;
    }
    fprintf(stderr, "[VIS-RVV] 8x8 转换失败（尺寸/内存），保持 mode-0 标量路径\n");
    fflush(stderr);
    return -1;
}

/* ================================================================
 * ViT Block Forward Pass
 * ================================================================ */

/**
 * Single ViT block (fused): attention + MLP with residual connections.
 * x: [n_patches, hidden_dim] input/output
 * layer_idx: which ViT layer to use
 *
 * Fusion: the whole block runs in just TWO OMP parallel domains per layer
 * (vs ~10 before), so the per-domain fork/join and re-sweeps vanish:
 *   domain 1: LN1 → QKV (single-patch Q8 matvec) → RoPE
 *   domain 2: AVX2 attention → proj → residual → LN2 → fc1 → GELU → fc2 → residual
 * Each thread owns a patch and runs its whole pipeline serially, keeping the
 * patch's activations in L1/L2 while the Q8 weight rows stream from L3.
 */
typedef struct {
    STVisionState *vis; const STModelConfig *cfg;
    float *x, *attn_buf, *mlp_buf, *qkv_buf, *fc1_buf;
    float *score_scratch;   /* [nthreads * n_patches] attention score staging */
    const float *norm1_w, *norm1_b, *norm2_w, *norm2_b;
    const float *fc1_b, *fc2_b;
    const uint8_t *q8_qkv, *q8_proj, *q8_fc1, *q8_fc2;
    int vh, nh, hd, vf, vf_pad, layer_idx, n_patches;
    float scale;
} vvis_vit_ctx;

/* Domain 1a: LN1 -> attn_buf (patch-parallel). */
static void vvis_vit_ln1(void *ctx_, int p) {
    vvis_vit_ctx *c = (vvis_vit_ctx *)ctx_;
    layer_norm_f32(c->attn_buf + (size_t)p * c->vh,
                   c->x + (size_t)p * c->vh,
                   c->norm1_w, c->norm1_b, c->vh, c->cfg->norm_eps);
}

/* Domain 1b: RoPE on qkv (patch-parallel). qkv_buf layout is the
 * head-contiguous [3][nh][n_patches][hd] from st_q8_matvec_batch_qkv:
 * q = (0*nh+h), k = (1*nh+h), v = (2*nh+h) row slabs. */
static void vvis_vit_rope(void *ctx_, int p) {
    vvis_vit_ctx *c = (vvis_vit_ctx *)ctx_;
    const int half = c->hd / 2;
    const float *rc = c->vis->rot_cos + (size_t)p * c->hd;
    const float *rs = c->vis->rot_sin + (size_t)p * c->hd;
    const size_t slab = (size_t)c->n_patches * c->hd;
    for (int h = 0; h < c->nh; h++) {
        float *q = c->qkv_buf + ((size_t)h) * slab + (size_t)p * c->hd;
        float *k = c->qkv_buf + ((size_t)(c->nh + h)) * slab + (size_t)p * c->hd;
        for (int d = 0; d < half; d++) {
            float cc = rc[d], ss = rs[d];
            float q0 = q[d], q1 = q[d + half];
            q[d] = q0 * cc - q1 * ss;
            q[d + half] = q1 * cc + q0 * ss;
            float k0 = k[d], k1 = k[d + half];
            k[d] = k0 * cc - k1 * ss;
            k[d + half] = k1 * cc + k0 * ss;
        }
    }
}

/* Domain 2a: attention over a PATCH BLOCK (block-parallel). Each worker
 * processes VVIS_ATTN_PB query patches; a K/V row (4 cache lines) is then
 * reused by all PB queries, instead of the per-patch kernel where every
 * query re-scans the whole key set alone (qkv's head-major strided layout
 * made each scan touch the full 8.4 MB K/V at cache-line granularity ->
 * ~8.6 GB DRAM/layer -> ~2 s/layer on RK3588). Blocking cuts that to
 * 8.6 GB/PB per layer. scores[] = per-thread [PB * n_patches] scratch. */
#define VVIS_ATTN_PB 32

/* acc/accv 是 [v4n] 个 float32x4 的向量累加器；v4n = vis_head_dim/4。
 * 历史代码按 2B 的 vis head_dim=64 写死 16 个向量；Qwen3-VL 8B/32B 的
 * vis head_dim=72 → v4n=18 > 16 会越界写栈（实测 stack smashing，
 * 2026-09-07）。容量放宽到 32 个向量（head_dim<=128 全覆盖）。 */
#define VVIS_ATTN_ACC 32

/* ================================================================
 * RISC-V RVV 注意力内核（VLLM_VIS_RVV=1 时接管）
 *
 * 位级约束（perf_axiom_bitwise_gate_v1）：两个内层循环都选「lane 之间
 * 互不相关」的切分，因此不需要在「树归约」和「顺序归约」之间取舍：
 *   - scores  : lane = query，每 lane 沿 head_dim i 升序 mul → add
 *               （vfmul + vfadd，禁 vfmacc / -ffp-contract=off），与标量
 *               `dot += q[i]*k[i]` 的累加顺序逐位相同；
 *   - attn@V  : lane = head_dim，每 lane 沿 key s 升序 mul → add，与标量
 *               `a += sq[s]*v[s*hd+i]` 逐位相同。
 * VLLM_VIS_ATTN_CHK=1 时对 h==0 用标量重算并逐位比对（见下）。
 *
 * 板上位级证据（K1-X, VLEN=256，内核级对拍，非端到端）：
 *   hd=64（2B 1024/16）与 hd=72（8B 1152/16）在 n_patches=33/100/256/1024、
 *   pb_n 尾块（1/7/31）与 nv 尾向量（hd=80/96/128）下，scores、attn@V 与
 *   整 block 的 attn_buf 全部 0 失配。标量参考分别用默认 -O2 与
 *   -fno-tree-vectorize 各编译一次（前者含编译器自动向量化的
 *   vfredosum.vs，后者 0 条），两版都 PASS —— 排除「参考侧也被向量化成
 *   同一形状」的自证。8B 权重不在板上，故 hd=72 只验内核、未跑端到端。
 * ================================================================ */
#if ST_HAVE_RVV
#include <riscv_vector.h>

static int vvis_rvv_on(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("VLLM_VIS_RVV");
        const char *a = getenv("VLLM_VIS_ATTN");
        /* VLLM_VIS_ATTN=0：只保留 Q8 权重 8x8 路径，注意力回标量，
         * 用于单独量出注意力内核的收益。 */
        v = (e && e[0] == '1' && !(a && a[0] == '0')) ? 1 : 0;
    }
    return v;
}

static int vvis_attn_chk_on(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("VLLM_VIS_ATTN_CHK");
        v = (e && e[0] == '1') ? 1 : 0;
    }
    return v;
}

/* scores[q*n_patches + s]（q < pb_n，s < n_patches）。
 * lane = query：先把 q 块转置成 [hd][VVIS_ATTN_PB]（每个 i 上 32 个 query
 * 连续），内层用一次 k 广播喂 4 组 query（PB=32 → 4 x 8 lane）。 */
static void vvis_attn_scores_rvv(const float *q_blk, const float *k_h,
                                 float *scores, int pb_n, int n_patches,
                                 int hd, float scale) {
    float qt[(size_t)hd * VVIS_ATTN_PB];
    for (int i = 0; i < hd; i++) {
        float *dst = qt + (size_t)i * VVIS_ATTN_PB;
        for (int q = 0; q < VVIS_ATTN_PB; q++)
            dst[q] = (q < pb_n) ? q_blk[(size_t)q * hd + i] : 0.0f;
    }
    const size_t vl = __riscv_vsetvl_e32m1(8);
    const ptrdiff_t st = (ptrdiff_t)n_patches * 4;
    for (int s = 0; s < n_patches; s++) {
        const float *k_s = k_h + (size_t)s * hd;
        vfloat32m1_t a0 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        vfloat32m1_t a1 = a0, a2 = a0, a3 = a0;
        for (int i = 0; i < hd; i++) {
            vfloat32m1_t kv = __riscv_vfmv_v_f_f32m1(k_s[i], vl);
            const float *qr = qt + (size_t)i * VVIS_ATTN_PB;
            a0 = __riscv_vfadd_vv_f32m1(__riscv_vfmul_vv_f32m1(
                     __riscv_vle32_v_f32m1(qr, vl), kv, vl), a0, vl);
            a1 = __riscv_vfadd_vv_f32m1(__riscv_vfmul_vv_f32m1(
                     __riscv_vle32_v_f32m1(qr + 8, vl), kv, vl), a1, vl);
            a2 = __riscv_vfadd_vv_f32m1(__riscv_vfmul_vv_f32m1(
                     __riscv_vle32_v_f32m1(qr + 16, vl), kv, vl), a2, vl);
            a3 = __riscv_vfadd_vv_f32m1(__riscv_vfmul_vv_f32m1(
                     __riscv_vle32_v_f32m1(qr + 24, vl), kv, vl), a3, vl);
        }
        /* 每个 query 自己的 s 连续 → 跨 lane 是跨行，用带步长的散写。
         * （GCC 不允许 vfloat32m1_t 数组，故逐组展开。） */
        size_t vq;
        vq = (size_t)(pb_n < 8 ? pb_n : 8);
        __riscv_vsse32_v_f32m1(scores + s, st,
                               __riscv_vfmul_vf_f32m1(a0, scale, vl), vq);
        if (pb_n > 8) {
            vq = (size_t)(pb_n - 8 < 8 ? pb_n - 8 : 8);
            __riscv_vsse32_v_f32m1(scores + (size_t)8 * n_patches + s, st,
                                   __riscv_vfmul_vf_f32m1(a1, scale, vl), vq);
        }
        if (pb_n > 16) {
            vq = (size_t)(pb_n - 16 < 8 ? pb_n - 16 : 8);
            __riscv_vsse32_v_f32m1(scores + (size_t)16 * n_patches + s, st,
                                   __riscv_vfmul_vf_f32m1(a2, scale, vl), vq);
        }
        if (pb_n > 24) {
            vq = (size_t)(pb_n - 24 < 8 ? pb_n - 24 : 8);
            __riscv_vsse32_v_f32m1(scores + (size_t)24 * n_patches + s, st,
                                   __riscv_vfmul_vf_f32m1(a3, scale, vl), vq);
        }
    }
}

/* out[q][i] = sum_s scores[q][s] * v_h[s][i]。
 * lane = head_dim（8 维一组），一次取 v 同时喂 a/b 两行 query：
 * V 的重复读次数从「每 query 一遍」降到「每两 query 一遍」。 */
static void vvis_attn_av_rvv(const float *sq_a, const float *sq_b,
                             const float *v_h, float *o_a, float *o_b,
                             int n_patches, int hd) {
    const size_t vl = __riscv_vsetvl_e32m1(8);
    const int nv = hd / 8;
    int i0 = 0;
    for (; i0 + 3 <= nv; i0 += 3) {
        vfloat32m1_t a0 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        vfloat32m1_t a1 = a0, a2 = a0, b0 = a0, b1 = a0, b2 = a0;
        for (int s = 0; s < n_patches; s++) {
            const float *vs = v_h + (size_t)s * hd + (size_t)i0 * 8;
            vfloat32m1_t sv = __riscv_vfmv_v_f_f32m1(sq_a[s], vl);
            vfloat32m1_t tv = __riscv_vfmv_v_f_f32m1(sq_b[s], vl);
            vfloat32m1_t x;
            x = __riscv_vle32_v_f32m1(vs, vl);
            a0 = __riscv_vfadd_vv_f32m1(__riscv_vfmul_vv_f32m1(sv, x, vl), a0, vl);
            b0 = __riscv_vfadd_vv_f32m1(__riscv_vfmul_vv_f32m1(tv, x, vl), b0, vl);
            x = __riscv_vle32_v_f32m1(vs + 8, vl);
            a1 = __riscv_vfadd_vv_f32m1(__riscv_vfmul_vv_f32m1(sv, x, vl), a1, vl);
            b1 = __riscv_vfadd_vv_f32m1(__riscv_vfmul_vv_f32m1(tv, x, vl), b1, vl);
            x = __riscv_vle32_v_f32m1(vs + 16, vl);
            a2 = __riscv_vfadd_vv_f32m1(__riscv_vfmul_vv_f32m1(sv, x, vl), a2, vl);
            b2 = __riscv_vfadd_vv_f32m1(__riscv_vfmul_vv_f32m1(tv, x, vl), b2, vl);
        }
        float *pa = o_a + (size_t)i0 * 8;
        float *pb = o_b + (size_t)i0 * 8;
        __riscv_vse32_v_f32m1(pa, a0, vl);
        __riscv_vse32_v_f32m1(pa + 8, a1, vl);
        __riscv_vse32_v_f32m1(pa + 16, a2, vl);
        __riscv_vse32_v_f32m1(pb, b0, vl);
        __riscv_vse32_v_f32m1(pb + 8, b1, vl);
        __riscv_vse32_v_f32m1(pb + 16, b2, vl);
    }
    /* hd/8 的尾数（72 → 0；非 3 倍数的 head_dim 才有）：单向量逐次处理。 */
    for (; i0 < nv; i0++) {
        vfloat32m1_t a = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        vfloat32m1_t b = a;
        for (int s = 0; s < n_patches; s++) {
            vfloat32m1_t x = __riscv_vle32_v_f32m1(
                v_h + (size_t)s * hd + (size_t)i0 * 8, vl);
            a = __riscv_vfadd_vv_f32m1(
                    __riscv_vfmul_vv_f32m1(__riscv_vfmv_v_f_f32m1(sq_a[s], vl), x, vl), a, vl);
            b = __riscv_vfadd_vv_f32m1(
                    __riscv_vfmul_vv_f32m1(__riscv_vfmv_v_f_f32m1(sq_b[s], vl), x, vl), b, vl);
        }
        __riscv_vse32_v_f32m1(o_a + (size_t)i0 * 8, a, vl);
        __riscv_vse32_v_f32m1(o_b + (size_t)i0 * 8, b, vl);
    }
}

/* 与标量参考路径同构的 RVV 版本（h 循环 + 三步流水线）。 */
static void vvis_vit_attn_block_rvv(vvis_vit_ctx *c, int pb) {
    int tid = vllm_tp_worker_id();
    if (tid < 0) tid = 0;
    const int vh = c->vh, hd = c->hd, nh = c->nh, n_patches = c->n_patches;
    const int p0 = pb * VVIS_ATTN_PB;
    const int pb_n = (p0 + VVIS_ATTN_PB <= n_patches) ? VVIS_ATTN_PB
                                                      : (n_patches - p0);
    float *scores = c->score_scratch + (size_t)tid * VVIS_ATTN_PB * n_patches;
    const size_t slab = (size_t)n_patches * hd;
    /* 逐位闸门：只在首层首个 patch 块的 head 0 上重算整块（scores 全矩阵 +
     * 整个 head 的输出），代价可忽略，覆盖两个内核的全部索引与累加序。 */
    const int chk = vvis_attn_chk_on() && c->layer_idx == 0 && pb == 0;
    long bad_sc = 0, bad_av = 0;

    for (int h = 0; h < nh; h++) {
        const float *k_h = c->qkv_buf + ((size_t)(nh + h)) * slab;   /* seg 1 */
        const float *q_h0 = c->qkv_buf + ((size_t)h) * slab;         /* seg 0 */
        vvis_attn_scores_rvv(q_h0 + (size_t)p0 * hd, k_h, scores,
                             pb_n, n_patches, hd, c->scale);

        if (chk && h == 0) {
            for (int q = 0; q < pb_n; q++) {
                const float *qh = q_h0 + (size_t)(p0 + q) * hd;
                for (int s = 0; s < n_patches; s++) {
                    const float *k_s = k_h + (size_t)s * hd;
                    float dot = 0.0f;
                    for (int i = 0; i < hd; i++) dot += qh[i] * k_s[i];
                    float want = dot * c->scale;
                    if (memcmp(&want, &scores[(size_t)q * n_patches + s],
                               sizeof(float)) != 0)
                        bad_sc++;
                }
            }
        }

        for (int q = 0; q < pb_n; q++)
            softmax_f32(scores + (size_t)q * n_patches, n_patches);

        const float *v_h = c->qkv_buf + ((size_t)(2 * nh + h)) * slab;  /* seg 2 */
        int q = 0;
        for (; q + 1 < pb_n; q += 2) {
            float *oa = c->attn_buf + (size_t)(p0 + q) * vh + (size_t)h * hd;
            float *ob = c->attn_buf + (size_t)(p0 + q + 1) * vh + (size_t)h * hd;
            vvis_attn_av_rvv(scores + (size_t)q * n_patches,
                             scores + (size_t)(q + 1) * n_patches,
                             v_h, oa, ob, n_patches, hd);
        }
        if (q < pb_n) {
            float *oa = c->attn_buf + (size_t)(p0 + q) * vh + (size_t)h * hd;
            vvis_attn_av_rvv(scores + (size_t)q * n_patches,
                             scores + (size_t)q * n_patches,
                             v_h, oa, oa, n_patches, hd);
        }
        if (chk && h == 0) {
            for (int qq = 0; qq < pb_n; qq++) {
                const float *sq = scores + (size_t)qq * n_patches;
                const float *oh = c->attn_buf + (size_t)(p0 + qq) * vh
                                  + (size_t)h * hd;
                for (int i = 0; i < hd; i++) {
                    float a = 0.0f;
                    for (int s = 0; s < n_patches; s++)
                        a += sq[s] * v_h[(size_t)s * hd + i];
                    if (memcmp(&a, &oh[i], sizeof(float)) != 0) bad_av++;
                }
            }
        }
    }
    if (chk) {
        fprintf(stderr, "[VISCHK] l%d pb%d hd=%d np=%d scores bad=%ld  av bad=%ld\n",
                c->layer_idx, pb, hd, n_patches, bad_sc, bad_av);
        fflush(stderr);
    }
}
#endif /* ST_HAVE_RVV */

static void vvis_vit_attn_block(void *ctx_, int pb) {
    vvis_vit_ctx *c = (vvis_vit_ctx *)ctx_;
#if ST_HAVE_RVV
    /* 要求 head_dim = vis_hidden / vis_heads 是 8 的倍数，否则落回标量。
     * 本仓两个 checkpoint 都满足：2B 为 1024/16 = 64，8B 为 1152/16 = 72。
     * 内核固定用 8 lane（vsetvl(8)），故 VLEN<256 时 VLMAX 不足 8，落回标量。 */
    if (vvis_rvv_on() && (c->hd & 7) == 0 && c->hd <= 128 &&
        __riscv_vsetvlmax_e32m1() >= 8) {
        vvis_vit_attn_block_rvv(c, pb);
        return;
    }
#endif
    int tid = vllm_tp_worker_id();
    if (tid < 0) tid = 0;
    const int vh = c->vh, hd = c->hd, nh = c->nh, n_patches = c->n_patches;
    const int v4n = hd / 4;
    const int p0 = pb * VVIS_ATTN_PB;
    const int pb_n = (p0 + VVIS_ATTN_PB <= n_patches) ? VVIS_ATTN_PB
                                                      : (n_patches - p0);
    float *scores = c->score_scratch + (size_t)tid * VVIS_ATTN_PB * n_patches;
    const size_t slab = (size_t)n_patches * hd;   /* per (seg, head) slab */

    for (int h = 0; h < nh; h++) {
        /* score: stream keys once (contiguous per head), dot all PB queries. */
        const float *k_h = c->qkv_buf + ((size_t)(nh + h)) * slab;   /* seg 1 */
        const float *q_h0 = c->qkv_buf + ((size_t)h) * slab;         /* seg 0 */
        for (int s = 0; s < n_patches; s++) {
            const float *k_s = k_h + (size_t)s * hd;
            for (int q = 0; q < pb_n; q++) {
                const float *q_h = q_h0 + (size_t)(p0 + q) * hd;
#if ST_HAVE_NEON
                float32x4_t acc[VVIS_ATTN_ACC];
                for (int i = 0; i < v4n; i++)
                    acc[i] = vmulq_f32(vld1q_f32(q_h + i * 4), vld1q_f32(k_s + i * 4));
                /* 归约 v4n 个 4 宽累加器：`v4n` 为 2 的幂时与旧的「逐次减半二分树」
                 * 逐位一致（2B 的 hd=64 → v4n=16，行为不变）；否则把**奇数尾项**
                 * 折回 acc[0]，避免漏项。旧实现 `for (w=v4n/2; w>=1; w>>=1)` 只覆盖
                 * 2 的幂：Qwen3-VL-8B 的 vis head_dim=72 → v4n=18 会静默丢掉
                 * acc[8]/acc[17]，即注意力点积漏掉 head 维 32-35 与 68-71
                 * （实测：8B 视觉语义全错、2B 正常，见 K-BENCH-020 排查）。 */
                for (int nacc = v4n; nacc > 1; ) {
                    int hf = nacc >> 1;
                    for (int i = 0; i < hf; i++)
                        acc[i] = vaddq_f32(acc[i], acc[i + hf]);
                    if (nacc & 1) acc[0] = vaddq_f32(acc[0], acc[nacc - 1]);
                    nacc = hf;
                }
                float32x2_t lo = vget_low_f32(acc[0]);
                float32x2_t hi = vget_high_f32(acc[0]);
                scores[(size_t)q * n_patches + s] =
                    (vget_lane_f32(lo, 0) + vget_lane_f32(lo, 1) +
                     vget_lane_f32(hi, 0) + vget_lane_f32(hi, 1)) * c->scale;
#else
                float dot = 0.0f;
                for (int i = 0; i < hd; i++) dot += q_h[i] * k_s[i];
                scores[(size_t)q * n_patches + s] = dot * c->scale;
#endif
            }
        }
        for (int q = 0; q < pb_n; q++)
            softmax_f32(scores + (size_t)q * n_patches, n_patches);

        /* attn@V: stream values once (contiguous), accumulate all PB queries. */
        const float *v_h = c->qkv_buf + ((size_t)(2 * nh + h)) * slab;  /* seg 2 */
        for (int q = 0; q < pb_n; q++) {
#if ST_HAVE_NEON
            float32x4_t accv[VVIS_ATTN_ACC];
            for (int i = 0; i < v4n; i++) accv[i] = vdupq_n_f32(0.0f);
            const float *sq = scores + (size_t)q * n_patches;
            for (int s = 0; s < n_patches; s++) {
                const float *v_s = v_h + (size_t)s * hd;
                float32x4_t sc = vdupq_n_f32(sq[s]);
                for (int i = 0; i < v4n; i++)
                    accv[i] = vfmaq_f32(accv[i], sc, vld1q_f32(v_s + i * 4));
            }
            float *out_h = c->attn_buf + (size_t)(p0 + q) * vh + h * hd;
            for (int i = 0; i < v4n; i++) vst1q_f32(out_h + i * 4, accv[i]);
#else
            const float *sq = scores + (size_t)q * n_patches;
            float *out_h = c->attn_buf + (size_t)(p0 + q) * vh + h * hd;
            for (int i = 0; i < hd; i++) {
                float a = 0.0f;
                for (int s = 0; s < n_patches; s++)
                    a += sq[s] * v_h[(size_t)s * hd + i];
                out_h[i] = a;
            }
#endif
        }
    }
}

/* Domain 2b: residual add after the attention projection (patch-parallel). */
static void vvis_vit_add1(void *ctx_, int p) {
    vvis_vit_ctx *c = (vvis_vit_ctx *)ctx_;
    float *xp = c->x + (size_t)p * c->vh;
    const float *mp = c->mlp_buf + (size_t)p * c->vh;
    for (int i = 0; i < c->vh; i++) xp[i] += mp[i];
}

/* Domain 2c: LN2 -> mlp_buf (patch-parallel). */
static void vvis_vit_ln2(void *ctx_, int p) {
    vvis_vit_ctx *c = (vvis_vit_ctx *)ctx_;
    layer_norm_f32(c->mlp_buf + (size_t)p * c->vh,
                   c->x + (size_t)p * c->vh,
                   c->norm2_w, c->norm2_b, c->vh, c->cfg->norm_eps);
}

/* Domain 2d: GELU on fc1_buf, zero the pad tail (patch-parallel). */
static void vvis_vit_gelu(void *ctx_, int p) {
    vvis_vit_ctx *c = (vvis_vit_ctx *)ctx_;
    float *fp = c->fc1_buf + (size_t)p * c->vf_pad;
    for (int i = 0; i < c->vf; i++) fp[i] = gelu_approx(fp[i]);
    for (int i = c->vf; i < c->vf_pad; i++) fp[i] = 0.0f;
}

/* Domain 2e: residual add after FC2 (patch-parallel). */
static void vvis_vit_add2(void *ctx_, int p) {
    vvis_vit_ctx *c = (vvis_vit_ctx *)ctx_;
    float *xp = c->x + (size_t)p * c->vh;
    const float *ap = c->attn_buf + (size_t)p * c->vh;
    for (int i = 0; i < c->vh; i++) xp[i] += ap[i];
}

static void vit_block_forward(STVisionState *vis, int layer_idx, int n_patches) {
    const STModelConfig *cfg = vis->cfg;
    static int vis_dbg = -1;
    if (vis_dbg < 0) vis_dbg = getenv("VLLM_MMDBG") != NULL;
    double t0 = vis_dbg ? st_now_sec() : 0.0;
    int vh = cfg->vis_hidden;
    int nh = cfg->vis_heads;
    int hd = vh / nh;  /* head_dim = 72 */
    int vf = cfg->vis_ffn;  /* 4304 */
    int vf_pad = VF_PAD(vf);

    float *x = vis->hidden_states;
    float *attn_buf = vis->attn_buf;
    float *mlp_buf = vis->mlp_buf;
    float *qkv_buf = vis->qkv_buf;
    float *fc1_buf = vis->fc1_buf;
    const float *norm1_w = vis->w.norm1_weight + (size_t)layer_idx * vh;
    const float *norm1_b = vis->w.norm1_bias + (size_t)layer_idx * vh;
    const float *norm2_w = vis->w.norm2_weight + (size_t)layer_idx * vh;
    const float *norm2_b = vis->w.norm2_bias + (size_t)layer_idx * vh;
    const float *fc1_b = vis->w.mlp_fc1_bias + (size_t)layer_idx * vf;
    const float *fc2_b = vis->w.mlp_fc2_bias + (size_t)layer_idx * vh;

    const uint8_t *q8_qkv = vis->w.q8_attn_qkv_weight + (size_t)layer_idx * VQ8_BYTES((size_t)vh * 3 * vh);
    const uint8_t *q8_proj = vis->w.q8_attn_proj_weight + (size_t)layer_idx * VQ8_BYTES((size_t)vh * vh);
    const uint8_t *q8_fc1  = vis->w.q8_mlp_fc1_weight  + (size_t)layer_idx * VQ8_BYTES((size_t)vf * vh);
    const uint8_t *q8_fc2  = vis->w.q8_mlp_fc2_weight  + (size_t)layer_idx * VQ8_BYTES((size_t)vh * vf_pad);

    vvis_vit_ctx vc;
    vc.vis = vis; vc.cfg = cfg;
    vc.x = x; vc.attn_buf = attn_buf; vc.mlp_buf = mlp_buf; vc.qkv_buf = qkv_buf; vc.fc1_buf = fc1_buf;
    vc.norm1_w = norm1_w; vc.norm1_b = norm1_b; vc.norm2_w = norm2_w; vc.norm2_b = norm2_b;
    vc.fc1_b = fc1_b; vc.fc2_b = fc2_b;
    /* Attention score staging: mlp_buf is idle until the proj matvec below,
     * so [nthreads * VVIS_ATTN_PB * n_patches] floats of it can stage the
     * per-thread softmax inputs of the blocked attention kernel. */
    vc.score_scratch = mlp_buf;
    vc.q8_qkv = q8_qkv; vc.q8_proj = q8_proj; vc.q8_fc1 = q8_fc1; vc.q8_fc2 = q8_fc2;
    vc.vh = vh; vc.nh = nh; vc.hd = hd; vc.vf = vf; vc.vf_pad = vf_pad;
    vc.layer_idx = layer_idx; vc.n_patches = n_patches;
    vc.scale = 1.0f / sqrtf((float)hd);

    /* ===== Domain 1: LN1 (patch-parallel) -> batch QKV (weights streamed
     * once, reused across all patches) -> RoPE (patch-parallel).
     * The batched matvec eliminates the per-patch full-weight re-read that
     * made large images (many patches) take minutes on RK3588. ===== */
    vllm_tp_parfor(0, n_patches, vvis_vit_ln1, &vc);
    st_q8_matvec_batch_qkv(qkv_buf, q8_qkv,
                           vis->w.attn_qkv_bias + (size_t)layer_idx * 3 * vh,
                           attn_buf, n_patches, 3 * vh, vh, nh, hd);
    vllm_tp_parfor(0, n_patches, vvis_vit_rope, &vc);
    if (vis_dbg)
        fprintf(stderr, "[VIS]   l%d d1_qkv %.3fs\n", layer_idx, st_now_sec() - t0);

    /* ===== Domain 2: attention (patch-block parallel, K/V rows reused by
     * VVIS_ATTN_PB queries per worker) -> batch proj -> residual -> LN2 ->
     * batch fc1 -> GELU -> batch fc2 -> residual ===== */
    vllm_tp_parfor(0, (n_patches + VVIS_ATTN_PB - 1) / VVIS_ATTN_PB,
                   vvis_vit_attn_block, &vc);
    if (vis_dbg)
        fprintf(stderr, "[VIS]   l%d attn   %.3fs\n", layer_idx, st_now_sec() - t0);
    st_q8_matvec_batch(mlp_buf, q8_proj,
                       vis->w.attn_proj_bias + (size_t)layer_idx * vh,
                       attn_buf, n_patches, vh, vh);
    vllm_tp_parfor(0, n_patches, vvis_vit_add1, &vc);
    vllm_tp_parfor(0, n_patches, vvis_vit_ln2, &vc);
    if (vis_dbg)
        fprintf(stderr, "[VIS]   l%d d2a     %.3fs\n", layer_idx, st_now_sec() - t0);
    st_q8_matvec_batch(fc1_buf, q8_fc1, fc1_b, mlp_buf, n_patches, vf, vh);
    vllm_tp_parfor(0, n_patches, vvis_vit_gelu, &vc);
    st_q8_matvec_batch(attn_buf, q8_fc2, fc2_b, fc1_buf, n_patches, vh, vf_pad);
    vllm_tp_parfor(0, n_patches, vvis_vit_add2, &vc);
    if (vis_dbg) {
        double _dt = st_now_sec() - t0;
        fprintf(stderr, "[VIS]   l%d d2b     %.3fs\n", layer_idx, _dt);
        fflush(stderr);
    }
}

/* ================================================================
 * Image Preprocessing & Patch Embedding
 * ================================================================ */

/**
 * Preprocess image: resize to compatible dimensions and normalize.
 * Qwen3-VL checkpoint normalization (preprocessor_config.json):
 *   image_mean = image_std = [0.5, 0.5, 0.5]
 *   pixel = (p/255 - 0.5) / 0.5 = p/127.5 - 1  →  [-1, 1]
 * (Qwen2.5-VL used ImageNet stats; Qwen3-VL overrides them to 0.5/0.5.
 *  Using the wrong stats shifts every ViT input by ~1.86x scale + offset.)
 */
static int preprocess_image(float *__restrict dst,
                             const uint8_t *__restrict src,
                             int src_w, int src_h,
                             int *out_w, int *out_h,
                             int patch_size, int max_grid) {
    /* Choose the resize target. The vision encoder's buffers are sized for
     * `max_grid` patches (vis_max_pos, e.g. 48x48=2304), so a photo that
     * would exceed that grid must be downscaled (aspect-preserving) instead
     * of rejected/crashing. */
    int gw = (src_w + patch_size - 1) / patch_size;
    int gh = (src_h + patch_size - 1) / patch_size;
    int target_w, target_h;
    if (gw * gh > max_grid) {
        double scale = sqrt((double)max_grid / ((double)gw * (double)gh));
        target_w = (int)((double)src_w * scale / patch_size) * patch_size;
        target_h = (int)((double)src_h * scale / patch_size) * patch_size;
        if (target_w < patch_size) target_w = patch_size;
        if (target_h < patch_size) target_h = patch_size;
        /* The snap can overshoot the grid for extreme aspect ratios; shrink
         * the longer side until it fits. */
        while ((target_w / patch_size) * (target_h / patch_size) > max_grid) {
            if (target_w >= target_h) target_w -= patch_size;
            else                     target_h -= patch_size;
            if (target_w < patch_size || target_h < patch_size) break;
        }
    } else {
        target_w = gw * patch_size;
        target_h = gh * patch_size;
    }

    /* Clamp to reasonable size */
    if (target_w > 2048) target_w = 2048;
    if (target_h > 2048) target_h = 2048;

    *out_w = target_w;
    *out_h = target_h;

    for (int y = 0; y < target_h; y++) {
        int sy = y * src_h / target_h;
        for (int x = 0; x < target_w; x++) {
            int sx = x * src_w / target_w;
            int src_idx = (sy * src_w + sx) * 3;
            int dst_idx = (y * target_w + x) * 3;
            for (int c = 0; c < 3; c++) {
                float v = (float)src[src_idx + c];
                dst[dst_idx + c] = v / 127.5f - 1.0f;
            }
        }
    }
    return 0;
}

/**
 * Patch embedding: split image into patches and project to hidden_dim.
 * Input: pixel_values [height * width * 3], normalized to [-1, 1]
 * Output: patch_features [grid_h * grid_w, vis_hidden]
 *
 * The patch_embed is a Conv3d: weight [vis_hidden, 3, temporal_patch, patch, patch]
 * For images: temporal=1 (single frame)
 * For video: temporal=temporal_patch_size
 *
 * We implement this as direct patch extraction + matvec, not actual convolution.
 */
typedef struct {
    STVisionState *vis; const float *pw, *pb, *pixels;
    int grid_h, grid_w, vh, vt, ps, img_w;
} vvis_conv_ctx;

static void vvis_patch_conv_worker(void *ctx_, int gy) {
    vvis_conv_ctx *c = (vvis_conv_ctx *)ctx_;
#if defined(__aarch64__)
    const int npx = 3 * c->vt * c->ps * c->ps;   /* 1536 for ps=16, vt=2 */
    /* stack staging in the contiguous [cc][t][py][px] layout the conv weight
     * uses; images are embedded as vt identical frames, so t just repeats. */
    float patch_px[3 * 2 * 16 * 16];
    for (int gx = 0; gx < c->grid_w; gx++) {
        int p_idx = gy * c->grid_w + gx;
        for (int cc = 0; cc < 3; cc++)
            for (int t = 0; t < c->vt; t++)
                for (int py = 0; py < c->ps; py++) {
                    const float *row = c->pixels +
                        (((size_t)(gy * c->ps + py) * c->img_w + gx * c->ps) * 3 + cc);
                    float *dst = patch_px + ((cc * c->vt + t) * c->ps + py) * c->ps;
                    for (int px = 0; px < c->ps; px++)
                        dst[px] = row[px * 3];
                }
        float *feat = c->vis->patch_features + (size_t)p_idx * c->vh;
        for (int oc = 0; oc < c->vh; oc++) {
            const float *w_oc = c->pw + (size_t)oc * npx;
            float32x4_t acc = vdupq_n_f32(0.0f);
            int i = 0;
            for (; i + 4 <= npx; i += 4)
                acc = vfmaq_f32(acc, vld1q_f32(w_oc + i), vld1q_f32(patch_px + i));
            float sum = vaddvq_f32(acc);
            for (; i < npx; i++) sum += w_oc[i] * patch_px[i];
            feat[oc] = sum + c->pb[oc];
        }
    }
#else
    for (int gx = 0; gx < c->grid_w; gx++) {
        int p_idx = gy * c->grid_w + gx;
        float *feat = c->vis->patch_features + (size_t)p_idx * c->vh;
        for (int oc = 0; oc < c->vh; oc++) {
            double sum = 0.0;
            const float *w_oc = c->pw + (size_t)oc * 3 * c->vt * c->ps * c->ps;
            for (int cc = 0; cc < 3; cc++) {
                for (int t = 0; t < c->vt; t++) {
                    for (int py = 0; py < c->ps; py++) {
                        for (int px = 0; px < c->ps; px++) {
                            int pixel_y = gy * c->ps + py;
                            int pixel_x = gx * c->ps + px;
                            int pixel_idx = (pixel_y * c->img_w + pixel_x) * 3 + cc;
                            float pixel = c->pixels[pixel_idx];
                            int w_idx = cc * c->vt * c->ps * c->ps + t * c->ps * c->ps + py * c->ps + px;
                            sum += (double)w_oc[w_idx] * pixel;
                        }
                    }
                }
            }
            feat[oc] = (float)sum + c->pb[oc];
        }
    }
#endif
}

static int patch_embed_image(STVisionState *vis,
                              const float *pixels, int img_h, int img_w) {
    const STModelConfig *cfg = vis->cfg;
    int vh = cfg->vis_hidden;
    int ps = cfg->vis_patch;   /* 16 */
    int vt = cfg->vis_temporal; /* 2: images are embedded as 2 identical frames */

    int grid_h = img_h / ps;
    int grid_w = img_w / ps;
    int n_patches = grid_h * grid_w;

    if (n_patches > cfg->vis_max_pos) {
        fprintf(stderr, "[VIS] Too many patches: %d > max %d\n", n_patches, cfg->vis_max_pos);
        return -1;
    }

    vis->grid_h = grid_h;
    vis->grid_w = grid_w;
    vis->n_frames_eff = 1;

    /* Extract patches and project.
     * Conv weight layout: [vh, 3, vt, ps, ps] (temporal kernel first for
     * each channel). Qwen treats a still image as vt identical frames, so
     * the temporal kernel is summed over vt duplicate frames. */
    const float *pw = vis->w.patch_embed_weight;
    const float *pb = vis->w.patch_embed_bias;

    vvis_conv_ctx vc = { vis, pw, pb, pixels, grid_h, grid_w, vh, vt, ps, img_w };
    vllm_tp_parfor(0, grid_h, vvis_patch_conv_worker, &vc);

    return n_patches;
}

/* ================================================================
 * Vision RoPE + interpolated position embedding
 * ================================================================ */

/**
 * Vision RoPE (Qwen3-VL reference: Qwen3VLVisionRotaryEmbedding +
 * get_vision_position_ids + apply_rotary_pos_emb_vision).
 *
 * Each patch token carries a 2D (h, w) grid position. The rotary table has
 *   inv_freq[i] = 1 / theta^(2*i / (head_dim/2))   theta=10000, i in [0, hd/4)
 * and the flattened rotary embedding for a patch is
 *   [h * inv_freq, w * inv_freq]  (hd/2 entries), repeated once (hd entries).
 *
 * Also adds the interpolated position embedding here (bilinear resample of
 * the square vis_max_pos table — 48x48 grid — onto the actual grid), which
 * replaces the old direct row-major index.
 *
 * Patch layout is raster order: images [row, col]; videos [t, row, col]
 * (each frame shares the same spatial positions). This matches the official
 * processor: sequence order is arbitrary for the (non-causal) ViT attention,
 * what matters is that each token's position code matches its grid location.
 */
typedef struct {
    STVisionState *vis; const float *pe, *inv_freq;
    int n_patches, n_spatial, grid_h, grid_w, side, vh, hd, half, n_freq;
} vvis_pos_ctx;

static void vvis_pos_worker(void *ctx_, int p) {
    vvis_pos_ctx *c = (vvis_pos_ctx *)ctx_;
    int within = p % c->n_spatial;
    int row = within / c->grid_w;
    int col = within % c->grid_w;

    float sh = (c->grid_h > 1) ? (float)row * (c->side - 1) / (float)(c->grid_h - 1) : 0.0f;
    float sw = (c->grid_w > 1) ? (float)col * (c->side - 1) / (float)(c->grid_w - 1) : 0.0f;
    int h0 = (int)floorf(sh);
    if (h0 > c->side - 1) h0 = c->side - 1;
    int h1 = (h0 + 1 < c->side) ? h0 + 1 : h0;
    int w0 = (int)floorf(sw);
    if (w0 > c->side - 1) w0 = c->side - 1;
    int w1 = (w0 + 1 < c->side) ? w0 + 1 : w0;
    float wh = sh - h0;
    float ww = sw - w0;

    float *feat = c->vis->patch_features + (size_t)p * c->vh;
    const float *p00 = c->pe + ((size_t)h0 * c->side + w0) * c->vh;
    const float *p01 = c->pe + ((size_t)h0 * c->side + w1) * c->vh;
    const float *p10 = c->pe + ((size_t)h1 * c->side + w0) * c->vh;
    const float *p11 = c->pe + ((size_t)h1 * c->side + w1) * c->vh;
    for (int d = 0; d < c->vh; d++) {
        float v = (1.0f - wh) * (1.0f - ww) * p00[d]
                + (1.0f - wh) * ww       * p01[d]
                + wh       * (1.0f - ww) * p10[d]
                + wh       * ww          * p11[d];
        feat[d] += v;
    }

    /* RoPE cos/sin table ([p, hd]) */
    float *rc = c->vis->rot_cos + (size_t)p * c->hd;
    float *rs = c->vis->rot_sin + (size_t)p * c->hd;
    for (int d = 0; d < c->half; d++) {
        int fi = (d < c->n_freq) ? d : d - c->n_freq;
        float ang = ((d < c->n_freq) ? (float)row : (float)col) * c->inv_freq[fi];
        float cc = cosf(ang), ss = sinf(ang);
        rc[d] = cc; rs[d] = ss;
        rc[d + c->half] = cc; rs[d + c->half] = ss;
    }
}

static void vision_pos_encode(STVisionState *vis, int n_patches,
                              int grid_t, int grid_h, int grid_w) {
    const STModelConfig *cfg = vis->cfg;
    int vh = cfg->vis_hidden;
    int hd = vh / cfg->vis_heads;          /* 72 */
    int half = hd / 2;                     /* 36 */
    int n_freq = half / 2;                 /* 18 */
    int side = (int)sqrtf((float)cfg->vis_max_pos);  /* 48 */

    float inv_freq[64];
    for (int i = 0; i < n_freq; i++) {
        /* arange(0, dim, 2)/dim with dim = half -> exponent 2*i/half */
        inv_freq[i] = 1.0f / powf(10000.0f, (float)(2 * i) / (float)half);
    }

    const float *pe = vis->w.pos_embed;    /* [side*side, vh] */
    int n_spatial = grid_h * grid_w;
    if (n_spatial <= 0) n_spatial = 1;

    vvis_pos_ctx vc = { vis, pe, inv_freq, n_patches, n_spatial, grid_h, grid_w,
                        side, vh, hd, half, n_freq };
    vllm_tp_parfor(0, n_patches, vvis_pos_worker, &vc);
}

/* ================================================================
 * Spatial Merge & Merger
 * ================================================================ */

/**
 * Spatial merge: group 2x2 patches and concatenate their features.
 * Input: [grid_h, grid_w, vh]
 * Output: [grid_h/2, grid_w/2, vh*4]
 */
typedef struct { float *merged; const float *patches; int grid_h, grid_w, vh; } vvis_merge_ctx;

static void vvis_merge_worker(void *ctx_, int my) {
    vvis_merge_ctx *c = (vvis_merge_ctx *)ctx_;
    const int merge = 2;
    int mg_w = c->grid_w / merge;
    int out_dim = c->vh * merge * merge;  /* vh * 4 */
    for (int mx = 0; mx < mg_w; mx++) {
        float *out = c->merged + ((size_t)my * mg_w + mx) * out_dim;
        for (int dy = 0; dy < merge; dy++) {
            for (int dx = 0; dx < merge; dx++) {
                int py = my * merge + dy;
                int px = mx * merge + dx;
                const float *src = c->patches + ((size_t)py * c->grid_w + px) * c->vh;
                float *dst_slot = out + (dy * merge + dx) * c->vh;
                memcpy(dst_slot, src, c->vh * sizeof(float));
            }
        }
    }
}

static void spatial_merge(float *__restrict merged,
                           const float *__restrict patches,
                           int grid_h, int grid_w, int vh) {
    int merge = 2;
    int mg_h = grid_h / merge;
    int mg_w = grid_w / merge;
    int out_dim = vh * merge * merge;  /* vh * 4 */

    vvis_merge_ctx c = { merged, patches, grid_h, grid_w, vh };
    vllm_tp_parfor(0, mg_h, vvis_merge_worker, &c);
}

/**
 * Merger forward: normalized → fc1 → GELU → fc2
 */
typedef struct {
    float *output; const float *input;
    int n_tokens, in_dim, out_dim, chunk_dim, full_norm, n_chunks, mid_dim;
    const float *norm_w, *norm_b, *fc1_w, *fc1_b, *fc2_w, *fc2_b;
    float norm_eps;
} vvis_merger_ctx;

static void vvis_merger_worker(void *ctx_, int i) {
    vvis_merger_ctx *c = (vvis_merger_ctx *)ctx_;
    const float *inp = c->input + (size_t)i * c->in_dim;
    float *outp = c->output + (size_t)i * c->out_dim;

    float *normed = (float*)alloca((size_t)c->in_dim * sizeof(float));
    if (c->full_norm) {
        layer_norm_f32(normed, inp, c->norm_w, c->norm_b, c->in_dim, c->norm_eps);
    } else {
        for (int chunk = 0; chunk < c->n_chunks; chunk++) {
            layer_norm_f32(normed + (size_t)chunk * c->chunk_dim,
                           inp + (size_t)chunk * c->chunk_dim,
                           c->norm_w, c->norm_b, c->chunk_dim, c->norm_eps);
        }
    }

    float *mid = (float*)alloca((size_t)c->mid_dim * sizeof(float));
    matvec_f32_bias(mid, c->fc1_w, normed, c->fc1_b, c->mid_dim, c->in_dim);
    for (int j = 0; j < c->mid_dim; j++) mid[j] = gelu_approx(mid[j]);
    matvec_f32_bias(outp, c->fc2_w, mid, c->fc2_b, c->out_dim, c->mid_dim);
}

static void merger_forward(float *__restrict output,
                            const float *__restrict input,
                            int n_tokens, int in_dim, int out_dim,
                            const float *norm_w, const float *norm_b,
                            const float *fc1_w, const float *fc1_b,
                            const float *fc2_w, const float *fc2_b,
                            int mid_dim, float norm_eps,
                            int chunk_dim, int full_norm) {
    int n_chunks = in_dim / chunk_dim;  /* 4 for 2x2 spatial merge */
    vvis_merger_ctx c = { output, input, n_tokens, in_dim, out_dim, chunk_dim,
                          full_norm, n_chunks, mid_dim, norm_w, norm_b,
                          fc1_w, fc1_b, fc2_w, fc2_b, norm_eps };
    vllm_tp_parfor(0, n_tokens, vvis_merger_worker, &c);
}

/**
 * DeepStack merger: run spatial merge + the deepstack layer's own merger
 * (full-norm, matching the official VisionSpatialMerger with
 * use_postshuffle_norm=True) on the BLOCK OUTPUT of a deepstack layer.
 * Produces ds_features[d] = [n_merged, vis_out_dim] for later injection
 * into the LLM's first layers (official _deepstack_process).
 */
static void vision_ds_merge(STVisionState *vis, int d,
                            int grid_t, int grid_h, int grid_w, int n_merged) {
    const STModelConfig *cfg = vis->cfg;
    int vh = cfg->vis_hidden;
    int merge = cfg->vis_merge;
    int merge_in = vh * merge * merge;
    int mg_h = grid_h / merge;
    int mg_w = grid_w / merge;
    int n_spatial = grid_h * grid_w;
    int out_dim = cfg->vis_out_dim;
    int ds_mid = merge_in;

    float *merged_buf = (float *)calloc((size_t)n_merged * merge_in, sizeof(float));
    for (int t = 0; t < grid_t; t++) {
        spatial_merge(merged_buf + (size_t)(t * mg_h * mg_w) * merge_in,
                      vis->hidden_states + (size_t)(t * n_spatial) * vh,
                      grid_h, grid_w, vh);
    }
    merger_forward(vis->ds_features[d], merged_buf, n_merged, merge_in, out_dim,
                   vis->w.ds_norm_weight + (size_t)d * merge_in,
                   vis->w.ds_norm_bias   + (size_t)d * merge_in,
                   vis->w.ds_fc1_weight  + (size_t)d * merge_in * ds_mid,
                   vis->w.ds_fc1_bias    + (size_t)d * ds_mid,
                   vis->w.ds_fc2_weight  + (size_t)d * ds_mid * out_dim,
                   vis->w.ds_fc2_bias    + (size_t)d * out_dim,
                   ds_mid, cfg->norm_eps, merge_in, 1);
    free(merged_buf);
}

/* ================================================================
 * Full Image Encoding Pipeline
 * ================================================================ */

int st_vision_encode_image(STVisionState *vis,
                            const uint8_t *image_data,
                            int width, int height) {
    const STModelConfig *cfg = vis->cfg;
    if (!cfg->has_vision) return -1;

    const int vis_dbg = getenv("VLLM_MMDBG") != NULL;
    double t_v = vis_dbg ? st_now_sec() : 0.0;
#define VIS_STEP(tag) do { if (vis_dbg) { \
        double _now = st_now_sec(); \
        fprintf(stderr, "[VIS] %-10s %.3fs (cum %.3fs)\n", tag, \
                _now - t_v, _now - t_v); \
        fflush(stderr); t_v = _now; } } while (0)

    int patch_size = cfg->vis_patch;
    int vh = cfg->vis_hidden;

    /* Step 1: Preprocess (resize + normalize) */
    int img_w, img_h;
    /* Allocate pixel buffer */
    free(vis->pixel_values);
    int max_pixels = (2048 / patch_size) * patch_size;
    max_pixels = max_pixels * max_pixels * 3;
    vis->pixel_values = calloc(max_pixels, sizeof(float));

    preprocess_image(vis->pixel_values, image_data, width, height,
                     &img_w, &img_h, patch_size, cfg->vis_max_pos);
    vis->img_width = img_w;
    vis->img_height = img_h;
    vis->n_frames = 1;
    VIS_STEP("preprocess");

    /* Step 2: Patch embedding */
    int n_patches = patch_embed_image(vis, vis->pixel_values, img_h, img_w);
    if (n_patches < 0) return -1;
    VIS_STEP("patch_embed");

    /* Step 3: Interpolated position embedding + vision RoPE table */
    vision_pos_encode(vis, n_patches, 1, vis->grid_h, vis->grid_w);
    VIS_STEP("pos_embed");

    /* Step 4: Copy to hidden_states for ViT processing */
    memcpy(vis->hidden_states, vis->patch_features,
           (size_t)n_patches * vh * sizeof(float));

    /* Step 5: ViT forward with DeepStack (features taken from the block
     * OUTPUT, matching the official forward: blk() then deepstack merger) */
    int grid_h = vis->grid_h;
    int grid_w = vis->grid_w;

    for (int l = 0; l < cfg->vis_depth; l++) {
        vit_block_forward(vis, l, n_patches);
        for (int d = 0; d < cfg->vis_ds_count; d++) {
            if (cfg->vis_ds_idx[d] == l) {
                /* Run the deepstack layer's own merger (full-norm) over the
                 * merged grid → ds_features[d] [n_merged, out_dim]. */
                int mg_h = grid_h / cfg->vis_merge;
                int mg_w = grid_w / cfg->vis_merge;
                vision_ds_merge(vis, d, 1, grid_h, grid_w, mg_h * mg_w);
                break;
            }
        }
        if (vis_dbg && (l == 0 || l == 5 || l == 10 || l == 15 || l == 20 ||
                        l == cfg->vis_depth - 1)) {
            double _now = st_now_sec();
            fprintf(stderr, "[VIS] vit l=%d/%d  %.3fs (cum %.3fs)\n", l,
                    cfg->vis_depth, _now - t_v, _now - t_v);
            fflush(stderr);
            t_v = _now;
        }
    }
    VIS_STEP("vit_all");

    /* Step 6: Spatial merge of final ViT output */
    int merge = cfg->vis_merge;
    int mg_h = grid_h / merge;
    int mg_w = grid_w / merge;
    int merge_in = vh * merge * merge;
    int n_merged = mg_h * mg_w;

    float *merged_buf = (float*)calloc((size_t)n_merged * merge_in, sizeof(float));
    spatial_merge(merged_buf, vis->hidden_states, grid_h, grid_w, vh);
    VIS_STEP("merge");

    /* Step 7: Main merger */
    int out_dim = cfg->vis_out_dim;  /* 4096 */
    float *main_tokens = (float*)calloc((size_t)n_merged * out_dim, sizeof(float));
    int merge_mid = merge_in;
    merger_forward(main_tokens, merged_buf, n_merged, merge_in, out_dim,
                   vis->w.merger_norm_weight, vis->w.merger_norm_bias,
                   vis->w.merger_fc1_weight, vis->w.merger_fc1_bias,
                   vis->w.merger_fc2_weight, vis->w.merger_fc2_bias,
                   merge_mid, cfg->norm_eps, vh, 0);

    /* Step 8: Visual token output = main merged tokens only.
     * DeepStack features (vis->ds_features) are kept for future layer-wise
     * injection into the LLM (official Qwen3-VL _deepstack_process) — they
     * are NOT part of the visual-token sequence. */
    vis->n_vis_tokens = n_merged;

    /* Reallocate visual tokens buffer if needed */
    free(vis->visual_tokens);
    vis->visual_tokens = calloc((size_t)vis->n_vis_tokens * out_dim, sizeof(float));

    /* Copy main merged tokens */
    memcpy(vis->visual_tokens, main_tokens,
           (size_t)n_merged * out_dim * sizeof(float));

    free(merged_buf);
    free(main_tokens);

    printf("[VIS] Image encoded: %dx%d → %d patches → %d visual tokens\n",
           img_w, img_h, n_patches, vis->n_vis_tokens);
    return vis->n_vis_tokens;
}

/* ================================================================
 * Video Encoding Pipeline
 * ================================================================ */

typedef struct {
    STVisionState *vis; const float *pw, *pb;
    int grid_h, grid_w, vh, temporal, patch_size, img_w, img_h, n_frames;
} vvis_vconv_ctx;

static void vvis_vconv_worker(void *ctx_, int t) {
    vvis_vconv_ctx *c = (vvis_vconv_ctx *)ctx_;
    for (int gy_v = 0; gy_v < c->grid_h; gy_v++) {
        for (int gx_v = 0; gx_v < c->grid_w; gx_v++) {
            int p_idx = (t * c->grid_h + gy_v) * c->grid_w + gx_v;
            float *feat = c->vis->patch_features + (size_t)p_idx * c->vh;

            for (int oc = 0; oc < c->vh; oc++) {
                double sum = 0.0;
                const float *w_oc = c->pw + (size_t)oc * 3 * c->temporal * c->patch_size * c->patch_size;

                int f_start = t * c->temporal;
                int f_end = (t + 1) * c->temporal;
                if (f_end > c->n_frames) f_end = c->n_frames;
                int frames_in_patch = f_end - f_start;

                for (int df = 0; df < frames_in_patch; df++) {
                    int frame_idx = f_start + df;
                    const float *frame = c->vis->pixel_values +
                        (size_t)frame_idx * c->img_w * c->img_h * 3;

                    for (int cc = 0; cc < 3; cc++) {
                        for (int py = 0; py < c->patch_size; py++) {
                            for (int px = 0; px < c->patch_size; px++) {
                                int pixel_y = gy_v * c->patch_size + py;
                                int pixel_x = gx_v * c->patch_size + px;
                                float pixel = frame[(pixel_y * c->img_w + pixel_x) * 3 + cc];
                                int w_idx = cc * c->temporal * c->patch_size * c->patch_size +
                                            df * c->patch_size * c->patch_size +
                                            py * c->patch_size + px;
                                sum += (double)w_oc[w_idx] * pixel;
                            }
                        }
                    }
                }
                feat[oc] = (float)sum + c->pb[oc];
            }
        }
    }
}

int st_vision_encode_video(STVisionState *vis,
                            const uint8_t **frame_data,
                            int width, int height, int n_frames) {
    const STModelConfig *cfg = vis->cfg;
    if (!cfg->has_vision) return -1;

    int patch_size = cfg->vis_patch;
    int temporal = cfg->vis_temporal;
    int vh = cfg->vis_hidden;

    /* Step 1: Preprocess each frame. grid_t is known up front (depends only
     * on n_frames and temporal), so each frame is downscaled to fit
     * vis_max_pos / grid_t spatial patches — the total grid then always
     * fits vis_max_pos. */
    int grid_t = (n_frames + temporal - 1) / temporal;
    int frame_max_grid = cfg->vis_max_pos / grid_t;
    if (frame_max_grid < 1) frame_max_grid = 1;

    int img_w, img_h;
    free(vis->pixel_values);
    int max_pixels = (2048 / patch_size) * patch_size;
    max_pixels = max_pixels * max_pixels * 3 * n_frames;
    vis->pixel_values = calloc(max_pixels, sizeof(float));

    vis->n_frames = n_frames;
    vis->img_width = width;
    vis->img_height = height;

    int frame_elems = 0;
    for (int f = 0; f < n_frames; f++) {
        int fw, fh;
        float *frame_pixels = vis->pixel_values + frame_elems;
        preprocess_image(frame_pixels, frame_data[f], width, height,
                         &fw, &fh, patch_size, frame_max_grid);
        if (f == 0) { img_w = fw; img_h = fh; }
        frame_elems += img_w * img_h * 3;
    }

    /* Step 2: Temporal patch embedding
     * Group frames into temporal patches of size `temporal`.
     * For each temporal patch, do 3D patch embedding. */
    int grid_h = img_h / patch_size;
    int grid_w = img_w / patch_size;
    int n_spatial = grid_h * grid_w;
    int n_patches = grid_t * n_spatial;

    vis->grid_h = grid_h;
    vis->grid_w = grid_w;
    vis->n_frames_eff = grid_t;

    if (n_patches > cfg->vis_max_pos) {
        fprintf(stderr, "[VIS] Too many video patches: %d > max %d\n",
                n_patches, cfg->vis_max_pos);
        return -1;
    }

    const float *pw = vis->w.patch_embed_weight;
    const float *pb = vis->w.patch_embed_bias;

    vvis_vconv_ctx vc = { vis, pw, pb, grid_h, grid_w, vh, temporal,
                          patch_size, img_w, img_h, n_frames };
    vllm_tp_parfor(0, grid_t, vvis_vconv_worker, &vc);

    /* Step 3: Interpolated position embedding + vision RoPE table
     * (each temporal patch reuses the same spatial positions) */
    vision_pos_encode(vis, n_patches, grid_t, grid_h, grid_w);

    /* Step 4: ViT forward (same as image; DeepStack features from block
     * output, matching the official forward) */
    memcpy(vis->hidden_states, vis->patch_features,
           (size_t)n_patches * vh * sizeof(float));

    for (int l = 0; l < cfg->vis_depth; l++) {
        vit_block_forward(vis, l, n_patches);
        for (int d = 0; d < cfg->vis_ds_count; d++) {
            if (cfg->vis_ds_idx[d] == l) {
                /* DeepStack merger per temporal group → [n_merged, out_dim] */
                int mg_h = grid_h / cfg->vis_merge;
                int mg_w = grid_w / cfg->vis_merge;
                vision_ds_merge(vis, d, grid_t, grid_h, grid_w, grid_t * mg_h * mg_w);
                break;
            }
        }
    }

    /* Step 5-8: Same merge + merger pipeline as image */
    int merge = cfg->vis_merge;
    int mg_h = grid_h / merge;
    int mg_w = grid_w / merge;
    int merge_in = vh * merge * merge;
    int n_merged = grid_t * mg_h * mg_w;

    float *merged_buf = (float*)calloc((size_t)n_merged * merge_in, sizeof(float));
    /* Spatial merge per temporal group */
    for (int t2 = 0; t2 < grid_t; t2++) {
        int src_off = t2 * n_spatial;
        int dst_off = t2 * mg_h * mg_w;
        spatial_merge(merged_buf + (size_t)dst_off * merge_in,
                      vis->hidden_states + (size_t)src_off * vh,
                      grid_h, grid_w, vh);
    }

    int out_dim = cfg->vis_out_dim;
    float *main_tokens = (float*)calloc((size_t)n_merged * out_dim, sizeof(float));
    int merge_mid = merge_in;
    merger_forward(main_tokens, merged_buf, n_merged, merge_in, out_dim,
                   vis->w.merger_norm_weight, vis->w.merger_norm_bias,
                   vis->w.merger_fc1_weight, vis->w.merger_fc1_bias,
                   vis->w.merger_fc2_weight, vis->w.merger_fc2_bias,
                   merge_mid, cfg->norm_eps, vh, 0);

    /* Visual token output = main merged tokens only (DeepStack features are
     * kept for future layer-wise injection, NOT part of the sequence). */
    vis->n_vis_tokens = n_merged;

    free(vis->visual_tokens);
    vis->visual_tokens = calloc((size_t)vis->n_vis_tokens * out_dim, sizeof(float));
    memcpy(vis->visual_tokens, main_tokens,
           (size_t)n_merged * out_dim * sizeof(float));

    free(merged_buf);
    free(main_tokens);

    printf("[VIS] Video encoded: %d frames %dx%d → %d patches → %d visual tokens\n",
           n_frames, img_w, img_h, n_patches, vis->n_vis_tokens);
    return vis->n_vis_tokens;
}

/* ================================================================
 * Accessor Functions
 * ================================================================ */

const float* st_vision_get_tokens(const STVisionState *vis) {
    return vis->visual_tokens;
}

int st_vision_get_num_tokens(const STVisionState *vis) {
    return vis->n_vis_tokens;
}
