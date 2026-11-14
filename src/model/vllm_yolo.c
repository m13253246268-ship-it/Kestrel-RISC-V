/**
 * vllm_yolo.c - YOLO 目标检测引擎实现（自有格式 VQF v4，纯 CPU 执行器）
 *
 * 本文件 = 图/权重的 C 侧执行器，与 tools/yolo/yolo_vqf_run.py（numpy 参考执行器）
 * 语义一一对应：同格式、同算子、同后处理。
 *
 * M0：CPU F32，正确性优先。
 * K-YOLO-002：int8 权重 + im2col 的 CPU dotprod 卷积（ARM NEON；其他平台标量回退），
 *     失败/不适用时显式回退标量（公理 silent_fallback_false_pass）。
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "vqf_format.h"
#include "vllm_yolo.h"
#include "vllm_platform.h"         /* ST_HAVE_NEON / ST_HAVE_RVV (int8 点积 CPU 卷积) */

#if ST_HAVE_RVV
#include <riscv_vector.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif
#include <time.h>
#ifdef __linux__
#include <sched.h>          /* V56：IME 只允许跑在带 ime 扩展的 hart0-3，需要绑核 */
#endif
static double ynow(void) {
#ifdef __linux__
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
#else
    return (double)clock() / (double)CLOCKS_PER_SEC;
#endif
}

/* ---- 图算子（与 tools/yolo/yolo_to_vqf.py 冻结表一致） ---- */
#define YOP_CONV      1
#define YOP_ADD       2
#define YOP_CONCAT    3
#define YOP_MAXPOOL   4
#define YOP_UPSAMPLE  5
#define YOP_SIGMOID   6
#define YOP_MUL       7
#define YOP_SLICE     8
#define YOP_SOFTMAX   9
#define YOP_ATTN_PSA  10
#define YOP_DETECT    11
/* ---- 阶段 P2：基于半环公理扩展的通用轻量图算子 ---- */
#define YOP_MATMUL    12            /* 线性代数半环 (+, ×)：矩阵乘（F32 / Q8 per-channel） */
#define YOP_RMSNORM   13            /* Bregman 流形归一化投影：RMSNorm */
#define YOP_ROPE      14            /* 正交旋转群同构：旋转位置编码 RoPE */
#define YOP_SWIGLU    15            /* 门控非线性合成：SwiGLU(gate, up) */
#define YOP_MAX       15

#define YGRAPH_MAGIC  0x50524759u   /* 'YGRP' */
#define YACT_SILU     1

/* ---- P1 精度门：逐层回调接口（编译期启用） ---- */
#ifdef YOLO_PRECISION_GATE_CB
typedef void (*yolo_layer_cb_t)(int op_idx, uint32_t opcode,
    uint32_t out_slot, const float *buf, int c, int h, int w, void *ud);
static yolo_layer_cb_t g_yolo_layer_cb = NULL;
static void *g_yolo_layer_cb_ud = NULL;
void st_yolo_set_layer_cb(yolo_layer_cb_t cb, void *ud) {
    g_yolo_layer_cb = cb; g_yolo_layer_cb_ud = ud;
}
#endif

/* ================================================================
 * 模型结构
 * ================================================================ */
typedef struct {
    char     name[32];
    uint32_t qtype, rows, cols, layout;
    uint64_t off, bytes;
    const uint8_t *ptr;
} YTens;

typedef struct {
    uint32_t op, nin, nparam, out;
    const uint32_t *ins;    /* -> graph blob */
    const uint32_t *prm;
} YOp;

typedef struct {
    size_t len, off;
    int def, first, last, layout;
    int planned_first, planned_last, planned_layout;
} YViewPlan;

struct StYoloModel {
    uint8_t  *blob;
    size_t    size;
    VQFArchDet det;
    int       n_tens;
    YTens    *tens;
    const uint8_t *graph;
    int       n_ops;
    YOp      *ops;
    int       n_slots;
    /* 阶段 P3：静态生命周期区间终点 last_use[slot] 与持久化连续激活 Arena */
    int      *slot_last_use;
    float    *arena;
    size_t    arena_cap;    /* float 元素容量 */
    size_t   *arena_offsets; /* 最近成功的主表示计划；SIZE_MAX 表示未分配 */
    int       arena_ih, arena_iw;
    size_t    arena_plan_hits, arena_plan_builds;
    YViewPlan *views;
    int view_step, view_complete, view_ready;
    size_t arena_main_elems, arena_view_elems;
    /* 卷积/矩阵权重索引（按 cid 直查张量）：F32 或 int8 + per-channel scale */
    const float  **cw;      /* F32 权重（Q8 模型为 NULL） */
    const int8_t **cq;      /* int8 权重（Q8） */
    const float  **cs;      /* per-channel scale [oc]（Q8） */
    const float  **cb;      /* bias（恒 F32；无则 NULL） */
    int          *crow, *ccol;   /* 权重几何（rows=oc, cols=ic*kh*kw） */
    uint32_t     *clayout;       /* VQF_LAYOUT_*（0=legacy, 4=HWC_TAP, 5=IME_K1） */
    int8_t      **cq_chw;        /* 当 clayout!=0 且回退标量 yconv_q8 时的逆向解包缓存 */
    int        max_cid;
    /* CPU dotprod 卷积（K-YOLO-002）缓存：补齐到 Kp 的 int8 权重 [OC][Kp] */
    int8_t     **nw;
    uint8_t     *nw_owned;  /* 阶段 P1：1=堆分配需 free，0=零拷贝直指 m->blob */
    /* V56：IME(vmadot) 路径的权重打包 [mt][kb][mv][32] */
    int8_t     **iw;
    uint8_t     *iw_owned;  /* 阶段 P1：1=堆分配需 free，0=零拷贝直指 m->blob */
    /* V56：IME 的激活打包 scratch（[nb][kb][t][r][8]，按需增长复用） */
    int8_t      *ib;
    size_t       ib_cap;
    /* V56：IME 路径的 rowsc scratch（跨层复用，避免每层 malloc 的首次触页） */
    float       *rsc;
    size_t       rsc_cap;
    /* V50：im2col 输入侧 HWC 化 scratch */
    float      *hx;
    size_t      hx_cap;
    /* 类别名（names blob） */
    const char *names;
    size_t      names_len;
    /* 诊断 */
    double    t_conv, t_detect, t_attn;
    double    t_i2c;             /* im2col 构建耗时（CPU dotprod 累计） */
    long      cpu_convs;
    long      nw_fallback_count; /* P1：nw/scratch 分配失败，降级到 scalar yconv_q8 的累计次数 */
    /* Phase-0（K-YOLO-002）CPU 分解计时（env VLLM_YOLO_CPU_TIMING 门控；默认不累加） */
    double    t_elem, t_lb, t_post, t_fwd;
    long      n_conv_ops, n_elem_ops;
    double    t_opk[YOP_MAX + 1];   /* Phase 2：逐算子类型耗时（index = opcode） */
    long      n_opk[YOP_MAX + 1];
};

/* ================================================================
 * VQF v4 加载器（FOPEN 读整文件 + 结构/FNV 自校验）
 * ================================================================ */
static uint64_t yfnv(const uint64_t *h0, const uint8_t *p, size_t n) {
    uint64_t h = *h0;
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 0x100000001B3ull; }
    return h;
}

static int ytens_find(const StYoloModel *m, const char *name) {
    for (int i = 0; i < m->n_tens; i++)
        if (!strcmp(m->tens[i].name, name)) return i;
    return -1;
}

static const float *ytens_f32(const StYoloModel *m, int idx, size_t *nelem) {
    if (idx < 0) return NULL;
    const YTens *t = &m->tens[idx];
    if (t->qtype == VQF_QT_F32) {
        if (nelem) *nelem = (size_t)t->rows * t->cols;
        return (const float *)t->ptr;
    }
    return NULL;                 /* F16/Q8 在 M1 接入 */
}

StYoloModel *st_yolo_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[YOLO] cannot open %s\n", path); return NULL; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= (long)sizeof(VQFHeader)) { fclose(f); return NULL; }
    StYoloModel *m = (StYoloModel *)calloc(1, sizeof(*m));
    if (!m) { fclose(f); return NULL; }
    m->blob = (uint8_t *)malloc((size_t)sz);
    if (!m->blob || fread(m->blob, 1, (size_t)sz, f) != (size_t)sz) {
        free(m->blob); free(m); fclose(f); return NULL;
    }
    fclose(f);
    m->size = (size_t)sz;

    const VQFHeader *h = (const VQFHeader *)m->blob;
    if (h->magic != VQF_MAGIC) {
        fprintf(stderr, "[YOLO] bad magic (not a VQF file)\n");
        goto fail;
    }
    if (h->version != VQF_VERSION_DET) {
        fprintf(stderr, "[YOLO] version %u is not a DETECTION model (need %u)\n",
                h->version, VQF_VERSION_DET);
        goto fail;
    }
    if (!(h->flags & VQF_FLAG_DET))
        fprintf(stderr, "[YOLO] warning: VQF_FLAG_DET not set\n");
    memcpy(&m->det, &h->arch, sizeof(m->det));

    uint64_t data_off = h->data_offset, file_len = h->file_len;
    size_t tag_len = (h->flags & VQF_FLAG_ENC) ? 32 : 8;
    if (h->flags & VQF_FLAG_ENC) {
        fprintf(stderr, "[YOLO] encrypted VQF not supported yet (M0)\n");
        goto fail;
    }
    if (file_len != 0 && m->size != file_len + tag_len) {
        fprintf(stderr, "[YOLO] size mismatch: file=%zu header=%llu\n",
                m->size, (unsigned long long)file_len);
        goto fail;
    }
    if (data_off < 448 || data_off > m->size) {
        fprintf(stderr, "[YOLO] bad data_offset %llu\n", (unsigned long long)data_off);
        goto fail;
    }
    if (m->size < data_off + 8) goto fail;

    /* FNV-1a（目录 + 数据区）对照尾部 8B —— 与 python 写出侧同规 */
    /* fix 2a: file_len==0 表示"不校验"，跳过整个块；否则 file_len-data_off 会 uint64 下溢 */
    if (file_len != 0) {
        size_t dir_off = 448, dn = (size_t)h->n_tensors * 64;
        if (dir_off + dn > m->size) { fprintf(stderr, "[YOLO] dir overflow\n"); goto fail; }
        uint64_t sum = 0xCBF29CE484222325ull;
        sum = yfnv(&sum, m->blob + dir_off, dn);
        sum = yfnv(&sum, m->blob + data_off, (size_t)(file_len - data_off));
        uint64_t want; memcpy(&want, m->blob + file_len, 8);
        if (sum != want) {
            fprintf(stderr, "[YOLO] FNV mismatch (got %016llx want %016llx) - corrupt?\n",
                    (unsigned long long)sum, (unsigned long long)want);
            goto fail;
        }
    }

    /* 目录 */
    m->n_tens = (int)h->n_tensors;
    m->tens = (YTens *)calloc((size_t)m->n_tens, sizeof(YTens));
    if (!m->tens) goto fail;
    for (int i = 0; i < m->n_tens; i++) {
        const VQFTensor *t = (const VQFTensor *)(m->blob + 448 + (size_t)i * 64);
        YTens *d = &m->tens[i];
        memcpy(d->name, t->name, 32); d->name[31] = 0;
        d->qtype = t->qtype; d->rows = t->rows; d->cols = t->cols;
        d->layout = t->layout;
        d->off = t->offset; d->bytes = t->bytes;
        if (t->offset > m->size || t->bytes > m->size - t->offset) {
            fprintf(stderr, "[YOLO] tensor %s out of range\n", d->name);
            goto fail;
        }
        d->ptr = m->blob + t->offset;
    }

    /* graph */
    {
        int gi = ytens_find(m, "graph");
        if (gi < 0 || m->tens[gi].qtype != VQF_QT_BLOB) {
            fprintf(stderr, "[YOLO] graph tensor missing\n"); goto fail;
        }
        if (m->tens[gi].bytes < 12) { fprintf(stderr, "[YOLO] graph tensor too small\n"); goto fail; }
        m->graph = m->tens[gi].ptr;
        uint32_t magic, gver, nops;
        memcpy(&magic, m->graph, 4); memcpy(&gver, m->graph + 4, 4);
        memcpy(&nops, m->graph + 8, 4);
        if (magic != YGRAPH_MAGIC) { fprintf(stderr, "[YOLO] bad graph magic\n"); goto fail; }
        if (gver != 1) { fprintf(stderr, "[YOLO] graph version %u\n", gver); goto fail; }
        m->n_ops = (int)nops;
        m->ops = (YOp *)calloc((size_t)m->n_ops, sizeof(YOp));
        if (!m->ops && m->n_ops) goto fail;
        const uint8_t *p = m->graph + 12, *end = m->graph + m->tens[gi].bytes;
        int maxslot = 0, maxcid = -1;
        for (int i = 0; i < m->n_ops; i++) {
            if (p + 16 > end) { fprintf(stderr, "[YOLO] graph truncated @%d\n", i); goto fail; }
            YOp *o = &m->ops[i];
            memcpy(&o->op, p, 4); memcpy(&o->nin, p + 4, 4);
            memcpy(&o->nparam, p + 8, 4); memcpy(&o->out, p + 12, 4);
            p += 16;
            /* Arena planning requires one definition per slot. */
            if (o->out == 0) goto fail;
            for (int j = 0; j < i; j++)
                if (m->ops[j].out == o->out) goto fail;
            if (p + 4 * ((size_t)o->nin + o->nparam) > end) {
                fprintf(stderr, "[YOLO] graph op %d overflow\n", i); goto fail;
            }
            o->ins = (const uint32_t *)p; p += 4 * o->nin;
            o->prm = (const uint32_t *)p; p += 4 * o->nparam;
            if ((int)o->out > maxslot) maxslot = (int)o->out;
            for (uint32_t k = 0; k < o->nin; k++)
                if ((int)o->ins[k] > maxslot) maxslot = (int)o->ins[k];
            if ((o->op == YOP_CONV || o->op == YOP_MATMUL || o->op == YOP_RMSNORM) &&
                o->nparam >= 1 && (int)o->prm[0] > maxcid)
                maxcid = (int)o->prm[0];
            if (o->op == YOP_ATTN_PSA && o->nparam >= 6)
                for (int k = 3; k < 6; k++)
                    if ((int)o->prm[k] > maxcid) maxcid = (int)o->prm[k];
        }
        m->n_slots = maxslot + 1;
        /* 阶段 P3：离线/加载期计算每个 slot 的生命周期终点 last_use[s]
         * [SHS 公理映射: set_5foperations_91154c82 (集合不交集公理)] */
        m->slot_last_use = (int *)calloc((size_t)m->n_slots, sizeof(int));
        if (!m->slot_last_use) goto fail;
        for (int s = 0; s < m->n_slots; s++) m->slot_last_use[s] = -1;
        for (int i = 0; i < m->n_ops; i++) {
            const YOp *o = &m->ops[i];
            if ((int)o->out < m->n_slots && m->slot_last_use[o->out] < i)
                m->slot_last_use[o->out] = i;
            for (uint32_t k = 0; k < o->nin; k++) {
                uint32_t in_s = o->ins[k];
                if ((int)in_s < m->n_slots && m->slot_last_use[in_s] < i)
                    m->slot_last_use[in_s] = i;
            }
        }
        /* 最终输出槽位保持存活至全图末尾 */
        if (m->n_ops > 0) {
            uint32_t last_out = m->ops[m->n_ops - 1].out;
            for (int i = 0; i < m->n_ops; i++) {
                uint32_t op = m->ops[i].op;
                if (op == YOP_DETECT || op == YOP_SOFTMAX || op == YOP_MATMUL ||
                    op == YOP_RMSNORM || op == YOP_ROPE || op == YOP_SWIGLU)
                    last_out = m->ops[i].out;
            }
            if ((int)last_out < m->n_slots)
                m->slot_last_use[last_out] = m->n_ops;
        }
        m->max_cid = maxcid;
        m->cw = (const float **)calloc((size_t)maxcid + 1, sizeof(float *));
        m->cq = (const int8_t **)calloc((size_t)maxcid + 1, sizeof(int8_t *));
        m->cs = (const float **)calloc((size_t)maxcid + 1, sizeof(float *));
        m->cb = (const float **)calloc((size_t)maxcid + 1, sizeof(float *));
        m->crow    = (int *)calloc((size_t)maxcid + 1, sizeof(int));
        m->ccol    = (int *)calloc((size_t)maxcid + 1, sizeof(int));
        m->clayout = (uint32_t *)calloc((size_t)maxcid + 1, sizeof(uint32_t));
        m->cq_chw  = (int8_t **)calloc((size_t)maxcid + 1, sizeof(int8_t *));
        m->nw       = (int8_t **)calloc((size_t)maxcid + 1, sizeof(int8_t *));
        m->nw_owned = (uint8_t *)calloc((size_t)maxcid + 1, sizeof(uint8_t));
        m->iw       = (int8_t **)calloc((size_t)maxcid + 1, sizeof(int8_t *));
        m->iw_owned = (uint8_t *)calloc((size_t)maxcid + 1, sizeof(uint8_t));
        if (maxcid >= 0 && (!m->cw || !m->cq || !m->cs || !m->cb ||
            !m->crow || !m->ccol || !m->clayout || !m->cq_chw ||
            !m->nw || !m->nw_owned || !m->iw || !m->iw_owned)) goto fail;
        int n_prepack_hwc = 0, n_prepack_ime = 0;
        for (int c = 0; c <= maxcid; c++) {
            char nm[32];
            snprintf(nm, sizeof(nm), "c%04d_w", c);
            int wi = ytens_find(m, nm);
            if (wi >= 0) {
                m->crow[c]    = (int)m->tens[wi].rows;
                m->ccol[c]    = (int)m->tens[wi].cols;
                m->clayout[c] = m->tens[wi].layout;
                if (m->tens[wi].qtype == VQF_QT_F32)
                    m->cw[c] = (const float *)m->tens[wi].ptr;
                else if (m->tens[wi].qtype == VQF_QT_Q8_0) {
                    m->cq[c] = (const int8_t *)m->tens[wi].ptr;
                    /* 阶段 P1：离线预打包零拷贝直挂
                     * layout == VQF_LAYOUT_CONV_HWC_TAP(4): ptr 已是 [OC][Kp] tap-major
                     * layout == VQF_LAYOUT_CONV_IME_K1(5):  ptr 已是 [mt][kb][mv][32] IME pack_a */
                    if (m->clayout[c] == VQF_LAYOUT_CONV_HWC_TAP) {
                        n_prepack_hwc++;
                    } else if (m->clayout[c] == VQF_LAYOUT_CONV_IME_K1) {
                        m->iw[c] = (int8_t *)m->tens[wi].ptr;
                        m->iw_owned[c] = 0;
                        n_prepack_ime++;
                    }
                } else
                    fprintf(stderr, "[YOLO] conv %s qtype %u unsupported (M1: F32/Q8)\n",
                            nm, m->tens[wi].qtype);
            }
            snprintf(nm, sizeof(nm), "c%04d_s", c);
            m->cs[c] = ytens_f32(m, ytens_find(m, nm), NULL);
            snprintf(nm, sizeof(nm), "c%04d_b", c);
            m->cb[c] = ytens_f32(m, ytens_find(m, nm), NULL);
        }
        {
            int nq8 = 0, nf32 = 0;
            for (int c = 0; c <= maxcid; c++) { if (m->cq[c]) nq8++; else if (m->cw[c]) nf32++; }
            if (nq8) fprintf(stderr, "[YOLO] conv weights: F32=%d int8=%d (per-channel scale, prepack hwc=%d ime=%d)\n",
                             nf32, nq8, n_prepack_hwc, n_prepack_ime);
        }
    }

    /* names */
    {
        int ni = ytens_find(m, "names");
        if (ni >= 0) { m->names = (const char *)m->tens[ni].ptr; m->names_len = m->tens[ni].bytes; }
    }

    fprintf(stderr, "[YOLO] loaded %s: family=%u nc=%u reg_max=%u in=%ux%u "
                    "ops=%d slots=%d tensors=%d\n",
            path, m->det.family, m->det.n_classes, m->det.reg_max,
            m->det.input_w, m->det.input_h, m->n_ops, m->n_slots, m->n_tens);
    return m;
fail:
    st_yolo_free(m);
    return NULL;
}

void st_yolo_free(StYoloModel *m) {
    if (!m) return;
    free(m->tens); free(m->ops);
    free(m->slot_last_use);
    free(m->arena);
    free(m->arena_offsets);
    free(m->views);
    free(m->cw); free(m->cq); free(m->cs); free(m->cb);
    free(m->crow); free(m->ccol); free(m->clayout);
    if (m->cq_chw) {
        int nc = m->max_cid + 1;
        for (int c = 0; c < nc; c++) free(m->cq_chw[c]);
        free(m->cq_chw);
    }
    if (m->nw) {
        int nc = m->max_cid + 1;
        for (int c = 0; c < nc; c++)
            if (m->nw_owned && m->nw_owned[c]) free(m->nw[c]);
        free(m->nw);
    }
    free(m->nw_owned);
    if (m->iw) {
        int nc = m->max_cid + 1;
        for (int c = 0; c < nc; c++)
            if (m->iw_owned && m->iw_owned[c]) free(m->iw[c]);
        free(m->iw);
    }
    free(m->iw_owned);
    free(m->ib);
    free(m->rsc);
    free(m->hx);
    free(m->blob);
    free(m);
}

int st_yolo_input_size(const StYoloModel *m, int *w, int *h) {
    if (!m) return -1;
    if (w) *w = (int)m->det.input_w;
    if (h) *h = (int)m->det.input_h;
    return 0;
}
int st_yolo_num_classes(const StYoloModel *m) { return m ? (int)m->det.n_classes : 0; }
int st_yolo_family(const StYoloModel *m) { return m ? (int)m->det.family : -1; }

int st_yolo_class_name(const StYoloModel *m, int idx, char *buf, int buflen) {
    if (!m || !m->names || idx < 0 || !buf || buflen <= 0) return -1;
    const char *p = m->names;
    for (int i = 0; i < idx; i++) {
        const char *nl = memchr(p, '\n', m->names_len - (size_t)(p - m->names));
        if (!nl) return -1;
        p = nl + 1;
    }
    const char *nl = memchr(p, '\n', m->names_len - (size_t)(p - m->names));
    size_t n = nl ? (size_t)(nl - p) : strlen(p);
    if ((int)n >= buflen) n = (size_t)buflen - 1;
    memcpy(buf, p, n); buf[n] = 0;
    return 0;
}

/* ================================================================
 * 算子
 * ================================================================ */
static void yconv(const float *x, int C, int H, int W,
                  const float *w, const float *b, int OC, int KH, int KW,
                  int sh, int sw, int ph, int pw, int G, int act,
                  float *y, int OH, int OW) {
    const int icg = C / G, ocg = OC / G;
    if (KH == 1 && KW == 1 && sh == 1 && sw == 1 && ph == 0 && pw == 0) {
        const int P = H * W;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if((long)OC * P > 262144)
#endif
        for (int oc = 0; oc < OC; oc++) {
            int g = oc / ocg, ic0 = g * icg;
            const float *wo = w + (size_t)oc * icg;
            float bias = b ? b[oc] : 0.f;
            float *yo = y + (size_t)oc * P;
            for (int q = 0; q < P; q++) {
                float acc = bias;
                for (int icl = 0; icl < icg; icl++)
                    acc += wo[icl] * x[(size_t)(ic0 + icl) * P + q];
                if (act == YACT_SILU) acc = acc / (1.f + expf(-acc));
                yo[q] = acc;
            }
        }
        return;
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(OC > 16)
#endif
    for (int oc = 0; oc < OC; oc++) {
        int g = oc / ocg, ic0 = g * icg;
        const float *wo = w + (size_t)oc * icg * KH * KW;
        float bias = b ? b[oc] : 0.f;
        for (int oy = 0; oy < OH; oy++) {
            for (int ox = 0; ox < OW; ox++) {
                float acc = bias;
                for (int icl = 0; icl < icg; icl++) {
                    const float *xc = x + (size_t)(ic0 + icl) * H * W;
                    const float *wk = wo + (size_t)icl * KH * KW;
                    for (int ky = 0; ky < KH; ky++) {
                        int iy = oy * sh - ph + ky;
                        if (iy < 0 || iy >= H) continue;
                        for (int kx = 0; kx < KW; kx++) {
                            int ix = ox * sw - pw + kx;
                            if (ix < 0 || ix >= W) continue;
                            acc += wk[ky * KW + kx] * xc[iy * W + ix];
                        }
                    }
                }
                if (act == YACT_SILU) acc = acc / (1.f + expf(-acc));
                y[((size_t)oc * OH + oy) * OW + ox] = acc;
            }
        }
    }
}

/* int8 权重卷积（per-output-channel scale；激活与 F32 路径同式）。
 * 量化坍缩公理：collapse_to_classic + collapse_priority_rule（细粒度 per-channel scale）。
 * 数值口径（勿按「int32 累加」理解）：本函数是 **int8 权重 × float 激活、
 * float 累加**，收尾为 acc = (Σ (float)q*x) * sc + bias。这与
 * tools/yolo/yolo_vqf_run.py 的参考执行器不同 —— 后者先把 int8 权重反量化
 * 成 f32（q*sc），再做 f32 einsum（Σ (q*sc)*x）。两者是同一量化模型的不同
 * 舍入次序，数值近似而非位级等价；板端 Q8 门的分类分数 max|d|≈8.8e-3
 * 就来自这里。int8×int8→int32 的累加见于 ARM 专用路径 yconv_cpu_dotprod
 * （NEON dotprod），与 RISC-V 无关。 */
static void yconv_q8(const float *x, int C, int H, int W,
                     const int8_t *wq, const float *wsc, const float *b,
                     int OC, int KH, int KW, int sh, int sw, int ph, int pw,
                     int G, int act, float *y, int OH, int OW) {
    const int icg = C / G, ocg = OC / G;
    if (KH == 1 && KW == 1 && sh == 1 && sw == 1 && ph == 0 && pw == 0) {
        const int P = H * W;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if((long)OC * P > 262144)
#endif
        for (int oc = 0; oc < OC; oc++) {
            int g = oc / ocg, ic0 = g * icg;
            const int8_t *wo = wq + (size_t)oc * icg;
            float sc = wsc ? wsc[oc] : 1.f;
            float bias = b ? b[oc] : 0.f;
            float *yo = y + (size_t)oc * P;
            for (int q = 0; q < P; q++) {
                float acc = 0.f;
                for (int icl = 0; icl < icg; icl++)
                    acc += (float)wo[icl] * x[(size_t)(ic0 + icl) * P + q];
                acc = acc * sc + bias;
                if (act == YACT_SILU) acc = acc / (1.f + expf(-acc));
                yo[q] = acc;
            }
        }
        return;
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(OC > 16)
#endif
    for (int oc = 0; oc < OC; oc++) {
        int g = oc / ocg, ic0 = g * icg;
        const int8_t *wo = wq + (size_t)oc * icg * KH * KW;
        float sc = wsc ? wsc[oc] : 1.f;
        float bias = b ? b[oc] : 0.f;
        for (int oy = 0; oy < OH; oy++) {
            for (int ox = 0; ox < OW; ox++) {
                float acc = 0.f;
                for (int icl = 0; icl < icg; icl++) {
                    const float *xc = x + (size_t)(ic0 + icl) * H * W;
                    const int8_t *wk = wo + (size_t)icl * KH * KW;
                    for (int ky = 0; ky < KH; ky++) {
                        int iy = oy * sh - ph + ky;
                        if (iy < 0 || iy >= H) continue;
                        for (int kx = 0; kx < KW; kx++) {
                            int ix = ox * sw - pw + kx;
                            if (ix < 0 || ix >= W) continue;
                            acc += (float)wk[ky * KW + kx] * xc[iy * W + ix];
                        }
                    }
                }
                acc = acc * sc + bias;
                if (act == YACT_SILU) acc = acc / (1.f + expf(-acc));
                y[((size_t)oc * OH + oy) * OW + ox] = acc;
            }
        }
    }
}

#if ST_HAVE_RVV
/* Opt-in CHW Q8 depthwise: preserve the scalar tap and epilogue order. */
static void yconv_dw_q8_rvv(const float *x, int C, int H, int W,
                           const int8_t *wq, const float *wsc, const float *b,
                           float *y) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(C > 16)
#endif
    for (int oc = 0; oc < C; oc++) {
        const float *xc = x + (size_t)oc * H * W;
        const int8_t *wo = wq + (size_t)oc * 9;
        float sc = wsc ? wsc[oc] : 1.f;
        float bias = b ? b[oc] : 0.f;
        float *yo = y + (size_t)oc * H * W;
        for (int oy = 0; oy < H; oy++) {
            for (int ox = 0; ox < W; ox++) {
                if (oy >= 1 && oy < H - 1 && ox >= 1 && ox < W - 1) {
                    size_t vl = __riscv_vsetvl_e32m1(W - 1 - ox);
                    vfloat32m1_t va = __riscv_vfmv_v_f_f32m1(0.f, vl);
                    for (int ky = 0; ky < 3; ky++) {
                        for (int kx = 0; kx < 3; kx++) {
                            vfloat32m1_t vx = __riscv_vle32_v_f32m1(
                                xc + (oy - 1 + ky) * W + ox - 1 + kx, vl);
                            vfloat32m1_t vp = __riscv_vfmul_vf_f32m1(vx, (float)wo[ky * 3 + kx], vl);
                            va = __riscv_vfadd_vv_f32m1(va, vp, vl);
                        }
                    }
                    float tmp[vl];
                    __riscv_vse32_v_f32m1(tmp, va, vl);
                    for (size_t j = 0; j < vl; j++) {
                        float acc = tmp[j] * sc + bias;
                        yo[(size_t)oy * W + ox + j] = acc / (1.f + expf(-acc));
                    }
                    ox += (int)vl - 1;
                    continue;
                }
                float acc = 0.f;
                for (int ky = 0; ky < 3; ky++) {
                    int iy = oy - 1 + ky;
                    if (iy < 0 || iy >= H) continue;
                    for (int kx = 0; kx < 3; kx++) {
                        int ix = ox - 1 + kx;
                        if (ix < 0 || ix >= W) continue;
                        acc += (float)wo[ky * 3 + kx] * xc[iy * W + ix];
                    }
                }
                acc = acc * sc + bias;
                yo[(size_t)oy * W + ox] = acc / (1.f + expf(-acc));
            }
        }
    }
}
#endif

/* ================================================================
 * V50：im2col 输入侧 HWC 化（唯一默认路径）
 *
 * 先把该层输入按 8×8 分块转成 HWC（buf[p*C+c]），再 im2col。
 * HWC 下固定 (ky,kx) 的 C 个通道是连续的，im2col 退化为 9 段连续拷贝
 * + vfmax，全程 vle32/vse32 无 gather/scatter。
 * k 排列：HWC 下 k = (ky*KW+kx)*C + ic（tap-major）。
 * ================================================================ */

/* CHW [C][H*W] → HWC [H*W][C]，8×8 分块：目的侧连续写、源侧 8 条流，
 * 使两侧都落在少数 cache 行内（朴素逐元素转置会两边都打穿）。 */
static void yolo_chw2hwc_blocked(const float *src, float *dst, int C, int H, int W) {
    const int P = H * W;
    const int TB = 8;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) collapse(2) if((long)C * P > 65536)
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

/* im2col（HWC 输入）→ int8，k = tap*C + ic。
 * 逐像素对 9 个 tap 各拷 C 个连续 float（vle32→vse32），同时求 |·| 最大值。
 * packed != NULL 时不落 imc，而是把量化结果直接按 IME 的 B 布局散写
 * （bpack[nb][kb][t][r][8]）。 */
static int yolo_im2col_q8_hwc(const float *x, int C, int H, int W,
                              int KH, int KW, int sh, int sw, int ph, int pw,
                              int OH, int OW, int K, int Kp,
                              int8_t *imc, float *rowsc, int8_t *packed) {
    if (K != C * KH * KW) return -1;
    const int kb_n = Kp / 8;
#ifdef _OPENMP
    int nth = omp_get_max_threads(); if (nth < 1) nth = 1;
#else
    int nth = 1;
#endif
    float **rb = (float **)calloc((size_t)nth, sizeof(float *));
    int8_t **qb = packed ? (int8_t **)calloc((size_t)nth, sizeof(int8_t *)) : NULL;
    if (!rb || (packed && !qb)) { free(rb); free(qb); return -1; }
    for (int i = 0; i < nth; i++) {
        rb[i] = (float *)malloc((size_t)K * sizeof(float));
        if (!rb[i]) {
            for (int j = 0; j < i; j++) free(rb[j]);
            free(rb);
            if (packed) { for (int j = 0; j < i; j++) free(qb[j]); }  /* fix: release already-allocated qb[j] */
            free(qb);
            return -1;
        }
        if (packed) {
            qb[i] = (int8_t *)malloc((size_t)Kp);
            if (!qb[i]) {
                free(rb[i]);
                for (int j = 0; j < i; j++) { free(rb[j]); free(qb[j]); }
                free(rb); free(qb); return -1;
            }
        }
    }
#if !ST_HAVE_RVV
    const float quant_magic = 12582912.0f;   /* 2^23 */
#endif

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if((long)OH * OW > 1024)
#endif
    for (int oy = 0; oy < OH; oy++) {
#ifdef _OPENMP
        float *row = rb[omp_get_thread_num()];
        int8_t *qrow = packed ? qb[omp_get_thread_num()] : NULL;
#else
        float *row = rb[0];
        int8_t *qrow = packed ? qb[0] : NULL;
#endif
        for (int ox = 0; ox < OW; ox++) {
            const int pp = oy * OW + ox;
            int8_t *arow = packed ? qrow : (imc + (size_t)pp * Kp);
            const int iy0 = oy * sh - ph, ix0 = ox * sw - pw;
            float mx = 0.f;
            for (int ky = 0; ky < KH; ky++) {
                const int iy = iy0 + ky;
                for (int kx = 0; kx < KW; kx++) {
                    const int tap = ky * KW + kx;
                    float *dst = row + (size_t)tap * C;
                    const int ix = ix0 + kx;
                    if (iy < 0 || iy >= H || ix < 0 || ix >= W) {
                        memset(dst, 0, (size_t)C * sizeof(float));
                        continue;
                    }
                    const float *src = x + ((size_t)iy * W + ix) * C;
                    float t = 0.f;
#if ST_HAVE_RVV
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
                    t = __riscv_vfmv_f_s_f32m1_f32(
                            __riscv_vfredmax_vs_f32m1_f32m1(
                                vm1, __riscv_vfmv_v_f_f32m1(0.0f, vmaxf), lastvl));
#else
                    for (int ic = 0; ic < C; ic++) {
                        float v = src[ic];
                        dst[ic] = v;
                        float av = fabsf(v); if (av > t) t = av;
                    }
#endif
                    if (t > mx) mx = t;
                }
            }
            const float s = (mx > 0.f) ? (mx / 127.f) : 1.f;
            const float inv = 1.f / s;
            if (Kp > K) memset(arow + K, 0, (size_t)(Kp - K));
#if ST_HAVE_RVV
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
#pragma omp simd
            for (int j = 0; j < K; j++) {
                int q = (int)(row[j] * inv + quant_magic) - 12582912;
                if (q > 127) q = 127; else if (q < -127) q = -127;
                arow[j] = (int8_t)q;
            }
#endif
            if (packed) {
                int8_t *dst = packed + (size_t)(pp >> 4) * kb_n * 128 +
                              ((pp >> 2) & 3) * 32 + (pp & 3) * 8;
                for (int kb = 0; kb < kb_n; kb++)
                    memcpy(dst + (size_t)kb * 128, arow + kb * 8, 8);
            }
            rowsc[pp] = s;
        }
    }
    for (int i = 0; i < nth; i++) { free(rb[i]); if (packed) free(qb[i]); }
    free(rb); free(qb);
    return 0;
}

/* ================================================================
 * PC 侧 int8 dotprod 卷积（K-YOLO-002 Phase 1）
 *
 * 复用 yolo_im2col_q8_row 的 im2col int8 + 逐像素 scale，
 * 权重用同一份补齐到 Kp 的 int8 重排 m->nw[cid]=[OC][Kp]。GEMM 转置取向：
 *   out[oc][p] = Σ_k W[oc][k]·im2col[p][k]
 * 整数域在 ARM 用 vdotq_s32（ARMv8.2 dotprod，A76）；**其他平台（x86/RISC-V）
 * 走同函数的 `#else` 标量 int8×int8 累加**，不是 SIMD，但 int8 权重带宽 + 补齐到
 * Kp 的预打包仍带来大幅收益 —— RISC-V 板端实测（V40）：本路径 conv 5016ms vs
 * 标量 yconv_q8 20936ms = **4.2×**，vs F32 模型 24585ms = **4.8×**。
 * per-(oc,位置) 反量化 = cs[oc]·rowsc[p]。
 * 数学等价性：per-block 反量化中权重/激活 scale 逐块相乘 ⇒ 合成 scale
 *   cs[oc]·rowsc[p] 与「先整段 int32 点积、再一次乘 scale」**数值等价**（int32 精确）。
 * 返回 1 = 已处理；0 = 不适用（调用方回退标量 yconv_q8，J-CPU-4 显式）。 */

#if ST_HAVE_RVV
/* ---- 向量化 SiLU 快速 exp（Cody-Waite 两段 ln2 归约 + 7 阶 Taylor，唯一默认） ---- */
static inline vfloat32m4_t yolo_vexp_f32m4(vfloat32m4_t x, size_t vl) {
    const float LOG2E = 1.4426950408889634f;
    const float LN2HI = 6.93145751953125e-1f;    /* ln2 高位（尾数低 12 位为 0） */
    const float LN2LO = 1.4286067653301870e-6f;  /* ln2 - LN2HI */
    x = __riscv_vfmin_vv_f32m4(x, __riscv_vfmv_v_f_f32m4(88.0f, vl), vl);
    x = __riscv_vfmax_vv_f32m4(x, __riscv_vfmv_v_f_f32m4(-87.0f, vl), vl);
    vint32m4_t n = __riscv_vfcvt_x_f_v_i32m4(__riscv_vfmul_vf_f32m4(x, LOG2E, vl), vl);
    vfloat32m4_t nf = __riscv_vfcvt_f_x_v_f32m4(n, vl);
    vfloat32m4_t r = __riscv_vfnmsub_vf_f32m4(nf, LN2HI, x, vl);
    r = __riscv_vfnmsub_vf_f32m4(nf, LN2LO, r, vl);
    vfloat32m4_t p = __riscv_vfmv_v_f_f32m4(1.0f / 5040.0f, vl);
    p = __riscv_vfmadd_vv_f32m4(p, r, __riscv_vfmv_v_f_f32m4(1.0f / 720.0f, vl), vl);
    p = __riscv_vfmadd_vv_f32m4(p, r, __riscv_vfmv_v_f_f32m4(1.0f / 120.0f, vl), vl);
    p = __riscv_vfmadd_vv_f32m4(p, r, __riscv_vfmv_v_f_f32m4(1.0f / 24.0f, vl), vl);
    p = __riscv_vfmadd_vv_f32m4(p, r, __riscv_vfmv_v_f_f32m4(1.0f / 6.0f, vl), vl);
    p = __riscv_vfmadd_vv_f32m4(p, r, __riscv_vfmv_v_f_f32m4(0.5f, vl), vl);
    p = __riscv_vfmadd_vv_f32m4(p, r, __riscv_vfmv_v_f_f32m4(1.0f, vl), vl);
    p = __riscv_vfmadd_vv_f32m4(p, r, __riscv_vfmv_v_f_f32m4(1.0f, vl), vl);
    const vint32m4_t e = __riscv_vsll_vx_i32m4(
                             __riscv_vadd_vx_i32m4(n, 127, vl), 23, vl);
    return __riscv_vfmul_vv_f32m4(p, __riscv_vreinterpret_v_i32m4_f32m4(e), vl);
}

static inline vfloat32m2_t yolo_vexp_f32m2(vfloat32m2_t x, size_t vl) {
    const float LOG2E = 1.4426950408889634f;
    const float LN2HI = 6.93145751953125e-1f;
    const float LN2LO = 1.4286067653301870e-6f;
    x = __riscv_vfmin_vv_f32m2(x, __riscv_vfmv_v_f_f32m2(88.0f, vl), vl);
    x = __riscv_vfmax_vv_f32m2(x, __riscv_vfmv_v_f_f32m2(-87.0f, vl), vl);
    vint32m2_t n = __riscv_vfcvt_x_f_v_i32m2(__riscv_vfmul_vf_f32m2(x, LOG2E, vl), vl);
    vfloat32m2_t nf = __riscv_vfcvt_f_x_v_f32m2(n, vl);
    vfloat32m2_t r = __riscv_vfnmsub_vf_f32m2(nf, LN2HI, x, vl);
    r = __riscv_vfnmsub_vf_f32m2(nf, LN2LO, r, vl);
    vfloat32m2_t p = __riscv_vfmv_v_f_f32m2(1.0f / 5040.0f, vl);
    p = __riscv_vfmadd_vv_f32m2(p, r, __riscv_vfmv_v_f_f32m2(1.0f / 720.0f, vl), vl);
    p = __riscv_vfmadd_vv_f32m2(p, r, __riscv_vfmv_v_f_f32m2(1.0f / 120.0f, vl), vl);
    p = __riscv_vfmadd_vv_f32m2(p, r, __riscv_vfmv_v_f_f32m2(1.0f / 24.0f, vl), vl);
    p = __riscv_vfmadd_vv_f32m2(p, r, __riscv_vfmv_v_f_f32m2(1.0f / 6.0f, vl), vl);
    p = __riscv_vfmadd_vv_f32m2(p, r, __riscv_vfmv_v_f_f32m2(0.5f, vl), vl);
    p = __riscv_vfmadd_vv_f32m2(p, r, __riscv_vfmv_v_f_f32m2(1.0f, vl), vl);
    p = __riscv_vfmadd_vv_f32m2(p, r, __riscv_vfmv_v_f_f32m2(1.0f, vl), vl);
    const vint32m2_t e = __riscv_vsll_vx_i32m2(
                             __riscv_vadd_vx_i32m2(n, 127, vl), 23, vl);
    return __riscv_vfmul_vv_f32m2(p, __riscv_vreinterpret_v_i32m2_f32m2(e), vl);
}

/* RISC-V RVV：p 方向向量化 int8 GEMM（无归约，OC 非 8 倍数或非 IME 时的最优 RVV 内核）。 */
static void yconv_dot_rvv_p(const int8_t *Wq, const float *cs, const float *b,
                            const int8_t *imc, const float *rowsc,
                            int OC4, int Kp, int P, int act, float *y) {
    const size_t vmax = __riscv_vsetvlmax_e8m1();   /* VLEN/8 */
    const long nb = (long)OC4 / 4;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(nb > 2)
#endif
    for (long blk = 0; blk < nb; blk++) {
        const int oc0 = (int)blk * 4;
        const int8_t *w0 = Wq + (size_t)(oc0 + 0) * Kp;
        const int8_t *w1 = Wq + (size_t)(oc0 + 1) * Kp;
        const int8_t *w2 = Wq + (size_t)(oc0 + 2) * Kp;
        const int8_t *w3 = Wq + (size_t)(oc0 + 3) * Kp;
        const float s0 = cs[oc0 + 0], s1 = cs[oc0 + 1];
        const float s2 = cs[oc0 + 2], s3 = cs[oc0 + 3];
        const float b0 = b ? b[oc0 + 0] : 0.f, b1 = b ? b[oc0 + 1] : 0.f;
        const float b2 = b ? b[oc0 + 2] : 0.f, b3 = b ? b[oc0 + 3] : 0.f;
        float *y0 = y + (size_t)(oc0 + 0) * P;
        float *y1 = y + (size_t)(oc0 + 1) * P;
        float *y2 = y + (size_t)(oc0 + 2) * P;
        float *y3 = y + (size_t)(oc0 + 3) * P;

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
            const vfloat32m4_t rs = __riscv_vle32_v_f32m4(rowsc + p0, vmax);
#define YOLO_RVV_EPI(ACC, DSTP, SV, BV) do {                                  \
                vfloat32m4_t f_ = __riscv_vfmul_vv_f32m4(                     \
                                      __riscv_vfcvt_f_x_v_f32m4(ACC, vmax),   \
                                      __riscv_vfmul_vf_f32m4(rs, SV, vmax),   \
                                      vmax);                                  \
                f_ = __riscv_vfadd_vf_f32m4(f_, BV, vmax);                    \
                if (act == YACT_SILU) {                                       \
                    vfloat32m4_t e_ = yolo_vexp_f32m4(                        \
                                          __riscv_vfneg_v_f32m4(f_, vmax), vmax); \
                    f_ = __riscv_vfdiv_vv_f32m4(                              \
                             f_, __riscv_vfadd_vf_f32m4(e_, 1.0f, vmax), vmax); \
                }                                                             \
                __riscv_vse32_v_f32m4(DSTP, f_, vmax);                        \
            } while (0)
            YOLO_RVV_EPI(a0, y0 + p0, s0, b0);
            YOLO_RVV_EPI(a1, y1 + p0, s1, b1);
            YOLO_RVV_EPI(a2, y2 + p0, s2, b2);
            YOLO_RVV_EPI(a3, y3 + p0, s3, b3);
#undef YOLO_RVV_EPI
        }
        /* p 尾部（< VLMAX）：逐位置标量 */
        for (; p0 < P; p0++) {
            const int8_t *ir = imc + (size_t)p0 * Kp;
            float *dst[4] = { y0, y1, y2, y3 };
            const int8_t *wp[4] = { w0, w1, w2, w3 };
            const float sv[4] = { s0, s1, s2, s3 };
            const float bv[4] = { b0, b1, b2, b3 };
            for (int u = 0; u < 4; u++) {
                int32_t t = 0;
                for (int k = 0; k < Kp; k++) t += (int32_t)wp[u][k] * (int32_t)ir[k];
                float r = (float)t * (sv[u] * rowsc[p0]) + bv[u];
                if (act == YACT_SILU) r = r / (1.f + expf(-r));
                dst[u][p0] = r;
            }
        }
    }
}

/* ================================================================
 * V56：IME（vmadot）卷积路径 —— SpacemiT K1 的 s8×s8 外积单元（自动探测启用）
 *
 * 数据布局：
 *   A（权重）[mt][kb][mv][32]：byte[q*8+j] = W[mt*8+mv*4+q][kb*8+j]
 *   B（激活）[nb][kb][t][32] ：byte[r*8+j] = imc[nb*16+t*4+r][kb*8+j]
 *
 * 硬约束：vmadot 只在 hart0-3 可用 ⇒ 必须绑核并在退出时还原。
 * ================================================================ */
#define YIME_STR(x) #x
#define YIME_VMADOT(vd, vs1, vs2)                                                \
    ".insn 4, " YIME_STR(0x2b | ((vd) << 7) | (3 << 12) | ((vs1) << 15) |        \
                         ((vs2) << 20) | (0x71 << 25)) "\n\t"

static int yolo_ime_on(void) {
    static int g = -1;
#ifdef _OPENMP
    if (g < 0) {
        g = 0;
        FILE *fp = fopen("/proc/cpuinfo", "r");
        if (fp) {
            char line[512];
            while (fgets(line, sizeof(line), fp)) {
                if (strstr(line, "_ime") || strstr(line, "spacemit") || strstr(line, "SpacemiT")) {
                    g = 1;
                    break;
                }
            }
            fclose(fp);
        }
    }
    return g;
#else
    return 0;   /* 绑核依赖 OpenMP 线程池；无 OMP 的构建不启用该路径 */
#endif
}

static void yime_pin_hart(int h) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(h & 3, &set);   /* 只有 hart0-3 带 ime 扩展 */
    sched_setaffinity(0, sizeof(set), &set);
}

/* V56 诊断：VLLM_YOLO_IME_DEBUG=1 时逐层打印内核+epilogue 耗时 */
static double yime_acc_conv = 0.0;
static int yime_timing;
static double yime_kernel_sum, yime_epi_sum, yime_wall;
static int yime_dbg(void) {
    static int g = -1;
    if (g < 0) { const char *e = getenv("VLLM_YOLO_IME_DEBUG"); g = (e && e[0] == '1'); }
    return g;
}

/* 8(M) × 16(N) × Kp 微内核：acc[mv*4+t][q*4+r] */
static void yime_kernel(const int8_t *a, const int8_t *b, int kb_n, int32_t acc[8][16]) {
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

/* 权重打包（每层一次，惰性） */
static int8_t *yime_pack_a(const int8_t *Wq, int OC, int Kp) {
    const int mt_n = OC / 8, kb_n = Kp / 8;
    int8_t *p = (int8_t *)calloc((size_t)mt_n * kb_n * 64, 1);
    if (!p) return NULL;
    for (int mt = 0; mt < mt_n; mt++)
        for (int kb = 0; kb < kb_n; kb++)
            for (int mv = 0; mv < 2; mv++)
                for (int q = 0; q < 4; q++)
                    memcpy(p + ((size_t)mt * kb_n + kb) * 64 + mv * 32 + q * 8,
                           Wq + (size_t)(mt * 8 + mv * 4 + q) * Kp + kb * 8, 8);
    return p;
}

/* 整层：nb 方向并行（绑 hart0-3），epilogue 与生产逐字相同 */
static void yconv_ime(const int8_t *apack, const int8_t *bpack,
                      const float *cs, const float *b, const float *rowsc,
                      int OC, int Kp, int P, int act, float *y, int out_hwc) {
    const int mt_n = OC / 8, kb_n = Kp / 8, nb_n = (P + 15) / 16;
    double kt[4] = {0}, et[4] = {0};
    const double wall0 = yime_timing ? ynow() : 0.0;
    int nt = 4;
#ifdef _OPENMP
    { int v = omp_get_max_threads(); if (v < nt) nt = v; }
#endif

#ifdef _OPENMP
#pragma omp parallel num_threads(nt) if(nb_n > 8)
    {
        /* 绑核只在区内生效：区外必须还原，否则 OMP 线程池被永久钉在 hart0-3，
         * 后续所有算子都退化成 4 线程（实测 elem 42.5→159.9ms）。 */
        cpu_set_t old;
        const int have_old = (sched_getaffinity(0, sizeof(old), &old) == 0);
        yime_pin_hart(omp_get_thread_num());
#pragma omp for schedule(static)
#endif
        for (int nb = 0; nb < nb_n; nb++) {
            double kernel_local = 0.0, epi_local = 0.0;
            const int px0 = nb * 16;
            const size_t left = (size_t)(P - px0);
            const size_t npx = left < 16 ? left : 16;
            const int8_t *bp = bpack + (size_t)nb * kb_n * 128;
            for (int mt = 0; mt < mt_n; mt++) {
                int32_t acc[8][16];
                double t0 = yime_timing ? ynow() : 0.0;
                yime_kernel(apack + (size_t)mt * kb_n * 64, bp, kb_n, acc);
                double t1 = yime_timing ? ynow() : 0.0;
                if (yime_timing) kernel_local += t1 - t0;
                for (int mv = 0; mv < 2; mv++) {
                    for (int q = 0; q < 4; q++) {
                        const int oc = mt * 8 + mv * 4 + q;
                        /* 4 个 t × 4 个 r = 16 个连续像素，拼成一条 m2 向量 */
                        /* vslideup 的 vl 是「结果里被处理到的下标上界」，不是搬运元素数：
                         * 要把 4 个元素搬到 [off, off+4) 必须传 vl = off+4。 */
                        vint32m2_t v = __riscv_vle32_v_i32m2(acc[mv * 4 + 0] + q * 4, 4);
                        v = __riscv_vslideup_vx_i32m2(
                                v, __riscv_vle32_v_i32m2(acc[mv * 4 + 1] + q * 4, 4), 4, 8);
                        v = __riscv_vslideup_vx_i32m2(
                                v, __riscv_vle32_v_i32m2(acc[mv * 4 + 2] + q * 4, 4), 8, 12);
                        v = __riscv_vslideup_vx_i32m2(
                                v, __riscv_vle32_v_i32m2(acc[mv * 4 + 3] + q * 4, 4), 12, 16);
                        const size_t vl = __riscv_vsetvl_e32m2(npx);
                        vfloat32m2_t f = __riscv_vfmul_vv_f32m2(
                            __riscv_vfcvt_f_x_v_f32m2(v, vl),
                            __riscv_vfmul_vf_f32m2(
                                __riscv_vle32_v_f32m2(rowsc + px0, vl), cs[oc], vl), vl);
                        f = __riscv_vfadd_vf_f32m2(f, b ? b[oc] : 0.f, vl);
                        if (act == YACT_SILU) {
                            vfloat32m2_t e_ = yolo_vexp_f32m2(
                                                  __riscv_vfneg_v_f32m2(f, vl), vl);
                            f = __riscv_vfdiv_vv_f32m2(
                                    f, __riscv_vfadd_vf_f32m2(e_, 1.0f, vl), vl);
                        }
                        if (out_hwc)
                            __riscv_vsse32_v_f32m2(y + (size_t)px0 * OC + oc, (ptrdiff_t)OC * sizeof(float), f, vl);
                        else __riscv_vse32_v_f32m2(y + (size_t)oc * P + px0, f, vl);
                    }
                }
                if (yime_timing) epi_local += ynow() - t1;
            }
            if (yime_timing) {
                int tid = 0;
#ifdef _OPENMP
                tid = omp_get_thread_num();
#endif
                kt[tid] += kernel_local; et[tid] += epi_local;
            }
        }
#ifdef _OPENMP
        if (have_old) sched_setaffinity(0, sizeof(old), &old);
    }
#endif
    if (yime_timing) {
        for (int t = 0; t < 4; t++) { yime_kernel_sum += kt[t]; yime_epi_sum += et[t]; }
        yime_wall += ynow() - wall0;
    }
}

#endif

/* V56 诊断：im2col 拆成「CHW→HWC 转置」和「gather+量化」两段分别累计 */
static double yacc_trans = 0.0, yacc_gather = 0.0;

/* 阶段 P1：双向可逆权重布局转换辅助函数（保障任何 layout 与任何逃生舱组合 100% 兼容） */
static void yconv_unpack_ime_to_hwc(int8_t *dst_hwc, const int8_t *src_ime, int OC, int Kp) {
    const int mt_n = OC / 8, kb_n = Kp / 8;
    for (int mt = 0; mt < mt_n; mt++)
        for (int kb = 0; kb < kb_n; kb++)
            for (int mv = 0; mv < 2; mv++)
                for (int q = 0; q < 4; q++)
                    memcpy(dst_hwc + (size_t)(mt * 8 + mv * 4 + q) * Kp + kb * 8,
                           src_ime + ((size_t)mt * kb_n + kb) * 64 + mv * 32 + q * 8, 8);
}

static void yconv_unpack_hwc_to_chw(int8_t *dst_chw, const int8_t *src_hwc,
                                    int OC, int C, int KH, int KW, int Kp) {
    const int K = C * KH * KW, KHW = KH * KW;
    for (int oc = 0; oc < OC; oc++) {
        int8_t *do_row = dst_chw + (size_t)oc * K;
        const int8_t *wo = src_hwc + (size_t)oc * Kp;
        for (int ic = 0; ic < C; ic++)
            for (int tap = 0; tap < KHW; tap++)
                do_row[(size_t)ic * KHW + tap] = wo[(size_t)tap * C + ic];
    }
}

/* 获取 CHW [OC, K] 布局的 int8 权重指针（供标量 yconv_q8 逃生舱或分组卷积回退） */
static const int8_t *yconv_ensure_chw_q8(StYoloModel *m, uint32_t cid,
                                         int OC, int C, int KH, int KW, int G) {
    const uint32_t lay = m->clayout[cid];
    if (lay == VQF_LAYOUT_LEGACY) return m->cq[cid];
    if (m->cq_chw[cid]) return m->cq_chw[cid];
    const int icg = C / (G > 0 ? G : 1);
    const int K = icg * KH * KW;
    const int Kp = (K + 31) & ~31;
    int8_t *chw = (int8_t *)calloc((size_t)OC * K, 1);
    if (!chw) return NULL;
    if (lay == VQF_LAYOUT_CONV_HWC_TAP) {
        yconv_unpack_hwc_to_chw(chw, m->cq[cid], OC, icg, KH, KW, Kp);
    } else if (lay == VQF_LAYOUT_CONV_IME_K1) {
        int8_t *hwc_tmp = (int8_t *)calloc((size_t)OC * Kp, 1);
        if (!hwc_tmp) { free(chw); return NULL; }
        yconv_unpack_ime_to_hwc(hwc_tmp, m->cq[cid], OC, Kp);
        yconv_unpack_hwc_to_chw(chw, hwc_tmp, OC, icg, KH, KW, Kp);
        free(hwc_tmp);
    } else { free(chw); return NULL; }
    m->cq_chw[cid] = chw;
    return chw;
}

/* V45 试过「p 外层 + oc 内层」GEMM 分块（pblk），想让 imc 的 32×Kp 块驻留 L1
 * 被所有 oc 共享。板端实测（gcc13，yolov8n/bus.jpg，8 线程）：oc-outer 609ms vs
 * pblk 775ms = **慢 27%**，gcc15 交叉编译下亦持平或略慢 ⇒ 已回退删除。
 * 原因：点积是 compute-bound（V41），分块只改 cache 复用；且 p 外层使权重
 * （OC×Kp）被每个 p 块重读 P/vmax 次，权重访问反而恶化。
 * 结论：不要重复此改造，除非先证明点积转为带宽受限。详见指导书 V45。 */
static int yconv_cpu_dotprod(StYoloModel *m, const float *x, int C, int H, int W,
                             uint32_t cid, const float *b, int OC, int KH, int KW,
                             int sh, int sw, int ph, int pw, int G, int act,
                             float *y, int OH, int OW, int in_hwc, int *out_hwc) {
    if (out_hwc) *out_hwc = 0;
    if (G != 1) return 0;                              /* 分组卷积：回退标量 */
    if ((int)cid > m->max_cid || !m->cq[cid] || !m->cs[cid]) return 0;  /* 需 int8 权重 */
    const int K = C * KH * KW;
    const int Kp = (K + 31) & ~31;
    const int P = OH * OW;
    if (K <= 0 || P <= 0 || OC <= 0) return 0;
    /* V50：im2col 走 HWC 输入（k = tap*C + ic）。Kp 补齐段两种排列都在尾部
     * （K 之后全 0），故 GEMM 侧无需区分。 */
    const int hwc = 1;
    const uint32_t lay = m->clayout[cid];

    /* 阶段 P1：离线预打包零拷贝直挂 + 逃生舱全兼容构建 [OC][Kp]。
     * [SHS 公理映射: blas_matrix_block_natural_isomorphism + tensor_5fcontraction_02096446]
     * 当 hwc==1 且 lay == VQF_LAYOUT_CONV_HWC_TAP(4) 时，直接零拷贝挂载 m->cq[cid]！ */
    if (!m->nw[cid]) {
        if (hwc && lay == VQF_LAYOUT_CONV_HWC_TAP) {
            m->nw[cid] = (int8_t *)m->cq[cid];
            m->nw_owned[cid] = 0;   /* 零拷贝直指 m->blob，禁止 free */
        } else {
#ifdef YOLO_P1_FORCE_NW_FAIL
            int8_t *w = NULL;   /* F1 测试：强制 nw 分配失败 */
#else
            int8_t *w = (int8_t *)calloc((size_t)OC * Kp, 1);
#endif
            if (!w) {
                fprintf(stderr, "[WARN][YOLO-P1] nw alloc failed cid=%u OC=%d Kp=%d size=%zu,"
                        " falling back to scalar yconv_q8\n",
                        (unsigned)cid, OC, Kp, (size_t)OC * Kp);
                m->nw_fallback_count++;
                return 0;
            }
            if (lay == VQF_LAYOUT_CONV_IME_K1) {
                /* 从 IME pack_a 逆向还原到 HWC [OC][Kp] */
                yconv_unpack_ime_to_hwc(w, m->cq[cid], OC, Kp);
                if (!hwc) {
                    const int8_t *chw = yconv_ensure_chw_q8(m, cid, OC, C, KH, KW, G);
                    if (!chw) { free(w); return 0; }
                    memset(w, 0, (size_t)OC * Kp);
                    for (int oc = 0; oc < OC; oc++)
                        memcpy(w + (size_t)oc * Kp, chw + (size_t)oc * K, (size_t)K);
                }
            } else if (lay == VQF_LAYOUT_CONV_HWC_TAP) {
                /* lay==4 但 hwc==0（逃生舱关闭 HWC）：逆向解包回 CHW 并补 0 到 Kp */
                const int8_t *chw = yconv_ensure_chw_q8(m, cid, OC, C, KH, KW, G);
                if (!chw) { free(w); return 0; }
                for (int oc = 0; oc < OC; oc++)
                    memcpy(w + (size_t)oc * Kp, chw + (size_t)oc * K, (size_t)K);
            } else {
                /* lay==0 (legacy CHW [OC][K]) */
                const int8_t *src = m->cq[cid];
                if (hwc) {
                    for (int oc = 0; oc < OC; oc++) {
                        int8_t *wo = w + (size_t)oc * Kp;
                        const int8_t *so = src + (size_t)oc * K;
                        for (int ic = 0; ic < C; ic++)
                            for (int tap = 0; tap < KH * KW; tap++)
                                wo[(size_t)tap * C + ic] = so[(size_t)ic * KH * KW + tap];
                    }
                } else {
                    for (int oc = 0; oc < OC; oc++)
                        memcpy(w + (size_t)oc * Kp, src + (size_t)oc * K, (size_t)K);
                }
            }
            m->nw[cid] = w;
            m->nw_owned[cid] = 1;
        }
    }

    const int8_t *Wq = m->nw[cid];
    const float  *cs = m->cs[cid];

    /* V56：先决定是否走 IME（要求 hwc 输入、OC 为 8 的倍数、scratch 就绪）。
     * 走 IME 时：若 lay==VQF_LAYOUT_CONV_IME_K1(5)，st_yolo_load 已将 m->iw[cid]
     * 零拷贝直挂到 m->blob（iw_owned=0），省去运行期 yime_pack_a 分配与重排！ */
    int use_ime = 0;
#if ST_HAVE_RVV
    if (yolo_ime_on() && (OC & 7) == 0) {
        const int nb_n = (P + 15) / 16;
        const size_t need = (size_t)nb_n * Kp * 16;   /* 每 nb 块 = kb_n*128 = Kp*16 */
        if (need > m->ib_cap) {
            int8_t *nbuf = (int8_t *)realloc(m->ib, need);
            if (nbuf) { memset(nbuf, 0, need); m->ib = nbuf; m->ib_cap = need; }
        }
        if ((size_t)P > m->rsc_cap) {
            float *nrs = (float *)realloc(m->rsc, (size_t)P * sizeof(float));
            if (nrs) { m->rsc = nrs; m->rsc_cap = (size_t)P; }
        }
        if (!m->iw[cid]) {
            m->iw[cid] = yime_pack_a(Wq, OC, Kp);
            m->iw_owned[cid] = (m->iw[cid] != NULL) ? 1 : 0;
        }
        if (m->ib && m->ib_cap >= need && m->rsc && m->rsc_cap >= (size_t)P && m->iw[cid]) {
            /* 最后一个不完整 nb 块（像素 P..nb_n*16-1）不会被写 ⇒ 先清零残留 */
            const size_t blk = (size_t)(P / 16) * Kp * 16;
            if (need > blk) memset(m->ib + blk, 0, need - blk);
            use_ime = 1;
        }
    }
#endif

    int8_t *imc = use_ime ? NULL : (int8_t *)malloc((size_t)P * Kp);
    float  *rowsc = use_ime ? m->rsc : (float *)malloc((size_t)P * sizeof(float));
    if (!rowsc || (!use_ime && !imc)) {
        fprintf(stderr, "[WARN][YOLO-P1] scratch alloc failed cid=%u P=%d Kp=%d,"
                " falling back to scalar yconv_q8\n",
                (unsigned)cid, P, Kp);
        m->nw_fallback_count++;
        free(imc); if (!use_ime) free(rowsc); return 0;
    }

    double _t0 = ynow();
    int i2c_rc;
    /* 输入 CHW → HWC（scratch 复用，避免每层 malloc/free 首次触页）。 */
    {
        const size_t need = (size_t)C * H * W;
        if (!in_hwc && need > m->hx_cap) {
            float *nb = (float *)realloc(m->hx, need * sizeof(float));
            if (!nb) { free(imc); if (!use_ime) free(rowsc); return 0; }
            m->hx = nb; m->hx_cap = need;
        }
        if (!in_hwc) yolo_chw2hwc_blocked(x, m->hx, C, H, W);
        const double _tt = ynow();
        yacc_trans += _tt - _t0;
        i2c_rc = yolo_im2col_q8_hwc(in_hwc ? x : m->hx, C, H, W, KH, KW, sh, sw, ph, pw,
                                    OH, OW, K, Kp, imc, rowsc, use_ime ? m->ib : NULL);
        yacc_gather += ynow() - _tt;
    }
    if (i2c_rc != 0) {
        if (!use_ime) { free(imc); free(rowsc); }
        return 0;
    }
    m->t_i2c += ynow() - _t0;

#if ST_HAVE_RVV
    if (use_ime) {
        const double _t1 = ynow();
        yconv_ime(m->iw[cid], m->ib, cs, b, rowsc, OC, Kp, P, act, y, out_hwc != NULL);
        if (out_hwc) *out_hwc = 1;
        const double _t2 = ynow();
        yime_acc_conv += _t2 - _t1;
        if (yime_dbg())
            fprintf(stderr, "[YIME] cid=%u OC=%d Kp=%d P=%d | conv=%.2fms | %.1f G MAC/s\n",
                    cid, OC, Kp, P, (_t2 - _t1) * 1e3,
                    (double)OC * P * Kp / ((_t2 - _t1) * 1e9));
        return 1;   /* imc/rowsc 都是 scratch，无需释放 */
    }
    const int oc4 = OC & ~3;
    if (oc4 > 0)
        yconv_dot_rvv_p(Wq, cs, b, imc, rowsc, oc4, Kp, P, act, y);
    for (int oc = oc4; oc < OC; oc++) {   /* OC 非 4 倍数时的余数通道 */
        const int8_t *wrow = Wq + (size_t)oc * Kp;
        const float oc_s = cs[oc];
        const float bias = b ? b[oc] : 0.f;
        float *yo = y + (size_t)oc * P;
        for (int p = 0; p < P; p++) {
            const int8_t *ir = imc + (size_t)p * Kp;
            int32_t s = 0;
            for (int k = 0; k < Kp; k++) s += (int32_t)wrow[k] * (int32_t)ir[k];
            float r = (float)s * (oc_s * rowsc[p]) + bias;
            if (act == YACT_SILU) r = r / (1.f + expf(-r));
            yo[p] = r;
        }
    }
    free(imc); free(rowsc);
    return 1;
#endif

    const int Kp16 = Kp & ~15;              /* Kp 为 32 倍数 ⇒ Kp16 == Kp（k=16 步长尾） */

    /* Phase 1.2：4-way ILP —— 同时算 4 个位置、4 条独立累加链，掩盖 vdotq 与
     * vaddvq_s32（水平归约）的延迟（小 Kp 时归约占比高，单链效率仅 ~50%）。
     * 每个位置的点积和顺序不变 ⇒ 逐位等价。 */
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(OC > 4)
#endif
    for (int oc = 0; oc < OC; oc++) {
        const int8_t *wrow = Wq + (size_t)oc * Kp;
        const float oc_s = cs[oc];
        const float bias = b ? b[oc] : 0.f;
        float *yo = y + (size_t)oc * P;
        int p = 0;
#if ST_HAVE_NEON && defined(__ARM_FEATURE_DOTPROD)
        for (; p + 4 <= P; p += 4) {
            const int8_t *r0 = imc + (size_t)(p + 0) * Kp;
            const int8_t *r1 = imc + (size_t)(p + 1) * Kp;
            const int8_t *r2 = imc + (size_t)(p + 2) * Kp;
            const int8_t *r3 = imc + (size_t)(p + 3) * Kp;
            int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0);
            int32x4_t a2 = vdupq_n_s32(0), a3 = vdupq_n_s32(0);
            for (int k = 0; k < Kp16; k += 16) {
                int8x16_t wv = vld1q_s8(wrow + k);
                a0 = vdotq_s32(a0, wv, vld1q_s8(r0 + k));
                a1 = vdotq_s32(a1, wv, vld1q_s8(r1 + k));
                a2 = vdotq_s32(a2, wv, vld1q_s8(r2 + k));
                a3 = vdotq_s32(a3, wv, vld1q_s8(r3 + k));
            }
            int32_t s0 = vaddvq_s32(a0), s1 = vaddvq_s32(a1);
            int32_t s2 = vaddvq_s32(a2), s3 = vaddvq_s32(a3);
            for (int k = Kp16; k < Kp; k++) {
                s0 += (int32_t)wrow[k] * (int32_t)r0[k];
                s1 += (int32_t)wrow[k] * (int32_t)r1[k];
                s2 += (int32_t)wrow[k] * (int32_t)r2[k];
                s3 += (int32_t)wrow[k] * (int32_t)r3[k];
            }
            float f0 = (float)s0 * (oc_s * rowsc[p + 0]) + bias;
            float f1 = (float)s1 * (oc_s * rowsc[p + 1]) + bias;
            float f2 = (float)s2 * (oc_s * rowsc[p + 2]) + bias;
            float f3 = (float)s3 * (oc_s * rowsc[p + 3]) + bias;
            if (act == YACT_SILU) {
                f0 = f0 / (1.f + expf(-f0)); f1 = f1 / (1.f + expf(-f1));
                f2 = f2 / (1.f + expf(-f2)); f3 = f3 / (1.f + expf(-f3));
            }
            yo[p + 0] = f0; yo[p + 1] = f1; yo[p + 2] = f2; yo[p + 3] = f3;
        }
#endif
        for (; p < P; p++) {
            const int8_t *irow = imc + (size_t)p * Kp;
#if ST_HAVE_NEON && defined(__ARM_FEATURE_DOTPROD)
            int32x4_t acc = vdupq_n_s32(0);
            for (int k = 0; k < Kp16; k += 16)
                acc = vdotq_s32(acc, vld1q_s8(wrow + k), vld1q_s8(irow + k));
            int32_t s = vaddvq_s32(acc);
            for (int k = Kp16; k < Kp; k++) s += (int32_t)wrow[k] * (int32_t)irow[k];
#else
            int32_t s = 0;
            for (int k = 0; k < Kp; k++) s += (int32_t)wrow[k] * (int32_t)irow[k];
#endif
            float r = (float)s * (oc_s * rowsc[p]) + bias;
            if (act == YACT_SILU) r = r / (1.f + expf(-r));
            yo[p] = r;
        }
    }
    free(imc); free(rowsc);
    return 1;
}

static int yconv_dispatch(StYoloModel *m, const float *x, int C, int H, int W,
                           uint32_t cid, int hb, int OC, int KH, int KW,
                           int sh, int sw, int ph, int pw, int G, int act,
                           float *y, int OH, int OW) {
    const float *b = (hb && (int)cid <= m->max_cid) ? m->cb[cid] : NULL;
    m->cpu_convs++;
    if (m->cw[cid]) {
        /* F32 权重：标量 F32 卷积（F32 模型无 int8 权重，dotprod 不适用）。 */
        yconv(x, C, H, W, m->cw[cid], b, OC, KH, KW, sh, sw, ph, pw, G, act, y, OH, OW);
        return 0;
    }
    /* Phase 1（K-YOLO-002）：CPU int8 dotprod 卷积。
     * 不适用（分组/无 int8 权重等）时显式回退标量 yconv_q8（公理
     * silent_fallback_false_pass，J-CPU-4）。 */
    if (yconv_cpu_dotprod(m, x, C, H, W, cid, b, OC, KH, KW, sh, sw,
                          ph, pw, G, act, y, OH, OW, 0, NULL))
        return 0;
    /* V51 诊断：VLLM_YOLO_SCALAR_PROF=1 时打印回退到标量 yconv_q8 的层
     * （分组 depthwise 与 dotprod 不适用者）及其耗时，用于定位残余耗时归属。 */
    const int prof = getenv("VLLM_YOLO_SCALAR_PROF") != NULL;
    const double t0 = prof ? ynow() : 0.0;
    const int8_t *wq_chw = yconv_ensure_chw_q8(m, cid, OC, C, KH, KW, G);
    if (!wq_chw) return -1;
    yconv_q8(x, C, H, W, wq_chw, m->cs[cid], b, OC, KH, KW, sh, sw, ph, pw,
             G, act, y, OH, OW);
    if (prof)
        fprintf(stderr, "[SCALAR-CONV] %6.2fms cid=%u C=%d %dx%d k%d g%d oc=%d P=%d act=%d\n",
                (ynow() - t0) * 1e3, cid, C, H, W, KH, G, OC, OH * OW, act);
    return 0;
}

static int yattn_psa(StYoloModel *m, const float *x, int C, int H, int W,
                      int nh, int hd, int kd,
                      uint32_t cqkv, uint32_t cproj, uint32_t cpe, float *y) {
    const int N = H * W;
    const int hq = nh * (2 * kd + hd);
    float *qkv = (float *)malloc((size_t)hq * N * sizeof(float));
    float *att = (float *)malloc((size_t)nh * N * N * sizeof(float));
    float *vbuf = (float *)malloc((size_t)C * N * sizeof(float));   /* 连续 v（pe 输入） */
    float *vm  = (float *)malloc((size_t)C * N * sizeof(float));    /* 注意力输出 o */
    float *out = (float *)malloc((size_t)C * N * sizeof(float));
    if (!qkv || !att || !vbuf || !vm || !out) goto fail;
    if (yconv_dispatch(m, x, C, H, W, cqkv, m->cb[cqkv] != NULL, hq, 1, 1, 1, 1, 0, 0, 1, 0, qkv, H, W) != 0) goto fail;
    const float scale = 1.f / sqrtf((float)kd);
    /* qkv[c][p], 每头块内布局 [q(kd) k(kd) v(hd)]；v 拷成连续 [nh,hd,N] */
    for (int h = 0; h < nh; h++) {
        const float *qh = qkv + (size_t)(h * (2 * kd + hd)) * N;
        const float *kh_ = qh + (size_t)kd * N;
        const float *vh = qh + (size_t)2 * kd * N;
        float *at = att + (size_t)h * N * N;
        for (int d = 0; d < hd; d++)
            memcpy(vbuf + (size_t)(h * hd + d) * N, vh + (size_t)d * N, (size_t)N * sizeof(float));
        /* K-YOLO-002 Phase 2：QK^T / softmax / att·V 每个 (h,i) 输出独立 ⇒ OMP over i。 */
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(N > 64)
#endif
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                float s = 0.f;
                for (int d = 0; d < kd; d++) s += qh[(size_t)d * N + i] * kh_[(size_t)d * N + j];
                at[(size_t)i * N + j] = s * scale;
            }
        }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(N > 64)
#endif
        for (int i = 0; i < N; i++) {           /* softmax over j */
            float mx = at[(size_t)i * N];
            for (int j = 1; j < N; j++) { float v = at[(size_t)i * N + j]; if (v > mx) mx = v; }
            float sum = 0.f;
            for (int j = 0; j < N; j++) { float e = expf(at[(size_t)i * N + j] - mx); at[(size_t)i * N + j] = e; sum += e; }
            float inv = 1.f / sum;
            for (int j = 0; j < N; j++) at[(size_t)i * N + j] *= inv;
        }
        for (int d = 0; d < hd; d++) {
            float *od = vm + (size_t)(h * hd + d) * N;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(N > 64)
#endif
            for (int i = 0; i < N; i++) {
                float s = 0.f;
                for (int j = 0; j < N; j++) s += vh[(size_t)d * N + j] * at[(size_t)i * N + j];
                od[i] = s;
            }
        }
    }
    /* o += pe(v)（pe 为 depthwise 3×3 分组卷积） */
    if (yconv_dispatch(m, vbuf, C, H, W, cpe, m->cb[cpe] != NULL, C, 3, 3, 1, 1, 1, 1, C, 0, out, H, W) != 0) goto fail;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if((long long)C * N > 65536)
#endif
    for (long long i = 0; i < (long long)C * N; i++) vm[i] += out[i];
    if (yconv_dispatch(m, vm, C, H, W, cproj, m->cb[cproj] != NULL, C, 1, 1, 1, 1, 0, 0, 1, 0, out, H, W) != 0) goto fail;
    memcpy(y, out, (size_t)C * N * sizeof(float));
    free(qkv); free(att); free(vbuf); free(vm); free(out);
    return 0;
fail:
    free(qkv); free(att); free(vbuf); free(vm); free(out);
    return -1;
}

/* ---- 槽位与阶段 P3 静态 Arena 规划器 ----
 * [SHS 公理映射: set_5foperations_91154c82 (集合不交集公理)]
 * 生命周期区间 L(s_i) ∩ L(s_j) = ∅ 时共享同一连续 Arena 偏移，
 * 消除每帧 60+ 次独立 malloc/free 与首次触页开销。 */
typedef struct {
    float *buf; int c, h, w; int owned; int layout; float *other;
    int other_owned, other_valid;
    StYoloModel *model;
    YViewPlan *view;
} YSlot;

static size_t slot_idx(const YSlot *s, int c, size_t p) {
    return s->layout ? p * s->c + c : (size_t)c * s->h * s->w + p;
}

static int slot_view(YSlot *s, int layout, YSlot *v) {
    *v = *s;
    v->owned = 0;
    v->other = NULL;
    if (s->layout != layout) {
        YViewPlan *p = s->view;
        if (p) {
            int step = s->model->view_step;
            if (p->first < 0 || step < p->first) p->first = step;
            if (step > p->last) p->last = step;
            p->layout = s->layout;
            if (s->other && !s->other_owned &&
                (step < p->planned_first || step > p->planned_last ||
                 s->layout != p->planned_layout)) {
                s->other = NULL;
                s->other_valid = 0;
            }
        }
        if (!s->other_valid) {
            size_t n = (size_t)s->c * s->h * s->w;
            if (!s->other) {
                s->other = (float *)malloc(n * sizeof(float));
                if (!s->other) return -1;
                s->other_owned = 1;
            }
            if (layout) yolo_chw2hwc_blocked(s->buf, s->other, s->c, s->h, s->w);
            else for (int c = 0; c < s->c; c++)
                for (size_t p = 0; p < (size_t)s->h * s->w; p++)
                    s->other[(size_t)c * s->h * s->w + p] = s->buf[p * s->c + c];
            s->other_valid = 1;
        }
        v->buf = s->other;
        v->layout = layout;
    }
    return 0;
}

/* Keep main offsets fixed; place recorded views against all intersecting intervals. */
static void yolo_plan_views(StYoloModel *m) {
    if (!m->views || !m->view_complete || m->view_ready) return;
    size_t peak = m->arena_main_elems;
    for (int s = 0; s < m->n_slots; s++) {
        YViewPlan *p = &m->views[s];
        p->off = SIZE_MAX;
        if (p->first < 0) continue;
        size_t off = 0;
        for (;;) {
            size_t next = off;
            for (int j = 0; j < m->n_slots; j++) {
                YViewPlan *q = &m->views[j];
                if (q->len && q->def <= p->last && m->slot_last_use[j] >= p->first &&
                    off < m->arena_offsets[j] + q->len && m->arena_offsets[j] < off + p->len)
                    if (next < m->arena_offsets[j] + q->len) next = m->arena_offsets[j] + q->len;
                if (j < s && q->off != SIZE_MAX && q->first <= p->last && q->last >= p->first &&
                    off < q->off + q->len && q->off < off + p->len)
                    if (next < q->off + q->len) next = q->off + q->len;
            }
            if (next == off) break;
            off = next;
        }
        p->off = off;
        p->planned_first = p->first; p->planned_last = p->last;
        p->planned_layout = p->layout;
        if (peak < off + p->len) peak = off + p->len;
    }
    if (peak > m->arena_cap) {
        float *a = (float *)realloc(m->arena, peak * sizeof(float));
        if (!a) return;
        m->arena = a; m->arena_cap = peak;
    }
    m->arena_view_elems = peak - m->arena_main_elems;
    m->view_ready = 1;
}

static void yolo_bind_views(StYoloModel *m, YSlot *S) {
    m->view_complete = 0;
    if (!m->views) return;
    for (int s = 0; s < m->n_slots; s++) {
        YViewPlan *p = &m->views[s];
        S[s].model = m; S[s].view = p;
        S[s].other_valid = 0;
        if (m->view_ready && p->off != SIZE_MAX) S[s].other = m->arena + p->off;
    }
}

/* 依据输入尺寸 (ih, iw) 静态推导全图各 slot 形状与生命周期不交集 Arena 偏移 */
static int yolo_plan_arena(StYoloModel *m, int ih, int iw, YSlot *S) {
    if (!m->slot_last_use || m->n_slots <= 0) return 0;
    const int ns = m->n_slots;
    /* 主表示计划仅由不可变图和输入尺寸决定，与后端/layout 无关。 */
    if (m->arena_offsets && m->arena_ih == ih && m->arena_iw == iw) {
        yolo_plan_views(m);
        yolo_bind_views(m, S);
        for (int s = 0; s < ns; s++) {
            if (m->arena_offsets[s] != SIZE_MAX) {
                S[s].buf = m->arena + m->arena_offsets[s];
                S[s].owned = 0;
            }
        }
        m->arena_plan_hits++;
        return 1;
    }
    size_t *s_elems = (size_t *)calloc((size_t)ns, sizeof(size_t));
    size_t *s_off   = (size_t *)calloc((size_t)ns, sizeof(size_t));
    int    *s_def   = (int *)calloc((size_t)ns, sizeof(int));
    if (!s_elems || !s_off || !s_def) {
        free(s_elems); free(s_off); free(s_def); return 0;
    }
    for (int s = 0; s < ns; s++) s_def[s] = -1;

    S[0].c = 3; S[0].h = ih; S[0].w = iw;
    s_elems[0] = (((size_t)3 * ih * iw) + 15u) & ~15u;   /* 64B (16 float) 对齐 */
    s_def[0] = 0;

    for (int i = 0; i < m->n_ops; i++) {
        const YOp *o = &m->ops[i];
        const uint32_t *P = o->prm;
        if ((int)o->out >= ns) continue;
        int oc = 0, oh = 0, ow = 0;
        switch (o->op) {
        case YOP_CONV: {
            uint32_t cid = P[0];
            int sh = (int)P[2], sw = (int)P[3], ph = (int)P[4], pw = (int)P[5];
            int KH = (int)P[8], KW = (int)P[9];
            YSlot *x = &S[o->ins[0]];
            oc = (cid <= (uint32_t)m->max_cid) ? m->crow[cid] : 0;
            oh = (x->h + 2 * ph - KH) / sh + 1;
            ow = (x->w + 2 * pw - KW) / sw + 1;
            break;
        }
        case YOP_CONCAT: {
            oh = S[o->ins[0]].h; ow = S[o->ins[0]].w;
            for (uint32_t k = 0; k < o->nin; k++) oc += S[o->ins[k]].c;
            break;
        }
        case YOP_MAXPOOL: {
            YSlot *x = &S[o->ins[0]];
            int k = (int)P[0], st = (int)P[1], p = (int)P[2];
            oc = x->c;
            oh = (x->h + 2 * p - k) / st + 1;
            ow = (x->w + 2 * p - k) / st + 1;
            break;
        }
        case YOP_UPSAMPLE: {
            YSlot *x = &S[o->ins[0]];
            int sc = (int)P[0];
            oc = x->c; oh = x->h * sc; ow = x->w * sc;
            break;
        }
        case YOP_SLICE: {
            YSlot *x = &S[o->ins[0]];
            oc = (int)P[2] - (int)P[1]; oh = x->h; ow = x->w;
            break;
        }
        case YOP_DETECT: {
            int nc = (int)P[0], A = 0;
            for (int s = 0; s < 3; s++) A += S[o->ins[s]].h * S[o->ins[s]].w;
            oc = 4 + nc; oh = A; ow = 1;
            break;
        }
        case YOP_MATMUL: {
            uint32_t cid = P[0];
            YSlot *x = &S[o->ins[0]];
            oc = (cid <= (uint32_t)m->max_cid) ? m->crow[cid] : 0;
            oh = x->h; ow = x->w;
            break;
        }
        case YOP_ADD:
        case YOP_SIGMOID:
        case YOP_MUL:
        case YOP_SOFTMAX:
        case YOP_ATTN_PSA:
        case YOP_RMSNORM:
        case YOP_ROPE:
        case YOP_SWIGLU: {
            YSlot *x = &S[o->ins[0]];
            oc = x->c; oh = x->h; ow = x->w;
            break;
        }
        default:
            free(s_elems); free(s_off); free(s_def);
            return 0;
        }
        if (oc <= 0 || oh <= 0 || ow <= 0) {
            free(s_elems); free(s_off); free(s_def);
            return 0;
        }
        S[o->out].c = oc; S[o->out].h = oh; S[o->out].w = ow;
        s_elems[o->out] = (((size_t)oc * oh * ow) + 15u) & ~15u;
        if (s_def[o->out] < 0) s_def[o->out] = i + 1;
    }

    /* 线性扫描不交集区间复用分配器（First-Fit Free List + Coalescing） */
    typedef struct { size_t off, len; int free_at; int active_slot; } YBlk;
    YBlk *blks = (YBlk *)calloc((size_t)ns * 2 + 4, sizeof(YBlk));
    if (!blks) { free(s_elems); free(s_off); free(s_def); return 0; }
    int nblk = 0;
    size_t total_elems = 0, sum_raw = 0;

    for (int step = 0; step <= m->n_ops; step++) {
        /* 1. 释放所有在当前算子 (op_idx = step - 1) 之前已结束生命周期
         *    (last_use < step - 1) 的块并合并相邻空闲块。
         *    注意：last_use == step - 1 的槽位正是当前算子的输入，绝不可与当前输出重叠！ */
        for (int b = 0; b < nblk; b++) {
            if (blks[b].active_slot >= 0 && m->slot_last_use[blks[b].active_slot] < step - 1)
                blks[b].active_slot = -1;
        }
        for (int b = 0; b + 1 < nblk; ) {
            if (blks[b].active_slot < 0 && blks[b + 1].active_slot < 0 &&
                blks[b].off + blks[b].len == blks[b + 1].off) {
                blks[b].len += blks[b + 1].len;
                memmove(&blks[b + 1], &blks[b + 2], (size_t)(nblk - b - 2) * sizeof(YBlk));
                nblk--;
            } else {
                b++;
            }
        }
        /* 2. 为在当前 step 定义的 slot 分配区间 */
        int s = (step == 0) ? 0 : (int)m->ops[step - 1].out;
        if (s < 0 || s >= ns || s_elems[s] == 0) continue;
        size_t need = s_elems[s];
        sum_raw += need;
        int best_b = -1;
        for (int b = 0; b < nblk; b++) {
            if (blks[b].active_slot < 0 && blks[b].len >= need) {
                if (best_b < 0 || blks[b].len < blks[best_b].len) best_b = b;
            }
        }
        if (best_b >= 0) {
            s_off[s] = blks[best_b].off;
            if (blks[best_b].len > need) {
                memmove(&blks[best_b + 2], &blks[best_b + 1],
                        (size_t)(nblk - best_b - 1) * sizeof(YBlk));
                blks[best_b + 1].off = blks[best_b].off + need;
                blks[best_b + 1].len = blks[best_b].len - need;
                blks[best_b + 1].active_slot = -1;
                blks[best_b].len = need;
                nblk++;
            }
            blks[best_b].active_slot = s;
        } else {
            s_off[s] = total_elems;
            blks[nblk].off = total_elems;
            blks[nblk].len = need;
            blks[nblk].active_slot = s;
            nblk++;
            total_elems += need;
        }
    }
    free(blks);

    if (total_elems > m->arena_cap) {
        float *na = (float *)realloc(m->arena, total_elems * sizeof(float));
        if (!na) { free(s_elems); free(s_off); free(s_def); return 0; }
        m->arena = na;
        m->arena_cap = total_elems;
        if (getenv("VLLM_YOLO_DEBUG"))
            fprintf(stderr, "[YOLO-ARENA] planned %d slots: peak=%.2f MB vs raw=%.2f MB (saved %.1f%%)\n",
                    ns, (double)(total_elems * sizeof(float)) / 1048576.0,
                    (double)(sum_raw * sizeof(float)) / 1048576.0,
                    sum_raw > 0 ? 100.0 * (1.0 - (double)total_elems / (double)sum_raw) : 0.0);
    }
    for (int s = 0; s < ns; s++) {
        if (s_elems[s] > 0) {
            S[s].buf = m->arena + s_off[s];
            S[s].owned = 0;
        } else {
            s_off[s] = SIZE_MAX;
        }
    }
    free(m->arena_offsets);
    m->arena_offsets = s_off;
    m->arena_ih = ih; m->arena_iw = iw;
    m->arena_plan_builds++;
    free(m->views);
    m->views = (YViewPlan *)calloc((size_t)ns, sizeof(YViewPlan));
    m->view_ready = m->view_complete = 0;
    m->arena_main_elems = total_elems; m->arena_view_elems = 0;
    if (m->views) for (int s = 0; s < ns; s++) {
        m->views[s].len = s_elems[s];
        m->views[s].def = s_def[s] - 1;
        m->views[s].first = m->views[s].last = -1;
        m->views[s].off = SIZE_MAX;
    }
    yolo_bind_views(m, S);
    free(s_elems); free(s_def);
    return 1;
}

static float *slot_new(YSlot *S, uint32_t s, int c, int h, int w) {
    if (S[s].other_owned) { free(S[s].other); S[s].other = NULL; }
    S[s].other_owned = S[s].other_valid = 0;
    S[s].layout = 0;
    S[s].c = c; S[s].h = h; S[s].w = w;
    if (S[s].buf && !S[s].owned) return S[s].buf;   /* 阶段 P3：静态 Arena 已预分配 */
    if (S[s].owned) free(S[s].buf);
    size_t n = (size_t)c * h * w;
    S[s].buf = (float *)malloc(n * sizeof(float));
    S[s].owned = 1;
    return S[s].buf;
}

static void slots_free(YSlot *S, int n_slots) {
    if (!S) return;
    for (int i = 0; i < n_slots; i++) {
        if (S[i].owned) free(S[i].buf);
        if (S[i].other_owned) free(S[i].other);
    }
    free(S);
}

/* ================================================================
 * 前向（图解释器）
 * ================================================================ */
static int yolo_forward(StYoloModel *m, const float *in, int iw, int ih,
                        YSlot *S, uint32_t *final_slot, int *final_c, int *final_a) {
    /* Phase-0 CPU 分解（K-YOLO-002）：env 门控，默认关（零开销）。 */
    int cpu_tim = 0;
    { const char *e = getenv("VLLM_YOLO_CPU_TIMING");
      cpu_tim = (e && e[0] && e[0] != '0'); }
    m->t_conv = m->t_detect = m->t_attn = m->t_elem = 0.0;
    m->n_conv_ops = m->n_elem_ops = 0;
#if ST_HAVE_RVV
    yime_acc_conv = 0.0;
    { const char *e = getenv("VLLM_YOLO_IME_TIMING");
      yime_timing = e && e[0] && e[0] != '0'; }
    yime_kernel_sum = yime_epi_sum = yime_wall = 0.0;
#endif
    yacc_trans = yacc_gather = 0.0;
    if (cpu_tim) {
        m->t_i2c = 0.0;
        for (int k = 0; k <= YOP_MAX; k++) { m->t_opk[k] = 0.0; m->n_opk[k] = 0; }
    }
    double _fwd0 = cpu_tim ? ynow() : 0.0;
    yolo_plan_arena(m, ih, iw, S);
    if (!slot_new(S, 0, 3, ih, iw)) return -1;
    memcpy(S[0].buf, in, (size_t)3 * ih * iw * sizeof(float));

    for (int i = 0; i < m->n_ops; i++) {
        const YOp *o = &m->ops[i];
        const uint32_t *P = o->prm;
        m->view_step = i;
        double _op0 = cpu_tim ? ynow() : 0.0;
        switch (o->op) {
        case YOP_CONV: {
            uint32_t cid = P[0]; int hb = (int)P[1];
            int sh = (int)P[2], sw = (int)P[3], ph = (int)P[4], pw = (int)P[5];
            int G = (int)P[6], act = (int)P[7], KH = (int)P[8], KW = (int)P[9];
            YSlot *x = &S[o->ins[0]];
            int OH = (x->h + 2 * ph - KH) / sh + 1;
            int OW = (x->w + 2 * pw - KW) / sw + 1;
            int OC = m->crow[cid];
            if (OC <= 0) { fprintf(stderr, "[YOLO] conv cid=%u weight missing\n", cid); return -1; }
            float *y = slot_new(S, o->out, OC, OH, OW);
            if (!y) return -1;
            if (!m->cw[cid] &&
                yconv_cpu_dotprod(m, x->buf, x->c, x->h, x->w, cid, hb ? m->cb[cid] : NULL,
                    OC, KH, KW, sh, sw, ph, pw, G, act, y, OH, OW, x->layout, &S[o->out].layout)) {
                m->cpu_convs++;
            } else {
                YSlot v;
                if (slot_view(x, 0, &v) != 0) return -1;
#if ST_HAVE_RVV
                const char *dw_rvv = getenv("VLLM_YOLO_DW_RVV");
                if (dw_rvv && strcmp(dw_rvv, "1") == 0 &&
                    !m->cw[cid] && m->cq[cid] && v.layout == 0 &&
                    (v.c == 64 || v.c == 80 || v.c == 128 || v.c == 256) &&
                    OC == v.c && G == v.c && G > 1 &&
                    KH == 3 && KW == 3 && sh == 1 && sw == 1 && ph == 1 && pw == 1 &&
                    OH == v.h && OW == v.w && v.h > 0 && v.w > 0 && act == YACT_SILU) {
                    m->cpu_convs++;
                    const int8_t *wq = yconv_ensure_chw_q8(m, cid, OC, v.c, KH, KW, G);
                    if (!wq) return -1;
                    yconv_dw_q8_rvv(v.buf, v.c, v.h, v.w, wq, m->cs[cid],
                                    hb ? m->cb[cid] : NULL, y);
                } else
#endif
                if (yconv_dispatch(m, v.buf, v.c, v.h, v.w, cid, hb, OC, KH, KW,
                                   sh, sw, ph, pw, G, act, y, OH, OW) != 0) return -1;
            }
            break;
        }
        case YOP_ADD: {
            YSlot *a = &S[o->ins[0]], bv, *b = &bv;
            if (slot_view(&S[o->ins[1]], a->layout, &bv) != 0) return -1;
            long long n = (long long)a->c * a->h * a->w;
            float *y = slot_new(S, o->out, a->c, a->h, a->w);
            if (!y) return -1;
            S[o->out].layout = a->layout;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(n > 65536)
#endif
            for (long long k = 0; k < n; k++) y[k] = a->buf[k] + b->buf[k];
            break;
        }
        case YOP_CONCAT: {
            int C = 0, h = S[o->ins[0]].h, w = S[o->ins[0]].w;
            for (uint32_t k = 0; k < o->nin; k++) C += S[o->ins[k]].c;
            float *y = slot_new(S, o->out, C, h, w);
            if (!y) return -1;
            const size_t hw = (size_t)h * w;
            int layout = 0;
            for (uint32_t k = 0; k < o->nin; k++) layout |= S[o->ins[k]].layout;
            if (layout) {
                S[o->out].layout = 1;
                int off = 0;
                for (uint32_t k = 0; k < o->nin; k++) {
                    YSlot v;
                    if (slot_view(&S[o->ins[k]], 1, &v) != 0) return -1;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(hw * C > 65536)
#endif
                    for (size_t p = 0; p < hw; p++)
                        memcpy(y + p * C + off, v.buf + p * v.c, (size_t)v.c * sizeof(float));
                    off += v.c;
                }
                break;
            }
            /* 各输入写入 y 的不相交区间 ⇒ 可并行（K-YOLO-002 Phase 2；前缀偏移先算）。 */
            if (o->nin <= 32) {
                size_t off[32];
                size_t acc = 0;
                for (uint32_t k = 0; k < o->nin; k++) {
                    off[k] = acc;
                    acc += (size_t)S[o->ins[k]].c * hw;
                }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(o->nin > 1 && acc > 65536)
#endif
                for (int k = 0; k < (int)o->nin; k++) {
                    YSlot *s = &S[o->ins[k]];
                    memcpy(y + off[k], s->buf, (size_t)s->c * hw * sizeof(float));
                }
            } else {
                size_t off = 0;
                for (uint32_t k = 0; k < o->nin; k++) {
                    YSlot *s = &S[o->ins[k]];
                    memcpy(y + off, s->buf, (size_t)s->c * hw * sizeof(float));
                    off += (size_t)s->c * hw;
                }
            }
            break;
        }
        case YOP_MAXPOOL: {
            YSlot *x = &S[o->ins[0]];
            int k = (int)P[0], st = (int)P[1], p = (int)P[2];
            int OH = (x->h + 2 * p - k) / st + 1, OW = (x->w + 2 * p - k) / st + 1;
            float *y = slot_new(S, o->out, x->c, OH, OW);
            if (!y) return -1;
            S[o->out].layout = x->layout;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(x->c > 2)
#endif
            for (int c = 0; c < x->c; c++)
                for (int oy = 0; oy < OH; oy++)
                    for (int ox = 0; ox < OW; ox++) {
                        float mx = -INFINITY;
                        for (int ky = 0; ky < k; ky++) {
                            int iy = oy * st - p + ky; if (iy < 0 || iy >= x->h) continue;
                            for (int kx = 0; kx < k; kx++) {
                                int ix = ox * st - p + kx; if (ix < 0 || ix >= x->w) continue;
                                float v = x->buf[slot_idx(x, c, (size_t)iy * x->w + ix)];
                                if (v > mx) mx = v;
                            }
                        }
                        y[slot_idx(&S[o->out], c, (size_t)oy * OW + ox)] = mx;
                    }
            break;
        }
        case YOP_UPSAMPLE: {
            YSlot *x = &S[o->ins[0]];
            int sc = (int)P[0];
            int OH = x->h * sc, OW = x->w * sc;
            float *y = slot_new(S, o->out, x->c, OH, OW);
            if (!y) return -1;
            S[o->out].layout = x->layout;
            if (x->layout) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(x->h > 1)
#endif
                for (int iy = 0; iy < x->h; iy++) {
                    float *row = y + (size_t)iy * sc * OW * x->c;
                    const float *src = x->buf + (size_t)iy * x->w * x->c;
                    for (int ix = 0; ix < x->w; ix++)
                        for (int k = 0; k < sc; k++)
                            memcpy(row + ((size_t)ix * sc + k) * x->c,
                                   src + (size_t)ix * x->c, (size_t)x->c * sizeof(float));
                    for (int k = 1; k < sc; k++)
                        memcpy(row + (size_t)k * OW * x->c, row, (size_t)OW * x->c * sizeof(float));
                }
                break;
            }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(x->c > 2)
#endif
            for (int c = 0; c < x->c; c++)
                for (int oy = 0; oy < OH; oy++)
                    for (int ox = 0; ox < OW; ox++)
                        y[((size_t)c * OH + oy) * OW + ox] =
                            x->buf[((size_t)c * x->h + oy / sc) * x->w + ox / sc];
            break;
        }
        case YOP_SIGMOID: {
            YSlot *x = &S[o->ins[0]];
            long long n = (long long)x->c * x->h * x->w;
            float *y = slot_new(S, o->out, x->c, x->h, x->w);
            if (!y) return -1;
            S[o->out].layout = x->layout;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(n > 65536)
#endif
            for (long long k = 0; k < n; k++) y[k] = 1.f / (1.f + expf(-x->buf[k]));
            break;
        }
        case YOP_MUL: {
            YSlot *a = &S[o->ins[0]], bv, *b = &bv;
            if (slot_view(&S[o->ins[1]], a->layout, &bv) != 0) return -1;
            long long n = (long long)a->c * a->h * a->w;
            float *y = slot_new(S, o->out, a->c, a->h, a->w);
            if (!y) return -1;
            S[o->out].layout = a->layout;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(n > 65536)
#endif
            for (long long k = 0; k < n; k++) y[k] = a->buf[k] * b->buf[k];
            break;
        }
        case YOP_SLICE: {
            YSlot *x = &S[o->ins[0]];
            int a = (int)P[1], b2 = (int)P[2];
            int C = b2 - a;
            float *y = slot_new(S, o->out, C, x->h, x->w);
            if (!y) return -1;
            S[o->out].layout = x->layout;
            if (x->layout) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if((size_t)C * x->h * x->w > 65536)
#endif
                for (size_t p = 0; p < (size_t)x->h * x->w; p++)
                    memcpy(y + p * C, x->buf + p * x->c + a, (size_t)C * sizeof(float));
                break;
            }
            const float *src = x->buf + (size_t)a * x->h * x->w;
            long long tot = (long long)C * x->h * x->w;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(tot > 65536)
#endif
            for (long long i = 0; i < tot; i++) y[i] = src[i];
            break;
        }
        case YOP_SOFTMAX: {
            /* 热带/概率归一化映射：默认沿 channel 维（axis=0）执行数值稳定 Softmax */
            YSlot xv, *x = &xv;
            if (slot_view(&S[o->ins[0]], 0, &xv) != 0) return -1;
            int axis = (o->nparam >= 1) ? (int)P[0] : 0;
            float *y = slot_new(S, o->out, x->c, x->h, x->w);
            if (!y) return -1;
            const int C = x->c, HW = x->h * x->w;
            if (axis == 0) {
                for (int p = 0; p < HW; p++) {
                    float mx = x->buf[p];
                    for (int c = 1; c < C; c++) {
                        float v = x->buf[(size_t)c * HW + p];
                        if (v > mx) mx = v;
                    }
                    float sum = 0.f;
                    for (int c = 0; c < C; c++) {
                        float e = expf(x->buf[(size_t)c * HW + p] - mx);
                        y[(size_t)c * HW + p] = e;
                        sum += e;
                    }
                    float inv = 1.f / sum;
                    for (int c = 0; c < C; c++) y[(size_t)c * HW + p] *= inv;
                }
            } else {
                for (int c = 0; c < C; c++) {
                    const float *xr = x->buf + (size_t)c * HW;
                    float *yr = y + (size_t)c * HW;
                    float mx = xr[0];
                    for (int p = 1; p < HW; p++) if (xr[p] > mx) mx = xr[p];
                    float sum = 0.f;
                    for (int p = 0; p < HW; p++) { float e = expf(xr[p] - mx); yr[p] = e; sum += e; }
                    float inv = 1.f / sum;
                    for (int p = 0; p < HW; p++) yr[p] *= inv;
                }
            }
            *final_slot = o->out; *final_c = x->c; *final_a = HW;
            break;
        }
        case YOP_ATTN_PSA: {
            YSlot xv, *x = &xv;
            if (slot_view(&S[o->ins[0]], 0, &xv) != 0) return -1;
            int nh = (int)P[0], hd = (int)P[1], kd = (int)P[2];
            float *y = slot_new(S, o->out, x->c, x->h, x->w);
            if (!y) return -1;
            if (yattn_psa(m, x->buf, x->c, x->h, x->w, nh, hd, kd,
                          P[3], P[4], P[5], y) != 0) return -1;
            break;
        }
        case YOP_MATMUL: {
            /* 阶段 P2：线性代数半环 (+, ×) 通用矩阵乘算子
             * [SHS 公理映射: shs_axiom_tensor_graph_mapping_v1 + blas_matrix_block_natural_isomorphism]
             * 复用 1×1/s1/p0/g1 yconv_dispatch，天然继承 F32 / Q8 dotprod / RVV / IME 零拷贝加速 */
            uint32_t cid = P[0];
            int hb  = (o->nparam >= 2) ? (int)P[1] : 0;
            int act = (o->nparam >= 3) ? (int)P[2] : 0;
            YSlot xv, *x = &xv;
            if (slot_view(&S[o->ins[0]], 0, &xv) != 0) return -1;
            int OC = (cid <= (uint32_t)m->max_cid) ? m->crow[cid] : 0;
            if (OC <= 0) { fprintf(stderr, "[YOLO] matmul cid=%u weight missing\n", cid); return -1; }
            float *y = slot_new(S, o->out, OC, x->h, x->w);
            if (!y) return -1;
            if (yconv_dispatch(m, x->buf, x->c, x->h, x->w, cid, hb, OC, 1, 1,
                               1, 1, 0, 0, 1, act, y, x->h, x->w) != 0) return -1;
            *final_slot = o->out; *final_c = OC; *final_a = x->h * x->w;
            break;
        }
        case YOP_RMSNORM: {
            /* 阶段 P2：Bregman 流形归一化投影算子 RMSNorm
             * [SHS 公理映射: shs_axiom_bregman_divergence_v1]
             * y[c, p] = x[c, p] * rsqrt(mean_c(x[:, p]^2) + eps) * w[c] */
            uint32_t cid = P[0];
            float eps = 1e-6f;
            if (o->nparam >= 2 && P[1] != 0) memcpy(&eps, &P[1], sizeof(float));
            YSlot xv, *x = &xv;
            if (slot_view(&S[o->ins[0]], 0, &xv) != 0) return -1;
            const float *w = (cid <= (uint32_t)m->max_cid) ? m->cw[cid] : NULL;
            float *y = slot_new(S, o->out, x->c, x->h, x->w);
            if (!y) return -1;
            const int C = x->c, P_len = x->h * x->w;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(P_len > 256)
#endif
            for (int p = 0; p < P_len; p++) {
                float ss = 0.f;
                for (int c = 0; c < C; c++) {
                    float v = x->buf[(size_t)c * P_len + p];
                    ss += v * v;
                }
                float scale = 1.f / sqrtf(ss / (float)C + eps);
                for (int c = 0; c < C; c++) {
                    float wc = w ? w[c] : 1.f;
                    y[(size_t)c * P_len + p] = x->buf[(size_t)c * P_len + p] * scale * wc;
                }
            }
            *final_slot = o->out; *final_c = x->c; *final_a = P_len;
            break;
        }
        case YOP_ROPE: {
            /* 阶段 P2：正交旋转群 SO(2)^d 同构算子 RoPE
             * 对每个空间/序列位置 p 与每个注意力头，对半拆分通道 (d, d + hd/2) 施加旋转 */
            YSlot xv, *x = &xv;
            if (slot_view(&S[o->ins[0]], 0, &xv) != 0) return -1;
            int hd = (o->nparam >= 1 && P[0] > 0) ? (int)P[0] : x->c;
            float theta = 10000.0f;
            if (o->nparam >= 2 && P[1] != 0) memcpy(&theta, &P[1], sizeof(float));
            int pos_off = (o->nparam >= 3) ? (int)P[2] : 0;
            float *y = slot_new(S, o->out, x->c, x->h, x->w);
            if (!y) return -1;
            const int C = x->c, P_len = x->h * x->w;
            const int nh = C / hd, half = hd / 2;
            for (int d = 0; d < half; d++) {
                float freq = powf(theta, -2.0f * (float)d / (float)hd);
                for (int p = 0; p < P_len; p++) {
                    float ang = (float)(pos_off + p) * freq;
                    float cs_v = cosf(ang), sn_v = sinf(ang);
                    for (int h = 0; h < nh; h++) {
                        int c0 = h * hd + d;
                        int c1 = c0 + half;
                        float x0 = x->buf[(size_t)c0 * P_len + p];
                        float x1 = x->buf[(size_t)c1 * P_len + p];
                        y[(size_t)c0 * P_len + p] = x0 * cs_v - x1 * sn_v;
                        y[(size_t)c1 * P_len + p] = x1 * cs_v + x0 * sn_v;
                    }
                }
            }
            *final_slot = o->out; *final_c = x->c; *final_a = P_len;
            break;
        }
        case YOP_SWIGLU: {
            /* 阶段 P2：门控非线性合成算子 SwiGLU(gate, up) = SiLU(gate) * up */
            YSlot gv, uv, *gate = &gv, *up = &uv;
            if (slot_view(&S[o->ins[0]], 0, &gv) != 0 ||
                slot_view(&S[o->ins[1]], 0, &uv) != 0) return -1;
            long long n = (long long)gate->c * gate->h * gate->w;
            float *y = slot_new(S, o->out, gate->c, gate->h, gate->w);
            if (!y) return -1;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(n > 65536)
#endif
            for (long long k = 0; k < n; k++) {
                float g = gate->buf[k];
                y[k] = (g / (1.f + expf(-g))) * up->buf[k];
            }
            *final_slot = o->out; *final_c = gate->c; *final_a = gate->h * gate->w;
            break;
        }
        case YOP_DETECT: {
            int nc = (int)P[0], rm = (int)P[1];
            int st[3] = { (int)P[2], (int)P[3], (int)P[4] };
            /* box slots = ins[0..2], cls slots = ins[3..5] */
            int A = 0, shapes[3][2];
            for (int s = 0; s < 3; s++) {
                YSlot *b = &S[o->ins[s]];
                shapes[s][0] = b->h; shapes[s][1] = b->w;
                A += b->h * b->w;
            }
            int C = 4 + nc;
            float *y = slot_new(S, o->out, C, A, 1);
            if (!y) return -1;
            int base = 0;
            for (int s = 0; s < 3; s++) {
                YSlot *bx = &S[o->ins[s]];
                YSlot *cl = &S[o->ins[3 + s]];
                int h = shapes[s][0], w = shapes[s][1];
                /* 每个 anchor p 写 y[*][base+p] 互不相交 ⇒ 可并行（K-YOLO-002 Phase 2）。 */
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(h * w > 256)
#endif
                for (int p = 0; p < h * w; p++) {
                    int px = p % w, py = p / w;
                    float ax = px + 0.5f, ay = py + 0.5f;
                    float d4[4];
                    for (int d = 0; d < 4; d++) {
                        const float *bin = bx->buf + slot_idx(bx, d * rm, (size_t)p);
                        size_t bstep = bx->layout ? 1 : (size_t)h * w;
                        float mx = bin[0], sum = 0.f;
                        for (int j = 1; j < rm; j++) { float v = bin[(size_t)j * bstep]; if (v > mx) mx = v; }
                        float acc = 0.f;
                        for (int j = 0; j < rm; j++) {
                            float e = expf(bin[(size_t)j * bstep] - mx);
                            acc += e * (float)j; sum += e;
                        }
                        d4[d] = acc / sum;
                    }
                    float x1 = ax - d4[0], y1 = ay - d4[1], x2 = ax + d4[2], y2 = ay + d4[3];
                    float cx = (x1 + x2) * 0.5f, cy = (y1 + y2) * 0.5f;
                    float bw = x2 - x1, bh = y2 - y1;
                    int q = base + p;
                    y[(size_t)0 * A + q] = cx * st[s];
                    y[(size_t)1 * A + q] = cy * st[s];
                    y[(size_t)2 * A + q] = bw * st[s];
                    y[(size_t)3 * A + q] = bh * st[s];
                    for (int c = 0; c < nc; c++) {
                        float v = cl->buf[slot_idx(cl, c, (size_t)p)];
                        y[(size_t)(4 + c) * A + q] = 1.f / (1.f + expf(-v));
                    }
                }
                base += h * w;
            }
            *final_slot = o->out; *final_c = C; *final_a = A;
            break;
        }
        default:
            fprintf(stderr, "[YOLO] unknown op %u @op%d\n", o->op, i);
            return -1;
        }
#ifdef YOLO_PRECISION_GATE_CB
        if (g_yolo_layer_cb && o->out < (uint32_t)m->n_slots && S[o->out].buf) {
            g_yolo_layer_cb(i, o->op, o->out, S[o->out].buf,
                           S[o->out].c, S[o->out].h, S[o->out].w,
                           g_yolo_layer_cb_ud);
        }
#endif
        if (cpu_tim) {
            const double dt = ynow() - _op0;
            if (o->op == YOP_CONV || o->op == YOP_MATMUL) { m->t_conv += dt; m->n_conv_ops++; }
            else if (o->op == YOP_DETECT){ m->t_detect += dt; }
            else if (o->op == YOP_ATTN_PSA) { m->t_attn += dt; }
            else                        { m->t_elem  += dt; m->n_elem_ops++; }
            if (o->op <= YOP_MAX) { m->t_opk[o->op] += dt; m->n_opk[o->op]++; }
        }
    }
    m->view_step = m->n_ops;
    m->view_complete = 1;
#if ST_HAVE_RVV
    if (yime_timing)
        fprintf(stderr, "[YOLO-IME] kernel_sum=%.3fms epi_sum=%.3fms wall=%.3fms\n",
                yime_kernel_sum * 1e3, yime_epi_sum * 1e3, yime_wall * 1e3);
#endif
    if (cpu_tim) {
        m->t_fwd = ynow() - _fwd0;
        fprintf(stderr, "[YOLO-CPU] forward breakdown: total=%.1fms | conv=%.1fms (%ld ops, "
                        "im2col=%.1fms) elem=%.1fms (%ld ops) attn=%.1fms detect=%.1fms | fwd_other=%.1fms\n",
                m->t_fwd * 1e3, m->t_conv * 1e3, m->n_conv_ops, m->t_i2c * 1e3,
                m->t_elem * 1e3, m->n_elem_ops, m->t_attn * 1e3, m->t_detect * 1e3,
                (m->t_fwd - m->t_conv - m->t_elem - m->t_attn - m->t_detect) * 1e3);
#if ST_HAVE_RVV
        if (yime_acc_conv > 0.0)
            fprintf(stderr, "[YOLO-CPU] IME: kernel+epi=%.1fms（打包已并入 im2col）\n",
                    yime_acc_conv * 1e3);
#endif
        fprintf(stderr, "[YOLO-CPU] im2col 拆分: CHW→HWC=%.1fms  gather+量化=%.1fms\n",
                yacc_trans * 1e3, yacc_gather * 1e3);
        {   /* 逐算子类型（Phase 2 定位） */
            static const char *nm[YOP_MAX + 1] = {
                "-", "CONV", "ADD", "CONCAT", "MAXPOOL",
                "UPSAMPLE", "SIGMOID", "MUL", "SLICE",
                "SOFTMAX", "ATTN_PSA", "DETECT",
                "MATMUL", "RMSNORM", "ROPE", "SWIGLU"
            };
            fprintf(stderr, "[YOLO-CPU] ops:");
            for (int k = 1; k <= YOP_MAX; k++)
                if (m->n_opk[k])
                    fprintf(stderr, " %s=%.1fms/%ld", nm[k], m->t_opk[k] * 1e3, m->n_opk[k]);
            fprintf(stderr, "\n");
        }
    }
    return 0;
}

int st_yolo_forward_raw(StYoloModel *m, const float *nchw,
                        float *out, int *out_c, int *out_a) {
    if (!m || !nchw || !out) return -1;
    YSlot *S = (YSlot *)calloc((size_t)m->n_slots, sizeof(YSlot));
    if (!S) return -1;
    uint32_t fs = 0; int fc = 0, fa = 0;
    int rc = yolo_forward(m, nchw, (int)m->det.input_w, (int)m->det.input_h,
                          S, &fs, &fc, &fa);
    YSlot v;
    if (rc == 0) rc = slot_view(&S[fs], 0, &v);
    if (rc == 0) {
        memcpy(out, v.buf, (size_t)fc * fa * sizeof(float));
        if (out_c) *out_c = fc;
        if (out_a) *out_a = fa;
    }
    slots_free(S, m->n_slots);
    return rc;
}

/* ================================================================
 * letterbox + 后处理（NMS）
 * ================================================================ */
/* letterbox（rect/auto=True 语义，对齐 ultralytics predict）：
 *   缩放到长边 = tw/th，再只补到 stride 倍数（最小矩形，不补成正方形）。
 * 输出尺寸 (oh, ow) 为 stride 倍数；返回 malloc 的 CHW float（[0,1]，114/255 填充）。 */
static float *letterbox_alloc(const uint8_t *rgb, int w, int h, int tw, int th, int stride,
                              int *out_w, int *out_h, float *scale,
                              float *padx, float *pady) {
    float r = fminf((float)tw / w, (float)th / h);
    int nw = (int)lroundf(w * r), nh = (int)lroundf(h * r);
    if (nw < 1) nw = 1; if (nh < 1) nh = 1;
    int dw = tw - nw, dh = th - nh;
    dw %= stride; dh %= stride;                    /* rect：只补到 stride 倍数 */
    int left = (int)lroundf(dw * 0.5f - 0.1f), right = (int)lroundf(dw * 0.5f + 0.1f);
    int top  = (int)lroundf(dh * 0.5f - 0.1f), bottom = (int)lroundf(dh * 0.5f + 0.1f);
    int ow = nw + left + right, oh = nh + top + bottom;
    float *out = (float *)malloc((size_t)3 * oh * ow * sizeof(float));
    if (!out) return NULL;
    size_t n = (size_t)3 * oh * ow;
    for (size_t i = 0; i < n; i++) out[i] = 114.f / 255.f;
    /* K-YOLO-002 Phase 2：x/y 映射只与 x（或 y）有关，预计算一次——原码把 x 映射
     * 放在每 y 行内重算整整 nh 次（nh×nw 次多余的除/floor/地址算）。同时把 c 提到
     * 最外层，使每个 (c,y) 行的写出连续（原码 c 内层为跨 oh·ow 的 3 路散射写）。
     * 逐点浮点表达式不变 ⇒ **逐位一致**。 */
    int   *xm = (int *)malloc((size_t)nw * 2 * sizeof(int));
    float *xf = (float *)malloc((size_t)nw * sizeof(float));
    int   *ym = (int *)malloc((size_t)nh * 2 * sizeof(int));
    float *yf = (float *)malloc((size_t)nh * sizeof(float));
    if (!xm || !xf || !ym || !yf) {
        free(xm); free(xf); free(ym); free(yf); free(out); return NULL;
    }
    for (int x = 0; x < nw; x++) {
        float sx = (x + 0.5f) * w / nw - 0.5f;
        int x0 = (int)floorf(sx); float fx = sx - x0;
        int x1 = x0 + 1; if (x0 < 0) x0 = 0; if (x1 > w - 1) x1 = w - 1;
        xm[x * 2] = x0; xm[x * 2 + 1] = x1; xf[x] = fx;
    }
    for (int y = 0; y < nh; y++) {
        float sy = (y + 0.5f) * h / nh - 0.5f;
        int y0 = (int)floorf(sy); float fy = sy - y0;
        int y1 = y0 + 1; if (y0 < 0) y0 = 0; if (y1 > h - 1) y1 = h - 1;
        ym[y * 2] = y0; ym[y * 2 + 1] = y1; yf[y] = fy;
    }
    const int crow = 3 * nh;    /* (c,y) 行数：行间写不相交 ⇒ 可并行 */
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(crow > 64)
#endif
    for (int ci = 0; ci < crow; ci++) {
        const int c = ci / nh, y = ci - c * nh;
        const int y0 = ym[y * 2], y1 = ym[y * 2 + 1];
        const float fy = yf[y], nfy = 1.f - fy;
        const uint8_t *r0 = rgb + (size_t)y0 * w * 3 + c;
        const uint8_t *r1 = rgb + (size_t)y1 * w * 3 + c;
        float *orow = out + ((size_t)c * oh + (y + top)) * ow + left;
        for (int x = 0; x < nw; x++) {
            const int x0 = xm[x * 2], x1 = xm[x * 2 + 1];
            const float fx = xf[x], nfx = 1.f - fx;
            float p00 = r0[(size_t)x0 * 3];
            float p01 = r0[(size_t)x1 * 3];
            float p10 = r1[(size_t)x0 * 3];
            float p11 = r1[(size_t)x1 * 3];
            float v = p00 * nfy * nfx + p01 * nfy * fx +
                      p10 * fy * nfx + p11 * fy * fx;
            orow[x] = v / 255.f;
        }
    }
    free(xm); free(xf); free(ym); free(yf);
    *out_w = ow; *out_h = oh; *scale = r; *padx = (float)left; *pady = (float)top;
    return out;
}

static float iou_xyxy(const float *a, const float *b) {
    float x1 = fmaxf(a[0], b[0]), y1 = fmaxf(a[1], b[1]);
    float x2 = fminf(a[2], b[2]), y2 = fminf(a[3], b[3]);
    float iw = x2 - x1, ih = y2 - y1;
    if (iw <= 0 || ih <= 0) return 0.f;
    float inter = iw * ih;
    float ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
    return inter / ua;
}

typedef struct { float s; int i; } Cand;
/* 升序（qsort 约定：a<b 返回负）；NMS 从数组末尾（最高分）向前遍历 */
static int cand_cmp(const void *pa, const void *pb) {
    const Cand *a = (const Cand *)pa, *b = (const Cand *)pb;
    return (a->s > b->s) - (a->s < b->s);
}

int st_yolo_detect(StYoloModel *m, const uint8_t *rgb, int w, int h,
                   float conf, float iou, YoloDet *out, int max_out) {
    if (!m || !rgb || w <= 0 || h <= 0) return -1;
    int tw = (int)m->det.input_w, th = (int)m->det.input_h;
    int nc = (int)m->det.n_classes;
    if (conf <= 0) conf = m->det.conf_thres > 0 ? m->det.conf_thres : 0.25f;
    if (iou  <= 0) iou  = m->det.iou_thres  > 0 ? m->det.iou_thres  : 0.7f;

    float *in = NULL;
    float scale = 1.f, pw = 0.f, ph = 0.f;
    int ow = 0, oh = 0;
    int stride = (int)m->det.strides[2] > 0 ? (int)m->det.strides[2] : 32;
    int cpu_tim = 0;
    { const char *e = getenv("VLLM_YOLO_CPU_TIMING");
      cpu_tim = (e && e[0] && e[0] != '0'); }
    double _l0 = cpu_tim ? ynow() : 0.0;
    in = letterbox_alloc(rgb, w, h, tw, th, stride, &ow, &oh, &scale, &pw, &ph);
    if (!in) return -1;
    if (cpu_tim) m->t_lb = ynow() - _l0;
    {   /* 诊断：转储 letterbox 后的 NCHW 输入（VLLM_YOLO_DUMP_LB=<file>） */
        const char *dp = getenv("VLLM_YOLO_DUMP_LB");
        if (dp && dp[0]) {
            FILE *df = fopen(dp, "wb");
            if (df) { fwrite(in, sizeof(float), (size_t)3 * oh * ow, df); fclose(df); }
        }
    }

    YSlot *S = (YSlot *)calloc((size_t)m->n_slots, sizeof(YSlot));
    if (!S) { free(in); return -1; }
    uint32_t fs = 0; int fc = 0, A = 0;
    int rc = yolo_forward(m, in, ow, oh, S, &fs, &fc, &A);
    free(in);
    if (rc != 0) { slots_free(S, m->n_slots); return -1; }
    double _p0 = cpu_tim ? ynow() : 0.0;
    const float *Y = S[fs].buf;

    /* 每 anchor 取最大类 */
    Cand *cand = (Cand *)malloc((size_t)A * sizeof(Cand));
    int ncan = 0;
    float *boxes = (float *)malloc((size_t)A * 4 * sizeof(float));
    int *clsv = (int *)malloc((size_t)A * sizeof(int));
    if (!cand || !boxes || !clsv) {
        free(cand); free(boxes); free(clsv); slots_free(S, m->n_slots); return -1;
    }
    for (int q = 0; q < A; q++) {
        float best = -1.f; int bc = 0;
        for (int c = 0; c < nc; c++) {
            float v = Y[(size_t)(4 + c) * A + q];
            if (v > best) { best = v; bc = c; }
        }
        if (best < conf) continue;
        float cx = Y[(size_t)0 * A + q], cy = Y[(size_t)1 * A + q];
        float bw = Y[(size_t)2 * A + q], bh = Y[(size_t)3 * A + q];
        float *bx = boxes + (size_t)ncan * 4;
        bx[0] = cx - bw * 0.5f; bx[1] = cy - bh * 0.5f;
        bx[2] = cx + bw * 0.5f; bx[3] = cy + bh * 0.5f;
        clsv[ncan] = bc;
        cand[ncan].s = best; cand[ncan].i = ncan;
        ncan++;
    }
    qsort(cand, (size_t)ncan, sizeof(Cand), cand_cmp);   /* 升序 */
    if (getenv("VLLM_YOLO_DEBUG")) {
        fprintf(stderr, "[YOLO-dbg] ncan=%d conf=%.3f iou=%.3f\n", ncan, conf, iou);
        for (int t = 0; t < 5 && t < ncan; t++) {
            int ii = cand[ncan - 1 - t].i;
            fprintf(stderr, "[YOLO-dbg]   #%d score=%.4f cls=%d box640=[%.2f,%.2f,%.2f,%.2f]\n",
                    t, cand[ncan - 1 - t].s, clsv[ii],
                    boxes[ii*4], boxes[ii*4+1], boxes[ii*4+2], boxes[ii*4+3]);
        }
    }

    /* class-wise NMS（按 score 降序） */
    char *dead = (char *)calloc((size_t)(ncan > 0 ? ncan : 1), 1);
    if (!dead) {
        free(cand); free(boxes); free(clsv); slots_free(S, m->n_slots); return -1;
    }
    int nout = 0;
    for (int a = ncan - 1; a >= 0 && nout < max_out; a--) {
        int ia = cand[a].i;
        if (dead[ia]) continue;
        for (int b = a - 1; b >= 0; b--) {
            int ib = cand[b].i;
            if (dead[ib] || clsv[ib] != clsv[ia]) continue;
            if (iou_xyxy(boxes + (size_t)ia * 4, boxes + (size_t)ib * 4) > iou)
                dead[ib] = 1;
        }
        float *bx = boxes + (size_t)ia * 4;
        YoloDet *d = &out[nout++];
        /* letterbox 反变换 → 原图坐标 */
        d->x1 = (bx[0] - pw) / scale; d->y1 = (bx[1] - ph) / scale;
        d->x2 = (bx[2] - pw) / scale; d->y2 = (bx[3] - ph) / scale;
        if (d->x1 < 0) d->x1 = 0; if (d->y1 < 0) d->y1 = 0;
        if (d->x2 > w) d->x2 = (float)w; if (d->y2 > h) d->y2 = (float)h;
        d->score = cand[a].s;
        d->cls = clsv[ia];
    }
    free(cand); free(boxes); free(clsv); free(dead);
    slots_free(S, m->n_slots);
    if (cpu_tim) {
        m->t_post = ynow() - _p0;
        fprintf(stderr, "[YOLO-CPU] letterbox=%.1fms post(NMS)=%.1fms | fwd_total=%.1fms "
                        "| image_total=%.1fms (fwd+lb+post)\n",
                m->t_lb * 1e3, m->t_post * 1e3, m->t_fwd * 1e3,
                (m->t_fwd + m->t_lb + m->t_post) * 1e3);
    }
    return nout;
}

/* ---------- P1 诊断接口 ---------- */
long st_yolo_nw_fallback_count(const StYoloModel *m) {
    return m ? m->nw_fallback_count : 0;
}
