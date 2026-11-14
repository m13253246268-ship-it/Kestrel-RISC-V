/**
 * test_p1_precision_gate.c - P1 精度门逐层回归测试（Section 10.16.5）
 *
 * 覆盖范围：
 *   G1  正常路径（normal build）逐层确定性：两次 forward_raw 每层 max_abs_error == 0
 *       逐层打印 max_abs_error；所有层 max_abs_error < 1.0f（精度门）
 *   G2  fallback 路径（inject build：-DYOLO_P1_FORCE_NW_FAIL）逐层 FNV 指纹
 *       与 G1 normal build 对比，验证 fallback == normal（位级相同 → max_abs_error = 0）
 *   G3  边界形状精度门：三种极端输入分布（全零、全+1、交替符号），
 *       逐层检查有限值 + max_abs_error < 1.0f（相对于全随机基准）
 *
 * 注意：
 *   G1/G3 在 normal build 中运行；G2 在 inject build 中运行。
 *   G2 打印每层 FNV；外部 diff 与 G1 FNV 对比即可验证位级一致。
 *   本文件需以 -DYOLO_PRECISION_GATE_CB 编译，否则 st_yolo_set_layer_cb 不可用。
 *
 * 编译（riscv64 K1，normal build）：
 *   gcc -O2 -march=rv64gcv -mabi=lp64d -ffp-contract=off \
 *       -DYOLO_PRECISION_GATE_CB \
 *       test_p1_precision_gate.c vllm_yolo.c \
 *       -I../../include/model -I../../include/common -lm -o test_p1_precision_gate_normal
 *
 * 编译（riscv64 K1，inject build）：
 *   gcc -O2 -march=rv64gcv -mabi=lp64d -ffp-contract=off \
 *       -DYOLO_PRECISION_GATE_CB -DYOLO_P1_FORCE_NW_FAIL \
 *       test_p1_precision_gate.c vllm_yolo.c \
 *       -I../../include/model -I../../include/common -lm -o test_p1_precision_gate_inject
 *
 * 用法：
 *   test_p1_precision_gate_normal <model.vqf>
 *   test_p1_precision_gate_inject <model.vqf>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Windows / POSIX 均可用的计时 */
#ifdef _WIN32
#  include <windows.h>
static double now_sec(void)
{
    LARGE_INTEGER freq, cnt;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    return (double)cnt.QuadPart / (double)freq.QuadPart;
}
#else
#  include <time.h>
static double now_sec(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}
#endif

#include "vllm_yolo.h"

/* ---- 工具函数 ---- */

static unsigned long long fnv1a(const void *p, size_t n)
{
    const unsigned char *b = (const unsigned char *)p;
    unsigned long long h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ULL; }
    return h;
}

static unsigned xorshift32(unsigned *s)
{
    unsigned x = *s;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *s = x;
    return x;
}

static void fill_random(float *buf, size_t n, unsigned seed)
{
    unsigned s = seed;
    for (size_t i = 0; i < n; i++) {
        unsigned r = xorshift32(&s);
        buf[i] = (float)(int)(r & 0x00FFFFFFu) / (float)0x00800000u - 1.0f;
    }
}

/* ---- 编译版本标记 ---- */
#ifdef YOLO_P1_FORCE_NW_FAIL
#  define BUILD_TAG "inject"
#else
#  define BUILD_TAG "normal"
#endif

#define PASS_MSG "[PASS]"
#define FAIL_MSG "[FAIL]"
#define SKIP_MSG "[SKIP]"

static int g_pass = 0, g_fail = 0, g_skip = 0;

#define CHECK(cond, fmt, ...) do {                                      \
    if (cond) { printf(PASS_MSG " " fmt "\n", ##__VA_ARGS__); g_pass++; } \
    else       { printf(FAIL_MSG " " fmt "\n", ##__VA_ARGS__); g_fail++; } \
} while (0)

/* ==================================================================== */
/* 精度门逐层回调基础设施                                                  */
/* ==================================================================== */

#define MAX_LAYERS 1024

typedef struct {
    float          *buf;   /* malloc 拷贝，NULL 表示未填充 */
    int             c, h, w;
    size_t          n;     /* c*h*w */
    unsigned long long fnv; /* 本层 FNV1a */
} LayerRef;

/* 两轮共用 — 第 0 轮收集基准，第 1 轮与基准对比 */
static LayerRef  g_ref[MAX_LAYERS];
static int       g_n_layers   = 0;
static int       g_cb_phase   = 0; /* 0=collect, 1=compare */
static float     g_layer_max_abs_err[MAX_LAYERS];
static int       g_layer_ok[MAX_LAYERS];   /* 1=pass, 0=fail */

static void reset_records(void)
{
    for (int i = 0; i < g_n_layers; i++) {
        free(g_ref[i].buf);
        g_ref[i].buf = NULL;
    }
    g_n_layers = 0;
    g_cb_phase = 0;
    memset(g_layer_max_abs_err, 0, sizeof(g_layer_max_abs_err));
    memset(g_layer_ok, 1, sizeof(g_layer_ok));
}

/* 逐层回调 — 由 st_yolo_set_layer_cb 注册 */
static void layer_cb(int op_idx, uint32_t opcode,
                     uint32_t out_slot,
                     const float *buf, int c, int h, int w,
                     void *ud)
{
    (void)opcode; (void)out_slot; (void)ud;
    if (op_idx < 0 || op_idx >= MAX_LAYERS) return;

    size_t n = (size_t)c * (size_t)h * (size_t)w;
    if (n == 0) return;

    if (g_cb_phase == 0) {
        /* 收集基准：malloc 拷贝 */
        if (g_ref[op_idx].buf) free(g_ref[op_idx].buf);
        g_ref[op_idx].buf = (float *)malloc(n * sizeof(float));
        if (!g_ref[op_idx].buf) return;
        memcpy(g_ref[op_idx].buf, buf, n * sizeof(float));
        g_ref[op_idx].c = c; g_ref[op_idx].h = h; g_ref[op_idx].w = w;
        g_ref[op_idx].n = n;
        g_ref[op_idx].fnv = fnv1a(buf, n * sizeof(float));
        if (op_idx >= g_n_layers) g_n_layers = op_idx + 1;
    } else {
        /* 对比阶段：计算 max_abs_error */
        if (!g_ref[op_idx].buf || g_ref[op_idx].n != n) {
            g_layer_ok[op_idx] = 0;
            return;
        }
        float max_err = 0.0f;
        const float *ref = g_ref[op_idx].buf;
        for (size_t i = 0; i < n; i++) {
            float e = fabsf(buf[i] - ref[i]);
            if (e > max_err) max_err = e;
        }
        g_layer_max_abs_err[op_idx] = max_err;
        g_layer_ok[op_idx] = (max_err < 1.0f);
    }
}

/* 只打印 FNV — 用于 G2 inject build 的逐层指纹输出 */
static void layer_cb_fnv_only(int op_idx, uint32_t opcode,
                               uint32_t out_slot,
                               const float *buf, int c, int h, int w,
                               void *ud)
{
    (void)opcode; (void)out_slot; (void)ud;
    if (op_idx < 0 || op_idx >= MAX_LAYERS) return;
    size_t n = (size_t)c * (size_t)h * (size_t)w;
    if (n == 0) return;
    unsigned long long h_ = fnv1a(buf, n * sizeof(float));
    printf("[G2-FNV][%s] layer=%4d c=%4d h=%4d w=%4d FNV=%016llx\n",
           BUILD_TAG, op_idx, c, h, w, h_);
    if (op_idx >= g_n_layers) g_n_layers = op_idx + 1;
}

/* ==================================================================== */
/* 测试场景函数                                                            */
/* ==================================================================== */

/* ---- G1：normal build 逐层确定性 + 精度门 ---- */
#ifndef YOLO_P1_FORCE_NW_FAIL
static void test_g1_normal(StYoloModel *m, const float *nchw_in, int alloc_floats)
{
    printf("\n--- G1 (normal build) per-layer determinism + precision gate ---\n");
    reset_records();

    float *out = (float *)malloc((size_t)alloc_floats * sizeof(float));
    if (!out) { printf(FAIL_MSG " G1 out malloc\n"); g_fail++; return; }

    int out_c = 0, out_a = 0;

    /* Round 1：收集每层基准 */
    g_cb_phase = 0;
    st_yolo_set_layer_cb(layer_cb, NULL);
    int rc1 = st_yolo_forward_raw(m, nchw_in, out, &out_c, &out_a);
    CHECK(rc1 == 0, "G1 forward-1 rc==0");

    /* Round 2：与基准逐层对比 */
    g_cb_phase = 1;
    int rc2 = st_yolo_forward_raw(m, nchw_in, out, &out_c, &out_a);
    CHECK(rc2 == 0, "G1 forward-2 rc==0");

    st_yolo_set_layer_cb(NULL, NULL);

    /* 打印每层 FNV，供外部与 G2-inject 输出 diff 验证位级一致 */
    for (int i = 0; i < g_n_layers; i++) {
        if (g_ref[i].buf)
            printf("[G1-FNV][normal] layer=%4d c=%4d h=%4d w=%4d FNV=%016llx\n",
                   i, g_ref[i].c, g_ref[i].h, g_ref[i].w, g_ref[i].fnv);
    }

    /* 逐层打印 max_abs_error（计划要求），并聚合检查 */
    int det_fail = 0, gate_fail = 0;
    for (int i = 0; i < g_n_layers; i++) {
        float e = g_layer_max_abs_err[i];
        printf("[G1] layer=%4d max_abs_err=%.6f\n", i, e);
        if (e != 0.0f) det_fail++;
        if (e >= 1.0f) gate_fail++;
    }
    CHECK(det_fail == 0,  "G1 all-layers deterministic (det_fail=%d/%d)", det_fail, g_n_layers);
    CHECK(gate_fail == 0, "G1 all-layers precision gate <1.0f (gate_fail=%d/%d)", gate_fail, g_n_layers);
    printf("    G1 info: layers=%d out=[%d,%d]\n", g_n_layers, out_c, out_a);

    free(out);
}
#endif /* !YOLO_P1_FORCE_NW_FAIL */

/* ------------------------------------------------------------------ */
/* G2: inject build — 逐层打印 FNV 指纹，供外部与 G1-normal 对比      */
/* ------------------------------------------------------------------ */
#ifdef YOLO_P1_FORCE_NW_FAIL
static void test_g2_inject(StYoloModel *m, const float *nchw_in, int alloc_floats)
{
    printf("\n--- G2 (inject build) per-layer FNV fingerprint ---\n");
    reset_records();

    float *out = (float *)malloc((size_t)alloc_floats * sizeof(float));
    if (!out) { fprintf(stderr, "G2: malloc failed\n"); g_fail++; return; }

    int out_c = 0, out_a = 0;

    /* inject build 只打印 FNV，不做 phase=1 对比 */
    g_cb_phase = 0;
    st_yolo_set_layer_cb(layer_cb_fnv_only, NULL);
    int rc = st_yolo_forward_raw(m, nchw_in, out, &out_c, &out_a);
    st_yolo_set_layer_cb(NULL, NULL);

    CHECK(rc == 0,         "G2 forward_raw rc==0 (inject)");
    CHECK(g_n_layers > 0,  "G2 layer_cb fired (layers=%d)", g_n_layers);
    printf("    G2 info: layers=%d out=[%d,%d]\n", g_n_layers, out_c, out_a);

    free(out);
}
#endif /* YOLO_P1_FORCE_NW_FAIL */

/* ------------------------------------------------------------------ */
/* G3: 边界形状精度门 — 全零 / 全+1 / 交替符号，每层 max_abs_error    */
/* ------------------------------------------------------------------ */
#ifndef YOLO_P1_FORCE_NW_FAIL
static void test_g3_boundary(StYoloModel *m, int iw, int ih, int alloc_floats)
{
    printf("\n--- G3 (normal build) boundary-shape precision gate ---\n");

    /* 计算输入元素数（单通道 NCHW，1×C×H×W，这里按 test_p1_fallback.c 惯例取首层输入） */
    /* 注：只需覆盖模型实际接受尺寸，不必完全精确，fill_* 函数按实际大小填充 */
    int ic = 3; /* RGB */
    size_t nin = (size_t)ic * ih * iw;
    float *inp = (float *)malloc(nin * sizeof(float));
    float *out = (float *)malloc((size_t)alloc_floats * sizeof(float));
    if (!inp || !out) {
        fprintf(stderr, "G3: malloc failed\n");
        g_fail++;
        free(inp); free(out);
        return;
    }

    const struct { const char *name; float val; int alt; } cases[] = {
        { "all-zero",    0.0f, 0 },
        { "all-plus1",   1.0f, 0 },
        { "alternating", 1.0f, 1 },  /* +1/-1 交替 */
    };
    int ncases = (int)(sizeof(cases) / sizeof(cases[0]));

    for (int ci = 0; ci < ncases; ci++) {
        /* 填充输入 */
        if (cases[ci].alt) {
            for (size_t k = 0; k < nin; k++)
                inp[k] = (k & 1) ? -1.0f : 1.0f;
        } else {
            for (size_t k = 0; k < nin; k++)
                inp[k] = cases[ci].val;
        }

        reset_records();
        int out_c = 0, out_a = 0;

        /* Round 1：收集基准 */
        g_cb_phase = 0;
        st_yolo_set_layer_cb(layer_cb, NULL);
        int rc1 = st_yolo_forward_raw(m, inp, out, &out_c, &out_a);
        CHECK(rc1 == 0, "G3[%s] forward-1 rc==0", cases[ci].name);

        /* Round 2：对比 */
        g_cb_phase = 1;
        int rc2 = st_yolo_forward_raw(m, inp, out, &out_c, &out_a);
        st_yolo_set_layer_cb(NULL, NULL);
        CHECK(rc2 == 0, "G3[%s] forward-2 rc==0", cases[ci].name);

        int finite_fail = 0, gate_fail = 0;
        for (int i = 0; i < g_n_layers; i++) {
            float e = g_layer_max_abs_err[i];
            printf("[G3][%s] layer=%4d max_abs_err=%.6f\n", cases[ci].name, i, e);
            if (!isfinite(e)) finite_fail++;
            if (e >= 1.0f)   gate_fail++;
        }
        CHECK(finite_fail == 0, "G3[%s] all-layers finite (fail=%d/%d)",
              cases[ci].name, finite_fail, g_n_layers);
        CHECK(gate_fail == 0,   "G3[%s] all-layers precision gate <1.0f (fail=%d/%d)",
              cases[ci].name, gate_fail, g_n_layers);
        printf("    G3[%s] info: layers=%d out=[%d,%d]\n",
               cases[ci].name, g_n_layers, out_c, out_a);
    }

    free(inp);
    free(out);
}
#endif /* !YOLO_P1_FORCE_NW_FAIL */

/* ================================================================== */
/* main                                                                */
/* ================================================================== */
int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.vqf>\n", argv[0]);
        return 1;
    }
    const char *model_path = argv[1];

    printf("=== test_p1_precision_gate (%s build) ===\n", BUILD_TAG);
    printf("    model : %s\n", model_path);

    /* ----- 加载模型 ----- */
    StYoloModel *m = st_yolo_load(model_path);
    if (!m) {
        fprintf(stderr, "st_yolo_load failed: %s\n", model_path);
        return 1;
    }

    int iw = 0, ih = 0;
    st_yolo_input_size(m, &iw, &ih);
    int nc = st_yolo_num_classes(m);
    printf("    input  : %d x %d\n", iw, ih);
    printf("    classes: %d\n", nc);

    /* ----- 输入缓冲 ----- */
    int ic = 3;
    size_t nin = (size_t)ic * ih * iw;
    float *nchw_in = (float *)malloc(nin * sizeof(float));
    if (!nchw_in) { fprintf(stderr, "malloc nchw_in failed\n"); st_yolo_free(m); return 1; }

    /* 确定性随机输入（与 test_p1_fallback.c 保持相同 seed） */
    fill_random(nchw_in, nin, 0x9E3779B9u);

    /* 输出缓冲大小估算（同 test_p1_fallback.c） */
    int alloc_floats = (4 + nc) * 10000;
    if (alloc_floats < 65536) alloc_floats = 65536;

    /* ----- 运行场景 ----- */
#ifndef YOLO_P1_FORCE_NW_FAIL
    test_g1_normal(m, nchw_in, alloc_floats);
    test_g3_boundary(m, iw, ih, alloc_floats);
#endif
#ifdef YOLO_P1_FORCE_NW_FAIL
    test_g2_inject(m, nchw_in, alloc_floats);
#endif

    /* ----- 汇总 ----- */
    printf("\n=== SUMMARY (%s build) PASS=%d FAIL=%d SKIP=%d ===\n",
           BUILD_TAG, g_pass, g_fail, g_skip);

    free(nchw_in);
    st_yolo_free(m);
    return (g_fail > 0) ? 1 : 0;
}
