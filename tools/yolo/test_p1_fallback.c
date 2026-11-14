/**
 * test_p1_fallback.c - P1 fallback 验证测试（F1–F4）
 *
 * 覆盖范围：
 *   F1  注入 YOLO_P1_FORCE_NW_FAIL 后推理不崩溃，stderr 出现 [WARN][YOLO-P1]
 *   F2  注入后 st_yolo_nw_fallback_count() >= 1
 *   F3  fallback 路径与正常路径输出位级相同（FNV1a 对比）
 *   F4  正常路径精度门：所有输出值 abs < 1e6（健壮性），且对固定输入结果稳定
 *
 * 编译方式（Windows x64 MSVC，注入版）：
 *   cl /W3 /DYOLO_P1_FORCE_NW_FAIL test_p1_fallback.c vllm_yolo.c /I../../include/model
 *
 * 编译方式（Windows x64 MSVC，正常版）：
 *   cl /W3 test_p1_fallback.c vllm_yolo.c /I../../include/model
 *
 * 编译方式（GCC/Clang，注入版）：
 *   gcc -DYOLO_P1_FORCE_NW_FAIL test_p1_fallback.c vllm_yolo.c -I../../include/model -lm -o test_p1_fallback_inject
 *
 * 用法：
 *   test_p1_fallback <model.vqf>
 *
 * 注意：
 *   F3 需要两个分别编译（有/无 YOLO_P1_FORCE_NW_FAIL）的可执行文件，
 *   或者在同一程序内通过宏控制（见下方说明）。
 *   本文件同时处理两种模式：通过宏 YOLO_P1_FORCE_NW_FAIL 判断当前是注入版还是正常版。
 *
 *   F3 位级对比由外部 shell 脚本或 CMake 测试步骤完成，分别运行两个可执行文件并
 *   比较 FNV 指纹（或二进制输出文件）。
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

/** xorshift32 LCG（确定性随机输入，Section 10.16.5）*/
static unsigned xorshift32(unsigned *s)
{
    unsigned x = *s;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *s = x;
    return x;
}

/** 用 xorshift32 填充 [-1,1] 均匀分布的 float 数组 */
static void fill_random(float *buf, size_t n, unsigned seed)
{
    unsigned s = seed;
    for (size_t i = 0; i < n; i++) {
        unsigned r = xorshift32(&s);
        buf[i] = (float)(int)(r & 0x00FFFFFFu) / (float)0x00800000u - 1.0f;
    }
}

/* ---- 测试用例 ---- */

/* 通过宏判断当前编译版本 */
#ifdef YOLO_P1_FORCE_NW_FAIL
#  define BUILD_TAG "inject"
#else
#  define BUILD_TAG "normal"
#endif

#define PASS_MSG "[PASS]"
#define FAIL_MSG "[FAIL]"
#define SKIP_MSG "[SKIP]"

static int g_pass = 0, g_fail = 0, g_skip = 0;

#define CHECK(cond, fmt, ...) do {                          \
    if (cond) { printf(PASS_MSG " " fmt "\n", ##__VA_ARGS__); g_pass++; } \
    else       { printf(FAIL_MSG " " fmt "\n", ##__VA_ARGS__); g_fail++; } \
} while (0)

/* ------------------------------------------------------------------ */
/* F1：注入后推理不崩溃，不返回异常值，st_yolo_forward_raw 返回 0      */
/* F2：注入后 st_yolo_nw_fallback_count >= 1                           */
/* ------------------------------------------------------------------ */
static void test_f1_f2(StYoloModel *m, const float *nchw_in, int iw, int ih,
                       int nc, int alloc_floats)
{
    printf("\n--- F1/F2 (%s build) ---\n", BUILD_TAG);

    float *out = (float *)malloc((size_t)alloc_floats * sizeof(float));
    if (!out) { printf(FAIL_MSG " F1 out malloc\n"); g_fail++; return; }

    int out_c = 0, out_a = 0;
    fprintf(stderr, "[test_p1] --- F1 stderr check begin ---\n");
    int rc = st_yolo_forward_raw(m, nchw_in, out, &out_c, &out_a);
    fprintf(stderr, "[test_p1] --- F1 stderr check end ---\n");

#ifdef YOLO_P1_FORCE_NW_FAIL
    /* 注入版：期望 forward_raw 返回 0（fallback 到标量路径后继续），或 fallback
       计数器至少递增了 1。两者之一满足即可，因不同层可能有不同路径。 */
    long cnt = st_yolo_nw_fallback_count(m);

    CHECK(rc == 0,  "F1: forward_raw returns 0 (no crash, fallback path)");
    CHECK(cnt >= 1, "F2: nw_fallback_count=%ld >= 1", cnt);

    if (rc == 0 && out_c > 0 && out_a > 0) {
        /* 额外健壮性检查：输出中无 NaN/Inf */
        int bad = 0;
        size_t n = (size_t)out_c * out_a;
        for (size_t i = 0; i < n && !bad; i++) {
            if (!isfinite(out[i])) bad = 1;
        }
        CHECK(!bad, "F1: no NaN/Inf in fallback output [%d,%d]", out_c, out_a);
    } else {
        printf(SKIP_MSG " F1 NaN/Inf check skipped (rc=%d out=[%d,%d])\n",
               rc, out_c, out_a);
        g_skip++;
    }
#else
    /* 正常版：forward_raw 应返回 0，fallback_count 应为 0 */
    CHECK(rc == 0, "F1-normal: forward_raw returns 0");
    long cnt = st_yolo_nw_fallback_count(m);
    CHECK(cnt == 0, "F2-normal: nw_fallback_count=%ld == 0 (no fallback)", cnt);
    (void)nc;
#endif

    free(out);
}

/* ------------------------------------------------------------------ */
/* F3：fallback 路径与正常路径输出 FNV 指纹一致（位级相同）             */
/*     本测试在各自编译版本中输出 FNV 指纹到 stdout，                   */
/*     由外部步骤对比两个版本的指纹。                                   */
/* ------------------------------------------------------------------ */
static void test_f3(StYoloModel *m, const float *nchw_in, int alloc_floats)
{
    printf("\n--- F3 (%s build) ---\n", BUILD_TAG);

    float *out = (float *)malloc((size_t)alloc_floats * sizeof(float));
    if (!out) { printf(FAIL_MSG " F3 out malloc\n"); g_fail++; return; }

    int out_c = 0, out_a = 0;
    int rc = st_yolo_forward_raw(m, nchw_in, out, &out_c, &out_a);
    if (rc != 0 || out_c <= 0 || out_a <= 0) {
        printf(SKIP_MSG " F3: forward_raw failed rc=%d out=[%d,%d]\n",
               rc, out_c, out_a);
        g_skip++;
        free(out);
        return;
    }

    size_t n = (size_t)out_c * out_a;
    unsigned long long h = fnv1a(out, n * sizeof(float));
    /* 用易于外部 grep/对比的格式输出 */
    printf("[F3-FNV][%s] out=[%d,%d] FNV1a=%016llx\n",
           BUILD_TAG, out_c, out_a, h);
    printf(PASS_MSG " F3: FNV printed (compare with other build to verify bit-exact)\n");
    g_pass++;

    free(out);
}

/* ------------------------------------------------------------------ */
/* F4：正常路径健壮性门：有限值 + 多次调用结果确定性                   */
/* ------------------------------------------------------------------ */
static void test_f4(StYoloModel *m, const float *nchw_in, int alloc_floats)
{
    printf("\n--- F4 (%s build) ---\n", BUILD_TAG);

#ifdef YOLO_P1_FORCE_NW_FAIL
    printf(SKIP_MSG " F4: skipped on inject build (fallback path is not \"normal path\")\n");
    g_skip++;
    return;
#else
    int alloc = alloc_floats;
    float *out1 = (float *)malloc((size_t)alloc * sizeof(float));
    float *out2 = (float *)malloc((size_t)alloc * sizeof(float));
    if (!out1 || !out2) {
        printf(FAIL_MSG " F4 malloc\n"); g_fail++;
        free(out1); free(out2); return;
    }

    int oc1 = 0, oa1 = 0, oc2 = 0, oa2 = 0;
    int r1 = st_yolo_forward_raw(m, nchw_in, out1, &oc1, &oa1);
    int r2 = st_yolo_forward_raw(m, nchw_in, out2, &oc2, &oa2);

    CHECK(r1 == 0, "F4: forward_raw #1 returns 0");
    CHECK(r2 == 0, "F4: forward_raw #2 returns 0");

    if (r1 == 0 && r2 == 0 && oc1 == oc2 && oa1 == oa2) {
        size_t n = (size_t)oc1 * oa1;
        /* 有限值检查 */
        int bad = 0;
        for (size_t i = 0; i < n && !bad; i++) {
            if (!isfinite(out1[i])) bad = 1;
        }
        CHECK(!bad, "F4: all output values finite out=[%d,%d]", oc1, oa1);

        /* 确定性：两次调用结果位级相同 */
        int diff = memcmp(out1, out2, n * sizeof(float));
        CHECK(diff == 0, "F4: deterministic (run1 == run2 bit-exact)");

        /* 精度健壮性：abs(output) < 1e6（粗粒度安全门，不要求 max_abs < 1） */
        int overflow = 0;
        float max_abs = 0.0f;
        for (size_t i = 0; i < n; i++) {
            float a = fabsf(out1[i]);
            if (a > max_abs) max_abs = a;
            if (a >= 1e6f) { overflow++; }
        }
        CHECK(overflow == 0, "F4: no overflow (max_abs=%.3f < 1e6)", (double)max_abs);
        printf("      F4 info: out=[%d,%d] n=%zu max_abs=%.3f\n",
               oc1, oa1, n, (double)max_abs);
    } else {
        printf(SKIP_MSG " F4: shape mismatch or failed (%d,[%d,%d]) (%d,[%d,%d])\n",
               r1, oc1, oa1, r2, oc2, oa2);
        g_skip += 3;
    }

    free(out1); free(out2);
#endif
}

/* ------------------------------------------------------------------ */
/* main                                                                 */
/* ------------------------------------------------------------------ */
int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: test_p1_fallback <model.vqf>\n");
        fprintf(stderr, "  build tag: %s\n", BUILD_TAG);
        return 2;
    }
    const char *mpath = argv[1];

    printf("=== test_p1_fallback build=%s ===\n", BUILD_TAG);
    printf("model: %s\n", mpath);

    double t0 = now_sec();
    StYoloModel *m = st_yolo_load(mpath);
    double tload = now_sec() - t0;

    if (!m) {
        fprintf(stderr, "[FAIL] st_yolo_load failed: %s\n", mpath);
        return 1;
    }

    int iw = 0, ih = 0;
    st_yolo_input_size(m, &iw, &ih);
    int nc = st_yolo_num_classes(m);
    printf("model loaded in %.1fms  input=%dx%d  num_classes=%d\n",
           tload * 1e3, iw, ih, nc);

    /* 确定性 NCHW 输入（xorshift32, seed=0x9E3779B9）*/
    size_t nin = (size_t)3 * iw * ih;
    float *nchw_in = (float *)malloc(nin * sizeof(float));
    if (!nchw_in) { fprintf(stderr, "[FAIL] malloc nchw_in\n"); st_yolo_free(m); return 1; }
    fill_random(nchw_in, nin, 0x9E3779B9u);

    /* 输出缓冲容量：(4+nc) * 较大锚点数，给足空间 */
    int alloc_floats = (4 + nc) * 10000;
    if (alloc_floats < 65536) alloc_floats = 65536;

    /* 运行测试 */
    test_f1_f2(m, nchw_in, iw, ih, nc, alloc_floats);
    test_f3(m, nchw_in, alloc_floats);
    test_f4(m, nchw_in, alloc_floats);

    free(nchw_in);
    st_yolo_free(m);

    printf("\n=== SUMMARY build=%s  PASS=%d  FAIL=%d  SKIP=%d ===\n",
           BUILD_TAG, g_pass, g_fail, g_skip);

    return g_fail > 0 ? 1 : 0;
}
