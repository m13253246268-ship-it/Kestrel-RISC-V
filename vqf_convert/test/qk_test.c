/* ================================================================
 * qk_test.c - 量化内核一致性 harness（两端共用同一源码）
 *
 * 确定性 LCG 生成 f32 权重（含全零块与极端块，触发 scale guard），
 * 走量化 + 各 repack，逐段输出 FNV-1a 64 哈希。
 * 对拍：x86（qk_repack.c 标量）与 aarch64（qk_ref_engine.c NEON）
 * 两个编译产物打印的哈希必须逐行一致（位级一致红线）。
 * ================================================================ */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ---- 量化内核（quant_kernels.c）---- */
void f32_to_q8_0(uint8_t *out, const float *in, int n);
void f32_to_q4_0(uint8_t *out, const float *in, int n);
void f32_to_q4i8(uint8_t *out, const float *in, int n);
void f32_to_g256q8(uint8_t *out, const float *in, int n);

/* ---- repack（x86 用标量 qk_repack.c / 板端用 NEON qk_ref_engine.c）---- */
int repack_q4_0_4x4_inplace(uint8_t *buf, int rows, int cols);
int repack_q4_0_lane_inplace(uint8_t *buf, int rows, int cols);
int repack_q4_0_lane_col_inplace(uint8_t *buf, int rows, int cols);
int repack_q8_0_8x8_inplace(uint8_t *buf, int rows, int cols);
int repack_q8_0_4x4_inplace(uint8_t *buf, int rows, int cols);
int repack_q8_0_tiled_inplace(uint8_t *buf, int rows, int cols);
int repack_q4_0_8x8l(uint8_t *dst, const uint8_t *src, int rows, int cols);
int repack_conv_q8_hwc_tap(int8_t *dst, const int8_t *src, int OC, int C, int KH, int KW);
int unpack_conv_q8_hwc_tap(int8_t *dst, const int8_t *src, int OC, int C, int KH, int KW);
int repack_conv_q8_ime_k1(int8_t *dst, const int8_t *src_hwc_kp, int OC, int Kp);
int unpack_conv_q8_ime_k1(int8_t *dst_hwc_kp, const int8_t *src_ime, int OC, int Kp);

#define Q8B(n) (((size_t)(n) + 31) / 32 * 34)
#define Q4B(n) (((size_t)(n) + 31) / 32 * 18)

static uint64_t fnv(const void *p, size_t n) {
    const uint8_t *b = (const uint8_t *)p;
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 0x100000001b3ull; }
    return h;
}

/* 确定性 LCG（xorshift32），序列固定，跨平台一致 */
static uint32_t g_seed = 0x9E3779B9u;
static uint32_t lcg(void) {
    uint32_t x = g_seed;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    g_seed = x;
    return x;
}

/* ---- Conv shape boundary roundtrip helper ----
 * HWC roundtrip always tested; IME roundtrip tested when OC%8==0.
 * When OC%8!=0, verifies IME repack returns -1 (rejects non-aligned OC).
 * Returns 0 on success, nonzero on failure. */
static int conv_boundary_test(const char *label, int OC, int C, int KH, int KW) {
    int K = C * KH * KW;
    int Kp = (K + 31) & ~31;
    size_t cn = (size_t)OC * K;
    size_t hwc_bytes = (size_t)OC * Kp;
    int8_t *src = (int8_t *)malloc(cn);
    int8_t *hwc = (int8_t *)malloc(hwc_bytes);
    int8_t *hwc2 = (int8_t *)malloc(hwc_bytes);
    int8_t *ime = (int8_t *)malloc(hwc_bytes);
    int8_t *rec = (int8_t *)malloc(cn);
    if (!src || !hwc || !hwc2 || !ime || !rec) {
        free(src); free(hwc); free(hwc2); free(ime); free(rec);
        printf("%-14s OOM\n", label);
        return -2;
    }
    for (size_t i = 0; i < cn; i++) src[i] = (int8_t)((int)(lcg() & 0xFFu) - 128);
    int fail = 0;

    /* HWC tap-major roundtrip */
    if (repack_conv_q8_hwc_tap(hwc, src, OC, C, KH, KW) != 0 ||
        unpack_conv_q8_hwc_tap(rec, hwc, OC, C, KH, KW) != 0 ||
        memcmp(src, rec, cn) != 0) {
        printf("%-14s hwc FAIL\n", label);
        fail = 1;
    } else {
        printf("%-14s hwc %016llx\n", label, (unsigned long long)fnv(hwc, hwc_bytes));
    }

    /* IME K1 roundtrip (only when OC%8==0; Kp is always %32 hence %8) */
    if (OC % 8 == 0) {
        memset(hwc2, 0, hwc_bytes);
        memset(rec, 0, cn);
        if (repack_conv_q8_ime_k1(ime, hwc, OC, Kp) != 0 ||
            unpack_conv_q8_ime_k1(hwc2, ime, OC, Kp) != 0 ||
            memcmp(hwc, hwc2, hwc_bytes) != 0 ||
            unpack_conv_q8_hwc_tap(rec, hwc2, OC, C, KH, KW) != 0 ||
            memcmp(src, rec, cn) != 0) {
            printf("%-14s ime FAIL\n", label);
            fail = 1;
        } else {
            printf("%-14s ime %016llx\n", label, (unsigned long long)fnv(ime, hwc_bytes));
        }
    } else {
        /* Verify IME rejects non-aligned OC */
        if (repack_conv_q8_ime_k1(ime, hwc, OC, Kp) != -1) {
            printf("%-14s ime should reject OC%%8!=0 but didn't\n", label);
            fail = 1;
        } else {
            printf("%-14s ime reject OK (OC%%8!=0)\n", label);
        }
    }

    free(src); free(hwc); free(hwc2); free(ime); free(rec);
    return fail ? -1 : 0;
}

int main(void) {
    const int rows = 96;     /* %8==0 && %4==0 */
    const int cols = 4096;   /* %32==0 */
    size_t n = (size_t)rows * cols;

    float *w = (float *)malloc(n * sizeof(float));
    uint8_t *q8 = (uint8_t *)malloc(Q8B(n));
    uint8_t *q4 = (uint8_t *)malloc(Q4B(n));
    uint8_t *q4i = (uint8_t *)malloc(Q8B(n));
    uint8_t *x8 = (uint8_t *)malloc(Q4B(n));
    if (!w || !q8 || !q4 || !q4i || !x8) { fprintf(stderr, "OOM\n"); return 2; }

    /* 确定性权重：[-1,1] 混合；行 32-39 全零（scale guard）；行 80 置大值 */
    for (size_t i = 0; i < n; i++) {
        uint32_t u = lcg();
        float v = (float)(u >> 8) / 16777215.0f * 2.0f - 1.0f;
        w[i] = v;
    }
    for (int c = 0; c < cols; c++) w[(size_t)32 * cols + c] = 0.0f;
    for (int c = 0; c < cols; c++) w[(size_t)39 * cols + c] = 0.0f;
    for (int c = 0; c < cols; c++) w[(size_t)80 * cols + c] = (float)(c % 7) - 1000.0f;

    int fail = 0;
    /* 1) q8_0 量化 + tiled(默认8x8) repack */
    f32_to_q8_0(q8, w, (int)n);
    printf("q8_0         %016llx\n", (unsigned long long)fnv(q8, Q8B(n)));
    f32_to_q8_0(q8, w, (int)n);   /* repack in-place on copy */
    if (repack_q8_0_tiled_inplace(q8, rows, cols) != 0) {
        printf("q8 tiled     repack rc=-1 (geometry?)\n");
        fail = 1;
    } else {
        printf("q8 tiled     %016llx\n", (unsigned long long)fnv(q8, Q8B(n)));
    }
    /* 2) q8_0 4x4 repack */
    f32_to_q8_0(q8, w, (int)n);
    if (repack_q8_0_4x4_inplace(q8, rows, cols) != 0) {
        printf("q8 4x4       repack rc=-1\n");
        fail = 1;
    } else {
        printf("q8 4x4       %016llx\n", (unsigned long long)fnv(q8, Q8B(n)));
    }
    /* 3) q4_0 量化 + 4x4 / lane / col repack */
    f32_to_q4_0(q4, w, (int)n);
    printf("q4_0         %016llx\n", (unsigned long long)fnv(q4, Q4B(n)));
    f32_to_q4_0(q4, w, (int)n);
    if (repack_q4_0_4x4_inplace(q4, rows, cols) != 0) {
        printf("q4 4x4       repack rc=-1\n");
        fail = 1;
    } else {
        printf("q4 4x4       %016llx\n", (unsigned long long)fnv(q4, Q4B(n)));
    }
    f32_to_q4_0(q4, w, (int)n);
    if (repack_q4_0_lane_inplace(q4, rows, cols) != 0) {
        printf("q4 lane      repack rc=-1\n");
        fail = 1;
    } else {
        printf("q4 lane      %016llx\n", (unsigned long long)fnv(q4, Q4B(n)));
    }
    f32_to_q4_0(q4, w, (int)n);
    if (repack_q4_0_lane_col_inplace(q4, rows, cols) != 0) {
        printf("q4 col       repack rc=-1\n");
        fail = 1;
    } else {
        printf("q4 col       %016llx\n", (unsigned long long)fnv(q4, Q4B(n)));
    }
    /* 4) q4_0 -> x8 8x8l 副本 */
    f32_to_q4_0(q4, w, (int)n);
    if (repack_q4_0_8x8l(x8, q4, rows, cols) != 0) {
        printf("x8 8x8l      repack rc=-1\n");
        fail = 1;
    } else {
        printf("x8 8x8l      %016llx\n", (unsigned long long)fnv(x8, Q4B(n)));
    }
    /* 5) q4i8 预解包 */
    f32_to_q4i8(q4i, w, (int)n);
    printf("q4i8         %016llx\n", (unsigned long long)fnv(q4i, Q8B(n)));
    /* 6) g256 */
    f32_to_g256q8(q8, w, (int)n);
    printf("g256         %016llx\n", (unsigned long long)fnv(q8, Q8B(n)));

    /* 7) Conv Q8 HWC tap-major + 32B Kp 补零重排 & 双射逆变换恒等校验 */
    {
        const int coc = 48, cic = 20, ckh = 3, ckw = 3; /* K=180 -> Kp=192 */
        const int ckp = (cic * ckh * ckw + 31) & ~31;
        const size_t cn = (size_t)coc * cic * ckh * ckw;
        const size_t hwc_bytes = (size_t)coc * ckp;
        int8_t *csrc = (int8_t *)malloc(cn);
        int8_t *chwc = (int8_t *)malloc(hwc_bytes);
        int8_t *chwc2 = (int8_t *)malloc(hwc_bytes);
        int8_t *cime = (int8_t *)malloc(hwc_bytes);
        int8_t *crec = (int8_t *)malloc(cn);
        for (size_t i = 0; i < cn; i++) csrc[i] = (int8_t)((int)(lcg() & 0xFFu) - 128);
        if (repack_conv_q8_hwc_tap(chwc, csrc, coc, cic, ckh, ckw) != 0 ||
            unpack_conv_q8_hwc_tap(crec, chwc, coc, cic, ckh, ckw) != 0 ||
            memcmp(csrc, crec, cn) != 0) {
            printf("conv hwc     roundtrip FAIL\n");
            fail = 1;
        } else {
            printf("conv hwc     %016llx\n", (unsigned long long)fnv(chwc, hwc_bytes));
        }

        /* 8) Conv Q8 SpacemiT K1 IME pack_a 重排 & 双射逆变换恒等校验 */
        memset(chwc2, 0, hwc_bytes);
        memset(crec, 0, cn);
        if (repack_conv_q8_ime_k1(cime, chwc, coc, ckp) != 0 ||
            unpack_conv_q8_ime_k1(chwc2, cime, coc, ckp) != 0 ||
            memcmp(chwc, chwc2, hwc_bytes) != 0 ||
            unpack_conv_q8_hwc_tap(crec, chwc2, coc, cic, ckh, ckw) != 0 ||
            memcmp(csrc, crec, cn) != 0) {
            printf("conv ime     roundtrip FAIL\n");
            fail = 1;
        } else {
            printf("conv ime     %016llx\n", (unsigned long long)fnv(cime, hwc_bytes));
        }
        free(csrc); free(chwc); free(chwc2); free(cime); free(crec);
    }

    /* ---- Section 10.12.2 形状边界覆盖（7 项未覆盖边界）---- */

    /* b1: OC%8!=0 (OC=5) — IME 以 8 为粒度打包，尾块处理若有误不会被 OC%8=0 检出
     *     K=180->Kp=192，HWC 应成功，IME 应拒绝 */
    if (conv_boundary_test("b1 oc%8!=0", 5, 20, 3, 3) != 0) fail = 1;

    /* b2: K%32=0 无尾块补零 (OC=16,C=4,KH=4,KW=2 -> K=32,Kp=32)
     *     补零分支不走，恒等比较不能覆盖 Kp==K 时的边界索引 */
    if (conv_boundary_test("b2 k%32=0", 16, 4, 4, 2) != 0) fail = 1;

    /* b3: K%32=0 且 OC%8!=0 组合 (OC=5,C=4,KH=4,KW=2 -> K=32,Kp=32)
     *     两个边界同时命中，HWC 成功，IME 拒绝 */
    if (conv_boundary_test("b3 k32+oc", 5, 4, 4, 2) != 0) fail = 1;

    /* b4: 1×1 卷积 KH=KW=1，K=C (OC=16,C=32 -> K=32,Kp=32)
     *     tap 只有 C 个，补零行为与 3×3 有本质差异 */
    if (conv_boundary_test("b4 1x1", 16, 32, 1, 1) != 0) fail = 1;

    /* b5: G>1 分组卷积 per-group slice (OC_g=4,C_g=1,KH=3,KW=3)
     *     分组路径走 legacy 格不经 IME repack；OC%8!=0 验证 IME 拒绝 */
    if (conv_boundary_test("b5 group", 4, 1, 3, 3) != 0) fail = 1;

    /* b6: 非 RVV 构建路径 — 确认标量 repack 在各构建目标下的路径标识
     *     repack_conv_q8_hwc_tap / ime_k1 均为纯标量实现，无 RVV/NEON intrinsics */
    {
        const char *build_mode =
#if defined(__riscv_v)
            "RVV"
#elif defined(__ARM_NEON) || defined(__aarch64__)
            "NEON"
#else
            "scalar"
#endif
            ;
        printf("b6 build       %s path\n", build_mode);
    }

    /* b7a: OC=8 极小值 — 最小合法 IME 打包粒度 (K=27->Kp=32) */
    if (conv_boundary_test("b7a oc=8", 8, 3, 3, 3) != 0) fail = 1;

    /* b7b: OC=1 极端值 — 单输出通道，HWC 成功，IME 应拒绝 */
    if (conv_boundary_test("b7b oc=1", 1, 3, 3, 3) != 0) fail = 1;

    free(w); free(q8); free(q4); free(q4i); free(x8);
    printf(fail ? "QK FAIL\n" : "QK OK\n");
    return fail ? 1 : 0;
}
