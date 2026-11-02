/* r10 d4：RISC-V attention RVV 核 vs 标量核 位级对拍 + 单核计时。
 * 核体与 vllm_safetensors.c 中 st_attn_qk_block8 / st_attn_softmax_vkq_scalar 同式。
 * gcc -O2 -march=rv64gcv -mabi=lp64d -ffp-contract=off bench_attnrvv.c -lm */
#include <riscv_vector.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static void qk_s(const float *q, const float *kbh, float *sc, int n, int hd, float scale) {
    for (int s = 0; s < n; s++) {
        const float *kb = kbh + (size_t)(s >> 3) * hd * 8 + (s & 7);
        float dot = 0.0f;
        for (int i = 0; i < hd; i++) dot += q[i] * kb[(size_t)i * 8];
        sc[s] = dot * scale;
    }
}
static void qk_v(const float *q, const float *kbh, float *sc, int n, int hd, float scale) {
    int s = 0, b = 0;
    for (; s + 16 <= n; s += 16, b += 2) {
        const float *k0 = kbh + (size_t)b * hd * 8, *k1 = k0 + (size_t)hd * 8;
        vfloat32m1_t a0 = __riscv_vfmv_v_f_f32m1(0.0f, 8), a1 = __riscv_vfmv_v_f_f32m1(0.0f, 8);
        for (int i = 0; i < hd; i++) {
            a0 = __riscv_vfadd_vv_f32m1(a0, __riscv_vfmul_vf_f32m1(__riscv_vle32_v_f32m1(k0 + (size_t)i * 8, 8), q[i], 8), 8);
            a1 = __riscv_vfadd_vv_f32m1(a1, __riscv_vfmul_vf_f32m1(__riscv_vle32_v_f32m1(k1 + (size_t)i * 8, 8), q[i], 8), 8);
        }
        __riscv_vse32_v_f32m1(sc + s, __riscv_vfmul_vf_f32m1(a0, scale, 8), 8);
        __riscv_vse32_v_f32m1(sc + s + 8, __riscv_vfmul_vf_f32m1(a1, scale, 8), 8);
    }
    for (; s < n; s += 8, b++) {
        size_t vl = (size_t)(n - s < 8 ? n - s : 8);
        const float *k0 = kbh + (size_t)b * hd * 8;
        vfloat32m1_t a0 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        for (int i = 0; i < hd; i++)
            a0 = __riscv_vfadd_vv_f32m1(a0, __riscv_vfmul_vf_f32m1(__riscv_vle32_v_f32m1(k0 + (size_t)i * 8, vl), q[i], vl), vl);
        __riscv_vse32_v_f32m1(sc + s, __riscv_vfmul_vf_f32m1(a0, scale, vl), vl);
    }
}
static void vkq_s(const float *w, const float *vp_head, float *o, int n, int hd) {
    for (int i = 0; i < hd; i++) o[i] = 0.0f;
    for (int s = 0; s < n; s++) {
        const float *vp = vp_head + (size_t)s * hd;
        for (int i = 0; i < hd; i++) o[i] += w[s] * vp[i];
    }
}
static void vkq_v(const float *w, const float *vp_head, float *o, int n, int hd) {
    for (int i0 = 0; i0 < hd; ) {
        size_t vl = __riscv_vsetvl_e32m8((size_t)(hd - i0));
        vfloat32m8_t acc = __riscv_vfmv_v_f_f32m8(0.0f, vl);
        const float *vp = vp_head + i0;
        for (int s = 0; s < n; s++, vp += hd)
            acc = __riscv_vfadd_vv_f32m8(acc, __riscv_vfmul_vf_f32m8(__riscv_vle32_v_f32m8(vp, vl), w[s], vl), vl);
        __riscv_vse32_v_f32m8(o + i0, acc, vl);
        i0 += (int)vl;
    }
}
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static float frand(void) { return (float)rand() / RAND_MAX * 2.0f - 1.0f; }

int main(void) {
    const int hd = 64, NMAX = 1100;
    int nb8 = (NMAX + 7) / 8;
    float *kb = calloc((size_t)nb8 * hd * 8, 4), *vp = malloc((size_t)NMAX * hd * 4);
    float q[64], s1[1104], s2[1104], o1[64], o2[64], w[1104];
    srand(12345);
    for (size_t i = 0; i < (size_t)nb8 * hd * 8; i++) kb[i] = frand();
    for (size_t i = 0; i < (size_t)NMAX * hd; i++) vp[i] = frand();
    for (int i = 0; i < hd; i++) q[i] = frand();
    long bad_qk = 0, bad_vk = 0;
    for (int n = 1; n <= NMAX; n++) {
        memset(s1, 0x55, sizeof s1); memset(s2, 0x55, sizeof s2);
        qk_s(q, kb, s1, n, hd, 0.125f); qk_v(q, kb, s2, n, hd, 0.125f);
        for (int s = 0; s < n; s++) if (memcmp(&s1[s], &s2[s], 4)) bad_qk++;
        if (memcmp(&s1[n], &s2[n], 4)) bad_qk += 1000000;   /* 越界写探针 */
        for (int s = 0; s < n; s++) w[s] = s1[s] * 0.01f;
        vkq_s(w, vp, o1, n, hd); vkq_v(w, vp, o2, n, hd);
        for (int i = 0; i < hd; i++) if (memcmp(&o1[i], &o2[i], 4)) bad_vk++;
    }
    printf("BITCHECK n=1..%d qk_mismatch=%ld vkq_mismatch=%ld\n", NMAX, bad_qk, bad_vk);
    int ns[3] = {128, 512, 1024};
    for (int k = 0; k < 3; k++) {
        int n = ns[k], R = 200000 / n;
        double t0, ta = 1e9, tb = 1e9, tc = 1e9, td = 1e9;
        for (int rep = 0; rep < 3; rep++) {
            t0 = now(); for (int r = 0; r < R; r++) qk_s(q, kb, s1, n, hd, 0.125f); ta = fmin(ta, now() - t0);
            t0 = now(); for (int r = 0; r < R; r++) qk_v(q, kb, s2, n, hd, 0.125f); tb = fmin(tb, now() - t0);
            t0 = now(); for (int r = 0; r < R; r++) vkq_s(w, vp, o1, n, hd); tc = fmin(tc, now() - t0);
            t0 = now(); for (int r = 0; r < R; r++) vkq_v(w, vp, o2, n, hd); td = fmin(td, now() - t0);
        }
        printf("n=%4d QK scalar %.2fus rvv %.2fus (%.2fx) | VKQ scalar %.2fus rvv %.2fus (%.2fx)\n",
               n, ta / R * 1e6, tb / R * 1e6, ta / tb, tc / R * 1e6, td / R * 1e6, tc / td);
    }
    return (bad_qk || bad_vk) ? 1 : 0;
}
