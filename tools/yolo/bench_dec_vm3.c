/* ================================================================
 * bench_dec_vm3.c — TCM 验证：真实遍历（cache miss）下 DRAM vs TCM
 *
 * 复用 bench_dec_vmadot.c（vm_tile16、mk、prep、mat_t）。
 * 遍历全部 16 行 tile：
 *   DRAM : vm_tile16 从 w288（DRAM，重排后）
 *   TCM  : memcpy 该 tile 权重到本线程 TCM 块 + vm_tile16 从 TCM（串行，无重叠）
 * 形状：gate(K=2560,608 tile) 与 down(K=9728,160 tile)。
 * 编译：gcc -O2 -march=rv64gcv -mabi=lp64d -ffp-contract=off -fopenmp bench_dec_vm3.c -o bench_dec_vm3 -ldl
 * ================================================================ */
#define _GNU_SOURCE
#define main bench_dec_vmadot_main
#include "bench_dec_vmadot.c"
#undef main
#include <dlfcn.h>

static void *(*tcm_mget)(int) = NULL;
static void tcm_init(void) {
    void *h = dlopen("libspine_tcm.so", RTLD_LAZY | RTLD_GLOBAL);
    if (h) tcm_mget = (void *(*)(int))dlsym(h, "spine_tcm_runtime_mem_get");
}

static int g_badc = 0;

/* 遍历 ntile 个 16 行 tile；from_tcm 决定 memcpy 到 TCM 后 gemm 还是直接 gemm */
static void run_tiles(mat_t *m, int nt, int from_tcm) {
    const size_t wbytes = (size_t)m->nblk * 288;
    const int ntile = m->rows / 16;
    int next = 0;
    #pragma omp parallel num_threads(nt)
    {
        float o[16];
        const int th = omp_get_thread_num();
        void *t = from_tcm ? tcm_mget(th) : NULL;
        if (from_tcm && !t) __atomic_add_fetch(&g_badc, 1, __ATOMIC_RELAXED);
        for (;;) {
            const int ti = __atomic_fetch_add(&next, 1, __ATOMIC_RELAXED);
            if (ti >= ntile) break;
            const uint8_t *src;
            if (from_tcm) { memcpy(t, m->w288 + (size_t)ti * wbytes, wbytes); src = (const uint8_t *)t; }
            else src = m->w288 + (size_t)ti * wbytes;
            vm_tile16(src, m->xr, m->xd, m->corr, m->nblk, o);
        }
    }
}

static double bench(mat_t *m, int nt, int from_tcm, int R) {
    double *ts = malloc(sizeof(double) * R);
    for (int r = 0; r < R; r++) {
        double a = now();
        run_tiles(m, nt, from_tcm);
        ts[r] = now() - a;
    }
    qsort(ts, R, sizeof(double), cmp_dbl);
    double med = ts[R / 2];
    free(ts);
    return med;
}

int main(int argc, char **argv) {
    const int R = argc > 1 ? atoi(argv[1]) : 21;
    cpu_set_t orig; sched_getaffinity(0, sizeof orig, &orig);
    omp_set_dynamic(0);
    { cpu_set_t s; CPU_ZERO(&s); CPU_SET(0, &s); sched_setaffinity(0, sizeof s, &s); }
    tcm_init();
    printf("bench_dec_vm3  DRAM vs TCM(串行)  R=%d  tcm_mget=%p\n", R, (void *)tcm_mget);

    mat_t L[2] = { mk("gate", 9728, 2560, 0), mk("down", 2560, 9728, 1) };
    for (int i = 0; i < 2; i++) {
        mat_t *m = &L[i];
        prep(m, M_VM);
        const double wbytes = (double)m->nblk * 288 * (m->rows / 16);
        for (int nt = 1; nt <= 4; nt += 3) {
            if (nt > 1) pin_team(nt);
            double d = bench(m, nt, 0, R);
            if (!tcm_mget) {
                printf("  %-5s t%d  DRAM med %.3f ms (%.2f GB/s)   TCM 不可用，跳过\n",
                       m->name, nt, d * 1e3, wbytes / d / 1e9);
                continue;
            }
            double t = bench(m, nt, 1, R);
            printf("  %-5s K=%-5d tiles=%-4d total=%.1fKB  t%d  DRAM med %.3f ms (%.2f GB/s)   TCM串行 med %.3f ms (%.2f GB/s)   相对 %+.1f%%\n",
                   m->name, m->nblk * 32, m->rows / 16, wbytes / 1024, nt, d * 1e3, wbytes / d / 1e9,
                   t * 1e3, wbytes / t / 1e9, (d / t - 1) * 100);
        }
    }
    printf("badc=%d\n", g_badc);
    sched_setaffinity(0, sizeof orig, &orig);
    return 0;
}
