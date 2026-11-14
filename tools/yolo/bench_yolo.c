/**
 * bench_yolo.c - VQF 检测引擎板端基准（稳态帧延迟 / 加载 / 首次 / 输出指纹）
 *
 * 用法：
 *   bench_yolo <model.vqf> <image> [runs]
 *   bench_yolo <model.vqf> --raw <in.f32> <out.f32>   # 位级门：固定 NCHW 输入
 *
 * 只依赖 vllm_yolo.c 的公开接口（st_yolo_load / st_yolo_detect / st_yolo_forward_raw），
 * 与引擎主二进制的执行路径同源。
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/resource.h>

#include "vllm_yolo.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static unsigned long long fnv1a(const void *p, size_t n, unsigned long long h)
{
    const unsigned char *b = (const unsigned char *)p;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ULL; }
    return h;
}

static int cmpd(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: bench_yolo <model.vqf> <image> [runs]\n"); return 2; }
    const char *mpath = argv[1];

    double t0 = now();
    StYoloModel *m = st_yolo_load(mpath);
    double tload = now() - t0;
    if (!m) { fprintf(stderr, "[bench] load failed\n"); return 1; }
    int iw = 0, ih = 0;
    st_yolo_input_size(m, &iw, &ih);
    printf("[bench] model=%s in=%dx%d load=%.1fms\n", mpath, iw, ih, tload * 1e3);

    /* ---- 位级门模式 ---- */
    if (argc >= 5 && strcmp(argv[2], "--raw") == 0) {
        const char *rin = argv[3], *rout = argv[4];
        size_t n = (size_t)3 * iw * ih;
        float *in = (float *)malloc(n * sizeof(float));
        float *out = (float *)malloc((size_t)(4 + st_yolo_num_classes(m)) * 2100 * sizeof(float));
        FILE *f = fopen(rin, "rb");
        if (!f || fread(in, sizeof(float), n, f) != n) { fprintf(stderr, "[bench] raw read failed\n"); return 1; }
        fclose(f);
        int oc = 0, oa = 0;
        double s = now();
        int rc = st_yolo_forward_raw(m, in, out, &oc, &oa);
        double el = now() - s;
        if (rc != 0) { fprintf(stderr, "[bench] forward_raw failed\n"); return 1; }
        size_t on = (size_t)oc * oa;
        unsigned long long h = fnv1a(out, on * sizeof(float), 1469598103934665603ULL);
        if (rout && strcmp(rout, "-") != 0) {
            FILE *g = fopen(rout, "wb");
            if (g) { fwrite(out, sizeof(float), on, g); fclose(g); }
        }
        printf("[bench] raw fwd=%.1fms out=[%d,%d] FNV=%016llx\n", el * 1e3, oc, oa, h);
        free(in); free(out); st_yolo_free(m);
        return 0;
    }

    /* ---- 图像稳态模式 ---- */
    int runs = argc >= 4 ? atoi(argv[3]) : 20;
    if (runs < 1) runs = 1;
    int w = 0, h = 0, ch = 0;
    unsigned char *img = stbi_load(argv[2], &w, &h, &ch, 3);
    if (!img) { fprintf(stderr, "[bench] image load failed: %s\n", argv[2]); st_yolo_free(m); return 1; }

    YoloDet det[300];
    double s0 = now();
    int nd = st_yolo_detect(m, img, w, h, 0.25f, 0.45f, det, 300);
    double first = now() - s0;
    unsigned long long hd = fnv1a(det, (size_t)(nd > 0 ? nd : 0) * sizeof(YoloDet), 1469598103934665603ULL);
    printf("[bench] first_detect=%.1fms dets=%d detFNV=%016llx\n", first * 1e3, nd, hd);
    for (int i = 0; i < (nd < 5 ? nd : 5); i++)
        printf("[bench]   #%d cls=%d score=%.4f box=[%.1f,%.1f,%.1f,%.1f]\n",
               i, det[i].cls, det[i].score, det[i].x1, det[i].y1, det[i].x2, det[i].y2);

    double *lat = (double *)malloc((size_t)runs * sizeof(double));
    for (int i = 0; i < runs; i++) {
        double s = now();
        int n2 = st_yolo_detect(m, img, w, h, 0.25f, 0.45f, det, 300);
        lat[i] = now() - s;
        if (n2 != nd) printf("[bench] warn: det count changed %d -> %d\n", nd, n2);
    }
    qsort(lat, (size_t)runs, sizeof(double), cmpd);
    double sum = 0;
    for (int i = 0; i < runs; i++) sum += lat[i];
    printf("[bench] steady runs=%d min=%.1f p50=%.1f mean=%.1f p90=%.1f ms\n",
           runs, lat[0] * 1e3, lat[runs / 2] * 1e3, sum / runs * 1e3,
           lat[(int)(0.9 * (runs - 1))] * 1e3);
    { struct rusage ru; getrusage(RUSAGE_SELF, &ru);
      printf("[bench] maxrss=%ld KB\n", (long)ru.ru_maxrss); }

    free(lat);
    stbi_image_free(img);
    st_yolo_free(m);
    return 0;
}
