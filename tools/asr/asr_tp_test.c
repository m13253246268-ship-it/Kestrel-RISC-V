/* Exercise the shared pool's partial-creation fallback without OS exhaustion. */
#include <pthread.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>

static int create_limit, create_count;
static int test_create(pthread_t *thread, const pthread_attr_t *attr,
                       void *(*entry)(void *), void *arg) {
    if (create_count++ >= create_limit) return EAGAIN;
    return pthread_create(thread, attr, entry, arg);
}
#define pthread_create test_create
#include "../../src/core/vllm_tp.c"
#undef pthread_create

static void increment(void *ctx, int i) {
    int *values = ctx;
    values[i]++;
}
static void nested(void *ctx, int i) {
    int *values = ctx;
    vllm_tp_parfor(0, 4, increment, values + i * 4);
}
static void slot(void *ctx) {
    int *values = ctx;
    values[vllm_tp_worker_id()]++;
}
int main(void) {
    for (int limit = 0; limit <= 7; limit++) {
        int values[128], slots[64];
        create_limit = limit;
        create_count = 0;
        int actual = vllm_tp_init(8);
        if (actual != limit + 1 || vllm_tp_threads() != actual) return 1;
        for (int iteration = 0; iteration < 32; iteration++) {
            memset(values, 0, sizeof values);
            vllm_tp_parfor(0, 128, increment, values);
            vllm_tp_parfor_dyn(0, 128, 7, increment, values);
            vllm_tp_parfor(0, 32, nested, values);
            for (int i = 0; i < 128; i++) if (values[i] != 3) return 2;
            if (actual > 1) {
                memset(slots, 0, sizeof slots);
                vllm_tp_parcall(slot, slots);
                for (int i = 0; i < actual; i++) if (slots[i] != 1) return 3;
            }
        }
        vllm_tp_shutdown();
        vllm_tp_shutdown();
        printf("create_limit=%d actual=%d static/dynamic/nested/shutdown PASS\n", limit, actual);
    }
    return 0;
}
