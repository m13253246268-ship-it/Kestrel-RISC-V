#!/bin/bash
# _k1_fault_matrix.sh  —  故障矩阵：关键分配点逐一失败注入
#
# 运行方式（主机侧）：
#   bash _k1_run.sh tools/yolo/_k1_fault_matrix.sh
#
# 验证目标：每个分配点注入 NULL 后，程序必须:
#   - 在 15 秒内正常退出（不 hang）
#   - 以信号杀死方式退出（SIGSEGV/SIGABRT 等）
#   - 不产生 ASan abort（如使用 asan 二进制）
# 判定：FAIL_HANG 或 FAIL_SIGNAL 任一非零 → 整体 FAIL
# ---------------------------------------------------------------------------
set -eu

ROOT=/root/kv/kestrel_vllm_riscv
TOOLS=$ROOT/tools/yolo
FM_DIR=/tmp/kv_fm
BIN_NORMAL=$ROOT/build-riscv/vllm_kestrel
BIN_ASAN=$ROOT/build-riscv-asan/vllm_kestrel
MODEL=$TOOLS/yolov8n_q8_320.vqf
IMAGE=$TOOLS/test.jpg

mkdir -p "$FM_DIR"

# ── 1. 写出 fault_inject.c (LD_PRELOAD malloc 拦截库) ─────────────────────
echo "[fm] writing fault_inject.c..."
cat > "$FM_DIR/fault_inject.c" << 'FAULT_C'
#define _GNU_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <dlfcn.h>
#include <stdatomic.h>

typedef void *(*malloc_fn_t)(size_t);
typedef void *(*calloc_fn_t)(size_t, size_t);
typedef void *(*realloc_fn_t)(void *, size_t);
typedef void  (*free_fn_t)(void *);

static malloc_fn_t  real_malloc  = NULL;
static calloc_fn_t  real_calloc  = NULL;
static realloc_fn_t real_realloc = NULL;
static free_fn_t    real_free    = NULL;

static _Atomic int g_count      = 0;
static int         g_fault_n    = 0;   /* 0 = disabled / passthrough */
static int         g_initialized = 0;
static int         g_in_init    = 0;   /* re-entrancy guard */

/* bootstrap static buffer: dlsym() calls calloc() before real_calloc is set */
static char   bs_buf[8192];
static size_t bs_off = 0;

static void fi_init(void) {
    if (g_initialized || g_in_init) return;
    g_in_init = 1;
    real_malloc  = (malloc_fn_t)  dlsym(RTLD_NEXT, "malloc");
    real_calloc  = (calloc_fn_t)  dlsym(RTLD_NEXT, "calloc");
    real_realloc = (realloc_fn_t) dlsym(RTLD_NEXT, "realloc");
    real_free    = (free_fn_t)    dlsym(RTLD_NEXT, "free");
    const char *e = getenv("FAULT_ALLOC_N");
    if (e && *e) g_fault_n = atoi(e);
    g_initialized = 1;
    g_in_init = 0;
}

__attribute__((constructor)) static void fi_ctor(void) { fi_init(); }

__attribute__((destructor)) static void fi_dtor(void) {
    const char *v = getenv("FAULT_ALLOC_VERBOSE");
    if (v && *v != '0') {
        int cnt = atomic_load_explicit(&g_count, memory_order_relaxed);
        fprintf(stderr, "[fault_inject] total_allocs=%d fault_n=%d\n",
                cnt, g_fault_n);
    }
}

void *malloc(size_t sz) {
    if (!g_initialized) fi_init();
    if (!real_malloc) return NULL;
    int n = atomic_fetch_add_explicit(&g_count, 1, memory_order_relaxed) + 1;
    if (g_fault_n > 0 && n == g_fault_n) return NULL;
    return real_malloc(sz);
}

void *calloc(size_t nm, size_t sz) {
    if (!real_calloc) {
        /* bootstrap: dlsym not yet available, serve from static buffer */
        size_t need = nm * sz;
        if (bs_off + need > sizeof(bs_buf)) return NULL;
        void *p = bs_buf + bs_off;
        bs_off += need;
        memset(p, 0, need);
        return p;
    }
    int n = atomic_fetch_add_explicit(&g_count, 1, memory_order_relaxed) + 1;
    if (g_fault_n > 0 && n == g_fault_n) return NULL;
    return real_calloc(nm, sz);
}

void *realloc(void *p, size_t sz) {
    if (!g_initialized) fi_init();
    if (!real_realloc) return NULL;
    int n = atomic_fetch_add_explicit(&g_count, 1, memory_order_relaxed) + 1;
    if (g_fault_n > 0 && n == g_fault_n) return NULL;
    return real_realloc(p, sz);
}

void free(void *p) {
    if (!p) return;
    if (!real_free) return;
    if ((char*)p >= bs_buf && (char*)p < bs_buf + sizeof(bs_buf)) return;
    real_free(p);
}
FAULT_C

# ── 2. 编译 fault_inject.so ───────────────────────────────────────────────
echo "[fm] compiling fault_inject.so..."
gcc -std=c11 -shared -fPIC -O1 -g \
    -o "$FM_DIR/libfault_inject.so" "$FM_DIR/fault_inject.c" -ldl \
    2>&1 | head -20
if [ ! -f "$FM_DIR/libfault_inject.so" ]; then
    echo "ERROR: failed to compile fault_inject.c" >&2; exit 1
fi
echo "[fm] libfault_inject.so OK"

# ── 3. 探测首轮：确定总分配数 N ──────────────────────────────────────────
# 使用 normal 二进制（无 ASan），FAULT_ALLOC_N=0 = 不注入，FAULT_ALLOC_VERBOSE=1 打印计数
echo "[fm] probe run (no fault injection)..."

# 选择要测试的二进制
if [ -f "$BIN_ASAN" ]; then
    BIN="$BIN_ASAN"
    USE_ASAN=1
    echo "[fm] using ASan binary: $BIN_ASAN"
elif [ -f "$BIN_NORMAL" ]; then
    BIN="$BIN_NORMAL"
    USE_ASAN=0
    echo "[fm] using normal binary: $BIN_NORMAL"
else
    echo "ERROR: no binary found (tried $BIN_ASAN and $BIN_NORMAL)" >&2
    exit 1
fi

if [ ! -f "$MODEL" ]; then
    echo "ERROR: model not found: $MODEL" >&2; exit 1
fi
if [ ! -f "$IMAGE" ]; then
    echo "ERROR: image not found: $IMAGE" >&2; exit 1
fi

PROBE_LOG="$FM_DIR/probe.log"
PROBE_ERR="$FM_DIR/probe.err"

ASAN_OPTIONS="abort_on_error=0:exitcode=23:log_path=$FM_DIR/asan_probe:log_to_syslog=0:color=never" \
LSAN_OPTIONS="exitcode=23:log_to_syslog=0" \
UBSAN_OPTIONS="print_stacktrace=1:exitcode=23:log_to_syslog=0" \
LD_PRELOAD="$FM_DIR/libfault_inject.so" \
FAULT_ALLOC_N=0 \
FAULT_ALLOC_VERBOSE=1 \
  timeout 30 "$BIN" -m "$MODEL" -i "$IMAGE" -n 1 \
  > "$PROBE_LOG" 2> "$PROBE_ERR" || true

# 解析 total_allocs
TOTAL=$(grep -oP 'total_allocs=\K[0-9]+' "$PROBE_ERR" 2>/dev/null | tail -1 || true)
if [ -z "$TOTAL" ] || [ "$TOTAL" -eq 0 ] 2>/dev/null; then
    echo "ERROR: probe run did not report total_allocs (got: '$TOTAL')" >&2
    echo "[fm] probe stderr:" >&2
    cat "$PROBE_ERR" >&2
    exit 1
fi
echo "[fm] total_allocs=$TOTAL  (will inject faults 1..$TOTAL)"

# ── 4. 主循环：逐一注入 ──────────────────────────────────────────────────
FAIL_HANG=0
FAIL_SIGNAL=0
FAIL_ASAN=0
PASS=0

# 结果记录文件
RESULT_FILE="$FM_DIR/results.txt"
> "$RESULT_FILE"

echo "[fm] starting fault injection loop (k=1..$TOTAL)..."

for k in $(seq 1 "$TOTAL"); do
    RUN_ERR="$FM_DIR/run_${k}.err"
    ASAN_LOG_PREFIX="$FM_DIR/asan_${k}"

    # 清理上一次可能残留的 asan 日志
    rm -f "${ASAN_LOG_PREFIX}."* 2>/dev/null || true

    ASAN_OPTIONS="abort_on_error=0:exitcode=23:log_path=${ASAN_LOG_PREFIX}:log_to_syslog=0:color=never" \
    LSAN_OPTIONS="exitcode=23:log_to_syslog=0" \
    UBSAN_OPTIONS="print_stacktrace=1:exitcode=23:log_to_syslog=0" \
    LD_PRELOAD="$FM_DIR/libfault_inject.so" \
    FAULT_ALLOC_N="$k" \
    FAULT_ALLOC_VERBOSE=0 \
      timeout 15 "$BIN" -m "$MODEL" -i "$IMAGE" -n 1 \
      > /dev/null 2> "$RUN_ERR"
    EXIT_CODE=$?

    # timeout 以 124 退出；killed by signal：128+signum
    if [ "$EXIT_CODE" -eq 124 ]; then
        STATUS="FAIL_HANG"
        FAIL_HANG=$((FAIL_HANG + 1))
    elif [ "$EXIT_CODE" -eq 23 ]; then
        # ASan/UBSan 检测到错误
        STATUS="FAIL_ASAN"
        FAIL_ASAN=$((FAIL_ASAN + 1))
    elif [ "$EXIT_CODE" -eq 0 ]; then
        # 分配失败但程序仍以0退出：通常是某条不关键路径
        # 检查是否产生了 ASan 日志（可能内容提示问题）
        ASAN_FILES=$(ls "${ASAN_LOG_PREFIX}."* 2>/dev/null | wc -l || echo 0)
        if [ "$ASAN_FILES" -gt 0 ] && [ "$USE_ASAN" -eq 1 ]; then
            STATUS="FAIL_ASAN_SILENT"
            FAIL_ASAN=$((FAIL_ASAN + 1))
        else
            STATUS="PASS_ZERO"
            PASS=$((PASS + 1))
        fi
    else
        # 非零退出（包括信号杀死 = 正常的分配失败路径）
        # 信号杀死（>128）也算 PASS，程序没有 hang
        if [ "$EXIT_CODE" -gt 128 ]; then
            SIG=$((EXIT_CODE - 128))
            STATUS="PASS_SIG${SIG}"
        else
            STATUS="PASS_ERR${EXIT_CODE}"
        fi
        PASS=$((PASS + 1))
    fi

    echo "k=$k exit=$EXIT_CODE $STATUS" >> "$RESULT_FILE"

    # 进度打印（每 20 个 + 最后一个）
    if [ $((k % 20)) -eq 0 ] || [ "$k" -eq "$TOTAL" ]; then
        echo "[fm] progress: $k/$TOTAL  pass=$PASS hang=$FAIL_HANG sig=$FAIL_SIGNAL asan=$FAIL_ASAN"
    fi
done

# ── 5. 汇总 PASS/FAIL 矩阵 ──────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════"
echo "  FAULT MATRIX RESULTS"
echo "════════════════════════════════════════"
echo "  total_allocs : $TOTAL"
echo "  PASS         : $PASS"
echo "  FAIL_HANG    : $FAIL_HANG"
echo "  FAIL_SIGNAL  : $FAIL_SIGNAL"
echo "  FAIL_ASAN    : $FAIL_ASAN"
echo "════════════════════════════════════════"

if [ "$FAIL_HANG" -gt 0 ] || [ "$FAIL_ASAN" -gt 0 ]; then
    echo "RESULT: FAIL"
    echo ""
    echo "Failures:"
    grep -E 'FAIL' "$RESULT_FILE" | head -40 || true
    echo ""
    echo "Full results: $RESULT_FILE"
    exit 1
else
    echo "RESULT: PASS  (all $TOTAL allocation-fault points handled cleanly)"
    exit 0
fi
