#!/bin/bash
# _k1_asan_longrun.sh - ASan 有界长跑测试
#
# 验证门：1200 次推理（A/B/C 三轮各 400 帧） + 30 次加载/释放，
#         零 sanitizer 错误报告。
#
# 用法（主机端）：bash _k1_run.sh _k1_asan_longrun.sh
# 用法（板上直接）：bash _k1_asan_longrun.sh
#
# 依赖：
#   - build-riscv-asan/vllm_kestrel  （先执行 _k1_asan_build.sh）
#   - yolov8n_q8_320.vqf 和 test.jpg 在 /root/kv/kestrel_vllm_riscv/tools/yolo/
set -eu

ROOT=/root/kv/kestrel_vllm_riscv
DIR=$ROOT/tools/yolo
ASAN_BIN=$ROOT/build-riscv-asan/vllm_kestrel
VQF=$DIR/yolov8n_q8_320.vqf
IMG=$DIR/test.jpg
LOG=/tmp/asan_longrun.log

cd "$DIR"

# --- 前置检查 ---
for f in "$ASAN_BIN" "$VQF" "$IMG"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: required file missing: $f" >&2
        exit 1
    fi
done

# ASan 公共选项：
#   abort_on_error=0   让进程正常报告所有错误后退出（不 abort），便于统计
#   log_path=/tmp/asan_longrun  每次运行写独立日志
#   exitcode=23        ASan 检测到错误时用 23 退出（脚本据此判断）
export ASAN_OPTIONS="abort_on_error=0:exitcode=23:log_path=/tmp/asan_longrun:log_to_syslog=0:color=never"
export LSAN_OPTIONS="exitcode=23:log_to_syslog=0"
export UBSAN_OPTIONS="print_stacktrace=1:exitcode=23:log_to_syslog=0"

FAIL=0
TOTAL_INFER=0

echo "LONGRUN_START $(date)" | tee "$LOG"

# ===== Part A: 三轮各 400 帧推理（A=1线程 B=4线程 C=8线程）=====
for THREADS in 1 4 8; do
    LABEL="threads_${THREADS}"
    echo "--- infer round $LABEL (400 frames) ---" | tee -a "$LOG"

    rm -f /tmp/asan_longrun.*

    VLLM_THREADS=$THREADS \
    OMP_NUM_THREADS=$THREADS \
    VLLM_YOLO_CPU_TIMING=0 \
    VLLM_YOLO_FRAMES=400 \
      "$ASAN_BIN" \
        --yolo-model "$VQF" \
        --yolo-image "$IMG" \
        --yolo-conf 0.25 \
        --yolo-iou 0.45 \
      >> "$LOG" 2>&1
    EC=$?

    # 收集 ASan 日志碎片
    ASAN_LOGS=$(ls /tmp/asan_longrun.* 2>/dev/null | wc -l)
    if [ "$EC" -ne 0 ] || [ "$ASAN_LOGS" -gt 0 ]; then
        echo "FAIL $LABEL exit=$EC asan_log_files=$ASAN_LOGS" | tee -a "$LOG"
        if [ "$ASAN_LOGS" -gt 0 ]; then
            cat /tmp/asan_longrun.* >> "$LOG" 2>/dev/null || true
        fi
        FAIL=$(( FAIL + 1 ))
    else
        echo "OK   $LABEL" | tee -a "$LOG"
    fi
    TOTAL_INFER=$(( TOTAL_INFER + 400 ))
done

# ===== Part B: 30 次加载 / 释放循环 =====
# 使用 --yolo-frames 1 触发单次推理（加载→推理→释放），循环 30 次
echo "--- load/release cycle x30 ---" | tee -a "$LOG"
CYCLE_FAIL=0
for i in $(seq 1 30); do
    rm -f /tmp/asan_longrun.*

    VLLM_THREADS=4 \
    OMP_NUM_THREADS=4 \
    VLLM_YOLO_CPU_TIMING=0 \
    VLLM_YOLO_FRAMES=1 \
      "$ASAN_BIN" \
        --yolo-model "$VQF" \
        --yolo-image "$IMG" \
        --yolo-conf 0.25 \
        --yolo-iou 0.45 \
      >> "$LOG" 2>&1
    EC=$?

    ASAN_LOGS=$(ls /tmp/asan_longrun.* 2>/dev/null | wc -l)
    if [ "$EC" -ne 0 ] || [ "$ASAN_LOGS" -gt 0 ]; then
        echo "FAIL cycle $i exit=$EC" | tee -a "$LOG"
        [ "$ASAN_LOGS" -gt 0 ] && cat /tmp/asan_longrun.* >> "$LOG" 2>/dev/null || true
        CYCLE_FAIL=$(( CYCLE_FAIL + 1 ))
    fi
done

if [ "$CYCLE_FAIL" -gt 0 ]; then
    echo "FAIL load_release $CYCLE_FAIL/30 cycles failed" | tee -a "$LOG"
    FAIL=$(( FAIL + 1 ))
else
    echo "OK   load_release 30/30" | tee -a "$LOG"
fi

# ===== 汇总 =====
echo "" | tee -a "$LOG"
echo "LONGRUN_END total_infer=$TOTAL_INFER cycles=30 fail=$FAIL" | tee -a "$LOG"

if [ "$FAIL" -gt 0 ]; then
    echo "RESULT: FAIL (see $LOG)" | tee -a "$LOG"
    exit 1
else
    echo "RESULT: PASS" | tee -a "$LOG"
fi
