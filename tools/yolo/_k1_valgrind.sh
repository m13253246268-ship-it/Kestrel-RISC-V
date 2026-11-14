#!/bin/bash
# _k1_valgrind.sh - Valgrind Memcheck 补充门
#
# 用法（主机端）：bash _k1_run.sh tools/yolo/_k1_valgrind.sh
# 用法（板上直接）：bash _k1_valgrind.sh
#
# 通过条件：valgrind 报告零 "definitely lost" / "Invalid read" /
#           "Invalid write" / "Use of uninitialised value"
# 退出码：
#   0  - PASS（memcheck 干净）
#   1  - FAIL（发现内存错误）
#   2  - SKIP（valgrind 不可用；不阻塞 CI）
#   3  - 环境缺失（模型/图片/二进制不存在）
set -eu

ROOT=/root/kv/kestrel_vllm_riscv
BIN=$ROOT/build-riscv/vllm_kestrel
MODEL=$ROOT/tools/yolo/yolov8n_q8_320.vqf
IMAGE=$ROOT/tools/yolo/test.jpg
LOG=/tmp/kv_valgrind.log
SUPP=/tmp/kv_valgrind.supp    # 可选 suppression 文件（不存在则跳过）

# ── 1. 检查 valgrind 可用性 ───────────────────────────────────────────────
echo "[valgrind] checking valgrind availability..."
if ! command -v valgrind >/dev/null 2>&1; then
    echo "[valgrind] SKIP: valgrind not found on this board."
    echo "           Install with: apt install valgrind"
    echo "           (Skipping is non-fatal; ASan/fault_matrix gates are primary.)"
    exit 2
fi

VG_VER=$(valgrind --version 2>&1 | head -1)
echo "[valgrind] $VG_VER"

# valgrind on RISC-V requires >= 3.20 for rv64gc support
VG_MAJOR=$(echo "$VG_VER" | grep -oP '(?<=valgrind-)\d+' | head -1 || true)
VG_MINOR=$(echo "$VG_VER" | grep -oP '(?<=valgrind-\d{1,2}\.)\d+' | head -1 || true)
if [ -n "$VG_MAJOR" ] && [ -n "$VG_MINOR" ]; then
    if [ "$VG_MAJOR" -lt 3 ] || { [ "$VG_MAJOR" -eq 3 ] && [ "$VG_MINOR" -lt 20 ]; }; then
        echo "[valgrind] SKIP: valgrind $VG_MAJOR.$VG_MINOR is too old for rv64gc (need >= 3.20)."
        exit 2
    fi
fi

# ── 2. 检查依赖文件 ───────────────────────────────────────────────────────
echo "[valgrind] checking required files..."
MISSING=0
for F in "$BIN" "$MODEL" "$IMAGE"; do
    if [ ! -f "$F" ]; then
        echo "ERROR: missing file: $F" >&2
        MISSING=1
    fi
done
if [ "$MISSING" -eq 1 ]; then
    echo "[valgrind] FAIL: required files missing (see above)."
    exit 3
fi
echo "[valgrind] binary : $BIN ($(stat -c %s "$BIN") bytes)"
echo "[valgrind] model  : $MODEL ($(stat -c %s "$MODEL") bytes)"
echo "[valgrind] image  : $IMAGE ($(stat -c %s "$IMAGE") bytes)"

# ── 3. 构造 suppression 参数 ─────────────────────────────────────────────
SUPP_ARGS=""
if [ -f "$SUPP" ]; then
    SUPP_ARGS="--suppressions=$SUPP"
    echo "[valgrind] using suppressions: $SUPP"
fi

# ── 4. 运行 memcheck ──────────────────────────────────────────────────────
echo "[valgrind] running memcheck (1 frame, full leak check)..."
echo "[valgrind] log -> $LOG"

# --error-exitcode=42: valgrind 自身检测到错误时以 42 退出
# --trace-children=yes: 跟踪 fork/exec 子进程（若 vllm_kestrel 有 helper 进程）
# --track-origins=yes: 追踪未初始化值来源（性能较慢，但单帧可接受）
# --malloc-fill/--free-fill: 检测 use-after-free 更灵敏
set +e
valgrind \
    --tool=memcheck \
    --leak-check=full \
    --show-leak-kinds=definite,indirect \
    --track-origins=yes \
    --malloc-fill=0xAB \
    --free-fill=0xCD \
    --error-exitcode=42 \
    --trace-children=yes \
    --time-stamp=yes \
    $SUPP_ARGS \
    --log-file="$LOG" \
    "$BIN" -m "$MODEL" -i "$IMAGE" --yolo-frames 1 \
    >/tmp/kv_valgrind_stdout.log 2>&1
VG_EXIT=$?
set -e

echo "[valgrind] exit code: $VG_EXIT"

# ── 5. 解析 memcheck 报告 ─────────────────────────────────────────────────
echo "[valgrind] --- summary from $LOG ---"
# 显示 LEAK SUMMARY 和 ERROR SUMMARY
grep -E "^==[0-9]+== (LEAK SUMMARY|ERROR SUMMARY|definitely lost|indirectly lost|possibly lost|still reachable|Invalid read|Invalid write|Use of uninitialised)" \
    "$LOG" 2>/dev/null || true
echo "[valgrind] ---"

# 提取各项计数
extract_count() {
    # $1: pattern to grep; returns the first integer found on the line
    grep -m1 "$1" "$LOG" 2>/dev/null \
        | grep -oP '\d[\d,]*' | head -1 | tr -d ',' || echo "0"
}

DEF_LOST=$(extract_count "definitely lost:")
INVAL_RD=$(grep -c "Invalid read" "$LOG" 2>/dev/null || echo "0")
INVAL_WR=$(grep -c "Invalid write" "$LOG" 2>/dev/null || echo "0")
UNINIT=$(grep -c "Use of uninitialised" "$LOG" 2>/dev/null || echo "0")
ERR_TOTAL=$(extract_count "ERROR SUMMARY:.*error")

echo "[valgrind] definitely lost    : $DEF_LOST bytes"
echo "[valgrind] Invalid read       : $INVAL_RD"
echo "[valgrind] Invalid write      : $INVAL_WR"
echo "[valgrind] Uninitialised use  : $UNINIT"
echo "[valgrind] ERROR SUMMARY total: $ERR_TOTAL"

# ── 6. 判断通过/失败 ──────────────────────────────────────────────────────
FAIL=0

# valgrind 以 42 退出 → 检测到错误
if [ "$VG_EXIT" -eq 42 ]; then
    echo "[valgrind] FAIL: valgrind reported errors (exit 42)."
    FAIL=1
fi

# 独立判断各类严重错误
if [ "$DEF_LOST" -ne 0 ] 2>/dev/null; then
    echo "[valgrind] FAIL: 'definitely lost' bytes = $DEF_LOST"
    FAIL=1
fi
if [ "$INVAL_RD" -ne 0 ] 2>/dev/null; then
    echo "[valgrind] FAIL: Invalid read count = $INVAL_RD"
    FAIL=1
fi
if [ "$INVAL_WR" -ne 0 ] 2>/dev/null; then
    echo "[valgrind] FAIL: Invalid write count = $INVAL_WR"
    FAIL=1
fi
if [ "$UNINIT" -ne 0 ] 2>/dev/null; then
    echo "[valgrind] FAIL: Uninitialised value use count = $UNINIT"
    FAIL=1
fi

if [ "$FAIL" -eq 0 ]; then
    echo "[valgrind] PASS: memcheck clean."
    exit 0
else
    echo "[valgrind] FAIL: see $LOG for full report."
    echo "           Tip: re-run with --track-origins=yes (already on) for source info."
    exit 1
fi
