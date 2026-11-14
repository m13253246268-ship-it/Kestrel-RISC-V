#!/bin/bash
# _k1_asan_build.sh - 在 K1 板上原生构建 ASan/LSan/UBSan 二进制
#
# 用法（主机端）：bash _k1_run.sh _k1_asan_build.sh
# 用法（板上直接）：bash _k1_asan_build.sh
#
# 产物：/root/kv/kestrel_vllm_riscv/build-riscv-asan/vllm_kestrel
#
# 注意：
#   - ASan 在 RISC-V Linux 上要求 gcc >= 10 且 libasan 存在
#   - 若 gcc 不支持 -fsanitize=address，脚本会给出明确错误
#   - 板上原生 gcc 不需要 VLLM_RISCV_CC，自动探测
set -eu

ROOT=/root/kv/kestrel_vllm_riscv
BUILD_SH=$ROOT/tools/build/build_riscv.sh

cd "$ROOT"

# --- 检查 gcc 是否支持 ASan ---
echo "[asan_build] checking gcc sanitizer support..."
GCC_VER=$(gcc --version | head -1)
echo "[asan_build] gcc: $GCC_VER"

# 写一个最小测试程序验证 -fsanitize=address 可用
TMPDIR_ASAN=$(mktemp -d)
trap 'rm -rf "$TMPDIR_ASAN"' EXIT

cat > "$TMPDIR_ASAN/asan_probe.c" << 'PROBE'
int main(void) { return 0; }
PROBE

if ! gcc -fsanitize=address,leak,undefined -fno-omit-frame-pointer \
         "$TMPDIR_ASAN/asan_probe.c" -o "$TMPDIR_ASAN/asan_probe" 2>&1; then
    echo "ERROR: gcc does not support -fsanitize=address on this board." >&2
    echo "       Install libasan (e.g. apt install libasan6) or use Valgrind instead." >&2
    exit 1
fi
echo "[asan_build] sanitizer probe OK"

# --- 构建 ---
echo "[asan_build] building with ASan/LSan/UBSan..."
export VLLM_RISCV_OUTDIR="$ROOT/build-riscv-asan"
export VLLM_RISCV_EXTRA="-fsanitize=address,leak,undefined -fno-omit-frame-pointer -g"

bash "$BUILD_SH"

OUT="$VLLM_RISCV_OUTDIR/vllm_kestrel"
if [ ! -f "$OUT" ]; then
    echo "ERROR: build produced no output: $OUT" >&2
    exit 1
fi

OUT_BYTES=$(stat -c %s "$OUT")
echo "[asan_build] OK: $OUT ($OUT_BYTES bytes)"
file "$OUT" || true

# --- 快速烟测（确认 binary 可加载，无 immediate ASan 崩溃）---
echo "[asan_build] smoke test: vllm_kestrel --help ..."
# ASan 需要 /proc/self/maps 和 shadow memory；RISC-V Linux 一般支持
# ASAN_OPTIONS: abort_on_error=0 确保 --help 路径可以正常退出
ASAN_OPTIONS="abort_on_error=0:log_to_syslog=0" \
LSAN_OPTIONS="exitcode=0" \
  "$OUT" --help 2>&1 | head -5 || true

echo "[asan_build] DONE"
