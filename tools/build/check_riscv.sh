#!/bin/sh
# ================================================================
# check_riscv.sh - RISC-V 回归闸门：构建 + L3/Sparse 自检计数
#
# 目的：把「RISC-V 上能编译、且数值自检与 x86/ARM 同源」固化成一条命令。
#   期望：--test-l3 16 PASS / 0 FAIL、--test-sparse 9 PASS / 0 FAIL。
#   无 NEON 构建下 (1b)「P2 restore NEON vs scalar bitwise」打 [SKIP]，不计 PASS。
#
# 用法：
#   ./tools/build/check_riscv.sh            # 板上原生跑
#   ./tools/build/check_riscv.sh --build    # 只构建（交叉编译主机无 qemu 时）
# 环境变量：
#   VLLM_RISCV_QEMU_CPU  qemu 的 -cpu 串，缺省 max（RVV 1.0）；
#                        VLEN 用 max,vlen=256 指定。原生运行不需要。
#   其余变量透传 build_riscv.sh（VLLM_RISCV_CC / _MARCH / _OUTDIR）。
#
# 退出码：0 = 构建成功且 16/0 + 9/0；1 = 构建失败或计数不符
#
# 边界（照实标注）：本脚本只证明「可编译 + 自检通过 + 与 x86 同为 16/9 条」，
#   **不**构成性能结论。默认热内核是 RVV =3（VLLM_RVV=0 回标量参考路径），
#   收益只在板端实测界定（8 线程，Qwen3VL-2B Q8_0）：
#     prefill 32 token：48.4 / 33.1 / 12.7 / 6.2 s（=0/1/2/3），
#     decode 1 token  ：1.75 / 1.28 / 0.43 / 0.43 s；
#     四档逐行比位全 0 失配（VLLM_RVV_CHK=1）、TOKIDS 与标量一致。
#   同板同模型官方基线 llama.cpp（SpacemiT 包）pp32 3.96 t/s、tg16 2.55 t/s；
#   =3 的 prefill 为 4.8–6.4 t/s（领先 1.2–1.6×），decode 2.3 t/s（0.9×，已到
#   板端顺序读上限 7.6 GB/s）。跨会话绝对值受板端热/缓存状态影响
#   （5.0–6.7 s），同会话极差 0.70%。未实测规模不写进默认
#   （perf_axiom_measured_benefit_region_v1）。
# ================================================================
set -u

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
OUTDIR="${VLLM_RISCV_OUTDIR:-$ROOT/build-riscv}"
BIN="$OUTDIR/vllm_kestrel"
BUILD_ONLY=0
[ "${1:-}" = "--build" ] && BUILD_ONLY=1

echo "=== [1/3] build riscv ==="
START=$(date +%s)
"$ROOT/tools/build/build_riscv.sh" || { echo "RISCV CHECK FAILED: build error"; exit 1; }
[ -x "$BIN" ] || { echo "RISCV CHECK FAILED: no $BIN"; exit 1; }
if [ "$(date -r "$BIN" +%s 2>/dev/null || echo 0)" -lt "$START" ]; then
    echo "RISCV CHECK FAILED: 产物早于本次构建开始（构建未产出新文件）"
    exit 1
fi
if [ "$BUILD_ONLY" = 1 ]; then
    echo "RISCV BUILD OK (self-test skipped)"; exit 0
fi

# 运行器：原生物理机直跑；交叉编译主机用 qemu-user
if [ "$(uname -m)" = "riscv64" ]; then
    RUN=""
else
    command -v qemu-riscv64 >/dev/null 2>&1 || { echo "RISCV CHECK FAILED: 无 qemu-riscv64（可用 --build 只构建）"; exit 1; }
    RUN="qemu-riscv64 -cpu ${VLLM_RISCV_QEMU_CPU:-max} -L /usr/riscv64-linux-gnu"
fi

check() { # $1=flag $2=want
    out=$($RUN "$BIN" "$1" 2>&1)
    p=$(printf '%s\n' "$out" | grep -c '\[PASS\]')
    f=$(printf '%s\n' "$out" | grep -c '\[FAIL\]')
    s=$(printf '%s\n' "$out" | grep -c '\[SKIP\]')
    if [ "$p" -eq "$2" ] && [ "$f" -eq 0 ]; then
        printf '     %-14s PASS=%s FAIL=%s SKIP=%s  (want %s/0)  -> OK\n' "$1" "$p" "$f" "$s" "$2"
        return 0
    fi
    printf '     %-14s PASS=%s FAIL=%s SKIP=%s  (want %s/0)  -> MISMATCH\n' "$1" "$p" "$f" "$s" "$2"
    printf '%s\n' "$out" | tail -20
    return 1
}

echo "=== [2/3] self-tests（RISC-V 口径：16 PASS + 1 SKIP / 9）==="
check --test-l3 16 || exit 1
check --test-sparse 9 || exit 1

echo "=== [3/3] artifact ==="
ls -l "$BIN"
if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$BIN"
fi
echo "RISCV CHECK PASSED"
