#!/bin/sh
# ================================================================
# build_riscv.sh - RISC-V 64 (Linux) 构建脚本
#
# 定位：与 build_x64.ps1 对称的第三架构分支。RISC-V 侧（ST_ARCH_RISCV=1）
#       的位级基准是标量参考路径：VLLM_RVV=0 可随时回到它（逃生舱）；
#       -march 含 v 时 ST_HAVE_RVV=1，mode-2 的 Q8 内核按 VLLM_RVV 选档
#       （lane=行；四档共用同一条累加链，彼此位级一致，可同二进制 A/B）：
#         缺省 = 3  1 = vlse8 逐列 gather（初版）
#                   2 = vle32 连续取数 + 移位取列
#                   3 = 2 的取数 + prefill 一取数喂 4 token（默认）
#       板端实测（8 线程，Qwen3VL-2B Q8_0）：32 token prefill 的 GEMM
#       48.4 s / 33.1 s / 12.7 s / 6.2 s；1 token decode 1.75/1.28/0.43/0.43 s。
#
# 用法（板上原生构建）：
#   ./tools/build/build_riscv.sh
# 用法（x86 主机交叉编译，需 riscv64-linux-gnu-gcc）：
#   CC=riscv64-linux-gnu-gcc ./tools/build/build_riscv.sh
#
# 环境变量：
#   VLLM_RISCV_CC      指定编译器（缺省自动探测 gcc / riscv64-*-gcc）
#   VLLM_RISCV_MARCH   -march 取值，缺省 rv64gcv（RVV 1.0 目标板；
#                      无向量板改 rv64gc；VLEN 由硬件决定，不进 -march）
#   VLLM_RISCV_OUTDIR  产物目录，缺省 <repo>/build-riscv
#   VLLM_RISCV_EXTRA   追加编译/链接参数
#
# 确定性红线：-ffp-contract=off 必须保留（禁止编译器把 mul+add 收缩成
#   fma，否则与参考实现的舍入链不再逐项同序，位级对拍失效）。
# ================================================================
set -e

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)

CC="$VLLM_RISCV_CC"
if [ -z "$CC" ]; then
    if [ "$(uname -m)" = "riscv64" ] && command -v gcc >/dev/null 2>&1; then
        CC=gcc
    elif command -v riscv64-linux-gnu-gcc >/dev/null 2>&1; then
        CC=riscv64-linux-gnu-gcc
    elif command -v riscv64-unknown-linux-gnu-gcc >/dev/null 2>&1; then
        CC=riscv64-unknown-linux-gnu-gcc
    else
        echo "ERROR: 未找到 riscv64 编译器（设置 VLLM_RISCV_CC 或安装交叉工具链）" >&2
        exit 1
    fi
fi

MARCH="${VLLM_RISCV_MARCH:-rv64gcv}"
OUTDIR="${VLLM_RISCV_OUTDIR:-$ROOT/build-riscv}"
mkdir -p "$OUTDIR"
OUT="$OUTDIR/vllm_kestrel"

SRC="
src/main.c
src/core/vllm_superpos.c src/core/vllm_attention.c src/core/vllm_scheduler.c
src/core/vllm_transformer.c src/core/vllm_tokenizer.c src/core/vllm_matmul.c
src/core/vllm_tp.c src/core/vllm_ep.c src/core/vllm_l3.c
src/core/vllm_kvstore.c
src/common/vllm_device.c src/common/vllm_crypto.c
src/serve/vllm_http.c src/serve/vllm_server.c src/serve/vllm_batch.c
src/serve/vllm_admin.c src/serve/vllm_attest.c src/serve/embedded_web.c
src/serve/vllm_rulebook.c src/serve/vllm_session.c src/serve/vllm_audit.c
src/model/vllm_safetensors.c src/model/vqf.c
src/model/vllm_tokenizer_qwen.c src/model/vllm_vision.c src/model/vllm_media.c
src/model/vllm_laya.c src/model/vllm_yolo.c
src/media/vllm_vsrc.c src/media/vllm_rtsp.c src/media/h264_dec.c
src/media/vllm_mp4.c
"

INCS="-Iinclude -Iinclude/common -Iinclude/core -Iinclude/serve -Iinclude/model \
-Iinclude/media -Isrc"

# GCC 14+ 把 incompatible-pointer-types / int-conversion 等提升为 error，
# 与 build_x64.ps1 保持同一套兼容旗标（只降级，不隐藏新问题）。
CFLAGS="-std=gnu11 -O2 -D_GNU_SOURCE -march=$MARCH -mabi=lp64d
-ffp-contract=off -fopenmp
-Wno-error=incompatible-pointer-types -Wno-error=int-conversion
-Wno-error=implicit-function-declaration"

cd "$ROOT"
echo "[riscv] CC=$CC MARCH=$MARCH OUT=$OUT"
# shellcheck disable=SC2086
$CC $CFLAGS $INCS $SRC -o "$OUT" -lm -lpthread $VLLM_RISCV_EXTRA

echo "[riscv] OK: $OUT"
if command -v file >/dev/null 2>&1; then file "$OUT"; fi
