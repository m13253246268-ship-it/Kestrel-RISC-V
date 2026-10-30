#!/bin/sh
# =============================================================================
# laya_gate.sh —— Laya 决策模型：位级门 + 热态极差（板端脚本）
#
# 做什么：
#   A/B 两臂同二进制、同权重、同 case、同线程与亲和性，逐字节比对
#   `VLLM_LAYA_DUMP` 落盘的全部中间量（逐层 l00..l27 / enc / head / qkv / ao /
#   wo / mh / wom）与每个 case 的原始 logits（logitsNNN，raw float32），
#   再各跑 N 次量极差。KV 哈希在本模型上不存在（非自回归、单次前向无 KV cache）。
#
# 用法（板端，pack 目录里要有 laya_f16.bin + laya_manifest.txt + cases）：
#   sh tools/bench/laya_gate.sh
#   PACK=/home/haliu/kv/laya CASES=cases.txt sh tools/bench/laya_gate.sh
#
# 环境变量：
#   EXE      引擎可执行文件          默认 ./vllm_kestrel（相对 PACK）
#   PACK     laya 权重包目录         默认 ./
#   CASES    全量 case 文件          默认 cases.txt
#   OUT      产物目录                默认 ./laya_gate_out
#   N        极差重复次数            默认 5
#   A_ENV    A 臂额外环境变量        默认空（= 部署默认档）
#   B_ENV    B 臂额外环境变量        默认 "VLLM_LAYA_ATTN_RVV=0"（回全标量）
#   CPUS     taskset 核集合          默认 0-7
#   THREADS  OMP/VLLM 线程数         默认 8
#
# 典型用法：
#   验 M10 注意力 RVV：B_ENV="VLLM_LAYA_ATTN_RVV=0"   → 应逐字节相同
#   验 M9 数值档：    B_ENV="VLLM_LAYA_RVV=2"          → 必然不同（非位级），
#                     只看 A 臂极差与 B 臂耗时，不改默认
#
# 产物：$OUT/{outA.txt,outB.txt,oA.txt,oB.txt,A/,B/,c1.txt}
# 依赖：sha256sum/cmp（busybox 亦可）、awk、taskset
# =============================================================================
set -u

PACK=${PACK:-./}
CASES=${CASES:-cases.txt}
EXE=${EXE:-$PACK/vllm_kestrel}
OUT=${OUT:-./laya_gate_out}
N=${N:-5}
A_ENV=${A_ENV:-}
B_ENV=${B_ENV:-VLLM_LAYA_ATTN_RVV=0}
CPUS=${CPUS:-0-7}
THREADS=${THREADS:-8}
export OMP_NUM_THREADS=$THREADS
export VLLM_THREADS=$THREADS

rm -rf "$OUT"; mkdir -p "$OUT/A" "$OUT/B"
head -1 "$PACK/$CASES" > "$OUT/c1.txt"

echo "GOV=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null) NPROC=$(nproc) CPUS=$CPUS THREADS=$THREADS"
echo "BIN=$(sha256sum "$EXE" | awk '{print $1}')"
echo "A_ENV='${A_ENV}'  B_ENV='${B_ENV}'"

run_arm() {   # $1=臂名 A/B  $2=cases  $3=额外环境
    local arm=$1 cs=$2 extra=$3
    # shellcheck disable=SC2086
    env $extra VLLM_LAYA_TIMING=1 VLLM_LAYA_DUMP="$OUT/$arm/x" \
        taskset -c "$CPUS" "$EXE" --laya-pack "$PACK" --laya-cases "$PACK/$cs" \
        > "$OUT/out$arm.txt" 2>&1
}

echo "=== 全量 case：A / B ==="
run_arm A "$CASES" "$A_ENV"
run_arm B "$CASES" "$B_ENV"

echo "=== 1) 档位与 stdout logits ==="
grep -h 'LAYA-RVV' "$OUT/outA.txt" "$OUT/outB.txt" | sed 's/^/  /'
grep '^\[LAYA\] q=' "$OUT/outA.txt" > "$OUT/oA.txt"
grep '^\[LAYA\] q=' "$OUT/outB.txt" > "$OUT/oB.txt"
if cmp -s "$OUT/oA.txt" "$OUT/oB.txt"; then
    echo "STDOUT: IDENTICAL ($(wc -l < "$OUT/oA.txt") cases)"
else
    echo "STDOUT: DIFFERENT"; diff "$OUT/oA.txt" "$OUT/oB.txt" | head -8
fi
head -1 "$OUT/oA.txt" | sed 's/^/  q0: /'

echo "=== 2) 逐字节（中间量 + 原始 logits）==="
same=0; diff_n=0
for f in "$OUT"/A/x.*; do
    b=$(basename "$f")
    if cmp -s "$f" "$OUT/B/$b"; then
        same=$((same + 1))
    else
        diff_n=$((diff_n + 1))
        echo "  DIFF $b"
    fi
done
echo "FILES: SAME=$same DIFF=$diff_n"

echo "=== 3) 极差（单 case，各 $N 次）==="
for arm in A B; do
    extra=$( [ "$arm" = A ] && echo "$A_ENV" || echo "$B_ENV" )
    printf '%s:' "$arm"
    i=0
    while [ "$i" -lt "$N" ]; do
        # shellcheck disable=SC2086
        env $extra VLLM_LAYA_TIMING=1 taskset -c "$CPUS" "$EXE" \
            --laya-pack "$PACK" --laya-cases "$OUT/c1.txt" 2>&1 \
            | grep -oE 'total=[0-9.]+ms' | tr '\n' ' '
        i=$((i + 1))
    done
    echo
done

if [ "$diff_n" -eq 0 ] && [ "$same" -gt 0 ]; then
    echo "GATE: PASS（$same 个文件逐字节相同）"
else
    echo "GATE: DIFFER（$diff_n 个文件不同）"
fi
