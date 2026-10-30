#!/bin/bash
# ================================================================
# bitwise_ab.sh - 完整模型位级对拍（②级 gate：RVVon vs RVVoff）
#
# 目的：把「完整模型全层 × 全 token 的位级 gate」固化成一条命令。
#   同一推理在板上跑两臂：
#     A = VLLM_Q4_RVV=3（RVV on）
#     B = VLLM_Q4_RVV=0（RVV off）
#   两臂同为 f32 直算链，代码用 VLLM_PFDUMP / VLLM_PFDUMPTAG 逐层落盘
#   pf_<tag>_hidden.bin（PFH1+base+l+nb+d+u64[nb] 自描述段），拉回本机后
#   用 tools/build/bitwise_hidden.py 比对各 (pos,l) 的 FNV-1a 全哈希。
#   位级 gate 要求 FULL MATCH（退出码 0）。
#
# 只对"f32 直算链"成立（lane-major 模型）。col/int(mode3) 是换数值路径
#   （激活整数量化 + int8 MAC），两臂 hidden 必然 diff，属预期，请走 Part F
#   质量门，不要用本脚本判位级 —— 脚本照实报 diff 并退出码 2。
#
# 用法（Windows 本机需在 WSL 里跑 bash；路径按 /mnt/e/... 访问）：
#   bash tools/build/bitwise_ab.sh [lane|col] [--keep]
#     lane  (默认) : lane-major 模型, A=RVVon(=3) B=RVVoff(=0) INT=0  → 判位级 FULL MATCH
#     col          : col 模型,        A=RVVon(=3,int=1) B=RVVoff(=0,int=1) → 仅验格式/两臂跑通（预期 diff,非位级）
#     --keep       : 保留本地 /tmp 重心转储与日志，不清理
#   env：
#     BRD_USER / BRD_HOST / BRD_SSHPASS  板端 ssh 凭证（缺省 haliu / 192.168.1.25 / 8153019）
#     BRD_MODEL   板端模型路径（缺省 lane→$HOME/kv/model_q4_lane，col→$HOME/kv/model_q4_col）
#     VLLM_THREADS 板上线程数（缺省 8）
#     PFDIR_R   板上临时目录（缺省 /tmp/pfhl）
#
# 退出码透传 bitwise_hidden.py：0=全等 / 1=key不符或无数据 / 2=diff>0 / 3=解析失败
# ================================================================
set -u

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
PY="$ROOT/tools/build/bitwise_hidden.py"
[ -f "$PY" ] || { echo "FATAL: 缺 $PY"; exit 3; }

WHICH=${1:-lane}
KEEP=0
{ [ "${2:-}" = "--keep" ] || [ "${1:-}" = "--keep" ]; } && KEEP=1

BRD_USER=${BRD_USER:-haliu}
BRD_HOST=${BRD_HOST:-192.168.1.25}
BRD_SSHPASS=${BRD_SSHPASS:-8153019}
SSHOPT="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

echo "=== bitwise_ab: WHICH=$WHICH 两臂对拍，本机解析=$(basename "$PY") ==="

# 远端脚本如何感知 WHICH / BRD_MODEL：把 BRD_MODEL 显式传进 heredoc（避免 $1 穿透失效）
if [ "$WHICH" = "col" ]; then
  MODELSEL="${BRD_MODEL:-\$HOME/kv/model_q4_col}"
  RVVON=3; RVOFF=0; INT=1
  echo "MODE=col(int) — 换数值路径，RVVon(vwmacc) vs RVVoff(标量 f32) hidden 必然 diff 属预期；仅验证格式/两臂跑通。"
else
  WHICH=lane
  MODELSEL="${BRD_MODEL:-\$HOME/kv/model_q4_lane}"
  RVVON=3; RVOFF=0; INT=0
  echo "MODE=lane(f32) — 位级口径：两臂同为 f32 直算链，应 FULL MATCH"
fi

export SSHPASS="$BRD_SSHPASS"

sshpass -e ssh $SSHOPT "$BRD_USER@$BRD_HOST" "MODELSEL='${MODELSEL}' WHICH='${WHICH}' RVVON=${RVVON} RVOFF=${RVOFF} INT=${INT} VLLM_THREADS=${VLLM_THREADS:-8} PFDIR_R='${PFDIR_R:-/tmp/pfhl}' bash -s" <<'EOF'
set -u
cd "$HOME/kv" || exit 1
B=./kestrel_src/build-riscv/vllm_kestrel
[ -x "$B" ] || { echo "FATAL: 板端无 $B"; exit 1; }
MODEL="$MODELSEL"
[ -d "$MODEL" ] || { echo "FATAL: 板端无模型 $MODEL"; exit 1; }
export VLLM_THREADS="$VLLM_THREADS" OMP_NUM_THREADS="$VLLM_THREADS"
PF="$PFDIR_R"; rm -rf "$PF"; mkdir -p "$PF"
CTX=8; GEN=2

run_arm() { # $1=tag $2=rvvmode
  local tag=$1 rvv=$2
  VLLM_PFDUMP="$PF" VLLM_PFDUMPTAG="$tag" VLLM_Q4_RVV=$rvv VLLM_Q4_INT=$INT \
    $B --stream-test --model "$MODEL" --stream-ctx $CTX --stream-n $GEN --threads "$VLLM_THREADS" \
    > "$PF/run_$tag.log" 2>&1
  echo "arm[$tag] rvv=$rvv rc=$? hidden_bytes=$(stat -c%s "$PF/pf_${tag}_hidden.bin" 2>/dev/null)"
}
run_arm A $RVVON
run_arm B $RVOFF
echo "PFDIR_R=$PF"
EOF
SSH_RC=$?
[ $SSH_RC -eq 0 ] || { echo "远端执行失败 rc=$SSH_RC"; exit 3; }

# 拉回 payload 文件
LOCAL_T=$(mktemp -d)
sshpass -e scp $SSHOPT -q "$BRD_USER@$BRD_HOST:${PFDIR_R:-/tmp/pfhl}/pf_A_hidden.bin" "$LOCAL_T/pf_A_hidden.bin" 2>/dev/null || true
sshpass -e scp $SSHOPT -q "$BRD_USER@$BRD_HOST:${PFDIR_R:-/tmp/pfhl}/pf_B_hidden.bin" "$LOCAL_T/pf_B_hidden.bin" 2>/dev/null || true

if [ ! -s "$LOCAL_T/pf_A_hidden.bin" ] || [ ! -s "$LOCAL_T/pf_B_hidden.bin" ]; then
  echo "FATAL: 拉回 payload 不完整（A=$(stat -c%s "$LOCAL_T/pf_A_hidden.bin" 2>/dev/null) B=$(stat -c%s "$LOCAL_T/pf_B_hidden.bin" 2>/dev/null)）"
  [ $KEEP -eq 1 ] || rm -rf "$LOCAL_T"
  exit 3
fi

echo "=== hidden 逐 (pos,l) 比对（bitwise_hidden.py）==="
TOL=""
[ "$WHICH" = "col" ] && TOL="--tol-nobits"
python3 "$PY" "$LOCAL_T/pf_A_hidden.bin" "$LOCAL_T/pf_B_hidden.bin" $TOL
RC=$?

[ $KEEP -eq 1 ] && echo "payload kept: $LOCAL_T" || rm -rf "$LOCAL_T"
echo "RESULT_RC=$RC"
exit $RC