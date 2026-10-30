#!/bin/bash
# ================================================================
# bitwise_close.sh - ③级 关闭回归 字节级对拍 + ④级 多形状扫描
#
# ③级 关闭回归：加速臂(=3) vs 逃生舱(=0) 跑同预填，比较「原始输出字节」。
#   依据 perf_axiom_bitwise_gate_v1 第 3 条：逃生舱关闭后，原始输出字节、
#   history_tokens/TOKIDS 与基线**逐字节一致**（仅文本/top-1 相同不算数）。
#   复用 pf_dump_post 的 PFD2 格式：f32[vc] logits 原始 bytes + per(l,pos) KV 行哈希。
#   正文红线：必须逐字节比较原始 float + TOKIDS，不可只对文本。
#
# ④级 多形状：同一脚本可对多个 (ctx, gen) 形状跑同一对拍，遍历验证
#   gate 在 hd / n / token 数变化时仍成立。
#
# 用法:
#   bash tools/build/bitwise_close.sh [lane|col] [ctxA,ctxB,...] [gen] [--keep]
#     参数1  = lane(默认,f32位级口径) | col(int,预期值非位级,仅验证格式)
#     参数2  = 逗号分隔的 ctx 列表（默认 "8,16,32"）→ ④多形状扫描
#     参数3  = gen（默认 2）
#     --keep 保留板上 /tmp 与本地拉回产物
#   env: BRD_USER/BRD_HOST/BRD_SSHPASS/BRD_MODEL/VLLM_THREADS/PFDIR_R（同 bitwise_ab.sh）
#   env: DISABLE_CLOSE=1  跳过③字节比对（仅④形状对齐检查）
#
# 退出码:
#   0 = 全部形状通过（logits bytes FULL MATCH + KV FULL MATCH + TOKIDS 一致）
#   1 = key/header 不符或某臂缺数据
#   2 = 任一形状存在字节级差异
#   3 = 解析失败 / 远端执行失败
# ================================================================
set -u

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
PY_PFD2="$ROOT/tools/build/bitwise_pfd2.py"
PY_HID="$ROOT/tools/build/bitwise_hidden.py"
[ -f "$PY_PFD2" ] || { echo "FATAL: 缺 $PY_PFD2"; exit 3; }
[ -f "$PY_HID" ] || { echo "FATAL: 缺 $PY_HID"; exit 3; }

WHICH=${1:-lane}
CTXS=${2:-8,16,32}
GEN=${3:-2}
KEEP=0
for a in "$@"; do [ "$a" = "--keep" ] && KEEP=1; done

BRD_USER=${BRD_USER:-haliu}
BRD_HOST=${BRD_HOST:-192.168.1.25}
BRD_SSHPASS=${BRD_SSHPASS:-8153019}
SSHOPT="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

if [ "$WHICH" = "col" ]; then
  MODELSEL="${BRD_MODEL:-\$HOME/kv/model_q4_col}"; A_RVV=3; B_RVV=0; INT=1
  echo "MODE=col(int) — 换数值路径，两臂 hidden/logits 非位级属预期；仅验证格式+TOKIDS。"
else
  WHICH=lane
  MODELSEL="${BRD_MODEL:-\$HOME/kv/model_q4_lane}"; A_RVV=3; B_RVV=0; INT=0
  echo "MODE=lane(f32) — 关闭回归位级口径：加速(=3) vs 逃生舱(=0) logits 字节须 FULL MATCH。"
fi

export SSHPASS="$BRD_SSHPASS"

# 远端：对每个 ctx 跑 A/B 两臂，产出 PFD2 bin + run log，规范化文件名为 pf_{tag}_{ctx}
sshpass -e ssh $SSHOPT "$BRD_USER@$BRD_HOST" \
  "MODELSEL='${MODELSEL}' WHICH='${WHICH}' A_RVV=${A_RVV} B_RVV=${B_RVV} INT=${INT} GEN=${GEN} CTXS='${CTXS}' VLLM_THREADS=${VLLM_THREADS:-8} PFDIR_R='${PFDIR_R:-/tmp/pfcl}' bash -s" <<'EOF'
set -u
cd "$HOME/kv" || exit 1
B=./kestrel_src/build-riscv/vllm_kestrel
[ -x "$B" ] || { echo "FATAL: 板端无 $B"; exit 1; }
MODEL="$MODELSEL"
[ -d "$MODEL" ] || { echo "FATAL: 板端无模型 $MODEL"; exit 1; }
export VLLM_THREADS="$VLLM_THREADS" OMP_NUM_THREADS="$VLLM_THREADS"
PF="$PFDIR_R"; rm -rf "$PF"; mkdir -p "$PF"

run_arm() { # $1=tag $2=rvv $3=ctx
  local tag=$1 rvv=$2 ctx=$3
  VLLM_PFDUMP="$PF" VLLM_PFDUMPTAG="$tag" VLLM_Q4_RVV=$rvv VLLM_Q4_INT=$INT \
    $B --stream-test --model "$MODEL" --stream-ctx $ctx --stream-n $GEN --threads "$VLLM_THREADS" \
    > "$PF/run_${tag}_ctx${ctx}.log" 2>&1
  echo "arm[$tag] ctx=$ctx rvv=$rvv rc=$? hidden=$(stat -c%s "$PF/pf_${tag}_hidden.bin" 2>/dev/null)"
}

IFS=',' read -ra CTXLIST <<< "$CTXS"
for ctx in "${CTXLIST[@]}"; do
  run_arm A $A_RVV $ctx
  run_arm B $B_RVV $ctx
  # PFD2 文件名固定 b0_n32，多次会被覆盖；按 ctx 立即改名保留（hidden 为追加累计不用管）
  for t in A B; do
    for p in "$PF"/pf_${t}_b*_n*.bin; do
      [ -e "$p" ] && mv -f "$p" "$PF/pf_${t}_ctx${ctx}.bin"
    done
  done
done
echo "PFDIR_R=$PF"
EOF
SSH_RC=$?
[ $SSH_RC -eq 0 ] || { echo "远端执行失败 rc=$SSH_RC"; exit 3; }

LOCAL_T=$(mktemp -d)
overall=0
# ④ 多形状核心：hidden 为追加累计，覆盖全部 ctx —— 拉一次即可全形状对拍
HA="$LOCAL_T/pf_A_hidden.bin"; HB="$LOCAL_T/pf_B_hidden.bin"
sshpass -e scp $SSHOPT -q "$BRD_USER@$BRD_HOST:${PFDIR_R:-/tmp/pfcl}/pf_A_hidden.bin" "$HA" 2>/dev/null || true
sshpass -e scp $SSHOPT -q "$BRD_USER@$BRD_HOST:${PFDIR_R:-/tmp/pfcl}/pf_B_hidden.bin" "$HB" 2>/dev/null || true
if [ -s "$HA" ] && [ -s "$HB" ]; then
  echo "=== ④ hidden 多形状对拍（累计全部 ctx）==="
  python3 "$PY_HID" "$HA" "$HB"
  rc=$?; [ $rc -gt $overall ] && overall=$rc
else
  echo "FATAL: 拉回 hidden 不完整（A=$(stat -c%s "$HA" 2>/dev/null) B=$(stat -c%s "$HB" 2>/dev/null)）"
  overall=3
fi

IFS=',' read -ra CTXLIST <<< "$CTXS"
for ctx in "${CTXLIST[@]}"; do
  echo "=== ③ 关闭回归 字节级 ctx=$ctx gen=$GEN ==="
  PA="$LOCAL_T/pf_A_ctx${ctx}.bin"; PB="$LOCAL_T/pf_B_ctx${ctx}.bin"
  sshpass -e scp $SSHOPT -q "$BRD_USER@$BRD_HOST:${PFDIR_R:-/tmp/pfcl}/pf_A_ctx${ctx}.bin" "$PA" 2>/dev/null || true
  sshpass -e scp $SSHOPT -q "$BRD_USER@$BRD_HOST:${PFDIR_R:-/tmp/pfcl}/pf_B_ctx${ctx}.bin" "$PB" 2>/dev/null || true
  sshpass -e scp $SSHOPT -q "$BRD_USER@$BRD_HOST:${PFDIR_R:-/tmp/pfcl}/run_A_ctx${ctx}.log" "$LOCAL_T/run_A_ctx${ctx}.log" 2>/dev/null || true
  sshpass -e scp $SSHOPT -q "$BRD_USER@$BRD_HOST:${PFDIR_R:-/tmp/pfcl}/run_B_ctx${ctx}.log" "$LOCAL_T/run_B_ctx${ctx}.log" 2>/dev/null || true
  if [ ! -s "$PA" ] || [ ! -s "$PB" ]; then
    echo "  [ctx=$ctx] FATAL: 拉回 PFD2 不完整（A=$(stat -c%s "$PA" 2>/dev/null) B=$(stat -c%s "$PB" 2>/dev/null)）"
    overall=3; continue
  fi
  if [ -n "${DISABLE_CLOSE:-}" ]; then
    python3 "$PY_PFD2" "$PA" "$PB" || overall=$?
    continue
  fi
  python3 "$PY_PFD2" "$PA" "$PB" --tagA "$LOCAL_T/run_A_ctx${ctx}.log" --tagB "$LOCAL_T/run_B_ctx${ctx}.log"
  rc=$?
  [ $rc -gt $overall ] && overall=$rc
done

[ $KEEP -eq 1 ] && echo "payload kept: $LOCAL_T" || rm -rf "$LOCAL_T"
echo "RESULT_RC=$overall"
exit $overall