#!/bin/bash
# V51 验收：SiLU 两档
#   默认（=0，精确）：标量 expf + **向量除法** —— 必须仍与锚点逐字节一致
#   VLLM_YOLO_SILU=1（近似）：向量多项式 exp —— 非位级，走误差门
set -u
cd ~/kv/yolo
B=~/kv/kestrel_src/build-riscv/vllm_kestrel
M320=yolov8n_q8_320.vqf
M640=yolov8n_q8.vqf

echo "===== 1. 精确路径（默认）必须仍与改造前锚点逐字节一致 ====="
for M in yolov8n_q8 yolov8n_q8_320; do
  VLLM_THREADS=8 OMP_NUM_THREADS=8 $B --yolo-model "${M}.vqf" \
    --yolo-raw raw.f32 --yolo-raw-out "/tmp/e_${M}.f32" >/dev/null 2>&1
  if cmp -s "anchor_${M}.f32" "/tmp/e_${M}.f32"; then echo "  $M: ANCHOR_MATCH（位级保持）"; else echo "  $M: ANCHOR_DIFF !!"; fi
  VLLM_THREADS=8 OMP_NUM_THREADS=8 VLLM_YOLO_SILU=1 $B --yolo-model "${M}.vqf" \
    --yolo-raw raw.f32 --yolo-raw-out "/tmp/a_${M}.f32" >/dev/null 2>&1
done

echo
echo "===== 2. 近似路径误差门（近似 vs 精确，原始输出 [84,A] 分通道）====="
python3 - <<'EOF'
import numpy as np
for M in ("yolov8n_q8", "yolov8n_q8_320"):
    a = np.fromfile(f"/tmp/e_{M}.f32", np.float32)
    b = np.fromfile(f"/tmp/a_{M}.f32", np.float32)
    if a.size != b.size:
        print(f"  {M}: SIZE MISMATCH"); continue
    A = a.size // 84
    a = a.reshape(84, A); b = b.reshape(84, A)
    ib, ic = a[:4], a[4:]
    jb, jc = b[:4], b[4:]
    sig = lambda v: 1.0 / (1.0 + np.exp(-v.astype(np.float64)))
    sb, sc = sig(ic), sig(jc)
    print(f"  {M}  A={A}")
    print(f"    box 通道  : max|d|={np.abs(ib-jb).max():.3e} mean|d|={np.abs(ib-jb).mean():.3e} (px)")
    print(f"    类别 logit: max|d|={np.abs(ic-jc).max():.3e} mean|d|={np.abs(ic-jc).mean():.3e}")
    print(f"    类别 sigmoid(得分): max|d|={np.abs(sb-sc).max():.3e} p99.9|d|={np.percentile(np.abs(sb-sc),99.9):.3e}")
    top = sc.max(axis=0)
    m = top > 0.25
    print(f"    最高分>0.25 的 anchor: n={int(m.sum())}  得分 max|d|={np.abs(sb[:,m]-sc[:,m]).max():.3e}  "
          f"box max|d|={np.abs(ib[:,m]-jb[:,m]).max():.3e}px")
EOF

echo
echo "===== 3. 检测框一致性（bus.jpg）====="
for M in "$M320" "$M640"; do
  for S in "" "1"; do
    tag="exact"; [ -n "$S" ] && tag="fast"
    VLLM_THREADS=8 OMP_NUM_THREADS=8 VLLM_YOLO_SILU=${S:-0} $B --yolo-model $M \
      --yolo-image bus.jpg --yolo-conf 0.25 --yolo-iou 0.45 2>/dev/null | tail -1 > /tmp/det_${M}_${tag}.json
  done
  if cmp -s /tmp/det_${M}_exact.json /tmp/det_${M}_fast.json; then
    echo "  $M: 检测框 JSON IDENTICAL"
  else
    echo "  $M: 检测框 JSON DIFF"
    echo "    精确: $(cat /tmp/det_${M}_exact.json)"
    echo "    近似: $(cat /tmp/det_${M}_fast.json)"
  fi
done

echo
echo "===== 4. 稳态性能（6 帧，第 2..6 帧）====="
for M in "$M320" "$M640"; do
  for S in 0 1; do
    VLLM_THREADS=8 OMP_NUM_THREADS=8 VLLM_YOLO_CPU_TIMING=1 VLLM_YOLO_FRAMES=6 VLLM_YOLO_SILU=$S \
    $B --yolo-model $M --yolo-image bus.jpg --yolo-conf 0.25 --yolo-iou 0.45 > /tmp/pf_${M}_$S.log 2>&1
    echo "  $M SILU=$S:"
    grep "image_total" /tmp/pf_${M}_$S.log | tail -n +2 | awk '
      { for (i=1;i<=NF;i++){ m=split($i,x,"="); if(m==2 && x[1]=="image_total"){ s+=x[2]+0; n++ } } }
      END { printf "     稳态 image_total 均值 = %.1f ms (n=%d)\n", s/n, n }'
  done
done
