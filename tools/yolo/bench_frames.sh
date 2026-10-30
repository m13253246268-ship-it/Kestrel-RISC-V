#!/bin/bash
# V51 稳态口径：连跑 N 帧，对比首帧（冷）与后续帧（稳态）
#   每帧同一张图，输入相同 ⇒ 稳态后每帧数值完全一致，只看时间。
set -u
cd ~/kv/yolo
B=~/kv/kestrel_src/build-riscv/vllm_kestrel
N=${1:-6}

sumf() { # $1=log $2=grep-pattern $3..=keys
  local log=$1 pat=$2; shift 2
  grep "$pat" "$log" | tail -n +2 | awk -v keys="$*" '
    { for (i = 1; i <= NF; i++) { m = split($i, a, "="); if (m == 2) { k = a[1]; v = a[2] + 0;
        for (j = 1; j <= nk; j++) if (k == key[j]) s[j] += v } } cnt++ }
    BEGIN { nk = split(keys, key, " ") }
    END { printf "  n=%d", cnt; for (j = 1; j <= nk; j++) printf "  %s=%.1fms", key[j], s[j] / cnt; printf "\n" }'
}

run() { # $1=model $2=tag
  echo "=== $2，连续 $N 帧，8 线程 ==="
  VLLM_THREADS=8 OMP_NUM_THREADS=8 VLLM_YOLO_CPU_TIMING=1 VLLM_YOLO_FRAMES=$N \
  $B --yolo-model "$1" --yolo-image bus.jpg --yolo-conf 0.25 --yolo-iou 0.45 \
  > /tmp/frames_$2.log 2>&1
  grep "image_total" /tmp/frames_$2.log | nl -w2 -s'  ' | sed 's/(fwd+lb+post)//'
  echo "--- 稳态（第 2..$N 帧）均值 ---"
  sumf /tmp/frames_$2.log "forward breakdown" total conv im2col elem detect
  sumf /tmp/frames_$2.log "letterbox=" letterbox image_total
  echo
}

run yolov8n_q8_320.vqf 320
run yolov8n_q8.vqf 640
