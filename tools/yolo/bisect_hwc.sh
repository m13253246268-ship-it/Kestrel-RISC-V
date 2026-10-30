#!/bin/bash
# 定位 HWC 位级不一致：区分「排列/转置结构问题」与「RVV 路径问题」
set -u
cd ~/kv/yolo
B=~/kv/kestrel_src/build-riscv/vllm_kestrel
M=yolov8n_q8_320.vqf

run() { # $1=hwc $2=rvv $3=out
  VLLM_THREADS=8 OMP_NUM_THREADS=8 VLLM_YOLO_HWC=$1 VLLM_YOLO_RVV=$2 \
  $B --yolo-model $M --yolo-raw raw.f32 --yolo-raw-out "$3" >/dev/null 2>&1
  echo "  hwc=$1 rvv=$2 -> $(stat -c%s "$3" 2>/dev/null) bytes"
}
cmpf() { # $1 $2 $3=label
  if cmp -s "$1" "$2"; then echo "  $3: BIT-EXACT"; else
    echo "  $3: DIFF"
    python3 -c "
import numpy as np,sys
a=np.fromfile('$1',np.float32);b=np.fromfile('$2',np.float32)
n=min(a.size,b.size)
d=np.abs(a-b)
print('    diff=%d/%d  max=%.3e  nonfinite_a=%d nonfinite_b=%d'%((a[:n]!=b[:n]).sum(),n,d[:n].max(),(~np.isfinite(a)).sum(),(~np.isfinite(b)).sum()))
" 2>/dev/null
  fi
}

run 0 0 /tmp/a_chw_sca.f32
run 1 0 /tmp/b_hwc_sca.f32
run 0 1 /tmp/c_chw_rvv.f32
run 1 1 /tmp/d_hwc_rvv.f32
run 1 1 /tmp/d_hwc_rvv2.f32

echo "== 对比 =="
cmpf /tmp/a_chw_sca.f32 /tmp/b_hwc_sca.f32 "标量: CHW vs HWC（纯排列/转置问题）"
cmpf /tmp/a_chw_sca.f32 /tmp/c_chw_rvv.f32 "CHW: 标量 vs RVV"
cmpf /tmp/a_chw_sca.f32 /tmp/d_hwc_rvv.f32 "标量CHW vs RVV-HWC"
cmpf /tmp/d_hwc_rvv.f32 /tmp/d_hwc_rvv2.f32 "HWC RVV 自身可复现性"
