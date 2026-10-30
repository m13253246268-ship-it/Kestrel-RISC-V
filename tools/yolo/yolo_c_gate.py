#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yolo_c_gate.py — C 引擎数值门：C 前向 vs torch golden（同权重、同输入）。

M0 的 J1/J2 前置门：证明**自有格式 VQF v4** 被 C 引擎（src/model/vllm_yolo.c）
完整复现。

适用范围（重要）：本门只对 **F32 权重 VQF** 成立（rel tol 1e-4）。Q8 权重
VQF 不能拿 torch golden 当基准 —— 量化本身就有 ~6e-2 的固有偏差（见
yolo_vqf_run.py 的同一 seed 对比）。Q8 的正确验证方式是 **C 引擎 vs
yolo_vqf_run.py 的 numpy 参考执行器**（同一份 VQF、同一量化语义），二者
应当数值近似；这属于「参考执行器对标」，不构成对 torch golden 的证明，
也不构成 C 与 Python 之间位级等价的证明（两者共同实现偏差无法被这种对标
发现）。另见 vllm_yolo.c 的 yconv_q8 注释：C 走 (Σ q*x)*s，Python 参考走
Σ (q*s)*x，舍入次序不同，故 Q8 对标只能给量级而非位级结论。

用法:
  python yolo_c_gate.py --pt <yolov8n.pt> --vqf <yolov8n_f32.vqf> --exe <vllm_kestrel_x64.exe>
"""
import argparse
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True)
    ap.add_argument("--vqf", required=True)
    ap.add_argument("--exe", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--h", type=int, default=640)
    ap.add_argument("--w", type=int, default=640)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    x = rng.standard_normal((3, a.h, a.w)).astype(np.float32)

    tmp = os.environ.get("TEMP", "/tmp")
    fin = os.path.join(tmp, "yolo_gate_in.f32")
    fout = os.path.join(tmp, "yolo_gate_out.f32")
    x.tofile(fin)

    cmd = [a.exe, "--yolo-model", a.vqf, "--yolo-raw", fin, "--yolo-raw-out", fout]
    r = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    if r.returncode != 0:
        print("[gate] C engine FAILED"); return 1

    info = [l for l in r.stdout.splitlines() if l.startswith("YOLO-RAW")]
    oc = oa = 0
    if info:
        parts = dict(p.split("=") for p in info[0].split()[1:])
        oc, oa = int(parts["out_c"]), int(parts["out_a"])
    mine = np.fromfile(fout, dtype=np.float32).reshape(oc, oa)

    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from yolo_vqf_run import load_net
    net = load_net(a.pt)
    with torch.no_grad():
        out = net(torch.from_numpy(x)[None])
    golden = out[0].numpy()[0] if isinstance(out, (tuple, list)) else out.numpy()[0]

    print(f"[gate] C shape={mine.shape}  torch shape={golden.shape}")
    if mine.shape != golden.shape:
        print("[gate] SHAPE MISMATCH"); return 1
    d = np.abs(mine - golden)
    scale = float(np.abs(golden).max()) + 1e-9
    print(f"[gate] max|Δ|={d.max():.3e} mean|Δ|={d.mean():.3e} rel={d.max()/scale:.3e}")
    ok = (d.max() / scale) < 1e-4
    print(f"[gate] C NUMERIC GATE: {'PASS' if ok else 'FAIL'} (rel tol 1e-4)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
