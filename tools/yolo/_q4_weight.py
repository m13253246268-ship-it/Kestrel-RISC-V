"""统计 YOLO 各 conv 层的工作量加权 Kp 分布 + Q4 净收益估算。

读 yolov8n/s/m/l/x，前向 hook 拿每层真实 (OC, C, KH, KW, OH, OW)，
按 OC*P*Kp 加权统计 Kp 落在哪一档，再套用板端实测的 Q4/Q8 比值。
"""
import argparse
import sys

import numpy as np
import torch

# 板端实测（K1, VLEN=256, 4-oc 块, 真实层形状）：Kp -> Q4/Q8 时间比
# 取自 bench_q4_vs_q8.c 两遍（reps=7/11）中位数的中位，非外推。
MEAS = {32: 1.085, 96: 1.150, 256: 1.163, 576: 1.218, 1152: 0.984,
        1728: 1.342, 2304: 1.024, 3456: 1.036, 5184: 1.032}


def ratio_for(kp):
    """分段线性插值；超出 [32,1152] 用端点值。"""
    xs = sorted(MEAS)
    if kp <= xs[0]:
        return MEAS[xs[0]]
    if kp >= xs[-1]:
        return MEAS[xs[-1]]
    for i in range(len(xs) - 1):
        a, b = xs[i], xs[i + 1]
        if a <= kp <= b:
            t = (kp - a) / (b - a)
            return MEAS[a] * (1 - t) + MEAS[b] * t
    return 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    from ultralytics.nn.tasks import DetectionModel  # noqa: F401
    ck = torch.load(args.pt, map_location="cpu", weights_only=False)
    net = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    net.float().eval()

    recs = []

    def hook(mod, inp, out):
        w = mod.conv.weight
        oc, ic, kh, kw = w.shape
        g = int(mod.conv.groups)
        oh, ow = int(out.shape[2]), int(out.shape[3])
        recs.append((oc, ic // g, kh, kw, oh, ow, g))

    hs = []
    for m in net.modules():
        if type(m).__name__ in ("Conv", "DWConv") and hasattr(m, "conv"):
            hs.append(m.register_forward_hook(hook))
    with torch.no_grad():
        net(torch.zeros(1, 3, args.imgsz, args.imgsz))
    for h in hs:
        h.remove()

    # 归一化到与 C 侧一致的 layer 集合：分组卷积（depthwise）走标量回退，不计入
    tot = 0.0
    buckets = {}
    for oc, ic, kh, kw, oh, ow, g in recs:
        if g != 1:
            continue
        K = ic * kh * kw
        Kp = (K + 31) & ~31
        P = oh * ow
        work = oc * P * Kp
        tot += work
        buckets.setdefault(Kp, [0, 0.0])
        buckets[Kp][0] += 1
        buckets[Kp][1] += work

    print(f"\n{args.pt}  @{args.imgsz}")
    print(f"非分组 conv 层数={sum(v[0] for v in buckets.values())}  "
          f"总工作量(OC*P*Kp)={tot/1e6:.1f} M-MAC")
    print(f"{'Kp':>6} {'层数':>5} {'工作量占比':>10} {'实测Q4/Q8':>10} {'加权时间':>10}")
    est = 0.0
    for Kp in sorted(buckets):
        n, wk = buckets[Kp]
        share = wk / tot
        r = ratio_for(Kp)
        est += share * r
        print(f"{Kp:6d} {n:5d} {100*share:9.1f}% {r:10.3f} {share*r:10.4f}")
    print(f"\n合计 Q4/Q8 时间比 ≈ {est:.4f}  "
          f"=> Q4 相对 Q8 {'提速' if est < 1 else '变慢'} {abs(1-est)*100:.1f}%")


if __name__ == "__main__":
    main()
