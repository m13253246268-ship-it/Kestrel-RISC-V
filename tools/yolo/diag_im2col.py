"""诊断 yolov8n im2col 工作量按 1x1 vs 3x3 的占比（320 与 640 两档）。

im2col 成本 ∝ P×K（每输出位置 gather K 个输入），按 kh×kw 分组统计。
"""
import argparse
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

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

    groups = {}  # (kh,kw) -> [层数, im2col工作量(P*K) 加权]
    tot = 0.0
    for oc, ic, kh, kw, oh, ow, g in recs:
        if g != 1:
            continue
        K = ic * kh * kw
        P = oh * ow
        work = P * K   # im2col gather 元素数
        tot += work
        key = (kh, kw)
        groups.setdefault(key, [0, 0.0])
        groups[key][0] += 1
        groups[key][1] += work

    print(f"yolov8n @{args.imgsz}  非分组 conv 层")
    print(f"{'kernel':>8} {'层数':>4} {'im2col工作量占比':>16} {'工作量(MPK)':>12}")
    for key in sorted(groups):
        n, wk = groups[key]
        print(f"{str(key):>8} {n:>4} {100*wk/tot:>15.1f}% {wk/1e6:>12.1f}")

    # 1x1 vs 3x3 汇总
    w11 = sum(wk for (kh, kw), (n, wk) in groups.items() if kh == 1 and kw == 1)
    w33 = sum(wk for (kh, kw), (n, wk) in groups.items() if kh == 3 and kw == 3)
    w_other = tot - w11 - w33
    print(f"\n1x1: {100*w11/tot:.1f}%  3x3: {100*w33/tot:.1f}%  其他: {100*w_other/tot:.1f}%")
    print(f"(3x3 im2col 当前是标量 gather，1x1 已 RVV 化)")


if __name__ == "__main__":
    main()
