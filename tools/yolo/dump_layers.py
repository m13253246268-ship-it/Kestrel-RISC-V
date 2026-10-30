"""导出 yolov8n 在指定输入尺寸下的真实卷积层形状为 C 数组（供 im2col 微基准用）。

只导出非分组卷积（groups==1），因为分组/depthwise 走的是标量 yconv_q8 回退路径，
不在 int8 dotprod + im2col 的目标范围内。

输出形如：
    static const YLayer k_layers[] = {
        /* C, H, W, KH, KW, sh, sw, ph, pw, OC, OH, OW */
        { 3, 320, 320, 3, 3, 2, 2, 1, 1, 16, 160, 160 },
        ...
    };
"""
import argparse
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--out", required=True)
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
        ih, iw = int(inp[0].shape[2]), int(inp[0].shape[3])
        sh, sw = mod.conv.stride if isinstance(mod.conv.stride, tuple) else (mod.conv.stride,) * 2
        ph, pw = mod.conv.padding if isinstance(mod.conv.padding, tuple) else (mod.conv.padding,) * 2
        recs.append((ic, ih, iw, kh, kw, sh, sw, ph, pw, oc, oh, ow, g))

    hs = []
    for m in net.modules():
        if type(m).__name__ in ("Conv", "DWConv") and hasattr(m, "conv"):
            hs.append(m.register_forward_hook(hook))
    with torch.no_grad():
        net(torch.zeros(1, 3, args.imgsz, args.imgsz))
    for h in hs:
        h.remove()

    lines = []
    total33 = 0
    total11 = 0
    n33 = n11 = 0
    for (ic, ih, iw, kh, kw, sh, sw, ph, pw, oc, oh, ow, g) in recs:
        if g != 1:
            continue
        P = oh * ow
        K = ic * kh * kw
        if kh == 3:
            total33 += P * K
            n33 += 1
        elif kh == 1:
            total11 += P * K
            n11 += 1
        lines.append((ic, ih, iw, kh, kw, sh, sw, ph, pw, oc, oh, ow))

    with open(args.out, "w") as f:
        f.write("/* 自动生成：yolov8n @%dx%d 非分组卷积层形状（勿手改） */\n" % (args.imgsz, args.imgsz))
        f.write("static const YLayer k_layers[] = {\n")
        f.write("    /*   C    H    W  KH KW sh sw ph pw  OC   OH   OW */\n")
        for r in lines:
            f.write("    { %3d, %3d, %3d, %d, %d, %d, %d, %d, %d, %3d, %3d, %3d },\n" % r)
        f.write("};\n")
        f.write("#define K_NLAYERS ((int)(sizeof(k_layers) / sizeof(k_layers[0])))\n")

    print("layers(groups==1) = %d   (3x3: %d, 1x1: %d)" % (len(lines), n33, n11))
    print("work 3x3 = %.1f MPK  1x1 = %.1f MPK  3x3 share = %.1f%%"
          % (total33 / 1e6, total11 / 1e6, 100.0 * total33 / (total33 + total11)))


if __name__ == "__main__":
    main()
