#!/usr/bin/env python3
"""bitwise_hidden.py — 逐层 hidden 全哈希对拍（②级位级 gate，V32 配套）。

用法:
  python3 bitwise_hidden.py <pf_A_hidden.bin> <pf_B_hidden.bin> [--tol-nobits]

输入格式（vllm_safetensors.c prefill 层循环写出，每段自描述小端）:
  "PFH1" + u64 base + u64 l + u64 nb + u64 d + u64[nb] hash
  第 i 个 hash = token(base+i) 在这层的 hidden 全向量 hash

语义:
  A / B 两臂（如 VLLM_Q4_RVV=3 vs =0）对同一 (pos,l) 比对。位级 gate 要求 FULL MATCH。
  对 col/int（mode3，换数值路径）两臂值非位级本就应不相等——此脚本照实报 diff 数，
  由调用方结合口径解读；--tol-nobits 下 diff>0 也仅提示"非位级口径"，不判失败。

退出码:
  0 = 两臂 key 完全重叠 且 diff==0
  1 = key 不重叠或一臂无数据
  2 = diff>0（位级不一致，或非位级口径预期）
  3 = 解析失败
"""
import struct, sys

MAGIC = b"PFH1"

def parse(path):
    d = {}
    nseg = 0
    try:
        f = open(path, "rb")
    except FileNotFoundError:
        return d, -1
    except OSError as e:
        print(f"[parse] {path}: {e}", file=sys.stderr)
        return d, -2
    with f:
        while True:
            m = f.read(4)
            if len(m) < 4:
                break
            if m != MAGIC:
                print(f"[parse] {path}: bad magic at offset {f.tell()-4} ({m!r})", file=sys.stderr)
                return d, -3
            hdr = f.read(32)
            if len(hdr) < 32:
                print(f"[parse] {path}: truncated header", file=sys.stderr)
                return d, -3
            base, l, nb, dd = struct.unpack("<4Q", hdr)
            hs = f.read(8 * nb)
            if len(hs) < 8 * nb:
                print(f"[parse] {path}: truncated body lb={l}", file=sys.stderr)
                return d, -3
            for i in range(nb):
                h = int.from_bytes(hs[i*8:(i+1)*8], "little")
                d[(base+i, l)] = h
            nseg += 1
    return d, nseg

def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 3
    a, sa = parse(sys.argv[1])
    b, sb = parse(sys.argv[2])
    if sa < 0 or sb < 0:
        return 3
    print(f"A segments={sa} entries={len(a)}")
    print(f"B segments={sb} entries={len(b)}")
    keys = sorted(set(a) | set(b))
    if not keys:
        print("NO DATA — 两臂都没产出 hidden")
        return 1
    ka, kb = set(a), set(b)
    print(f"key 交集={len(ka & kb)}  A独有={len(ka - kb)}  B独有={len(kb - ka)}")
    if len(ka ^ kb):
        print("KEY MISMATCH — 两臂覆盖的 (pos,l) 不一致")
        return 1
    diff = [k for k in keys if a[k] != b[k]]
    print(f"total (pos,l)={len(keys)}  diff={len(diff)}")
    if diff:
        d0 = diff[0]
        print(f"首差异 (pos,l)={d0}  A={a[d0]:016x}  B={b[d0]:016x}")
        from collections import Counter
        c = Counter(l for (_, l) in diff)
        layer_dist = " ".join(f"l{l}:{c[l]}" for l in sorted(c))
        print(f"diff 按层分布: {layer_dist}")
        return 2
    print(f"BITWISE HIDDEN: 全等（{len(keys)} entries FULL MATCH）")
    return 0

if __name__ == "__main__":
    sys.exit(main())