#!/usr/bin/env python3
"""bitwise_pfd2.py — ③级关闭回归字节级对拍（原始输出 bytes + KV 行哈希）。

用法:
  python3 bitwise_pfd2.py <pf_A_b<base>_n<n>.bin> <pf_B_b<base>_n<n>.bin> [--tagA A --tagB B]

输入（vllm_safetensors.c pf_dump_post 写出，PFD2 格式）:
  "PFD2" + u32 mode + u64[8] (n_tokens, base, nl, kv_bs, kv_dim, nkv, hd, vc)
       + u64 flags + f32[vc] logits + per(l,pos) 6×u64（K/V f32、K/V q8、K/V scale 行哈希）

语义:
  A/B 两臂（如 VLLM_Q4_RVV=3 加速 vs =0 逃生舱）跑同一预填。
  ③级要求"原始输出字节"逐字节一致：即 logits 区 f32[vc] 的 **字节** 完全相等
  （不是数值近似），K/V 行哈希亦全等。正文红线：生成文本/top-1 相同不构成证明，
  必须逐字节比较原始 float。

  旁路: --tagA/--tagB 同时给两个 run log，脚本抓取 `[STREAM] TOKIDS:` 行并逐字节比较。

退出码:
  0 = 两臂 header 一致、logits 字节全等、KV 行哈希全等（TOKIDS 若给出亦全等）
  1 = key 覆盖不一致 / 一臂缺文件 / header 不匹配
  2 = logits 字节有差异 或 KV 行哈希有差异 或 TOKIDS 不一致
  3 = 解析失败
"""
import struct, sys, re

MAGIC = b"PFD2"

def parse(path):
    f = open(path, "rb")
    with f:
        m = f.read(4)
        if m != MAGIC:
            print(f"[parse] {path}: bad magic {m!r}", file=sys.stderr); return None
        mode = struct.unpack("<I", f.read(4))[0]
        hdr = struct.unpack("<8Q", f.read(64))
        flags = struct.unpack("<Q", f.read(8))[0]
        n_tokens, base, nl, kv_bs, kv_dim, nkv, hd, vc = hdr
        logits = f.read(4 * vc)                       # 原始 float bytes
        kv = []                                        # per(l,pos): 6×u64
        for _ in range(nl):
            for _ in range(n_tokens):
                k = f.read(48)
                if len(k) < 48:
                    return None
                kv.append(struct.unpack("<6Q", k))
    return {"mode": mode, "hdr": hdr, "flags": flags, "logits": logits,
            "kv": kv, "n_tokens": n_tokens, "base": base, "nl": nl, "vc": vc,
            "kv_dim": kv_dim}

def grab_tokids(path):
    if not path:
        return None
    try:
        txt = open(path, "r", errors="replace").read()
    except OSError:
        return None
    for line in txt.splitlines():
        if "[STREAM] TOKIDS:" in line:
            return line.split("TOKIDS:", 1)[1].strip()
    return None  # 找不到（该 run 无 decode 阶段）

def main():
    args = sys.argv[1:]
    if len(args) < 2:
        print(__doc__); return 3
    tagA = tagB = None
    rest = []
    i = 0
    while i < len(args):
        if args[i] == "--tagA" and i + 1 < len(args): tagA = args[i+1]; i += 2
        elif args[i] == "--tagB" and i + 1 < len(args): tagB = args[i+1]; i += 2
        else: rest.append(args[i]); i += 1
    if len(rest) < 2:
        print(__doc__); return 3
    A = parse(rest[0]); B = parse(rest[1])
    if A is None or B is None:
        return 3
    if A["hdr"] != B["hdr"]:
        print(f"HEADER MISMATCH\n  A={A['hdr']}\n  B={B['hdr']}")
        return 1
    print(f"header A==B: n_tokens={A['n_tokens']} base={A['base']} nl={A['nl']} "
          f"kv_dim={A['kv_dim']} vc={A['vc']} flags={A['flags']:#x}")
    # 1) logits 原始 float 字节直接比
    la, lb = A["logits"], B["logits"]
    blen = min(len(la), len(lb))
    nbyte = sum(1 for i in range(blen) if la[i] != lb[i])
    if len(la) != len(lb):
        print(f"LOGITS BYTE LENGTH DIFF: A={len(la)} B={len(lb)}")
        return 2
    print(f"logits bytes={len(la)}  diff_bytes={nbyte}")
    if nbyte:
        # 报首差分位（字节级）
        first = next(i for i in range(blen) if la[i] != lb[i])
        float_idx = first // 4
        print(f"首 byte 差异 @ off={first} (float[{float_idx}])  "
              f"A={struct.unpack('<f', la[float_idx*4:float_idx*4+4])[0]!r} "
              f"B={struct.unpack('<f', lb[float_idx*4:float_idx*4+4])[0]!r}")
        return 2
    # 2) KV 行哈希全等
    if A["kv"] != B["kv"]:
        nk = sum(1 for a, b in zip(A["kv"], B["kv"]) if a != b)
        l0 = next(i for i,(a,b) in enumerate(zip(A["kv"],B["kv"])) if a != b)
        print(f"KV 行哈希 DIFF: {nk} 行不同, 首差异 idx={l0} "
              f"(l={l0//A['n_tokens']} pos={l0%A['n_tokens']+A['base']})  A={A['kv'][l0]} B={B['kv'][l0]}")
        return 2
    print("KV 行哈希: 全等")
    # 3) TOKIDS（若给 log）
    ta, tb = grab_tokids(tagA), grab_tokids(tagB)
    if ta is not None and tb is not None:
        same = ta == tb
        print(f"TOKIDS: {'一致' if same else '不一致'}")
        if not same:
            print(f"  A: {ta}\n  B: {tb}")
            return 2
    elif tagA and tagB:
        print("TOKIDS: 一臂/两臂日志缺 [STREAM] TOKIDS: 行，跳过（logits 字节已判）")
    print("BYTEWISE CLOSE-REGRESSION: 通过（logits 字节=FULL MATCH, KV=FULL MATCH）")
    return 0

if __name__ == "__main__":
    sys.exit(main())