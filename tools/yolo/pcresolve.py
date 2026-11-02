#!/usr/bin/env python3
# pcresolve.py <pc.out> <lib-substring> <lib-path>：pcsamp 输出按 nm -DC 符号聚合
import subprocess, sys, bisect, collections
pcf, key, lib = sys.argv[1:4]
syms = []
for ln in subprocess.run(["nm", "-DC", "--defined-only", lib], capture_output=True, text=True).stdout.splitlines():
    p = ln.split(None, 2)
    if len(p) == 3 and p[1] in "tTwWiu":
        syms.append((int(p[0], 16), p[2]))
syms.sort()
addrs = [a for a, _ in syms]
agg = collections.Counter(); tot = 0
for ln in open(pcf):
    p = ln.split()
    if len(p) == 3 and key in p[1]:
        n, off = int(p[0]), int(p[2], 16)
        i = bisect.bisect_right(addrs, off) - 1
        agg[syms[i][1][:110] if i >= 0 else "?"] += n; tot += n
print("total", tot)
for k, v in agg.most_common(15):
    print("%5d %5.1f%% %s" % (v, 100.0 * v / tot, k))
