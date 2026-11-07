"""Strict KDBYTES1 validation and byte comparison for single-token Q8/nof32 runs."""
import argparse
import json
import re
import struct
import sys
from pathlib import Path

NONE = 0xFFFFFFFF


def validate(path, steps):
    counts = {str(i): 0 for i in range(1, 10)}
    tokens = []
    with Path(path).open("rb") as f:
        def read(n):
            data = f.read(n)
            if len(data) != n:
                raise ValueError(f"{path}: truncated at {f.tell()}, need {n} bytes")
            return data

        if read(8) != b"KDBYTES1":
            raise ValueError(f"{path}: bad magic")
        header = struct.unpack("<14I", read(56))
        version, endian, dim, layers, heads, kvheads, hd, ffn, vocab, bs, capacity, sparse, k, minctx = header
        if version != 1 or endian != 0x01020304:
            raise ValueError(f"{path}: unsupported version/byte order")
        if not all((dim, layers, heads, kvheads, hd, ffn, vocab, bs, capacity)) or layers > 64:
            raise ValueError(f"{path}: invalid shape")
        if heads % kvheads or steps > capacity or sparse not in (0, 1):
            raise ValueError(f"{path}: inconsistent shape/steps")
        read(12)  # raw norm_eps, rope_theta, sparse_ratio
        total = 0

        def record(tag, pos, layer, size, capture=False):
            nonlocal total
            actual = struct.unpack("<4I", read(16))
            expected = (tag, pos, layer, size)
            if actual != expected:
                raise ValueError(f"{path}: record at {f.tell()-16}: {actual} != {expected}")
            if capture:
                data = read(size)
            else:
                remaining = size
                while remaining:
                    n = min(remaining, 65536)
                    read(n)
                    remaining -= n
                data = None
            counts[str(tag)] += 1
            total += 1
            return data

        for pos in range(steps):
            start, start_count = f.tell(), total
            token, = struct.unpack("<I", record(1, pos, NONE, 4, True))
            if token >= vocab:
                raise ValueError(f"{path}: invalid token at {pos}")
            tokens.append(token)
            for layer in range(layers):
                record(2, pos, layer, dim * 4)
            for layer in range(layers):
                for tag in (3, 4):
                    record(tag, pos, layer, (pos + 1) * kvheads * hd)
                for tag in (5, 6):
                    record(tag, pos, layer, (pos + 1) * kvheads * 4)
            record(7, pos, NONE, vocab * 4)
            nbytes, nrecords = f.tell() - start, total - start_count
            footer = struct.unpack("<QQ", record(8, pos, NONE, 16, True))
            if footer != (nrecords, nbytes):
                raise ValueError(f"{path}: bad step footer at {pos}")
        nbytes, nrecords = f.tell(), total
        footer = struct.unpack("<QQ", record(9, steps, NONE, 16, True))
        if footer != (nrecords, nbytes) or f.read(1):
            raise ValueError(f"{path}: invalid final footer/trailing bytes")
        return {"steps": steps, "layers": layers, "dim": dim, "kv_heads": kvheads,
                "head_dim": hd, "vocab": vocab, "records": total, "counts": counts,
                "bytes": f.tell(), "tokens": tokens,
                "kv_row_snapshots_per_kind": layers * steps * (steps + 1) // 2}


def tokids(path, count, steps):
    text = Path(path).read_bytes().decode("utf-8", errors="replace")
    prefix = "[STREAM] TOKIDS:"
    if text.count(prefix) != 1:
        raise ValueError(f"{path}: missing/duplicate TOKIDS prefix")
    prompts = re.findall(r"\[STREAM\] prompt tokens = (\d+)", text)
    decoded = re.findall(r"\[STREAM\] decoded (\d+) tokens,", text)
    if len(prompts) != 1 or decoded != [str(count)] or int(prompts[0]) + count != steps:
        raise ValueError(f"{path}: inconsistent prompt/decode/expected step count")
    block = text.split(prefix, 1)[1].split("[STREAM] decoded", 1)[0]
    rows = []
    for line in block.splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r"\s*(\d+) t=(\d+) ms=\d+ rss=-?\d+kB\s*", line)
        if not match:
            raise ValueError(f"{path}: malformed/interleaved TOKIDS row: {line!r}")
        rows.append(tuple(map(int, match.groups())))
    if len(rows) != count or [r[1] for r in rows] != list(range(1, count + 1)):
        raise ValueError(f"{path}: incomplete TOKIDS (including prefixed first row)")
    return [r[0] for r in rows]


def compare_bytes(a, b):
    offset = 0
    with Path(a).open("rb") as fa, Path(b).open("rb") as fb:
        while True:
            x, y = fa.read(65536), fb.read(65536)
            if x != y:
                mismatch = next((i for i, (u, v) in enumerate(zip(x, y)) if u != v), min(len(x), len(y)))
                raise ValueError(f"raw byte mismatch at offset {offset + mismatch}")
            if not x:
                return offset
            offset += len(x)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("baseline")
    p.add_argument("candidate")
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--tokens", type=int, required=True, help="expected generated TOKIDS count")
    p.add_argument("--log-a", required=True)
    p.add_argument("--log-b", required=True)
    args = p.parse_args()
    try:
        if args.steps <= 0 or args.tokens <= 0 or args.tokens >= args.steps:
            raise ValueError("expected steps/tokens must be positive, tokens < steps")
        a, b = validate(args.baseline, args.steps), validate(args.candidate, args.steps)
        ta = tokids(args.log_a, args.tokens, args.steps)
        tb = tokids(args.log_b, args.tokens, args.steps)
        if ta != tb or a["tokens"][-args.tokens:] != ta or b["tokens"][-args.tokens:] != tb:
            raise ValueError("complete TOKIDS or dump input-token suffix mismatch")
        compared = compare_bytes(args.baseline, args.candidate)
        a.pop("tokens")
        print(json.dumps({"result": "PASS", "comparison": "raw bytes (no hash equality inference)",
                          "compared_bytes": compared, "integrity": a, "TOKIDS": ta}, indent=2))
        return 0
    except (OSError, ValueError, struct.error) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
