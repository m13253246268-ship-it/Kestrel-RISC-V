"""Exercise malformed W8 directory and scale rejection in the production loader."""
import argparse
from pathlib import Path
import struct
import subprocess
import tempfile


def fnv(data, h=14695981039346656037):
    for b in data:
        h = ((h ^ b) * 1099511628211) & ((1 << 64) - 1)
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('exe', type=Path)
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for case in ('length', 'layout', 'checksum', 'w8_overflow', 'f32_overflow'):
            header = bytearray(448)
            struct.pack_into('<IIII', header, 0, 0x57465146, 6, 4096, 1)
            struct.pack_into('<QQ', header, 200, 4096, 4102)
            entry = bytearray(64)
            entry[:2] = b'w\0'
            struct.pack_into('<IIIIQQ', entry, 32, 0x100, 1, 2, 2 if case == 'layout' else 1, 4096, 5 if case == 'length' else 6)
            payload = b'\x01\x02' + struct.pack('<f', 1.0)
            if case in ('w8_overflow', 'f32_overflow'):
                # Both dimensions are uint32, but byte-length arithmetic can wrap uint64.
                qtype, rows, cols, layout = (0x100, 0xfffffffe, 0xfffffffe, 1) if case == 'w8_overflow' else (1, 0x80000000, 0x80000000, 0)
                struct.pack_into('<IIIIQQ', entry, 32, qtype, rows, cols, layout, 4096, 0)
                struct.pack_into('<QQ', header, 200, 4096, 4096)
                payload = b''
            digest = fnv(payload, fnv(entry))
            model = root / 'bad.vqf'
            model.write_bytes(header + entry + bytes(4096 - 512) + payload + struct.pack('<Q', digest ^ (1 if case == 'checksum' else 0)))
            mapping = root / 'bad.map'
            mapping.write_bytes(b'ASRLM001' + struct.pack('<QI', digest, 103))
            p = subprocess.run([str(args.exe), str(model), str(mapping)], capture_output=True, text=True)
            expected = 'checksum/model map mismatch' if case == 'checksum' else 'invalid directory/shape'
            assert p.returncode and expected in p.stderr, (case, p.returncode, p.stderr)
            print(case, 'rejected')
        for case in ('nan', 'zero', 'negative', 'infinity'):
            p = subprocess.run([str(args.exe), '--bad-scale', case], capture_output=True, text=True)
            assert p.returncode and 'invalid W8 scale' in p.stderr, (case, p.stderr)
            print(case, 'scale rejected')


if __name__ == '__main__':
    main()
