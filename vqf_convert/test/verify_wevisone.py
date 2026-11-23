"""Check the local WeVisOne Q4 VQF against its F32 checkpoint (Python + NumPy)."""
import hashlib
import json
from pathlib import Path
import struct
import sys

import numpy as np


def verify(model, vqf):
    with (model / 'model.safetensors').open('rb') as src, vqf.open('rb') as out:
        tensors = json.loads(src.read(int.from_bytes(src.read(8), 'little')))
        base = src.tell()
        header = out.read(448)
        magic, version, flags, count = struct.unpack_from('<4I', header)
        assert (magic, version, count) == (0x57465146, 5, 41) and flags & 64
        arch = struct.unpack_from('<45I', header, 16)
        assert arch[:7] == (2048, 28, 16, 8, 128, 6144, 152680)
        assert arch[10:12] == (151643, 151645)
        assert arch[16:29] == (1, 24, 1024, 16, 4096, 16, 2, 2, 2048, 3, 2304, 3, 3)
        assert arch[29:32] == (5, 11, 17) and arch[33:36] == (24, 20, 20)
        start, end = struct.unpack_from('<QQ', header, 200)
        assert vqf.stat().st_size == end + 8 and start >= 448 + count * 64
        entries = {}
        last = start
        for _ in range(count):
            name, qt, rows, cols, layout, off, size = struct.unpack('<32s4I2Q', out.read(64))
            name = name.rstrip(b'\0').decode()
            assert name not in entries and off % 64 == 0 and last <= off < off + size <= end, name
            entries[name] = qt, rows, cols, off, size
            last = off + size

        def compare(section, names):
            qt, rows, cols, off, size = entries[section]
            assert rows == len(names) and size == rows * (cols * 4 if qt == 1 else cols // 32 * 34)
            for row, name in enumerate(names):
                t = tensors['model.visual.' + name]
                assert t['dtype'] == 'F32' and np.prod(t['shape']) == cols, name
                srcpos = base + t['data_offsets'][0]
                dstpos = off + row * size // rows
                if qt == 1:
                    src.seek(srcpos)
                    out.seek(dstpos)
                    for i in range(0, cols * 4, 1 << 20):
                        n = min(1 << 20, cols * 4 - i)
                        assert src.read(n) == out.read(n), name
                else:
                    assert qt == 2
                    for block in (0, cols // 64, cols // 32 - 1):
                        src.seek(srcpos + block * 128)
                        out.seek(dstpos + block * 34)
                        ref = np.frombuffer(src.read(128), '<f4')
                        scale, *q = struct.unpack('<e32b', out.read(34))
                        assert np.max(np.abs(ref - scale * np.array(q))) <= scale * 0.65 + 1e-7, name

        for section, name in [('patch_embed', 'patch_embed.proj.weight'), ('patch_bias', 'patch_embed.proj.bias'), ('pos_embed', 'pos_embed.weight')]:
            compare('v_' + section, [name])
        for section, name in [('qkv', 'attn.qkv'), ('proj', 'attn.proj'), ('fc1', 'mlp.linear_fc1'), ('fc2', 'mlp.linear_fc2')]:
            for field, key in [('weight', 'q8_' + section), ('bias', section + '_bias')]:
                compare('v_' + key, [f'blocks.{i}.{name}.{field}' for i in range(24)])
        for field, suffix in [('weight', 'w'), ('bias', 'b')]:
            for n in ('norm1', 'norm2'):
                compare(f'v_{n}_{suffix}', [f'blocks.{i}.{n}.{field}' for i in range(24)])
            for key, n in [('n', 'norm'), ('f1', 'linear_fc1'), ('f2', 'linear_fc2')]:
                compare(f'v_merger_{key}{suffix}', [f'merger.{n}.{field}'])
                compare(f'v_ds_{key}{suffix}', [f'deepstack_merger_list.{i}.{n}.{field}' for i in range(3)])
        out.seek(0)
        digest = hashlib.file_digest(out, 'sha256').hexdigest()
        if len(sys.argv) > 3 and digest != sys.argv[3].lower():
            raise ValueError(f'SHA256 mismatch: {digest}')
        print(f'PASS: {count} sections, 27 vision sections checked, {end + 8} bytes')
        print('SHA256', digest)


if __name__ == '__main__':
    verify(Path(sys.argv[1]), Path(sys.argv[2]))
