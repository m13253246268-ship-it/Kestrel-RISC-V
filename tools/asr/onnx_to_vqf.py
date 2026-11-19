"""Offline ONNX bundle to experimental ASR VQF v6 (not supported by C loaders)."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

MAGIC, VERSION, FLAGS = 0x57465146, 6, 1 << 12
HEADER, ENTRY = 448, 64


def fnv_update(h, data):
    for byte in data:
        h = ((h ^ byte) * 1099511628211) & 0xffffffffffffffff
    return h


def checksum(stream, count, h=14695981039346656037):
    while count:
        data = stream.read(min(count, 4 * 1024 * 1024))
        if not data:
            raise ValueError('Truncated VQF')
        h = fnv_update(h, data)
        count -= len(data)
    return h


def aligned(n, alignment):
    return (n + alignment - 1) // alignment * alignment


class Writer:
    def __init__(self, path):
        self.path = Path(path)
        self.entries = []
        self.arrays = {}
        self.hashes = {}
        self.shared = 0
        # Payload is streamed to a sibling temporary file, avoiding a second RAM copy.
        self.spool = self.path.with_suffix(self.path.suffix + '.payload')
        self.f = self.spool.open('w+b')

    def add(self, array):
        a = np.asarray(array)
        if a.dtype.kind == 'f':
            a = a.astype('<f4', copy=False)
        else:
            a = a.astype(a.dtype.newbyteorder('<'), copy=False)
        a = np.ascontiguousarray(a) if a.ndim else a.copy()
        raw = memoryview(a).cast('B')
        digest = hashlib.sha256(raw).digest()
        key = (digest, a.dtype.str, a.shape)
        if key in self.hashes:
            name = self.hashes[key]
            off, size = self.arrays[name]['offset'], len(raw)
            self.f.flush()
            self.f.seek(off)
            for start in range(0, size, 1024 * 1024):
                if self.f.read(min(size - start, 1024 * 1024)) != raw[start:start + 1024 * 1024]:
                    raise ValueError('Hash collision during full tensor comparison')
            self.shared += 1
            return name
        name = f't{len(self.entries):06d}'
        self.f.seek(0, 2)
        offset = aligned(self.f.tell(), 64)
        self.f.write(bytes(offset - self.f.tell()))
        self.f.write(raw)
        self.entries.append((name, 1 if a.dtype == np.dtype('<f4') else 6,
                             int(a.shape[0]) if a.ndim else 1,
                             int(np.prod(a.shape[1:])) if a.ndim else 1, offset, len(raw)))
        self.arrays[name] = dict(shape=list(a.shape), dtype=a.dtype.str, offset=offset)
        self.hashes[key] = name
        return name

    def finish(self, metadata):
        metadata['arrays'] = {k: {f: v[f] for f in ('shape', 'dtype')} for k, v in self.arrays.items()}
        raw = json.dumps(metadata, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
        self.f.seek(0, 2)
        off = aligned(self.f.tell(), 64)
        self.f.write(bytes(off - self.f.tell()))
        self.f.write(raw)
        self.entries.append(('asr_metadata', 6, 1, len(raw), off, len(raw)))
        dataoff = aligned(HEADER + ENTRY * len(self.entries), 4096)
        filelen = dataoff + self.f.tell()
        header = bytearray(HEADER)
        struct.pack_into('<IIII', header, 0, MAGIC, VERSION, FLAGS, len(self.entries))
        header[16:32] = b'ASR-NUMPY-v1\0\0\0\0'
        struct.pack_into('<QQ', header, 200, dataoff, filelen)
        directory = b''.join(n.encode().ljust(32, b'\0') + struct.pack('<IIIIQQ', q, r, c, 0, dataoff + o, s)
                             for n, q, r, c, o, s in self.entries)
        with self.path.open('w+b') as out:
            out.write(header)
            out.write(directory)
            out.write(bytes(dataoff - out.tell()))
            self.f.seek(0)
            while True:
                block = self.f.read(4 * 1024 * 1024)
                if not block:
                    break
                out.write(block)
            out.flush()
            print(f'Hashing {filelen:,} bytes (FNV-1a)...', flush=True)
            h = fnv_update(14695981039346656037, directory)
            out.seek(dataoff)
            h = checksum(out, filelen - dataoff, h)
            out.seek(filelen)
            out.write(struct.pack('<Q', h))
        self.f.close()
        self.spool.unlink()
        print(f'Wrote {self.path}: {len(self.entries)} entries, {self.shared} fully verified shared tensors', flush=True)


def convert(bundle, output):
    import onnx
    from onnx import numpy_helper, helper
    bundle = Path(bundle)
    metadata = json.loads((bundle / 'metadata.json').read_text(encoding='utf-8'))
    writer = Writer(output)
    result = dict(format='experimental-asr-v6', schema=1, bundle=metadata, graphs={}, assets={})
    try:
        for label in ('audio_hidden', 'lm_cache_prefill', 'lm_cache_decode'):
            path = bundle / metadata['graphs'][label]['path']
            model = onnx.load(path, load_external_data=False)
            graph = dict(inputs=[x.name for x in model.graph.input], outputs=[x.name for x in model.graph.output], tensors={}, nodes=[])
            head_weights = {node.input[1] for node in model.graph.node if node.name == '/lm_head/MatMul'}
            graph['weight_roles'] = {node.name: node.input[1] for node in model.graph.node if node.op_type in ('MatMul', 'Gemm') and len(node.input) > 1}
            for t in model.graph.initializer:
                array = numpy_helper.to_array(t, base_dir=str(path.parent))
                if t.name in head_weights:
                    canonical = t.name + '.stored_transposed'
                    graph['tensors'][canonical] = writer.add(array.T)
                    graph['nodes'].append(dict(op='Transpose', name='vqf_shared_lm_head', inputs=[canonical], outputs=[t.name], attrs={'perm': [1, 0]}))
                else:
                    graph['tensors'][t.name] = writer.add(array)
            for node in model.graph.node:
                attrs = {}
                for attr in node.attribute:
                    value = helper.get_attribute_value(attr)
                    if isinstance(value, onnx.TensorProto):
                        value = {'tensor': writer.add(numpy_helper.to_array(value))}
                    elif isinstance(value, bytes):
                        value = value.decode('utf-8')
                    attrs[attr.name] = value
                if node.op_type == 'Constant':
                    if 'value' not in attrs:
                        raise ValueError(f'Unsupported constant: {node.name}')
                    graph['tensors'][node.output[0]] = attrs['value']['tensor']
                else:
                    graph['nodes'].append(dict(op=node.op_type, name=node.name, inputs=list(node.input), outputs=list(node.output), attrs=attrs))
            result['graphs'][label] = graph
            print(f'{label}: {len(graph["nodes"])} operators exported', flush=True)
        result['embedding'] = writer.add(np.load(bundle / metadata['weights']['token_embedding'], mmap_mode='r'))
        for label in ('lm_cache_prefill', 'lm_cache_decode'):
            graph = result['graphs'][label]
            head = graph['weight_roles']['/lm_head/MatMul'] + '.stored_transposed'
            if graph['tensors'][head] != result['embedding']:
                raise ValueError(f'{label}: full lm_head/embedding sharing verification failed')
        print('Full lm_head matrices equal embedding (both graphs); stored once.', flush=True)
        with np.load(bundle / metadata['weights']['audio_projector']) as p:
            result['projector'] = {k: writer.add(p[k]) for k in p.files}
        for name, path in [('tokenizer', bundle / 'tokenizer.json'), ('feature', bundle / 'qwen3_asr_feature_extractor' / 'preprocessor_config.json')]:
            result['assets'][name] = json.loads(path.read_text(encoding='utf-8'))
        writer.finish(result)
    finally:
        writer.f.close()
        if writer.spool.exists():
            writer.spool.unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    convert(args.bundle, args.out)
