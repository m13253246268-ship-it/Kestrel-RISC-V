"""Offline, lossless ParaformerStreaming checkpoint to independent VQF v7.

Deployment needs a Paraformer v7 C loader; Audio8 v6 cannot load this family.
Tensor names retain the official names in the embedded manifest. Directory names
are stable ordinal IDs; convolution dimensions remain in the manifest.
"""
import argparse
import json
import math
from pathlib import Path
import re
import struct

import numpy as np
import torch
import yaml

MAGIC, VERSION, FLAGS = 0x57465146, 7, 1 << 13
HEADER, ENTRY = 448, 64


def expected_shapes():
    shapes = {}

    def add(prefix, name, shape):
        shapes[prefix + '.' + name] = shape

    def norm(prefix, width):
        add(prefix, 'weight', (width,))
        add(prefix, 'bias', (width,))

    def linear(prefix, rows, cols, bias=True):
        add(prefix, 'weight', (rows, cols))
        if bias:
            add(prefix, 'bias', (rows,))

    for i in range(50):
        p = 'encoder.encoders0.0' if i == 0 else f'encoder.encoders.{i - 1}'
        width = 560 if i == 0 else 512
        linear(p + '.self_attn.linear_out', 512, 512)
        linear(p + '.self_attn.linear_q_k_v', 1536, width)
        add(p, 'self_attn.fsmn_block.weight', (512, 1, 11))
        linear(p + '.feed_forward.w_1', 2048, 512)
        linear(p + '.feed_forward.w_2', 512, 2048)
        norm(p + '.norm1', width)
        norm(p + '.norm2', 512)
    norm('encoder.after_norm', 512)
    for i in range(17):
        p = f'decoder.decoders.{i}' if i < 16 else 'decoder.decoders3.0'
        linear(p + '.feed_forward.w_1', 2048, 512)
        linear(p + '.feed_forward.w_2', 512, 2048, False)
        norm(p + '.feed_forward.norm', 2048)
        norm(p + '.norm1', 512)
        if i < 16:
            add(p, 'self_attn.fsmn_block.weight', (512, 1, 11))
            linear(p + '.src_attn.linear_q', 512, 512)
            linear(p + '.src_attn.linear_k_v', 1024, 512)
            linear(p + '.src_attn.linear_out', 512, 512)
            norm(p + '.norm2', 512)
            norm(p + '.norm3', 512)
    norm('decoder.after_norm', 512)
    linear('decoder.output_layer', 8404, 512)
    add('decoder', 'embed.0.weight', (8404, 512))
    add('predictor', 'cif_conv1d.weight', (512, 512, 3))
    add('predictor', 'cif_conv1d.bias', (512,))
    linear('predictor.cif_output', 1, 512)
    return shapes


def load_assets(model):
    config = yaml.safe_load((model / 'config.yaml').read_text(encoding='utf-8'))
    required = {
        'model': 'ParaformerStreaming', 'encoder': 'SANMEncoderChunkOpt',
        'decoder': 'ParaformerSANMDecoder', 'predictor': 'CifPredictorV2',
        'frontend': 'WavFrontendOnline',
    }
    for key, value in required.items():
        if config.get(key) != value:
            raise ValueError(f'Unsupported {key}: {config.get(key)!r}')
    checks = {
        'encoder_conf': dict(output_size=512, attention_heads=4, linear_units=2048,
                             num_blocks=50, input_layer='pe_online', normalize_before=True,
                             kernel_size=11, sanm_shfit=0),
        'decoder_conf': dict(attention_heads=4, linear_units=2048, num_blocks=16,
                             att_layer_num=16, kernel_size=11, sanm_shfit=5),
        'predictor_conf': dict(idim=512, threshold=1.0, l_order=1, r_order=1,
                              tail_threshold=0.45),
        'frontend_conf': dict(fs=16000, window='hamming', n_mels=80,
                             frame_length=25, frame_shift=10, lfr_m=7, lfr_n=6),
    }
    for section, values in checks.items():
        for key, value in values.items():
            if config[section].get(key) != value:
                raise ValueError(f'Unsupported {section}.{key}: {config[section].get(key)!r}')
    tokens = json.loads((model / 'tokens.json').read_text(encoding='utf-8'))
    if not isinstance(tokens, list) or len(tokens) != 8404 or not all(isinstance(t, str) for t in tokens):
        raise ValueError('Expected 8404 UTF-8 tokens')
    text = (model / 'am.mvn').read_text(encoding='utf-8')
    vectors = []
    for tag in ('AddShift', 'Rescale'):
        match = re.search(r'<' + tag + r'>.*?<LearnRateCoef>[^\[]*\[([^\]]+)\]', text, re.S)
        if match is None:
            raise ValueError(f'Missing CMVN {tag}')
        values = np.fromstring(match.group(1), sep=' ', dtype=np.float32)
        if values.shape != (560,) or not np.isfinite(values).all():
            raise ValueError(f'Invalid CMVN {tag}')
        vectors.append(values)
    state = torch.load(model / 'model.pt', map_location='cpu', weights_only=True)
    expected = expected_shapes()
    if set(state) != set(expected):
        raise ValueError(f'Checkpoint keys differ: missing={set(expected) - set(state)}, extra={set(state) - set(expected)}')
    for name, shape in expected.items():
        tensor = state[name]
        if tensor.dtype != torch.float32 or tuple(tensor.shape) != shape:
            raise ValueError(f'{name}: expected FP32 {shape}, got {tensor.dtype} {tuple(tensor.shape)}')
        if not torch.isfinite(tensor).all():
            raise ValueError(f'{name}: nonfinite weights')
    return config, tokens, vectors, state


def align(value, boundary=64):
    return (value + boundary - 1) // boundary * boundary


def fnv(data, value=14695981039346656037):
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & 0xffffffffffffffff
    return value


def convert(model, output, check=False):
    torch.set_num_threads(1)
    config, tokens, cmvn, state = load_assets(model)
    arrays = [(name, tensor.detach().numpy()) for name, tensor in sorted(state.items())]
    arrays += [('frontend.cmvn_shift', cmvn[0]), ('frontend.cmvn_scale', cmvn[1])]
    manifest = dict(family='paraformer-streaming', schema=1, config=config, tensors={})
    entries, payloads = [], []
    for i, (name, array) in enumerate(arrays):
        ident = f'p{i:04d}'
        array = np.ascontiguousarray(array, dtype='<f4')
        manifest['tensors'][name] = dict(name=ident, shape=list(array.shape))
        payloads.append((ident, 1, array.shape[0], math.prod(array.shape[1:]), memoryview(array).cast('B')))
    for name, obj in [('pf_manifest', manifest), ('pf_tokens', tokens)]:
        raw = json.dumps(obj, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
        payloads.append((name, 6, 1, len(raw), raw))
    data_offset = align(HEADER + ENTRY * len(payloads), 4096)
    offset = data_offset
    for name, qtype, rows, cols, raw in payloads:
        offset = align(offset)
        entries.append(struct.pack('<32sIIIIQQ', name.encode(), qtype, rows, cols, 0, offset, len(raw)))
        offset += len(raw)
    header = bytearray(HEADER)
    struct.pack_into('<IIII', header, 0, MAGIC, VERSION, FLAGS, len(entries))
    header[16:32] = b'PARAFORMER-v1\0\0\0'
    struct.pack_into('<QQ', header, 200, data_offset, offset)
    directory = b''.join(entries)
    temporary = output.with_suffix(output.suffix + '.tmp')
    try:
        with temporary.open('w+b') as stream:
            stream.write(header)
            stream.write(directory)
            stream.write(bytes(data_offset - stream.tell()))
            for entry, (_, _, _, _, raw) in zip(entries, payloads):
                pos = struct.unpack_from('<Q', entry, 48)[0]
                stream.write(bytes(pos - stream.tell()))
                stream.write(raw)
            stream.flush()
            print(f'Hashing {offset:,} bytes; FP32 payloads preserved', flush=True)
            digest = fnv(directory)
            stream.seek(data_offset)
            remaining = offset - data_offset
            while remaining:
                block = stream.read(min(4 * 1024 * 1024, remaining))
                digest = fnv(block, digest)
                remaining -= len(block)
            stream.write(struct.pack('<Q', digest))
            if check:
                for entry, (_, _, _, _, raw) in zip(entries, payloads):
                    stream.seek(struct.unpack_from('<Q', entry, 48)[0])
                    if stream.read(len(raw)) != raw:
                        raise ValueError('Payload readback mismatch')
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f'Wrote {output}: {len(state)} weights, CMVN and tokens; native forward not verified')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--check', action='store_true', help='Compare every payload byte after writing')
    args = parser.parse_args()
    convert(args.model, args.out, args.check)
