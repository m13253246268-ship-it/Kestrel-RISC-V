"""Experimental ASR VQF v6: independent NumPy forward; ORT only with --verify-bundle."""
import argparse
from collections import Counter
import json
import math
import re
from pathlib import Path
import struct
import wave

import numpy as np
from scipy.special import erf

from onnx_to_vqf import MAGIC, VERSION, FLAGS, checksum, fnv_update


class RowW8:
    """Compressed storage; reference operators materialize only their current weight."""
    dtype = np.dtype('<f4')

    def __init__(self, path, off, shape):
        self.shape = shape
        self.size = math.prod(shape)
        self.ndim = len(shape)
        self.q = np.memmap(path, mode='r', offset=off, dtype='i1', shape=shape)
        self.scales = np.memmap(path, mode='r', offset=off + self.size, dtype='<f4', shape=(shape[0],))
        if not np.all(np.isfinite(self.scales)) or np.any(self.scales <= 0):
            raise ValueError('Invalid W8 scales')

    def __array__(self, dtype=None, copy=None):
        a = self.q.astype(np.float32) * self.scales.reshape((-1,) + (1,) * (self.ndim - 1))
        return a.astype(dtype, copy=False) if dtype is not None else a

    def __getitem__(self, key):
        return self.q[key].astype(np.float32) * np.asarray(self.scales[key]).reshape((-1,) + (1,) * (self.ndim - 1)) if not np.isscalar(key) else self.q[key].astype(np.float32) * self.scales[key]

    @property
    def T(self):
        return np.asarray(self).T


class VQF:
    def __init__(self, path, verify=True):
        self.path = Path(path)
        with self.path.open('rb') as f:
            header = f.read(448)
            if len(header) != 448:
                raise ValueError('Truncated header')
            magic, version, flags, count = struct.unpack_from('<IIII', header)
            if (magic, version, flags) != (MAGIC, VERSION, FLAGS):
                raise ValueError('Expected experimental ASR VQF v6, bit12 (C loaders unsupported)')
            dataoff, filelen = struct.unpack_from('<QQ', header, 200)
            if dataoff % 4096 or dataoff < 448 + count * 64 or self.path.stat().st_size != filelen + 8:
                raise ValueError('Invalid container bounds')
            directory = f.read(count * 64)
            self.entries = {}
            self.shapes = {}
            end = dataoff
            for i in range(count):
                entry = directory[i * 64:(i + 1) * 64]
                name = entry[:32].split(b'\0')[0].decode('ascii')
                qtype, rows, cols, layout, off, size = struct.unpack('<IIIIQQ', entry[32:])
                if name in self.entries or qtype not in (1, 6, 0x100) or layout != (1 if qtype == 0x100 else 0) or off % 64 or off < end or off + size > filelen:
                    raise ValueError('Invalid tensor directory')
                if not rows or not cols or (qtype == 1 and size != rows * cols * 4) or (qtype == 0x100 and size != rows * cols + rows * 4):
                    raise ValueError('Invalid tensor payload length')
                self.shapes[name] = (rows, cols)
                self.entries[name] = (off, size, qtype)
                end = off + size
            if verify:
                print('Verifying VQF FNV-1a...', flush=True)
                f.seek(dataoff)
                h = checksum(f, filelen - dataoff, fnv_update(14695981039346656037, directory))
                if h != struct.unpack('<Q', f.read(8))[0]:
                    raise ValueError('VQF checksum mismatch')
            off, size, _ = self.entries['asr_metadata']
            f.seek(off)
            self.meta = json.loads(f.read(size))
        if self.meta.get('format') != 'experimental-asr-v6' or self.meta.get('schema') != 1:
            raise ValueError('Unknown ASR metadata schema')
        self.tensors = {}
        for name, spec in self.meta['arrays'].items():
            off, size, qtype = self.entries[name]
            dtype = np.dtype(spec['dtype'])
            shape = tuple(spec['shape'])
            if qtype == 0x100:
                if len(shape) < 2 or any(n <= 0 for n in shape) or dtype != np.dtype('i1') or spec.get('quantization') != 'signed-row-f32-v1' or self.shapes[name] != (shape[0], math.prod(shape[1:])) or size != math.prod(shape) + shape[0] * 4:
                    raise ValueError('Invalid W8 metadata')
                self.tensors[name] = RowW8(self.path, off, shape)
                continue
            if 'quantization' in spec:
                raise ValueError('Quantization metadata on unquantized tensor')
            if dtype.hasobject or dtype.kind not in 'fibu' or math.prod(shape) * dtype.itemsize != size:
                raise ValueError(f'Invalid tensor shape/dtype: {name}')
            if qtype == 1 and dtype != np.dtype('<f4'):
                raise ValueError('Invalid F32 tensor')
            self.tensors[name] = np.memmap(self.path, mode='r', offset=off, dtype=dtype, shape=shape)


CAST = {1: np.float32, 2: np.uint8, 3: np.int8, 6: np.int32, 7: np.int64, 9: np.bool_, 10: np.float16, 11: np.float64}


def conv(x, w, b, a):
    if a.get('group', 1) != 1 or a.get('dilations', [1, 1]) != [1, 1]:
        raise ValueError('Only group1, dilation1 convolution supported')
    pads = a.get('pads', [0, 0, 0, 0])
    x = np.pad(x, ((0, 0), (0, 0), (pads[0], pads[2]), (pads[1], pads[3])))
    kh, kw = w.shape[2:]
    sh, sw = a.get('strides', [1, 1])
    windows = np.lib.stride_tricks.sliding_window_view(x, (kh, kw), axis=(2, 3))[:, :, ::sh, ::sw]
    n, _, h, width, _, _ = windows.shape
    result = np.empty((n, w.shape[0], h, width), np.float32)
    weights = w.reshape(w.shape[0], -1).T
    for i in range(n):
        cols = windows[i].transpose(1, 2, 0, 3, 4).reshape(h * width, -1)
        y = cols @ weights
        if b is not None:
            y += b
        result[i] = y.reshape(h, width, -1).transpose(2, 0, 1)
    return result


def operator(op, x, a, tensors):
    x = [np.asarray(v) if isinstance(v, RowW8) else v for v in x]
    if op == 'Identity': return x[0]
    if op == 'Add': return x[0] + x[1]
    if op == 'Sub': return x[0] - x[1]
    if op == 'Mul': return x[0] * x[1]
    if op == 'Div':
        if x[0].dtype.kind in 'iu':
            return np.trunc(x[0] / x[1]).astype(x[0].dtype)
        return x[0] / x[1]
    if op == 'Mod': return np.fmod(*x) if a.get('fmod', 0) else np.mod(*x)
    if op == 'Pow': return np.power(*x)
    if op == 'Sqrt': return np.sqrt(x[0])
    if op == 'Neg': return -x[0]
    if op == 'Erf': return erf(x[0])
    if op == 'Cos': return np.cos(x[0])
    if op == 'Sin': return np.sin(x[0])
    if op == 'Sigmoid': return 1 / (1 + np.exp(-x[0]))
    if op == 'Equal': return np.equal(*x)
    if op == 'Less': return np.less(*x)
    if op == 'LessOrEqual': return np.less_equal(*x)
    if op == 'Greater': return np.greater(*x)
    if op == 'And': return np.logical_and(*x)
    if op == 'Or': return np.logical_or(*x)
    if op == 'Xor': return np.logical_xor(*x)
    if op == 'Not': return np.logical_not(x[0])
    if op == 'Where': return np.where(*x)
    if op == 'Cast': return x[0].astype(CAST[a['to']])
    if op == 'Transpose': return np.transpose(x[0], a.get('perm'))
    if op == 'Reshape':
        shape = [int(v) for v in x[1]]
        if not a.get('allowzero', 0): shape = [x[0].shape[i] if v == 0 else v for i, v in enumerate(shape)]
        return x[0].reshape(shape)
    if op == 'Unsqueeze': return np.expand_dims(x[0], tuple(int(v) for v in (x[1] if len(x) > 1 else a['axes'])))
    if op == 'Squeeze': return np.squeeze(x[0], axis=tuple(int(v) for v in x[1]) if len(x) > 1 else None)
    if op == 'Shape': return np.asarray(x[0].shape[a.get('start', 0):a.get('end', x[0].ndim)], dtype=np.int64)
    if op == 'Gather': return np.take(x[0], x[1], axis=a.get('axis', 0))
    if op == 'Concat': return np.concatenate(x, axis=a['axis'])
    if op == 'Expand': return np.broadcast_to(x[0], np.broadcast_shapes(x[0].shape, tuple(x[1])))
    if op == 'ConstantOfShape':
        value = tensors[a['value']['tensor']] if 'value' in a else np.array([0], np.float32)
        return np.full(tuple(x[0]), value.item(), dtype=value.dtype)
    if op == 'Slice':
        slices = [slice(None)] * x[0].ndim
        axes = x[3] if len(x) > 3 and x[3] is not None else range(len(x[1]))
        steps = x[4] if len(x) > 4 and x[4] is not None else [1] * len(x[1])
        for axis, start, end, step in zip(axes, x[1], x[2], steps):
            slices[int(axis)] = slice(int(start), int(end), int(step))
        return x[0][tuple(slices)]
    if op == 'Clip': return np.clip(x[0], x[1] if len(x) > 1 else None, x[2] if len(x) > 2 else None)
    if op == 'CumSum':
        if a.get('exclusive', 0) or a.get('reverse', 0): raise ValueError('Unsupported cumulative sum')
        return np.cumsum(x[0], axis=int(x[1]), dtype=x[0].dtype)
    if op == 'Trilu': return (np.triu if a.get('upper', 1) else np.tril)(x[0], int(x[1]) if len(x) > 1 else 0)
    if op == 'ScatterElements':
        if a.get('reduction', 'none') != 'none': raise ValueError('Unsupported scatter reduction')
        out = x[0].copy()
        np.put_along_axis(out, x[1], x[2], axis=a.get('axis', 0))
        return out
    if op == 'ReduceMean':
        axes = x[1] if len(x) > 1 else a.get('axes')
        return np.mean(x[0], axis=tuple(int(v) for v in axes) if axes is not None else None, keepdims=bool(a.get('keepdims', 1)))
    if op == 'MatMul': return x[0] @ x[1]
    if op == 'Gemm':
        left = x[0].T if a.get('transA', 0) else x[0]
        right = x[1].T if a.get('transB', 0) else x[1]
        y = (left @ right) * np.float32(a.get('alpha', 1))
        return y + x[2] * np.float32(a.get('beta', 1)) if len(x) > 2 else y
    if op == 'LayerNormalization':
        axes = tuple(range(a.get('axis', -1) % x[0].ndim, x[0].ndim))
        mean = x[0].mean(axis=axes, keepdims=True)
        delta = x[0] - mean
        inv = 1 / np.sqrt((delta * delta).mean(axis=axes, keepdims=True) + np.float32(a.get('epsilon', 1e-5)))
        y = delta * inv * x[1]
        if len(x) > 2: y = y + x[2]
        return (y, mean, inv)
    if op == 'Softmax':
        axis = a.get('axis', -1)
        y = np.exp(x[0] - np.max(x[0], axis=axis, keepdims=True))
        return y / y.sum(axis=axis, keepdims=True)
    if op == 'Conv': return conv(x[0], x[1], x[2] if len(x) > 2 else None, a)
    raise ValueError(f'Unsupported operator {op}')


class Graph:
    def __init__(self, vqf, name):
        self.graph = vqf.meta['graphs'][name]
        self.tensors = vqf.tensors
        self.constants = {k: self.tensors[v] for k, v in self.graph['tensors'].items()}
        self.uses = Counter(n for node in self.graph['nodes'] for n in node['inputs'] if n)
        self.uses.update(self.graph['outputs'])

    def run_outputs(self, feeds, outputs):
        """Evaluate only ancestors of requested reference outputs (mask audits)."""
        needed = set(outputs)
        selected = []
        for node in reversed(self.graph['nodes']):
            if needed.intersection(node['outputs']):
                selected.append(node)
                needed.update(node['inputs'])
        values = dict(self.constants)
        values.update(feeds)
        for node in reversed(selected):
            result = operator(node['op'], [values[n] if n else None for n in node['inputs']], node['attrs'], self.tensors)
            if not isinstance(result, tuple):
                result = (result,)
            values.update(zip(node['outputs'], result))
        return [values[n] for n in outputs]

    def run(self, feeds):
        values = dict(self.constants)
        values.update(feeds)
        uses = self.uses.copy()
        for node in self.graph['nodes']:
            try:
                result = operator(node['op'], [values[n] if n else None for n in node['inputs']], node['attrs'], self.tensors)
            except Exception as exc:
                raise RuntimeError(f"{node['name']} ({node['op']}): {exc}") from exc
            if not isinstance(result, tuple): result = (result,)
            for name, value in zip(node['outputs'], result):
                if name: values[name] = value
            for name in node['inputs']:
                uses[name] -= 1
                if uses[name] == 0 and name in values: del values[name]
        return [values[n] for n in self.graph['outputs']]


def layer_norm(x, w, b):
    delta = x - x.mean(axis=-1, keepdims=True)
    return delta / np.sqrt((delta * delta).mean(axis=-1, keepdims=True) + np.float32(1e-5)) * w + b


def project(hidden, mask, count, p):
    x = hidden[mask.astype(bool)]
    if len(x) != count:
        x = np.stack([x[math.floor(i * len(x) / count):math.ceil((i + 1) * len(x) / count)].mean(axis=0) for i in range(count)])
    return layer_norm(x, p['norm_weight'], p['norm_bias']) @ p['linear_weight'].T + p['linear_bias']


def load_audio(path, rate):
    with wave.open(str(path), 'rb') as f:
        if f.getsampwidth() != 2: raise ValueError('Expected PCM16 WAV')
        audio = np.frombuffer(f.readframes(f.getnframes()), '<i2').astype(np.float32) / 32768
        audio = audio.reshape(-1, f.getnchannels()).mean(axis=1)
        source_rate = f.getframerate()
    if source_rate != rate:
        duration = len(audio) / source_rate
        audio = np.interp(np.linspace(0, duration, max(1, round(duration * rate)), endpoint=False), np.linspace(0, duration, len(audio), endpoint=False), audio).astype(np.float32)
    return audio


def prepare(vqf, path):
    from tokenizers import Tokenizer
    from transformers import WhisperFeatureExtractor
    meta = vqf.meta['bundle']
    tokenizer = Tokenizer.from_str(json.dumps(vqf.meta['assets']['tokenizer']))
    extractor = WhisperFeatureExtractor(**vqf.meta['assets']['feature'])
    audio = load_audio(path, meta['sampling_rate'])[:meta['sampling_rate'] * meta['max_audio_seconds']]
    if not len(audio): raise ValueError('Empty audio')
    features = extractor([audio], sampling_rate=meta['sampling_rate'], return_tensors='np', return_attention_mask=False, padding='longest', max_length=meta['sampling_rate'] * meta['max_audio_seconds'])['input_features'].astype(np.float32)
    length = min(max(1, math.ceil(len(audio) / extractor.hop_length)), features.shape[-1])
    features = np.pad(features[:, :, :3000], ((0, 0), (0, 0), (0, max(0, 3000 - features.shape[-1]))))
    count = max(((len(audio) // extractor.hop_length + 1) // 2) // meta['prompt_audio']['merge_factor'], 1)
    t = meta['tokens']
    prompt = t['user_token'] + t['bos_audio_token'] + t['audio_token'] * count + t['eos_audio_token'] + 'Please transcribe this audio.' + t['assistant_token'] + (meta.get('response_prefix') or '')
    ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    return features, length, count, ids, tokenizer


def normalize_text(text):
    for marker in ('<|user|>', '<|assistant|>', '<|im_end|>'):
        text = text.split(marker, 1)[0]
    for marker in ('<|text|>', '<asr_text>'):
        if marker in text: text = text.split(marker, 1)[1]
    text = re.sub(r'^\s*language\s+[A-Za-z]+\s+', '', text)
    text = re.sub(r'<\|[^>]+\|>', '', text).strip()
    text = re.sub(r'\s+', ' ', text).strip()
    return re.sub(r'^[\s,.;:!?-]+', '', text).strip()


def error(a, b):
    d = a.astype(np.float64) - b.astype(np.float64)
    return dict(max_abs=float(np.max(np.abs(d))), rms=float(np.sqrt(np.mean(d * d))))


def generation(prefill, decode, embeds, embedding, tokens, max_tokens, observer=None):
    length = len(embeds)
    if length >= 512: raise ValueError('Prompt exceeds cache capacity')
    outputs = prefill.run({'inputs_embeds': embeds[None], 'cache_position': np.arange(length, dtype=np.int64)})
    caches = [np.zeros((1, 8, 512, 64), np.float32) for _ in range(16)]
    for cache, delta in zip(caches, outputs[1:]):
        cache[:, :, :length] = delta
    generated = []
    logits_history = []
    stop = None
    for step in range(min(max_tokens, 513 - length)):
        logits = outputs[0].reshape(-1, outputs[0].shape[-1])[-1].copy()
        logits_history.append(logits.copy())
        if observer: observer(step, outputs)
        cutoff = tokens.get('asr_block_token_id_from', -1)
        if cutoff >= 0: logits[cutoff:] = -np.inf
        logits[tokens.get('extra_block_token_ids', [])] = -np.inf
        token = int(np.argmax(logits))
        if token in tokens['eos_token_ids'] or token == tokens['pad_token_id']:
            stop = token
            break
        generated.append(token)
        if step + 1 == max_tokens or length >= 512: break
        attention_mask = (np.arange(512) <= length).astype(np.int64)[None]
        feeds = {'inputs_embeds': embedding[[token]][None], 'cache_position': np.asarray([length], np.int64), 'attention_mask': attention_mask}
        for i in range(8):
            feeds[f'cache_key_{i}'] = caches[2 * i]
            feeds[f'cache_value_{i}'] = caches[2 * i + 1]
        outputs = decode.run(feeds)
        for cache, delta in zip(caches, outputs[1:]):
            cache[:, :, length:length + 1] = delta
        length += 1
    return generated, stop, logits_history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vqf', type=Path, required=True)
    parser.add_argument('--audio', type=Path, required=True)
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--verify-bundle', type=Path)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--skip-checksum', action='store_true', help='Explicitly skip costly integrity scan')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if args.max_new_tokens < 1: parser.error('--max-new-tokens must be positive')
    v = VQF(args.vqf, not args.skip_checksum)
    features, length, count, ids, tokenizer = prepare(v, args.audio)
    audio_graph = Graph(v, 'audio_hidden')
    feeds = {'audios': features, 'audio_feature_lengths': np.asarray([length], np.int64)}
    print(f'NumPy audio forward: feature length={length}, prompt audio tokens={count}', flush=True)
    hidden, mask = audio_graph.run(feeds)
    projector = {k: v.tensors[n] for k, n in v.meta['projector'].items()}
    projected = project(hidden, mask, count, projector)
    embedding = v.tensors[v.meta['embedding']]
    embeds = embedding[ids].copy()
    positions = np.asarray(ids) == v.meta['bundle']['tokens']['audio_token_id']
    if positions.sum() != count: raise ValueError('Prompt/audio token mismatch')
    embeds[positions] = projected
    prefill, decode = Graph(v, 'lm_cache_prefill'), Graph(v, 'lm_cache_decode')
    print('NumPy prefill and autoregressive decode...', flush=True)
    observed = []
    def observe(step, outputs):
        if step < 2: observed.append([a.copy() for a in outputs])
    generated, stop, history = generation(prefill, decode, embeds, embedding, v.meta['bundle']['tokens'], args.max_new_tokens, observe if args.verify_bundle else None)
    raw = tokenizer.decode(generated, skip_special_tokens=False)
    report = dict(audio=str(args.audio), feature_length=length, audio_tokens=count, tokens=generated, stop_token=stop, raw=raw, text=normalize_text(raw), backend='independent_numpy', format='experimental_vqf_v6_C_unsupported')
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if args.verify_bundle:
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = args.threads
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        class Reference:
            def __init__(self, name):
                self.session = ort.InferenceSession(str(args.verify_bundle / v.meta['bundle']['graphs'][name]['path']), sess_options=options, providers=['CPUExecutionProvider'])
            def run(self, inputs): return self.session.run(None, inputs)
        reference_audio = Reference('audio_hidden')
        rh, rm = reference_audio.run(feeds)
        rp = project(rh, rm, count, projector)
        report['audio_hidden_error'] = error(hidden, rh)
        report['audio_valid_hidden_error'] = error(hidden[mask.astype(bool)], rh[rm.astype(bool)])
        report['mask_equal'] = bool(np.array_equal(mask, rm))
        report['projector_error'] = error(projected, rp)
        reference_embeds = embeds.copy()
        reference_embeds[positions] = rp
        ref_prefill, ref_decode = Reference('lm_cache_prefill'), Reference('lm_cache_decode')
        comparisons = []
        def compare(step, outputs):
            if step < len(observed):
                comparisons.append(dict(step=step, logits=error(observed[step][0], outputs[0]), kv_max_abs=max(error(a, b)['max_abs'] for a, b in zip(observed[step][1:], outputs[1:]))))
        rg, rs, r_history = generation(ref_prefill, ref_decode, reference_embeds, embedding, v.meta['bundle']['tokens'], args.max_new_tokens, compare)
        report.update(reference_tokens=rg, reference_stop_token=rs, reference_raw=tokenizer.decode(rg, skip_special_tokens=False), tokens_equal=generated == rg and stop == rs, prefill_decode_errors=comparisons, generation_logits_max_abs=max(error(a, b)['max_abs'] for a, b in zip(history, r_history)))
        # Isolate LM numeric error using identical NumPy audio embeddings.
        identical = ref_prefill.run({'inputs_embeds': embeds[None], 'cache_position': np.arange(len(embeds), dtype=np.int64)})
        report['prefill_identical_input_error'] = error(observed[0][0], identical[0])
        report['text_equal'] = normalize_text(raw) == normalize_text(report['reference_raw'])
        numeric_errors = [report['audio_hidden_error']['max_abs'], report['projector_error']['max_abs'], report['generation_logits_max_abs'], report['prefill_identical_input_error']['max_abs']]
        numeric_errors.extend(item['kv_max_abs'] for item in comparisons)
        report['numeric_tolerance'] = 1e-3
        report['numeric_pass'] = all(np.isfinite(value) and value <= report['numeric_tolerance'] for value in numeric_errors)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        if not report['mask_equal'] or not report['tokens_equal'] or not report['text_equal'] or not report['numeric_pass']:
            if args.report: args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            raise AssertionError('ORT mask/token alignment failed')
    if args.report: args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
