"""Offline FunASR golden capture and native Paraformer numerical comparison."""
import argparse
import json
import os
from pathlib import Path
import struct
import subprocess
import wave

import numpy as np
import torch
import funasr
from funasr import AutoModel


BOUNDARIES = (0, 1, 399, 400, 959, 960, 961, 9599, 9600, 9601)


def compare(actual, ref):
    if actual.shape != ref.shape:
        return {'passed': False, 'shape_mismatch': [list(ref.shape), list(actual.shape)]}
    finite = bool(np.isfinite(actual).all() and np.isfinite(ref).all())
    diff = np.abs(actual.astype(np.float64) - ref.astype(np.float64))
    return {'passed': finite and bool(np.allclose(actual, ref, atol=0.001, rtol=0.0001)),
            'max_abs': float(diff.max(initial=0)),
            'rmse': float(np.sqrt(np.mean(diff ** 2))) if diff.size else 0.0}


def read_exact(f, size):
    data = f.read(size)
    if len(data) != size:
        raise ValueError(f'truncated {f.name}: expected {size}, got {len(data)} bytes')
    return data


def read_features(path):
    records = []
    with path.open('rb') as f:
        assert read_exact(f, 8) == b'PFCH0001'
        while True:
            header = f.read(16)
            if not header:
                break
            nf, final, tail, reset = struct.unpack('<4I', header)
            x = np.frombuffer(read_exact(f, nf * 560 * 4), '<f4').reshape(nf, 560)
            records.append({'features': x, 'final': bool(final), 'tail': bool(tail), 'reset': bool(reset)})
    return records


def write_features(path, records):
    with path.open('wb') as f:
        f.write(b'PFCH0001')
        for r in records:
            x = r['features']
            # Official tail input is already positional-encoded overlap. Native tail
            # consumes its own overlap, so the PFCH tail payload must be empty.
            if r['tail']:
                x = np.empty((0, 560), np.float32)
            f.write(struct.pack('<4I', len(x), r['final'], r['tail'], r['reset']))
            f.write(x.astype('<f4').tobytes())


def read_outputs(path, count):
    outputs = []
    with path.open('rb') as f:
        assert read_exact(f, 8) == b'PFOUT001'
        for _ in range(count):
            ne, nt = struct.unpack('<2I', read_exact(f, 8))
            item = {}
            for name, shape in [('encoder', (ne, 512)), ('alpha', (ne,)),
                                ('predictor', (nt, 512)), ('decoder', (nt, 8404))]:
                item[name] = np.frombuffer(read_exact(f, int(np.prod(shape)) * 4), '<f4').reshape(shape)
            outputs.append(item)
        assert not f.read(1), 'unexpected extra native chunks'
    return outputs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--wav', required=True)
    p.add_argument('--vqf', required=True)
    p.add_argument('--exe', required=True)
    p.add_argument('--out-dir', required=True, type=Path)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    auto = AutoModel(model=args.model, device='cpu', disable_update=True,
                     frontend_conf={'dither': 0.0}, disable_pbar=True)
    model = auto.model.eval()
    auto.kwargs['frontend'].dither = 0.0
    records = []
    original = model.generate_chunk

    def capture(speech, speech_lengths=None, **kwargs):
        r = {'features': speech.detach().cpu().numpy()[0].copy(),
             'final': bool(kwargs.get('is_final', False)),
             'tail': bool(kwargs['cache']['encoder']['tail_chunk']), 'reset': not records}
        records.append(r)
        result = original(speech, speech_lengths, **kwargs)
        assert 'encoder' in r and 'predictor' in r and 'alpha' in r
        if r['predictor'].shape[0] == 0:
            assert result == [] and 'decoder' not in r, 'zero-token path changed'
            r['decoder'] = np.empty((0, 8404), np.float32)
        else:
            assert 'decoder' in r, 'decoder capture missing for nonempty predictor'
        return result

    model.generate_chunk = capture

    def alpha_hook(module, inputs, output):
        predictor = model.predictor
        alpha = torch.relu(torch.sigmoid(output) * predictor.smooth_factor - predictor.noise_threshold)
        alpha = alpha.detach().cpu().numpy()[0, :, 0].copy()
        if not records[-1]['final']:
            alpha[10:] = 0
        records[-1]['alpha'] = alpha

    model.predictor.cif_output.register_forward_hook(alpha_hook)
    for name, module in [('encoder', model.encoder), ('predictor', model.predictor),
                         ('decoder', model.decoder)]:
        method = module.forward_chunk

        def wrapped(*a, _name=name, _method=method, **kw):
            value = _method(*a, **kw)
            tensor = value if _name == 'decoder' else value[0]
            if _name == 'predictor':
                count = int(value[1][0])
                # FunASR returns hidden states (not embeddings) on zero tokens.
                tensor = tensor[:, :count]
            assert isinstance(tensor, torch.Tensor), f'unexpected {_name} result'
            records[-1][_name] = tensor.detach().cpu().numpy()[0].copy()
            return value
        module.forward_chunk = wrapped

    with wave.open(args.wav, 'rb') as f:
        assert (f.getnchannels(), f.getsampwidth(), f.getframerate()) == (1, 2, 16000)
        pcm = np.frombuffer(f.readframes(f.getnframes()), '<i2')
    assert len(pcm) >= 48000, 'fixture must exercise multiple decoder-history chunks'
    cases = [('example', pcm)]
    # Each boundary is tested independently; tail cases use actual speech history.
    cases += [(f'short_{n}', pcm[:n]) for n in BOUNDARIES]
    cases += [(f'tail_{n}', pcm[:38400 + n]) for n in (0, 1, 959, 960, 961)]
    report = {'funasr_version': funasr.__version__, 'atol': 0.001, 'rtol': 0.0001,
              'boundaries': list(BOUNDARIES), 'cases': [], 'passed': True}
    env = dict(os.environ)
    env.setdefault('OMP_NUM_THREADS', '4')
    for name, samples in cases:
        work = args.out_dir / name
        work.mkdir(exist_ok=True)
        wav = work / 'input.wav'
        with wave.open(str(wav), 'wb') as f:
            f.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
            f.writeframes(samples.astype('<i2').tobytes())
        records.clear()
        # Array input explicitly finalizes even empty audio, avoiding file-loader
        # empty-file special cases while preserving exact PCM normalization.
        result = auto.generate(input=samples.astype(np.float32) / 32768.0,
                               is_final=True, cache={}, chunk_size=[0, 10, 5],
                               encoder_chunk_look_back=4, decoder_chunk_look_back=1)
        assert isinstance(result, list), 'unexpected official result type'
        assert all(isinstance(r, dict) and isinstance(r.get('text'), str) for r in result)
        assert len(records) == len(samples) // 9600 + 1
        official_text = ''.join(r['text'] for r in result).replace(' ', '')
        if not result:
            assert all(len(r['predictor']) == 0 for r in records), 'empty result with predicted tokens'
        item = {'name': name, 'samples': len(samples), 'official': result,
                'chunks': [], 'frontend': [], 'passed': True}
        native_path = work / 'native_features.pfch'
        run = subprocess.run([args.exe, args.vqf, str(wav)], check=True,
                             env=dict(env, KESTREL_PF_FEATURE_DUMP=str(native_path)),
                             capture_output=True, encoding='utf-8')
        item['native_wav_text'] = run.stdout.strip()
        item['text_equal'] = item['native_wav_text'] == official_text
        item['passed'] &= item['text_equal']
        native_features = read_features(native_path)
        assert len(native_features) == len(records), 'frontend chunk count mismatch'
        for i, (actual, ref) in enumerate(zip(native_features, records)):
            meta = all(actual[k] == ref[k] for k in ('final', 'tail', 'reset'))
            check = {'chunk': i, 'metadata_equal': meta, 'final': ref['final'], 'tail': ref['tail']}
            if ref['tail']:
                check['passed'] = meta and actual['features'].shape == (0, 560)
            else:
                check.update(compare(actual['features'], ref['features']))
                check['passed'] &= meta
            item['frontend'].append(check)
            item['passed'] &= check['passed']
        # Replay the official features twice in one process to verify reset, then
        # native frontend features to check the full numerical WAV path as well.
        stream = records + records + native_features
        inp, out = work / 'features.pfch', work / 'native.pfout'
        write_features(inp, stream)
        subprocess.run([args.exe, args.vqf, str(inp), str(out)], check=True, env=env,
                       capture_output=True)
        outputs = read_outputs(out, len(stream))
        for j, actual in enumerate(outputs):
            i = j % len(records)
            repeat = j // len(records)
            ref = records[i]
            check = {'path': ('golden', 'reset', 'wav')[repeat], 'chunk': i,
                     'tokens': len(actual['predictor'])}
            for key in ('encoder', 'alpha', 'predictor', 'decoder'):
                check[key] = compare(actual[key], ref[key])
                item['passed'] &= check[key]['passed']
            check['ids_equal'] = bool(np.array_equal(actual['decoder'].argmax(-1), ref['decoder'].argmax(-1)))
            item['passed'] &= check['ids_equal']
            if repeat == 1:
                check['reset_exact'] = all(np.array_equal(actual[k], outputs[i][k]) for k in actual)
                item['passed'] &= check['reset_exact']
            item['chunks'].append(check)
        report['cases'].append(item)
        report['passed'] &= item['passed']
        (args.out_dir / 'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({'case': name, 'samples': len(samples), 'chunks': len(records),
                          'text': item['native_wav_text'], 'passed': item['passed']}, ensure_ascii=False), flush=True)
    print('PASS', report['passed'], 'report:', args.out_dir / 'comparison.json', flush=True)
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
