"""离线导出独立 C WAV 运行时资产及现有模型清单。"""
import argparse
import json
from pathlib import Path
import struct

import hashlib
import math
import shutil


def copy_assets(source, out, vqf, audio, reference):
    source, out, reference = source.resolve(), out.resolve(), reference.resolve()
    for root in (source, reference):
        if out.is_relative_to(root) or root.is_relative_to(out):
            raise ValueError('Output must be separate from assets and reference')
    manifest = load_manifest(source, vqf, audio, reference)
    out.mkdir(parents=True, exist_ok=False)
    for name in PACKAGE_FILES:
        shutil.copyfile(checked_file(source, name), out / name)
    frozen_reference = out / 'reference'
    frozen_reference.mkdir()
    for name in REFERENCE_FILES:
        shutil.copyfile(checked_file(reference, name), frozen_reference / name)
    (out / 'assets_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    load_manifest(out, vqf, audio, frozen_reference)
    return manifest, frozen_reference


FEATURE_CASES = ('real', 'silence', 'short201', 'hop319', 'hop321', 'stereo',
                 'rate22050', 'ties_even', 'ties_down', 'over30', 'extra_chunk')
REFERENCE_FILES = ('alignment.json', 'mel.bin', 'input.bin', 'output.bin', 'hidden_reference.npz')
PACKAGE_FILES = ('asr_lm.map', 'asr_wav.assets', 'decoder_reference.json') + tuple(
    name + '.python.features' for name in FEATURE_CASES)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def checked_file(root, name):
    path = root / name
    if Path(name).is_absolute() or '..' in Path(name).parts or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Asset path escapes directory: ' + name)
    if not path.is_file():
        raise ValueError('Missing asset: ' + name)
    return path


def reference_hashes(reference):
    report = json.loads(checked_file(reference, 'alignment.json').read_text(encoding='utf-8'))
    if (report.get('numeric_pass') is not True or report.get('tokens_equal') is not True
            or report.get('hidden_regression_pass') is not True
            or report.get('numeric_tolerance') != 1e-3
            or not report.get('expected_tokens') or not isinstance(report.get('reference_text'), str)):
        raise ValueError('Unqualified external reference')
    errors = [report['hidden_error']['max_abs'], report['projector_error']['max_abs'],
              report['hidden_regression_max_abs']]
    errors += [value for step in report['steps'] for value in (step['logits']['max_abs'], step['kv_max_abs'])]
    if len(report['steps']) != len(report['expected_tokens']) or not all(
            math.isfinite(value) and 0 <= value <= 1e-3 for value in errors):
        raise ValueError('Invalid reference errors')
    if len(report.get('mask_boundaries', {})) != 21 or not all(report['mask_boundaries'].values()):
        raise ValueError('Unqualified mask reference')
    if len(report.get('rejection_tests', {})) != 9 or not all(report['rejection_tests'].values()):
        raise ValueError('Unqualified rejection reference')
    return {name: sha256(checked_file(reference, name)) for name in REFERENCE_FILES}


def write_manifest(out, vqf, audio, reference, feature_audio):
    manifest = dict(schema=1, source_vqf_sha256=sha256(vqf), audio_sha256=sha256(audio),
                    reference=reference_hashes(reference), feature_audio=feature_audio,
                    files={name: sha256(checked_file(out, name)) for name in PACKAGE_FILES})
    frozen = out / 'reference'
    frozen.mkdir(exist_ok=False)
    for name in REFERENCE_FILES:
        shutil.copyfile(checked_file(reference, name), frozen / name)
    if reference_hashes(frozen) != manifest['reference']:
        raise ValueError('Copied reference SHA256 mismatch')
    (out / 'assets_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    load_manifest(out, vqf, audio, frozen)


def load_manifest(source, vqf, audio, reference):
    manifest = json.loads(checked_file(source, 'assets_manifest.json').read_text(encoding='utf-8'))
    if type(manifest.get('schema')) is not int or manifest['schema'] != 1 or set(manifest.get('files', {})) != set(PACKAGE_FILES):
        raise ValueError('Unsupported or incomplete asset manifest')
    if set(manifest.get('feature_audio', {})) != set(FEATURE_CASES):
        raise ValueError('Incomplete feature audio bindings')
    for name, digest in manifest['files'].items():
        if sha256(checked_file(source, name)) != digest:
            raise ValueError('Asset SHA256 mismatch: ' + name)
    if manifest.get('source_vqf_sha256') != sha256(vqf):
        raise ValueError('Source VQF SHA256 mismatch')
    if manifest.get('audio_sha256') != sha256(audio):
        raise ValueError('Input audio SHA256 mismatch')
    if manifest.get('reference') != reference_hashes(reference):
        raise ValueError('Reference SHA256 mismatch')
    return manifest


def export_assets(v, out):
    import numpy as np
    from tokenizers import Tokenizer
    from transformers import WhisperFeatureExtractor
    import torch
    from onnx_to_vqf import fnv_update

    meta = v.meta['bundle']
    config = v.meta['assets']['feature']
    if (meta['sampling_rate'], meta['max_audio_seconds'], meta['prompt_audio']['merge_factor']) != (16000, 30, 4):
        raise ValueError('Unsupported audio configuration')
    if any(config.get(k) != value for k, value in dict(n_fft=400, hop_length=160, feature_size=128, sampling_rate=16000, dither=0.0, padding_value=0.0).items()):
        raise ValueError('Unsupported feature configuration')
    spec = v.meta['assets']['tokenizer']
    if spec['decoder']['type'] != 'ByteLevel':
        raise ValueError('Expected ByteLevel decoder')
    tok = Tokenizer.from_str(json.dumps(spec))
    t = meta['tokens']
    if (t['audio_token_id'], t['pad_token_id'], t['eos_token_ids'], t['asr_block_token_id_from'], sorted(t['extra_block_token_ids'])) != (151646, 151643, [151645], -1, [151646, 151647, 151648, 151649, 151650]):
        raise ValueError('Unsupported generation rules')
    before = t['user_token'] + t['bos_audio_token']
    after = t['eos_audio_token'] + 'Please transcribe this audio.' + t['assistant_token'] + (meta.get('response_prefix') or '')
    prefix = tok.encode(before, add_special_tokens=False).ids
    suffix = tok.encode(after, add_special_tokens=False).ids
    for count in range(1, 376):
        if tok.encode(before + t['audio_token'] * count + after, add_special_tokens=False).ids != prefix + [151646] * count + suffix:
            raise ValueError('Prompt token boundaries differ')
    # GPT ByteLevel 的可见字符反映射为真实字节，特殊 token 按原文保留以匹配 skip_special_tokens=False。
    visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    mapping = {chr(b): b for b in visible}
    extra = 256
    for b in range(256):
        if b not in visible:
            mapping[chr(extra)] = b
            extra += 1
    added = {a['id']: a['content'] for a in spec['added_tokens']}
    offsets = [0]
    data = bytearray()
    for i in range(151936):
        text = tok.id_to_token(i)
        raw = added[i].encode('utf-8') if i in added else bytes(mapping[c] for c in text) if text is not None else b''
        data.extend(raw)
        offsets.append(len(data))
    extractor = WhisperFeatureExtractor(**config)
    payload = struct.pack('<II', len(prefix), len(suffix))
    payload += np.asarray(prefix + suffix, '<u4').tobytes()
    payload += torch.hann_window(400).numpy().astype('<f4').tobytes()
    payload += extractor.mel_filters.astype('<f4').tobytes()
    payload += np.asarray(offsets, '<u4').tobytes() + data
    with v.path.open('rb') as f:
        f.seek(-8, 2)
        digest = f.read(8)
    header = b'ASRWAV01' + digest + struct.pack('<QQ', len(payload), fnv_update(14695981039346656037, payload))
    path = out / 'asr_wav.assets'
    path.write_bytes(header + payload)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--vqf', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()
    from asr_vqf_run import VQF
    from asr_c_gate import export, export_audio
    args.out.mkdir(parents=True, exist_ok=True)
    v = VQF(args.vqf, verify=True)
    export(v, args.out)
    export_audio(v, args.out)
    print(export_assets(v, args.out))


if __name__ == '__main__':
    main()
