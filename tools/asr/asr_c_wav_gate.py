"""编译、导出并验证独立 C WAV 转写；全部运行产物保存在指定 TEMP 目录。"""
import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import struct
import subprocess
import wave

import numpy as np
from asr_c_assets import export_assets, copy_assets, write_manifest, sha256
from asr_c_gate import parse_cflags, export, export_audio


def error(a, b):
    d = a.astype(np.float64) - b.astype(np.float64)
    return dict(max_abs=float(np.max(np.abs(d))), rms=float(np.sqrt(np.mean(d * d))))


def regression_baseline(directory, reference, out, vqf, audio):
    directory, out = directory.resolve(), out.resolve()
    if directory.is_relative_to(out) or out.is_relative_to(directory):
        raise ValueError('C baseline and output must be separate directories')
    proof = json.loads((directory / 'c_regression_baseline.json').read_text(encoding='utf-8'))
    if (proof['platform'] != [platform.system(), platform.machine()]
            or proof['vqf_sha256'] != sha256(vqf) or proof['audio_sha256'] != sha256(audio)):
        raise ValueError('C baseline platform/model/audio mismatch')
    names = {'alignment.json', 'input.bin', 'mel.bin', 'hidden_reference.npz', 'asr_lm.map', 'output.bin'}
    if set(proof['files']) != names or any(sha256(directory / n) != proof['files'][n] for n in names):
        raise ValueError('C baseline hash mismatch')
    for name in ('input.bin', 'mel.bin', 'hidden_reference.npz'):
        if sha256(directory / name) != sha256(reference / name):
            raise ValueError('C baseline frozen input mismatch: ' + name)
    if sha256(directory / 'asr_lm.map') != sha256(out / 'asr_lm.map'):
        raise ValueError('C baseline model map mismatch')
    report = json.loads((directory / 'alignment.json').read_text(encoding='utf-8'))
    if not all(report.get(k) is True for k in ('numeric_pass', 'tokens_equal', 'hidden_regression_pass')):
        raise ValueError('C baseline hidden gate did not pass')
    if not report.get('rejection_tests') or not all(report['rejection_tests'].values()):
        raise ValueError('C baseline rejection tests did not pass')
    target = out / 'mel_regression.bin'
    if target.exists() and target.samefile(directory / 'output.bin'):
        raise ValueError('C baseline self comparison')
    return (directory / 'output.bin').read_bytes(), proof


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--vqf', required=True, type=Path)
    ap.add_argument('--audio', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--reference', required=True, type=Path)
    ap.add_argument('--c-baseline', type=Path, help='successful same-platform hidden gate directory for bytewise Mel regression')
    ap.add_argument('--cc', default='gcc')
    ap.add_argument('--cflags', type=parse_cflags, default='', metavar='FLAGS',
                    help='quoted target/ASR switches, e.g. --cflags="-march=rv64gcv -DVLLM_ASR_W8_RVV"; native execution required')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--assets-dir', type=Path, help='reuse a validated frozen Python reference asset package; no re-export fallback')
    ap.add_argument('--export-manifest', action='store_true', help='freeze Python feature/decoder references and publish a manifest after the full gate passes')
    args = ap.parse_args()
    if args.export_manifest and (args.assets_dir or args.quick):
        ap.error('--export-manifest requires a full default gate')
    if args.export_manifest and args.out.exists():
        ap.error('--export-manifest requires a new output directory')
    if args.out.resolve().is_relative_to(args.reference.resolve()) or args.reference.resolve().is_relative_to(args.out.resolve()):
        ap.error('--out must be separate from the external reference directory')
    manifest = None
    feature_audio = {}
    if args.assets_dir:
        if args.out.resolve().is_relative_to(args.assets_dir.resolve()) or args.assets_dir.resolve().is_relative_to(args.out.resolve()):
            ap.error('--out and --assets-dir must be separate directories')
        manifest, args.reference = copy_assets(args.assets_dir, args.out, args.vqf, args.audio, args.reference)
        args.assets_dir = args.out
        assets = args.out / 'asr_wav.assets'
        model_map = args.out / 'asr_lm.map'
    else:
        from asr_vqf_run import VQF, prepare, normalize_text
        args.out.mkdir(parents=True, exist_ok=True)
        v = VQF(args.vqf, False)
        export(v, args.out)
        export_audio(v, args.out)
        assets = export_assets(v, args.out)
        model_map = args.out / 'asr_lm.map'
    args.out.mkdir(parents=True, exist_ok=True)
    baseline, baseline_proof = None, None
    if args.c_baseline:
        baseline, baseline_proof = regression_baseline(args.c_baseline, args.reference, args.out, args.vqf, args.audio)
    exe = args.out / 'asr.exe'
    root = Path(__file__).resolve().parents[2]
    cmd = [args.cc, '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror', '-pedantic', '-ffp-contract=off', str(root / 'src/model/vllm_asr.c'), str(root / 'src/core/vllm_tp.c'), '-I' + str(root / 'include/core'), '-I' + str(root / 'include/common'), '-D_GNU_SOURCE', '-pthread', '-o', str(exe), '-lm']
    cmd[1:1] = args.cflags
    compiled = subprocess.run(cmd, capture_output=True, text=True)
    (args.out / 'compile.log').write_text(subprocess.check_output([args.cc, '--version'], text=True) + '\n' + repr(cmd) + '\n' + compiled.stdout + compiled.stderr, encoding='utf-8')
    compiled.check_returncode()
    imports = subprocess.check_output(['objdump', '-p', str(exe)], text=True)
    (args.out / 'binary_imports.log').write_text(imports, encoding='utf-8')
    dependencies = [line.strip() for line in imports.splitlines() if 'DLL Name:' in line]
    assert not any(name.lower() in imports.lower() for name in ('python', 'onnxruntime', 'tokenizers', 'torch'))
    report = dict(compile=cmd, dependencies=dependencies, binary_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(), binary_mtime_ns=exe.stat().st_mtime_ns, features={}, rejected={}, decoder={})
    def save():
        (args.out / 'wav_alignment.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    def check_audio(label, path):
        output = args.out / (label + '.features')
        result = subprocess.run([str(exe), '--features', str(assets), str(path), str(output)], capture_output=True)
        result.check_returncode()
        raw = output.read_bytes()
        length, count = struct.unpack_from('<II', raw)
        feature_audio[label] = sha256(path)
        tokenizer = None
        if manifest is not None:
            if feature_audio[label] != manifest['feature_audio'][label]:
                raise ValueError('Feature audio SHA256 mismatch: ' + label)
            frozen = (args.assets_dir / (label + '.python.features')).read_bytes()
            expected_length, expected_count = struct.unpack_from('<II', frozen)
            features = np.frombuffer(frozen, '<f4', offset=8).reshape(1, 128, 3000)
        else:
            features, expected_length, expected_count, _, tokenizer = prepare(v, path)
            if args.export_manifest:
                (args.out / (label + '.python.features')).write_bytes(
                    struct.pack('<II', expected_length, expected_count) + np.asarray(features, '<f4').tobytes())
        value = error(np.frombuffer(raw, '<f4', offset=8).reshape(1, 128, 3000), features)
        report['features'][label] = dict(**value, length=length, count=count)
        save()
        assert (length, count) == (expected_length, expected_count), (label, length, expected_length, count, expected_count)
        assert value['max_abs'] <= 1e-3, (label, value)
        return tokenizer
    tokenizer = check_audio('real', args.audio)
    rng = np.random.default_rng(17)
    cases = [('silence', 16000, 1, 320), ('short201', 16000, 1, 201), ('hop319', 16000, 1, 319), ('hop321', 16000, 1, 321), ('stereo', 16000, 2, 1801), ('rate22050', 22050, 1, 2303), ('ties_even', 32000, 1, 643), ('ties_down', 32000, 1, 641), ('over30', 16000, 1, 480161)]
    for label, rate, channels, n in cases:
        path = args.out / (label + '.wav')
        data = np.zeros((n, channels), '<i2') if label in ('silence', 'over30') else rng.integers(-16000, 16000, (n, channels), dtype=np.int16)
        with wave.open(str(path), 'wb') as f:
            f.setnchannels(channels)
            f.setsampwidth(2)
            f.setframerate(rate)
            f.writeframes(data.tobytes())
        check_audio(label, path)
    original = (args.out / 'stereo.wav').read_bytes()
    extra = original[:12] + b'JUNK' + struct.pack('<I', 3) + b'abc\0' + original[12:]
    extra = extra[:4] + struct.pack('<I', len(extra)-8) + extra[8:]
    path = args.out / 'extra_chunk.wav'
    path.write_bytes(extra)
    check_audio('extra_chunk', path)
    malformed = {'truncated': original[:-1], 'chunk_overflow': original[:40] + struct.pack('<I', 0xffffffff) + original[44:], 'empty': original[:4] + struct.pack('<I', 36) + original[8:40] + struct.pack('<I', 0), 'nonpcm': original[:20] + struct.pack('<H', 3) + original[22:], 'short200': original[:4] + struct.pack('<I', 36+800) + original[8:40] + struct.pack('<I', 800) + original[44:844]}
    for label, raw in malformed.items():
        path = args.out / (label + '.wav')
        path.write_bytes(raw)
        result = subprocess.run([str(exe), '--features', str(assets), str(path), str(args.out / 'bad.features')], capture_output=True)
        report['rejected'][label] = result.returncode != 0
        assert result.returncode != 0, label
    for label, raw in [('asset_truncated', assets.read_bytes()[:-1]), ('asset_corrupt', assets.read_bytes()[:-1] + bytes([assets.read_bytes()[-1] ^ 1]))]:
        path = args.out / (label + '.assets')
        path.write_bytes(raw)
        result = subprocess.run([str(exe), '--decode', str(path), '1'], capture_output=True)
        report['rejected'][label] = result.returncode != 0
        assert result.returncode != 0
    texts = ['language English ', 'language English', 'language English  <|audio|>', 'a  <|audio|>  b', '欢迎大家来体验达摩院推出的语音识别模型。', '  language Chinese 你好 世界 ', '<|text|>中文<|user|>ignored', '<asr_text> ,.;:!?-你好\u3000世界', 'a<|audio|>b', '中文😀 café\n\t test', '\u0085language English\u3000 hello']
    random_groups = [[i] for i in range(256)] + [list(map(int, rng.integers(0, 256, 20))) for _ in range(20)]
    if manifest is not None:
        decoder = json.loads((args.assets_dir / 'decoder_reference.json').read_text(encoding='utf-8'))
        groups, expected_texts = decoder['groups'], decoder['expected']
        if decoder['texts'] != texts or len(groups) != len(texts) + 276 or groups[len(texts):] != random_groups or len(expected_texts) != len(groups):
            raise ValueError('Incomplete or incompatible decoder reference')
    else:
        groups = [tokenizer.encode(t, add_special_tokens=False).ids for t in texts] + random_groups
        expected_texts = [normalize_text(tokenizer.decode(ids, skip_special_tokens=False)) for ids in groups]
        if args.export_manifest:
            (args.out / 'decoder_reference.json').write_text(json.dumps(
                dict(texts=texts, groups=groups, expected=expected_texts), ensure_ascii=False), encoding='utf-8')
    for i, ids in enumerate(groups):
        actual = subprocess.check_output([str(exe), '--decode', str(assets), *map(str, ids)]).decode('utf-8').rstrip('\r\n')
        expected = expected_texts[i]
        assert actual == expected, (i, ids, repr(actual), repr(expected))
    report['decoder']['cases'] = len(groups)
    save()
    print('Feature/decoder tests passed', flush=True)
    if args.quick:
        return
    command = [str(exe), '--wav', str(args.vqf), str(model_map), str(assets), str(args.audio), str(args.out / 'transcript.txt'), str(args.out / 'wav_trace.bin')]
    with (args.out / 'wav_run.log').open('wb') as log:
        subprocess.run(command, stdout=log, stderr=log, check=True)
    report['run'] = command
    reference = json.loads((args.reference / 'alignment.json').read_text(encoding='utf-8'))
    mel = (args.reference / 'mel.bin').read_bytes()
    length, count = struct.unpack_from('<II', mel, 8)
    raw = (args.out / 'wav_trace.bin').read_bytes()
    old = (args.reference / 'output.bin').read_bytes()
    valid = int(np.frombuffer(raw, '<u4', 390).sum())
    assert raw[:1560] == old[:1560]
    offset = 1560
    def compare(n):
        nonlocal offset
        a = np.frombuffer(raw, '<f4', n, offset)
        b = np.frombuffer(old, '<f4', n, offset)
        offset += n*4
        return error(a, b)['max_abs']
    hidden = compare(valid*1024)
    projected = compare(count*512)
    steps, tokens = [], []
    for expected in reference['expected_tokens']:
        logits = compare(151936)
        kv = compare(16*512)
        token = struct.unpack_from('<I', raw, offset)[0]
        assert token == expected
        tokens.append(token)
        offset += 4
        steps.append(dict(logits_vs_prior_c=logits, kv_vs_prior_c=kv))
    assert offset == len(raw) == len(old)
    # 三角不等式提供新 C WAV 与原 Python 各层误差的保守上界。
    bounds = [hidden + reference['hidden_error']['max_abs'], projected + reference['projector_error']['max_abs']]
    for step, ref in zip(steps, reference['steps']):
        bounds += [step['logits_vs_prior_c'] + ref['logits']['max_abs'], step['kv_vs_prior_c'] + ref['kv_max_abs']]
    report.update(tokens=tokens, text=(args.out / 'transcript.txt').read_text(encoding='utf-8').strip(), hidden_vs_prior_c=hidden, projector_vs_prior_c=projected, steps=steps, python_error_upper_bound=max(bounds))
    assert all(np.isfinite(bound) and 0 <= bound <= 1e-3 for bound in bounds)
    assert report['text'] == reference['reference_text']
    for mode in ('mel', 'hidden'):
        fixture = args.reference / ('mel.bin' if mode == 'mel' else 'input.bin')
        output = args.out / (mode + '_regression.bin')
        with (args.out / (mode + '_regression.log')).open('wb') as log:
            subprocess.run([str(exe), str(args.vqf), str(model_map), str(fixture), str(output)], stdout=log, stderr=log, check=True)
        if mode == 'mel':
            assert output.read_bytes() == (baseline if baseline is not None else old)
            report['c_regression_baseline'] = baseline_proof
        else:
            r = output.read_bytes()
            off = count*512*4
            rtokens = []
            for _ in tokens:
                off += (151936+16*512)*4
                rtokens.append(struct.unpack_from('<I', r, off)[0])
                off += 4
            assert rtokens == tokens and off == len(r)
        report[mode + '_regression'] = True
    save()
    if args.export_manifest:
        write_manifest(args.out, args.vqf, args.audio, args.reference, feature_audio)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
