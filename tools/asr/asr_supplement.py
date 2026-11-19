"""Prepare/replay frozen WAV boundaries or run serial full gates after profiling."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys

CASES = 'silence short201 hop319 hop321 stereo rate22050 ties_even ties_down over30 extra_chunk'.split()
BAD = 'truncated chunk_overflow empty nonpcm short200'.split()
ERRORS = dict(zip(BAD + ['asset_truncated', 'asset_corrupt'], [
    'truncated/invalid RIFF length', 'truncated RIFF chunk/padding', 'empty WAV',
    'only PCM16 WAV supported', 'WAV needs more than 200 resampled samples',
    'truncated file', 'WAV asset checksum/length mismatch']))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def idle(root):
    if json.loads((root / 'model_status.json').read_text()) != {'status': 'complete'}:
        raise ValueError('Formal runner is not complete; supplementation forbidden')
    records = json.loads((root / 'model_execution.json').read_text())
    expected = {f'profile_{m}_{c}_t{t}_r{r}'
                for m, cs in [('f32', ['A', 'default', 'B', 'D']), ('w8', ['A', 'default', 'B', 'C', 'D'])]
                for c in cs for t in (1, 8) for r in range(1, 6)}
    formal = [r for r in records if r['label'].startswith('profile_')]
    if len(formal) != 90 or {r['label'] for r in formal} != expected or any(r['exit_code'] != 0 for r in records):
        raise ValueError('Require all 90 unique successful formal samples')


def prepare(source, out):
    report = json.loads((source / 'wav_alignment.json').read_text(encoding='utf-8'))
    names = ['asr_wav.assets', 'asset_corrupt.assets', 'asset_truncated.assets', 'wav_alignment.json']
    names += [c + ext for c in CASES for ext in ('.wav', '.features')]
    names += [c + '.wav' for c in BAD]
    for c in CASES:
        value = report['features'][c]['max_abs']
        if not math.isfinite(value) or not 0 <= value <= 1e-3:
            raise ValueError('Unqualified Python reference: ' + c)
    if not all(report['rejected'].values()):
        raise ValueError('Reference rejection tests failed')
    out.mkdir(parents=True, exist_ok=False)
    for name in names:
        shutil.copy2(source / name, out / name)
    save(out / 'manifest.json', {'files': {n: sha(out / n) for n in names},
                               'scope': '10 frozen feature cases and 7 rejection cases; not full dual gates'})


def boundary(exe, fixtures, out):
    manifest = json.loads((fixtures / 'manifest.json').read_text())
    for name, digest in manifest['files'].items():
        if sha(fixtures / name) != digest:
            raise ValueError('Fixture hash mismatch: ' + name)
    reference = json.loads((fixtures / 'wav_alignment.json').read_text(encoding='utf-8'))
    out.mkdir(parents=True, exist_ok=False)
    results = {'binary_sha256': sha(exe), 'scope': manifest['scope'], 'cases': []}
    env = os.environ.copy()
    env['VLLM_THREADS'] = '1'
    env.pop('VLLM_ASR_PROFILE', None)
    for name in CASES + BAD + ['asset_truncated', 'asset_corrupt']:
        feature = out / (name + '.features')
        if name.startswith('asset_'):
            cmd = [str(exe), '--decode', str(fixtures / (name + '.assets')), '1']
        else:
            cmd = [str(exe), '--features', str(fixtures / 'asr_wav.assets'), str(fixtures / (name + '.wav')), str(feature)]
        with (out / (name + '.log')).open('wb') as log:
            run = subprocess.run(cmd, stdout=log, stderr=log, env=env)
        item = {'name': name, 'command': cmd, 'exit_code': run.returncode}
        if name in CASES and run.returncode == 0:
            raw, old = feature.read_bytes(), (fixtures / (name + '.features')).read_bytes()
            valid = len(raw) == len(old) == 8 + 128 * 3000 * 4 and raw[:8] == old[:8]
            delta = float('inf')
            if valid:
                pairs = zip(struct.iter_unpack('<f', raw[8:]), struct.iter_unpack('<f', old[8:]))
                delta = max(abs(a[0] - b[0]) if math.isfinite(a[0]) and math.isfinite(b[0]) else float('inf') for a, b in pairs)
            bound = delta + reference['features'][name]['max_abs']
            item.update(max_abs_vs_frozen=delta if math.isfinite(delta) else None,
                        python_error_upper_bound=bound if math.isfinite(bound) else None,
                        passed=valid and math.isfinite(bound) and bound <= 1e-3)
        else:
            item['expected_error'] = ERRORS.get(name)
            item['passed'] = (name not in CASES and run.returncode == 1
                              and ERRORS[name] in (out / (name + '.log')).read_text(errors='replace'))
        results['cases'].append(item)
        save(out / 'results.json', results)
        if not item['passed']:
            raise ValueError('Boundary failed: ' + name)
    results['passed'] = True
    save(out / 'results.json', results)
    print('17/17 frozen boundary checks passed; not full dual-gate acceptance')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='mode', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--source', required=True, type=Path)
    prep.add_argument('--out', required=True, type=Path)
    replay = sub.add_parser('boundary')
    replay.add_argument('--exe', required=True, type=Path)
    replay.add_argument('--fixtures', required=True, type=Path)
    replay.add_argument('--out', required=True, type=Path)
    guard = replay.add_mutually_exclusive_group(required=True)
    guard.add_argument('--runner-root', type=Path)
    guard.add_argument('--local-check', action='store_true')
    full = sub.add_parser('full')
    full_guard = full.add_mutually_exclusive_group(required=True)
    full_guard.add_argument('--runner-root', type=Path)
    full_guard.add_argument('--local-check', action='store_true')
    full.add_argument('--vqf', required=True, type=Path)
    full.add_argument('--audio', required=True, type=Path)
    full.add_argument('--out', required=True, type=Path)
    full.add_argument('--cflags', default='')
    full.add_argument('--assets-dir', type=Path, help='reuse a qualified complete dual-gate package')
    full.add_argument('--export-manifest', action='store_true')
    args = ap.parse_args()
    if args.mode == 'full' and args.assets_dir and args.export_manifest:
        ap.error('--export-manifest requires default reference generation')
    if args.mode == 'prepare':
        prepare(args.source.resolve(), args.out.resolve())
        return
    if getattr(args, 'local_check', False):
        if sys.platform != 'win32':
            ap.error('--local-check is restricted to the Windows preparation host')
    else:
        idle(args.runner_root.resolve())
    if args.mode == 'boundary':
        boundary(args.exe.resolve(), args.fixtures.resolve(), args.out.resolve())
        return
    from asr_c_gate import parse_cflags
    parse_cflags(args.cflags)
    root = Path(__file__).resolve().parent
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env['VLLM_THREADS'] = '1'
    env.pop('VLLM_ASR_PROFILE', None)
    records = []
    for kind, script in [('hidden', 'asr_c_gate.py'), ('wav', 'asr_c_wav_gate.py')]:
        cmd = [sys.executable, '-B', str(root / script), '--vqf', str(args.vqf.resolve()),
               '--audio', str(args.audio.resolve()), '--out', str(out / kind), '--cflags=' + args.cflags]
        if args.assets_dir:
            cmd += ['--assets-dir', str(args.assets_dir.resolve()),
                    '--reference', str(args.assets_dir.resolve() / 'reference')]
        elif kind == 'wav':
            cmd += ['--reference', str(out / 'hidden')]
            if args.export_manifest:
                cmd += ['--export-manifest']
        if kind == 'wav':
            cmd += ['--c-baseline', str(out / 'hidden')]
        with (out / (kind + '.log')).open('wb') as log:
            run = subprocess.run(cmd, stdout=log, stderr=log, env=env)
        records.append({'command': cmd, 'exit_code': run.returncode})
        save(out / 'execution.json', records)
        run.check_returncode()


if __name__ == '__main__':
    main()
