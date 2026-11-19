"""Local functional checks for a package produced by the full WAV gate."""
import argparse
import builtins
import json
import os
import subprocess
from pathlib import Path
import sys
import tempfile

from asr_c_assets import PACKAGE_FILES, REFERENCE_FILES, copy_assets, load_manifest, sha256


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('assets', 'reference', 'vqf', 'other-vqf', 'audio', 'prior-assets'):
        ap.add_argument('--' + name, required=True, type=Path)
    ap.add_argument('--replay-out', type=Path)
    ap.add_argument('--c-baseline', type=Path)
    args = ap.parse_args()
    original_import = builtins.__import__

    def guarded(name, *a, **kw):
        if name.split('.')[0] in ('torch', 'transformers', 'tokenizers', 'asr_vqf_run'):
            raise AssertionError('Export dependency imported: ' + name)
        return original_import(name, *a, **kw)

    builtins.__import__ = guarded
    import asr_c_wav_gate
    assert callable(asr_c_wav_gate.main)
    manifest = load_manifest(args.assets, args.vqf, args.audio, args.reference)
    for name in ('asr_lm.map', 'asr_wav.assets'):
        assert sha256(args.assets / name) == sha256(args.prior_assets / name), name
    passed = ['no_export_imports', 'prior_export_bytes_equal']
    with tempfile.TemporaryDirectory(prefix='asr_r4_tests_') as temporary:
        root = Path(temporary)
        copied = root / 'copied'
        _, reference = copy_assets(args.assets, copied, args.vqf, args.audio, args.reference)
        assert all(sha256(copied / n) == sha256(args.assets / n) for n in PACKAGE_FILES)
        assert all(sha256(reference / n) == sha256(args.reference / n) for n in REFERENCE_FILES)
        passed.append('copied_bytes_equal')

        def reject(label, action):
            try:
                action()
            except (ValueError, FileNotFoundError):
                passed.append(label)
            else:
                raise AssertionError('Accepted negative: ' + label)

        if args.c_baseline:
            import shutil
            baseline = root / 'baseline'
            shutil.copytree(args.c_baseline, baseline)
            check = lambda: asr_c_wav_gate.regression_baseline(baseline, reference, copied, args.vqf, args.audio)
            check()
            passed.append('current_baseline_accepted')
            reject('baseline_self_comparison', lambda: asr_c_wav_gate.regression_baseline(baseline, reference, baseline, args.vqf, args.audio))
            reject('baseline_wrong_model', lambda: asr_c_wav_gate.regression_baseline(baseline, reference, copied, args.other_vqf, args.audio))
            proof_path = baseline / 'c_regression_baseline.json'
            original_proof = proof_path.read_text()
            for label, change in (
                    ('baseline_wrong_platform', lambda p: p.update(platform=['wrong', 'wrong'])),
                    ('baseline_wrong_output', lambda p: p['files'].update({'output.bin': '0' * 64}))):
                proof = json.loads(original_proof)
                change(proof)
                proof_path.write_text(json.dumps(proof))
                reject(label, check)
            proof_path.write_text(original_proof)
            frozen_input = reference / 'mel.bin'
            original_input = frozen_input.read_bytes()
            frozen_input.write_bytes(original_input + b'x')
            reject('baseline_wrong_frozen_input', check)
            frozen_input.write_bytes(original_input)
            alignment_path = baseline / 'alignment.json'
            original_alignment = alignment_path.read_text(encoding='utf-8')
            failed = json.loads(original_alignment)
            failed['numeric_pass'] = False
            alignment_path.write_text(json.dumps(failed), encoding='utf-8')
            proof = json.loads(original_proof)
            proof['files']['alignment.json'] = sha256(alignment_path)
            proof_path.write_text(json.dumps(proof))
            reject('baseline_failed_hidden', check)
            alignment_path.write_text(original_alignment, encoding='utf-8')
            proof_path.write_text(original_proof)
            target = copied / 'mel_regression.bin'
            os.link(baseline / 'output.bin', target)
            reject('baseline_hardlink_self_comparison', check)
            target.unlink()
            proof_path.unlink()
            reject('baseline_missing_success_proof', check)
        reject('different_vqf', lambda: load_manifest(copied, args.other_vqf, args.audio, reference))
        wrong_audio = root / 'wrong.wav'
        wrong_audio.write_bytes(args.audio.read_bytes() + b'x')
        reject('different_audio', lambda: load_manifest(copied, args.vqf, wrong_audio, reference))
        reject('overlapping_output', lambda: copy_assets(args.assets, args.assets / 'nested', args.vqf, args.audio, args.reference))
        for name in PACKAGE_FILES:
            path = copied / name
            raw = path.read_bytes()
            path.write_bytes(raw + b'x')
            reject('corrupt:' + name, lambda: load_manifest(copied, args.vqf, args.audio, reference))
            path.unlink()
            reject('missing:' + name, lambda: load_manifest(copied, args.vqf, args.audio, reference))
            path.write_bytes(raw)
        for name in REFERENCE_FILES:
            path = reference / name
            raw = path.read_bytes()
            path.write_bytes(raw.replace(b'0.', b'1.', 1) if name == 'alignment.json' else raw + b'x')
            reject('reference_corrupt:' + name, lambda: load_manifest(copied, args.vqf, args.audio, reference))
            path.unlink()
            reject('reference_missing:' + name, lambda: load_manifest(copied, args.vqf, args.audio, reference))
            path.write_bytes(raw)
        from unittest.mock import patch
        import shutil
        real_copy = shutil.copyfile
        def corrupt_copy(source, target):
            result = real_copy(source, target)
            if Path(source).name == 'hidden_reference.npz':
                with Path(target).open('ab') as stream:
                    stream.write(b'x')
            return result
        with patch('asr_c_assets.shutil.copyfile', side_effect=corrupt_copy):
            reject('copy_corruption', lambda: copy_assets(args.assets, root / 'bad_copy', args.vqf, args.audio, args.reference))
        manifest_path = copied / 'assets_manifest.json'
        for label, mutate in (
                ('version', lambda m: m.update(schema=2)),
                ('boolean_version', lambda m: m.update(schema=True)),
                ('traversal', lambda m: m['files'].update({'../escape': '0' * 64})),
                ('absolute_path', lambda m: m['files'].update({str(root / 'escape'): '0' * 64})),
                ('missing_entry', lambda m: m['files'].pop('asr_lm.map')),
                ('missing_reference_entry', lambda m: m['reference'].pop('input.bin'))):
            changed = json.loads(json.dumps(manifest))
            mutate(changed)
            manifest_path.write_text(json.dumps(changed), encoding='utf-8')
            reject(label, lambda: load_manifest(copied, args.vqf, args.audio, reference))
        manifest_path.unlink()
        reject('missing_manifest', lambda: load_manifest(copied, args.vqf, args.audio, reference))
    assert load_manifest(args.assets, args.vqf, args.audio, args.reference) == manifest
    print(json.dumps({'passed': passed, 'count': len(passed)}, indent=2), flush=True)
    if args.replay_out:
        with tempfile.TemporaryDirectory(prefix='asr_import_guard_') as temporary:
            guard = Path(temporary)
            (guard / 'sitecustomize.py').write_text(
                "import sys, atexit, json, os\n"
                "class Guard:\n"
                "    def find_spec(self, fullname, path=None, target=None):\n"
                "        if fullname.split('.')[0] in ('torch', 'transformers', 'tokenizers', 'asr_vqf_run', 'onnx_to_vqf'):\n"
                "            raise RuntimeError('Forbidden reference import: ' + fullname)\n"
                "sys.meta_path.insert(0, Guard())\n"
                "@atexit.register\n"
                "def audit():\n"
                "    print('IMPORT_AUDIT ' + json.dumps(dict(pid=os.getpid(), modules=sorted(set(n.split('.')[0] for n in sys.modules)))))\n",
                encoding='utf-8')
            env = os.environ.copy()
            env['PYTHONPATH'] = str(guard)
            subprocess.run([sys.executable, '-B', str(Path(__file__).with_name('asr_supplement.py')),
                            'full', '--local-check', '--vqf', str(args.vqf), '--audio', str(args.audio),
                            '--assets-dir', str(args.assets), '--out', str(args.replay_out)], env=env, check=True)
        for kind in ('hidden', 'wav'):
            log = (args.replay_out / (kind + '.log')).read_text(encoding='utf-8')
            audits = [json.loads(line.removeprefix('IMPORT_AUDIT ')) for line in log.splitlines() if line.startswith('IMPORT_AUDIT ')]
            assert len(audits) == 1
            modules = set(audits[0]['modules'])
            assert not modules.intersection(('torch', 'transformers', 'tokenizers', 'asr_vqf_run', 'onnx_to_vqf'))
            print(kind + ' dependencies: ' + json.dumps({name: name in modules for name in ('numpy', 'scipy')}), flush=True)
            assert all(sha256(args.replay_out / kind / n) == sha256(args.assets / n) for n in PACKAGE_FILES)
        for name in ('input.bin', 'mel.bin', 'output.bin', 'hidden_reference.npz'):
            assert sha256(args.replay_out / 'hidden' / name) == sha256(args.reference / name), name
        for name in ('wav_trace.bin', 'mel_regression.bin', 'hidden_regression.bin', 'transcript.txt'):
            assert sha256(args.replay_out / 'wav' / name) == sha256(args.assets / name), name
        print('Full dual-gate replay passed with exporter imports blocked; reference and C output bytes unchanged', flush=True)


if __name__ == '__main__':
    main()
