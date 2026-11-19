"""Compare board PFOUT replay against a frozen, passing native gate baseline (NumPy only)."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np


KEYS = ('encoder', 'alpha', 'predictor', 'decoder')
CASES = ('example',) + tuple(f'short_{n}' for n in
    (0, 1, 399, 400, 959, 960, 961, 9599, 9600, 9601)) + tuple(
    f'tail_{n}' for n in (0, 1, 959, 960, 961))


def exact(f, size):
    data = f.read(size)
    if len(data) != size:
        raise ValueError(f'truncated {f.name}')
    return data


def features(path):
    records = []
    with path.open('rb') as f:
        if exact(f, 8) != b'PFCH0001':
            raise ValueError(f'invalid PFCH: {path}')
        while header := f.read(16):
            nf, final, tail, reset = struct.unpack('<4I', header)
            payload = exact(f, nf * 560 * 4)
            records.append((nf, final, tail, reset, payload))
    return records


def outputs(path, count):
    records = []
    with path.open('rb') as f:
        if exact(f, 8) != b'PFOUT001':
            raise ValueError(f'invalid PFOUT: {path}')
        for _ in range(count):
            ne, nt = struct.unpack('<2I', exact(f, 8))
            record = {}
            for key, shape in zip(KEYS, ((ne, 512), (ne,), (nt, 512), (nt, 8404))):
                record[key] = np.frombuffer(exact(f, int(np.prod(shape)) * 4), '<f4').reshape(shape)
            records.append(record)
        if f.read(1):
            raise ValueError(f'extra PFOUT chunks: {path}')
    return records


def compare(actual, ref):
    if actual.shape != ref.shape:
        return {'passed': False, 'shape_mismatch': [list(ref.shape), list(actual.shape)]}
    finite = bool(np.isfinite(actual).all() and np.isfinite(ref).all())
    diff = np.abs(actual.astype(np.float64) - ref.astype(np.float64))
    return {'passed': finite and bool(np.allclose(actual, ref, atol=0.001, rtol=0.0001)),
            'max_abs': float(diff.max(initial=0)),
            'rmse': float(np.sqrt(np.mean(diff ** 2))) if diff.size else 0.0}


def sha256(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', required=True, type=Path)
    p.add_argument('--actual', required=True, type=Path)
    p.add_argument('--report', required=True, type=Path)
    args = p.parse_args()
    source = args.baseline / 'comparison.json'
    frozen = json.loads(source.read_text(encoding='utf-8'))
    if (frozen.get('passed') is not True or
            tuple(case['name'] for case in frozen['cases']) != CASES or
            any(case.get('passed') is not True for case in frozen['cases'])):
        raise ValueError('baseline must contain all 16 passing frozen gate cases')
    report = {'reference': 'frozen native gate PFOUT, not original PyTorch tensors',
              'baseline_report_sha256': sha256(source), 'atol': 0.001, 'rtol': 0.0001,
              'cases': [], 'passed': True}
    for case in frozen['cases']:
        name = case['name']
        inp = args.baseline / name / 'features.pfch'
        records = features(inp)
        n = len(case['frontend'])
        if not n or len(records) != 3 * n or records[:n] != records[n:2*n]:
            raise ValueError(f'{name}: invalid golden/reset replay input')
        if any(bool(r[3]) != (i % n == 0) for i, r in enumerate(records)):
            raise ValueError(f'{name}: invalid reset markers')
        ref_path = args.baseline / name / 'native.pfout'
        out_path = args.actual / name / 'native.pfout'
        reference = outputs(ref_path, len(records))
        actual = outputs(out_path, len(records))
        item = {'name': name, 'input_sha256': sha256(inp),
                'reference_sha256': sha256(ref_path), 'actual_sha256': sha256(out_path),
                'chunks': [], 'passed': True}
        for j, (a, b) in enumerate(zip(actual, reference)):
            check = {'path': ('golden', 'reset', 'wav')[j // n], 'chunk': j % n,
                     'tokens': len(a['predictor'])}
            for key in KEYS:
                check[key] = compare(a[key], b[key])
                item['passed'] &= check[key]['passed']
            check['ids_equal'] = bool(np.array_equal(a['decoder'].argmax(-1), b['decoder'].argmax(-1)))
            item['passed'] &= check['ids_equal']
            if n <= j < 2*n:
                check['reset_exact'] = all(np.array_equal(a[k], actual[j-n][k]) for k in KEYS)
                item['passed'] &= check['reset_exact']
            item['chunks'].append(check)
        report['cases'].append(item)
        report['passed'] &= item['passed']
        print(json.dumps({'case': name, 'passed': item['passed'], 'chunks': len(records)}), flush=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('PASS' if report['passed'] else 'FAIL', args.report, flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
