"""固定 Audio8：冻结 Mel，验证 C encoder、投影器与完整 LM，并回归 hidden 输入。"""
import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import re
import shlex
import struct
import subprocess

import numpy as np

GRAPH_HASH = 'da046a67c2fd414ad5b70a53a11decf1d495aa807eda290aae6cf42eb6dda756'


def export(v, out):
    g = v.meta['graphs']['lm_cache_decode']
    if hashlib.sha256(json.dumps(g, sort_keys=True).encode()).hexdigest() != GRAPH_HASH:
        raise ValueError('Unsupported Audio8 LM graph')
    refs = []

    def add(name, shape):
        a = v.tensors[name]
        if a.dtype != np.dtype('<f4') or a.shape != shape:
            raise ValueError(f'Unsupported shape/dtype: {name}: {a.shape}')
        refs.append((list(v.entries).index(name), *shape))

    add(v.meta['embedding'], (151936, 512))
    # 清单统一用二维行列；一维向量按一行处理。
    def vector(name, n):
        a = v.tensors[name]
        if v.entries[name][2] != 1 or a.dtype != np.dtype('<f4') or a.size != n:
            raise ValueError('Unsupported vector')
        refs.append((list(v.entries).index(name), 1, n))
    vector(g['tensors']['core.norm.weight'], 512)
    vector(g['tensors']['onnx::Expand_120'], 32)
    p = v.meta['projector']
    vector(p['norm_weight'], 1024)
    vector(p['norm_bias'], 1024)
    add(p['linear_weight'], (512, 1024))
    vector(p['linear_bias'], 512)
    for i in range(8):
        suffix = f'_{i}' if i else ''
        prefix = f'core.layers.{i}.'
        for n in ('input_layernorm', 'post_attention_layernorm'):
            vector(g['tensors'][prefix + n + '.weight'], 512)
        for n in ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'mlp/gate_proj', 'mlp/up_proj', 'mlp/down_proj'):
            # ONNX 权重为 [in,out]；C 加载后转置一次。
            role = '/' + n + suffix + '/MatMul'
            shape = (1408, 512) if n.endswith('down_proj') else (512, 1408) if n.startswith('mlp/') else (512, 512)
            add(g['tensors'][g['weight_roles'][role]], shape)
        for n in ('q_proj', 'k_proj', 'v_proj'):
            vector(g['tensors'][prefix + 'self_attn.' + n + '.bias'], 512)
    with v.path.open('rb') as f:
        f.seek(-8, 2)
        digest = f.read(8)
    with (out / 'asr_lm.map').open('wb') as f:
        f.write(b'ASRLM001' + digest + struct.pack('<I', len(refs)))
        for ref in refs:
            f.write(struct.pack('<III', *ref))


AUDIO_GRAPH_HASH = 'c03f10a6fbb6c5b413b0b4bc08927dee4e59334e91c2db3dca6f095cf9492e16'


def export_audio(v, out):
    g = v.meta['graphs']['audio_hidden']
    if hashlib.sha256(json.dumps(g, sort_keys=True).encode()).hexdigest() != AUDIO_GRAPH_HASH:
        raise ValueError('Unsupported Audio8 audio graph')
    refs = []
    def add(name, shape):
        key = g['tensors'][name]
        a = v.tensors[key]
        if a.dtype != np.dtype('<f4') or a.shape != shape:
            raise ValueError(f'Audio shape/dtype mismatch: {name} {a.shape}')
        refs.append((list(v.entries).index(key), shape[0] if shape else 1, int(np.prod(shape[1:])) if shape else 1))
    def pair(prefix, rows, cols=None):
        add(prefix + '.weight', (rows, cols) if cols else (rows,))
        add(prefix + '.bias', (rows,))
    for i in range(3):
        add(f'audio_encoder.conv2d{i+1}.weight', (480, 1 if i == 0 else 480, 3, 3))
        add(f'audio_encoder.conv2d{i+1}.bias', (480,))
    add(g['weight_roles']['/conv_out/MatMul'], (7680, 896))
    add('/Constant_19_output_0', (1, 13, 896))
    for name in ('/Constant_9_output_0', '/layers.0/self_attn/Constant_10_output_0', '/layers.0/self_attn/Constant_11_output_0'):
        add(name, ())
    for i in range(18):
        prefix = f'audio_encoder.layers.{i}.'
        for name in ('self_attn_layer_norm', 'final_layer_norm'):
            pair(prefix + name, 896)
        for name in ('q_proj', 'k_proj', 'v_proj', 'out_proj'):
            pair(prefix + 'self_attn.' + name, 896, 896)
        pair(prefix + 'fc1', 3584, 896)
        pair(prefix + 'fc2', 896, 3584)
    pair('audio_encoder.ln_post', 896)
    pair('audio_encoder.proj1', 896, 896)
    pair('audio_encoder.proj2', 1024, 896)
    for i in range(4):
        prefix = f'audio_mlp_tower.layers.{i}.'
        pair(prefix + 'norm', 1024)
        pair(prefix + 'fc1', 4096, 1024)
        pair(prefix + 'fc2', 1024, 4096)
    pair('audio_mlp_tower.final_norm', 1024)
    assert len(refs) == 331
    path = out / 'asr_lm.map'
    data = path.read_bytes()
    path.write_bytes(b'ASRLM002' + data[8:] + AUDIO_GRAPH_HASH.encode() + b''.join(struct.pack('<III', *r) for r in refs))


def parse_cflags(value):
    """Only expose target selection and existing ASR switches; keep FP flags fixed."""
    try:
        flags = shlex.split(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    switches = ('SCALAR', 'W8_RVV', 'CONV_PACK', 'CONV_SCALAR', 'FFN_SCALAR',
                'TAIL_SCALAR', 'QKV_SCALAR', 'OUT_SCALAR')
    for flag in flags:
        target = re.fullmatch(r'-m(?:arch|abi|cpu|tune)=[A-Za-z0-9_.+\-]+', flag)
        macro = any(flag in ('-DVLLM_ASR_' + name, '-DVLLM_ASR_' + name + '=1')
                    for name in switches)
        if not target and not macro:
            raise argparse.ArgumentTypeError(
                f'unsupported cflag {flag!r}; only -march/-mabi/-mcpu/-mtune=VALUE '
                'and existing ASR path switches are allowed; FP flags are fixed')
    return flags


MASK_BOUNDARIES = (-1, 0, 1, 2, 7, 8, 9, 15, 16, 17, 95, 96, 97, 99, 100, 101, 799, 800, 2999, 3000, 3001)


def error(a, b):
    d = a.astype(np.float64) - b.astype(np.float64)
    return dict(max_abs=float(np.max(np.abs(d))), rms=float(np.sqrt(np.mean(d * d))))


def prepare_reference(args):
    from asr_vqf_run import VQF, Graph, prepare, project, generation, normalize_text
    v = VQF(args.vqf, verify=False)
    export(v, args.out)
    export_audio(v, args.out)
    features, length, count, ids, tokenizer = prepare(v, args.audio)
    if not 0 < len(ids) < 512:
        raise ValueError('Prompt exceeds capacity')
    print('NumPy audio encoder and reference generation', flush=True)
    graph = Graph(v, 'audio_hidden')
    hidden, mask = graph.run({'audios': features, 'audio_feature_lengths': np.asarray([length], np.int64)})
    valid = hidden[mask.astype(bool)]
    projected = project(hidden, mask, count, {k: v.tensors[n] for k, n in v.meta['projector'].items()})
    embeds = v.tensors[v.meta['embedding']][ids].copy()
    positions = np.asarray(ids) == 151646
    if int(positions.sum()) != count:
        raise ValueError('Audio token count mismatch')
    embeds[positions] = projected
    for name, magic, size, values in [('input.bin', b'ASRIN001', len(valid), valid),
                                      ('mel.bin', b'ASRMEL01', length, features)]:
        with (args.out / name).open('wb') as f:
            f.write(magic + struct.pack('<IIII', size, count, len(ids), args.max_new_tokens))
            f.write(np.asarray(values, '<f4').tobytes())
            f.write(np.asarray(ids, '<u4').tobytes())
    observed = []
    def observe(step, outputs):
        observed.append([np.asarray(a[:, :, -1:, :]).copy() for a in outputs[1:]])
    tokens, stop, history = generation(Graph(v, 'lm_cache_prefill'), Graph(v, 'lm_cache_decode'), embeds,
                                       v.tensors[v.meta['embedding']], v.meta['bundle']['tokens'], args.max_new_tokens, observe)
    masks, allowed = zip(*(graph.run_outputs({'audio_feature_lengths': np.asarray([b], np.int64)},
                                           ['/Less_output_0', '/Or_output_0']) for b in MASK_BOUNDARIES))
    np.savez(args.out / 'hidden_reference.npz', mask=mask, valid=valid, projected=projected,
             history=np.asarray(history), observed=np.asarray(observed), masks=np.asarray(masks),
             allowed=np.asarray(allowed), expected=np.asarray(tokens + ([stop] if stop is not None else [])),
             reference_text=np.asarray(normalize_text(tokenizer.decode(tokens, skip_special_tokens=False))),
             max_new_tokens=np.asarray(args.max_new_tokens))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--vqf', type=Path, required=True)
    ap.add_argument('--audio', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=Path(os.environ.get('TEMP', '/tmp')) / 'asr_c')
    ap.add_argument('--cc', default='gcc')
    ap.add_argument('--cflags', type=parse_cflags, default='', metavar='FLAGS',
                    help='quoted target/ASR switches, e.g. --cflags="-march=rv64gcv -DVLLM_ASR_W8_RVV"; native execution required')
    ap.add_argument('--max-new-tokens', type=int, default=32)
    ap.add_argument('--assets-dir', type=Path)
    ap.add_argument('--reference', type=Path)
    args = ap.parse_args()
    (args.out / 'c_regression_baseline.json').unlink(missing_ok=True)
    if not 1 <= args.max_new_tokens <= 512:
        ap.error('max-new-tokens must be in 1..512')
    if args.assets_dir:
        import shutil
        from asr_c_assets import copy_assets, sha256
        reference = args.reference or args.assets_dir / 'reference'
        manifest, frozen = copy_assets(args.assets_dir, args.out, args.vqf, args.audio, reference)
        for name in ('input.bin', 'mel.bin', 'hidden_reference.npz'):
            shutil.copyfile(frozen / name, args.out / name)
            if sha256(args.out / name) != manifest['reference'][name]:
                raise ValueError('Copied reference SHA256 mismatch: ' + name)
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        prepare_reference(args)
    with np.load(args.out / 'hidden_reference.npz', allow_pickle=False) as frozen:
        mask, valid, projected, history, observed, masks, allowed_masks = (
            frozen[name] for name in ('mask', 'valid', 'projected', 'history', 'observed', 'masks', 'allowed'))
        expected = frozen['expected'].tolist()
        reference_text = str(frozen['reference_text'])
        if int(frozen['max_new_tokens']) != args.max_new_tokens:
            raise ValueError('Reference max-new-tokens mismatch')
    count = len(projected)
    if (mask.size != 390 or valid.shape != (int(mask.sum()), 1024)
            or projected.shape != (count, 512) or not 0 < count <= 375
            or history.shape != (len(expected), 151936) or not expected
            or observed.shape[:2] != (len(expected), 16) or observed.size != len(expected) * 16 * 512
            or len(masks) != len(MASK_BOUNDARIES) or len(allowed_masks) != len(MASK_BOUNDARIES)):
        raise ValueError('Invalid hidden reference dimensions')
    if not all(np.isfinite(a).all() for a in (valid, projected, history, observed)):
        raise ValueError('Nonfinite hidden reference')
    exe = args.out / ('asr_lm.exe' if os.name == 'nt' else 'asr_lm')
    source = Path(__file__).resolve().parents[2] / 'src/model/vllm_asr.c'
    cmd = [args.cc, '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror', '-pedantic', '-ffp-contract=off', str(source), str(source.parents[1] / 'core/vllm_tp.c'), '-I' + str(source.parents[2] / 'include/core'), '-I' + str(source.parents[2] / 'include/common'), '-D_GNU_SOURCE', '-pthread', '-o', str(exe), '-lm']
    cmd[1:1] = args.cflags
    print('Compile:', cmd, flush=True)
    subprocess.run(cmd, check=True)
    run = [str(exe), str(args.vqf.resolve()), str((args.out / 'asr_lm.map').resolve()), str((args.out / 'input.bin').resolve()), str((args.out / 'output.bin').resolve())]
    print('Run:', run, flush=True)
    with (args.out / 'hidden_run.log').open('w', encoding='utf-8') as log:
        subprocess.run(run, check=True, stdout=log, stderr=log)
    hidden_output = (args.out / 'output.bin').read_bytes()
    mel_run = run.copy()
    mel_run[3] = str((args.out / 'mel.bin').resolve())
    with (args.out / 'mel_run.log').open('w', encoding='utf-8') as log:
        subprocess.run(mel_run, check=True, stdout=log, stderr=log)
    comparisons = []
    actual_tokens = []
    with (args.out / 'output.bin').open('rb') as f:
        def array(n):
            raw = f.read(n * 4)
            if len(raw) != n * 4:
                raise ValueError('Truncated C output')
            return np.frombuffer(raw, '<f4')
        actual_mask = np.frombuffer(f.read(390 * 4), '<u4')
        if not np.array_equal(actual_mask, mask):
            raise AssertionError('C mask mismatch')
        he = error(array(len(valid) * 1024).reshape(valid.shape), valid)
        pe = error(array(count * 512).reshape(count, 512), projected)
        for step, logits in enumerate(history):
            actual = array(151936)
            kv = array(16 * 512).reshape(16, 512)
            token = struct.unpack('<I', f.read(4))[0]
            actual_tokens.append(token)
            comparisons.append(dict(step=step, logits=error(actual, logits), kv_max_abs=max(error(kv[i], observed[step][i].reshape(-1))['max_abs'] for i in range(16))))
        if f.read(1):
            raise ValueError('Unexpected extra C generation steps')
    # 负例使用独立临时文件，不修改原模型和有效 fixture。
    rejected = {}
    def reject(label, index, payload, expected_message):
        bad = args.out / (label + '.bin')
        bad.write_bytes(payload)
        command = run.copy()
        command[index] = str(bad.resolve())
        command[4] = str((args.out / 'rejected_output.bin').resolve())
        result = subprocess.run(command, capture_output=True, text=True)
        rejected[label] = result.returncode != 0 and expected_message in result.stderr
        if not rejected[label]:
            raise AssertionError(f'{label}: {result.returncode}, {result.stderr}')
    with args.vqf.open('rb') as stream:
        header = bytearray(stream.read(448))
    struct.pack_into('<I', header, 4, 5)
    reject('wrong_vqf_version', 1, header, 'unsupported VQF header')
    manifest = bytearray((args.out / 'asr_lm.map').read_bytes())
    struct.pack_into('<I', manifest, 24, 1)
    reject('wrong_tensor_shape', 2, manifest, 'unsupported fixed model shape')
    fixture = bytearray((args.out / 'input.bin').read_bytes())
    struct.pack_into('<I', fixture, 16, 512)
    reject('prompt_capacity_512', 3, fixture, 'invalid input dimensions/capacity')
    reject('truncated_input', 3, (args.out / 'input.bin').read_bytes()[:-4], 'truncated file')
    audio_manifest = bytearray((args.out / 'asr_lm.map').read_bytes())
    audio_manifest[20 + 103 * 12] ^= 1
    reject('wrong_audio_graph', 2, audio_manifest, 'audio graph identity mismatch')
    audio_manifest = bytearray((args.out / 'asr_lm.map').read_bytes())
    struct.pack_into('<I', audio_manifest, 20 + 103 * 12 + 64 + 4, 1)
    reject('wrong_audio_shape', 2, audio_manifest, 'unsupported audio tensor shape')
    mel_fixture = bytearray((args.out / 'mel.bin').read_bytes())
    struct.pack_into('<I', mel_fixture, 8, 3001)
    reject('mel_length_3001', 3, mel_fixture, 'invalid input dimensions/capacity')
    mel_fixture = bytearray((args.out / 'mel.bin').read_bytes())
    struct.pack_into('<f', mel_fixture, 24, float('nan'))
    reject('mel_nonfinite', 3, mel_fixture, 'nonfinite tensor')
    reject('mel_truncated_features', 3, (args.out / 'mel.bin').read_bytes()[:100], 'truncated file')
    mask_tests = {}
    for index, boundary in enumerate(MASK_BOUNDARIES):
        vm, allowed = masks[index], allowed_masks[index]
        text = subprocess.check_output([str(exe), '--mask', str(boundary)], text=True).strip()
        cm = np.asarray([int(c) for c in text], bool)
        group = (np.cumsum(cm) - 1) // 104
        ca = (cm[:, None] & cm[None, :] & (group[:, None] == group[None, :])) | (~cm[:, None] & np.eye(390, dtype=bool))
        mask_tests[str(boundary)] = np.array_equal(vm, cm) and np.array_equal(allowed, ca)
        if not mask_tests[str(boundary)]:
            raise AssertionError(f'Mask boundary failed: {boundary}')
    offset = count * 512 * 4
    regression_errors = [error(np.frombuffer(hidden_output[:offset], '<f4').reshape(count, 512), projected)['max_abs']]
    regression_tokens = []
    for step, logits in enumerate(history):
        regression_errors.append(error(np.frombuffer(hidden_output, '<f4', 151936, offset), logits)['max_abs'])
        offset += 151936 * 4
        kv = np.frombuffer(hidden_output, '<f4', 16 * 512, offset).reshape(16, 512)
        regression_errors.extend(error(kv[i], observed[step][i].reshape(-1))['max_abs'] for i in range(16))
        offset += 16 * 512 * 4
        regression_tokens.append(struct.unpack_from('<I', hidden_output, offset)[0])
        offset += 4
    regression_pass = offset == len(hidden_output) and regression_tokens == expected and max(regression_errors) <= 1e-3
    if not regression_pass:
        raise AssertionError('Hidden input regression failed')
    numeric = [he['max_abs'], pe['max_abs']] + [c['logits']['max_abs'] for c in comparisons] + [c['kv_max_abs'] for c in comparisons]
    report = dict(boundary='Python WAV/features -> C F32 audio encoder + projector + LM (existing double accumulations); Python tokenizer decoding',
                  mask_boundaries=mask_tests, hidden_regression_pass=regression_pass, hidden_regression_max_abs=max(regression_errors),
                  hidden_error=he, binary_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(), binary_mtime_ns=exe.stat().st_mtime_ns, mel_run=mel_run,
                  projector_error=pe, steps=comparisons, tokens=actual_tokens, expected_tokens=expected,
                  tokens_equal=actual_tokens == expected, numeric_tolerance=1e-3,
                  numeric_pass=all(np.isfinite(e) and e <= 1e-3 for e in numeric),
                  reference_text=reference_text,
                  rejection_tests=rejected, compile=cmd, run=run)
    (args.out / 'alignment.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not report['numeric_pass'] or not report['tokens_equal']:
        raise AssertionError('C alignment failed')
    from asr_c_assets import sha256
    proof = dict(platform=[platform.system(), platform.machine()],
                 vqf_sha256=sha256(args.vqf), audio_sha256=sha256(args.audio),
                 files={name: sha256(args.out / name) for name in
                        ('alignment.json', 'input.bin', 'mel.bin', 'hidden_reference.npz', 'asr_lm.map', 'output.bin')})
    (args.out / 'c_regression_baseline.json').write_text(json.dumps(proof, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
