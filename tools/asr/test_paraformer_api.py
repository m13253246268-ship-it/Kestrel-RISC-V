"""Public C ABI regression. Artifacts and malformed VQFs live in TEMP only."""
import argparse
import ctypes as C
import hashlib
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, type=Path)
    ap.add_argument('--gcc', default='gcc')
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    work = Path(tempfile.mkdtemp(prefix='paraformer_api_'))
    dll = work / 'paraformer.dll'
    subprocess.run([args.gcc, '-std=c11', '-O2', '-ffp-contract=off', '-shared',
                    '-DVLLM_PARAFORMER_NO_MAIN', str(root / 'src/model/vllm_paraformer.c'),
                    '-o', str(dll), '-lm'], check=True)
    dll_directory = os.add_dll_directory(str(Path(args.gcc).resolve().parent)) if os.name == 'nt' else None
    lib = C.CDLL(str(dll))
    ptr = C.c_void_p
    callback = C.CFUNCTYPE(None, ptr, C.POINTER(C.c_int), C.c_size_t, C.POINTER(C.c_float))
    create = lib.vllm_paraformer_model_create
    create.argtypes = [C.c_char_p, C.POINTER(ptr), C.c_char_p, C.c_size_t]
    destroy = lib.vllm_paraformer_model_destroy
    destroy.argtypes = [ptr]
    new = lib.vllm_paraformer_session_create
    new.argtypes = [ptr, callback, ptr, C.POINTER(ptr)]
    feed = lib.vllm_paraformer_session_feed
    feed.argtypes = [ptr, C.POINTER(C.c_int16), C.c_size_t]
    finish = lib.vllm_paraformer_session_finish
    reset = lib.vllm_paraformer_session_reset
    delete = lib.vllm_paraformer_session_destroy
    for fn in (finish, reset, delete):
        fn.argtypes = [ptr]
    err = C.create_string_buffer(160)
    models = []
    for _ in range(2):
        m = ptr()
        assert create(os.fsencode(args.model), C.byref(m), err, len(err)) == 0, err.value
        models.append(m)
    assert create(None, C.byref(ptr()), err, len(err)) == -1
    assert feed(None, None, 0) == -1
    assert reset(None) == -1
    sessions = []
    records = [[], [], []]
    callbacks = []
    for i, model in enumerate((models[0], models[0], models[1])):
        def capture(user, ids, count, logits, i=i):
            raw = C.string_at(logits, count * 8404 * 4) if count else b''
            records[i].append((tuple(ids[j] for j in range(count)), hashlib.sha256(raw).hexdigest()))
        cb = callback(capture)
        callbacks.append(cb)
        s = ptr()
        assert new(model, cb, None, C.byref(s)) == 0
        sessions.append(s)
    # Caller releases both model handles before any inference.
    for model in models:
        destroy(model)
    pcm = (C.c_int16 * 22000)(*((i * 7919 % 24001) - 12000 for i in range(22000)))

    def push(s, start, count):
        p = C.cast(C.byref(pcm, start * 2), C.POINTER(C.c_int16))
        assert feed(s, p, count) == 0

    def clean():
        for i, s in enumerate(sessions):
            assert reset(s) == 0
            records[i].clear()

    for n in (0, 1, 399, 400, 959, 960, 961, 9599, 9600, 9601, 10560, 20160):
        clean()
        assert feed(sessions[0], None, 0) == 0
        assert feed(sessions[0], None, 1) == -1
        push(sessions[0], 0, n)
        # Two independent models, with interleaved irregular input partitions.
        at = [0, 0]
        while min(at) < n:
            for j, step in enumerate((137, 997)):
                count = min(step, n - at[j])
                push(sessions[j + 1], at[j], count)
                at[j] += count
        for s in sessions:
            assert finish(s) == 0
            assert finish(s) == -3
            assert feed(s, None, 0) == -3
        assert records[0] == records[1] == records[2], n
        assert len(records[0]) == n // 9600 + 1
        print(f'PASS samples={n} chunks={len(records[0])} tokens={sum(len(x[0]) for x in records[0])}', flush=True)
    baseline = list(records[0])
    # Reset after partial audio and after a complete network chunk plus residue.
    for prefix in (351, 10003):
        clean()
        push(sessions[0], 0, prefix)
        assert reset(sessions[0]) == 0
        records[0].clear()
        push(sessions[0], 0, 20160)
        assert finish(sessions[0]) == 0
        assert records[0] == baseline
    # Real speech exercises nonempty decoder histories across several chunks.
    import wave
    example = root.parent / 'modls/iic--speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online/example/asr_example.wav'
    if example.exists():
        with wave.open(str(example), 'rb') as f:
            assert (f.getnchannels(), f.getsampwidth(), f.getframerate()) == (1, 2, 16000)
            speech = f.readframes(48000)
        synthetic = pcm
        pcm = (C.c_int16 * (len(speech) // 2)).from_buffer_copy(speech)
        clean()
        push(sessions[0], 0, len(pcm))
        for at in range(0, len(pcm), 1777):
            for s in sessions[1:]:
                push(s, at, min(1777, len(pcm) - at))
        for s in sessions:
            assert finish(s) == 0
        assert records[0] == records[1] == records[2]
        assert sum(bool(x[0]) for x in records[0]) > 1, 'speech must exercise decoder history'
        speech_baseline = list(records[0])
        assert reset(sessions[0]) == 0
        records[0].clear()
        push(sessions[0], 0, len(pcm))
        assert finish(sessions[0]) == 0 and records[0] == speech_baseline
        print('PASS real speech decoder cache, interleaving, reset; tokens=', sum(len(x[0]) for x in speech_baseline), flush=True)
        pcm = synthetic
    else:
        raise AssertionError('real speech fixture unavailable: ' + str(example))
    for s in sessions:
        delete(s)
    print('PASS reset discards frontend/network state; independent model lifetime', flush=True)
    # WAV CLI -> PFCH -> PFOUT must match the public PCM API exactly.
    import wave
    cli = work / 'paraformer.exe'
    subprocess.run([args.gcc, '-std=c11', '-O2', '-ffp-contract=off',
                    str(root / 'src/model/vllm_paraformer.c'), '-o', str(cli), '-lm'], check=True)
    wav = work / 'input.wav'
    with wave.open(str(wav), 'wb') as f:
        f.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
        f.writeframes(bytes(pcm)[:20160 * 2])
    env = dict(os.environ, KESTREL_PF_FEATURE_DUMP=str(work / 'input.pfch'))
    env['PATH'] = str(Path(args.gcc).resolve().parent) + os.pathsep + env.get('PATH', '')
    subprocess.run([str(cli), str(args.model), str(wav)], check=True, env=env)
    subprocess.run([str(cli), str(args.model), str(work / 'input.pfch'), str(work / 'output.pfout')], check=True, env=env)
    output = (work / 'output.pfout').read_bytes()
    assert output[:8] == b'PFOUT001'
    at, protocol = 8, []
    while at < len(output):
        ne, count = struct.unpack_from('<II', output, at)
        at += 8 + ne * 512 * 4 + ne * 4 + count * 512 * 4
        raw = output[at:at + count * 8404 * 4]
        values = struct.unpack('<' + 'f' * (count * 8404), raw)
        ids = tuple(max(range(8404), key=lambda j: values[t * 8404 + j]) for t in range(count))
        protocol.append((ids, hashlib.sha256(raw).hexdigest()))
        at += len(raw)
    assert at == len(output) and protocol == baseline
    print('PASS WAV CLI/PFCH/PFOUT exactly match public API logits', flush=True)

    def rejected(path):
        out = ptr(123)
        assert create(os.fsencode(path), C.byref(out), err, len(err)) == -2
        assert not out.value and err.value
        print('PASS reject', path.name, err.value.decode(), flush=True)

    header = args.model.open('rb').read(448)
    for size in (0, 447, 448, 456, 1024, 61887):
        p = work / f'truncated_{size}.vqf'
        data = bytearray(header + bytes(max(0, size - 448)))[:size]
        if size >= 448:
            struct.pack_into('<QQ', data, 200, 448 + 960 * 64, size - 8)
        p.write_bytes(data)
        rejected(p)
    corrupt = work / 'corrupt.vqf'
    shutil.copyfile(args.model, corrupt)
    for offset, replacement in ((496, struct.pack('<Q', 2**64 - 64)),
                                (504, struct.pack('<Q', 2**64 - 1)),
                                (448, b'x' * 32), (12, struct.pack('<I', 2**32 - 1)),
                                (200, struct.pack('<Q', 2**64 - 1))):
        with corrupt.open('r+b') as f:
            f.seek(offset)
            original = f.read(len(replacement))
            f.seek(offset)
            f.write(replacement)
        rejected(corrupt)
        with corrupt.open('r+b') as f:
            f.seek(offset)
            f.write(original)
    with corrupt.open('r+b') as f:
        f.seek(-1, 2)
        value = f.read(1)
        f.seek(-1, 2)
        f.write(bytes([value[0] ^ 1]))
    rejected(corrupt)
    corrupt.unlink()
    # A valid load after all failures verifies the process remains usable.
    m = ptr()
    assert create(os.fsencode(args.model), C.byref(m), err, len(err)) == 0
    destroy(m)
    print('PASS all public API regressions; artifacts:', work, flush=True)


if __name__ == '__main__':
    main()
