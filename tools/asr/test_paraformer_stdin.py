"""Compare WAV and live PCM pipe; require output before closing input."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import wave


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--exe', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--wav', required=True)
    args = p.parse_args()
    command = [args.exe, args.model]
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', VLLM_THREADS='1')
    env.pop('KESTREL_PF_FEATURE_DUMP', None)
    with wave.open(args.wav, 'rb') as f:
        assert (f.getnchannels(), f.getsampwidth(), f.getframerate()) == (1, 2, 16000)
        pcm = f.readframes(f.getnframes())
    reference = subprocess.run(command + [args.wav], env=env, capture_output=True, check=True).stdout
    print('PASS WAV reference', reference.decode('utf-8').strip(), flush=True)
    child = subprocess.Popen(command + ['--pcm-stdin'], env=env, stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    output = bytearray()
    emitted = threading.Event()

    def drain():
        while True:
            data = child.stdout.read(4096)
            if not data:
                return
            output.extend(data)
            emitted.set()

    reader = threading.Thread(target=drain)
    reader.start()
    try:
        # Supply 3 seconds, then wait with stdin OPEN: an EOF-only recognizer fails.
        split = min(96000, len(pcm))
        at = 0
        sizes = (1, 3199, 7, 641, 4093)
        while at < split:
            n = min(sizes[at % len(sizes)], split - at)
            child.stdin.write(pcm[at:at+n])
            at += n
            time.sleep(0.001)
        assert emitted.wait(180), 'no incremental text before EOF'
        print('PASS incremental output while stdin remains open', flush=True)
        while at < len(pcm):
            n = min(3201, len(pcm) - at)
            child.stdin.write(pcm[at:at+n])
            at += n
            time.sleep(0.001)
        child.stdin.close()
        assert child.wait(timeout=600) == 0, child.stderr.read()
        reader.join(timeout=5)
        assert not reader.is_alive()
        assert bytes(output) == reference, (bytes(output), reference)
        print('PASS irregular byte partitions equal WAV', flush=True)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
        reader.join(timeout=5)
        for stream in (child.stdin, child.stdout, child.stderr):
            stream.close()
    with tempfile.TemporaryDirectory(prefix='pf_empty_') as work:
        wav = Path(work) / 'empty.wav'
        with wave.open(str(wav), 'wb') as f:
            f.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
            f.writeframes(b'')
        empty = subprocess.run(command + [str(wav)], env=env, capture_output=True, check=True)
        result = subprocess.run(command + ['--pcm-stdin'], input=b'', env=env, capture_output=True, check=True)
        assert result.stdout == empty.stdout
    print('PASS empty EOF equals empty WAV finish', flush=True)
    result = subprocess.run(command + ['--pcm-stdin'], input=b'\x01', env=env, capture_output=True)
    assert result.returncode != 0 and b'incomplete PCM16 sample' in result.stderr
    print('PASS odd byte EOF rejected', flush=True)


if __name__ == '__main__':
    main()
