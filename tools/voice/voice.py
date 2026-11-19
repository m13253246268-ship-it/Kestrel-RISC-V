#!/usr/bin/env python3
import argparse
import array
import collections
import ctypes
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import threading
import uuid
import wave

RATE = 16000
CHUNK = 1600


class Kws:
    def __init__(self, config):
        self.lib = ctypes.CDLL(config['bridge'])
        self.lib.voice_open.argtypes = [ctypes.c_char_p] * 5
        self.lib.voice_open.restype = ctypes.c_void_p
        self.lib.voice_feed.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        self.lib.voice_reset.argtypes = [ctypes.c_void_p]
        self.lib.voice_close.argtypes = [ctypes.c_void_p]
        self.ptr = self.lib.voice_open(*(os.fsencode(config[k]) for k in ('encoder', 'decoder', 'joiner', 'tokens', 'keywords')))
        if not self.ptr:
            raise RuntimeError('KWS initialization failed')

    def feed(self, pcm):
        samples = array.array('h', pcm)
        if sys.byteorder != 'little':
            samples.byteswap()
        floats = (ctypes.c_float * len(samples))(*(x / 32768 for x in samples))
        word = ctypes.create_string_buffer(4096)
        result = self.lib.voice_feed(self.ptr, floats, len(samples), word, len(word))
        if result < 0:
            raise RuntimeError('keyword exceeds buffer')
        return word.value.decode('utf-8') if result else ''

    def reset(self):
        if self.lib.voice_reset(self.ptr):
            raise RuntimeError('KWS reset failed')

    def close(self):
        self.lib.voice_close(self.ptr)


def energy(pcm):
    samples = array.array('h', pcm)
    if sys.byteorder != 'little':
        samples.byteswap()
    return math.sqrt(sum(x * x for x in samples) / max(1, len(samples))) / 32768


class Microphone:
    def __init__(self, device):
        self.device = device
        self.process = None
        self.reader = None
        self.condition = threading.Condition()
        self.queue = collections.deque(maxlen=2)
        self.generation = 0
        self.busy = self.closed = self.eof = False
        self.error = None
        self.dropped = 0

    def set_busy(self, busy):
        with self.condition:
            self.busy = busy
            self.generation += 1
            self.dropped += len(self.queue)
            self.queue.clear()
            self.condition.notify_all()

    def start(self):
        if self.closed:
            raise RuntimeError('microphone is closed')
        if self.process is None:
            argv = (['pw-record', '--target', self.device[len('pipewire:'):],
                     '--rate', str(RATE), '--channels', '1', '--format', 's16', '-']
                    if self.device.startswith('pipewire:') else
                    ['arecord', '-q', '-D', self.device, '-t', 'raw', '-f', 'S16_LE',
                     '-r', str(RATE), '-c', '1'])
            self.process = subprocess.Popen(
                argv, shell=False, stdout=subprocess.PIPE,
                stderr=None, bufsize=0)
            self.reader = threading.Thread(target=self._drain, name='voice-pcm-reader')
            self.reader.start()

    def _drain(self):
        try:
            while True:
                with self.condition:
                    if self.closed:
                        return
                    generation = self.generation
                pcm = bytearray()
                while len(pcm) < CHUNK * 2:
                    data = self.process.stdout.read(CHUNK * 2 - len(pcm))
                    if not data:
                        if pcm:
                            raise OSError('partial microphone block at EOF: %d bytes' % len(pcm))
                        code = self.process.wait(timeout=2)
                        if code:
                            raise OSError('arecord exited with code %d' % code)
                        return
                    pcm.extend(data)
                with self.condition:
                    if self.closed:
                        return
                    if self.busy or generation != self.generation:
                        self.dropped += 1
                    else:
                        if len(self.queue) == self.queue.maxlen:
                            self.dropped += 1
                        self.queue.append(bytes(pcm))
                    self.condition.notify_all()
        except Exception as exc:
            with self.condition:
                if not self.closed:
                    self.error = exc
        finally:
            with self.condition:
                self.eof = True
                self.condition.notify_all()

    def read(self, timeout=None):
        self.start()
        with self.condition:
            ready = self.condition.wait_for(
                lambda: self.queue or self.eof or self.closed, timeout)
            if not ready:
                raise TimeoutError('microphone session deadline')
            if self.error is not None:
                raise self.error
            return self.queue.popleft() if self.queue else b''

    def close(self):
        with self.condition:
            self.closed = True
            self.queue.clear()
            self.condition.notify_all()
        process = self.process
        if process is None:
            return
        try:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        finally:
            if self.reader is not None:
                self.reader.join(timeout=3)
            process.stdout.close()
        if self.reader is not None and self.reader.is_alive():
            raise RuntimeError('PCM reader did not stop')


def listen_microphone(v, device, seconds):
    mic = Microphone(device)
    v.microphone = mic
    deadline = time.monotonic() + seconds
    frames = samples = 0
    heartbeat = time.monotonic()
    v.emit('mic_limit', device=device, seconds=seconds, end_timestamp=time.time()+seconds,
           detail='single continuous PCM pipe; bounded queue; busy audio drained and discarded')
    try:
        while time.monotonic() < deadline:
            try:
                pcm = mic.read(timeout=max(0, deadline-time.monotonic()))
            except TimeoutError:
                break
            if not pcm:
                v.emit('capture_eof')
                break
            frames += 1
            samples += len(pcm)//2
            if frames == 1 or time.monotonic() >= heartbeat:
                v.emit('capture_health', device=device, frames=frames, samples=samples,
                       rms=energy(pcm), dropped_blocks=mic.dropped, capture_pid=mic.process.pid)
                heartbeat = time.monotonic()+2
            v.feed(pcm)
    except Exception as exc:
        v.emit('capture_error', error=str(exc))
        raise
    finally:
        mic.close()
        v.microphone = None
        v.emit('mic_closed', dropped_blocks=mic.dropped)


class Voice:
    def __init__(self, config, kws, output):
        self.c, self.kws = config, kws
        self.microphone = None
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.state = 'LISTENING'
        self.pos = self.requests = self.asr_calls = 0
        self.request_id = None
        self.keyword = ''
        self.pre = collections.deque(maxlen=int(config['pre_roll_ms'] / 100))
        self.capture = bytearray()
        self.quiet = self.elapsed = self.voiced = 0
        self.emit('ready', kws_loads=1, demo_only=config.get('demo_only', False), asr_resident=False)

    def emit(self, event, **fields):
        print(json.dumps(dict(event=event, state=self.state, timestamp=time.time(), monotonic_s=time.monotonic(), audio_sample=self.pos,
                              request_id=self.request_id, keyword=self.keyword, **fields), ensure_ascii=False), flush=True)

    def transition(self, state):
        self.state = state
        self.emit('state')

    def recover(self):
        self.capture.clear()
        self.pre.clear()
        self.kws.reset()
        self.quiet = self.elapsed = 0
        self.transition('RECOVERING')
        if self.microphone is not None:
            self.microphone.set_busy(False)

    def finish(self, reason):
        if self.voiced < self.c.get('min_speech_ms', 200)*16:
            self.emit('capture_rejected', error='empty_audio', stop=reason)
            self.recover()
            return
        wav = self.output / (self.request_id + '.wav')
        try:
            with wave.open(str(wav), 'wb') as f:
                f.setparams((1, 2, RATE, 0, 'NONE', 'not compressed'))
                f.writeframes(self.capture)
        except OSError as exc:
            self.emit('capture_error', error=str(exc))
            self.recover()
            return
        self.emit('capture', path=str(wav), start_sample=self.start, end_sample=self.pos, samples=len(self.capture)//2, stop=reason, endpoint='energy_not_learned_vad')
        if self.microphone is not None:
            self.microphone.set_busy(True)
        self.transition('ASR_BUSY')
        self.emit('listening_paused', detail='microphone continuously drained; busy PCM discarded' if self.microphone else 'file replay pauses during synchronous ASR')
        paraformer = self.c.get('asr_backend', 'audio8') == 'paraformer'
        argv = ([self.c['asr_cli'], self.c['asr_model'], str(wav)] if paraformer else
                [self.c['asr_cli'], '--wav', self.c['asr_model'], self.c['asr_map'], self.c['asr_assets'], str(wav), '-'])
        env = dict(os.environ, VLLM_THREADS='1' if paraformer else '8')
        if paraformer:
            env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        self.emit('asr_start', backend='paraformer' if paraformer else 'audio8', streaming=False)
        self.asr_calls += 1
        start = time.monotonic()
        try:
            result = subprocess.run(argv, env=env, shell=False, capture_output=True, timeout=self.c['asr_timeout_s'])
            stderr = result.stderr.decode('utf-8', 'replace')
            tokens = [int(x) for x in re.findall(r'^token (\d+)', stderr, re.M)]
            self.emit('asr_result', argv=argv, seconds=time.monotonic()-start, text=result.stdout.decode('utf-8', 'replace').strip(), tokens=tokens,
                      stop='eos' if tokens and tokens[-1] in (151643, 151645) else 'exit', exit_code=result.returncode,
                      error=None if result.returncode == 0 else stderr[-4096:])
        except subprocess.TimeoutExpired:
            self.emit('asr_result', seconds=time.monotonic()-start, text='', tokens=[], stop='timeout', error='child_timeout')
        except OSError as exc:
            self.emit('asr_result', text='', tokens=[], stop='spawn_error', error=str(exc))
        self.recover()

    def feed(self, pcm):
        if not pcm:
            return
        if len(pcm) % 2 or len(pcm) > CHUNK*2:
            raise ValueError('feed expects at most 100ms complete PCM16 samples')
        n = len(pcm)//2
        self.pos += n
        loud = energy(pcm) >= self.c['energy_threshold']
        if self.state == 'RECOVERING':
            self.elapsed += n
            self.quiet = 0 if loud else self.quiet+n
            if self.elapsed >= self.c['cooldown_ms']*16 and self.quiet >= self.c['recovery_silence_ms']*16:
                self.kws.reset()
                self.transition('LISTENING')
            return
        if self.state == 'ASR_BUSY':
            self.emit('busy_drop', samples=n)
            return
        if self.state == 'LISTENING':
            self.pre.append(pcm)
            hit = self.kws.feed(pcm)
            if not hit:
                return
            self.keyword = hit
            self.requests += 1
            self.request_id = uuid.uuid4().hex
            self.emit('keyword')
            if self.c['mode'] == 'commands':
                command = self.c.get('commands', {}).get(hit)
                self.emit('command' if command else 'unmapped_keyword', command_id=command, executed=False, asr_calls=self.asr_calls)
                feedback = self.c.get('feedback_wav')
                if command and feedback:
                    if self.microphone is not None:
                        self.microphone.set_busy(True)
                    self.transition('FEEDBACK_BUSY')
                    try:
                        result = subprocess.run(['aplay', '-q', '-D', 'default', feedback],
                                                shell=False, capture_output=True, timeout=3)
                        self.emit('feedback', exit_code=result.returncode,
                                  error=result.stderr.decode('utf-8', 'replace')[-1024:] or None)
                    except (OSError, subprocess.TimeoutExpired) as exc:
                        self.emit('feedback', error=str(exc))
                self.recover()
                return
            self.capture = bytearray(b''.join(self.pre))
            self.start = self.pos-len(self.capture)//2
            self.elapsed = self.quiet = self.voiced = 0
            self.transition('CAPTURING')
            return
        self.capture.extend(pcm)
        self.elapsed += n
        if loud:
            self.voiced += n
            self.quiet = 0
        else:
            self.quiet += n
        if len(self.capture)//2 >= self.c['max_capture_ms']*16:
            self.finish('max_duration')
        elif not self.voiced and self.elapsed >= self.c['speech_wait_ms']*16:
            self.finish('speech_wait_timeout')
        elif self.voiced and self.quiet >= self.c['tail_silence_ms']*16:
            self.finish('tail_silence')


def validate(c):
    if not isinstance(c, dict):
        raise ValueError('config must be an object')
    for key in ('pre_roll_ms', 'cooldown_ms', 'recovery_silence_ms', 'speech_wait_ms', 'tail_silence_ms', 'max_capture_ms', 'min_speech_ms'):
        value = c.get(key)
        if type(value) is not int or value < 100 or value % 100:
            raise ValueError(key + ' must be positive multiples of 100ms')
    if not 100 <= c['max_capture_ms'] <= 30000 or c['pre_roll_ms'] >= c['max_capture_ms']:
        raise ValueError('capture bound must be 100..30000ms and exceed pre-roll')
    if c['min_speech_ms'] > c['max_capture_ms'] - c['pre_roll_ms']:
        raise ValueError('min speech exceeds capture capacity')
    for key, upper in (('energy_threshold', 1), ('asr_timeout_s', 600)):
        value = c.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= upper:
            raise ValueError('invalid ' + key)
    if c.get('mode') not in ('commands', 'wake_asr'):
        raise ValueError('mode must be commands or wake_asr')
    backend = c.get('asr_backend', 'audio8')
    if backend not in ('audio8', 'paraformer'):
        raise ValueError('asr_backend must be audio8 or paraformer')
    paths = ('bridge', 'encoder', 'decoder', 'joiner', 'tokens', 'keywords', 'asr_cli', 'asr_model')
    if backend == 'audio8':
        paths += ('asr_map', 'asr_assets')
    for key in paths:
        if not isinstance(c.get(key), str) or not c[key] or '\x00' in c[key]:
            raise ValueError('invalid path: ' + key)
    if 'feedback_wav' in c and (c['mode'] != 'commands' or
            not isinstance(c['feedback_wav'], str) or not c['feedback_wav'] or '\x00' in c['feedback_wav']):
        raise ValueError('feedback_wav requires commands mode and a fixed WAV path')
    commands = c.get('commands', {})
    if not isinstance(commands, dict) or any(not isinstance(k, str) or not isinstance(v, str) or not re.fullmatch(r'[A-Za-z0-9_]{1,64}', v) for k, v in commands.items()):
        raise ValueError('commands must map keywords to safe event identifiers')


def main():
    p = argparse.ArgumentParser(description='Resident KWS + per-request ASR. File playback pauses while ASR runs; microphone continuously drains and discards busy audio. PCM16 mono 16kHz only.')
    p.add_argument('--config', required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--wav')
    group.add_argument('--mic', help='ALSA capture device, explicit opt-in')
    p.add_argument('--mic-seconds', type=int, default=15)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    c = json.loads(Path(args.config).read_text(encoding='utf-8'))
    validate(c)
    kws = Kws(c)
    try:
        v = Voice(c, kws, args.output)
        if args.wav:
            with wave.open(args.wav, 'rb') as f:
                if (f.getnchannels(), f.getsampwidth(), f.getframerate(), f.getcomptype()) != (1, 2, RATE, 'NONE'):
                    raise ValueError('requires PCM16 mono 16kHz WAV')
                while True:
                    pcm = f.readframes(CHUNK)
                    if not pcm:
                        break
                    v.feed(pcm)
        else:
            limit = 300 if c['mode'] == 'commands' else 30
            if not 1 <= args.mic_seconds <= limit:
                raise ValueError('mic session must be 1..%d seconds' % limit)
            listen_microphone(v, args.mic, args.mic_seconds)
        if v.state == 'CAPTURING':
            v.finish('eof')
        v.emit('summary', requests=v.requests, asr_calls=v.asr_calls, input_samples=v.pos)
    finally:
        kws.close()


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(json.dumps({'event': 'fatal', 'error': str(exc)}, ensure_ascii=False), flush=True)
        sys.exit(1)
