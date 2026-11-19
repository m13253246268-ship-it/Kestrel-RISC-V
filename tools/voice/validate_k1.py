#!/usr/bin/env python3
"""Run on K1: python3 validate_k1.py --kws-work DIR --asr-cli ELF --asr-model VQF --asr-map MAP --asr-assets ASSETS --command-wav WAV --output DIR"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import wave


def run(argv, **kwargs):
    r = subprocess.run([str(x) for x in argv], capture_output=True, **kwargs)
    return {'argv': [str(x) for x in argv], 'exit': r.returncode, 'stdout': r.stdout.decode('utf-8', 'replace'), 'stderr': r.stderr.decode('utf-8', 'replace')}


def main():
    p = argparse.ArgumentParser()
    for name in ('kws-work', 'asr-cli', 'asr-model', 'asr-map', 'asr-assets', 'command-wav', 'output'):
        p.add_argument('--'+name, required=True)
    a = p.parse_args()
    src = Path(__file__).resolve().parent
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    work = Path(a.kws_work)
    model = work/'sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01'
    lib = work/'sherpa-onnx-v2.0.1-linux-riscv64-spacemit-shared/lib'
    report = {'status': 'running', 'tests': [], 'limitations': ['demo_only official keywords; no FAR/FRR measurement', 'ASR reloads every request; not resident ASR', 'File stream is sequential replay, paused during ASR, not real-time continuous microphone experience', 'Energy endpoint is not learned VAD', 'Microphone is optional bounded acquisition with gaps; no microphone speech recognition claim']}
    def save():
        (out/'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    report['build'] = run(['sh', src/'build.sh', work, lib, out/'libvoice_kws.so'])
    if report['build']['exit']:
        save()
        raise RuntimeError(report['build'])
    report['devices'] = run(['arecord', '-l']) if shutil.which('arecord') else {'error': 'arecord unavailable'}
    report['hashes'] = {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in [src/'kws_bridge.c', src/'voice.py', src/'build.sh', out/'libvoice_kws.so', Path(a.asr_cli)]}
    c = json.loads((src/'demo.json').read_text())
    c['bridge'] = str(out/'libvoice_kws.so')
    for key in ('encoder', 'decoder', 'joiner', 'tokens', 'keywords'):
        c[key] = c[key].replace('MODEL', str(model))
    for key in ('asr_cli', 'asr_model', 'asr_map', 'asr_assets'):
        c[key] = getattr(a, key)
    def pcm(path):
        with wave.open(str(path), 'rb') as w:
            assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 16000)
            return w.readframes(w.getnframes())
    wake = pcm(model/'test_wavs/3.wav')
    command = pcm(a.command_wav)
    c['asr_timeout_s'] = 600
    c['tail_silence_ms'] = 2000
    silence = bytes(32000*4)
    def wav(name, data):
        path = out/(name+'.wav')
        with wave.open(str(path), 'wb') as w:
            w.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
            w.writeframes(data)
        return path
    combined = wav('continuous_two_rounds', (wake + bytes(3200) + command + silence)*2)
    report['fixture'] = 'Artificial concatenation: official KWS positive + 100ms silence + known welcome ASR WAV + 4s silence, twice; not natural spoken command'
    negative = wav('silence', bytes(32000*10))
    empty = wav('empty', b'')
    def test(name, audio, cfg):
        config = out/(name+'.json')
        config.write_text(json.dumps(cfg, ensure_ascii=False), encoding='utf-8')
        result = run([sys.executable, src/'voice.py', '--config', config, '--wav', audio, '--output', out/name], timeout=1300)
        result['name'] = name
        result['events'] = [json.loads(line) for line in result['stdout'].splitlines() if line.startswith('{')]
        report['tests'].append(result)
        save()
        return result
    real = test('real_two_rounds', combined, c)
    test('silence', negative, c)
    test('empty', empty, c)
    fixed = dict(c, mode='commands')
    fixed['commands'] = {line.split('@')[-1].strip(): 'demo_'+str(i) for i, line in enumerate(Path(c['keywords']).read_text().splitlines()) if '@' in line}
    test('fixed_commands', combined, fixed)
    captures = [e for e in real['events'] if e['event'] == 'capture']
    results = [e for e in real['events'] if e['event'] == 'asr_result']
    if captures:
        direct = run([c['asr_cli'], '--wav', c['asr_model'], c['asr_map'], c['asr_assets'], captures[0]['path'], '-'], env=dict(os.environ, VLLM_THREADS='8'), timeout=600)
        direct['tokens'] = [int(x) for x in re.findall(r'^token (\d+)', direct['stderr'], re.M)]
        direct['matches_orchestrated_tokens'] = bool(results) and direct['tokens'] == results[0]['tokens']
        direct['matches_orchestrated_text'] = bool(results) and direct['stdout'].strip() == results[0]['text']
        direct['stop'] = 'eos' if direct['tokens'] and direct['tokens'][-1] in (151643,151645) else 'exit'
        direct['matches_stop'] = bool(results) and direct['stop'] == results[0]['stop']
        direct['input_sha256'] = hashlib.sha256(Path(captures[0]['path']).read_bytes()).hexdigest()
        report['direct_same_capture'] = direct
    report['status'] = 'completed'
    report['checks'] = {
        'two_asr_results': len(results) == 2 and all(e.get('exit_code') == 0 for e in results),
        'command_audio_captured': len(captures) == 2 and all(e['start_sample'] <= i*(len(wake)+3200+len(command)+len(silence))//2+(len(wake)+3200)//2 and e['end_sample'] >= i*(len(wake)+3200+len(command)+len(silence))//2+(len(wake)+3200+len(command))//2 for i,e in enumerate(captures)),
        'same_capture_text_stop': report.get('direct_same_capture', {}).get('matches_orchestrated_text', False) and report.get('direct_same_capture', {}).get('matches_stop', False),
        'silence_no_trigger': not any(e['event'] == 'keyword' for e in report['tests'][1]['events']),
        'fixed_zero_asr': any(e['event'] == 'command' for e in report['tests'][3]['events']) and not any(e['event'] == 'asr_result' for e in report['tests'][3]['events']),
        'same_capture_tokens': report.get('direct_same_capture', {}).get('matches_orchestrated_tokens', False)
    }
    save()
    print(json.dumps(report['checks']))


if __name__ == '__main__':
    main()
