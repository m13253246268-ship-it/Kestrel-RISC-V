import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import voice

SILENCE = bytes(3200)
SPEECH = b'\x00\x20'*1600

class FakeKws:
    def __init__(self):
        self.hit = 'demo'
        self.resets = 0
    def feed(self, pcm):
        result, self.hit = self.hit, ''
        return result
    def reset(self):
        self.resets += 1

class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.log = io.StringIO()
        self.redirect = contextlib.redirect_stdout(self.log)
        self.redirect.__enter__()
        self.addCleanup(self.redirect.__exit__, None, None, None)
        self.c = json.loads(Path(__file__).with_name('demo.json').read_text(encoding='utf-8'))
        self.kws = FakeKws()
    def make(self):
        voice.validate(self.c)
        return voice.Voice(self.c, self.kws, self.tmp.name)
    def events(self):
        return [json.loads(x) for x in self.log.getvalue().splitlines()]
    def capture(self, v):
        v.feed(SPEECH)
        v.feed(SPEECH)
        v.feed(SPEECH)
        for _ in range(8): v.feed(SILENCE)
    def test_paraformer_backend(self):
        self.c['asr_backend'] = 'paraformer'
        del self.c['asr_map'], self.c['asr_assets']
        result = subprocess.CompletedProcess([], 0, '你好\n'.encode(), b'')
        with patch('voice.subprocess.run', return_value=result) as run:
            self.capture(self.make())
        argv = run.call_args.args[0]
        self.assertEqual(argv[:2], [self.c['asr_cli'], self.c['asr_model']])
        self.assertTrue(argv[2].endswith('.wav'))
        self.assertEqual(len(argv), 3)
        self.assertEqual(run.call_args.kwargs['env']['OMP_NUM_THREADS'], '1')
        self.assertFalse(run.call_args.kwargs['shell'])
        self.assertFalse(next(e for e in self.events() if e['event'] == 'asr_start')['streaming'])
        self.assertEqual(next(e for e in self.events() if e['event'] == 'asr_result')['text'], '你好')

    def test_invalid_backend(self):
        self.c['asr_backend'] = 'invalid'
        with self.assertRaises(ValueError):
            voice.validate(self.c)

    def test_shell_text_is_only_data(self):
        result = subprocess.CompletedProcess([], 0, b'$(touch bad); & echo nope', b'token 151645\n')
        with patch('voice.subprocess.run', return_value=result) as run:
            self.capture(self.make())
        self.assertFalse(run.call_args.kwargs['shell'])
        self.assertEqual(run.call_args.kwargs['env']['VLLM_THREADS'], '8')
        self.assertNotIn('$(touch bad)', str(run.call_args.args))
        self.assertEqual([e for e in self.events() if e['event']=='asr_result'][0]['text'], result.stdout.decode())
    def test_timeout_spawn_exit_recover(self):
        for outcome in (subprocess.TimeoutExpired('stub', 1), OSError('stub'), subprocess.CompletedProcess([], 9, b'', b'failure')):
            self.kws.hit = 'demo'
            v = self.make()
            with patch('voice.subprocess.run', side_effect=outcome if isinstance(outcome, Exception) else None, return_value=outcome):
                self.capture(v)
            self.assertEqual(v.state, 'RECOVERING')
            self.assertFalse(v.pre)
            for _ in range(10): v.feed(SILENCE)
            self.assertEqual(v.state, 'LISTENING')
    def test_command_no_asr(self):
        self.c.update(mode='commands', commands={'demo': 'demo_event'})
        with patch('voice.subprocess.run') as run:
            v = self.make(); v.feed(SPEECH)
        run.assert_not_called()
        event = next(e for e in self.events() if e['event']=='command')
        self.assertFalse(event['executed'])
    def test_feedback_resumes_without_asr_or_recording(self):
        from unittest.mock import Mock
        self.c.update(mode='commands', commands={'demo': 'wake'}, feedback_wav='/fixed/tone.wav')
        v = self.make()
        mic = v.microphone = Mock()
        with patch('voice.subprocess.run', return_value=subprocess.CompletedProcess([], 0, b'', b'')) as run:
            v.feed(SPEECH)
            self.assertEqual(run.call_args.args[0], ['aplay', '-q', '-D', 'default', '/fixed/tone.wav'])
            self.assertFalse(run.call_args.kwargs['shell'])
            for _ in range(10): v.feed(SILENCE)
            self.kws.hit = 'demo'
            v.feed(SPEECH)
        self.assertEqual(v.requests, 2)
        self.assertEqual(v.asr_calls, 0)
        self.assertEqual([x.args for x in mic.set_busy.call_args_list], [(True,), (False,), (True,), (False,)])
        self.assertEqual(list(Path(self.tmp.name).glob('*.wav')), [])

    def test_mic_duration_bounds(self):
        for mode, seconds, allowed in [('commands', 300, True), ('commands', 301, False), ('wake_asr', 31, False)]:
            self.c['mode'] = mode
            config = Path(self.tmp.name)/'config.json'
            config.write_text(json.dumps(self.c), encoding='utf-8')
            argv = ['voice', '--config', str(config), '--mic', 'mock', '--mic-seconds', str(seconds), '--output', self.tmp.name]
            with patch('sys.argv', argv), patch('voice.Kws'), patch('voice.listen_microphone') as listen:
                if allowed:
                    voice.main()
                    self.assertEqual(listen.call_args.args[2], 300)
                else:
                    with self.assertRaises(ValueError): voice.main()
                    listen.assert_not_called()

    def test_empty_short_and_onset_timeout(self):
        with patch('voice.subprocess.run') as run:
            v = self.make(); v.feed(SILENCE)
            for _ in range(30): v.feed(SILENCE)
            self.assertEqual(v.state, 'RECOVERING')
            self.kws.hit='demo'; v=self.make(); v.feed(SPEECH);v.feed(SPEECH);v.finish('eof')
        run.assert_not_called()
    def test_max_bound_and_busy_drop(self):
        self.c['max_capture_ms']=1000
        v=self.make()
        with patch('voice.subprocess.run', return_value=subprocess.CompletedProcess([],0,b'',b'')):
            for _ in range(11): v.feed(SPEECH)
        event=next(e for e in self.events() if e['event']=='capture')
        self.assertLessEqual(event['samples'],16000)
        v.state='ASR_BUSY'; v.feed(SPEECH)
        self.assertEqual(v.requests,1)
    def test_invalid_config_and_pcm(self):
        for key,value in [('mode','asr'),('max_capture_ms',30100),('energy_threshold',float('nan')),('pre_roll_ms',True),('commands',{'demo':'echo ; bad'}),('asr_timeout_s',0)]:
            c=dict(self.c);c[key]=value
            with self.assertRaises(ValueError): voice.validate(c)
        v=self.make()
        with self.assertRaises(ValueError):v.feed(b'1')
        with self.assertRaises(ValueError):v.feed(bytes(3202))
    def test_two_wakes_reset_without_model_reload(self):
        v=self.make()
        with patch('voice.subprocess.run',return_value=subprocess.CompletedProcess([],0,b'ok',b'token 151645 stop\n')) as run:
            self.capture(v)
            for _ in range(10):v.feed(SILENCE)
            self.kws.hit='demo';self.capture(v)
        self.assertEqual(run.call_count,2)
        self.assertEqual(v.requests,2)
        self.assertGreaterEqual(self.kws.resets,3)
        self.assertEqual(sum(e['event']=='ready' for e in self.events()),1)
    def test_asr_marks_busy_then_flushes_recovery(self):
        from unittest.mock import Mock
        mic = Mock()
        v = self.make()
        v.microphone = mic
        def asr(*args, **kwargs):
            mic.set_busy.assert_called_once_with(True)
            self.assertEqual(v.state, 'ASR_BUSY')
            return subprocess.CompletedProcess([], 0, b'ok', b'')
        with patch('voice.subprocess.run', side_effect=asr):
            self.capture(v)
        self.assertEqual([c.args for c in mic.set_busy.call_args_list], [(True,), (False,)])
        mic.close.assert_not_called()
        self.assertEqual(v.state, 'RECOVERING')
        self.assertFalse(v.pre)

    def test_capture_write_error(self):
        v=self.make();v.feed(SPEECH);v.feed(SPEECH);v.feed(SPEECH)
        with patch('voice.wave.open',side_effect=OSError('disk')),patch('voice.subprocess.run') as run:
            v.finish('eof')
        run.assert_not_called();self.assertEqual(v.state,'RECOVERING')

class PipeTests(unittest.TestCase):
    def child(self, script):
        import sys
        spawn = subprocess.Popen
        def factory(*args, **kwargs):
            return spawn([sys.executable, '-u', '-c', script], stdout=subprocess.PIPE,
                         stdin=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        return patch('voice.subprocess.Popen', side_effect=factory)

    def wait(self, mic, predicate):
        with mic.condition:
            self.assertTrue(mic.condition.wait_for(predicate, 3))

    def test_short_reads_single_spawn_clean_eof(self):
        script = "import os,time; os.write(1,b'a'); time.sleep(.05); os.write(1,b'b'*3199)"
        with self.child(script) as spawn:
            mic = voice.Microphone('mock')
            self.addCleanup(mic.close)
            self.assertEqual(mic.read(3), b'a'+b'b'*3199)
            self.assertEqual(mic.read(3), b'')
            self.assertEqual(spawn.call_count, 1)
            self.assertNotIn('-s', spawn.call_args.args[0])
            mic.close()
            self.assertFalse(mic.reader.is_alive())
            self.assertIsNotNone(mic.process.poll())
            mic.process.stdin.close(); mic.process.stderr.close()

    def test_partial_eof_and_nonzero(self):
        for script, message in [("import os; os.write(1,b'abc')", 'partial microphone'),
                                ("raise SystemExit(7)", 'code 7')]:
            with self.child(script):
                mic = voice.Microphone('mock')
                try:
                    with self.assertRaisesRegex(OSError, message): mic.read(3)
                finally:
                    mic.close()
                    mic.process.stdin.close(); mic.process.stderr.close()

    def test_busy_drain_generation_flush_bounded_queue_shutdown(self):
        script = "import os\nwhile True:\n c=os.read(0,1)\n if not c: break\n os.write(1,c*3200)\n"
        with self.child(script) as spawn:
            mic = voice.Microphone('mock')
            mic.start()
            try:
                mic.process.stdin.write(b'AAAAAA')
                self.wait(mic, lambda: mic.dropped >= 4)
                with mic.condition: self.assertEqual(len(mic.queue), 2)
                mic.set_busy(True)
                before = mic.dropped
                mic.process.stdin.write(b'B'*40)
                self.wait(mic, lambda: mic.dropped >= before+40)
                with mic.condition: self.assertFalse(mic.queue)
                mic.set_busy(False)
                # The read already in flight belongs to the old generation.
                mic.process.stdin.write(b'CF')
                self.assertEqual(mic.read(3), b'F'*3200)
                self.assertEqual(spawn.call_count, 1)
            finally:
                mic.close()
                mic.process.stdin.close(); mic.process.stderr.close()
            self.assertFalse(mic.reader.is_alive())
            self.assertIsNotNone(mic.process.poll())
            mic.close()
            with self.assertRaises(RuntimeError): mic.read(1)

    def test_no_spawn_until_explicit_read_and_spawn_failure(self):
        with patch('voice.subprocess.Popen', side_effect=OSError('no device')) as spawn:
            mic = voice.Microphone('mock')
            spawn.assert_not_called()
            with self.assertRaisesRegex(OSError, 'no device'): mic.read(1)
            mic.close()
            self.assertEqual(spawn.call_count, 1)

    def test_timeout_releases_blocked_child(self):
        with self.child("import os; os.read(0,1)"):
            mic = voice.Microphone('mock')
            try:
                with self.assertRaises(TimeoutError): mic.read(.05)
            finally:
                mic.close()
                mic.process.stdin.close(); mic.process.stderr.close()
            self.assertFalse(mic.reader.is_alive())

if __name__=='__main__': unittest.main()
