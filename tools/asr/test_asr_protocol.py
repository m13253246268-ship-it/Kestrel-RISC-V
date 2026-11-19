"""Fast simulated protocol runs. All generated artifacts stay in TEMP."""
import copy
import json
from pathlib import Path
import struct
import sys
import os
import tempfile
import unittest
from unittest.mock import patch

import asr_protocol as p


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='asr_protocol_')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        asset = self.root / 'asset.bin'
        asset.write_bytes(b'frozen fixture')
        ref = {'path': str(asset), 'sha256': p.sha(asset)}
        self.expected = {'text': 'test transcript', 'tokens': [10, 151645], 'decode': [1, 1, 2], 'projected_count': 1}
        self.config = {'schema_version': 1, 'protocol': {'warmups': 1, 'repeats': 5, 'interval_seconds': 3, 'cache': 'same_boot_warm'}, 'models': [{'id': 'model1', 'family': 'family1', 'model': ref, 'map': ref, 'wav_assets': ref, 'tiers': ['A'], 'expected': {'audio1': self.expected}}], 'audios': [{'id': 'audio1', 'audio': ref}], 'tiers': [{'id': 'A', 'binary': ref}], 'threads': [{'threads': 1, 'cpus': [0]}]}
        self.config_path = self.root / 'config.json'
        p.save(self.config_path, self.config)
        self.clock = 0
        self.calls = 0

    def tick(self):
        self.clock += 4
        return self.clock

    def snap(self, cpus):
        value = {'value': '1000', 'missing_reason': None}
        return {'time_ns': 1, 'monotonic_ns': 1, 'boot_id': 'boot1', 'loadavg': value, 'governor': {'0': value}, 'frequency': {'0': value}, 'thermal': {'zone': value}}

    def emit(self, command, env, directory, cpus):
        self.calls += 1
        self.assertEqual(command[:3], ['taskset', '-c', '0'])
        self.assertEqual(env['VLLM_THREADS'], '1')
        values = {stage: 1 for stage in p.LEAVES}
        values['encoder'] = 9
        values['total'] = 19
        lines = ['ASR_THREADS requested=1 actual=1', 'token 10', 'token 151645 stop']
        for stage, value in values.items():
            suffix = ' decode_steps=1 generated_tokens=1 selected_tokens=2 trace_includes_finite=1' if stage == 'total' else ''
            lines.append('ASR_PROFILE stage=%s seconds=%.9f%s' % (stage, value, suffix))
        (directory / 'log.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        (directory / 'text.txt').write_text('test transcript\n', encoding='utf-8')
        raw = bytearray(1560 + 512 * 4)
        for token in self.expected['tokens']:
            raw.extend(bytes((151936 + 8192) * 4))
            raw.extend(struct.pack('<I', token))
        (directory / 'trace.bin').write_bytes(raw)
        p.save(directory / 'affinity.json', {'observations': [{'tasks': {'123': {'cpus': [0]}}}]})
        return 0

    def run_fixture(self, name='run', execute=None, snapshot=None):
        out = self.root / name
        with patch.object(p, 'topology', return_value={'1': {'cpus': [0]}}), patch.object(p, 'boot_id', return_value='boot1'), patch.object(p, 'snapshot', side_effect=snapshot or self.snap), patch.object(p, 'execute', side_effect=execute or self.emit), patch.object(p.time, 'sleep'), patch.object(p.time, 'monotonic', side_effect=self.tick):
            code = p.run(self.config_path, out)
        return code, out

    def test_complete_warmup_excluded_and_no_overwrite(self):
        code, out = self.run_fixture()
        self.assertEqual(code, 0)
        self.assertEqual(self.calls, 6)
        result = p.summarize(out)
        self.assertTrue(result['integrity_passed'])
        self.assertTrue(result['stability_passed'])
        self.assertIsNone(result['environment_review_passed'])
        self.assertIsNone(result['performance_passed'])
        self.assertEqual(result['groups'][0]['statistics']['total']['values'], [19] * 5)
        with self.assertRaises(FileExistsError):
            self.run_fixture()
        self.assertEqual(len(list(out.glob('*/r*/trace.bin'))), 5)

    def test_failure_no_replacement(self):
        def fail(command, env, d, cpus):
            self.emit(command, env, d, cpus)
            return 1 if d.name == 'r2' else 0
        code, out = self.run_fixture(execute=fail)
        self.assertEqual(code, 1)
        self.assertEqual(self.calls, 3)
        self.assertFalse(p.summarize(out)['integrity_passed'])
        self.assertEqual(p.load(out / 'finish.json')['status'], 'failed')
        self.assertFalse((out / 'model1__audio1__A__t1' / 'r3').exists())

    def test_interrupt_and_unfinished(self):
        def interrupt(*args):
            raise KeyboardInterrupt()
        code, out = self.run_fixture(execute=interrupt)
        self.assertEqual(code, 1)
        self.assertEqual(p.summarize(out)['runner_status'], 'interrupted')
        (out / 'finish.json').unlink()
        self.assertFalse(p.summarize(out)['is_final'])
        self.assertFalse(p.summarize(out)['integrity_passed'])

    def test_boot_change_rejected(self):
        counter = [0]
        def snapshot(cpus):
            counter[0] += 1
            s = self.snap(cpus)
            if counter[0] >= 2:
                s['boot_id'] = 'boot2'
            return s
        code, out = self.run_fixture(snapshot=snapshot)
        self.assertEqual(code, 1)
        self.assertFalse(p.summarize(out)['integrity_passed'])

    def test_parser_rejects_corruption(self):
        _, out = self.run_fixture()
        d = out / 'model1__audio1__A__t1' / 'r1'
        log = (d / 'log.txt').read_text()
        for corrupted in [log + 'ASR_PROFILE stage=load seconds=1\n', log.replace('stage=load seconds=1.000000000', 'stage=load seconds=nan'), log.replace('stage=load seconds=1.000000000', 'stage=load seconds=-1'), log.replace('stage=total seconds=19', 'stage=total seconds=29'), log.replace('actual=1', 'actual=8'), log.replace('generated_tokens=1', 'generated_tokens=2'), log.replace('token 10\n', 'token 11\n')]:
            (d / 'log.txt').write_text(corrupted)
            with self.assertRaises(ValueError):
                p.parse_sample(d, self.expected, 1)
        (d / 'log.txt').write_text(log)
        raw = (d / 'trace.bin').read_bytes()
        (d / 'trace.bin').write_bytes(raw[:-1])
        with self.assertRaises(ValueError):
            p.parse_sample(d, self.expected, 1)
        (d / 'trace.bin').write_bytes(raw[:-4] + struct.pack('<I', 5))
        with self.assertRaises(ValueError):
            p.parse_sample(d, self.expected, 1)

    def test_real_child_process_log_and_explicit_environment(self):
        directory = self.root / 'child'
        directory.mkdir()
        env = os.environ.copy()
        env['VLLM_THREADS'] = '8'
        code = p.execute([sys.executable, '-c', 'import os; print(os.environ["VLLM_THREADS"])'], env, directory, [0])
        self.assertEqual(code, 0)
        self.assertEqual((directory / 'log.txt').read_text().strip(), '8')
        self.assertTrue((directory / 'affinity.json').exists())

    def test_freeze_and_hash_rejection(self):
        frozen = self.root / 'frozen.json'
        p.freeze(self.config_path, frozen)
        p.validate(p.load(frozen))
        with self.assertRaises(FileExistsError):
            p.freeze(self.config_path, frozen)
        (self.root / 'asset.bin').write_bytes(b'changed')
        with self.assertRaises(ValueError):
            p.validate(p.load(frozen))

    def test_population_statistics_and_thresholds(self):
        result = p.stats([1, 2, 3, 4, 5])
        self.assertEqual(result['median'], 3)
        self.assertAlmostEqual(result['population_rsd'], (2 ** .5) / 3)
        _, out = self.run_fixture()
        d = out / 'model1__audio1__A__t1' / 'r5' / 'log.txt'
        text = d.read_text().replace('seconds=1.000000000', 'seconds=1.040000000').replace('seconds=19.000000000', 'seconds=19.760000000')
        d.write_text(text)
        self.assertFalse(p.summarize(out)['stability_passed'])
        self.assertEqual(p.summarize(out)['groups'][0]['stability_limit'], .03)

    def test_stage_regression_and_dc_separate_threshold(self):
        self.config['tiers'] += [{'id': x, 'binary': self.config['tiers'][0]['binary']} for x in ('C', 'D')]
        self.config['models'][0]['tiers'] = ['A', 'C', 'D']
        self.config_path.write_text(json.dumps(self.config))
        _, out = self.run_fixture()
        for d in out.glob('model1__audio1__D__t1/r*/log.txt'):
            text = d.read_text().replace('stage=load seconds=1.000000000', 'stage=load seconds=1.060000000')
            d.write_text(text)
        result = p.summarize(out)
        row = result['groups'][2]
        self.assertIn('load', row['regressions_vs_A'])
        self.assertIn('load', row['regressions_vs_C'])
        self.assertFalse(row['regression_passed'])
        # Tiny buckets: >5% alone is D/C evidence but not the generic >10ms gate.
        for d in out.glob('*/r*/log.txt'):
            text = d.read_text().replace('stage=unclassified seconds=1.000000000', 'stage=unclassified seconds=0.001000000')
            text = text.replace('stage=total seconds=19.000000000', 'stage=total seconds=18.001000000')
            if '__D__' in str(d):
                text = text.replace('stage=unclassified seconds=0.001000000', 'stage=unclassified seconds=0.002000000')
            d.write_text(text)
        row = p.summarize(out)['groups'][2]
        self.assertNotIn('unclassified', row['regressions_vs_A'])
        self.assertIn('unclassified', row['regressions_vs_C'])

    def test_eight_thread_registered_limit(self):
        self.config['threads'] = [{'threads': 8, 'cpus': list(range(8))}]
        self.config_path.write_text(json.dumps(self.config))
        def emit(command, env, d, cpus):
            self.emit(['taskset', '-c', '0'], {'VLLM_THREADS': '1'}, d, cpus)
            log = d / 'log.txt'
            text = log.read_text().replace('requested=1 actual=1', 'requested=8 actual=8')
            if d.name == 'r5':
                text = text.replace('seconds=1.000000000', 'seconds=1.040000000').replace('seconds=19.000000000', 'seconds=19.760000000')
            log.write_text(text)
            return 0
        _, out = self.run_fixture(execute=emit)
        result = p.summarize(out)
        self.assertTrue(result['integrity_passed'])
        self.assertTrue(result['stability_passed'])
        self.assertEqual(result['groups'][0]['stability_limit'], .05)

    def test_cluster_local_core_ids(self):
        c = {'threads': [{'threads': 8, 'cpus': list(range(8))}]}
        def read(path):
            cpu = int(path.parent.parent.name[3:])
            return {'value': str(cpu) if path.name == 'thread_siblings_list' else str(cpu % 4) if path.name == 'core_id' else '0'}
        with patch.object(p.sys, 'platform', 'linux'), patch.object(p.os, 'sched_getaffinity', return_value=set(range(8)), create=True), patch.object(p, 'read_value', side_effect=read):
            self.assertEqual(len(p.topology(c)['8']['physical_cores']), 8)
        def shared(path):
            value = read(path)
            if path.name == 'thread_siblings_list':
                value['value'] = '0-7'
            return value
        with patch.object(p.sys, 'platform', 'linux'), patch.object(p.os, 'sched_getaffinity', return_value=set(range(8)), create=True), patch.object(p, 'read_value', side_effect=shared):
            with self.assertRaises(ValueError):
                p.topology(c)

    def test_affinity_all_engine_tids_and_transition(self):
        launcher = {'phase': 'launcher', 'tasks': {'1': {'cpus': [0, 1]}}}
        good = {'phase': 'engine', 'tasks': {'1': {'cpus': [0]}, '2': {'cpus': [0]}}}
        bad = {'phase': 'engine', 'tasks': {'1': {'cpus': [0]}, '2': {'cpus': [1]}}}
        self.assertTrue(p.affinity_review([launcher, good], [0]))
        self.assertFalse(p.affinity_review([good, bad, good], [0]))
        self.assertIsNone(p.affinity_review([launcher], [0]))
        self.assertIsNone(p.affinity_review([{'phase': 'engine', 'tasks': {'1': {'missing_reason': 'gone'}}}], [0]))

    def test_environment_and_tristate(self):
        policy = {'frequency': {'0': '1000'}, 'governor': {'0': '1000'}, 'thermal_limits': {'zone': 2000}}
        snap = self.snap([0])
        self.assertTrue(p.environment_review([snap], [0], 'boot1', policy)[0])
        self.assertIsNone(p.environment_review([snap], [0], 'boot1', {})[0])
        self.assertFalse(p.environment_review([snap], [0], 'boot2', {})[0])
        snap['thermal']['zone'] = {'value': None}
        self.assertIsNone(p.environment_review([snap], [0], 'boot1', policy)[0])
        self.assertFalse(p.tri_all([True, None, False]))
        self.assertIsNone(p.tri_all([True, None]))
        self.assertTrue(p.tri_all([True, True]))

    def test_multi_model_config_and_bad_cpu(self):
        c = copy.deepcopy(self.config)
        second = copy.deepcopy(c['models'][0])
        second.update(id='model2', family='family2')
        c['models'].append(second)
        p.validate(c)
        self.assertEqual(len(list(p.groups(c))), 2)
        c['threads'] = [{'threads': 8, 'cpus': [0] * 8}]
        with self.assertRaises(ValueError):
            p.validate(c)


if __name__ == '__main__':
    unittest.main()
