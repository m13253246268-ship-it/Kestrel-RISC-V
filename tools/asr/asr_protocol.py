"""R1/R2 protocol v1: freeze, run and summarize; Python standard library only."""
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import struct
import subprocess
import sys
import time

LEAVES = 'unclassified load assets features encoder_other conv0 conv1 conv2 conv_out norm_qkv attention outprojection ffn tail_mlp projector prefill decode trace text'.split()


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def assets(config):
    for model in config['models']:
        yield from (model[k] for k in ('model', 'map', 'wav_assets'))
    yield from (audio['audio'] for audio in config['audios'])
    yield from (tier['binary'] for tier in config['tiers'])


def validate(c, verify=True):
    require(c['schema_version'] == 1, 'unsupported schema_version')
    require(c['protocol'] == {'warmups': 1, 'repeats': 5, 'interval_seconds': 3, 'cache': 'same_boot_warm'}, 'protocol must be same boot, one warmup, five repeats, 3s interval')
    for key in ('models', 'audios', 'tiers'):
        ids = [item['id'] for item in c[key]]
        require(ids and len(ids) == len(set(ids)), 'empty/duplicate ' + key)
        require(all(re.fullmatch(r'[A-Za-z0-9_-]+', x) and '__' not in x for x in ids), 'unsafe ID or reserved double underscore')
    require(len({x['threads'] for x in c['threads']}) == len(c['threads']) and c['threads'], 'duplicate/empty threads')
    for t in c['threads']:
        require(t['threads'] in (1, 8), 'v1 supports registered 1/8 threads only')
        require(len(t['cpus']) == t['threads'] and len(set(t['cpus'])) == len(t['cpus']) and all(type(x) is int and x >= 0 for x in t['cpus']), 'invalid CPU set')
    tiers = {t['id'] for t in c['tiers']}
    audios = {a['id'] for a in c['audios']}
    for m in c['models']:
        require(m.get('family') and m['tiers'] and len(m['tiers']) == len(set(m['tiers'])) and set(m['tiers']) <= tiers and 'A' in m['tiers'], 'model family/tiers/A required')
        require(set(m['expected']) == audios, 'expectations must cover every audio')
        for e in m['expected'].values():
            require(isinstance(e['text'], str) and e['tokens'] and all(type(x) is int and x >= 0 for x in e['tokens']), 'invalid expected text/tokens')
            require(len(e['decode']) == 3 and all(type(x) is int and x >= 0 for x in e['decode']) and e['decode'][2] == len(e['tokens']), 'invalid decode counts')
            require(type(e['projected_count']) is int and e['projected_count'] > 0, 'projected_count required for trace layout')
    for a in assets(c):
        require(Path(a['path']).is_absolute() and re.fullmatch('[0-9a-f]{64}', a['sha256']), 'absolute path and SHA256 required')
        if verify:
            require(sha(a['path']) == a['sha256'], 'asset SHA mismatch: ' + a['path'])


def freeze(source, target):
    c = load(source)
    for a in assets(c):
        a['path'] = str(Path(a['path']).resolve())
        a['sha256'] = sha(a['path'])
    validate(c)
    save(target, c)


def read_value(path):
    try:
        return {'value': Path(path).read_text().strip(), 'missing_reason': None}
    except OSError as exc:
        return {'value': None, 'missing_reason': str(exc)}


def boot_id():
    value = read_value('/proc/sys/kernel/random/boot_id')
    require(value['value'], 'boot_id unavailable: ' + str(value['missing_reason']))
    return value['value']


def snapshot(cpus):
    result = {'time_ns': time.time_ns(), 'monotonic_ns': time.monotonic_ns(), 'boot_id': boot_id(), 'loadavg': read_value('/proc/loadavg')}
    for name, field in [('governor', 'scaling_governor'), ('frequency', 'scaling_cur_freq')]:
        result[name] = {str(cpu): read_value('/sys/devices/system/cpu/cpu%d/cpufreq/%s' % (cpu, field)) for cpu in cpus}
    paths = list(Path('/sys/class/thermal').glob('thermal_zone*/temp'))
    result['thermal'] = {str(p): read_value(p) for p in paths}
    result['thermal_trip_points'] = {str(p): read_value(p) for zone in paths for p in zone.parent.glob('trip_point_*')}
    result['thermal_missing_reason'] = None if paths else 'no readable thermal zone paths found'
    return result


def topology(c):
    require(sys.platform.startswith('linux'), 'real run requires Linux and taskset')
    allowed = os.sched_getaffinity(0)
    result = {}
    for t in c['threads']:
        cores = []
        for cpu in t['cpus']:
            require(cpu in allowed, 'CPU outside allowed affinity: ' + str(cpu))
            root = Path('/sys/devices/system/cpu/cpu%d/topology' % cpu)
            pair = tuple(read_value(root / k)['value'] for k in ('physical_package_id', 'core_id'))
            require(None not in pair, 'CPU topology unavailable')
            siblings = read_value(root / 'thread_siblings_list')['value']
            require(siblings is not None, 'thread sibling topology unavailable')
            members = set()
            for part in siblings.split(','):
                bounds = list(map(int, part.split('-')))
                members.update(range(bounds[0], bounds[-1] + 1))
            require(cpu in members, 'invalid thread sibling topology')
            require(not any(set(core['thread_siblings']) & members for core in cores), 'CPU set shares a physical core')
            cores.append({'package_core': pair, 'thread_siblings': sorted(members)})
        result[str(t['threads'])] = {'cpus': t['cpus'], 'physical_cores': cores}
    return result


def parse_sample(directory, expected, threads):
    d = Path(directory)
    text = (d / 'log.txt').read_text(encoding='utf-8', errors='replace')
    pairs = re.findall(r'^ASR_PROFILE stage=(\w+) seconds=(\S+)', text, re.M)
    require(len(pairs) == 21 and len(dict(pairs)) == 21, 'missing/duplicate profile stages')
    stages = {k: float(v) for k, v in pairs}
    require(set(stages) == set(LEAVES) | {'encoder', 'total'}, 'unexpected profile stages')
    require(all(math.isfinite(v) and v >= 0 for v in stages.values()) and stages['total'] > 0, 'nonfinite/negative timing')
    closure = abs(sum(stages[k] for k in LEAVES) - stages['total']) / stages['total']
    require(closure <= .02, '19-leaf closure exceeds 2%')
    banners = re.findall(r'ASR_THREADS requested=(\d+) actual=(\d+)', text)
    require(banners == [(str(threads), str(threads))], 'thread banner mismatch')
    counts = re.findall(r'decode_steps=(\d+) generated_tokens=(\d+) selected_tokens=(\d+)', text)
    require(len(counts) == 1 and list(map(int, counts[0])) == expected['decode'], 'decode count mismatch')
    tokens = list(map(int, re.findall(r'^token (\d+)(?: stop)?$', text, re.M)))
    require(tokens == expected['tokens'], 'log token mismatch')
    require((d / 'text.txt').read_text(encoding='utf-8') == expected['text'] + '\n', 'text mismatch')
    raw = (d / 'trace.bin').read_bytes()
    require(len(raw) >= 1560, 'truncated trace mask')
    mask = struct.unpack_from('<390I', raw)
    require(all(x in (0, 1) for x in mask), 'invalid trace mask')
    offset = 1560 + (sum(mask) * 1024 + expected['projected_count'] * 512) * 4
    stride = (151936 + 8192) * 4 + 4
    require(len(raw) == offset + stride * len(tokens), 'trace length mismatch')
    trace_tokens = [struct.unpack_from('<I', raw, offset + stride * (i + 1) - 4)[0] for i in range(len(tokens))]
    require(trace_tokens == tokens, 'trace token mismatch')
    # All floating fields are validated, excluding the integer mask/token fields.
    segments = [(1560, offset)] + [(offset + i * stride, offset + (i + 1) * stride - 4) for i in range(len(tokens))]
    require(all(math.isfinite(x[0]) for a, b in segments for x in struct.iter_unpack('<f', raw[a:b])), 'nonfinite trace')
    return {'stages': stages, 'closure_relative_error': closure, 'tokens': tokens, 'thread_banner': banners[0], 'trace_sha256': sha(d / 'trace.bin'), 'text_sha256': sha(d / 'text.txt')}


def tri_all(values):
    values = list(values)
    return False if False in values else (None if None in values or not values else True)


def affinity_review(observations, cpus):
    states = []
    for obs in observations:
        if obs.get('phase') == 'launcher':
            continue
        if obs.get('phase') != 'engine' or not obs.get('tasks'):
            states.append(None)
            continue
        states.extend(set(task['cpus']) == set(cpus) if 'cpus' in task else None
                      for task in obs['tasks'].values())
    return tri_all(states)


def environment_review(observations, cpus, initial_boot, policy):
    checks, reasons = [], []
    if not observations:
        return None, ['periodic environment observations unavailable']
    for obs in observations:
        checks.append(obs.get('boot_id') == initial_boot if obs.get('boot_id') else None)
        for kind in ('frequency', 'governor'):
            readings = obs.get(kind, {})
            for cpu in cpus:
                value = readings.get(str(cpu), {}).get('value')
                expected = policy.get(kind, {}).get(str(cpu))
                checks.append(None if value is None or expected is None else str(value) == str(expected))
        limits = policy.get('thermal_limits', {})
        thermal = obs.get('thermal', {})
        if not limits or not thermal:
            checks.append(None)
        for path in set(limits) | set(thermal):
            value = thermal.get(path, {}).get('value')
            limit = limits.get(path)
            try:
                checks.append(float(value) < float(limit) if value is not None and limit is not None else None)
            except (ValueError, TypeError):
                checks.append(None)
    if None in checks:
        reasons.append('missing sensor reading or frozen sysfs-based threshold/policy; unknown')
    if False in checks:
        reasons.append('observed boot/frequency/governor/temperature policy violation')
    return tri_all(checks), reasons


def execute(command, env, directory, cpus):
    observations = []
    environment = []
    next_snapshot = 0
    engine = str(Path(command[3] if command[0] == 'taskset' else command[0]).resolve())
    with (directory / 'log.txt').open('xb') as log:
        p = subprocess.Popen(command, stdout=log, stderr=log, env=env)
        try:
            while p.poll() is None:
                tasks = {}
                for task in Path('/proc/%d/task' % p.pid).glob('*'):
                    try:
                        tasks[task.name] = {'cpus': sorted(os.sched_getaffinity(int(task.name)))}
                    except OSError as exc:
                        tasks[task.name] = {'missing_reason': str(exc)}
                try:
                    executable = os.readlink('/proc/%d/exe' % p.pid)
                    phase = 'engine' if executable == engine else ('launcher' if Path(executable).name == 'taskset' else 'unknown')
                except OSError:
                    executable, phase = None, 'unknown'
                observations.append({'time_ns': time.time_ns(), 'tasks': tasks, 'executable': executable, 'phase': phase})
                if time.monotonic() >= next_snapshot:
                    try:
                        environment.append(snapshot(cpus))
                    except Exception as exc:
                        environment.append({'time_ns': time.time_ns(), 'missing_reason': str(exc)})
                    next_snapshot = time.monotonic() + 1
                time.sleep(.1)
        except BaseException:
            p.terminate()
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
            raise
        finally:
            save(directory / 'affinity.json', {'pid': p.pid, 'observations': observations, 'missing_reason': None if observations else 'process exited before observation'})
            save(directory / 'environment.json', {'interval_seconds': 1, 'observations': environment})
    return p.returncode


def groups(c):
    for model, audio, thread in itertools.product(c['models'], c['audios'], c['threads']):
        for tier_id in model['tiers']:
            tier = next(t for t in c['tiers'] if t['id'] == tier_id)
            label = '%s__%s__%s__t%d' % (model['id'], audio['id'], tier_id, thread['threads'])
            yield label, model, audio, tier, thread


def run(config_path, output):
    c = load(config_path)
    validate(c)
    topo = topology(c)
    initial_boot = boot_id()
    out = Path(output)
    out.mkdir(parents=False, exist_ok=False)
    save(out / 'config.json', c)
    save(out / 'start.json', {'status': 'running', 'boot_id': initial_boot, 'topology': topo, 'config_sha256': sha(out / 'config.json'), 'time_ns': time.time_ns()})
    failed = False
    last_end = None
    try:
        for label, m, a, tier, t in groups(c):
            group = out / label
            group.mkdir()
            for repeat in range(6):
                if last_end is not None:
                    time.sleep(max(0, 3 - (time.monotonic() - last_end)))
                d = group / ('warmup' if repeat == 0 else 'r%d' % repeat)
                d.mkdir()
                record = {'label': label + '__' + d.name, 'repeat': repeat, 'formal': repeat != 0, 'exit_code': None, 'passed': False}
                try:
                    pre = snapshot(t['cpus'])
                    save(d / 'before.json', pre)
                    require(pre['boot_id'] == initial_boot, 'boot changed before sample')
                    command = ['taskset', '-c', ','.join(map(str, t['cpus'])), tier['binary']['path'], '--wav', m['model']['path'], m['map']['path'], m['wav_assets']['path'], a['audio']['path'], str((d / 'text.txt').resolve()), str((d / 'trace.bin').resolve())]
                    env = {k: v for k, v in os.environ.items() if not k.startswith(('VLLM_', 'OMP_', 'GOMP_', 'KMP_'))}
                    explicit = {'VLLM_THREADS': str(t['threads']), 'VLLM_ASR_PROFILE': '1', 'OMP_NUM_THREADS': str(t['threads']), 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
                    env.update(explicit)
                    record.update(command=command, environment=explicit, start_ns=time.time_ns(), start_monotonic=time.monotonic())
                    record['exit_code'] = execute(command, env, d, t['cpus'])
                    require(record['exit_code'] == 0, 'subprocess failed')
                    affinity_path = d / 'affinity.json'
                    if affinity_path.exists():
                        require(affinity_review(load(affinity_path)['observations'], t['cpus']) is not False, 'observed engine affinity violation')
                    environment_path = d / 'environment.json'
                    if environment_path.exists():
                        state, reasons = environment_review(load(environment_path)['observations'], t['cpus'], initial_boot, c.get('environment_policy', {}))
                        require(state is not False, '; '.join(reasons))
                    record['process_end_ns'] = time.time_ns()
                    record['validation'] = parse_sample(d, m['expected'][a['id']], t['threads'])
                    record['passed'] = True
                except (Exception, KeyboardInterrupt) as exc:
                    record['error'] = str(exc) or type(exc).__name__
                    raise
                finally:
                    last_end = time.monotonic()
                    record.update(end_ns=time.time_ns(), end_monotonic=last_end)
                    try:
                        post = snapshot(t['cpus'])
                        save(d / 'after.json', post)
                        if post['boot_id'] != initial_boot:
                            record.update(passed=False, error='boot changed after sample')
                    except Exception as exc:
                        record.update(passed=False, error='post snapshot: ' + str(exc))
                    save(d / 'execution.json', record)
                if not record['passed']:
                    raise ValueError(record['error'])
        validate(c)  # Detect inputs changed during the matrix.
    except (Exception, KeyboardInterrupt) as exc:
        failed = True
        save(out / 'finish.json', {'status': 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', 'error': str(exc), 'time_ns': time.time_ns()})
    else:
        save(out / 'finish.json', {'status': 'complete', 'time_ns': time.time_ns()})
    save(out / 'results.json', summarize(out))
    return 1 if failed else 0


def stats(values):
    median = statistics.median(values)
    return {'values': values, 'median': median, 'min': min(values), 'max': max(values), 'range_over_median': (max(values) - min(values)) / median if median else None, 'population_rsd': statistics.pstdev(values) / statistics.mean(values) if any(values) else None}


def summarize(output):
    out = Path(output)
    c = load(out / 'config.json')
    validate(c, verify=False)
    start = load(out / 'start.json')
    finish = load(out / 'finish.json') if (out / 'finish.json').exists() else {'status': 'running_or_interrupted'}
    rows = []
    previous_end = None
    for label, m, a, tier, t in groups(c):
        row = {'label': label, 'model': m['id'], 'audio': a['id'], 'tier': tier['id'], 'threads': t['threads'], 'errors': [], 'environment_review_passed': None, 'environment_reasons': ['endpoint snapshots do not prove full-duration frequency/no throttling', 'temperature threshold not frozen']}
        row['environment_reasons'] = []
        environment_states = []
        samples = []
        for repeat in range(6):
            d = out / label / ('warmup' if repeat == 0 else 'r%d' % repeat)
            try:
                e = load(d / 'execution.json')
                require(e['label'] == label + '__' + d.name and e['repeat'] == repeat and e['formal'] == (repeat != 0), 'sample identity mismatch')
                require(e['passed'] and e['exit_code'] == 0, 'failed execution')
                require(previous_end is None or e['start_monotonic'] - previous_end >= 2.999, 'sample interval below 3s')
                previous_end = e['end_monotonic']
                for side in ('before', 'after'):
                    snap = load(d / (side + '.json'))
                    require(snap['boot_id'] == start['boot_id'], 'boot_id mismatch')
                    require(all(k in snap for k in ('time_ns', 'monotonic_ns', 'governor', 'frequency', 'thermal', 'loadavg')), 'snapshot incomplete')
                    for kind in ('governor', 'frequency', 'thermal'):
                        readings = snap[kind]
                        if not readings or any(v.get('value') is None for v in readings.values()):
                            row['environment_reasons'].append(d.name + '/' + side + ': missing ' + kind + '; see snapshot missing_reason')
                    if snap['loadavg'].get('value') is None:
                        row['environment_reasons'].append(d.name + '/' + side + ': missing loadavg')
                sample = parse_sample(d, m['expected'][a['id']], t['threads'])
                affinity = load(d / 'affinity.json')
                binding = affinity_review(affinity['observations'], t['cpus'])
                environment_states.append(binding)
                if binding is not True:
                    row['environment_reasons'].append(d.name + ': actual binding ' + ('failed' if binding is False else 'unknown'))
                periodic_path = d / 'environment.json'
                periodic = load(periodic_path)['observations'] if periodic_path.exists() else []
                state, reasons = environment_review(periodic, t['cpus'], start['boot_id'], c.get('environment_policy', {}))
                times = [obs.get('time_ns') for obs in periodic]
                if not times or any(value is None for value in times):
                    coverage = None
                else:
                    gaps = [times[0] - e['start_ns'], e.get('process_end_ns', e['end_ns']) - times[-1]] + [b - a for a, b in zip(times, times[1:])]
                    coverage = True if all(0 <= gap <= 2_000_000_000 for gap in gaps) else None
                if coverage is None:
                    reasons.append('periodic environment coverage incomplete; unknown')
                environment_states.extend([state, coverage])
                row['environment_reasons'].extend(d.name + ': ' + reason for reason in reasons)
                if repeat:
                    samples.append(sample)
            except (OSError, ValueError, KeyError, TypeError, struct.error) as exc:
                row['errors'].append(d.name + ': ' + str(exc))
        row['environment_review_passed'] = tri_all(environment_states + ([None] if row['errors'] else []))
        row['integrity_passed'] = not row['errors'] and len(samples) == 5
        row['stability_passed'] = None
        if row['integrity_passed']:
            row['statistics'] = {stage: stats([s['stages'][stage] for s in samples]) for stage in LEAVES + ['encoder', 'total']}
            row['stability_limit'] = .03 if t['threads'] == 1 else .05
            row['stability_passed'] = row['statistics']['total']['range_over_median'] <= row['stability_limit']
        rows.append(row)
    for row in rows:
        row.update(regression_passed=None, benefit_vs_A=None)
        for baseline_id in ('A', 'C'):
            if baseline_id == 'C' and row['tier'] != 'D':
                continue
            baseline = next((x for x in rows if (x['model'], x['audio'], x['threads'], x['tier']) == (row['model'], row['audio'], row['threads'], baseline_id)), None)
            if not baseline or not baseline['integrity_passed'] or not row['integrity_passed']:
                continue
            med = lambda r, s: r['statistics'][s]['median']
            regressions = {s: {'baseline_seconds': med(baseline, s), 'candidate_seconds': med(row, s)} for s in LEAVES + ['encoder'] if med(row, s) > med(baseline, s) * 1.05 and (baseline_id == 'C' or med(row, s) - med(baseline, s) > .01)}
            row['regressions_vs_' + baseline_id] = regressions
            row['speedup_vs_' + baseline_id] = med(baseline, 'total') / med(row, 'total')
            if baseline_id == 'A':
                row['regression_passed'] = not regressions
                target = c.get('benefit_targets', {}).get(row['model'], {}).get(row['tier'], {}).get(str(row['threads']))
                row['benefit_target'] = target
                row['benefit_vs_A'] = True if row['tier'] == 'A' else (row['speedup_vs_A'] >= target if target is not None else None)
            elif regressions:
                row['regression_passed'] = False
    integrity = finish['status'] == 'complete' and sha(out / 'config.json') == start['config_sha256'] and all(r['integrity_passed'] for r in rows)
    components = {key: tri_all(r[key] for r in rows) for key in ('stability_passed', 'environment_review_passed', 'regression_passed', 'benefit_vs_A')}
    return {'schema_version': 1, 'is_final': finish['status'] in ('complete', 'failed', 'interrupted'), 'runner_status': finish['status'], 'integrity_passed': integrity, **components, 'performance_passed': tri_all([integrity] + list(components.values())), 'multi_model_expansion_completed': False, 'scope': 'protocol evidence only; distinct model families and full correctness gates require separate acceptance', 'groups': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    for name in ('freeze', 'run', 'summarize'):
        cmd = commands.add_parser(name)
        cmd.add_argument('--config' if name != 'summarize' else '--input', required=True)
        cmd.add_argument('--out', required=True)
    args = parser.parse_args()
    if args.action == 'freeze':
        freeze(args.config, args.out)
    elif args.action == 'run':
        def interrupted(signum, frame):
            raise KeyboardInterrupt('signal %d' % signum)
        signal.signal(signal.SIGTERM, interrupted)
        return run(args.config, args.out)
    else:
        save(args.out, summarize(args.input))
    return 0


if __name__ == '__main__':
    sys.exit(main())
