"""EXP-P38-001 TRAIN-only fixed-step resource measurement, protocol p38_train_fixed_step_resource_v2 (AUTHOR ONLY).

The v1 smoke gate (8,192 TRAIN rows x 2 epochs) REFUSED at 1.2528 > 1.25 on 2026-10-05 15:44 KST and stays the
original record.  This v2 METHOD was written AFTER seeing that result: same metric (report.seconds / epochs_run, median
per arm), same threshold 1.25, same arms, but the full TRAIN rows/vocabulary with fixed epochs (patience == epochs).
It is a resource rule, not a science result, not the original pre-registration and not an independent validation.

Exactly ONE run: the out dir is fixed in the spec, created exclusively, and the CLI refuses any other or existing out.
Serial only (one live child): the reviewed TwoSlotSupervisor (gpu3_two_slot.py) is reused with slots=1 for the common
GPU3 flock, own-session children, own-group cleanup on timeout/signal and lock release; preflight is gpu3_two_slot's
plus the P38 'not started' paths.

  cd $B/code-<commit>/experiments/pitchmdp && python scripts/p38_resource_measure.py measure \
      --spec docs/launch/p38-resource-v2-001.json --out $B/p38-resource-v2-001
"""
import argparse
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu3_two_slot as two  # reviewed supervisor/preflight (c17a092 lineage), reused unchanged

PROTOCOL = 'p38_train_fixed_step_resource_v2'
REFERENCE = 'periodic_residual_counts'
CANDIDATE = 'pitcher_id_periodic_residual_counts'
RATIO_MAX = 1.25
LAST_TRAIN_DATE = 20250430
# not 'p38_resource_measure': the outer timeout / ssh shell of this very launch carries that string (pgrep self-match)
OTHER_WORK = two.OTHER_GPU_WORK + '|p38_resource_child'


# review p38-review-execute-002: the measurement code itself is part of the source identity (registration checks these
# bytes against the reviewed commit), next to the frozen runner/model map in pins.source_sha256
MEASUREMENT_CODE = ('p38_resource_measure.py', 'p38_resource_child.py', 'gpu3_two_slot.py', 'gpu3_bench_child.py')


def measurement_code_sha256(scripts=Path(__file__).resolve().parent):
    return {name: two.sha256_file(Path(scripts) / name) for name in MEASUREMENT_CODE}


def gpu3_apps(uuid=two.GPU3):
    """[(gpu_uuid, pid)] of compute apps on GPU3 ONLY (nvidia-smi --id filter): one-card rule, no other GPU is queried.
    A child cannot reach another GPU anyway: CUDA_VISIBLE_DEVICES is the GPU3 UUID and bind_gpu's bound_uuid is checked."""
    out = subprocess.run(['nvidia-smi', f'--id={uuid}', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True, timeout=20, check=True).stdout
    rows = [(u.strip(), int(p)) for u, p in (line.split(',') for line in out.strip().splitlines() if line.strip())]
    if any(u != uuid for u, _ in rows):
        raise RuntimeError(f'nvidia-smi --id={uuid} answered for another GPU: {rows}')
    return rows


def gpu3_preflight(spec):
    return two.preflight(spec, apps=gpu3_apps)


def gpu3_sampler(uuid):
    return two.Sampler(uuid, apps=gpu3_apps)


def check_spec(spec):
    """Fail closed on anything that would change the predeclared measurement."""
    assert spec.get('protocol') == PROTOCOL, 'unknown protocol'
    assert spec['gpu_uuid'] == two.GPU3 and spec['gpu_uuid'] not in two.RELEASED, 'GPU3 only'
    assert spec['concurrent_slots'] == 1, 'v2 measurement is serial (max 1 fit)'
    assert spec['decision']['ratio_max'] == RATIO_MAX, 'threshold is fixed at 1.25'
    assert spec['workload']['train_rows'] is None and spec['workload']['earlystop_rows'] is None, 'full TRAIN only'
    assert isinstance(spec['workload']['epochs'], int) and spec['workload']['epochs'] >= 2
    assert spec['budget']['total_wall_cap_seconds'] <= 600, 'wall cap <= 600 s'
    arms = [t['arm'] for t in spec['tasks']]
    assert arms == [REFERENCE, CANDIDATE, CANDIDATE, REFERENCE], 'ABBA order of the two P38 arms'
    seeds = {t['seed'] for t in spec['tasks']}
    assert not seeds & set(range(39, 49)) and not seeds & {900, 901}, 'measurement seeds must be outside P37/P38/bench'
    assert spec['provenance']['revision_after_observation'] is True
    return spec


def evaluate(spec, run):
    """Pure: run = supervisor record with per-task 'result' (child JSON) and 'oom'. Returns the decision record."""
    errors, per_arm, uuids, vocab = [], {REFERENCE: [], CANDIDATE: []}, set(), set()
    tasks = spec['tasks']
    epochs = spec['workload']['epochs']
    if run.get('completed') != len(tasks):
        errors.append(f'{run.get("completed")}/{len(tasks)} fits completed')
    if run.get('max_live', 0) != 1:
        errors.append(f'max_live {run.get("max_live")} != 1 (serial protocol)')
    if run.get('foreign_gpu_apps'):
        errors.append(f'a child ran on another GPU: {run["foreign_gpu_apps"]}')
    updates = {REFERENCE: set(), CANDIDATE: set()}
    for record in run.get('tasks', []):
        task, result = record['task'], record.get('result')
        name = f'{task["arm"]}-{task["seed"]}'
        if record.get('oom'):
            errors.append(f'{name}: out of memory')
        if not result:
            errors.append(f'{name}: no result')
            continue
        if result.get('protocol') != PROTOCOL or result.get('arm') != task['arm'] or result.get('seed') != task['seed']:
            errors.append(f'{name}: result identity differs')
        if result.get('source_sha256') != spec['pins']['source_sha256']:
            errors.append(f'{name}: sources differ from pins')
        if result.get('dev_used_for_fit_or_selection') is not False or result.get('splits_loaded') != ['train', 'earlystop']:
            errors.append(f'{name}: not TRAIN/earlystop-only')
        uuids.add((result.get('runtime') or {}).get('bound_uuid'))
        if result.get('epochs_run') != epochs:
            errors.append(f'{name}: epochs_run {result.get("epochs_run")} != fixed {epochs}')
        seconds = result.get('report_seconds')
        if not isinstance(seconds, (int, float)) or isinstance(seconds, bool) or not math.isfinite(seconds) or seconds <= 0:
            errors.append(f'{name}: report_seconds invalid {seconds!r}')
        elif result.get('epochs_run') == epochs:
            per_arm.setdefault(task['arm'], []).append(seconds / epochs)
        updates.setdefault(task['arm'], set()).add(result.get('optimizer_updates'))
        record_id = result.get('pitcher_id')
        if task['arm'] == REFERENCE and record_id is not None:
            errors.append(f'{name}: reference carries a pitcher-ID record')
        if task['arm'] == CANDIDATE:
            if not isinstance(record_id, dict) or not isinstance(record_id.get('n_vocab'), int) or record_id['n_vocab'] < 1 \
                    or record_id.get('train_max_date') is None or int(record_id['train_max_date']) > LAST_TRAIN_DATE:
                errors.append(f'{name}: full-TRAIN vocabulary record missing, empty or after the cutoff')
            else:
                vocab.add(json.dumps({k: v for k, v in record_id.items() if k != 'unk_share'}, sort_keys=True))
    for arm, values in updates.items():
        if len(values) > 1:
            errors.append(f'{arm}: optimizer_updates differ across seeds {sorted(map(str, values))}')
    if len(vocab) > 1:
        errors.append('candidate fits carry different vocabulary records (must be TRAIN-only, seed-free)')
    if uuids != {two.GPU3}:
        errors.append(f'bound UUID set {sorted(map(str, uuids))} != [GPU3]')
    counts = {arm: len(v) for arm, v in per_arm.items()}
    medians = {arm: statistics.median(v) if len(v) == 2 else None for arm, v in per_arm.items()}
    if counts != {REFERENCE: 2, CANDIDATE: 2}:
        errors.append(f'per-arm valid fit counts {counts} != 2 each')
    cand, ref = medians.get(CANDIDATE), medians.get(REFERENCE)
    ratio = cand / ref if cand is not None and ref else None
    passed = not errors and ratio is not None and math.isfinite(ratio) and ratio <= RATIO_MAX
    return {'protocol': PROTOCOL, 'pass': passed, 'errors': errors, 'candidate_over_reference': ratio,
            'ratio_max': RATIO_MAX, 'train_seconds_per_epoch_median': medians,
            'train_seconds_per_epoch': per_arm, 'bound_uuids': sorted(map(str, uuids)),
            'vocabulary': json.loads(next(iter(vocab))) if len(vocab) == 1 else None, 'tasks': len(tasks),
            'note': 'resource rule only (method revised after observing the v1 smoke refusal 1.2528); '
                    'not a science result; no DEV/2026 read, nothing registered or scored'}


def preflight(spec, base=gpu3_preflight, procs=two.pgrep):
    for path in spec['absent_before']:
        if Path(path).exists():
            raise RuntimeError(f'{path} exists: P38 already registered/started; the v2 measurement is pre-registration only')
    others = procs(OTHER_WORK)
    if others:
        raise RuntimeError(f'Pitcheezy GPU work still running: pids {others}')
    return base(spec)


def child_command(spec, spec_path):
    def command(task, out_dir):
        return [spec['python'], str(Path(__file__).resolve().parent / 'p38_resource_child.py'), '--spec', str(spec_path),
                '--arm', task['arm'], '--seed', str(task['seed']), '--result', str(Path(out_dir) / f'{task["id"]}.json')]
    return command


def measure(spec_path, out, *, preflight_fn=preflight, sampler_cls=gpu3_sampler, command_for=None):
    spec_path, out = Path(spec_path).resolve(), Path(out)
    spec = check_spec(json.loads(spec_path.read_text()))
    if str(out) != spec['out']:
        raise ValueError(f'out must be the predeclared {spec["out"]} (exactly one measurement)')
    if out.exists():
        raise FileExistsError(f'{out} exists: the one v2 measurement was already used; no rerun')
    sup = two.TwoSlotSupervisor(spec['lock_dir'], spec['gpu_uuid'])
    sup.acquire()  # non-blocking; refuses if any P37/P38/bench queue holds the common GPU3 lock
    try:
        checked = preflight_fn(spec)
    except BaseException:
        sup.release()  # refused before any CUDA: the out dir is NOT created, the one measurement is not consumed
        raise
    out.mkdir(parents=True, exist_ok=False)  # exclusive from here on: a second run of this protocol is refused
    two.write_new_json(out / 'spec-copy.json', {'sha256': two.sha256_file(spec_path), 'spec': spec})
    tasks = [{**t, 'id': f'{t["arm"]}-{t["seed"]}'} for t in spec['tasks']]
    command_for = command_for or child_command(spec, spec_path)
    record = {'started_utc': two.utc(), 'spec_sha256': two.sha256_file(spec_path), 'protocol': PROTOCOL,
              'measurement_code_sha256': measurement_code_sha256(), 'code_dir': str(Path(__file__).resolve().parents[1]),
              'preflight': checked}
    try:
        budget, start = spec['budget'], time.monotonic()
        sampler = sampler_cls(spec['gpu_uuid'])
        sampler.start()
        try:
            run = sup.run(tasks, command_for, out / 'serial', 1, task_timeout=budget['task_timeout_seconds'],
                          total_timeout=budget['total_wall_cap_seconds'])
        finally:
            sampler.halt.set()
            sampler.join(5)
        run = two.attach_results(run, out / 'serial', sampler.app_rows)
        run.update(gpu3_peak_used_bytes=max((s['used'] for s in sampler.samples), default=None),
                   gpu3_peak_util=max((s['util'] for s in sampler.samples), default=None),
                   sampler_errors=sampler.errors)
        record['serial'] = run
        record['total_wall_seconds'] = time.monotonic() - start
        record['decision'] = evaluate(spec, run)
    except BaseException as error:
        record['decision'] = {'protocol': PROTOCOL, 'pass': False, 'errors': [f'measurement aborted: {error!r}']}
        raise
    finally:
        sup.release()
        record['finished_utc'] = two.utc()
        two.write_new_json(out / 'measure.json', record)
    return record


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    m = sub.add_parser('measure')
    m.add_argument('--spec', type=Path, required=True)
    m.add_argument('--out', type=Path, required=True)
    args = p.parse_args(argv)
    two.install_signal_handlers()
    result = measure(args.spec, args.out)
    out = Path(args.out) / 'measure.json'
    print('P38_RESOURCE_V2', 'PASS' if result['decision']['pass'] else 'REFUSED', two.sha256_file(out),
          json.dumps(result['decision'].get('errors', [])[:5]), flush=True)
    return 0 if result['decision']['pass'] else 3


if __name__ == '__main__':
    sys.exit(main())
