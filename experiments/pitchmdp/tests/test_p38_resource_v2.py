"""CPU-only tests of the P38 v2 TRAIN-only fixed-step resource measurement (p38-recovery-001; nothing launched)."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import threading

import pytest

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
LAUNCH = PROJECT / 'docs' / 'launch'
sys.path[:0] = [str(PROJECT), str(PROJECT / 'scripts'), str(LAUNCH)]
import gpu3_two_slot as two  # noqa: E402
import make_p38_registered_config as reg  # noqa: E402
import p38_resource_measure as m  # noqa: E402
import p38_smoke_gate as gate  # noqa: E402
import run_cuda_direct as rc  # noqa: E402
from pitchmdp import direct_pitcher_id_registry as pid  # noqa: E402

SPEC = LAUNCH / 'p38-resource-v2-001.json'
GPU4 = pid.RELEASED_GPUS[0]
V1 = {'periodic_residual_counts': 0.6056096930005879, 'pitcher_id_periodic_residual_counts': 0.7587012064977898}


def spec():
    return json.loads(SPEC.read_text())


def child_result(task, seconds, epochs=3, n_vocab=812, uuid=two.GPU3, **over):
    cand = task['arm'] == m.CANDIDATE
    record = {'protocol': m.PROTOCOL, 'arm': task['arm'], 'seed': task['seed'], 'source_sha256': spec()['pins']['source_sha256'],
              'dev_used_for_fit_or_selection': False, 'splits_loaded': ['train', 'earlystop'],
              'runtime': {'bound_uuid': uuid}, 'epochs_run': epochs, 'report_seconds': seconds,
              'optimizer_updates': epochs * (1300 if cand else 1300),
              'pitcher_id': {'n_vocab': n_vocab, 'ids_sha256': 'a' * 64, 'min_count': 100, 'train_rows': 1,
                             'train_pitchers': 2, 'train_unk_rows': 0, 'train_max_date': 20250430,
                             'last_train_date': 20250430, 'unk_share': {'train': 0.1, 'earlystop': 0.2}} if cand else None}
    record.update(over)
    return record


def run_record(per_epoch, **over):
    """per_epoch: {arm: seconds/epoch}; builds a completed serial supervisor record in ABBA order."""
    tasks = [{**t, 'id': f'{t["arm"]}-{t["seed"]}'} for t in spec()['tasks']]
    records = [{'task': t, 'pid': 100 + i, 'returncode': 0, 'oom': False, 'result': child_result(t, 3 * per_epoch[t['arm']])}
               for i, t in enumerate(tasks)]
    return {'tasks': records, 'completed': 4, 'max_live': 1, 'foreign_gpu_apps': [], **over}


@pytest.mark.skip(reason='P40 branch (2026-10-06) changes the runner/model/registry/scorer sources on purpose; this P38 registration-time pin (current tree == frozen 5b3335e source map) passed at ba0a8f1 and P38 is complete')
def test_spec_is_predeclared_gpu3_serial_full_train_same_threshold_and_marked_post_observation():
    s = m.check_spec(spec())
    assert s['status'] == 'AUTHOR_ONLY_NOT_LAUNCHED' and s['concurrent_slots'] == 1 and s['gpu_uuid'] == two.GPU3
    assert GPU4 not in SPEC.read_text() and s['decision']['ratio_max'] == pid.RESOURCE_RATIO_MAX == 1.25
    assert s['workload'] == {**s['workload'], 'train_rows': None, 'earlystop_rows': None, 'epochs': 3}
    assert s['budget']['total_wall_cap_seconds'] == 600 and 4 * s['budget']['task_timeout_seconds'] <= 800
    assert s['provenance']['revision_after_observation'] is True and '1.2528' in s['provenance']['observed_before_this_protocol']
    assert s['pins']['config_sha256'] == gate.DRAFT_SHA and s['pins']['export_manifest_sha256'] == gate.MANIFEST_SHA
    assert s['pins']['source_sha256'] == rc.own_hashes(), 'pins must be the frozen 5b3335e runner/model map'
    assert s['out'].endswith('/p38-resource-v2-001') and 'code-5b3335e' in s['config']
    for bad in ({'concurrent_slots': 2}, {'gpu_uuid': GPU4}):
        with pytest.raises(AssertionError):
            m.check_spec({**s, **bad})
    with pytest.raises(AssertionError):
        m.check_spec({**s, 'decision': {**s['decision'], 'ratio_max': 1.3}})
    with pytest.raises(AssertionError):
        m.check_spec({**s, 'tasks': [{'arm': m.REFERENCE, 'seed': 44}, *s['tasks'][1:]]})


def test_evaluate_passes_at_or_below_and_refuses_above_the_unchanged_threshold():
    ok = m.evaluate(spec(), run_record({m.REFERENCE: 10., m.CANDIDATE: 12.5}))
    assert ok['pass'] and ok['errors'] == [] and ok['candidate_over_reference'] == pytest.approx(1.25)
    assert ok['vocabulary']['n_vocab'] == 812 and ok['bound_uuids'] == [two.GPU3] and ok['tasks'] == 4
    bad = m.evaluate(spec(), run_record({m.REFERENCE: 10., m.CANDIDATE: 12.6}))
    assert not bad['pass'] and bad['errors'] == [] and bad['candidate_over_reference'] == pytest.approx(1.26)


@pytest.mark.parametrize('mutate', [
    lambda r: r.update(completed=3),
    lambda r: r.update(max_live=2),
    lambda r: r.update(foreign_gpu_apps=[[GPU4, 100]]),
    lambda r: r['tasks'][0].update(oom=True),
    lambda r: r['tasks'][1].update(result=None),
    lambda r: r['tasks'][1]['result'].update(epochs_run=2),
    lambda r: [r['tasks'][i]['result']['pitcher_id'].update(n_vocab=0) for i in (1, 2)],  # all-UNK is not full TRAIN
    lambda r: r['tasks'][2]['result']['pitcher_id'].update(train_max_date=20250501),
    lambda r: r['tasks'][2]['result']['pitcher_id'].update(ids_sha256='b' * 64),
    lambda r: r['tasks'][0]['result'].update(pitcher_id={'n_vocab': 1}),
    lambda r: r['tasks'][3]['result'].update(runtime={'bound_uuid': GPU4}),
    lambda r: r['tasks'][3]['result'].update(optimizer_updates=1),
    lambda r: r['tasks'][3]['result'].update(source_sha256={}),
    lambda r: r['tasks'][3]['result'].update(splits_loaded=['train', 'earlystop', 'dev']),
    lambda r: r['tasks'][0]['result'].update(report_seconds=float('nan')),
])
def test_evaluate_fails_closed_on_validity_problems(mutate):
    record = run_record({m.REFERENCE: 10., m.CANDIDATE: 11.})
    mutate(record)
    out = m.evaluate(spec(), record)
    assert not out['pass'] and out['errors']


FAKE = '''import json, sys
json.dump(json.loads(sys.argv[2]), open(sys.argv[1], 'x'))
'''


class NoSampler(threading.Thread):
    def __init__(self, uuid):
        super().__init__(daemon=True)
        self.samples, self.app_rows, self.errors, self.halt = [{'used': 1, 'util': 2}], [], [], threading.Event()

    def run(self):
        self.halt.wait(5)


def local_spec(tmp_path):
    s = spec()
    s.update(lock_dir=str(tmp_path / 'locks'), out=str(tmp_path / 'p38-resource-v2-001'))
    (tmp_path / 'locks').mkdir()
    path = tmp_path / 'spec.json'
    path.write_text(json.dumps(s))
    return path, s


def fake_command(tmp_path, per_epoch):
    fake = tmp_path / 'fake_child.py'
    fake.write_text(FAKE)
    return lambda task, out: [sys.executable, str(fake), str(Path(out) / f'{task["id"]}.json'),
                              json.dumps(child_result(task, 3 * per_epoch[task['arm']]))]


def test_measure_runs_serial_abba_once_writes_record_and_releases_the_common_lock(tmp_path):
    path, s = local_spec(tmp_path)
    out = Path(s['out'])
    rec = m.measure(path, out, preflight_fn=lambda spec: {'ok': True}, sampler_cls=NoSampler,
                    command_for=fake_command(tmp_path, {m.REFERENCE: 10., m.CANDIDATE: 12.}))
    assert rec['decision']['pass'] and rec['serial']['max_live'] == 1
    assert [r['task']['arm'] for r in rec['serial']['tasks']] == [m.REFERENCE, m.CANDIDATE, m.CANDIDATE, m.REFERENCE]
    saved = json.loads((out / 'measure.json').read_text())
    assert saved['decision']['candidate_over_reference'] == pytest.approx(1.2)
    assert json.loads((out / 'spec-copy.json').read_text())['sha256'] == two.sha256_file(path)
    lock = two.TwoSlotSupervisor(s['lock_dir'], two.GPU3)
    lock.acquire()  # released: another queue could now take GPU3
    lock.release()
    with pytest.raises(FileExistsError, match='no rerun'):  # exactly one measurement
        m.measure(path, out, preflight_fn=lambda spec: {}, sampler_cls=NoSampler, command_for=None)
    with pytest.raises(ValueError, match='predeclared'):
        m.measure(path, tmp_path / 'other', preflight_fn=lambda spec: {}, sampler_cls=NoSampler)


def test_preflight_refusal_does_not_consume_the_one_measurement_and_releases_lock(tmp_path):
    path, s = local_spec(tmp_path)

    def refuse(spec):
        raise RuntimeError('GPU3 has compute processes [1]')
    with pytest.raises(RuntimeError, match='compute'):
        m.measure(path, Path(s['out']), preflight_fn=refuse, sampler_cls=NoSampler)
    assert not Path(s['out']).exists()
    holder = two.TwoSlotSupervisor(s['lock_dir'], two.GPU3)
    holder.acquire()
    try:  # a held common lock refuses without waiting and without creating the out dir
        with pytest.raises(Exception):
            m.measure(path, Path(s['out']), preflight_fn=lambda spec: {}, sampler_cls=NoSampler)
    finally:
        holder.release()
    assert not Path(s['out']).exists()


def test_preflight_refuses_when_p38_already_started_or_other_work_runs(tmp_path):
    s = spec()
    marker = tmp_path / 'EXP-P38-001.json'
    marker.write_text('{}')
    with pytest.raises(RuntimeError, match='already registered'):
        m.preflight({**s, 'absent_before': [str(marker)]}, base=lambda spec: {}, procs=lambda pattern: [])
    with pytest.raises(RuntimeError, match='still running'):
        m.preflight({**s, 'absent_before': []}, base=lambda spec: {}, procs=lambda pattern: [4242])
    seen = []
    m.preflight({**s, 'absent_before': []}, base=lambda spec: {'base': True}, procs=lambda p: seen.append(p) or [])
    assert 'p38_resource_child' in seen[0] and 'p38_resource_measure' not in seen[0]  # no ssh/timeout self-match


# ------------------------------------------------------------------ registration with v1 validity + v2 resource

def v1_gate(tmp_path, **over):
    record = {'protocol': gate.PROTOCOL, 'pass': False, 'errors': [], 'train_seconds_per_epoch_median': dict(V1),
              'candidate_over_reference': V1[m.CANDIDATE] / V1[m.REFERENCE], 'ratio_max': 1.25,
              'bound_uuids': [two.GPU3], 'tasks': pid.FITS, 'family_projection_seconds': pid.FAMILY_PROJECTION_SECONDS,
              'config_sha256': gate.DRAFT_SHA, 'export_manifest_sha256': gate.MANIFEST_SHA,
              'source_sha256': rc.own_hashes(), **over}
    path = tmp_path / 'gate.json'
    path.write_text(json.dumps(record, sort_keys=True))
    return path


def v2_measure(tmp_path, per_epoch=None, spec_sha=None, code=None, **over):
    decision = m.evaluate(spec(), run_record(per_epoch or {m.REFERENCE: 10., m.CANDIDATE: 11.}))
    decision.update(over)
    record = {'protocol': m.PROTOCOL, 'spec_sha256': spec_sha or two.sha256_file(SPEC), 'decision': decision,
              'serial': {'max_live': 1}, 'measurement_code_sha256': code or m.measurement_code_sha256()}
    path = tmp_path / 'measure.json'
    path.write_text(json.dumps(record, sort_keys=True))
    return path


@pytest.mark.skip(reason='P40 branch (2026-10-06) changes the runner/model/registry/scorer sources on purpose; this P38 registration-time pin (current tree == frozen 5b3335e source map) passed at ba0a8f1 and P38 is complete')
def test_registration_needs_v1_validity_and_passing_v2_and_records_both_ratios(tmp_path):
    when = reg.at('2026-10-05', '20:30')
    config = reg.build(v1_gate(tmp_path), v2_measure(tmp_path), '20:30', '5b3335e', 'root go', when)
    smoke, res = config['registration']['smoke'], config['registration']['resource']
    assert smoke['v1_verdict'].startswith('REFUSED') and smoke['v1_candidate_over_reference_train_seconds_per_epoch'] \
        == pytest.approx(1.2527890739969603)
    assert res['candidate_over_reference_train_seconds_per_epoch'] == pytest.approx(1.1) and res['ratio_max'] == 1.25
    assert res['method_revised_after_observing_v1'] is True and res['protocol'] == m.PROTOCOL
    assert res['measurement_sha256'] == hashlib.sha256((tmp_path / 'measure.json').read_bytes()).hexdigest()
    assert config['device']['max_parallel'] == 1 and config['device']['allowed_gpu_uuids'] == [two.GPU3]
    rc.config_check(config)
    cases = [  # (v1 overrides, v2 kwargs)
        ({'errors': ['identity']}, {}), ({'tasks': 9}, {}), ({'bound_uuids': [two.GPU3, GPU4]}, {}), ({'pass': True}, {}),
        ({'source_sha256': {}}, {}), ({}, {'per_epoch': {m.REFERENCE: 10., m.CANDIDATE: 12.6}}),
        ({}, {'spec_sha': '0' * 64}), ({}, {'pass': False, 'errors': ['3/4 fits completed']}), ({}, {'ratio_max': 1.3}), ({}, {'bound_uuids': [GPU4]}), ({}, {'tasks': 3}),
        ({}, {'code': {**m.measurement_code_sha256(), 'p38_resource_child.py': '0' * 64}}),
    ]
    for v1_over, v2_kw in cases:
        for name in ('gate.json', 'measure.json'):
            (tmp_path / name).unlink()
        with pytest.raises(AssertionError):
            reg.build(v1_gate(tmp_path, **v1_over), v2_measure(tmp_path, **v2_kw), '20:30', '5b3335e', 'go', when)


def test_template_requires_v2_measurement_and_keeps_one_fit():
    text = (LAUNCH / 'run-p38-pipeline.TEMPLATE.py').read_text()
    assert "ressha='FILL_AT_REGISTRATION'" in text and "p38-resource-v2-001/measure.json" in text
    assert "rd['pass'] is True" in text and "rd['ratio_max']==1.25" in text and "r['serial']['max_live']==1" in text
    assert "new['device']['max_parallel']==1" in text and GPU4 not in text
    assert "g['pass'] is True" not in text, 'v1 resource verdict is replaced by v2, not silently re-passed'


# ------------------------------------------------------------------ review p38-review-execute-002

@pytest.mark.skip(reason='P40 branch (2026-10-06) changes the runner/model/registry/scorer sources on purpose; this P38 registration-time pin (current tree == frozen 5b3335e source map) passed at ba0a8f1 and P38 is complete')
def test_measure_records_its_own_code_hashes_and_registration_records_them(tmp_path):
    path, s = local_spec(tmp_path)
    rec = m.measure(path, Path(s['out']), preflight_fn=lambda spec: {'ok': True}, sampler_cls=NoSampler,
                    command_for=fake_command(tmp_path, {m.REFERENCE: 10., m.CANDIDATE: 11.}))
    assert rec['measurement_code_sha256'] == {n: two.sha256_file(PROJECT / 'scripts' / n) for n in m.MEASUREMENT_CODE}
    assert rec['code_dir'] == str(PROJECT)
    config = reg.build(v1_gate(tmp_path), v2_measure(tmp_path), '20:30', '5b3335e', 'go', reg.at('2026-10-05', '20:30'))
    assert config['registration']['resource']['measurement_code_sha256'] == rec['measurement_code_sha256']


def test_gpu_queries_name_gpu3_only(monkeypatch):
    calls = []

    class Done:
        def __init__(self, stdout):
            self.stdout = stdout

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return Done(fake_run.out)
    monkeypatch.setattr(m.subprocess, 'run', fake_run)
    fake_run.out = f'{two.GPU3}, 4242\n'
    assert m.gpu3_apps() == [(two.GPU3, 4242)]
    assert calls[-1][1] == f'--id={two.GPU3}' and GPU4 not in ' '.join(calls[-1])
    fake_run.out = f'{GPU4}, 1\n'
    with pytest.raises(RuntimeError, match='another GPU'):
        m.gpu3_apps()
    fake_run.out = ''
    assert m.gpu3_apps() == []
    import inspect
    assert inspect.signature(m.measure).parameters['sampler_cls'].default is m.gpu3_sampler
    assert inspect.signature(m.preflight).parameters['base'].default is m.gpu3_preflight
    text = (LAUNCH / 'run-p38-pipeline.TEMPLATE.py').read_text()
    assert "'nvidia-smi','--id='+GPU3,'--query-compute-apps" in text
    assert "['nvidia-smi','--query-compute-apps" not in text, 'template must not list every GPU'
