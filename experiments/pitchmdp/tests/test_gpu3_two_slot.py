"""CPU-only tests for the same-GPU3 two-slot scheduler and benchmark comparison (no CUDA, no SSH)."""
import copy
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import gpu3_two_slot as g  # noqa: E402
from cuda_direct_queue import GpuLock  # noqa: E402

GPU3, GPU4 = g.GPU3, g.RELEASED[0]
SPEC = json.loads((SCRIPTS.parent / 'docs/launch/gpu3-two-slot-bench-001.json').read_text())

FAKE = r'''
import json, os, sys, time, fcntl
out, name, sleep, mode = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
open(os.path.join(out, name + '.start'), 'w').write(f"{time.time()} {os.getpid()} {os.getpgid(0)} {os.environ.get('CUDA_VISIBLE_DEVICES')}")
if mode == 'lockprobe':
    h = open(sys.argv[5], 'a+')
    try:
        fcntl.flock(h.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB); got = True
    except BlockingIOError:
        got = False
    open(os.path.join(out, name + '.lock'), 'w').write(str(got))
if mode == 'spawn':
    import subprocess; subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
time.sleep(0.1 if mode == 'fail' else sleep)
open(os.path.join(out, name + '.end'), 'w').write(str(time.time()))
sys.exit(3 if mode == 'fail' else 0)
'''


@pytest.fixture
def fake(tmp_path):
    path = tmp_path / 'fake_child.py'
    path.write_text(FAKE)
    return path


def cmd(fake, sleep=.6, modes=None, extra=()):
    modes = modes or {}
    return lambda task, out: [sys.executable, str(fake), str(out), task['id'], str(sleep), modes.get(task['id'], 'ok'), *extra]


def tasks(n):
    return [{'id': f't{i}', 'arm': 'control', 'seed': 900 + i} for i in range(n)]


def intervals(out, n):
    rows = []
    for i in range(n):
        start = float((out / f't{i}.start').read_text().split()[0])
        rows.append((start, float((out / f't{i}.end').read_text())))
    return rows


def max_overlap(rows):
    events = sorted([(s, 1) for s, _ in rows] + [(e, -1) for _, e in rows])
    live = peak = 0
    for _, step in events:
        live += step
        peak = max(peak, live)
    return peak


def supervisor(tmp_path):
    sup = g.TwoSlotSupervisor(tmp_path / 'locks', GPU3, grace=2., poll=.05)
    sup.acquire()
    return sup


# ------------------------------------------------------------------ GPU / slot guards

@pytest.mark.parametrize('uuid', [GPU4, 'GPU-00000000-0000-0000-0000-000000000000', '0', ''])
def test_only_gpu3_is_accepted(uuid):
    with pytest.raises(ValueError):
        g.check_gpu(uuid, 1)


@pytest.mark.parametrize('slots', [0, 3, 4])
def test_slot_cap_is_two(slots):
    with pytest.raises(ValueError):
        g.check_gpu(GPU3, slots)


def test_spec_is_gpu3_two_slots_train_only_and_not_launched():
    assert SPEC['gpu_uuid'] == GPU3 and SPEC['concurrent_slots'] == 2 and SPEC['status'] == 'AUTHOR_ONLY_NOT_LAUNCHED'
    assert SPEC['lock_dir'] == '/home/sds733/pitcheezy-autonomy/gpu-locks'  # the P37/P38 common lock dir
    assert {t['seed'] for t in SPEC['tasks']}.isdisjoint(range(39, 49))
    assert SPEC['decision'] == {**SPEC['decision'], 'equivalence': 'exact', 'tolerance': 0, 'min_throughput_gain': 1.15}
    assert SPEC['budget']['total_wall_cap_seconds'] <= 600 and SPEC['workload']['train_rows'] is None


# ------------------------------------------------------------------ scheduling

def test_two_slots_overlap_but_never_exceed_two_and_bind_gpu3(tmp_path, fake):
    sup = supervisor(tmp_path)
    run = sup.run(tasks(4), cmd(fake), tmp_path / 'c', 2, task_timeout=30, total_timeout=60)
    sup.release()
    rows = intervals(tmp_path / 'c', 4)
    assert run['completed'] == 4 and run['max_live'] == 2 and max_overlap(rows) == 2
    for i in range(4):
        _, pid, pgid, visible = (tmp_path / 'c' / f't{i}.start').read_text().split()
        assert visible == GPU3 and pid == pgid  # own session per child, bound to GPU3 only
    assert {r['pid'] for r in run['tasks']} == {int(p.read_text().split()[1]) for p in (tmp_path / 'c').glob('*.start')}
    assert all(r['cpu_seconds'] is not None and r['returncode'] == 0 for r in run['tasks'])


def test_serial_mode_never_overlaps(tmp_path, fake):
    sup = supervisor(tmp_path)
    run = sup.run(tasks(2), cmd(fake, .3), tmp_path / 's', 1, task_timeout=30, total_timeout=60)
    sup.release()
    assert run['max_live'] == 1 and max_overlap(intervals(tmp_path / 's', 2)) == 1


# ------------------------------------------------------------------ common lock

def test_existing_owner_blocks_without_waiting(tmp_path):
    other = GpuLock(tmp_path / 'locks', GPU3)  # e.g. the P37/P38 queue holding GPU3
    assert other.acquire()
    sup = g.TwoSlotSupervisor(tmp_path / 'locks', GPU3)
    started = time.monotonic()
    with pytest.raises(RuntimeError, match='held by another owner'):
        sup.acquire()
    assert time.monotonic() - started < 1 and not sup.owned
    other.release()


def test_supervisor_lock_blocks_queue_and_children_do_not_reacquire(tmp_path, fake):
    sup = supervisor(tmp_path)
    assert not GpuLock(tmp_path / 'locks', GPU3).acquire()  # a P38/other queue sees GPU3 busy
    lock_file = str(tmp_path / 'locks' / f'{GPU3}.lock')
    run = sup.run(tasks(2), cmd(fake, .2, {'t0': 'lockprobe', 't1': 'lockprobe'}, (lock_file,)), tmp_path / 'c', 2,
                  task_timeout=10, total_timeout=20)
    assert run['completed'] == 2  # no child waited on the parent-owned lock (no deadlock)
    assert [(tmp_path / 'c' / f't{i}.lock').read_text() for i in range(2)] == ['False', 'False']
    sup.release()
    q = GpuLock(tmp_path / 'locks', GPU3)
    assert q.acquire()  # released exactly once, by its owner
    q.release()


def test_launch_requires_owned_lock(tmp_path, fake):
    sup = g.TwoSlotSupervisor(tmp_path / 'locks', GPU3)
    with pytest.raises(RuntimeError, match='own the common GPU3 lock'):
        sup.run(tasks(1), cmd(fake), tmp_path / 'c', 1, task_timeout=5, total_timeout=5)


# ------------------------------------------------------------------ failure / timeout / cleanup

def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return subprocess.run(['ps', '-o', 'stat=', '-p', str(pid)], capture_output=True, text=True).stdout.strip()[:1] not in ('Z', '')


def test_failure_stops_sibling_group_only(tmp_path, fake):
    bystander = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True)
    sup = supervisor(tmp_path)
    run = sup.run(tasks(3), cmd(fake, 5, {'t0': 'fail'}), tmp_path / 'c', 2, task_timeout=30, total_timeout=60)
    sup.release()
    codes = {r['task']['id']: r['returncode'] for r in run['tasks']}
    assert codes['t0'] == 3 and 't2' not in codes and run['completed'] == 0  # t1 stopped, t2 never launched
    assert alive(bystander.pid)  # a process this supervisor did not create is untouched
    bystander.kill()
    bystander.wait()


def test_timeout_kills_whole_own_group_including_grandchild(tmp_path, fake):
    sup = supervisor(tmp_path)
    run = sup.run(tasks(1), cmd(fake, 30, {'t0': 'spawn'}), tmp_path / 'c', 2, task_timeout=1, total_timeout=10)
    sup.release()
    assert run['tasks'][0]['timed_out'] and run['completed'] == 0
    pgid = int((tmp_path / 'c' / 't0.start').read_text().split()[2])
    time.sleep(.3)
    left = subprocess.run(['pgrep', '-g', str(pgid)], capture_output=True, text=True).stdout.split()
    assert not [p for p in left if alive(int(p))]


def test_release_refused_while_children_alive(tmp_path):
    sup = supervisor(tmp_path)
    sup.running[12345] = {'pid': 12345}
    with pytest.raises(RuntimeError, match='children are alive'):
        sup.release()
    sup.running.clear()
    sup.release()


# ------------------------------------------------------------------ no overwrite / no partial resume

def test_mode_and_bench_directories_are_never_reused(tmp_path, fake):
    sup = supervisor(tmp_path)
    (tmp_path / 'c').mkdir()
    with pytest.raises(FileExistsError):
        sup.run(tasks(1), cmd(fake), tmp_path / 'c', 1, task_timeout=5, total_timeout=5)
    sup.release()
    (tmp_path / 'bench').mkdir()
    with pytest.raises(FileExistsError):
        g.bench(SCRIPTS.parent / 'docs/launch/gpu3-two-slot-bench-001.json', tmp_path / 'bench')
    with pytest.raises(FileExistsError):
        g.write_new_json(tmp_path / 'x.json', {}) or g.write_new_json(tmp_path / 'x.json', {})


# ------------------------------------------------------------------ preflight (fail closed)

def preflight_spec(tmp_path, status='complete'):
    state = tmp_path / 'p37.json'
    state.write_text(json.dumps({'status': status}))
    return {**SPEC, 'wait_for_terminal': [str(state)]}


OK = dict(mem_available=lambda: 64 << 30, gpu=lambda u: (20 << 30, 0, 0), apps=lambda: [(GPU4, 77)], procs=lambda p: [])


def test_preflight_passes_with_room(tmp_path):
    record = g.preflight(preflight_spec(tmp_path), **OK)
    assert record['host_required'] == 28 << 30 and record['gpu3_required'] == 10 << 30


@pytest.mark.parametrize('change, match', [
    ({'mem_available': lambda: 27 << 30}, 'MemAvailable'),
    ({'gpu': lambda u: (9 << 30, 0, 0)}, 'GPU3 free'),
    ({'apps': lambda: [(GPU3, 27851)]}, 'compute processes'),
    ({'procs': lambda p: [27849]}, 'still running'),
])
def test_preflight_fails_closed(tmp_path, change, match):
    with pytest.raises(RuntimeError, match=match):
        g.preflight(preflight_spec(tmp_path), **{**OK, **change})


@pytest.mark.parametrize('status', ['running', 'p37-full-001', None])
def test_preflight_waits_for_p37_terminal(tmp_path, status):
    with pytest.raises(RuntimeError, match='not terminal'):
        g.preflight(preflight_spec(tmp_path, status), **OK)


# ------------------------------------------------------------------ predeclared comparison and mutants

def result(arm, seed):
    rt = {'bound_uuid': GPU3, 'cuda_visible_devices': GPU3, 'device_uuid': GPU3[4:]}
    return {'arm': arm, 'seed': seed, 'state_dict_sha256': f's{seed}', 'tensor_sha256': {'w': f'w{seed}'},
            'initial_param_sha256': f'i{seed}', 'first_permutation_sha256': f'p{seed}',
            'history': [{'epoch': 1, 'train_weighted_nll': 1.25, 'earlystop_conditional_nll': 1.5}],
            'best_epoch': 1, 'epochs_run': 5, 'optimizer_updates': 6120, 'train_inputs_sha256': 't',
            'earlystop_inputs_sha256': 'e', 'source_sha256': SPEC['pins']['source_sha256'], 'training_rows': 10,
            'earlystop_rows': 5, 'train_only': True, 'runtime': rt, 'fit_seconds': 40., 'pid': seed}


def run_of(wall, live, fit=40.):
    recs = [{'pid': i, 'cuda_visible_devices': GPU3, 'oom': False, 'result': {**result(t['arm'], t['seed']), 'fit_seconds': fit}}
            for i, t in enumerate(SPEC['tasks'])]
    return {'tasks': recs, 'completed': 2, 'wall_seconds': wall, 'max_live': live, 'foreign_gpu_apps': []}


def test_equivalent_and_fast_enough_is_only_a_candidate():
    out = g.compare(SPEC, run_of(100., 1), run_of(80., 2, 55.))
    assert out['equivalent'] and out['throughput_gain_total_wall'] == pytest.approx(1.25)
    assert out['verdict'] == 'adoption_candidate_for_review'


def test_timing_fields_are_excluded_from_equivalence():
    conc = run_of(80., 2)
    for r in conc['tasks']:
        r['result'].update(fit_seconds=99., pid=1, report_seconds=3., peak_device_bytes=5)
    assert g.compare(SPEC, run_of(100., 1), conc)['equivalent']


def test_gain_below_threshold_retains_serial():
    out = g.compare(SPEC, run_of(100., 1), run_of(100. / 1.14, 2))
    assert out['equivalent'] and out['verdict'] == 'retain_serial_record_negative'


MUTANTS = [
    lambda r: r['tasks'][0]['result']['tensor_sha256'].update(w='x'),
    lambda r: r['tasks'][1]['result'].update(state_dict_sha256='x'),
    lambda r: r['tasks'][0]['result']['history'][0].update(earlystop_conditional_nll=1.5000000001),
    lambda r: r['tasks'][0]['result'].update(optimizer_updates=6119),
    lambda r: r['tasks'][0]['result'].update(best_epoch=2),
    lambda r: r['tasks'][0]['result'].update(initial_param_sha256='x'),
    lambda r: r['tasks'][0]['result'].update(first_permutation_sha256='x'),
    lambda r: r['tasks'][0]['result'].update(train_inputs_sha256='x'),
    lambda r: r['tasks'][0]['result'].pop('history'),
    lambda r: r['tasks'][0]['result']['runtime'].update(bound_uuid=GPU4),
    lambda r: r['tasks'][0]['result']['runtime'].update(device_uuid=''),
    lambda r: r['tasks'][0].update(cuda_visible_devices=GPU4),
    lambda r: r['tasks'][0]['result'].update(source_sha256={}),
    lambda r: r['tasks'][0]['result'].update(train_only=False),
    lambda r: r['tasks'][0].update(oom=True),
    lambda r: r.update(completed=1),
    lambda r: r['tasks'].pop(),
    lambda r: r['tasks'][1]['result'].update(seed=900, arm='control'),
    lambda r: r.update(max_live=3),
    lambda r: r.update(max_live=1),
    lambda r: r.update(foreign_gpu_apps=[(GPU4, 0)]),
]


@pytest.mark.parametrize('mutate', MUTANTS)
def test_comparison_rejects_mutants_even_when_fast(mutate):
    conc = copy.deepcopy(run_of(50., 2))
    mutate(conc)
    out = g.compare(SPEC, run_of(100., 1), conc)
    assert not out['equivalent'] and out['verdict'] == 'retain_serial_record_negative'


def test_post_hoc_tolerance_is_refused():
    spec = copy.deepcopy(SPEC)
    spec['decision']['tolerance'] = 1e-6
    assert g.compare(spec, run_of(100., 1), run_of(50., 2))['verdict'] == 'retain_serial_record_negative'


# ------------------------------------------------------------------ bench orchestration (fake children)

def test_bench_holds_one_lock_across_modes_and_records_everything(tmp_path, fake, monkeypatch):
    spec = {**preflight_spec(tmp_path), 'lock_dir': str(tmp_path / 'locks'),
            'budget': {**SPEC['budget'], 'task_timeout_seconds': 10, 'min_seconds_to_start_mode': 1}}
    path = tmp_path / 'spec.json'
    path.write_text(json.dumps(spec))
    seen_busy = []

    def command(task, out):
        seen_busy.append(not GpuLock(tmp_path / 'locks', GPU3).acquire())
        return [sys.executable, str(fake), str(out), task['id'], '.2', 'ok']

    class NoSampler(g.Sampler):
        def __init__(self, uuid):
            super().__init__(uuid, gpu=lambda u: (0, 9, 1 << 30), apps=lambda: [], period=.05)

    record = g.bench(path, tmp_path / 'bench', preflight_fn=lambda s: {'ok': True}, sampler_cls=NoSampler,
                     command_for=command)
    assert seen_busy == [True] * 4  # lock held by the supervisor during every launch in both modes
    assert record['concurrent']['max_live'] == 2 and record['serial']['max_live'] == 1
    assert record['serial']['gpu3_peak_used_bytes'] == 1 << 30 and 'total_wall_seconds' in record
    assert record['decision']['verdict'] == 'retain_serial_record_negative'  # fake children write no fit evidence
    assert json.loads((tmp_path / 'bench' / 'bench.json').read_text())['spec_sha256'] == g.sha256_file(path)
    assert GpuLock(tmp_path / 'locks', GPU3).acquire()  # released at the end


def test_bench_refuses_when_preflight_fails_and_releases_lock(tmp_path):
    spec = {**preflight_spec(tmp_path, 'running'), 'lock_dir': str(tmp_path / 'locks')}
    path = tmp_path / 'spec.json'
    path.write_text(json.dumps(spec))
    with pytest.raises(RuntimeError, match='not terminal'):
        g.bench(path, tmp_path / 'bench', preflight_fn=lambda s: g.preflight(s, **OK))
    assert GpuLock(tmp_path / 'locks', GPU3).acquire()


# ------------------------------------------------------------------ frozen route

@pytest.mark.skip(reason='P40 branch (2026-10-06) changes the runner/model/registry sources on purpose; this pin (bench child == frozen 5b3335e route) passed at ba0a8f1; a two-slot run on P40 sources would need its own equivalence bench')
def test_child_uses_frozen_runner_route_and_frozen_sources_unchanged():
    import gpu3_bench_child as child
    import run_cuda_direct as runner
    assert child.runner is runner
    text = (SCRIPTS / 'gpu3_bench_child.py').read_text()
    for name in ('bind_gpu', 'config_check', 'verify_export', 'load_split', 'train_subset', 'arm_inputs', 'build_model'):
        assert f'runner.{name}(' in text
    for forbidden in ("'dev'", "'mlb_dev'", "'blend'", "'temperature'", 'fast_fit', 'run_task('):
        assert forbidden not in text
    # p38-recovery-001 (on P38 review head 5b3335e): bench-001 pins the P37 route; here only the runner (P38 dispatch,
    # reviewed in 5b3335e) differs and the two pitcher-ID modules are added; every other pinned model file is equal
    own, p37 = runner.own_hashes(), SPEC['pins']['source_sha256']
    assert {k for k in p37 if own.get(k) != p37[k]} == {'scripts/run_cuda_direct.py'}
    assert set(own) - set(p37) == {'pitchmdp/direct_pitcher_id_model.py', 'pitchmdp/direct_pitcher_id_registry.py'}
    frozen = ['experiments/pitchmdp/' + p for p in own] + ['configs/EXP-P37-001.json',
              'configs/DRAFT-CUDA-PITCHER-ID-20261005.json', 'experiments/pitchmdp/scripts/cuda_direct_queue.py']
    root = SCRIPTS.parents[2]
    diff = subprocess.run(['git', 'diff', '--stat', '5b3335e', '--', *frozen], cwd=root, capture_output=True, text=True)
    assert diff.returncode == 0 and diff.stdout == ''


def test_child_evidence_is_deterministic_on_cpu_with_real_fit():
    torch = pytest.importorskip('torch')
    import gpu3_bench_child as child
    from pitchmdp.matrix_models import MatrixModel
    rng = np.random.default_rng(0)
    n, m = 600, 120
    def inputs(k):
        tok = rng.normal(size=(k, 6, 38)).astype(np.float32)
        tok[:, -1, -11:] = 0  # current-outcome channels are zero by contract
        return (tok, np.ones((k, 6), dtype=bool),
                rng.normal(size=(k, 52)).astype(np.float32))
    tr, ea = inputs(n), inputs(m)
    y, ey = rng.integers(0, 10, n), rng.integers(0, 10, m)
    runs = [child.fit_and_record(MatrixModel('flatten_mlp', seed=900, width=16, device='cpu'), tr, y, ea, ey,
                                 seed=900, epochs=3, batch_size=128, learning_rate=5e-4) for _ in range(2)]
    other = child.fit_and_record(MatrixModel('flatten_mlp', seed=901, width=16, device='cpu'), tr, y, ea, ey,
                                 seed=901, epochs=3, batch_size=128, learning_rate=5e-4)
    keys = [k for k in g.COMPARED if k in runs[0]]
    assert runs[0]['optimizer_updates'] == 3 * 5 and runs[0]['epochs_run'] == 3  # patience == epochs: fixed steps
    assert runs[0]['initial_param_sha256'] and all(runs[0][k] == runs[1][k] for k in keys)
    assert other['initial_param_sha256'] != runs[0]['initial_param_sha256']
    assert other['state_dict_sha256'] != runs[0]['state_dict_sha256']


# ------------------------------------------------------------------ review (execute-004): outer-timeout signal

SIGNAL_DRIVER = r"""
import sys, time
sys.path.insert(0, sys.argv[1])
import gpu3_two_slot as g
g.install_signal_handlers()
sup = g.TwoSlotSupervisor(sys.argv[2], g.GPU3, grace=2.)
sup.acquire()
cmd = lambda task, out: [sys.executable, sys.argv[3], str(out), task['id'], '60', 'spawn']
try:
    sup.run([{'id': 't0'}, {'id': 't1'}], cmd, sys.argv[4], 2, task_timeout=120, total_timeout=120)
except RuntimeError as error:
    sup.release()
    print('STOPPED', error, flush=True)
"""


def test_sigterm_stops_own_child_groups_and_releases_lock(tmp_path, fake):
    """Without a handler, SIGTERM from the outer `timeout` kills only the supervisor; the children (own sessions)
    would keep running on GPU3 after the lock is gone."""
    driver = tmp_path / 'driver.py'
    driver.write_text(SIGNAL_DRIVER)
    out = tmp_path / 'm'
    proc = subprocess.Popen([sys.executable, str(driver), str(SCRIPTS), str(tmp_path / 'locks'), str(fake), str(out)],
                            stdout=subprocess.PIPE, text=True)
    deadline = time.time() + 20
    while time.time() < deadline and len(list(out.glob('*.start'))) < 2:
        time.sleep(.05)
    assert len(list(out.glob('*.start'))) == 2
    time.sleep(.3)  # let each child spawn its grandchild
    groups = [int(s.read_text().split()[2]) for s in out.glob('*.start')]
    proc.send_signal(signal.SIGTERM)
    try:
        stdout, _ = proc.communicate(timeout=30)
        assert 'STOPPED supervisor interrupted by signal' in stdout
        time.sleep(.3)
        for pgid in groups:
            left = subprocess.run(['pgrep', '-g', str(pgid)], capture_output=True, text=True).stdout.split()
            assert not [p for p in left if alive(int(p))], f'own group {pgid} outlived the supervisor'
    finally:
        for pgid in groups:  # test hygiene only: never leave a sleeping fake behind
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    lock = GpuLock(tmp_path / 'locks', GPU3)
    assert lock.acquire()
    lock.release()


def test_preflight_refuses_while_a_p3x_pipeline_is_alive(tmp_path):
    procs = lambda pattern: [4242] if 'run-p3[0-9]-pipeline' in pattern else []
    spec = copy.deepcopy(SPEC)
    state = tmp_path / 'p37.json'
    state.write_text(json.dumps({'status': 'complete'}))
    spec['wait_for_terminal'] = [str(state)]
    with pytest.raises(RuntimeError, match='still running'):
        g.preflight(spec, mem_available=lambda: 1 << 40, gpu=lambda u: (1 << 40, 0, 0), apps=lambda: [], procs=procs)



def test_exit_status_survives_a_sampler_that_spawns_subprocesses(tmp_path, fake):
    """Real GPU3 run 2026-10-05 (bench-002): the 1 s nvidia-smi sampler's subprocess.run reaped a finished child
    whose Popen object had been dropped, so a successful fit got returncode None and the mode counted 1/2."""
    halt = __import__('threading').Event()

    def spawner():
        while not halt.is_set():
            subprocess.run(['true'])
            time.sleep(.01)
    thread = __import__('threading').Thread(target=spawner, daemon=True)
    thread.start()
    try:
        sup = supervisor(tmp_path)
        run = sup.run(tasks(2), cmd(fake, .4), tmp_path / 'c', 2, task_timeout=20, total_timeout=30)
        sup.release()
    finally:
        halt.set()
        thread.join(5)
    assert [r['returncode'] for r in run['tasks']] == [0, 0] and run['completed'] == 2
    assert all(r['cpu_seconds'] is not None and 'proc' not in r for r in run['tasks'])
