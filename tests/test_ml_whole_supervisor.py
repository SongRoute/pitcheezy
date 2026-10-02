"""CPU-only checks of the whole-MLB supervisor's durable launch boundaries."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'scripts' / 'supervise_ml_whole.py'
spec = importlib.util.spec_from_file_location('supervise_ml_whole', SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture_plan(tmp_path, *, seconds=2, stages=2, behavior='ok'):
    repo = tmp_path / 'repo'
    registration = tmp_path / 'registration'
    repo.mkdir()
    registration.mkdir()
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    python = registration / '.venv' / 'bin' / 'python'
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    worker = repo / 'worker.py'
    worker.write_text('''import pathlib, subprocess, sys, time
mode, output = sys.argv[1:3]
if mode == "sleep": time.sleep(30)
if mode == "fail": sys.exit(7)
if mode == "tree":
    subprocess.Popen([sys.executable, "-c", "import pathlib,time,sys,signal; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(1.5); pathlib.Path(sys.argv[1]).write_text('orphan')", output + '.orphan'], start_new_session=False)
    time.sleep(30)
pathlib.Path(output).parent.mkdir(parents=True, exist_ok=True)
pathlib.Path(output).write_text("complete")
''')
    config = registration / 'config.json'
    config.write_text('{}')
    subprocess.run(['git', '-C', str(repo), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(repo), '-c', 'user.name=Test', '-c', 'user.email=test@example.test', 'commit', '-qm', 'init'], check=True)
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    root = tmp_path / 'artifacts'
    (root / 'runs' / 'ML-MATRIX-20260924').mkdir(parents=True)
    entries = []
    for number in range(stages):
        output = root / 'runs' / f'whole-{number}' / 'done.txt'
        entries.append({'id': f'stage{number}', 'phase': 'evaluation',
                        'argv': [str(python), str(worker), behavior if number == 0 else 'ok', str(output), str(config)],
                        'config_path': str(config), 'config_sha256': module.sha(config),
                        'source_hashes': {str(worker): module.sha(worker)},
                        'completion_path': str(output), 'max_seconds': seconds, 'gates': []})
    plan = {'protocol': module.PROTOCOL, 'repo': str(repo), 'registration_root': str(registration), 'repo_commit': commit,
            'python_executable': str(python), 'ledger_dir': str(root / 'runs' / 'whole-ledger'),
            'artifact_root': str(root),
            'worker_heavy_lock': str(root / 'runs' / 'ML-MATRIX-20260924' / '.heavy.lock'),
            'termination_grace_seconds': 0.1,
            'budgets_seconds': {'evaluation': 7200, 'improvement': 3600}, 'stages': entries}
    plan_path = registration / 'plan.json'
    plan_path.write_text(json.dumps(plan))
    return module.validate(plan, plan_path), plan_path, worker, config


def test_order_completion_and_hash_guard(tmp_path):
    plan, _, _, _ = fixture_plan(tmp_path)
    digest = module.canonical(plan)
    with pytest.raises(RuntimeError, match='ordering'):
        module.run_stage(plan, digest, 'stage1')
    result = module.run_stage(plan, digest, 'stage0')
    assert result['elapsed_seconds'] > 0
    assert module.run_stage(plan, digest, 'stage1')['stage'] == 'stage1'
    with pytest.raises(RuntimeError, match='ordering'):
        module.run_stage(plan, digest, 'stage0')
    Path(plan['stages'][0]['completion_path']).write_text('tampered')
    with pytest.raises(RuntimeError, match='changed completion'):
        module.ledger_state(plan, digest)


def test_source_pin_and_gate_block_before_reservation(tmp_path):
    plan, _, worker, _ = fixture_plan(tmp_path, stages=1)
    worker.write_text(worker.read_text() + '\n# changed\n')
    with pytest.raises(RuntimeError, match='source file changed'):
        module.run_stage(plan, module.canonical(plan), 'stage0')
    assert not (Path(plan['ledger_dir']) / 'starts' / 'stage0.json').exists()


def test_failed_job_blocks_retry_and_next(tmp_path):
    plan, _, _, _ = fixture_plan(tmp_path, behavior='fail')
    digest = module.canonical(plan)
    with pytest.raises(RuntimeError, match='failed'):
        module.run_stage(plan, digest, 'stage0')
    jobs, charged = module.ledger_state(plan, digest)
    assert jobs[0][1]['exit_code'] == 7 and charged['evaluation'] > 0
    with pytest.raises(RuntimeError, match='manual review'):
        module.run_stage(plan, digest, 'stage1')


def test_orphan_start_reserves_cap_and_blocks(tmp_path):
    plan, _, _, _ = fixture_plan(tmp_path, stages=1)
    digest = module.canonical(plan)
    module.atomic(Path(plan['ledger_dir']) / 'starts' / 'stage0.json',
                  {'protocol': module.PROTOCOL, 'plan_sha256': digest,
                   'stage_id': 'stage0', 'phase': 'evaluation', 'cap_seconds': 2})
    jobs, charged = module.ledger_state(plan, digest)
    assert jobs[0][1] is None and charged['evaluation'] == 2
    with pytest.raises(RuntimeError, match='manual review'):
        module.run_stage(plan, digest, 'stage0')


def test_timeout_reaps_worker_process(tmp_path):
    plan, _, _, _ = fixture_plan(tmp_path, seconds=0.5, stages=1, behavior='sleep')
    digest = module.canonical(plan)
    start = time.monotonic()
    with pytest.raises(RuntimeError, match='timeout'):
        module.run_stage(plan, digest, 'stage0')
    assert time.monotonic() - start < 4
    jobs, _ = module.ledger_state(plan, digest)
    assert jobs[0][1]['outcome'] == 'timeout'
    assert jobs[0][1]['elapsed_seconds'] >= 0.4


def test_timeout_reaps_descendant_process_group(tmp_path):
    plan, _, _, _ = fixture_plan(tmp_path, seconds=0.4, stages=1, behavior='tree')
    with pytest.raises(RuntimeError, match='timeout'):
        module.run_stage(plan, module.canonical(plan), 'stage0')
    time.sleep(1.6)
    assert not Path(plan['stages'][0]['completion_path'] + '.orphan').exists()


def test_numeric_gate_blocks_first_prediction(tmp_path):
    plan, _, _, _ = fixture_plan(tmp_path, stages=2)
    plan['stages'][0]['completion_path'] = str(Path(plan['artifact_root']) / 'runs' / 'profile.json')
    plan['stages'][0]['argv'][3] = plan['stages'][0]['completion_path']
    plan['stages'][1]['gates'] = [{'source_stage_id': 'stage0', 'source': 'ledger_end',
                                  'field': 'elapsed_seconds', 'multiplier': 1000000,
                                  'reserve_seconds': 0}]
    digest = module.canonical(plan)
    module.run_stage(plan, digest, 'stage0')
    with pytest.raises(RuntimeError, match='resource gate'):
        module.run_stage(plan, digest, 'stage1')
    assert not (Path(plan['ledger_dir']) / 'starts' / 'stage1.json').exists()


def test_signal_supervisor_writes_terminal_failure(tmp_path):
    plan, plan_path, _, _ = fixture_plan(tmp_path, seconds=5, stages=1, behavior='sleep')
    virtual_env = Path(plan['registration_root']) / '.venv'
    shutil.rmtree(virtual_env)
    subprocess.run([sys.executable, '-m', 'venv', str(virtual_env)], check=True)
    plan['python_executable'] = str(virtual_env / 'bin' / 'python')
    plan['stages'][0]['argv'][0] = plan['python_executable']
    plan_path.write_text(json.dumps(plan))
    process = subprocess.Popen([plan['python_executable'], str(SOURCE), '--plan', str(plan_path), 'run', '--stage', 'stage0'],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    start_file = Path(plan['ledger_dir']) / 'starts' / 'stage0.json'
    deadline = time.monotonic() + 3
    while not start_file.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    if not start_file.exists():
        out, err = process.communicate(timeout=1)
        pytest.fail(f'No start record: {out} {err}')
    time.sleep(0.1)
    os.kill(process.pid, signal.SIGTERM)
    process.communicate(timeout=3)
    assert process.returncode != 0
    jobs, _ = module.ledger_state(plan, module.canonical(plan))
    assert jobs[0][1]['outcome'] == 'failed'
    assert 'SupervisorInterrupted' in jobs[0][1]['note']


def test_ledger_rejects_out_of_order_and_corrupt_terminal(tmp_path):
    plan, _, _, _ = fixture_plan(tmp_path)
    digest = module.canonical(plan)
    starts = Path(plan['ledger_dir']) / 'starts'
    module.atomic(starts / 'stage1.json', {'protocol': module.PROTOCOL, 'plan_sha256': digest,
                                         'stage_id': 'stage1', 'phase': 'evaluation', 'cap_seconds': 2})
    with pytest.raises(RuntimeError, match='exact registered prefix'):
        module.ledger_state(plan, digest)
    (starts / 'stage1.json').unlink()
    module.run_stage(plan, digest, 'stage0')
    end_path = Path(plan['ledger_dir']) / 'ends' / 'stage0.json'
    end = module.read(end_path)
    end['cap_seconds'] = 999
    module.atomic(end_path, end)
    with pytest.raises(RuntimeError, match='Invalid terminal'):
        module.ledger_state(plan, digest)


def test_negative_gate_evidence_fails_closed(tmp_path):
    plan, _, _, _ = fixture_plan(tmp_path, stages=2)
    gate = {'source_stage_id': 'stage0', 'source': 'completion',
            'field': 'projection', 'multiplier': 1, 'reserve_seconds': 0}
    plan['stages'][1]['gates'] = [gate]
    digest = module.canonical(plan)
    module.run_stage(plan, digest, 'stage0')
    completion = Path(plan['stages'][0]['completion_path'])
    completion.write_text(json.dumps({'projection': -1}))
    end_path = Path(plan['ledger_dir']) / 'ends' / 'stage0.json'
    end = module.read(end_path)
    end['completion_sha256'] = module.sha(completion)
    module.atomic(end_path, end)
    with pytest.raises(RuntimeError, match='resource gate'):
        module.run_stage(plan, digest, 'stage1')
