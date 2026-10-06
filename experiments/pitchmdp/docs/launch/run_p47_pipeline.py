"""P47 registered serial queue and scorer. Run only from a reviewed, staged source copy on newsrv.

The queue owns durable task status and bounded retries; this wrapper refuses a second
start after any state file exists. Manual recovery reviews queue/status before a new
attempt. Scoring requires sixteen matching sealed tasks.
"""
import argparse
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'scripts'), str(Path(__file__).resolve().parent)]
import p47_launch
import run_cuda_direct as runner
from pitchmdp import direct_pitcher_id_registry as pid

GPU3 = pid.RETAINED_GPU
LOCKS = '/home/sds733/pitcheezy-autonomy/gpu-locks'
SCORE_SECONDS = 900
REPORT_SECONDS = 300


def clock(queue, now):
    if now.date().isoformat() != queue['date'] or now > p47_launch.at(queue['date'], queue['launch_by']):
        raise ValueError('Actual launch is outside the dated window')
    no_new = p47_launch.at(queue['date'], queue['no_new_fit_after'])
    kill = p47_launch.at(queue['date'], queue['kill_own_after'])
    aggregate = p47_launch.at(queue['date'], queue['aggregate_by'])
    if now + timedelta(seconds=pid.P47_FAMILY_PROJECTION_SECONDS) > no_new \
            or kill - no_new < timedelta(minutes=15) or aggregate - kill < timedelta(minutes=20):
        raise ValueError('Actual fit projection/grace/scorer reserve does not fit')
    return int((kill - now).total_seconds()), aggregate - timedelta(seconds=SCORE_SECONDS + REPORT_SECONDS)


def preflight(config, check, draft, config_bytes, check_bytes, source_commit):
    if config['status'] != 'REGISTERED' or config['execution'] != {'enabled': True, 'executable': True} \
            or config[pid.DESIGN_KEY] != pid.P47_DESIGN:
        raise ValueError('Registered P47 config required')
    reg = config['registration']
    if reg['source_commit'] != source_commit or reg['draft_sha256'] != hashlib.sha256(draft).hexdigest() \
            or reg['preflight']['check_sha256'] != hashlib.sha256(check_bytes).hexdigest():
        raise ValueError('Commit/DRAFT/probe pin differs')
    if reg.get('execution_code_sha256') != p47_launch.execution_hashes():
        raise ValueError('Queue/scorer/registration/pipeline code differs from registration pin')
    if reg['undated_draft_sha256'] != p47_launch.UNDATED_SHA or \
            config['device']['allowed_gpu_uuids'] != [GPU3] or config['device']['max_parallel'] != 1:
        raise ValueError('Undated or GPU3 serial pin differs')
    if not isinstance(reg.get('seed_id_ledger_review'), dict) or not reg['seed_id_ledger_review'].get('sha256') \
            or reg.get('go_reference') in (None, ''):
        raise ValueError('Global ledger reservation and PM acceptance missing')
    p47_launch.dated_check(draft.decode(), check)
    if {k for k in set(config) | set(json.loads(draft)) if config.get(k) != json.loads(draft).get(k)} \
            != {'status', 'decision', 'execution', 'registration'}:
        raise ValueError('Registered config changes more than registration fields')
    if check['valid'] is not True or check['tasks'] != 17 or check['bound_uuids'] != [GPU3] \
            or check['config_sha256'] != p47_launch.UNDATED_SHA \
            or not check['source_sha256'] == reg['preflight']['source_sha256'] == runner.own_hashes():
        raise ValueError('TRAIN-only probe identity differs')
    if check['export_manifest_sha256'] != config['export']['manifest_sha256'] \
            or check['sidecar_sha256'] != config[pid.BATTER_SIDECAR_KEY]['sidecar_sha256']:
        raise ValueError('Export or sidecar pin differs')
    if not config_bytes:
        raise ValueError('Registered config is empty')


def full_payload_check(config, check, export, sidecar):
    """Separate post-registration hash stage; metadata-only probe never calls this."""
    manifest = runner.verify_export(export, config, check['export_manifest_sha256'])
    if runner.sha256_file(sidecar / 'sidecar.json') != check['sidecar_sha256']:
        raise ValueError('Staged batter sidecar differs')
    if len(manifest['splits']) != len(config['export']['splits']):
        raise ValueError('Frozen export is incomplete')


def sealed_family(config, output, manifest_sha, sources):
    if [a['name'] for a in config['arms']] != list(pid.P47_ARM_ORDER) or config['seeds'] != pid.P47_SEEDS:
        raise ValueError('P47 arm/seed set differs')
    for arm in config['arms']:
        for seed in config['seeds']:
            dest = runner.task_dir(output, arm['name'], seed, False)
            runner.verify_task(dest, runner.expected_identity(config, manifest_sha, arm['name'], seed,
                                                                False, 'cuda', sources))
    return pid.P47_FITS


def run(args):
    base = args.base.resolve()
    base.mkdir(parents=True, exist_ok=True)
    state = base / 'score-p47-state.json'
    with state.open('x') as handle:
        json.dump({'status': 'created', 'pid': os.getpid()}, handle)
    child = None

    def interrupted(number, frame):
        raise RuntimeError(f'Pipeline interrupted by signal {number}')

    for number in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(number, interrupted)

    def save(status, **extra):
        temp = state.with_name(state.name + '.tmp')
        temp.write_text(json.dumps({'status': status, 'pid': os.getpid(),
                                    'updated_kst': datetime.now(p47_launch.KST).isoformat(), **extra}, indent=2))
        temp.replace(state)

    def execute(label, command, timeout):
        nonlocal child
        with (base / (label + '.log')).open('x') as log:
            child = subprocess.Popen(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT,
                                     env={**os.environ, 'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2',
                                          'MKL_NUM_THREADS': '2', 'PYTHONUNBUFFERED': '1'}, start_new_session=True)
            save(label, child_pid=child.pid, command=command)
            try:
                code = child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=60)
                raise RuntimeError(label + ' timed out')
            if code:
                raise RuntimeError(label + ' failed: ' + str(code))
        child = None

    try:
        config_bytes, draft, check_bytes = args.config.read_bytes(), args.draft.read_bytes(), args.check.read_bytes()
        config, check = json.loads(config_bytes), json.loads(check_bytes)
        save('checking_registered_identity')
        preflight(config, check, draft, config_bytes, check_bytes, args.source_commit)
        save('hashing_full_payload_after_registration',
             note='This stage opens <=2025 payloads for integrity only; no prediction/metric')
        full_payload_check(config, check, args.export, args.sidecar)
        save('payload_verified_after_registration', splits_hashed=list(config['export']['splits']),
             note='All payload hashes checked; no predictions or metrics read yet')
        queue_timeout, score_by = clock(config['queue'], datetime.now(p47_launch.KST))
        apps = subprocess.run(['nvidia-smi', f'--id={GPU3}', '--query-compute-apps=gpu_uuid,pid',
                               '--format=csv,noheader'], capture_output=True, text=True, check=True).stdout
        if apps.strip():
            raise RuntimeError('GPU3 compute process present')
        output, queue_dir = base / 'runs/EXP-P47-001', base / 'queue-p47-001'
        if output.exists() or queue_dir.exists() or (base / 'scores/EXP-P47-001').exists():
            raise FileExistsError('P47 queue/run/score already exists; manual review required')
        sha = hashlib.sha256(config_bytes).hexdigest()
        command = [args.python, str(PROJECT / 'scripts/cuda_direct_queue.py'), '--config', str(args.config),
                   '--config-sha256', sha, '--export', str(args.export), '--export-manifest-sha256',
                   check['export_manifest_sha256'], '--output', str(output), '--queue-dir', str(queue_dir),
                   '--lock-dir', LOCKS, '--gpu', GPU3, 'run']
        execute('p47-full', command, queue_timeout)
        counts = json.loads((queue_dir / 'summary.json').read_text())['counts']
        if counts != {'done': pid.P47_FITS}:
            save('incomplete', counts=counts, note='No partial scoring')
            return
        sources = runner.own_hashes()
        if sources != check['source_sha256'] or sealed_family(config, output, check['export_manifest_sha256'], sources) != 16:
            raise ValueError('Sixteen matching sealed fits required')
        if datetime.now(p47_launch.KST) > score_by:
            save('incomplete', counts=counts, note='Scorer/report reserve expired; no partial scoring')
            return
        destination = base / 'scores/EXP-P47-001'
        score = [args.python, str(PROJECT / 'scripts/score_cuda_direct.py'), '--config', str(args.config),
                 '--config-sha256', sha, '--export', str(args.export), '--export-manifest-sha256',
                 check['export_manifest_sha256'], '--family', str(output), '--destination', str(destination)]
        execute('p47-score', score, min(SCORE_SECONDS, int((p47_launch.at(config['queue']['date'],
                           config['queue']['aggregate_by']) - datetime.now(p47_launch.KST)).total_seconds()) - REPORT_SECONDS))
        result = json.loads((destination / 'score.json').read_text())
        if result['family'] != 'EXP-P47-001' or result['decision'] is not None:
            raise ValueError('P47 scorer returned unexpected family/verdict')
        save('complete', score=str(destination / 'score.json'))
    except BaseException as error:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        save('blocked', error=repr(error), child_pid=child.pid if child else None)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--draft', type=Path, required=True)
    p.add_argument('--check', type=Path, required=True)
    p.add_argument('--export', type=Path, required=True)
    p.add_argument('--sidecar', type=Path, required=True)
    p.add_argument('--source-commit', required=True)
    p.add_argument('--python', required=True)
    run(p.parse_args())


if __name__ == '__main__':
    main()
