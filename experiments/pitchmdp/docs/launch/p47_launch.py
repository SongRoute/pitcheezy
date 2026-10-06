"""P47 dated DRAFT and registration gates. No real data or GPU access in this module."""
import argparse
import hashlib
import json
import math
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[2]
REPO = PROJECT.parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'scripts')]
import run_cuda_direct as runner
from p40_launch import KST, at
from pitchmdp import direct_pitcher_id_registry as pid

UNDATED = REPO / 'configs/DRAFT-CUDA-BATTER-DROPOUT-20261006.json'
UNDATED_SHA = 'b2ca6336bef81ce4db22272da7a70c0f15fb7e1c605e763a83a07309f474c763'
MANIFEST_SHA = '49097109b4f522a5a94d1557e00c52d88a8089c725977fe263c592214fb1f578'
PREREG = 'experiments/pitchmdp/docs/PREREG-DRAFT-EXP-P47-001-batter-dropout-2026-10-06.md'
CLOCK_FIELDS = ('date', 'launch_by', 'no_new_fit_after', 'kill_own_after', 'aggregate_by',
                'preflight_seconds_measured')
EXECUTION_FILES = ('scripts/cuda_direct_queue.py', 'scripts/score_cuda_direct.py',
                   'docs/launch/p47_launch.py', 'docs/launch/run_p47_pipeline.py')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def execution_hashes():
    return {name: sha((PROJECT / name).read_bytes()) for name in EXECUTION_FILES}


def clock(day, launch_by, preflight_seconds):
    if not isinstance(preflight_seconds, (int, float)) or isinstance(preflight_seconds, bool) \
            or not math.isfinite(preflight_seconds) or preflight_seconds <= 0:
        raise ValueError('Measured preflight seconds required')
    start = at(day, launch_by)
    no_new = start + timedelta(seconds=pid.P47_FAMILY_PROJECTION_SECONDS + preflight_seconds)
    no_new += timedelta(seconds=-no_new.second, microseconds=-no_new.microsecond)
    if no_new < start + timedelta(seconds=pid.P47_FAMILY_PROJECTION_SECONDS + preflight_seconds):
        no_new += timedelta(minutes=1)
    no_new += timedelta(minutes=-no_new.minute % 5)
    kill = no_new + timedelta(minutes=15)
    aggregate = kill + timedelta(minutes=20)
    if aggregate.date() != start.date():
        raise ValueError('P47 window must finish on the launch date')
    return {'date': day, 'launch_by': launch_by, 'no_new_fit_after': no_new.strftime('%H:%M'),
            'kill_own_after': kill.strftime('%H:%M'), 'aggregate_by': aggregate.strftime('%H:%M'),
            'preflight_seconds_measured': preflight_seconds}


def dated(base_text, day, launch_by, check):
    if sha(base_text.encode()) != UNDATED_SHA:
        raise ValueError('Undated P47 DRAFT differs from pin')
    if check.get('protocol') != 'p47_train_earlystop_probe_v1' or check.get('valid') is not True \
            or check.get('errors') != [] or check.get('config_sha256') != UNDATED_SHA \
            or check.get('tasks') != 17 or check.get('bound_uuids') != [pid.RETAINED_GPU]:
        raise ValueError('P47 TRAIN-only probe has not passed')
    config = json.loads(base_text)
    config['queue'].update(clock(day, launch_by, check['preflight_seconds_measured']))
    return config


def dated_check(text, check):
    if sha(UNDATED.read_bytes()) != UNDATED_SHA:
        raise ValueError('Undated DRAFT pin differs')
    value = json.loads(text)
    expect = dated(UNDATED.read_text(), value['queue']['date'], value['queue']['launch_by'], check)
    if value != expect or {k for k in value['queue'] if value['queue'][k] != json.loads(UNDATED.read_text())['queue'].get(k)} \
            != set(CLOCK_FIELDS):
        raise ValueError('Dated P47 DRAFT differs beyond the measured clock fields')
    return value


def build_registered(draft_text, check, check_sha, hhmm, source_commit, go_reference, ledger_review, now):
    value = dated_check(draft_text, check)
    if not go_reference or not ledger_review or not Path(go_reference).is_file() or not Path(ledger_review).is_file():
        raise ValueError('PM acceptance and global seed/ID ledger reservation required')
    reservation = json.loads(Path(ledger_review).read_text())
    if reservation.get('experiment_id') != 'EXP-P47-001' or reservation.get('seeds') != pid.P47_SEEDS \
            or reservation.get('reserved') is not True:
        raise ValueError('Global ledger must reserve EXP-P47-001 and seeds 80-83')
    checked = datetime.fromisoformat(reservation['checked_kst'])
    if checked.tzinfo is None or checked > now or now - checked > timedelta(minutes=15):
        raise ValueError('Global ledger reservation must be rechecked within 15 minutes of registration')
    if check.get('source_sha256') is None or check.get('export_manifest_sha256') != MANIFEST_SHA \
            or check.get('sidecar_sha256') != value[pid.BATTER_SIDECAR_KEY]['sidecar_sha256']:
        raise ValueError('Probe source/export/sidecar pins differ')
    if check['source_sha256'] != runner.own_hashes():
        raise ValueError('Registration source differs from the reviewed probe source map')
    if len(source_commit) < 7 or any(c not in '0123456789abcdef' for c in source_commit):
        raise ValueError('Source commit must be a hex SHA')
    q = value['queue']
    registered = at(q['date'], hhmm)
    if registered > at(q['date'], q['launch_by']) or registered > now + timedelta(minutes=1) \
            or now - registered > timedelta(minutes=15):
        raise ValueError('Registration must be current and before launch_by')
    if registered + timedelta(seconds=pid.P47_FAMILY_PROJECTION_SECONDS + q['preflight_seconds_measured']) \
            > at(q['date'], q['no_new_fit_after']):
        raise ValueError('Measured preflight plus projected fits exceed no-new-fit deadline')
    if not (at(q['date'], q['no_new_fit_after']) + timedelta(minutes=15) <= at(q['date'], q['kill_own_after'])
            and at(q['date'], q['kill_own_after']) + timedelta(minutes=20) <= at(q['date'], q['aggregate_by'])):
        raise ValueError('Stop grace or scorer/report reserve missing')
    if q['task_timeout_seconds'] != 3600 or q['family_budget_seconds'] != 36000 or q['max_attempts'] != 2:
        raise ValueError('Inherited task/family/attempt caps differ')
    value.update(status='REGISTERED', decision='None. Registered exploratory repeated-DEV execution; no verdict.',
                 execution={'enabled': True, 'executable': True},
                 registration={'registered_kst': q['date'] + ' ' + hhmm,
                               'draft_sha256': sha(draft_text.encode()), 'undated_draft_sha256': UNDATED_SHA,
                               'source_commit': source_commit, 'go_reference': go_reference,
                               'go_reference_sha256': sha(Path(go_reference).read_bytes()),
                               'execution_code_sha256': execution_hashes(),
                               'seed_id_ledger_review': {'path': ledger_review,
                                                          'sha256': sha(Path(ledger_review).read_bytes())},
                               'seed_rule_correction': 'Historical P47 DRAFT cites P44 seeds 75-79 as reserved; '
                                                       'P44 actual DRAFT uses 49-53. This registration relies on '
                                                       'the separately checked global ledger reservation for 80-83.',
                               'prereg': PREREG,
                               'preflight': {'protocol': check['protocol'], 'check_sha256': check_sha,
                                             'source_sha256': check['source_sha256'],
                                             'probe_code_sha256': check['probe_code_sha256'],
                                             'measured_seconds': check['preflight_seconds_measured']},
                               'payload_hashing': 'main pipeline after registration; no DEV payload read in preflight'})
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    d = sub.add_parser('date')
    d.add_argument('day'); d.add_argument('launch_by'); d.add_argument('check', type=Path); d.add_argument('out', type=Path)
    r = sub.add_parser('register')
    r.add_argument('draft', type=Path); r.add_argument('check', type=Path); r.add_argument('hhmm')
    r.add_argument('source_commit'); r.add_argument('go_reference'); r.add_argument('ledger_review')
    r.add_argument('out', type=Path)
    args = p.parse_args()
    check_bytes = args.check.read_bytes()
    check = json.loads(check_bytes)
    if args.command == 'date':
        value = dated(UNDATED.read_text(), args.day, args.launch_by, check)
    else:
        head = subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], capture_output=True,
                              text=True, check=True).stdout.strip()
        if head != args.source_commit:
            raise ValueError('Registration source commit must be this checkout HEAD')
        value = build_registered(args.draft.read_text(), check, sha(check_bytes), args.hhmm, args.source_commit,
                                 args.go_reference, args.ledger_review, datetime.now(KST))
    with args.out.open('x') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
    print(args.out, sha(args.out.read_bytes()))


if __name__ == '__main__':
    main()
