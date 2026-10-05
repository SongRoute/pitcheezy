"""EXP-P40-001 launch helpers (pitcher-ID controls): TRAIN-only smoke validity check and registration.

  # server, from code-<commit>/experiments/pitchmdp, after the 20-task CUDA smoke (DRAFT config):
  python docs/launch/p40_launch.py smoke-check <DRAFT config> <smoke queue dir> <smoke runs dir (-smoke)> <out.json>
  # Mac, repo root:
  python experiments/pitchmdp/docs/launch/p40_launch.py register <smoke-check.json> <registered HH:MM KST> \
      <source commit> <go reference>

smoke-check reads only the queue summary.json and identity.json / fit.json of the 20 smoke tasks (via the runner's
verify_task); never a prediction archive, a DEV / mlb_dev metric or 2026 data. Exit 0 = valid, 3 = refused (summary
still written, NEW file). Validity: 20/20 done, identities, bound UUID set == {GPU3}, no released GPU,
dev_used_for_fit_or_selection False, the three id arms carry ONE TRAIN-only vocabulary record, the row-shuffle and
ID-dropout arms record their fixed mode, the reference none. Per-arm median train seconds / epoch is recorded
DESCRIPTIVELY (no ratio gate: the id arms share the P38 candidate path measured 1.0729 in P38 v2).

register writes configs/EXP-P40-001.json: only status / decision / execution / registration differ from the DRAFT;
registered HH:MM on queue.date, <= launch_by, not in the future and within 15 min of the real clock;
registered + P40 projection (6,117.5 s) <= no_new_fit_after.
"""
import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
import statistics
import sys

PROJECT = Path(__file__).resolve().parents[2]
REPO = PROJECT.parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'scripts')]
from pitchmdp import direct_pitcher_id_registry as pid  # noqa: E402

SMOKE_PROTOCOL = 'p40_train_smoke_validity_v1'
DRAFT = REPO / 'configs' / 'DRAFT-CUDA-PITCHER-ID-CONTROLS-20261006.json'
DRAFT_SHA = '3dba5445b3a76dbc5f19c73f1cd60f6a69fd18312c775a6f82e06db656ab6944'
MANIFEST_SHA = '49097109b4f522a5a94d1557e00c52d88a8089c725977fe263c592214fb1f578'
PREREG = 'experiments/pitchmdp/docs/PREREG-DRAFT-EXP-P40-001-pitcher-id-controls-2026-10-06.md'
OUT = REPO / 'configs' / 'EXP-P40-001.json'
KST = timezone(timedelta(hours=9))
CLOCK_SKEW = timedelta(minutes=15)
MODES = {pid.CANDIDATE: {'index_source': None, 'shuffle_seed': None, 'id_dropout': None},
         pid.SHUFFLE: {'index_source': 'row_shuffle', 'shuffle_seed': pid.SHUFFLE_SEED, 'id_dropout': 0.},
         pid.DROPOUT: {'index_source': 'identity', 'shuffle_seed': None, 'id_dropout': pid.ID_DROPOUT}}


def train_epoch_seconds(fit):
    report = fit.get('report') or {}
    seconds, epochs = report.get('seconds'), report.get('epochs_run')
    if not isinstance(seconds, (int, float)) or isinstance(seconds, bool) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(f'report.seconds missing or invalid: {seconds!r}')
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs <= 0:
        raise ValueError(f'report.epochs_run missing or zero: {epochs!r}')
    return seconds / epochs


def evaluate(fits, allowed_uuids=(pid.RETAINED_GPU,)):
    """fits: {arm: {seed: fit.json dict}} -> validity record (pure; no IO)."""
    errors, per_arm, uuids, vocab = [], {}, set(), set()
    for arm in pid.P40_ARM_ORDER:
        values = []
        for seed in pid.P40_SEEDS:
            fit = (fits.get(arm) or {}).get(seed)
            if fit is None:
                errors.append(f'{arm}/{seed}: fit.json missing')
                continue
            uuids.add((fit.get('runtime') or {}).get('bound_uuid'))
            if fit.get('dev_used_for_fit_or_selection') is not False:
                errors.append(f'{arm}/{seed}: dev_used_for_fit_or_selection is not False')
            record = fit.get(pid.FLAG)
            if arm == pid.REFERENCE:
                if record is not None:
                    errors.append(f'{arm}/{seed}: reference carries a pitcher-ID record')
            elif not isinstance(record, dict) or not isinstance(record.get('n_vocab'), int) or record['n_vocab'] < 0 \
                    or record.get('train_max_date') is None or int(record['train_max_date']) > pid.LAST_TRAIN_DATE:
                errors.append(f'{arm}/{seed}: TRAIN vocabulary record missing or after the cutoff')
            else:
                vocab.add(json.dumps({k: record.get(k) for k in pid.VOCAB_KEYS}, sort_keys=True))
                mode = {k: record.get(k) for k in ('index_source', 'shuffle_seed', 'id_dropout')}
                if mode != MODES[arm]:
                    errors.append(f'{arm}/{seed}: id mode record {mode} != {MODES[arm]}')
            try:
                values.append(train_epoch_seconds(fit))
            except ValueError as error:
                errors.append(f'{arm}/{seed}: {error}')
        per_arm[arm] = statistics.median(values) if len(values) == len(pid.P40_SEEDS) else None
    if len(vocab) > 1:
        errors.append('id-arm smoke fits carry different vocabulary records (must be TRAIN-only, seed-free)')
    if uuids != set(allowed_uuids):
        errors.append(f'smoke bound UUID set {sorted(map(str, uuids))} != allowed {sorted(allowed_uuids)}')
    if uuids & set(pid.RELEASED_GPUS):
        errors.append('smoke used a released GPU')
    ref = per_arm.get(pid.REFERENCE)
    return {'protocol': SMOKE_PROTOCOL, 'valid': not errors, 'errors': errors,
            'train_seconds_per_epoch_median': per_arm,
            'over_reference_descriptive': {a: (v / ref if v is not None and ref else None) for a, v in per_arm.items()},
            'bound_uuids': sorted(map(str, uuids)),
            'smoke_vocabulary': json.loads(next(iter(vocab))) if len(vocab) == 1 else None,
            'tasks': sum(len(fits.get(a) or {}) for a in pid.P40_ARM_ORDER),
            'family_projection_seconds': pid.P40_FAMILY_PROJECTION_SECONDS,
            'note': 'TRAIN-only smoke (8,192 rows x smoke epochs): validity only, no resource gate; projection fixed '
                    'from P38 actual task walls in the registry; no prediction / DEV / 2026 read'}


def smoke_check(argv):
    draft, queue_dir, runs, out = map(Path, argv)
    assert not out.exists(), 'smoke-check summary already exists'
    assert runs.name.endswith('-smoke'), 'smoke runs dir must end -smoke'
    assert hashlib.sha256(draft.read_bytes()).hexdigest() == DRAFT_SHA, 'DRAFT config sha differs'
    import run_cuda_direct as runner
    config = json.loads(draft.read_text())
    assert config['device'] == pid.DEVICE and config['seeds'] == pid.P40_SEEDS
    assert [a['name'] for a in config['arms']] == list(pid.P40_ARM_ORDER)
    counts = json.loads((queue_dir / 'summary.json').read_text())['counts']
    sources = runner.own_hashes()
    fits, identity_errors = {}, []
    for arm in pid.P40_ARM_ORDER:
        for seed in pid.P40_SEEDS:
            d = runner.task_dir(runs, arm, seed, True)
            try:
                runner.verify_task(d, runner.expected_identity(config, MANIFEST_SHA, arm, seed, True, 'cuda', sources))
                fits.setdefault(arm, {})[seed] = json.loads((d / 'fit.json').read_text())
            except Exception as error:  # noqa: BLE001 - every identity failure is a refusal
                identity_errors.append(f'{arm}/{seed}: identity {error!r}')
    record = evaluate(fits, config['device']['allowed_gpu_uuids'])
    if counts != {'done': pid.P40_FITS}:
        record['errors'].append(f'smoke queue counts {counts} != done {pid.P40_FITS}')
    record['errors'] += identity_errors
    record['valid'] = not record['errors']
    record.update(config_sha256=DRAFT_SHA, export_manifest_sha256=MANIFEST_SHA, smoke_queue=str(queue_dir),
                  smoke_runs=str(runs), source_sha256=sources)
    with out.open('x') as handle:
        handle.write(json.dumps(record, indent=2, sort_keys=True) + '\n')
    print('P40_SMOKE_CHECK', 'VALID' if record['valid'] else 'REFUSED', hashlib.sha256(out.read_bytes()).hexdigest(),
          json.dumps(record['errors'][:5]))
    return 0 if record['valid'] else 3


def at(day, hhmm):
    return datetime.strptime(day + ' ' + hhmm, '%Y-%m-%d %H:%M').replace(tzinfo=KST)


def build_registered(draft_text, smoke, smoke_sha, registered_hhmm, commit, go_reference, now):
    if hashlib.sha256(draft_text.encode()).hexdigest() != DRAFT_SHA:
        raise ValueError('DRAFT config differs from its pinned sha')
    if smoke.get('protocol') != SMOKE_PROTOCOL or smoke.get('valid') is not True or smoke.get('errors') != []:
        raise ValueError('smoke check is not valid')
    if smoke.get('config_sha256') != DRAFT_SHA or smoke.get('export_manifest_sha256') != MANIFEST_SHA \
            or smoke.get('tasks') != pid.P40_FITS or smoke.get('bound_uuids') != [pid.RETAINED_GPU]:
        raise ValueError('smoke check identity (DRAFT / manifest / 20 tasks / {GPU3}) differs')
    if smoke.get('family_projection_seconds') != pid.P40_FAMILY_PROJECTION_SECONDS:
        raise ValueError('projection constant differs')
    config = json.loads(draft_text)
    q = config['queue']
    registered = at(q['date'], registered_hhmm)
    if registered > at(q['date'], q['launch_by']):
        raise ValueError('registration after launch_by: re-date the DRAFT first')
    if registered > now + timedelta(minutes=1) or now - registered > CLOCK_SKEW:
        raise ValueError('registration time must be the real clock (not future, within 15 min)')
    if registered + timedelta(seconds=pid.P40_FAMILY_PROJECTION_SECONDS) > at(q['date'], q['no_new_fit_after']):
        raise ValueError('projection does not end before no_new_fit_after')
    if len(commit) < 7 or not all(c in '0123456789abcdef' for c in commit):
        raise ValueError('source commit must be a hex sha')
    config.update(status='REGISTERED',
                  decision='None. Registered for execution only; descriptive repeated-DEV contrasts, no verdict.',
                  execution={'enabled': True, 'executable': True},
                  registration={'registered_kst': q['date'] + ' ' + registered_hhmm, 'draft_sha256': DRAFT_SHA,
                                'source_commit': commit, 'go_reference': go_reference, 'prereg': PREREG,
                                'smoke': {'protocol': SMOKE_PROTOCOL, 'check_sha256': smoke_sha,
                                          'train_seconds_per_epoch_median': smoke['train_seconds_per_epoch_median'],
                                          'source_sha256': smoke['source_sha256']}})
    return config


def register(argv):
    smoke_path, hhmm, commit, go_reference = argv
    smoke_bytes = Path(smoke_path).read_bytes()
    config = build_registered(DRAFT.read_text(), json.loads(smoke_bytes), hashlib.sha256(smoke_bytes).hexdigest(),
                              hhmm, commit, go_reference, datetime.now(KST))
    text = json.dumps(config, indent=2, ensure_ascii=False) + '\n'
    with OUT.open('x') as handle:
        handle.write(text)
    print(OUT.relative_to(REPO), hashlib.sha256(text.encode()).hexdigest())
    return 0


if __name__ == '__main__':
    command, *rest = sys.argv[1:]
    sys.exit({'smoke-check': smoke_check, 'register': register}[command](rest))
