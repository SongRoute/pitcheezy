"""EXP-P43-001 launch helpers (batter-ID family): dated-DRAFT check, TRAIN-only smoke validity check and registration.

The committed DRAFT (configs/DRAFT-CUDA-BATTER-ID-20261006.json, sha UNDATED_SHA) has no queue window because P40 /
P42 hold GPU3.  When GPU3 is free the GPU lead writes a dated copy with make_p43_config.py --clock (only the five queue
clock fields change; `dated-check` proves it), smokes THAT file, then registers it.

  # Mac, repo root (dated DRAFT, a NEW file name, e.g. configs/DRAFT-CUDA-BATTER-ID-20261006-dated-<HHMM>.json):
  python experiments/pitchmdp/docs/launch/p43_launch.py dated-check <dated DRAFT>
  # server, from code-<commit>/experiments/pitchmdp, after the 20-task CUDA smoke on the dated DRAFT:
  python docs/launch/p43_launch.py smoke-check <dated DRAFT> <smoke queue dir> <smoke runs dir (-smoke)> <out.json>
  # Mac, repo root:
  python experiments/pitchmdp/docs/launch/p43_launch.py register <dated DRAFT> <smoke-check.json> <HH:MM KST> \
      <source commit> <go reference>

smoke-check reads only the queue summary.json and identity.json / fit.json of the 20 smoke tasks (runner verify_task);
never a prediction archive, a DEV / mlb_dev metric or 2026 data.  Validity: 20/20 done, identities, bound UUID set ==
{GPU3}, no released GPU, dev_used_for_fit_or_selection False, every new id arm names exactly its registry columns and
the pitcher-only arm carries the unmoded P38 record, with ONE TRAIN-only vocabulary record per column across arms and
seeds; the row-shuffle arm records its fixed seed, the reference none.  Works for EXP-P43-001 (seeds 70-72, 15 tasks)
and the clock-only extension EXP-P43-002 (seeds 73-74, 10 tasks); the design key picks the family.  Per-arm median train seconds / epoch is DESCRIPTIVE (no ratio gate).
register writes configs/EXP-P43-001.json (or EXP-P43-002.json): only status / decision / execution / registration differ from the dated
DRAFT; registered HH:MM on queue.date, <= launch_by, real clock (not future, within 15 min), + projection <=
no_new_fit_after.
"""
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
import runpy
import statistics
import sys

PROJECT = Path(__file__).resolve().parents[2]
REPO = PROJECT.parents[1]
LAUNCH = Path(__file__).resolve().parent
sys.path[:0] = [str(PROJECT), str(PROJECT / 'scripts'), str(LAUNCH)]
from pitchmdp import direct_pitcher_id_registry as pid  # noqa: E402
from p40_launch import KST, at, train_epoch_seconds  # noqa: E402  same clock / epoch-seconds rules as P40

SMOKE_PROTOCOL = 'p43_train_smoke_validity_v1'
# design key -> (undated DRAFT, its sha, experiment id, seeds, fits, projection, registered config path)
FAMILIES = {
    pid.P43_DESIGN: (REPO / 'configs' / 'DRAFT-CUDA-BATTER-ID-20261006.json',
                     'bc94ae86a2482c920ebd3cb8a898774bcbbb4c5862d58639556621f4b7a442e2', 'EXP-P43-001',
                     pid.P43_SEEDS, pid.P43_FITS, pid.P43_FAMILY_PROJECTION_SECONDS, REPO / 'configs' / 'EXP-P43-001.json'),
    pid.P43_EXT_DESIGN: (REPO / 'configs' / 'DRAFT-CUDA-BATTER-ID-EXT-20261006.json',
                         '389c81b369a206460202aeda8b683a66da5df3e565dda0c380417e61f998782a', 'EXP-P43-002',
                         pid.P43_EXT_SEEDS, pid.P43_EXT_FITS, pid.P43_EXT_PROJECTION_SECONDS,
                         REPO / 'configs' / 'EXP-P43-002.json')}
MANIFEST_SHA = '49097109b4f522a5a94d1557e00c52d88a8089c725977fe263c592214fb1f578'
PREREG = 'experiments/pitchmdp/docs/PREREG-DRAFT-EXP-P43-001-batter-id-2026-10-06.md'
CLOCK_FIELDS = ('date', 'launch_by', 'no_new_fit_after', 'kill_own_after', 'aggregate_by')
CLOCK_SKEW = timedelta(minutes=15)
MODES = {pid.BATTER: (['batter'], 'identity', None), pid.BATTER_SHUFFLE: (['batter'], 'row_shuffle',
                                                                         pid.BATTER_SHUFFLE_SEED),
         pid.PITCHER_BATTER: (['pitcher', 'batter'], 'identity', None)}  # the pitcher-only arm keeps the P38 record


def family(config):
    key = config.get(pid.DESIGN_KEY)
    if key not in FAMILIES:
        raise ValueError('Not a P43 design: ' + str(key))
    return FAMILIES[key]


def dated_check(text):
    """The dated DRAFT differs from the committed undated DRAFT only in the five queue clock fields, and those equal
    make_p43_config.clock(date, launch_by) (deterministic window; no other field may move)."""
    dated = json.loads(text)
    undated, undated_sha, *_ = family(dated)
    if hashlib.sha256(undated.read_bytes()).hexdigest() != undated_sha:
        raise ValueError('Undated DRAFT differs from its pinned sha')
    base = json.loads(undated.read_text())
    if {k for k in set(base) | set(dated) if base.get(k) != dated.get(k)} - {'queue'}:
        raise ValueError('Dated DRAFT changes more than the queue')
    moved = {k for k in set(base['queue']) | set(dated['queue']) if base['queue'].get(k) != dated['queue'].get(k)}
    if moved != set(CLOCK_FIELDS):
        raise ValueError(f'Dated DRAFT must change exactly the clock fields, changed {sorted(moved)}')
    q = dated['queue']
    window = runpy.run_path(str(LAUNCH / 'make_p43_config.py'))['clock'](
        q['date'], q['launch_by'], extension=dated[pid.DESIGN_KEY] == pid.P43_EXT_DESIGN)
    if {k: q[k] for k in CLOCK_FIELDS} != window:
        raise ValueError('Dated clock is not the make_p43_config window for its date / launch_by')
    return dated


def evaluate(fits, allowed_uuids=(pid.RETAINED_GPU,), design=pid.P43_DESIGN):
    """fits: {arm: {seed: fit.json dict}} -> validity record (pure; no IO)."""
    _, _, _, seeds, n_fits, projection, _ = FAMILIES[design]
    errors, per_arm, uuids, vocab = [], {}, set(), {}
    for arm in pid.P43_ARM_ORDER:
        values = []
        for seed in seeds:
            fit = (fits.get(arm) or {}).get(seed)
            if fit is None:
                errors.append(f'{arm}/{seed}: fit.json missing')
                continue
            uuids.add((fit.get('runtime') or {}).get('bound_uuid'))
            if fit.get('dev_used_for_fit_or_selection') is not False:
                errors.append(f'{arm}/{seed}: dev_used_for_fit_or_selection is not False')
            block = fit.get(pid.FLAG)
            if arm == pid.REFERENCE:
                if block is not None:
                    errors.append(f'{arm}/{seed}: reference carries an id record')
            elif arm == pid.CANDIDATE:  # P38 entry verbatim: P38 record (one pitcher vocabulary at top level)
                if not isinstance(block, dict) or 'vocabs' in block or not isinstance(block.get('n_vocab'), int) \
                        or block.get('train_max_date') is None or int(block['train_max_date']) > pid.LAST_TRAIN_DATE \
                        or block.get('index_source') is not None or block.get('id_dropout') is not None:
                    errors.append(f'{arm}/{seed}: P38 pitcher vocabulary record missing, moded or after the cutoff')
                else:
                    vocab.setdefault('pitcher', set()).add(json.dumps({k: block.get(k) for k in pid.VOCAB_KEYS},
                                                                      sort_keys=True))
            else:
                columns, source, shuffle = MODES[arm]
                if not isinstance(block, dict) or block.get('id_columns') != columns \
                        or set(block.get('vocabs') or {}) != set(columns):
                    errors.append(f'{arm}/{seed}: id record does not name exactly {columns}')
                elif (block.get('index_source'), block.get('shuffle_seed')) != (source, shuffle):
                    errors.append(f'{arm}/{seed}: id mode {block.get("index_source"), block.get("shuffle_seed")}')
                else:
                    for c in columns:
                        record = block['vocabs'][c]
                        if not isinstance(record.get('n_vocab'), int) or record.get('train_max_date') is None \
                                or int(record['train_max_date']) > pid.LAST_TRAIN_DATE:
                            errors.append(f'{arm}/{seed}: {c} TRAIN vocabulary record missing or after the cutoff')
                        vocab.setdefault(c, set()).add(json.dumps({k: record.get(k) for k in pid.VOCAB_KEYS},
                                                                  sort_keys=True))
            try:
                values.append(train_epoch_seconds(fit))
            except ValueError as error:
                errors.append(f'{arm}/{seed}: {error}')
        per_arm[arm] = statistics.median(values) if len(values) == len(seeds) else None
    if set(vocab) != {'pitcher', 'batter'} or any(len(v) != 1 for v in vocab.values()):
        errors.append('smoke fits must carry one TRAIN-only vocabulary record per id column (pitcher, batter)')
    if uuids != set(allowed_uuids):
        errors.append(f'smoke bound UUID set {sorted(map(str, uuids))} != allowed {sorted(allowed_uuids)}')
    if uuids & set(pid.RELEASED_GPUS):
        errors.append('smoke used a released GPU')
    ref = per_arm.get(pid.REFERENCE)
    return {'protocol': SMOKE_PROTOCOL, 'valid': not errors, 'errors': errors,
            'train_seconds_per_epoch_median': per_arm,
            'over_reference_descriptive': {a: (v / ref if v is not None and ref else None) for a, v in per_arm.items()},
            'bound_uuids': sorted(map(str, uuids)),
            'smoke_vocabulary': {c: json.loads(next(iter(v))) for c, v in vocab.items() if len(v) == 1},
            'tasks': sum(len(fits.get(a) or {}) for a in pid.P43_ARM_ORDER), 'expected_tasks': n_fits,
            'design': design, 'family_projection_seconds': projection,
            'note': 'TRAIN-only smoke (8,192 rows x smoke epochs): validity only, no resource gate; projection fixed '
                    'from P38 actual task walls in the registry; no prediction / DEV / 2026 read'}


def smoke_check(argv):
    draft, queue_dir, runs, out = map(Path, argv)
    assert not out.exists(), 'smoke-check summary already exists'
    assert runs.name.endswith('-smoke'), 'smoke runs dir must end -smoke'
    text = draft.read_text()
    config = dated_check(text)
    draft_sha = hashlib.sha256(text.encode()).hexdigest()
    import run_cuda_direct as runner
    _, undated_sha, _, seeds, n_fits, _, _ = family(config)
    assert config['device'] == pid.DEVICE and config['seeds'] == seeds
    assert [a['name'] for a in config['arms']] == list(pid.P43_ARM_ORDER)
    counts = json.loads((queue_dir / 'summary.json').read_text())['counts']
    sources = runner.own_hashes()
    fits, identity_errors = {}, []
    for arm in pid.P43_ARM_ORDER:
        for seed in seeds:
            d = runner.task_dir(runs, arm, seed, True)
            try:
                runner.verify_task(d, runner.expected_identity(config, MANIFEST_SHA, arm, seed, True, 'cuda', sources))
                fits.setdefault(arm, {})[seed] = json.loads((d / 'fit.json').read_text())
            except Exception as error:  # noqa: BLE001 - every identity failure is a refusal
                identity_errors.append(f'{arm}/{seed}: identity {error!r}')
    record = evaluate(fits, config['device']['allowed_gpu_uuids'], config[pid.DESIGN_KEY])
    if counts != {'done': n_fits}:
        record['errors'].append(f'smoke queue counts {counts} != done {n_fits}')
    record['errors'] += identity_errors
    record['valid'] = not record['errors']
    record.update(config_sha256=draft_sha, undated_draft_sha256=undated_sha, export_manifest_sha256=MANIFEST_SHA,
                  batter_sidecar_sha256=config[pid.BATTER_SIDECAR_KEY]['sidecar_sha256'], smoke_queue=str(queue_dir),
                  smoke_runs=str(runs), source_sha256=sources)
    with out.open('x') as handle:
        handle.write(json.dumps(record, indent=2, sort_keys=True) + '\n')
    print('P43_SMOKE_CHECK', 'VALID' if record['valid'] else 'REFUSED', hashlib.sha256(out.read_bytes()).hexdigest(),
          json.dumps(record['errors'][:5]))
    return 0 if record['valid'] else 3


def build_registered(draft_text, smoke, smoke_sha, registered_hhmm, commit, go_reference, now):
    config = dated_check(draft_text)
    _, undated_sha, _, _, n_fits, projection, _ = family(config)
    draft_sha = hashlib.sha256(draft_text.encode()).hexdigest()
    if smoke.get('protocol') != SMOKE_PROTOCOL or smoke.get('valid') is not True or smoke.get('errors') != []:
        raise ValueError('smoke check is not valid')
    if smoke.get('config_sha256') != draft_sha or smoke.get('export_manifest_sha256') != MANIFEST_SHA \
            or smoke.get('tasks') != n_fits or smoke.get('bound_uuids') != [pid.RETAINED_GPU] \
            or smoke.get('batter_sidecar_sha256') != config[pid.BATTER_SIDECAR_KEY]['sidecar_sha256']:
        raise ValueError('smoke check identity (dated DRAFT / manifest / sidecar / task count / {GPU3}) differs')
    if smoke.get('family_projection_seconds') != projection or smoke.get('design') != config[pid.DESIGN_KEY]:
        raise ValueError('projection constant differs')
    q = config['queue']
    registered = at(q['date'], registered_hhmm)
    if registered > at(q['date'], q['launch_by']):
        raise ValueError('registration after launch_by: re-date the DRAFT first')
    if registered > now + timedelta(minutes=1) or now - registered > CLOCK_SKEW:
        raise ValueError('registration time must be the real clock (not future, within 15 min)')
    if registered + timedelta(seconds=projection) > at(q['date'], q['no_new_fit_after']):
        raise ValueError('projection does not end before no_new_fit_after')
    if len(commit) < 7 or not all(c in '0123456789abcdef' for c in commit):
        raise ValueError('source commit must be a hex sha')
    config.update(status='REGISTERED',
                  decision='None. Registered for execution only; descriptive repeated-DEV contrasts, no verdict.',
                  execution={'enabled': True, 'executable': True},
                  registration={'registered_kst': q['date'] + ' ' + registered_hhmm, 'draft_sha256': draft_sha,
                                'undated_draft_sha256': undated_sha, 'source_commit': commit,
                                'go_reference': go_reference, 'prereg': PREREG,
                                'smoke': {'protocol': SMOKE_PROTOCOL, 'check_sha256': smoke_sha,
                                          'train_seconds_per_epoch_median': smoke['train_seconds_per_epoch_median'],
                                          'source_sha256': smoke['source_sha256']}})
    return config


def register(argv):
    draft, smoke_path, hhmm, commit, go_reference = argv
    smoke_bytes = Path(smoke_path).read_bytes()
    config = build_registered(Path(draft).read_text(), json.loads(smoke_bytes),
                              hashlib.sha256(smoke_bytes).hexdigest(), hhmm, commit, go_reference, datetime.now(KST))
    text = json.dumps(config, indent=2, ensure_ascii=False) + '\n'
    out = family(config)[6]
    with out.open('x') as handle:
        handle.write(text)
    print(out.relative_to(REPO), hashlib.sha256(text.encode()).hexdigest())
    return 0


def dated(argv):
    (path,) = argv
    text = Path(path).read_text()
    dated_check(text)
    print('P43_DATED_OK', hashlib.sha256(text.encode()).hexdigest())
    return 0


if __name__ == '__main__':
    command, *rest = sys.argv[1:]
    sys.exit({'dated-check': dated, 'smoke-check': smoke_check, 'register': register}[command](rest))
