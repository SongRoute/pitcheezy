"""EXP-P37-001 TRAIN-only smoke gate (pre-registration section 4, review 6b66436 M1). Server-side, read-only on the smoke.

Run from the fresh code dir (code-<commit>/experiments/pitchmdp) after the 15-task CUDA smoke:
  python docs/launch/p37_smoke_gate.py <DRAFT config> <smoke queue dir> <smoke runs dir (ends -smoke)> <out summary.json>
Reads only queue summary.json, identity.json/fit.json (via the runner's verify_task) of the 15 smoke tasks; never a
prediction archive, a DEV/mlb_dev metric or 2026 data. Exit 0 = pass, 3 = gate refused (summary still written), other =
error. The summary is a NEW file ('x'); its sha256 is printed and goes into registration.smoke.gate_summary_sha256.

Gate (fixed before any smoke): per arm, the median over the 5 seeds of report.seconds / report.epochs_run (training
loop only; fit wall = fit_seconds incl. export load/hash is recorded separately and never gated). Stop if
linear_residual_counts / periodic_residual_counts > 1.5, or any record is missing, nonfinite, nonpositive or has 0 epochs.
The smoke is 8,192 TRAIN rows x 2 epochs: it is NOT a cap-60 measurement. The family projection below is fixed from
P36 actuals (extract-p36.json @ ce2861b, per-arm task wall max_s), not from the smoke.
"""
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'scripts')]

PROTOCOL = 'p37_train_smoke_gate_v1'
DRAFT_SHA = '75265fc715f0fec998f609f0316ffe1de486c609677588d0fbd7241fc427ace2'
MANIFEST_SHA = '49097109b4f522a5a94d1557e00c52d88a8089c725977fe263c592214fb1f578'
RATIO_MAX = 1.5  # linear / periodic median training seconds per epoch
# Conservative per-task wall bound (s): 2 x the P36 per-arm max task wall at cap 60 (control 100.13, periodic 235.20;
# P36 epochs_run 10-13, so 2x covers up to ~20-26 epochs); linear = RATIO_MAX x periodic bound (the gate maximum).
P36_MAX_TASK_SECONDS = {'control': 100.13, 'periodic_residual_counts': 235.20}
SAFETY = 2.0
TASK_BOUND_SECONDS = {'control': SAFETY * P36_MAX_TASK_SECONDS['control'],
                      'periodic_residual_counts': SAFETY * P36_MAX_TASK_SECONDS['periodic_residual_counts'],
                      'linear_residual_counts': RATIO_MAX * SAFETY * P36_MAX_TASK_SECONDS['periodic_residual_counts']}
SEEDS_PER_ARM = 5
FAMILY_PROJECTION_SECONDS = SEEDS_PER_ARM * sum(TASK_BOUND_SECONDS.values())  # serial, one GPU: 6,881.4 s
SCORER_TIMEOUT_SECONDS = 900


def train_epoch_seconds(fit):
    """report.seconds / report.epochs_run of one fit; fail closed on missing, nonfinite, nonpositive or 0 epochs."""
    report = fit.get('report') or {}
    seconds, epochs = report.get('seconds'), report.get('epochs_run')
    if not isinstance(seconds, (int, float)) or isinstance(seconds, bool) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(f'report.seconds missing or invalid: {seconds!r}')
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs <= 0:
        raise ValueError(f'report.epochs_run missing or zero: {epochs!r}')
    return seconds / epochs


def evaluate(fits, arms, seeds, allowed_uuids, released_uuids):
    """fits: {arm: {seed: fit.json dict}}. Returns the gate record (pure; no IO)."""
    errors, per_arm, overhead, uuids = [], {}, {}, set()
    for arm in arms:
        values, walls = [], []
        for seed in seeds:
            fit = (fits.get(arm) or {}).get(seed)
            if fit is None:
                errors.append(f'{arm}/{seed}: fit.json missing')
                continue
            uuids.add((fit.get('runtime') or {}).get('bound_uuid'))
            if fit.get('dev_used_for_fit_or_selection') is not False:
                errors.append(f'{arm}/{seed}: dev_used_for_fit_or_selection is not False')
            try:
                values.append(train_epoch_seconds(fit))
                walls.append(float(fit['fit_seconds']) - float(fit['report']['seconds']))
            except (ValueError, KeyError, TypeError) as error:
                errors.append(f'{arm}/{seed}: {error}')
        per_arm[arm] = statistics.median(values) if len(values) == len(seeds) else None
        overhead[arm] = statistics.median(walls) if len(walls) == len(seeds) else None
    if uuids != set(allowed_uuids):
        errors.append(f'smoke bound UUID set {sorted(map(str, uuids))} != allowed {sorted(allowed_uuids)}')
    if uuids & set(released_uuids):
        errors.append('smoke used a released GPU')
    lin, per = per_arm.get('linear_residual_counts'), per_arm.get('periodic_residual_counts')
    ratio = lin / per if lin is not None and per else None
    passed = not errors and ratio is not None and math.isfinite(ratio) and ratio <= RATIO_MAX
    return {'protocol': PROTOCOL, 'pass': passed, 'errors': errors,
            'train_seconds_per_epoch_median': per_arm, 'linear_over_periodic': ratio, 'ratio_max': RATIO_MAX,
            'fit_wall_overhead_seconds_median': overhead, 'bound_uuids': sorted(map(str, uuids)),
            'tasks': sum(len((fits.get(a) or {})) for a in arms),
            'family_projection_seconds': FAMILY_PROJECTION_SECONDS, 'task_bound_seconds': TASK_BOUND_SECONDS,
            'scorer_timeout_seconds': SCORER_TIMEOUT_SECONDS,
            'note': 'TRAIN-only smoke (8,192 rows x 2 epochs): not a cap-60 measurement; projection fixed from P36 '
                    'actuals; fit wall overhead recorded, not gated; no prediction/DEV/2026 read'}


def main(argv):
    draft, queue_dir, runs, out = map(Path, argv)
    assert not out.exists(), 'gate summary already exists'
    assert runs.name.endswith('-smoke'), 'smoke runs dir must end -smoke'
    assert hashlib.sha256(draft.read_bytes()).hexdigest() == DRAFT_SHA, 'DRAFT config sha differs'
    import run_cuda_direct as runner
    from pitchmdp import direct_embed_type_registry as et
    config = json.loads(draft.read_text())
    assert config['device'] == et.DEVICE and config['seeds'] == et.SEEDS
    assert [a['name'] for a in config['arms']] == list(et.ARM_ORDER)
    counts = json.loads((queue_dir / 'summary.json').read_text())['counts']
    sources = runner.own_hashes()
    fits, identity_errors = {}, []
    for arm in et.ARM_ORDER:
        for seed in et.SEEDS:
            d = runner.task_dir(runs, arm, seed, True)
            try:
                runner.verify_task(d, runner.expected_identity(config, MANIFEST_SHA, arm, seed, True, 'cuda', sources))
                fits.setdefault(arm, {})[seed] = json.loads((d / 'fit.json').read_text())
            except Exception as error:  # noqa: BLE001 - every identity failure is a gate refusal
                identity_errors.append(f'{arm}/{seed}: identity {error!r}')
    record = evaluate(fits, et.ARM_ORDER, et.SEEDS, config['device']['allowed_gpu_uuids'], et.RELEASED_GPUS)
    if counts != {'done': et.FITS}:
        record['errors'].append(f'smoke queue counts {counts} != done {et.FITS}')
    record['errors'] += identity_errors
    record['pass'] = record['pass'] and not record['errors']
    record.update(config_sha256=DRAFT_SHA, export_manifest_sha256=MANIFEST_SHA, smoke_queue=str(queue_dir),
                  smoke_runs=str(runs), source_sha256=sources)
    with out.open('x') as handle:
        handle.write(json.dumps(record, indent=2, sort_keys=True) + '\n')
    print('P37_SMOKE_GATE', 'PASS' if record['pass'] else 'REFUSED', hashlib.sha256(out.read_bytes()).hexdigest(),
          json.dumps(record['errors'][:5]))
    return 0 if record['pass'] else 3


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
