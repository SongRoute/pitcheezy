"""EXP-P38-001 TRAIN-only smoke resource gate (pre-registration section 4, fixed before any P38 measurement).

Run from the fresh code dir (code-<commit>/experiments/pitchmdp) after the 10-task CUDA smoke:
  python docs/launch/p38_smoke_gate.py <DRAFT config> <smoke queue dir> <smoke runs dir (ends -smoke)> <out summary.json>
Reads only queue summary.json and identity.json / fit.json (via the runner's verify_task) of the 10 smoke tasks; never a
prediction archive, a DEV/mlb_dev metric or 2026 data. Exit 0 = pass, 3 = gate refused (summary still written), other =
error. The summary is a NEW file ('x'); its sha256 goes into registration.smoke.gate_summary_sha256.

ONE resource gate (not the P37 linear/periodic gate): per arm, the median over the 5 seeds of report.seconds /
report.epochs_run (training loop only); refuse if candidate / reference > RESOURCE_RATIO_MAX (1.25) or any record is
missing, nonfinite, nonpositive or has 0 epochs.  Validity (not resource) checks that also refuse: 10/10 identities,
bound UUID set == {GPU3}, no released GPU, dev_used_for_fit_or_selection False, candidate fits carry one TRAIN-only
vocabulary record (same hash on every seed, TRAIN max_date <= cutoff, n_vocab >= 1), reference fits carry none.  The
smoke is 8,192 TRAIN rows x 2 epochs (its vocabulary is the smoke rows' vocabulary): NOT a cap-60 or full-vocabulary
measurement.  The family projection is fixed from P36 actuals in the registry, not from the smoke.
"""
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'scripts')]
from pitchmdp import direct_pitcher_id_registry as pid  # noqa: E402

PROTOCOL = 'p38_train_smoke_gate_v1'
DRAFT_SHA = '8518f754236447fd054550dee75a5c3da7c510c3dd646f028662f097a02ee537'
MANIFEST_SHA = '49097109b4f522a5a94d1557e00c52d88a8089c725977fe263c592214fb1f578'
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


def evaluate(fits, seeds=pid.SEEDS, allowed_uuids=(pid.RETAINED_GPU,), released_uuids=pid.RELEASED_GPUS):
    """fits: {arm: {seed: fit.json dict}}. Returns the gate record (pure; no IO)."""
    errors, per_arm, uuids, vocab = [], {}, set(), set()
    for arm in pid.ARM_ORDER:
        values = []
        for seed in seeds:
            fit = (fits.get(arm) or {}).get(seed)
            if fit is None:
                errors.append(f'{arm}/{seed}: fit.json missing')
                continue
            uuids.add((fit.get('runtime') or {}).get('bound_uuid'))
            if fit.get('dev_used_for_fit_or_selection') is not False:
                errors.append(f'{arm}/{seed}: dev_used_for_fit_or_selection is not False')
            record = fit.get(pid.FLAG)
            if arm == pid.REFERENCE and record is not None:
                errors.append(f'{arm}/{seed}: reference carries a pitcher-ID record')
            if arm == pid.CANDIDATE:
                if not isinstance(record, dict) or int(record.get('n_vocab') or 0) < 1 \
                        or record.get('train_max_date') is None or int(record['train_max_date']) > pid.LAST_TRAIN_DATE:
                    errors.append(f'{arm}/{seed}: TRAIN vocabulary record missing or after the cutoff')
                else:
                    vocab.add(json.dumps({k: record.get(k) for k in pid.VOCAB_KEYS}, sort_keys=True))
            try:
                values.append(train_epoch_seconds(fit))
            except ValueError as error:
                errors.append(f'{arm}/{seed}: {error}')
        per_arm[arm] = statistics.median(values) if len(values) == len(seeds) else None
    if len(vocab) > 1:
        errors.append('candidate smoke fits carry different vocabulary records (must be TRAIN-only, seed-free)')
    if uuids != set(allowed_uuids):
        errors.append(f'smoke bound UUID set {sorted(map(str, uuids))} != allowed {sorted(allowed_uuids)}')
    if uuids & set(released_uuids):
        errors.append('smoke used a released GPU')
    cand, ref = per_arm.get(pid.CANDIDATE), per_arm.get(pid.REFERENCE)
    ratio = cand / ref if cand is not None and ref else None
    passed = not errors and ratio is not None and math.isfinite(ratio) and ratio <= pid.RESOURCE_RATIO_MAX
    return {'protocol': PROTOCOL, 'pass': passed, 'errors': errors,
            'train_seconds_per_epoch_median': per_arm, 'candidate_over_reference': ratio,
            'ratio_max': pid.RESOURCE_RATIO_MAX, 'bound_uuids': sorted(map(str, uuids)),
            'smoke_vocabulary': json.loads(next(iter(vocab))) if len(vocab) == 1 else None,
            'tasks': sum(len(fits.get(a) or {}) for a in pid.ARM_ORDER),
            'family_projection_seconds': pid.FAMILY_PROJECTION_SECONDS, 'task_bound_seconds': pid.TASK_BOUND_SECONDS,
            'scorer_timeout_seconds': SCORER_TIMEOUT_SECONDS,
            'note': 'TRAIN-only smoke (8,192 rows x 2 epochs): not a cap-60 or full-vocabulary measurement; '
                    'projection fixed from P36 actuals; no prediction/DEV/2026 read'}


def main(argv):
    draft, queue_dir, runs, out = map(Path, argv)
    assert not out.exists(), 'gate summary already exists'
    assert runs.name.endswith('-smoke'), 'smoke runs dir must end -smoke'
    assert hashlib.sha256(draft.read_bytes()).hexdigest() == DRAFT_SHA, 'DRAFT config sha differs'
    import run_cuda_direct as runner
    config = json.loads(draft.read_text())
    assert config['device'] == pid.DEVICE and config['seeds'] == pid.SEEDS
    assert [a['name'] for a in config['arms']] == list(pid.ARM_ORDER)
    counts = json.loads((queue_dir / 'summary.json').read_text())['counts']
    sources = runner.own_hashes()
    fits, identity_errors = {}, []
    for arm in pid.ARM_ORDER:
        for seed in pid.SEEDS:
            d = runner.task_dir(runs, arm, seed, True)
            try:
                runner.verify_task(d, runner.expected_identity(config, MANIFEST_SHA, arm, seed, True, 'cuda', sources))
                fits.setdefault(arm, {})[seed] = json.loads((d / 'fit.json').read_text())
            except Exception as error:  # noqa: BLE001 - every identity failure is a gate refusal
                identity_errors.append(f'{arm}/{seed}: identity {error!r}')
    record = evaluate(fits, pid.SEEDS, config['device']['allowed_gpu_uuids'], pid.RELEASED_GPUS)
    if counts != {'done': pid.FITS}:
        record['errors'].append(f'smoke queue counts {counts} != done {pid.FITS}')
    record['errors'] += identity_errors
    record['pass'] = record['pass'] and not record['errors']
    record.update(config_sha256=DRAFT_SHA, export_manifest_sha256=MANIFEST_SHA, smoke_queue=str(queue_dir),
                  smoke_runs=str(runs), source_sha256=sources)
    with out.open('x') as handle:
        handle.write(json.dumps(record, indent=2, sort_keys=True) + '\n')
    print('P38_SMOKE_GATE', 'PASS' if record['pass'] else 'REFUSED', hashlib.sha256(out.read_bytes()).hexdigest(),
          json.dumps(record['errors'][:5]))
    return 0 if record['pass'] else 3


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
