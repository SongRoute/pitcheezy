"""End-to-end EXP-BC-CAL-01 attempt on synthetic data (smoke -> decide) through the production A3
readers (synthetic processed parquet with pre-TRAIN and June rows, raw key files, synthetic S6-shaped
ledger). Not collected by default (file name); run by
test_behavior_calibration_runner.test_end_to_end_synthetic_attempt_isolated in a fresh interpreter
with DYLD_LIBRARY_PATH=<torch/lib> (one OpenMP runtime for torch + LightGBM)."""
from datetime import datetime
import json

import pytest

from behavior_calibration_synthetic import (SyntheticSource, estimator_ess, extra_rows, load_config, make_ledger,
                                            make_world, synthetic_config)
import run_behavior_calibration as runner
from pitchmdp import behavior_calibration_registry as reg


@pytest.fixture(scope='module')
def attempt(tmp_path_factory):
    root = tmp_path_factory.mktemp('a3')
    world = make_world(0)
    ledger = make_ledger(root / 's6', world, games=8)
    config = synthetic_config(load_config(), world, ledger, estimator_ess(ledger))
    guard = runner.Guard(reg.CAPS, reg.PROTECTION_WINDOW_KST, now=lambda: datetime(2026, 10, 27, 15, 0, tzinfo=runner.KST),
                         alarm=False)
    output_root = root / 'EXP-BC-CAL-01'
    output_root.mkdir()
    destination = output_root / 'attempt-1'
    destination.mkdir()
    source = SyntheticSource(world, ledger, root / 'data', extras=extra_rows())
    run = runner.Run(config, destination=destination, source=source, guard=guard, commit='0' * 40)
    statuses = {stage: runner.execute_stage(run, stage) for stage in reg.STAGES}
    return {'run': run, 'statuses': statuses, 'destination': destination, 'config': config, 'world': world,
            'ledger': ledger, 'guard': guard, 'source': source}


def stage_json(attempt, stage, name):
    return json.loads((attempt['run'].attempt.stage_path(stage) / name).read_text())


def pyarrow_reads(entries):
    return [e for e in entries if e['kind'] == 'read']


def test_every_stage_passes_in_order_with_records(attempt):
    failures = {s: stage_json(attempt, s, 'stage.json')['error'] for s, v in attempt['statuses'].items() if v != 'PASS'}
    assert not failures, failures
    records = attempt['run'].attempt.records()
    assert [r['stage'] for r in records] == list(reg.STAGES)
    for stage in reg.STAGES:
        path = attempt['run'].attempt.stage_path(stage)
        assert runner.verify_sha256sums(path) and attempt['run'].attempt.passed(stage)
        record = stage_json(attempt, stage, 'stage.json')
        assert record['config_sha256'] == attempt['run'].config_sha256 and record['git_commit'] == '0' * 40
        assert record['seeds'] == attempt['config']['seeds'] and record['peak_rss_bytes'] > 0
        assert record['seconds'] >= 0 and record['citable'] is True and record['revalidation']['stage'] == stage
    assert [a['addendum'] for a in attempt['run'].attempt.addenda()] == ['smoke', 'freeze-primary', 'freeze-xf']
    assert attempt['source'].opens == 1 and attempt['source'].revalidations == list(reg.STAGES)


def test_smoke_reads_train_row_groups_only_and_computes_no_metric(attempt):
    smoke = stage_json(attempt, 'smoke', 'smoke.json')
    assert smoke['metrics_computed'] is None and all(smoke['checks'].values())
    assert smoke['checks']['b2_deterministic'] and smoke['checks']['no_row_after_train_end_decoded']
    assert smoke['rows_decoded_after_train_end'] == 0 and 'nll' not in json.dumps(smoke).lower()
    assert smoke['checks']['processed_layout_split_aligned'] and smoke['checks']['no_dev_bearing_guarded_chunk_read']
    assert stage_json(attempt, 'smoke', 'layout.json')['split_aligned'] is True
    record = stage_json(attempt, 'smoke', 'stage.json')
    reads = pyarrow_reads(record['reads'])
    assert reads and all(e['rows_at_or_after']['2025-05-01'] == 0 and e['validation'] == 'passed' for e in reads)
    summary = record['read_summary']
    processed = [e for e in reads if e['file'] == 'pitches.split-aligned.parquet']
    assert processed and not any(any(e['rows_dropped'].values()) for e in processed)  # split-aligned input
    assert summary['row_groups_skipped'] > 0 and summary['dev_bearing_guarded_reads'] == 0
    assert 'split-aligned' in summary['disclosure']


def test_census_is_label_free_and_dev_labels_wait_for_the_gate(attempt):
    census = pyarrow_reads(stage_json(attempt, 'census', 'stage.json')['reads'])
    assert census and not any(e['guarded'] for e in census)
    fit_record = stage_json(attempt, 'fit', 'stage.json')
    fit = pyarrow_reads(fit_record['reads'])
    assert any(e['guarded'] for e in fit) and not any(e['guarded'] and e['dev_bearing'] for e in fit)
    skipped = [e for e in fit_record['reads'] + stage_json(attempt, 'census', 'stage.json')['reads']
               if e['kind'] == 'skip' and e['file'] == 'pitches.split-aligned.parquet']
    assert {(e['group_low'], e['group_high']) for e in skipped} >= {('2025-06-01', '2025-06-30')}  # June never read
    assert any(e['group_high'] < '2023-05-15' for e in skipped)  # rows before TRAIN never read
    xf_record = stage_json(attempt, 'xf-fit', 'stage.json')
    assert xf_record['read_summary']['dev_bearing_guarded_reads'] > 0
    for stage in reg.STAGES:
        if stage != 'xf-fit':
            reads = pyarrow_reads(stage_json(attempt, stage, 'stage.json')['reads'])
            assert not any(e['guarded'] and e['dev_bearing'] for e in reads), stage
    events = runner.read_chain(attempt['destination'] / 'reads.jsonl')  # persisted, hash-chained, event by event
    begun = {e['index'] for e in events if e['event'] == 'begin'}
    assert begun and all(e['index'] in begun for e in events if e['event'] == 'finish')
    first = {}
    for position, e in enumerate(events):
        first.setdefault((e['index'], e['event']), position)
    assert all(first[(i, 'begin')] < first[(i, 'finish')] for i in begun)  # logged before the read completed


def test_freeze_primary_precedes_dev_labels_and_reproduces(attempt):
    manifest = json.loads((attempt['run'].attempt.stage_path('freeze-primary') / 'manifest.json').read_text())
    assert manifest['dev_guarded_rows_decoded'] == 0 and manifest['inputs_on_or_before_2025_05_31'] is True
    assert manifest['dev_bearing_guarded_chunks_read'] == 0 and manifest['read_summary']['pyarrow_reads'] > 0
    assert manifest['latest_input_row_date'] <= '2025-05-31' and manifest['reads']
    assert not any(e['guarded'] and e['dev_bearing'] for e in pyarrow_reads(manifest['reads']))
    assert set(manifest['groups']) == {'B1', 'B2', 'calibrators', 'oof_encoders', 'style_snapshots', 'train_volume'}
    runner.verify_manifest(attempt['destination'], manifest)
    pin = attempt['run'].attempt.pin('freeze-primary')
    assert pin['manifest_sha256'] == runner.sha256_file(attempt['run'].attempt.stage_path('freeze-primary') / 'manifest.json')
    assert pin['dev_guarded_rows_decoded'] == 0
    summary = stage_json(attempt, 'freeze-primary', 'stage.json')['summary']
    assert summary['roundtrip_max_abs'] == {'B1': 0.0, 'B2': 0.0}
    xf = stage_json(attempt, 'xf-fit', 'xf/xf.json')
    assert xf['not_run'] == ['B2-XF'] and xf['dev_release']['dev_rows'] > 0
    assert attempt['source'].inputs.released == 'xf-fit'


def test_integrity_scoring_weights_and_decision(attempt):
    integrity = stage_json(attempt, 's6-integrity', 's6_integrity.json')
    assert integrity['checks'] == {'S6_identity': True, 'R1': True, 'R1b': True, 'R2': True}
    assert integrity['dev_metric'] is None and 'never decoded' in integrity['ledger']['parsing']
    score = stage_json(attempt, 'score-dev', 'score.json')
    assert {'B0', 'B1', 'B1_raw', 'B2', 'B2_raw', 'B0-T', 'B0-XF', 'B1-XF', 'B1-XF_raw'} <= set(score['arms'])
    for arm in ('B1', 'B2'):
        assert score['paired_vs_b0'][arm]['status'] == 'measured'
    assert set(score['holm']) == {'B1', 'B2'} and score['populations']['S_off'] > 0
    assert score['populations']['TRAIN0'] > 0 and 'B0-XF' in score['xf_rescued']
    weights = stage_json(attempt, 's6-weights', 'weights.json')
    assert set(weights['E5']) >= {'B0', 'B1', 'B2'}
    assert weights['arms']['B0']['policies']['candidate']['ess'] == \
        attempt['config']['ope_weight_diagnostics']['integrity_gates']['R2']['expected']['candidate']
    assert weights['arms']['B1']['weight_shift_D']['paired'] >= 0
    kl = weights['arms']['B1']['kl_new_vs_b0']
    assert {'mean', 'finite_only_mean', 'decisions', 'infinite_decisions'} <= set(kl)
    decision = stage_json(attempt, 'decide', 'decision.json')
    assert decision['label'] in reg.DECISION_LABELS and set(decision['candidate_status'].values()) <= set(reg.CANDIDATE_STATUSES)
    final = json.loads((attempt['destination'] / 'final.json').read_text())
    assert final['label'] == decision['label'] and final['citable'] is True


def test_rerun_and_rescoring_are_refused(attempt):
    with pytest.raises(runner.Refused, match='closed'):
        attempt['run'].require_order('score-dev')
    sibling = attempt['destination'].parent / 'attempt-2'
    sibling.mkdir()
    with pytest.raises(runner.Refused, match='already scored'):
        runner.one_shot_guard(sibling)
