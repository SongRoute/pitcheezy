"""End-to-end EXP-BC-CAL-01 attempt on synthetic data (smoke -> decide). Not collected by default
(file name); run by test_behavior_calibration_runner.test_end_to_end_synthetic_attempt_isolated in
a fresh interpreter with DYLD_LIBRARY_PATH=<torch/lib> (one OpenMP runtime for torch + LightGBM)."""
from datetime import datetime
import json
from pathlib import Path

import pytest

from behavior_calibration_synthetic import SyntheticSource, estimator_ess, make_ledger, make_world, synthetic_config
import run_behavior_calibration as runner
from pitchmdp import behavior_calibration_registry as reg

CONFIG = Path(__file__).resolve().parents[3] / 'configs' / 'EXP-BC-CAL-01.json'


@pytest.fixture(scope='module')
def attempt(tmp_path_factory):
    root = tmp_path_factory.mktemp('a3')
    world = make_world(0)
    ledger = make_ledger(root / 's6', world, games=8)
    config = synthetic_config(json.loads(CONFIG.read_text()), world, ledger, estimator_ess(ledger))
    guard = runner.Guard(reg.CAPS, reg.PROTECTION_WINDOW_KST, now=lambda: datetime(2026, 10, 27, 15, 0, tzinfo=runner.KST),
                         alarm=False)
    output_root = root / 'EXP-BC-CAL-01'
    output_root.mkdir()
    destination = output_root / 'attempt-1'
    destination.mkdir()
    source = SyntheticSource(world, ledger)
    run = runner.Run(config, destination=destination, source=source, guard=guard, commit='0' * 40)
    statuses = {stage: runner.execute_stage(run, stage) for stage in reg.STAGES}
    return {'run': run, 'statuses': statuses, 'destination': destination, 'config': config, 'world': world,
            'ledger': ledger, 'guard': guard, 'source': source}


def stage_json(attempt, stage, name):
    return json.loads((attempt['run'].attempt.stage_path(stage) / name).read_text())


def test_every_stage_passes_in_order_with_records(attempt):
    failures = {s: stage_json(attempt, s, 'stage.json')['error'] for s, v in attempt['statuses'].items() if v != 'PASS'}
    assert not failures, failures
    records = attempt['run'].attempt.records()
    assert [r['stage'] for r in records] == list(reg.STAGES)
    for stage in reg.STAGES:
        path = attempt['run'].attempt.stage_path(stage)
        assert runner.verify_sha256sums(path)
        record = stage_json(attempt, stage, 'stage.json')
        assert record['config_sha256'] == attempt['run'].config_sha256 and record['git_commit'] == '0' * 40
        assert record['seeds'] == attempt['config']['seeds'] and record['peak_rss_bytes'] > 0
        assert record['seconds'] >= 0 and record['citable'] is True
    assert [a['addendum'] for a in attempt['run'].attempt.addenda()] == ['smoke', 'freeze-primary', 'freeze-xf']
    assert attempt['source'].loads == 1


def test_smoke_computes_no_metric_and_checks_integrity(attempt):
    smoke = stage_json(attempt, 'smoke', 'smoke.json')
    assert smoke['metrics_computed'] is None and all(smoke['checks'].values())
    assert smoke['checks']['b2_deterministic'] and smoke['checks']['dev_labels_sealed']
    assert 'nll' not in json.dumps(smoke).lower()


def test_freeze_primary_precedes_dev_labels_and_reproduces(attempt):
    manifest = json.loads((attempt['run'].attempt.stage_path('freeze-primary') / 'manifest.json').read_text())
    assert manifest['dev_pitch_type_labels_read'] is False and manifest['inputs_on_or_before_2025_05_31'] is True
    assert manifest['latest_input_row_date'] <= '2025-05-31'
    assert set(manifest['groups']) == {'B1', 'B2', 'calibrators', 'oof_encoders', 'style_snapshots', 'train_volume'}
    runner.verify_manifest(attempt['destination'], manifest)
    pin = attempt['run'].attempt.pin('freeze-primary')
    assert pin['manifest_sha256'] == runner.sha256_file(attempt['run'].attempt.stage_path('freeze-primary') / 'manifest.json')
    summary = stage_json(attempt, 'freeze-primary', 'stage.json')['summary']
    assert summary['roundtrip_max_abs'] == {'B1': 0.0, 'B2': 0.0}
    xf = stage_json(attempt, 'xf-fit', 'xf/xf.json')
    assert xf['not_run'] == ['B2-XF'] and xf['dev_labels_sha256'] == attempt['run'].inputs().sealed.sha256


def test_integrity_scoring_weights_and_decision(attempt):
    integrity = stage_json(attempt, 's6-integrity', 's6_integrity.json')
    assert integrity['checks'] == {'S6_identity': True, 'R1': True, 'R1b': True, 'R2': True}
    assert integrity['dev_metric'] is None
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
    decision = stage_json(attempt, 'decide', 'decision.json')
    assert decision['label'] in reg.RUN_LABELS and set(decision['candidate_status'].values()) <= set(reg.CANDIDATE_STATUSES)
    final = json.loads((attempt['destination'] / 'final.json').read_text())
    assert final['label'] == decision['label'] and final['citable'] is True


def test_rerun_and_rescoring_are_refused(attempt):
    with pytest.raises(runner.Refused, match='already exists'):
        attempt['run'].require_order('score-dev')
    sibling = attempt['destination'].parent / 'attempt-2'
    sibling.mkdir()
    with pytest.raises(runner.Refused, match='already scored'):
        runner.one_shot_guard(sibling)
