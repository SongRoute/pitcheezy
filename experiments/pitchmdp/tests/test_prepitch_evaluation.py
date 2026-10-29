import copy
import json
import math
from pathlib import Path
import random
import subprocess
import sys

import pytest

from pitchmdp.prepitch_evaluation import LABELS, MODELS, evaluate, synthetic_fixture

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / 'configs/EVAL-PREPITCH-001.json'
SCRIPT = ROOT / 'experiments/pitchmdp/scripts/evaluate_prepitch.py'


@pytest.fixture
def sample():
    config = json.loads(CONFIG.read_text())
    return config, *synthetic_fixture(config)


def test_hand_calculated_scores_and_paired_difference(sample):
    result = evaluate(*sample)
    control, candidate = [result['paired_metrics'][m] for m in MODELS]
    assert control['nll'] == pytest.approx(math.log(2))
    assert candidate['nll'] == pytest.approx(-math.log(.75))
    assert control['brier_class_sum'] == .5
    assert candidate['brier_class_sum'] == .125
    assert control['accuracy'] == .5  # ties use the first fixed label
    assert candidate['accuracy'] == 1
    assert result['comparison']['nll']['delta'] == pytest.approx(math.log(2/3))
    assert result['comparison']['brier']['delta'] == -.375
    assert result['comparison']['nll']['ci95'] == pytest.approx([math.log(2/3)] * 2)
    assert 'p_less' not in result['comparison']['nll']
    assert result['scientific_winner'] is None


def test_order_invariant_with_unequal_games_and_nonconstant_losses(sample):
    config, roster, predictions = sample
    roster[-1]['game_id'] = 3
    for p in predictions[-2:]:
        p['game_id'] = 3
    predictions[1]['probabilities'] = [.2, .8] + [0.] * 8
    reference = evaluate(config, roster, predictions)
    rng = random.Random(9)
    for _ in range(6):
        rng.shuffle(roster)
        rng.shuffle(predictions)
        assert evaluate(config, roster, predictions) == reference


def test_failure_denominators_alignment_and_descriptive_strata(sample):
    config, roster, predictions = sample
    predictions = predictions[1:]  # control missing pitch 1
    p = predictions[1]  # control pitch 2
    p.clear()
    p.update({k: roster[1][k] for k in ('game_id', 'at_bat_number', 'pitch_number')})
    p.update(model=MODELS[0], status='abstain')
    predictions[-1]['generated_at'] = roster[-1]['deadline']  # exact deadline is late
    result = evaluate(config, roster, predictions)
    assert result['coverage'][MODELS[0]]['statuses'] == dict(ok=2, missing=1, error=0, abstain=1, late=0)
    assert result['coverage'][MODELS[1]]['statuses']['late'] == 1
    assert result['paired_keys'] == [[2, 1, 1]]
    assert result['comparison']['status'] == 'insufficient_games'
    assert result['comparison']['nll']['ci95'] is None
    assert not result['complete_common_coverage']
    assert sum(r['eligible'] for r in result['strata_descriptive_only']['count']) == 4
    assert result['scientific_winner'] is None


def test_all_failures_and_empty_pair(sample):
    config, roster, predictions = sample
    predictions = [{**{k: p[k] for k in ('game_id', 'at_bat_number', 'pitch_number', 'model')},
                    'status': 'error'} for p in predictions]
    result = evaluate(config, roster, predictions)
    assert result['comparison']['status'] == 'unmeasured'
    assert result['paired_metrics'][MODELS[0]]['n'] == 0
    assert result['coverage'][MODELS[0]]['statuses']['error'] == 4
    assert result['coverage'][MODELS[0]]['eligible'] == 4


def test_zero_target_probability_is_counted_and_capped(sample):
    config, roster, predictions = sample
    predictions[0]['probabilities'] = [0., 1.] + [0.] * 8
    result = evaluate(config, roster[:1], predictions[:2])
    score = result['paired_metrics'][MODELS[0]]
    assert score['zero_true_probability'] == 1
    assert score['nll'] == pytest.approx(-math.log(1e-12))
    assert score['brier_class_sum'] == 2


@pytest.mark.parametrize('kind', ['roster', 'prediction', 'conflicting'])
def test_duplicate_physical_identity_rejected(sample, kind):
    config, roster, predictions = sample
    if kind == 'roster':
        roster.append(copy.deepcopy(roster[0]))
    else:
        predictions.append(copy.deepcopy(predictions[0]))
        if kind == 'conflicting':
            predictions[-1]['probabilities'] = [1.] + [0.] * 9
    with pytest.raises(ValueError, match='duplicate'):
        evaluate(config, roster, predictions)


@pytest.mark.parametrize('field,value', [
    ('labels', list(reversed(LABELS))), ('labels', list(LABELS[:7])),
    ('labels', list(LABELS) + ['extra']), ('probabilities', [float('nan')] * 10),
    ('probabilities', [float('inf')] * 10), ('probabilities', [-.1, 1.1] + [0.] * 8),
    ('probabilities', [.5, .500002] + [0.] * 8), ('probabilities', [.1] * 7), ('probabilities', ['0.1'] * 10),
    ('probabilities', [True] + [False] * 9),
    ('input_fingerprint', '0' * 64), ('model', 'unregistered'), ('game_id', 999),
    ('generated_at', '2000-01-02T11:59:59'),
    ('input_available_at', '2000-01-02T12:00:00Z'),
    ('input_available_at', '2000-01-02T11:59:59.5Z'),
    ('input_available_at', 'invalid'), ('status', 'unknown')])
def test_invalid_prediction_rejected(sample, field, value):
    config, roster, predictions = sample
    predictions[0][field] = value
    with pytest.raises(ValueError):
        evaluate(config, roster, predictions)


@pytest.mark.parametrize('field,value', [('label', 'unknown'), ('game_id', True),
    ('pitch_number', 0), ('deadline', '1999-01-01T00:00:00Z'), ('extra', 'unregistered')])
def test_invalid_roster_rejected(sample, field, value):
    config, roster, predictions = sample
    roster[0][field] = value
    with pytest.raises(ValueError):
        evaluate(config, roster, predictions)


@pytest.mark.parametrize('mutation', ['real', 'gate', 'schema', 'pair', 'cutoff', 'threshold', 'strata'])
def test_unregistered_configuration_and_real_data_refused(sample, mutation):
    config, roster, predictions = sample
    if mutation == 'real': config['mode'] = 'REAL_DATA'
    if mutation == 'gate': config['real_data_gate'] = 'OPEN'
    if mutation == 'schema': config['schema'] = 'eleven-class'
    if mutation == 'pair': config['primary_pair'].reverse()
    if mutation == 'cutoff': config['training_cutoffs'][MODELS[0]] = config['synthetic_holdout_start']
    if mutation == 'threshold': config['real_data_prerequisites']['practical_improvement_rule'] = .003
    if mutation == 'strata': config['strata']['prior_history'] = ['low', 'high']
    with pytest.raises(ValueError): evaluate(config, roster, predictions)


def test_cli_deterministic_serialization_and_no_overwrite(tmp_path):
    first, second = tmp_path / 'one.json', tmp_path / 'two.json'
    args = [sys.executable, str(SCRIPT), '--config', str(CONFIG), '--output']
    for path in (first, second):
        run = subprocess.run([*args, str(path)], text=True, capture_output=True)
        assert run.returncode == 0, run.stderr
        assert 'SYNTHETIC_ONLY' in run.stdout
    assert first.read_bytes() == second.read_bytes()
    previous = first.read_bytes()
    assert subprocess.run([*args, str(first)], capture_output=True).returncode != 0
    assert first.read_bytes() == previous
    assert subprocess.run([*args, str(tmp_path / 'three.json'), '--input', 'real.json'], capture_output=True).returncode != 0


@pytest.mark.parametrize('generated', ['1999-01-01T00:00:00Z', '2000-01-01T00:00:00Z'])
def test_generation_before_or_equal_training_cutoff_rejected(sample, generated):
    config, roster, predictions = sample
    predictions[0]['input_available_at'] = '1998-12-31T23:59:00Z'
    predictions[0]['generated_at'] = generated
    with pytest.raises(ValueError, match='training cutoff'):
        evaluate(config, roster, predictions)


def test_generation_strictly_after_training_cutoff_allowed(sample):
    config, roster, predictions = sample
    predictions[0]['input_available_at'] = '2000-01-01T00:00:00Z'
    predictions[0]['generated_at'] = '2000-01-01T00:00:00.000001Z'
    result = evaluate(config, roster, predictions)
    assert result['coverage'][MODELS[0]]['timely_ok'] == 4
