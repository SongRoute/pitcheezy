import copy
import json
import math
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pytest

from pitchmdp.pitch_type_evaluation import (LABELS, MODELS, evaluate, frequency_predictions,
                                          probability_scores, synthetic_fixture)

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / 'configs/EVAL-PITCH-TYPE-003.json'
SCRIPT = ROOT / 'experiments/pitchmdp/scripts/evaluate_pitch_type.py'


@pytest.fixture
def sample():
    config = json.loads(CONFIG.read_text())
    return config, *synthetic_fixture(config)


def test_hand_smoothed_counts_and_fallbacks(sample):
    config, history, query, _ = sample
    predictions, support = frequency_predictions(config, history, query)
    global_p = np.array([.4, .3, .2, .1])  # (FIT counts [3,2,1,0] + 4/4) / (6+4)
    n = np.array([2, 1, 1, 0])
    pitcher = (n + 4 * global_p) / 8
    context = pitcher.copy()
    for _ in range(3):  # count, matchup, full situation
        context = (n + 4 * context) / 8
    assert predictions[MODELS[0]][(11, 1, 1)] == pytest.approx(pitcher)
    assert predictions[MODELS[1]][(11, 1, 1)] == pytest.approx(context)
    assert support[0]['support'] == {'global': 6, 'pitcher': 4, 'count': 4, 'matchup': 4, 'situation': 4}
    assert support[1]['effective_level'] == 'pitcher'  # unseen count, no query labels added
    assert predictions[MODELS[1]][(11, 1, 2)] == pytest.approx(pitcher)
    assert support[-1]['unseen_pitcher']
    assert support[-1]['effective_level'] == 'global'
    assert predictions[MODELS[1]][(12, 1, 2)] == pytest.approx(global_p)
    assert all(min(p) > 0 for model in predictions.values() for p in model.values())


def test_query_labels_cannot_change_frozen_baselines(sample):
    config, history, query, _ = sample
    original = frequency_predictions(config, history, query)
    for row in query:
        row['label'] = 'OTHER'
    assert frequency_predictions(config, history, query) == original
    assert original[1][0]['support']['global'] == 6


def test_hand_metrics_top2_ties_and_zero_probability():
    scores, losses = probability_scores(np.array([0, 2]), [[.5, .5, 0, 0], [.25] * 4], 4)
    assert scores['nll'] == pytest.approx((math.log(2) + math.log(4)) / 2)
    assert scores['brier_class_sum'] == .625
    assert scores['top1_accuracy'] == .5
    assert scores['top2_coverage'] == .5  # class2 loses uniform top2 tie
    assert losses[:, 1] == pytest.approx([.5, .75])
    wrong, _ = probability_scores(np.array([0]), [[0, 1, 0, 0]], 4)
    assert wrong['nll'] == pytest.approx(-math.log(1e-12))
    assert wrong['zero_true_probability'] == 1
    assert wrong['brier_class_sum'] == 2


def test_shuffle_history_query_and_prediction_order_invariant(sample):
    config, history, query, candidate = sample
    candidate[0]['probabilities'] = [.1, .2, .3, .4]
    query[-1]['game_id'] = 13  # unequal game sizes
    candidate[-1]['game_id'] = 13
    expected = evaluate(config, history, query, candidate)
    randomizer = random.Random(6)
    for rows in (history, query, candidate): randomizer.shuffle(rows)
    assert evaluate(config, history, query, candidate) == expected
    assert expected['comparison']['status'] == 'measured'
    assert expected['scientific_winner'] is None
    assert expected['candidate_kind'] == 'HYPOTHETICAL_NOT_FITTED_TREE'
    assert set(expected['source_sha256']) == {'pitch_type_evaluation.py', 'matrix_metrics.py'}


def test_game_bootstrap_is_pitch_weighted_hand_delta(sample):
    config, history, query, candidate = sample
    query[-1]['game_id'] = 11
    query[-1]['pitch_number'] = 3
    candidate[-1]['game_id'] = 11
    candidate[-1]['pitch_number'] = 3
    result = evaluate(config, history, query, candidate)
    probs = {tuple(row['key']): row['probabilities'] for row in result['baseline_predictions'][MODELS[1]]}
    deltas = []
    for row, pred in zip(query, candidate):
        k = (row['game_id'], row['at_bat_number'], row['pitch_number'])
        y = LABELS.index(row['label'])
        deltas.append(-math.log(pred['probabilities'][y]) + math.log(probs[k][y]))
    assert result['comparison']['nll']['delta'] == pytest.approx(np.mean(deltas))
    assert result['comparison']['nll']['delta'] != pytest.approx((np.mean([deltas[i] for i in (0, 1, 3)]) + deltas[2]) / 2)


def test_missing_abstain_late_and_error_preserve_denominator(sample):
    config, history, query, candidate = sample
    candidate = candidate[1:]  # absent first query
    candidate[0] = {k: candidate[0][k] for k in ('game_id', 'at_bat_number', 'pitch_number')}
    candidate[0]['status'] = 'abstain'
    candidate[-1]['generated_at'] = '2000-01-03T21:00:00+09:00'  # exact deadline
    result = evaluate(config, history, query, candidate)
    coverage = result['coverage'][MODELS[2]]
    assert coverage['eligible'] == 4
    assert coverage['statuses'] == dict(ok=1, missing=1, error=0, abstain=1, late=1)
    assert result['paired_keys'] == [[12, 1, 1]]
    assert result['comparison']['status'] == 'insufficient_games'
    assert result['comparison']['nll']['ci95'] is None
    assert result['coverage'][MODELS[1]]['timely_ok'] == 4
    candidate[1] = {k: candidate[1][k] for k in ('game_id', 'at_bat_number', 'pitch_number')}
    candidate[1]['status'] = 'error'
    empty = evaluate(config, history, query, candidate)
    assert empty['comparison']['status'] == 'unmeasured'
    assert empty['paired_metrics'][MODELS[2]]['nll'] is None
    assert empty['coverage'][MODELS[2]]['statuses']['error'] == 1


@pytest.mark.parametrize('label', [None, '', 'UNKNOWN', 'FA', 'PO', 'OT'])
def test_unknown_labels_never_silently_become_other(sample, label):
    config, history, query, candidate = sample
    for rows in (history, query):
        changed = copy.deepcopy(rows)
        changed[0]['label'] = label
        with pytest.raises(ValueError, match='label'):
            evaluate(config, changed if rows is history else history,
                     changed if rows is query else query, candidate)


@pytest.mark.parametrize('field,value', [('plate_x', .1), ('plate_z', 2.5), ('pitch_type', 'FF'),
    ('release_speed', 95), ('events', 'single'), ('prev_pitch_type', 'SL'), ('post_home_score', 1),
    ('balls', 4), ('strikes', 3), ('outs', 3), ('bases', 8), ('pitcher', True),
    ('score_diff', .5), ('batter_side', 'S'), ('pitcher_hand', None)])
def test_only_valid_pre_pitch_inputs(sample, field, value):
    config, history, query, candidate = sample
    query[0]['context'][field] = value
    with pytest.raises(ValueError): evaluate(config, history, query, candidate)


@pytest.mark.parametrize('kind', ['FIT_duplicate', 'query_duplicate', 'prediction_duplicate', 'key_overlap', 'game_overlap'])
def test_identity_and_role_boundaries(sample, kind):
    config, history, query, candidate = sample
    if kind == 'FIT_duplicate': history.append(copy.deepcopy(history[0]))
    if kind == 'query_duplicate': query.append(copy.deepcopy(query[0]))
    if kind == 'prediction_duplicate': candidate.append(copy.deepcopy(candidate[0]))
    if kind == 'key_overlap':
        for key in ('game_id', 'at_bat_number', 'pitch_number'): query[0][key] = history[0][key]
    if kind == 'game_overlap': query[0]['game_id'] = history[0]['game_id']; query[0]['pitch_number'] = 99
    with pytest.raises(ValueError): evaluate(config, history, query, candidate)


@pytest.mark.parametrize('observed', ['2000-01-02T00:00:00Z', '2000-01-02T00:00:00.000001Z', '2000-01-01T12:00:00'])
def test_fit_labels_must_be_strictly_prior_and_aware(sample, observed):
    config, history, query, candidate = sample
    history[0]['observed_at'] = observed
    with pytest.raises(ValueError): evaluate(config, history, query, candidate)


@pytest.mark.parametrize('kind', ['query_before_fit', 'query_equal_fit', 'candidate_equal_fit',
    'input_after_generation', 'input_at_deadline', 'holdout_at_cutoff', 'query_before_holdout'])
def test_chronology(sample, kind):
    config, history, query, candidate = sample
    if kind in ('query_before_fit', 'query_equal_fit'):
        query[0]['input_available_at'] = '1999-12-31T00:00:00Z'
        query[0]['generated_at'] = '2000-01-01T00:00:00Z' if kind == 'query_before_fit' else config['training_cutoff']
    if kind == 'candidate_equal_fit': candidate[0]['generated_at'] = config['training_cutoff']
    if kind == 'input_after_generation': query[0]['input_available_at'] = '2000-01-03T11:59:59.5Z'
    if kind == 'input_at_deadline': query[0]['input_available_at'] = query[0]['deadline']
    if kind == 'holdout_at_cutoff': config['holdout_start'] = config['training_cutoff']
    if kind == 'query_before_holdout': query[0]['deadline'] = '2000-01-02T12:00:00Z'
    with pytest.raises(ValueError): evaluate(config, history, query, candidate)


@pytest.mark.parametrize('field,value', [('labels', ['FF', 'SL', 'CH']), ('labels', list(reversed(LABELS))),
    ('schema', 'pitchmdp-ten-v1'), ('probabilities', [.1] * 10),
    ('probabilities', [.2] * 4), ('probabilities', [float('nan')] * 4),
    ('probabilities', [float('inf'), 0, 0, 0]), ('probabilities', [1.1, -.1, 0, 0]),
    ('probabilities', ['.25'] * 4), ('probabilities', [True, False, False, False]),
    ('game_id', 999), ('input_fingerprint', '0' * 64),
    ('input_available_at', '2000-01-03T11:59:57Z'), ('status', 'unknown')])
def test_invalid_candidate_rejected(sample, field, value):
    config, history, query, candidate = sample
    candidate[0][field] = value
    with pytest.raises(ValueError): evaluate(config, history, query, candidate)


@pytest.mark.parametrize('field,value', [('mode', 'REAL_DATA'), ('real_data_gate', 'OPEN'),
    ('candidate_kind', 'FITTED_TREE'), ('label_map', {'UNKNOWN': 'OTHER'}),
    ('smoothing', {'global': 0, 'pitcher': 4, 'count': 4, 'matchup': 4, 'situation': 4})])
def test_unregistered_config_refused(sample, field, value):
    config, history, query, candidate = sample
    config[field] = value
    with pytest.raises(ValueError): evaluate(config, history, query, candidate)


def test_cli_deterministic_exclusive_and_no_external_input(tmp_path):
    args = [sys.executable, str(SCRIPT), '--config', str(CONFIG), '--output']
    one, two = tmp_path / 'one.json', tmp_path / 'two.json'
    for path in (one, two):
        run = subprocess.run([*args, str(path)], capture_output=True, text=True)
        assert run.returncode == 0, run.stderr
        assert 'hypothetical candidate' in run.stdout
    assert one.read_bytes() == two.read_bytes()
    result = json.loads(one.read_text())
    assert 'evaluate_pitch_type.py' in result['source_sha256']
    before = one.read_bytes()
    assert subprocess.run([*args, str(one)], capture_output=True).returncode != 0
    assert one.read_bytes() == before
    assert subprocess.run([*args, str(tmp_path / 'real.json'), '--input', 'payload.json'], capture_output=True).returncode != 0


@pytest.mark.parametrize('index', [1, 2, 3])
def test_row_generators_rejected_instead_of_consumed_before_provenance(sample, index):
    args = list(sample)
    args[index] = iter(args[index])
    with pytest.raises(ValueError, match='JSON array'):
        evaluate(*args)
