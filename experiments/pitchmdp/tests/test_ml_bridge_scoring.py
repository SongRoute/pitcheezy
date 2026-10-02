"""F1 bridge scoring rules on synthetic paired archives; no real data."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'scripts'), str(Path(__file__).parent)]
from pitchmdp.matrix_bridge import batter_train_volume
from score_ml_matrix import summarize_cell
import score_ml_bridge as scorer
from bridge_synthetic import synthetic_family, synthetic_metadata


def comparison(delta=-.01, upper=-.004, lower=-.02, p=.01, brier_upper=.0005, brier_lower=-.001):
    return {'status': 'measured', 'nll': {'delta': delta, 'ci95': [lower, upper], 'p_less': p},
            'brier': {'delta': 0., 'ci95': [brier_lower, brier_upper], 'p_less': .5}}


def test_primary_decision_requires_every_registered_criterion():
    passed = scorer.bridge_decision(comparison(), [-.01, -.02, .001])
    assert passed['status'] == 'predictive_improvement' and all(passed['criteria'].values())
    assert passed['multiplicity'].startswith('single')
    for change, criterion in [({'delta': -.0029}, 'practical_improvement'),
                              ({'upper': 0.}, 'paired_ci_below_zero'),
                              ({'p': .0501}, 'one_sided_p'),
                              ({'brier_upper': .0011}, 'brier_noninferior')]:
        result = scorer.bridge_decision(comparison(**change), [-.01, -.02, -.03])
        assert result['status'] == 'inconclusive' and result['criteria'][criterion] is False
    one_seed = scorer.bridge_decision(comparison(), [-.01, .02, .001])
    assert one_seed['status'] == 'inconclusive' and not one_seed['criteria']['seed_direction_stable']
    worse = scorer.bridge_decision(comparison(delta=.01, upper=.02, lower=.004, p=.99), [.01, .02, .01])
    assert worse['status'] == 'worse_or_guardrail_failure' and worse['reverse_direction_point_estimate']
    assert 'no automatic promotion' in worse['reverse_direction_note']
    brier_fail = scorer.bridge_decision(comparison(brier_lower=.0012, brier_upper=.003), [-.01, -.01, -.01])
    assert brier_fail['status'] == 'worse_or_guardrail_failure'
    unmeasured = scorer.bridge_decision({'status': 'insufficient_games', 'nll': {'p_less': None}}, [0., 0., 0.])
    assert unmeasured['status'] == 'unmeasured'
    with pytest.raises(ValueError):
        scorer.bridge_decision(comparison(), [-.01, -.01])


def test_robustness_keeps_24_slots_and_marks_structural_zero_volume():
    full, masked, baseline = synthetic_family()
    metadata = synthetic_metadata(baseline)
    _, f = summarize_cell(full, baseline)
    _, m = summarize_cell(masked, baseline)
    result = scorer.robustness(baseline['dev_y'], f['primary'], m['primary'], baseline['dev_game_pk'], metadata)
    assert result['family_size'] == 24 and len(result['slots']) == 24
    assert result['per_bound_alpha'] == pytest.approx(.05 / 24)
    assert result['draws'] == 100000 and result['seed'] == 20260924
    zero = result['groups']['volume_zero']
    assert zero['structural_missing'] and zero['reporting']['n'] == 0
    assert zero['nll']['passed'] is None and zero['brier']['passed'] is None
    assert 'structurally unobserved' in zero['missing_reason']
    assert result['status'] == 'unconfirmed'
    assert set(result['groups']) == {'role_starter', 'role_relief', 'hand_L', 'hand_R', 'volume_low',
        'volume_middle', 'volume_high', 'volume_zero', 'two_strikes', 'less_two_strikes', 'runners_on', 'bases_empty'}
    measured = [g for g in result['groups'].values() if g['paired'] is not None]
    assert measured and all(g['nll']['passed'] is True for g in measured)
    # Reversing the arms makes full much worse and must fail the guardrail.
    worse = scorer.robustness(baseline['dev_y'], m['primary'], f['primary'], baseline['dev_game_pk'], metadata)
    assert worse['status'] == 'failed'


def test_batter_volume_groups_are_descriptive_with_train_only_q25():
    full, masked, baseline = synthetic_family()
    metadata = synthetic_metadata(baseline)
    volume = batter_train_volume(np.repeat([100, 101, 102, 103], [1, 2, 3, 40]))
    _, f = summarize_cell(full, baseline)
    _, m = summarize_cell(masked, baseline)
    result = scorer.batter_descriptive(baseline['dev_y'], f['primary'], m['primary'], baseline['dev_game_pk'],
                                       metadata.batter.to_numpy(), volume)
    assert result['status'].startswith('descriptive')
    assert result['q25'] == pytest.approx(1.75)
    batters = metadata.batter.to_numpy()
    assert result['groups']['zero']['n'] == int(np.isin(batters, [104, 105]).sum())
    assert result['groups']['low']['n'] == int(np.isin(batters, [100]).sum())
    assert result['groups']['high']['n'] == int(np.isin(batters, [101, 102, 103]).sum())
    assert result['per_batter']['103']['d100_train_pitches_as_batter'] == 40
    assert result['per_batter']['104']['volume_group'] == 'zero'


def test_analyze_end_to_end_reuses_verified_full_and_rejects_tampered_full():
    full, masked, baseline = synthetic_family()
    metadata = synthetic_metadata(baseline)
    report, values = summarize_cell(full, baseline)
    stored = {name: baseline['dev_' + name] for name in ('keys', 'y', 'game_pk', 'pitcher')}
    stored.update({'G0-global_' + k: v for k, v in values.items()})
    frozen = json.loads(json.dumps(report))
    volume = batter_train_volume(np.repeat([100, 101, 102], [5, 10, 20]))
    core, f, m = scorer.analyze(full, masked, baseline, metadata, volume, stored, frozen)
    assert np.array_equal(f['primary'], values['primary'])
    decision = core['primary']['decision']
    assert core['primary']['paired']['draws'] == 10000 and core['primary']['paired']['seed'] == 20260924
    assert decision['status'] == 'predictive_improvement'
    assert len(decision['seed_deltas']) == 3 and all(d < 0 for d in decision['seed_deltas'])
    assert core['robustness']['family_size'] == 24
    assert core['robustness']['groups']['volume_zero']['structural_missing']
    tampered = {k: v.copy() for k, v in stored.items()}
    tampered['G0-global_primary'][0, 0] += 1e-9
    with pytest.raises(ValueError, match='reconstruction'):
        scorer.analyze(full, masked, baseline, metadata, volume, tampered, frozen)
    shuffled = metadata.iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match='metadata'):
        scorer.analyze(full, masked, baseline, shuffled, volume, stored, frozen)


def test_seed_deltas_pair_by_seed_index():
    full, masked, baseline = synthetic_family()
    _, f = summarize_cell(full, baseline)
    _, m = summarize_cell(masked, baseline)
    deltas = scorer.seed_deltas(baseline['dev_y'], f['seed_primary'], m['seed_primary'])
    same = scorer.seed_deltas(baseline['dev_y'], f['seed_primary'], f['seed_primary'])
    assert len(deltas) == 3 and same == [0., 0., 0.]
    with pytest.raises(ValueError):
        scorer.seed_deltas(baseline['dev_y'], f['seed_primary'][:2], m['seed_primary'])
