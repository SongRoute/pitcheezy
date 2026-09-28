"""Synthetic N3/R78 scorer checks for ML-JUNE-CALIBRATION-v1; no real data or scores."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pitchmdp import matrix_june_calibration as mj
from pitchmdp import matrix_june_calibration_metrics as mm
from pitchmdp.matrix_group_metrics import bootstrap_difference
from pitchmdp.matrix_metrics import holm_adjust, paired_game_comparison, pitch_losses

REPO = Path(__file__).resolve().parents[3]
SCIENTIFIC = REPO / 'configs' / 'ML-JUNE-CALIBRATION-v1.json'


def simplex(rng, n, y=None, strength=0.0):
    p = rng.gamma(1.0, size=(n, 10))
    if y is not None:
        p[np.arange(n), y] += strength
    return p / p.sum(axis=1, keepdims=True)


def population(seed=3, games=60, per_game=(20, 40)):
    rng = np.random.default_rng(seed)
    sizes = rng.integers(*per_game, size=games)
    game = np.repeat(np.arange(games) + 7000, sizes)
    n = len(game)
    y = rng.integers(0, 10, n)
    base = simplex(rng, n, y, 0.4)
    variants = {'B0': base, 'B1': 0.7 * base + 0.3 * simplex(rng, n, y, 1.0), 'B2': 0.5 * base + 0.5 * simplex(rng, n, y, 1.2)}
    predictions = {v: {'primary': p, 'seed_primary': np.stack([0.9 * p + 0.1 * simplex(rng, n) for _ in mj.SEEDS])}
                   for v, p in variants.items()}
    metadata = pd.DataFrame({
        'game_role': np.where(rng.random(n) < .6, 'starter', 'relief'),
        'throwing_hand': np.where(rng.random(n) < .3, 'L', 'R'),
        'train_volume': rng.choice(np.asarray(mj.GROUPS), n, p=[.02, .18, .4, .4]),
        'two_strikes': pd.array(rng.random(n) < .3, dtype='boolean'),
        'runners_on': pd.array(rng.random(n) < .45, dtype='boolean'),
        'seen_pitcher': rng.random(n) < .9, 'seen_batter': rng.random(n) < .8,
        'month': np.where(game < 7030, '2025-07', '2025-08')})
    cpanel = np.isin(game, game[::97]) & (rng.random(n) < .5)
    pitchers = rng.integers(100, 160, n)
    return y, game, pitchers, metadata, cpanel, predictions


def test_frozen_scientific_config_matches_the_implemented_constants():
    sci = json.loads(SCIENTIFIC.read_text())
    assert mj.check_scientific_config(sci) is sci and mm.check_scoring_config(sci) is sci
    for path, value in ((('robustness', 'family_size'), 52), (('primary', 'holm_order'), ['B2-B0', 'B1-B0', 'B2-B1']),
                        (('optimizer', 'xatol'), 1e-4), (('variants', 'B2', 'shrinkage_pitches'), 500)):
        changed = copy.deepcopy(sci)
        target = changed
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        with pytest.raises(ValueError):
            mj.check_scientific_config(changed)
            mm.check_scoring_config(changed)


def test_shared_draws_reproduce_each_legacy_contrast_exactly():
    y, game, _, _, _, predictions = population()
    losses = {v: pitch_losses(y, predictions[v]['primary']) for v in mj.VARIANTS}
    deltas = {cid: losses[c] - losses[k] for cid, c, k in mm.CONTRASTS}
    shared = mm.shared_bootstrap(deltas, game, draws=3000, seed=20260924, chunk_rule=mm.legacy_paired_chunk)
    bounds = mm.shared_bootstrap(deltas, game, draws=3000, seed=20260924, chunk_rule=mm.legacy_bound_chunk)
    for cid, c, k in mm.CONTRASTS:
        old = paired_game_comparison(y, predictions[c]['primary'], predictions[k]['primary'], game, draws=3000)
        reps = shared['replicates'][cid]
        for j, loss in enumerate(('nll', 'brier')):
            observed = shared['observed'][cid][j]
            assert old[loss]['delta'] == float(observed)
            assert old[loss]['ci95'] == np.quantile(reps[:, j], [.025, .975]).tolist()
            assert old[loss]['p_less'] == float((1 + ((reps[:, j] - observed) <= observed).sum()) / 3001)
        legacy = bootstrap_difference(deltas[cid], game, draws=3000, upper_alpha=.05 / 78)
        assert legacy['simultaneous_upper'] == np.quantile(bounds['replicates'][cid], 1 - .05 / 78, axis=0).tolist()


def test_estimand_is_pitch_weighted_not_equal_game():
    game = np.asarray([1] * 90 + [2] * 10)
    values = np.zeros((100, 2))
    values[:90] = 1.0
    boot = mm.shared_bootstrap({'x': values}, game, draws=200, seed=1, chunk_rule=mm.legacy_paired_chunk)
    assert boot['observed']['x'].tolist() == [0.9, 0.9]  # equal-game mean would be 0.5
    assert boot['games'] == 2


def test_additivity_is_checked_at_every_draw_and_misalignment_stops():
    y, game, _, _, _, predictions = population()
    losses = {v: pitch_losses(y, predictions[v]['primary']) for v in mj.VARIANTS}
    deltas = {cid: losses[c] - losses[k] for cid, c, k in mm.CONTRASTS}
    boot = mm.shared_bootstrap(deltas, game, draws=500, seed=20260924, chunk_rule=mm.legacy_paired_chunk)
    report = mm.additivity_report(boot['observed'], boot['replicates'], scope='t')
    assert report['passed'] and report['draws_checked'] == 500 and report['draw_max_abs_error'] < 1e-14
    tampered = {k: v.copy() for k, v in boot['replicates'].items()}
    tampered['B2-B1'][417, 1] += 1e-11
    with pytest.raises(mm.ContrastAdditivityError):
        mm.require_additivity(mm.additivity_report(boot['observed'], tampered, scope='t'))
    shuffled = dict(deltas)
    shuffled['B2-B1'] = deltas['B2-B1'][np.random.default_rng(0).permutation(len(y))]
    misaligned = mm.shared_bootstrap(shuffled, game, draws=50, seed=20260924, chunk_rule=mm.legacy_paired_chunk)
    assert not mm.additivity_report(misaligned['observed'], misaligned['replicates'], scope='t')['passed']


def _slot(delta, ci, brier_ci, seeds, status='measured'):
    return {'status': status, 'nll': {'delta': delta, 'ci95': ci}, 'brier': {'delta': 0, 'ci95': brier_ci},
            'seed_nll_deltas': seeds}


def test_n_decision_rules_and_holm_with_unmeasured_slot():
    good = _slot(-.004, [-.006, -.001], [-.001, .0009], [-1, -1, -1, -1, .1])
    assert mm.n_decision(good, .04)['status'] == 'development_improvement'
    assert mm.n_decision(good, .051)['status'] == 'inconclusive'
    assert mm.n_decision(_slot(-.004, [-.006, -.001], [-.001, .0011], [-1] * 5), .01)['status'] == 'inconclusive'
    assert mm.n_decision(_slot(-.004, [-.006, -.001], [-.001, .001], [-1, -1, -1, .1, .1]), .01)['status'] == 'inconclusive'
    assert mm.n_decision(_slot(-.0029, [-.006, -.001], [-.001, .001], [-1] * 5), .01)['status'] == 'inconclusive'
    assert mm.n_decision(_slot(.002, [.001, .003], [0, .001], [1] * 5), .9)['status'] == 'worse_or_guardrail_failure'
    assert mm.n_decision(_slot(0, [-.001, .001], [.0011, .002], [1] * 5), .9)['status'] == 'worse_or_guardrail_failure'
    assert mm.n_decision({'status': 'unmeasured'}, 1.0)['status'] == 'unmeasured'
    assert holm_adjust([.01, 1.0, .02]) == [.03, 1.0, .04]


def test_r78_retains_every_slot_and_marks_unsupported_groups_unconfirmed():
    y, game, _, metadata, cpanel, predictions = population()
    losses = {v: pitch_losses(y, predictions[v]['primary']) for v in mj.VARIANTS}
    result, additivity = mm.r_bounds(game, losses, mm.r_group_masks(metadata, cpanel))
    assert len(result['slots']) == 78 and result['family_size'] == 78 and result['upper_alpha'] == .05 / 78
    zero = result['groups']['volume_zero']
    assert zero['reporting']['status'] != 'reporting_eligible'
    assert all(zero['contrasts'][cid]['status'] == 'unconfirmed' for cid in mm.CONTRAST_IDS)
    assert all(result['contrast_status'][cid] in ('unconfirmed', 'failed') for cid in mm.CONTRAST_IDS)
    assert all(a['passed'] for a in additivity) and len(additivity) == sum(
        g['reporting']['status'] == 'reporting_eligible' for g in result['groups'].values())
    measured = [g for g in result['groups'].values() if g['reporting']['status'] == 'reporting_eligible']
    assert measured and all(c['draws'] == 100000 for g in measured for c in g['contrasts'].values())
    with pytest.raises(ValueError, match='thirteen'):
        mm.r_bounds(game, losses, {k: v for k, v in mm.r_group_masks(metadata, cpanel).items() if k != 'non_cpanel'})


def test_candidate_rules_need_both_b2_contrasts_but_only_c1_for_b1():
    n = {cid: {'decision': {'status': 'development_improvement'}} for cid in mm.CONTRAST_IDS}
    r = {cid: 'passed' for cid in mm.CONTRAST_IDS}
    assert all(v['research_improvement_candidate'] for v in mm.candidate_eligibility(n, r).values())
    n['B2-B1']['decision']['status'] = 'inconclusive'
    out = mm.candidate_eligibility(n, r)
    assert out['B1']['research_improvement_candidate'] and not out['B2']['research_improvement_candidate']
    n['B2-B1']['decision']['status'] = 'development_improvement'
    n['B1-B0']['decision']['status'] = 'inconclusive'
    out = mm.candidate_eligibility(n, {**r, 'B2-B0': 'passed'})
    assert not out['B1']['research_improvement_candidate'] and out['B2']['research_improvement_candidate']
    out = mm.candidate_eligibility(n, {**r, 'B2-B0': 'unconfirmed'})
    assert not out['B2']['research_improvement_candidate'] and out['B2']['service_promotion'] is False


def test_aborted_family_keeps_three_and_seventy_eight_slots():
    result = mm.unmeasured_family_result('fit failed')
    assert list(result['N']) == list(mm.CONTRAST_IDS)
    assert result['multiplicity']['placeholder_p_for_complete_vector'] == [1.0, 1.0, 1.0]
    assert result['multiplicity']['measured_p'] is None and len(result['R']['slots']) == 78
    assert result['independent_confirmation'] is None and result['held_out_confirmation'] is False
    assert result['policy_effect'] is None and result['service_adoption'] is None


def test_score_family_requires_both_candidates_and_all_five_seeds():
    y, game, pitchers, metadata, cpanel, predictions = population()
    partial = {k: v for k, v in predictions.items() if k != 'B2'}
    with pytest.raises(ValueError, match='B0, B1 and B2'):
        mm.score_family(y=y, games=game, pitchers=pitchers, metadata=metadata, cpanel_mask=cpanel, predictions=partial)
    four = {k: dict(v) for k, v in predictions.items()}
    four['B1']['seed_primary'] = four['B1']['seed_primary'][:4]
    with pytest.raises(ValueError, match='five aligned'):
        mm.score_family(y=y, games=game, pitchers=pitchers, metadata=metadata, cpanel_mask=cpanel, predictions=four)


def test_score_family_complete_result_shape():
    y, game, pitchers, metadata, cpanel, predictions = population()
    result = mm.score_family(y=y, games=game, pitchers=pitchers, metadata=metadata, cpanel_mask=cpanel,
                             predictions=predictions)
    assert result['contrast_additivity_passed'] is True
    assert list(result['N']) == list(mm.CONTRAST_IDS) and result['multiplicity']['family_size'] == 3
    assert all(len(result['N'][c]['seed_nll_deltas']) == 5 for c in mm.CONTRAST_IDS)
    assert result['R']['family_size'] == 78 and len(result['R']['slots']) == 78
    assert set(result['descriptive']['populations']) == {'whole', 'cpanel', 'non_cpanel'}
    assert {'month_2025-07', 'month_2025-08', 'seen_pitcher', 'unseen_batter'} <= set(result['descriptive']['slices'])
    assert result['independent_confirmation'] is None and result['held_out_confirmation'] is False
    assert result['policy_effect'] is None and result['service_adoption'] is None
    whole = result['descriptive']['populations']['whole']
    assert whole['B0']['log_loss'] == pytest.approx(pitch_losses(y, predictions['B0']['primary'])[:, 0].mean())
    json.dumps(result, allow_nan=False)
