"""EXP-BC-CAL-01 metrics (T8): 10-class input reproduces matrix_metrics; the paired whole-game
bootstrap reproduces matrix_metrics.paired_game_comparison; the ECE / P1 sufficient statistics are
exact; Holm; SC-1 and SC-3 on synthetic data."""
import numpy as np
import pytest

from pitchmdp import behavior_metrics as bm
from pitchmdp import matrix_metrics as mm


def draw(n, k, seed):
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(n, k)) * 1.3
    p = np.exp(logits)
    p /= p.sum(axis=1, keepdims=True)
    y = np.array([rng.choice(k, p=row) for row in p])
    q = np.exp(logits * .8 + rng.normal(size=(n, k)) * .2)
    q /= q.sum(axis=1, keepdims=True)
    games = rng.integers(0, 40, size=n)
    return y, p, q, games


def test_t8_ten_class_input_equals_matrix_metrics():
    y, p, _, _ = draw(4000, 10, 0)
    ours = bm.prediction_summary(y, p, p1=1e-3, p2=1e-2)
    theirs = mm.prediction_metrics(y, p)
    assert ours['nll'] == pytest.approx(theirs['log_loss'], abs=1e-12)
    assert ours['brier'] == pytest.approx(theirs['brier_multiclass'], abs=1e-12)
    assert ours['top_label_ece10'] == pytest.approx(theirs['top_label_ece10'], abs=1e-12)
    assert ours['accuracy'] == theirs['accuracy']
    assert ours['classwise_ece']['per_class'] == pytest.approx([c['ece'] for c in theirs['classes']], abs=1e-12)


def test_t8_paired_bootstrap_reproduces_matrix_metrics_and_is_seeded():
    y, p, q, games = draw(3000, 10, 1)
    ours = bm.paired_game_bootstrap(y, q, p, games, draws=500, seed=20260924, p1_threshold=1e-3)
    theirs = mm.paired_game_comparison(y, q, p, games, draws=500, seed=20260924)
    for name in ('nll', 'brier'):
        assert ours[name]['delta'] == theirs[name]['delta']
        assert ours[name]['ci95'] == theirs[name]['ci95']
        assert ours[name]['p_less'] == theirs[name]['p_less']
    again = bm.paired_game_bootstrap(y, q, p, games, draws=500, seed=20260924, p1_threshold=1e-3)
    assert again == ours
    assert ours['ece']['delta'] == pytest.approx(bm.top_label_ece(y, q) - bm.top_label_ece(y, p), abs=1e-12)
    assert ours['ece']['ci95'][0] <= ours['ece']['delta'] + 0.05 and ours['status'] == 'measured'


def test_ece_and_p1_sufficient_statistics_are_exact_on_resampled_games():
    y, p, _, games = draw(2000, 18, 2)
    unique, inverse = np.unique(games, return_inverse=True)
    stats = bm._game_bin_stats(y, p, inverse, len(unique), 10)
    assert bm._ece_from(stats.sum(axis=0), len(y), 10) == pytest.approx(bm.top_label_ece(y, p), abs=1e-12)
    rng = np.random.default_rng(5)
    sample = rng.integers(0, len(unique), size=len(unique))
    multiplicity = np.bincount(sample, minlength=len(unique)).astype(float)
    rows = np.concatenate([np.flatnonzero(inverse == g) for g in sample])
    total = multiplicity @ np.bincount(inverse).astype(float)
    assert bm._ece_from(multiplicity @ stats, total, 10) == pytest.approx(bm.top_label_ece(y[rows], p[rows]), abs=1e-12)
    near = np.bincount(inverse, (p[np.arange(len(y)), y] < .05).astype(float), len(unique))
    assert (multiplicity @ near) / total == pytest.approx(bm.near_zero_share(y[rows], p[rows], .05), abs=1e-12)


def test_holm_family_keeps_missing_slots():
    assert bm.holm_adjust([0.01, 0.04]) == [0.02, 0.04]
    assert bm.holm_adjust([0.03, None]) == [0.06, None]  # a CALIBRATION_FAILED candidate keeps its slot
    assert bm.holm_adjust([0.04, 0.01]) == [0.04, 0.02]


def test_sc1_groups_and_sc3_bins():
    vocabulary = ('FF', 'SL', 'CH', 'KN')
    groups = {'fastball': ['FF'], 'breaking': ['SL'], 'offspeed': ['CH'], 'other': ['KN']}
    y = np.array([0, 0, 1, 2])
    p = np.array([[.5, .5, 0, 0], [.7, .1, .1, .1], [.2, .8, 0, 0], [.25, .25, .25, .25]])
    sc1 = bm.group_calibration(y, p, vocabulary, groups)
    assert sc1['groups']['fastball']['observed_minus_predicted'] == pytest.approx(.5 - p[:, 0].mean())
    assert sc1['codes']['KN']['observed'] == 0
    bins = bm.ratio_bins(y, p, [0, .001, .01, .05, .2, .5, 1.])
    assert sum(b['pairs'] for b in bins) == p.size
    last = bins[-1]  # [0.5, 1] closed: (row0 FF .5 hit), (row0 SL .5), (row1 FF .7 hit), (row2 SL .8 hit)
    assert last['pairs'] == 4 and last['observed'] == 3 and last['expected'] == pytest.approx(2.5)
    first = bins[0]  # zeros: expected 0 -> ratio None
    assert first['expected'] == 0 and first['observed_over_expected'] is None


def test_strata_and_near_zero():
    y, p, q, games = draw(600, 18, 3)
    strata = bm.strata_metrics(y, {'B0': p, 'B1': q}, {'two_strikes': np.arange(600) % 2 == 0}, p1=1e-3, p2=1e-2)
    assert set(strata['two_strikes']) == {'True', 'False'}
    assert strata['two_strikes']['True']['B0']['n'] == 300
    assert bm.near_zero_share(y, p, 1.01) == 1.0
    with pytest.raises(ValueError):
        bm.paired_game_bootstrap(y, q, p, games, draws=10, seed=1, p1_threshold=1e-3)
