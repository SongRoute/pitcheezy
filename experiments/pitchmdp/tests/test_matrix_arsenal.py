"""Synthetic checks for EXP-P13-001 arsenal features (no real data)."""
import numpy as np
import pandas as pd
import pytest

from pitchmdp.matrix_arsenal import (WIDTH, ArsenalContext, CandidateArsenalModel, fit_pitcher_arsenal,
                                     with_candidate_arsenal)
from pitchmdp.matrix_models import MatrixModel
from pitchmdp.matrix_sharing import SharingContext, SharingPredictor, fit_pitcher_clusters, training_arrays

VOCAB = ('CH', 'FF', 'SL')
T = len(VOCAB)


class Base:
    def transform(self, frame):
        return np.column_stack([frame.balls / 3, frame.strikes / 2]).astype(np.float32)

    def report(self):
        return {'base': 'synthetic'}


def pitches(split='train'):
    rng = np.random.default_rng(0)
    rows = []
    for pid, (n_ff, n_sl, n_ch, velo) in {10: (200, 100, 0, 95.), 11: (40, 0, 10, 90.),
                                         12: (0, 3, 30, 88.), 13: (60, 60, 60, 93.)}.items():
        for kind, n, dv, px in (('FF', n_ff, 0., -6.), ('SL', n_sl, -9., 4.), ('CH', n_ch, -8., -12.)):
            for _ in range(n):
                rows.append({'pitcher': pid, 'pitch_type': kind, 'split': split, 'p_throws': 'R',
                             'effective_speed': velo + dv + rng.normal(0, .5),
                             'release_spin_rate': 2300. + rng.normal(0, 30), 'pfx_x': px + rng.normal(0, .3),
                             'pfx_z': 1. + rng.normal(0, .1), 'balls': 1, 'strikes': 2})
    return pd.DataFrame(rows)


def unstandardize(arsenal, row):
    row = np.asarray(row).reshape(T, WIDTH)
    return np.concatenate([row[:, :-1] * arsenal['scale'] + arsenal['mean'], row[:, -1:]], axis=1)


def tokens_for(types):
    tokens = np.zeros((len(types), 6, 8 + T + 1 + 11), dtype=np.float32)
    for i, t in enumerate(types):
        tokens[i, -1, 8 + (VOCAB.index(t) + 1 if t in VOCAB else 0)] = 1
    return tokens, np.ones((len(types), 6), bool)


def test_train_only_and_dev_rows_do_not_change_features():
    train = pitches()
    first = fit_pitcher_arsenal(train, VOCAB)
    mixed = pd.concat([train, pitches('dev').iloc[:50]])
    with pytest.raises(ValueError, match='TRAIN'):
        fit_pitcher_arsenal(mixed, VOCAB)
    # The contract: callers pass TRAIN rows; mutating a DEV row cannot reach the fit.
    dev = pitches('dev')
    before = fit_pitcher_arsenal(mixed.loc[mixed.split.eq('train')], VOCAB)
    dev.loc[dev.index[0], 'effective_speed'] = 500.
    after = fit_pitcher_arsenal(pd.concat([train, dev]).query("split == 'train'"), VOCAB)
    assert first == before == after
    # A DEV-only pitcher is unknown and gets the league fallback row.
    frame = dev.iloc[:2].assign(pitcher=[10, 999])
    clusters = fit_pitcher_clusters(train)
    context = ArsenalContext(SharingContext(Base(), clusters), first).transform(frame)
    block = context[:, -7 - T * WIDTH:-7]
    np.testing.assert_allclose(block[1], first['fallback'], rtol=0, atol=1e-6)
    np.testing.assert_allclose(block[0], first['pitcher_arsenal']['10'], rtol=0, atol=1e-6)
    shuffled = fit_pitcher_arsenal(train.sample(frac=1, random_state=3), VOCAB)
    np.testing.assert_allclose(shuffled['pitcher_arsenal']['13'], first['pitcher_arsenal']['13'], atol=1e-9)


def test_shrinkage_toward_league_type_mean():
    train = pitches()
    arsenal = fit_pitcher_arsenal(train, VOCAB, tau=50.)
    league = np.asarray(arsenal['league_mean'])
    sl = VOCAB.index('SL')
    heavy, light, none = (unstandardize(arsenal, arsenal['pitcher_arsenal'][p]) for p in ('10', '12', '11'))
    # 100 sliders: mostly own mean (95-9=86). 3 sliders: mostly league slider speed.
    assert abs(heavy[sl, 1] - 86.) < abs(league[sl, 0] - 86.) * .5
    w = 3 / 53
    own = train.query('pitcher == 12 and pitch_type == "SL"').effective_speed.mean()
    assert heavy[sl, -1] == pytest.approx(100 / 150) and light[sl, -1] == pytest.approx(w)
    assert light[sl, 1] == pytest.approx(w * own + (1 - w) * league[sl, 0])
    # Zero sliders: exactly the league per-type mean, reliability zero.
    assert none[sl, 1] == pytest.approx(league[sl, 0]) and none[sl, -1] == 0
    # Share shrinks to league share; primary fastball and differences.
    assert heavy[sl, 0] == pytest.approx((100 + 50 * arsenal['league_share'][sl]) / 350)
    assert arsenal['pitcher_primary_fastball']['10'] == 'FF'
    ff = VOCAB.index('FF')
    assert heavy[ff, 5] == pytest.approx(0) and heavy[sl, 5] == pytest.approx(heavy[sl, 1] - heavy[ff, 1])
    # Pitcher 12 has no fastball: difference uses the league FF row.
    assert arsenal['pitcher_primary_fastball']['12'] == 'FF'
    assert light[ff, 1] == pytest.approx(league[ff, 0])
    fallback = unstandardize(arsenal, arsenal['fallback'])
    np.testing.assert_allclose(fallback[:, 0], arsenal['league_share'])
    assert (fallback[:, -1] == 0).all()


def test_flag_off_is_byte_identical_to_g0():
    train = pitches()
    clusters = fit_pitcher_clusters(train)
    frame = train.iloc[::37].assign(pitcher=lambda f: f.pitcher.where(f.index % 2 == 0, 999))
    sharing = SharingContext(Base(), clusters)
    g0 = sharing.transform(frame)
    off = ArsenalContext(sharing, None).transform(frame)
    assert off.dtype == g0.dtype and off.tobytes() == g0.tobytes()
    assert ArsenalContext(sharing, None).report() == sharing.report()
    tokens, valid = tokens_for(frame.pitch_type)
    model = MatrixModel('linear', seed=0, device='cpu')
    y = np.arange(len(frame)) % 10
    model.fit(training_arrays((tokens, valid, g0)), y, training_arrays((tokens, valid, g0)), y, epochs=1)
    a = SharingPredictor('G0-global', model, clusters).logits((tokens, valid, g0))
    b = SharingPredictor('G0-global', model, clusters).logits((tokens, valid, off))
    assert a.tobytes() == b.tobytes()


def test_shapes_routing_and_candidate_row():
    train = pitches()
    clusters, arsenal = fit_pitcher_clusters(train), fit_pitcher_arsenal(train, VOCAB)
    frame = train.iloc[::41]
    sharing = SharingContext(Base(), clusters)
    g0, on = sharing.transform(frame), ArsenalContext(sharing, arsenal).transform(frame)
    assert on.shape == (len(frame), g0.shape[1] + T * WIDTH)
    assert on[:, -7:].tobytes() == g0[:, -7:].tobytes()  # routing untouched
    assert on[:, :g0.shape[1] - 7].tobytes() == g0[:, :-7].tobytes()  # G0 columns untouched
    candidates = ['SL'] * (len(frame) - 1) + ['KN']  # last: out-of-vocabulary candidate
    tokens, valid = tokens_for(candidates)
    stripped = training_arrays((tokens, valid, on))
    augmented = with_candidate_arsenal(stripped, T)[2]
    assert augmented.shape[1] == g0.shape[1] - 7 + (T + 1) * WIDTH
    sl = VOCAB.index('SL')
    for i, pid in enumerate(frame.pitcher.iloc[:-1]):
        np.testing.assert_allclose(augmented[i, -WIDTH:], np.reshape(arsenal['pitcher_arsenal'][str(pid)], (T, WIDTH))[sl],
                                   atol=1e-6)
    assert (augmented[-1, -WIDTH:] == 0).all()
    # End to end through the frozen G0 wrapper with the candidate model.
    y = np.arange(len(frame)) % 10
    model = MatrixModel('flatten_mlp', seed=0, width=8, device='cpu')
    model.fit(with_candidate_arsenal(stripped, T), y, with_candidate_arsenal(stripped, T), y, epochs=1)
    logits = SharingPredictor('G0-global', CandidateArsenalModel(model, T), clusters).logits((tokens, valid, on))
    assert logits.shape == (len(frame), 10) and np.isfinite(logits).all()
