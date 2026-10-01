"""Synthetic wiring checks for scripts/run_ml_direct_head.py (no real data)."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'scripts')]
import run_ml_direct_head as runner  # noqa: E402
from pitchmdp.matrix_features import MatrixHistoryStore  # noqa: E402

CONFIG = json.loads((PROJECT.parents[1] / 'configs' / 'EXP-P15-001.yaml').read_text())


def test_registered_config_matches_runner():
    runner.config_check(CONFIG, smoke=True)
    assert 'scripts/run_ml_direct_head.py' in runner.SOURCES and 'scripts/run_ml_delivery_pool.py' in runner.SOURCES
    pointer = json.loads((PROJECT.parents[1] / 'configs' / 'EXP-P15-002.yaml').read_text())
    assert pointer['family_config'] == 'configs/EXP-P15-001.yaml' and pointer['arm'] == runner.MIX
    off = copy.deepcopy(CONFIG)
    off['execution']['enabled'] = False
    with pytest.raises(ValueError, match='not enabled'):
        runner.config_check(off)
    weight = copy.deepcopy(CONFIG)
    weight['mix']['direct_weight'] = .7
    with pytest.raises(ValueError, match='one half'):
        runner.config_check(weight, smoke=True)
    family = copy.deepcopy(CONFIG)
    family['family'] = family['family'][:1]
    with pytest.raises(ValueError, match='family'):
        runner.config_check(family, smoke=True)


def frame(speed_shift=0.):
    n = 8
    return pd.DataFrame({'game_date': pd.to_datetime(['2024-04-01'] * 6 + ['2024-04-02'] * 2),
                         'game_pk': [1] * 6 + [2] * 2, 'at_bat_number': [1] * 3 + [2] * 3 + [1] * 2,
                         'pitch_number': [1, 2, 3, 1, 2, 3, 1, 2], 'split': ['train'] * 6 + ['dev'] * 2,
                         'pitch_type': ['FF', 'SL', 'FF', 'SL', 'FF', 'SL', 'FF', 'FF'],
                         'description': ['ball', 'called_strike', 'foul', 'ball', 'ball', 'foul', 'ball', 'ball'],
                         'events': [None] * n, 'supported_pa': [True] * n,
                         'strikes': [0, 0, 1, 0, 0, 0, 0, 0], 'balls': [0, 1, 1, 0, 1, 2, 0, 1],
                         'pitcher': [10] * n, 'p_throws': ['R'] * n, 'stand': ['L'] * n,
                         'effective_speed': np.arange(n) + 90. + speed_shift, 'release_spin_rate': np.arange(n) + 2100.,
                         'spin_axis': np.arange(n) * 10., 'pfx_x': [.1] * n, 'pfx_z': [.2] * n,
                         'plate_x': np.arange(n) / 10, 'plate_z': [2.5] * n})


class Context:
    def transform(self, part):
        return np.zeros((len(part), 9), np.float32)


def test_direct_inputs_never_contain_current_physics():
    store = MatrixHistoryStore.from_frame(frame())
    conditional, _ = store.gather([2, 7])
    assert np.any(conditional[:, -1, :8] != 0)
    tokens, valid, context = runner.direct_arrays(store, Context(), [2, 7])
    assert np.all(tokens[:, -1, :8] == 0) and valid[:, -1].all() and context.shape == (2, 9)
    assert np.array_equal(tokens[:, :-1], conditional[:, :-1])          # observed past pitches are unchanged
    assert np.array_equal(tokens[:, -1, 8:], conditional[:, -1, 8:])    # candidate type and empty outcome kept
    # Changing only the logged physics of the current rows must not change the direct inputs of those rows.
    other = frame()
    other.loc[[2, 7], ['effective_speed', 'plate_x']] += 5.
    changed = MatrixHistoryStore.from_frame(other, normalizer=store.normalizer, type_vocabulary=store.type_vocabulary)
    assert np.array_equal(runner.direct_arrays(changed, Context(), [2, 7])[0], tokens)


def test_no_physics_delivery_is_one_zero_draw():
    delivery = runner.NoPhysicsDelivery(8)
    points, levels = delivery.sample(frame().iloc[:3])
    assert points.shape == (3, 1, 8) and not points.any() and delivery.draws == 1
    assert levels.tolist() == [runner.NO_PHYSICS_LEVEL] * 3

    class Model:
        delivery_temperature = temperature = 1.
        report = {}

        def logits(self, arrays):
            tokens, _, _ = arrays
            assert not tokens[:, -1, :8].any()
            return np.tile(np.log(np.arange(1, 11) / 55.), (len(tokens), 1))

    store = MatrixHistoryStore.from_frame(frame())
    logits, used = delivery.logits(Model(), store, Context(), [0, 1, 2])
    assert logits.shape == (3, 1, 10) and used.tolist() == [runner.NO_PHYSICS_LEVEL] * 3
    assert np.allclose(delivery.predict(Model(), store, Context(), [0, 1]).sum(1), 1.)


def archives(rng, n=600, games=40):
    keys = np.column_stack([np.repeat(np.arange(games), n // games), np.zeros(n, int), np.arange(n)]).astype(np.int64)
    y = rng.integers(0, 10, n)
    out = {}
    for part in ('blend', 'dev'):
        p = rng.dirichlet(np.ones(10), n)
        out.update({part: p, part + '_raw': p, part + '_keys': keys, part + '_y': y,
                    part + '_game_pk': keys[:, 0], part + '_pitcher': np.full(n, 7, np.int64),
                    part + '_delivery_level': np.where(np.arange(n) % 3 == 0, 1, 3)})
    return out


def arm(baseline, seeds):
    names = ('dev', 'blend', 'dev_raw', 'blend_raw')
    return [{**baseline, **{k: v for k, v in archives(np.random.default_rng(s)).items() if k in names}} for s in seeds]


def test_family_scoring_and_fixed_mix():
    baseline = archives(np.random.default_rng(0))
    g0 = arm(baseline, (1, 2, 3))
    same = runner.score_arrays(g0, g0, baseline, CONFIG)
    rows = same['cpanel_dev']['comparison']
    assert set(rows) == {runner.DIRECT, runner.MIX}
    assert all(row['paired']['nll']['delta'] == 0 and row['N']['status'] != 'predictive_improvement' for row in rows.values())
    assert same['C'][runner.DIRECT]['status'] == 'not_shown' and same['C'][runner.DIRECT]['cost_reduction'] is None
    better = [{**m, 'dev': (m['dev'] + np.eye(10)[m['dev_y']]) / 2, 'blend': (m['blend'] + np.eye(10)[m['blend_y']]) / 2,
               'dev_raw': m['dev_raw'], 'blend_raw': m['blend_raw']} for m in g0]
    result = runner.score_arrays(better, g0, baseline, CONFIG, direct_seconds=[10., 10., 10.], reference_seconds=[350.] * 3)
    rows = result['cpanel_dev']['comparison']
    assert rows[runner.DIRECT]['N']['status'] == 'predictive_improvement'
    assert rows[runner.MIX]['paired']['nll']['delta'] < 0
    assert rows[runner.DIRECT]['paired']['nll']['delta'] < rows[runner.MIX]['paired']['nll']['delta']
    assert rows[runner.DIRECT]['N']['p_holm'] >= rows[runner.DIRECT]['paired']['nll']['p_less']
    assert result['C'][runner.DIRECT]['status'] == 'efficiency_improvement'
    assert result['C'][runner.DIRECT]['cost_reduction'] == pytest.approx(1 - 30 / 1050)
    mix = runner.mixed(better, g0, .5)
    assert np.allclose(mix[0]['dev'], (better[0]['dev'] + g0[0]['dev']) / 2) and np.allclose(mix[0]['dev'].sum(1), 1.)
    unpaired = [{**m, 'dev_y': np.roll(m['dev_y'], 1)} for m in better]
    with pytest.raises(ValueError, match='Unpaired'):
        runner.score_arrays(unpaired, g0, baseline, CONFIG)


def test_whole_mlb_secondaries_require_identical_keys():
    baseline = archives(np.random.default_rng(4))
    g0 = arm(baseline, (1, 2, 3))
    mlb = archives(np.random.default_rng(5))
    g0_mlb = [{'dev': mlb['dev'], 'dev_keys': mlb['dev_keys'], 'dev_y': mlb['dev_y'], 'dev_game_pk': mlb['dev_game_pk'],
               'dev_delivery_level': mlb['dev_delivery_level']}] * 3
    mine_mlb = [{'mlb_dev': mlb['dev'], 'mlb_dev_keys': mlb['dev_keys'], 'mlb_dev_y': mlb['dev_y']}] * 3
    frequency = {'mlb_dev': np.full((len(mlb['dev_y']), 10), .1), 'mlb_dev_keys': mlb['dev_keys']}
    result = runner.score_arrays(g0, g0, baseline, CONFIG, mine_mlb, g0_mlb, frequency)
    for block in ('mlb_dev', 'mlb_dev_blended'):
        assert all(row['paired']['nll']['delta'] == 0 for row in result[block]['comparison'].values())
    shuffled = [{**m, 'mlb_dev_keys': m['mlb_dev_keys'][::-1]} for m in mine_mlb]
    with pytest.raises(ValueError, match='whole-MLB'):
        runner.score_arrays(g0, g0, baseline, CONFIG, shuffled, g0_mlb, frequency)
