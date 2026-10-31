"""Synthetic paired prediction archives for F1 bridge tests (not real data)."""
import numpy as np
import pandas as pd


def _part(rng, games, per_game, offset):
    game_pk = np.repeat(np.arange(games) + offset, per_game).astype(np.int64)
    at_bat = np.tile(np.arange(per_game) // 4 + 1, games).astype(np.int64)
    pitch = np.tile(np.arange(per_game) % 4 + 1, games).astype(np.int64)
    y = rng.integers(0, 10, len(game_pk)).astype(np.int64)
    pitcher = rng.choice([10, 20, 30, 40], len(game_pk)).astype(np.int64)
    return {'keys': np.column_stack([game_pk, at_bat, pitch]), 'y': y, 'game_pk': game_pk, 'pitcher': pitcher}


def _probabilities(rng, y, sharpness):
    p = rng.dirichlet(np.ones(10), len(y))
    p[np.arange(len(y)), y] += sharpness
    return p / p.sum(1, keepdims=True)


def synthetic_family(seed=0, games=40, per_game=40, full_sharpness=.6, masked_sharpness=.2):
    rng = np.random.default_rng(seed)
    baseline = {}
    for name, g, offset in (('blend', 12, 5000), ('dev', games, 9000)):
        for key, value in _part(rng, g, per_game, offset).items():
            baseline[name + '_' + key] = value
        baseline[name] = _probabilities(rng, baseline[name + '_y'], 0.)
        baseline[name + '_raw'] = baseline[name]

    def member(sharpness):
        values = {k: v for k, v in baseline.items() if not k.endswith(('_raw',)) and k not in ('blend', 'dev')}
        for name in ('blend', 'dev'):
            values[name] = _probabilities(rng, baseline[name + '_y'], sharpness)
            values[name + '_raw'] = _probabilities(rng, baseline[name + '_y'], sharpness)
        return values

    full = [member(full_sharpness) for _ in range(3)]
    masked = [member(masked_sharpness) for _ in range(3)]
    return full, masked, baseline


def synthetic_metadata(baseline, seed=0):
    rng = np.random.default_rng(seed)
    n = len(baseline['dev_y'])
    keys = baseline['dev_keys']
    return pd.DataFrame({'game_pk': keys[:, 0], 'at_bat_number': keys[:, 1], 'pitch_number': keys[:, 2],
                         'pitcher': baseline['dev_pitcher'], 'batter': rng.choice([100, 101, 102, 103, 104, 105], n),
                         'game_role': rng.choice(['starter', 'relief'], n),
                         'throwing_hand': rng.choice(['L', 'R'], n),
                         # TRAIN-selected panel: no zero-TRAIN pitcher is ever present.
                         'train_volume': rng.choice(['low', 'middle', 'high'], n),
                         'two_strikes': pd.array(rng.random(n) < .4, dtype='boolean'),
                         'runners_on': pd.array(rng.random(n) < .5, dtype='boolean')})
