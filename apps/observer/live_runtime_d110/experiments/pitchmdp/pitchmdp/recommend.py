"""Counterfactual target queries under an explicit, unidentifiable control kernel."""
from __future__ import annotations

from dataclasses import asdict
from itertools import product
import time

import numpy as np
import pandas as pd

from .model import OUTCOMES

TARGET_X = (-.7, 0., .7)
TARGET_Z = (1.65, 2.45, 3.25)


def supported_actions(train: pd.DataFrame, pitcher: int, stand: str):
    pool = train[(train.pitcher == pitcher) & (train.stand == stand)]
    type_counts = pool.pitch_type.value_counts()
    actions = []
    for pitch_type, count in type_counts.items():
        if count < 100:
            continue
        g = pool[pool.pitch_type == pitch_type]
        x, z = g.plate_x.to_numpy(), g.plate_z.to_numpy()
        for tx, tz in product(TARGET_X, TARGET_Z):
            local_n = int((((x-tx)**2+(z-tz)**2) <= .65**2).sum())
            if local_n >= 30:
                actions.append({'pitch_type': pitch_type, 'target_x_ft': tx, 'target_z_ft': tz,
                                'training_local_n': local_n, 'training_pitch_type_n': int(count)})
    return actions


def probability_tensor(model, row, actions, control_sigma=.3):
    """Three-point Gauss-Hermite per axis integrates a bivariate normal.

    Hypothetical preceding type is updated in every branch. No logged future
    pitch, actual current location, or actual future count enters the planner.
    """
    previous = ['START'] + sorted(({a['pitch_type'] for a in actions} | {str(row.prev_pitch_type)}) - {'START'})
    offsets = np.sqrt(3)*np.array([-1., 0., 1.])*control_sigma
    weights = np.array([1/6, 2/3, 1/6])
    from .model import CATEGORICAL, NUMERIC_INPUTS
    columns = set(CATEGORICAL) | set(NUMERIC_INPUTS)
    if getattr(getattr(model, 'encoder', None), 'variant', '').startswith('archetype'):
        from .archetypes import HISTORY_COLUMNS
        columns |= set(HISTORY_COLUMNS)
    base = {k: row[k] for k in columns}
    records = []
    for balls, strikes, prev, action, ix, iz in product(range(4), range(3), previous, actions, range(3), range(3)):
        r = base.copy()
        r.update(balls=balls, strikes=strikes, prev_pitch_type=prev, pitch_type=action['pitch_type'],
                 plate_x=action['target_x_ft']+offsets[ix], plate_z=action['target_z_ft']+offsets[iz])
        records.append(r)
    frame = pd.DataFrame.from_records(records)
    p = model.predict(frame).reshape(4, 3, len(previous), len(actions), 3, 3, len(OUTCOMES))
    p = np.einsum('bspaxyo,x,y->bspao', p, weights, weights)
    # A double play cannot occur with two outs or empty bases. Pool to ordinary out.
    if int(row.outs_when_up) == 2 or int(row.bases) == 0:
        p[..., 3] += p[..., 9]
        p[..., 9] = 0
    assert np.allclose(p.sum(-1), 1., atol=1e-6)
    assert np.isfinite(p).all() and (p >= 0).all()
    next_prev = np.array([previous.index(a['pitch_type']) for a in actions])
    usage = np.array([a['training_pitch_type_n'] / sum(b['pitch_type'] == a['pitch_type'] for b in actions)
                      for a in actions], dtype=float)
    usage /= usage.sum()
    return p, previous, next_prev, usage


def recommend(model, row, actions, we, advancement, control_sigma=.3, objective='we'):
    from .game import GameState, terminal_values
    from .planner import solve_pa
    start = time.perf_counter()
    state = GameState(inning=int(row.inning), half=str(row.inning_topbot), outs=int(row.outs_when_up),
                      bases=int(row.bases), home_score=int(row.home_score), away_score=int(row.away_score))
    p, previous, next_prev, baseline = probability_tensor(model, row, actions, control_sigma)
    values = terminal_values(state, we, advancement)
    if objective != 'we':
        raise ValueError('Only WE implemented for initial result')
    result = solve_pa(p, values, next_prev, baseline)
    s = (int(row.balls), int(row.strikes), previous.index(str(row.prev_pitch_type)))
    q = result.q_values[s]
    myopic = result.myopic_q_values[s]
    ranking = np.argsort(-q)
    start_we = float(we.predict_defense(state, defender_is_home=state.defender_is_home))
    reference = float(result.baseline_values[s])
    best = int(ranking[0])
    top = []
    for i in ranking[:5]:
        top.append({**actions[int(i)], 'defense_win_probability': float(q[i]),
                    'delta_we_from_start': float(q[i]-start_we),
                    'advantage_vs_reference_pp': float((q[i]-reference)*100)})
    return {
        'input': {'game_pk': int(row.game_pk), 'game_date': str(row.game_date.date()),
                  'at_bat_number': int(row.at_bat_number), 'pitch_number': int(row.pitch_number),
                  'pitcher': int(row.pitcher), 'batter': int(row.batter), 'stand': str(row.stand),
                  'balls': int(row.balls), 'strikes': int(row.strikes),
                  'prev_pitch_type': str(row.prev_pitch_type), 'state': asdict(state)},
        'n_actions': len(actions), 'control_sigma_ft': control_sigma,
        'start_defense_we': start_we, 'planned_defense_we': float(result.values[s]),
        'reference_defense_we': reference,
        'model_internal_advantage_pp': float((result.values[s]-reference)*100),
        'top_k': top,
        'myopic_action': actions[int(np.argmax(myopic))],
        'myopic_defense_we': float(np.max(myopic)),
        'planned_first_action_equals_myopic': bool(best == np.argmax(myopic)),
        'observed_pitch_type_for_description_only': str(row.pitch_type),
        'terminal_values': {k: float(v) for k, v in values.items()},
        'solver': result.diagnostics, 'seconds': time.perf_counter()-start,
        'evidence': 'Model-dependent, same-model optimization; not causal or observational OPE.',
        'baseline_definition': 'Training pitch-type usage; uniform over supported target cells within type.'}
