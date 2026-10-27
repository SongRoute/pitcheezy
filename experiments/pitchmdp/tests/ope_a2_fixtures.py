"""SYNTHETIC-ONLY fixtures for the EXP-OPE-RES-01 tests: random ledgers in the runtime-v3 row format
covering every L1-R resolution (complete, rho_zero positivity, rho_zero_before_node, natural course
with/without end, shared end unknown, worst-case residual), PAs outside E0 and no-decision PAs. No
real data, model or file is read.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pitchmdp import policy_estimator as est  # noqa: E402

N_ACTIONS = 4
KINDS = ('complete', 'complete', 'complete', 'complete', 'positivity', 'rho_zero_before', 'natural_end',
         'natural_open', 'shared_end', 'worst_refusal', 'worst_no_terminal', 'outside', 'no_decision')


def _decision(rng, pa_id, t, pitcher, *, positivity=False, off_mask=False, weight_scale=1.0):
    while True:
        mask = rng.random(N_ACTIONS) < .8
        if mask.sum() >= 2:
            break
    logging = rng.dirichlet(np.ones(N_ACTIONS) * 2.0)
    if positivity:
        a = int(np.flatnonzero(~mask)[0]) if (~mask).any() else 0
        mask[a] = False
        if mask.sum() < 1:
            mask[(a + 1) % N_ACTIONS] = True
        logging[a] = 0.
        logging /= logging.sum()
    reference = np.where(mask, logging, 0.)
    reference /= reference.sum()
    tilt = np.exp(rng.normal(0, weight_scale, N_ACTIONS))
    candidate = np.where(mask, logging * tilt, 0.)
    candidate /= candidate.sum()
    q = [float(rng.uniform(.3, .7)) if m else None for m in mask]
    if positivity:
        index = a
    elif off_mask and (~mask).any():
        index = int(np.flatnonzero(~mask)[0])
    else:
        index = int(rng.choice(N_ACTIONS, p=logging))
        if off_mask:
            index = int(np.flatnonzero(mask)[0])
    result = {'mask': mask.tolist(), 'logging': logging.tolist(), 'reference': reference.tolist(),
              'candidate': candidate.tolist(), 'logged_index': index, 'q_reference': q}
    if positivity:
        status = est.POSITIVITY
        result.update(rho_reference=0.0, rho_candidate=0.0)
    else:
        status = 'SUPPORTED' if mask[index] else 'OUTSIDE_POLICY_SUPPORT'
        result.update(rho_reference=float(reference[index] / logging[index]),
                      rho_candidate=float(candidate[index] / logging[index]))
    return {'pa_id': pa_id, 'decision_index': t, 'pitcher': str(pitcher), 'status': status, 'result': result,
            'balls': int(min(t, 3)), 'strikes': int(min(t, 2))}


def synthetic_pa(rng, pa_id, game, pitcher, kind, *, weight_scale=1.0):
    """(decision rows, facts) for one PA of the requested kind."""
    T = int(rng.integers(2, 6))
    reward = float(rng.uniform(.2, .8))
    info = {'game': game, 'in_population': True, 'reward': reward, 'start_pitcher': str(pitcher),
            'problem': None, 'problem_index': None}
    rows = [_decision(rng, pa_id, t, pitcher, weight_scale=weight_scale) for t in range(T)]
    k = int(rng.integers(1, T))

    def refuse(at, status):
        return rows[:at] + [{**rows[at], 'status': status, 'result': None}] + [
            {**r, 'status': 'UNSUPPORTED_MID_PA', 'result': None} for r in rows[at + 1:]]
    if kind == 'complete':
        return rows, info
    if kind == 'positivity':
        rows[k] = _decision(rng, pa_id, k, pitcher, positivity=True, weight_scale=weight_scale)
        rows = rows[:k + 1] + [{**r, 'status': 'UNSUPPORTED_MID_PA', 'result': None} for r in rows[k + 1:]]
        return rows, info
    if kind == 'rho_zero_before':
        rows[0] = _decision(rng, pa_id, 0, pitcher, off_mask=True, weight_scale=weight_scale)
        if rows[0]['status'] != 'OUTSIDE_POLICY_SUPPORT':
            return rows, info
        return refuse(k, 'UNSUPPORTED_MISSING_ACTION_LABEL'), info
    if kind == 'natural_end':
        return refuse(k, 'UNSUPPORTED_NO_LOGGED_ACTION'), info
    if kind == 'natural_open':
        return refuse(k, 'UNSUPPORTED_NO_LOGGED_ACTION'), {**info, 'reward': None, 'kind': est.NO_TERMINAL,
                                                           'reason': 'no_terminal_event'}
    if kind == 'shared_end':
        return rows, {**info, 'reward': None, 'kind': est.TERMINAL_VALUE_MISSING, 'reason': 'end_state_gap'}
    if kind == 'worst_refusal':
        return refuse(k, 'UNSUPPORTED_MISSING_ACTION_LABEL'), info
    if kind == 'worst_no_terminal':
        return rows, {**info, 'reward': None, 'kind': est.NO_TERMINAL, 'reason': 'no_terminal_event'}
    if kind == 'outside':
        return rows, {**info, 'in_population': False, 'start_reason': 'UNSUPPORTED_UNKNOWN_PITCHER'}
    if kind == 'no_decision':
        return [], {**info, 'problem': 'no_decision', 'problem_index': 0}
    raise ValueError(kind)


def synthetic_ledger(seed=0, n_pas=120, n_games=12, n_pitchers=8, kinds=KINDS, weight_scale=1.0):
    """(ledger decision rows in facts order, facts {pa_id: info}) for a random window."""
    rng = np.random.default_rng(seed)
    decisions, facts = [], {}
    for i in range(n_pas):
        game = int(1000 + i % n_games)
        pitcher = int(500 + (i * 7) % n_pitchers)
        kind = kinds[i % len(kinds)] if i >= len(kinds) else kinds[i]
        pa_id = f'{game}:{i}'
        rows, info = synthetic_pa(rng, pa_id, game, pitcher, kind, weight_scale=weight_scale)
        decisions.extend(rows)
        facts[pa_id] = info
    return decisions, facts


# ---------------------------------------------------------------- synthetic TAB / S5-X layer inputs
class ToyWE:
    """Deterministic defensive WE in (0, 1): depends on outs, runners, score difference and inning."""
    def predict_defense(self, state, defender_is_home):
        if state.winner is not None:
            return float((state.winner == 'home') == defender_is_home)
        diff = (state.home_score - state.away_score) * (1 if defender_is_home else -1)
        z = 0.35 * diff + 0.12 * state.outs - 0.09 * bin(state.bases).count('1') + 0.03 * state.inning
        return float(1 / (1 + np.exp(-z)))


class ToyAdvancement:
    """Deterministic apply_terminal except an out with a runner on third and < 2 outs, which scores the
    runner from third with probability 0.4 (so the zeta layer has a scoring mass to tilt)."""
    def distribution(self, state, event):
        from pitchmdp.game import _advance, apply_terminal
        if event == 'out' and state.bases & 4 and state.outs < 2:
            return [(0.6, apply_terminal(state, 'out')), (0.4, _advance(state, 1, state.bases & ~4, 1))]
        return [(1.0, apply_terminal(state, event))]


def synthetic_situations(rng, n=64):
    from pitchmdp.game import GameState
    from pitchmdp.ope_synthetic_a2 import Situation
    out = []
    for _ in range(n):
        state = GameState(int(rng.integers(1, 10)), str(rng.choice(['Top', 'Bot'])), int(rng.integers(0, 3)),
                          int(rng.integers(0, 8)), int(rng.integers(0, 6)), int(rng.integers(0, 6)))
        out.append(Situation.from_models(state, ToyWE(), ToyAdvancement()))
    return out


def synthetic_table(rng, n_rows=40000, m=20):
    from pitchmdp import ope_surrogate as osg
    from pitchmdp.ope_redesign_registry import N_CELLS
    cells = rng.integers(0, N_CELLS, size=n_rows)
    base = np.array([.62, .2, .07, .01, .05, .05])
    classes = rng.choice(6, size=n_rows, p=base)
    classes[(cells % 2 == 0) & (classes == 5)] = 0  # no double play at dp = 0
    return osg.fit_table(osg.class_counts(cells, classes), m)


def synthetic_tab_marginals(seed=0, n_situations=64, n_train=20000):
    from pitchmdp.ope_synthetic_a2 import TabMarginals
    from pitchmdp.ope_redesign_registry import N_X
    rng = np.random.default_rng(seed)
    G = 4
    outcome = rng.dirichlet([35, 30, 15, 1, 19], size=(12, G))
    table = synthetic_table(rng)
    return TabMarginals(
        groups=('FB', 'FC', 'BR', 'OS'), outcome_freq=outcome, count_group_freq=rng.dirichlet(np.ones(G) * 5, size=12),
        league_shares=np.array([.45, .1, .3, .15]),
        archetype_shares=np.array([rng.dirichlet(np.ones(G) * 3) * (rng.random(G) > .2) + 1e-9 * 0 for _ in range(8)]),
        x_dist=rng.dirichlet(np.ones(N_X) * .5, size=(G, 2)), bip_mix=rng.dirichlet(np.ones(G * 4)).reshape(G, 2, 2),
        class_table=table, first_half_table=synthetic_table(rng), second_half_table=synthetic_table(rng),
        situations=synthetic_situations(rng, n_situations), signs=np.array([1, -1, 1, 0]),
        signs_prime=np.array([-1, 1, 0, 1]), kappa_p=0.2, zeta_p=0.3, n_train_bip=n_train)


# ---------------------------------------------------------------- synthetic TRAIN rows (A2-S0 code path)
PITCH_CODES = ('FF', 'SI', 'FC', 'SL', 'ST', 'CU', 'CH', 'FS', 'KN')
BIP_EVENTS = ('field_out', 'field_out', 'field_out', 'force_out', 'single', 'single', 'double', 'triple', 'home_run',
              'grounded_into_double_play', 'sac_fly', 'field_error', 'fielders_choice', 'triple_play', 'double_play')


def synthetic_train_frame(seed=0, n_pas=6000, n_pitchers=12, start='2023-06-01', days=300):
    """Synthetic TRAIN pitch rows with the processed-frame and raw-join columns (no real data)."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    rows = []
    dates = pd.date_range(start, periods=days, freq='D')
    for pa in range(n_pas):
        game = 700000 + pa // 60
        ab = pa % 60 + 1
        date = dates[(pa // 60) % days]
        pitcher = 600 + int(rng.integers(n_pitchers))
        stand = str(rng.choice(['R', 'L']))
        inning, half = int(rng.integers(1, 10)), str(rng.choice(['Top', 'Bot']))
        outs, bases = int(rng.integers(0, 3)), int(rng.integers(0, 8))
        home, away = int(rng.integers(0, 6)), int(rng.integers(0, 6))
        balls = strikes = 0
        for number in range(1, 9):
            code = str(rng.choice(PITCH_CODES, p=[.25, .1, .08, .15, .05, .08, .15, .1, .04]))
            u = rng.random()
            final = number == 8 or u < .22
            base = {'game_pk': game, 'at_bat_number': ab, 'pitch_number': number, 'game_date': date,
                    'pitch_type': code, 'pitcher': pitcher, 'stand': stand, 'balls': balls, 'strikes': strikes,
                    'outs_when_up': outs, 'inning': inning, 'inning_topbot': half, 'home_score': home,
                    'away_score': away, 'post_home_score': home, 'post_away_score': away,
                    'on_1b': 1 if bases & 1 else np.nan, 'on_2b': 1 if bases & 2 else np.nan,
                    'on_3b': 1 if bases & 4 else np.nan, 'events': None, 'description': None,
                    'launch_speed': np.nan, 'launch_angle': np.nan, 'hc_x': np.nan, 'hc_y': np.nan, 'bb_type': None,
                    'split': 'train'}
            if final:
                kind = rng.random()
                if kind < .2:
                    base.update(description='called_strike', events='strikeout')
                elif kind < .3:
                    base.update(description='ball', events='walk')
                elif kind < .32:
                    base.update(description='hit_by_pitch', events='hit_by_pitch')
                else:
                    event = str(rng.choice(BIP_EVENTS))
                    if event in ('grounded_into_double_play', 'double_play') and not (bases & 1 and outs < 2):
                        event = 'field_out'
                    runs = 1 if event == 'sac_fly' and bases & 4 else 0
                    side = 'post_home_score' if half == 'Bot' else 'post_away_score'
                    base.update(description='hit_into_play', events=event, bb_type='line_drive',
                                launch_speed=float(rng.normal(90, 12)) if rng.random() > .05 else np.nan,
                                launch_angle=float(rng.normal(12, 25)), hc_x=float(rng.normal(125, 40)) if rng.random() > .1 else np.nan,
                                hc_y=float(rng.normal(130, 30)))
                    base[side] = base[side] + runs
                rows.append(base)
                break
            outcome = str(rng.choice(['ball', 'called_strike', 'foul'], p=[.4, .35, .25]))
            base.update(description=outcome)
            rows.append(base)
            if outcome == 'ball':
                balls = min(balls + 1, 3)
            elif strikes < 2:
                strikes += 1
    return pd.DataFrame(rows)
