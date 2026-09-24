"""Leakage-restricted adapters for the frozen G ensemble and rollout policies.

No data loading, model fitting or source mutation occurs at import time.
Physical token vectors are already normalized joint TRAIN deliveries; do not
normalize them a second time or reconstruct current observed physics.
"""
from collections import Counter
import hashlib

import numpy as np
import pandas as pd

from .archetypes import STYLE_COLUMNS, RELIABILITY_COLUMNS
from .data import KEY
from .game import GameState, terminal_values
from .rollout_policy import (BCRecord, CategoricalBC, PAState, PastPitch,
                             DeliveryPool, calibrated_conditional)
from .planner import OUTCOMES

SAFE_COLUMNS = (*KEY, 'game_date', 'pitcher', 'batter', 'p_throws', 'stand',
                'inning', 'inning_topbot', 'outs_when_up', 'bases', 'home_score',
                'away_score', 'balls', 'strikes', *STYLE_COLUMNS, *RELIABILITY_COLUMNS)


def context_key(row):
    return ':'.join(str(int(row[k])) for k in KEY)


def safe_rows(frame):
    """Hard whitelist excludes current type, physics, outcomes and PA support."""
    return frame.loc[:, list(SAFE_COLUMNS)].copy()


def state_from_row(store, position, *, bc_only=False):
    row = store.frame.iloc[position]
    history = []
    indices = store.indices[position]
    if bc_only:
        indices = indices[-1:]
    for previous in indices[indices >= 0]:
        prior = store.frame.iloc[previous]
        # Explicit key/order audit avoids trusting a supplied future history map.
        if any(prior[k] != row[k] for k in KEY[:2]) or prior.pitch_number >= row.pitch_number:
            raise ValueError('History is not strictly previous within this PA')
        label = int(store.outcome_channels[previous].argmax())
        history.append(PastPitch(str(prior.pitch_type) if pd.notna(prior.pitch_type) else '<UNKNOWN>',
            () if bc_only else tuple(float(v) for v in store.physical[previous]),
            OUTCOMES[label] if label < 10 else 'unknown', int(prior.balls), int(prior.strikes)))
    return PAState(int(row.balls), int(row.strikes), str(int(row.pitcher)), str(row.stand),
                   tuple(history), context_key(row))


def fit_bc(store, train, prior_strength=20., minimum_action_count=1):
    dates = pd.to_datetime(train.game_date)
    if not len(train) or not train.split.eq('train').all() or not dates.between('2023-05-15', '2025-04-30').all():
        raise ValueError('BC requires frozen TRAIN dates and records')
    if train.pitch_type.isna().any():
        raise ValueError('Missing observed TRAIN action')
    records = (BCRecord(state_from_row(store, int(i), bc_only=True), str(action), 'train')
               for i, action in zip(train.index, train.pitch_type))
    return CategoricalBC(prior_strength, minimum_action_count).fit(records)


def select_pa_requests(frame, panel, split, requested_count, seed):
    """Select requested PA starts before support/outcome checks; no replacement.

    TRAIN role strata alternate starter/relief; hash order only within role.
    Selection accesses PA IDs, chronology and frozen TRAIN role, not outcomes.
    Returns ALL requested starts with selection flags; unsupported selected starts
    remain selected and are later marked unsupported, never replaced.
    """
    if split not in ('temperature', 'blend', 'dev') or requested_count < 1:
        raise ValueError('Invalid policy selection split/count')
    periods = {'temperature': ('2025-05-16', '2025-05-31'),
               'blend': ('2025-06-01', '2025-06-30'), 'dev': ('2025-07-01', '2025-09-30')}
    subset = frame.loc[frame.split.eq(split) & frame.pitcher.isin(panel['pitcher_ids'])]
    if not pd.to_datetime(subset.game_date).between(*periods[split]).all():
        raise ValueError('Policy split dates changed')
    starts = subset.sort_values(KEY).drop_duplicates(KEY[:2], keep='first')
    lookup = {int(r['pitcher']): r['train_role'] for r in panel['train_players']}
    result = starts[[*KEY, 'pitcher', 'game_date']].copy()
    result['position'] = starts.index.to_numpy()
    result['train_role'] = [lookup[int(pid)] for pid in result.pitcher]
    result['selection_hash'] = [hashlib.sha256(f'{seed}|{split}|{int(g)}|{int(a)}'.encode()).hexdigest()
                                for g, a in zip(result.game_pk, result.at_bat_number)]
    rank = []
    groups = [result.loc[result.train_role.eq(role)].sort_values(['selection_hash', *KEY]).index.tolist()
              for role in ('starter', 'relief')]
    for i in range(max((len(g) for g in groups), default=0)):
        for group in groups:
            if i < len(group): rank.append(group[i])
    selected = set(rank[:requested_count])
    result['selected'] = result.index.isin(selected)
    result['selection_rank'] = result.index.map({index: rank for rank, index in enumerate(rank)})
    return result.sort_values('selection_rank').reset_index(drop=True)


class PolicyInputs:
    """Frozen safe context and exact TRAIN pools; common support across counts."""
    def __init__(self, rows, context, delivery, type_vocabulary, source_hash, bc):
        self.rows = {context_key(row): row.to_dict() for _, row in safe_rows(rows).iterrows()}
        if len(self.rows) != len(rows):
            raise ValueError('Duplicate context keys')
        self.context_encoder, self.delivery, self.source_hash = context, delivery, source_hash
        self.types = tuple(type_vocabulary)
        self.type_map = {value: i+1 for i, value in enumerate(self.types)}
        self.bc = bc
        self.contexts = {}
        self.pool_cache, self.support_cache = {}, {}
        self.pool_tiers = Counter()
        if delivery.draws != 400:
            raise ValueError('Policy requires the exact frozen 400-draw delivery model')
        if len(rows):
            encoded = context.transform(safe_rows(rows))
            for key, value in zip(self.rows, encoded): self.contexts[key] = np.array(value, copy=True)

    def query(self, state, action):
        row = dict(self.rows[state.context_key])
        if str(int(row['pitcher'])) != state.pitcher or row['stand'] != state.batter_side:
            raise ValueError('State identity does not match frozen context')
        row.update(balls=state.balls, strikes=state.strikes, pitch_type=action)
        return row

    def pool(self, state, action, *, count_access=True):
        row = self.query(state, action)
        signature = tuple(row[k] for k in ('pitcher', 'pitch_type', 'p_throws', 'stand', 'balls', 'strikes'))
        if signature not in self.pool_cache:
            selected = None
            for level in reversed(range(len(self.delivery.TIERS))):
                key = tuple(row[k] for k in self.delivery.TIERS[level])
                if (level, key) in self.delivery.pools:
                    selected = (level, key, self.delivery.pools[(level, key)])
                    break
            if selected is None or action not in self.type_map:
                raise ValueError('No frozen action-specific TRAIN delivery support')
            level, key, values = selected
            self.pool_cache[signature] = (DeliveryPool(values, 'train', self.source_hash,
                repr((level, key))), level)
        pool, level = self.pool_cache[signature]
        if count_access: self.pool_tiers[level] += 1
        return pool

    def support(self, state):
        row = self.rows[state.context_key]
        signature = (state.pitcher, row['p_throws'], state.batter_side)
        if signature not in self.support_cache:
            support = self.bc.support(state).copy()
            for i, action in enumerate(self.bc.actions):
                if not support[i]: continue
                for balls in range(4):
                    for strikes in range(3):
                        probe = PAState(balls, strikes, state.pitcher, state.batter_side, (), state.context_key)
                        try: self.pool(probe, action, count_access=False)
                        except ValueError: support[i] = False
            support.setflags(write=False)
            self.support_cache[signature] = support
        return self.support_cache[signature].copy()

    def arrays(self, states, actions, physical):
        physical = np.asarray(physical, dtype=np.float32)
        n, nt = len(states), len(self.types)+1
        if physical.shape != (n, 8) or not np.isfinite(physical).all() or len(actions) != n:
            raise ValueError('One finite normalized 8-channel delivery per state required')
        tokens = np.zeros((n, 6, 8+nt+11), dtype=np.float32)
        valid = np.zeros((n, 6), dtype=bool)
        context = np.stack([self.contexts[s.context_key].copy() for s in states])
        for i, (state, action) in enumerate(zip(states, actions)):
            if action not in self.type_map:
                raise ValueError('Action outside frozen TRAIN token vocabulary')
            self.query(state, action)
            context[i, 0], context[i, 1] = state.balls/3, state.strikes/2
            for j, token in enumerate(state.history[-5:], start=5-min(5, len(state.history))):
                if len(token.physics) != 8:
                    raise ValueError('Outcome history requires physical vectors, not BC-only history')
                tokens[i, j, :8] = token.physics
                tokens[i, j, 8+self.type_map.get(token.action, 0)] = 1
                label = OUTCOMES.index(token.outcome) if token.outcome in OUTCOMES else 10
                tokens[i, j, 8+nt+label] = 1
                valid[i, j] = True
            tokens[i, 5, :8] = physical[i]
            tokens[i, 5, 8+self.type_map[action]] = 1
            valid[i, 5] = True
        return tokens, valid, context


class SupportedBC:
    def __init__(self, inputs):
        self.inputs, self.actions = inputs, inputs.bc.actions

    def support(self, state): return self.inputs.support(state)
    def fallback(self, state): return self.inputs.bc.fallback(state)

    def probabilities(self, state, *, frequency=False):
        p = self.inputs.bc.probabilities(state, frequency=frequency) * self.support(state)
        if p.sum() <= 0: raise ValueError('No common supported actions; abstain')
        return p/p.sum()


class FrozenGEnsemble:
    """Conditional G ensemble; fixed G baseline temperature and blend weight."""
    def __init__(self, inputs, models, temperatures, baseline, baseline_temperature, neural_weight):
        if len(models) != 3 or len(temperatures) != 3:
            raise ValueError('Frozen G screen requires seeds 0,1,2')
        self.inputs, self.models, self.temperatures = inputs, models, temperatures
        if not np.isfinite(baseline_temperature) or baseline_temperature <= 0:
            raise ValueError('Frozen baseline temperature must be positive')
        self.baseline, self.baseline_temperature, self.neural_weight = baseline, baseline_temperature, neural_weight
        self.frequency_cache = {}
        self.actual_neural_network_rows = 0

    def __call__(self, states, actions, physical):
        from scipy.special import softmax
        arrays = self.inputs.arrays(states, actions, physical)
        # SharingPredictor evaluates global plus routed subnetworks. Count actual
        # subnetwork rows instead of assuming exactly three networks per row.
        for model in self.models:
            routing = arrays[2][:, -2:].astype(int)
            n = len(states)
            if model.cell == 'G2-feature': n += int((routing[:, 0] >= 0).sum())
            if model.cell in ('G3-cluster', 'G4-partial'):
                n += sum(int((routing[:, 0] == int(cid)).sum()) for cid in model.cluster_models)
            if model.cell in ('G1-personal', 'G4-partial'):
                n += sum(int((routing[:, 1] == int(pid)).sum()) for pid in model.personal_models)
            self.actual_neural_network_rows += n
        logits = np.stack([model.logits(arrays) for model in self.models])
        missing = {}
        keys = [(s.context_key, s.balls, s.strikes, a) for s, a in zip(states, actions)]
        for key, state, action in zip(keys, states, actions):
            if key not in self.frequency_cache: missing[key] = self.inputs.query(state, action)
        if missing:
            raw = self.baseline.predict(pd.DataFrame(list(missing.values())))
            freq = softmax(np.log(np.clip(raw, 1e-12, 1))/self.baseline_temperature, axis=-1)
            self.frequency_cache.update(zip(missing, freq))
        frequency = np.stack([self.frequency_cache[key] for key in keys])
        return calibrated_conditional(logits, self.temperatures, frequency, self.neural_weight)


class FrozenWE:
    def __init__(self, inputs, game_values):
        self.terminal_tables, self.initial = {}, {}
        we, advancement = game_values['we'], game_values['advancement']
        for key, row in inputs.rows.items():
            game = GameState.from_row(row)
            self.terminal_tables[key] = terminal_values(game, we, advancement)
            self.initial[key] = float(we.predict_defense(game, game.defender_is_home))

    def terminal(self, state, event): return self.terminal_tables[state.context_key][event]
    def cutoff(self, state): return self.initial[state.context_key]


POLICY_INFERENCE = {'minimum_games': 30, 'minimum_supported_pa_starts': 50,
    'primary_comparisons': [['P1', 'P0'], ['P2', 'P1'], ['P3', 'P2']],
    'holm_alpha': .05, 'minimum_imputed_delta_we': .0001,
    'robust_test': 'max(imputed_p,worst_case_p); same game bootstrap draws',
    'bootstrap_draws': 10000, 'bootstrap_seed': 20260924}


def game_policy_comparisons(values, games, *, truncated=None, draws=10000, seed=20260924):
    return paired_policy_comparisons(values, games, POLICY_INFERENCE['primary_comparisons'],
                                     truncated=truncated, draws=draws, seed=seed)


def paired_policy_comparisons(values, games, pairs, *, truncated=None, draws=10000, seed=20260924):
    """PA-weighted conditional game bootstrap with explicit truncation robustness.

    Imputed-only Holm is a preliminary, tail-dependent screen. Strong model-
    internal improvement uses the intersection of imputed and worst-case tests,
    max(p_imputed,p_worst), then Holm across the caller's preregistered three/four contrasts.
    Below 30 games / 50 supported starts all inference slots stay null.
    """
    from .matrix_metrics import holm_adjust
    games = np.asarray(games)
    if games.ndim != 1 or not len(games) or draws < 1:
        raise ValueError('Nonempty game vector and bootstrap draws required')
    if pd.isna(games).any(): raise ValueError('Missing game IDs')
    unique, inverse = np.unique(games, return_inverse=True)
    counts = np.bincount(inverse)
    eligible = len(unique) >= 30 and len(games) >= 50
    samples = np.random.default_rng(seed).integers(len(unique), size=(draws, len(unique))) if eligible else None
    pairs = tuple(tuple(pair) for pair in pairs)
    if (len(pairs) not in (3, 4) or len(set(pairs)) != len(pairs)
            or any(len(pair) != 2 or pair[0] == pair[1] for pair in pairs)):
        raise ValueError('Complete registered three/four comparison family required')
    names = sorted({name for pair in pairs for name in pair})
    if set(values) != set(names): raise ValueError('Policy value family differs from declared comparisons')
    arrays = {name: np.asarray(values[name], dtype=float) for name in names}
    shape = arrays[names[0]].shape
    if (len(shape) != 2 or shape[0] != len(games) or shape[1] < 2
        or any(a.shape != shape or not np.isfinite(a).all() or ((a < 0) | (a > 1)).any() for a in arrays.values())):
        raise ValueError('Finite paired [PAstarts,rollouts>=2] WE arrays in [0,1] required')
    flags = None
    if truncated is not None:
        flags = {name: np.asarray(truncated[name]) for name in arrays}
        if any(a.shape != shape or a.dtype != np.bool_ for a in flags.values()):
            raise ValueError('Boolean truncation flags must match rollout arrays')

    def estimate(delta):
        mean = float(delta.mean())
        if not eligible: return mean, None, None
        sums = np.bincount(inverse, weights=delta)
        boot = sums[samples].sum(axis=1)/counts[samples].sum(axis=1)
        p = float((1+np.count_nonzero(boot-mean >= mean))/(draws+1))
        return mean, np.quantile(boot, [.025, .975]).tolist(), p

    result = []
    for candidate, control in pairs:
        a, b = arrays[candidate], arrays[control]
        mean, ci, p = estimate((a-b).mean(axis=1))
        lower, worst_ci, worst_p = None, None, None
        if flags is not None:
            worst = np.where(flags[candidate], 0., a)-np.where(flags[control], 1., b)
            lower, worst_ci, worst_p = estimate(worst.mean(axis=1))
        result.append({'candidate': candidate, 'control': control, 'mean_delta_we': mean,
            'game_bootstrap_ci95': ci, 'p_greater': p, 'games': len(unique), 'pa_starts': len(games),
            'family_size': len(pairs), 'reporting': {'status': 'reporting_eligible' if eligible else 'descriptive_only',
                          'minimum_games': 30, 'minimum_supported_pa_starts': 50},
            'worst_case_mean_delta_lower': lower, 'worst_case_game_bootstrap_ci95': worst_ci,
            'worst_case_p_greater': worst_p,
            'combined_p_greater': max(p, worst_p) if p is not None and worst_p is not None else None})
    preliminary = holm_adjust([row['p_greater'] for row in result])
    robust = holm_adjust([row['combined_p_greater'] for row in result])
    for row, imputed_holm, robust_holm in zip(result, preliminary, robust):
        row['imputed_holm_p'] = imputed_holm
        row['holm_p'] = robust_holm
        screen = bool(eligible and row['mean_delta_we'] >= .0001 and
                      row['game_bootstrap_ci95'][0] > 0 and imputed_holm <= .05)
        row['model_internal_P_screen'] = screen
        row['preliminary_screen_interpretation'] = 'Imputed-only, tail-assumption-dependent; not strong confirmation'
        strong = bool(screen and row['worst_case_game_bootstrap_ci95'] is not None and
            row['worst_case_game_bootstrap_ci95'][0] > 0 and row['worst_case_mean_delta_lower'] > 0
            and robust_holm is not None and robust_holm <= .05)
        row['untruncated_pa_improvement_confirmed'] = strong
        row['model_internal_screen'] = ('improved_under_model_and_truncation_bound' if strong else
            'tail_assumption_dependent' if screen else 'inconclusive' if eligible else 'descriptive_only')
        row['causal_P'] = None
        row['observational_ope'] = None
    return result
