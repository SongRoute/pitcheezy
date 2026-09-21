"""Independent July-only evaluator of *observed* pitch-location outcomes.

Code for pre-pitch context is reused, learned parameters and all count priors
are independently fitted. This is not an evaluator of identified target intent.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.special import expit

PROJECT = Path(__file__).resolve().parents[3]/'experiments/pitchmdp'
sys.path[:0] = [str(PROJECT/'scripts'), str(PROJECT)]
from regularized_policy_evaluator import RegularizedPolicyEvaluator


CENTERS = np.array([(x, z) for z in (1.5, 2.5, 3.5) for x in (-.83, 0., .83)], dtype=float)


def location_features(frame):
    xy = frame[['plate_x', 'plate_z']].to_numpy(dtype=float)
    if not np.isfinite(xy).all():
        raise ValueError('Observed location evaluator requires finite plate coordinates')
    x, z = np.clip(xy[:, 0], -5, 5), np.clip(xy[:, 1]-2.5, -5, 5)
    rbf = np.exp(-np.square(xy[:, None, :]-CENTERS[None, :, :]).sum(-1)/(2*.65**2))
    edges = expit(np.column_stack([(.83-xy[:, 0]), (.83+xy[:, 0]),
                                   (xy[:, 1]-1.5), (3.5-xy[:, 1])])/.15)
    return np.column_stack([x, z, x*x, z*z, x*z, rbf, edges])


class LocationEvaluator(RegularizedPolicyEvaluator):
    NUMERIC_NAMES = [*RegularizedPolicyEvaluator.NUMERIC_NAMES,
                     'location_x', 'location_z', 'location_x2', 'location_z2', 'location_xz',
                     *[f'location_rbf_{i}' for i in range(9)],
                     'inside_right', 'inside_left', 'above_bottom', 'below_top']

    def _numeric(self, frame):
        return np.column_stack([super()._numeric(frame), location_features(frame)])

    def _features(self, frame):
        base = super()._features(frame)
        # Shared location shape plus type-specific residuals, avoiding sparse
        # independent outcome tables for every exact location and game state.
        loc = (location_features(frame)-self.mean[-18:])/self.scale[-18:]
        kinds = frame.pitch_type.astype(str).to_numpy()
        indicators = np.column_stack([kinds == kind for kind in self.pitch_types])
        interactions = (indicators[:, :, None]*loc[:, None, :]).reshape(len(frame), -1)
        return np.column_stack([base, interactions])

    def fit(self, frame):
        super().fit(frame)
        self.report['kind'] = 'july_independent_observed_location_residual_logistic'
        self.report['feature_names'] += [f'{kind}:{name}' for kind in self.pitch_types for name in self.NUMERIC_NAMES[-18:]]
        self.report['scope'] = ('July-only fit/calibration; observed-location conditional association, '
                                'not intended-target intervention. No frozen MVP learned weights or count priors.')
        train = frame.loc[pd.to_datetime(frame.game_date).le('2025-07-21')].copy()
        train['location_cell'] = location_cells(train)
        self.location_counts = train.groupby([*self.SUPPORT_KEYS, 'location_cell'], observed=True).size().to_dict()
        return self

    def local_support(self, frame):
        work = frame[self.SUPPORT_KEYS].copy()
        work['location_cell'] = location_cells(frame)
        return np.array([self.location_counts.get(key, 0) for key in work.itertuples(index=False, name=None)])


def location_cells(frame):
    x, z = frame.plate_x.to_numpy(float), frame.plate_z.to_numpy(float)
    column = np.clip(((x+.83)/1.66*3).astype(int), 0, 2)
    row = np.clip(((z-1.5)/2*3).astype(int), 0, 2)
    cell = row*3+column
    # Four deterministic outside regions, corners assigned horizontally first.
    return np.where(x < -.83, 9, np.where(x > .83, 10, np.where(z < 1.5, 11, np.where(z > 3.5, 12, cell))))


def legal(probabilities, frame):
    result = np.array(probabilities, dtype=float, copy=True)
    impossible = frame.outs_when_up.eq(2).to_numpy() | frame.bases.eq(0).to_numpy()
    result[impossible, 9] = 0
    result /= result.sum(axis=1, keepdims=True)
    return result


def row_scores(y, probabilities):
    return {'log_loss': -np.log(np.clip(probabilities[np.arange(len(y)), y], 1e-15, 1)),
            'brier': np.square(probabilities-np.eye(10)[y]).sum(axis=1)}


def paired_games(difference, game_ids, replicates=2000, seed=20260921):
    values = pd.DataFrame({'game': game_ids, 'difference': difference}).groupby('game').difference.agg(['sum', 'count'])
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(values), (replicates, len(values)))
    totals, sizes = values['sum'].to_numpy(), values['count'].to_numpy()
    draws = totals[samples].sum(axis=1)/sizes[samples].sum(axis=1)
    return {'difference': float(np.mean(difference)), 'ci95': np.quantile(draws, [.025, .975]).tolist(),
            'games': len(values), 'rows': len(difference), 'bootstrap_replicates': replicates, 'seed': seed}
