"""Past-only batting profiles and training-fitted, soft similarity groups.

Player IDs are history join keys, never inference inputs. Memberships describe a
dated statistical profile, not an immutable player class or calibrated posterior.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


STYLE_NAMES = ('contact', 'swing', 'walk', 'strikeout', 'isolated_power', 'groundball')
STYLE_COLUMNS = tuple(f'batter_style_{name}_prior' for name in STYLE_NAMES)
RELIABILITY_COLUMNS = tuple(f'batter_style_{name}_reliability' for name in STYLE_NAMES)
HISTORY_COLUMNS = STYLE_COLUMNS + RELIABILITY_COLUMNS
# Fixed initialization assumptions, not rates estimated from held-out data.
INITIAL_RATES = np.array([.76, .47, .085, .225, .165, .43])
PRIOR_STRENGTH = np.array([100., 200., 60., 60., 60., 60.])


def add_batter_style_history(frame: pd.DataFrame) -> pd.DataFrame:
    """Append six shrunken rates + evidence weights in place; return frame.

    Call on the complete chronological pitch pool BEFORE outcome/support filters.
    Input order is immaterial. Each date, including doubleheaders, shares one
    strictly prior-date profile. League fallback also excludes the entire date.
    Missing launch angles reduce evidence rather than count as non-groundballs.
    """
    required = {'batter', 'game_date', 'events', 'description', 'is_pa_terminal'}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f'Missing style history columns: {sorted(missing)}')
    dates = pd.to_datetime(frame.game_date).dt.normalize()
    if frame.batter.isna().any() or dates.isna().any():
        raise ValueError('Style history requires batter and game_date join keys')
    if dates.dt.year.ge(2026).any():
        raise ValueError('This experiment does not allow 2026 or later records')
    d = frame.description.fillna('')
    e = frame.events.fillna('')
    terminal = frame.is_pa_terminal.fillna(False).astype(bool)
    contact = d.isin(['foul', 'foul_tip', 'hit_into_play'])
    swing = contact | d.isin(['swinging_strike', 'swinging_strike_blocked'])
    decision = swing | d.isin(['ball', 'blocked_ball', 'called_strike'])
    # Bunts, pitchouts and intentional balls are outside the swing profile.
    pa = terminal & ~e.isin(['intent_walk', 'catcher_interf', 'truncated_pa', ''])
    ab = terminal & e.isin([
        'single', 'double', 'triple', 'home_run', 'field_out', 'force_out',
        'fielders_choice', 'fielders_choice_out', 'field_error', 'strikeout',
        'strikeout_double_play', 'grounded_into_double_play', 'double_play',
        'triple_play',
    ])
    angle = pd.to_numeric(frame.get('launch_angle', pd.Series(np.nan, index=frame.index)), errors='coerce')
    measured_bip = d.eq('hit_into_play') & angle.notna()
    numerators = [contact, swing, pa & e.eq('walk'),
                  pa & e.isin(['strikeout', 'strikeout_double_play']),
                  e.map({'double': 1, 'triple': 2, 'home_run': 3}).fillna(0) * ab,
                  measured_bip & angle.lt(10)]
    denominators = [swing, decision, pa, pa, ab, measured_bip]
    counts = pd.DataFrame({'batter': frame.batter.to_numpy(), 'game_date': dates.to_numpy()})
    for j, (num, den) in enumerate(zip(numerators, denominators)):
        counts[f'n{j}'] = np.asarray(num, dtype=np.float64)
        counts[f'd{j}'] = np.asarray(den, dtype=np.float64)
    daily = counts.groupby(['batter', 'game_date'], sort=True).sum()
    prior = daily.groupby(level='batter').cumsum() - daily
    league_daily = daily.groupby(level='game_date').sum().sort_index()
    league_prior = league_daily.cumsum() - league_daily
    lookup = pd.MultiIndex.from_arrays([frame.batter.to_numpy(), dates.to_numpy()], names=['batter', 'game_date'])
    daily_dates = daily.index.get_level_values('game_date')
    for j, (rate_col, weight_col) in enumerate(zip(STYLE_COLUMNS, RELIABILITY_COLUMNS)):
        # A small fixed league pseudo-sample handles the first observed date.
        league_rate = (league_prior[f'n{j}'] + 500 * INITIAL_RATES[j]) / (league_prior[f'd{j}'] + 500)
        fallback = league_rate.reindex(daily_dates).to_numpy()
        den = prior[f'd{j}'].to_numpy()
        rate = (prior[f'n{j}'].to_numpy() + PRIOR_STRENGTH[j] * fallback) / (den + PRIOR_STRENGTH[j])
        weight = den / (den + PRIOR_STRENGTH[j])
        frame[rate_col] = pd.Series(rate, index=daily.index).reindex(lookup).to_numpy(np.float32)
        frame[weight_col] = pd.Series(weight, index=daily.index).reindex(lookup).to_numpy(np.float32)
    return frame


class Archetypes:
    """Weighted k-means geometry with soft, uncertainty-attenuated membership.

    Five centers are an exploratory capacity choice, not five biological types.
    Centers/scaling are fitted once on TRAIN snapshots and frozen thereafter.
    """
    def __init__(self, n_clusters: int = 5, seed: int = 42):
        if n_clusters < 1:
            raise ValueError('n_clusters must be positive')
        self.n_clusters = n_clusters
        self.seed = seed

    @staticmethod
    def _values(frame):
        x = frame[list(STYLE_COLUMNS)].to_numpy(dtype=np.float64)
        evidence = frame[list(RELIABILITY_COLUMNS)].to_numpy(dtype=np.float64)
        if not np.isfinite(x).all() or not np.isfinite(evidence).all():
            raise ValueError('Style profiles must be finite; construct chronological histories first')
        if (evidence < 0).any() or (evidence > 1).any():
            raise ValueError('Style reliability must lie in [0, 1]')
        return x, evidence

    def fit(self, train: pd.DataFrame):
        if 'split' not in train or not train.split.eq('train').all():
            raise ValueError('Archetype fitting requires only explicitly marked train rows')
        # One snapshot per player-date, no repeated weighting by pitch count.
        snapshots = train.drop_duplicates(['batter', 'game_date'])
        x, evidence = self._values(snapshots)
        if len(x) < self.n_clusters:
            raise ValueError('Fewer training snapshots than requested centers')
        counts = snapshots.groupby('batter').batter.transform('size').to_numpy(float)
        weight = evidence.mean(axis=1) / counts
        if weight.sum() <= 0:
            raise ValueError('Training snapshots contain no prior batting evidence')
        weight /= weight.sum()
        self.mean = np.sum(x * weight[:, None], axis=0)
        self.scale = np.maximum(np.sqrt(np.sum((x-self.mean)**2 * weight[:, None], axis=0)), .01)
        z = (x-self.mean)/self.scale
        rng = np.random.default_rng(self.seed)
        centers = [z[rng.choice(len(z), p=weight)].copy()]
        # Weighted k-means++ initialization; duplicate centers are permitted only
        # when the entire pool has fewer distinct profiles than requested groups.
        for _ in range(1, self.n_clusters):
            distance = np.min(((z[:, None]-np.asarray(centers))**2).sum(axis=2), axis=1)
            probability = weight * distance
            ix = rng.choice(len(z), p=probability/probability.sum() if probability.sum() > 1e-12 else weight)
            centers.append(z[ix].copy())
        self.centers = np.asarray(centers)
        for iteration in range(100):
            distance = ((z[:, None]-self.centers)**2).sum(axis=2)
            assignment = distance.argmin(axis=1)
            updated = self.centers.copy()
            for k in range(self.n_clusters):
                mask = assignment == k
                if weight[mask].sum() > 0:
                    updated[k] = np.average(z[mask], axis=0, weights=weight[mask])
            converged = np.max(np.abs(updated-self.centers)) < 1e-6
            self.centers = updated
            if converged:
                break
        # Stable presentation order, no semantic labels inferred from group IDs.
        self.centers = self.centers[np.argsort(self.centers[:, 4], kind='stable')]
        residual = np.min(((z[:, None]-self.centers)**2).sum(axis=2), axis=1)
        self.bandwidth_squared = max(float(np.sum(weight*residual)), .25)
        self.n_snapshots = len(snapshots)
        self.n_batters = int(snapshots.batter.nunique())
        self.iterations = iteration + 1
        self.fit_date_max = str(pd.to_datetime(snapshots.game_date).max().date())
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        x, evidence = self._values(frame)
        z = (x-self.mean)/self.scale
        distance = ((z[:, None]-self.centers)**2).sum(axis=2)
        logits = -distance/(2*self.bandwidth_squared)
        logits -= logits.max(axis=1, keepdims=True)
        membership = np.exp(logits)
        membership /= membership.sum(axis=1, keepdims=True)
        # An unseen player has no defensible cluster assignment. His observed
        # batting side remains available separately in the model's stand field.
        confidence = evidence.mean(axis=1, keepdims=True)
        return (confidence*membership + (1-confidence)/self.n_clusters).astype(np.float32)

    def numeric_features(self, frame: pd.DataFrame) -> np.ndarray:
        x, evidence = self._values(frame)
        return np.column_stack([(x-self.mean)/self.scale, evidence, self.transform(frame)]).astype(np.float32)

    def report(self) -> dict:
        return {
            'method': 'training-only weighted k-means with soft radial memberships',
            'n_clusters': self.n_clusters, 'seed': self.seed,
            'style_columns': list(STYLE_COLUMNS), 'reliability_columns': list(RELIABILITY_COLUMNS),
            'n_snapshots': self.n_snapshots, 'n_batters': self.n_batters,
            'fit_date_max': self.fit_date_max, 'iterations': self.iterations,
            'mean': self.mean.tolist(), 'scale': self.scale.tolist(),
            'centers_raw_rates': (self.centers*self.scale+self.mean).tolist(),
            'bandwidth_squared': self.bandwidth_squared,
            'weighting': 'equal total player weight before prior-evidence attenuation; unique player-date snapshots',
            'membership': 'similarity weights, not calibrated probabilities; shrink toward uniform by mean reliability',
            'handedness': 'observed stand is a separate model category, excluded from clustering',
            'limitations': 'fixed exploratory K=5 default; cumulative profiles without aging or opponent adjustment; no stable taxonomy claim',
        }
