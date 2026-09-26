"""Fixed-input T3 stress adapters; no fitting, calibration, selection or scoring.

Only historical tokens are perturbed. Every random variate is a function of the
observed historical pitch key, so repeated queries and delivery draws share one
measurement. Original stores, context encoders and models are never modified.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

from .archetypes import INITIAL_RATES, RELIABILITY_COLUMNS, STYLE_COLUMNS
from .data import KEY
from .matrix_benchmark import predict_streamed
from .matrix_features import MatrixHistoryStore
from .matrix_sharing import SharingContext


SEED = 20260924
SCENARIOS = ('clean', 'H2', 'H0', 'mask20', 'mask50', 'unknown_type_outcome',
             'sensor01', 'sensor03', 'unknown_pitcher', 'unknown_batter')
NON_ANGLE = np.array([0, 1, 4, 5, 6, 7])


def protocol():
    """Exact feature contract to copy into a preregistration before any scores."""
    return {'version': 'ml_stress_h5_v1', 'seed': SEED, 'scenarios': list(SCENARIOS),
        'history_length': 5, 'draws': 400,
        'mask_probability': {'mask20': .2, 'mask50': .5},
        'sensor_noise': {'sensor01': {'normalized_sigma': .1, 'spin_sd_degrees': 5.},
                         'sensor03': {'normalized_sigma': .3, 'spin_sd_degrees': 15.}},
        'randomness': 'SHA256(version|seed|domain|game_pk|at_bat_number|pitch_number); Box-Muller normal',
        'shared_randomness': 'same historical key across query positions, batches, models and all delivery draws; nested mask/noise strengths',
        'current': 'candidate type unchanged; current outcome channels zero; supplied delivery physics mandatory and untouched',
        'pitcher_missing': 'zero standardized profile/count, unknown cluster flag, cluster=-1 and routing pitcher=-1; frozen delivery still uses original identity',
        'batter_missing': {'rates': INITIAL_RATES.tolist(), 'evidence': 'zero',
                           'membership': 'recomputed by frozen encoder, hence uniform'},
        'frozen': ['weights', 'normalizers', 'vocabularies', 'centroids', 'delivery pools/draws',
                   'temperature', 'blend weights', 'requested keys and labels'],
        'scope': 'hypothetical input perturbations; not an empirical sensor-error distribution; no training or calibration'}


def _scenario(value):
    if value not in SCENARIOS:
        raise ValueError('Unregistered H5 stress scenario')
    return value


def _uniform(key, domain):
    payload = '|'.join(map(str, ('ml_stress_h5_v1', SEED, domain, *key))).encode('ascii')
    # Exactly representable 52-bit midpoint; neither endpoint can occur.
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], 'big') >> 12
    return (integer + .5) / 2**52


def _normal(key, domain):
    return math.sqrt(-2. * math.log(_uniform(key, domain + ':u'))) * math.cos(
        2. * math.pi * _uniform(key, domain + ':v'))


class StressHistoryStore:
    """Read-only wrapper compatible with JointDelivery and predict_streamed."""
    def __init__(self, base, scenario):
        if not isinstance(base, MatrixHistoryStore) or base.indices.shape[1] != 5:
            raise ValueError('T3 requires the frozen enriched H5 store')
        self.base, self.scenario = base, _scenario(scenario)
        self.frame, self.indices, self.normalizer = base.frame, base.indices, base.normalizer
        self.type_vocabulary, self.n_types = base.type_vocabulary, base.n_types
        if self.frame[KEY].isna().any().any() or self.frame.duplicated(KEY).any():
            raise ValueError('Unique complete historical pitch keys required')
        raw_keys = self.frame[KEY].to_numpy()
        self.keys = raw_keys.astype(np.int64)
        if not np.array_equal(raw_keys, self.keys):
            raise ValueError('Integral historical pitch keys required')

    def gather(self, rows, current=None, candidate_pitch_types=None):
        if current is None:
            raise ValueError('T3 inference requires supplied current delivery draws')
        rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        tokens, valid = self.base.gather(rows, current, candidate_pitch_types)
        # MatrixHistoryStore allocates new arrays; copies make wrapper ownership explicit.
        tokens, valid = tokens.copy(), valid.copy()
        if self.scenario in ('H2', 'H0'):
            keep = 2 if self.scenario == 'H2' else 0
            valid[:, :5 - keep] = False
        elif self.scenario == 'unknown_type_outcome':
            tokens[:, :-1, 8:] = 0.
            tokens[:, :-1, 8] = 1.
            tokens[:, :-1, -1] = 1.
        elif self.scenario in ('mask20', 'mask50', 'sensor01', 'sensor03'):
            history = self.indices[rows]
            present = history >= 0
            unique, inverse = np.unique(history[present], return_inverse=True)
            keys = self.keys[unique]
            if self.scenario.startswith('mask'):
                threshold = .2 if self.scenario == 'mask20' else .5
                dropped = np.array([_uniform(k, 'missing') < threshold for k in keys], dtype=bool)
                valid[:, :-1][present] &= ~dropped[inverse]
            else:
                sigma, degrees = (.1, 5.) if self.scenario == 'sensor01' else (.3, 15.)
                noise = np.array([[_normal(k, f'physical:{j}') for j in NON_ANGLE]
                                  for k in keys], dtype=float).reshape(-1, 6)
                angle = np.array([_normal(k, 'spin') for k in keys]) * np.deg2rad(degrees)
                values = tokens[:, :-1][present].astype(float)
                values[:, NON_ANGLE] += sigma * noise[inverse]
                # Rotate unscaled sine/cosine together. Preserve the radius of an
                # imputed pair rather than inventing a measured unit-length angle.
                mean, scale = np.asarray(self.normalizer.mean), np.asarray(self.normalizer.scale)
                sine, cosine = (values[:, 2:4] * scale[2:4] + mean[2:4]).T
                c, s = np.cos(angle[inverse]), np.sin(angle[inverse])
                rotated = np.column_stack((sine*c + cosine*s, cosine*c - sine*s))
                values[:, 2:4] = (rotated - mean[2:4]) / scale[2:4]
                tokens[:, :-1][present] = values
        tokens[~valid] = 0.
        return np.ascontiguousarray(tokens), valid

    def report(self):
        return {'base': self.base.report(), 'scenario': self.scenario, 'protocol': protocol()}


class StressSharingContext:
    """Reconstruct a missing feature block through the existing frozen encoder."""
    def __init__(self, base, scenario):
        if not isinstance(base, SharingContext):
            raise ValueError('T3 G predictors require SharingContext including routing metadata')
        self.base, self.scenario = base, _scenario(scenario)

    def transform(self, frame):
        query = frame
        if self.scenario == 'unknown_batter':
            query = frame.copy()
            for name, value in zip(STYLE_COLUMNS, INITIAL_RATES):
                query[name] = value
            for name in RELIABILITY_COLUMNS:
                query[name] = 0.
        result = self.base.transform(query).copy()
        if self.scenario == 'unknown_pitcher':
            width = len(self.base.clusters['columns'])
            # Last columns: profile[width], count[1], flags[5], cluster[1], ID[1].
            result[:, -(width + 8):-7] = 0.
            result[:, -7:-2] = 0.
            result[:, -3] = 1.  # fifth cluster flag = unknown
            result[:, -2:] = -1.  # disable BOTH cluster and personal routing
        return result

    def report(self):
        return {'base': self.base.report(), 'scenario': self.scenario, 'protocol': protocol()}


def predict_stress_streamed(model, delivery, store, context, rows, scenario, *, chunk_size=16):
    """Return calibrated, raw, tier arrays; caller applies archived blend weights.

    The orchestration layer must verify frozen artifact hashes, complete members,
    shared heavy-lock ownership, configuration and common ordered evaluation keys.
    This adapter never recalibrates and never writes to the supplied objects.
    """
    if delivery.draws != 400:
        raise ValueError('T3 fixes the existing 400 delivery draws')
    if not np.isfinite(model.delivery_temperature) or model.delivery_temperature <= 0:
        raise ValueError('An archived positive delivery temperature is required')
    return predict_streamed(model, delivery, StressHistoryStore(store, scenario),
                            StressSharingContext(context, scenario), rows, chunk_size)
