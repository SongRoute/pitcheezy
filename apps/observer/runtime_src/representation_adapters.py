"""Generated prediction runtime: exact selected definitions from a hash-verified capture.
Training methods may remain for class identity; the service does not invoke them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pitchmdp.archetypes import Archetypes


def _require_train(frame):
    if not len(frame) or 'split' not in frame or not frame.split.eq('train').all():
        raise ValueError('Adapter fitting requires nonempty TRAIN rows only')
    if 'game_date' in frame and not pd.to_datetime(frame.game_date).between('2023-05-15', '2025-04-30').all():
        raise ValueError('Adapter fit dates outside approved TRAIN period')

def _base_features(frame):
    """Exactly the frozen eleven game/count/hand channels; no batting outcomes."""
    def col(name):
        return pd.to_numeric(frame[name], errors='coerce').fillna(0).to_numpy(np.float32)
    bases = col('bases').astype(int)
    top = frame.inning_topbot.eq('Top').to_numpy()
    return np.column_stack([col('balls')/3, col('strikes')/2, col('outs_when_up')/2,
        col('inning')/9, np.where(top, 1., -1.)*(col('home_score')-col('away_score'))/5,
        top.astype(float), (bases & 1).astype(float), ((bases >> 1) & 1).astype(float),
        ((bases >> 2) & 1).astype(float), frame.stand.eq('L').to_numpy(float),
        frame.p_throws.eq('L').to_numpy(float)]).astype(np.float32)

def _batter_keys(frame):
    keys = pd.to_numeric(frame.batter, errors='coerce')
    if (keys.notna() & ((keys % 1) != 0)).any():
        raise ValueError('Batter join keys must be integer IDs')
    return keys.astype('Int64')

class BatterContext:
    def __init__(self, base_context, mode, clusters=None):
        if mode not in ('hand', 'id', 'continuous', 'clusters', 'reference'):
            raise ValueError('Unknown batter representation')
        if mode == 'clusters' and clusters not in (3, 5, 10, 20):
            raise ValueError('Cluster-only representation requires K in 3, 5, 10, 20')
        if mode != 'clusters' and clusters is not None:
            raise ValueError('Cluster count is only applicable to cluster-only mode')
        self.base_context, self.mode, self.clusters = base_context, mode, clusters

    def fit(self, full_train, id_train=None):
        _require_train(full_train)
        self.fit_rows = len(full_train)
        self.fit_date_max = str(pd.to_datetime(full_train.game_date).max().date()) if 'game_date' in full_train else None
        if self.mode in ('continuous', 'reference') and self.fit_date_max is not None:
            base_max = self.base_context.archetypes.report().get('fit_date_max')
            if base_max is None or pd.Timestamp(base_max) > pd.Timestamp(self.fit_date_max):
                raise ValueError('Reference batting geometry extends beyond this TRAIN pool')
        if self.mode == 'clusters':
            self.archetypes = Archetypes(n_clusters=self.clusters, seed=42).fit(full_train)
        if self.mode == 'id':
            if id_train is None:
                raise ValueError('ID vocabulary requires explicit shared neural TRAIN sample')
            _require_train(id_train)
            if self.fit_date_max is not None and 'game_date' in id_train and pd.to_datetime(id_train.game_date).max() > pd.Timestamp(self.fit_date_max):
                raise ValueError('ID vocabulary dates extend beyond full TRAIN pool')
            key_columns = ['game_pk', 'at_bat_number', 'pitch_number']
            if set(key_columns).issubset(full_train.columns) and set(key_columns).issubset(id_train.columns):
                full_keys = pd.MultiIndex.from_frame(full_train[key_columns])
                if not pd.MultiIndex.from_frame(id_train[key_columns]).isin(full_keys).all():
                    raise ValueError('ID vocabulary rows must belong to full TRAIN pool')
            ids = sorted(int(value) for value in _batter_keys(id_train).dropna().unique())
            self.id_map = {value: index+1 for index, value in enumerate(ids)}
            self.vocab_size = len(self.id_map)+1
            self.vocabulary_rows = len(id_train)
        self.fitted = True
        return self

    def transform(self, frame):
        if not getattr(self, 'fitted', False):
            raise ValueError('BatterContext must be fitted before transformation')
        if self.mode == 'reference':
            return self.base_context.transform(frame)
        base = _base_features(frame)
        if self.mode == 'hand':
            return base
        if self.mode == 'id':
            indices = _batter_keys(frame).map(self.id_map).fillna(0).to_numpy(np.float32)
            return np.column_stack([base, indices]).astype(np.float32)
        if self.mode == 'continuous':
            batting = self.base_context.archetypes.numeric_features(frame)[:, :12]
        else:
            batting = self.archetypes.transform(frame)
        return np.column_stack([base, batting]).astype(np.float32)

    def report(self):
        result = {'mode': self.mode, 'base_channels': 11, 'fit_rows': self.fit_rows,
                  'fit_date_max': self.fit_date_max, 'game_count_handedness_preserved': True,
                  'continuous_normalization': 'Frozen reference TRAIN geometry',
                  'batter_id_input': self.mode == 'id'}
        if self.mode == 'id':
            result.update(vocab_size=self.vocab_size, observed_train_ids=len(self.id_map),
                          vocabulary_rows=self.vocabulary_rows, unknown_index=0,
                          unknown_embedding='Fixed zero; padding_idx=0', embedding_width=16)
        if self.mode == 'clusters':
            result['archetypes'] = self.archetypes.report()
        result['n_context'] = {'hand': 11, 'id': 12, 'continuous': 23, 'reference': 28,
                               'clusters': 11+(self.clusters or 0)}[self.mode]
        return result
