"""Shared pre-pitch information contract for the ML2 architecture comparison.

Physical TRAIN targets remain conditional. At pre-pitch inference JointDelivery
must replace the eight current physical channels. Every architecture receives
identical ordered past physical/type/outcome channels and padding masks.
"""
from __future__ import annotations

import numpy as np

from .model import OUTCOMES, outcome_labels
from .sequence_data import HistoryStore


class MatrixHistoryStore:
    def __init__(self, base, type_vocabulary):
        self.base = base
        self.frame, self.physical = base.frame, base.physical
        self.indices, self.normalizer = base.indices, base.normalizer
        self.type_vocabulary = tuple(type_vocabulary)
        if len(set(self.type_vocabulary)) != len(self.type_vocabulary):
            raise ValueError('Duplicate pitch-type vocabulary')
        self.type_map = {value: i + 1 for i, value in enumerate(self.type_vocabulary)}
        self.n_types = len(self.type_map) + 1
        self.type_channels = self._encode_types(self.frame.pitch_type)
        labels = outcome_labels(self.frame)
        # Whole-PA supported_pa depends on later events and MUST NOT be an input.
        # A previous pitch's description/events are already observed at query time.
        labels = np.where(labels >= 0, labels, len(OUTCOMES))
        self.outcome_channels = np.eye(len(OUTCOMES) + 1, dtype=np.float32)[labels]

    @classmethod
    def from_frame(cls, frame, normalizer=None, history_length=5, type_vocabulary=None):
        if not isinstance(history_length, int) or history_length < 0:
            raise ValueError('history_length must be a nonnegative integer')
        base = HistoryStore.from_frame(frame, normalizer, max(1, history_length))
        if history_length == 0:
            base.indices = base.indices[:, :0]
        if type_vocabulary is None:
            train = frame.loc[frame.split.eq('train')]
            if not len(train):
                raise ValueError('Type vocabulary requires TRAIN or an explicit frozen vocabulary')
            type_vocabulary = sorted(train.pitch_type.dropna().astype(str).unique())
        return cls(base, type_vocabulary)

    def _encode_types(self, values):
        ids = [self.type_map.get(str(value), 0) for value in values]
        return np.eye(self.n_types, dtype=np.float32)[ids]

    def gather(self, rows, current=None, candidate_pitch_types=None):
        rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        physical, valid = self.base.gather(rows, current=current)
        history = self.indices[rows]
        n, length = valid.shape
        types = np.zeros((n, length, self.n_types), dtype=np.float32)
        outcomes = np.zeros((n, length, len(OUTCOMES) + 1), dtype=np.float32)
        types[:, :-1] = self.type_channels[np.maximum(history, 0)]
        outcomes[:, :-1] = self.outcome_channels[np.maximum(history, 0)]
        if candidate_pitch_types is not None and len(candidate_pitch_types) != n:
            raise ValueError('One candidate pitch type per requested row is required')
        types[:, -1] = (self._encode_types(self.frame.iloc[rows].pitch_type) if candidate_pitch_types is None
                        else self._encode_types(candidate_pitch_types))
        # No current outcome, including no current unknown/missingness flag.
        tokens = np.concatenate((physical, types, outcomes), axis=-1)
        tokens[~valid] = 0.
        return np.ascontiguousarray(tokens), valid

    def report(self):
        return {'version': 'enriched_h5_v1', 'history_length': self.indices.shape[1],
                'physical_channels': 8, 'type_vocabulary': list(self.type_vocabulary),
                'type_channels': self.n_types, 'unknown_type_index': 0,
                'outcome_order': [*OUTCOMES, 'unknown'], 'outcome_channels': 11,
                'token_channels': 8 + self.n_types + 11,
                'current_outcome': 'all zero; no outcome or outcome availability is exposed',
                'past_outcome': '10-class observed past outcome; unmapped label unknown; retrospective PA support never used',
                'current_physics': 'conditional TRAIN only; JointDelivery must override at pre-pitch inference',
                'history': 'same PA, strictly previous rows, oldest to newest, left padding masked',
                'type_fit': 'TRAIN vocabulary; unknown category for unseen or missing pitch type',
                'candidate_type': 'read from current frame row at gather time; candidate query must update frame.pitch_type or pass candidate_pitch_types'}
