"""Compact as-of batter history links and lazy common-input minibatches.

The second stream excludes the entire current PA. Same-game completed PAs are
available immediately; other same-date games are withheld. Earlier dates use
recorded date/game-key ordering, approximate within historical doubleheaders.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import KEY
from .matrix_features import MatrixHistoryStore


class MatrixLongHistoryStore:
    """O(N) integer links, with O(batch*128) history expansion only on demand."""
    MAX_HISTORY = 128

    def __init__(self, base, long_length=32):
        if long_length not in (0, 32, 128):
            raise ValueError('F4 fixes second-stream lengths to 0, 32 or 128')
        self.base, self.long_length = base, long_length
        self.frame, self.physical, self.normalizer = base.frame, base.physical, base.normalizer
        if self.frame.batter.isna().any():
            raise ValueError('Batter identities are required to construct observed history')
        if len(self.frame) > np.iinfo(np.int32).max:
            raise ValueError('Frame exceeds int32 link capacity')
        work = self.frame[['batter', 'game_date', 'game_pk', 'at_bat_number']].copy()
        work['position'] = np.arange(len(work), dtype=np.int32)
        dates = pd.to_datetime(work.game_date).dt.normalize()
        if dates.isna().any() or dates.dt.year.ge(2026).any() or not dates.is_monotonic_increasing:
            raise ValueError('Expected chronological permitted dates')
        work['game_date'] = dates
        if work.groupby('game_pk').game_date.nunique().gt(1).any():
            raise ValueError('A recorded game spanning dates needs an explicit suspended-game chronology contract')
        if work.groupby(['game_pk', 'at_bat_number']).batter.nunique().gt(1).any():
            raise ValueError('A PA cannot have multiple batter identities')
        self.previous_global = work.groupby('batter', sort=False).position.shift().fillna(-1).to_numpy(np.int32)
        self.previous_game = work.groupby(['game_pk', 'batter'], sort=False).position.shift().fillna(-1).to_numpy(np.int32)
        first_date = work.groupby(['batter', 'game_date'], sort=False).position.transform('min').to_numpy(np.int32)
        first_pa = work.groupby(['game_pk', 'at_bat_number'], sort=False).position.transform('min').to_numpy(np.int32)
        self.date_root = self.previous_global[first_date]
        self.game_root = self.previous_game[first_pa]

    @classmethod
    def from_frame(cls, frame, normalizer=None, type_vocabulary=None, long_length=32):
        base = MatrixHistoryStore.from_frame(frame, normalizer=normalizer, history_length=5,
                                             type_vocabulary=type_vocabulary)
        return cls(base, long_length)

    def with_length(self, long_length):
        """Share immutable O(N) links across 0/32/128 arms without refitting."""
        if long_length not in (0, 32, 128):
            raise ValueError('F4 fixes second-stream lengths to 0, 32 or 128')
        other = object.__new__(type(self))
        other.__dict__.update(self.__dict__)
        other.long_length = long_length
        return other

    def long_indices(self, rows):
        rows = np.asarray(rows, dtype=np.int64)
        if rows.ndim != 1 or (rows < 0).any() or (rows >= len(self.frame)).any():
            raise ValueError('Rows are outside the history store')
        indices = np.full((len(rows), self.MAX_HISTORY), -1, dtype=np.int32)
        game, dated = self.game_root[rows].copy(), self.date_root[rows].copy()
        for offset in range(self.long_length):
            use_game = game >= 0
            chosen = np.where(use_game, game, dated)
            indices[:, -1 - offset] = chosen
            observed = chosen >= 0
            from_game = use_game & observed
            from_date = ~use_game & observed
            game[from_game] = self.previous_game[chosen[from_game]]
            dated[from_date] = self.previous_global[chosen[from_date]]
        return indices

    def gather(self, rows, current=None, candidate_pitch_types=None):
        rows = np.asarray(rows, dtype=np.int64)
        h5, h5_valid = self.base.gather(rows, current=current, candidate_pitch_types=candidate_pitch_types)
        indices = self.long_indices(rows)
        valid = indices >= 0
        safe = np.maximum(indices, 0)
        long = np.concatenate((self.base.physical[safe], self.base.type_channels[safe],
                               self.base.outcome_channels[safe]), axis=-1)
        long[~valid] = 0.
        return h5, h5_valid, np.ascontiguousarray(long), valid

    def report(self):
        return {'version': 'batter_dual_stream_v1', 'h5': self.base.report(),
                'long_length': self.long_length, 'max_capacity': self.MAX_HISTORY,
                'link_storage_bytes': sum(a.nbytes for a in (self.previous_global, self.previous_game, self.date_root, self.game_root)),
                'scope': 'same batter; previous completed PAs in same game plus strictly prior dates; current PA excluded',
                'same_date_other_games': 'excluded without trustworthy game start chronology',
                'prior_date_doubleheaders': 'date/game-key order; approximate ordering within a historical date',
                'online_dev_history': 'earlier observed DEV pitches allowed by the same as-of rule',
                'retrospective_pa_support': 'never enters outcome tokens or history availability',
                'allocation': 'four int32 links per source row; batch-only expansion of max128 tokens'}


@dataclass
class LazyPitchBatch:
    store: MatrixLongHistoryStore
    context: object
    rows: np.ndarray
    current: np.ndarray | None = None
    candidate_pitch_types: np.ndarray | None = None

    def __post_init__(self):
        self.rows = np.asarray(self.rows, dtype=np.int64)
        if self.rows.ndim != 1 or (self.rows < 0).any() or (self.rows >= len(self.store.frame)).any():
            raise ValueError('Invalid lazy row references')
        if self.current is not None:
            self.current = np.asarray(self.current, dtype=np.float32)
            if self.current.shape != (len(self.rows), 8) or not np.isfinite(self.current).all():
                raise ValueError('Current override must be normalized finite eight-channel physics')
        if self.candidate_pitch_types is not None:
            self.candidate_pitch_types = np.asarray(self.candidate_pitch_types)
            if self.candidate_pitch_types.shape != (len(self.rows),):
                raise ValueError('One candidate type per requested row is required')

    def __len__(self):
        return len(self.rows)

    def subset(self, indices):
        return type(self)(self.store, self.context, self.rows[indices],
                          None if self.current is None else self.current[indices],
                          None if self.candidate_pitch_types is None else self.candidate_pitch_types[indices])

    def frame(self):
        frame = self.store.frame.iloc[self.rows]
        if self.candidate_pitch_types is not None:
            frame = frame.copy()
            frame['pitch_type'] = self.candidate_pitch_types
        return frame

    def gather(self):
        inputs = self.store.gather(self.rows, current=self.current,
                                  candidate_pitch_types=self.candidate_pitch_types)
        context = np.asarray(self.context.transform(self.frame()), dtype=np.float32)
        if context.ndim != 2 or len(context) != len(self.rows) or not np.isfinite(context).all():
            raise ValueError('Invalid lazy context encoding')
        return (*inputs, context)
