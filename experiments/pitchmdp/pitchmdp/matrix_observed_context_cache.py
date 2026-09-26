"""Optional exact memoization of the frozen F4 observed-row context encoder.

Not adopted by any experiment runner. This cache is invalid for simulated
policy states: game/count/hand/style/identity changes are rejected. Candidate
pitch_type and current physical replacements do not enter this encoder.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import time

import numpy as np
import pandas as pd

from .archetypes import Archetypes, STYLE_COLUMNS, RELIABILITY_COLUMNS
from .data import KEY
from .matrix_data import canonical_hash, ordered_key_hash
from .matrix_sharing import ContinuousPitcherContext, SharingContext
from .sequence_model import SequenceContext


# Exact dependencies of SequenceContext/Archetypes/SharingContext; identity and
# chronology fields additionally prevent treating another observed row as this one.
CONTEXT_COLUMNS = ('balls', 'strikes', 'outs_when_up', 'inning', 'home_score',
    'away_score', 'inning_topbot', 'bases', 'stand', 'p_throws', 'pitcher',
    *STYLE_COLUMNS, *RELIABILITY_COLUMNS)
GUARDED_COLUMNS = tuple(dict.fromkeys((*KEY, 'game_date', 'batter', *CONTEXT_COLUMNS)))


class FrozenObservedContext:
    """Read-only float32 context rows plus exact (not hashed) input guards.

    Build on the union of the registered observed query rows, in bounded chunks.
    No fitting or RNG calls. Returned fancy-indexed arrays are independent copies.
    The unchanged ``report`` is the scientific feature contract; cache diagnostics
    are separate, so a new run must explicitly source-pin/register this adapter.
    """
    def __init__(self, base, frame, rows, *, chunk_size=8192):
        if (type(base) is not ContinuousPitcherContext or type(base.sharing) is not SharingContext
            or type(base.sharing.base) is not SequenceContext
            or type(base.sharing.base.archetypes) is not Archetypes):
            raise ValueError('Only the frozen ContinuousPitcherContext/SequenceContext encoder is supported')
        if not frame.index.equals(pd.RangeIndex(len(frame))):
            raise ValueError('Cache requires canonical global observed row positions')
        if type(chunk_size) is not int or chunk_size < 1:
            raise ValueError('Positive context construction chunk size required')
        rows = np.asarray(rows)
        if (rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer) or not len(rows)
            or (rows < 0).any() or (rows >= len(frame)).any() or len(frame) > np.iinfo(np.int32).max):
            raise ValueError('Nonempty integer observed query rows within int32 store capacity required')
        rows = np.unique(rows.astype(np.int64))
        started = time.perf_counter()
        self._report = deepcopy(base.report())
        # Preserve the frozen encoder's existing empty-frame shape as well.
        self._empty = np.asarray(base.transform(frame.iloc[:0]), dtype=np.float32).copy()
        self._inputs = frame.iloc[rows].loc[:, GUARDED_COLUMNS].copy(deep=True)
        self._positions = np.full(len(frame), -1, dtype=np.int32)
        self._positions[rows] = np.arange(len(rows), dtype=np.int32)
        self._values = np.empty((len(rows), 52), dtype=np.float32)
        for begin in range(0, len(rows), chunk_size):
            end = min(begin+chunk_size, len(rows))
            values = base.transform(frame.iloc[rows[begin:end]])
            if values.dtype != np.float32 or values.shape != (end-begin, 52) or not np.isfinite(values).all():
                raise ValueError('Frozen F4 context must produce exactly 52 finite float32 channels')
            self._values[begin:end] = values
        if base.report() != self._report:
            raise ValueError('Context encoder changed during cache construction')
        self._positions.flags.writeable = False
        self._values.flags.writeable = False
        self._diagnostics = {'version': 'f4_observed_context_cache_v1', 'rows': len(rows),
            'rows_sha256': ordered_key_hash(frame.iloc[rows]), 'context_sha256': canonical_hash(self._report),
            'context_columns': 52, 'context_dtype': 'float32', 'construction_chunk_size': chunk_size,
            'context_values_sha256': hashlib.sha256(self._values).hexdigest(),
            'value_bytes': self._values.nbytes, 'row_lookup_bytes': self._positions.nbytes,
            'exact_input_snapshot_bytes': int(self._inputs.memory_usage(index=True, deep=True).sum()),
            'construction_seconds': time.perf_counter()-started,
            'guarded_columns': list(GUARDED_COLUMNS),
            'invariant_overrides': ['pitch_type', 'current normalized physical vector'],
            'policy_synthetic_states': 'unsupported; changed observed-row dependencies rejected',
            'model_training_or_calibration_changed': False, 'adoption': None}

    def transform(self, frame):
        rows = frame.index.to_numpy()
        if (rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer)
            or (rows < 0).any() or (rows >= len(self._positions)).any()):
            raise ValueError('Cache lookup needs canonical observed row indices')
        positions = self._positions[rows]
        if (positions < 0).any():
            raise ValueError('Observed query row was not registered in this cache')
        # Candidate type replacement makes a frame copy but retains its index.
        # Exact dtype/value/index comparison also detects source-frame mutation.
        expected = self._inputs.iloc[positions]
        if not frame.loc[:, GUARDED_COLUMNS].equals(expected):
            raise ValueError('Observed context dependency changed; synthetic policy states cannot use this cache')
        return self._empty.copy() if not len(rows) else self._values[positions]

    def report(self):
        return deepcopy(self._report)

    def cache_report(self):
        return deepcopy(self._diagnostics)
