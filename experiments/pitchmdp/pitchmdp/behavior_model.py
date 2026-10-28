"""Behavior-policy (pi_b_hat) candidates of EXP-BC-CAL-01 behind one ``BehaviorModel`` protocol.

The OPE runtime (``policy_runtime.PolicyRuntime``) reads its logging law through the
``CategoricalBC`` surface: ``actions``, ``support(state)``, ``probabilities(state)`` and
``fallback(state)``. Every candidate here exposes the same surface (B0 is the registered
``CategoricalBC`` itself, wrapped), plus ``batch(table, rows)`` (the same law, vectorized over a
frozen pre-pitch context table) and ``identity()`` (parameters, artifacts and code sha256).

- ``RegisteredBC`` (B0): the BC_P artifact loaded by sha256 (``policy_artifacts.load_train_bc``),
  never refit. ``VectorBC`` is its vectorized recipe (OOF encodings, cross-fit arms, R1b).
- ``BackoffBC`` (B1): the hierarchical back-off chain league -> pitcher -> x batter side ->
  x count -> x previous type (= B0 key) -> x base-out -> x times through order -> x batter
  cluster with p_l = (n_l + alpha_l p_{l-1}) / (n_l + alpha_l), zero outside R_j and renormalized
  inside at every level; alpha = inf skips a level, a level without a TRAIN count for the key keeps
  its parent. B0 is the special case alpha = (0, inf, inf, 20, inf, inf, inf).
- ``LGBMBehavior`` (B2): LightGBM 18-class over the 63 registered pre-pitch features, masked to
  R_j and renormalized (lightgbm is imported lazily).
- ``Calibrated``: a frozen temperature / vector calibrator over any of them.

Every candidate uses the B0 support: the pitcher's TRAIN repertoire R_j (>= minimum_action_count
TRAIN rows); a pitcher without TRAIN rows takes the league TRAIN support and is flagged by
``fallback`` (the OPE runtime refuses such pitchers). Probabilities are 0 outside the support and
sum to 1 inside. Nothing here reads data at import time.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from .behavior_calibration_registry import (ACTION_CODES, B1_LEVELS, B2_CATEGORICAL, B2_FEATURES, alpha_grid,
                                            b0_alphas)
from .policy_artifacts import IntegrityError, TrainBCArtifact, load_train_bc

START, UNKNOWN = '<START>', '<UNKNOWN>'
SIDES = ('L', 'R')
NLL_CLIP = 1e-12  # matrix_metrics.pitch_losses convention
PACKAGE = Path(__file__).resolve().parent
CODE_FILES = ('behavior_model.py', 'behavior_calibration.py', 'behavior_calibration_registry.py',
              'rollout_policy.py', 'policy_artifacts.py')

# Composite integer key layout (bits): pitcher 24 | side 2 | balls 2 | strikes 2 | previous 8 |
# base_out 4 | times-through-order group 2 | batter cluster 3. The previous component is the index
# of the RAW previous label in the count table's own sorted vocabulary; UNSEEN is never stored, so a
# label absent from that vocabulary matches no cell (exactly CategoricalBC's raw-string key).
_LAYOUT = (('pitcher', 24), ('side', 2), ('balls', 2), ('strikes', 2), ('previous', 8), ('base_out', 4),
           ('tto', 2), ('cluster', 3))
PREVIOUS_POSITION = 4
UNSEEN = (1 << 8) - 1
LEVEL_COMPONENTS = {'pitcher': 1, 'batter_side': 2, 'count': 4, 'previous_type': 5, 'base_out': 6,
                    'times_through_order': 7, 'batter_cluster': 8}  # number of leading layout components


def trailing_bits(components):
    """Bits after the first ``components`` layout entries (``composite(c) >> trailing_bits(c)``
    recovers the leading component when c == 1)."""
    return sum(bits for _, bits in _LAYOUT[components:])


def code_sha256(files=CODE_FILES):
    digest = hashlib.sha256()
    for name in files:
        digest.update(name.encode() + b'\0' + (PACKAGE / name).read_bytes() + b'\0')
    return digest.hexdigest()


def side_code(values):
    out = np.array([SIDES.index(v) if v in SIDES else -1 for v in values], dtype=np.int64)
    if (out < 0).any():
        raise IntegrityError('batter side must be L or R')
    return out


def previous_index(labels, vocabulary):
    """Index of each raw previous label in ``vocabulary``; UNSEEN where absent."""
    if not len(vocabulary):
        return np.full(len(labels), UNSEEN, dtype=np.int64)
    index = pd.Index(list(vocabulary), dtype=object).get_indexer(pd.Index(list(labels), dtype=object))
    return np.where(index >= 0, index, UNSEEN).astype(np.int64)


class Keys:
    """Pre-pitch key components of n rows: integers plus the raw previous-pitch label strings."""
    INTEGER = (('pitcher', 24), ('side', 2), ('balls', 2), ('strikes', 2), ('base_out', 4), ('tto', 2), ('cluster', 3))

    def __init__(self, pitcher, side, balls, strikes, previous, base_out=None, tto=None, cluster=None):
        n = len(pitcher)
        zeros = np.zeros(n, dtype=np.int64)
        self.values = {}
        for (name, bits), value in zip(self.INTEGER, (pitcher, side, balls, strikes,
                                                      zeros if base_out is None else base_out,
                                                      zeros if tto is None else tto,
                                                      zeros if cluster is None else cluster)):
            array = np.asarray(value, dtype=np.int64)
            if array.shape != (n,) or (array < 0).any() or (array >= 1 << bits).any():
                raise IntegrityError(f'key component {name} outside its registered range')
            self.values[name] = array
        labels = list(previous)
        if len(labels) != n or not all(isinstance(v, str) and v for v in labels):
            raise IntegrityError('previous-pitch labels must be one nonempty string per row')
        self.previous = np.array(labels, dtype=object)
        self.n = n

    def __len__(self):
        return self.n

    @classmethod
    def from_frame(cls, frame):
        """From columns pitcher, batter_side, balls, strikes, previous (+ base_out, tto_group, batter_cluster)."""
        get = lambda name: frame[name].to_numpy(np.int64) if name in frame else None
        return cls(frame.pitcher.to_numpy(np.int64), side_code(frame.batter_side.tolist()),
                   frame.balls.to_numpy(np.int64), frame.strikes.to_numpy(np.int64),
                   [str(v) for v in frame.previous.tolist()], get('base_out'), get('tto_group'), get('batter_cluster'))

    @classmethod
    def from_state(cls, state):
        previous = state.history[-1].action if state.history else START
        return cls([int(state.pitcher)], side_code([state.batter_side]), [int(state.balls)], [int(state.strikes)],
                   [str(previous)])

    def composite(self, components, vocabulary=None):
        """int64 key over the first ``components`` layout entries (later entries zero); a key that
        includes the previous label needs the table's vocabulary."""
        key = np.zeros(self.n, dtype=np.int64)
        for i, (name, bits) in enumerate(_LAYOUT):
            key = key << bits
            if i >= components:
                continue
            if name == 'previous':
                if vocabulary is None:
                    raise IntegrityError('a previous-type key needs its count-table vocabulary')
                key = key | previous_index(self.previous, vocabulary)
            else:
                key = key | self.values[name]
        return key

    def take(self, rows):
        out = Keys.__new__(Keys)
        out.values = {k: v[rows] for k, v in self.values.items()}
        out.previous = self.previous[rows]
        out.n = len(out.previous)
        return out

    @staticmethod
    def concat(parts):
        out = Keys.__new__(Keys)
        out.values = {name: np.concatenate([p.values[name] for p in parts]) for name, _ in Keys.INTEGER}
        out.previous = np.concatenate([p.previous for p in parts])
        out.n = len(out.previous)
        return out


class CountTable:
    """Exact-match counts for one key level: sorted unique composite keys -> counts [keys, V] (int32).
    A level whose key includes the previous label carries its own sorted label vocabulary."""
    def __init__(self, keys, counts, components, vocabulary=None):
        keys, counts = np.asarray(keys, dtype=np.int64), np.asarray(counts)
        if keys.ndim != 1 or counts.ndim != 2 or len(keys) != len(counts) or (counts < 0).any() \
                or (counts.size and counts.max() >= 1 << 31):
            raise IntegrityError('count table shape')
        if (components > PREVIOUS_POSITION) != (vocabulary is not None):
            raise IntegrityError('exactly the previous-type levels carry a label vocabulary')
        if vocabulary is not None:
            vocabulary = tuple(str(v) for v in vocabulary)
            if list(vocabulary) != sorted(set(vocabulary)) or len(vocabulary) >= UNSEEN:
                raise IntegrityError('label vocabulary must be sorted, unique and below the UNSEEN code')
        order = np.argsort(keys, kind='stable')
        keys, counts = keys[order], counts[order].astype(np.int32)
        if len(keys) > 1 and (np.diff(keys) <= 0).any():
            raise IntegrityError('count table keys must be unique')
        self.keys, self.counts, self.components, self.vocabulary = keys, counts, components, vocabulary

    @classmethod
    def from_rows(cls, keys, actions, width, components):
        actions = np.asarray(actions, dtype=np.int64)
        if (actions < 0).any() or (actions >= width).any():
            raise IntegrityError('action index outside the vocabulary')
        vocabulary = tuple(sorted(set(keys.previous.tolist()))) if components > PREVIOUS_POSITION else None
        unique, inverse = np.unique(keys.composite(components, vocabulary), return_inverse=True)
        counts = np.zeros(len(unique) * width, dtype=np.int64)
        np.add.at(counts, inverse * width + actions, 1)
        return cls(unique, counts.reshape(len(unique), width), components, vocabulary)

    @classmethod
    def from_counts(cls, keys, counts, components):
        """One row of counts per key row (artifact cells); distinct raw labels stay distinct."""
        vocabulary = tuple(sorted(set(keys.previous.tolist()))) if components > PREVIOUS_POSITION else None
        composite = keys.composite(components, vocabulary)
        if len(np.unique(composite)) != len(composite):
            raise IntegrityError('duplicate cell keys')
        return cls(composite, counts, components, vocabulary)

    def lookup(self, keys):
        query = keys.composite(self.components, self.vocabulary)
        position = np.minimum(np.searchsorted(self.keys, query), max(len(self.keys) - 1, 0))
        found = (self.keys[position] == query) if len(self.keys) else np.zeros(len(query), dtype=bool)
        out = np.zeros((len(query), self.counts.shape[1]), dtype=np.int32)
        out[found] = self.counts[position[found]]
        return out, found

    def arrays(self, prefix):
        out = {f'{prefix}_keys': self.keys, f'{prefix}_counts': self.counts}
        if self.vocabulary is not None:
            out[f'{prefix}_vocabulary'] = np.array(self.vocabulary, dtype=str)
        return out

    @classmethod
    def from_arrays(cls, arrays, prefix, components):
        vocabulary = arrays.get(f'{prefix}_vocabulary')
        return cls(arrays[f'{prefix}_keys'], arrays[f'{prefix}_counts'], components,
                   None if vocabulary is None else [str(v) for v in vocabulary.tolist()])


class RepertoireSupport:
    """R_j = actions with >= minimum_action_count TRAIN rows of pitcher j; unknown -> league support."""
    def __init__(self, league, pitchers, counts, minimum_action_count):
        self.league = np.asarray(league, dtype=np.int64)
        pitchers, counts = np.asarray(pitchers, dtype=np.int64), np.asarray(counts, dtype=np.int64)
        order = np.argsort(pitchers, kind='stable')
        self.pitchers, self.pitcher_counts = pitchers[order], counts[order]
        if (len(self.pitchers) > 1 and (np.diff(self.pitchers) <= 0).any()) or self.pitcher_counts.ndim != 2 \
                or self.pitcher_counts.shape != (len(self.pitchers), len(self.league)) or not self.league.sum():
            raise IntegrityError('support tables need unique pitchers and a nonempty league row of one width')
        if type(minimum_action_count) is not int or minimum_action_count < 1:
            raise IntegrityError('minimum_action_count must be a positive integer')
        self.minimum = minimum_action_count

    @classmethod
    def from_rows(cls, pitchers, actions, width, minimum_action_count=1):
        unique, inverse = np.unique(np.asarray(pitchers, dtype=np.int64), return_inverse=True)
        counts = np.zeros(len(unique) * width, dtype=np.int64)
        np.add.at(counts, inverse * width + np.asarray(actions, dtype=np.int64), 1)
        counts = counts.reshape(len(unique), width)
        return cls(counts.sum(axis=0), unique, counts, minimum_action_count)

    def known(self, pitchers):
        pitchers = np.asarray(pitchers, dtype=np.int64)
        position = np.minimum(np.searchsorted(self.pitchers, pitchers), max(len(self.pitchers) - 1, 0))
        return (self.pitchers[position] == pitchers) if len(self.pitchers) else np.zeros(len(pitchers), dtype=bool)

    def counts(self, pitchers):
        pitchers = np.asarray(pitchers, dtype=np.int64)
        known = self.known(pitchers)
        position = np.minimum(np.searchsorted(self.pitchers, pitchers), max(len(self.pitchers) - 1, 0))
        out = np.broadcast_to(self.league, (len(pitchers), len(self.league))).copy()
        out[known] = self.pitcher_counts[position[known]]
        return out, known

    def mask(self, pitchers):
        return self.counts(pitchers)[0] >= self.minimum

    def arrays(self, prefix):
        return {f'{prefix}_league': self.league, f'{prefix}_pitchers': self.pitchers,
                f'{prefix}_pitcher_counts': self.pitcher_counts}

    @classmethod
    def from_arrays(cls, arrays, prefix, minimum_action_count):
        return cls(arrays[f'{prefix}_league'], arrays[f'{prefix}_pitchers'], arrays[f'{prefix}_pitcher_counts'],
                   minimum_action_count)


def _normalized(values, mask, name):
    values = values * mask
    total = values.sum(axis=1, keepdims=True)
    if not (total > 0).all() or not np.isfinite(values).all():
        raise ValueError(f'{name}: no positive mass on the TRAIN support; abstain')
    return values / total


def pitch_key(game_pk, at_bat_number, pitch_number):
    """Composite int64 of the pitch key (game 31 bits | at-bat 16 | pitch 16)."""
    g, a, p = (np.asarray(v, dtype=np.int64) for v in (game_pk, at_bat_number, pitch_number))
    if (g < 0).any() or (g >= 1 << 31).any() or (a < 0).any() or (a >= 1 << 16).any() or (p < 0).any() \
            or (p >= 1 << 16).any():
        raise IntegrityError('pitch key outside its range')
    return (g << 32) | (a << 16) | p


def parse_context_key(context_key):
    parts = str(context_key).split(':')
    if len(parts) != 3:
        raise IntegrityError(f'context key {context_key!r} is not game:at_bat:pitch')
    return pitch_key(*([int(x)] for x in parts))[0]


class BehaviorTable:
    """Frozen pre-pitch context rows keyed by the pitch key (``context_key`` = game:at_bat:pitch).

    A ``PAState`` carries only count and history; game state, batter style and workload come from
    this table (rollout_policy.PAState: context_key = the caller's frozen pre-pitch context)."""
    REQUIRED = ('game_pk', 'at_bat_number', 'pitch_number', 'pitcher', 'batter_side', 'balls', 'strikes', 'previous')

    def __init__(self, frame):
        missing = [c for c in self.REQUIRED if c not in frame.columns]
        if missing:
            raise IntegrityError(f'behavior table lacks {missing}')
        self.frame = frame.reset_index(drop=True)
        self.pitch_keys = pitch_key(self.frame.game_pk, self.frame.at_bat_number, self.frame.pitch_number)
        self._order = np.argsort(self.pitch_keys, kind='stable')
        ordered = self.pitch_keys[self._order]
        if len(ordered) > 1 and (np.diff(ordered) == 0).any():
            raise IntegrityError('duplicate pitch keys in a behavior table')
        self.keys = Keys.from_frame(self.frame)

    def __len__(self):
        return len(self.frame)

    def positions(self, pitch_keys):
        """Table positions of composite pitch keys; -1 where absent."""
        keys = np.asarray(pitch_keys, dtype=np.int64)
        ordered = self.pitch_keys[self._order]
        at = np.minimum(np.searchsorted(ordered, keys), max(len(ordered) - 1, 0))
        found = (ordered[at] == keys) if len(ordered) else np.zeros(len(keys), dtype=bool)
        return np.where(found, self._order[at] if len(ordered) else -1, -1)

    def row_of(self, state):
        """Table position of ``state`` after checking that the state is that frozen row."""
        i = int(self.positions([parse_context_key(state.context_key)])[0])
        if i < 0:
            raise IntegrityError(f'state context {state.context_key!r} is not in the frozen table')
        row = self.frame.iloc[i]
        previous = state.history[-1].action if state.history else START
        if (int(row.pitcher) != int(state.pitcher) or row.batter_side != state.batter_side
                or int(row.balls) != int(state.balls) or int(row.strikes) != int(state.strikes)
                or str(row.previous) != str(previous)):
            raise IntegrityError(f'state {state.context_key} disagrees with its frozen context row')
        return i

    def matrix(self, names, rows, dtype=np.float32):
        missing = [c for c in names if c not in self.frame.columns]
        if missing:
            raise IntegrityError(f'behavior table lacks feature columns {missing}')
        part = self.frame if rows is None else self.frame.iloc[np.asarray(rows, dtype=np.int64)]
        return np.ascontiguousarray(part[list(names)].to_numpy(dtype))


def _rows(table, rows):
    return np.arange(len(table)) if rows is None else np.asarray(rows, dtype=np.int64)


@runtime_checkable
class BehaviorModel(Protocol):
    """Logging-law surface shared with ``CategoricalBC`` plus vectorized ``batch`` and ``identity``."""
    actions: tuple

    def support(self, state) -> np.ndarray: ...

    def probabilities(self, state) -> np.ndarray: ...

    def fallback(self, state) -> bool: ...

    def batch(self, table, rows) -> np.ndarray: ...

    def identity(self) -> dict: ...


class _SupportMixin:
    """Support and fallback from a ``RepertoireSupport`` (shared by every candidate)."""
    def support(self, state):
        return self.repertoire.mask(np.array([int(state.pitcher)]))[0]

    def fallback(self, state):
        return not bool(self.repertoire.known(np.array([int(state.pitcher)]))[0])

    def support_batch(self, table, rows=None):
        return self.repertoire.mask(table.keys.values['pitcher'][_rows(table, rows)])

    def bind(self, table):
        """Frozen context table used by ``probabilities(state)`` (B1/B2 need it; B0 ignores it)."""
        self.table = table
        return self


class _GuardCallback:
    """LightGBM after-iteration callback running the runner's cooperative guard."""
    order, before_iteration = 30, False

    def __init__(self, guard):
        self.guard = guard

    def __call__(self, env):
        self.guard.check('b2-fit')


class VectorBC(_SupportMixin):
    """The B0 recipe (CategoricalBC arithmetic) vectorized over integer keys."""
    name = 'VectorBC'

    def __init__(self, vocabulary, repertoire, cells, prior_strength, minimum_action_count, provenance=None):
        if not math.isfinite(prior_strength) or prior_strength <= 0:
            raise IntegrityError('prior_strength must be positive')
        if repertoire.minimum != minimum_action_count:
            raise IntegrityError('support minimum differs from the BC minimum_action_count')
        self.actions, self.repertoire, self.cells = tuple(vocabulary), repertoire, cells
        self.prior_strength, self.minimum_action_count = float(prior_strength), minimum_action_count
        self.provenance = dict(provenance or {})

    @classmethod
    def fit(cls, keys, actions, vocabulary=ACTION_CODES, prior_strength=20., minimum_action_count=1, provenance=None):
        width = len(vocabulary)
        repertoire = RepertoireSupport.from_rows(keys.values['pitcher'], actions, width, minimum_action_count)
        cells = CountTable.from_rows(keys, actions, width, LEVEL_COMPONENTS['previous_type'])
        return cls(vocabulary, repertoire, cells, prior_strength, minimum_action_count, provenance)

    @classmethod
    def from_artifact(cls, artifact, vocabulary=ACTION_CODES):
        """Count tables of a reloaded TRAIN BC artifact (no refit); raw previous labels kept distinct."""
        bc = artifact.bc if isinstance(artifact, TrainBCArtifact) else artifact
        if tuple(bc.actions) != tuple(vocabulary):
            raise IntegrityError('BC vocabulary differs from the registered 18 codes')
        width = len(vocabulary)
        row = lambda counter: [int(counter.get(a, 0)) for a in vocabulary]
        pitchers = sorted(bc.pitchers, key=int)
        repertoire = RepertoireSupport(row(bc.league), [int(p) for p in pitchers], [row(bc.pitchers[p]) for p in pitchers],
                                       bc.minimum_action_count)
        cell_keys = list(bc.cells)
        keys = Keys([int(p) for p, *_ in cell_keys], side_code([k[3] for k in cell_keys]), [k[1] for k in cell_keys],
                    [k[2] for k in cell_keys], [str(k[4]) for k in cell_keys])
        counts = np.array([row(bc.cells[k]) for k in cell_keys], dtype=np.int64).reshape(len(cell_keys), width)
        provenance = {'artifact_file_sha256': getattr(artifact, 'file_sha256', None),
                      'bc_sha256': getattr(artifact, 'sha256', None)}
        cells = CountTable.from_counts(keys, counts, LEVEL_COMPONENTS['previous_type'])
        return cls(vocabulary, repertoire, cells, bc.prior_strength, bc.minimum_action_count, provenance)

    def batch_keys(self, keys):
        counts, _ = self.repertoire.counts(keys.values['pitcher'])
        mask = counts >= self.minimum_action_count
        prior = counts * mask.astype(np.float64)
        total = prior.sum(axis=1, keepdims=True)
        if not (total > 0).all():
            raise ValueError('no TRAIN-supported actions; abstain')
        prior /= total
        cell, _ = self.cells.lookup(keys)
        p = (cell + self.prior_strength * prior) * mask
        return p / p.sum(axis=1, keepdims=True)

    def batch(self, table, rows=None):
        return self.batch_keys(table.keys.take(_rows(table, rows)))

    def probabilities(self, state):
        return self.batch_keys(Keys.from_state(state))[0]

    def arrays(self):
        return {**self.repertoire.arrays('support'), **self.cells.arrays('cell')}

    @classmethod
    def from_arrays(cls, arrays, prior_strength, minimum_action_count, vocabulary=ACTION_CODES, provenance=None):
        repertoire = RepertoireSupport.from_arrays(arrays, 'support', minimum_action_count)
        cells = CountTable.from_arrays(arrays, 'cell', LEVEL_COMPONENTS['previous_type'])
        return cls(vocabulary, repertoire, cells, prior_strength, minimum_action_count, provenance)

    def identity(self):
        return {'class': f'{__name__}.VectorBC', 'prior_strength': self.prior_strength,
                'minimum_action_count': self.minimum_action_count, 'vocabulary': list(self.actions),
                'tables_sha256': arrays_sha256(self.arrays()), 'provenance': self.provenance,
                'code_sha256': code_sha256()}


class RegisteredBC(_SupportMixin):
    """B0: the registered BC_P CategoricalBC (loaded by sha256, never refit)."""
    name = 'B0'

    def __init__(self, artifact, vocabulary=ACTION_CODES):
        if not isinstance(artifact, TrainBCArtifact):
            raise IntegrityError('B0 must be a reloaded TrainBCArtifact')
        if tuple(artifact.vocabulary) != tuple(vocabulary):
            raise IntegrityError('BC_P vocabulary differs from the registered 18 codes')
        self.artifact, self.bc, self.actions = artifact, artifact.bc, tuple(vocabulary)
        self.vector = VectorBC.from_artifact(artifact, vocabulary)
        self.repertoire = self.vector.repertoire
        self.pitchers = artifact.bc.pitchers

    @classmethod
    def load(cls, path, sha256, vocabulary=ACTION_CODES):
        return cls(load_train_bc(path, sha256), vocabulary)

    def support(self, state):
        return np.asarray(self.bc.support(state), dtype=bool)

    def probabilities(self, state):
        return np.asarray(self.bc.probabilities(state), dtype=np.float64)

    def fallback(self, state):
        return bool(self.bc.fallback(state))

    def batch(self, table, rows=None):
        return self.vector.batch(table, rows)

    def identity(self):
        return {'class': f'{__name__}.RegisteredBC', 'name': 'B0', **self.artifact.identity(),
                'parameters': {'prior_strength': self.bc.prior_strength,
                               'minimum_action_count': self.bc.minimum_action_count},
                'refit': False, 'code_sha256': code_sha256()}


def arrays_sha256(arrays):
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode() + b'\0' + str(array.dtype).encode() + b'\0' + str(array.shape).encode() + b'\0')
        digest.update(array.tobytes())
    return digest.hexdigest()


def mean_nll(p, labels):
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    return float(-np.log(np.clip(p[np.arange(len(y)), y], NLL_CLIP, 1)).mean())


class BackoffBC(_SupportMixin):
    """B1: hierarchical back-off chain over ``B1_LEVELS`` (alpha per level; inf skips a level)."""
    name = 'B1'

    def __init__(self, vocabulary, repertoire, league, tables, alphas, provenance=None):
        self.actions, self.repertoire = tuple(vocabulary), repertoire
        self.league = np.asarray(league, dtype=np.int64)
        if set(tables) != set(B1_LEVELS[1:]) or set(alphas) != set(B1_LEVELS[1:]):
            raise IntegrityError('B1 needs one count table and one alpha per chain level')
        self.tables = tables
        self.alphas = {level: float(alphas[level]) for level in B1_LEVELS[1:]}
        if any(not (a == math.inf or (math.isfinite(a) and a >= 0)) for a in self.alphas.values()):
            raise IntegrityError('B1 alphas must be finite nonnegative or inf')
        if any(a == 0 for level, a in self.alphas.items() if level != 'pitcher'):
            raise IntegrityError('alpha 0 is registered for the pitcher level only')
        self.provenance = dict(provenance or {})
        self.tune_record = None

    @classmethod
    def fit(cls, keys, actions, vocabulary=ACTION_CODES, repertoire=None, alphas=None, provenance=None):
        """Count tables from the fit rows; ``repertoire`` defaults to the support of the same rows."""
        width = len(vocabulary)
        actions = np.asarray(actions, dtype=np.int64)
        if repertoire is None:
            repertoire = RepertoireSupport.from_rows(keys.values['pitcher'], actions, width)
        tables = {level: CountTable.from_rows(keys, actions, width, LEVEL_COMPONENTS[level]) for level in B1_LEVELS[1:]}
        league = np.bincount(actions, minlength=width).astype(np.int64)
        return cls(vocabulary, repertoire, league, tables, b0_alphas() if alphas is None else alphas, provenance)

    def _start(self, keys):
        mask = self.repertoire.mask(keys.values['pitcher'])
        p0 = _normalized(np.broadcast_to(self.league.astype(np.float64), mask.shape), mask, 'B1 league level')
        return mask, p0

    def level_counts(self, keys, mask):
        return [self.tables[level].lookup(keys)[0] * mask for level in B1_LEVELS[1:]]

    @staticmethod
    def chain(p0, mask, counts, alphas):
        p = p0
        for level, n in zip(B1_LEVELS[1:], counts):
            a = alphas[level]
            if a == math.inf:
                continue
            has = n.sum(axis=1) > 0
            q = (n if a == 0 else n + a * p) * mask
            total = q.sum(axis=1, keepdims=True)
            p = np.where(has[:, None], q / np.where(total > 0, total, 1.), p)
        return p

    def batch_keys(self, keys):
        mask, p0 = self._start(keys)
        return self.chain(p0, mask, self.level_counts(keys, mask), self.alphas)

    def batch(self, table, rows=None):
        return self.batch_keys(table.keys.take(_rows(table, rows)))

    def probabilities(self, state):
        table = getattr(self, 'table', None)
        if table is None:
            raise IntegrityError('B1 needs a bound frozen context table for a single state')
        return self.batch(table, [table.row_of(state)])[0]

    def tune(self, keys, labels, sweeps=2, guard=None):
        """Coordinate descent on the earlystop S_in NLL in chain order from the current (B0) setting;
        ``sweeps`` passes; ties (exact float64 equality) -> larger alpha."""
        mask, p0 = self._start(keys)
        counts = self.level_counts(keys, mask)
        current = dict(self.alphas)
        start = mean_nll(self.chain(p0, mask, counts, current), labels)
        history = []
        for sweep in range(sweeps):
            for level in B1_LEVELS[1:]:
                trials = []
                for value in alpha_grid(level):
                    trial = {**current, level: value}
                    trials.append((value, mean_nll(self.chain(p0, mask, counts, trial), labels)))
                    if guard is not None:
                        guard.check('b1-tune')
                best = min(v for _, v in trials)
                chosen = max(a for a, v in trials if v == best)
                history.append({'sweep': sweep + 1, 'level': level,
                                'trials': [[_alpha_token(a), v] for a, v in trials], 'chosen': _alpha_token(chosen)})
                current[level] = chosen
        self.alphas = current
        final = mean_nll(self.chain(p0, mask, counts, current), labels)
        self.tune_record = {'objective': 'earlystop S_in mean NLL', 'rows': int(len(keys)), 'sweeps': sweeps,
                            'start_alphas': {k: _alpha_token(v) for k, v in b0_alphas().items()},
                            'start_nll': start, 'alphas': {k: _alpha_token(v) for k, v in current.items()},
                            'final_nll': final, 'history': history}
        return self.tune_record

    def arrays(self):
        out = {'league': self.league, **self.repertoire.arrays('support')}
        for level, table in self.tables.items():
            out.update(table.arrays(level))
        return out

    @staticmethod
    def array_names():
        names = ['league', 'support_league', 'support_pitchers', 'support_pitcher_counts']
        for level in B1_LEVELS[1:]:
            names += [f'{level}_keys', f'{level}_counts']
            if LEVEL_COMPONENTS[level] > PREVIOUS_POSITION:
                names.append(f'{level}_vocabulary')
        return names

    @classmethod
    def from_arrays(cls, arrays, alphas, minimum_action_count=1, vocabulary=ACTION_CODES, provenance=None):
        repertoire = RepertoireSupport.from_arrays(arrays, 'support', minimum_action_count)
        tables = {level: CountTable.from_arrays(arrays, level, LEVEL_COMPONENTS[level]) for level in B1_LEVELS[1:]}
        return cls(vocabulary, repertoire, arrays['league'], tables, alphas, provenance)

    def identity(self):
        return {'class': f'{__name__}.BackoffBC', 'name': 'B1', 'levels': list(B1_LEVELS),
                'alphas': {k: _alpha_token(v) for k, v in self.alphas.items()},
                'tables_sha256': arrays_sha256(self.arrays()), 'provenance': self.provenance,
                'code_sha256': code_sha256()}


def _alpha_token(value):
    return 'inf' if value == math.inf else (int(value) if float(value).is_integer() else float(value))


def alpha_from_token(token):
    return math.inf if token == 'inf' else float(token)


class LGBMBehavior(_SupportMixin):
    """B2: LightGBM multiclass over the registered 63 features, masked to R_j and renormalized."""
    name = 'B2'

    def __init__(self, model_string, best_round, repertoire, record=None):
        import lightgbm as lgb  # optional dependency (pyproject research-a3)
        self.model_string, self.best_round = model_string, int(best_round)
        self.booster = lgb.Booster(model_str=model_string)
        if self.booster.num_model_per_iteration() != len(ACTION_CODES) or not 1 <= self.best_round <= \
                self.booster.current_iteration():
            raise IntegrityError('B2 booster classes or selected round invalid')
        if list(self.booster.feature_name()) != list(B2_FEATURES):
            raise IntegrityError('B2 booster features differ from the registered 63 features')
        self.actions, self.repertoire = ACTION_CODES, repertoire
        self.record = dict(record or {})

    @staticmethod
    def train(features, labels, early_features, early_labels, early_mask, *, params, rounds, guard=None):
        """All ``rounds`` rounds; the masked earlystop S_in NLL of every round comes from a feval on the
        valid set (LightGBM passes softmax probabilities there). First minimum (ties -> earliest)."""
        import lightgbm as lgb
        params = dict(params)
        train_set = lgb.Dataset(features, label=labels, feature_name=list(B2_FEATURES),
                                categorical_feature=list(B2_CATEGORICAL), free_raw_data=True)
        early_set = lgb.Dataset(early_features, label=early_labels, reference=train_set,
                                feature_name=list(B2_FEATURES), categorical_feature=list(B2_CATEGORICAL))
        curve = []
        y_early = np.asarray(early_labels, dtype=np.int64)
        mask = np.asarray(early_mask, dtype=bool)

        def masked_nll(preds, data):
            p = np.asarray(preds, dtype=np.float64).reshape(len(y_early), -1)
            curve.append(mean_nll(_normalized(p, mask, 'B2 earlystop'), y_early))
            return 'masked_s_in_nll', curve[-1], False

        callbacks = [] if guard is None else [_GuardCallback(guard)]
        booster = lgb.train(params, train_set, num_boost_round=int(rounds), valid_sets=[early_set],
                            valid_names=['earlystop_s_in'], feval=masked_nll, callbacks=callbacks)
        if len(curve) != rounds:
            raise IntegrityError('B2 earlystop curve does not cover every round')
        best = int(np.argmin(curve)) + 1
        return booster.model_to_string(), best, {'rounds': int(rounds), 'curve': curve, 'best_round': best,
                                                 'best_nll': curve[best - 1], 'cap_hit': best == rounds,
                                                 'lightgbm': lgb.__version__, 'params': params}

    @staticmethod
    def train_only(features, labels, *, params, rounds, guard=None):
        """Smoke cost/determinism probe: the registered params for ``rounds`` rounds, no valid set."""
        import lightgbm as lgb
        train_set = lgb.Dataset(features, label=labels, feature_name=list(B2_FEATURES),
                                categorical_feature=list(B2_CATEGORICAL), free_raw_data=True)
        callbacks = [] if guard is None else [_GuardCallback(guard)]
        return lgb.train(dict(params), train_set, num_boost_round=int(rounds), callbacks=callbacks).model_to_string()

    def raw(self, features):
        p = np.asarray(self.booster.predict(np.asarray(features, dtype=np.float32), num_iteration=self.best_round),
                       dtype=np.float64)
        if p.shape != (len(features), len(ACTION_CODES)) or not np.isfinite(p).all():
            raise IntegrityError('B2 prediction shape or values invalid')
        return p

    def batch(self, table, rows=None):
        rows = _rows(table, rows)
        return _normalized(self.raw(table.matrix(B2_FEATURES, rows)), self.support_batch(table, rows), 'B2')

    def probabilities(self, state):
        table = getattr(self, 'table', None)
        if table is None:
            raise IntegrityError('B2 needs a bound frozen context table for a single state')
        return self.batch(table, [table.row_of(state)])[0]

    def identity(self):
        return {'class': f'{__name__}.LGBMBehavior', 'name': 'B2', 'best_round': self.best_round,
                'model_sha256': hashlib.sha256(self.model_string.encode()).hexdigest(),
                'support_sha256': arrays_sha256(self.repertoire.arrays('support')),
                'features': list(B2_FEATURES), 'record': {k: v for k, v in self.record.items() if k != 'curve'},
                'code_sha256': code_sha256()}


class Calibrated:
    """A frozen calibrator over a candidate (same support; only probabilities inside it change)."""
    def __init__(self, model, calibrator, name):
        self.model, self.calibrator, self.name = model, calibrator, name
        self.actions, self.repertoire = model.actions, model.repertoire

    def support(self, state):
        return self.model.support(state)

    def fallback(self, state):
        return self.model.fallback(state)

    def support_batch(self, table, rows=None):
        return self.model.support_batch(table, rows)

    def probabilities(self, state):
        return self.calibrator.apply(self.model.probabilities(state)[None, :], self.model.support(state)[None, :])[0]

    def batch(self, table, rows=None):
        rows = _rows(table, rows)
        return self.calibrator.apply(self.model.batch(table, rows), self.model.support_batch(table, rows))

    def bind(self, table):
        self.model.bind(table)
        return self

    def identity(self):
        return {'class': f'{__name__}.Calibrated', 'name': self.name, 'model': self.model.identity(),
                'calibrator': self.calibrator.identity()}


def to_frame_keys(pitcher, side, balls, strikes, previous):
    """Convenience: a key frame for ``Keys.from_frame`` from parallel lists (tests, ledgers)."""
    return pd.DataFrame({'pitcher': np.asarray(pitcher, dtype=np.int64), 'batter_side': list(side),
                         'balls': np.asarray(balls, dtype=np.int64), 'strikes': np.asarray(strikes, dtype=np.int64),
                         'previous': list(previous)})
