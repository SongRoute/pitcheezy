"""T2 exclusion masks and two-view feature reconstruction, with no model fitting.

Statistics consume sanitized primitive observations. H5 tokens and ground truth
retain original observations, guarded by an explicit per-query visibility mask.
Never use the sanitized statistics frame to derive evaluation labels/eligibility.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
import pandas as pd

from .archetypes import HISTORY_COLUMNS, add_batter_style_history
from .data import KEY, PA_KEY, add_history
from .matrix_data import ordered_key_hash
from .matrix_features import MatrixHistoryStore
from .matrix_panel import SELECTOR_VERSION
from .model import eligible, outcome_labels


SEED = 20260924
FIT_SPLITS = ('train', 'earlystop', 'temperature', 'blend')
REGIMES = ('Z', 'W', 'O')
LEGACY_PRIORS = ('batter_pa_prior', 'batter_obp_prior', 'batter_k_prior')
UNSAFE_PREVIOUS = ('prev_pitch_type', 'prev_description', 'prev_plate_x', 'prev_plate_z')
DEV_START = pd.Timestamp('2025-07-01')
DEV_END = pd.Timestamp('2025-09-30')


def _hash(axis, identity):
    return hashlib.sha256(f'ml_t2_{axis}_v1|{SEED}|{int(identity)}'.encode('ascii')).hexdigest()


def _identities(values):
    raw = np.asarray(list(values))
    if pd.isna(raw).any():
        raise ValueError('Missing entity identity')
    ids = raw.astype(np.int64)
    if not np.array_equal(raw, ids) or (ids < 0).any():
        raise ValueError('Nonnegative integral entity identities required')
    return ids


def select_heldout_pitchers(panel):
    if panel.get('version') != SELECTOR_VERSION:
        raise ValueError('Expected frozen Cpanel manifest')
    panel_ids = _identities(panel['pitcher_ids']).tolist()
    if len(set(panel_ids)) != len(panel_ids):
        raise ValueError('Repeated panel pitcher')
    records, selected, covered = [], [], []
    for stratum in panel['strata']:
        ids = _identities(stratum['pitcher_ids']).tolist()
        if not set(ids) <= set(panel_ids):
            raise ValueError('Stratum includes an unselected panel pitcher')
        ordered = sorted(ids, key=lambda pid: (_hash('pitcher', pid), pid))
        chosen = ordered[:1]
        covered.extend(ids)
        selected.extend(chosen)
        records.append({name: stratum[name] for name in ('role', 'hand', 'volume')} |
                       {'available_panel_ids': ids, 'selected_ids': chosen,
                        'status': 'selected' if chosen else 'empty'})
    if sorted(covered) != sorted(panel_ids):
        raise ValueError('Panel strata must partition selected pitchers exactly')
    return {'axis': 'pitcher', 'seed': SEED, 'ids': sorted(selected), 'strata': records,
            'hash_namespace': 'ml_t2_pitcher_v1', 'rule': 'first SHA256 rank per nonempty panel stratum',
            'replacement': 'none; no DEV coverage or labels consumed'}


def select_heldout_batters(train):
    if train.empty or not train.split.eq('train').all():
        raise ValueError('Nonempty actual TRAIN pool required')
    ids = sorted(set(_identities(train.batter)))
    selected = [int(pid) for pid in ids if int(_hash('batter', pid), 16) % 5 == 0]
    return {'axis': 'batter', 'seed': SEED, 'ids': selected, 'train_batters': len(ids),
            'train_keys_sha256': ordered_key_hash(train),
            'train_counts': {str(int(pid)): int(n) for pid, n in train.groupby('batter').size().items()},
            'hash_namespace': 'ml_t2_batter_v1', 'rule': 'full SHA256 integer modulo five equals zero',
            'replacement': 'none; no DEV coverage or labels consumed'}


def _metadata(frame):
    needed = {*KEY, 'game_date', 'split', 'pitcher', 'batter'}
    if not needed <= set(frame.columns):
        raise ValueError('Missing dated pitch/entity metadata')
    ordered_key_hash(frame)
    raw = frame[KEY].to_numpy()
    if not np.array_equal(raw, raw.astype(np.int64)):
        raise ValueError('Integral pitch keys required')
    _identities(frame.pitcher)
    _identities(frame.batter)
    dates = pd.to_datetime(frame.game_date).dt.normalize()
    if dates.isna().any() or not dates.dt.year.isin([2023, 2024, 2025]).all():
        raise ValueError('Only approved 2023–2025 history is allowed')
    if frame.assign(_date=dates).groupby('game_pk')._date.nunique().gt(1).any():
        raise ValueError('A game must have one observation date')
    return dates


@dataclass
class ReconstructedFold:
    statistics_frame: pd.DataFrame
    feature_frame: pd.DataFrame
    fold: 'GeneralizationFold'
    regime: str

    def history_store(self, *, normalizer, type_vocabulary):
        """Only this guarded store is valid for model access to token primitives.

        The caller must freshly fit the supplied normalizer/vocabulary using the
        fold's refit plan; this API deliberately has no implicit fit or aux reuse.
        """
        training = self.fold.frame.loc[self.fold.allowed_fit['train']]
        expected_types = tuple(sorted(training.pitch_type.dropna().astype(str).unique()))
        expected_date = str(self.fold.dates[self.fold.allowed_fit['train']].max().date())
        if (getattr(normalizer, 'n_train', None) != len(training)
                or getattr(normalizer, 'fit_date_max', None) != expected_date
                or tuple(type_vocabulary) != expected_types):
            raise ValueError('Fresh allowed-TRAIN normalizer and exact TRAIN type vocabulary required')
        # Counts/cutoff/vocabulary catch ordinary parent-aux reuse. The runner
        # must additionally pin the actual fitting-key/source hashes.
        base = MatrixHistoryStore.from_frame(self.feature_frame, normalizer,
            history_length=5, type_vocabulary=type_vocabulary)
        return GeneralizationHistoryStore(base, self.fold, self.regime)


class GeneralizationFold:
    def __init__(self, frame, axis, heldout_ids):
        if axis not in ('pitcher', 'batter'):
            raise ValueError('Use separate pitcher and batter exclusion folds')
        dates = _metadata(frame)
        needed = {'description', 'events', 'is_pa_terminal', 'launch_angle', 'pitch_type',
                  'plate_x', 'plate_z', 'balls', 'strikes', 'supported_pa'}
        if not needed <= set(frame.columns):
            raise ValueError('Primitive history and original eligibility columns required')
        if not dates.is_monotonic_increasing:
            raise ValueError('Chronological source order is required; row positions are preserved')
        if not frame.split.isin([*FIT_SPLITS, 'dev', 'history', 'unused']).all():
            raise ValueError('Use explicitly separated early-stop/temperature/blend splits')
        periods = {'train': ('2023-05-15', '2025-04-30'),
                   'earlystop': ('2025-05-01', '2025-05-15'),
                   'temperature': ('2025-05-16', '2025-05-31'),
                   'blend': ('2025-06-01', '2025-06-30')}
        for split, (start, end) in periods.items():
            if not dates[frame.split.eq(split)].between(start, end).all():
                raise ValueError('Fitting split dates differ from the 2025 temporal fold')
        if not dates[frame.split.isin(['history', 'unused'])].lt('2023-05-15').all():
            raise ValueError('History-only rows must precede TRAIN')
        if not dates[frame.split.eq('dev')].between(DEV_START, DEV_END).all():
            raise ValueError('The first T2 fold fixes July–September 2025 DEV')
        ids = _identities(heldout_ids)
        if not len(ids) or len(set(ids)) != len(ids):
            raise ValueError('Nonempty unique held-out identities required')
        if not set(ids) <= set(frame.loc[frame.split.eq('train'), axis]):
            raise ValueError('Exclusion identities must be selected from TRAIN')
        self.frame, self.axis, self.ids, self.dates = frame, axis, tuple(sorted(ids.tolist())), dates
        self.heldout = frame[axis].isin(ids).to_numpy()
        self.keys = frame[KEY].to_numpy(np.int64)
        pa = pd.MultiIndex.from_frame(frame[PA_KEY])
        self.affected_pa = pa.isin(pa[self.heldout].unique())
        self.allowed_fit = {split: frame.split.eq(split).to_numpy() & ~self.affected_pa for split in FIT_SPLITS}
        # Truth is captured before any observation is blanked. Retrospective
        # eligibility gates targets only, never the availability of past tokens.
        self.truth_labels = outcome_labels(frame)
        self.truth_eligible = np.asarray(eligible(frame), dtype=bool)
        self.prefix = np.zeros(len(frame), dtype=bool)
        self.target_eval = np.zeros(len(frame), dtype=bool)
        self.prefix_records = []
        dev = frame.split.eq('dev').to_numpy()
        for identity in self.ids:
            member = frame[axis].eq(identity).to_numpy()
            appearances = frame.loc[member & dev, ['game_pk']].assign(
                game_date=dates[member & dev].to_numpy()).drop_duplicates().sort_values(['game_date', 'game_pk'])
            first = appearances.head(2)
            self.prefix |= member & dev & frame.game_pk.isin(first.game_pk).to_numpy()
            cutoff = first.game_date.iloc[-1] if len(first) == 2 else None
            evaluation = member & dev & (dates.gt(cutoff).to_numpy() if cutoff is not None else False)
            self.target_eval |= evaluation
            self.prefix_records.append({'id': identity, 'games': first.game_pk.astype(int).tolist(),
                'prefix_pitches': int((member & self.prefix).sum()),
                'cutoff_date': str(cutoff.date()) if cutoff is not None else None,
                'requested_evaluation_pitches': int(evaluation.sum()),
                'status': 'paired_evaluation' if evaluation.any() else 'unmeasured_no_post_prefix_rows'})
        self.observation_masks = {'Z': ~self.heldout, 'W': ~self.heldout,
                                  'O': ~self.heldout | self.prefix}
        self.query_allowed = self.target_eval.copy()
        for mask in self.allowed_fit.values():
            self.query_allowed |= mask
        self.conditional_allowed = self.allowed_fit['train'] | self.allowed_fit['earlystop']
        for array in [self.heldout, self.prefix, self.target_eval, self.truth_labels, self.truth_eligible,
                      self.query_allowed, self.conditional_allowed, *self.allowed_fit.values(),
                      *self.observation_masks.values()]:
            array.setflags(write=False)

    def observation_mask(self, regime):
        if regime not in REGIMES:
            raise ValueError('Expected Z, W or O observation regime')
        return self.observation_masks[regime]

    def evaluation_mask(self, regime, *, eligible_only=False):
        self.observation_mask(regime)  # validate; all regimes use the same target rows
        return self.target_eval & self.truth_eligible if eligible_only else self.target_eval.copy()

    def reconstruct(self, regime):
        allowed = self.observation_mask(regime)
        statistics = self.frame.copy()
        statistics['description'] = statistics.description.astype(object)
        statistics['events'] = statistics.events.astype(object)
        statistics.loc[~allowed, 'description'] = ''
        statistics.loc[~allowed, 'events'] = ''
        statistics.loc[~allowed, 'is_pa_terminal'] = False
        statistics.loc[~allowed, 'launch_angle'] = np.nan
        # Both functions overwrite every cumulative feature they expose. Mask
        # entire primitive observations BEFORE player and league aggregation.
        add_history(statistics)
        add_batter_style_history(statistics)
        features = self.frame.copy()
        for name in (*LEGACY_PRIORS, *HISTORY_COLUMNS):
            features[name] = statistics[name].to_numpy()
        # These old single-pitch columns have no visibility guard and are outside
        # the G input contract. Drop rather than silently offer unsafe features.
        features.drop(columns=list(UNSAFE_PREVIOUS), errors='ignore', inplace=True)
        features.attrs['t2_feature_contract'] = {
            'axis': self.axis, 'heldout_ids': list(self.ids), 'regime': regime,
            'token_access': 'GeneralizationHistoryStore mandatory; direct MatrixHistoryStore is invalid',
            'truth': 'original fold.truth_labels/truth_eligible only; never statistics_frame',
            'legacy_previous_columns': 'forbidden and removed', 'aux_reuse': 'forbidden'}
        return ReconstructedFold(statistics, features, self, regime)

    def h5_visibility(self, rows, history_indices, regime):
        source_allowed = self.observation_mask(regime)
        rows, indices = np.asarray(rows, dtype=np.int64), np.asarray(history_indices, dtype=np.int64)
        if rows.ndim != 1 or indices.shape != (len(rows), 5) or (rows < 0).any() or (rows >= len(self.frame)).any():
            raise ValueError('Aligned H5 row positions required')
        if (indices < -1).any() or (indices >= rows[:, None]).any():
            raise ValueError('History must refer strictly to previous source rows')
        present = indices >= 0
        safe = np.maximum(indices, 0)
        if np.any(present & np.any(self.keys[safe, :2] != self.keys[rows, None, :2], axis=-1)):
            raise ValueError('H5 cannot cross PA boundaries')
        visible = present & source_allowed[safe]
        if regime == 'Z':
            visible[self.heldout[rows]] = False
        else:
            query = self.target_eval[rows]
            visible[query] = present[query]  # observed current-PA tokens, never cumulative evidence
        return visible

    def auxiliary_fit_plan(self):
        """Fresh fitting inputs, intentionally not fitted auxiliary objects."""
        features = self.reconstruct('Z').feature_frame
        parts = {split: features.loc[mask & self.truth_eligible].copy()
                 for split, mask in self.allowed_fit.items()}
        normalizer_train = features.loc[self.allowed_fit['train']].copy()
        if not len(normalizer_train) or not len(parts['train']):
            raise ValueError('Exclusion leaves no allowed TRAIN rows')
        return {'normalizer_train': normalizer_train, 'type_vocabulary_train': normalizer_train,
            'supervised_train': parts['train'], 'fit_partitions': parts,
            'report': {'version': 't2_sanitized_aux_refit_v1', 'axis': self.axis, 'heldout_ids': list(self.ids),
                'aux_reuse': False,
                'normalizer_type_keys_sha256': ordered_key_hash(normalizer_train),
                'partition_keys_sha256': {name: ordered_key_hash(part) for name, part in parts.items()},
                'refit_on_supervised_train': ['frequency model', 'batter archetype encoder',
                    'pitcher continuous profiles/scaler/centroids', '400-draw delivery pools and league fallback'],
                'refit_on_all_allowed_train': ['physical normalizer', 'pitch-type vocabulary'],
                'calibration': 'allowed temperature/blend rows only; candidate/control share fold',
                'prefix': self.prefix_records,
                'history_scope': 'all source rows including pre-TRAIN history; no inherited cumulative features',
                'dropped_previous_columns': list(UNSAFE_PREVIOUS)}}


class GeneralizationHistoryStore:
    """Guard original token payload with T2 availability before model access."""
    def __init__(self, base, fold, regime):
        if not np.array_equal(base.frame[KEY].to_numpy(np.int64), fold.keys):
            raise ValueError('History source positions differ from fold')
        fold.observation_mask(regime)
        self.base, self.fold, self.regime = base, fold, regime
        self.frame, self.indices, self.normalizer = base.frame, base.indices, base.normalizer
        self.type_vocabulary, self.n_types = base.type_vocabulary, base.n_types

    def gather(self, rows, current=None, candidate_pitch_types=None):
        rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        if (rows < 0).any() or (rows >= len(self.frame)).any():
            raise IndexError('Query row outside source')
        if not self.fold.query_allowed[rows].all():
            raise ValueError('Query is neither an allowed fit/CAL row nor a paired evaluation row')
        if current is None and not self.fold.conditional_allowed[rows].all():
            raise ValueError('CAL/DEV queries require supplied current delivery physics')
        tokens, valid = self.base.gather(rows, current, candidate_pitch_types)
        valid[:, :-1] &= self.fold.h5_visibility(rows, self.indices[rows], self.regime)
        tokens[~valid] = 0.
        return tokens, valid

    def report(self):
        return {'base': self.base.report(), 'version': 't2_guarded_h5_v1', 'axis': self.fold.axis,
                'heldout_ids': list(self.fold.ids), 'regime': self.regime,
                'current_outcome': 'always zero', 'source_positions': 'unchanged',
                'truth': 'original source; statistics-frame labels are forbidden'}


def audit_new_matchups(frame, *, history_sources, fitting_train):
    """Audit all supplied historical sources, including dates before TRAIN.

    Returns an all-row mask for the first DEV game of each previously absent
    pair. Caller must declare the complete source manifest; this helper cannot
    prove that an undisclosed external history source does not exist.
    """
    if (not history_sources or {'query_pool', 'actual_train'} & set(history_sources)
            or any(not isinstance(name, str) or not name for name in history_sources)):
        raise ValueError('Named complete history sources required; query_pool and actual_train are reserved')
    if fitting_train.empty or not fitting_train.split.eq('train').all():
        raise ValueError('Actual TRAIN fitting identities are required')
    train_dates = _metadata(fitting_train)
    if not train_dates.between('2023-05-15', '2025-04-30').all():
        raise ValueError('Actual fitting TRAIN must precede early-stop and CAL')
    dates = _metadata(frame)
    sources = {'query_pool': frame, 'actual_train': fitting_train, **history_sources}
    prior, dev, manifests = [], [], {}
    for name, source in sources.items():
        source_dates = _metadata(source)
        prior.append(source.loc[source_dates.lt(DEV_START), ['pitcher', 'batter']])
        observations = source.loc[source_dates.between(DEV_START, DEV_END), ['pitcher', 'batter', 'game_pk']].copy()
        observations['game_date'] = source_dates[source_dates.between(DEV_START, DEV_END)].to_numpy()
        dev.append(observations)
        manifests[name] = {'rows': len(source), 'keys_sha256': ordered_key_hash(source),
                           'pre_dev_rows': int(source_dates.lt(DEV_START).sum())}
    seen = pd.MultiIndex.from_frame(pd.concat(prior, ignore_index=True).drop_duplicates())
    pairs = pd.MultiIndex.from_frame(frame[['pitcher', 'batter']])
    first = pd.concat(dev, ignore_index=True).drop_duplicates().sort_values(['game_date', 'game_pk']).drop_duplicates(['pitcher', 'batter'])
    first_game = first.set_index(['pitcher', 'batter']).game_pk.reindex(pairs).to_numpy()
    known = frame.pitcher.isin(fitting_train.pitcher).to_numpy() & frame.batter.isin(fitting_train.batter).to_numpy()
    mask = (frame.split.eq('dev').to_numpy() & dates.between(DEV_START, DEV_END).to_numpy()
            & known & ~pairs.isin(seen) & frame.game_pk.eq(first_game).to_numpy())
    return {'mask': mask, 'report': {'version': 't2_new_matchup_audit_v1', 'sources': manifests,
        'requested_pitches': int(mask.sum()), 'requested_pairs': len(pairs[mask].unique()),
        'rule': 'both IDs in actual TRAIN; pair absent from every supplied pre-DEV source; first observed DEV game only',
        'eligibility': 'apply original truth eligibility separately; no labels consumed by this audit',
        'source_completeness': 'caller must pin all feature-history and fitting sources'}}
