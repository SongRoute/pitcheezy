"""Stage-specific acquisition of the EXP-BC-CAL-01 rows (no shared full-file loader is used).

The processed columns are split into FREE (keys, date, game type, pre-pitch fields) and GUARDED
(pitch_type and the outcome columns description, events, is_pa_terminal, launch_angle). Every decode
goes through ``ProcessedReader`` / ``RawKeysReader`` and is written to the ``ReadLog`` with its
phase, file, row group, columns and the exact row range decoded:

- smoke: FREE + GUARDED columns of row groups whose game_date statistics end on or before the TRAIN
  end (2025-04-30) only, so no row of a later split is decoded; raw keys likewise;
- census: FREE columns only (no GUARDED column is decoded);
- pre-gate (fit, calibrate, freeze-primary): FREE columns of every row group; GUARDED columns of each
  row group only up to its last row dated <= 2025-05-31 (pyarrow ``iter_batches`` with batch_size =
  that prefix; reading stops after the first batch). Row groups must be date-sorted, and a DEV row
  inside a straddling row group must lie at least ``margin`` rows after the prefix;
- after the gate (freeze-primary pinned AND s6-integrity passed): GUARDED columns of the row groups
  holding DEV rows.

Rows dated before 2023-05-15 that share a row group with TRAIN rows are decoded and dropped (logged).
File bytes are sha256-hashed before the first decode and after the last (authorized raw-byte
hashing); a change stops the run. A 2026 row or a date after 2025-09-30 stops the run.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from . import behavior_calibration_registry as reg
from .behavior_features import SEALED, label_free_view
from .policy_artifacts import IntegrityError

FREE = list(reg.FREE_COLUMNS)
GUARDED = list(reg.GUARDED_COLUMNS)
RAW_KEY_COLUMNS = ['game_date', 'game_pk', 'at_bat_number', 'pitch_number', 'pitcher']
TRAIN_END = pd.Timestamp(reg.SPLIT_DATES['train'][1])
PRE_GATE_LIMIT = pd.Timestamp(reg.READ_POLICY['pre_gate_limit'])
DEV_START = pd.Timestamp(reg.SPLIT_DATES['dev'][0])
LAST_DATE = pd.Timestamp(reg.LAST_DATE)
STRING_GUARDED = ('pitch_type', 'description', 'events')


class ReadRefused(IntegrityError):
    """The requested read would decode rows or columns this phase may not read (fail closed)."""


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _timestamp(value):
    if value is None:
        return None
    stamp = pd.Timestamp(value)
    return (stamp.tz_localize(None) if stamp.tzinfo is not None else stamp).normalize()


def split_of(dates):
    """Registered split of each date (run_temporal_blend.assign_fold(frame, 2025) windows); 'unused'
    for 2023-03-30..05-14."""
    dates = pd.to_datetime(pd.Series(dates)).dt.normalize()
    out = np.full(len(dates), 'unused', dtype=object)
    for split, (start, end) in reg.SPLIT_DATES.items():
        out[dates.between(pd.Timestamp(start), pd.Timestamp(end)).to_numpy()] = split
    return out


def check_dates(dates, where):
    dates = pd.to_datetime(pd.Series(dates)).dt.normalize()
    if dates.isna().any() or dates.dt.year.ge(2026).any() or (dates > LAST_DATE).any():
        raise ReadRefused(f'{where}: a 2026 row or a date after 2025-09-30 was found; stop')


class ReadLog:
    """Every decode of the stage-specific readers (the 'actual reads' of the stage records)."""
    def __init__(self):
        self.entries = []

    def add(self, **entry):
        self.entries.append(entry)

    def for_phase(self, phase):
        return [e for e in self.entries if e['phase'] == phase]

    def rows_at_or_after(self, since, *, guarded_only, phases=None):
        """Rows decoded with a date >= ``since`` ('2025-07-01' or '2025-05-01'), optionally only for
        guarded reads and only in ``phases``."""
        return int(sum(e['rows_at_or_after'][since] for e in self.entries
                       if (e['guarded'] or not guarded_only) and (phases is None or e['phase'] in phases)))


@dataclass
class Group:
    index: int
    offset: int
    rows: int
    low: object
    high: object


def _groups(parquet, path, column='game_date'):
    import pyarrow.parquet  # noqa: F401  (pyarrow is a project dependency)
    meta = parquet.metadata
    if meta.num_row_groups == 0:
        raise ReadRefused(f'{path.name}: no row groups')
    names = [meta.row_group(0).column(j).path_in_schema for j in range(meta.num_columns)]
    if column not in names:
        raise ReadRefused(f'{path.name}: no {column} column')
    index, groups, offset = names.index(column), [], 0
    for i in range(meta.num_row_groups):
        group = meta.row_group(i)
        stats = group.column(index).statistics
        low, high = (_timestamp(stats.min), _timestamp(stats.max)) if stats is not None and stats.has_min_max \
            else (None, None)
        groups.append(Group(i, offset, group.num_rows, low, high))
        offset += group.num_rows
    return groups


def _log_entry(log, *, phase, path, group, columns, start, stop, dates, guarded):
    dates = pd.to_datetime(pd.Series(dates)).dt.normalize()
    log.add(phase=phase, file=path.name, row_group=group.index, columns=list(columns), row_start=int(start),
            row_stop=int(stop), rows=int(stop - start), guarded=bool(guarded),
            date_min=None if not len(dates) else str(dates.min().date()),
            date_max=None if not len(dates) else str(dates.max().date()),
            rows_at_or_after={str(DEV_START.date()): int((dates >= DEV_START).sum()),
                              str((TRAIN_END + pd.Timedelta(days=1)).date()): int((dates > TRAIN_END).sum())})


class ProcessedReader:
    """Phase-limited reads of the pinned processed parquet (see the module docstring)."""
    def __init__(self, path, *, sha256, log, margin):
        import pyarrow.parquet as pq
        self.path, self.sha256, self.log, self.margin = Path(path), sha256, log, int(margin)
        if file_sha256(self.path) != sha256:
            raise ReadRefused('processed parquet differs from its registered sha256')
        self.file = pq.ParquetFile(self.path)
        missing = set(FREE + GUARDED) - set(self.file.schema_arrow.names)
        if missing:
            raise ReadRefused(f'processed parquet lacks {sorted(missing)}')
        self.groups = _groups(self.file, self.path)
        for group in self.groups:
            if group.high is not None and (group.high > LAST_DATE or group.high.year >= 2026):
                raise ReadRefused('processed row-group statistics show a 2026 row or a date after 2025-09-30; stop')
        self.rows = sum(g.rows for g in self.groups)
        self.dates = {}

    def verify_unchanged(self):
        if file_sha256(self.path) != self.sha256:
            raise IntegrityError('processed parquet changed while the run read it')

    def groups_ending_by(self, limit):
        if any(g.high is None for g in self.groups):
            raise ReadRefused('row-group date statistics are required to restrict a read by date')
        return [g for g in self.groups if g.high <= limit]

    def free(self, phase, groups=None):
        """FREE columns of ``groups`` (default: all), with a global ``row_id``."""
        parts = []
        for group in self.groups if groups is None else groups:
            frame = self.file.read_row_group(group.index, columns=FREE).to_pandas()
            dates = pd.to_datetime(frame.game_date).dt.normalize()
            check_dates(dates, f'{self.path.name} row group {group.index}')
            frame['game_date'] = dates.to_numpy()
            frame['row_id'] = np.arange(group.offset, group.offset + group.rows)
            self.dates[group.index] = dates.to_numpy()
            _log_entry(self.log, phase=phase, path=self.path, group=group, columns=FREE, start=0, stop=group.rows,
                       dates=dates, guarded=False)
            parts.append(frame)
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=[*FREE, 'row_id'])

    def guarded_prefix(self, phase, limit, groups=None):
        """GUARDED columns of every row dated <= ``limit``; each row group is decoded only up to its
        last such row (no later row's guarded value is decoded)."""
        parts = []
        for group in self.groups if groups is None else groups:
            if group.index not in self.dates:
                raise ReadRefused('guarded reads need the row group dates first (free read)')
            dates = self.dates[group.index]
            if len(dates) > 1 and (np.diff(dates.astype('datetime64[ns]').astype(np.int64)) < 0).any():
                raise ReadRefused(f'row group {group.index} is not date-sorted; a prefix read cannot be bounded')
            prefix = int((dates <= np.datetime64(limit)).sum())
            if prefix == 0:
                continue
            columns = ['game_date', *GUARDED]
            if prefix == group.rows:
                table = self.file.read_row_group(group.index, columns=columns)
            else:
                dev = np.flatnonzero(dates >= np.datetime64(DEV_START))
                if len(dev) and dev[0] - prefix < self.margin:
                    raise ReadRefused(f'row group {group.index}: a DEV row lies within {self.margin} rows of the '
                                      f'{limit.date()} prefix')
                import pyarrow as pa
                batches = self.file.iter_batches(batch_size=prefix, row_groups=[group.index], columns=columns)
                batch = next(batches)
                del batches  # stop: no further batch of this row group is decoded
                table = pa.Table.from_batches([batch])
                if table.num_rows != prefix:
                    raise ReadRefused(f'row group {group.index}: prefix read returned {table.num_rows} rows')
            frame = table.to_pandas()
            decoded = pd.to_datetime(frame.game_date).dt.normalize()
            if not np.array_equal(decoded.to_numpy(), dates[:prefix]):
                raise ReadRefused(f'row group {group.index}: guarded rows are not the dated prefix')
            frame['row_id'] = np.arange(group.offset, group.offset + prefix)
            _log_entry(self.log, phase=phase, path=self.path, group=group, columns=columns, start=0, stop=prefix,
                       dates=decoded, guarded=True)
            parts.append(frame.drop(columns='game_date'))
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=[*GUARDED, 'row_id'])

    def guarded_after_gate(self, phase, gate):
        """GUARDED columns of the row groups holding DEV rows; only after freeze-primary is pinned and
        s6-integrity passed."""
        gate.require_pinned('freeze-primary', f'{phase}: DEV guarded columns')
        gate.require_passed('s6-integrity', f'{phase}: DEV guarded columns')
        parts = []
        for group in self.groups:
            dates = self.dates.get(group.index)
            if dates is None:
                raise ReadRefused('DEV guarded reads need the row group dates first (free read)')
            if not (dates >= np.datetime64(DEV_START)).any():
                continue
            columns = ['game_date', *GUARDED]
            frame = self.file.read_row_group(group.index, columns=columns).to_pandas()
            decoded = pd.to_datetime(frame.game_date).dt.normalize()
            frame['row_id'] = np.arange(group.offset, group.offset + group.rows)
            _log_entry(self.log, phase=phase, path=self.path, group=group, columns=columns, start=0, stop=group.rows,
                       dates=decoded, guarded=True)
            parts.append(frame.drop(columns='game_date'))
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=[*GUARDED, 'row_id'])


class RawKeysReader:
    """Keys/date/pitcher of the pinned raw Statcast files (workload source; label-free columns only)."""
    def __init__(self, root, pins, log):
        self.root, self.pins, self.log = Path(root), dict(pins), log

    def read(self, phase, games, *, max_date=None):
        import pyarrow.parquet as pq
        games, parts = set(int(g) for g in games), []
        for name, sha in self.pins.items():
            path = self.root / name
            if file_sha256(path) != sha:
                raise ReadRefused(f'raw Statcast file differs from its pin: {name}')
            parquet = pq.ParquetFile(path)
            groups = _groups(parquet, path)
            if any(g.high is not None and (g.high > LAST_DATE or g.high.year >= 2026) for g in groups):
                raise ReadRefused(f'{name}: row-group statistics show a 2026 row or a date after 2025-09-30; stop')
            for group in groups:
                if max_date is not None and (group.high is None or group.high > max_date):
                    continue  # not decoded (date statistics end after the phase limit)
                frame = parquet.read_row_group(group.index, columns=RAW_KEY_COLUMNS).to_pandas()
                frame['game_date'] = pd.to_datetime(frame.game_date).dt.normalize()
                check_dates(frame.game_date, f'{name} row group {group.index}')
                _log_entry(self.log, phase=phase, path=path, group=group, columns=RAW_KEY_COLUMNS, start=0,
                           stop=group.rows, dates=frame.game_date, guarded=False)
                parts.append(frame.loc[frame.game_pk.isin(games), RAW_KEY_COLUMNS])
            if file_sha256(path) != sha:
                raise IntegrityError(f'raw Statcast file changed while it was read: {name}')
        raw = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=RAW_KEY_COLUMNS)
        return raw.reset_index(drop=True)


def assemble(free, guarded, keep):
    """Regular-season rows of the ``keep`` splits; DEV rows without decoded guarded values carry
    SEALED (strings) / missing (numbers), never a value."""
    frame = free.merge(guarded, on='row_id', how='left', validate='one_to_one', indicator=True)
    frame = frame.loc[frame.game_type.eq('R')].copy()
    frame['split'] = split_of(frame.game_date)
    frame = frame.loc[frame.split.isin(keep)].copy()
    missing = frame.pop('_merge').eq('left_only').to_numpy()
    if (missing & frame.split.ne('dev').to_numpy()).any():
        raise ReadRefused('a non-DEV row lacks its guarded columns')
    for column in STRING_GUARDED:
        frame[column] = frame[column].astype(object)
        frame.loc[missing, column] = SEALED
    frame['is_pa_terminal'] = frame['is_pa_terminal'].astype(object)
    frame.loc[missing, 'is_pa_terminal'] = None
    frame.loc[missing, 'launch_angle'] = np.nan
    frame = frame.sort_values(['game_date', 'game_pk', 'at_bat_number', 'pitch_number'], kind='stable')
    return frame.reset_index(drop=True)


class A3Inputs:
    """The inputs of one attempt behind phase-specific accessors (every decode is logged)."""
    def __init__(self, *, reader, raw_reader, b0, support, hands, archetypes, ledger, identity, log):
        self.reader, self.raw_reader, self.b0 = reader, raw_reader, b0
        self.support, self.hands, self.archetypes = support, hands, archetypes
        self.ledger, self.identity, self.log = ledger, identity, log
        self._free = self._frame = None
        self.released = None

    def smoke_frame(self, phase):
        """TRAIN rows of the row groups that end on or before 2025-04-30 (no later row is decoded)."""
        groups = self.reader.groups_ending_by(TRAIN_END)
        if not groups:
            raise ReadRefused('no processed row group ends on or before 2025-04-30; the smoke cannot stay TRAIN-only')
        free = self.reader.free(phase, groups)
        return assemble(free, self.reader.guarded_prefix(phase, TRAIN_END, groups), keep=('train',))

    def _free_all(self, phase):
        if self._free is None:
            self._free = self.reader.free(phase)
        return self._free

    def census_view(self, phase):
        free = self._free_all(phase)
        frame = free.loc[free.game_type.eq('R')].copy()
        frame['split'] = split_of(frame.game_date)
        return label_free_view(frame.loc[frame.split.isin(reg.USED_SPLITS)])

    def pre_gate_frame(self, phase):
        if self._frame is None:
            free = self._free_all(phase)
            self._frame = assemble(free, self.reader.guarded_prefix(phase, PRE_GATE_LIMIT), keep=reg.USED_SPLITS)
        return self._frame

    def raw_keys(self, phase, games, max_date=None):
        """Raw Statcast keys of ``games``; with ``max_date`` only raw row groups ending by that date."""
        return self.raw_reader.read(phase, games, max_date=max_date)

    def verify_unchanged(self):
        """The processed parquet bytes still equal the sha256 hashed before the first decode."""
        self.reader.verify_unchanged()
        return True

    def release_dev(self, gate, phase):
        """Decode the DEV guarded columns (after the gate) into the pre-gate frame."""
        frame = self.pre_gate_frame(phase)
        guarded = self.reader.guarded_after_gate(phase, gate).set_index('row_id')
        dev = frame.split.eq('dev').to_numpy()
        ids = frame.loc[dev, 'row_id'].to_numpy()
        missing = ~np.isin(ids, guarded.index.to_numpy())
        if missing.any():
            raise IntegrityError(f'{int(missing.sum())} DEV rows have no decoded guarded values')
        values = guarded.loc[ids]
        for column in GUARDED:
            frame.loc[dev, column] = values[column].to_numpy()
        self.released = phase
        digest = hashlib.sha256('\n'.join('' if pd.isna(v) else str(v) for v in frame.loc[dev, 'pitch_type']).encode())
        return {'dev_rows': int(dev.sum()), 'dev_pitch_type_sha256': digest.hexdigest()}

    def reads(self, phase=None):
        return list(self.log.entries) if phase is None else self.log.for_phase(phase)
