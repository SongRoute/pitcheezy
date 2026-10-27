"""Stage-specific acquisition of the EXP-BC-CAL-01 rows (no shared full-file loader is used).

Columns are FREE (keys, date, game type, pre-pitch fields) or GUARDED (pitch_type and the outcome
columns description, events, is_pa_terminal, launch_angle). Every pyarrow read goes through ``_read``:
the read is written to the ``ReadLog`` BEFORE pyarrow is called (file, row group, columns, the row
group's footer statistics) and completed afterwards (rows decoded, kept and dropped, validation).
The log hands every event to its sink at once (the runner appends them to the attempt's hash-chained
reads.jsonl), so a failed or interrupted acquisition is never lost.

Row groups are chosen from footer statistics (game_date minimum, maximum and null count):
- a row group wholly outside the phase window is skipped and never read (recorded as a skip);
- a row group straddling a window boundary is read, and its out-of-window rows are dropped and
  counted (decoded-then-dropped, disclosed in the stage record; straddling row groups only);
- before the gate, GUARDED columns are never read from a DEV-bearing row group (game_date maximum on
  or after 2025-07-01, or statistics missing): the reader refuses before pyarrow is called. No DEV
  pitch type, description, event, launch angle or dictionary page can be decoded before the gate,
  because those column chunks are never passed to pyarrow (physically safe acquisition).

Phases: smoke = row groups lying wholly on or before 2025-04-30 (FREE + GUARDED); census = FREE only;
pre-gate (fit .. freeze-primary) = GUARDED of the row groups overlapping 2023-05-15..2025-05-31;
after the gate (freeze-primary pinned AND s6-integrity passed) = GUARDED of the row groups overlapping
DEV. Raw Statcast files: key columns only. Each accessor is allowed only in its own phases.

pyarrow parses a file's footer (metadata and statistics of every column chunk) when the file is
opened; the reader uses only game_date statistics and row counts. File bytes are sha256-hashed before
the first decode and after the last (authorized raw-byte hashing). A 2026 date, a date after
2025-09-30 or a null date stops the run (from statistics before a decode, from values after it).
"""
from __future__ import annotations

from collections import Counter
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
TRAIN_START, TRAIN_END = (pd.Timestamp(d) for d in reg.SPLIT_DATES['train'])
PRE_GATE_START, PRE_GATE_LIMIT = (pd.Timestamp(d) for d in reg.READ_POLICY['pre_gate_window'])
DEV_START, DEV_END = (pd.Timestamp(d) for d in reg.SPLIT_DATES['dev'])
LAST_DATE = pd.Timestamp(reg.LAST_DATE)
WINDOWS = {'train': ((TRAIN_START, TRAIN_END),), 'pre_gate': ((PRE_GATE_START, PRE_GATE_LIMIT),),
           'dev': ((DEV_START, DEV_END),), 'used': ((PRE_GATE_START, PRE_GATE_LIMIT), (DEV_START, DEV_END))}
STRING_GUARDED = ('pitch_type', 'description', 'events')
PHASES = {'smoke_frame': ('smoke',), 'census_view': ('census',),
          'pre_gate_frame': ('fit', 'calibrate', 'freeze-primary', 'xf-fit'), 'raw_keys': ('smoke', 'fit'),
          'release_dev': ('xf-fit',)}
DISCLOSURE = ('row groups wholly outside the phase window are skipped from footer statistics; rows outside the '
              'window that share a row group with in-window rows (straddling row groups only) are decoded and '
              'dropped, counted in rows_decoded_then_dropped; guarded columns of a DEV-bearing row group are never '
              'read before the gate; pyarrow parses each file footer (metadata and statistics of every column chunk) '
              'on open, of which only game_date statistics and row counts are used')


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


def _day(value):
    return None if value is None else str(pd.Timestamp(value).date())


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
        raise ReadRefused(f'{where}: a null date, a 2026 row or a date after 2025-09-30 was found; stop')


class ReadLog:
    """Every acquisition of an attempt. ``begin`` writes the read BEFORE pyarrow is called, ``finish``
    completes it; skips, file opens and refusals are notes. ``sink`` receives each event at once."""
    def __init__(self, sink=None):
        self.entries, self.sink = [], sink

    def _emit(self, event):
        if self.sink is not None:
            self.sink(event)

    def begin(self, **entry):
        entry = {'index': len(self.entries), 'kind': 'read', 'status': 'attempted', **entry}
        self.entries.append(entry)
        self._emit({'event': 'begin', **entry})
        return entry

    def finish(self, entry, **fields):
        entry.update(fields)
        self._emit({'event': 'finish', 'index': entry['index'], **fields})
        return entry

    def note(self, **entry):
        entry = {'index': len(self.entries), **entry}
        self.entries.append(entry)
        self._emit({'event': entry['kind'], **entry})
        return entry

    def for_phase(self, phase):
        return [e for e in self.entries if e.get('phase') == phase]

    def reads(self, phases=None):
        return [e for e in self.entries if e['kind'] == 'read' and (phases is None or e['phase'] in phases)]

    def rows_at_or_after(self, since, *, guarded_only, phases=None):
        """Rows decoded with a date >= ``since`` ('2025-07-01' or '2025-05-01')."""
        return int(sum((e.get('rows_at_or_after') or {}).get(since, 0) for e in self.reads(phases)
                       if e['guarded'] or not guarded_only))

    def dev_bearing_guarded_reads(self, phases=None):
        """pyarrow reads of a GUARDED column chunk of a DEV-bearing row group (whatever came back)."""
        return sum(1 for e in self.reads(phases) if e['guarded'] and e['dev_bearing'])

    def summary(self, phases=None):
        reads = self.reads(phases)
        notes = [e for e in self.entries if e['kind'] != 'read' and (phases is None or e.get('phase') in phases)]
        dropped = Counter()
        for e in reads:
            dropped.update(e.get('rows_dropped') or {})
        return {'pyarrow_reads': len(reads), 'guarded_reads': sum(1 for e in reads if e['guarded']),
                'dev_bearing_guarded_reads': sum(1 for e in reads if e['guarded'] and e['dev_bearing']),
                'rows_decoded': int(sum(e.get('rows_decoded', 0) for e in reads)),
                'rows_kept': int(sum(e.get('rows_kept', 0) for e in reads)),
                'rows_decoded_then_dropped': dict(sorted(dropped.items())),
                'row_groups_skipped': sum(1 for e in notes if e['kind'] == 'skip'),
                'refusals': sum(1 for e in notes if e['kind'] == 'refused'),
                'files_opened': sorted({e['file'] for e in notes if e['kind'] == 'open'}),
                'incomplete_or_failed': [e['index'] for e in reads
                                         if e['status'] != 'decoded' or e.get('validation') != 'passed'],
                'disclosure': DISCLOSURE}


@dataclass
class Group:
    index: int
    offset: int
    rows: int
    low: object
    high: object
    nulls: object

    @property
    def known(self):
        return self.low is not None and self.high is not None and self.nulls == 0

    @property
    def dev_bearing(self):
        return not self.known or self.high >= DEV_START

    def overlaps(self, windows):
        return not self.known or any(self.low <= end and self.high >= start for start, end in windows)

    def describe(self):
        return {'row_group': self.index, 'group_rows': self.rows, 'group_low': _day(self.low),
                'group_high': _day(self.high), 'group_null_dates': self.nulls, 'dev_bearing': self.dev_bearing}


def _groups(parquet, path, column='game_date'):
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
        has = stats is not None and stats.has_min_max
        nulls = stats.null_count if stats is not None and stats.has_null_count else None
        groups.append(Group(i, offset, group.num_rows, _timestamp(stats.min) if has else None,
                            _timestamp(stats.max) if has else None, nulls))
        offset += group.num_rows
    for group in groups:
        if group.high is not None and (group.high > LAST_DATE or group.high.year >= 2026):
            raise ReadRefused(f'{path.name} row group {group.index}: statistics show a 2026 row or a date after '
                              '2025-09-30; stop')
        if group.nulls:
            raise ReadRefused(f'{path.name} row group {group.index}: statistics show null dates; stop')
    return groups


def _windows(windows):
    return [[_day(start), _day(end)] for start, end in windows]


def _window_mask(dates, windows, game_type=None):
    inside = np.zeros(len(dates), dtype=bool)
    for start, end in windows:
        inside |= dates.between(start, end).to_numpy()
    before = (dates < windows[0][0]).to_numpy()
    after = (dates > windows[-1][1]).to_numpy()
    regular = np.ones(len(dates), dtype=bool) if game_type is None else game_type.eq('R').to_numpy()
    keep = inside & regular
    return keep, {'before_window': int(before.sum()), 'between_windows': int((~inside & ~before & ~after).sum()),
                  'after_window': int(after.sum()), 'not_regular_season': int((inside & ~regular).sum())}


def _read(log, parquet, path, group, columns, *, phase, guarded, windows, regular_season=False):
    """One pyarrow read of one row group: logged before the call, completed after (decode and
    validation outcome attached to the same entry). Returns (frame, dates, keep mask)."""
    entry = log.begin(phase=phase, file=path.name, columns=list(columns), guarded=bool(guarded), pyarrow_called=True,
                      windows=_windows(windows), **group.describe())
    try:
        frame = parquet.read_row_group(group.index, columns=list(columns)).to_pandas()
    except Exception as error:
        log.finish(entry, status=f'error: {type(error).__name__}: {error}')
        raise
    log.finish(entry, status='decoded', rows_decoded=int(len(frame)))
    try:
        dates = pd.to_datetime(frame.game_date).dt.normalize()
        log.finish(entry, date_min=_day(dates.min()) if dates.notna().any() else None,
                   date_max=_day(dates.max()) if dates.notna().any() else None,
                   rows_at_or_after={_day(DEV_START): int((dates >= DEV_START).sum()),
                                     _day(TRAIN_END + pd.Timedelta(days=1)): int((dates > TRAIN_END).sum())})
        check_dates(dates, f'{path.name} row group {group.index}')
    except Exception as error:
        log.finish(entry, validation=f'failed: {error}')
        raise ReadRefused(f'{path.name} row group {group.index}: {error}') from error
    keep, dropped = _window_mask(dates, windows, frame.game_type if regular_season else None)
    log.finish(entry, validation='passed', rows_kept=int(keep.sum()), rows_dropped=dropped)
    return frame, dates, keep


def _select(log, path, groups, phase, windows, columns, upper=None):
    """Row groups to read; a row group wholly outside ``windows`` (or not wholly on or before
    ``upper``) is skipped from its statistics and recorded."""
    chosen = []
    for group in groups:
        if group.known and not group.overlaps(windows):
            reason = 'wholly outside the phase window'
        elif upper is not None and not (group.known and group.high <= upper):
            reason = f'not wholly on or before {_day(upper)} (or no statistics)'
        else:
            chosen.append(group)
            continue
        log.note(kind='skip', phase=phase, file=path.name, columns=list(columns), reason=reason,
                 windows=_windows(windows), **group.describe())
    return chosen


class ProcessedReader:
    """Phase-limited reads of the pinned processed parquet (see the module docstring)."""
    def __init__(self, path, *, sha256, log, phase='open'):
        import pyarrow.parquet as pq
        self.path, self.sha256, self.log = Path(path), sha256, log
        log.note(kind='open', phase=phase, file=self.path.name, what='sha256 of the bytes, then the Parquet footer')
        if file_sha256(self.path) != sha256:
            raise ReadRefused('processed parquet differs from its registered sha256')
        self.file = pq.ParquetFile(self.path)
        missing = set(FREE + GUARDED) - set(self.file.schema_arrow.names)
        if missing:
            raise ReadRefused(f'processed parquet lacks {sorted(missing)}')
        self.groups = _groups(self.file, self.path)
        self.rows = sum(g.rows for g in self.groups)
        self.dates = {}

    def verify_unchanged(self):
        if file_sha256(self.path) != self.sha256:
            raise IntegrityError('processed parquet changed while the run read it')

    def layout(self):
        """Footer-statistics layout check (nothing decoded): every row group has game_date statistics
        without nulls, no row group overlapping the pre-gate window holds DEV rows, and the smoke has a
        TRAIN row group lying wholly on or before 2025-04-30."""
        problems = []
        for g in self.groups:
            if not g.known:
                problems.append(f'row group {g.index}: game_date statistics missing')
            elif g.overlaps(WINDOWS['pre_gate']) and g.dev_bearing:
                problems.append(f'row group {g.index} ({_day(g.low)}..{_day(g.high)}) holds pre-gate and DEV rows')
        if not any(g.known and g.high <= TRAIN_END and g.overlaps(WINDOWS['train']) for g in self.groups):
            problems.append('no row group with TRAIN rows lies wholly on or before 2025-04-30')
        return {'row_groups': [g.describe() for g in self.groups], 'split_aligned': not problems,
                'problems': problems}

    def free(self, phase, windows, *, upper=None):
        """FREE columns of the row groups overlapping ``windows`` (in-window regular-season rows kept)."""
        parts = []
        for g in _select(self.log, self.path, self.groups, phase, windows, FREE, upper):
            frame, dates, keep = _read(self.log, self.file, self.path, g, FREE, phase=phase, guarded=False,
                                       windows=windows, regular_season=True)
            frame['game_date'] = dates.to_numpy()
            frame['row_id'] = np.arange(g.offset, g.offset + g.rows)
            self.dates[g.index] = dates.to_numpy()
            parts.append(frame.loc[keep])
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=[*FREE, 'row_id'])

    def guarded(self, phase, windows, *, upper=None, gate=None):
        """GUARDED columns of the row groups overlapping ``windows``. Without ``gate`` (before the gate)
        a DEV-bearing row group is refused before pyarrow is called; with it, freeze-primary must be
        pinned and s6-integrity passed first."""
        columns = ['game_date', *GUARDED]
        if gate is not None:
            gate.require_pinned('freeze-primary', f'{phase}: DEV guarded columns')
            gate.require_passed('s6-integrity', f'{phase}: DEV guarded columns')
        groups = _select(self.log, self.path, self.groups, phase, windows, columns, upper)
        unsafe = [g for g in groups if g.dev_bearing] if gate is None else []
        for g in unsafe:
            self.log.note(kind='refused', phase=phase, file=self.path.name, columns=columns, guarded=True,
                          pyarrow_called=False, reason='DEV-bearing row group before the gate', **g.describe())
        if unsafe:
            raise ReadRefused(f'guarded columns of DEV-bearing row groups {[g.index for g in unsafe]} before the gate '
                              '(the processed layout is not split-aligned)')
        parts = []
        for g in groups:
            if g.index not in self.dates:
                raise ReadRefused('guarded reads need the row group dates of a free read first')
            frame, dates, keep = _read(self.log, self.file, self.path, g, columns, phase=phase, guarded=True,
                                       windows=windows)
            if not np.array_equal(dates.to_numpy(), self.dates[g.index]):
                raise ReadRefused(f'row group {g.index}: guarded rows are not the free-read rows')
            frame['row_id'] = np.arange(g.offset, g.offset + g.rows)
            parts.append(frame.loc[keep].drop(columns='game_date'))
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=[*GUARDED, 'row_id'])


class RawKeysReader:
    """Keys/date/pitcher of the pinned raw Statcast files (workload source; label-free columns only)."""
    def __init__(self, root, pins, log):
        self.root, self.pins, self.log = Path(root), dict(pins), log

    def read(self, phase, games, windows, *, upper=None):
        import pyarrow.parquet as pq
        games, parts = set(int(g) for g in games), []
        for name, sha in self.pins.items():
            path = self.root / name
            self.log.note(kind='open', phase=phase, file=name, what='sha256 of the bytes, then the Parquet footer')
            if file_sha256(path) != sha:
                raise ReadRefused(f'raw Statcast file differs from its pin: {name}')
            parquet = pq.ParquetFile(path)
            for group in _select(self.log, path, _groups(parquet, path), phase, windows, RAW_KEY_COLUMNS, upper):
                frame, dates, keep = _read(self.log, parquet, path, group, RAW_KEY_COLUMNS, phase=phase, guarded=False,
                                           windows=windows)
                frame['game_date'] = dates.to_numpy()
                part = frame.loc[keep]
                parts.append(part.loc[part.game_pk.isin(games), RAW_KEY_COLUMNS])
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
    """The inputs of one attempt behind phase-specific accessors (every acquisition is logged)."""
    def __init__(self, *, reader, raw_reader, b0, support, hands, archetypes, ledger, identity, log):
        self.reader, self.raw_reader, self.b0 = reader, raw_reader, b0
        self.support, self.hands, self.archetypes = support, hands, archetypes
        self.ledger, self.identity, self.log = ledger, identity, log
        self._free = self._frame = None
        self.released = None

    def _allow(self, accessor, phase):
        if phase not in PHASES[accessor]:
            self.log.note(kind='refused', phase=phase, accessor=accessor, pyarrow_called=False,
                          reason='accessor not allowed in this phase')
            raise ReadRefused(f'{accessor} is not allowed in phase {phase}')

    def layout(self):
        return self.reader.layout()

    def smoke_frame(self, phase):
        """TRAIN rows of the row groups lying wholly on or before 2025-04-30."""
        self._allow('smoke_frame', phase)
        free = self.reader.free(phase, WINDOWS['train'], upper=TRAIN_END)
        if free.empty:
            raise ReadRefused('no processed row group with TRAIN rows lies wholly on or before 2025-04-30')
        return assemble(free, self.reader.guarded(phase, WINDOWS['train'], upper=TRAIN_END), keep=('train',))

    def _free_all(self, phase):
        if self._free is None:
            self._free = self.reader.free(phase, WINDOWS['used'])
        return self._free

    def census_view(self, phase):
        self._allow('census_view', phase)
        frame = self._free_all(phase).copy()
        frame['split'] = split_of(frame.game_date)
        return label_free_view(frame.loc[frame.split.isin(reg.USED_SPLITS)])

    def pre_gate_frame(self, phase):
        self._allow('pre_gate_frame', phase)
        if self._frame is None:
            free = self._free_all(phase)
            self._frame = assemble(free, self.reader.guarded(phase, WINDOWS['pre_gate']), keep=reg.USED_SPLITS)
        return self._frame

    def raw_keys(self, phase, games, *, upper=None):
        """Raw Statcast keys of ``games``: TRAIN window in the smoke (with ``upper``), used windows after."""
        self._allow('raw_keys', phase)
        return self.raw_reader.read(phase, games, WINDOWS['train'] if phase == 'smoke' else WINDOWS['used'],
                                    upper=upper)

    def verify_unchanged(self):
        """The processed parquet bytes still equal the sha256 hashed before the first decode."""
        self.reader.verify_unchanged()
        return True

    def release_dev(self, gate, phase):
        """Decode the DEV guarded columns (after the gate) into the pre-gate frame."""
        self._allow('release_dev', phase)
        gate.require_pinned('freeze-primary', f'{phase}: DEV guarded columns')
        gate.require_passed('s6-integrity', f'{phase}: DEV guarded columns')
        frame = self.pre_gate_frame(phase)
        guarded = self.reader.guarded(phase, WINDOWS['dev'], gate=gate).set_index('row_id')
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
