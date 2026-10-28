"""EXP-BC-CAL-01 stage-specific acquisition (pitchmdp.behavior_data) on synthetic processed parquets.
A spy on pyarrow.parquet.ParquetFile records every column chunk passed to pyarrow (file, row group,
columns) and is compared with the reader's own log: the smoke reads only TRAIN row groups, the census
no guarded column, and before the gate no guarded column chunk of a DEV-bearing row group is ever
passed to pyarrow (an unsafe layout is refused before the call); wholly excluded row groups are
skipped and straddling ones decoded-then-dropped with counts; every acquisition (also a failed one) is
logged before pyarrow is called. No real data."""
import hashlib
from pathlib import Path
import sys

import pandas as pd
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from behavior_calibration_synthetic import (PROCESSED_COLUMNS, default_cuts, extra_rows, make_frame,  # noqa: E402
                                            processed_order, straddling_cuts, write_parquet, write_processed,
                                            write_raw)
from pitchmdp import behavior_calibration_registry as reg  # noqa: E402
from pitchmdp.behavior_data import (DEV_START, PRE_GATE_LIMIT, RAW_KEY_COLUMNS, TRAIN_END, WINDOWS,  # noqa: E402
                                    A3Inputs, ProcessedReader, RawKeysReader, ReadLog, ReadRefused, assemble,
                                    split_of)
from pitchmdp.behavior_features import CENSUS_COLUMNS, SEALED  # noqa: E402

GUARDED = set(reg.GUARDED_COLUMNS)


@pytest.fixture(scope='module')
def frames():
    world = make_frame(0)
    return world, pd.concat([world, extra_rows()], ignore_index=True)


@pytest.fixture()
def files(tmp_path, frames):
    _, frame = frames
    path, sha, cuts = write_processed(tmp_path, frame)
    root, pins = write_raw(tmp_path, frame)
    return {'path': path, 'sha': sha, 'cuts': cuts, 'raw_root': root, 'raw_pins': pins,
            'ordered': processed_order(frame)}


@pytest.fixture()
def spy(monkeypatch):
    """Every column chunk passed to pyarrow: (file, row groups, columns); iter_batches is never used."""
    calls = []
    init, read_row_group = pq.ParquetFile.__init__, pq.ParquetFile.read_row_group

    def spy_init(self, source, *args, **kwargs):
        self._spy_file = Path(str(source)).name
        init(self, source, *args, **kwargs)

    def spy_read(self, i, columns=None, **kwargs):
        calls.append({'file': self._spy_file, 'row_group': i, 'columns': list(columns or [])})
        return read_row_group(self, i, columns=columns, **kwargs)

    def spy_iter(self, *args, **kwargs):
        raise AssertionError('the A3 reader never uses iter_batches (row-group reads only)')

    monkeypatch.setattr(pq.ParquetFile, '__init__', spy_init)
    monkeypatch.setattr(pq.ParquetFile, 'read_row_group', spy_read)
    monkeypatch.setattr(pq.ParquetFile, 'iter_batches', spy_iter)
    return calls


class Gate:
    """Stand-in for the runner gate: refuses until the pins are granted."""
    def __init__(self):
        self.pinned, self.passed = set(), set()

    def require_pinned(self, name, purpose):
        if name not in self.pinned:
            raise ReadRefused(f'{purpose} needs the {name} manifest pinned first')

    def require_passed(self, stage, purpose):
        if stage not in self.passed:
            raise ReadRefused(f'{purpose} needs a passed {stage} stage in this attempt')


def inputs_for(files, events=None, path=None, sha=None):
    log = ReadLog(sink=None if events is None else events.append)
    reader = ProcessedReader(path or files['path'], sha256=sha or files['sha'], log=log)
    return A3Inputs(reader=reader, raw_reader=RawKeysReader(files['raw_root'], files['raw_pins'], log), b0=None,
                    support=None, hands=None, archetypes=None, ledger=None, identity={}, log=log)


def group_dates(files, index, cuts=None):
    cuts = cuts or files['cuts']
    return files['ordered'].game_date.iloc[cuts[index]:cuts[index + 1]]


def log_calls(log, phases=None):
    return [{'file': e['file'], 'row_group': e['row_group'], 'columns': e['columns']} for e in log.reads(phases)]


def test_split_of_matches_the_registered_assign_fold():
    from run_temporal_blend import assign_fold
    dates = pd.date_range('2023-03-30', '2025-09-30', freq='D')
    expected = assign_fold(pd.DataFrame({'game_date': dates}), 2025).split.to_numpy()
    assert (split_of(dates) == expected).all()
    assert set(split_of(dates)) == {'unused', 'train', 'earlystop', 'temperature', 'blend', 'dev'}


def test_the_reader_refuses_a_layout_that_is_not_split_aligned_from_the_footer(tmp_path, files, frames, spy):
    inputs = inputs_for(files)
    layout = inputs.layout()
    assert layout['split_aligned'] and not layout['problems'] and spy == []
    for name, kwargs, message in (('straddling', {'cuts': straddling_cuts}, 'spans the split dates'),
                                  ('bare', {'statistics': False}, 'statistics missing')):
        path, sha, _ = write_processed(tmp_path / name, frames[1], **kwargs)
        log = ReadLog()
        with pytest.raises(ReadRefused, match='not split-aligned'):
            ProcessedReader(path, sha256=sha, log=log)
        assert any(message in p for p in log.entries[-1]['problems']) and log.entries[-1]['kind'] == 'refused'
    assert spy == []  # refused from the footer: nothing decoded


def test_smoke_reads_only_train_row_groups_and_counts_dropped_rows(files, spy):
    inputs = inputs_for(files)
    frame = inputs.smoke_frame('smoke')
    assert frame.split.eq('train').all() and len(frame) > 0
    processed = [c for c in spy if c['file'] == files['path'].name]
    assert processed and all(group_dates(files, c['row_group']).max() <= TRAIN_END for c in processed)
    assert 0 not in {c['row_group'] for c in processed}  # wholly pre-TRAIN row group skipped
    assert log_calls(inputs.log) == spy  # the reader's own log is exactly what pyarrow received
    summary = inputs.log.summary(['smoke'])
    assert group_dates(files, 0).max() < pd.Timestamp('2023-05-15')  # the pre-TRAIN row group: skipped
    assert not any(summary['rows_decoded_then_dropped'].values())  # split-aligned: nothing decoded then dropped
    assert summary['row_groups_skipped'] > 0 and summary['dev_bearing_guarded_reads'] == 0
    assert inputs.log.rows_at_or_after('2025-05-01', guarded_only=False) == 0
    raw = inputs.raw_keys('smoke', frame.game_pk.unique(), upper=TRAIN_END)
    assert set(frame.game_pk) <= set(raw.game_pk) and raw.game_date.max() <= TRAIN_END
    raw_calls = [c for c in spy if c['file'] != files['path'].name]
    assert raw_calls and all(c['columns'] == RAW_KEY_COLUMNS for c in raw_calls)
    assert inputs.log.rows_at_or_after('2025-05-01', guarded_only=False, phases=['smoke']) == 0


def test_census_reads_free_columns_only_and_skips_wholly_excluded_groups(files, spy):
    inputs = inputs_for(files)
    view = inputs.census_view('census')
    assert list(view.columns) == list(CENSUS_COLUMNS) and set(view.split) == set(reg.USED_SPLITS)
    assert spy and all(not GUARDED & set(c['columns']) for c in spy)
    read = {c['row_group'] for c in spy}
    june_only = [i for i in range(len(files['cuts']) - 1) if group_dates(files, i).between('2025-06-01',
                                                                                             '2025-06-30').all()]
    assert june_only and not read & set(june_only) and 0 not in read
    summary = inputs.log.summary(['census'])
    assert not any(summary['rows_decoded_then_dropped'].values())  # June and pre-TRAIN rows never decoded
    assert summary['guarded_reads'] == 0 and log_calls(inputs.log) == spy


def test_pre_gate_never_passes_a_dev_bearing_guarded_chunk_to_pyarrow(files, spy):
    inputs = inputs_for(files)
    frame = inputs.pre_gate_frame('fit')
    guarded = [c for c in spy if GUARDED & set(c['columns'])]
    assert guarded
    for call in guarded:  # independent of the reader: the dates actually written in that row group
        assert group_dates(files, call['row_group']).max() < DEV_START
    assert inputs.log.dev_bearing_guarded_reads() == 0 and log_calls(inputs.log) == spy
    assert inputs.log.rows_at_or_after('2025-07-01', guarded_only=True) == 0
    assert not any(inputs.log.summary(['fit'])['rows_decoded_then_dropped'].values())
    dev = frame.split.eq('dev')
    assert dev.any() and frame.loc[dev, 'pitch_type'].eq(SEALED).all()
    assert frame.loc[dev, 'description'].eq(SEALED).all() and frame.loc[dev, 'events'].eq(SEALED).all()
    assert frame.loc[dev, 'is_pa_terminal'].isna().all() and frame.loc[dev, 'launch_angle'].isna().all()
    assert not frame.loc[~dev, 'pitch_type'].astype(object).eq(SEALED).any()
    assert set(frame.split) == set(reg.USED_SPLITS)  # June and 2023-03-30..05-14 dropped
    assert inputs.pre_gate_frame('calibrate') is frame and not inputs.log.reads(['calibrate'])  # cached


def test_a_dev_bearing_guarded_chunk_is_refused_before_pyarrow_is_called(files, spy):
    """Defense in depth: even if a pre-gate row group looked DEV-bearing, no guarded chunk reaches pyarrow."""
    inputs = inputs_for(files)
    inputs.census_view('census')
    target = next(g for g in inputs.reader.groups if g.overlaps(WINDOWS['pre_gate']))
    target.high = DEV_START  # as if the footer showed a DEV row in a pre-gate row group
    with pytest.raises(ReadRefused, match='DEV-bearing'):
        inputs.pre_gate_frame('fit')
    assert not [c for c in spy if GUARDED & set(c['columns'])]
    refused = [e for e in inputs.log.entries if e['kind'] == 'refused']
    assert refused and all(e['pyarrow_called'] is False and e['dev_bearing'] for e in refused)
    assert inputs.log.dev_bearing_guarded_reads() == 0


def test_dev_guarded_columns_only_after_the_gate(files, frames):
    world, _ = frames
    inputs = inputs_for(files)
    frame = inputs.pre_gate_frame('fit')
    gate = Gate()
    with pytest.raises(ReadRefused, match='freeze-primary'):
        inputs.release_dev(gate, 'xf-fit')
    gate.pinned.add('freeze-primary')
    with pytest.raises(ReadRefused, match='s6-integrity'):
        inputs.release_dev(gate, 'xf-fit')
    assert inputs.log.dev_bearing_guarded_reads() == 0 and inputs.released is None
    gate.passed.add('s6-integrity')
    released = inputs.release_dev(gate, 'xf-fit')
    dev = frame.loc[frame.split.eq('dev')]
    original = processed_order(world).loc[lambda f: f.split.eq('dev')]
    assert released['dev_rows'] == len(dev) == len(original)
    clean = lambda s: s.astype(object).where(s.notna(), None).tolist()
    assert clean(dev.pitch_type) == clean(original.pitch_type) and clean(dev.events) == clean(original.events)
    assert inputs.log.dev_bearing_guarded_reads(['xf-fit']) > 0
    assert inputs.log.dev_bearing_guarded_reads(['fit', 'census', 'smoke']) == 0


def test_missing_statistics_null_dates_and_failed_reads_stay_in_the_log(tmp_path, files, frames, spy, monkeypatch):
    world = frames[0]
    # no statistics: refused from the footer (nothing can be skipped or proven DEV-free), logged
    bare_path, bare_sha, _ = write_processed(tmp_path / 'bare', world, statistics=False)
    events = []
    with pytest.raises(ReadRefused, match='statistics missing'):
        ProcessedReader(bare_path, sha256=bare_sha, log=ReadLog(sink=events.append))
    assert [e['event'] for e in events] == ['open', 'refused'] and spy == []
    # a raw key file without statistics holding a 2026 date: decoded, then refused; the entry keeps
    # the failed validation and was logged before the read
    late = processed_order(world)
    late.loc[late.index[-3:], 'game_date'] = pd.Timestamp('2026-04-01')
    raw = write_parquet(tmp_path / 'raw' / 'statcast_x.parquet', late, RAW_KEY_COLUMNS, [0, len(late)], statistics=False)
    events = []
    log = ReadLog(sink=events.append)
    with pytest.raises(ReadRefused, match='2026'):
        RawKeysReader(raw.parent, {raw.name: hashlib.sha256(raw.read_bytes()).hexdigest()}, log).read(
            'fit', world.game_pk.unique(), WINDOWS['used'])
    entry = log.reads()[-1]
    assert entry['status'] == 'decoded' and entry['validation'].startswith('failed')
    order = [e['event'] for e in events if e.get('index') == entry['index']]
    assert order[0] == 'begin' and order[-1] == 'finish'
    assert any(e['event'] == 'finish' and str(e.get('validation', '')).startswith('failed') for e in events)
    # null dates with statistics: refused from the footer, nothing decoded
    nulls = processed_order(world)
    nulls['game_date'] = nulls.game_date.astype(object)
    nulls.loc[nulls.index[5], 'game_date'] = None
    null_path = write_parquet(tmp_path / 'nulls.parquet', nulls, PROCESSED_COLUMNS, [0, len(nulls)])
    calls = len(spy)
    with pytest.raises(ReadRefused, match='null dates'):
        ProcessedReader(null_path, sha256=hashlib.sha256(null_path.read_bytes()).hexdigest(), log=ReadLog())
    assert len(spy) == calls
    # pyarrow itself fails: the attempted read is already in the log, with the error attached
    events = []
    broken = inputs_for(files, events)
    monkeypatch.setattr(pq.ParquetFile, 'read_row_group', lambda self, i, columns=None, **k: (_ for _ in ()).throw(
        OSError('disk error')))
    with pytest.raises(OSError):
        broken.census_view('census')
    assert broken.log.reads()[-1]['status'].startswith('error: OSError')
    assert [e['event'] for e in events][-2:] == ['begin', 'finish']


@pytest.mark.parametrize('late', ['2026-04-01', '2025-10-01'])
def test_2026_and_late_rows_are_refused_before_any_decode(tmp_path, frames, spy, late):
    world, _ = frames
    bad = world.copy()
    bad.loc[bad.index[-5:], 'game_date'] = pd.Timestamp(late)
    path, sha, _ = write_processed(tmp_path, bad)
    with pytest.raises(ReadRefused, match='2026 row or a date after 2025-09-30'):
        ProcessedReader(path, sha256=sha, log=ReadLog())
    assert spy == []  # refused from row-group statistics: nothing decoded
    with pytest.raises(ReadRefused, match='registered sha256'):
        ProcessedReader(path, sha256='0' * 64, log=ReadLog())


def test_raw_key_reader_reads_keys_only_and_checks_bytes(tmp_path, files, frames, spy):
    log = ReadLog()
    raw = RawKeysReader(files['raw_root'], files['raw_pins'], log)
    games = frames[0].game_pk.unique()
    keys = raw.read('fit', games, WINDOWS['used'])
    assert list(keys.columns) == RAW_KEY_COLUMNS and len(keys) == len(frames[0])
    assert spy and all(c['columns'] == RAW_KEY_COLUMNS for c in spy) and log_calls(log) == spy
    name = next(iter(files['raw_pins']))
    with pytest.raises(ReadRefused, match='differs from its pin'):
        RawKeysReader(files['raw_root'], {**files['raw_pins'], name: '0' * 64}, log).read('fit', games, WINDOWS['used'])
    late = processed_order(frames[0])
    late.loc[late.index[-3:], 'game_date'] = pd.Timestamp('2026-03-27')
    path = write_parquet(tmp_path / 'raw26' / 'statcast_x.parquet', late, RAW_KEY_COLUMNS, [0, 100, len(late)])
    calls = len(spy)
    with pytest.raises(ReadRefused, match='2026 row'):  # even when that row group would be skipped by date
        RawKeysReader(path.parent, {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}, log).read(
            'smoke', games, WINDOWS['train'], upper=TRAIN_END)
    assert len(spy) == calls  # refused from statistics, nothing decoded


def test_assemble_orders_rows_and_refuses_a_non_dev_row_without_guarded_values(files):
    inputs = inputs_for(files)
    frame = inputs.pre_gate_frame('fit')
    order = frame[['game_date', 'game_pk', 'at_bat_number', 'pitch_number']]
    assert order.equals(order.sort_values(list(order.columns), kind='stable').reset_index(drop=True))
    free = inputs.reader.free('again', WINDOWS['used'])
    guarded = inputs.reader.guarded('again', WINDOWS['pre_gate'])
    with pytest.raises(ReadRefused, match='non-DEV row lacks'):
        assemble(free, guarded.iloc[:-5], keep=reg.USED_SPLITS)  # temperature rows without guarded values
    with pytest.raises(ReadRefused, match='dates of a free read'):
        ProcessedReader(files['path'], sha256=files['sha'], log=ReadLog()).guarded('fit', WINDOWS['pre_gate'])


def test_accessors_are_phase_bound(files):
    inputs = inputs_for(files)
    for call in (lambda: inputs.smoke_frame('fit'), lambda: inputs.census_view('smoke'),
                 lambda: inputs.pre_gate_frame('census'), lambda: inputs.raw_keys('census', []),
                 lambda: inputs.release_dev(Gate(), 'fit')):
        with pytest.raises(ReadRefused, match='not allowed in phase'):
            call()
    assert not inputs.log.reads() and len([e for e in inputs.log.entries if e['kind'] == 'refused']) == 5
    assert PRE_GATE_LIMIT == pd.Timestamp('2025-05-31') and default_cuts is not None
