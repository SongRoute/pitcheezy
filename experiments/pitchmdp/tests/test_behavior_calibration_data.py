"""EXP-BC-CAL-01 stage-specific acquisition (pitchmdp.behavior_data) on a synthetic processed parquet:
the smoke decodes only TRAIN row groups, the census no guarded column, the pre-gate frame guarded
values only up to 2025-05-31 (pyarrow prefix reads, never a DEV value), DEV guarded columns only after
freeze-primary is pinned and s6-integrity passed; 2026 / post-2025-09-30 rows are refused before any
decode. Every pyarrow decode call is recorded by a spy and compared with the read log. No real data."""
import hashlib
from pathlib import Path
import sys

import pandas as pd
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from behavior_calibration_synthetic import (extra_rows, make_frame, processed_order, write_parquet,  # noqa: E402
                                            write_processed, write_raw, PROCESSED_COLUMNS)
from pitchmdp import behavior_calibration_registry as reg  # noqa: E402
from pitchmdp.behavior_data import (DEV_START, PRE_GATE_LIMIT, TRAIN_END, A3Inputs, ProcessedReader,  # noqa: E402
                                    RawKeysReader, ReadLog, ReadRefused, assemble, split_of)
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
    """Every ParquetFile decode call: how, row groups, columns, batch size and batches consumed."""
    calls = []
    read_row_group, iter_batches = pq.ParquetFile.read_row_group, pq.ParquetFile.iter_batches

    def spy_read(self, i, columns=None, **kwargs):
        calls.append({'how': 'row_group', 'groups': [i], 'columns': list(columns or []), 'rows': None})
        table = read_row_group(self, i, columns=columns, **kwargs)
        calls[-1]['rows'] = table.num_rows
        return table

    def spy_iter(self, batch_size=65536, row_groups=None, columns=None, **kwargs):
        record = {'how': 'batches', 'groups': list(row_groups), 'columns': list(columns or []),
                  'batch_size': batch_size, 'consumed': 0, 'rows': 0}
        calls.append(record)
        for batch in iter_batches(self, batch_size=batch_size, row_groups=row_groups, columns=columns, **kwargs):
            record['consumed'] += 1
            record['rows'] += batch.num_rows
            yield batch

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


def inputs_for(files, margin=8):
    log = ReadLog()
    reader = ProcessedReader(files['path'], sha256=files['sha'], log=log, margin=margin)
    return A3Inputs(reader=reader, raw_reader=RawKeysReader(files['raw_root'], files['raw_pins'], log), b0=None,
                    support=None, hands=None, archetypes=None, ledger=None, identity={}, log=log)


def group_dates(files, index):
    a, b = files['cuts'][index], files['cuts'][index + 1]
    return files['ordered'].game_date.iloc[a:b]


def test_split_of_matches_the_registered_assign_fold():
    from run_temporal_blend import assign_fold
    dates = pd.date_range('2023-03-30', '2025-09-30', freq='D')
    expected = assign_fold(pd.DataFrame({'game_date': dates}), 2025).split.to_numpy()
    assert (split_of(dates) == expected).all()
    assert set(split_of(dates)) == {'unused', 'train', 'earlystop', 'temperature', 'blend', 'dev'}


def test_smoke_decodes_only_train_row_groups(files, spy):
    inputs = inputs_for(files)
    frame = inputs.smoke_frame('smoke')
    assert frame.split.eq('train').all() and len(frame) > 0
    ends = {i: group_dates(files, i).max() for i in range(len(files['cuts']) - 1)}
    processed = [c for c in spy if 'stand' in c['columns'] or 'pitch_type' in c['columns']]
    assert processed and all(ends[c['groups'][0]] <= TRAIN_END for c in processed)
    assert inputs.log.rows_at_or_after('2025-05-01', guarded_only=False) == 0
    assert inputs.log.rows_at_or_after('2025-07-01', guarded_only=True) == 0
    decoded_pre = sum(e['rows'] for e in inputs.log.entries if e['date_min'] and e['date_min'] < '2023-05-15')
    assert decoded_pre > 0 and frame.game_date.min() >= pd.Timestamp('2023-05-15')  # decoded and dropped
    raw = inputs.raw_keys('smoke', frame.game_pk.unique(), max_date=TRAIN_END)
    assert set(frame.game_pk) <= set(raw.game_pk) and raw.game_date.max() <= TRAIN_END
    raw_calls = [c for c in spy if c['columns'] == ['game_date', 'game_pk', 'at_bat_number', 'pitch_number', 'pitcher']]
    assert raw_calls and inputs.log.rows_at_or_after('2025-05-01', guarded_only=False, phases=['smoke']) == 0
    assert all('pitch_type' not in c['columns'] for c in raw_calls)


def test_census_decodes_no_guarded_column(files, spy):
    inputs = inputs_for(files)
    view = inputs.census_view('census')
    assert list(view.columns) == list(CENSUS_COLUMNS) and set(view.split) <= set(reg.USED_SPLITS)
    assert spy and all(not GUARDED & set(c['columns']) for c in spy)
    assert all(not e['guarded'] for e in inputs.reads('census')) and 'release_speed' not in str(spy)


def test_pre_gate_frame_never_materializes_a_dev_guarded_value(files, spy):
    inputs = inputs_for(files)
    frame = inputs.pre_gate_frame('fit')
    guarded = [c for c in spy if GUARDED & set(c['columns'])]
    assert guarded
    for call in guarded:
        dates = group_dates(files, call['groups'][0])
        limit_rows = int((dates <= PRE_GATE_LIMIT).sum())
        if call['how'] == 'row_group':
            assert dates.max() <= PRE_GATE_LIMIT
        else:  # straddling group: exactly one batch of the dated prefix
            assert call['batch_size'] == limit_rows == call['rows'] and call['consumed'] == 1
            assert (dates >= DEV_START).any()
    assert inputs.log.rows_at_or_after('2025-07-01', guarded_only=True) == 0
    dev = frame.split.eq('dev')
    assert dev.any() and frame.loc[dev, 'pitch_type'].eq(SEALED).all()
    assert frame.loc[dev, 'description'].eq(SEALED).all() and frame.loc[dev, 'events'].eq(SEALED).all()
    assert frame.loc[dev, 'is_pa_terminal'].isna().all() and frame.loc[dev, 'launch_angle'].isna().all()
    assert not frame.loc[~dev, 'pitch_type'].astype(object).eq(SEALED).any()
    assert set(frame.split) == set(reg.USED_SPLITS)  # June and 2023-03-30..05-14 dropped
    assert inputs.pre_gate_frame('calibrate') is frame and not inputs.reads('calibrate')  # cached, no new read


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
    assert inputs.log.rows_at_or_after('2025-07-01', guarded_only=True) == 0 and inputs.released is None
    gate.passed.add('s6-integrity')
    released = inputs.release_dev(gate, 'xf-fit')
    dev = frame.loc[frame.split.eq('dev')]
    original = processed_order(world).loc[lambda f: f.split.eq('dev')]
    assert released['dev_rows'] == len(dev) == len(original)
    clean = lambda s: s.astype(object).where(s.notna(), None).tolist()
    assert clean(dev.pitch_type) == clean(original.pitch_type) and clean(dev.events) == clean(original.events)
    assert inputs.log.rows_at_or_after('2025-07-01', guarded_only=True, phases=['xf-fit']) == len(dev)
    assert inputs.log.rows_at_or_after('2025-07-01', guarded_only=True, phases=['fit']) == 0


def test_prefix_margin_unsorted_groups_and_missing_statistics_are_refused(tmp_path, files, frames):
    inputs = inputs_for(files, margin=10_000)  # June rows (< 10,000) separate the prefix from DEV
    with pytest.raises(ReadRefused, match='DEV row lies within'):
        inputs.pre_gate_frame('fit')
    ordered = files['ordered'].copy()
    a, b = files['cuts'][2], files['cuts'][3]
    ordered.iloc[a:b] = ordered.iloc[a:b].iloc[::-1].to_numpy()  # reverse one straddling group
    path = write_parquet(tmp_path / 'unsorted.parquet', ordered, PROCESSED_COLUMNS, files['cuts'])
    log = ReadLog()
    reader = ProcessedReader(path, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), log=log, margin=8)
    reader.free('fit')
    with pytest.raises(ReadRefused, match='not date-sorted'):
        reader.guarded_prefix('fit', PRE_GATE_LIMIT)


@pytest.mark.parametrize('late', ['2026-04-01', '2025-10-01'])
def test_2026_and_late_rows_are_refused_before_any_decode(tmp_path, frames, spy, late):
    world, _ = frames
    bad = world.copy()
    bad.loc[bad.index[-5:], 'game_date'] = pd.Timestamp(late)
    path, sha, _ = write_processed(tmp_path, bad)
    with pytest.raises(ReadRefused, match='2026 row or a date after 2025-09-30'):
        ProcessedReader(path, sha256=sha, log=ReadLog(), margin=8)
    assert spy == []  # refused from row-group statistics: nothing decoded
    with pytest.raises(ReadRefused, match='registered sha256'):
        ProcessedReader(path, sha256='0' * 64, log=ReadLog(), margin=8)


def test_raw_key_reader_reads_keys_only_and_checks_bytes(tmp_path, files, frames, spy):
    log = ReadLog()
    raw = RawKeysReader(files['raw_root'], files['raw_pins'], log)
    games = frames[0].game_pk.unique()
    keys = raw.read('fit', games)
    assert list(keys.columns) == ['game_date', 'game_pk', 'at_bat_number', 'pitch_number', 'pitcher']
    assert len(keys) == len(frames[0]) and all('pitch_type' not in c['columns'] for c in spy)
    name = next(iter(files['raw_pins']))
    with pytest.raises(ReadRefused, match='differs from its pin'):
        RawKeysReader(files['raw_root'], {**files['raw_pins'], name: '0' * 64}, log).read('fit', games)
    late = processed_order(frames[0])
    late.loc[late.index[-3:], 'game_date'] = pd.Timestamp('2026-03-27')
    path = write_parquet(tmp_path / 'raw26' / 'statcast_x.parquet', late,
                         ['game_date', 'game_pk', 'at_bat_number', 'pitch_number', 'pitcher'], [0, 100, len(late)])
    calls = len(spy)
    with pytest.raises(ReadRefused, match='2026 row'):  # even when that row group would be skipped by date
        RawKeysReader(path.parent, {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}, log).read(
            'smoke', games, max_date=TRAIN_END)
    assert len(spy) == calls  # refused from statistics, nothing decoded


def test_assemble_orders_rows_and_refuses_a_non_dev_row_without_guarded_values(files):
    inputs = inputs_for(files)
    frame = inputs.pre_gate_frame('fit')
    order = frame[['game_date', 'game_pk', 'at_bat_number', 'pitch_number']]
    assert order.equals(order.sort_values(list(order.columns), kind='stable').reset_index(drop=True))
    free = inputs.reader.free('again')
    guarded = inputs.reader.guarded_prefix('again', PRE_GATE_LIMIT)
    with pytest.raises(ReadRefused, match='non-DEV row lacks'):
        assemble(free, guarded.iloc[:-5], keep=reg.USED_SPLITS)  # temperature rows without guarded values
    with pytest.raises(ReadRefused, match='guarded reads need the row group dates'):
        fresh = ProcessedReader(files['path'], sha256=files['sha'], log=ReadLog(), margin=8)
        fresh.guarded_prefix('fit', PRE_GATE_LIMIT)
