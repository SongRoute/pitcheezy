"""EXP-BC-CAL-01 one-time input preparation (scripts/prepare_behavior_calibration_input.py) on synthetic
parquets: the rewrite has row-group boundaries exactly at the split dates and is row-for-row identical
(independently re-checked with Arrow equality); the identity check fails on a changed, swapped, missing
or nulled row and a failed check leaves no usable output; the A3 reader refuses the misaligned
original; paths, pins, overwrites, unsorted input and the protection window are refused; and the tool
reads no label column into Python objects (static check of its source, and pandas is never imported in
a fresh interpreter). The identity hash streams every column's bytes, pitch_type included: a byte-level
identity check, not a label read. No real data."""
import ast
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import prepare_behavior_calibration_input as tool  # noqa: E402
from behavior_calibration_synthetic import (PROJECT, extra_rows, make_frame, processed_order,  # noqa: E402
                                            straddling_cuts, write_parquet, write_processed, PROCESSED_COLUMNS)
from pitchmdp.behavior_data import ProcessedReader, ReadLog, ReadRefused  # noqa: E402

AFTERNOON = datetime(2026, 10, 27, 15, 0, tzinfo=tool.KST)
MORNING = datetime(2026, 10, 28, 6, 30, tzinfo=tool.KST)


@pytest.fixture(scope='module')
def frame():
    return pd.concat([make_frame(0), extra_rows()], ignore_index=True)


@pytest.fixture()
def source(tmp_path, frame):
    path, sha, _ = write_processed(tmp_path / 'processed', frame, straddling_cuts)  # size-based, not aligned
    (tmp_path / 'A3').mkdir()
    output = tmp_path / 'A3' / 'inputs' / 'pitches.split-aligned.parquet'
    return {'path': path, 'sha': sha, 'output': output,
            'manifest': output.with_name('pitches.split-aligned.identity.json')}


def run(source, **kwargs):
    return tool.prepare(source['path'], source['output'], source_sha256=source['sha'],
                        now=kwargs.pop('now', lambda: AFTERNOON), **kwargs)


def rewrite_like(table, path, sizes):
    """Write ``table`` with the given row-group sizes (a tampered copy of the rewritten file)."""
    with pq.ParquetWriter(path, table.schema) as writer:
        start = 0
        for size in sizes:
            writer.write_table(table.slice(start, size), row_group_size=size)
            start += size
    return path


def test_the_rewrite_is_split_aligned_identical_and_accepted_by_the_reader(source, frame):
    record = run(source)
    assert record['result'] == 'PASS' and record['aligned'] and record['identity']['result'] == 'PASS'
    ranges = record['rewritten']['row_groups']
    dates = processed_order(frame).game_date
    edges = [pd.Timestamp('1900-01-01'), *(pd.Timestamp(b) for b in tool.BOUNDARIES), pd.Timestamp('2025-10-01')]
    counts = [int(((dates >= a) & (dates < b)).sum()) for a, b in zip(edges[:-1], edges[1:])]
    assert [r['rows'] for r in ranges] == [c for c in counts if c]  # one row group per non-empty segment
    assert [r['date_min'] for r in ranges[1:]] == list(tool.BOUNDARIES)  # boundaries exactly at the split dates
    assert all(not any(r['date_min'] < b <= r['date_max'] for b in tool.BOUNDARIES) for r in ranges)
    original, rewritten = pq.read_table(source['path']), pq.read_table(source['output'])
    assert original.equals(rewritten) and original.schema.equals(rewritten.schema, check_metadata=True)
    manifest = json.loads(source['manifest'].read_text())
    assert manifest['result'] == 'PASS' and manifest['source']['sha256'] == source['sha']
    assert manifest['rewritten']['sha256'] == hashlib.sha256(source['output'].read_bytes()).hexdigest()
    assert manifest['rewritten']['rows'] == manifest['source']['rows'] == len(frame)
    assert set(manifest['identity']['columns']) == set(PROCESSED_COLUMNS) and 'byte-level' in manifest['disclosure']
    reader = ProcessedReader(source['output'], sha256=manifest['rewritten']['sha256'], log=ReadLog())
    assert reader.layout()['split_aligned']
    with pytest.raises(ReadRefused, match='not split-aligned'):  # the original layout is refused
        ProcessedReader(source['path'], sha256=source['sha'], log=ReadLog())


def change(table, name, row, value):
    values = table.column(name).to_pylist()  # test-only (synthetic data)
    values[row] = value
    return table.set_column(table.schema.get_field_index(name), table.schema.field(name),
                            pa.array(values, type=table.schema.field(name).type))


@pytest.mark.parametrize('damage', ['value', 'swap', 'missing', 'null'])
def test_the_identity_check_fails_on_a_tampered_row(source, tmp_path, damage):
    record = run(source)
    table = pq.read_table(source['output'])
    sizes = [r['rows'] for r in record['rewritten']['row_groups']]
    types = table.column('pitch_type').to_pylist()
    row = next(i for i in range(len(types) - 1) if types[i] and types[i + 1] and types[i] != types[i + 1])
    if damage == 'value':
        table = change(table, 'pitch_type', row, 'SL' if types[row] != 'SL' else 'CH')
    elif damage == 'swap':
        order = list(range(table.num_rows))
        order[row], order[row + 1] = order[row + 1], order[row]
        table = table.take(order)
    elif damage == 'missing':
        table = pa.concat_tables([table.slice(0, row), table.slice(row + 1)])
        sizes[-1] -= 1
    else:
        angles = table.column('launch_angle').to_pylist()
        table = change(table, 'launch_angle', next(i for i, a in enumerate(angles) if a == a and a is not None), None)
    tampered = rewrite_like(table, tmp_path / 'tampered.parquet', sizes)
    identity = tool.verify_identity(pq.ParquetFile(source['path']), pq.ParquetFile(tampered))
    assert identity['result'] == 'FAIL'
    if damage in ('value', 'swap'):
        assert 'pitch_type' in identity['differing_columns']
    elif damage == 'missing':
        assert identity['rows_equal'] is False
    else:
        assert identity['differing_columns'] == ['launch_angle']


def test_a_failed_check_leaves_no_usable_output(source, monkeypatch):
    real = tool.rewrite

    def faulty(original, target, plan, check):  # a copy that changes one value
        real(original, target, plan, check)
        table = pq.read_table(target)
        types = table.column('pitch_type').to_pylist()
        row = next(i for i, t in enumerate(types) if t)
        bad = change(table, 'pitch_type', row, 'SL' if types[row] != 'SL' else 'CH')
        rewrite_like(bad, target, [stop - start for _, start, stop in plan])
    monkeypatch.setattr(tool, 'rewrite', faulty)
    record = run(source)
    assert record['result'] == 'FAIL' and 'pitch_type' in record['identity']['differing_columns']
    assert not source['output'].exists() and source['output'].with_name(source['output'].name + '.failed').exists()
    assert json.loads(source['manifest'].read_text())['result'] == 'FAIL'
    cli = tool.main(['--source', str(source['path']), '--output', str(source['output'].with_name('again.parquet')),
                     '--source-sha256', source['sha']])
    assert cli == 1


def test_paths_pins_overwrites_order_and_the_window_are_refused(source, tmp_path, frame):
    for output, message in ((tmp_path / 'A3' / 'elsewhere.parquet', 'inputs/ folder'),
                            (tool.REPO / 'inputs' / 'x.parquet', 'inside a repository')):
        with pytest.raises(tool.PrepareRefused, match=message):
            tool.prepare(source['path'], output, source_sha256=source['sha'], now=lambda: AFTERNOON)
    with pytest.raises(tool.PrepareRefused, match='pinned processed sha256'):
        tool.prepare(source['path'], source['output'], source_sha256='0' * 64, now=lambda: AFTERNOON)
    with pytest.raises(tool.PrepareRefused, match='protection window'):
        run(source, now=lambda: MORNING)
    calls = {'n': 0}

    def now():  # the window begins while the tool runs: it stops and removes its partial output
        calls['n'] += 1
        return AFTERNOON if calls['n'] <= 6 else MORNING
    with pytest.raises(tool.PrepareRefused, match='protection window'):
        run(source, now=now)
    assert not any(source['output'].parent.iterdir())
    assert run(source)['result'] == 'PASS'
    with pytest.raises(tool.PrepareRefused, match='never overwrites'):
        run(source)
    shuffled = processed_order(frame).sample(frac=1, random_state=1)
    path = write_parquet(tmp_path / 'shuffled.parquet', shuffled, PROCESSED_COLUMNS, [0, len(shuffled)])
    with pytest.raises(tool.PrepareRefused, match='game_date order'):
        tool.prepare(path, tmp_path / 'A3' / 'inputs' / 'shuffled.parquet',
                     source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), now=lambda: AFTERNOON)


def test_the_tool_reads_no_label_column_into_python_objects(source):
    """Static: the tool's source never calls a value-materializing method, imports no pandas, and turns
    only game_date into numpy values. Runtime: a fresh interpreter prepares a synthetic input without
    ever importing pandas."""
    tree = ast.parse(Path(tool.__file__).read_text())
    forbidden = {'to_pylist', 'to_pydict', 'to_pandas', 'as_py', 'tolist', 'to_list', 'iterrows', 'itertuples',
                 'to_string', 'decode', 'to_numpy'}
    owners = {}
    for function in ast.walk(tree):
        if isinstance(function, ast.FunctionDef):
            for node in ast.walk(function):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    owners.setdefault(node.func.attr, set()).add(function.name)
    assert not forbidden & set(owners), forbidden & set(owners)
    assert owners['frombuffer'] == {'read_dates', '_bits', '_update'}  # game_date values; hashing byte views
    imported = {alias.name.split('.')[0] for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
                for alias in node.names} | {node.module.split('.')[0] for node in ast.walk(tree)
                                            if isinstance(node, ast.ImportFrom) and node.module}
    assert 'pandas' not in imported
    code = (f'import json, sys; sys.path[:0] = [{str(SCRIPTS)!r}, {str(PROJECT)!r}]\n'
            'from datetime import datetime\n'
            'import prepare_behavior_calibration_input as tool\n'
            f'record = tool.prepare({str(source["path"])!r}, {str(source["output"])!r}, source_sha256={source["sha"]!r},'
            ' now=lambda: datetime(2026, 10, 27, 15, 0, tzinfo=tool.KST))\n'
            "print(json.dumps({'result': record['result'], 'pandas': 'pandas' in sys.modules}))\n")
    done = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    assert json.loads(done.stdout.strip().splitlines()[-1]) == {'result': 'PASS', 'pandas': False}
