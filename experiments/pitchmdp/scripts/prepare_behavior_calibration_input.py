"""One-time A3 input preparation (EXP-BC-CAL-01): a split-aligned, row-for-row identical copy of the
pinned processed parquet.

Why: before the gate the A3 reader may read guarded columns (pitch_type and the outcomes) only from
row groups that hold no DEV row. The pinned processed parquet was written with size-based row
groups, so one row group can hold TRAIN and DEV rows together. This tool rewrites it once, before
registration, into row groups whose boundaries are exactly the split dates of the design: TRAIN
start 2023-05-15, earlystop 2025-05-01, temperature 2025-05-16, blend (June) 2025-06-01, DEV
2025-07-01, and the end 2025-09-30. Every non-empty segment is one row group (rows before 2023-05-15
form their own segment). Row order, schema (with its metadata) and every value stay unchanged;
pyarrow only (pandas is never imported).

Identity proof: for every column, a canonical sha256 of its values in file order is computed on the
original and on the rewritten file (validity per value, fixed-width value bytes with null slots
zeroed, string lengths and string bytes; independent of how rows are chunked into row groups). With
equal row counts and equal schemas, equal digests for every column prove the copy row for row. The
identity hash necessarily streams the bytes of every column, pitch_type and the outcome columns
included: it is a byte-level identity check, not a label read. No value of any column is turned into
a Python object (pyarrow buffers and numpy byte views only); game_date is the only column read as
values (numpy datetime64), to place the boundaries. The copy moves Arrow tables from reader to writer.

Output: the rewritten parquet and an identity manifest (original sha256, rewritten sha256, row-group
date ranges, per-column digests, the result). Both must lie in an ``inputs/`` folder outside any
repository (the A3 output_root/inputs) and are never overwritten. A failed check renames the output
``*.failed``, records FAIL and exits 1. The tool refuses to start inside the recorder/publisher
protection window 06:30-13:30 KST and stops (removing its partial output) if the window begins.

Run once (PM, after code review passes, outside 06:30-13:30 KST; never by the A3 runner):
  /Users/song/Projects/pitcheezy/.venv/bin/python experiments/pitchmdp/scripts/prepare_behavior_calibration_input.py \
      --source "<artifact_root>/processed/pitches.parquet" \
      --output "/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/ML-MATRIX-20260924/EXP-BC-CAL-01/inputs/pitches.split-aligned.parquet"
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT))
from pitchmdp import behavior_calibration_registry as reg  # noqa: E402  (stdlib-only module)

KST = timezone(timedelta(hours=9))
BOUNDARIES = tuple(reg.A3_INPUT['boundaries'])
END = reg.A3_INPUT['end']
SEGMENTS = ('before_train', 'train', 'earlystop', 'temperature', 'blend', 'dev')
DISCLOSURE = ('the identity hash streams the bytes of every column, pitch_type and the outcome columns included: a '
              'byte-level identity check, not a label read; no value is turned into a Python object (pyarrow '
              'buffers and numpy byte views only); game_date is the only column read as values')


class PrepareRefused(Exception):
    """A precondition failed; nothing usable was written."""


def file_sha256(path, check=None):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while True:
            if check is not None:
                check()
            chunk = stream.read(8 << 20)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def _minutes(text):
    hours, minutes = (int(x) for x in text.split(':'))
    return hours * 60 + minutes


def window_check(now=None):
    """A callable that refuses inside the protection window (06:30-13:30 KST, every day)."""
    start, end = (_minutes(reg.PROTECTION_WINDOW_KST[k]) for k in ('start', 'end'))
    now = now or (lambda: datetime.now(KST))

    def check():
        t = now().astimezone(KST)
        if start <= t.hour * 60 + t.minute + t.second / 60 < end:
            raise PrepareRefused('inside the recorder/publisher protection window 06:30-13:30 KST')
    return check


def check_paths(source, output, manifest):
    source, output, manifest = (Path(p).expanduser().resolve() for p in (source, output, manifest))
    for path in (output, manifest):
        if path.parent.name != 'inputs':
            raise PrepareRefused(f'{path} must lie in an inputs/ folder (the A3 output_root/inputs)')
        if REPO.resolve() in path.parents or any((parent / '.git').exists() for parent in path.parents):
            raise PrepareRefused(f'{path} lies inside a repository')
        for name in (path, path.with_name(path.name + '.partial'), path.with_name(path.name + '.failed')):
            if name.exists():
                raise PrepareRefused(f'{name} exists; this tool never overwrites')
    if output == source or manifest == source or output == manifest:
        raise PrepareRefused('source, output and manifest must be three different files')
    return source, output, manifest


def read_dates(parquet):
    """game_date of every row in file order as datetime64[D] (the only column read as values; numpy
    views of the Arrow value buffers, so pyarrow's pandas-based conversion is never used)."""
    column = parquet.read(columns=['game_date']).column('game_date')
    kind = column.type
    if pa.types.is_date32(kind):
        dtype, unit = np.int32, 'D'
    elif pa.types.is_date64(kind):
        dtype, unit = np.int64, 'ms'
    elif pa.types.is_timestamp(kind) and kind.tz is None:
        dtype, unit = np.int64, kind.unit
    else:
        raise PrepareRefused(f'game_date must be a date or a naive timestamp, not {kind}')
    if column.null_count:
        raise PrepareRefused('game_date has null values')
    parts = [np.frombuffer(chunk.buffers()[1], dtype=dtype)[chunk.offset:chunk.offset + len(chunk)]
             for chunk in column.chunks]
    values = np.concatenate(parts) if parts else np.zeros(0, dtype=dtype)
    return values.astype(np.int64).view(f'datetime64[{unit}]').astype('datetime64[D]')


def plan_segments(dates):
    """(segment, first row, stop row) of every non-empty segment between the split dates."""
    if len(dates) > 1 and (np.diff(dates.astype(np.int64)) < 0).any():
        raise PrepareRefused('rows are not in game_date order; the order must stay unchanged')
    if len(dates) and dates.max() > np.datetime64(END):
        raise PrepareRefused(f'rows after {END} (or 2026 rows) found; stop')
    cuts = [0, *(int(np.searchsorted(dates, np.datetime64(b), side='left')) for b in BOUNDARIES), len(dates)]
    return [(name, a, b) for name, a, b in zip(SEGMENTS, cuts[:-1], cuts[1:]) if b > a]


def rewrite(source, target, plan, check):
    """Copy the source rows segment by segment, one row group per segment (Arrow tables only)."""
    meta = source.metadata
    offsets = np.cumsum([0, *(meta.row_group(i).num_rows for i in range(meta.num_row_groups))])
    codec = meta.row_group(0).column(0).compression.lower() if meta.num_row_groups else 'zstd'
    cached = (None, None)
    with pq.ParquetWriter(target, source.schema_arrow, compression=codec, write_statistics=True) as writer:
        for _, start, stop in plan:
            parts = []
            for g in range(meta.num_row_groups):
                lo, hi = int(offsets[g]), int(offsets[g + 1])
                if hi <= start or lo >= stop:
                    continue
                check()
                if cached[0] != g:
                    cached = (g, source.read_row_group(g))
                first, last = max(start, lo), min(stop, hi)
                parts.append(cached[1].slice(first - lo, last - first))
            segment = pa.concat_tables(parts)
            writer.write_table(segment, row_group_size=segment.num_rows)


def _bits(buffer, offset, n):
    return np.unpackbits(np.frombuffer(buffer, dtype=np.uint8), bitorder='little')[offset:offset + n]


def _update(array, streams):
    """Canonical bytes of one Arrow array (no value becomes a Python object)."""
    if pa.types.is_dictionary(array.type):
        array = array.dictionary_decode()
    n, kind, buffers = len(array), array.type, array.buffers()
    valid = _bits(buffers[0], array.offset, n) if buffers[0] is not None and array.null_count else \
        np.ones(n, dtype=np.uint8)
    streams[0].update(valid.tobytes())
    if pa.types.is_boolean(kind):
        streams[1].update((_bits(buffers[1], array.offset, n) & valid).tobytes())
    elif pa.types.is_string(kind) or pa.types.is_binary(kind) or pa.types.is_large_string(kind) \
            or pa.types.is_large_binary(kind):
        wide = pa.types.is_large_string(kind) or pa.types.is_large_binary(kind)
        offsets = np.frombuffer(buffers[1], dtype=np.int64 if wide else np.int32)[array.offset:array.offset + n + 1]
        offsets = offsets.astype(np.int64)
        streams[1].update((np.diff(offsets) * valid).astype('<i8').tobytes())
        data = memoryview(buffers[2]) if buffers[2] is not None else memoryview(b'')
        rows = np.flatnonzero(valid)
        if len(rows):
            breaks = np.flatnonzero(np.diff(rows) != 1)
            for first, last in zip(np.r_[rows[0], rows[breaks + 1]], np.r_[rows[breaks], rows[-1]]):
                streams[2].update(data[int(offsets[first]):int(offsets[last + 1])])
    elif pa.types.is_null(kind):
        return
    else:
        try:
            width = kind.bit_width
        except ValueError as error:
            raise PrepareRefused(f'unsupported column type {kind}') from error
        if width % 8 or pa.types.is_nested(kind):
            raise PrepareRefused(f'unsupported column type {kind}')
        width //= 8
        raw = np.frombuffer(buffers[1], dtype=np.uint8)[array.offset * width:(array.offset + n) * width]
        streams[1].update((raw.reshape(n, width) * valid[:, None]).astype(np.uint8).tobytes())


def column_digests(parquet, check):
    """{column: canonical sha256 of its values in file order} (byte-level; see the module docstring)."""
    out = {}
    for index, name in enumerate(parquet.schema_arrow.names):
        streams = [hashlib.sha256() for _ in range(3)]
        for g in range(parquet.metadata.num_row_groups):
            check()
            for chunk in parquet.read_row_group(g, columns=[name]).column(0).chunks:
                _update(chunk, streams)
        kind = str(parquet.schema_arrow.field(index).type).encode()
        out[name] = hashlib.sha256(kind + b'\0' + b''.join(s.digest() for s in streams)).hexdigest()
    return out


def schema_sha256(schema):
    return hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


def _raw_day(value, kind):
    """A raw (physical) game_date statistic as YYYY-MM-DD, without converting through Python scalars."""
    if pa.types.is_date32(kind):
        return str(np.datetime64(int(value), 'D'))
    if pa.types.is_date64(kind):
        return str(np.datetime64(int(value), 'ms').astype('datetime64[D]'))
    return str(np.datetime64(int(value), kind.unit).astype('datetime64[D]'))


def row_group_ranges(parquet):
    """Rows and footer game_date range of every row group (raw statistics; nothing decoded)."""
    meta, kind = parquet.metadata, parquet.schema_arrow.field('game_date').type
    names = [meta.schema.column(j).path for j in range(meta.num_columns)]
    index, out = names.index('game_date'), []
    for i in range(meta.num_row_groups):
        stats = meta.row_group(i).column(index).statistics
        has = stats is not None and stats.has_min_max
        out.append({'row_group': i, 'rows': meta.row_group(i).num_rows,
                    'date_min': _raw_day(stats.min_raw, kind) if has else None,
                    'date_max': _raw_day(stats.max_raw, kind) if has else None})
    return out


def aligned(ranges, plan):
    """Every row group is exactly one planned segment and no row group spans a split date."""
    if [r['rows'] for r in ranges] != [stop - start for _, start, stop in plan]:
        return False
    for r in ranges:
        if r['date_min'] is None or r['date_max'] is None:
            return False
        if any(r['date_min'] < b <= r['date_max'] for b in BOUNDARIES) or r['date_max'] > END:
            return False
    return True


def verify_identity(original, rewritten, check=lambda: None):
    """Row-for-row identity of two parquet files: rows, schema (with metadata), column digests."""
    rows = original.metadata.num_rows == rewritten.metadata.num_rows
    schema = original.schema_arrow.equals(rewritten.schema_arrow, check_metadata=True)
    left, right = column_digests(original, check), column_digests(rewritten, check)
    columns = {name: {'original': left.get(name), 'rewritten': right.get(name),
                      'equal': left.get(name) == right.get(name)} for name in sorted(set(left) | set(right))}
    passed = rows and schema and all(c['equal'] for c in columns.values())
    return {'rows_equal': rows, 'schema_equal': schema, 'columns': columns,
            'differing_columns': [n for n, c in columns.items() if not c['equal']],
            'result': 'PASS' if passed else 'FAIL'}


def prepare(source, output, manifest=None, *, source_sha256=reg.DATA_PINS['processed_sha256'], now=None):
    """Rewrite ``source`` into split-aligned row groups at ``output``, prove identity, write the
    manifest. Returns the manifest (result PASS or FAIL)."""
    manifest = manifest or Path(output).with_name(Path(output).stem + '.identity.json')
    source, output, manifest = check_paths(source, output, manifest)
    check = window_check(now)
    check()
    if file_sha256(source, check) != source_sha256:
        raise PrepareRefused('the source bytes differ from the pinned processed sha256')
    original = pq.ParquetFile(source)
    plan = plan_segments(read_dates(original))
    partial = output.with_name(output.name + '.partial')
    output.parent.mkdir(parents=False, exist_ok=True)
    try:
        rewrite(original, partial, plan, check)
        rewritten = pq.ParquetFile(partial)
        ranges = row_group_ranges(rewritten)
        identity = verify_identity(original, rewritten, check)
        if file_sha256(source, check) != source_sha256:
            raise PrepareRefused('the source bytes changed while the tool ran')
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    is_aligned = aligned(ranges, plan)
    result = 'PASS' if identity['result'] == 'PASS' and is_aligned else 'FAIL'
    final = output if result == 'PASS' else output.with_name(output.name + '.failed')
    partial.rename(final)
    record = {'experiment': reg.EXPERIMENT_ID, 'tool': 'experiments/pitchmdp/scripts/prepare_behavior_calibration_input.py',
              'tool_sha256': file_sha256(Path(__file__)), 'created_utc': datetime.now(timezone.utc).isoformat(),
              'python': platform.python_version(), 'pyarrow': pa.__version__,
              'source': {'path': str(source), 'sha256': source_sha256, 'bytes': source.stat().st_size,
                         'rows': original.metadata.num_rows, 'row_groups': row_group_ranges(original),
                         'schema_sha256': schema_sha256(original.schema_arrow)},
              'rewritten': {'path': str(final), 'sha256': file_sha256(final), 'bytes': final.stat().st_size,
                            'rows': rewritten.metadata.num_rows, 'row_groups': ranges,
                            'schema_sha256': schema_sha256(rewritten.schema_arrow)},
              'boundaries': list(BOUNDARIES), 'end': END,
              'segments': [{'segment': name, 'first_row': start, 'stop_row': stop} for name, start, stop in plan],
              'aligned': is_aligned, 'identity': identity, 'result': result, 'disclosure': DISCLOSURE}
    manifest.write_text(json.dumps(record, indent=1, sort_keys=True) + '\n')
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--manifest')
    parser.add_argument('--source-sha256', default=reg.DATA_PINS['processed_sha256'])
    args = parser.parse_args(argv)
    try:
        record = prepare(args.source, args.output, args.manifest, source_sha256=args.source_sha256)
    except PrepareRefused as refusal:
        print(json.dumps({'refused': str(refusal)}), file=sys.stderr)
        return 2
    print(json.dumps({'result': record['result'], 'rewritten_sha256': record['rewritten']['sha256'],
                      'row_groups': record['rewritten']['row_groups']}))
    return 0 if record['result'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
