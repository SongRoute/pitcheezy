"""Hash-pinned `batter` id sidecar for the frozen CUDA direct export 49097109 (EXP-P43-001 DRAFT).

The export carries `pitcher` per row but no `batter`.  The batter of a pitch is an identity fact of the row (who stood
at the plate), not a statistic, so it is joined by the export's own ordered `keys` (game_pk, at_bat_number,
pitch_number) to the pinned raw Statcast files.  Only id columns are read from raw (RAW_COLUMNS: keys, batter, pitcher);
no outcome, physics, date or text column.  Every export row must match exactly one raw row and the raw `pitcher` must
equal the export `pitcher` on every row (a row-level integrity check of the join).  Raw files must be named
statcast_YYYY.parquet with YYYY <= 2025 and no path part may contain `holdout`; anything else is refused before it is
opened (the snapshot directory name itself carries a 2026 collection date, so the season is read from the file name).

Layout (a NEW sibling directory of the export; the read-only export is never touched):
  <dir>/sidecar.json            format, export manifest sha, per split: n, keys rows_sha256 (== export manifest),
                                batter.npy sha256, distinct batters, pitcher_mismatch 0; raw file names + sha256
  <dir>/<split>/batter.npy      int64 [n], same row order as the export split
The TRAIN-only vocabulary is built later by the runner from TRAIN rows (direct_pitcher_id_registry.build_vocab); this
module fits nothing.  numpy only at import; pyarrow is imported inside build() only.
"""
from __future__ import annotations

import hashlib
import re
import json
from pathlib import Path

import numpy as np

FORMAT = 'cuda_direct_batter_sidecar_v1'
COLUMN = 'batter'
RAW_COLUMNS = ('game_pk', 'at_bat_number', 'pitch_number', 'batter', 'pitcher')  # ids only
SPLITS = ('train', 'earlystop', 'temperature', 'blend', 'dev', 'mlb_dev')
RAW_NAME = re.compile(r'statcast_(\d{4})\.parquet')
LAST_SEASON = 2025
AB_LIMIT, PITCH_LIMIT = 1000, 100  # combined key = (game_pk * AB_LIMIT + at_bat) * PITCH_LIMIT + pitch (exact int64)


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def rows_sha256(keys):
    return hashlib.sha256(np.ascontiguousarray(keys, dtype=np.int64).tobytes()).hexdigest()


def combined_key(game_pk, at_bat, pitch):
    game_pk, at_bat, pitch = (np.asarray(a, dtype=np.int64) for a in (game_pk, at_bat, pitch))
    if (game_pk < 0).any() or game_pk.max(initial=0) >= (1 << 62) // (AB_LIMIT * PITCH_LIMIT):
        raise ValueError('game_pk outside the exact combined-key range')
    if (at_bat < 0).any() or (at_bat >= AB_LIMIT).any() or (pitch < 0).any() or (pitch >= PITCH_LIMIT).any():
        raise ValueError('at_bat_number / pitch_number outside the exact combined-key range')
    return (game_pk * AB_LIMIT + at_bat) * PITCH_LIMIT + pitch


def join(keys, pitcher, raw):
    """Batter per export row.  `raw` = dict of RAW_COLUMNS arrays.  Fails closed on duplicate raw keys, a missing row
    or any raw/export pitcher mismatch."""
    keys = np.asarray(keys, dtype=np.int64)
    if keys.ndim != 2 or keys.shape[1] != 3:
        raise ValueError('Export keys must be [n, 3] (game_pk, at_bat_number, pitch_number)')
    raw_key = combined_key(raw['game_pk'], raw['at_bat_number'], raw['pitch_number'])
    order = np.argsort(raw_key, kind='stable')
    raw_key = raw_key[order]
    if len(raw_key) > 1 and np.any(raw_key[1:] == raw_key[:-1]):
        raise ValueError('Raw Statcast has duplicate (game_pk, at_bat_number, pitch_number) keys')
    want = combined_key(keys[:, 0], keys[:, 1], keys[:, 2])
    pos = np.searchsorted(raw_key, want)
    hit = (pos < len(raw_key)) & (raw_key[np.minimum(pos, len(raw_key) - 1)] == want)
    if not hit.all():
        raise ValueError(f'{int((~hit).sum())} export rows have no raw Statcast row')
    source = order[pos]
    raw_pitcher = np.asarray(raw['pitcher'], dtype=np.int64)[source]
    mismatch = int((raw_pitcher != np.asarray(pitcher, dtype=np.int64)).sum())
    if mismatch:
        raise ValueError(f'{mismatch} rows: raw pitcher differs from the export pitcher (join is not row-exact)')
    batter = np.asarray(raw['batter'], dtype=np.int64)[source]
    if (batter <= 0).any():
        raise ValueError('Non-positive batter id')
    return batter


def season_of(name):
    match = RAW_NAME.fullmatch(Path(name).name)
    if not match or int(match.group(1)) > LAST_SEASON:
        raise ValueError(f'Refused raw file (not statcast_YYYY.parquet with YYYY <= {LAST_SEASON}): {name}')
    return int(match.group(1))


def check_raw_names(raw_files):
    paths = [Path(p) for p in raw_files]
    for path in paths:
        if any('holdout' in part.lower() for part in path.parts):
            raise ValueError(f'Refused raw file (holdout path): {path}')
        season_of(path)
    if len({p.name for p in paths}) != len(paths):
        raise ValueError('Duplicate raw season file')
    return paths


def read_raw(raw_files):
    """Only RAW_COLUMNS from each (name-checked) parquet file."""
    import pyarrow.parquet as pq
    parts = [pq.read_table(path, columns=list(RAW_COLUMNS)) for path in check_raw_names(raw_files)]
    return {c: np.concatenate([np.asarray(t.column(c).to_numpy(), dtype=np.int64) for t in parts]) for c in RAW_COLUMNS}


def build(export, manifest_sha256, raw_files, out):
    """Write the sidecar into the NEW directory `out` (refuses an existing one)."""
    export, out = Path(export), Path(out)
    raw_files = check_raw_names(raw_files)
    manifest_path = export / 'manifest.json'
    if sha256_file(manifest_path) != manifest_sha256:
        raise ValueError('Export manifest differs from the pinned sha')
    manifest = json.loads(manifest_path.read_text())
    if out.exists():
        raise FileExistsError(f'{out} exists; sidecars are never overwritten')
    raw = read_raw(raw_files)
    work = out.with_name(f'.{out.name}.tmp')
    work.mkdir(parents=True)
    record = {'format': FORMAT, 'column': COLUMN, 'export_manifest_sha256': manifest_sha256,
              'raw_columns_read': list(RAW_COLUMNS),
              'raw_files': {p.name: sha256_file(p) for p in raw_files}, 'splits': {}}
    for split in SPLITS:
        files = manifest['splits'][split]['files']
        keys = np.load(export / files['keys']['path'], allow_pickle=False)
        pitcher = np.load(export / files['pitcher']['path'], allow_pickle=False)
        if rows_sha256(keys) != manifest['splits'][split]['rows_sha256']:
            raise ValueError(f'{split}: export keys differ from the manifest rows hash')
        batter = join(keys, pitcher, raw)
        (work / split).mkdir()
        np.save(work / split / 'batter.npy', batter)
        record['splits'][split] = {'n': int(len(batter)), 'rows_sha256': manifest['splits'][split]['rows_sha256'],
                                   'path': f'{split}/batter.npy', 'sha256': sha256_file(work / split / 'batter.npy'),
                                   'distinct_batters': int(len(np.unique(batter))), 'pitcher_mismatch': 0}
    (work / 'sidecar.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    work.rename(out)
    return record, sha256_file(out / 'sidecar.json')


def verify(directory, manifest, sidecar_sha256, hash_arrays=True):
    """The sidecar record after pin checks: sidecar.json sha, export manifest link, per-split rows hash and n."""
    directory = Path(directory)
    path = directory / 'sidecar.json'
    if not path.is_file() or sha256_file(path) != sidecar_sha256:
        raise ValueError('Batter sidecar record differs from the config pin')
    record = json.loads(path.read_text())
    if record.get('format') != FORMAT or record.get('column') != COLUMN or set(record['splits']) != set(SPLITS):
        raise ValueError('Batter sidecar format / splits differ')
    if set(record['raw_columns_read']) - set(RAW_COLUMNS):
        raise ValueError('Batter sidecar read a non-id raw column')
    for name in record['raw_files']:
        season_of(name)  # refuses a 2026 (or unnamed) season file
    for split, one in record['splits'].items():
        if one['rows_sha256'] != manifest['splits'][split]['rows_sha256'] or one['n'] != manifest['splits'][split]['n']:
            raise ValueError(f'{split}: batter sidecar rows differ from the export split')
        if one.get('pitcher_mismatch') != 0:
            raise ValueError(f'{split}: batter sidecar join was not row-exact')
        if hash_arrays and sha256_file(directory / one['path']) != one['sha256']:
            raise ValueError(f'{split}: batter sidecar array hash differs')
    return record


def load(directory, record, split, limit=None):
    """`batter` int64 for a split (first `limit` rows when given, matching runner.load_split)."""
    one = record['splits'][split]
    value = np.load(Path(directory) / one['path'], mmap_mode='r', allow_pickle=False)
    if value.dtype != np.int64 or value.shape != (one['n'],):
        raise ValueError(f'{split}: batter sidecar shape or dtype differs')
    return value[:limit] if limit else value


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--export', required=True)
    p.add_argument('--export-manifest-sha256', required=True)
    p.add_argument('--raw', nargs='+', required=True, help='pinned raw Statcast parquet files (<= 2025 only)')
    p.add_argument('--out', required=True, help='NEW sibling directory of the export')
    args = p.parse_args(argv)
    record, digest = build(args.export, args.export_manifest_sha256, args.raw, args.out)
    print('BATTER_SIDECAR', digest, json.dumps({s: {k: v[k] for k in ('n', 'distinct_batters')}
                                                for s, v in record['splits'].items()}))


if __name__ == '__main__':
    main()
