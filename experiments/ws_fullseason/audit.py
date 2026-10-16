"""Registered first-window quality audit. Invoke only through the reviewed supervisor."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.ws_fullseason import data


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def checked_json(path, cap=1 << 20):
    path = Path(path)
    if path.stat().st_size > cap:
        raise ValueError('JSON byte cap')
    return json.loads(path.read_text())


def gates(regpath, grantpath):
    reg, grant = checked_json(regpath), checked_json(grantpath)
    if not (grant.get('execute') is True and grant.get('verdict') == 'PASS'
            and grant.get('registration_sha256') == sha(regpath)
            and grant.get('author') == reg.get('author') and reg.get('author')
            and grant.get('reviewer') != reg['author'] and grant.get('reviewer')
            and grant.get('authorized_by') == 'root'):
        raise ValueError('independent Root grant required')
    for path, digest in reg['source_pins'].items():
        if sha(path) != digest:
            raise ValueError('source pin mismatch')
    recipe = checked_json(reg['recipe'])
    if recipe['data_role'] != '2026 EXPOSED DEVELOPMENT':
        raise ValueError('data role')
    return reg, recipe


def run(regpath, grantpath):
    reg, recipe = gates(regpath, grantpath)
    fd = int(os.environ['PITCHEEZY_FULLSEASON_LOCK_FD'])
    os.fstat(fd)  # inherited same lock description; parent death cannot release it
    signal.signal(signal.SIGALRM, lambda *_: os._exit(124))
    signal.alarm(reg['bounds']['wall_seconds'] + 15)
    import pandas as pd
    import pyarrow.parquet as pq
    M = data.closure()
    source = reg['source']
    for field, cap in [('manifest', reg['bounds']['manifest_bytes']), ('parquet', reg['bounds']['parquet_bytes'])]:
        p = Path(source[field])
        if not stat.S_ISREG(p.lstat().st_mode) or p.stat().st_size > cap or sha(p) != source[field + '_sha256']:
            raise ValueError('input byte cap/hash ' + field)
    manifest = checked_json(source['manifest'], reg['bounds']['manifest_bytes'])
    M.cp.check_manifest(manifest, source['manifest_binding'], source['parquet'], source['manifest'])
    pf = pq.ParquetFile(source['parquet'])
    if pf.metadata.num_rows != source['rows'] or len(pf.schema_arrow.names) != source['manifest_binding']['columns']:
        raise ValueError('footer mismatch')
    if M.cp.PROJECT != reg['columns']:
        raise ValueError('projection allowlist mismatch')
    raw = pq.read_table(source['parquet'], columns=reg['columns'], filters=[
        ('game_date', '>=', pd.Timestamp(recipe['window']['fit'][0])),
        ('game_date', '<=', pd.Timestamp(recipe['window']['oof'][1]))], use_threads=False).to_pandas()
    g, census, fit, oof, y7 = data.project(raw, recipe, M)
    result = data.audit_summary(g, census, fit, oof, y7, M)
    result.update(status='COMPLETED', stage='QUALITY_AUDIT_NOT_FIT_READY',
        data_role=recipe['data_role'], source_sha256=source['parquet_sha256'],
        registration_sha256=sha(regpath), recipe_sha256=sha(reg['recipe']),
        bytes_hash_scope='whole pinned S2 file', decoded_columns=reg['columns'],
        decoded_window=recipe['window'], original_arrival_time='UNKNOWN',
        snapshot_mode='corrected retrospective reconstruction; not prospective',
        learned_assets_produced=0, fits=0, raw_rows_output=0)
    encoded = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if len(encoded.encode()) > reg['bounds']['output_bytes']:
        raise ValueError('output byte cap')
    with (Path(reg['output']) / 'receipt.pending.json').open('x') as f:
        f.write(encoded)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--registration', required=True); p.add_argument('--grant', required=True)
    a = p.parse_args()
    run(a.registration, a.grant)
