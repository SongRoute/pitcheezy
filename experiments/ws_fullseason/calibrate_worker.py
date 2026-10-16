"""Registered CPU-only pooled calibration worker; invoke via owned supervisor."""
import argparse
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import stat
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from experiments.ws_fullseason import calibrate as cal
from experiments.ws_fullseason.audit import gates, sha
from experiments.ws_fullseason.data import key_digest

ROW_CAP = 1_000_000
FILE_CAP = 128 << 20
TOTAL_CAP = 512 << 20
CORPUS = {'keys', 'y10', 'y7', 'games', 'fit', 'tuning', 'calibration',
          'evaluation', 'narrow_whiff', 'ground_ball_per_pitch'}
WINDOWS = {
    'july': {'fit': ['2026-03-25', '2026-06-30'], 'oof': ['2026-07-01', '2026-07-31']},
    'august': {'fit': ['2026-03-25', '2026-07-31'], 'oof': ['2026-08-01', '2026-08-31']},
    'september': {'fit': ['2026-03-25', '2026-08-31'], 'oof': ['2026-09-01', '2026-09-30']},
}
RECIPE = {'id': cal.SCHEMA, 'data_role': '2026 EXPOSED DEVELOPMENT',
          'folds': list(cal.FOLDS), 'seeds': list(cal.SEEDS), 'clip': cal.CLIP,
          'minimum_rows': cal.MIN_ROWS, 'bounds': list(cal.base.T_BOUNDS),
          'order': 'mean_raw_seeds_then_temperature', 'weighting': 'equal_pitch',
          'support_failure': 'identity', 'optimizer_failure': 'identity',
          'invalid_input': 'STOP', 'final_fit_recalibration': False}


def file_digest(stream):
    stream.seek(0); h = hashlib.sha256()
    for block in iter(lambda: stream.read(1 << 20), b''):
        h.update(block)
    stream.seek(0)
    return h.hexdigest()


def pinned(binding, stack, *, json_file=False):
    path = Path(binding['path']); size = binding['bytes']
    cap = (1 << 20) if json_file else FILE_CAP
    if not isinstance(size, int) or isinstance(size, bool) or not 0 < size <= cap:
        raise ValueError('input byte bound')
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError('regular nonsymlink input required')
    stream = stack.enter_context(path.open('rb'))
    before = os.fstat(stream.fileno())
    if not stat.S_ISREG(before.st_mode) or before.st_size != size or file_digest(stream) != binding['sha256']:
        raise ValueError('input byte custody')
    if json_file:
        value = json.loads(stream.read(cap + 1)); stream.seek(0)
        if not isinstance(value, dict): raise ValueError('object receipt required')
        return stream, value
    return stream, None


def header_check(stream, members, selected):
    """Reject array allocation claims before numpy reads any array body."""
    stream.seek(0)
    with zipfile.ZipFile(stream) as archive:
        infos = archive.infolist()
        if (len(infos) != len(members) or {i.filename for i in infos} != {n + '.npy' for n in members}
                or sum(i.file_size for i in infos) > FILE_CAP
                or any(i.compress_type != zipfile.ZIP_STORED or i.flag_bits & 1 for i in infos)):
            raise ValueError('bounded exact stored NPZ members required')
        for name, (shape, kind) in selected.items():
            info = archive.getinfo(name + '.npy')
            with archive.open(info) as entry:
                version = np.lib.format.read_magic(entry)
                readers = {(1, 0): np.lib.format.read_array_header_1_0,
                           (2, 0): np.lib.format.read_array_header_2_0}
                if version not in readers: raise ValueError('NPY header version')
                actual, fortran, dtype = readers[version](entry, max_header_size=10000)
                if (actual != shape or fortran or dtype.hasobject or dtype.kind not in kind
                        or dtype.itemsize not in (1, 2, 4, 8)
                        or entry.tell() + math.prod(actual) * dtype.itemsize != info.file_size):
                    raise ValueError('NPY shape/dtype/byte-length contract')
    stream.seek(0)


def load_windows(reg):
    """Only the explicit fold bindings are opened; no model or aggregate scores."""
    if set(reg['folds']) != set(cal.FOLDS): raise ValueError('exact fold bindings')
    windows = {}; bindings = []; decoded = {}
    with ExitStack() as stack:
        # Complete byte custody for every input before decoding any NPZ array.
        for fold in cal.FOLDS:
            spec = reg['folds'][fold]
            if set(spec['files']) != {'manifest', 'corpus', 'parent', 'acceptance', 'recipe', 'fit_grant'}:
                raise ValueError('exact fold file allowlist')
            if set(spec['seeds']) != {'0', '1', '2'}: raise ValueError('exact seed bindings')
            pairs = [(name, b, name != 'corpus') for name, b in spec['files'].items()]
            for seed in cal.SEEDS:
                group = spec['seeds'][str(seed)]
                if set(group) != {'receipt', 'supervisor', 'predictions'}: raise ValueError('exact seed file allowlist')
                pairs += [(f'{seed}:{name}', b, name != 'predictions') for name, b in group.items()]
            for name, binding, is_json in pairs:
                bindings.append(binding)
                if sum(b['bytes'] for b in bindings) > TOTAL_CAP: raise ValueError('total input byte cap')
                decoded[fold, name] = pinned(binding, stack, json_file=is_json)
        total_rows = 0
        for fold in cal.FOLDS:
            spec = reg['folds'][fold]; e = spec['expected']; files = spec['files']
            rows, nfit, noof = e['rows'], e['FIT'], e['OOF']
            if (any(type(n) is not int for n in (rows, nfit, noof))
                    or not 0 < nfit < rows <= ROW_CAP or nfit + noof != rows or noof <= 0):
                raise ValueError('registered row caps')
            total_rows += noof
            if total_rows > ROW_CAP: raise ValueError('pooled row cap')
            meta = lambda name: decoded[fold, name][1]
            manifest, parent, recipe, grant = (meta(n) for n in ('manifest', 'parent', 'recipe', 'fit_grant'))
            if (not meta('acceptance') or recipe['data_role'] != '2026 EXPOSED DEVELOPMENT'
                    or recipe['arm'] != 'T' or recipe['seeds'] != [0, 1, 2]
                    or recipe['training_rule']['epochs'] != 30
                    or recipe['window'] != WINDOWS[fold] or recipe['window'] != e['window']
                    or grant['commit'] != e['source_commit'] or grant['seeds'] != [0, 1, 2]
                    or grant['input_manifest_sha256'] != files['manifest']['sha256']
                    or grant['expected_rows'] != {'FIT': nfit, 'OOF': noof}
                    or grant['expected_key_sha256'] != e['key_sha256']
                    or manifest['status'] != 'COMPLETED' or manifest['recipe_sha256'] != files['recipe']['sha256']
                    or manifest['roles'] != {'FIT': nfit, 'OOF': noof}
                    or manifest['files']['corpus-subset.npz'] != {k: files['corpus'][k] for k in ('sha256', 'bytes')}
                    or parent['status'] != 'COMPLETED' or parent['fits'] != 3 or parent['calibration_fits'] != 0
                    or parent['grant_sha256'] != files['fit_grant']['sha256']
                    or [r['seed'] for r in parent['seeds']] != list(cal.SEEDS)):
                raise ValueError('accepted fold identity')
            corpus = decoded[fold, 'corpus'][0]
            header_check(corpus, CORPUS, {'keys': ((rows, 3), 'iu'), 'y7': ((rows,), 'iu'),
                                         'fit': ((rows,), 'b'), 'evaluation': ((rows,), 'b')})
            predictions = {}
            for seed, record in zip(cal.SEEDS, parent['seeds']):
                group = spec['seeds'][str(seed)]; receipt = meta(f'{seed}:receipt'); supervisor = meta(f'{seed}:supervisor')
                if (record['receipt_sha256'] != group['receipt']['sha256']
                        or record['supervisor_sha256'] != group['supervisor']['sha256']
                        or receipt['status'] != 'COMPLETED' or receipt['seed'] != seed
                        or receipt['source_commit'] != e['source_commit']
                        or receipt['grant_sha256'] != files['fit_grant']['sha256']
                        or receipt['recipe_sha256'] != files['recipe']['sha256']
                        or receipt['input_manifest_sha256'] != files['manifest']['sha256']
                        or receipt['outputs_sha256']['oof-predictions.npz'] != group['predictions']['sha256']
                        or receipt['OOF_rows'] != noof or receipt['OOF_keys_sha256'] != e['key_sha256']['OOF']
                        or receipt['epochs'] != 30 or receipt['temperature'] != 1 or receipt['calibration_fits'] != 0
                        or receipt['SELECT_queries'] != 0 or receipt['checkpoint'] != 'final epoch'
                        or supervisor['status'] != 'COMPLETED' or supervisor['child_reaped'] is not True
                        or supervisor['returncode'] != 0 or supervisor['receipt_sha256'] != group['receipt']['sha256']
                        or supervisor['grant_sha256'] != files['fit_grant']['sha256']):
                    raise ValueError('completed fixed seed custody')
                trace = receipt['consumption']
                if len(trace) != 30 or any(r['epoch'] != i or r['rows'] != nfit
                    or r['ordered_keys_sha256'] != e['key_sha256']['FIT'] or r['every_key_once'] is not True
                    or r['positive_parent_weight_every_row'] is not True for i, r in enumerate(trace, 1)):
                    raise ValueError('complete fixed FIT consumption')
                stream = decoded[fold, f'{seed}:predictions'][0]
                header_check(stream, {'keys', 'probabilities'}, {'keys': ((noof, 3), 'iu'), 'probabilities': ((noof, 7), 'f')})
            # Every header for this fold is now bounded; only selected members load.
            with np.load(corpus, allow_pickle=False) as c:
                keys, labels, fit, keep = c['keys'], c['y7'], c['fit'], c['evaluation']
            cal.checked_keys(keys)
            if fit.sum() != nfit or keep.sum() != noof or (fit & keep).any() or not (fit | keep).all():
                raise ValueError('complete disjoint registered roles')
            if any(key_digest(keys[m]) != e['key_sha256'][role] for m, role in ((fit, 'FIT'), (keep, 'OOF'))):
                raise ValueError('accepted role keys')
            for seed in cal.SEEDS:
                with np.load(decoded[fold, f'{seed}:predictions'][0], allow_pickle=False) as z:
                    predictions[seed] = (z['keys'], z['probabilities'])
            windows[fold] = {'keys': keys[keep], 'labels': labels[keep], 'seeds': predictions}
        # Same open file descriptions, so path replacement cannot redirect reads.
        for fold in cal.FOLDS:
            spec = reg['folds'][fold]
            pairs = list(spec['files'].items()) + [(f'{s}:{n}', b) for s, group in spec['seeds'].items() for n, b in group.items()]
            for name, b in pairs:
                stream = decoded[fold, name][0]
                if os.fstat(stream.fileno()).st_size != b['bytes'] or file_digest(stream) != b['sha256']:
                    raise ValueError('input changed during decode')
    return windows


def run(regpath, grantpath):
    reg, recipe = gates(regpath, grantpath)
    if recipe != RECIPE or recipe != reg['calibration_recipe']:
        raise ValueError('exact calibration recipe')
    if reg['accepted_folds'] != list(cal.FOLDS): raise ValueError('all folds independently accepted')
    bounds = reg['bounds']
    if (bounds['wall_seconds'] != 300 or bounds['rss_bytes'] != 2 << 30
            or bounds['free_memory_bytes'] != 4 << 30 or bounds['free_disk_bytes'] != 1 << 30
            or bounds['output_bytes'] != 64 << 10): raise ValueError('fixed CPU calibration bounds')
    out = Path(reg['output'])
    if not out.is_dir() or any(p.name != 'worker.log' for p in out.iterdir()):
        raise FileExistsError('fresh supervisor output required before calibration')
    os.fstat(int(os.environ['PITCHEEZY_FULLSEASON_LOCK_FD']))
    signal.signal(signal.SIGALRM, lambda *_: os._exit(124)); signal.alarm(315)
    result = cal.fit(load_windows(reg))
    for name, value in [('calibrator.json', result), ('receipt.pending.json', {
            'status': 'COMPLETED', 'stage': 'POOLED_CALIBRATION_NOT_RELEASE',
            'registration_sha256': sha(regpath), 'recipe_sha256': sha(reg['recipe']),
            'grant_sha256': sha(grantpath), 'data_role': recipe['data_role'],
            'calibrator': result, 'model_fits': 0, 'optimizer_calls': result['optimizer_calls'],
            'metrics_computed': False, 'independent_test': False})]:
        body = json.dumps(value, indent=2, allow_nan=False) + '\n'
        if len(body.encode()) > 16 << 10: raise ValueError('receipt byte cap')
        with (out / name).open('x') as stream: stream.write(body)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--registration', required=True); parser.add_argument('--grant', required=True)
    args = parser.parse_args(); run(args.registration, args.grant)
