"""Separate C experiment: immutable T producers/helpers, no old grant bypass.

The DRAFT gate precedes payload I/O and torch import. A reviewed phase grant pins
all real paths; outputs are fresh, and each of the twelve fits has one shared
exclusive task claim. Scientific primitives remain in their accepted modules.
"""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import importlib
import io
import json
import os
import plistlib
import re
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
THREAD_ENV = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')
FOLDS = ('july', 'august', 'september')
SEEDS = (0, 1, 2)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def document(path):
    p = Path(path)
    if not stat.S_ISREG(p.lstat().st_mode) or p.stat().st_size > 2 << 20: raise ValueError('bounded regular JSON')
    return json.loads(p.read_text())


def binding(meta):
    return {'path': meta['path'], 'bytes': meta.get('recorded_bytes', meta.get('bytes')), 'sha256': meta.get('recorded_sha256', meta.get('sha256'))}


def verify_file(b, cap=1 << 30):
    p = Path(b['path'])
    if (type(b['bytes']) is not int or not 0 < b['bytes'] <= cap or not stat.S_ISREG(p.lstat().st_mode)
            or p.stat().st_size != b['bytes'] or sha(p) != b['sha256']): raise ValueError('input byte custody: ' + str(p))
    return p


def verify_sources(cfg, manifest_path=None):
    p = Path(manifest_path or cfg['source_manifest']['path'])
    if sha(p) != cfg['source_manifest']['sha256']: raise ValueError('source manifest pin')
    pins = document(p)['files']
    for rel, digest in pins.items():
        path = (ROOT / rel).resolve()
        if not path.is_relative_to(ROOT) or sha(path) != digest: raise ValueError('source pin: ' + rel)
    for name, module in tuple(sys.modules.items()):
        if name.startswith(('experiments.ws_', 'pitcheezy.seq_shadow', 'scripts.fullseason_')) or name in ('first_small_features','history_transfer','outcome_transfer','scale_safe','abc_corpus','scale_runner'):
            path = getattr(module, '__file__', None)
            if path and (not Path(path).resolve().is_relative_to(ROOT) or str(Path(path).resolve().relative_to(ROOT)) not in pins): raise ValueError('foreign/unpinned same-name scientific module')
    return pins


def gates(config_path, grant_path, grant_sha, task):
    cfg = document(config_path)
    # Do not move this below source, grant, metadata or scientific imports.
    if cfg.get('status') != 'REGISTERED' or cfg.get('execute') is not True: raise ValueError('DRAFT: real I/O disabled')
    if sha(grant_path) != grant_sha: raise ValueError('exact independently reviewed grant')
    g = document(grant_path)
    if (g.get('execute') is not True or g.get('verdict') != 'PASS' or g.get('authorized_by') != 'root'
            or g.get('author') != cfg['author'] or not g.get('reviewer') or g['reviewer'] == g['author']
            or g.get('config_sha256') != sha(config_path)): raise ValueError('independent phase grant')
    if task not in g['tasks']: raise ValueError('task absent from reviewed grant')
    g = dict(g, **g['tasks'][task], task=task)
    for name in ('output', 'output_root'):
        value = g[name]
        if '${GPU_JOB_DIR}' in value:
            if not os.environ.get('GPU_JOB_DIR'): raise ValueError('gpu run job directory absent')
            value = value.replace('${GPU_JOB_DIR}', os.environ['GPU_JOB_DIR'])
        if '$' in value or '~' in value: raise ValueError('unregistered path expansion')
        g[name] = value
    jobs = {j['id']: j for j in cfg['jobs']}
    if task in jobs:
        if jobs[task]['enabled'] is not True: raise ValueError('fit task disabled')
        recipe = cfg['folds'][jobs[task]['fold']]['recipe']
        if recipe['arm'] != 'C' or recipe['seeds'] != list(SEEDS) or recipe['training_rule']['epochs'] != 30: raise ValueError('fixed C30')
    elif task not in cfg['phases'] or cfg['phases'][task] is not True: raise ValueError('phase disabled')
    if cfg['arm'] != 'C' or cfg['data_role'] != '2026 EXPOSED DEVELOPMENT' or cfg['fit_ceiling'] != 12: raise ValueError('study identity')
    verify_sources(cfg, g['source_manifest_path'])
    if sha(g['input_metadata_path']) != cfg['input_metadata']['sha256']: raise ValueError('input metadata pin')
    if any(os.environ.get(k) != '1' for k in THREAD_ENV) or os.environ.get('PYTHONPATH') or os.environ.get('PYTHONHOME'): raise ValueError('fixed thread/import environment')
    fit = task in jobs; limits = cfg['limits']
    if (g['rss_bytes'] != limits['rss_bytes'] or g['wall_seconds'] != limits['fit_wall_seconds' if fit else 'cpu_wall_seconds'] or g['output_bytes'] != limits['output_bytes_per_fit' if fit else 'cpu_output_bytes']): raise ValueError('fixed phase resource caps')
    if g['free_disk_bytes'] != cfg['resource_admission']['remote_disk_free_bytes' if fit else 'local_T7_free_bytes']: raise ValueError('fixed output headroom')
    if fit and g['free_memory_bytes'] != cfg['resource_admission']['remote_host_available_bytes']: raise ValueError('fixed host headroom')
    if fit:
        if g['min_free_mib'] != 20480: raise ValueError('fixed GPU free-memory threshold')
        devices = g['gpu_candidates']
        if len(devices) != 5 or len({v['uuid'] for v in devices}) != 5 or {v['index'] for v in devices} != set(range(5)): raise ValueError('registered five-device allowlist')
        selected = [v for v in devices if str(v['index']) == os.environ.get('CUDA_VISIBLE_DEVICES')]
        if len(selected) != 1: raise ValueError('gpu run one allowed numeric index')
        g.update(gpu_index=selected[0]['index'], gpu_uuid=selected[0]['uuid'], lock=str(Path(g['gpu_lock_root']) / (selected[0]['uuid'] + '.lock')))
    return cfg, g, document(g['input_metadata_path'])


def same_recipe(cfg, fold):
    spec = cfg['folds'][fold]; p = ROOT / spec['producer_recipe']
    if sha(p) != spec['producer_recipe_sha256']: raise ValueError('T producer recipe pin')
    producer = document(p); c = dict(spec['recipe']); t = dict(producer)
    for value in (c, t):
        for k in ('id', 'arm', 'selection_basis', 'budgets_proposed'): value.pop(k, None)
    if c != t: raise ValueError('C must retain exact T data/training recipe')
    return spec['recipe']


def package_binding(meta, fold, grant):
    spec = meta['training']['packages'][fold]
    root = Path(grant['package_root'])
    files = {Path(v['path']).name: binding(v) for v in spec['files']}
    man_binding = binding(spec['manifest']); man_binding['path'] = str(root / 'manifest.json')
    manifest = document(verify_file(man_binding, 1 << 20))
    if (manifest['recipe_sha256'] != spec['producer_recipe']['recorded_sha256'] or manifest['roles'] != spec['roles']
            or set(manifest['files']) != set(files) or {p.name for p in root.iterdir()} != set(files) | {'manifest.json'}): raise ValueError('exact producer package')
    for name, b in files.items():
        if {k: b[k] for k in ('bytes', 'sha256')} != manifest['files'][name]: raise ValueError('producer file binding')
        verify_file(dict(b, path=str(root / name)))
    return root, spec


def fit_worker(cfg, g, meta, out):
    job = next(j for j in cfg['jobs'] if j['id'] == g['task']); recipe = same_recipe(cfg, job['fold'])
    inputs, spec = package_binding(meta, job['fold'], g)
    import numpy as np
    import torch
    from experiments.ws_seq import train_seq as base
    from experiments.ws_fullseason.data import key_digest
    from experiments.ws_fullseason_c import train
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1: raise ValueError('one CUDA device')
    torch.cuda.set_per_process_memory_fraction(cfg['limits']['vram_bytes'] / torch.cuda.get_device_properties(0).total_memory)
    package = base.load_package(inputs, recipe['base']); _, corpus, _ = package
    for role, field in (('FIT', 'fit'), ('OOF', 'evaluation')):
        if int(corpus[field].sum()) != spec['roles'][role] or key_digest(corpus['keys'][corpus[field]]) != spec['expected']['key_sha256'][role]: raise ValueError('fixed role keys')
    result = train.run(recipe, job['seed'], package, out)
    names = ['model-state.pt'] + ([] if job['fold'] == 'final' else ['oof-predictions.npz'])
    result.update(task=job['id'], fold=job['fold'], model_recipe_sha256=object_sha(recipe), producer_recipe_sha256=spec['producer_recipe']['recorded_sha256'], input_manifest_sha256=spec['manifest']['recorded_sha256'], outputs_sha256={n: sha(out / n) for n in names}, gpu_max_reserved_bytes=torch.cuda.max_memory_reserved(), gpu_index=g['gpu_index'], gpu_uuid=g['gpu_uuid'])
    if result['gpu_max_reserved_bytes'] > cfg['limits']['vram_bytes']: raise ValueError('VRAM cap')
    package_binding(meta, job['fold'], g)  # Recheck immutable mmap inputs after the fit.
    return result


def completed_C(record, cfg, fold, seed, output_name):
    receipt = document(verify_file(binding(record['receipt']), 1 << 20))
    supervisor = document(verify_file(binding(record['supervisor']), 1 << 20))
    spec = cfg['folds'][fold]
    if (receipt.get('status') != 'COMPLETED' or receipt.get('arm') != 'C' or receipt.get('seed') != seed
            or receipt.get('fold') != fold or receipt.get('task') != f'C-{fold}-s{seed}'
            or receipt.get('model_recipe_sha256') != object_sha(spec['recipe'])
            or receipt.get('producer_recipe_sha256') != spec['producer_recipe_sha256']
            or supervisor.get('status') != 'COMPLETED' or supervisor.get('returncode') != 0
            or supervisor.get('child_reaped') is not True
            or supervisor.get('receipt_sha256') != binding(record['receipt'])['sha256']
            or supervisor.get('grant_sha256') != receipt.get('grant_sha256')
            or receipt.get('outputs_sha256', {}).get(output_name) != binding(record['model' if fold == 'final' else 'predictions'])['sha256']):
        raise ValueError('accepted C fit receipt chain')
    return receipt


def own_calibrator(g, meta):
    record = g['C_calibration']; receipt = document(verify_file(record['receipt'], 1 << 20))
    supervisor = document(verify_file(record['supervisor'], 1 << 20))
    if (receipt.get('status') != 'COMPLETED' or receipt.get('task') != 'calibrate'
            or receipt.get('outputs_sha256', {}).get('calibrator.json') != record['calibrator']['sha256']
            or supervisor.get('status') != 'COMPLETED' or supervisor.get('child_reaped') is not True
            or supervisor.get('returncode') != 0 or supervisor.get('receipt_sha256') != record['receipt']['sha256']
            or supervisor.get('grant_sha256') != receipt.get('grant_sha256')): raise ValueError('own C calibration completion chain')
    cal = document(verify_file(record['calibrator'], 1 << 20))
    if cal != receipt['calibrator'] or cal['rows'] != sum(meta['training']['packages'][f]['roles']['OOF'] for f in FOLDS): raise ValueError('complete C pooled calibration')
    return cal


def load_windows(meta, records, arm, cfg=None):
    """Exact accepted labels and all 3 seeds; no intersection/drop or T fit reuse."""
    import numpy as np
    from experiments.ws_fullseason import calibrate as cal
    from experiments.ws_fullseason.calibrate_worker import pinned, header_check, CORPUS, file_digest
    from experiments.ws_fullseason.data import key_digest
    if {(r['fold'], r['seed']) for r in records} != {(f, s) for f in FOLDS for s in SEEDS} or len(records) != 9: raise ValueError('exact nine OOF predictions')
    windows = {}
    for fold in FOLDS:
        spec = meta['training']['packages'][fold]; expected = spec['expected']; n = expected['rows']; noof = expected['OOF']
        corpus_b = next(binding(v) for v in spec['files'] if Path(v['path']).name == 'corpus-subset.npz')
        with ExitStack() as stack:
            stream, _ = pinned(corpus_b, stack)
            header_check(stream, CORPUS, {'keys': ((n, 3), 'iu'), 'y7': ((n,), 'iu'), 'fit': ((n,), 'b'), 'evaluation': ((n,), 'b')})
            with np.load(stream, allow_pickle=False) as z: keys, labels, fit, keep = (z[k] for k in ('keys', 'y7', 'fit', 'evaluation'))
            if int(fit.sum()) != expected['FIT'] or int(keep.sum()) != noof or (fit & keep).any() or not (fit | keep).all(): raise ValueError('full fixed roles')
            if any(key_digest(keys[m]) != expected['key_sha256'][r] for m, r in ((fit, 'FIT'), (keep, 'OOF'))): raise ValueError('fixed corpus keys')
            predictions = {}
            for record in [r for r in records if r['fold'] == fold]:
                if arm == 'C': completed_C(record, cfg, fold, record['seed'], 'oof-predictions.npz')
                receipt_b = binding(record['receipt']); pred_b = binding(record['predictions'])
                _, receipt = pinned(receipt_b, stack, json_file=True)
                if (receipt['status'] != 'COMPLETED' or receipt['arm'] != arm or receipt['seed'] != record['seed']
                        or receipt['epochs'] != 30 or receipt['calibration_fits'] != 0 or receipt['temperature'] != 1
                        or receipt['SELECT_queries'] != 0 or receipt['checkpoint'] != 'final epoch'
                        or receipt['input_manifest_sha256'] != spec['manifest']['recorded_sha256']
                        or receipt['OOF_rows'] != noof or receipt['OOF_keys_sha256'] != expected['key_sha256']['OOF']
                        or receipt['outputs_sha256']['oof-predictions.npz'] != pred_b['sha256']): raise ValueError('fixed seed identity')
                trace = receipt['consumption']
                if len(trace) != 30 or any(t['epoch'] != i or t['rows'] != expected['FIT'] or t['ordered_keys_sha256'] != expected['key_sha256']['FIT'] or t['every_key_once'] is not True for i, t in enumerate(trace, 1)): raise ValueError('exact FIT consumption')
                ps, _ = pinned(pred_b, stack)
                header_check(ps, {'keys', 'probabilities'}, {'keys': ((noof, 3), 'iu'), 'probabilities': ((noof, 7), 'f')})
                with np.load(ps, allow_pickle=False) as z: pk, q = z['keys'], z['probabilities']
                if not np.array_equal(pk, keys[keep]): raise ValueError('seed ordered keys')
                predictions[record['seed']] = (pk, cal.probabilities(q))
                if file_digest(ps) != pred_b['sha256']: raise ValueError('prediction changed')
            if file_digest(stream) != corpus_b['sha256']: raise ValueError('corpus changed')
            windows[fold] = {'keys': keys[keep], 'labels': labels[keep], 'seeds': predictions}
    cal.pool(windows)
    return windows


def oof_comparison(c, t, ccal, tcal):
    import numpy as np
    from experiments.ws_fullseason import calibrate as cal
    from experiments.ws_seq.evaluate_seq import row_loss, ece
    # The frozen metric module's CLI-era default sets 2; CPU comparison stays 1.
    for k in THREAD_ENV: os.environ[k] = '1'
    result = {}
    for scope in (*FOLDS, 'pooled'):
        if scope == 'pooled':
            ck, cy, cq = cal.pool(c); tk, ty, tq = cal.pool(t)
            cp = {s: np.concatenate([c[f]['seeds'][s][1] for f in FOLDS]) for s in SEEDS}
            tp = {s: np.concatenate([t[f]['seeds'][s][1] for f in FOLDS]) for s in SEEDS}
            keys = np.concatenate([c[f]['keys'] for f in FOLDS]); order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
            cp = {s: q[order] for s, q in cp.items()}; tp = {s: q[order] for s, q in tp.items()}
        else:
            ck, cy, tk, ty = c[scope]['keys'], c[scope]['labels'], t[scope]['keys'], t[scope]['labels']
            cp = {s: c[scope]['seeds'][s][1] for s in SEEDS}; tp = {s: t[scope]['seeds'][s][1] for s in SEEDS}
            cq, tq = cal.mean_seeds(cp), cal.mean_seeds(tp)
        if not np.array_equal(ck, tk) or not np.array_equal(cy, ty): raise ValueError('paired exact keys and labels')
        rows = {}
        for label, a, b in [('raw_OOF', cq, tq), ('own_pooled_calibration_resubstitution', cal.apply(cp, ccal), cal.apply(tp, tcal))]:
            av, bv = row_loss(a, cy), row_loss(b, cy)
            cm = {'nll': float(av[0].mean()), 'brier': float(av[1].mean()), 'ece10': ece(a, cy)}
            tm = {'nll': float(bv[0].mean()), 'brier': float(bv[1].mean()), 'ece10': ece(b, cy)}
            rows[label] = {'C': cm, 'T': tm, 'C_minus_T': {k: cm[k] - tm[k] for k in cm}}
        result[scope] = {'rows': len(ck), 'metrics': rows}
    return {'comparison': result, 'nll_clip': 1e-15, 'final_weights_evaluated': False, 'hypothesis_test': None, 'independent_test': False}


def c_ensemble(cfg, state_bindings, encoder_binding, calibrator_binding):
    """Explicit C loader; keep the existing T-only loader and its gates intact."""
    import numpy as np
    from pitcheezy.seq_shadow import fullseason as fs
    from experiments.ws_fullseason import calibrate as cal
    recipe = same_recipe(cfg, 'final')
    if set(state_bindings) != {'0', '1', '2'}: raise ValueError('three FINAL C seeds')
    raw = {s: fs.read_artifact(state_bindings[str(s)], fs.STATE_CAP) for s in SEEDS}
    encoder = json.loads(fs.read_artifact(encoder_binding, fs.JSON_CAP)); fs._encoder(encoder)
    calibrator = json.loads(fs.read_artifact(calibrator_binding, fs.JSON_CAP))
    cal.apply({s: np.full((1, 7), 1 / 7) for s in SEEDS}, calibrator)
    for value in raw.values():
        with zipfile.ZipFile(io.BytesIO(value)) as archive:
            info = archive.infolist()
            if len(info) > 1024 or len({v.filename for v in info}) != len(info) or sum(v.file_size for v in info) > fs.STATE_CAP or any(v.compress_type != zipfile.ZIP_STORED or v.flag_bits & 1 for v in info): raise ValueError('bounded stored C state')
    import torch
    from experiments.ws_campaign.models import build_model
    torch.set_num_threads(1); nets = {}; geometry = cfg['expected_C_state_geometry']
    for seed in SEEDS:
        state = torch.load(io.BytesIO(raw[seed]), map_location='cpu', weights_only=True)
        if (not isinstance(state, dict) or len(state) != geometry['tensors'] or not all(isinstance(v, torch.Tensor) and v.layout == torch.strided and v.dtype == torch.float32 and torch.isfinite(v).all() for v in state.values()) or sum(v.numel() for v in state.values()) != geometry['parameters']): raise ValueError('C state geometry/finite fp32')
        net = build_model(recipe, 'C', seed).cpu(); net.load_state_dict(state, strict=True); nets[seed] = net.eval()
    return fs.Ensemble(nets, recipe, encoder, calibrator, cal)


def paired_final_records(crows, trows):
    import numpy as np
    from pitcheezy.seq_shadow.summary import pitch_statistics
    if len(crows) != 1659 or len(trows) != 1659: raise ValueError('fixed quality denominator')
    cp, tp, labels, keys = [], [], [], []; seen = set()
    for c, t in zip(crows, trows):
        if any(c[k] != t[k] for k in ('key', 'status', 'reason', 'label', 'observed_pitch_type')): raise ValueError('same physical primary key/status/label')
        key = tuple(c['key'])
        if key in seen: raise ValueError('duplicate final key')
        seen.add(key)
        if c['status'] == 'scored':
            for q in (c['probabilities'], t['probabilities']):
                q = np.asarray(q)
                if q.shape != (7,) or not np.isfinite(q).all() or (q < 0).any() or (q > 1).any() or abs(float(q.sum()) - 1) > 1e-10: raise ValueError('strict final simplex')
            keys.append(key); labels.append(c['label']); cp.append(c['probabilities']); tp.append(t['probabilities'])
    if len(keys) != 1323: raise ValueError('fixed primary scored denominator')
    from experiments.ws_fullseason.data import key_digest
    keys, labels, cp, tp = np.asarray(keys), np.asarray(labels), np.asarray(cp), np.asarray(tp)
    result = {}
    for scope in ('pooled', *[str(v) for v in (849812,849820,849813,849811,849810,849818)]):
        keep = np.ones(len(keys), bool) if scope == 'pooled' else keys[:, 0] == int(scope)
        cm = pitch_statistics(cp[keep], labels[keep], nll_clip=1e-12, bins=10)
        tm = pitch_statistics(tp[keep], labels[keep], nll_clip=1e-12, bins=10)
        result[scope] = {'rows': int(keep.sum()), 'C': {k: float(cm[k].mean()) for k in ('nll', 'brier')}, 'T': {k: float(tm[k].mean()) for k in ('nll', 'brier')}, 'C_minus_T': {k: float((cm[k]-tm[k]).mean()) for k in ('nll', 'brier')}}
        for arm, stats in (('C', cm), ('T', tm)): result[scope][arm]['ece10'] = float(np.abs(stats['gap'].sum(0)).sum() / keep.sum())
        result[scope]['C_minus_T']['ece10'] = result[scope]['C']['ece10'] - result[scope]['T']['ece10']
    return {'scored_keys_sha256': key_digest(keys), 'comparison': result, 'nll_clip': 1e-12, 'hypothesis_test': None, 'independent_test': False}


def final_compare(cfg, g, meta, out):
    from experiments.ws_fullseason.data import closure
    from scripts.fullseason_postcutoff_offline import _source, score_population
    from pitcheezy.seq_shadow.postcutoff_offline import prepare
    from pitcheezy.seq_shadow.fullseason_runtime import Runtime
    from pitcheezy.seq_shadow.fullseason_profiles import load_bundle
    from pitcheezy.seq_shadow.ledger import sha as canonical_sha
    from pitcheezy.seq_shadow.fullseason import LEAVES
    f = meta['final_comparison']
    records = g['C_final_records']
    if [r['seed'] for r in records] != list(SEEDS): raise ValueError('fixed final three seeds')
    for record in records:
        receipt = completed_C(record, cfg, 'final', record['seed'], 'model-state.pt')
        spec = meta['training']['packages']['final']
        if receipt.get('execution_stage') != 'FINAL_REFIT' or receipt.get('OOF_rows') != 0 or receipt.get('epochs') != 30 or receipt.get('calibration_fits') != 0 or receipt.get('input_manifest_sha256') != spec['manifest']['recorded_sha256']: raise ValueError('fixed FINAL fit provenance')
        if len(receipt['consumption']) != 30 or any(v['rows'] != spec['roles']['FIT'] or v['ordered_keys_sha256'] != spec['expected']['key_sha256']['FIT'] or v['every_key_once'] is not True for v in receipt['consumption']): raise ValueError('FINAL exact FIT consumption')
    expected_encoder = next(binding(v) for v in spec['files'] if Path(v['path']).name == 'encoder.json')
    if g['C_encoder'] != expected_encoder: raise ValueError('same final FIT-only encoder')
    own_calibrator(g, meta)
    if g['old_score_registration']['sha256'] != f['primary_review']['registration_sha256']: raise ValueError('original six-game registration pin')
    oldreg = document(verify_file(g['old_score_registration'], 1 << 20))
    if oldreg['profile_bundle']['manifest_sha256'] != f['profile_manifest']['recorded_sha256']: raise ValueError('frozen profile identity')
    frame, source_sha, _ = _source(oldreg, closure()); prepared = prepare(frame, source_sha, closure())
    if prepared['census'] != document(verify_file(binding(f['primary_receipt']), 1 << 20))['census']: raise ValueError('exact original primary census')
    old = document(verify_file(binding(f['primary_score_rows']), 1 << 20))
    if old['leaf_order'] != list(LEAVES) or old['source_sha256'] != source_sha: raise ValueError('old T prediction identity')
    class CStandaloneRuntime(Runtime):
        def __init__(self):
            states = {str(r['seed']): binding(r['model']) for r in records}
            self.ensemble = c_ensemble(cfg, states, g['C_encoder'], g['C_calibration']['calibrator'])
            self.profiles = load_bundle(str(Path(f['profile_manifest']['path']).parent), f['profile_manifest']['recorded_sha256'])
            self.identity = {'arm': 'C', 'states': states, 'encoder': g['C_encoder'], 'calibrator': g['C_calibration']['calibrator'], 'profile_manifest_sha256': f['profile_manifest']['recorded_sha256']}
            self.bundle_sha256 = canonical_sha(self.identity)
    runtime = CStandaloneRuntime()
    metrics, records = score_population(prepared, runtime.predict, source_sha256=source_sha, bundle_sha256=runtime.bundle_sha256)
    result = paired_final_records(records['rows'], old['rows'])
    result.update(metrics=metrics, old_T_score_rows_sha256=f['primary_score_rows']['recorded_sha256'], quality_keys_sha256=f['primary_census']['accepted_keys_sha256'], source_arrival='UNKNOWN', prospective_claim=False, operational_adoption=False)
    write_json(out / 'score_rows.json', records)
    return result


def write_json(path, value):
    raw = json.dumps(value, indent=2, allow_nan=False) + '\n'
    if len(raw.encode()) > 16 << 20: raise ValueError('output JSON cap')
    with Path(path).open('x') as f: f.write(raw)


def fresh_worker_output(out):
    names = {p.name for p in out.iterdir()} if out.is_dir() else set()
    if not {'tmp', 'worker.log'} <= names or names - {'tmp', 'worker.log', '._tmp', '._worker.log'}:
        raise ValueError('fresh supervised worker output')


def worker(cfg, g, meta, out, config_sha, grant_sha):
    fd = int(os.environ['PITCHEEZY_C_LOCK_FD']); held = os.fstat(fd); lock = Path(g['lock']).stat()
    with Path(g['lock']).open('rb') as probe:
        try: fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: pass
        else: raise ValueError('inherited lock not actually held')
    if (held.st_dev, held.st_ino) != (lock.st_dev, lock.st_ino): raise ValueError('inherited exact lock')
    fresh_worker_output(Path(g['output']))
    if g['task'].startswith('C-'): result = fit_worker(cfg, g, meta, out)
    else:
        from experiments.ws_fullseason import calibrate as cal
        if g['task'] in ('calibrate', 'compare_oof'):
            c = load_windows(meta, g['C_oof_predictions'], 'C', cfg)
            if g['task'] == 'calibrate':
                result = cal.fit(c); write_json(out / 'calibrator.json', result)
                result = {'calibrator': result, 'calibration_fits': result['optimizer_calls'], 'metrics_computed': False}
            else:
                t = load_windows(meta, meta['training']['T_oof_predictions'], 'T')
                cc = own_calibrator(g, meta); tc = document(verify_file(binding(meta['training']['T_pooled_calibration']['calibrator.json']), 1 << 20))
                result = oof_comparison(c, t, cc, tc)
        else: result = final_compare(cfg, g, meta, out)
    verify_sources(cfg, g['source_manifest_path'])
    result.update(status='COMPLETED', task=g['task'], source_commit=g['source_commit'], source_manifest_sha256=cfg['source_manifest']['sha256'], input_metadata_sha256=cfg['input_metadata']['sha256'], runtime={n: getattr(sys.modules.get(n), '__version__', None) for n in ('numpy','scipy','pandas','pyarrow','torch')} | {'python':sys.version}, config_sha256=config_sha, grant_sha256=grant_sha, data_role=cfg['data_role'], independent_test=False, outputs_sha256={p.name: sha(p) for p in out.iterdir() if p.is_file() and p.name not in ('worker.log', 'receipt.json')})
    write_json(out / 'receipt.json', result)


def gpu_check(g):
    if sys.platform != 'linux' or not os.environ.get('GPU_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES') != str(g['gpu_index']): raise ValueError('registered gpu run device')
    expected = Path.home() / 'pitcheezy-autonomy/gpu-locks' / (g['gpu_uuid'] + '.lock')
    if Path(g['lock']).resolve() != expected.resolve(): raise ValueError('exact common GPU lock')
    rows = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()
    devices = [tuple(v.strip() for v in row.split(',')) for row in rows]
    found = [v for v in devices if v[1] == g['gpu_uuid']]
    if len(found) != 1 or int(found[0][0]) != g['gpu_index'] or int(found[0][2]) < g['min_free_mib']: raise ValueError('fresh GPU identity/free memory')
    active = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    if any(row.split(',')[0].strip() == g['gpu_uuid'] for row in active.splitlines()): raise ValueError('GPU compute process exists')


def headroom(g):
    if sys.platform == 'linux':
        mem = dict((line.split(':')[0], int(line.split()[1]) * 1024) for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
        if mem['MemAvailable'] < g['free_memory_bytes']: raise ValueError('host memory headroom')
    elif sys.platform == 'darwin':
        value = subprocess.check_output(['memory_pressure', '-Q'], text=True)
        found = re.search(r'System-wide memory free percentage: ([0-9]+)%', value)
        if not found or int(found[1]) < 20: raise ValueError('recorder-first system memory pressure')
    else: raise ValueError('unregistered runtime platform')


def supervise(cfg, g, args):
    fit = g['task'].startswith('C-'); out = Path(g['output']); root = Path(g['output_root']).resolve()
    if out.exists() or out.is_symlink() or not out.resolve().is_relative_to(root) or not root.is_dir(): raise ValueError('fresh registered output under existing root')
    if g['device_mount'] is not None:
        if not os.path.ismount(g['device_mount']): raise ValueError('registered output mount absent')
        info = plistlib.loads(subprocess.check_output(['diskutil','info','-plist',g['device_mount']]))
        if info.get('VolumeUUID') != g['volume_uuid']: raise ValueError('registered volume UUID')
    output_device = root.stat().st_dev
    headroom(g)
    if shutil.disk_usage(root).free < g['free_disk_bytes']: raise ValueError('output filesystem headroom')
    if fit and not out.resolve().is_relative_to(Path(os.environ['GPU_JOB_DIR']).resolve() / 'code'): raise ValueError('pullable GPU job output')
    if not fit and os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise ValueError('CPU-only phase')
    with Path(g['lock']).open('rb') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if fit: gpu_check(g)
        if fit:
            claim = Path(g['task_claim'])
            if claim.name != g['task'] + '.json' or claim.parent.resolve() != Path(g['claim_root']).resolve(): raise ValueError('exact shared task claim')
            write_json(claim, {'task': g['task'], 'grant_sha256': args.grant_sha256, 'output': str(out), 'state': 'CLAIMED_NO_AUTO_RETRY', 'gpu_index': g['gpu_index'], 'gpu_uuid': g['gpu_uuid']})
        out.mkdir(parents=True, mode=0o700)
        env = dict(os.environ, PITCHEEZY_C_LOCK_FD=str(lock.fileno()), TMPDIR=str(out / 'tmp'))
        (out / 'tmp').mkdir()
        cmd = [sys.executable, '-B', str(Path(__file__)), '--config', str(args.config), '--grant', str(args.grant), '--grant-sha256', args.grant_sha256, '--task', args.task, '--worker']
        start = time.monotonic(); peak = 0; reason = None
        with (out / 'worker.log').open('x') as log:
            child = subprocess.Popen(cmd, env=env, pass_fds=(lock.fileno(),), stdout=log, stderr=log)
            try:
                while child.poll() is None:
                    rss = subprocess.run(['ps','-o','rss=','-p',str(child.pid)],text=True,capture_output=True)
                    if rss.returncode == 0 and rss.stdout.strip(): peak = max(peak, int(rss.stdout.strip()) * 1024)
                    elif child.poll() is None: reason = 'unobservable-owned-child-rss'; break
                    try: headroom(g)
                    except (ValueError, OSError, subprocess.CalledProcessError): reason = 'host-memory-headroom'; break
                    if peak > g['rss_bytes']: reason = 'rss'; break
                    if time.monotonic() - start > g['wall_seconds']: reason = 'wall'; break
                    if sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) > g['output_bytes']: reason = 'output'; break
                    if root.stat().st_dev != output_device or (g['device_mount'] is not None and not os.path.ismount(g['device_mount'])): reason = 'mount-lost'; break
                    if shutil.disk_usage(root).free < g['free_disk_bytes']: reason = 'disk'; break
                    time.sleep(.25)
            finally:
                if child.poll() is None:
                    child.terminate()
                    try: child.wait(timeout=5)
                    except subprocess.TimeoutExpired: child.kill(); child.wait()
        if sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) > g['output_bytes']: reason = 'output'
        completed = child.returncode == 0 and reason is None and (out / 'receipt.json').is_file() and document(out / 'receipt.json').get('status') == 'COMPLETED'
        write_json(out / 'SUPERVISOR.json', {'status': 'COMPLETED' if completed else 'STOP', 'returncode': child.returncode, 'reason': reason, 'child_reaped': True, 'child_pid': child.pid, 'peak_sampled_rss_bytes': peak, 'elapsed_seconds': time.monotonic() - start, 'grant_sha256': args.grant_sha256, 'receipt_sha256': sha(out / 'receipt.json') if completed else None})
        return 0 if completed else 87


def main():
    p = argparse.ArgumentParser(); p.add_argument('--config', type=Path, required=True); p.add_argument('--grant', type=Path, required=True); p.add_argument('--grant-sha256', required=True); p.add_argument('--task', required=True); p.add_argument('--worker', action='store_true'); a = p.parse_args()
    cfg, g, meta = gates(a.config, a.grant, a.grant_sha256, a.task)
    if a.worker: worker(cfg, g, meta, Path(g['output']), sha(a.config), a.grant_sha256); return 0
    return supervise(cfg, g, a)


if __name__ == '__main__': sys.exit(main())
