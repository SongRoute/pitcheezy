"""Bounded exposed-OOF study; existing T fits are controls, never repeated.

Pure custody/resource primitives are reused from the accepted C runner. Its C
study gates are not called or modified. Real I/O starts after this study's gate.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import plistlib
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from experiments.ws_fullseason_c import run as custody
from experiments.gpu14h import train

sha = custody.sha
object_sha = custody.object_sha
document = custody.document
binding = custody.binding
verify_file = custody.verify_file
write_json = custody.write_json
FOLDS = ('july', 'august', 'september')
THREAD_ENV = custody.THREAD_ENV
DEADLINE = '2026-11-03T03:21:56Z'


def sources(cfg, path):
    pins = custody.verify_sources(cfg, path)
    for name, module in tuple(sys.modules.items()):
        if name.startswith('experiments.gpu14h') and getattr(module, '__file__', None):
            p = Path(module.__file__).resolve()
            if not p.is_relative_to(ROOT) or str(p.relative_to(ROOT)) not in pins: raise ValueError('foreign study source')
    return pins


def effective_wall(cfg, now=None):
    now = time.time() if now is None else now
    end = datetime.fromisoformat(cfg['deadline_utc'].replace('Z', '+00:00')).timestamp()
    wall = min(cfg['limits']['fit_wall_seconds'], end - cfg['recovery_grace_seconds'] - now)
    if wall < cfg['minimum_fit_seconds']: raise ValueError('deadline admission: no complete fit budget')
    return wall


def producer(cfg, fold):
    spec = cfg['folds'][fold]
    p = ROOT / spec['producer_recipe']
    if sha(p) != spec['producer_recipe_sha256']: raise ValueError('frozen producer recipe')
    return document(p)


def completed_phase(record, cfg, task):
    r = document(verify_file(binding(record['receipt']), 2 << 20))
    s = document(verify_file(binding(record['supervisor']), 1 << 20))
    if (r.get('status') != 'COMPLETED' or r.get('task') != task
            or r.get('source_manifest_sha256') != cfg['source_manifest']['sha256']
            or r.get('config_sha256') != record['config_sha256']
            or s.get('status') != 'COMPLETED' or s.get('returncode') != 0
            or s.get('child_reaped') is not True or s.get('receipt_sha256') != binding(record['receipt'])['sha256']
            or s.get('grant_sha256') != r.get('grant_sha256')): raise ValueError('accepted predecessor chain')
    return r


def selected_screen(record, cfg, config_sha):
    if record['config_sha256'] != config_sha: raise ValueError('same study screen config')
    r = completed_phase(record, cfg, 'score_screen')
    selected = r['selected_variants']
    if len(selected) > 2 or len(set(selected)) != len(selected) or any(v not in train.VARIANTS for v in selected): raise ValueError('finite selected variants')
    if selected != select_variants(r['candidates']): raise ValueError('frozen selection replay')
    return selected


def gates(config_path, grant_path, grant_sha, task):
    cfg = document(config_path)
    if cfg.get('status') != 'REGISTERED' or cfg.get('execute') is not True: raise ValueError('DRAFT: real I/O disabled')
    if (cfg['id'] != 'EXP-GPU14H-001' or cfg['variants'] != train.VARIANTS or cfg['fit_ceiling'] != 42
            or cfg['data_role'] != '2026 EXPOSED DEVELOPMENT' or cfg['seeds'] != [0, 1, 2]
            or cfg['deadline_utc'] != DEADLINE or cfg['recovery_grace_seconds'] != 900
            or cfg['minimum_fit_seconds'] != 900): raise ValueError('fixed study contract')
    expected = {(v, f, s) for v in train.VARIANTS for f in FOLDS for s in (0, 1, 2)}
    if {(j['variant'], j['fold'], j['seed']) for j in cfg['jobs']} != expected or len(cfg['jobs']) != 90: raise ValueError('finite potential task identities')
    for j in cfg['jobs']:
        if j['id'] != f"H14-{j['variant']}-{j['fold']}-s{j['seed']}": raise ValueError('unique task identity')
    if sha(grant_path) != grant_sha: raise ValueError('exact reviewed grant')
    raw = document(grant_path)
    if (raw.get('execute') is not True or raw.get('verdict') != 'PASS' or raw.get('authorized_by') != 'root'
            or raw.get('author') != cfg['author'] or not raw.get('reviewer') or raw['reviewer'] == raw['author']
            or raw.get('config_sha256') != sha(config_path) or task not in raw['tasks']): raise ValueError('independent phase grant')
    g = dict(raw, **raw['tasks'][task], task=task)
    for key in ('output', 'output_root'):
        value = g[key]
        if '${GPU_JOB_DIR}' in value:
            if not os.environ.get('GPU_JOB_DIR'): raise ValueError('gpu run missing')
            value = value.replace('${GPU_JOB_DIR}', os.environ['GPU_JOB_DIR'])
        if '$' in value or '~' in value: raise ValueError('unregistered expansion')
        g[key] = value
    sources(cfg, g['source_manifest_path'])
    if sha(g['input_metadata_path']) != cfg['input_metadata']['sha256']: raise ValueError('input metadata pin')
    if any(os.environ.get(k) != '1' for k in THREAD_ENV) or os.environ.get('PYTHONPATH') or os.environ.get('PYTHONHOME'): raise ValueError('fixed import/thread environment')
    jobs = {j['id']: j for j in cfg['jobs']}; fit = task in jobs
    if fit:
        job = jobs[task]
        if not job['enabled']: raise ValueError('disabled fit')
        train.recipe(producer(cfg, job['fold']), job['variant'])
        if job['seed'] != 0 and job['variant'] not in selected_screen(g['screen'], cfg, sha(config_path)): raise ValueError('nonselected stability fit')
        if g['min_free_mib'] != 20480 or g['free_memory_bytes'] != 72 << 30: raise ValueError('GPU host admission')
        devices = g['gpu_candidates']
        if len(devices) != 5 or len({v['uuid'] for v in devices}) != 5 or {v['index'] for v in devices} != set(range(5)): raise ValueError('five candidate GPUs')
        chosen = [v for v in devices if str(v['index']) == os.environ.get('CUDA_VISIBLE_DEVICES')]
        if len(chosen) != 1: raise ValueError('gpu run one numeric device')
        g.update(gpu_index=chosen[0]['index'], gpu_uuid=chosen[0]['uuid'], lock=str(Path(g['gpu_lock_root']) / (chosen[0]['uuid'] + '.lock')))
    elif task not in cfg['phases'] or cfg['phases'][task] is not True: raise ValueError('disabled phase')
    limits = cfg['limits']
    if (g['rss_bytes'] != limits['rss_bytes'] or g['wall_seconds'] != limits['fit_wall_seconds' if fit else 'cpu_wall_seconds']
            or g['output_bytes'] != limits['output_bytes_per_fit' if fit else 'cpu_output_bytes']
            or g['free_disk_bytes'] != cfg['resource_admission']['remote_disk_free_bytes' if fit else 'local_T7_free_bytes']): raise ValueError('fixed resource bounds')
    if fit: effective_wall(cfg)
    return cfg, g, document(g['input_metadata_path'])


def fit_worker(cfg, g, meta, out):
    job = next(j for j in cfg['jobs'] if j['id'] == g['task'])
    original = producer(cfg, job['fold']); recipe = train.recipe(original, job['variant'])
    inputs, spec = custody.package_binding(meta, job['fold'], g)
    import torch
    from experiments.ws_seq import train_seq as base
    from experiments.ws_fullseason.data import key_digest
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1: raise ValueError('one CUDA device')
    torch.cuda.set_per_process_memory_fraction(cfg['limits']['vram_bytes'] / torch.cuda.get_device_properties(0).total_memory)
    package = base.load_package(inputs, original['base']); _, corpus, _ = package
    for role, field in (('FIT', 'fit'), ('OOF', 'evaluation')):
        if int(corpus[field].sum()) != spec['roles'][role] or key_digest(corpus['keys'][corpus[field]]) != spec['expected']['key_sha256'][role]: raise ValueError('fixed role keys')
    r = train.run(original, job['variant'], job['seed'], package, out)
    r.update(variant=job['variant'], fold=job['fold'], model_recipe_sha256=object_sha(recipe), producer_recipe_sha256=spec['producer_recipe']['recorded_sha256'], input_manifest_sha256=spec['manifest']['recorded_sha256'], gpu_index=g['gpu_index'], gpu_uuid=g['gpu_uuid'], gpu_max_reserved_bytes=torch.cuda.max_memory_reserved())
    if r['gpu_max_reserved_bytes'] > cfg['limits']['vram_bytes']: raise ValueError('VRAM cap')
    custody.package_binding(meta, job['fold'], g)
    return r


def prediction(record, cfg, spec, fold, seed, keys, variant=None):
    import numpy as np
    from experiments.ws_fullseason.calibrate import probabilities
    from experiments.ws_fullseason.calibrate_worker import pinned, header_check, file_digest
    rb = binding(record['receipt']); pb = binding(record['predictions'])
    r = document(verify_file(rb, 1 << 20)); expected = spec['expected']
    if variant is not None:
        if (record['config_sha256'] != cfg['_config_sha256'] or r.get('variant') != variant or r.get('fold') != fold
                or r.get('model_recipe_sha256') != object_sha(train.recipe(producer(cfg, fold), variant))): raise ValueError('candidate recipe identity')
        completed_phase(record, cfg, f'H14-{variant}-{fold}-s{seed}')
    if (r['status'] != 'COMPLETED' or r['arm'] != 'T' or r['seed'] != seed or r['epochs'] != 30
            or r['calibration_fits'] != 0 or r['temperature'] != 1 or r['SELECT_queries'] != 0
            or r['checkpoint'] != 'final epoch' or r['input_manifest_sha256'] != spec['manifest']['recorded_sha256']
            or r['OOF_rows'] != expected['OOF'] or r['OOF_keys_sha256'] != expected['key_sha256']['OOF']
            or r['outputs_sha256']['oof-predictions.npz'] != pb['sha256']): raise ValueError('fixed prediction provenance')
    trace = r['consumption']
    if len(trace) != 30 or any(t['epoch'] != i or t['rows'] != expected['FIT'] or t['ordered_keys_sha256'] != expected['key_sha256']['FIT'] or t['every_key_once'] is not True for i, t in enumerate(trace, 1)): raise ValueError('exact FIT consumption')
    with ExitStack() as stack:
        stream, _ = pinned(pb, stack)
        header_check(stream, {'keys', 'probabilities'}, {'keys': ((len(keys), 3), 'iu'), 'probabilities': ((len(keys), 7), 'f')})
        with np.load(stream, allow_pickle=False) as z: pk, q = z['keys'], z['probabilities']
        if not np.array_equal(pk, keys): raise ValueError('exact ordered OOF prediction keys')
        q = probabilities(q)
        if file_digest(stream) != pb['sha256']: raise ValueError('prediction changed')
    return q


def labels(meta, fold):
    import numpy as np
    from experiments.ws_fullseason.calibrate_worker import pinned, header_check, CORPUS, file_digest
    from experiments.ws_fullseason.data import key_digest
    spec = meta['training']['packages'][fold]; e = spec['expected']; n = e['rows']
    b = next(binding(v) for v in spec['files'] if Path(v['path']).name == 'corpus-subset.npz')
    with ExitStack() as stack:
        stream, _ = pinned(b, stack)
        header_check(stream, CORPUS, {'keys': ((n, 3), 'iu'), 'y7': ((n,), 'iu'), 'fit': ((n,), 'b'), 'evaluation': ((n,), 'b')})
        with np.load(stream, allow_pickle=False) as z: k, y, fit, keep = (z[v] for v in ('keys', 'y7', 'fit', 'evaluation'))
        if int(fit.sum()) != e['FIT'] or int(keep.sum()) != e['OOF'] or (fit & keep).any() or not (fit | keep).all(): raise ValueError('fixed full roles')
        if any(key_digest(k[m]) != e['key_sha256'][role] for m, role in ((fit, 'FIT'), (keep, 'OOF'))): raise ValueError('fixed corpus keys')
        if y.min() < 0 or y.max() >= 7 or file_digest(stream) != b['sha256']: raise ValueError('fixed labels')
    return k[keep], y[keep]


def metrics(q, y):
    from experiments.ws_seq.evaluate_seq import row_loss, ece
    for key in THREAD_ENV: os.environ[key] = '1'  # Legacy metric import sets 2; CPU contract remains 1.
    loss, brier = row_loss(q, y)
    return {'rows': len(y), 'nll': float(loss.mean()), 'brier': float(brier.mean()), 'ece10': ece(q, y)}


def comparison(candidate, baseline, y):
    c, b = metrics(candidate, y), metrics(baseline, y)
    return {'candidate': c, 'baseline': b, 'delta': {k: c[k] - b[k] for k in ('nll', 'brier', 'ece10')}}


def select_variants(rows):
    # ponytail: ten candidates only; deterministic tolerance grouping avoids a
    # non-transitive pairwise float comparator. No general ranking framework.
    eligible = [r for r in rows if r['status'] == 'COMPLETE' and r['pooled']['delta']['nll'] < 0 and r['pooled']['delta']['brier'] <= 0]
    chosen = []
    while eligible and len(chosen) < 2:
        best = min(r['pooled']['delta']['nll'] for r in eligible)
        tied = [r for r in eligible if r['pooled']['delta']['nll'] - best <= 1e-6]
        winner = min(tied, key=lambda r: r['variant'])
        chosen.append(winner['variant']); eligible.remove(winner)
    return chosen


def score_worker(cfg, g, meta):
    import numpy as np
    screen = g['task'] == 'score_screen'
    variants = list(train.VARIANTS) if screen else selected_screen(g['screen'], cfg, cfg['_config_sha256'])
    seeds = [0] if screen else [0, 1, 2]
    expected = {(v, f, s) for v in variants for f in FOLDS for s in seeds}
    records = g['results']; unavailable = g['unavailable_tasks']
    by = {(r['variant'], r['fold'], r['seed']): r for r in records}
    absent = {(r['variant'], r['fold'], r['seed']) for r in unavailable}
    if len(by) != len(records) or len(absent) != len(unavailable) or set(by) & absent or set(by) | absent != expected: raise ValueError('exact complete/incomplete task accounting')
    if any(r['status'] not in ('FAILED', 'NOT_RUN', 'INCOMPLETE') for r in unavailable): raise ValueError('explicit unavailable status')
    complete = [v for v in variants if all((v, f, s) in by for f in FOLDS for s in seeds)]
    all_labels = {}; baseline = {}
    for f in FOLDS:
        k, y = labels(meta, f); all_labels[f] = (k, y)
        baseline[f] = {s: prediction(next(r for r in meta['training']['T_oof_predictions'] if r['fold'] == f and r['seed'] == s), cfg, meta['training']['packages'][f], f, s, k) for s in seeds}
    ys = np.concatenate([all_labels[f][1] for f in FOLDS])
    base_pooled = np.concatenate([np.mean([baseline[f][s] for s in seeds], axis=0, dtype=np.float64) for f in FOLDS])
    rows = []
    for v in variants:
        if v not in complete:
            rows.append({'variant': v, 'status': 'INCOMPLETE_NOT_SELECTED', 'pooled': None, 'unavailable': [r for r in unavailable if r['variant'] == v]}); continue
        own = {}; monthly = {}; per_seed = {}
        for f in FOLDS:
            k, y = all_labels[f]
            own[f] = {s: prediction(by[v, f, s], cfg, meta['training']['packages'][f], f, s, k, v) for s in seeds}
            monthly[f] = comparison(np.mean(list(own[f].values()), axis=0, dtype=np.float64), np.mean(list(baseline[f].values()), axis=0, dtype=np.float64), y)
        pooled = np.concatenate([np.mean(list(own[f].values()), axis=0, dtype=np.float64) for f in FOLDS])
        for s in seeds: per_seed[str(s)] = comparison(np.concatenate([own[f][s] for f in FOLDS]), np.concatenate([baseline[f][s] for f in FOLDS]), ys)
        rows.append({'variant': v, 'status': 'COMPLETE', 'pooled': comparison(pooled, base_pooled, ys), 'monthly': monthly, 'per_seed': per_seed})
    return {'candidates': rows, 'selected_variants': select_variants(rows) if screen else variants, 'selection_applied': screen,
            'matched_seeds': seeds, 'baseline': 'accepted frozen T same seeds; mean probabilities before metrics',
            'purpose': 'exposed OOF development screening' if screen else 'seed stability on same exposed OOF, not independent confirmation',
            'new_backbone_fits_in_phase': 0, 'calibration_fits': 0, 'FINAL_evaluations': 0,
            'independent_test': False, 'significance_test': False, 'promotion': False,
            'unavailable_tasks': unavailable, 'selection_bias': 'seed0 selection and multiple candidates on repeatedly exposed OOF'}


def worker(cfg, g, meta, out, config_sha, grant_sha):
    fd = int(os.environ['PITCHEEZY_H14_LOCK_FD']); held = os.fstat(fd); lock = Path(g['lock']).stat()
    with Path(g['lock']).open('rb') as probe:
        try: fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: pass
        else: raise ValueError('inherited lock absent')
    if (held.st_dev, held.st_ino) != (lock.st_dev, lock.st_ino): raise ValueError('exact inherited lock')
    custody.fresh_worker_output(out)
    cfg['_config_sha256'] = config_sha
    result = fit_worker(cfg, g, meta, out) if g['task'].startswith('H14-') else score_worker(cfg, g, meta)
    sources(cfg, g['source_manifest_path'])
    result.update(status='COMPLETED', task=g['task'], source_commit=g['source_commit'], source_manifest_sha256=cfg['source_manifest']['sha256'], input_metadata_sha256=cfg['input_metadata']['sha256'], config_sha256=config_sha, grant_sha256=grant_sha, data_role=cfg['data_role'], independent_test=False, runtime={n: getattr(sys.modules.get(n), '__version__', None) for n in ('numpy','scipy','pandas','pyarrow','torch')} | {'python': sys.version}, outputs_sha256={p.name: sha(p) for p in out.iterdir() if p.is_file() and p.name not in ('worker.log','receipt.json')})
    write_json(out / 'receipt.json', result)


def supervise(cfg, g, args):
    fit = g['task'].startswith('H14-'); out = Path(g['output']); root = Path(g['output_root']).resolve()
    if out.exists() or out.is_symlink() or not out.resolve().is_relative_to(root) or not root.is_dir(): raise ValueError('fresh registered output under existing root')
    if g['device_mount'] is not None:
        if not os.path.ismount(g['device_mount']): raise ValueError('registered output mount absent')
        info = plistlib.loads(subprocess.check_output(['diskutil','info','-plist',g['device_mount']]))
        if info.get('VolumeUUID') != g['volume_uuid']: raise ValueError('registered volume UUID')
    output_device = root.stat().st_dev
    custody.headroom(g)
    if shutil.disk_usage(root).free < g['free_disk_bytes']: raise ValueError('output filesystem headroom')
    if fit and not out.resolve().is_relative_to(Path(os.environ['GPU_JOB_DIR']).resolve() / 'code'): raise ValueError('pullable GPU job output')
    if not fit and os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise ValueError('CPU-only phase')
    with Path(g['lock']).open('rb') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if fit: custody.gpu_check(g)
        if fit:
            claim = Path(g['task_claim'])
            if claim.name != g['task'] + '.json' or claim.parent.resolve() != Path(g['claim_root']).resolve(): raise ValueError('exact shared task claim')
            write_json(claim, {'task': g['task'], 'grant_sha256': args.grant_sha256, 'output': str(out), 'state': 'CLAIMED_NO_AUTO_RETRY', 'gpu_index': g['gpu_index'], 'gpu_uuid': g['gpu_uuid']})
        out.mkdir(parents=True, mode=0o700)
        env = dict(os.environ, PITCHEEZY_H14_LOCK_FD=str(lock.fileno()), TMPDIR=str(out / 'tmp'))
        (out / 'tmp').mkdir()
        cmd = [sys.executable, '-B', str(Path(__file__)), '--config', str(args.config), '--grant', str(args.grant), '--grant-sha256', args.grant_sha256, '--task', args.task, '--worker']
        absolute_end = datetime.fromisoformat(cfg['deadline_utc'].replace('Z', '+00:00')).timestamp() - (900 if fit else 60)
        wall = effective_wall(cfg) if fit else min(g['wall_seconds'], absolute_end - time.time())
        if wall <= 0: raise ValueError('absolute phase deadline')
        start = time.monotonic(); peak = 0; reason = None
        with (out / 'worker.log').open('x') as log:
            child = subprocess.Popen(cmd, env=env, pass_fds=(lock.fileno(),), stdout=log, stderr=log)
            try:
                while child.poll() is None:
                    rss = subprocess.run(['ps','-o','rss=','-p',str(child.pid)],text=True,capture_output=True)
                    if rss.returncode == 0 and rss.stdout.strip(): peak = max(peak, int(rss.stdout.strip()) * 1024)
                    elif child.poll() is None: reason = 'unobservable-owned-child-rss'; break
                    try: custody.headroom(g)
                    except (ValueError, OSError, subprocess.CalledProcessError): reason = 'host-memory-headroom'; break
                    if peak > g['rss_bytes']: reason = 'rss'; break
                    if time.time() >= absolute_end: reason = 'absolute-deadline'; break
                    if time.monotonic() - start > wall: reason = 'wall'; break
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
        write_json(out / 'SUPERVISOR.json', {'status': 'COMPLETED' if completed else 'STOP', 'returncode': child.returncode, 'reason': reason, 'child_reaped': True, 'child_pid': child.pid, 'peak_sampled_rss_bytes': peak, 'elapsed_seconds': time.monotonic() - start, 'effective_wall_seconds': wall, 'grant_sha256': args.grant_sha256, 'receipt_sha256': sha(out / 'receipt.json') if completed else None})
        return 0 if completed else 87


def main():
    p = argparse.ArgumentParser(); p.add_argument('--config', type=Path, required=True); p.add_argument('--grant', type=Path, required=True); p.add_argument('--grant-sha256', required=True); p.add_argument('--task', required=True); p.add_argument('--worker', action='store_true'); a = p.parse_args()
    cfg, g, meta = gates(a.config, a.grant, a.grant_sha256, a.task)
    if a.worker: worker(cfg, g, meta, Path(g['output']), sha(a.config), a.grant_sha256); return 0
    return supervise(cfg, g, a)


if __name__ == '__main__': sys.exit(main())
