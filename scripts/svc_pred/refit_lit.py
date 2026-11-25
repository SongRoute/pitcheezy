"""EXP-SVC-PRED-001 refit: re-run one EXP-LIT-BASE-001 final (same cache, hp, budget, seeds, tau) and SAVE the model.

    python -B scripts/svc_pred/refit_lit.py check  --family B-SEQ   # pins + original records only (no fit, no label)
    python -B scripts/svc_pred/refit_lit.py refit  --family B-SEQ   # fresh process; refuses unless refit.json is REGISTERED + execute
    python -B scripts/svc_pred/refit_lit.py pin    --family B-SEQ   # prints the svc-lit-bundle-v1 pin after the gates passed

Why: EXP-LIT-BASE-001 final saved E_PS predictions only. The refit calls the original ``run.fit_one`` (byte copy of
eb8e8b1b:experiments/lit_base) and captures what the fit built: seq nets (state_dict), the MLR parameters (LBFGS params),
the LightGBM booster, the TAB / feature TargetEncoders (refitted on the same FIT rows with the same code; checked against
the cached features). MPS (seq families) is not bit-reproducible -> a new draw of the same recipe; table / mlr / gbt are
expected to reproduce the scored predictions. Gates (configs/EXP-SVC-PRED-001/refit.json, fixed before any refit):
  A  label-free agreement refit vs scored E_PS q_tau (per-family bounds)
  B  |NLL(refit) - NLL(scored)| on the EXP-LIT-BASE-001 E_PS rows (same CR labels, floor, tool) <= bound; R - refit reported
  C  pitcheezy.svc_lit.bundle.Bundle (CPU, raw inputs -> its own features) reproduces the refit E_PS q (max |dq| bound)
Any gate failing -> refit.json pass false and the pin / predict / build steps refuse.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
for p in (REPO, REPO / 'src'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
REFIT = REPO / 'configs/EXP-SVC-PRED-001/refit.json'
KST = dt.timezone(dt.timedelta(hours=9))


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def git_state():
    h = subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(['git', '-C', str(REPO), 'status', '--porcelain', '--', 'experiments/lit_base', 'configs', 'scripts/svc_pred',
                            'scripts/svc_p9', 'scripts/svc_r1', 'scripts/svc_r3', 'scripts/svc_r4', 'src/pitcheezy'],
                           capture_output=True, text=True, check=True).stdout.strip()
    return h, bool(dirty)


def load_refit(path=REFIT):
    raw = pathlib.Path(path).read_bytes(); c = json.loads(raw)
    if c.get('id') != 'EXP-SVC-PRED-001' or c.get('part') != 'refit':
        raise SystemExit('not the EXP-SVC-PRED-001 refit config')
    return c, hashlib.sha256(raw).hexdigest()


def check_sources(g, fam):
    src = g['source']
    if fam not in g['families']:
        raise SystemExit(f'{fam} is not a registered refit family')
    for rel, want in src['files_sha256'].items():
        if sha(REPO / rel) != want:
            raise SystemExit(f'{rel} differs from its pin')
    for a, b in src['vendored_equal']:
        if sha(REPO / a) != sha(REPO / b):
            raise SystemExit(f'{a} != {b}')
    from experiments.lit_base import run as LR
    cfg = LR.load_cfg(REPO / src['lit_config'])
    if cfg['_sha256'] != src['lit_config_sha256']:
        raise SystemExit('lit-base config sha differs')
    root = LR.out_root(cfg, False)
    if str(root) != src['attempt_root']:
        raise SystemExit('attempt root differs')
    fin = root / 'final' / fam / 'final.json'; selp = root / 'tune' / fam / 'selection.json'
    LR.same_config(cfg, fin); LR.same_config(cfg, selp)
    final = json.loads(fin.read_text()); sel = json.loads(selp.read_text()); f = g['families'][fam]
    if sha(root / 'final' / fam / 'eps-pred.npz') != final['pred_sha256'] or final['pred_sha256'] != f['scored_pred_sha256']:
        raise SystemExit('scored eps-pred.npz differs from final.json / the pin')
    if (final['hp'], final['tau'], final['budget']) != (sel['hp'], sel['tau'], sel['budget']) or final['seeds'] != f['seeds'] or final['tau'] != f['tau']:
        raise SystemExit('final.json hp / tau / budget / seeds differ from selection.json or the pin')
    if cfg['families'][fam]['kind'] != f['kind']:
        raise SystemExit('family kind differs')
    return LR, cfg, root, final, sel


@contextlib.contextmanager
def capturing(kind, M):
    """Record what ``run.fit_one`` builds: seq nets (models.build_seq), MLR LBFGS params, LightGBM boosters."""
    captured = {}
    if kind == 'seq':
        orig = M.build_seq

        def capture(*a, **k):
            net = orig(*a, **k); captured.setdefault('nets', []).append(net); return net
        M.build_seq = capture
    elif kind == 'mlr':
        import torch
        orig = torch.optim.LBFGS

        class Cap(orig):
            def __init__(self, params, *a, **k):
                params = list(params); captured.setdefault('params', []).append(params); super().__init__(params, *a, **k)
        torch.optim.LBFGS = Cap
    elif kind == 'gbt':
        import lightgbm as lgb
        orig = lgb.train

        def cap_train(*a, **k):
            bst = orig(*a, **k); captured.setdefault('boosters', []).append(bst); return bst
        lgb.train = cap_train
    try:
        yield captured
    finally:
        if kind == 'seq':
            M.build_seq = orig
        elif kind == 'mlr':
            torch.optim.LBFGS = orig
        elif kind == 'gbt':
            lgb.train = orig


def save_fit(kind, s, captured, tr, sel, out, D, LB):
    """Write the model of the fit that just ran (seed s) into out/ (bundle file names of svc_lit.bundle.files_for)."""
    if kind == 'seq':
        import torch
        sd = {k: v.detach().to('cpu') for k, v in captured['nets'][-1].state_dict().items()}
        tmp = out / f'seed{s}.pt.tmp'; torch.save(sd, tmp); os.replace(tmp, out / f'seed{s}.pt')
    elif kind == 'mlr':
        ps = captured['params'][-1]; W, b = ps[0], ps[1]; P = ps[2] if len(ps) > 2 else None
        mu = tr['X'].mean(0); sd = tr['X'].std(0); sd[sd < 1e-6] = 1.0
        np.savez(out / 'mlr.npz', W=W.detach().numpy(), b=b.detach().numpy(),
                 P=np.zeros((0, 9), np.float32) if P is None else P.detach().numpy(), mu=mu, sd=sd)
    elif kind == 'gbt':
        captured['boosters'][-1].save_model(str(out / 'gbt.txt'))
    elif kind == 'table':
        te = D.TargetEncoder(sel['hp']['alpha']).fit(tr['pitcher'], tr['batter'], tr['balls'], tr['strikes'], tr['stand'], tr['y'])
        LB.save_encoder(out / 'tab-encoder.npz', te)


def fit_and_save(LR, cfg, fam, kind, sel, final, tr, ev, out, log, captured, D, LB):
    qs, fits = [], []
    for s in final['seeds']:
        t0 = time.monotonic(); before = {k: len(v) for k, v in captured.items()}
        q, info = LR.fit_one(cfg, fam, sel['hp'], tr, ev, s, log, dev=False, budget=sel['budget'])
        key = {'seq': 'nets', 'mlr': 'params', 'gbt': 'boosters'}.get(kind)
        if key and len(captured.get(key, [])) != before.get(key, 0) + 1:
            raise SystemExit(f'expected exactly one captured {key} per fit')
        save_fit(kind, s, captured, tr, sel, out, D, LB)
        np.save(out / f'seed{s}-eps.npy', q.astype(np.float64)); qs.append(q)
        fits.append({'seed': s, 'seconds': time.monotonic() - t0, 'info': {k: v for k, v in info.items() if k != 'trace'},
                     'trace': info.get('trace')})
    return qs, fits


def prepare_inputs(g, LR, cfg, kind, D):
    """Label-free consistency checks on the FIT cache (no fit): vocabulary, state-encoder round trip, feature encoder."""
    tr, ev, rep = LR.load_cache(cfg, 'final', False)
    if rep['cache_sha256'] != g['source']['cache_sha256']:
        raise SystemExit('cache sha differs from the pin')
    pv, bv = D.vocab(tr['pitcher']), D.vocab(tr['batter'])
    if not all(np.array_equal(D.index(a, v), c) for a, v, c in ((tr['pitcher'], pv, tr['pidx']), (tr['batter'], bv, tr['bidx']),
                                                                (ev['pitcher'], pv, ev['pidx']), (ev['batter'], bv, ev['bidx']))):
        raise SystemExit('vocabulary rebuilt from the cache does not reproduce the cached indices')
    enc = g['source']['state_encoder']
    if sha(enc['path']) != enc['sha256']:
        raise SystemExit('state encoder sha differs')
    b_, s_, st_, _ = D.raw_from_state(ev['state11'], enc['path'])
    if not (np.array_equal(b_, ev['balls']) and np.array_equal(s_, ev['strikes']) and np.array_equal(st_, ev['stand'])):
        raise SystemExit('raw_from_state(eval state11) differs from the cached balls / strikes / stand')
    fe = None
    if kind in ('mlr', 'gbt'):
        fe = D.TargetEncoder(cfg['features']['target_encoding_alpha']).fit(tr['pitcher'], tr['batter'], tr['balls'], tr['strikes'], tr['stand'], tr['y'])
        Xev = D.tabular(ev['state11'], ev['balls'], ev['strikes'], fe.transform(ev['pitcher'], ev['batter'], ev['balls'], ev['strikes'], ev['stand']), ev['tok'])
        if not np.array_equal(Xev, ev['X']):
            raise SystemExit('refitted feature encoder does not reproduce the cached E_PS features')
    return tr, ev, rep, pv, bv, fe


def precheck(g, fam):
    LR, cfg, root, final, sel = check_sources(g, fam)
    from experiments.lit_base import data as D
    tr, ev, rep, pv, bv, fe = prepare_inputs(g, LR, cfg, g['families'][fam]['kind'], D)
    return {'family': fam, 'train_rows': int(len(tr['y'])), 'eval_rows': int(len(ev['state11'])), 'pitcher_vocab': int(len(pv)),
            'batter_vocab': int(len(bv)), 'feature_encoder_reproduces_cache': fe is not None or None, 'labels_read': False}


def refit(g, g_sha, fam):
    if g['status'] != 'REGISTERED' or g['execute'] is not True:
        raise SystemExit('EXP-SVC-PRED-001 refit is not REGISTERED with execute true')
    commit = git_state()
    if commit[1]:
        raise SystemExit('uncommitted changes in the refit code / configs')
    out = pathlib.Path(g['families'][fam]['out_dir'])
    if out.exists():
        raise SystemExit(f'{out} exists; refit writes a new directory only')
    LR, cfg, root, final, sel = check_sources(g, fam)
    lock = admission(g, fam)
    from experiments.lit_base import models as M
    from experiments.lit_base import data as D
    from pitcheezy.svc_lit import bundle as LB
    kind = g['families'][fam]['kind']
    out.mkdir(parents=True)
    logf = open(out / 'refit.log', 'a')

    def log(m):
        logf.write(f'{dt.datetime.now(KST).isoformat()} {m}\n'); logf.flush()
    tr, ev, rep, pv, bv, fe = prepare_inputs(g, LR, cfg, kind, D)
    np.savez(out / 'vocab.npz', pitcher=pv, batter=bv)
    if fe is not None:
        LB.save_encoder(out / 'feature-encoder.npz', fe)
    if kind == 'gbt' and 'torch' in sys.modules:
        raise SystemExit('torch is loaded: LightGBM must run in a torch-free process (libomp clash)')
    with capturing(kind, M) as captured:
        qs, fits = fit_and_save(LR, cfg, fam, kind, sel, final, tr, ev, out, log, captured, D, LB)
    qm = np.mean(qs, 0); qm /= qm.sum(1, keepdims=True); qt = M.temper(qm, sel['tau'])
    np.save(out / 'eps-q_tau.npy', qt); np.save(out / 'eps-q_tau1.npy', qm)
    meta = {'format': 'svc-lit-bundle-meta-v1', 'family': fam, 'kind': kind, 'hp': sel['hp'], 'tau': sel['tau'], 'seeds': final['seeds'],
            'budget': sel['budget'], 'n_pitcher': int(len(pv)), 'n_batter': int(len(bv)), 'train_rows': rep['train_rows'],
            'train_date_range': rep.get('train_date_range'), 'cache_sha256': rep['cache_sha256'], 'lit_config_sha256': cfg['_sha256'],
            'refit_config_sha256': g_sha, 'commit': commit[0], 'device': cfg['resources']['seq_device'] if kind == 'seq' else 'cpu',
            'threads': cfg['resources']['threads'][cfg['families'][fam]['lane']], 'fits': fits}
    (out / 'bundle.json').write_text(json.dumps(meta, indent=1, default=float) + '\n')
    gates = gate_report(g, fam, LR, cfg, root, ev, qt, qm, qs, out)
    gates.update(family=fam, refit_config_sha256=g_sha, commit=commit, at=dt.datetime.now(KST).isoformat())
    (out / 'refit.json').write_text(json.dumps(gates, indent=1, allow_nan=False) + '\n')
    log(f'refit done pass={gates["pass"]}')
    return gates


def regate(g, g_sha, fam):
    """Recompute the gates from the saved refit files when the gate step itself failed (refit.json absent). No fit."""
    if g['status'] != 'REGISTERED' or g['execute'] is not True:
        raise SystemExit('EXP-SVC-PRED-001 refit is not REGISTERED with execute true')
    commit = git_state()
    if commit[1]:
        raise SystemExit('uncommitted changes in the refit code / configs')
    out = pathlib.Path(g['families'][fam]['out_dir'])
    if (out / 'refit.json').exists():
        raise SystemExit('refit.json exists; gates are computed once')
    meta = json.loads((out / 'bundle.json').read_text())
    if meta['refit_config_sha256'] != g_sha:
        raise SystemExit('bundle.json was made under another refit config')
    LR, cfg, root, final, sel = check_sources(g, fam)
    from experiments.lit_base import data as D
    tr, ev, rep, pv, bv, fe = prepare_inputs(g, LR, cfg, meta['kind'], D)
    qs = [np.load(out / f'seed{s}-eps.npy') for s in meta['seeds']]
    qm = np.load(out / 'eps-q_tau1.npy'); qt = np.load(out / 'eps-q_tau.npy')
    gates = gate_report(g, fam, LR, cfg, root, ev, qt, qm, qs, out)
    gates.update(family=fam, refit_config_sha256=g_sha, commit=commit, at=dt.datetime.now(KST).isoformat(), regate=True)
    (out / 'refit.json').write_text(json.dumps(gates, indent=1, allow_nan=False) + '\n')
    return gates


def admission(g, fam, now=None):
    """Disk floor, memory floor (memory_pressure), no overlap of [now, now + max_minutes] with a forbidden window, and the
    common heavy lock held (non-blocking) for the whole refit. Returns the open lock file (released at process exit)."""
    import fcntl
    r = g['admission']
    if shutil.disk_usage(pathlib.Path.home()).free < r['internal_free_bytes_min']:
        raise SystemExit('internal disk below the floor')
    try:
        mp = subprocess.run(['memory_pressure'], capture_output=True, text=True, timeout=20).stdout
        free = int(mp.strip().splitlines()[-1].split(':')[-1].strip().rstrip('%'))
    except Exception:  # noqa: BLE001
        raise SystemExit('memory_pressure unavailable; refusing') from None
    if free < r['memory_free_percent_min']:
        raise SystemExit(f'memory free {free}% < {r["memory_free_percent_min"]}%')
    if window_overlap(r['forbidden_windows_kst'], g['families'][fam]['max_minutes'], now):
        raise SystemExit('the refit would overlap a forbidden window')
    lock = open(r['heavy_lock'], 'a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit('the common heavy lock is held by another job') from None
    return lock


def window_overlap(windows, minutes, now=None):
    now = (now or dt.datetime.now(KST)).replace(tzinfo=None); end = now + dt.timedelta(minutes=minutes)
    return any(dt.datetime.fromisoformat(a) < end and now < dt.datetime.fromisoformat(b) for a, b in windows)


def agreement(qa, qb):
    tv = 0.5 * np.abs(qa - qb).sum(1)
    kl = (qa * (np.log(np.clip(qa, 1e-15, 1)) - np.log(np.clip(qb, 1e-15, 1)))).sum(1)
    return {'rows': int(len(qa)), 'top1_agreement': float((qa.argmax(1) == qb.argmax(1)).mean()), 'tv_mean': float(tv.mean()),
            'tv_max': float(tv.max()), 'max_abs_diff': float(np.abs(qa - qb).max()), 'kl_mean_scored_to_refit': float(kl.mean())}


def gate_report(g, fam, LR, cfg, root, ev, qt, qm, qs, out):
    """A and C first (no label); B (reads the already scored E_PS labels) only when A and C pass."""
    a = g['families'][fam]['gates']
    z = np.load(root / 'final' / fam / 'eps-pred.npz', allow_pickle=False)
    if z['q_tau'].shape != qt.shape:
        raise SystemExit('scored / refit E_PS rows differ')
    # A: ensemble vs scored ensemble (before temperature) + every refit seed vs the scored ensemble + tempered max |dq|
    ens = agreement(z['q_tau1'], qm); tempered = agreement(z['q_tau'], qt)
    per_seed = [agreement(z['q_tau1'], q / q.sum(1, keepdims=True)) for q in qs]
    ga_pass = (ens['tv_mean'] <= a['A_ens_tv_mean_max'] and ens['top1_agreement'] >= a['A_ens_top1_min']
               and all(r['tv_mean'] <= a['A_seed_tv_mean_max'] and r['top1_agreement'] >= a['A_seed_top1_min'] for r in per_seed)
               and tempered['max_abs_diff'] <= a['A_tempered_max_abs_diff_max'])
    from pitcheezy.svc_lit import bundle as LB
    pin = pin_doc(g, fam, out)
    pin_path = out / 'bundle-pin.json'; pin_path.write_text(json.dumps(pin, indent=1) + '\n')
    bundle = LB.Bundle.load(pin_path, threads=g['predict_threads'])
    step = 4096
    qc = np.concatenate([bundle.predict(ev['state11'][i:i + step], ev['pitcher'][i:i + step], ev['batter'][i:i + step],
                                        {k: v[i:i + step] for k, v in ev['tok'].items()})['q'] for i in range(0, len(qt), step)])
    gc = {'max_abs_diff': float(np.abs(qc - qt).max()), 'top1_agreement': float((qc.argmax(1) == qt.argmax(1)).mean())}
    gc_pass = gc['max_abs_diff'] <= a['C_bundle_max_abs_diff']
    rep = {'A_agreement': {'ensemble_tau1': ens, 'per_seed_tau1': per_seed, 'tempered': tempered, 'pass': bool(ga_pass)},
           'C_bundle': {**gc, 'pass': bool(gc_pass)}, 'thresholds': a}
    if not (ga_pass and gc_pass):
        rep.update(pass_=False, B_eps_sanity={'skipped': 'A or C failed; E_PS labels not read'})
        rep['pass'] = rep.pop('pass_')
        return rep
    gb = eps_sanity(g, fam, LR, cfg, root, z, qt)
    gb_pass = gb['self_check_ok'] and abs(gb['nll_refit'] - gb['nll_scored']) <= a['B_abs_nll_diff_max']
    rep.update({'pass': bool(gb_pass), 'B_eps_sanity': {**gb, 'pass': bool(gb_pass)}})
    return rep


def eps_sanity(g, fam, LR, cfg, root, z, qt):
    sys.path.insert(0, str(REPO / 'scripts/svc_p9'))
    import eval_prediction as E
    s = cfg['scoring']; cr = pathlib.Path(s['cr_dir'])
    receipt = json.loads((cr / 'receipt.json').read_text())
    for name in ('candidate', 'baselines'):
        if sha(cr / f'{name}.jsonl') != receipt['stage_files_sha256'][name]:
            raise SystemExit(f'CR {name}.jsonl vs receipt')
    cand = [json.loads(l) for l in open(cr / 'candidate.jsonl')]
    base = {(b['game_pk'], b['n']): b for b in map(json.loads, open(cr / 'baselines.jsonl'))}
    y = np.array([E.label(base[(c['game_pk'], c['n'])]['actual']) for c in cand]); games = np.array([c['game_pk'] for c in cand])
    present = np.array([c['cand_R'] is not None for c in cand]); U = np.full(9, 1 / 9)
    if not (np.array_equal(z['eps_built'], present) and np.array_equal(z['eps_game_pk'], games)
            and np.array_equal(z['eps_n'], np.array([c['n'] for c in cand]))
            and all(a == json.dumps(c['service_key']) for a, c in zip(z['eps_service_key'], cand))):
        raise SystemExit('E_PS rows differ from CR')
    QR = np.array([c['cand_R'] if c['cand_R'] is not None else U for c in cand])
    QF = np.tile(U, (len(cand), 1)); QF[present] = qt
    QS = np.tile(U, (len(cand), 1)); QS[present] = z['q_tau']
    # self-check: R - scored recomputed here must equal EXP-LIT-BASE-001 result.json (same rows / labels / tool)
    res_p = root / 'score' / 'result.json'
    if sha(res_p) != g['source']['result_sha256']:
        raise SystemExit('result.json differs from its pin')
    res = json.loads(res_p.read_text())
    again = LR.contrast(E, QR, QS, y, games)
    want = res['contrasts'][f'R-{fam}']
    ok = (abs(again['point']['nll'] - want['point']['nll']) < 1e-9
          and all(abs(u - v) < 1e-9 for u, v in zip(again['ci95']['nll'], want['ci95']['nll'])))
    return {'rows': int(len(cand)), 'self_check_ok': bool(ok), 'R_minus_scored_recomputed': again, 'R_minus_scored_result_json': want,
            'nll_refit': LR.metrics(E, QF, y)['nll'], 'nll_scored': LR.metrics(E, QS, y)['nll'], 'nll_R': LR.metrics(E, QR, y)['nll'],
            'R_minus_refit': LR.contrast(E, QR, QF, y, games), 'scored_minus_refit': LR.contrast(E, QS, QF, y, games),
            'labels': 'CR attempt-001 (already scored by EXP-LIT-BASE-001; descriptive re-use, not a new test)'}


def pin_doc(g, fam, out):
    from pitcheezy.svc_lit import bundle as LB
    meta = json.loads((out / 'bundle.json').read_text())
    names = LB.files_for(meta['kind'], meta['seeds'])
    return {'format': 'svc-lit-bundle-v1', 'family': fam, 'dir': str(out), 'hp': meta['hp'], 'tau': meta['tau'], 'seeds': meta['seeds'],
            'state_encoder': g['source']['state_encoder'], 'files_sha256': {n: sha(out / n) for n in names}, 'refit_commit': meta['commit']}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('kind', choices=('refit', 'regate', 'pin', 'check', 'precheck'))
    ap.add_argument('--family', required=True)
    ap.add_argument('--config', type=pathlib.Path, default=REFIT)
    a = ap.parse_args(argv)
    if a.kind in ('refit', 'regate', 'pin') and pathlib.Path(a.config).resolve() != REFIT.resolve():
        raise SystemExit('refit / regate / pin use the registered configs/EXP-SVC-PRED-001/refit.json only')
    g, g_sha = load_refit(a.config)
    if a.kind == 'check':
        check_sources(g, a.family); print('sources OK')
    elif a.kind == 'precheck':
        print(json.dumps(precheck(g, a.family), indent=1))
    elif a.kind == 'refit':
        print(json.dumps(refit(g, g_sha, a.family), indent=1))
    elif a.kind == 'regate':
        print(json.dumps(regate(g, g_sha, a.family), indent=1))
    else:
        out = pathlib.Path(g['families'][a.family]['out_dir'])
        if not json.loads((out / 'refit.json').read_text())['pass']:
            raise SystemExit('refit gates did not pass')
        gate_pin = json.loads((out / 'bundle-pin.json').read_text())
        if pin_doc(g, a.family, out) != gate_pin:
            raise SystemExit('bundle files changed since the gates ran')
        print(json.dumps(gate_pin, indent=1))


if __name__ == '__main__':
    main()
