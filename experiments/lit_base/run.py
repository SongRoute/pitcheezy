"""EXP-LIT-BASE-001 실행기. configs/EXP-LIT-BASE-001.yaml만 읽는다(파라미터 하드코딩 없음).

단계(각각 새 프로세스, 산출물은 cfg.output_root/<stage>/...):
  prepare --phase tune|final   표 특징·ID 색인 캐시(라벨: package FIT 행만. E_PS 라벨은 읽지 않음)
  tune    --family F           FIT_tr로 학습 -> DEV 예측. 설정별 DEV NLL, 최고 설정, (신경망) seed 1·2 추가, 온도 tau(DEV)
  select                       모든 family의 tune 결과 -> 최고 기준 모델(DEV NLL, tau 적용) 고정 = best.json (E_PS 채점 전)
  final   --family F           FIT 전체로 다시 학습(선정 설정·epoch/rounds 그대로) -> E_PS 입력 예측(라벨 없음) -> tau 적용
  score                        모든 family final + best.json이 있을 때만. CR 행(라벨)과 R·C·표를 읽어 등록 규칙대로 채점(1회)
  driver                       공통 .heavy.lock을 쥐고 CPU 줄(TAB->MLR->GBT)과 MPS 줄(SEQ->TAK)을 병렬로 돌린다

usage: .venv/bin/python -B -m experiments.lit_base.run <stage> [--family F] [--phase P] [--synthetic-day DATE]
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import threading
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
CONFIG = REPO / 'configs/EXP-LIT-BASE-001.yaml'
KST = dt.timezone(dt.timedelta(hours=9))


IDS = ('EXP-LIT-BASE-001', 'EXP-EXPL-1121-A', 'EXP-EXPL-1121-A2', 'EXP-EXPL-1121-A3', 'EXP-EXPL-1121-A4', 'EXP-EXPL-1121-A4b', 'EXP-EXPL-1121-A5', 'EXP-EXPL-1121-B', 'EXP-EXPL-1121-FINAL')


def _expand(v):
    """Public repo: config paths are written with ${PITCHEEZY_*} variables (see docs/REPRODUCE.md); expand them here."""
    if isinstance(v, str):
        return os.path.expandvars(v)
    if isinstance(v, dict):
        return {k: _expand(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_expand(x) for x in v]
    return v


def load_cfg(path=CONFIG):
    import yaml
    raw = pathlib.Path(path).read_bytes()
    c = _expand(yaml.safe_load(raw)); c['_sha256'] = hashlib.sha256(raw).hexdigest(); c['_path'] = str(path)
    if c.get('id') not in IDS:
        raise SystemExit('wrong config')
    return c


def rpath(p):
    """Config paths: absolute as given, relative = repo root (server snapshot links: data/final-1124, runs)."""
    p = pathlib.Path(p)
    return p if p.is_absolute() else REPO / p


def git_state():
    snap = os.environ.get('PITCHEEZY_SNAPSHOT_COMMIT')
    if not snap and os.environ.get('GPU_JOB_DIR'):
        env = pathlib.Path(os.environ['GPU_JOB_DIR']) / 'job.env'
        if env.exists():
            for line in env.read_text().splitlines():
                k, _, v = line.partition('=')
                if k.strip() in ('GPU_COMMIT', 'COMMIT', 'PITCHEEZY_SNAPSHOT_COMMIT'):
                    snap = v.strip().strip('"\'')
    if not (REPO / '.git').exists() and snap:
        dirty = None
        if os.environ.get('GPU_JOB_DIR') and (pathlib.Path(os.environ['GPU_JOB_DIR']) / 'job.env').exists():
            for line in (pathlib.Path(os.environ['GPU_JOB_DIR']) / 'job.env').read_text().splitlines():
                k, _, v = line.partition('=')
                if k.strip() == 'DIRTY':
                    dirty = v.strip().strip('"\'').lower() not in ('0', 'false', 'no', '')
        if os.environ.get('PITCHEEZY_SNAPSHOT_DIRTY'):
            dirty = os.environ['PITCHEEZY_SNAPSHOT_DIRTY'].lower() not in ('0', 'false', 'no', '')
        return snap, dirty   # None (dirty state unknown) -> require_registered refuses   # gpu run snapshot: no .git, files = git ls-files at the commit (a5000 runner contract)
    try:
        h = subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(['git', '-C', str(REPO), 'status', '--porcelain', '--', 'experiments/lit_base', 'configs/EXP-LIT-BASE-001.yaml', 'configs/EXP-EXPL-1121-A.yaml', 'configs/EXP-EXPL-1121-A2.yaml', 'configs/EXP-EXPL-1121-A3.yaml', 'configs/EXP-EXPL-1121-A4.yaml', 'configs/EXP-EXPL-1121-A4b.yaml', 'configs/EXP-EXPL-1121-A5.yaml', 'configs/EXP-EXPL-1121-B.yaml', 'configs/EXP-EXPL-1121-FINAL.yaml',
                                     'scripts/lit_base'], capture_output=True, text=True).stdout.strip())
        return h, dirty
    except Exception:  # noqa: BLE001
        return None, None


def out_root(cfg, synthetic):
    r = pathlib.Path(os.environ['PITCHEEZY_OUT_ROOT']) if os.environ.get('PITCHEEZY_OUT_ROOT') else rpath(cfg['resources']['output_root'])
    lane = os.environ.get('PITCHEEZY_LANE')
    if lane:
        r = r / lane
    return r / ('dryrun' if synthetic else 'attempt-001')


def logger(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, 'a')
    def log(msg):
        line = f'{dt.datetime.now(KST).isoformat(timespec="seconds")} {msg}'
        print(line, flush=True); f.write(line + '\n'); f.flush()
    return log


def require_registered(cfg, synthetic):
    if synthetic:
        return
    if cfg.get('status') != 'REGISTERED' or cfg.get('execute') is not True:
        raise SystemExit(f'{cfg["id"]} is not REGISTERED/execute (independent review gate): real-data run refused')
    if pathlib.Path(cfg['_path']).resolve() != (REPO / 'configs' / f'{cfg["id"]}.yaml').resolve():
        raise SystemExit('real-data runs use only the committed default config path')
    h, dirty = git_state()
    if h is None or dirty is None:
        raise SystemExit('git state unavailable: real-data run refused')
    if dirty:
        raise SystemExit('uncommitted changes in experiment code/config: commit before a real-data run')


def same_config(cfg, path):
    """Every stage output must come from the same registered config bytes."""
    got = json.loads(pathlib.Path(path).read_text()).get('config_sha256')
    if got != cfg['_sha256']:
        raise SystemExit(f'config sha mismatch for {path}: {got} != {cfg["_sha256"]} (config changed between stages)')


def save_npy(path, arr):
    tmp = pathlib.Path(str(path) + '.tmp.npy'); np.save(tmp, arr); os.replace(tmp, path)


def admission(cfg, log):
    r = cfg['resources']
    st = shutil.disk_usage(str(pathlib.Path.home()))
    if st.free < r['internal_free_bytes_min']:
        raise SystemExit(f'internal disk free {st.free} < {r["internal_free_bytes_min"]}')
    try:
        mp = subprocess.run(['memory_pressure'], capture_output=True, text=True, timeout=20).stdout
        free = int(mp.strip().splitlines()[-1].split(':')[-1].strip().rstrip('%'))
    except Exception:  # noqa: BLE001
        free = None
    if free is not None and free < r['memory_free_percent_min']:
        raise SystemExit(f'memory free {free}% < {r["memory_free_percent_min"]}%')
    if in_window(cfg):
        raise SystemExit('inside a forbidden heavy-work window')
    log(f'admission ok: internal free {st.free / 2**30:.1f} GiB, memory free {free}%')


def in_window(cfg, now=None):
    now = now or dt.datetime.now(KST)
    for a, b in cfg['resources']['forbidden_windows_kst']:
        if dt.datetime.fromisoformat(a).replace(tzinfo=KST) <= now < dt.datetime.fromisoformat(b).replace(tzinfo=KST):
            return True
    return False


def sha(p):
    from .data import sha256_file
    return sha256_file(p)


# ---------------------------------------------------------------- prepare
def prepare(cfg, phase, synthetic, log):
    from . import data as D
    root = out_root(cfg, synthetic) / 'prepare' / phase
    if (root / 'DONE').exists():
        same_config(cfg, root / 'report.json'); log(f'prepare {phase}: DONE exists'); return root
    root.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    day = cfg['dry_run']['package_days'] if synthetic else None
    pk = D.load_package(cfg, check_sha=True, day_slice=day)
    enc_path = cfg['data']['package']['root'] + '/encoder.json'
    balls, strikes, stand, _ = D.raw_from_state(pk['state11'], enc_path)
    D.need(np.array_equal(balls, pk['balls']) and np.array_equal(strikes, pk['strikes']), 'state11 count == seq-inputs count')
    sp = cfg['splits']; date = pk['date']
    lab = pk['y'] >= 0
    if phase == 'tune':
        d64 = lambda v: np.datetime64(v)  # noqa: E731
        tr = lab & (date <= d64(sp['fit_tr_end'])); ev = lab & (date >= d64(sp['dev_start'])) & (date <= d64(sp['fit_window'][1]))
        if synthetic:
            days = sorted(set(date.tolist())); tr = lab & (date == days[0]); ev = lab & (date == days[-1])
        D.need(not (tr & ev).any() and tr.any() and ev.any(), 'tune split')
        E = dict(state11=pk['state11'][ev], pitcher=pk['pitcher'][ev], batter=pk['batter'][ev], balls=balls[ev], strikes=strikes[ev],
                 stand=stand[ev], tok={k: v[ev] for k, v in pk['tok'].items()}, y=pk['y'][ev], keys=pk['keys'][ev], game_pk=pk['game_pk'][ev])
    else:
        tr = lab.copy()
        if synthetic:
            days = sorted(set(date.tolist())); tr = lab & (date <= days[-1])
        eps = D.load_eps(cfg)   # label-free inputs; the dry run predicts them too (it never scores)
        b = eps['built']
        _, _, st_e, _ = D.raw_from_state(eps['state11'][b], enc_path)
        stand_e = np.zeros(len(b), np.int64); stand_e[b] = st_e
        E = dict(state11=eps['state11'][b], pitcher=eps['pitcher'][b], batter=eps['batter'][b], balls=eps['balls'][b], strikes=eps['strikes'][b],
                 stand=stand_e[b], tok={k: v[b] for k, v in eps['tok'].items()}, y=None, built=b,
                 game_pk=eps['game_pk'], n=eps['n'], service_key=eps['service_key'], w1_key=eps['w1_key'])
    T = dict(state11=pk['state11'][tr], pitcher=pk['pitcher'][tr], batter=pk['batter'][tr], balls=balls[tr], strikes=strikes[tr],
             stand=stand[tr], tok={k: v[tr] for k, v in pk['tok'].items()}, y=pk['y'][tr], keys=pk['keys'][tr], game_pk=pk['game_pk'][tr])
    a = cfg['features']['target_encoding_alpha']; folds = cfg['features']['target_encoding_folds']
    enc_tr = D.oof_encodings(a, folds, T['game_pk'], T['pitcher'], T['batter'], T['balls'], T['strikes'], T['stand'], T['y'])
    te = D.TargetEncoder(a).fit(T['pitcher'], T['batter'], T['balls'], T['strikes'], T['stand'], T['y'])
    enc_ev = te.transform(E['pitcher'], E['batter'], E['balls'], E['strikes'], E['stand'])
    Xtr = D.tabular(T['state11'], T['balls'], T['strikes'], enc_tr, T['tok'])
    Xev = D.tabular(E['state11'], E['balls'], E['strikes'], enc_ev, E['tok'])
    pv = D.vocab(T['pitcher']); bv = D.vocab(T['batter'])
    save = dict(Xtr=Xtr, ytr=T['y'], Xev=Xev, state11_tr=T['state11'], state11_ev=E['state11'],
                pidx_tr=D.index(T['pitcher'], pv), bidx_tr=D.index(T['batter'], bv), pidx_ev=D.index(E['pitcher'], pv), bidx_ev=D.index(E['batter'], bv),
                pitcher_tr=T['pitcher'], batter_tr=T['batter'], balls_tr=T['balls'], strikes_tr=T['strikes'], stand_tr=T['stand'],
                pitcher_ev=E['pitcher'], batter_ev=E['batter'], balls_ev=E['balls'], strikes_ev=E['strikes'], stand_ev=E['stand'],
                **{f'tok_tr_{k}': v for k, v in T['tok'].items()}, **{f'tok_ev_{k}': v for k, v in E['tok'].items()})
    if phase == 'tune':
        save['yev'] = E['y']; save['keys_ev'] = E['keys']
    else:
        for k in ('built', 'game_pk', 'n', 'service_key', 'w1_key'):
            save[f'eps_{k}'] = E[k]
    np.savez(root / 'cache.npz', **save)
    rep = {'phase': phase, 'synthetic': synthetic, 'train_rows': int(tr.sum()), 'eval_rows': int(len(E['state11'])),
           'train_date_range': [str(date[tr].min()), str(date[tr].max())], 'feature_dim': int(Xtr.shape[1]),
           'pitcher_vocab': int(len(pv)), 'batter_vocab': int(len(bv)),
           'eval_unknown_pitcher_rows': int((save['pidx_ev'] == 0).sum()), 'eval_unknown_batter_rows': int((save['bidx_ev'] == 0).sum()),
           'eps_labels_read': False, 'seconds': time.monotonic() - t0, 'config_sha256': cfg['_sha256'], 'commit': git_state(),
           'cache_sha256': sha(root / 'cache.npz')}
    if phase == 'tune':
        rep['eval_date_range'] = [str(date[ev].min()), str(date[ev].max())]
    (root / 'report.json').write_text(json.dumps(rep, indent=1) + '\n'); (root / 'DONE').write_text('')
    log(f'prepare {phase}: {rep}')
    return root


def synthetic_eps(pk, lab, balls, strikes, stand):
    """dry run only: the last package day stands in for E_PS (no real E_PS byte)."""
    days = sorted(set(pk['date'].tolist())); m = lab & (pk['date'] == days[-1])
    b = np.ones(m.sum(), bool); b[::37] = False
    return dict(built=b, state11=pk['state11'][m], pitcher=pk['pitcher'][m], batter=pk['batter'][m], balls=balls[m], strikes=strikes[m],
                tok={k: v[m] for k, v in pk['tok'].items()}, game_pk=pk['game_pk'][m], n=np.arange(m.sum()),
                service_key=np.array(['s'] * m.sum()), w1_key=np.array(['w'] * m.sum()))


def load_cache(cfg, phase, synthetic):
    root = out_root(cfg, synthetic) / 'prepare' / phase
    same_config(cfg, root / 'report.json'); rep = json.loads((root / 'report.json').read_text())
    if sha(root / 'cache.npz') != rep['cache_sha256']:
        raise SystemExit('cache sha')
    z = dict(np.load(root / 'cache.npz', allow_pickle=False))
    tok = lambda s: {k: z[f'tok_{s}_{k}'] for k in ('valid', 'type', 'out', 'balls', 'strikes', 'same')}  # noqa: E731
    tr = dict(X=z['Xtr'], y=z['ytr'], state11=z['state11_tr'], pidx=z['pidx_tr'], bidx=z['bidx_tr'], tok=tok('tr'),
              pitcher=z['pitcher_tr'], batter=z['batter_tr'], balls=z['balls_tr'], strikes=z['strikes_tr'], stand=z['stand_tr'])
    ev = dict(X=z['Xev'], y=z.get('yev'), state11=z['state11_ev'], pidx=z['pidx_ev'], bidx=z['bidx_ev'], tok=tok('ev'),
              pitcher=z['pitcher_ev'], batter=z['batter_ev'], balls=z['balls_ev'], strikes=z['strikes_ev'], stand=z['stand_ev'])
    ev['eps'] = {k[4:]: z[k] for k in z if k.startswith('eps_')}
    return tr, ev, rep


# ---------------------------------------------------------------- fit one config
def fit_one(cfg, fam, hp, tr, ev, seed, log, dev=True, budget=None):
    from . import models as M
    f = cfg['families'][fam]; kind = f['kind']; th = cfg['resources']['threads'].get(f['lane'], 2)
    if kind == 'table':
        return M.tab_predict(tr, ev, hp)
    if kind == 'mlr':
        return M.mlr_fit_predict(tr['X'], tr['y'], tr['pidx'], ev['X'], ev['pidx'], hp, seed, th, log)
    if kind == 'gbt':
        if dev:
            return M.gbt_fit_predict(tr['X'], tr['y'], ev['X'], hp, seed, th, log, dev=(ev['X'], ev['y']))
        return M.gbt_fit_predict(tr['X'], tr['y'], ev['X'], hp, seed, th, log, rounds=budget)
    if kind == 'seq':
        device = os.environ.get('PITCHEEZY_DEVICE') or f.get('device', cfg['resources'].get('seq_device', 'cpu'))
        if dev:
            return M.seq_fit_predict(tr, ev, hp, seed, device, th, log, dev=ev, max_seconds=f.get('max_seconds_per_fit'))
        return M.seq_fit_predict(tr, ev, hp, seed, device, th, log, epochs=budget)
    raise ValueError(kind)


def tune(cfg, fam, synthetic, log):
    from . import models as M
    root = out_root(cfg, synthetic) / 'tune' / fam
    if (root / 'DONE').exists():
        same_config(cfg, root / 'selection.json'); log(f'tune {fam}: DONE exists'); return
    root.mkdir(parents=True, exist_ok=True)
    tr, ev, rep = load_cache(cfg, 'tune', synthetic)
    f = cfg['families'][fam]; base = f.get('base', {}); configs = f['tune']['configs']
    if synthetic:
        configs = configs[:cfg['dry_run']['configs_per_family']]
        base = {**base, **cfg['dry_run'].get('override', {}).get(f['kind'], {})}
    results = []; t0 = time.monotonic()
    for i, c in enumerate(configs):
        hp = {**base, **c}
        done = root / f'config-{i:02d}.json'
        if done.exists():
            same_config(cfg, done); results.append(json.loads(done.read_text())); continue
        s0 = time.monotonic()
        q, info = fit_one(cfg, fam, hp, tr, ev, f['tune']['seed'], log)
        r = {'index': i, 'hp': hp, 'dev_nll': M.nll(q, ev['y']), 'info': {k: v for k, v in info.items() if k != 'trace'},
             'trace': info.get('trace'), 'seconds': time.monotonic() - s0, 'config_sha256': cfg['_sha256'],
             'hit_cap': bool((info.get('best_iteration') or 0) >= hp.get('max_rounds', 10**9) or (info.get('best_epoch') or 0) >= hp.get('max_epochs', 10**9)),
             'wall_budget_stop': any(t.get('stopped') == 'wall_budget' for t in (info.get('trace') or []))}
        save_npy(root / f'config-{i:02d}-dev.npy', q.astype(np.float32))
        done.write_text(json.dumps(r, indent=1) + '\n'); results.append(r)
        log(f'tune {fam} config {i} dev_nll {r["dev_nll"]:.5f} {r["seconds"]:.0f}s')
    best = min(results, key=lambda r: r['dev_nll'])
    q0 = np.load(root / f'config-{best["index"]:02d}-dev.npy').astype(np.float64)
    budget = best['info'].get('best_iteration') or best['info'].get('best_epoch')
    qs = [q0]; extra = []
    for s in f['final']['seeds'][1:] if f['kind'] == 'seq' else []:
        q, info = fit_one(cfg, fam, best['hp'], tr, ev, s, log, dev=False, budget=budget)
        save_npy(root / f'best-seed{s}-dev.npy', q.astype(np.float32)); qs.append(q)
        extra.append({'seed': s, 'dev_nll': M.nll(q, ev['y']), 'epochs': budget})
    qm = np.mean(qs, 0); qm /= qm.sum(1, keepdims=True)
    tau = M.fit_temperature(qm, ev['y']) if f.get('temperature', True) else 1.0
    sel = {'family': fam, 'best_index': best['index'], 'hp': best['hp'], 'budget': budget, 'seeds_dev': [f['tune']['seed']] + [e['seed'] for e in extra],
           'dev_nll_best_seed0': best['dev_nll'], 'extra_seeds': extra, 'dev_nll_ensemble_tau1': M.nll(qm, ev['y']),
           'tau': tau, 'dev_nll_ensemble_tau': M.nll(M.temper(qm, tau), ev['y']), 'dev_rows': int(len(ev['y'])),
           'dev_top1': float((qm.argmax(1) == ev['y']).mean()), 'all_configs': [{'index': r['index'], 'dev_nll': r['dev_nll'], 'seconds': r['seconds'],
           'hit_cap': r.get('hit_cap'), 'wall_budget_stop': r.get('wall_budget_stop'),
           **({'budget': r['info'].get('best_iteration') or r['info'].get('best_epoch')} if r['info'] else {})} for r in results],
           'seconds_total': time.monotonic() - t0, 'config_sha256': cfg['_sha256'], 'commit': git_state()}
    np.save(root / 'selected-dev.npy', qm.astype(np.float32))
    (root / 'selection.json').write_text(json.dumps(sel, indent=1) + '\n'); (root / 'DONE').write_text('')
    log(f'tune {fam} selected {sel["best_index"]} dev_nll(tau) {sel["dev_nll_ensemble_tau"]:.5f} tau {tau:.4f}')


def select(cfg, synthetic, log):
    root = out_root(cfg, synthetic)
    if (root / 'best.json').exists():
        same_config(cfg, root / 'best.json'); log('select: best.json exists (never overwritten)'); return
    fams = [f for f, v in cfg['families'].items() if v.get('enabled', True)]
    for f in fams:
        same_config(cfg, root / 'tune' / f / 'selection.json')
    sels = {f: json.loads((root / 'tune' / f / 'selection.json').read_text()) for f in fams}
    best = best_family(sels)
    out = {'rule': cfg['decision']['best_baseline_rule'], 'best_family': best,
           'dev_nll_tau': {f: s['dev_nll_ensemble_tau'] for f, s in sels.items()}, 'written_before_eps_scoring': True,
           'eps_labels_read': False, 'at': dt.datetime.now(KST).isoformat(), 'config_sha256': cfg['_sha256'], 'commit': git_state()}
    (root / 'best.json').write_text(json.dumps(out, indent=1) + '\n'); log(f'select: {out}')


def best_family(sels):
    return min(sorted(sels), key=lambda f: sels[f]['dev_nll_ensemble_tau'])


def final(cfg, fam, synthetic, log):
    from . import models as M
    root = out_root(cfg, synthetic) / 'final' / fam
    if (root / 'DONE').exists():
        same_config(cfg, root / 'final.json'); log(f'final {fam}: DONE exists'); return
    root.mkdir(parents=True, exist_ok=True)
    same_config(cfg, out_root(cfg, synthetic) / 'tune' / fam / 'selection.json'); same_config(cfg, out_root(cfg, synthetic) / 'best.json')
    sel = json.loads((out_root(cfg, synthetic) / 'tune' / fam / 'selection.json').read_text())
    tr, ev, rep = load_cache(cfg, 'final', synthetic)
    f = cfg['families'][fam]; seeds = f['final']['seeds'] if f['kind'] == 'seq' else f['final']['seeds'][:1]
    qs = []; infos = []
    for s in seeds:
        p = root / f'seed{s}-eps.npy'
        if p.exists():
            qs.append(np.load(p).astype(np.float64)); continue
        s0 = time.monotonic()
        q, info = fit_one(cfg, fam, sel['hp'], tr, ev, s, log, dev=False, budget=sel['budget'])
        save_npy(p, q.astype(np.float64)); qs.append(q); infos.append({'seed': s, 'seconds': time.monotonic() - s0, 'trace': info.get('trace')})
    qm = np.mean(qs, 0); qm /= qm.sum(1, keepdims=True)
    qt = M.temper(qm, sel['tau'])
    eps = ev['eps']
    np.savez(root / 'eps-pred.npz', q_tau=qt, q_tau1=qm, **{f'eps_{k}': v for k, v in eps.items()})
    r = {'family': fam, 'seeds': seeds, 'tau': sel['tau'], 'hp': sel['hp'], 'budget': sel['budget'], 'train_rows': rep['train_rows'],
         'eps_built_rows': int(len(qt)), 'eps_labels_read': False, 'fits': infos, 'pred_sha256': sha(root / 'eps-pred.npz'),
         'config_sha256': cfg['_sha256'], 'commit': git_state()}
    (root / 'final.json').write_text(json.dumps(r, indent=1) + '\n'); (root / 'DONE').write_text('')
    log(f'final {fam} done ({len(qt)} rows)')


# ---------------------------------------------------------------- score (reads E_PS labels; once)
def score(cfg, synthetic, log):
    root = out_root(cfg, synthetic)
    if (root / 'score' / 'result.json').exists():
        raise SystemExit('score already DONE: E_PS is scored exactly once')
    fams = [f for f, v in cfg['families'].items() if v.get('enabled', True)]
    for f in fams:
        if not (root / 'final' / f / 'DONE').exists():
            raise SystemExit(f'final {f} not DONE: every registered baseline must finish before scoring (no selective reporting)')
    same_config(cfg, root / 'best.json'); best = json.loads((root / 'best.json').read_text())
    sels = {}
    for f in fams:
        same_config(cfg, root / 'tune' / f / 'selection.json'); same_config(cfg, root / 'final' / f / 'final.json')
        sels[f] = json.loads((root / 'tune' / f / 'selection.json').read_text())
        if sha(root / 'final' / f / 'eps-pred.npz') != json.loads((root / 'final' / f / 'final.json').read_text())['pred_sha256']:
            raise SystemExit(f'{f}: eps-pred.npz sha differs from final.json')
    if best_family(sels) != best['best_family'] or set(best['dev_nll_tau']) != set(fams):
        raise SystemExit('best.json does not match the DEV selections of the enabled families')
    sys.path.insert(0, str(REPO / 'scripts/svc_p9'))
    import eval_prediction as E
    s = cfg['scoring']; cr = pathlib.Path(s['cr_dir'])
    receipt = json.loads((cr / 'receipt.json').read_text())
    for name in ('candidate', 'baselines'):
        if sha(cr / f'{name}.jsonl') != receipt['stage_files_sha256'][name]:
            raise SystemExit(f'CR {name}.jsonl vs receipt')
    cand = [json.loads(l) for l in open(cr / 'candidate.jsonl')]; base = {(b['game_pk'], b['n']): b for b in map(json.loads, open(cr / 'baselines.jsonl'))}
    for c in cand:
        b0 = base[(c['game_pk'], c['n'])]
        if b0['service_key'] != c['service_key'] or (c['cand_C'] is None) != (c['cand_R'] is None) or E.label(b0['actual']) is None:
            raise SystemExit('CR row check (service key / R-C omitted together / labelled) failed')
    y = np.array([E.label(base[(c['game_pk'], c['n'])]['actual']) for c in cand]); games = np.array([c['game_pk'] for c in cand])
    present = np.array([c['cand_R'] is not None for c in cand]); U = np.full(9, 1 / 9)
    Q = {'R': np.array([c['cand_R'] if c['cand_R'] is not None else U for c in cand]),
         'C': np.array([c['cand_C'] if c['cand_C'] is not None else U for c in cand]),
         'T_frozen_sept': np.array([base[(c['game_pk'], c['n'])]['freq'] for c in cand])}
    if len(cand) != s['rows'] or len(set(games.tolist())) != s['games'] or 849799 in set(games.tolist()) or (~present).sum() != s['omitted_rows']:
        raise SystemExit('E_PS rows / games / omitted rows differ from the registration')
    for f in fams:
        z = np.load(root / 'final' / f / 'eps-pred.npz', allow_pickle=False)
        key_ok = (np.array_equal(z['eps_game_pk'], games) and np.array_equal(z['eps_n'], np.array([c['n'] for c in cand]))
                  and all(a == json.dumps(c['service_key']) for a, c in zip(z['eps_service_key'], cand))
                  and all(a == json.dumps(c['w1_key']) for a, c in zip(z['eps_w1_key'], cand))
                  and np.array_equal(z['eps_built'], present))
        if not key_ok:
            raise SystemExit(f'{f}: E_PS rows / omitted rows differ from CR')
        for v in ('q_tau', 'q_tau1'):
            q = np.tile(U, (len(cand), 1)); q[present] = z[v]; Q[f if v == 'q_tau' else f'{f}@tau1'] = q
    met = {k: metrics(E, q, y) for k, q in Q.items()}
    con = {}
    for f in fams:
        for v in (f, f'{f}@tau1'):
            con[f'R-{v}'] = contrast(E, Q['R'], Q[v], y, games)
            con[f'{v}-C'] = contrast(E, Q[v], Q['C'], y, games)
            con[f'{v}-T_frozen_sept'] = contrast(E, Q[v], Q['T_frozen_sept'], y, games)
    b = best['best_family']; primary = con[f'R-{b}']
    lo, hi = primary['ci95']['nll']
    verdict = cfg['decision']['verdicts']['r_better'] if hi < 0 else (cfg['decision']['verdicts']['baseline_better'] if lo > 0 else cfg['decision']['verdicts']['no_difference'])
    out = {'experiment': 'EXP-LIT-BASE-001', 'best_baseline_from_DEV': b, 'primary': {'contrast': f'R - {b}', **primary}, 'verdict': verdict,
           'metrics': met, 'contrasts': con, 'rows': len(cand), 'omitted_uniform': int((~present).sum()), 'floor_eps': E.EPS,
           'bootstrap': {'unit': 'game_pk', 'replicates': E.REPLICATES, 'seed': E.SEED}, 'best_json_sha256': sha(root / 'best.json'),
           'cr_receipt_sha256': sha(cr / 'receipt.json'), 'config_sha256': cfg['_sha256'], 'commit': git_state(),
           'labels_read_here_first_time': True}
    (root / 'score').mkdir(parents=True, exist_ok=True)
    (root / 'score' / 'result.json').write_text(json.dumps(out, indent=1, allow_nan=False) + '\n')
    log(f'score: primary R-{b} {primary["point"]["nll"]:+.5f} [{lo:+.5f},{hi:+.5f}] -> {verdict}')


def metrics(E, Q, y):
    F = E.floor(Q); parts = E.row_parts(F, y); p = E.pooled(parts)
    top3 = float(np.mean((np.argsort(-F, 1)[:, :3] == y[:, None]).any(1)))
    cw = []
    for c in range(9):
        pc = F[:, c]; o = (y == c).astype(float); bb = np.minimum((pc * 10).astype(int), 9); tot = 0.0
        for k in range(10):
            m = bb == k
            if m.any():
                tot += abs(pc[m].mean() - o[m].mean()) * m.sum()
        cw.append(tot / len(y))
    return {**p, 'top3': top3, 'ece10_classwise_mean': float(np.mean(cw))}


def contrast(E, Qa, Qb, y, games):
    d = E.paired_contrast(E.floor(Qa), E.floor(Qb), y, games)
    return {'point': d['point'], 'ci95': d['ci95']}


# ---------------------------------------------------------------- driver (holds the common heavy lock)
def driver(cfg, synthetic, log):
    require_registered(cfg, synthetic)
    root = out_root(cfg, synthetic); root.mkdir(parents=True, exist_ok=True)
    lock = open(cfg['resources']['heavy_lock'], 'a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('common heavy lock held by another process: not started')
    (root / 'driver.pid').write_text(str(os.getpid()))
    py = sys.executable; extra = ['--synthetic'] if synthetic else []
    def stage(args, name):
        while True:
            while in_window(cfg):
                log(f'{name}: waiting (forbidden window)'); time.sleep(60)
            try:
                admission(cfg, log)
            except SystemExit as e:
                log(f'{name}: admission refused ({e}); retry in 5 min'); time.sleep(300); continue
            lf = open(root / 'logs' / f'{name}.log', 'a')
            p = subprocess.Popen([py, '-B', '-m', 'experiments.lit_base.run', *args, *extra], cwd=str(REPO), stdout=lf, stderr=subprocess.STDOUT)
            (root / 'pids' / f'{name}.pid').write_text(str(p.pid)); log(f'{name}: started pid {p.pid}')
            interrupted = False
            while p.poll() is None:
                if in_window(cfg):   # own child only (pid we started); recorder/booth processes are never signalled
                    p.terminate(); p.wait(); interrupted = True; log(f'{name}: interrupted for forbidden window (own child {p.pid})'); break
                time.sleep(20)
            (root / 'pids' / f'{name}.pid').unlink(missing_ok=True)
            if interrupted:
                continue
            if p.returncode != 0:
                raise RuntimeError(f'{name} failed rc {p.returncode}')
            log(f'{name}: done'); return
    (root / 'logs').mkdir(exist_ok=True); (root / 'pids').mkdir(exist_ok=True)
    fams = [f for f, v in cfg['families'].items() if v.get('enabled', True)]
    stage(['prepare', '--phase', 'tune'], 'prepare-tune'); stage(['prepare', '--phase', 'final'], 'prepare-final')
    lanes = {}
    for f in fams:
        lanes.setdefault(cfg['families'][f]['lane'], []).append(f)
    errors = []
    def lane_tune(fs):
        try:
            for f in fs:
                stage(['tune', '--family', f], f'tune-{f}')
        except BaseException as e:  # noqa: BLE001  (SystemExit too: a lane must never vanish silently)
            errors.append(repr(e))
    th = [threading.Thread(target=lane_tune, args=(fs,)) for fs in lanes.values()]
    [t.start() for t in th]; [t.join() for t in th]
    if errors:
        raise SystemExit(f'tune errors {errors}')
    stage(['select'], 'select')
    def lane_final(fs):
        try:
            for f in fs:
                stage(['final', '--family', f], f'final-{f}')
        except BaseException as e:  # noqa: BLE001  (SystemExit too: a lane must never vanish silently)
            errors.append(repr(e))
    th = [threading.Thread(target=lane_final, args=(fs,)) for fs in lanes.values()]
    [t.start() for t in th]; [t.join() for t in th]
    if errors:
        raise SystemExit(f'final errors {errors}')
    if synthetic:
        stage(['score'], 'score') if cfg['dry_run'].get('score', False) else None
    missing = [f for f in fams if not (root / 'final' / f / 'DONE').exists()]
    if missing:
        raise SystemExit(f'driver: final not DONE for {missing}')
    log('driver: all fits DONE (score is a separate, explicit step)')
    (root / 'driver.pid').unlink(missing_ok=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=('prepare', 'tune', 'select', 'final', 'score', 'driver'))
    ap.add_argument('--family'); ap.add_argument('--phase', choices=('tune', 'final')); ap.add_argument('--synthetic', action='store_true')
    ap.add_argument('--config', default=str(CONFIG))
    a = ap.parse_args(argv)
    cfg = load_cfg(a.config)
    if cfg['id'] != 'EXP-LIT-BASE-001':
        raise SystemExit('run.py stages are EXP-LIT-BASE-001 only; EXPL uses experiments.lit_base.expl')
    if a.stage != 'driver':
        require_registered(cfg, a.synthetic)
    log = logger(out_root(cfg, a.synthetic) / 'logs' / 'run.log')
    if a.stage == 'score' and a.synthetic:
        raise SystemExit('score never runs on the dry-run slice (it reads E_PS labels)')
    {'prepare': lambda: prepare(cfg, a.phase, a.synthetic, log), 'tune': lambda: tune(cfg, a.family, a.synthetic, log),
     'select': lambda: select(cfg, a.synthetic, log), 'final': lambda: final(cfg, a.family, a.synthetic, log),
     'score': lambda: score(cfg, a.synthetic, log), 'driver': lambda: driver(cfg, a.synthetic, log)}[a.stage]()


if __name__ == '__main__':
    main()
