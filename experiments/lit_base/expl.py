"""EXP-EXPL-1121 확장 탐색 실행기 (LIT-BASE 코드·분할·선택 규칙 승계). configs/EXP-EXPL-1121.yaml만 읽는다.

서버(A5000, GPU 2·3·4)와 맥의 일 분리(등록 사항):
  서버  lane --families F[,G]   : prepare(tune, fit) -> tune(FIT_tr -> DEV 선정, seed 묶음, tau) -> fitw(FIT 전체 재학습, .pt 저장)
                                   E_PS 입력·라벨은 서버에 없다.
  맥    lane --families X-GBT --mac : 같은 순서(GBT는 prepare final로 학습; booster·표적 인코딩 표 저장), 공통 heavy lock, 금지 시간 대기
  맥    preq (EXPL-B)           : 풀(A ∪ X-DATA ∪ LIT-BASE B-SEQ) DEV 1위 위에 U1 - DEV 안에서 엄격히 이른 날짜만 반영
  맥    select                  : A·B = 묶음 안 provisional, FINAL = 풀 전체에서 DEV로 B*_new 고정(best.json)
  맥    predict (FINAL)         : B*_new(또는 U1 바탕 팔) 한 팔만 E_PS 입력(라벨 없음) 예측
  맥    score (FINAL)           : 1회. R - B*_new, B*_new - B-SEQ(LIT-BASE), B*_new - C, top1 변화

usage: python -B -m experiments.lit_base.expl <stage> [--families A,B] [--mac] [--synthetic]
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import math
import os
import pathlib
import sys
import time

import numpy as np

from . import data as D
from . import models as M
from . import run as R

CONFIGS = {'A': R.REPO / 'configs/EXP-EXPL-1121-A.yaml', 'A2': R.REPO / 'configs/EXP-EXPL-1121-A2.yaml', 'A3': R.REPO / 'configs/EXP-EXPL-1121-A3.yaml', 'A4': R.REPO / 'configs/EXP-EXPL-1121-A4.yaml', 'A4b': R.REPO / 'configs/EXP-EXPL-1121-A4b.yaml', 'A5': R.REPO / 'configs/EXP-EXPL-1121-A5.yaml', 'B': R.REPO / 'configs/EXP-EXPL-1121-B.yaml', 'FINAL': R.REPO / 'configs/EXP-EXPL-1121-FINAL.yaml'}


def fams(cfg, kinds=None, include_preq=True):
    return [f for f, v in cfg.get('families', {}).items() if v.get('enabled', True) and (kinds is None or v['kind'] in kinds)
            and (include_preq or v['kind'] != 'preq')]


def cache_dir(cfg, syn, data, phase, tabular=False):
    return R.out_root(cfg, syn) / 'prepare' / (data + ('-tab' if tabular else '')) / phase


# ---------------------------------------------------------------- prepare
def prepare(cfg, data, phase, syn, log, tabular=False):
    """phase tune: train FIT_tr(+hist) / eval DEV.  fit: train FIT(+hist), no eval.  final (base only, Mac): train FIT / eval E_PS."""
    if phase == 'final' and cfg['id'] != 'EXP-EXPL-1121-FINAL':
        raise SystemExit(f'prepare final reads E_PS inputs: only the EXPL-FINAL registration may run it, not {cfg["id"]}')
    root = cache_dir(cfg, syn, data, phase, tabular)
    if (root / 'DONE').exists():
        R.same_config(cfg, root / 'report.json')
        log(f'prepare {data}/{phase}: DONE exists'); return
    root.mkdir(parents=True, exist_ok=True); t0 = time.monotonic()
    day = cfg['dry_run']['package_days'] if syn else None
    pk = D.load_package(cfg, check_sha=True, day_slice=day)
    pk['is_hist'] = np.zeros(len(pk['y']), bool)
    if data == 'plus':
        pk = D.load_hist(cfg, pk, day_slice=cfg['dry_run'].get('hist_days') if syn else None)
    enc_path = str(D.resolve(cfg['data']['package']['root'])) + '/encoder.json'
    balls, strikes, stand, _ = D.raw_from_state(pk['state11'], enc_path)
    D.need(np.array_equal(balls, pk['balls']) and np.array_equal(strikes, pk['strikes']), 'state11 count == source count')
    sp = cfg['splits']; date = pk['date']; lab = pk['y'] >= 0; h = pk['is_hist']; d64 = np.datetime64
    if phase == 'tune':
        tr = lab & (h | (date <= d64(sp['fit_tr_end']))); ev = lab & ~h & (date >= d64(sp['dev_start'])) & (date <= d64(sp['fit_window'][1]))
        if syn:
            days = sorted(set(date[~h].tolist())); tr = lab & (h | (date == days[0])); ev = lab & ~h & (date > days[0])
    else:
        tr = lab.copy(); ev = np.zeros(len(lab), bool)
    D.need(tr.any() and not (tr & ev).any() and (phase != 'tune' or ev.any()), 'split')
    sel = lambda m: dict(state11=pk['state11'][m], pitcher=pk['pitcher'][m], batter=pk['batter'][m], balls=balls[m], strikes=strikes[m],  # noqa: E731
                         stand=stand[m], tok={k: v[m] for k, v in pk['tok'].items()}, y=pk['y'][m], game_pk=pk['game_pk'][m], date=pk['date'][m],
                         is_hist=h[m])
    T = sel(tr)
    if phase == 'final':
        D.need(data == 'base', 'final cache (E_PS features) is for base-data tabular families only')
        eps = D.load_eps(cfg); b = eps['built']
        _, _, st_e, _ = D.raw_from_state(eps['state11'][b], enc_path)
        E = dict(state11=eps['state11'][b], pitcher=eps['pitcher'][b], batter=eps['batter'][b], balls=eps['balls'][b], strikes=eps['strikes'][b],
                 stand=st_e, tok={k: v[b] for k, v in eps['tok'].items()}, y=None)
    elif phase == 'tune':
        E = sel(ev)
    else:
        E = None
    pv = D.vocab(T['pitcher']); bv = D.vocab(T['batter'])
    save = dict(ytr=T['y'], state11_tr=T['state11'], pidx_tr=D.index(T['pitcher'], pv), bidx_tr=D.index(T['batter'], bv),
                pitcher_tr=T['pitcher'], batter_tr=T['batter'], balls_tr=T['balls'], strikes_tr=T['strikes'], stand_tr=T['stand'],
                is_hist_tr=T['is_hist'], pvocab=pv, bvocab=bv, **{f'tok_tr_{k}': v for k, v in T['tok'].items()})
    tabular = bool(tabular) and data == 'base'
    if tabular:
        a = cfg['features']['target_encoding_alpha']; folds = cfg['features']['target_encoding_folds']
        enc_tr = D.oof_encodings(a, folds, T['game_pk'], T['pitcher'], T['batter'], T['balls'], T['strikes'], T['stand'], T['y'])
        save['Xtr'] = D.tabular(T['state11'], T['balls'], T['strikes'], enc_tr, T['tok'])
    if E is not None:
        save.update(state11_ev=E['state11'], pidx_ev=D.index(E['pitcher'], pv), bidx_ev=D.index(E['batter'], bv), pitcher_ev=E['pitcher'],
                    batter_ev=E['batter'], balls_ev=E['balls'], strikes_ev=E['strikes'], stand_ev=E['stand'], **{f'tok_ev_{k}': v for k, v in E['tok'].items()})
        if tabular:
            te = D.TargetEncoder(a).fit(T['pitcher'], T['batter'], T['balls'], T['strikes'], T['stand'], T['y'])
            save['Xev'] = D.tabular(E['state11'], E['balls'], E['strikes'], te.transform(E['pitcher'], E['batter'], E['balls'], E['strikes'], E['stand']), E['tok'])
        if phase == 'tune':
            save.update(yev=E['y'], date_ev=E['date'].astype('datetime64[D]').astype(np.int64))
    np.savez(root / 'cache.npz', **save)
    rep = {'data': data, 'phase': phase, 'synthetic': syn, 'train_rows': int(tr.sum()), 'train_hist_rows': int(T['is_hist'].sum()),
           'eval_rows': int(0 if E is None else len(E['state11'])), 'train_2026_date_range': [str(T['date'][~T['is_hist']].min()), str(T['date'][~T['is_hist']].max())],
           'pitcher_vocab': int(len(pv)), 'batter_vocab': int(len(bv)), 'tabular': tabular, 'eps_labels_read': False, 'seconds': time.monotonic() - t0,
           'config_sha256': cfg['_sha256'], 'commit': R.git_state(), 'cache_sha256': R.sha(root / 'cache.npz')}
    if E is not None:
        rep['eval_unknown_pitcher_rows'] = int((save['pidx_ev'] == 0).sum())
    (root / 'report.json').write_text(json.dumps(rep, indent=1) + '\n'); (root / 'DONE').write_text('')
    log(f'prepare {data}/{phase}: {rep}')


def load_cache(cfg, syn, data, phase, tabular=False):
    root = cache_dir(cfg, syn, data, phase, tabular); R.same_config(cfg, root / 'report.json')
    rep = json.loads((root / 'report.json').read_text())
    if R.sha(root / 'cache.npz') != rep['cache_sha256']:
        raise SystemExit('cache sha')
    z = dict(np.load(root / 'cache.npz', allow_pickle=False))
    tok = lambda s: {k: z[f'tok_{s}_{k}'] for k in ('valid', 'type', 'out', 'balls', 'strikes', 'same')}  # noqa: E731
    tr = dict(X=z.get('Xtr'), y=z['ytr'], state11=z['state11_tr'], pidx=z['pidx_tr'], bidx=z['bidx_tr'], tok=tok('tr'), pitcher=z['pitcher_tr'],
              batter=z['batter_tr'], balls=z['balls_tr'], strikes=z['strikes_tr'], stand=z['stand_tr'], is_hist=z['is_hist_tr'])
    ev = None
    if 'state11_ev' in z:
        ev = dict(X=z.get('Xev'), y=z.get('yev'), state11=z['state11_ev'], pidx=z['pidx_ev'], bidx=z['bidx_ev'], tok=tok('ev'), pitcher=z['pitcher_ev'],
                  batter=z['batter_ev'], balls=z['balls_ev'], strikes=z['strikes_ev'], stand=z['stand_ev'], date=z.get('date_ev'))
    return tr, ev, rep, z['pvocab'], z['bvocab']


def hp_of(cfg, fam, c, syn):
    f = cfg['families'][fam]; base = f.get('base', {})
    if syn:
        base = {**base, **cfg['dry_run'].get('override', {}).get(f['kind'], {})}
    return {**base, **c}


# ---------------------------------------------------------------- tune (DEV only)
def tune(cfg, fam, syn, log, mac=False):
    root = R.out_root(cfg, syn) / 'tune' / fam
    if (root / 'DONE').exists():
        R.same_config(cfg, root / 'selection.json'); log(f'tune {fam}: DONE exists'); return
    root.mkdir(parents=True, exist_ok=True)
    f = cfg['families'][fam]; data = f.get('data', 'base')
    tr, ev, rep, _, _ = load_cache(cfg, syn, data, 'tune', f['kind'] in ('gbt', 'mlr'))
    configs = f['tune']['configs'][:cfg['dry_run']['configs_per_family']] if syn else f['tune']['configs']
    results = []; t0 = time.monotonic()
    for i, c in enumerate(configs):
        hp = hp_of(cfg, fam, c, syn); done = root / f'config-{i:02d}.json'
        if done.exists():
            R.same_config(cfg, done); results.append(json.loads(done.read_text())); continue
        if mac:
            wait_window(cfg, log); mac_admission(cfg, log)
        s0 = time.monotonic()
        q, info = R.fit_one(cfg, fam, hp, tr, ev, f['tune']['seed'], log)
        r = {'index': i, 'hp': hp, 'dev_nll': M.nll(q, ev['y']), 'dev_top1': float((q.argmax(1) == ev['y']).mean()),
             'info': {k: v for k, v in info.items() if k not in ('trace', 'model')}, 'trace': info.get('trace'), 'seconds': time.monotonic() - s0,
             'config_sha256': cfg['_sha256'],
             'hit_cap': bool((info.get('best_iteration') or 0) >= hp.get('max_rounds', 10**9) or (info.get('best_epoch') or 0) >= hp.get('max_epochs', 10**9)),
             'wall_budget_stop': any(t.get('stopped') == 'wall_budget' for t in (info.get('trace') or []))}
        R.save_npy(root / f'config-{i:02d}-dev.npy', q.astype(np.float32)); done.write_text(json.dumps(r, indent=1) + '\n'); results.append(r)
        log(f'tune {fam} config {i} dev_nll {r["dev_nll"]:.5f} {r["seconds"]:.0f}s cap {r["hit_cap"]}')
    best = min(results, key=lambda r: r['dev_nll'])
    budget = best['info'].get('best_iteration') or best['info'].get('best_epoch')
    fixed = f['tune'].get('fixed_budget')   # EXPL-A5: registered epoch count for every seed (no per-seed DEV early stop); config fits above are records only
    if fixed:
        budget = int(fixed)
    qs = [] if fixed else [np.load(root / f'config-{best["index"]:02d}-dev.npy').astype(np.float64)]; extra = []
    seeds = f['seeds'] if f['kind'] == 'seq' else f['seeds'][:1]
    for s in (seeds if fixed else seeds[1:]):
        p = root / f'best-seed{s}-dev.npy'
        if p.exists():
            qs.append(np.load(p).astype(np.float64)); continue
        if mac:
            wait_window(cfg, log)
        q, _ = R.fit_one(cfg, fam, best['hp'], tr, ev, s, log, dev=False, budget=budget)
        R.save_npy(p, q.astype(np.float32)); qs.append(q); extra.append({'seed': s, 'dev_nll': M.nll(q, ev['y'])})
    qm = np.mean(qs, 0); qm /= qm.sum(1, keepdims=True)
    tau = M.fit_temperature(qm, ev['y'])
    qt = M.temper(qm, tau)
    sel = {'family': fam, 'data': data, 'best_index': best['index'], 'hp': best['hp'], 'budget': budget, 'seeds': seeds, 'extra_seeds': extra, 'fixed_budget': fixed,
           'dev_nll_best_seed0': best['dev_nll'], 'dev_nll_ensemble_tau1': M.nll(qm, ev['y']), 'tau': tau, 'dev_nll_ensemble_tau': M.nll(qt, ev['y']),
           'dev_top1': float((qt.argmax(1) == ev['y']).mean()), 'dev_rows': int(len(ev['y'])), 'best_hit_cap': best['hit_cap'],
           'all_configs': [{'index': r['index'], 'dev_nll': r['dev_nll'], 'hit_cap': r['hit_cap'], 'wall_budget_stop': r['wall_budget_stop'],
                            'budget': r['info'].get('best_iteration') or r['info'].get('best_epoch') or r['info'].get('lbfgs_iters'), 'seconds': r['seconds']} for r in results],
           'seconds_total': time.monotonic() - t0, 'config_sha256': cfg['_sha256'], 'commit': R.git_state()}
    R.save_npy(root / 'selected-dev.npy', qm.astype(np.float32))
    (root / 'selection.json').write_text(json.dumps(sel, indent=1) + '\n'); (root / 'DONE').write_text('')
    log(f'tune {fam} selected {best["index"]} dev_nll(tau) {sel["dev_nll_ensemble_tau"]:.5f} tau {tau:.4f}')


# ---------------------------------------------------------------- fitw: full-FIT refit, weights saved
def fitw(cfg, fam, syn, log, mac=False):
    root = R.out_root(cfg, syn) / 'fitw' / fam
    if (root / 'DONE').exists():
        R.same_config(cfg, root / 'spec.json'); log(f'fitw {fam}: DONE exists'); return
    root.mkdir(parents=True, exist_ok=True)
    sp = R.out_root(cfg, syn) / 'tune' / fam / 'selection.json'; R.same_config(cfg, sp); sel = json.loads(sp.read_text())
    f = cfg['families'][fam]; data = f.get('data', 'base'); fits = []
    if f['kind'] == 'seq':
        import torch
        tr, _, rep, pv, bv = load_cache(cfg, syn, data, 'fit')
        np.savez(root / 'vocab.npz', pitcher=pv, batter=bv)
        for s in sel['seeds']:
            p = root / f'model-state-seed{s}.pt'; rec = root / f'fit-seed{s}.json'
            if p.exists() and rec.exists():
                r0 = json.loads(rec.read_text()); D.need(r0['state_sha256'] == R.sha(p), f'seed {s} weights sha'); fits.append(r0); continue
            s0 = time.monotonic()
            _, info = M.seq_fit_predict(tr, None, sel['hp'], s, os.environ.get('PITCHEEZY_DEVICE') or f['device'], cfg['resources']['threads'].get(f['lane'], 2), log, epochs=sel['budget'], return_model=True)
            sd = {k: v.detach().cpu() for k, v in info['model'].state_dict().items()}
            D.need(all(torch.isfinite(v).all() for v in sd.values() if v.is_floating_point()), 'finite weights')
            tmp = root / f'.tmp-seed{s}.pt'; torch.save(sd, tmp); tmp.replace(p)
            r0 = {'seed': s, 'seconds': time.monotonic() - s0, 'n_pitcher': info['n_pitcher'], 'n_batter': info['n_batter'],
                  'state_sha256': R.sha(p), 'trace': info['trace'], 'config_sha256': cfg['_sha256'], 'commit': R.git_state()}
            D.need(info['n_pitcher'] == len(pv) and info['n_batter'] == len(bv), 'embedding sizes == vocab')
            rec.write_text(json.dumps(r0, indent=1) + '\n'); fits.append(r0)
            log(f'fitw {fam} seed {s} saved')
        n_p, n_b = int(len(pv)), int(len(bv))
        D.need(sorted(x['seed'] for x in fits) == sorted(sel['seeds']), 'every registered seed has saved weights')
    else:
        import lightgbm as lgb
        tr, _, rep, pv, bv = load_cache(cfg, syn, 'base', 'fit', True)
        D.need(tr['X'] is not None, 'fit cache has tabular features')
        if mac:
            wait_window(cfg, log)
        hp = sel['hp']; s0 = time.monotonic()
        params = dict(objective='multiclass', num_class=9, learning_rate=hp['learning_rate'], num_leaves=hp['num_leaves'], min_data_in_leaf=hp['min_data_in_leaf'],
                      feature_fraction=hp['feature_fraction'], lambda_l2=hp['lambda_l2'], max_bin=hp.get('max_bin', 255),
                      num_threads=cfg['resources']['threads'].get(f['lane'], 2), seed=0, deterministic=True, force_row_wise=True, verbose=-1)
        booster = lgb.train(params, lgb.Dataset(tr['X'], tr['y']), num_boost_round=int(sel['budget']))
        booster.save_model(str(root / 'model-seed0.txt'))
        a = cfg['features']['target_encoding_alpha']
        te = D.TargetEncoder(a).fit(tr['pitcher'], tr['batter'], tr['balls'], tr['strikes'], tr['stand'], tr['y'])
        np.savez(root / 'target-encoder.npz', alpha=a, glob=te.glob, pu=te.pu, pc=te.pc, pcu=te.pcu, pcc=te.pcc, psu=te.psu, psc=te.psc, bu=te.bu, bc=te.bc)
        fits.append({'seed': 0, 'seconds': time.monotonic() - s0, 'rounds': int(sel['budget']), 'model_sha256': R.sha(root / 'model-seed0.txt')})
        n_p, n_b = int(len(pv)), int(len(bv))
    spec = {'family': fam, 'kind': f['kind'], 'data': data, 'hp': sel['hp'], 'budget': sel['budget'], 'seeds': [x['seed'] for x in fits] or sel['seeds'],
            'tau': sel['tau'], 'n_pitcher': n_p, 'n_batter': n_b, 'train_rows': rep['train_rows'], 'train_hist_rows': rep['train_hist_rows'],
            'fits': fits, 'information_set': cfg['features']['information_set'], 'eps_labels_read': False, 'eps_inputs_read': False,  # 'fit' cache has no eval set (prepare final refused without pool)

            'config_sha256': cfg['_sha256'], 'commit': R.git_state()}
    (root / 'spec.json').write_text(json.dumps(spec, indent=1) + '\n'); (root / 'DONE').write_text('')
    log(f'fitw {fam} done')


def mac_admission(cfg, log):
    while True:
        try:
            R.admission(cfg, log); return
        except SystemExit as e:
            log(f'admission refused ({e}); retry in 5 min'); time.sleep(300)


def wait_window(cfg, log):
    while R.in_window(cfg):
        log('waiting: forbidden heavy-work window'); time.sleep(60)


# ---------------------------------------------------------------- U1 (EXP-PREQ-001 tilt)
def tilt(q, m, c, kappa, clip):
    if math.isinf(kappa):
        return q.copy()
    n = c.sum(1, keepdims=True); post = (c + kappa * m) / (n + kappa)
    out = q * np.clip(post / np.maximum(m, 1e-300), clip[0], clip[1]); return out / out.sum(1, keepdims=True)


def asof_counts(pitcher, day, y):
    """counts of the same pitcher's 9-group labels over rows with a strictly earlier day (whole earlier days only)."""
    c = np.zeros((len(y), 9)); run = {}
    for d in np.unique(day):
        idx = np.flatnonzero(day == d)
        for i in idx:
            c[i] = run.get(int(pitcher[i]), 0)
        for i in idx:
            v = run.setdefault(int(pitcher[i]), np.zeros(9)); v[y[i]] += 1
    return c


def pitcher_marginal(alpha, tr_pitcher, tr_y, pitcher):
    te = D.TargetEncoder(alpha).fit(tr_pitcher, tr_pitcher * 0, tr_pitcher * 0, tr_pitcher * 0, tr_pitcher * 0, tr_y)
    z = np.zeros(len(pitcher), np.int64)
    return te.transform(pitcher, z, z, z, z)['p']



def select(cfg, syn, log):
    root = R.out_root(cfg, syn)
    if (root / 'provisional-best.json').exists():
        R.same_config(cfg, root / 'provisional-best.json'); log('select: provisional-best.json exists (never overwritten)'); return
    fs = fams(cfg, include_preq=False)     # U1 (update mode) is never ranked with static arms (data-roles.md:21)
    for f in fs:
        if not (root / 'tune' / f / 'DONE').exists():
            raise SystemExit(f'select: tune {f} not DONE (every registered arm finishes before selection)')
        R.same_config(cfg, root / 'tune' / f / 'selection.json')
    sels = {f: json.loads((root / 'tune' / f / 'selection.json').read_text()) for f in fs}
    best = min(sorted(sels), key=lambda f: sels[f]['dev_nll_ensemble_tau'])
    out = {'rule': cfg['decision']['provisional_rule'], 'provisional_best_family': best, 'final_selection': False, 'dev_nll_tau': {f: s['dev_nll_ensemble_tau'] for f, s in sels.items()},
           'eps_labels_read': False, 'at': dt.datetime.now(R.KST).isoformat(), 'config_sha256': cfg['_sha256'], 'commit': R.git_state()}
    (root / 'provisional-best.json').write_text(json.dumps(out, indent=1) + '\n'); log(f'select (provisional, within {cfg["id"]}): {best}')


def leaderboard(cfg, syn, log):
    root = R.out_root(cfg, syn); rows = []
    for name, m in cfg.get('pool', {}).items():
        mr = pathlib.Path(m['root']) / ('dryrun' if syn else 'attempt-001')
        for f in m['families']:
            p = mr / 'tune' / f / 'selection.json'
            if p.exists():
                s0 = json.loads(p.read_text()); rows.append((s0['dev_nll_ensemble_tau'], f, name, s0.get('tau'), s0.get('dev_top1'), 'DONE', f'budget {s0.get("budget")}, seed {len(s0.get("seeds", []))}개'))
            else:
                rows.append((math.inf, f, name, None, None, '진행 중/미회수', ''))
    for name, p in cfg['reference'].get('dev_selections', {}).items():
        s = json.loads(pathlib.Path(p).read_text())
        rows.append((s['dev_nll_ensemble_tau'], name, 'LIT-BASE 참고(같은 DEV)', s.get('tau'), s.get('dev_top1'), 'DONE', ''))
    for f in fams(cfg):
        p = root / 'tune' / f / 'selection.json'
        if p.exists():
            s = json.loads(p.read_text()); note = f'base {s["base_family"]}, κ {s["kappa"]}, DEV top1 변화 {s["dev_top1_change"]:+.4f}' if f == 'X-PREQ' else \
                f'설정 {s["best_index"]}, budget {s["budget"]}, seed {len(s["seeds"])}개, 상한 {sum(bool(c["hit_cap"]) for c in s["all_configs"])}/{len(s["all_configs"])}'
            rows.append((s['dev_nll_ensemble_tau'], f, cfg['families'][f]['lane'], s['tau'], s.get('dev_top1'), 'DONE', note))
        else:
            done = len(list((root / 'tune' / f).glob('config-*.json'))) if (root / 'tune' / f).exists() else 0
            rows.append((math.inf, f, cfg['families'][f]['lane'], None, None, f'진행 {done}/{len(cfg["families"][f].get("tune", {}).get("configs", []))}', ''))
    rows.sort(key=lambda r: r[0])
    lines = [f'# EXP-EXPL-1121 DEV 순위표 (DEV 09-01~09-27, 105,798행, tau 적용; 갱신 {dt.datetime.now(R.KST).strftime("%m/%d %H:%M")} KST)', '',
             'E_PS 수치는 이 표에 없다. 순위·선택은 DEV만으로 한다. R은 9월이 학습 안이라 DEV 값이 없다.', '']
    for i, (v, n, lane, tau, t1, st, note) in enumerate(rows, 1):
        vs = '미측정' if math.isinf(v) else f'{v:.5f}'
        lines.append(f'{i}. **{n}** — DEV NLL {vs}' + (f', top1 {t1:.4f}, tau {tau:.3f}' if t1 is not None else '') + f' · {lane} · {st}' + (f' · {note}' if note else ''))
    out = rpath_report(cfg) / ('DEV-LEADERBOARD.dryrun.md' if syn else 'DEV-LEADERBOARD.md'); out.write_text('\n'.join(lines) + '\n'); log(f'leaderboard -> {out}')


def rpath_report(cfg):
    return pathlib.Path(cfg['resources']['report_dir'])


# ---------------------------------------------------------------- predict (Mac, B*_new only) and score (once)


def lane(cfg, families, syn, log, mac=False):
    lock = None
    if mac:
        lock = open(cfg['resources']['heavy_lock'], 'a+')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('common heavy lock held by another process: not started')
    for fam in families:
        f = cfg['families'][fam]; data = f.get('data', 'base')
        if mac != (f['lane'] == 'mac-cpu'):
            raise SystemExit(f'{fam} runs on lane {f["lane"]} (--mac only for mac-cpu families)')
        if not syn and not mac and os.environ.get('PITCHEEZY_LANE') != f['lane']:
            raise SystemExit(f'server run needs PITCHEEZY_LANE={f["lane"]} (outputs runs/.../{f["lane"]}/)')
        if not syn and mac and not os.environ.get('PITCHEEZY_OUT_ROOT'):
            raise SystemExit('mac run needs PITCHEEZY_OUT_ROOT (T7), the repo runs link is Google Drive')
        if mac:
            wait_window(cfg, log); mac_admission(cfg, log)
        tab = f['kind'] in ('gbt', 'mlr')
        prepare(cfg, data, 'tune', syn, log, tabular=tab)
        prepare(cfg, data, 'fit', syn, log, tabular=tab)
        tune(cfg, fam, syn, log, mac=mac)
        fitw(cfg, fam, syn, log, mac=mac)
        if not mac:
            leaderboard_safe(cfg, syn, log)
    log(f'lane {families}: DONE')


def leaderboard_safe(cfg, syn, log):
    try:
        if pathlib.Path(cfg['resources']['report_dir']).exists():
            leaderboard(cfg, syn, log)
    except Exception as e:  # noqa: BLE001 (report only)
        log(f'leaderboard skipped: {e!r}')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=('lane', 'preq', 'select', 'leaderboard', 'predict', 'score'))
    ap.add_argument('--families', default=''); ap.add_argument('--mac', action='store_true'); ap.add_argument('--synthetic', action='store_true')
    ap.add_argument('--part', default='A', choices=sorted(CONFIGS))
    a = ap.parse_args(argv)
    cfg = R.load_cfg(CONFIGS[a.part])
    allowed = {'EXP-EXPL-1121-A': ('lane', 'select', 'leaderboard'), 'EXP-EXPL-1121-A2': ('lane', 'select', 'leaderboard'), 'EXP-EXPL-1121-A3': ('lane', 'select', 'leaderboard'), 'EXP-EXPL-1121-A4': ('lane', 'select', 'leaderboard'), 'EXP-EXPL-1121-A4b': ('lane', 'select', 'leaderboard'), 'EXP-EXPL-1121-A5': ('lane', 'select', 'leaderboard'), 'EXP-EXPL-1121-B': ('lane', 'preq', 'select', 'leaderboard'),
               'EXP-EXPL-1121-FINAL': ('select', 'predict', 'score', 'leaderboard')}[cfg['id']]
    if a.stage not in allowed:
        raise SystemExit(f'{a.stage} is not part of {cfg["id"]} (allowed: {allowed})')
    if a.stage != 'leaderboard':
        R.require_registered(cfg, a.synthetic)
    log = R.logger(R.out_root(cfg, a.synthetic) / 'logs' / f'{a.stage}-{a.families.replace(",", "_") or "all"}.log')
    fs = [x for x in a.families.split(',') if x]
    for f in fs:
        if f not in fams(cfg) or cfg['families'][f]['kind'] == 'preq':
            raise SystemExit(f'unknown/disabled lane family {f}')
    final = cfg['id'] == 'EXP-EXPL-1121-FINAL'
    if final and (os.environ.get('PITCHEEZY_OUT_ROOT') or os.environ.get('PITCHEEZY_LANE')):
        raise SystemExit('EXPL-FINAL writes only to its registered output_root: unset PITCHEEZY_OUT_ROOT / PITCHEEZY_LANE')
    if final and a.stage in ('predict', 'score') and not a.synthetic:
        wait_window(cfg, log)
    if final and cfg.get('combo'):
        from . import final2 as F2   # v2: single + combination candidates (CANDIDATES-1121 rule)
        {'select': lambda: F2.select(cfg, a.synthetic, log), 'predict': lambda: F2.predict(cfg, a.synthetic, log),
         'score': lambda: F2.score(cfg, a.synthetic, log), 'leaderboard': lambda: leaderboard(cfg, a.synthetic, log)}[a.stage]()
        return
    {'lane': lambda: lane(cfg, fs, a.synthetic, log, a.mac), 'preq': lambda: preq_pool(cfg, a.synthetic, log),
     'select': lambda: (final_select if final else select)(cfg, a.synthetic, log),
     'leaderboard': lambda: leaderboard(cfg, a.synthetic, log), 'predict': lambda: final_predict(cfg, a.synthetic, log),
     'score': lambda: final_score(cfg, a.synthetic, log)}[a.stage]()



# ================================================================ pool (EXPL-B U1 arm, EXPL-FINAL selection over A ∪ B ∪ LIT-BASE)
def members(cfg, syn):
    """name -> (attempt root, pinned config sha, families). Each member's config bytes must equal the pin."""
    out = {}
    for name, m in ({} if syn else cfg.get('pool', {})).items():   # dry run: own arms only (members' dry runs used other slices)
        got = R.sha(R.rpath(m['config']))
        if got != m['config_sha256']:
            raise SystemExit(f'pool member {name}: config bytes {got[:12]} != pin {m["config_sha256"][:12]}')
        if not m['families']:
            raise SystemExit(f'pool member {name}: empty families (fill the registered arm or delete the member)')
        out[name] = (pathlib.Path(m['root']) / ('dryrun' if syn else 'attempt-001'), m['config_sha256'], list(m['families']))
    if fams(cfg):
        out[cfg['id']] = (R.out_root(cfg, syn), cfg['_sha256'], fams(cfg))
    return out


def pool_selections(cfg, syn, skip_preq=False):
    sels = {}
    for name, (root, csha, fl) in members(cfg, syn).items():
        for f in fl:
            if skip_preq and f == 'X-PREQ':
                continue
            p = root / 'tune' / f / 'selection.json'
            if not (root / 'tune' / f / 'DONE').exists():
                raise SystemExit(f'pool arm {name}/{f} not DONE: every registered arm finishes before selection')
            s = json.loads(p.read_text())
            if s.get('config_sha256') != csha:
                raise SystemExit(f'pool arm {name}/{f}: selection config sha differs from its registration')
            s['_root'] = str(root); s['_member'] = name
            if f in sels:
                raise SystemExit(f'duplicate arm name {f}')
            sels[f] = s
    return sels


def best_of(sels):
    return min(sorted(sels), key=lambda f: sels[f]['dev_nll_ensemble_tau'])


def preq_pool(cfg, syn, log):
    """EXPL-B X-PREQ: U1 on top of the DEV-best arm of A ∪ B(X-DATA) ∪ LIT-BASE B-SEQ (DEV only, as-of within DEV)."""
    root = R.out_root(cfg, syn) / 'tune' / 'X-PREQ'
    if (root / 'DONE').exists():
        R.same_config(cfg, root / 'selection.json'); log('preq: DONE exists'); return
    sels = pool_selections(cfg, syn, skip_preq=True); base = best_of(sels); sb = sels[base]
    prepare(cfg, 'base', 'tune', syn, log)                       # DEV pitcher / date / label rows (package only; same order in every data variant)
    tr, ev, _, _, _ = load_cache(cfg, syn, 'base', 'tune')
    q1 = np.load(pathlib.Path(sb['_root']) / 'tune' / base / 'selected-dev.npy').astype(np.float64); q1 /= q1.sum(1, keepdims=True)
    y = ev['y']
    D.need(len(q1) == len(y) and abs(M.nll(q1, y) - sb['dev_nll_ensemble_tau1']) < 1e-6, 'base DEV predictions align with DEV rows (NLL reproduces)')
    q = M.temper(q1, sb['tau']); P = cfg['families']['X-PREQ']; clip = P['ratio_clip']
    m = pitcher_marginal(P['alpha'], tr['pitcher'], tr['y'], ev['pitcher'])  # base-data tune cache: 2026 FIT_tr only
    c = asof_counts(ev['pitcher'], ev['date'], y)
    D.need(np.all(c[ev['date'] == ev['date'].min()] == 0), 'first DEV day has no update data')
    grid = [float(k) for k in P['kappa_grid']]
    res = {str(k): M.nll(tilt(q, m, c, k, clip), y) for k in grid}
    best = min(res.values()); kstar = max(k for k in grid if res[str(k)] <= best + 1e-9)
    qu = tilt(q, m, c, kstar, clip)
    sel = {'family': 'X-PREQ', 'kind': 'preq', 'base_family': base, 'base_member': sb['_member'], 'base_dev_nll_tau': sb['dev_nll_ensemble_tau'],
           'kappa': 'inf' if math.isinf(kstar) else kstar, 'nll_by_kappa': res, 'dev_nll_ensemble_tau': M.nll(qu, y),
           'dev_top1': float((qu.argmax(1) == y).mean()), 'dev_top1_base': float((q.argmax(1) == y).mean()),
           'dev_top1_change': float((qu.argmax(1) == y).mean() - (q.argmax(1) == y).mean()), 'tau': sb['tau'], 'dev_rows': int(len(y)),
           'asof': 'DEV replayed by date; a row uses labels of strictly earlier DEV dates only (whole days); m = 2026 FIT_tr pitcher mix (alpha 4)',
           'config_sha256': cfg['_sha256'], 'commit': R.git_state()}
    root.mkdir(parents=True, exist_ok=True)
    (root / 'selection.json').write_text(json.dumps(sel, indent=1) + '\n'); (root / 'DONE').write_text('')
    log(f'preq: base {base} κ {sel["kappa"]} DEV NLL {sel["dev_nll_ensemble_tau"]:.5f} (base {sb["dev_nll_ensemble_tau"]:.5f}) top1 change {sel["dev_top1_change"]:+.4f}')


def final_select(cfg, syn, log):
    root = R.out_root(cfg, syn); root.mkdir(parents=True, exist_ok=True)
    if (root / 'best.json').exists():
        R.same_config(cfg, root / 'best.json'); log('final select: best.json exists (never overwritten)'); return
    sels = pool_selections(cfg, syn); static = {f: v for f, v in sels.items() if v.get('kind') != 'preq'}; b = best_of(static)
    pq = sels.get('X-PREQ')
    if pq is not None and pq['base_family'] != b:
        raise SystemExit(f'X-PREQ base {pq["base_family"]} != static B* {b}: pools differ')
    out = {'rule': cfg['decision']['final_rule'], 'best_family': b, 'best_member': sels[b]['_member'], 'preq_reported_separately': pq is not None,
           'dev_nll_tau': {f: s['dev_nll_ensemble_tau'] for f, s in sels.items()}, 'members': {k: v[1] for k, v in members(cfg, syn).items()},
           'eps_labels_read': False, 'at': dt.datetime.now(R.KST).isoformat(), 'config_sha256': cfg['_sha256'], 'commit': R.git_state()}
    (root / 'best.json').write_text(json.dumps(out, indent=1) + '\n'); log(f'final select: {b} ({sels[b]["_member"]})')


def _best(cfg, syn):
    root = R.out_root(cfg, syn); R.same_config(cfg, root / 'best.json'); best = json.loads((root / 'best.json').read_text())
    sels = pool_selections(cfg, syn)
    if best_of({f: v for f, v in sels.items() if v.get('kind') != 'preq'}) != best['best_family']:
        raise SystemExit('best.json does not match the pool DEV selections')
    b = best['best_family']
    return root, best, sels, b, b


def final_predict(cfg, syn, log):
    """E_PS inputs (no labels) -> predictions of the selected arm (or of the U1 base arm). LIT-BASE B-SEQ reuses its scored prediction file."""
    root, best, sels, b, fam = _best(cfg, syn); out = root / 'predict' / fam
    if (out / 'DONE').exists():
        log('predict: DONE exists'); return
    out.mkdir(parents=True, exist_ok=True); src = pathlib.Path(sels[fam]['_root'])
    if sels[fam]['_member'] == 'LIT-BASE':
        ref = cfg['reference']['bseq_eps_pred']; D.need(fam == 'B-SEQ' and R.sha(ref['path']) == ref['sha256'], 'LIT-BASE B-SEQ prediction pin')
        import shutil
        shutil.copyfile(ref['path'], out / 'eps-pred.npz'); unknown = None
    else:
        w = src / 'fitw' / fam; spec = json.loads((w / 'spec.json').read_text())
        D.need(spec['config_sha256'] == sels[fam]['config_sha256'], 'weights spec from the same registration')
        eps = D.load_eps(cfg); bt = eps['built']
        if spec['kind'] == 'seq':
            voc = np.load(w / 'vocab.npz'); pidx = D.index(eps['pitcher'][bt], voc['pitcher']); bidx = D.index(eps['batter'][bt], voc['batter'])
            tok = {k: v[bt] for k, v in eps['tok'].items()}; qs = []
            D.need(sorted(f['seed'] for f in spec['fits']) == sorted(spec['seeds']), 'all seeds saved')
            for f in spec['fits']:
                pth = w / f'model-state-seed{f["seed"]}.pt'; D.need(R.sha(pth) == f['state_sha256'], 'weights sha')
                qs.append(M.seq_predict_inputs(M.seq_load(spec, pth, 'cpu'), eps['state11'][bt], pidx, bidx, tok, 'cpu'))
            unknown = int((pidx == 0).sum())
        else:
            import lightgbm as lgb
            t = np.load(w / 'target-encoder.npz'); te = D.TargetEncoder(float(t['alpha']))
            for k in ('glob', 'pu', 'pc', 'pcu', 'pcc', 'psu', 'psc', 'bu', 'bc'):
                setattr(te, k, t[k])
            enc_path = str(D.resolve(cfg['data']['package']['root'])) + '/encoder.json'
            _, _, stand, _ = D.raw_from_state(eps['state11'][bt], enc_path)
            enc = te.transform(eps['pitcher'][bt], eps['batter'][bt], eps['balls'][bt], eps['strikes'][bt], stand)
            X = D.tabular(eps['state11'][bt], eps['balls'][bt], eps['strikes'][bt], enc, {k: v[bt] for k, v in eps['tok'].items()})
            qs = [np.asarray(lgb.Booster(model_file=str(w / 'model-seed0.txt')).predict(X), np.float64)]; unknown = None
        qm = np.mean(qs, 0); qm /= qm.sum(1, keepdims=True); qt = M.temper(qm, spec['tau'])
        np.savez(out / 'eps-pred.npz', q_tau=qt, q_tau1=qm, eps_game_pk=eps['game_pk'], eps_n=eps['n'], eps_service_key=eps['service_key'],
                 eps_w1_key=eps['w1_key'], eps_built=bt)
    rec = {'family': fam, 'for_best': b, 'unknown_pitcher_rows': unknown, 'pred_sha256': R.sha(out / 'eps-pred.npz'), 'eps_labels_read': False,
           'config_sha256': cfg['_sha256'], 'commit': R.git_state()}
    (out / 'predict.json').write_text(json.dumps(rec, indent=1) + '\n'); (out / 'DONE').write_text(''); log(f'predict {fam}: {rec}')


def final_score(cfg, syn, log):
    if syn:
        raise SystemExit('score never runs in a dry run')
    root, best, sels, b, fam = _best(cfg, syn)
    if (root / 'score' / 'result.json').exists():
        raise SystemExit('score already DONE: E_PS is scored exactly once')
    pr = root / 'predict' / fam; R.same_config(cfg, pr / 'predict.json')
    if R.sha(pr / 'eps-pred.npz') != json.loads((pr / 'predict.json').read_text())['pred_sha256']:
        raise SystemExit('eps-pred sha')
    sys.path.insert(0, str(R.REPO / 'scripts/svc_p9'))
    import eval_prediction as E
    s = cfg['scoring']
    if (E.EPS, E.REPLICATES, E.SEED) != (s['floor_eps'], s['bootstrap']['replicates'], s['bootstrap']['seed']):
        raise SystemExit('scoring constants differ from config')
    cr = pathlib.Path(s['cr_dir']); receipt = json.loads((cr / 'receipt.json').read_text())
    for name in ('candidate', 'baselines'):
        if R.sha(cr / f'{name}.jsonl') != receipt['stage_files_sha256'][name]:
            raise SystemExit(f'CR {name}.jsonl vs receipt')
    cand = [json.loads(l) for l in open(cr / 'candidate.jsonl')]; base = {(x['game_pk'], x['n']): x for x in map(json.loads, open(cr / 'baselines.jsonl'))}
    for c in cand:
        b0 = base[(c['game_pk'], c['n'])]
        if b0['service_key'] != c['service_key'] or (c['cand_C'] is None) != (c['cand_R'] is None) or E.label(b0['actual']) is None:
            raise SystemExit('CR row check failed')
    y = np.array([E.label(base[(c['game_pk'], c['n'])]['actual']) for c in cand]); games = np.array([c['game_pk'] for c in cand])
    pitcher = np.array([base[(c['game_pk'], c['n'])]['pitcher'] for c in cand])
    present = np.array([c['cand_R'] is not None for c in cand]); U = np.full(9, 1 / 9)
    if len(cand) != s['rows'] or len(set(games.tolist())) != s['games'] or 849799 in set(games.tolist()) or (~present).sum() != s['omitted_rows']:
        raise SystemExit('E_PS rows differ from registration')
    Q = {'R': np.array([c['cand_R'] if c['cand_R'] is not None else U for c in cand]), 'C': np.array([c['cand_C'] if c['cand_C'] is not None else U for c in cand])}
    def load_pred(path):
        z = np.load(path, allow_pickle=False)
        ok = (np.array_equal(z['eps_game_pk'], games) and np.array_equal(z['eps_n'], np.array([c['n'] for c in cand])) and np.array_equal(z['eps_built'], present)
              and all(a == json.dumps(c['service_key']) for a, c in zip(z['eps_service_key'], cand)) and all(a == json.dumps(c['w1_key']) for a, c in zip(z['eps_w1_key'], cand)))
        if not ok:
            raise SystemExit(f'rows differ {path}')
        q = np.tile(U, (len(cand), 1)); q[present] = z['q_tau']; return q
    ref = cfg['reference']['bseq_eps_pred']
    if R.sha(ref['path']) != ref['sha256']:
        raise SystemExit('B-SEQ prediction pin')
    Q['B-SEQ'] = load_pred(ref['path']); Q[fam] = load_pred(pr / 'eps-pred.npz')
    if 'X-PREQ' in sels:                         # U1 on top of the static B* (same base by construction; reported separately)
        P = sels['X-PREQ']; D.need(P['base_family'] == b, 'X-PREQ base == static B*'); k = math.inf if P['kappa'] == 'inf' else float(P['kappa']); spec = cfg['preq']
        pk = D.load_package(cfg, check_sha=True); lab = pk['y'] >= 0
        m = pitcher_marginal(spec['alpha'], pk['pitcher'][lab], pk['y'][lab], pitcher)          # 2026 FIT (full) pitcher mix
        fd = {g['game_pk']: g['official_date'] for g in json.loads((R.REPO / 'docs/pm/research/svc-p9/registration-facts.json').read_text())['games']}
        day = np.array([np.datetime64(fd[int(g)]).astype(np.int64) for g in games])
        c = asof_counts(pitcher, day, y)          # labels of E_PS games with a strictly earlier officialDate only (as-of)
        D.need(abs(P['tau'] - sels[b]['tau']) < 1e-12 and abs(P['base_dev_nll_tau'] - sels[b]['dev_nll_ensemble_tau']) < 1e-12, 'X-PREQ base tau/NLL == B*_static')
        eps_p = D.load_eps(cfg)['pitcher']
        D.need(np.array_equal(eps_p[present], pitcher[present]), 'CR pitcher == E_PS input pitcher on built rows (U1 counts and model inputs use the same pitcher)')
        q = Q[fam].copy(); q[present] = tilt(Q[fam][present], m[present], c[present], k, spec['ratio_clip']); Q['X-PREQ'] = q
    met = {kk: R.metrics(E, q, y) for kk, q in Q.items()}
    con = {f'R-{b}': R.contrast(E, Q['R'], Q[b], y, games), f'{b}-C': R.contrast(E, Q[b], Q['C'], y, games)}
    if b != 'B-SEQ':
        con[f'{b}-B-SEQ'] = R.contrast(E, Q[b], Q['B-SEQ'], y, games)
    if 'X-PREQ' in Q:
        con[f'X-PREQ-{b}'] = R.contrast(E, Q['X-PREQ'], Q[b], y, games)
        con['R-X-PREQ'] = R.contrast(E, Q['R'], Q['X-PREQ'], y, games)
        con['X-PREQ-B-SEQ'] = R.contrast(E, Q['X-PREQ'], Q['B-SEQ'], y, games)
    v = cfg['decision']['verdicts']
    def word(cn, w):
        lo, hi = cn['ci95']['nll']; return w['upper_lt_0'] if hi < 0 else (w['lower_gt_0'] if lo > 0 else w['else'])
    out = {'experiment': cfg['id'], 'best_static_from_DEV': b, 'preq_on_top': 'X-PREQ' in Q, 'verdict_R_vs_best': word(con[f'R-{b}'], v['R_minus_best']),
           'verdict_best_vs_BSEQ': word(con[f'{b}-B-SEQ'], v['best_minus_BSEQ']) if b != 'B-SEQ' else 'B*_new = B-SEQ (같은 모델)',
           'contrasts': con, 'metrics': met, 'top1_change': {kk: float(con[kk]['point']['top1']) for kk in con},
           'rows': len(cand), 'omitted_uniform': int((~present).sum()), 'best_json_sha256': R.sha(root / 'best.json'),
           'floor_eps': E.EPS, 'bootstrap': {'unit': 'game_pk', 'replicates': E.REPLICATES, 'seed': E.SEED}, 'cr_receipt_sha256': R.sha(cr / 'receipt.json'),
           'pred_sha256': R.sha(pr / 'eps-pred.npz'), 'preq_kappa': sels['X-PREQ']['kappa'] if 'X-PREQ' in sels else None,
           'config_sha256': cfg['_sha256'], 'commit': R.git_state(), 'labels_read_here_first_time_for_this_registration': True}
    (root / 'score').mkdir(parents=True, exist_ok=True)
    (root / 'score' / 'result.json').write_text(json.dumps(out, indent=1, allow_nan=False) + '\n')
    log(f'score: {out["verdict_R_vs_best"]} / {out["verdict_best_vs_BSEQ"]}')


if __name__ == '__main__':
    main()
