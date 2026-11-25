"""EXP-EXPL-1121-FINAL (v2): 단일 모델과 조합 후보를 CANDIDATES-1121과 같은 규칙으로 DEV에서 고르고, E_PS로 1회 확인한다.

성분 풀 = CANDIDATES D6(C, B-SEQ, B-GBT, B-TAK, B-MLR, B-TAB; 20261121-candidates/data/dev.npz, sha pin)
        + EXPL-A 4팔(X-GRU-A, X-GRU-B, X-TAK, X-GBT) + EXPL-A2 2팔(X-GRU-A2a, X-GRU-A2b) + EXPL-B X-DATA + EXPL-A3 2팔(X-GRU-A3-seeds, X-GRU-A3-plus; v3) (각 tune/selected-dev.npy, tau 적용; config sha pin).
후보 = 단일 성분 수(pool에서 정해짐; v3 15개, v4 17개) × 온도{없음, T} + 크기 2..max_subset 부분집합 × {AM-eq, GM-eq, AM-w, GM-w} × 온도 2 + 전체 집합 × 4 × 2.
점수 = DEV 교차적합 NLL(하한 0.001; 날짜 반 09-01..13 / 09-14..27, dev_half.npy pin); 맞출 값이 없으면 DEV 전체 NLL.
1순위 = 최저와 0.0005 이내 후보 중 (성분 수 적음 → 동일 가중 → 온도 없음 → 점수) 순 최소 (cand.py와 같은 규칙).
combine / fit 은 20261121-candidates/cand.py(sha 8987d487…)의 함수를 바이트 그대로 옮겼다.
기준 후보: CANDIDATES 1순위 C+SEQ+GBT GM-w (교차적합 1.2261915183584156 재현 게이트).
U1(X-PREQ 식)은 1순위 조합 위에 얹어 따로 보고한다(정적 순위에 넣지 않음).
"""
from __future__ import annotations

import datetime as dt
import itertools
import json
import math
import multiprocessing as mp
import pathlib
import sys
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

from . import data as D
from . import expl as X
from . import models as M
from . import run as R

SHORT = {'C': 'C', 'B-SEQ': 'SEQ', 'B-GBT': 'GBT', 'B-TAK': 'TAK', 'B-MLR': 'MLR', 'B-TAB': 'TAB',
         'X-GRU-A': 'XGRUA', 'X-GRU-B': 'XGRUB', 'X-TAK': 'XTAK', 'X-GBT': 'XGBT', 'X-DATA': 'XDATA',
         'X-GRU-A2a': 'XGRUA2a', 'X-GRU-A2b': 'XGRUA2b', 'X-GRU-A3-seeds': 'XGRUA3s', 'X-GRU-A3-plus': 'XGRUA3p', 'X-GRU-A4-seeds': 'XGRUA4s', 'X-GRU-A4-cal': 'XGRUA4c', 'X-GRU-A4b-seeds': 'XGRUA4bs', 'X-GRU-A5-b24': 'XGRUA5b24', 'X-GRU-A5-b30': 'XGRUA5b30'}
METHODS = ('AM-eq', 'GM-eq', 'AM-w', 'GM-w')
EPS = 1e-3


# ---------------------------------------------------------------- cand.py (sha 8987d487…) lines 53-97, verbatim
def fl(q):
    return (1 - EPS) * q + EPS / 9


def nll(q, y):
    return float(-np.log(fl(q)[np.arange(len(y)), y]).mean())


def combine(Qs, method, w=None, T=1.0):
    """Qs: [m, n, 9]. AM: sum w q. GM: softmax(sum w log q). Then temperature T on the result."""
    m = len(Qs)
    w = np.full(m, 1.0 / m) if w is None else np.asarray(w)
    if method.startswith('AM'):
        q = np.tensordot(w, Qs, axes=1)
        lq = np.log(np.clip(q, 1e-15, 1))
    else:
        lq = np.tensordot(w, np.log(np.clip(Qs, 1e-15, 1)), axes=1)
    return softmax(lq / T, axis=1) if (T != 1.0 or method.startswith('GM')) else q


def fit(Qs, y, method, temp):
    """DEV NLL(raw, clip) 최소 가중치(단체)·온도. 반환 (w, T)."""
    m = len(Qs)
    learn_w = method.endswith('-w') and m > 1
    LQ = np.log(np.clip(Qs, 1e-15, 1))
    idx = np.arange(len(y))

    def unpack(th):
        w = softmax(np.r_[0.0, th[:m - 1]]) if learn_w else np.full(m, 1.0 / m)
        T = float(np.exp(th[-1])) if temp else 1.0
        return w, T

    def obj(th):
        w, T = unpack(th)
        if method.startswith('AM'):
            lq = np.log(np.clip(np.tensordot(w, Qs, axes=1), 1e-15, 1))
        else:
            lq = np.tensordot(w, LQ, axes=1)
        z = lq / T
        return float(-(z[idx, y] - logsumexp(z, axis=1)).mean())

    k = (m - 1 if learn_w else 0) + (1 if temp else 0)
    if k == 0:
        return np.full(m, 1.0 / m), 1.0
    r = minimize(obj, np.zeros(k), method='L-BFGS-B')
    return unpack(r.x)
# ---------------------------------------------------------------- end of verbatim block


def cname(members, mth, temp):
    return '+'.join(SHORT[x] for x in members) + ('' if len(members) == 1 else f' {mth}') + (' +T' if temp else '')


# ---------------------------------------------------------------- DEV inputs
def dev_inputs(cfg, syn, log):
    c = cfg['combo']; cd = pathlib.Path(c['candidates_dir'])
    for k, f in (('dev_npz_sha256', 'data/dev.npz'), ('dev_half_sha256', 'data/dev_half.npy')):
        D.need(R.sha(cd / f) == c[k], f'CANDIDATES {f} sha pin')
    dz = np.load(cd / 'data/dev.npz', allow_pickle=False); y = dz['y'].astype(np.int64); half = np.load(cd / 'data/dev_half.npy')
    Q = {m: dz[f'q_{m}'].astype(np.float64) for m in c['d6']}
    # own DEV rows (expl base tune cache) must be the same rows: labels identical, and every arm's stored NLL reproduces
    X.prepare(cfg, 'base', 'tune', syn, log)
    _, ev, _, _, _ = X.load_cache(cfg, syn, 'base', 'tune')
    D.need(np.array_equal(ev['y'], y) and np.array_equal(ev['pitcher'], dz['pitcher']), 'DEV rows identical to CANDIDATES dev.npz (labels, pitchers, order)')
    sels = X.pool_selections(cfg, syn)
    for f, s in sels.items():
        if f in Q or s.get('kind') == 'preq':
            continue
        q1 = np.load(pathlib.Path(s['_root']) / 'tune' / f / 'selected-dev.npy').astype(np.float64); q1 /= q1.sum(1, keepdims=True)
        D.need(len(q1) == len(y) and abs(M.nll(q1, y) - s['dev_nll_ensemble_tau1']) < 1e-6, f'{f}: DEV predictions align (NLL reproduces)')
        Q[f] = M.temper(q1, s['tau'])
    return Q, y, half, sels


def candidate_list(members, kmax):
    out = []
    for r in range(1, kmax + 1):
        for s in itertools.combinations(members, r):
            for mth in (('single',) if r == 1 else METHODS):
                for temp in (False, True):
                    out.append((s, mth, temp))
    full = tuple(members)
    if len(full) > kmax:
        for mth in METHODS:
            for temp in (False, True):
                out.append((full, mth, temp))
    return out


_G = {}


def _init(Q, y, half):
    _G.update(Q=Q, y=y, half=half)


def _score(job):
    s, mth, temp = job; Q, y, half = _G['Q'], _G['y'], _G['half']
    Qd = np.stack([Q[x] for x in s]); mm = 'AM-eq' if mth == 'single' else mth
    if temp or mm.endswith('-w'):
        qcf = np.empty((len(y), 9))
        for h in (0, 1):
            tr, te = half != h, half == h
            w, T = fit(Qd[:, tr], y[tr], mm, temp)
            qcf[te] = combine(Qd[:, te], mm, w, T)
        cf = nll(qcf, y)
    else:
        cf = None
    w, T = fit(Qd, y, mm, temp)
    full = nll(combine(Qd, mm, w, T), y)
    return {'name': cname(s, mth, temp), 'members': list(s), 'method': mth, 'temp': temp, 'w': [float(v) for v in w], 'T': float(T),
            'dev_nll_full': full, 'dev_nll_crossfit': cf if cf is not None else full, 'n_members': len(s)}


def rank(A, tie):
    order = sorted(A, key=lambda k: A[k]['dev_nll_crossfit']); best = A[order[0]]['dev_nll_crossfit']
    simp = lambda k: (A[k]['n_members'], 0 if A[k]['method'] in ('single', 'AM-eq', 'GM-eq') else 1, 1 if A[k]['temp'] else 0, A[k]['dev_nll_crossfit'])  # noqa: E731
    near = [k for k in order if A[k]['dev_nll_crossfit'] - best <= tie]
    top1 = min(near, key=simp)
    rep = {}
    for k in order:
        rep.setdefault(tuple(A[k]['members']), k)
    ranked = [rep[s] for s in sorted(rep, key=lambda s: A[rep[s]]['dev_nll_crossfit'])]
    top = [top1] + [k for k in ranked if tuple(A[k]['members']) != tuple(A[top1]['members'])][:2]
    return order, best, near, top1, top


def select(cfg, syn, log):
    root = R.out_root(cfg, syn); root.mkdir(parents=True, exist_ok=True)
    if (root / 'best.json').exists():
        R.same_config(cfg, root / 'best.json'); log('final select: best.json exists (never overwritten)'); return
    t0 = time.time(); c = cfg['combo']
    X.wait_window(cfg, log)
    Q, y, half, sels = dev_inputs(cfg, syn, log)
    for f, sv in sels.items():
        if f not in c['d6'] and sv.get('kind') != 'preq':
            D.need((pathlib.Path(sv['_root']) / 'fitw' / f / 'DONE').exists(), f'{f}: fitw weights DONE before selection (every candidate must be deployable)')
    members = [m for m in c['d6']] + [f for f in sels if f not in c['d6'] and sels[f].get('kind') != 'preq']
    jobs = candidate_list(members, c['max_subset'])
    log(f'final select: {len(members)} members, {len(jobs)} candidates, {c["processes"]} processes')
    with mp.get_context('fork').Pool(c['processes'], initializer=_init, initargs=(Q, y, half)) as pool:
        res = pool.map(_score, jobs, chunksize=8)
    A = {r['name']: r for r in res}
    ref = A[c['reference']['name']]
    D.need(abs(ref['dev_nll_crossfit'] - c['reference']['dev_nll_crossfit']) < 1e-9, 'reference GM-w reproduces CANDIDATES cross-fit NLL')
    order, best, near, top1, top = rank(A, c['tie'])
    # U1 on top of top1 (DEV, as-of within DEV; reported separately, never ranked)
    t = A[top1]; qd = combine(np.stack([Q[x] for x in t['members']]), 'AM-eq' if t['method'] == 'single' else t['method'], t['w'], t['T'])
    P = cfg['preq']; _, ev, _, _, _ = X.load_cache(cfg, syn, 'base', 'tune'); tr, _, _, _, _ = X.load_cache(cfg, syn, 'base', 'tune')
    m = X.pitcher_marginal(P['alpha'], tr['pitcher'], tr['y'], ev['pitcher']); cnt = X.asof_counts(ev['pitcher'], ev['date'], y)
    grid = [float(k) for k in P['kappa_grid']]; resk = {str(k): M.nll(X.tilt(qd, m, cnt, k, P['ratio_clip']), y) for k in grid}
    bk = min(resk.values()); kstar = max(k for k in grid if resk[str(k)] <= bk + 1e-9); qu = X.tilt(qd, m, cnt, kstar, P['ratio_clip'])
    preq = {'base': top1, 'kappa': 'inf' if math.isinf(kstar) else kstar, 'nll_by_kappa': resk, 'dev_nll_full_base': t['dev_nll_full'],
            'dev_nll_u1': M.nll(qu, y), 'dev_top1_base': float((qd.argmax(1) == y).mean()), 'dev_top1_u1': float((qu.argmax(1) == y).mean()),
            'note': 'κ는 DEV 전체에서, 기준은 1순위 조합의 DEV 전체 맞춤 가중치(교차적합 아님). 정적 순위에 넣지 않는다.'}
    out = {'rule': cfg['decision']['final_rule'], 'members': members, 'n_candidates': len(A), 'best_crossfit': best, 'tie': c['tie'],
           'near_best': near, 'top1': A[top1], 'top3': [A[k] for k in top], 'reference': ref,
           'reference_rank_raw': order.index(c['reference']['name']) + 1, 'preq_on_top1': preq,
           'pool_config_sha256': {k: v[1] for k, v in X.members(cfg, syn).items()}, 'eps_labels_read': False, 'seconds': time.time() - t0,
           'at': dt.datetime.now(R.KST).isoformat(), 'config_sha256': cfg['_sha256'], 'commit': R.git_state()}
    (root / 'candidates-dev.json').write_text(json.dumps(A, indent=1) + '\n')
    out['candidates_dev_sha256'] = R.sha(root / 'candidates-dev.json')
    (root / 'best.json').write_text(json.dumps(out, indent=1) + '\n')
    write_mix(cfg, root, out, sels)
    log(f'final select: top1 {top1} crossfit {A[top1]["dev_nll_crossfit"]:.5f} (ref {ref["dev_nll_crossfit"]:.5f}); U1 κ {preq["kappa"]}')


# ---------------------------------------------------------------- E_PS predictions of components (labels never read here)
def eps_member(cfg, syn, f, sels, log):
    """q over the 8,943 CR rows (built rows from the model, omitted rows 1/9) for one member, + key arrays for alignment."""
    root = R.out_root(cfg, syn); out = root / 'predict' / f
    if (out / 'DONE').exists():
        return out / 'eps-pred.npz'
    out.mkdir(parents=True, exist_ok=True)
    c = cfg['combo']
    if f == 'C':
        src = 'CR cand_C (read in score; no file)'; (out / 'eps-pred.json').write_text(json.dumps({'family': f, 'source': src}) + '\n')
        (out / 'DONE').write_text(''); return None
    if f in c['d6']:
        p = pathlib.Path(c['lit_final_root']) / 'final' / f / 'eps-pred.npz'
        D.need(R.sha(p) == json.loads((p.parent / 'final.json').read_text())['pred_sha256'], f'{f} LIT final prediction sha')
        if f == 'B-SEQ':
            D.need(R.sha(p) == cfg['reference']['bseq_eps_pred']['sha256'], 'B-SEQ prediction == config pin')
        import shutil
        shutil.copyfile(p, out / 'eps-pred.npz')
    else:
        predict_member(cfg, f, sels[f], out)
    (out / 'predict.json').write_text(json.dumps({'family': f, 'pred_sha256': R.sha(out / 'eps-pred.npz'), 'eps_labels_read': False,
                                                  'config_sha256': cfg['_sha256'], 'commit': R.git_state()}, indent=1) + '\n')
    (out / 'DONE').write_text(''); log(f'predict {f} done')
    return out / 'eps-pred.npz'


def predict_member(cfg, f, sel, out):
    """E_PS inputs (no labels) -> saved-weights prediction of one EXPL arm (same path as expl.final_predict)."""
    w = pathlib.Path(sel['_root']) / 'fitw' / f; spec = json.loads((w / 'spec.json').read_text())
    D.need(spec['config_sha256'] == sel['config_sha256'], f'{f}: weights spec from the same registration')
    eps = D.load_eps(cfg); bt = eps['built']
    if spec['kind'] == 'seq':
        voc = np.load(w / 'vocab.npz'); pidx = D.index(eps['pitcher'][bt], voc['pitcher']); bidx = D.index(eps['batter'][bt], voc['batter'])
        D.need(len(voc['pitcher']) == spec['n_pitcher'] and len(voc['batter']) == spec['n_batter'], f'{f}: vocab == spec')
        tok = {k: v[bt] for k, v in eps['tok'].items()}; qs = []
        D.need(sorted(x['seed'] for x in spec['fits']) == sorted(sel['seeds']), f'{f}: weights for every registered seed')
        for x in spec['fits']:
            pth = w / f'model-state-seed{x["seed"]}.pt'; D.need(R.sha(pth) == x['state_sha256'], 'weights sha')
            qs.append(M.seq_predict_inputs(M.seq_load(spec, pth, 'cpu'), eps['state11'][bt], pidx, bidx, tok, 'cpu'))
    else:
        import lightgbm as lgb
        D.need(R.sha(w / 'model-seed0.txt') == spec['fits'][0]['model_sha256'], f'{f}: booster sha')
        t = np.load(w / 'target-encoder.npz'); te = D.TargetEncoder(float(t['alpha']))
        for k in ('glob', 'pu', 'pc', 'pcu', 'pcc', 'psu', 'psc', 'bu', 'bc'):
            setattr(te, k, t[k])
        enc_path = str(D.resolve(cfg['data']['package']['root'])) + '/encoder.json'
        _, _, stand, _ = D.raw_from_state(eps['state11'][bt], enc_path)
        enc = te.transform(eps['pitcher'][bt], eps['batter'][bt], eps['balls'][bt], eps['strikes'][bt], stand)
        Xe = D.tabular(eps['state11'][bt], eps['balls'][bt], eps['strikes'][bt], enc, {k: v[bt] for k, v in eps['tok'].items()})
        qs = [np.asarray(lgb.Booster(model_file=str(w / 'model-seed0.txt')).predict(Xe), np.float64)]
    qm = np.mean(qs, 0); qm /= qm.sum(1, keepdims=True); qt = M.temper(qm, spec['tau'])
    D.need(abs(spec['tau'] - sel['tau']) < 1e-12, f'{f}: tau spec == selection')
    np.savez(out / 'eps-pred.npz', q_tau=qt, q_tau1=qm, eps_game_pk=eps['game_pk'], eps_n=eps['n'], eps_service_key=eps['service_key'],
             eps_w1_key=eps['w1_key'], eps_built=bt)


def verified_best(cfg, root):
    R.same_config(cfg, root / 'best.json'); best = json.loads((root / 'best.json').read_text())
    D.need(R.sha(root / 'candidates-dev.json') == best['candidates_dev_sha256'], 'candidates-dev.json sha == best.json')
    A = json.loads((root / 'candidates-dev.json').read_text())
    _, _, _, top1, top = rank(A, cfg['combo']['tie'])
    D.need(top1 == best['top1']['name'] and [k for k in top] == [t['name'] for t in best['top3']] and A[cfg['combo']['reference']['name']] == best['reference'],
           'best.json == rank(candidates-dev.json)')
    return best


def predict(cfg, syn, log):
    root = R.out_root(cfg, syn); best = verified_best(cfg, root)
    sels = X.pool_selections(cfg, syn)
    need_m = list(dict.fromkeys(best['top1']['members'] + best['reference']['members'] + ['B-SEQ']))
    for f in need_m:
        eps_member(cfg, syn, f, sels, log)
    info = {f: component_info(cfg, root, f, sels.get(f)) for f in need_m}
    (root / 'components-info.json').write_text(json.dumps(info, indent=1, ensure_ascii=False) + '\n')
    log(f'predict: members {need_m} DONE')


def component_info(cfg, root, f, sel):
    """Per-component facts for the service CPU Bundle check (EXP-SVC-PRED-001): E_PS prediction file, training window/data, code, device."""
    c = cfg['combo']; p = root / 'predict' / f / 'eps-pred.npz'
    d = {'name': f, 'eps_pred': None if f == 'C' else {'path': str(p), 'sha256': R.sha(p), 'content': 'q_tau (tau applied, seed mean) on built rows (8,668) + eps_* keys; CR order via (game_pk, n)'}}
    if f == 'C':
        d.update(eps_pred='CR candidate.jsonl cand_C', trained_window=['2026-03-25', '2026-08-31'], trained_through='2026-08-31', data='9월 package fold september',
                 code='EXP-SVC-P9-003 C pins (B_PSR9N6)', device='MPS')
        return d
    if f in c['d6']:
        lit = json.loads((pathlib.Path(c['lit_final_root']) / 'final' / f / 'final.json').read_text())
        d.update(trained_window=['2026-03-25', '2026-09-27'], trained_through='2026-09-27', data='9월 package FIT 전체(712,603 라벨 행)',
                 code={'commit': lit.get('commit'), 'repo_path': 'experiments/lit_base (EXP-LIT-BASE-001, eb8e8b1b)'},
                 device='MPS' if f in ('B-SEQ', 'B-TAK') else 'CPU', hp=lit.get('hp'), tau=lit.get('tau'), weights='없음(LIT final은 가중치 미저장; EXP-SVC-PRED-001 재학습)')
        return d
    w = pathlib.Path(sel['_root']) / 'fitw' / f; spec = json.loads((w / 'spec.json').read_text())
    data = '9월 package FIT 전체(2026-03-25..09-27)'
    mcfg = R.load_cfg(R.rpath(cfg['pool'][sel['_member']]['config']))
    if spec['data'] == 'plus':   # hist sha from the arm's own registration (EXPL-B X-DATA, EXPL-A3 X-GRU-A3-plus: same files)
        data += ' + 2023~2025 정규시즌 hist(sha ' + ', '.join(f'{k}: {v}' for k, v in mcfg['data']['hist']['sha256'].items()) + ')'
    d.update(trained_window=['2026-03-25', '2026-09-27'] if spec['data'] == 'base' else ['2023-03-30', '2026-09-27'], trained_through='2026-09-27',
             data=data, train_rows=spec['train_rows'], train_hist_rows=spec['train_hist_rows'], code={'commit': spec['commit'], 'models_py': 'experiments/lit_base/models.py'},
             device=mcfg['families'][f].get('device', 'cpu').upper(), kind=spec['kind'], hp=spec['hp'], budget=spec['budget'], tau=spec['tau'],
             seeds=spec['seeds'], weights_dir=str(w), n_pitcher=spec['n_pitcher'], n_batter=spec['n_batter'])
    if spec['kind'] == 'gbt':
        t = np.load(w / 'target-encoder.npz')
        d.update(features='data.tabular 114차원(state11 11 + 카운트 12 + 표적 인코딩 log 4×9 + exposure 1 + 토큰 특징 54); LightGBM num_feature 아래',
                 target_encoder={'alpha': float(t['alpha']), 'use': '표 특징용(투수·투수×카운트·투수×타석·타자 비율), FIT 전체'})
        try:
            import lightgbm as lgb
            d['lightgbm_num_feature'] = int(lgb.Booster(model_file=str(w / 'model-seed0.txt')).num_feature())
        except Exception as e:  # noqa: BLE001
            d['lightgbm_num_feature'] = f'미확인 ({e!r})'
    return d


def write_mix(cfg, root, best, sels):
    t = best['top1']
    comps = []
    for f in t['members']:
        if f == 'C':
            comps.append({'name': 'C', 'module': 'pitcheezy.svc_p9.predictor', 'candidate': 'C', 'source': 'configs/EXP-SVC-P9-003.json C pins (fold september)'})
        elif f in cfg['combo']['d6']:
            comps.append({'name': f, 'module': 'pitcheezy.svc_lit.bundle', 'candidate': f, 'source': 'EXP-LIT-BASE-001 final (weights not saved; EXP-SVC-PRED-001 refit)'})
        else:
            s = sels[f]
            comps.append({'name': f, 'module': 'pitcheezy.svc_lit.bundle', 'candidate': f, 'weights_dir': str(pathlib.Path(s['_root']) / 'fitw' / f),
                          'files': 'spec.json + vocab.npz + model-state-seed{s}.pt (seq) | spec.json + model-seed0.txt + target-encoder.npz (gbt)',
                          'models_py': 'experiments/lit_base/models.py at the registration commit (GRU num_layers)'})
    mix = {'format': 'svc-mix-bundle-v1-draft', 'format_note': 'svc-mix-bundle-v1 필드에 config/config_sha256(성분 pin)이 아직 없다. EXP-SVC-PRED-001이 성분 pin을 만들 때 채운다.',
           'name': t['name'].replace(' ', '_'), 'method': 'GM' if t['method'] in ('GM-eq', 'GM-w') else 'AM',
           'weights': t['w'] if t['method'] != 'single' else [1.0], 'temperature': None if t['T'] == 1.0 else t['T'], 'trained_window': ['2026-03-25', '2026-09-27'], 'components': comps,
           'selected_by': 'EXP-EXPL-1121-FINAL DEV cross-fit rule', 'config_sha256': cfg['_sha256']}
    (root / 'final-mix.json').write_text(json.dumps(mix, indent=1, ensure_ascii=False) + '\n')


def score(cfg, syn, log):
    if syn:
        raise SystemExit('score never runs in a dry run')
    root = R.out_root(cfg, syn)
    if (root / 'score' / 'result.json').exists():
        raise SystemExit('score already DONE: E_PS is scored exactly once')
    best = verified_best(cfg, root)
    pv = cfg.get('prior_view')   # v4: an earlier FINAL already scored E_PS; this run is a re-view (exploratory) and only under the pre-fixed v4_rule
    if pv:
        pr = pathlib.Path(pv['root'])
        D.need((pr / 'score' / 'result.json').exists() and R.sha(pr / 'score' / 'result.json') == pv['result_json_sha256']
               and R.sha(pr / 'best.json') == pv['best_json_sha256'], 'prior_view: the earlier FINAL was scored first (result/best sha pins)')
        D.need(pr.resolve() != root.resolve(), 'prior_view root differs from this root')
        t = best['top1']
        if t['name'] == pv['top1'] or not (t['dev_nll_crossfit'] < pv['top1_dev_nll_crossfit'] - cfg['combo']['tie']):
            (root / 'score').mkdir(parents=True, exist_ok=True)
            (root / 'score' / 'not-scored.json').write_text(json.dumps({'reason': 'v4_rule: top1 same as prior or DEV cross-fit not lower by more than tie; E_PS not re-viewed',
                'top1': t['name'], 'top1_dev_nll_crossfit': t['dev_nll_crossfit'], 'prior_view': pv, 'eps_labels_read': False, 'config_sha256': cfg['_sha256']}, indent=1, ensure_ascii=False) + '\n')
            log('score: v4_rule says no E_PS re-view (prior top1 stays)'); return
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
    QE = {'R': np.array([c['cand_R'] if c['cand_R'] is not None else U for c in cand]), 'C': np.array([c['cand_C'] if c['cand_C'] is not None else U for c in cand])}

    def load_pred(path):
        z = np.load(path, allow_pickle=False)
        pos = {(int(g), int(n)): i for i, (g, n) in enumerate(zip(z['eps_game_pk'], z['eps_n']))}
        order = np.array([pos[(c['game_pk'], c['n'])] for c in cand])
        ok = (len(pos) == len(cand) and np.array_equal(z['eps_built'][order], present)
              and all(str(z['eps_service_key'][i]) == json.dumps(c['service_key']) for i, c in zip(order, cand))
              and all(str(z['eps_w1_key'][i]) == json.dumps(c['w1_key']) for i, c in zip(order, cand)))
        if not ok:
            raise SystemExit(f'rows differ {path}')
        bi = np.cumsum(z['eps_built']) - 1; q = np.tile(U, (len(cand), 1)); q[present] = z['q_tau'][bi[order][present]]; return q
    need_m = list(dict.fromkeys(best['top1']['members'] + best['reference']['members'] + ['B-SEQ']))
    for f in need_m:
        if f == 'C':
            continue
        p = root / 'predict' / f / 'eps-pred.npz'; R.same_config(cfg, root / 'predict' / f / 'predict.json')
        D.need(R.sha(p) == json.loads((root / 'predict' / f / 'predict.json').read_text())['pred_sha256'], f'{f} eps-pred sha')
        QE[f] = load_pred(p)

    def mix(t):
        mm = 'AM-eq' if t['method'] == 'single' else t['method']
        return combine(np.stack([QE[x] for x in t['members']]), mm, t['w'], t['T'])
    t1 = best['top1']; rf = best['reference']
    Q = {'R': QE['R'], 'C': QE['C'], 'B-SEQ': QE['B-SEQ'], 'TOP1': mix(t1), 'REF_GMw': mix(rf)}
    # U1 on top of top1: m = 2026 FIT pitcher mix; c = labels of E_PS games with a strictly earlier officialDate (as-of)
    P = cfg['preq']; pq = best['preq_on_top1']; k = math.inf if pq['kappa'] == 'inf' else float(pq['kappa'])
    pk = D.load_package(cfg, check_sha=True); lab = pk['y'] >= 0
    eps_p = D.load_eps(cfg)['pitcher']; D.need(np.array_equal(eps_p[present], pitcher[present]), 'CR pitcher == E_PS input pitcher on built rows')
    m = X.pitcher_marginal(P['alpha'], pk['pitcher'][lab], pk['y'][lab], pitcher)
    fd = {g['game_pk']: g['official_date'] for g in json.loads((R.REPO / 'docs/pm/research/svc-p9/registration-facts.json').read_text())['games']}
    day = np.array([np.datetime64(fd[int(g)]).astype(np.int64) for g in games]); cnt = X.asof_counts(pitcher, day, y)
    qu = Q['TOP1'].copy(); qu[present] = X.tilt(Q['TOP1'][present], m[present], cnt[present], k, P['ratio_clip']); Q['TOP1+U1'] = qu
    g = cfg['combo']['eps_gates']
    D.need(abs(nll(Q['REF_GMw'], y) - g['reference_eps_nll']) < 1e-9, 'E_PS path reproduces CANDIDATES reference GM-w NLL')
    D.need(abs(nll(Q['B-SEQ'], y) - g['bseq_eps_nll']) < 1e-9, 'E_PS path reproduces LIT B-SEQ NLL')
    met = {kk: R.metrics(E, q, y) for kk, q in Q.items()}
    pairs = [('R', 'TOP1'), ('TOP1', 'B-SEQ'), ('TOP1', 'C'), ('TOP1', 'REF_GMw'), ('TOP1+U1', 'TOP1'), ('R', 'TOP1+U1'), ('TOP1+U1', 'B-SEQ')]
    con = {f'{a}-{b}': R.contrast(E, Q[a], Q[b], y, games) for a, b in pairs}
    v = cfg['decision']['verdicts']

    def word(cn, w):
        lo, hi = cn['ci95']['nll']; return w['upper_lt_0'] if hi < 0 else (w['lower_gt_0'] if lo > 0 else w['else'])
    out = {'experiment': cfg['id'], 'top1': t1['name'], 'reference': rf['name'], 'same_as_reference': t1['name'] == rf['name'],
           'verdict_R_vs_top1': word(con['R-TOP1'], v['R_minus_best']), 'verdict_top1_vs_BSEQ': word(con['TOP1-B-SEQ'], v['best_minus_BSEQ']),
           'verdict_top1_vs_reference': word(con['TOP1-REF_GMw'], v['best_minus_reference']),
           'contrasts': con, 'metrics': met, 'top1_change': {kk: float(con[kk]['point']['top1']) for kk in con}, 'preq_kappa': pq['kappa'],
           'rows': len(cand), 'omitted_uniform': int((~present).sum()), 'floor_eps': E.EPS,
           'bootstrap': {'unit': 'game_pk', 'replicates': E.REPLICATES, 'seed': E.SEED}, 'cr_receipt_sha256': R.sha(cr / 'receipt.json'),
           'best_json_sha256': R.sha(root / 'best.json'), 'config_sha256': cfg['_sha256'], 'commit': R.git_state(),
           'prior_view': pv, 'exploratory_reviewing': bool(pv)}
    # output-only addition for the service gate (EXP-SVC-PRED-001 M0): mix E_PS NLL + per-component prediction files. Selection untouched.
    mixp = root / 'final-mix.json'; mixd = json.loads(mixp.read_text())
    D.need(mixd['name'] == t1['name'].replace(' ', '_'), 'final-mix.json is the selected top1')
    mixd['eps_nll'] = met['TOP1']['nll']
    mixd['eps_nll_definition'] = 'E_PS 8,943 CR rows, omitted 275 = 1/9, floor q=(1-0.001)q+0.001/9, mean -log q[y] (= result.json metrics.TOP1.nll)'
    for comp in mixd['components']:
        f = comp['name']
        if f == 'C':
            comp['eps_pred'] = 'CR candidate.jsonl cand_C (no file)'
        else:
            pj = json.loads((root / 'predict' / f / 'predict.json').read_text())
            comp['eps_pred'] = {'path': str(root / 'predict' / f / 'eps-pred.npz'), 'predict_json': str(root / 'predict' / f / 'predict.json'), 'sha256': pj['pred_sha256']}
    mixp.write_text(json.dumps(mixd, indent=1, ensure_ascii=False) + '\n')
    out['final_mix_sha256'] = R.sha(mixp)
    (root / 'score').mkdir(parents=True, exist_ok=True)
    (root / 'score' / 'result.json').write_text(json.dumps(out, indent=1, allow_nan=False) + '\n')
    log(f'score: {out["verdict_R_vs_top1"]} / {out["verdict_top1_vs_BSEQ"]} / {out["verdict_top1_vs_reference"]}')
