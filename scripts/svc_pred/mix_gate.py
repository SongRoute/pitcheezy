"""EXP-SVC-PRED-001 mix gates (configs/EXP-SVC-PRED-001/mix-gates.json, fixed before any refit result).

    python -B scripts/svc_pred/mix_gate.py --mix-pin configs/EXP-SVC-PRED-001/pins/<name>-mix.json --out <json>

M0 recombining the SCORED components with the candidates.json weights reproduces the entry eps_nll (self-check)
M1 svc_p9 C on the cached E_PS inputs == CR cand_C (label-free)
M2 the mix Bundle on the cached inputs == recombination of the refit components (label-free)
M3 refit mix vs scored mix agreement (label-free)
M4 only after M0-M3: |NLL(refit mix) - entry eps_nll| <= bound (labels already scored; descriptive re-use)
The cache is read with numpy only: this process binds ``experiments`` to R's snapshot (svc_p9), so experiments.lit_base
must not be imported here.
"""
from __future__ import annotations

import os
import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
for p in (REPO / 'src', REPO / 'scripts/svc_p9'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
GATES = REPO / 'configs/EXP-SVC-PRED-001/mix-gates.json'
REFIT = REPO / 'configs/EXP-SVC-PRED-001/refit.json'
EPS = 0.001
NAME_TO = {'C': ('pitcheezy.svc_p9.predictor', 'C'), 'B-SEQ': ('pitcheezy.svc_lit.bundle', 'B-SEQ'), 'B-GBT': ('pitcheezy.svc_lit.bundle', 'B-GBT'),
           'B-TAK': ('pitcheezy.svc_lit.bundle', 'B-TAK'), 'B-MLR': ('pitcheezy.svc_lit.bundle', 'B-MLR'), 'B-TAB': ('pitcheezy.svc_lit.bundle', 'B-TAB'),
           'R': ('pitcheezy.svc_p9.predictor', 'R')}


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def nll(q, y):
    f = (1 - EPS) * q + EPS / 9
    return float(-np.log(f[np.arange(len(y)), y]).mean())


def agreement(a, b):
    tv = 0.5 * np.abs(a - b).sum(1)
    return {'rows': int(len(a)), 'tv_mean': float(tv.mean()), 'tv_max': float(tv.max()), 'top1_agreement': float((a.argmax(1) == b.argmax(1)).mean()),
            'max_abs_diff': float(np.abs(a - b).max())}


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument('--mix-pin', type=pathlib.Path, required=True); ap.add_argument('--out', type=pathlib.Path, required=True)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f'{a.out} exists')
    G = json.loads(GATES.read_text()); g = G['gates']; rf = json.loads(REFIT.read_text())
    cf = G['candidates']
    if sha(cf['file']) != cf['sha256']:
        raise SystemExit('candidates.json differs from its pin')
    entry = json.loads(pathlib.Path(cf['file']).read_text())['A'][cf['key']]
    mix = json.loads(a.mix_pin.read_text())
    want_method = {'GM-w': 'GM', 'AM-w': 'AM', 'GM-eq': 'GM', 'AM-eq': 'AM'}[entry['method']]
    want_T = entry['T'] if entry['temp'] else None
    if ([c['name'] for c in mix['components']] != entry['members'] or mix['method'] != want_method or mix.get('temperature') != want_T
            or not np.allclose(mix['weights'], entry['w'], rtol=0, atol=0)):
        raise SystemExit('mix pin differs from the candidates.json entry')
    for c in mix['components']:
        if (c['module'], c['candidate']) != NAME_TO[c['name']]:
            raise SystemExit(f"component {c['name']} module / candidate")
    from pitcheezy.svc_lit import mix as MX
    import eval_prediction as E
    # CR rows (same as EXP-LIT-BASE-001 score / CANDIDATES-1121)
    cr = pathlib.Path(os.path.expandvars('${PITCHEEZY_SVC_RUNS}/EXP-SVC-P9-003/CR/attempt-001'))
    receipt = json.loads((cr / 'receipt.json').read_text())
    for name in ('candidate', 'baselines'):
        if sha(cr / f'{name}.jsonl') != receipt['stage_files_sha256'][name]:
            raise SystemExit(f'CR {name}.jsonl vs receipt')
    cand = [json.loads(l) for l in open(cr / 'candidate.jsonl')]
    present = np.array([c['cand_C'] is not None for c in cand]); U = np.full(9, 1 / 9)
    QC = np.array([c['cand_C'] if c['cand_C'] is not None else U for c in cand], np.float64)
    root = pathlib.Path(rf['source']['attempt_root'])
    scored, refit = {}, {}
    for c in mix['components']:
        if c['module'] != 'pitcheezy.svc_lit.bundle':
            continue
        z = np.load(root / 'final' / c['name'] / 'eps-pred.npz', allow_pickle=False)
        if sha(root / 'final' / c['name'] / 'eps-pred.npz') != rf['families'][c['name']]['scored_pred_sha256']:
            raise SystemExit('scored eps-pred sha')
        if not (np.array_equal(z['eps_built'], present) and np.array_equal(z['eps_game_pk'], np.array([c_['game_pk'] for c_ in cand]))
                and np.array_equal(z['eps_n'], np.array([c_['n'] for c_ in cand]))):
            raise SystemExit('E_PS rows differ from CR')
        out = pathlib.Path(rf['families'][c['name']]['out_dir']); r = json.loads((out / 'refit.json').read_text())
        if r.get('pass') is not True or r.get('refit_config_sha256') != sha(REFIT):
            raise SystemExit(f"refit gates not passed for {c['name']}")
        pin = json.loads(pathlib.Path(c['config'] if pathlib.Path(c['config']).is_absolute() else REPO / c['config']).read_text())
        if pin != json.loads((out / 'bundle-pin.json').read_text()) or sha(REPO / c['config']) != c['config_sha256']:
            raise SystemExit(f"pin of {c['name']} differs from the gate-time pin")
        for d, arr in ((scored, z['q_tau']), (refit, np.load(out / 'eps-q_tau.npy'))):
            q = np.tile(U, (len(cand), 1)); q[present] = arr; d[c['name']] = q
    names = [c['name'] for c in mix['components']]
    full = lambda d: [QC if n == 'C' else d[n] for n in names]  # noqa: E731
    comb = lambda qs: MX.combine(qs, mix['weights'], mix['method'], mix.get('temperature'))  # noqa: E731
    q_scored, q_refit = comb(full(scored)), comb(full(refit))
    rep = {'entry': {k: entry[k] for k in ('members', 'method', 'w', 'T', 'temp', 'eps_nll', 'dev_nll_crossfit')}, 'mix_pin_sha256': sha(a.mix_pin)}
    # M1 / M2 on the cached E_PS inputs (label-free)
    cache = root / 'prepare/final/cache.npz'
    if sha(cache) != rf['source']['cache_sha256']:
        raise SystemExit('cache sha')
    zc = np.load(cache, allow_pickle=False)
    st, pit, bat = zc['state11_ev'], zc['pitcher_ev'], zc['batter_ev']
    tok = {k: zc[f'tok_ev_{k}'] for k in ('valid', 'type', 'out', 'balls', 'strikes', 'same')}
    tok['pos'] = np.broadcast_to(np.arange(1, 33), tok['valid'].shape).copy()
    from pitcheezy.svc_p9 import predictor as P9
    cb = P9.Bundle.load(config=REPO / 'configs/EXP-SVC-P9-003.json', threads=2, candidate='C')
    bx = MX.Bundle.load(a.mix_pin, threads=2)
    qc, qx = [], []
    for i in range(0, len(st), 2048):
        sl = slice(i, i + 2048); t = {k: v[sl] for k, v in tok.items()}
        qc.append(cb.predict(st[sl], pit[sl], bat[sl], t)['q']); qx.append(bx.predict(st[sl], pit[sl], bat[sl], t)['q'])
    qc, qx = np.concatenate(qc), np.concatenate(qx)
    m1 = agreement(QC[present], qc); m2 = agreement(q_refit[present], qx); m3 = agreement(q_scored[present], q_refit[present])
    rep['M1_c_path'] = {**m1, 'pass': m1['max_abs_diff'] <= g['M1_c_path_max_abs_diff']}
    rep['M2_mix_bundle'] = {**m2, 'pass': m2['max_abs_diff'] <= g['M2_mix_bundle_max_abs_diff']}
    rep['M3_agreement'] = {**m3, 'pass': m3['tv_mean'] <= g['M3_tv_mean_max'] and m3['top1_agreement'] >= g['M3_top1_min']}
    label_free_ok = rep['M1_c_path']['pass'] and rep['M2_mix_bundle']['pass'] and rep['M3_agreement']['pass']
    if not label_free_ok:
        rep.update(M0_self_check={'skipped': 'label-free gate failed; labels not read'}, M4={'skipped': True}, passed=False)
    else:
        base = {(b['game_pk'], b['n']): b for b in map(json.loads, open(cr / 'baselines.jsonl'))}
        y = np.array([E.label(base[(c['game_pk'], c['n'])]['actual']) for c in cand])
        n0 = nll(q_scored, y); n1 = nll(q_refit, y)
        rep['M0_self_check'] = {'nll_scored_recombined': n0, 'entry_eps_nll': entry['eps_nll'], 'pass': abs(n0 - entry['eps_nll']) < 1e-9}
        rep['M4'] = {'nll_refit_mix': n1, 'diff_vs_entry': n1 - entry['eps_nll'], 'nll_C': nll(QC, y),
                     'pass': rep['M0_self_check']['pass'] and abs(n1 - entry['eps_nll']) <= g['M4_abs_nll_diff_max'],
                     'labels': 'CR attempt-001 baselines (already scored; descriptive re-use, not a new test)'}
        rep['passed'] = bool(rep['M4']['pass'])
    rep['gates_sha256'] = sha(GATES)
    a.out.write_text(json.dumps(rep, indent=1, allow_nan=False) + '\n')
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
