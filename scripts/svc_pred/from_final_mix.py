"""EXP-SVC-PRED-001 x EXPL-FINAL: turn final-mix.json (svc-mix-bundle-v1 by the EXPL worker) into our pins, then gate it.

    python -B scripts/svc_pred/from_final_mix.py pins --final-mix <final-mix.json>              # pins/<X-*>.json + pins/<name>-mix.json
    python -B scripts/svc_pred/from_final_mix.py gate --final-mix <final-mix.json> --eps-pred X-GRU-A=<.../predict/X-GRU-A/eps-pred.npz> --out <json>
        # ext-expl gates: X1 / F1 / M3 label-free, then M0 / M4; receipt -> configs/EXP-SVC-PRED-001/gates/<name>.json

final-mix components: {name, module, candidate, weights_dir | source, eps_pred: {path, sha256} (reference E_PS q_tau, 8,668 built
rows in CR order)}. C -> svc_p9 (configs/EXP-SVC-P9-003.json); B-* -> the existing refit pins (configs/EXP-SVC-PRED-001/pins/B-*.json,
refit gates passed); X-* -> new svc-expl-fitw-v1 pins from weights_dir. Then build_pred.py new-instance --predictor-kind mix.
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
for p in (REPO / 'src', REPO / 'scripts/svc_p9', REPO / 'scripts/svc_pred'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
PINS = REPO / 'configs/EXP-SVC-PRED-001/pins'
REFIT = REPO / 'configs/EXP-SVC-PRED-001/refit.json'
EXT = REPO / 'configs/EXP-SVC-PRED-001/ext-expl.json'
GATES_DIR = REPO / 'configs/EXP-SVC-PRED-001/gates'
CR = pathlib.Path(os.path.expandvars('${PITCHEEZY_SVC_RUNS}/EXP-SVC-P9-003/CR/attempt-001'))
EPS = 0.001


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def rel(p):
    return str(pathlib.Path(p).resolve().relative_to(REPO.resolve()))


def safe(name):
    if not name or '/' in name or name in ('.', '..'):
        raise SystemExit(f'bad name {name!r}')
    return name


MAIN = pathlib.Path(os.path.expandvars('${PITCHEEZY_REPO}'))
FINAL_MIX_FORMATS = ('svc-mix-bundle-v1', 'svc-mix-bundle-v1-draft')


def committed_config(branch, path, want_sha):
    """The committed bytes of `path` on `branch` whose sha is want_sha, only if that version says status REGISTERED."""
    import subprocess
    import yaml
    revs = subprocess.run(['git', '-C', str(MAIN), 'log', '--format=%H', branch, '--', path], capture_output=True, text=True, check=True).stdout.split()
    for r in revs:
        raw = subprocess.run(['git', '-C', str(MAIN), 'show', f'{r}:{path}'], capture_output=True, check=True).stdout
        if hashlib.sha256(raw).hexdigest() == want_sha:
            doc = yaml.safe_load(raw)
            if doc.get('status') != 'REGISTERED' or doc.get('execute') is not True:
                raise SystemExit(f'{path} at {r[:8]} (sha {want_sha[:8]}) is status {doc.get("status")} execute {doc.get("execute")}, not REGISTERED + execute')
            return r, doc
    raise SystemExit(f'sha {str(want_sha)[:8]} is not a committed {path} on {branch}')


def verify_plus(spec):
    """X-DATA ('plus'): spec.config_sha256 -> a committed EXPL-B config whose data.hist.sha256 equals the ext-expl pins."""
    t = json.loads(EXT.read_text())['templates']['plus']
    found, misses = None, []
    for cp in t['config_paths']:
        try:
            found = (cp,) + committed_config(t['branch'], cp, spec.get('config_sha256')); break
        except SystemExit as ex:
            misses.append(str(ex))
    if found is None:
        raise SystemExit(f"{spec['family']}: no registered config matches the spec: {misses}")
    cp, r, doc = found
    hist = {str(k): v for k, v in doc['data']['hist']['sha256'].items()}
    if hist != t['hist_sha256']:
        raise SystemExit(f'{spec["family"]}: EXPL-B config at {r[:8]} pins other hist files')
    if not int(spec.get('train_hist_rows') or 0) > 0:
        raise SystemExit(f'{spec["family"]}: train_hist_rows is not > 0')
    return {'config_path': cp, 'config_commit': r, 'config_sha256': spec['config_sha256'], 'hist_sha256': hist, 'spec_commit': spec.get('commit'),
            'train_hist_rows': int(spec['train_hist_rows'])}


def recheck_pin(pin_path):
    """Gate-time re-check of an svc_expl pin: plus chain recomputed from the fitw spec and equal to the pinned one."""
    pin = json.loads(pathlib.Path(pin_path).read_text())
    if pin.get('data') == 'plus':
        spec = json.loads((pathlib.Path(pin['dir']) / 'spec.json').read_text())
        if pin.get('plus_chain') != verify_plus(spec):
            raise SystemExit(f"{pin['family']}: plus chain missing or differs")
    elif pin.get('data') != 'base':
        raise SystemExit(f"{pin['family']}: data {pin.get('data')} not registered")


def check_predict_json(pj):
    """EXPL-FINAL predict.json must come from a REGISTERED committed EXP-EXPL-1121-FINAL config."""
    t = json.loads(EXT.read_text())['final_config']
    if pj.get('config_sha256') != t['sha256']:
        raise SystemExit(f"predict.json config_sha256 {str(pj.get('config_sha256'))[:8]} != registered EXPL-FINAL {t['sha256'][:8]}")
    r, _ = committed_config(t['branch'], t['config_path'], pj.get('config_sha256'))
    return {'final_config_commit': r, 'final_config_sha256': pj['config_sha256'], 'predict_commit': pj.get('commit')}


def expl_pin(weights_dir, name=None):
    from pitcheezy.svc_expl import bundle as XB
    enc = json.loads(REFIT.read_text())['source']['state_encoder']
    pin = XB.make_pin(weights_dir, enc)
    if name is not None and pin['family'] != name:
        raise SystemExit(f'{name}: weights_dir spec family {pin["family"]}')
    if pin.get('data') == 'plus':
        pin['plus_chain'] = verify_plus(json.loads((pathlib.Path(weights_dir) / 'spec.json').read_text()))
    elif pin.get('data') != 'base':
        raise SystemExit(f"{pin['family']}: data {pin.get('data')} not registered")
    p = PINS / f"{safe(pin['family'])}.json"
    if p.exists():
        if json.loads(p.read_text()) != pin:
            raise SystemExit(f'{p} exists with other content')
    else:
        p.write_text(json.dumps(pin, indent=1) + '\n')
    return p


def pins(final_mix, name=None):
    fm = json.loads(pathlib.Path(final_mix).read_text())
    if fm.get('format') not in FINAL_MIX_FORMATS or fm.get('method') not in ('AM', 'GM'):
        raise SystemExit('final-mix is not svc-mix-bundle-v1')
    want = json.loads(EXT.read_text())['final_config']['sha256']
    if fm.get('format') != 'svc-mix-bundle-v1' and fm.get('config_sha256') != want:
        # the draft (no component config/config_sha256; our pins fill them) is accepted only from the pinned EXPL-FINAL config
        raise SystemExit(f"draft final-mix config_sha256 {str(fm.get('config_sha256'))[:8]} != registered EXPL-FINAL {want[:8]}")
    w = np.asarray(fm['weights'], float)
    if len(w) != len(fm['components']) or (w < 0).any() or abs(w.sum() - 1) > 1e-9:
        raise SystemExit('weights')
    names = [c['name'] for c in fm['components']]
    if len(set(names)) != len(names) or any(c.get('candidate', c['name']) != c['name'] for c in fm['components']):
        raise SystemExit('duplicate component names or candidate != name')
    if fm.get('temperature') is not None and float(fm['temperature']) <= 0:
        raise SystemExit('temperature must be > 0')
    comps = []
    for c in fm['components']:
        n = safe(c['name'])
        if n in ('C', 'R'):
            comps.append({'name': n, 'module': 'pitcheezy.svc_p9.predictor', 'config': 'configs/EXP-SVC-P9-003.json',
                          'config_sha256': sha(REPO / 'configs/EXP-SVC-P9-003.json'), 'candidate': n})
        elif n.startswith('B-'):
            p = PINS / f'{n}.json'
            if not p.exists():
                raise SystemExit(f'{n}: refit pin missing (refit_lit.py refit --family {n} first)')
            comps.append({'name': n, 'module': 'pitcheezy.svc_lit.bundle', 'config': rel(p), 'config_sha256': sha(p), 'candidate': n})
        elif n.startswith('X-'):
            p = expl_pin(c['weights_dir'], n)
            comps.append({'name': n, 'module': 'pitcheezy.svc_expl.bundle', 'config': rel(p), 'config_sha256': sha(p), 'candidate': n})
        else:
            raise SystemExit(f'unknown component {n}')
    name = safe(name or fm['name'])
    mix = {'format': 'svc-mix-bundle-v1', 'name': name, 'method': fm['method'], 'weights': fm['weights'], 'temperature': fm.get('temperature'),
           'source': {'final_mix': str(final_mix), 'sha256': sha(final_mix)}, 'components': comps}
    out = PINS / f'{name}-mix.json'
    if out.exists():
        raise SystemExit(f'{out} exists')
    out.write_text(json.dumps(mix, indent=1) + '\n')
    return {'mix_pin': rel(out), 'sha256': sha(out), 'components': [c['name'] for c in comps]}


def nll(q, y):
    f = (1 - EPS) * q + EPS / 9
    return float(-np.log(f[np.arange(len(y)), y]).mean())


def gate(final_mix, out, eps_preds, name=None):
    out = pathlib.Path(out)
    if out.exists():
        raise SystemExit(f'{out} exists')
    ext = json.loads(EXT.read_text()); g = ext['gates']
    if ext['status'] != 'REGISTERED':
        raise SystemExit('ext-expl.json is not REGISTERED')
    for r_, want in ext['vendored'].items():
        if r_ != 'source' and sha(REPO / r_) != want:
            raise SystemExit(f'{r_} differs from its ext-expl pin')
    fm = json.loads(pathlib.Path(final_mix).read_text()); name = safe(name or fm['name'])
    mp = PINS / f'{name}-mix.json'; mix = json.loads(mp.read_text())
    receipt_path = GATES_DIR / f'{name}.json'
    if receipt_path.exists():
        raise SystemExit(f'{receipt_path} exists (gates run once per mix pin)')
    if mix['source']['sha256'] != sha(final_mix) or ([c['name'] for c in mix['components']], mix['method'], mix['weights'], mix.get('temperature')) != \
            ([c['name'] for c in fm['components']], fm['method'], fm['weights'], fm.get('temperature')):
        raise SystemExit('mix pin differs from final-mix')
    from pitcheezy.svc_lit import mix as MX
    import eval_prediction as E
    import importlib
    rf = json.loads(REFIT.read_text())
    receipt = json.loads((CR / 'receipt.json').read_text())
    for nm in ('candidate', 'baselines'):
        if sha(CR / f'{nm}.jsonl') != receipt['stage_files_sha256'][nm]:
            raise SystemExit(f'CR {nm}.jsonl vs receipt')
    cand = [json.loads(l) for l in open(CR / 'candidate.jsonl')]
    present = np.array([c['cand_C'] is not None for c in cand]); U = np.full(9, 1 / 9)
    games = np.array([c['game_pk'] for c in cand]); ns = np.array([c['n'] for c in cand])
    root = pathlib.Path(rf['source']['attempt_root'])
    if sha(root / 'prepare/final/cache.npz') != rf['source']['cache_sha256']:
        raise SystemExit('cache sha')
    zc = np.load(root / 'prepare/final/cache.npz', allow_pickle=False)
    st, pit, bat = zc['state11_ev'], zc['pitcher_ev'], zc['batter_ev']
    tok = {k: zc[f'tok_ev_{k}'] for k in ('valid', 'type', 'out', 'balls', 'strikes', 'same')}
    tok['pos'] = np.broadcast_to(np.arange(1, 33), tok['valid'].shape).copy()

    def full(arr):
        q = np.tile(U, (len(cand), 1)); q[present] = arr; return q

    def rows_ok(z):
        return np.array_equal(z['eps_built'], present) and np.array_equal(z['eps_game_pk'], games) and np.array_equal(z['eps_n'], ns)

    def run(b):
        return np.concatenate([b.predict(st[i:i + 2048], pit[i:i + 2048], bat[i:i + 2048], {k: v[i:i + 2048] for k, v in tok.items()})['q']
                               for i in range(0, len(st), 2048)])
    loaded, scored = {}, {}   # loaded = what the mix Bundle uses (B-* refit); scored = B-* EXP-LIT-BASE-001 final
    rep = {'final_mix_sha256': sha(final_mix), 'mix_pin_sha256': sha(mp), 'ext_sha256': sha(EXT), 'X1': {}, 'eps_pred': {}}
    for c in mix['components']:
        n = c['name']
        if n in ('C', 'R'):
            key = 'cand_C' if n == 'C' else 'cand_R'
            loaded[n] = scored[n] = np.array([x[key] if x[key] is not None else U for x in cand], np.float64)
        elif n.startswith('B-'):
            d = pathlib.Path(rf['families'][n]['out_dir']); r = json.loads((d / 'refit.json').read_text())
            if r.get('pass') is not True:
                raise SystemExit(f'{n} refit gates')
            z = np.load(root / 'final' / n / 'eps-pred.npz', allow_pickle=False)
            if sha(root / 'final' / n / 'eps-pred.npz') != rf['families'][n]['scored_pred_sha256'] or not rows_ok(z):
                raise SystemExit(f'{n} scored eps-pred')
            loaded[n] = full(np.load(d / 'eps-q_tau.npy')); scored[n] = full(z['q_tau'])
        else:
            if n not in eps_preds:
                raise SystemExit(f'{n}: give --eps-pred {n}=<EXPL-FINAL predict/{n}/eps-pred.npz>')
            ep = pathlib.Path(eps_preds[n]); pj = json.loads((ep.parent / 'predict.json').read_text())
            if pj.get('family') != n or pj.get('pred_sha256') != sha(ep):
                raise SystemExit(f'{n}: eps-pred differs from its predict.json')
            fm_ep = next(x for x in fm['components'] if x['name'] == n).get('eps_pred') or {}
            if fm_ep.get('sha256') != sha(ep):
                raise SystemExit(f'{n}: eps-pred sha missing in final-mix or differs from it')
            z = np.load(ep, allow_pickle=False)
            if not rows_ok(z):
                raise SystemExit(f'{n}: eps-pred rows differ from CR')
            recheck_pin(REPO / c['config'])
            rep['eps_pred'][n] = {'path': str(ep), 'sha256': sha(ep), 'predict_json_sha256': sha(ep.parent / 'predict.json'), **check_predict_json(pj)}
            loaded[n] = scored[n] = full(z['q_tau'])
            b = importlib.import_module(c['module']).Bundle.load(config=REPO / c['config'], threads=2, candidate=n)
            d = float(np.abs(run(b) - z['q_tau']).max())
            rep['X1'][n] = {'max_abs_diff': d, 'pass': d <= g['X1_max_abs_diff']}
    names = [c['name'] for c in mix['components']]
    comb = lambda Q: MX.combine([Q[n] for n in names], mix['weights'], mix['method'], mix.get('temperature'))  # noqa: E731
    q_loaded, q_scored = comb(loaded), comb(scored)
    d = float(np.abs(run(MX.Bundle.load(mp, threads=2)) - q_loaded[present]).max())
    rep['F1'] = {'max_abs_diff': d, 'pass': d <= g['F1_max_abs_diff']}
    tv = 0.5 * np.abs(q_loaded[present] - q_scored[present]).sum(1); t1 = float((q_loaded[present].argmax(1) == q_scored[present].argmax(1)).mean())
    rep['M3'] = {'tv_mean': float(tv.mean()), 'top1_agreement': t1, 'pass': bool(tv.mean() <= g['M3_tv_mean_max'] and t1 >= g['M3_top1_min'])}
    ok = rep['F1']['pass'] and rep['M3']['pass'] and all(v['pass'] for v in rep['X1'].values())
    if ok:
        base = {(b_['game_pk'], b_['n']): b_ for b_ in map(json.loads, open(CR / 'baselines.jsonl'))}
        y = np.array([E.label(base[(x['game_pk'], x['n'])]['actual']) for x in cand])
        n_s, n_l = nll(q_scored, y), nll(q_loaded, y)
        ref = fm.get('eps_nll')
        rep['M0'] = {'nll_scored_recombined': n_s, 'final_mix_eps_nll': ref,
                     'pass': (abs(n_s - ref) < 1e-9) if ref is not None else None, 'note': None if ref is not None else 'final-mix gives no eps_nll: recorded only'}
        rep['M4'] = {'nll_loaded_mix': n_l, 'diff_vs_scored': n_l - n_s, 'pass': abs(n_l - n_s) <= g['M4_abs_nll_diff_max'],
                     'labels': 'CR attempt-001 (already scored; descriptive re-use, not a new test)'}
        ok = rep['M4']['pass'] and rep['M0']['pass'] is not False
    else:
        rep['M0'] = rep['M4'] = {'skipped': 'a label-free gate failed; labels not read'}
    rep['passed'] = bool(ok)
    out.write_text(json.dumps(rep, indent=1) + '\n')
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps({k: rep[k] for k in ('passed', 'mix_pin_sha256', 'final_mix_sha256', 'ext_sha256')} | {'report': str(out)}, indent=1) + '\n')
    return rep


def single_gate(family, eps_pred, out):
    """Single X arm: X1 only (label-free). Receipt configs/EXP-SVC-PRED-001/gates/<family>.json."""
    out = pathlib.Path(out)
    if out.exists():
        raise SystemExit(f'{out} exists')
    ext = json.loads(EXT.read_text()); g = ext['gates']
    if ext['status'] != 'REGISTERED':
        raise SystemExit('ext-expl.json is not REGISTERED')
    for r_, want in ext['vendored'].items():
        if r_ != 'source' and sha(REPO / r_) != want:
            raise SystemExit(f'{r_} differs from its ext-expl pin')
    fam = safe(family); pp = PINS / f'{fam}.json'; rp = GATES_DIR / f'{fam}.json'
    if rp.exists():
        raise SystemExit(f'{rp} exists')
    ep = pathlib.Path(eps_pred); pj = json.loads((ep.parent / 'predict.json').read_text())
    if pj.get('family') != fam or pj.get('pred_sha256') != sha(ep):
        raise SystemExit('eps-pred differs from its predict.json')
    recheck_pin(pp)
    provenance = check_predict_json(pj)
    cand = [json.loads(l) for l in open(CR / 'candidate.jsonl')]
    present = np.array([c['cand_C'] is not None for c in cand])
    z = np.load(ep, allow_pickle=False)
    if not (np.array_equal(z['eps_built'], present) and np.array_equal(z['eps_game_pk'], np.array([c['game_pk'] for c in cand]))
            and np.array_equal(z['eps_n'], np.array([c['n'] for c in cand]))):
        raise SystemExit('eps-pred rows differ from CR')
    rf = json.loads(REFIT.read_text()); root = pathlib.Path(rf['source']['attempt_root'])
    if sha(root / 'prepare/final/cache.npz') != rf['source']['cache_sha256']:
        raise SystemExit('cache sha')
    zc = np.load(root / 'prepare/final/cache.npz', allow_pickle=False)
    tok = {k: zc[f'tok_ev_{k}'] for k in ('valid', 'type', 'out', 'balls', 'strikes', 'same')}
    from pitcheezy.svc_expl import bundle as XB
    b = XB.Bundle.load(pp, threads=2, candidate=fam)
    qb = np.concatenate([b.predict(zc['state11_ev'][i:i + 2048], zc['pitcher_ev'][i:i + 2048], zc['batter_ev'][i:i + 2048],
                                   {k: v[i:i + 2048] for k, v in tok.items()})['q'] for i in range(0, len(tok['valid']), 2048)])
    d = float(np.abs(qb - z['q_tau']).max())
    rep = {'family': fam, 'pin_sha256': sha(pp), 'ext_sha256': sha(EXT), 'eps_pred': {'path': str(ep), 'sha256': sha(ep), **provenance},
           'X1': {'max_abs_diff': d, 'pass': d <= g['X1_max_abs_diff']}, 'labels_read': False}
    rep['passed'] = bool(rep['X1']['pass'])
    out.write_text(json.dumps(rep, indent=1) + '\n')
    GATES_DIR.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps({'passed': rep['passed'], 'pin_sha256': rep['pin_sha256'], 'ext_sha256': rep['ext_sha256'], 'report': str(out)}, indent=1) + '\n')
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('kind', choices=('pins', 'gate', 'single-pin', 'single-gate')); ap.add_argument('--final-mix', type=pathlib.Path)
    ap.add_argument('--weights-dir', type=pathlib.Path); ap.add_argument('--family')
    ap.add_argument('--out', type=pathlib.Path); ap.add_argument('--eps-pred', action='append', default=[], help='NAME=PATH')
    ap.add_argument('--name', help='mix pin / receipt / instance name (default: final-mix name); e.g. a rehearsal name')
    a = ap.parse_args(argv)
    if a.kind in ('gate', 'single-gate') and a.out is None:
        ap.error('--out is required for gates')
    eps = dict(x.split('=', 1) for x in a.eps_pred)
    if a.kind == 'pins':
        r = pins(a.final_mix, a.name)
    elif a.kind == 'gate':
        r = gate(a.final_mix, a.out, eps, a.name)
    elif a.kind == 'single-pin':
        p = expl_pin(a.weights_dir); r = {'pin': rel(p), 'sha256': sha(p)}
    else:
        if len(eps) != 1 or a.family not in eps:
            ap.error('single-gate needs --family F and --eps-pred F=PATH')
        r = single_gate(a.family, eps[a.family], a.out)
    print(json.dumps(r, indent=1))


if __name__ == '__main__':
    main()
