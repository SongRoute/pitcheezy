"""Export the mix E_PS probabilities (fan_numbers --pred .npz: game_pk, n, probs[PITCH_NAMES order], built). No labels read.
usage: python -B export_eps_mix.py <worktree> <mix pin> <final-mix.json> <out.npz>"""
import hashlib, json, pathlib, sys
import numpy as np
wt, mixp, fmp, out = map(pathlib.Path, sys.argv[1:5])
sys.path.insert(0, str(wt / 'src'))
from pitcheezy.svc_lit import mix as MX
from pitcheezy.svc_lit.bundle import CLASSES
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
mix = json.loads(mixp.read_text()); fm = json.loads(fmp.read_text())
assert mix['source']['sha256'] == sha(fmp)
qs, keys = [], None
for c in mix['components']:
    e = next(x for x in fm['components'] if x['name'] == c['name'])['eps_pred']
    assert sha(e['path']) == e['sha256'], c['name']
    z = np.load(e['path'], allow_pickle=False)
    k = (z['eps_game_pk'], z['eps_n'], z['eps_built'])
    if keys is None: keys = k
    assert all(np.array_equal(a, b) for a, b in zip(keys, k)), c['name']
    qs.append(np.asarray(z['q_tau'], np.float64))
q = MX.combine(qs, mix['weights'], mix['method'], mix.get('temperature'))
gpk, n, built = keys
probs = np.full((len(gpk), 9), 1 / 9); probs[built] = q
assert np.isfinite(probs).all() and (probs >= 0).all() and np.allclose(probs.sum(1), 1)
assert CLASSES == ('FF', 'SI', 'FC', 'SL', 'ST', 'CU', 'CH', 'FS', 'OT')
out.parent.mkdir(parents=True, exist_ok=True)
if out.exists(): raise SystemExit(f'{out} exists')
np.savez(out, game_pk=gpk, n=n, probs=probs, built=built)
print(json.dumps({'out': str(out), 'sha256': sha(out), 'rows': int(len(gpk)), 'built': int(built.sum()), 'mix_pin_sha256': sha(mixp),
                  'final_mix_sha256': sha(fmp), 'method': mix['method'], 'weights': mix['weights'], 'temperature': mix.get('temperature')}, indent=1))
