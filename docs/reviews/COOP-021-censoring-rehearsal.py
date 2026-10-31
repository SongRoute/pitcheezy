"""COOP-021 rehearsal of censoring designs on the sealed <=2025 S6-dr-a2 outputs.

Exposed development data (DEV 2025, already read and reported in D115); read-only; no 2026 row.
Usage: PYTHONPATH=experiments/pitchmdp python docs/reviews/COOP-021-censoring-rehearsal.py OUT.json
"""
import json
import sys

import numpy as np
import pandas as pd

from pitchmdp.policy_estimator import game_bootstrap

S = '/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/ML-MATRIX-20260924/ML-POLICY-VAL-v1/S6-dr-a2/'
d = pd.read_parquet(S + 'pa_values.parquet')
boot = json.load(open(S + 'dr.json'))['bootstrap']
DRAWS, SEED, MIN = boot['draws'], boot['seed'], {'games': 0, 'pa_starts': 0}

# ledger: |v_c(H) - v_r(H)| = |sum (pi_c - pi_r) q| on evaluated rows; first-row pitcher; positivity events
gaps, first, posit = [], {}, []
with open(S + 'ledger-dev-0.jsonl') as f:
    for line in f:
        r = json.loads(line)
        if r.get('kind') != 'decision':
            continue
        if r['decision_index'] == 0:
            first[r['pa_id']] = r['pitcher']
        if r['status'] == 'UNSUPPORTED_LOGGING_POSITIVITY':
            posit.append((r['pa_id'], r['pitcher'], r['logged_action']))
        res = r.get('result')
        if r['status'] in ('SUPPORTED', 'OUTSIDE_POLICY_SUPPORT') and res and res.get('q_reference'):
            q = np.array([np.nan if v is None else v for v in res['q_reference']])
            m = np.array(res['mask'])
            gaps.append(float(np.array(res['candidate'])[m] @ q[m] - np.array(res['reference'])[m] @ q[m]))
gaps = np.abs(np.array(gaps))
out = {'evaluated_rows': len(gaps), 'positivity_rows': len(posit),
       'positivity_pitchers': len({p for _, p, _ in posit}),
       'positivity_pitcher_action_pairs': len({(p, a) for _, p, a in posit}),
       'abs_vc_minus_vr': dict(zip(('median', 'q99', 'q999', 'max'),
                                   map(float, np.quantile(gaps, [.5, .99, .999, 1.]))))}

e0 = d[d.in_population].copy().reset_index(drop=True)
n = len(e0)
lo0 = e0.delta_bounds.map(lambda b: b[0]).to_numpy()
hi0 = e0.delta_bounds.map(lambda b: b[1]).to_numpy()
cb = e0.candidate_bounds.map(lambda b: b[0]).to_numpy()
rb = e0.reference_bounds.map(lambda b: b[0]).to_numpy()
sc, sr = e0.weight_candidate.to_numpy(), e0.weight_reference.to_numpy()
cens = (e0.status == 'CENSORED').to_numpy()
reason, kind = e0.reason.to_numpy(), e0.kind.to_numpy()
POS = reason == 'UNSUPPORTED_LOGGING_POSITIVITY'
HK = np.isin(reason, ['UNSUPPORTED_NO_LOGGED_ACTION', 'UNSUPPORTED_UNKNOWN_PITCHER'])  # fixed by H_k


def shared(mask, lo, hi, delta=0.):
    """Node unknown u in [0,1] shared by both policies plus a policy gap |V_c - V_r| <= delta."""
    lo, hi = lo.copy(), hi.copy()
    base, slope = cb[mask] - rb[mask], sc[mask] - sr[mask]
    lo[mask] = base + np.minimum(0, slope) - sc[mask] * delta
    hi[mask] = base + np.maximum(0, slope) + sc[mask] * delta
    return lo, hi


def summarize(name, lo, hi, keep=None, note=''):
    keep = np.ones(n, bool) if keep is None else keep
    rows = [{'game': g, 'status': s, 'delta_bounds': [a, b], 'delta': a if s == 'COMPLETE' else 0.}
            for g, s, a, b in zip(e0.game[keep], e0.status[keep], lo[keep], hi[keep])]
    bs = game_bootstrap(rows, draws=DRAWS, seed=SEED, invalid_share_max=1., minimum=MIN)
    out[name] = {'pas': int(keep.sum()), 'L1_delta_bounds': [float(lo[keep].mean()), float(hi[keep].mean())],
                 'width': float((hi[keep] - lo[keep]).mean()),
                 'lower_endpoint_ci95': bs['L1_lower_endpoint_ci95'],
                 'upper_endpoint_ci95': bs['L1_upper_endpoint_ci95'], 'note': note}


summarize('v1_current', lo0, hi0, note='reproduces sealed dr.json')
out['width_by_reason'] = {r: {'pas': int((reason == r).sum()), 'width_contrib': float((hi0 - lo0)[reason == r].sum() / n)}
                          for r in sorted(set(reason[cens]))}
out['width_by_node'] = {str(int(k)): float((hi0 - lo0)[(e0.node == k).to_numpy()].sum() / n)
                        for k in sorted(e0.node.dropna().unique())}
out['positivity_at_node0'] = int((POS & (e0.node == 0).to_numpy()).sum())
out['censored_mean_abs_weight_gap'] = float(np.abs(sc - sr)[cens].mean())

lo, hi = shared(cens & (kind == 'REFUSED'), lo0, hi0)
summarize('A_shared_all_refusals', lo, hi,
          note='A: one shared unknown at every refusal node; not a valid regime for POSITIVITY (post-decision)')
lo, hi = shared(HK, lo0, hi0)
summarize('A_hk_natural_course_only', lo, hi, note='H_k-fixed refusals -> natural course (shared end value) only')
for tag in ('max', 'q999'):
    delta = out['abs_vc_minus_vr'][tag]
    lo, hi = shared(HK, lo0, hi0)
    lo, hi = shared(POS, lo, hi, delta)
    summarize(f'R_delta_{tag}', lo, hi, note=f'HK natural course + POSITIVITY rho=0 with |v_c-v_r| <= {delta:.5f} '
                                              '(rehearsal proxy; 2026 runtime records v_pi(H_k) exactly)')
lo, hi = shared(HK, lo0, hi0)
lo, hi = shared(POS, lo, hi, 0.)
summarize('R_limit_gap0', lo, hi, note='limit v_c = v_r at positivity nodes')

# B: first-row pitcher revealed an off-TRAIN-support pitch type in a strictly earlier PA
ids = e0.pa_id.str.split(':', expand=True).astype(int)
e0['g'], e0['ab'], e0['p0'] = ids[0], ids[1], e0.pa_id.map(first)
month = dict(zip(e0.game, e0.stratum_month))
events = [(month[int(p.split(':')[0])], int(p.split(':')[0]), int(p.split(':')[1]), pit)
          for p, pit, _ in posit if int(p.split(':')[0]) in month]


def revealed(row, across):
    return any(pit == row.p0 and ((g == row.g and ab < row.ab) or
                                  (across and g != row.g and (m, g) < (row.stratum_month, row.g)))
               for m, g, ab, pit in events)


for across in (False, True):
    tag = 'within_and_across_games' if across else 'within_game'
    keep = ~e0.apply(lambda r: revealed(r, across), axis=1).to_numpy()
    summarize(f'B_{tag}', lo0, hi0, keep, note='v1 bounds; PAs of pre-PA revealed new-pitch pitchers leave E0'
              + ('; across-game order approximated by (month, game_pk)' if across else ''))
    out[f'B_{tag}']['positivity_left'] = int((POS & keep).sum())
    lo, hi = shared(HK, lo0, hi0)
    lo, hi = shared(POS, lo, hi, out['abs_vc_minus_vr']['max'])
    summarize(f'R_plus_B_{tag}', lo, hi, keep, note='recommended (delta max proxy) + B restriction')

for tag in ('q99', 'q999', 'max'):  # delta calibrated on <=2025 evaluated rows only
    delta = out['abs_vc_minus_vr'][tag]
    lo, hi = shared(cens & (kind != 'TERMINAL_VALUE_MISSING'), lo0, hi0, delta)
    summarize(f'C_delta_{tag}', lo, hi, note=f'C: shared unknown + |V_c - V_r| <= {delta:.5f} at every censored node')

# lower-endpoint SE (from the percentile CI) and a naive projection to a 2,430-game season
for name, v in list(out.items()):
    if isinstance(v, dict) and 'lower_endpoint_ci95' in v:
        se = (v['lower_endpoint_ci95'][1] - v['lower_endpoint_ci95'][0]) / 3.92
        v['lower_endpoint_se'], v['lower_endpoint_se_2430_games'] = se, se * (150 / 2430) ** .5

json.dump(out, open(sys.argv[1], 'w'), indent=1)
print(json.dumps(out, indent=1))
