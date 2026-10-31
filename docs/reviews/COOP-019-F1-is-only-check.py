"""COOP-019 F1 reviewer-side check: IS-only (q-hat := 0) vs DR on the sealed S5 V2 (pi_b_hat) world.

Read-only on the sealed stage. Independent re-implementation of the D89 recursion
(cross-checked against pitchmdp.policy_estimator.affine).

Missing from the ledger: the per-PA terminal reward (pas[pa_id]['reward'], the terminal WE /
cutoff value) is held in memory in run_world and never persisted (ledger rows carry only
decisions; v2.json carries per-start aggregates). Consequences:
  * Per-start MEAN IS-only is still exact: DR_j = (bc_j - br_j) + (sc_j - sr_j) r_j and
    IS_j = (sc_j - sr_j) r_j (q := 0 kills every base term), so mean IS = delta_dr - mean(bc - br).
  * Per-PA values (for per-start IS se and the per-PA DR reproduction) need r_j: recovered by
    solving the over-determined system given by v2.json per-start aggregates of the three
    planning seeds (logs are byte-identical across seeds; rewards are shared). The fit residual
    is the reproduction check.
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import least_squares

STAGE = '/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/ML-MATRIX-20260924/ML-POLICY-VAL-v1/S5-v2-a1'
OUT = os.path.dirname(os.path.abspath(__file__))
SEEDS = (0, 1, 2)
sys.dont_write_bytecode = True
sys.path.insert(0, '/Users/song/Projects/pitcheezy/experiments/pitchmdp')
from pitchmdp import policy_estimator as PE  # noqa: E402  (read-only use for the cross-check)


def lines_of(decisions):
    """Independent recursion: (base, slope) per policy; q-hat from q_reference on the mask."""
    out = {}
    for name in ('candidate', 'reference'):
        base, slope = 0., 1.
        for d in reversed(decisions):
            res = d['result']
            mask = np.array(res['mask'], bool)
            p = np.array(res[name], float)
            logging = np.array(res['logging'], float)
            q = np.array([np.nan if v is None else v for v in res['q_reference']], float)
            a = res['logged_index']
            rho = p[a] / logging[a]
            assert np.isclose(rho, res[f'rho_{name}'], rtol=1e-12, atol=0)
            v = float(p[mask] @ q[mask])
            base, slope = (v, 0.) if rho == 0 else (v + rho * (base - q[a]), rho * slope)
        out[name] = (base, slope)
    return out


def load(seed):
    by_pa = {}
    with open(f'{STAGE}/ledger-dev-{seed}-pi_b_hat.jsonl') as f:
        for line in f:
            r = json.loads(line)
            if r.get('kind') == 'decision':
                by_pa.setdefault(r['pa_id'], []).append(r)
    lines, maxdiff = {}, 0.
    for pa, ds in by_pa.items():
        ds.sort(key=lambda d: d['decision_index'])
        mine = lines_of(ds)
        steps = PE._steps(ds)
        for name in mine:
            ref = PE.affine(steps, name)
            maxdiff = max(maxdiff, abs(ref[0] - mine[name][0]), abs(ref[1] - mine[name][1]))
        lines[pa] = mine
    return lines, maxdiff


def main():
    v2 = json.load(open(f'{STAGE}/v2.json'))
    runs = {r['planning_seed']: r for r in v2['runs'] if r['law'] == 'pi_b_hat'}
    L, report = {}, {'cross_check_vs_repo_affine_maxdiff': {}}
    for s in SEEDS:
        L[s], md = load(s)
        report['cross_check_vs_repo_affine_maxdiff'][s] = md
    starts = sorted({int(p[1:].split(':')[0]) for p in L[0]})
    pas_of = {i: sorted([p for p in L[0] if int(p[1:].split(':')[0]) == i], key=lambda p: int(p.split(':')[1]))
              for i in starts}
    # reward recovery per start (shared across seeds)
    rewards, fit_res = {}, []
    for i in starts:
        pas = pas_of[i]
        n = len(pas)
        B = {s: np.array([[L[s][p]['candidate'][0], L[s][p]['reference'][0]] for p in pas]) for s in SEEDS}
        S = {s: np.array([[L[s][p]['candidate'][1], L[s][p]['reference'][1]] for p in pas]) for s in SEEDS}
        ps = {s: runs[s]['per_start'][i] for s in SEEDS}
        for s in SEEDS:
            assert ps[s]['start'] == i and ps[s]['logs'] == n

        def resid(r):
            out = []
            for s in SEEDS:
                c = B[s][:, 0] + S[s][:, 0] * r
                ref = B[s][:, 1] + S[s][:, 1] * r
                dl = c - ref
                out += [c.mean() - ps[s]['dr_mean']['candidate'], ref.mean() - ps[s]['dr_mean']['reference'],
                        dl.mean() - ps[s]['delta_dr'],
                        c.std(ddof=1) / np.sqrt(n) - ps[s]['dr_se']['candidate'],
                        ref.std(ddof=1) / np.sqrt(n) - ps[s]['dr_se']['reference'],
                        dl.std(ddof=1) / np.sqrt(n) - ps[s]['delta_dr_se']]
            return np.array(out)
        best = None
        rng = np.random.default_rng(i)
        for k in range(40):
            x0 = rng.uniform(0.05, 0.95, n)
            sol = least_squares(resid, x0, bounds=(0, 1), xtol=1e-15, ftol=1e-15, gtol=1e-15)
            if best is None or sol.cost < best.cost:
                best = sol
            if np.abs(best.fun).max() < 1e-12:
                break
        rewards[i] = best.x
        fit_res.append(float(np.abs(best.fun).max()))
    report['reward_recovery_max_abs_residual'] = max(fit_res)
    report['reward_recovery_starts_with_residual_gt_1e-9'] = int(sum(r > 1e-9 for r in fit_res))

    for s in SEEDS:
        run = runs[s]
        dr_gap, is_gap, dr_repro, is_exact_vs_pa, se_pipe_dr, se_pipe_is, truth_se = [], [], [], [], [], [], []
        for i in starts:
            pas, r = pas_of[i], rewards[i]
            ps, n = run['per_start'][i], len(pas_of[i])
            bc = np.array([L[s][p]['candidate'][0] for p in pas]); sc = np.array([L[s][p]['candidate'][1] for p in pas])
            br = np.array([L[s][p]['reference'][0] for p in pas]); sr = np.array([L[s][p]['reference'][1] for p in pas])
            dr_j = (bc - br) + (sc - sr) * r
            is_j = (sc - sr) * r
            is_mean_exact = ps['delta_dr'] - float(np.mean(bc - br))  # reward-free exact mean
            dr_repro.append(abs(dr_j.mean() - ps['delta_dr']))
            is_exact_vs_pa.append(abs(is_j.mean() - is_mean_exact))
            t = ps['truth']
            dr_gap.append(ps['delta_dr'] - t['delta']); is_gap.append(is_mean_exact - t['delta'])
            se_pipe_dr.append(ps['delta_dr_se']); se_pipe_is.append(is_j.std(ddof=1) / np.sqrt(n))
            truth_se.append(t['delta_mc_se'])
        m = len(starts)
        tse = np.array(truth_se)

        def summ(g, se):
            g, se = np.array(g), np.array(se)
            mean = float(g.mean())
            se_p = float(np.sqrt((se ** 2 + tse ** 2).sum()) / m)  # run_world formula
            se_e = float(g.std(ddof=1) / np.sqrt(m))              # empirical across independent starts
            return {'mean_gap': mean, 'se_pipeline_formula': se_p, 'ci95_pipeline': [mean - 1.96 * se_p, mean + 1.96 * se_p],
                    'se_empirical_starts': se_e, 'ci95_empirical': [mean - 1.96 * se_e, mean + 1.96 * se_e],
                    'z_pipeline': mean / se_p}
        report[f'seed{s}'] = {
            'dr_reproduction_max_abs_err_per_start': float(max(dr_repro)),
            'dr_gap_matches_v2_pooled': abs(float(np.mean(dr_gap)) - run['delta_gap']),
            'is_mean_pa_vs_exact_max_abs': float(max(is_exact_vs_pa)),
            'DR': summ(dr_gap, se_pipe_dr), 'IS_only_q0': summ(is_gap, se_pipe_is),
            'mean_candidate_weight': float(np.mean([L[s][p]['candidate'][1] for p in L[s]])),
            'mean_reference_weight': float(np.mean([L[s][p]['reference'][1] for p in L[s]])),
        }
    json.dump(report, open(f'{OUT}/is_only_result.json', 'w'), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
