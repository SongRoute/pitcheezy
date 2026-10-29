"""EXP-OPE-RES-02 (A2b) design-time TAB-R3 stand-in sizing. SYNTHETIC AND DATA-FREE.

Builds synthetic TAB worlds with the EXP-OPE-RES-01 test fixture (tests/ope_a2_fixtures.py,
synthetic_tab_marginals: toy WE, toy advancement, Dirichlet frequencies; its docstring says "No real
data, model or file is read"), constructs the A2b stand-in denominator D2 around the TAB logging law D0
as A2b design section 5.3 specifies, and computes by exact dynamic programming with the EXP-OPE-RES-01
module ope_synthetic_a2 (TabWorld, _expected_dr, _solve, _visitation; imported read-only) the DR bias
for both denominators under the logging-law family b_t ~ D0^(1-t) * D2^t, the induced PA-ESS ratio and
the decision-weight tails. It reads no real row, ledger, model artifact, TRAIN marginal, sealed output
or 2026 file, fits nothing to data and writes only the output JSON. Run with python -B.

Fixed in the A2b design v1 before any sizing run: omega = 0.5 (primary), KL target 0.048863351237125756
(recorded A3 mean KL(B2 || B0) on S6), t family {0, 1, 2} (0.5 and V3 descriptive). The adequacy band
numbers were fixed before the first run; its evaluation point is t = 1 (see README, revision note). The
output is a sizing, not a result.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time

import numpy as np

B = 1.26e-05
KL_RECORDED = 0.048863351237125756  # results/EXP-BC-CAL-01.json stages[09-s6-weights][weights.json].arms.B2.kl_new_vs_b0.mean
RECORDED = {
    'kl_B2_vs_B0': KL_RECORDED,
    'pa_ess_ratio_B2_over_B0': {'candidate': 6340.025838071803 / 9120.218896471748,
                                'reference': 6592.232520207458 / 9725.694335061527},
    'candidate_rho_q99': {'B0': 1.3805970621754051, 'B2': 1.9218155219752069},
    'candidate_rho_max': {'B0': 3.97351720470025, 'B2': 18.0788561729745},
    'reference_mean_rho': {'B0': 0.9895888876169783, 'B2': 0.9578918688562033},
    'real_TAB_V3_beta_0': 8.92541220275217e-06,  # results/EXP-OPE-RES-01.json S1_tab_summary.acceptance.T-A2.R1_0xR2_0.b_hat
    'real_TAB_eta_p': 0.16405664825560962,       # results/EXP-OPE-RES-01.json S1_tab_summary.eta_p
}
_LOSS = 1 - RECORDED['pa_ess_ratio_B2_over_B0']['reference']                                  # 0.322
_EXCESS = RECORDED['candidate_rho_q99']['B2'] / RECORDED['candidate_rho_q99']['B0'] - 1      # 0.392
ADEQUACY_BAND = {'reference_pa_ess_ratio': [1 - 1.5 * _LOSS, 1 - 0.5 * _LOSS],
                 'candidate_rho_q99_ratio': [1 + 0.5 * _EXCESS, 1 + 1.5 * _EXCESS]}
ADEQUACY_POINT = 't1'  # registered evaluation point (README revision note); t0 and t2 are reported too
A2B_BASE, D2_FIELD_ROLE = 20261028, 7
OMEGAS = (0.0, 0.5, 1.0)
KL_MULTIPLIERS = (0.5, 1.0, 2.0)
LAWS = ('t0', 't0.5', 't1', 't2', 'V3')
LAMBDAS = (100.0, 10.0, 3.0)
WORLD_SEEDS = (0, 1, 2, 3, 4)


def sha256_file(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def r5(x):
    return None if x is None else float(f'{x:.5g}')


def norm_g(x, support):
    x = np.where(support, x, 0.0)
    return x / x.sum(-1, keepdims=True)


def mean_kl(d2, d0, support, vis_ca):
    with np.errstate(divide='ignore', invalid='ignore'):
        kl = np.where(support & (d2 > 0), d2 * np.log(d2 / d0), 0.0).sum(-1)
    return float((vis_ca * kl).sum() / vis_ca.sum())


def tilt(d0, xi, theta, support):
    with np.errstate(divide='ignore', invalid='ignore'):
        logw = np.where(support, np.log(np.where(support, d0, 1.0)) + theta * xi, -np.inf)
    logw -= logw.max(-1, keepdims=True)
    return norm_g(np.exp(logw), support)


def calibrate_theta(d0, xi, support, vis_ca, target):
    lo, hi = 0.0, 0.25
    while mean_kl(tilt(d0, xi, hi, support), d0, support, vis_ca) < target:
        hi *= 2.0
        if hi > 1e3:
            raise RuntimeError('KL target unreachable')
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mean_kl(tilt(d0, xi, mid, support), d0, support, vis_ca) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-13 * max(1.0, hi):
            break
    return hi


def law_of(name, d0, d2, support):
    if name == 'V3':
        return norm_g(np.where(support, d0 ** 0.5, 0.0), support)
    t = float(name[1:])
    with np.errstate(divide='ignore', invalid='ignore'):
        logw = np.where(support, (1 - t) * np.log(np.where(support, d0, 1.0)) + t * np.log(np.where(support, d2, 1.0)), -np.inf)
    logw -= logw.max(-1, keepdims=True)
    return norm_g(np.exp(logw), support)


def shrunk_denominator(policy, den, lam):
    """A denominator whose ratio policy/den' is the shrinkage weight lam*rho/(rho^2 + lam)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        rho = np.where(den > 0, policy / den, 0.0)
        rt = lam * rho / (rho * rho + lam)
        return np.where((rt > 0) & (policy > 0), policy / np.where(rt > 0, rt, 1.0), den)


def weight_moment(policy, law, den, pb, ps, pf, k):
    """E[W^k], W = PA-end product of rho, exact recursion over the count tree (uniform starts);
    inf when the two-strike foul loop makes the k-th moment diverge."""
    S, A = policy.shape[1], policy.shape[2]
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.where(den > 0, policy / den, 0.0)
    f = law * r ** k
    pterm = 1.0 - pb - ps - pf
    M = np.zeros((4, 3, S, A))
    for b in (3, 2, 1, 0):
        for st in (2, 1, 0):
            c = b * 3 + st
            mball = (np.ones((S, A)) if b == 3 else M[b + 1, st])[..., None]
            if st == 2:
                loop = (f[c] * pf[c]).sum(-1)
                if np.any(loop >= 1.0):
                    return float('inf')
                M[b, 2] = (f[c] * (pb[c] * mball + ps[c] + pterm[c])).sum(-1) / (1.0 - loop)
            else:
                nxt = M[b, st + 1][..., None]
                M[b, st] = (f[c] * (pb[c] * mball + (ps[c] + pf[c]) * nxt + pterm[c])).sum(-1)
    return float(M[0, 0].mean())


def ess_per_n(policy, law, den, pb, ps, pf):
    m1 = weight_moment(policy, law, den, pb, ps, pf, 1)
    m2 = weight_moment(policy, law, den, pb, ps, pf, 2)
    return None if not math.isfinite(m2) or m2 <= 0 else m1 * m1 / m2


def rho_tail(policy, law, den, vis):
    weights = vis[..., None] * law
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.where(den > 0, policy / den, 0.0)
    keep = weights > 1e-15
    vals, ww = r[keep], weights[keep]
    order = np.argsort(vals, kind='stable')
    vals, ww = vals[order], ww[order]
    cum = np.cumsum(ww) / ww.sum()
    return float(vals[min(np.searchsorted(cum, 0.99), len(vals) - 1)]), float(vals.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    root = os.environ.get('A2_PITCHMDP')
    if not root:
        raise SystemExit('set A2_PITCHMDP to the read-only A2 worktree experiments/pitchmdp directory')
    sys.dont_write_bytecode = True
    sys.path.insert(0, root)
    from pitchmdp import ope_synthetic_a2 as syn  # read-only import, nothing modified
    fixture_path = os.path.join(root, 'tests', 'ope_a2_fixtures.py')
    spec = importlib.util.spec_from_file_location('ope_a2_fixtures_readonly', fixture_path)
    fx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fx)
    started = time.time()
    rows, worlds = [], []
    for seed in WORLD_SEEDS:
        mg = fx.synthetic_tab_marginals(seed=seed)
        eta_p = float(syn.TabWorld.eta_p(mg))
        worlds.append({'seed': seed, 'eta_p': r5(eta_p)})
        for eta_name, eta in (('0', 0.0), ('eta_p', eta_p)):
            w = syn.TabWorld(mg, law='V2', eta=eta, kappa=0.0, zeta=0.0)
            S, A, G = w.S, w.A, w.G
            sup = np.broadcast_to(w.support[None], (12, A, G))
            d0 = np.asarray(w.pi_b)
            vis_ca = np.asarray(w.visits).sum(axis=1)
            q_world = syn._solve(w.pi_b_full, w.pb, w.ps, w.pf, w.imm_world, w.walk, w.strikeout)[1]
            err = (q_world - w.q_hat).mean(axis=1)
            zbar = np.where(sup, (err - err[sup].mean()) / err[sup].std(), 0.0)
            eps = np.random.default_rng(np.random.SeedSequence([A2B_BASE, D2_FIELD_ROLE, seed])).standard_normal((12, A, G))
            full = lambda x: np.broadcast_to(x[:, None], (12, S, A, G))
            pols = {'cand': np.asarray(w.pi_c), 'ref': np.asarray(w.pi_b_full)}
            for klm in KL_MULTIPLIERS:
                for omega in OMEGAS:
                    xi = math.sqrt(1 - omega * omega) * eps + omega * zbar
                    theta = calibrate_theta(d0, xi, sup, vis_ca, klm * KL_RECORDED)
                    d2 = tilt(d0, xi, theta, sup)
                    dens = {'B0': full(d0), 'B2': full(d2)}
                    row = {'s': seed, 'eta': eta_name, 'klm': klm, 'om': omega, 'theta': r5(theta),
                           'kl': r5(mean_kl(d2, d0, sup, vis_ca)), 'dtrue': r5(w.delta_true),
                           'beta': {}, 'ess': {}, 'q99': {}, 'max': {}, 'adequacy': {}}
                    if klm == 1.0:
                        row['beta_lambda'] = {}
                    for law_name in LAWS:
                        law = full(law_of(law_name, d0, d2, sup))
                        betas = []
                        for den in (dens['B0'], dens['B2']):
                            y = {k: syn._expected_dr(p, law, den, w.q_hat, w.pb, w.ps, w.pf, w.imm_world, w.walk, w.strikeout)
                                 for k, p in pols.items()}
                            betas.append(r5(float((y['cand'][0, 0] - y['ref'][0, 0]).mean()) - w.delta_true))
                        row['beta'][law_name] = betas
                        if klm == 1.0 and law_name in ('t0', 't2'):
                            lam_out = {}
                            for lam in LAMBDAS:
                                pair = []
                                for den in (dens['B0'], dens['B2']):
                                    y = {k: syn._expected_dr(p, law, shrunk_denominator(p, den, lam), w.q_hat, w.pb, w.ps, w.pf,
                                                             w.imm_world, w.walk, w.strikeout) for k, p in pols.items()}
                                    pair.append(r5(float((y['cand'][0, 0] - y['ref'][0, 0]).mean()) - w.delta_true))
                                lam_out[str(int(lam))] = pair
                            row['beta_lambda'][law_name] = lam_out
                        if law_name in ('t0', 't1', 't2'):
                            vis = syn._visitation(law, w.pb, w.pf, w.ps)
                            ess, q99, mx = {}, {}, {}
                            for k, p in pols.items():
                                e0 = ess_per_n(p, law, dens['B0'], w.pb, w.ps, w.pf)
                                e2 = ess_per_n(p, law, dens['B2'], w.pb, w.ps, w.pf)
                                ess[k] = r5(e2 / e0) if (e0 and e2 is not None) else None
                                t0q, t0m = rho_tail(p, law, dens['B0'], vis)
                                t2q, t2m = rho_tail(p, law, dens['B2'], vis)
                                q99[k], mx[k] = [r5(t0q), r5(t2q)], [r5(t0m), r5(t2m)]
                            row['ess'][law_name], row['q99'][law_name], row['max'][law_name] = ess, q99, mx
                            ref_ess = ess['ref']
                            q_ratio = q99['cand'][1] / q99['cand'][0]
                            lo1, hi1 = ADEQUACY_BAND['reference_pa_ess_ratio']
                            lo2, hi2 = ADEQUACY_BAND['candidate_rho_q99_ratio']
                            row['adequacy'][law_name] = {'ref_ess': ref_ess, 'cand_q99_ratio': r5(q_ratio),
                                                         'in_band': bool(ref_ess is not None and lo1 <= ref_ess <= hi1 and lo2 <= q_ratio <= hi2)}
                    bt = row['beta']
                    row['swap_cells'] = {
                        'C1_t0_abs': abs(bt['t0'][1]) <= B,
                        'C2_V3_rel': abs(bt['V3'][1]) <= abs(bt['V3'][0]) + B,
                        'C5_t1_abs': abs(bt['t1'][1]) <= B,
                        'C6_t2_rel': abs(bt['t2'][1]) <= abs(bt['t2'][0]) + B,
                    }
                    rows.append(row)

    def sel(klm, om, eta):
        return [r for r in rows if r['klm'] == klm and r['om'] == om and r['eta'] == eta]

    def span(vals):
        vals = [v for v in vals if v is not None]
        return [r5(min(vals)), r5(max(vals))] if vals else None

    summary = {}
    for klm in KL_MULTIPLIERS:
        for om in OMEGAS:
            for eta in ('0', 'eta_p'):
                rs = sel(klm, om, eta)
                summary[f'kl{klm}_om{om}_eta{eta}'] = {
                    'abs_beta2_t0_over_B': span([abs(r['beta']['t0'][1]) / B for r in rs]),
                    'abs_beta0_t1_over_B': span([abs(r['beta']['t1'][0]) / B for r in rs]),
                    'abs_beta0_t2_over_B': span([abs(r['beta']['t2'][0]) / B for r in rs]),
                    'abs_beta2_t2_over_B': span([abs(r['beta']['t2'][1]) / B for r in rs]),
                    'abs_beta0_V3_over_B': span([abs(r['beta']['V3'][0]) / B for r in rs]),
                    'ref_ess_ratio': {t: span([r['ess'][t]['ref'] for r in rs]) for t in ('t0', 't1', 't2')},
                    'cand_q99_ratio': {t: span([r['adequacy'][t]['cand_q99_ratio'] for r in rs]) for t in ('t0', 't1', 't2')},
                    'in_band_count': {t: sum(r['adequacy'][t]['in_band'] for r in rs) for t in ('t0', 't1', 't2')},
                    'swap_C1_pass_count': sum(r['swap_cells']['C1_t0_abs'] for r in rs),
                    'swap_C2_pass_count': sum(r['swap_cells']['C2_V3_rel'] for r in rs),
                    'swap_C6_pass_count': sum(r['swap_cells']['C6_t2_rel'] for r in rs),
                    'worlds': len(rs),
                }
    lam_summary = {}
    for om in OMEGAS:
        rs = sel(1.0, om, 'eta_p')
        lam_summary[f'om{om}'] = {f'lambda{int(l)}': span([abs(r['beta_lambda']['t0'][str(int(l))][1]) / B for r in rs])
                                  for l in LAMBDAS}
    script_path = os.path.abspath(__file__)
    module_path = os.path.join(root, 'pitchmdp', 'ope_synthetic_a2.py')
    registry_path = os.path.join(root, 'pitchmdp', 'ope_redesign_registry.py')
    out = {
        'what': 'EXP-OPE-RES-02 design-time TAB-R3 stand-in sizing; synthetic and data-free; a sizing, not a result',
        'data_statement': 'no real row, ledger, model artifact, TRAIN marginal, sealed output or 2026 file is read; synthetic TAB marginals from the EXP-OPE-RES-01 test fixture; nothing is fitted to data; no registration is needed',
        'code': {'script_sha256': sha256_file(script_path), 'ope_synthetic_a2_sha256': sha256_file(module_path),
                 'ope_redesign_registry_sha256': sha256_file(registry_path), 'ope_a2_fixtures_sha256': sha256_file(fixture_path),
                 'a2_worktree_pitchmdp': root, 'numpy': np.__version__, 'python': sys.version.split()[0]},
        'definitions': {
            'D0': 'TAB logging law pi_b (archetype x count x group); reference policy = D0; candidate = KL tilt of D0 (TabWorld, mean KL 0.0074)',
            'xi': 'sqrt(1-omega^2)*eps + omega*zbar; eps ~ N(0,1) per (count, archetype, group) from SeedSequence([20261028, 7, world_seed]); zbar = situation average of (Q_world - q_hat) standardized over the support',
            'D2': 'D0*exp(theta*xi) renormalized on the D0 support; theta by bisection: D0-visitation-weighted mean KL(D2||D0) = klm x 0.048863351237125756',
            'laws': 't: b_t ~ D0^(1-t) D2^t; V3: D0^0.5 (the A2 T-A2 law)',
            'beta': '[beta_B0, beta_B2] = exact E[Delta_hat_DR] - Delta_true with the D0 / D2 denominator (pure baseline / pure swap), kappa = zeta = 0, uniform starts; beta_lambda = the same with the shrinkage weight lambda*rho/(rho^2+lambda) at a fixed lambda (descriptive)',
            'ess': 'ESS ratio D2 over D0 denominator, ESS/N = E[W]^2/E[W^2] with W the PA-end product of rho (exact); null = infinite second moment in the two-strike foul loop',
            'q99_max': '[D0, D2] decision-level rho quantile 0.99 and max, weights = visits under the law x the law',
            'adequacy_band_numbers_fixed_before_first_run': ADEQUACY_BAND,
            'adequacy_point': ADEQUACY_POINT,
            'swap_cells': 'exact pure-swap outcomes: C1 |beta_B2(t0)| <= B; C2 |beta_B2(V3)| <= |beta_B0(V3)| + B; C5 |beta_B2(t1)| <= B (identity); C6 |beta_B2(t2)| <= |beta_B0(t2)| + B',
            'B': B,
        },
        'grid': {'world_seeds': WORLD_SEEDS, 'eta': ['0', 'eta_p (A2 rule ECE10 = 2 x 0.0087, recalibrated per synthetic world)'],
                 'kl_multipliers': KL_MULTIPLIERS, 'omegas': OMEGAS, 'laws': LAWS, 'lambdas_t0_t2_kl1': LAMBDAS},
        'recorded_targets': RECORDED,
        'worlds': worlds,
        'summary': summary,
        'shrinkage_t0_beta2_over_B_eta_p_kl1': lam_summary,
        'rows': rows,
        'seconds': round(time.time() - started, 1),
    }
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, separators=(',', ':'))
        f.write('\n')
    print(json.dumps({'kl1.0_om0.5_etaeta_p': summary['kl1.0_om0.5_etaeta_p'], 'seconds': out['seconds']}, indent=1))


if __name__ == '__main__':
    main()
