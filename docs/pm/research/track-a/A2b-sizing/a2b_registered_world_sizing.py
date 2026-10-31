"""EXP-OPE-RES-02 (A2b) design-time TAB-R3 stand-in sizing on the REGISTERED A2 TAB world. DATA-FREE.

The registered EXP-OPE-RES-01 TAB world is rebuilt from the public, sealed A2 stage outputs that the PM
copied into registered-world-inputs.json (A2-S0-a1 s0.json tab_marginals: TRAIN marginals, 64 frozen C0
situations, signs, kappa_p, zeta_p; A2-S1-a1 tab.json eta_p and the baseline acceptance summary). The world
is built exactly as the A2 runner does it (run_ope_a2.py situation_from_doc / marginals_from_doc :1249-1262,
mirrored below because the runner module cannot be imported without its stage machinery; stage_s1 :1428-1433
builds TabWorld(marginals, law, eta = scale(eta_p), kappa, zeta)), with the A2 module ope_synthetic_a2
(TabWorld, _expected_dr, _solve, _visitation) imported read-only. Fidelity is checked against the sealed
record before anything else: eta_p recomputed by the A2 rule, and the exact baseline beta_0 of the four A2
acceptance cells T-A1..T-A4.

Then the A2b stand-in denominator D2 (design section 5.3) is sized on that world with the formulas of the
reviewed toy sizing script a2b_tab_r3_sizing.py (imported read-only; its 90 fixture rows are re-computed here
and must match its output file exactly): the same theta / omega / eta / t grid, plus three additions that
the toy run could not give: the exact pure-swap outcome in all eight acceptance cells C1..C8, the spread over
alternative eps draws (sizing-only streams, not registered), and a first-order worst-direction D2 at the same
KL (a feasible D2, so its bias is a lower bound on the worst case at that distance).

It reads no rehearsal row, ledger, reward, model artifact, T7 path or 2026 file; it fits nothing; the TRAIN
information it uses is the aggregate marginals already sealed in A2-S0 (TRAIN 2023-05-15..2025-04-30). It
writes only the output JSON. Run with python -B from outside the A2 worktree.
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

HERE = os.path.dirname(os.path.abspath(__file__))
REGISTERED_EPS_KEY = (20261028, 7)            # design 9.4: eps field SeedSequence([20261028, 7]) (role d2_field)
ALT_EPS_KEY = (20261028, 7, 1000)             # sizing-only alternative draws [20261028, 7, 1000, j]; not registered
ALT_DRAWS = 100
FD_STEP = 1e-4                                # central-difference step on log D2 for the worst-direction gradient


def sha256_file(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- mirror of run_ope_a2.py :1249-1262
def situation_from_doc(syn, GameState, doc):
    st = doc['state']
    state = GameState(st['inning'], st['half'], st['outs'], st['bases'], st['home_score'], st['away_score'])
    return syn.Situation(state, {e: [tuple(o) for o in rows] for e, rows in doc['outcomes'].items()})


def marginals_from_doc(syn, GameState, doc):
    values = {k: doc[k] for k in syn.TabMarginals.FIELDS}
    values['situations'] = [situation_from_doc(syn, GameState, s) for s in doc['situations']]
    for k in ('outcome_freq', 'count_group_freq', 'league_shares', 'archetype_shares', 'x_dist', 'bip_mix', 'class_table',
              'first_half_table', 'second_half_table'):
        values[k] = np.asarray(values[k], dtype=np.float64)
    values['groups'] = tuple(values['groups'])
    return syn.TabMarginals(**values)


# ---------------------------------------------------------------- shared row computation (toy formulas)
class Ctx:
    """The pieces of one TabWorld that the stand-in computations need."""
    def __init__(self, syn, w):
        self.syn, self.w = syn, w
        self.S, self.A, self.G = w.S, w.A, w.G
        self.sup = np.broadcast_to(w.support[None], (12, w.A, w.G))
        self.d0 = np.asarray(w.pi_b)
        self.vis_ca = np.asarray(w.visits).sum(axis=1)
        q_world = syn._solve(w.pi_b_full, w.pb, w.ps, w.pf, w.imm_world, w.walk, w.strikeout)[1]
        err = (q_world - w.q_hat).mean(axis=1)
        self.zbar = np.where(self.sup, (err - err[self.sup].mean()) / err[self.sup].std(), 0.0)
        self.pols = {'cand': np.asarray(w.pi_c), 'ref': np.asarray(w.pi_b_full)}

    def full(self, x):
        return np.broadcast_to(x[:, None], (12, self.S, self.A, self.G))

    def beta(self, law, den, policy_den=None):
        w, syn = self.w, self.syn
        y = {k: syn._expected_dr(p, law, den if policy_den is None else policy_den(p, den), w.q_hat, w.pb, w.ps, w.pf,
                                 w.imm_world, w.walk, w.strikeout) for k, p in self.pols.items()}
        return float((y['cand'][0, 0] - y['ref'][0, 0]).mean()) - w.delta_true


def row_for(tz, ctx, eps, klm, omega, *, label, eta_name, lambdas=True):
    """One sizing row with exactly the fields and formulas of a2b_tab_r3_sizing.py (toy) main()."""
    w, B = ctx.w, tz.B
    sup, d0, vis_ca = ctx.sup, ctx.d0, ctx.vis_ca
    xi = math.sqrt(1 - omega * omega) * eps + omega * ctx.zbar
    theta = tz.calibrate_theta(d0, xi, sup, vis_ca, klm * tz.KL_RECORDED)
    d2 = tz.tilt(d0, xi, theta, sup)
    dens = {'B0': ctx.full(d0), 'B2': ctx.full(d2)}
    row = {'s': label, 'eta': eta_name, 'klm': klm, 'om': omega, 'theta': tz.r5(theta),
           'kl': tz.r5(tz.mean_kl(d2, d0, sup, vis_ca)), 'dtrue': tz.r5(w.delta_true),
           'beta': {}, 'ess': {}, 'q99': {}, 'max': {}, 'adequacy': {}}
    if klm == 1.0 and lambdas:
        row['beta_lambda'] = {}
    for law_name in tz.LAWS:
        law = ctx.full(tz.law_of(law_name, d0, d2, sup))
        row['beta'][law_name] = [tz.r5(ctx.beta(law, dens['B0'])), tz.r5(ctx.beta(law, dens['B2']))]
        if klm == 1.0 and lambdas and law_name in ('t0', 't2'):
            row['beta_lambda'][law_name] = {
                str(int(lam)): [tz.r5(ctx.beta(law, den, lambda p, d, lam=lam: tz.shrunk_denominator(p, d, lam)))
                                for den in (dens['B0'], dens['B2'])] for lam in tz.LAMBDAS}
        if law_name in ('t0', 't1', 't2'):
            ess, q99, mx = adequacy_parts(tz, ctx, law, dens)
            row['ess'][law_name], row['q99'][law_name], row['max'][law_name] = ess, q99, mx
            row['adequacy'][law_name] = band_check(tz, ess, q99)
    bt = row['beta']
    row['swap_cells'] = {
        'C1_t0_abs': abs(bt['t0'][1]) <= B,
        'C2_V3_rel': abs(bt['V3'][1]) <= abs(bt['V3'][0]) + B,
        'C5_t1_abs': abs(bt['t1'][1]) <= B,
        'C6_t2_rel': abs(bt['t2'][1]) <= abs(bt['t2'][0]) + B,
    }
    return row, d2, theta


def adequacy_parts(tz, ctx, law, dens):
    w = ctx.w
    vis = ctx.syn._visitation(law, w.pb, w.pf, w.ps)
    ess, q99, mx = {}, {}, {}
    for k, p in ctx.pols.items():
        e0 = tz.ess_per_n(p, law, dens['B0'], w.pb, w.ps, w.pf)
        e2 = tz.ess_per_n(p, law, dens['B2'], w.pb, w.ps, w.pf)
        ess[k] = tz.r5(e2 / e0) if (e0 and e2 is not None) else None
        t0q, t0m = tz.rho_tail(p, law, dens['B0'], vis)
        t2q, t2m = tz.rho_tail(p, law, dens['B2'], vis)
        q99[k], mx[k] = [tz.r5(t0q), tz.r5(t2q)], [tz.r5(t0m), tz.r5(t2m)]
    return ess, q99, mx


def band_check(tz, ess, q99):
    ref_ess = ess['ref']
    q_ratio = q99['cand'][1] / q99['cand'][0]
    lo1, hi1 = tz.ADEQUACY_BAND['reference_pa_ess_ratio']
    lo2, hi2 = tz.ADEQUACY_BAND['candidate_rho_q99_ratio']
    return {'ref_ess': ref_ess, 'cand_q99_ratio': tz.r5(q_ratio),
            'in_band': bool(ref_ess is not None and lo1 <= ref_ess <= hi1 and lo2 <= q_ratio <= hi2)}


def quick_draw(tz, ctx, eps, klm, omega):
    """Compact per-draw outcome at one setting: theta, |beta_2|/B at t0 (C1), C2, C6 and the t1 adequacy."""
    B, sup, d0, vis_ca = tz.B, ctx.sup, ctx.d0, ctx.vis_ca
    xi = math.sqrt(1 - omega * omega) * eps + omega * ctx.zbar
    theta = tz.calibrate_theta(d0, xi, sup, vis_ca, klm * tz.KL_RECORDED)
    d2 = tz.tilt(d0, xi, theta, sup)
    dens = {'B0': ctx.full(d0), 'B2': ctx.full(d2)}
    out = {}
    for law_name in ('t0', 'V3', 't2'):
        law = ctx.full(tz.law_of(law_name, d0, d2, sup))
        out[law_name] = (ctx.beta(law, dens['B0']), ctx.beta(law, dens['B2']))
    ess, q99, _ = adequacy_parts(tz, ctx, ctx.full(tz.law_of('t1', d0, d2, sup)), dens)
    adq = band_check(tz, ess, q99)
    lo1, hi1 = tz.ADEQUACY_BAND['reference_pa_ess_ratio']
    return {'b2_t0_B': abs(out['t0'][1]) / B,
            'C1': abs(out['t0'][1]) <= B,
            'C2': abs(out['V3'][1]) <= abs(out['V3'][0]) + B,
            'C6': abs(out['t2'][1]) <= abs(out['t2'][0]) + B,
            'ref_ess_t1': adq['ref_ess'], 'q99_ratio_t1': adq['cand_q99_ratio'], 'in_band_t1': adq['in_band'],
            'ess_only_in_band_t1': bool(adq['ref_ess'] is not None and lo1 <= adq['ref_ess'] <= hi1)}


def quantiles(vals):
    v = np.sort(np.asarray(vals, dtype=np.float64))
    return {q: float(f'{float(np.quantile(v, p)):.4g}') for q, p in
            (('min', 0.0), ('q10', 0.1), ('q25', 0.25), ('median', 0.5), ('q75', 0.75), ('q90', 0.9), ('max', 1.0))}


def worst_direction(tz, ctx):
    """First-order bias-maximizing log-tilt direction for the pure swap in C1 (logging D0, denominator D2):
    G = d beta_2 / d log D2(c, a, g) at D2 = D0 by central differences; maximizing sum G * xi subject to the
    D0-visitation-weighted mean of Var_D0(xi | c, a) gives xi* = (G / D0 - sum_g G) / v(c, a) on the support."""
    sup, d0, vis_ca = ctx.sup, ctx.d0, ctx.vis_ca
    law = ctx.full(d0)
    grad = np.zeros_like(d0)
    for c, a, g in zip(*np.nonzero(sup)):
        vals = []
        for sgn in (1.0, -1.0):
            den = d0.copy()
            den[c, a, g] *= math.exp(sgn * FD_STEP)
            vals.append(ctx.beta(law, ctx.full(den)))
        grad[c, a, g] = (vals[0] - vals[1]) / (2 * FD_STEP)
    vbar = vis_ca / vis_ca.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        xi = np.where(sup, grad / np.where(sup, d0, 1.0) - grad.sum(-1, keepdims=True), 0.0)
        xi = np.where(vbar[..., None] > 0, xi / vbar[..., None], 0.0)
    xi = np.where(sup, xi, 0.0)
    xi = np.where(sup, (xi - xi[sup].mean()) / xi[sup].std(), 0.0)
    zb = ctx.zbar
    cos = float((xi * zb)[sup].sum() / math.sqrt((xi * xi)[sup].sum() * (zb * zb)[sup].sum()))
    return xi, cos, float(np.abs(grad).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', required=True)
    ap.add_argument('--toy-output', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    root = os.environ.get('A2_PITCHMDP')
    if not root:
        raise SystemExit('set A2_PITCHMDP to the read-only A2 worktree experiments/pitchmdp directory')
    sys.dont_write_bytecode = True
    sys.path.insert(0, root)
    from pitchmdp import ope_synthetic_a2 as syn  # read-only import, nothing modified
    from pitchmdp.game import GameState
    toy_path = os.path.join(HERE, 'a2b_tab_r3_sizing.py')
    tz = load_module('a2b_tab_r3_sizing_readonly', toy_path)
    fx = load_module('ope_a2_fixtures_readonly', os.path.join(root, 'tests', 'ope_a2_fixtures.py'))
    started = time.time()
    B = tz.B

    # 1. the toy rows are reproduced exactly by this script's row function (same formulas)
    toy = json.load(open(args.toy_output, encoding='utf-8'))
    toy_rows = {(r['s'], r['eta'], r['klm'], r['om']): r for r in toy['rows']}
    reproduced, mismatched = 0, []
    for seed in tz.WORLD_SEEDS:
        mg_t = fx.synthetic_tab_marginals(seed=seed)
        eta_t = float(syn.TabWorld.eta_p(mg_t))
        eps_t = np.random.default_rng(np.random.SeedSequence([tz.A2B_BASE, tz.D2_FIELD_ROLE, seed])).standard_normal((12, 8, 4))
        for eta_name, eta in (('0', 0.0), ('eta_p', eta_t)):
            ctx_t = Ctx(syn, syn.TabWorld(mg_t, law='V2', eta=eta, kappa=0.0, zeta=0.0))
            for klm in tz.KL_MULTIPLIERS:
                for om in tz.OMEGAS:
                    row, _, _ = row_for(tz, ctx_t, eps_t, klm, om, label=seed, eta_name=eta_name)
                    if json.loads(json.dumps(row)) == toy_rows[(seed, eta_name, klm, om)]:
                        reproduced += 1
                    else:
                        mismatched.append([seed, eta_name, klm, om])
    toy_check = {'rows_reproduced_exactly': reproduced, 'rows_total': len(toy['rows']), 'mismatched': mismatched}

    # 2. the registered world and its fidelity against the sealed A2 record
    inputs = json.load(open(args.inputs, encoding='utf-8'))
    mg = marginals_from_doc(syn, GameState, inputs['tab_marginals'])
    eta_p = float(syn.TabWorld.eta_p(mg))
    rec = inputs['tab_acceptance_baseline_summary']
    cell_specs = {'T-A1': ('V2', 0.0, 0.0), 'T-A2': ('V3', 0.0, 0.0), 'T-A3': ('V2', 2.0 * float(mg.kappa_p), 0.0),
                  'T-A4': ('V2', 0.0, 2.0 * float(mg.zeta_p))}
    fidelity = {'eta_p': {'recomputed': eta_p, 'recorded': inputs['eta_p'], 'abs_diff': abs(eta_p - inputs['eta_p'])}}
    worlds = {}
    for cid, (law, kappa, zeta) in cell_specs.items():
        wc = syn.TabWorld(mg, law=law, eta=eta_p, kappa=kappa, zeta=zeta)
        if law == 'V2':
            syn.tab_integrity(wc)
        recorded = rec[cid]['R1_0xR2_0']['b_hat']
        fidelity[cid] = {'law': law, 'beta_0_recomputed': wc.beta_0, 'beta_0_recorded': recorded,
                         'abs_diff': abs(wc.beta_0 - recorded), 'bit_identical': wc.beta_0 == recorded,
                         'delta_true': wc.delta_true, 'tau': wc.tau}
        worlds[cid] = wc
    fidelity['pass'] = bool(fidelity['eta_p']['abs_diff'] <= 1e-12 and
                            all(fidelity[c]['abs_diff'] <= 1e-15 for c in cell_specs))

    # 3. the toy grid on the registered world (registered eps field)
    eps = np.random.default_rng(np.random.SeedSequence(list(REGISTERED_EPS_KEY))).standard_normal((12, mg.archetype_shares.shape[0], len(mg.groups)))
    rows, d2_reg = [], {}
    ctx_eta = {'0': Ctx(syn, syn.TabWorld(mg, law='V2', eta=0.0, kappa=0.0, zeta=0.0)), 'eta_p': Ctx(syn, worlds['T-A1'])}
    for eta_name, ctx in ctx_eta.items():
        for klm in tz.KL_MULTIPLIERS:
            for om in tz.OMEGAS:
                row, d2, _ = row_for(tz, ctx, eps, klm, om, label='registered', eta_name=eta_name)
                rows.append(row)
                if eta_name == 'eta_p' and klm == 1.0:
                    d2_reg[om] = d2
    v3_row = next(r for r in rows if r['eta'] == 'eta_p' and r['klm'] == 1.0 and r['om'] == 0.5)
    fidelity['grid_beta0_V3_equals_T-A2'] = {'grid_value_5sig': v3_row['beta']['V3'][0],
                                            'T-A2_recorded_5sig': tz.r5(rec['T-A2']['R1_0xR2_0']['b_hat'])}

    # 4. exact pure-swap outcome in the eight acceptance cells (D2 built once in the C1 world, fixed across cells)
    cells = (('C1', 't0', 'T-A1'), ('C2', 'V3', 'T-A1'), ('C3', 't0', 'T-A3'), ('C4', 't0', 'T-A4'), ('C5', 't1', 'T-A1'),
             ('C6', 't2', 'T-A1'), ('C7', 't1', 'T-A3'), ('C8', 't1', 'T-A4'))
    b2_rule = {'C1': 'abs', 'C3': 'abs', 'C4': 'abs', 'C5': 'abs', 'C7': 'abs', 'C8': 'abs', 'C2': 'rel', 'C6': 'rel'}
    ctx_cell = {'T-A1': ctx_eta['eta_p'], 'T-A3': Ctx(syn, worlds['T-A3']), 'T-A4': Ctx(syn, worlds['T-A4'])}
    exact_cells = {}
    for om in tz.OMEGAS:
        d2 = d2_reg[om]
        out = {}
        for cid, law_name, wid in cells:
            ctx = ctx_cell[wid]
            sup, d0 = ctx.sup, ctx.d0
            law = ctx.full(tz.law_of(law_name, d0, d2, sup))
            b0, b2 = ctx.beta(law, ctx.full(d0)), ctx.beta(law, ctx.full(d2))
            bound = B if b2_rule[cid] == 'abs' else abs(b0) + B
            out[cid] = {'law': law_name, 'world': wid, 'beta_B0_over_B': tz.r5(b0 / B), 'beta_B2_over_B': tz.r5(b2 / B),
                        'rule': b2_rule[cid], 'bound_over_B': tz.r5(bound / B),
                        'pure_swap': 'EXACT_PASS' if abs(b2) <= bound else 'EXACT_FAIL'}
        exact_cells[f'om{om}'] = out

    # 5. spread over alternative eps draws (sizing-only streams)
    draws = {'om1.0': 'no eps component at omega = 1 (xi = zbar): identical to the grid row (deterministic)'}
    ctx = ctx_eta['eta_p']
    for om in (0.0, 0.5):
        recs = []
        for j in range(1, ALT_DRAWS + 1):
            e = np.random.default_rng(np.random.SeedSequence([*ALT_EPS_KEY, j])).standard_normal(eps.shape)
            recs.append(quick_draw(tz, ctx, e, 1.0, om))
        draws[f'om{om}'] = {
            'draws': ALT_DRAWS,
            'abs_beta2_t0_over_B': quantiles([r['b2_t0_B'] for r in recs]),
            'C1_pass': sum(r['C1'] for r in recs), 'C2_pass': sum(r['C2'] for r in recs), 'C6_pass': sum(r['C6'] for r in recs),
            'C1_C2_C6_pass': sum(r['C1'] and r['C2'] and r['C6'] for r in recs),
            'in_band_t1': sum(r['in_band_t1'] for r in recs), 'ess_only_in_band_t1': sum(r['ess_only_in_band_t1'] for r in recs),
            'C1_and_in_band_t1': sum(r['C1'] and r['in_band_t1'] for r in recs),
            'C1_and_ess_only_t1': sum(r['C1'] and r['ess_only_in_band_t1'] for r in recs),
            'ref_ess_t1': quantiles([r['ref_ess_t1'] for r in recs if r['ref_ess_t1'] is not None]),
            'cand_q99_ratio_t1': quantiles([r['q99_ratio_t1'] for r in recs]),
            'rows_compact': [[tz.r5(r['b2_t0_B']), int(r['C1']), int(r['in_band_t1']), tz.r5(r['ref_ess_t1']),
                              tz.r5(r['q99_ratio_t1'])] for r in recs],
        }

    # 6. first-order worst direction at the same distance (eta_p)
    xi_w, cos_w, gmax = worst_direction(tz, ctx)
    worst = {'cos_with_zbar': round(cos_w, 4), 'max_abs_gradient': tz.r5(gmax), 'rows': []}
    for klm in tz.KL_MULTIPLIERS:
        best = None
        for sign in (1.0, -1.0):
            theta = tz.calibrate_theta(ctx.d0, sign * xi_w, ctx.sup, ctx.vis_ca, klm * tz.KL_RECORDED)
            d2 = tz.tilt(ctx.d0, sign * xi_w, theta, ctx.sup)
            b2 = ctx.beta(ctx.full(ctx.d0), ctx.full(d2))
            if best is None or abs(b2) > abs(best[0]):
                best = (b2, sign, theta, d2)
        b2, sign, theta, d2 = best
        dens = {'B0': ctx.full(ctx.d0), 'B2': ctx.full(d2)}
        ess, q99, mx = adequacy_parts(tz, ctx, ctx.full(tz.law_of('t1', ctx.d0, d2, ctx.sup)), dens)
        worst['rows'].append({'klm': klm, 'sign': sign, 'theta': tz.r5(theta), 'kl': tz.r5(tz.mean_kl(d2, ctx.d0, ctx.sup, ctx.vis_ca)),
                              'abs_beta2_t0_over_B': tz.r5(abs(b2) / B), 'C1': abs(b2) <= B,
                              'adequacy_t1': band_check(tz, ess, q99), 'cand_rho_max_t1_D2': mx['cand'][1]})

    def sel(klm, om, eta):
        return next(r for r in rows if r['klm'] == klm and r['om'] == om and r['eta'] == eta)

    reg = sel(1.0, 0.5, 'eta_p')
    headline = {
        'setting': 'registered stand-in: KL x 1, omega 0.5, eta_p, registered eps field',
        'theta': reg['theta'], 'kl': reg['kl'],
        'abs_beta2_t0_over_B (C1, pure swap, B0 true)': tz.r5(abs(reg['beta']['t0'][1]) / B),
        'C1': reg['swap_cells']['C1_t0_abs'],
        'abs_beta0_t1_over_B (current baseline when B2 is true)': tz.r5(abs(reg['beta']['t1'][0]) / B),
        'abs_beta0_V3_over_B (= A2 T-A2)': tz.r5(abs(reg['beta']['V3'][0]) / B),
        'adequacy': {t: reg['adequacy'][t] for t in ('t0', 't1', 't2')},
        'omega_sensitivity_C1_over_B': {f'om{om}': tz.r5(abs(sel(1.0, om, 'eta_p')['beta']['t0'][1]) / B) for om in tz.OMEGAS},
        'kl_sensitivity_C1_over_B_om0.5': {f'kl{k}': tz.r5(abs(sel(k, 0.5, 'eta_p')['beta']['t0'][1]) / B) for k in tz.KL_MULTIPLIERS},
        'eta0_C1_over_B_om0.5': tz.r5(abs(sel(1.0, 0.5, '0')['beta']['t0'][1]) / B),
    }
    out = {
        'what': 'EXP-OPE-RES-02 design-time TAB-R3 stand-in sizing on the registered EXP-OPE-RES-01 TAB world; a sizing, not a result',
        'data_statement': ('reads only registered-world-inputs.json (public sealed A2-S0-a1 TRAIN marginals and C0 situations, '
                           'A2-S1-a1 eta_p and baseline acceptance summary) and the toy sizing output; no rehearsal row, ledger, '
                           'reward, model artifact, T7 path or 2026 file; nothing is fitted; no registration is needed'),
        'code': {'script_sha256': sha256_file(os.path.abspath(__file__)), 'toy_script_sha256': sha256_file(toy_path),
                 'toy_output_sha256': sha256_file(args.toy_output), 'inputs_sha256': sha256_file(args.inputs),
                 'inputs_sources': inputs['sources'],
                 'ope_synthetic_a2_sha256': sha256_file(os.path.join(root, 'pitchmdp', 'ope_synthetic_a2.py')),
                 'ope_redesign_registry_sha256': sha256_file(os.path.join(root, 'pitchmdp', 'ope_redesign_registry.py')),
                 'run_ope_a2_sha256_mirrored_lines_1249_1262': sha256_file(os.path.join(root, 'scripts', 'run_ope_a2.py')),
                 'game_sha256': sha256_file(os.path.join(root, 'pitchmdp', 'game.py')),
                 'ope_a2_fixtures_sha256': sha256_file(os.path.join(root, 'tests', 'ope_a2_fixtures.py')),
                 'numpy': np.__version__, 'python': sys.version.split()[0]},
        'toy_reproduction_check': toy_check,
        'fidelity_vs_sealed_A2_record': fidelity,
        'definitions': {
            'world': 'TabWorld(marginals from A2-S0-a1, law, eta, kappa, zeta) with the A2 fixed draws role_sequence(tab_world, 0, component)',
            'eps_registered': 'SeedSequence([20261028, 7]) standard normal (12, 8, 4) (design 9.4)',
            'eps_alternative': f'SeedSequence([20261028, 7, 1000, j]), j = 1..{ALT_DRAWS}; sizing-only, not registered streams',
            'grid_rows': 'same fields and formulas as a2b_tab_r3_sizing.py rows (D2 rebuilt from the world of each eta, as in the toy run)',
            'exact_cells': 'pure swap R1_0 x R2_0 x B2 by exact DP in C1..C8 with one D2 built in the C1 world (eta_p, kappa = zeta = 0) and kept fixed across cells; rule for B2 tuples: absolute in C1, C3, C4, C5, C7, C8, relative (|beta_0| + B) in C2, C6',
            'worst_direction': 'xi* = (G / D0 - sum_g G) / v(c, a) on the support, G = d beta_2(C1) / d log D2 at D2 = D0 (central differences, step 1e-4); theta calibrated to the same KL; sign maximizing |beta_2|; a feasible D2, so |beta_2| is a lower bound on the worst case at that distance',
            'adequacy_band': tz.ADEQUACY_BAND, 'B': B,
        },
        'headline': headline,
        'exact_cells_kl1_eta_p': exact_cells,
        'eps_draw_spread_kl1_eta_p': draws,
        'worst_direction_eta_p': worst,
        'rows': rows,
        'seconds': round(time.time() - started, 1),
    }
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, separators=(',', ':'))
        f.write('\n')
    print(json.dumps({'toy_check': toy_check, 'fidelity_pass': fidelity['pass'], 'headline': headline,
                      'exact_om0.5': {k: (v['beta_B2_over_B'], v['pure_swap']) for k, v in exact_cells['om0.5'].items()},
                      'draws': {om: ({k: v for k, v in d.items() if k != 'rows_compact'} if isinstance(d, dict) else d)
                                for om, d in draws.items()},
                      'worst': worst, 'seconds': out['seconds']}, indent=1, default=str))


if __name__ == '__main__':
    main()
