"""D-9 S3b: G0 P3 tau from ledger overlap and search noise, never from outcomes (COOP-018, D93).

Input is the candidate-mode ledger of June 2025 R1 requests (no reward, WE, events kind or
next-row state is read) and, per PA, only whether its last row carries a terminal marker. For
each tau of the registered grid the candidate law is recomputed from the recorded planning Q
exactly as the runtime does (``kl_policy(Q_planning, pi_ref, M, tau)``); the PA weight is the
product of per-decision ratios over the logged actions (OUTSIDE_POLICY_SUPPORT rows give 0). The
selection rule is registered before the ledger exists: the reference must pass the overlap
thresholds, else ARM-B is not evaluable; among the grid, the smallest tau whose candidate passes
overlap, the candidate/reference ESS ratio and the noise threshold is selected; none -> P3 not
evaluable. The grid is never widened within one registration.
"""
from __future__ import annotations

import numpy as np

from .policy_artifacts import _require
from .policy_estimator import EVALUATED, effective_sample_size
from .rollout_policy import kl_policy

THRESHOLD_KEYS = ('pa_ess_ratio_min', 'game_ess_min', 'ess_ratio_candidate_reference_min', 'noise_ratio_q90_max',
                  'safety_multiplier')


def _decision(row, tau):
    r = row['result']
    mask = np.asarray(r['mask'], dtype=bool)
    reference, logging = np.asarray(r['reference']), np.asarray(r['logging'])
    q = np.array([np.nan if v is None else v for v in r['q_planning']], dtype=np.float64)
    candidate = kl_policy(np.where(mask, q, -np.inf), reference, mask, tau)
    a = r['logged_index']
    noise = np.array([np.nan if v is None else v for v in r['q_planning_diff_se']], dtype=np.float64)
    kl = float(np.sum(candidate[mask] * np.log(candidate[mask] / reference[mask]), where=candidate[mask] > 0))
    return candidate[a] / logging[a], reference[a] / logging[a], kl, float(noise[mask].max()) / tau


def tau_table(ledger_decisions, pa_facts, taus, thresholds):
    """Overlap/noise table per tau and the registered selection. ``pa_facts``: {pa_id: {'game',
    'in_population', 'terminal_marker'}}; weights use E0 PAs whose decisions are all evaluated
    and whose last row carries a terminal marker (the rest are counted, not weighted)."""
    _require(set(thresholds) == set(THRESHOLD_KEYS) and all(isinstance(thresholds[k], (int, float))
             and thresholds[k] > 0 for k in THRESHOLD_KEYS), 'registered tau thresholds required')
    _require(len(taus) >= 1 and list(taus) == sorted(taus) and all(t > 0 for t in taus), 'registered tau grid')
    by_pa = {}
    for row in ledger_decisions:
        by_pa.setdefault(row['pa_id'], []).append(row)
    _require(set(by_pa) <= set(pa_facts), 'every ledger PA needs its facts')
    usable = [pa for pa, rows in by_pa.items() if pa_facts[pa]['in_population'] and pa_facts[pa]['terminal_marker']
              and pa_facts[pa].get('problem') is None and all(r['status'] in EVALUATED for r in rows)]
    _require(usable, 'no weighted PA: the tau table is undefined')
    games = [pa_facts[pa]['game'] for pa in usable]
    table = []
    for tau in taus:
        w_c, w_r, kls, noise = [], [], [], []
        for pa in usable:
            wc = wr = 1.
            for row in sorted(by_pa[pa], key=lambda r: r['decision_index']):
                rc, rr, kl, nz = _decision(row, tau)
                wc, wr = wc * rc, wr * rr
                kls.append(kl)
                noise.append(nz)
            w_c.append(wc)
            w_r.append(wr)
        entry = {'tau': tau, 'pas': len(usable)}
        for name, w in (('candidate', w_c), ('reference', w_r)):
            pa_ess = effective_sample_size(w)
            entry[name] = {'pa_ess': pa_ess, 'pa_ess_ratio': None if pa_ess is None else pa_ess / len(usable),
                           'game_ess': effective_sample_size(w, games)}
        entry['mean_kl_candidate_reference'] = float(np.mean(kls))
        _require(np.isfinite(noise).all(), 'paired-difference noise undefined (search samples <= 1): the D-9 rule '
                 'needs samples >= 3')
        entry['noise_ratio_quantiles'] = {str(q): float(np.quantile(noise, q)) for q in (.5, .9)}
        table.append(entry)

    def passes(entry, name):
        e = entry[name]
        return (e['pa_ess_ratio'] is not None and e['game_ess'] is not None
                and e['pa_ess_ratio'] >= thresholds['pa_ess_ratio_min'] * thresholds['safety_multiplier']
                and e['game_ess'] >= thresholds['game_ess_min'])
    if not passes(table[0], 'reference'):  # the reference weights do not depend on tau
        return {'table': table, 'selected_tau': None, 'status': 'ARM_B_NOT_EVALUABLE',
                'counted_not_weighted_pas': len(by_pa) - len(usable)}
    for entry in table:
        ratio = entry['candidate']['pa_ess'] / entry['reference']['pa_ess'] if entry['candidate']['pa_ess'] else 0.
        if (passes(entry, 'candidate') and ratio >= thresholds['ess_ratio_candidate_reference_min']
                and entry['noise_ratio_quantiles']['0.9'] <= thresholds['noise_ratio_q90_max']):
            return {'table': table, 'selected_tau': entry['tau'], 'status': 'SELECTED',
                    'counted_not_weighted_pas': len(by_pa) - len(usable)}
    return {'table': table, 'selected_tau': None, 'status': 'P3_NOT_EVALUABLE',
            'counted_not_weighted_pas': len(by_pa) - len(usable)}


def select_search_settings(profiles, row_budget):
    """D-9a: the largest registered (samples, pitch_cap) whose MEASURED conditional rows per evaluated
    decision (S3, one run per candidate on the same starts), scaled to the registered decision
    count, fit the registered row budget. Rollouts stop at the PA end, so rows are not assumed to
    scale with pitch_cap. ``profiles``: [{'samples', 'pitch_cap', 'decisions', 'conditional_rows'}].
    Returns None when nothing fits (P3 not evaluable at that budget)."""
    _require(profiles and all(p['decisions'] > 0 and p['conditional_rows'] > 0 for p in profiles),
             'S3 profile measured nothing')
    fitting = [p for p in profiles
               if p['conditional_rows'] / p['decisions'] * row_budget['decisions'] <= row_budget['rows']]
    best = max(fitting, key=lambda p: (p['samples'] * p['pitch_cap'], p['samples'])) if fitting else None
    return None if best is None else {'samples': best['samples'], 'pitch_cap': best['pitch_cap']}
