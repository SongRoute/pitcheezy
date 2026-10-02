"""Additive N3/R78 scorer for ML-JUNE-CALIBRATION-v1.

The old I1/I2 Holm2/R52 helpers stay unchanged.  Three fixed contrasts
(B1-B0, B2-B0, B2-B1) share every whole-game bootstrap draw: game indices are
drawn once per population/group with the legacy RNG protocol and applied to
each contrast's per-game loss sums with the legacy expressions, so each slot
equals what ``paired_game_comparison``/``bootstrap_difference`` would report.
The shared draws make ``C2 = C1 + C3`` checkable at the point estimate and at
every replicate; a violation means misaligned rows/draws and stops scoring.
"""
from __future__ import annotations

import numpy as np

from .matrix_group_metrics import masks as fixed_group_masks
from .matrix_june_calibration import FAMILY_ID, RESEARCH_STATUS, SEEDS, VARIANTS
from .matrix_metrics import holm_adjust, pitch_losses, prediction_metrics
from .matrix_panel import group_reporting_status

CONTRASTS = (('B1-B0', 'B1', 'B0'), ('B2-B0', 'B2', 'B0'), ('B2-B1', 'B2', 'B1'))
CONTRAST_IDS = tuple(c[0] for c in CONTRASTS)
LOSSES = ('NLL', 'Brier')
R_GROUPS = ('role_starter', 'role_relief', 'hand_L', 'hand_R', 'volume_low', 'volume_middle', 'volume_high',
            'volume_zero', 'two_strikes', 'less_two_strikes', 'runners_on', 'bases_empty', 'non_cpanel')
N_SETTINGS = {'draws': 10000, 'seed': 20260924, 'alpha': 0.05, 'delta_nll_max': -0.003, 'nll_upper_below': 0.0,
              'adjusted_p_max': 0.05, 'brier_upper_max': 0.001, 'min_negative_seeds': 4,
              'worse_nll_lower_above': 0.0, 'worse_brier_lower_above': 0.001, 'missing_slot_p': 1.0}
R_SETTINGS = {'draws': 100000, 'seed': 20260924, 'alpha': 0.05, 'family_size': 78, 'nll_margin': 0.010,
              'brier_margin': 0.002, 'min_games': 30, 'min_pitches': 500}
ADDITIVITY_ATOL = 1e-12
CANDIDATE_RULES = {'B1': ('B1-B0',), 'B2': ('B2-B0', 'B2-B1')}
DESCRIPTIVE_DRAWS = 10000


class ContrastAdditivityError(RuntimeError):
    """Row or draw misalignment: (B2-B0) != (B1-B0) + (B2-B1) beyond 1e-12."""


def check_scoring_config(sci):
    primary, robust = sci['primary'], sci['robustness']
    fixed = (primary['bootstrap_draws'], primary['bootstrap_seed'], primary['alpha'], primary['delta_nll_max'],
             primary['nll_ci95_upper_strictly_below'], primary['adjusted_p_max'], primary['brier_ci95_upper_max'],
             primary['min_negative_matched_seed_deltas'], primary['missing_slot_p'], tuple(primary['holm_order']),
             primary['quantile_method'], tuple(primary['ci_quantiles']), primary['family_size'],
             primary['worse_rule']['nll_ci95_lower_strictly_above'], primary['worse_rule']['brier_ci95_lower_strictly_above'])
    expected = (N_SETTINGS['draws'], N_SETTINGS['seed'], N_SETTINGS['alpha'], N_SETTINGS['delta_nll_max'],
                N_SETTINGS['nll_upper_below'], N_SETTINGS['adjusted_p_max'], N_SETTINGS['brier_upper_max'],
                N_SETTINGS['min_negative_seeds'], N_SETTINGS['missing_slot_p'], CONTRAST_IDS, 'linear',
                (0.025, 0.975), 3, N_SETTINGS['worse_nll_lower_above'], N_SETTINGS['worse_brier_lower_above'])
    if fixed != expected:
        raise ValueError('Primary N3 settings differ from the scorer')
    if (robust['bootstrap_draws'], robust['bootstrap_seed'], robust['alpha'], robust['family_size'], robust['nll_margin'],
            robust['brier_margin'], robust['min_games'], robust['min_pitches'], tuple(robust['groups']),
            robust['contrast_count'], robust['quantile_method'], robust['retain_missing_slots']) != (
            R_SETTINGS['draws'], R_SETTINGS['seed'], R_SETTINGS['alpha'], R_SETTINGS['family_size'],
            R_SETTINGS['nll_margin'], R_SETTINGS['brier_margin'], R_SETTINGS['min_games'], R_SETTINGS['min_pitches'],
            R_GROUPS, 3, 'linear', True):
        raise ValueError('R78 settings differ from the scorer')
    identity = sci['contrast_identity_check']
    if (identity['absolute_tolerance'], identity['relative_tolerance'], identity['completion_field']) != (
            ADDITIVITY_ATOL, 0, 'contrast_additivity_passed'):
        raise ValueError('Contrast additivity rule differs')
    rules = sci['candidate_eligibility']
    if (tuple(rules['B1']['require_N_and_R']), tuple(rules['B2']['require_N_and_R'])) != (
            CANDIDATE_RULES['B1'], CANDIDATE_RULES['B2']) or rules['service_promotion'] is not False:
        raise ValueError('Candidate eligibility rules differ')
    return sci


# ---------------------------------------------------------------- shared whole-game draws

def legacy_paired_chunk(n_games):
    """``paired_game_comparison`` chunk rule."""
    return max(1, min(256, 2_000_000 // n_games))


def legacy_bound_chunk(n_games):
    """``bootstrap_difference`` chunk rule."""
    return 256


def _game_index(games):
    games = np.asarray(games)
    if games.ndim != 1:
        raise ValueError('One game identity per pitch required')
    unique, inverse = np.unique(games, return_inverse=True)
    if any(g is None or str(g).lower() in ('nan', 'nat', '<na>') for g in unique.tolist()):
        raise ValueError('Missing game identity')
    return unique, inverse


def shared_bootstrap(values, games, *, draws, seed, chunk_rule):
    """Sorted unique games sampled with replacement; identical indices for every named column pair.

    ``values`` maps a name to aligned [n,2] pitch values.  Each replicate is
    sum(values in sampled games) / sum(sampled pitches), computed exactly as the
    legacy helpers do, per name, from the same index block.
    """
    names = list(values)
    arrays = {name: np.asarray(values[name], dtype=np.float64) for name in names}
    games = np.asarray(games)
    for name, array in arrays.items():
        if array.ndim != 2 or array.shape != (len(games), 2) or not np.isfinite(array).all():
            raise ValueError(f'{name}: aligned finite [n,2] values required')
    unique, inverse = _game_index(games)
    counts = np.bincount(inverse).astype(np.float64)
    sums = {name: np.column_stack([np.bincount(inverse, weights=arrays[name][:, j]) for j in range(2)])
            for name in names}
    observed = {name: sums[name].sum(axis=0) / counts.sum() for name in names}
    replicates = {name: np.empty((draws, 2), dtype=np.float64) for name in names}
    rng = np.random.default_rng(seed)
    chunk = chunk_rule(len(unique))
    for start in range(0, draws, chunk):
        size = min(chunk, draws - start)
        indices = rng.integers(0, len(unique), size=(size, len(unique)))
        denominator = counts[indices].sum(axis=1)[:, None]
        for name in names:
            replicates[name][start:start + size] = sums[name][indices].sum(axis=1) / denominator
    return {'games': int(len(unique)), 'pitches': int(len(games)), 'observed': observed, 'replicates': replicates,
            'draws': draws, 'seed': seed}


def additivity_report(points, replicates, *, scope):
    """(B2-B0) - [(B1-B0) + (B2-B1)] at the point estimate and every aligned replicate, both losses."""
    c1, c2, c3 = CONTRAST_IDS
    point = np.abs(np.asarray(points[c2]) - (np.asarray(points[c1]) + np.asarray(points[c3])))
    draws = np.abs(replicates[c2] - (replicates[c1] + replicates[c3]))
    point_error = float(point.max())
    draw_error = float(draws.max()) if draws.size else 0.0
    passed = bool(np.isfinite(point).all() and np.isfinite(draws).all() and point_error <= ADDITIVITY_ATOL
                  and draw_error <= ADDITIVITY_ATOL)
    return {'scope': scope, 'point_max_abs_error': point_error, 'draw_max_abs_error': draw_error,
            'draws_checked': int(len(replicates[c2])), 'atol': ADDITIVITY_ATOL, 'rtol': 0, 'passed': passed}


def require_additivity(report):
    if report['passed'] is not True:
        raise ContrastAdditivityError(f'Contrast additivity failed in {report["scope"]}: point '
                                      f'{report["point_max_abs_error"]:.3g}, draws {report["draw_max_abs_error"]:.3g}')
    return report


def _contrast_losses(losses):
    return {cid: losses[cand] - losses[ctrl] for cid, cand, ctrl in CONTRASTS}


# ---------------------------------------------------------------- N3

def n_slots(y, games, losses, seed_losses):
    """Three whole-DEV paired pitch-weighted NLL/Brier slots with shared draws and Holm3."""
    deltas = _contrast_losses(losses)
    boot = shared_bootstrap(deltas, games, draws=N_SETTINGS['draws'], seed=N_SETTINGS['seed'],
                            chunk_rule=legacy_paired_chunk)
    additivity = require_additivity(additivity_report(boot['observed'], boot['replicates'], scope='N3 whole DEV'))
    slots = {}
    for cid, cand, ctrl in CONTRASTS:
        observed, reps = boot['observed'][cid], boot['replicates'][cid]
        slot = {'contrast': cid, 'candidate': cand, 'control': ctrl, 'n': boot['pitches'], 'games': boot['games'],
                'draws': boot['draws'], 'seed': boot['seed'], 'status': 'measured',
                'estimand': 'pitch-weighted mean paired loss difference; negative favors candidate'}
        if boot['games'] < 2:
            slot['status'] = 'insufficient_games'
        for j, loss in enumerate(('nll', 'brier')):
            centered = reps[:, j] - observed[j]
            slot[loss] = {'delta': float(observed[j]),
                          'ci95': np.quantile(reps[:, j], [.025, .975]).tolist() if boot['games'] >= 2 else None,
                          'p_less': float((1 + (centered <= observed[j]).sum()) / (boot['draws'] + 1))
                          if boot['games'] >= 2 else None}
        slot['seed_nll_deltas'] = [float((seed_losses[cand][s] - seed_losses[ctrl][s])[:, 0].mean()) for s in SEEDS]
        slot['seed_brier_deltas'] = [float((seed_losses[cand][s] - seed_losses[ctrl][s])[:, 1].mean()) for s in SEEDS]
        slots[cid] = slot
    raw_p = [slots[cid]['nll']['p_less'] if slots[cid]['status'] == 'measured' else N_SETTINGS['missing_slot_p']
             for cid in CONTRAST_IDS]
    adjusted = holm_adjust(raw_p)
    for i, cid in enumerate(CONTRAST_IDS):
        slots[cid]['adjusted_p'] = adjusted[i]
        slots[cid]['decision'] = n_decision(slots[cid], adjusted[i])
    return slots, {'method': 'Holm', 'alpha': N_SETTINGS['alpha'], 'slot_order': list(CONTRAST_IDS),
                   'raw_p': raw_p, 'adjusted_p': adjusted, 'family_size': 3}, additivity


def n_decision(slot, adjusted_p):
    if slot.get('status') != 'measured':
        return {'status': 'unmeasured', 'criteria': None}
    nll, brier = slot['nll'], slot['brier']
    negatives = sum(v < 0 for v in slot['seed_nll_deltas'])
    criteria = {'delta_nll_at_most_minus_0.003': nll['delta'] <= N_SETTINGS['delta_nll_max'],
                'nll_ci95_upper_below_zero': nll['ci95'][1] < N_SETTINGS['nll_upper_below'],
                'holm_adjusted_p_at_most_0.05': adjusted_p <= N_SETTINGS['adjusted_p_max'],
                'brier_ci95_upper_at_most_0.001': brier['ci95'][1] <= N_SETTINGS['brier_upper_max'],
                'at_least_4_of_5_seed_nll_negative': negatives >= N_SETTINGS['min_negative_seeds']}
    if all(criteria.values()):
        status = 'development_improvement'
    elif nll['ci95'][0] > N_SETTINGS['worse_nll_lower_above'] or brier['ci95'][0] > N_SETTINGS['worse_brier_lower_above']:
        status = 'worse_or_guardrail_failure'
    else:
        status = 'inconclusive'
    return {'status': status, 'criteria': criteria, 'negative_seeds': int(negatives), 'required_negative_seeds': 4,
            'adjusted_p': adjusted_p}


# ---------------------------------------------------------------- R78

def r_group_masks(metadata, cpanel_mask):
    masks = {**fixed_group_masks(metadata), 'non_cpanel': ~np.asarray(cpanel_mask, dtype=bool)}
    if tuple(masks) != R_GROUPS:
        raise ValueError('R group inventory differs from the fixed thirteen groups')
    return masks


def r_bounds(games, losses, group_masks):
    """3 contrasts x 13 groups x 2 losses = 78 one-sided Bonferroni upper bounds; missing slots retained."""
    if tuple(group_masks) != R_GROUPS:
        raise ValueError('R group inventory differs from the fixed thirteen groups')
    games = np.asarray(games)
    upper_alpha = R_SETTINGS['alpha'] / R_SETTINGS['family_size']
    deltas = _contrast_losses(losses)
    groups, slots, additivity = {}, [], []
    for group in R_GROUPS:
        mask = np.asarray(group_masks[group], dtype=bool)
        gate = group_reporting_status(games, mask)
        record = {'reporting': gate, 'contrasts': {}}
        if gate['status'] == 'reporting_eligible':
            subset = {cid: d[mask] for cid, d in deltas.items()}
            boot = shared_bootstrap(subset, games[mask], draws=R_SETTINGS['draws'], seed=R_SETTINGS['seed'],
                                    chunk_rule=legacy_bound_chunk)
            points = {cid: subset[cid].mean(0) for cid in CONTRAST_IDS}
            additivity.append(require_additivity(additivity_report(points, boot['replicates'], scope='R78 ' + group)))
            for cid in CONTRAST_IDS:
                reps = boot['replicates'][cid]
                upper = np.quantile(reps, 1 - upper_alpha, axis=0).tolist()
                ci95 = np.quantile(reps, [.025, .975], axis=0).T.tolist()
                within = [upper[0] <= R_SETTINGS['nll_margin'], upper[1] <= R_SETTINGS['brier_margin']]
                record['contrasts'][cid] = {'status': 'measured', 'delta': points[cid].tolist(), 'ci95': ci95,
                    'simultaneous_upper': upper, 'upper_alpha': upper_alpha, 'within_margin': within,
                    'passed': bool(all(within)), 'deterioration_lower_bound_above_zero': [ci95[0][0] > 0, ci95[1][0] > 0],
                    'draws': boot['draws'], 'seed': boot['seed']}
        else:
            for cid in CONTRAST_IDS:
                record['contrasts'][cid] = {'status': 'unconfirmed', 'passed': None, 'within_margin': [None, None],
                                            'reason': gate['reasons']}
        groups[group] = record
        for cid in CONTRAST_IDS:
            item = record['contrasts'][cid]
            for j, loss in enumerate(LOSSES):
                slots.append({'slot': f'{cid}|{group}|{loss}', 'contrast': cid, 'group': group, 'loss': loss,
                              'margin': R_SETTINGS['nll_margin'] if j == 0 else R_SETTINGS['brier_margin'],
                              'status': item['status'],
                              'upper': item['simultaneous_upper'][j] if item['status'] == 'measured' else None,
                              'within_margin': item['within_margin'][j]})
    if len(slots) != R_SETTINGS['family_size']:
        raise ValueError('R78 slot inventory is incomplete')
    status = {}
    for cid in CONTRAST_IDS:
        mine = [s for s in slots if s['contrast'] == cid]
        if any(s['within_margin'] is False for s in mine):
            status[cid] = 'failed'
        elif all(s['within_margin'] is True for s in mine) and len(mine) == 26:
            status[cid] = 'passed'
        else:
            status[cid] = 'unconfirmed'
    return {'family_size': R_SETTINGS['family_size'], 'upper_alpha': upper_alpha, 'draws': R_SETTINGS['draws'],
            'seed': R_SETTINGS['seed'], 'groups': groups, 'slots': slots, 'contrast_status': status,
            'margins': {'NLL': R_SETTINGS['nll_margin'], 'Brier': R_SETTINGS['brier_margin']},
            'note': 'An upper bound above its margin means noninferiority unestablished; missing slots stay unconfirmed.'}, additivity


# ---------------------------------------------------------------- candidates and placeholders

def candidate_eligibility(n_results, r_status):
    out = {}
    for candidate, required in CANDIDATE_RULES.items():
        checks = {cid: {'N': n_results[cid]['decision']['status'], 'R': r_status[cid]} for cid in required}
        eligible = all(v['N'] == 'development_improvement' and v['R'] == 'passed' for v in checks.values())
        out[candidate] = {'required_contrasts': list(required), 'checks': checks,
                          'research_improvement_candidate': bool(eligible), 'service_promotion': False}
    return out


def unmeasured_family_result(reason):
    """Aborted family: three N slots unmeasured, p placeholders 1 over denominator 3, all 78 R slots unconfirmed."""
    slots = [{'slot': f'{cid}|{group}|{loss}', 'contrast': cid, 'group': group, 'loss': loss,
              'status': 'unmeasured', 'within_margin': None} for cid in CONTRAST_IDS for group in R_GROUPS for loss in LOSSES]
    return {'family_id': FAMILY_ID, 'status': 'aborted_unmeasured', 'reason': reason,
            'N': {cid: {'status': 'unmeasured', 'decision': {'status': 'unmeasured'}} for cid in CONTRAST_IDS},
            'multiplicity': {'method': 'Holm', 'slot_order': list(CONTRAST_IDS), 'family_size': 3,
                             'placeholder_p_for_complete_vector': [1.0, 1.0, 1.0], 'measured_p': None},
            'R': {'family_size': 78, 'slots': slots, 'contrast_status': {cid: 'unconfirmed' for cid in CONTRAST_IDS}},
            'candidates': {c: {'research_improvement_candidate': False, 'status': 'unmeasured'} for c in CANDIDATE_RULES},
            'contrast_additivity_passed': None, **RESEARCH_STATUS}


# ---------------------------------------------------------------- descriptive summaries

def _trimmed(metrics):
    return {k: metrics[k] for k in ('n', 'log_loss', 'brier_multiclass', 'top_label_ece10')}


def _descriptive(y, games, predictions, losses, population_masks, metadata, pitchers):
    out = {'populations': {}, 'slices': {}, 'intersections': {}, 'per_pitcher': {}, 'seed_absolute': {}}
    for name, mask in population_masks.items():
        record = {'n': int(mask.sum()), 'games': int(len(np.unique(games[mask])))}
        for variant in VARIANTS:
            record[variant] = prediction_metrics(y[mask], predictions[variant]['primary'][mask])
        absolute = shared_bootstrap({v: losses[v][mask] for v in VARIANTS}, games[mask], draws=DESCRIPTIVE_DRAWS,
                                    seed=N_SETTINGS['seed'], chunk_rule=legacy_bound_chunk)
        record['absolute_ci95'] = {v: np.quantile(absolute['replicates'][v], [.025, .975], axis=0).T.tolist()
                                   for v in VARIANTS}
        paired = shared_bootstrap({cid: d[mask] for cid, d in _contrast_losses(losses).items()}, games[mask],
                                  draws=DESCRIPTIVE_DRAWS, seed=N_SETTINGS['seed'], chunk_rule=legacy_paired_chunk)
        record['paired'] = {cid: {'delta': paired['observed'][cid].tolist(),
                                  'ci95': np.quantile(paired['replicates'][cid], [.025, .975], axis=0).T.tolist(),
                                  'status': 'descriptive; no additional hypothesis'} for cid in CONTRAST_IDS}
        record['paired_additivity'] = additivity_report(paired['observed'], paired['replicates'], scope='descriptive ' + name)
        out['populations'][name] = record
    slices = {}
    for column in ('seen_pitcher', 'seen_batter'):
        for seen in (True, False):
            slices[('seen_' if seen else 'unseen_') + column.removeprefix('seen_')] = metadata[column].eq(seen).to_numpy(bool)
    for month in sorted(metadata.month.astype(str).unique()):
        slices['month_' + month] = metadata.month.astype(str).eq(month).to_numpy(bool)
    for name, mask in slices.items():
        out['slices'][name] = {'status': 'descriptive', 'n': int(mask.sum()), 'games': int(len(np.unique(games[mask]))),
                               **{v: _trimmed(prediction_metrics(y[mask], predictions[v]['primary'][mask])) for v in VARIANTS}}
    for label, (left, right) in {'role_x_volume': ('game_role', 'train_volume'),
                                 'hand_x_volume': ('throwing_hand', 'train_volume')}.items():
        combined = metadata[left].astype(str) + '|' + metadata[right].astype(str)
        rows = {}
        for level in sorted(combined.unique()):
            mask = combined.eq(level).to_numpy(bool)
            rows[level] = {'status': 'descriptive only; no R or improvement label',
                           'reporting': group_reporting_status(games, mask),
                           'delta_mean': {cid: (losses[c] - losses[k])[mask].mean(0).tolist() for cid, c, k in CONTRASTS}}
        out['intersections'][label] = rows
    unique, inverse = np.unique(pitchers, return_inverse=True)
    counts = np.bincount(inverse)
    pairs = np.unique(np.column_stack([inverse, games]), axis=0)
    game_counts = np.bincount(pairs[:, 0], minlength=len(unique))
    means = {v: [(np.bincount(inverse, weights=losses[v][:, j]) / counts).tolist() for j in range(2)] for v in VARIANTS}
    out['per_pitcher'] = {'status': 'descriptive only', 'pitchers': unique.tolist(), 'n': counts.tolist(),
                          'games': game_counts.tolist(),
                          'mean_nll': {v: means[v][0] for v in VARIANTS}, 'mean_brier': {v: means[v][1] for v in VARIANTS}}
    return out


def score_family(*, y, games, pitchers, metadata, cpanel_mask, predictions):
    """Full N3/R78/descriptive result from sealed B0/B1/B2 archives; requires all five seeds and both candidates."""
    y = np.asarray(y)
    n = len(y)
    games, pitchers = np.asarray(games), np.asarray(pitchers)
    if games.shape != (n,) or pitchers.shape != (n,) or len(metadata) != n:
        raise ValueError('Labels, games, pitchers and metadata must align')
    if set(predictions) != set(VARIANTS):
        raise ValueError('B0, B1 and B2 archives are all required before scoring')
    losses, seed_losses = {}, {}
    for variant in VARIANTS:
        item = predictions[variant]
        if np.asarray(item['seed_primary']).shape != (len(SEEDS), n, 10):
            raise ValueError(f'{variant}: five aligned seed predictions required')
        losses[variant] = pitch_losses(y, item['primary'])
        seed_losses[variant] = [pitch_losses(y, item['seed_primary'][s]) for s in SEEDS]
    cpanel = np.asarray(cpanel_mask, dtype=bool)
    if cpanel.shape != (n,):
        raise ValueError('Cpanel mask must align')
    n_results, multiplicity, n_additivity = n_slots(y, games, losses, seed_losses)
    r_results, r_additivity = r_bounds(games, losses, r_group_masks(metadata, cpanel))
    candidates = candidate_eligibility(n_results, r_results['contrast_status'])
    whole = np.ones(n, dtype=bool)
    descriptive = _descriptive(y, games, predictions, losses, {'whole': whole, 'cpanel': cpanel, 'non_cpanel': ~cpanel},
                               metadata, pitchers)
    descriptive['seed_absolute'] = {v: [{'nll': float(seed_losses[v][s][:, 0].mean()),
                                         'brier': float(seed_losses[v][s][:, 1].mean())} for s in SEEDS] for v in VARIANTS}
    descriptive_additivity = [p['paired_additivity'] for p in descriptive['populations'].values()]
    class_support = [int((y == c).sum()) for c in range(10)]
    all_additivity = [n_additivity, *r_additivity, *descriptive_additivity]
    passed = all(a['passed'] is True for a in all_additivity)
    if not passed:
        raise ContrastAdditivityError('Descriptive paired draws violated contrast additivity')
    return {'family_id': FAMILY_ID, 'status': 'scored', 'scope': 'Previously exposed eligible 2025 whole-MLB DEV',
            'n': int(n), 'games': int(len(np.unique(games))), 'contrasts': [list(c) for c in CONTRASTS],
            'N': n_results, 'multiplicity': multiplicity, 'R': r_results, 'candidates': candidates,
            'contrast_additivity': {'N': n_additivity, 'R': r_additivity, 'descriptive': descriptive_additivity},
            'contrast_additivity_passed': passed,
            'class_reporting': {'minimum_events': 30, 'support': class_support,
                                'eligible_for_class_claims': [s >= 30 for s in class_support]},
            'descriptive': descriptive, **RESEARCH_STATUS,
            'limits': ['DEV was exposed in P11 and adaptive family choice followed it; no fresh alpha or confirmation.',
                       'Intervals condition on frozen base models, fitted calibration parameters and outputs.',
                       'B1-B0 changes calibration sample size and composition jointly; not a causal row-count effect.',
                       'Cpanel/non-Cpanel, month, seen/unseen and intersections are descriptive and add no hypotheses.']}
