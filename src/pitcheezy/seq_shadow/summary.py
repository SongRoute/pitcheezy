"""Summary of one sequence-model shadow run: gate V, the two metric populations, paired bootstrap against the
Step 5 (frozen 2024) ledger, context statistics and latency. Everything here was fixed before any real result.

Definitions (constants from ``configs/seq_shadow.json`` ``metrics`` / ``bootstrap``):
  top1_accuracy        argmax of the seven leaves (first maximum on ties) equals the realized leaf
  nll_mean             mean of -log(min(1, max(p[realized], nll_clip))); the clip is inside the logarithm only
  brier_mean           mean over pitches of the sum over the seven leaves of squared error
  ece_top_label_10bin  sum over bins of |sum(conf - hit)| / n, bin = min(floor(bins*conf), bins-1)
                       (calibrate_logits.py ``top`` / ``ece``, lines 101-104)
  leaf ECE             one-vs-rest for one leaf: sum over bins of |sum(p_leaf - 1[y = leaf])| / n
  reliability          Step 5 ``shadow_runner._calibration``: equal-width bins over the pooled one-vs-rest pairs
  paired difference    metric(first arm) - metric(second arm) on the same pitches
  bootstrap            paired, percentile interval; unit 'pitch' resamples pitches, unit 'plate_appearance' resamples
                       at_bat_numbers; each draw is a ratio of sums over the drawn units (pitch-weighted), the same
                       draw for both arms; ECE is recomputed per draw from the drawn units' bin sums. One game, so
                       there is no game-level bootstrap. No test, no label, no winner.

Populations (physical pitches of the last identity-verified body, labelled by THIS run's reconciliation):
  A "paired"          label had_attempt -> its attempt has a scored seq_forecasts row -> realized pitch type is one of
                      the 17 actions -> realized leaf placed -> the Step 5 ledger's cell of the realized type is
                      scored with seven leaves. Exclusions are counted by reason in that order.
  B "seq all linked"  the same without the Step 5 cell condition.
Gate V: every attempt scored by this run and present as scored in the Step 5 ledger must carry the same
``vectors_sha256`` and the same ``written_at_replay_index``; any mismatch stops the run before metrics.
"""
from __future__ import annotations

import json
import sqlite3

import numpy as np


# ----------------------------------------------------------------------------------------------- per-pitch statistics
def pitch_statistics(q, y, *, nll_clip, bins):
    """Per-pitch pieces of the metrics for probabilities ``q`` (n, 7) and realized leaf indices ``y`` (n,)."""
    q, y = np.asarray(q, dtype=np.float64).reshape(-1, 7), np.asarray(y, dtype=np.int64)
    n, i = len(y), np.arange(len(y))
    one_hot = np.zeros_like(q)
    one_hot[i, y] = 1
    conf = q.max(1) if n else np.zeros(0)
    hit = (q.argmax(1) == y).astype(np.float64) if n else np.zeros(0)
    gap = np.zeros((n, bins))
    gap[i, np.minimum((conf*bins).astype(int), bins-1)] = conf-hit
    return {'nll': -np.log(np.minimum(1., np.maximum(q[i, y], nll_clip))), 'brier': ((q-one_hot)**2).sum(1),
            'hit': hit, 'gap': gap}


def _point(stats):
    n = len(stats['nll'])
    return {'nll_mean': float(stats['nll'].mean()), 'brier_mean': float(stats['brier'].mean()),
            'top1_accuracy': float(stats['hit'].mean()),
            'ece_top_label': float(np.abs(stats['gap'].sum(0)).sum()/n)}


def leaf_ece(q, y, leaf, bins):
    p, hit = q[:, leaf], (y == leaf).astype(np.float64)
    b = np.minimum((p*bins).astype(int), bins-1)
    return float(np.abs(np.bincount(b, p-hit, bins)).sum()/len(y))


def metric_block(q, y, *, leaves, nll_clip, bins, leaf_ece_leaves, calibration):
    """All registered metrics of one model on one population. ``calibration``: shadow_runner._calibration."""
    q, y = np.asarray(q, dtype=np.float64).reshape(-1, 7), np.asarray(y, dtype=np.int64)
    n = len(y)
    if n == 0:
        return {'n': 0, 'top1_accuracy': None, 'nll_mean': None, 'brier_mean': None, 'ece_top_label_10bin': None,
                'leaf_ece_10bin': {name: None for name in leaf_ece_leaves}, 'calibration_one_vs_rest_pooled': calibration([], bins)}
    point = _point(pitch_statistics(q, y, nll_clip=nll_clip, bins=bins))
    pairs = [(float(q[r, k]), bool(y[r] == k)) for r in range(n) for k in range(7)]
    return {'n': n, 'top1_accuracy': point['top1_accuracy'], 'nll_mean': point['nll_mean'],
            'brier_mean': point['brier_mean'], 'ece_top_label_10bin': point['ece_top_label'],
            'leaf_ece_10bin': {name: leaf_ece(q, y, leaves.index(name), bins) for name in leaf_ece_leaves},
            'calibration_one_vs_rest_pooled': calibration(pairs, bins)}


# ----------------------------------------------------------------------------------------------- paired bootstrap
METRICS = ('nll_mean', 'brier_mean', 'top1_accuracy', 'ece_top_label')


def _cluster_sums(stats, inverse, groups):
    gap = np.zeros((groups, stats['gap'].shape[1]))
    np.add.at(gap, inverse, stats['gap'])
    return {'nll': np.bincount(inverse, stats['nll'], groups), 'brier': np.bincount(inverse, stats['brier'], groups),
            'hit': np.bincount(inverse, stats['hit'], groups), 'gap': gap}


def paired_bootstrap(first, second, clusters, *, draws, seed, chunk, quantiles):
    """Percentile interval of metric(first) - metric(second) under a paired bootstrap over ``clusters`` (one id per
    pitch; each pitch its own id = pitch-level). ``first`` / ``second``: ``pitch_statistics`` of the two arms on the
    SAME pitches in the same order."""
    clusters = np.asarray(clusters)
    n = len(clusters)
    if n == 0:
        return {'units': 0, 'pitches': 0, 'draws': draws, 'seed': seed, 'ci': {m: None for m in METRICS}}
    _, inverse = np.unique(clusters, return_inverse=True)
    groups = int(inverse.max())+1
    size = np.bincount(inverse, minlength=groups).astype(np.float64)
    a, b = _cluster_sums(first, inverse, groups), _cluster_sums(second, inverse, groups)
    rng = np.random.default_rng(seed)
    out = {m: np.empty(draws) for m in METRICS}
    for start in range(0, draws, chunk):
        pick = rng.integers(0, groups, (min(chunk, draws-start), groups))
        total = size[pick].sum(1)
        part = slice(start, start+len(pick))
        out['nll_mean'][part] = (a['nll']-b['nll'])[pick].sum(1)/total
        out['brier_mean'][part] = (a['brier']-b['brier'])[pick].sum(1)/total
        out['top1_accuracy'][part] = (a['hit']-b['hit'])[pick].sum(1)/total
        out['ece_top_label'][part] = (np.abs(a['gap'][pick].sum(1)).sum(1)-np.abs(b['gap'][pick].sum(1)).sum(1))/total
    low, high = quantiles
    return {'units': groups, 'pitches': n, 'draws': draws, 'seed': seed,
            'ci': {m: [float(np.quantile(v, low)), float(np.quantile(v, high))] for m, v in out.items()}}


def paired_difference(q_first, q_second, y, at_bats, *, metrics_config, bootstrap_config):
    """Point estimate and both bootstrap intervals of (first - second)."""
    kw = {'nll_clip': metrics_config['nll_clip'], 'bins': metrics_config['bins']}
    first, second = pitch_statistics(q_first, y, **kw), pitch_statistics(q_second, y, **kw)
    n = len(first['nll'])
    if n == 0:
        return {'n': 0, 'difference': {m: None for m in METRICS}, 'bootstrap': None}
    pa, pb = _point(first), _point(second)
    boot = {k: bootstrap_config[k] for k in ('draws', 'seed', 'chunk')} | {'quantiles': bootstrap_config['ci_quantiles']}
    units = {'pitch': np.arange(n), 'plate_appearance': np.asarray(at_bats)}
    return {'n': n, 'difference': {m: pa[m]-pb[m] for m in METRICS},
            'bootstrap': {unit: paired_bootstrap(first, second, units[unit], **boot) for unit in bootstrap_config['units']},
            'ci_method': 'percentile, numpy quantile (linear interpolation) at '+json.dumps(bootstrap_config['ci_quantiles'])}


# ----------------------------------------------------------------------------------------------- ledgers
def read_step5_forecasts(path):
    """{attempt_key: row} of the Step 5 shadow ledger, opened read-only and immutable (no lock, no journal)."""
    names = ('attempt_key', 'at_bat_number', 'candidate_pitch_number', 'written_at_replay_index', 'decision', 'outcome',
             'reason', 'action_probabilities', 'vectors_sha256', 'context_sha256')
    db = sqlite3.connect(f'file:{path}?mode=ro&immutable=1', uri=True)
    try:
        return {r[0]: dict(zip(names, r)) for r in db.execute(f"SELECT {','.join(names)} FROM shadow_forecasts")}
    finally:
        db.close()


def read_seq_forecasts(db):
    names = ('attempt_key', 'at_bat_number', 'candidate_pitch_number', 'written_at_replay_index', 'decision', 'reason',
             'vectors_sha256', 'context_sha256', 'tokens_json', 'c0_ensemble_json', 'sel_ensemble_json', 'compute_ms',
             'tokens_ms', 'forward_ms', 'calibration_ms', 'features_ms', 'replay_file', 'row_sha256')
    return {r[0]: dict(zip(names, r)) for r in db.execute(f"SELECT {','.join(names)} FROM seq_forecasts")}


def gate_v(seq_rows, step5_rows):
    """Identity of the pre-pitch input between this run and the Step 5 run, over attempts scored in both."""
    mine = {k: r for k, r in seq_rows.items() if r['decision'] == 'scored'}
    theirs = {k: r for k, r in step5_rows.items() if r['decision'] == 'scored'}
    common = sorted(set(mine) & set(theirs))
    vectors = [k for k in common if mine[k]['vectors_sha256'] != theirs[k]['vectors_sha256']]
    index = [k for k in common if mine[k]['written_at_replay_index'] != theirs[k]['written_at_replay_index']]
    context = [k for k in common if mine[k]['context_sha256'] != theirs[k]['context_sha256']]
    return {'keys_this_run': len(mine), 'keys_step5': len(theirs), 'keys_common': len(common),
            'keys_only_this_run': len(set(mine)-set(theirs)), 'keys_only_step5': len(set(theirs)-set(mine)),
            'vectors_sha256_mismatch': len(vectors), 'decision_index_mismatch': len(index),
            'context_sha256_mismatch_report_only': len(context),
            'first_mismatching_keys': (vectors+index)[:10],
            'passed': not vectors and not index,
            'rule': 'vectors_sha256_mismatch and decision_index_mismatch must both be 0 over keys_common'}


# ----------------------------------------------------------------------------------------------- populations
def populations(physical, realized, seq_rows, step5_rows, actions, leaves):
    """(rows of A, rows of B, counts). ``physical``: [(at_bat_number, pitch_number, attempt_key, label)] without
    phantoms; ``realized``: {(at_bat_number, pitch_number): latest observation}. A row: at_bat_number, pitch_number,
    y (leaf index), sel / c0 (seven leaves of the realized type) and, for A, step5."""
    a_rows, b_rows = [], []
    counts = {'physical_pitches': len(physical), 'physical_had_attempt': 0,
              'A': {'excluded': {}, 'n': 0}, 'B': {'excluded': {}, 'n': 0}}

    def skip(names, why):
        for name in names:
            counts[name]['excluded'][why] = counts[name]['excluded'].get(why, 0)+1
    for ab, number, key, label in physical:
        if label != 'had_attempt':
            continue
        counts['physical_had_attempt'] += 1
        seq = seq_rows.get(key)
        if seq is None or seq['decision'] != 'scored':
            skip('AB', 'seq_forecast_not_scored:'+str(None if seq is None else seq['reason']))
            continue
        seen = realized.get((ab, number)) or {}
        kind = seen.get('realized_pitch_type')
        if kind is None:
            skip('AB', 'realized_pitch_type_absent')
            continue
        if kind not in actions:
            skip('AB', 'realized_pitch_type_outside_17_actions')
            continue
        if seen.get('realized_leaf') is None:
            skip('AB', 'realized_leaf_unplaced:'+str(seen.get('realized_leaf_reason')))
            continue
        at, y = actions.index(kind), leaves.index(seen['realized_leaf'])
        row = {'at_bat_number': ab, 'pitch_number': number, 'attempt_key': key, 'y': y,
               'sel': json.loads(seq['sel_ensemble_json'])[at], 'c0': json.loads(seq['c0_ensemble_json'])[at]}
        b_rows.append(row)
        step5 = step5_rows.get(key)
        if step5 is None:
            skip('A', 'step5_attempt_absent')
            continue
        cell = json.loads(step5['action_probabilities'])[at]
        if cell.get('action') != kind:
            raise ValueError('Step 5 ledger cell order differs from the 17 actions')
        if cell.get('status') != 'scored' or not isinstance(cell.get('seven_leaf'), list) or len(cell['seven_leaf']) != 7:
            skip('A', 'step5_realized_type_cell_not_scored:'+str(cell.get('reason')))
            continue
        a_rows.append(row | {'step5': cell['seven_leaf']})
    counts['A']['n'], counts['B']['n'] = len(a_rows), len(b_rows)
    return a_rows, b_rows, counts


def metrics(a_rows, b_rows, counts, *, config, leaves, calibration):
    """The registered metric block. ``calibration``: shadow_runner._calibration (reused, not copied)."""
    m, boot = config['metrics'], config['bootstrap']
    if list(leaves) != list(m['leaf_order']):
        raise ValueError('pipeline leaf order differs from the registered leaf order')
    kw = {'leaves': list(leaves), 'nll_clip': m['nll_clip'], 'bins': m['bins'], 'leaf_ece_leaves': m['leaf_ece_leaves'],
          'calibration': calibration}
    column = {'N1_SEL': 'sel', 'N1_C0': 'c0', 'STEP5_2024': 'step5'}
    out = {'leaf_order': list(leaves), 'definitions': {'nll_clip_inside_log_only': m['nll_clip'], 'bins': m['bins'],
                                                        'ece': 'bin = min(floor(bins*conf), bins-1), pitch-weighted'},
           'physical_pitches': counts['physical_pitches'], 'physical_had_attempt': counts['physical_had_attempt']}
    for name, rows, models, label in (('A', a_rows, m['models_population_A'], 'paired: also scored by Step 5 at the realized type'),
                                      ('B', b_rows, m['models_population_B'], 'seq all linked')):
        y = np.array([r['y'] for r in rows], dtype=np.int64)
        block = {'label': label, 'n': len(rows), 'excluded_in_order': counts[name]['excluded'],
                 'plate_appearances': len({r['at_bat_number'] for r in rows}), 'models': {}}
        for model in models:
            block['models'][model] = metric_block(np.array([r[column[model]] for r in rows], dtype=np.float64), y, **kw)
        out['population_'+name] = block
    y = np.array([r['y'] for r in a_rows], dtype=np.int64)
    at_bats = np.array([r['at_bat_number'] for r in a_rows], dtype=np.int64)
    out['paired_differences_population_A'] = {
        f'{first}-{second}': paired_difference(np.array([r[column[first]] for r in a_rows], dtype=np.float64),
                                               np.array([r[column[second]] for r in a_rows], dtype=np.float64),
                                               y, at_bats, metrics_config=m, bootstrap_config=boot)
        for first, second in m['paired_differences']}
    out['note'] = ('descriptive only: one already exposed game; intervals condition on fixed predictions; no test, '
                   'no label, no winner. STEP5_2024 values are read from the Step 5 ledger, never from this run.')
    return out


# ----------------------------------------------------------------------------------------------- context / latency
def context_statistics(seq_rows):
    scored = [json.loads(r['tokens_json'])['counts'] for r in seq_rows.values() if r['decision'] == 'scored']
    n = sorted(c['n_context'] for c in scored)
    out = {'rows': len(scored), 'rows_with_zero_context': sum(v == 0 for v in n),
           'rows_truncated': sum(c['truncated'] > 0 for c in scored),
           'skipped_total_by_reason': {k: sum(c[k] for c in scored) for k in (
               'skipped_unmapped_outcome', 'skipped_count_range', 'skipped_no_type')},
           'rows_with_a_skip_by_reason': {k: sum(c[k] > 0 for c in scored) for k in (
               'skipped_unmapped_outcome', 'skipped_count_range', 'skipped_no_type')}}
    if n:
        out['n_context'] = {'n': len(n), 'min': n[0], 'p50': n[max(0, -(-len(n)//2)-1)], 'max': n[-1],
                            'mean': sum(n)/len(n), 'method': 'nearest rank'}
    else:
        out['n_context'] = {'n': 0}
    return out


def seq_denominators(seq_rows):
    reasons = {}
    for r in seq_rows.values():
        if r['decision'] != 'scored':
            reasons[str(r['reason'])] = reasons.get(str(r['reason']), 0)+1
    return {'seq_forecast_rows': len(seq_rows), 'seq_scored': sum(r['decision'] == 'scored' for r in seq_rows.values()),
            'seq_abstain_by_reason': dict(sorted(reasons.items()))}


def latency(seq_rows, quantiles):
    """Per scored decision, from the ledger's timing columns. ``quantiles``: shadow_runner._quantiles."""
    scored = [r for r in seq_rows.values() if r['decision'] == 'scored']
    return {name: quantiles([r[name] for r in scored if r[name] is not None])
            for name in ('compute_ms', 'tokens_ms', 'forward_ms', 'calibration_ms', 'features_ms')} | {
        'note': 'compute_ms = tokens_ms + forward_ms (three seeds) + calibration_ms, ledger commit excluded; '
                'features_ms = second build of the 17x199 matrix and the scale-002 encoding, reported apart'}


def physical_and_realized(db):
    physical = db.execute("SELECT at_bat_number, pitch_number, attempt_key, label FROM physical_reconciliation "
                          "WHERE label != 'phantom_unmatched_attempt' ORDER BY at_bat_number, pitch_number").fetchall()
    names = ('at_bat_number', 'pitch_number', 'observed_at_replay_index', 'realized_pitch_type', 'realized_leaf',
             'realized_leaf_reason')
    realized = {(r[0], r[1]): dict(zip(names, r)) for r in db.execute(
        f"SELECT {','.join(names)} FROM shadow_realized WHERE id IN (SELECT MAX(id) FROM shadow_realized "
        'GROUP BY at_bat_number, pitch_number)')}
    first_seen = {(a, n): i for a, n, i in db.execute(
        'SELECT at_bat_number, pitch_number, MIN(observed_at_replay_index) FROM shadow_realized GROUP BY 1, 2')}
    return physical, realized, first_seen


def order_violations(seq_rows, first_seen):
    """First-decision seq_forecasts rows (``replay_file`` set: written from the body of their own decision) that
    were not written strictly before the first body that showed their pitch (must be 0). Rows of attempts that never
    had a first decision are abstentions written when an observed event closed them, possibly the very body that
    shows the pitch; they carry no forecast and are not counted here."""
    return sum(not (first_seen.get((r['at_bat_number'], r['candidate_pitch_number']), 10**12) > r['written_at_replay_index'])
               for r in seq_rows.values() if r['replay_file'] is not None)
