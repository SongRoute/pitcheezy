"""W3 of EXP-SVC-P9-001: metric (a), next-pitch-type prediction on E_PS, exactly as registered (plan §5(a)).

Real-data runs refuse unless the registration is ACTIVE with execute true and a reviewed grant is pinned
(``require_active``). This commit only exercises the pure functions on synthetic rows.

Fixed before results (configs/EXP-SVC-P9-001.json metrics.a_prediction):
  rows        E_PS isPitch rows whose actual pitch type maps to the 9 groups (interfaces/pitch_types.py);
              UNKNOWN/PO/IN/UN and unmapped codes are dropped as TARGETS only (context unchanged);
              the three predictions use the same rows. 849799 is refused outright.
  candidate   B_PSR9N6 on W1 inputs built from replay snapshots (scripts/demo_precompute.py:replay_snapshots).
  baseline 1  service CategoricalBC.probabilities(state) on the service's own request states, folded to 9 groups:
              mass of codes outside the map is dropped and renormalised (rows counted); a refusal uses the BC league
              frequency (rows counted).
  baseline 2  frozen frequency, september FIT table (gpu14h_prepitch/run.py:frequency, alpha 4,
              global -> pitcher -> pitcher x count).
  floor       primary: q' = (1-0.001) q + 0.001/9 for all three; secondary: raw q with NLL clip 1e-12
              (gpu14h_prepitch/run.py:comparison9) and the count of zero-probability true classes.
  metrics     NLL, Brier (sum over 9), ECE10 top-label (10 equal bins on max prob, sum_bin |sum(conf-hit)| / N;
              experiments/ws_seq/evaluate_seq.py:122-127), top-1.
  bootstrap   game-level paired, 2000 replicates, seed 20261110, percentile 2.5/97.5.
  verdict     dNLL upper < 0 -> 낮다, lower > 0 -> 높다, else 차이 없음; gate = 낮다 AND dECE10 upper <= +0.01.

  omitted     (re-review N1) an E_PS row where W1 omits the candidate is NOT dropped: the candidate is scored as
              uniform 1/9 there (N6 needs W1 inputs, so it cannot fill); counts by reason are reported; baselines
              keep their own values. Descriptive: the same contrast on candidate-present rows only.
  bc zeros    (re-review N2) descriptive dNLL on rows where the unfloored service BC gives the true class > 0, with
              the counts of nonzero and zero rows (a zero row costs -log(0.001/9) ~ 9.105 nat after the floor).
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src'))
from pitcheezy.interfaces.pitch_types import PITCH_NAMES, STATCAST_TO_PITCH_ID  # noqa: E402

CONFIG = REPO / 'configs/EXP-SVC-P9-001.json'
FACTS = REPO / 'docs/pm/research/svc-p9/registration-facts.json'
DEMO_GAME = 849799
EPS = 1e-3
NLL_CLIP = 1e-12
BINS = 10
REPLICATES, SEED = 2000, 20261110
ROW_RULE = 'all_target_rows_uniform_fill'
UNIFORM = [1.0 / 9] * 9
K = len(PITCH_NAMES)


# ---------------------------------------------------------------- guards
def require_active(config=CONFIG, grant=None):
    c = json.loads(pathlib.Path(config).read_text())
    if c.get('status') != 'ACTIVE' or c.get('execute') is not True:
        raise SystemExit('EXP-SVC-P9-001 is not ACTIVE/execute: real-data evaluation refused')
    if grant is None or hashlib.sha256(pathlib.Path(grant).read_bytes()).hexdigest() != c.get('grant_sha256'):
        raise SystemExit('reviewed grant missing or not pinned')
    return c


def e_ps_games(facts=FACTS):
    f = json.loads(pathlib.Path(facts).read_text())
    games = list(f['E_PS']['game_pks'])
    if DEMO_GAME in games or len(games) != 31:
        raise SystemExit('E_PS list does not match the registration (31 games, 849799 excluded)')
    return games


def refuse_demo_game(game_pks):
    if DEMO_GAME in set(int(g) for g in game_pks):
        raise ValueError('849799 is demo-only: metric (a) is never computed on it')


# ---------------------------------------------------------------- labels and baselines
def label(code):
    """9-group index of an actual pitch type, or None (excluded as a target)."""
    if code is None:
        return None
    return STATCAST_TO_PITCH_ID.get(str(code))


def fold_bc(actions, p):
    """Raw-code BC probabilities -> 9 groups. Returns (q9, dropped_mass)."""
    q = np.zeros(K)
    dropped = 0.0
    for a, v in zip(actions, np.asarray(p, np.float64)):
        g = STATCAST_TO_PITCH_ID.get(str(a))
        if g is None:
            dropped += float(v)
        else:
            q[g] += float(v)
    if q.sum() <= 0:
        raise ValueError('BC has no mass on the 9 groups')
    return q / q.sum(), dropped


def frozen_frequency(table, pitcher, balls, strikes):
    """gpu14h_prepitch/run.py:frequency for one row (alpha 4; global -> pitcher -> pitcher:balls:strikes)."""
    if table['alpha'] != 4.0:
        raise ValueError('frozen frequency alpha')
    count = np.asarray(table['global'], np.float64)
    globalp = (count + 4 / 9) / (count.sum() + 4)
    pc = table['pitcher'].get(str(pitcher)); cc = table['context'].get(f'{pitcher}:{balls}:{strikes}')
    pp = globalp if pc is None else (np.asarray(pc) + 4 * globalp) / (sum(pc) + 4)
    return pp if cc is None else (np.asarray(cc) + 4 * pp) / (sum(cc) + 4)


def floor(q, eps=EPS):
    q = np.asarray(q, np.float64)
    return (1 - eps) * q + eps / K


# ---------------------------------------------------------------- metrics (per row parts, so the bootstrap reweights games)
def row_parts(q, y, clip=None):
    q = np.asarray(q, np.float64); y = np.asarray(y, np.int64)
    truth = q[np.arange(len(y)), y]
    with np.errstate(divide='ignore'):
        nll = -np.log(np.clip(truth, clip, 1.0)) if clip else -np.log(truth)
    onehot = np.eye(K)[y]
    brier = ((q - onehot) ** 2).sum(1)
    conf = q.max(1); hit = (q.argmax(1) == y).astype(np.float64)
    b = np.minimum((conf * BINS).astype(int), BINS - 1)
    return {'nll': nll, 'brier': brier, 'gap': conf - hit, 'bin': b, 'top1': hit, 'zero_truth': (truth == 0)}


def pooled(parts, weights=None):
    w = np.ones(len(parts['nll'])) if weights is None else np.asarray(weights, np.float64)
    total = w.sum()
    cells = np.zeros(BINS); np.add.at(cells, parts['bin'], w * parts['gap'])
    return {'nll': float(w @ parts['nll'] / total), 'brier': float(w @ parts['brier'] / total),
            'ece10': float(np.abs(cells).sum() / total), 'top1': float(w @ parts['top1'] / total)}


def game_weights(games, replicates=REPLICATES, seed=SEED):
    """Row weights per replicate: each game drawn with replacement; a row's weight = its game's draw count."""
    ids, inverse = np.unique(np.asarray(games), return_inverse=True)
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        counts = np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids))
        yield counts[inverse].astype(np.float64)


def paired_contrast(q_cand, q_base, y, games, *, clip=None, replicates=REPLICATES, seed=SEED):
    a, b = row_parts(q_cand, y, clip), row_parts(q_base, y, clip)
    point = {k: pooled(a)[k] - pooled(b)[k] for k in ('nll', 'brier', 'ece10', 'top1')}
    reps = {k: [] for k in point}
    for w in game_weights(games, replicates, seed):
        pa, pb = pooled(a, w), pooled(b, w)
        for k in reps:
            reps[k].append(pa[k] - pb[k])
    ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in reps.items()}
    return {'point': point, 'ci95': ci, 'candidate': pooled(a), 'baseline': pooled(b),
            'zero_truth': {'candidate': int(a['zero_truth'].sum()), 'baseline': int(b['zero_truth'].sum())}}


def verdict(contrast):
    lo, hi = contrast['ci95']['nll']
    text = '낮다' if hi < 0 else '높다' if lo > 0 else '차이 없음'
    gate = text == '낮다' and contrast['ci95']['ece10'][1] <= 0.01
    return {'nll': text, 'gate_pass': bool(gate), 'rule': 'dNLL upper<0 낮다 / lower>0 높다 / else 차이 없음; gate 낮다 AND dECE10 upper <= +0.01'}


def evaluate(rows, *, grant, config=CONFIG, facts=FACTS, replicates=REPLICATES, seed=SEED):
    """THE entry point for metric (a) (code review P1-1): the registration must be ACTIVE with execute true and the
    reviewed grant pinned, and the scored games must be exactly the registered E_PS list (no wildcard game, no
    missing game, no 849799). Only then are the rows scored."""
    c = require_active(config, grant)
    expected = set(e_ps_games(facts))
    got = {int(r['game_pk']) for r in rows}
    refuse_demo_game(got)
    if got != expected:
        raise SystemExit(f'scored games differ from E_PS: extra {sorted(got - expected)}, missing {sorted(expected - got)}')
    floor_value = ((c.get('metrics') or {}).get('a_prediction') or {}).get('validity', {}).get('min_candidate_computed_fraction')
    return _score(rows, replicates=replicates, seed=seed, min_computed=floor_value)


def validity(computed, rows, min_computed):
    """Pre-registered validity (code review P1-2 나): below the floor the verdict is '판정 없음'."""
    fraction = computed / rows
    if isinstance(min_computed, bool) or not isinstance(min_computed, (int, float)):
        return {'computed_rows': computed, 'rows': rows, 'fraction': fraction, 'floor': min_computed, 'valid': None}
    return {'computed_rows': computed, 'rows': rows, 'fraction': fraction, 'floor': float(min_computed),
            'valid': bool(fraction >= min_computed)}


def _score(rows, *, replicates=REPLICATES, seed=SEED, min_computed=None):
    """Scoring only (no registration guard; tests call this on synthetic rows). rows: dicts with game_pk, actual
    (raw code), cand (9 probs or None = W1 omitted, with omit_reason), bc (9 probs), freq (9 probs), known (bool),
    optional bc_masked (9 probs or None: the support-masked reference the service policy consumes; descriptive)."""
    refuse_demo_game(r['game_pk'] for r in rows)
    excluded, omitted, keep = {}, {}, []
    for r in rows:
        y = label(r['actual'])
        if y is None:
            excluded[str(r['actual'])] = excluded.get(str(r['actual']), 0) + 1
            continue
        if r['cand'] is None:
            reason = r.get('omit_reason') or 'unknown'
            omitted[reason] = omitted.get(reason, 0) + 1
            r = dict(r, cand=UNIFORM, filled=True)
        keep.append((r, y))
    if not keep:
        raise ValueError('no evaluable rows')
    y = np.array([k[1] for k in keep]); games = np.array([k[0]['game_pk'] for k in keep])
    Q = {name: np.array([k[0][name] for k in keep], np.float64) for name in ('cand', 'bc', 'freq')}
    for name, q in Q.items():
        if q.shape != (len(keep), K) or not np.isfinite(q).all() or not np.allclose(q.sum(1), 1, atol=1e-9) or (q < 0).any():
            raise ValueError(f'{name} probabilities')
    F = {n: floor(q) for n, q in Q.items()}
    primary = paired_contrast(F['cand'], F['bc'], y, games, replicates=replicates, seed=seed)
    report = {
        'row_rule': ROW_RULE, 'rows': len(keep), 'games': int(len(set(games.tolist()))),
        'excluded_targets': excluded, 'candidate_omitted': omitted,
        'primary_floor_eps': EPS,
        'candidate_minus_service_bc': primary, 'verdict': verdict(primary),
        'secondary': {
            'candidate_minus_frozen_frequency': paired_contrast(F['cand'], F['freq'], y, games, replicates=replicates, seed=seed),
            'unfloored_clip_1e-12_candidate_minus_service_bc': paired_contrast(Q['cand'], Q['bc'], y, games, clip=NLL_CLIP, replicates=replicates, seed=seed)},
        'descriptive': {}}
    present = np.array([not k[0].get('filled') for k in keep])
    report['candidate_filled_uniform_rows'] = int((~present).sum())
    report['validity'] = validity(int(present.sum()), len(keep), min_computed)
    if report['validity']['valid'] is False:
        report['verdict'] = {'nll': '판정 없음', 'gate_pass': False, 'computed_nll_text': report['verdict']['nll'],
                             'rule': 'candidate-computed fraction below the registered floor -> 판정 없음 (numbers descriptive only)'}
    elif report['validity']['valid'] is None:
        report['verdict'] = dict(report['verdict'], validity_floor='unset (synthetic or TBD_song)')
    if present.any() and (~present).any():
        report['descriptive']['candidate_present_rows_only'] = dict(rows=int(present.sum()), **paired_contrast(
            F['cand'][present], F['bc'][present], y[present], games[present], replicates=replicates, seed=seed))
    nonzero = Q['bc'][np.arange(len(y)), y] > 0
    report['descriptive']['bc_nonzero_rows'] = {'rows': int(nonzero.sum()), 'bc_zero_rows': int((~nonzero).sum())}
    if nonzero.any():
        report['descriptive']['bc_nonzero_rows'].update(paired_contrast(
            F['cand'][nonzero], F['bc'][nonzero], y[nonzero], games[nonzero], replicates=replicates, seed=seed))
    known = np.array([bool(k[0].get('known')) for k in keep])
    for name, m in (('known_pitcher', known), ('unseen_pitcher', ~known)):
        if m.any():
            report['descriptive'][name] = {'rows': int(m.sum()), 'candidate': pooled(row_parts(F['cand'][m], y[m])),
                                           'service_bc': pooled(row_parts(F['bc'][m], y[m]))}
    per_game = {}
    for g in sorted(set(games.tolist())):
        m = games == g
        per_game[str(g)] = pooled(row_parts(F['cand'][m], y[m]))['nll'] - pooled(row_parts(F['bc'][m], y[m]))['nll']
    report['descriptive']['per_game_dnll'] = per_game
    # PM decision 11/16: the support-masked reference the service policy actually consumes, descriptive only (no gate)
    masked = [i for i, k in enumerate(keep) if k[0].get('bc_masked') is not None]
    report['descriptive']['service_masked_reference'] = {'rows': len(masked), 'gate': False}
    if masked:
        idx = np.array(masked)
        M = floor(np.array([keep[i][0]['bc_masked'] for i in masked], np.float64))
        report['descriptive']['service_masked_reference'].update(paired_contrast(
            F['cand'][idx], M, y[idx], games[idx], replicates=replicates, seed=seed))
    return report


# ---------------------------------------------------------------- EXP-SVC-P9-002 (D198): retrained R vs current C on E_PS
CONFIG_P9_002 = REPO / 'configs/EXP-SVC-P9-002.json'
ADOPTION_RULE = 'use R only if validity is valid AND the game-level paired bootstrap 95% upper bound of dNLL(R - C) < 0; else keep C (B_PSR9N6)'


def evaluate_retrain(rows, *, grant, config=CONFIG_P9_002, facts=FACTS, replicates=REPLICATES, seed=SEED):
    """THE entry point for EXP-SVC-P9-002: ACTIVE + execute + pinned grant, and exactly the registered E_PS games."""
    c = require_active(config, grant)
    if c.get('id') != 'EXP-SVC-P9-002':
        raise SystemExit('not the EXP-SVC-P9-002 registration')
    expected = set(e_ps_games(facts))
    got = {int(r['game_pk']) for r in rows}
    refuse_demo_game(got)
    if got != expected:
        raise SystemExit(f'scored games differ from E_PS: extra {sorted(got - expected)}, missing {sorted(expected - got)}')
    m = c['metrics']
    return _score_retrain(rows, replicates=replicates, seed=seed, min_computed=m['validity']['min_candidate_computed_fraction'])


def _score_retrain(rows, *, replicates=REPLICATES, seed=SEED, min_computed=None):
    """rows: game_pk, actual, cand_R and cand_C (9 probs, or both None = W1 omitted, with omit_reason), bc, freq, known.
    Both candidates read the same W1 inputs (same Encoder11), so an omitted row is omitted for both; both are then
    scored uniform 1/9 there (dNLL 0 on that row). Primary: paired R - C. Descriptive: each vs service BC and frozen frequency."""
    refuse_demo_game(r['game_pk'] for r in rows)
    excluded, omitted, keep = {}, {}, []
    for r in rows:
        y = label(r['actual'])
        if y is None:
            excluded[str(r['actual'])] = excluded.get(str(r['actual']), 0) + 1
            continue
        if (r['cand_R'] is None) != (r['cand_C'] is None):
            raise ValueError('R and C must be omitted on the same rows (same W1 inputs)')
        if r['cand_R'] is None:
            reason = r.get('omit_reason') or 'unknown'
            omitted[reason] = omitted.get(reason, 0) + 1
            r = dict(r, cand_R=UNIFORM, cand_C=UNIFORM, filled=True)
        keep.append((r, y))
    if not keep:
        raise ValueError('no evaluable rows')
    y = np.array([k[1] for k in keep]); games = np.array([k[0]['game_pk'] for k in keep])
    Q = {name: np.array([k[0][name] for k in keep], np.float64) for name in ('cand_R', 'cand_C', 'bc', 'freq')}
    for name, q in Q.items():
        if q.shape != (len(keep), K) or not np.isfinite(q).all() or not np.allclose(q.sum(1), 1, atol=1e-9) or (q < 0).any():
            raise ValueError(f'{name} probabilities')
    F = {n: floor(q) for n, q in Q.items()}
    primary = paired_contrast(F['cand_R'], F['cand_C'], y, games, replicates=replicates, seed=seed)
    present = np.array([not k[0].get('filled') for k in keep])
    valid = validity(int(present.sum()), len(keep), min_computed)
    upper = primary['ci95']['nll'][1]
    use_r = bool(valid['valid'] is True and upper < 0)
    report = {'row_rule': ROW_RULE, 'rows': len(keep), 'games': int(len(set(games.tolist()))), 'excluded_targets': excluded,
              'candidate_omitted': omitted, 'candidate_filled_uniform_rows': int((~present).sum()), 'primary_floor_eps': EPS,
              'retrained_minus_current': primary, 'validity': valid,
              'adoption': {'rule': ADOPTION_RULE, 'dnll_upper95': upper, 'use_retrained': use_r, 'selected': 'R' if use_r else 'C',
                           'verdict_text': ('판정 없음' if valid['valid'] is False else verdict(primary)['nll'])},
              'calibration_note': 'tau_K, tau_N, w_b are the P9-013 september values for both R and C; for R September is inside FIT, so its calibration is not out-of-sample',
              'descriptive': {}}
    d = report['descriptive']
    for cand in ('cand_R', 'cand_C'):
        d[f'{cand}_minus_service_bc'] = paired_contrast(F[cand], F['bc'], y, games, replicates=replicates, seed=seed)
        d[f'{cand}_minus_frozen_frequency'] = paired_contrast(F[cand], F['freq'], y, games, replicates=replicates, seed=seed)
    nonzero = Q['bc'][np.arange(len(y)), y] > 0
    d['bc_nonzero_rows'] = {'rows': int(nonzero.sum()), 'bc_zero_rows': int((~nonzero).sum())}
    if nonzero.any():
        for cand in ('cand_R', 'cand_C'):
            d['bc_nonzero_rows'][cand] = paired_contrast(F[cand][nonzero], F['bc'][nonzero], y[nonzero], games[nonzero], replicates=replicates, seed=seed)
    if present.any() and (~present).any():
        d['candidate_present_rows_only'] = dict(rows=int(present.sum()), **paired_contrast(
            F['cand_R'][present], F['cand_C'][present], y[present], games[present], replicates=replicates, seed=seed))
    known = np.array([bool(k[0].get('known')) for k in keep])
    for name, m in (('known_pitcher', known), ('unseen_pitcher', ~known)):
        if m.any():
            d[name] = {'rows': int(m.sum()), 'cand_R': pooled(row_parts(F['cand_R'][m], y[m])), 'cand_C': pooled(row_parts(F['cand_C'][m], y[m]))}
    d['per_game_dnll_R_minus_C'] = {str(g): pooled(row_parts(F['cand_R'][games == g], y[games == g]))['nll'] - pooled(row_parts(F['cand_C'][games == g], y[games == g]))['nll']
                                    for g in sorted(set(games.tolist()))}
    return report


if __name__ == '__main__':
    # Real-data path intentionally not wired in this step: activation, grant and the X1 window come first.
    require_active(grant=pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else None)
    raise SystemExit('X1 wiring (replay snapshots -> W1 -> bundle, service BC, frequency table) is added after activation review')


# ---------------------------------------------------------------- EXP-SVC-P9-003 (POST-HOC W1 v3): same rule as P9-002 + references
CONFIG_P9_003 = REPO / 'configs/EXP-SVC-P9-003.json'
P9_003_PHASES = ('C', 'CR')
SENSITIVITY_EXCLUDED_GAMES = (849826, 849819)   # masked census v3 per-game f 0.383 and 0.753 (the only games < 0.95)


def evaluate_p9_003(rows, *, phase, grant, config=CONFIG_P9_003, facts=FACTS, replicates=REPLICATES, seed=SEED):
    """THE entry point for EXP-SVC-P9-003: ACTIVE + execute + pinned grant, exactly the registered E_PS games."""
    c = require_active(config, grant)
    if c.get('id') != 'EXP-SVC-P9-003':
        raise SystemExit('not the EXP-SVC-P9-003 registration')
    expected = set(e_ps_games(facts))
    got = {int(r['game_pk']) for r in rows}
    refuse_demo_game(got)
    if got != expected:
        raise SystemExit(f'scored games differ from E_PS: extra {sorted(got - expected)}, missing {sorted(expected - got)}')
    return _score_p9_003(rows, phase=phase, replicates=replicates, seed=seed,
                         min_computed=c['metrics']['validity']['min_candidate_computed_fraction'])


def reference_contrasts(F, Q, y, games, cand, masked_idx=None, M=None, *, replicates=REPLICATES, seed=SEED):
    """Descriptive (fixed 2026-11-19, before any E_PS outcome): candidate - each reference, all floored, game-level
    paired bootstrap. bc = service BC folded to 9 groups; freq = frozen september frequency; bc_masked = the
    support-masked reference the policy consumes (rows where it exists); bc_nonzero = rows where raw BC gives y > 0."""
    out = {}
    for ref, name in (('bc', 'minus_service_bc_folded9'), ('freq', 'minus_frozen_frequency')):
        con = paired_contrast(F[cand], F[ref], y, games, replicates=replicates, seed=seed)
        out[name] = dict(con, verdict_text=verdict(con)['nll'])
    if masked_idx is not None and len(masked_idx):
        con = paired_contrast(F[cand][masked_idx], M, y[masked_idx], games[masked_idx], replicates=replicates, seed=seed)
        out['minus_service_masked_reference'] = dict(con, rows=int(len(masked_idx)), verdict_text=verdict(con)['nll'])
    nonzero = Q['bc'][np.arange(len(y)), y] > 0
    out['bc_nonzero_rows'] = {'rows': int(nonzero.sum()), 'bc_zero_rows': int((~nonzero).sum())}
    if nonzero.any():
        out['bc_nonzero_rows'].update(paired_contrast(F[cand][nonzero], F['bc'][nonzero], y[nonzero], games[nonzero],
                                                      replicates=replicates, seed=seed))
    return out


def _score_p9_003(rows, *, phase, replicates=REPLICATES, seed=SEED, min_computed=None):
    """phase 'C': C only (R pins may still be PENDING): validity + C vs the references. phase 'CR': P9-002's primary,
    validity and adoption rule unchanged (``_score_retrain``) + both candidates vs the references.
    rows as ``_score_retrain`` (phase C: cand_R None everywhere, cand_C None = W1 omitted -> uniform 1/9)."""
    if phase not in P9_003_PHASES:
        raise ValueError(f'phase {phase!r}')
    refuse_demo_game(r['game_pk'] for r in rows)
    names = ('cand_C',) if phase == 'C' else ('cand_C', 'cand_R')
    if phase == 'C' and any(r.get('cand_R') is not None for r in rows):
        raise ValueError('phase C rows carry an R prediction')
    excluded, omitted, keep = {}, {}, []
    for r in rows:
        y = label(r['actual'])
        if y is None:
            excluded[str(r['actual'])] = excluded.get(str(r['actual']), 0) + 1
            continue
        if phase == 'CR' and (r['cand_R'] is None) != (r['cand_C'] is None):
            raise ValueError('R and C must be omitted on the same rows (same W1 inputs)')
        if r['cand_C'] is None:
            reason = r.get('omit_reason') or 'unknown'
            omitted[reason] = omitted.get(reason, 0) + 1
            r = dict(r, filled=True, **{n: UNIFORM for n in names})
        keep.append((r, y))
    if not keep:
        raise ValueError('no evaluable rows')
    y = np.array([k[1] for k in keep]); games = np.array([k[0]['game_pk'] for k in keep])
    Q = {n: np.array([k[0][n] for k in keep], np.float64) for n in names + ('bc', 'freq')}
    for n, q in Q.items():
        if q.shape != (len(keep), K) or not np.isfinite(q).all() or not np.allclose(q.sum(1), 1, atol=1e-9) or (q < 0).any():
            raise ValueError(f'{n} probabilities')
    F = {n: floor(q) for n, q in Q.items()}
    present = np.array([not k[0].get('filled') for k in keep])
    masked_idx = np.array([i for i, k in enumerate(keep) if k[0].get('bc_masked') is not None], np.int64)
    M = floor(np.array([keep[i][0]['bc_masked'] for i in masked_idx], np.float64)) if len(masked_idx) else None
    if phase == 'CR':
        report = _score_retrain(rows, replicates=replicates, seed=seed, min_computed=min_computed)   # P9-002 rule, unchanged
    else:
        report = {'row_rule': ROW_RULE, 'rows': len(keep), 'games': int(len(set(games.tolist()))),
                  'excluded_targets': excluded, 'candidate_omitted': omitted,
                  'candidate_filled_uniform_rows': int((~present).sum()), 'primary_floor_eps': EPS,
                  'validity': validity(int(present.sum()), len(keep), min_computed),
                  'adoption': {'rule': ADOPTION_RULE, 'phase': 'C', 'selected': None, 'use_retrained': None,
                               'note': 'R not scored in phase C; the R-vs-C rule applies in phase CR only'}}
    report['phase'] = phase
    report['candidate_omitted'] = omitted
    report['recommendation_references'] = {
        'gate': False,
        'note': 'descriptive, fixed before any E_PS outcome; input to the PM/owner decision on using the predictor in '
                'recommendations, not an automatic rule. Numbers are 판정 없음 when validity is not valid.',
        'valid': report['validity']['valid'],
        **{n: reference_contrasts(F, Q, y, games, n, masked_idx, M, replicates=replicates, seed=seed) for n in names}}
    if present.any() and (~present).any():
        report['recommendation_references']['candidate_present_rows_only'] = {
            n: dict(rows=int(present.sum()), **{ref: paired_contrast(F[n][present], F[ref][present], y[present], games[present],
                                                                     replicates=replicates, seed=seed)['point']
                                                for ref in ('bc', 'freq')}) for n in names}
    # fixed before phase C (review note, 2026-11-19): point estimates without the two low-f games of the masked census
    drop = np.isin(games, SENSITIVITY_EXCLUDED_GAMES)
    if drop.any() and (~drop).any():
        m = ~drop
        report['recommendation_references']['sensitivity_excluding_low_f_games'] = {
            'games_excluded': list(SENSITIVITY_EXCLUDED_GAMES), 'rows': int(m.sum()), 'point_only': True,
            **{n: {ref: paired_contrast(F[n][m], F[ref][m], y[m], games[m], replicates=1, seed=seed)['point']
                   for ref in ('bc', 'freq')} for n in names}}
    known = np.array([bool(k[0].get('known')) for k in keep])
    report['recommendation_references']['known_unseen'] = {
        name: {'rows': int(m.sum()), **{n: pooled(row_parts(F[n][m], y[m])) for n in names + ('bc', 'freq')}}
        for name, m in (('known_pitcher', known), ('unseen_pitcher', ~known)) if m.any()}
    return report
