"""EXP-OPE-RES-02 phase-2 synthetic drivers: S-2' (R1 surrogate re-certification), G-INT-S2EQ and the TAB re-run.

S-2' (design 5.2, decision 3 option 1): ``run_s2_prime`` is the EXP-OPE-RES-01 ``run_s5x`` copied with one change
- the seed scheme. ``scheme='A2'`` takes world_seed(k) = role_seed(v2) XOR k and the A2-base layer streams
SeedSequence([20261027, 3, k, layer, i, draw]) (G-INT-S2EQ reproduces the recorded A2 runs k = 4, 5, 9-12);
``scheme='A2b'`` takes the registered A2b world seed SeedSequence([20261028, 5, k]) and the A2b layer role
SeedSequence([20261028, 4, k, layer, i, draw]). Everything else is the same code path (tested bit for bit against
``ope_synthetic_a2.run_s5x`` on a fixture world at the same A2 seeds).

Path (G-CODE judgment, design 9.3 / review 004 note 3): the candidate planner seeds every Q search from a hash of
the full state (``rollout_policy.planning_seed`` :268-274, used by ``RolloutImprovement.q_values`` :382-396) and
caches results by the exact state (:386-391, capacity 4,096 without eviction, :417), so it holds no sequential
random state; but the bound G0 predictor caches its frequency-baseline rows across calls
(``matrix_policy.FrozenGEnsemble.__call__`` :229-234, keyed by context, count and action and filled in whatever
batch first needs them), and whether the pinned frequency model is batch-invariant to the last bit cannot be read
from the code. The truth values could therefore depend on whether the log phase ran first. The exact reproduction
and the new-seed runs both use the A2 order WITH the log phase (logs through the candidate runtime ledger, then
the CRN truth with the gamma record): ``S2P_PATH = 'a2_order_with_log_phase'``. The 433.9 s unit already includes
the logs, so the projection is unchanged. The run also records its S5-X repetitions (the A2 body), unused by S-2'.

Pooling (operator-blind): per-run results go to per-run sealed files; ``pool_cell`` refuses unless every registered
run of the cell (N' fixed: 10 / 55 / 5) is sealed and verified; b_hat = mean_k b_hat_k, SE = sqrt(sum SE_k^2) / N';
only fresh A2b runs are pooled (the A2 runs k = 4, 5 are reported beside, never pooled).

TAB re-run (design 5.7): the A2 world (``TabWorld`` fixed draws from the EXP-OPE-RES-01 role_sequence('tab_world', 0,
component), eta_p 0.16405...) with A2b replication streams SeedSequence([20261028, 3, 1, cell, replication]) at the
new E0 dataset size, and the exact-value cell rule for a tuple equal to its baseline in every replication.
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np

from . import ope_redesign_registry as a2
from . import ope_registry_a2b as rb
from . import ope_synthetic_a2 as osx
from .policy_artifacts import IntegrityError, _require

SCHEMES = ('A2', 'A2b')
COMPONENTS = ('R1_a', 'R1_b')


def scheme_seeds(scheme, k):
    """(world seed, layer-stream factory) of run k under a seed scheme (the only difference between the schemes)."""
    if scheme == 'A2':
        _require(type(k) is int and 1 <= k <= 15, 'the A2 seed scheme reproduces the registered A2 runs 1..15')
        return a2.world_seed(k), (lambda *key: a2.role_sequence('s5x_world', k, *key))
    _require(scheme == 'A2b', f'unknown seed scheme {scheme!r}')
    rb.registered_run(k)  # 1..70 only
    return rb.world_seed(k), (lambda *key: rb.role_sequence('s5x_layers', k, *key))


def run_s2_prime(runtime, components, starts, *, k, scheme, cell, law, layers, world_predictor=None, logs_per_start=6,
                 truth_rollouts=12, cap=12, repetitions=a2.S5X_REPETITIONS, budget=None, deadline=None):
    """One S-2' run: ``ope_synthetic_a2.run_s5x`` (logs through the candidate runtime ledger, CRN truth, the six
    repetitions, the Rao-Blackwellized gamma bias and, at zeta = 0, the gamma_a invariant) with the seed scheme's
    world seed and layer streams. ``cell``: '2*zeta_p', '2*kappa_p' or 'descriptive_kappa_prime' (A2 scheme: the
    recorded dose family cell of k)."""
    from . import ope_variants as ov
    from .policy_runtime import OUTSIDE_POLICY_SUPPORT, DecisionRequest
    from .policy_semisynthetic import generate_pa
    _require(cell in rb.S2P_CELLS, f'unknown S-2\' cell {cell!r}')
    _require(runtime.components is components and runtime.improvement is not None, 'candidate runtime required')
    _require(logs_per_start == repetitions and truth_rollouts == 2 * repetitions, 'repetitions split the S5 budget')
    if scheme == 'A2':  # the recorded run's own levels (k = 9-12 are kappa_p, 5 kappa_p, zeta_p, 5 zeta_p; NOTE-2)
        _require(k in rb.GINT_S2EQ_RUNS.get(cell, ()), f'run {k} is not a recorded run of the {cell} dose family')
        spec = next(r for r in a2.S5X_RUNS if r['k'] == k)
        kappa_label, zeta_label = spec['kappa'], spec['zeta']
    else:
        kappa_label, zeta_label = rb.S2P_CELL_LEVELS[cell]
    if scheme == 'A2b':
        _require(rb.registered_run(k) == cell, f'run {k} is not a registered run of the {cell} cell')
    run = {'k': k, 'cell': cell, 'scheme': scheme, 'setting': rb.S2P_WORLD['setting'], 'law': rb.S2P_WORLD['law'],
           'planning_seed': rb.S2P_WORLD['planning_seed'], 'kappa': kappa_label, 'zeta': zeta_label,
           'role': 'S2_prime', 'path': rb.S2P_PATH}
    gamma, invariant = True, zeta_label == '0'
    ws, layer_stream = scheme_seeds(scheme, k)
    budget = runtime.improvement.simulator.budget if budget is None else budget
    world = osx.QualityLayerWorld(components.inputs.pool, world_predictor or components.g0, components.we.terminal,
                                  budget, layers)
    pas, ends, excluded, kept = {}, {}, [], []
    for i, (start, hand) in enumerate(starts):
        inside, reason = runtime.start_population(start, hand)
        if not inside:
            excluded.append({'start': i, 'reason': reason})
            continue
        kept.append(i)
        for j in range(logs_per_start):
            if deadline is not None:
                deadline()
            pa_id = f'w{i}:{j}'
            rng = np.random.default_rng(np.random.SeedSequence([ws, 0, i, j]))
            world.bind(np.random.default_rng(layer_stream(0, i, j)).random((cap, 3)))
            states, actions, (end, value), last = generate_pa(runtime, world, start, law, rng, cap)
            rows = [runtime.submit(DecisionRequest(f'{pa_id}:{t}', pa_id, t, state, action, runtime.sha256, hand))
                    for t, (state, action) in enumerate(zip(states, actions))]
            _require(all(r['status'] in ('SUPPORTED', OUTSIDE_POLICY_SUPPORT) for r in rows),
                     'a generated in-E0 log was refused by the runtime')
            if end in ('absorbed', 'cap'):
                value = float(components.we.cutoff(last))
            pas[pa_id] = {'game': f'start{i}', 'in_population': True, 'reward': float(value), 'end_kind': end}
            ends[pa_id] = world.last if end == 'terminal' else None
    _require(kept, 'no registered start is inside E0')
    by_pa = {}
    for row in runtime.ledger.iter_decisions():
        by_pa.setdefault(row['pa_id'], []).append(row)
    lines = {pa_id: ov.pa_line(sorted(by_pa[pa_id], key=lambda d: d['decision_index']), info)
             for pa_id, info in pas.items()}
    # truth: CRN rollouts, the run_world stream layout, layer draws shared by both policies
    improvement = runtime.improvement
    play = getattr(runtime, 'planning_policy', None) or (lambda state: runtime.candidate(state)[0])
    policies = {'reference': lambda state, depth: (improvement.bc.actions, improvement.bc.probabilities(state)),
                'candidate': lambda state, depth: (improvement.bc.actions, play(state))}
    truth, g_sums = {}, {}
    gamma_cache, invariant_max = {}, 0.0
    for i in kept:
        if deadline is not None:
            deadline()
        seed = int(np.random.SeedSequence([ws, 1, i]).generate_state(1)[0])
        uniforms = np.random.default_rng(seed).random((truth_rollouts, cap, 3))
        layer = np.stack([np.random.default_rng(layer_stream(1, i, r)).random((cap, 3)) for r in range(truth_rollouts)])
        values = {}
        for name, policy in policies.items():
            sums = np.zeros((truth_rollouts, 2))
            record = None
            if gamma or invariant:
                def record(r, depth, state, names, probs, sums=sums):
                    nonlocal invariant_max
                    for a, p in zip(names, probs):
                        if p <= 0:
                            continue
                        key = (state, a)
                        if key not in gamma_cache:
                            gamma_cache[key] = osx._gamma(components, world, state, a, budget)
                        g_a, g_b = gamma_cache[key]
                        invariant_max = max(invariant_max, abs(g_a))
                        sums[r] += p * np.array([g_a, g_b])
            values[name] = osx.rollouts_x(world, [starts[i][0]] * truth_rollouts, policy, uniforms, layer,
                                          components.we.cutoff, record=record)
            g_sums[i, name] = sums
        truth[i] = values
    if invariant and invariant_max > rb.INTEGRITY_TOL:
        raise IntegrityError(f'S-2\' invariant violated: |gamma_a| = {invariant_max} > 1e-12 at zeta = 0 '
                             f'(deterministic integrity)')
    runtime.verify_components()
    # the six complete repetitions over the fixed starts (A2 body; recorded, not an S-2' input)
    reps = []
    for r in range(1, repetitions + 1):
        streams = osx.repetition_streams(r, repetitions)
        ids = [f'w{i}:{streams["log"]}' for i in kept]
        rep_lines = [lines[pa] for pa in ids]
        steps, lengths = osx._padded_lines(rep_lines)
        data = {'steps': steps, 'lengths': lengths, 'game': np.array(kept)}
        rewards = osx.s5x_rewards(rep_lines, [ends[pa] for pa in ids], layers)
        estimates, selected, bad = osx.pipeline(data, rewards)
        pick = list(streams['truth'])
        true_r = float(np.mean([np.mean(truth[i]['candidate'][pick] - truth[i]['reference'][pick]) for i in kept]))
        raw = {p: float(estimates[p] - true_r) for p in a2.PAIRS}
        bad_r = sorted(set(bad) | {p for p in a2.PAIRS if not math.isfinite(raw[p])})
        reps.append({'gaps': {p: (None if p in bad_r else raw[p]) for p in a2.PAIRS}, 'truth': true_r, 'invalid': bad_r,
                     'invalid_reason': {p: osx.INVALID_REPETITION_REASON for p in bad_r},
                     'selected': {n: ('inf' if v == rb.INF else str(v)) for n, v in selected.items()}})
    gaps = {p: np.array([np.nan if rep['gaps'][p] is None else rep['gaps'][p] for rep in reps]) for p in a2.PAIRS}
    invalid_reps = {p: sum(p in rep['invalid'] for rep in reps) for p in a2.PAIRS}
    result = {'run': dict(run), 'world_seed': ws, 'starts': len(starts), 'kept_starts': len(kept),
              'world_predictor': {'kind': 'bound_g0'} if world_predictor is None else dict(world_predictor.identity),
              'excluded_starts': excluded, 'repetitions': reps, 'pairs': {}}
    for p in a2.PAIRS:
        valid = invalid_reps[p] == 0 and np.isfinite(gaps[p]).all()
        result['pairs'][p] = {'gap': float(gaps[p].mean()) if valid else None,
                              'se': float(gaps[p].std(ddof=1) / math.sqrt(repetitions)) if valid else None,
                              'valid': bool(valid), 'invalid_repetitions': int(invalid_reps[p]),
                              'reason': None if valid else osx.INVALID_REPETITION_REASON}
    result['inherited_s5_se'] = osx._inherited_se(lines, ends, layers, kept, truth, logs_per_start)
    result['s2'] = {}
    exact = invariant and zeta_label == '0'  # the registered gamma_a = 0 invariant at zeta = 0
    for col, comp in ((0, 'R1_a'), (1, 'R1_b')):
        diffs = np.array([g_sums[i, 'candidate'][:, col] - g_sums[i, 'reference'][:, col] for i in kept])
        n = len(kept)
        b_hat = float(diffs.mean(axis=1).sum() / n)
        se = float(math.sqrt(float((diffs.std(axis=1, ddof=1) ** 2 / truth_rollouts).sum())) / n)
        result['s2'][comp] = {'b_hat': b_hat, 'se': se,
                              **rb.cell_outcome(b_hat, se, exact_zero_allowed=bool(exact and comp == 'R1_a'))}
    result['s2_pairs'] = osx.s2_pair_cells(result['s2'], result['pairs'], k)
    result['first_order_reference_approximation'] = {
        'status': 'NOT_RUN', 'reason': 'as EXP-OPE-RES-01: needs planner calls at every reference-visited state'}
    result['invariant'] = None if not invariant else {'max_abs_gamma_a': invariant_max, 'outcome': 'PASS'}
    return result


# ---------------------------------------------------------------- run seals and operator-blind pooling
def canonical(document):
    return (json.dumps(document, ensure_ascii=False, sort_keys=True, indent=1, allow_nan=False) + '\n').encode('utf-8')


def run_record(result):
    """The sealed per-run record (b_hat and SE stay sealed until every run of the cell is sealed)."""
    run = result['run']
    _require(run['scheme'] == 'A2b', 'only fresh A2b runs are sealed for pooling (A2 runs are never pooled)')
    s2 = {c: {'b_hat': result['s2'][c]['b_hat'], 'se': result['s2'][c]['se']} for c in COMPONENTS}
    finite = all(math.isfinite(v) for c in COMPONENTS for v in s2[c].values())
    invariant_ok = result['invariant'] is None or result['invariant']['outcome'] == 'PASS'
    return {'experiment': rb.EXPERIMENT_ID, 'stage': 'A2b-S2a', 'cell': run['cell'], 'k': run['k'],
            'world_seed': result['world_seed'], 'path': run['path'], 'kept_starts': result['kept_starts'],
            's2': s2, 'invariant': result['invariant'], 'pair_valid': {p: result['pairs'][p]['valid'] for p in a2.PAIRS},
            'valid': bool(finite and invariant_ok)}


def seal_run(record):
    data = canonical(record)
    return data, hashlib.sha256(data).hexdigest()


def run_public(record, sealed_sha256, *, wall_seconds, peak_rss_bytes):
    """Public item 12: wall time, RSS, integrity pass/fail and the seal of one run (no b_hat or SE)."""
    doc = {'experiment': rb.EXPERIMENT_ID, 'stage': 'A2b-S2a', 'cell': record['cell'], 'k': record['k'],
           'wall_seconds': float(wall_seconds), 'peak_rss_bytes': int(peak_rss_bytes),
           'integrity': 'PASS' if record['valid'] else 'FAIL', 'sealed_sha256': sealed_sha256}
    rb.validate_run_public(doc)
    return doc


S2P_TUPLES = {'R1_a': ('R1_axR2_0', 'R1_axR2_1'), 'R1_b': ('R1_bxR2_0', 'R1_bxR2_1')}  # the B0 candidates' A2 pairs


def pool_cell(cell, sealed_runs):
    """Pool one cell after EVERY registered run is sealed: ``sealed_runs`` = {k: (bytes, sha256)}; each seal is
    re-verified. b_hat = mean_k, SE = sqrt(sum SE_k^2) / N'; an invalid run makes the cell INVALID (never dropped);
    per tuple, an invalid repetition of its pair in any run makes its cell INCONCLUSIVE (the A2 s2_pair_cells
    rule). The 2 kappa_p runs keep the gamma_a invariant (R1_a: exact zero)."""
    _require(cell in rb.S2P_CELLS, f'unknown S-2\' cell {cell!r}')
    expected = rb.RUN_INDEX[cell]
    _require(sorted(sealed_runs) == list(expected),
             f'{cell}: pooling needs every registered run sealed first (operator-blind pooling): '
             f'{len(sealed_runs)} of {len(expected)}')
    records = []
    for k in expected:
        data, sha = sealed_runs[k]
        _require(hashlib.sha256(data).hexdigest() == sha, f'run {k} differs from its seal (deterministic integrity)')
        record = json.loads(data)
        _require(record['cell'] == cell and record['k'] == k and record['world_seed'] == rb.world_seed(k),
                 f'run {k} is not the registered run of {cell}')
        records.append(record)
    n = len(records)
    out = {'cell': cell, 'n_prime': n, 'runs': list(expected), 'components': {}, 'tuples': {},
           'decision_input': cell in rb.S2P_DECISION_CELLS}
    valid = all(r['valid'] for r in records)
    for comp in COMPONENTS:
        b = [r['s2'][comp]['b_hat'] for r in records]
        s = [r['s2'][comp]['se'] for r in records]
        if not valid:
            out['components'][comp] = {**rb.cell_outcome(None, None, valid=False), 'invalid_runs':
                                       [r['k'] for r in records if not r['valid']]}
            continue
        b_hat = float(sum(b) / n)
        se = float(math.sqrt(sum(x * x for x in s)) / n)
        exact = comp == 'R1_a' and rb.S2P_CELL_LEVELS[cell][1] == '0'  # gamma_a = 0 invariant at zeta = 0
        out['components'][comp] = {**rb.cell_outcome(b_hat, se, exact_zero_allowed=exact), 'runs': n}
    for comp, pairs in S2P_TUPLES.items():
        for pair in pairs:
            bad = [r['k'] for r in records if not r['pair_valid'][pair]]
            out['tuples'][f'{pair}xB0'] = (dict(out['components'][comp]) if not bad else
                                          a2.inconclusive_invalid(f'invalid repetitions of {pair} in runs {bad}'))
    out['invariant'] = ({'max_abs_gamma_a': max(r['invariant']['max_abs_gamma_a'] for r in records), 'outcome': 'PASS'}
                        if all(r['invariant'] for r in records) else None)
    return out


# ---------------------------------------------------------------- G-INT-S2EQ (A2b-S2a, before any new-seed run)
def recorded_runs(results):
    """{k: {R1_a: (b_hat, se), R1_b: (b_hat, se)}} of the recorded A2 dose-family runs (results file
    S2_s5x_summary.dose_response_s2_by_run.runs[*].per_r1_component)."""
    out = {}
    for run in results['S2_s5x_summary']['dose_response_s2_by_run']['runs']:
        out[int(run['k'])] = {c: (run['per_r1_component'][c]['b_hat'], run['per_r1_component'][c]['se'])
                              for c in COMPONENTS}
    return out


def reference_cross_check(recorded, sealed):
    """Review 004 note 3 c: the results-file values (k = 9-12 were PM-extracted) against the sealed A2-S2 run records
    ({k: {comp: (b_hat, se)}} or None when not supplied); key lists only."""
    if sealed is None:
        return {'sealed_a2_s2_run_records': 'not supplied', 'equal': None, 'mismatched_keys': []}
    mismatched = sorted(f'k{k}.{c}.{name}' for k in recorded for c in COMPONENTS for name, i in (('b_hat', 0), ('se', 1))
                        if k not in sealed or not rb.integrity_equal(recorded[k][c][i], sealed[k][c][i]))
    return {'sealed_a2_s2_run_records': 'compared', 'equal': not mismatched, 'mismatched_keys': mismatched}


def sealed_runs(directory, ks):
    """{k: {R1_a/R1_b: (b_hat, se)}} from the sealed A2-S2 run records run-kNN.json."""
    out = {}
    for k in ks:
        record = json.loads((directory / f'run-k{k:02d}.json').read_text())
        out[k] = {c: (record['s2'][c]['b_hat'], record['s2'][c]['se']) for c in COMPONENTS}
    return out


def gint_s2eq_record(cells, results_by_k, recorded, *, reference, sealed=None):
    """G-INT-S2EQ (public item 14): the new driver in the A2 seed scheme reproduces the recorded per-run R1_a /
    R1_b b_hat and SE of the opened cells' dose family (2 zeta_p -> k = 5, 11, 12; 2 kappa_p -> k = 4, 9, 10)
    within an ABSOLUTE 1e-12 (the registered exact form). The statistical form |delta b_hat| <= 3 sqrt(SE_rec^2 +
    SE_new^2) is an auxiliary record with its detection limit in B units (never a pass criterion). Both forms ran
    on the same path as the new-seed runs (review 004 note 3 a)."""
    cells = tuple(c for c in rb.S2P_DECISION_CELLS if c in cells)
    _require(cells, 'G-INT-S2EQ runs for the opened decision cells only')
    ks = sorted(k for c in cells for k in rb.GINT_S2EQ_RUNS[c])
    _require(sorted(results_by_k) == ks, f'G-INT-S2EQ needs exactly the recorded runs {ks}')
    compared, mismatched, max_delta = [], [], 0.0
    aux_ok, limits = True, {}
    for k in ks:
        result = results_by_k[k]
        _require(result['run']['scheme'] == 'A2' and result['world_seed'] == a2.world_seed(k),
                 f'run {k} did not use the A2 seed scheme')
        limits[f'k{k}'] = {}
        for comp in COMPONENTS:
            new_b, new_se = result['s2'][comp]['b_hat'], result['s2'][comp]['se']
            rec_b, rec_se = recorded[k][comp]
            for name, mine, theirs in (('b_hat', new_b, rec_b), ('se', new_se, rec_se)):
                key = f'k{k}.{comp}.{name}'
                compared.append(key)
                if not rb.integrity_equal(mine, theirs):
                    mismatched.append(key)
                if math.isfinite(mine) and math.isfinite(theirs):
                    max_delta = max(max_delta, abs(mine - theirs))
            bound = 3 * math.sqrt(rec_se ** 2 + new_se ** 2)
            aux_ok &= abs(new_b - rec_b) <= bound
            limits[f'k{k}'][comp] = float(bound / rb.B)
    record = {'experiment': rb.EXPERIMENT_ID, 'check': 'G-INT-S2EQ', 'result': 'FAIL' if mismatched else 'PASS',
              'max_abs_delta': float(max_delta), 'tolerance': dict(rb._TOLERANCE), 'run_indices': ks,
              'cells': list(cells), 'paths': {'exact_form': rb.S2P_PATH, 'statistical_form': rb.S2P_PATH,
                                              'new_seed_runs': rb.S2P_PATH},
              'compared_keys': compared, 'mismatched_keys': mismatched,
              'auxiliary_statistical_form': {'result': 'PASS' if aux_ok else 'FAIL', 'detection_limit_B': limits,
                                             'decision_input': False},
              'reference': dict(reference),
              'reference_cross_check': reference_cross_check({k: recorded[k] for k in ks}, sealed),
              'failure_scope': rb.GINT_S2EQ_FAILURE_SCOPE}  # design 9.6: not narrowed to the S-2' tuples (NOTE-8)
    rb.validate_integrity_record(record, 'gint_s2eq')
    return record


def gint_s2eq_scope(record):
    """On the same path an auxiliary miss beside an exact pass means the comparison code is wrong (review 004 note
    3 a): a deterministic integrity failure, not a report-only line."""
    if record['result'] == 'PASS' and record['auxiliary_statistical_form']['result'] == 'FAIL':
        raise IntegrityError('G-INT-S2EQ: an auxiliary statistical miss beside an exact pass on the same path '
                             '(comparison defect; deterministic integrity)')
    if record['result'] != 'PASS':
        raise IntegrityError(f'G-INT-S2EQ failed (deterministic integrity): {record["mismatched_keys"]}')
    return record


def kappa_prime(hit_probability, groups, diagnostics):
    """The descriptive kappa_p' rule (design 5.2, review NOTE N2): the A2 kappa rule (ope_surrogate.calibrate_kappa)
    over the four large groups FB, FC, BR, OS only. A2-S0 already applied exactly this rule to the same TRAIN inputs
    for its TAB world (``TAB_GROUPS``), so A2b-S0 reads the sealed ``kappa_calibration.TAB`` instead of recomputing it
    (G-CODE MINOR-1); this function documents and tests the identity."""
    from .ope_surrogate import calibrate_kappa
    return calibrate_kappa(hit_probability, groups, diagnostics, rb.KAPPA_PRIME_GROUPS)


# ---------------------------------------------------------------- TAB re-run (A2 world, A2b replication streams)
def tab_replication(world, cell_index, replication, *, n_train, n_pas, smoke=None, stream='tab_world'):
    """One replication: the A2 ``tab_replication`` with the A2b stream SeedSequence([20261028, 3, 1, cell,
    replication]) (smoke: [20261028, 8, 1, replication]) and the new dataset size ``n_pas``."""
    _require(world.seed_role == 'tab_world', 'the TAB re-run uses the A2 world (EXP-OPE-RES-01 fixed draws)')
    if stream == 'tab_world':
        rng = np.random.default_rng(rb.role_sequence('tab_world', 1, cell_index, replication))
    else:
        _require(stream == 'smoke', 'TAB re-run streams: tab_world or smoke')
        rng = np.random.default_rng(rb.role_sequence('smoke', 1, replication))
    folds = world.synthetic_train(rng, n_train)
    if smoke is None:
        try:
            m, _ = osx.select_m(folds)
        except IntegrityError:
            m = None
    else:
        m = smoke['m']
    table = None if m is None else osx.fit_table(folds[0] + folds[1], m)
    data = world.simulate(rng, n_pas=n_pas)
    rewards = osx.tab_rewards(world, data, table)
    estimates, selected, invalid = osx.pipeline(data, rewards, smoke=smoke)
    not_evaluable = [] if table is not None else [p for p in a2.PAIRS if p.startswith('R1_b')]
    invalid = [p for p in invalid if p not in not_evaluable]
    return estimates, {**selected, 'm': m}, invalid, not_evaluable


def tab_cell(world, cell, *, n_train, n_pas, replications=None, deadline=None):
    """The A2 ``tab_cell`` on the re-run streams and size, with the exact-value cell rule: a tuple equal to the
    baseline in EVERY replication (bit for bit) has b_hat = beta_0 exactly (dynamic programming) and SE = 0 by
    construction -> EXACT_PASS / EXACT_FAIL instead of the degenerate INCONCLUSIVE (design 5.1 (b))."""
    R = cell['R'] if replications is None else replications
    estimates = {p: np.full(R, np.nan) for p in a2.PAIRS}
    picks, invalid, not_evaluable = {}, {p: 0 for p in a2.PAIRS}, {p: 0 for p in a2.PAIRS}
    for j in range(R):
        if deadline is not None:
            deadline()
        est_j, sel, bad, skipped = tab_replication(world, cell['index'], j, n_train=n_train, n_pas=n_pas)
        for p in a2.PAIRS:
            estimates[p][j] = est_j[p]
            invalid[p] += p in bad
            not_evaluable[p] += p in skipped
        for name, value in sel.items():
            label = 'none' if value is None else ('inf' if value == rb.INF else str(value))
            picks.setdefault(name, {}).setdefault(label, 0)
            picks[name][label] += 1
    base = estimates[a2.BASELINE]
    out = {}
    for p in a2.PAIRS:
        diff = estimates[p] - base
        valid = invalid[p] == 0 and invalid[a2.BASELINE] == 0 and not_evaluable[p] == 0 and np.isfinite(diff).all()
        exact = bool(valid and p != a2.BASELINE and np.all(diff == 0))
        if p == a2.BASELINE:
            b_hat, se = world.beta_0, 0.0
        else:
            b_hat = float(np.mean(diff)) + world.beta_0 if valid else None
            se = float(np.std(diff, ddof=1) / math.sqrt(R)) if valid and R > 1 else None
        finite = np.isfinite(estimates[p])
        record = {'b_hat': b_hat, 'se': se, 'invalid_replications': invalid[p],
                  'not_evaluable_replications': not_evaluable[p], 'R': R,
                  'sd': float(np.std(estimates[p][finite], ddof=1)) if finite.sum() > 1 else None,
                  'rmse': float(np.sqrt(np.mean((estimates[p][finite] - world.delta_true) ** 2))) if finite.any() else None}
        relative = cell['law'] == 'V3'
        if cell['role'] == 'acceptance':
            if p == a2.BASELINE:
                record.update(outcome='EXACT_DP', note='beta_0 by exact dynamic programming; not a certification cell')
            elif not_evaluable[p]:
                record.update(rb.cell_outcome(None, None, not_evaluable=True))
            elif invalid[p] or invalid[a2.BASELINE]:
                record.update(a2.inconclusive_invalid(f'{invalid[p] + invalid[a2.BASELINE]} invalid replications'),
                              b_hat=b_hat, se=se)
            elif exact:
                record.update(rb.exact_cell_outcome(world.beta_0, beta_0=world.beta_0 if relative else None))
            else:
                record.update(rb.cell_outcome(b_hat, se, beta_0=world.beta_0 if relative else None, valid=valid))
        out[p] = record
    return {'cell': {k: cell[k] for k in ('id', 'law', 'eta', 'kappa', 'zeta', 'R', 'role', 'drift', 'index')},
            'dataset_pas': n_pas, 'delta_true': world.delta_true, 'beta_0': world.beta_0, 'tau': world.tau,
            'expected_delta_hat_0': world.expected_delta_hat_0, 'pairs': out, 'selections': picks,
            'streams': 'SeedSequence([20261028, 3, 1, cell, replication]); world fixed draws EXP-OPE-RES-01 tab_world'}


def tab_world(marginals, cell, eta_p=rb.ETA_P):
    """The registered A2 world of one acceptance cell (A2 fixed draws, eta_p reused, kappa_p / zeta_p of TAB)."""
    kappa = a2.scale(cell['kappa'], marginals.kappa_p)
    zeta = a2.scale(cell['zeta'], marginals.zeta_p)
    eta = a2.scale(cell['eta'], eta_p)
    world = osx.TabWorld(marginals, law=cell['law'], eta=eta, kappa=kappa, zeta=zeta, drift=cell.get('drift', False))
    osx.tab_integrity(world)
    return world


def tab_rerun(marginals, *, n_pas, n_train, eta_p=rb.ETA_P, replications=None, deadline=None, write_cell=None):
    """The four acceptance cells T-A1..T-A4 at the new dataset size (cells sealed one by one by ``write_cell``)."""
    out = {}
    for index, spec in enumerate(rb.TAB_ACCEPTANCE):
        cell = dict(spec, index=index, role='acceptance', drift=False)
        record = tab_cell(tab_world(marginals, cell, eta_p), cell, n_train=n_train, n_pas=n_pas,
                          replications=replications, deadline=deadline)
        out[cell['id']] = record
        if write_cell is not None:
            write_cell(cell, record)
    return out
