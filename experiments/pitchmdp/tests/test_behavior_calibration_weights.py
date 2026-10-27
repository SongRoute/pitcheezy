"""EXP-BC-CAL-01 S6 weights on synthetic ledgers: W_t / W_T rules and ESS (T9), the allowlisted
reader (T10), SC-2 (T12), E0 = PolicyRuntime.start_population, R1b and R2 against the
registered L1-R estimator. No real ledger is opened."""
import json

import numpy as np
import pytest

from behavior_calibration_synthetic import estimator_ess, make_ledger, make_world
from pitchmdp import behavior_weights as bw
from pitchmdp.policy_artifacts import IntegrityError, load_hand_registry, load_support_table
from pitchmdp.policy_estimator import effective_sample_size

V = 18


def row(pa, index, status, a=0, pi=None, logging=None, request=None):
    logging = np.full(V, 1 / V) if logging is None else np.asarray(logging, float)
    pi = np.zeros(V) if pi is None else np.asarray(pi, float)
    result = None
    if status in ('SUPPORTED', 'OUTSIDE_POLICY_SUPPORT', 'UNSUPPORTED_LOGGING_POSITIVITY'):
        rho = 0. if status != 'SUPPORTED' else float(pi[a] / logging[a])
        result = {'mask': (pi > 0).tolist(), 'logging': logging.tolist(), 'reference': pi.tolist(),
                  'candidate': pi.tolist(), 'logged_index': a, 'rho_reference': rho, 'rho_candidate': rho}
    return {'kind': 'decision', 'request_id': request or f'{pa}:{index + 1}', 'pa_id': pa, 'decision_index': index,
            'status': status, 'pitcher': '1', 'pitcher_hand': 'R', 'batter_side': 'L', 'balls': 0, 'strikes': 0,
            'history_actions': [], 'logged_action': 'FF', 'context_key': f'{pa}:{index + 1}', 'result': result,
            'runtime_sha256': 'a' * 64}


def law(entries):
    out = np.zeros(V)
    for i, v in entries.items():
        out[i] = v
    return out


def test_t9_weight_rules_and_ess():
    logging = law({0: .25, 1: .5, 2: .25})
    pi = law({0: .5, 1: .5})
    pas = [('1:1', [row('1:1', 0, 'SUPPORTED', 0, pi, logging), row('1:1', 1, 'SUPPORTED', 1, pi, logging)]),
           ('1:2', [row('1:2', 0, 'SUPPORTED', 0, pi, logging), row('1:2', 1, 'OUTSIDE_POLICY_SUPPORT', 2, pi, logging)]),
           ('2:1', [row('2:1', 0, 'SUPPORTED', 0, pi, logging),
                    row('2:1', 1, 'UNSUPPORTED_LOGGING_POSITIVITY', 3, pi, law({0: .5, 1: .5}))]),
           ('2:2', [row('2:2', 0, 'SUPPORTED', 0, pi, logging), row('2:2', 1, 'UNSUPPORTED_NO_LOGGED_ACTION'),
                    row('2:2', 2, 'UNSUPPORTED_MID_PA')]),
           ('3:1', [row('3:1', 0, 'UNSUPPORTED_NO_LOGGED_ACTION')])]
    weights = {pa: bw.pa_weights(rows) for pa, rows in pas}
    assert weights['1:1']['candidate']['rho'] == [2., 1.] and weights['1:1']['candidate']['w_T'] == 2.
    assert list(weights['1:1']['candidate']['residual']) == [2., 2.]
    assert weights['1:2']['reference']['rho'] == [2., 0.] and weights['1:2']['reference']['w_T'] == 0.
    assert weights['2:1']['candidate']['rho'] == [2., 0.] and weights['2:1']['candidate']['w_T'] == 0.
    assert weights['2:2']['candidate']['rho'] == [2.] and weights['2:2']['candidate']['w_T'] == 2.
    assert weights['3:1']['candidate']['rho'] == [] and weights['3:1']['candidate']['w_T'] == 1.
    e0 = {pa: (True, None) for pa, _ in pas}
    diagnostics = bw.weight_diagnostics(pas, e0, large=10, mei=1e-3, ess_min={'pa': 1, 'game': 1}, draws=200, seed=1)
    w = [weights[pa]['candidate']['w_T'] for pa, _ in pas]
    games = [bw.game_of(pa) for pa, _ in pas]
    assert diagnostics['policies']['candidate']['ess'] == {'pa': effective_sample_size(w),
                                                           'game': effective_sample_size(w, games)}
    # a new denominator: only SUPPORTED rows change; OUTSIDE / positivity rows stay 0 for any law
    vectors = {r['request_id']: law({0: .5, 1: .25, 2: .25}) for _, rows in pas for r in rows}
    shifted = bw.weight_diagnostics(pas, e0, new_vectors=vectors, baseline=diagnostics, large=10, mei=1e-3,
                                    ess_min={'pa': 1, 'game': 1}, draws=200, seed=1)
    assert shifted['residual'][1]['candidate']['rho'] == [1., 0.]
    assert shifted['weight_shift_D']['candidate'] > 0 and shifted['weight_shift_D']['paired_below_mei'] is False
    same = bw.weight_diagnostics(pas, e0, new_vectors={r['request_id']: np.asarray(r['result']['logging'])
                                                       for _, rows in pas for r in rows if r['result']},
                                 baseline=diagnostics, large=10, mei=1e-3, ess_min={'pa': 1, 'game': 1}, draws=200, seed=1)
    assert same['weight_shift_D']['paired'] == 0 and same['kl_new_vs_b0']['mean'] == 0
    with pytest.raises(IntegrityError):
        bad = [r.copy() for r in pas[0][1]]
        bad[0] = {**bad[0], 'result': {**bad[0]['result'], 'rho_candidate': 3.}}
        bw.pa_weights(bad)  # recorded ratio must match the recorded probabilities


def write_ledger(path, rows, header=None):
    header = header or {'kind': 'header', 'runtime_sha256': 'a' * 64, 'pins': {'candidate': 'x'}, 'seq': 0, 'prev': None,
                        'sha256': 'h'}
    path.write_text('\n'.join(json.dumps(r) for r in [header, *rows]) + '\n')
    manifest = path.parent / 'manifest.json'
    manifest.write_text('{"artifact_sha256": {}}')
    return bw.file_sha256(path), bw.file_sha256(manifest)


def test_t10_reader_allowlist_and_refusals(tmp_path):
    rows = [row('1:1', 0, 'SUPPORTED', 0, law({0: 1.}), law({0: .5, 1: .5}))]
    rows[0]['result'].update(q_reference=[0.4] * V, mc_se=[0.1] * V, q_planning=[0.4] * V, logging_mass_on_mask=1.)
    rows[0].update(fingerprint='f', detail='', context_sha256=None, value=0.7, reward=1.0)
    path = tmp_path / 'ledger-dev-0.jsonl'
    ledger_sha, manifest_sha = write_ledger(path, rows)
    kwargs = dict(manifest_path=tmp_path / 'manifest.json', ledger_sha256=ledger_sha, manifest_sha256=manifest_sha,
                  runtime_sha256='a' * 64)
    header, decisions, record = bw.read_ledger(path, **kwargs)
    assert set(header) == {'runtime_sha256', 'pins'}
    assert set(decisions[0]) <= set(bw.ALLOWED_DECISION) | {'result'}
    assert set(decisions[0]['result']) == set(bw.ALLOWED_RESULT)
    flat = json.dumps(decisions)
    for forbidden in ('q_reference', 'mc_se', 'q_planning', 'value', 'reward', 'fingerprint', 'logging_mass'):
        assert forbidden not in flat
    assert record['sha256_before'] == record['sha256_after']
    for name in ('pa_values.parquet', 'dr.json'):
        (tmp_path / name).write_text('x')
        with pytest.raises(bw.LedgerRefused, match='forbidden'):
            bw.read_ledger(tmp_path / name, **kwargs)
    with pytest.raises(bw.LedgerRefused):
        bw.read_ledger(path, **{**kwargs, 'ledger_sha256': '0' * 64})
    with pytest.raises(bw.LedgerRefused):
        bw.read_ledger(path, **{**kwargs, 'manifest_path': tmp_path / 'elsewhere' / 'manifest.json'})
    with pytest.raises(IntegrityError, match='runtime_sha256'):
        bw.read_ledger(path, **{**kwargs, 'runtime_sha256': 'b' * 64})
    aborted = tmp_path / 'x'
    aborted.mkdir()
    pins = write_ledger(aborted / 'ledger-dev-0.jsonl', [*rows, {'kind': 'aborted', 'detail': 'wall cap'}])
    with pytest.raises(IntegrityError, match='aborted'):
        bw.read_ledger(aborted / 'ledger-dev-0.jsonl', manifest_path=aborted / 'manifest.json', ledger_sha256=pins[0],
                       manifest_sha256=pins[1], runtime_sha256='a' * 64)


class SpyDecoder:
    """Records every value the scanner decodes (json raw_decode is the only decode path)."""
    def __init__(self):
        self.values = []
        self.inner = json.JSONDecoder()

    def raw_decode(self, text, index):
        value, end = self.inner.raw_decode(text, index)
        self.values.append(value)
        return value, end


def test_t10_forbidden_members_are_skipped_unread(tmp_path, monkeypatch):
    """Review finding 7: non-allowlisted members are skipped by the scanner, never decoded into a
    Python object (no json.loads of a line; raw_decode only for allowlisted values)."""
    sentinel = 987654.321
    rows = [row('1:1', 0, 'SUPPORTED', 0, law({0: 1.}), law({0: .5, 1: .5}))]
    rows[0]['result'].update(q_reference=[sentinel] * V, mc_se={'nested': [sentinel, {'deep': 'S"}{]['}]},
                             q_planning='quote \\" brace } bracket ] SENTINEL', logging_mass_on_mask=sentinel)
    rows[0].update(fingerprint='SENTINEL', detail={'a': [1, 2, {'b': 'SENTINEL'}]}, value=sentinel, reward=sentinel,
                   context_sha256=None)
    path = tmp_path / 'ledger-dev-0.jsonl'
    ledger_sha, manifest_sha = write_ledger(path, rows)
    spy = SpyDecoder()
    monkeypatch.setattr(bw, '_DECODER', spy)
    monkeypatch.setattr(bw.json, 'loads', lambda *a, **k: pytest.fail('json.loads must not parse a ledger line'))
    header, decisions, record = bw.read_ledger(path, manifest_path=tmp_path / 'manifest.json', ledger_sha256=ledger_sha,
                                               manifest_sha256=manifest_sha, runtime_sha256='a' * 64)
    decoded = json.dumps(spy.values)
    assert 'SENTINEL' not in decoded and str(sentinel) not in decoded and spy.values
    assert decisions[0]['result']['logging'] == rows[0]['result']['logging']  # allowlisted values intact
    assert decisions[0]['history_actions'] == [] and decisions[0]['context_key'] == '1:1:1'
    skipped = record['skipped_unread_members']
    for name in ('q_reference', 'mc_se', 'q_planning', 'logging_mass_on_mask', 'fingerprint', 'detail', 'value',
                 'reward', 'context_sha256', 'seq', 'prev', 'sha256'):
        assert skipped.get(name, 0) >= 1, name
    assert record['parsing'] == bw.PARSING_DISCLOSURE and 'never decoded' in record['parsing']
    with pytest.raises(IntegrityError):
        bw.scan_line(b'{"kind": "decision", "q_reference": [1, 2', bw.DECISION_SCHEMA, {})
    with pytest.raises(IntegrityError):
        bw.scan_line(b'{"kind": "decision"} trailing', bw.DECISION_SCHEMA, {})


def test_kl_is_infinite_when_new_mass_leaves_the_b0_support():
    """Review finding 9: the registered mean KL is infinite as soon as one decision puts new mass
    outside the B0 support; the finite-only statistic is reported under its own name."""
    logging = law({0: .5, 1: .5})
    pi = law({0: .5, 1: .5})
    pas = [('1:1', [row('1:1', 0, 'SUPPORTED', 0, pi, logging)]), ('1:2', [row('1:2', 0, 'SUPPORTED', 1, pi, logging)])]
    e0 = {pa: (True, None) for pa, _ in pas}
    common = dict(large=10, mei=1e-3, ess_min={'pa': 1, 'game': 1}, draws=50, seed=1)
    inside = {'1:1:1': law({0: .75, 1: .25}), '1:2:1': law({0: .5, 1: .5})}
    finite = bw.weight_diagnostics(pas, e0, new_vectors=inside, **common)['kl_new_vs_b0']
    expected = (.75 * np.log(1.5) + .25 * np.log(.5)) / 2
    assert finite['infinite_decisions'] == 0 and abs(finite['mean'] - expected) < 1e-15
    assert finite['finite_only_mean'] == finite['mean']
    outside = {**inside, '1:2:1': law({0: .5, 1: .25, 2: .25})}
    kl = bw.weight_diagnostics(pas, e0, new_vectors=outside, **common)['kl_new_vs_b0']
    assert kl['mean'] == float('inf') and kl['infinite_decisions'] == 1 and kl['decisions'] == 2
    assert abs(kl['finite_only_mean'] - (.75 * np.log(1.5) + .25 * np.log(.5))) < 1e-15


def test_t12_sc2_mean_weight_is_one_under_the_logging_law_and_detects_a_wrong_law():
    rng = np.random.default_rng(4)
    pas, true, wrong = [], {}, {}
    for i in range(4000):
        support = rng.random(V) < .4
        support[rng.integers(V)] = True
        b = np.where(support, rng.dirichlet(np.ones(V)), 0)
        b /= b.sum()
        policy_mask = support & (rng.random(V) < .7)
        policy_mask[np.argmax(b)] = True
        pi = np.where(policy_mask, b ** 2, 0)
        pi /= pi.sum()
        a = int(rng.choice(V, p=b))
        status = 'SUPPORTED' if policy_mask[a] else 'OUTSIDE_POLICY_SUPPORT'
        pa = f'{i % 400}:{i}'
        r = row(pa, 0, status, a, pi, b)
        pas.append((pa, [r]))
        true[r['request_id']] = b
        flat = support / support.sum()
        wrong[r['request_id']] = flat
    e0 = {pa: (True, None) for pa, _ in pas}
    kwargs = dict(large=10, mei=1e-3, ess_min={'pa': 1, 'game': 1}, draws=1000, seed=20260924)
    right = bw.weight_diagnostics(pas, e0, new_vectors=true, **kwargs)['policies']['candidate']
    assert right['sc2_mean_rho_ci95'][0] <= 1 <= right['sc2_mean_rho_ci95'][1]
    off = bw.weight_diagnostics(pas, e0, new_vectors=wrong, **kwargs)['policies']['candidate']
    assert off['sc2_mean_rho_ci95'][0] > 1 or off['sc2_mean_rho_ci95'][1] < 1


@pytest.fixture(scope='module')
def synthetic_s6(tmp_path_factory):
    world = make_world(0)
    ledger = make_ledger(tmp_path_factory.mktemp('s6'), world, games=8)
    return world, ledger


def test_e0_r1b_and_r2_on_a_runtime_written_ledger(synthetic_s6):
    world, ledger = synthetic_s6
    header, decisions, _ = bw.read_ledger(ledger['ledger'], manifest_path=ledger['manifest'],
                                          ledger_sha256=ledger['ledger_sha256'], manifest_sha256=ledger['manifest_sha256'],
                                          runtime_sha256=ledger['runtime'].sha256)
    assert header['runtime_sha256'] == ledger['runtime'].sha256
    pas = bw.by_pa(decisions)
    support, _ = load_support_table(ledger['support_path'], ledger['support_sha256'], world['artifact'])
    hands, _ = load_hand_registry(ledger['hands_path'], ledger['hands_sha256'], world['artifact'])
    e0 = bw.e0_from_ledger(pas, world['artifact'].bc, support, hands)
    assert {pa: e0[pa][0] for pa, _ in pas} == {pa: f['in_population'] for pa, f in ledger['facts'].items()}
    assert any(not v[0] and v[1] == 'UNSUPPORTED_UNKNOWN_PITCHER' for v in e0.values())
    assert any(not v[0] and str(v[1]).startswith('UNSUPPORTED_PITCHER_HAND') for v in e0.values())
    statuses = {d['status'] for d in decisions}
    assert {'SUPPORTED', 'UNSUPPORTED_LOGGING_POSITIVITY'} <= statuses
    r1b = bw.r1b_logging(decisions, world['b0'].vector)
    assert r1b['rows'] > 0 and r1b['max_abs_diff'] <= 1e-12
    expected = estimator_ess(ledger)
    r2 = bw.r2_reproduction(pas, e0, expected, 1e-9)
    assert r2['pass'], r2
    for policy in ('candidate', 'reference'):
        for level in ('pa', 'game'):
            assert abs(r2['actual'][policy][level] - expected[policy][level]) <= 1e-12 * expected[policy][level]
    nudged = {p: {k: v * (1 + 1e-6) for k, v in levels.items()} for p, levels in expected.items()}
    assert not bw.r2_reproduction(pas, e0, nudged, 1e-9)['pass']  # the gate trips on a perturbed aggregate
    tampered = [dict(d) for d in decisions]
    target = next(i for i, d in enumerate(tampered) if d['status'] == 'SUPPORTED')
    logging = list(tampered[target]['result']['logging'])
    a = tampered[target]['result']['logged_index']
    logging[a] *= 1.01
    tampered[target] = {**tampered[target], 'result': {**tampered[target]['result'], 'logging': logging}}
    assert bw.r1b_logging(tampered, world['b0'].vector)['max_abs_diff'] > 1e-12
    with pytest.raises(IntegrityError):
        bw.r2_reproduction(bw.by_pa(tampered), e0, expected, 1e-9)
