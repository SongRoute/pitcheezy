"""Synthetic checks for scripts/diagnose_policy_stability_aggregate.py (no real data, no model)."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'scripts')]
import diagnose_policy_stability_aggregate as agg  # noqa: E402

IDENTITY = {'sealed_path': '/sealed/S3b', 'sealed_manifest_sha256': 'm' * 64, 'sealed_final_identity_sha256': 'f' * 64,
            'sealed_ledger': 'ledger-blend-0.jsonl', 'sealed_evaluated_decisions': 400, 'tau': .1, 'pitch_cap': 12,
            'sealed_seed': 0, 'sealed_samples': 6}
RESULTS_BASE = {'smoke': False, 'outcomes_read': False, 'policy_identity_registered': False,
                'settings': {'tau': .1, 'samples': 6, 'pitch_cap': 12, 'seed': 0},
                'sealed_g0_run': {'path': '/sealed/S3b', 'manifest_sha256': 'm' * 64, 'final_identity_sha256': 'f' * 64,
                                  'ledger': 'ledger-blend-0.jsonl', 'requests': 430, 'evaluated_decisions': 400}}


def laws(rng, n, width=10):
    p = rng.random((n, width))
    return p / p.sum(axis=1, keepdims=True)


def make_variant(label, rng, n=400, kind='seed', seed=1, samples=6, with_law=True, ids=None, games=None, **kw):
    ids = np.array([f'r{i:05d}' for i in range(n)]) if ids is None else ids
    games = (np.arange(n) // 20) if games is None else games
    law = laws(rng, n) if with_law else None
    return agg.Variant(label=label, kind=kind, source='SYN', seed=seed, samples=samples, request_id=ids, game_pk=games,
                       tv=rng.uniform(0, .2, n), top1_same=rng.random(n) < .9, identity=dict(IDENTITY), law=law,
                       top1=None if law is None else law.argmax(axis=1), **kw)


def test_summaries_are_the_plain_means_with_game_bootstrap_intervals():
    rng = np.random.default_rng(0)
    v = make_variant('seed1', rng)
    s = agg.summarize_variant(v, v.game_pk, draws=300, seed=1)
    assert s['mean_tv_vs_sealed'] == pytest.approx(v.tv.mean()) and s['top1_agreement_vs_sealed'] == pytest.approx(v.top1_same.mean())
    assert s['mean_tv_ci95'][0] <= s['mean_tv_vs_sealed'] <= s['mean_tv_ci95'][1]
    assert s['top1_agreement_ci95'][0] <= s['top1_agreement_vs_sealed'] <= s['top1_agreement_ci95'][1]
    assert s['games'] == 20 and s['decisions'] == 400 and s['share_tv_above_0.05'] == pytest.approx((v.tv > .05).mean())
    (ci,), games = agg.game_bootstrap([np.ones(400)], v.game_pk, draws=50, seed=0)
    assert ci == [1., 1.] and games == 20


def test_cross_variant_pairwise_tv_and_top1_votes():
    rng = np.random.default_rng(1)
    a, b = make_variant('a', rng), make_variant('b', rng, seed=2)
    same = make_variant('a-copy', rng)
    same.law, same.top1 = a.law.copy(), a.top1.copy()
    sealed = agg.Sealed(request_id=a.request_id.copy(), law=a.law.copy(), top1=a.top1.copy(), mask=np.ones((400, 10), bool))
    out = agg.cross_variant([a, same, b], sealed)
    assert out['pairwise_mean_tv']['a|a-copy'] == 0
    expected = .5 * np.abs(a.law - b.law).sum(axis=1).mean()
    assert out['pairwise_mean_tv']['a|b'] == pytest.approx(expected) and 'b|a' not in out['pairwise_mean_tv']
    assert out['members'] == ['sealed', 'a', 'a-copy', 'b']
    unanimous = (a.top1 == b.top1).mean()      # sealed, a and a-copy always agree; b decides unanimity
    assert out['top1_unanimous_share'] == pytest.approx(unanimous)
    assert out['top1_modal_agreement_mean'] == pytest.approx(np.where(a.top1 == b.top1, 1., .75).mean())
    swap = make_variant('swap', rng, kind='predictor_replacement', seed=0)
    assert agg.cross_variant([swap], None)['status'] == 'unmeasured'   # replacement rows never enter the cross-seed statistic
    legacy = make_variant('legacy', rng, with_law=False)
    assert agg.cross_variant([legacy], None)['status'] == 'unmeasured'


def test_aggregate_separates_seed_samples_and_predictor_replacement():
    rng = np.random.default_rng(2)
    seed2 = make_variant('seed2-s6', rng, seed=2)
    s12 = make_variant('seed0-s12', rng, kind='samples', seed=0, samples=12, shares_prefix_draws=True)
    swap = make_variant('P17', rng, kind='predictor_replacement', seed=0, with_law=False,
                        registered_verdict='EXP-P17-001: DIFFERENT_RECOMMENDATIONS (registered, not recomputed)')
    sealed = agg.Sealed(request_id=seed2.request_id.copy(), law=laws(rng, 400), top1=np.zeros(400, int),
                        mask=np.array([[True] * 10] * 390 + [[True] + [False] * 9] * 10))
    report = agg.aggregate([seed2, s12, swap], sealed, draws=100, seed=3)
    assert [r['label'] for r in report['fixed_predictor_seed_variability']] == ['seed2-s6']
    assert [r['label'] for r in report['sample_count_variants']] == ['seed0-s12']
    assert report['sample_count_variants'][0]['shares_prefix_draws'] is True
    assert [r['label'] for r in report['predictor_replacement']] == ['P17']
    assert report['registered_verdicts_unchanged'] == ['EXP-P17-001: DIFFERENT_RECOMMENDATIONS (registered, not recomputed)']
    assert report['samples_trend'] == {'6': [{'label': 'seed2-s6', 'seed': 2, 'mean_tv_vs_sealed': pytest.approx(seed2.tv.mean()),
                                              'shares_prefix_draws': False}],
                                       '12': [{'label': 'seed0-s12', 'seed': 0, 'mean_tv_vs_sealed': pytest.approx(s12.tv.mean()),
                                               'shares_prefix_draws': True}]}
    cover = report['support_coverage']
    assert cover['coverage_share'] == 1. and cover['supported_actions']['share_decisions_with_single_action'] == pytest.approx(.025)
    assert report['cross_variant']['members'] == ['sealed', 'seed2-s6', 'seed0-s12']
    assert report['outcomes_read'] is False and 'policy utility or improvement' in report['claims_excluded']
    assert report['identity'] == IDENTITY
    json.dumps(report, allow_nan=False)   # serialisable, no NaN


def test_alignment_reorders_by_request_id_and_refuses_set_differences():
    rng = np.random.default_rng(3)
    a = make_variant('a', rng)
    perm = rng.permutation(400)
    b = make_variant('b', rng, ids=a.request_id[perm], games=a.game_pk[perm])
    b.tv = a.tv[perm].copy()           # same numbers, shuffled: after alignment they must coincide
    ref, games, aligned, _ = agg.align([a, b])
    assert np.array_equal(ref, a.request_id) and np.array_equal(aligned[1].tv, a.tv)
    assert np.array_equal(aligned[1].law, b.law[np.argsort(perm)]) and np.array_equal(games, a.game_pk)
    missing = make_variant('missing', rng, n=399)
    with pytest.raises(agg.AggregationError, match='missing 1, extra 0'):
        agg.align([a, missing])
    extra_ids = np.append(a.request_id, 'r99999')
    extra = make_variant('extra', rng, n=401, ids=extra_ids, games=np.append(a.game_pk, 0))
    with pytest.raises(agg.AggregationError, match='missing 0, extra 1'):
        agg.align([a, extra])
    dup_ids = a.request_id.copy()
    dup_ids[1] = dup_ids[0]
    with pytest.raises(agg.AggregationError, match='duplicate request ids'):
        agg.align([make_variant('dup', rng, ids=dup_ids)])
    other_games = make_variant('games', rng, games=a.game_pk[::-1].copy())
    with pytest.raises(agg.AggregationError, match='game_pk differs'):
        agg.align([a, other_games])
    sealed_short = agg.Sealed(request_id=a.request_id[:-1], law=a.law[:-1], top1=a.top1[:-1])
    with pytest.raises(agg.AggregationError, match='sealed: decision set differs'):
        agg.align([a], sealed_short)


def test_identity_checks_refuse_missing_outcome_flags_and_disagreeing_sealed_sides():
    ok = agg.identity_of(dict(RESULTS_BASE))
    assert ok == IDENTITY
    for key in ('outcomes_read', 'policy_identity_registered', 'smoke'):
        absent = copy.deepcopy(RESULTS_BASE)
        del absent[key]
        with pytest.raises(agg.AggregationError, match='missing'):
            agg.identity_of(absent)
        flagged = copy.deepcopy(RESULTS_BASE)
        flagged[key] = True
        with pytest.raises(agg.AggregationError, match='refused'):
            agg.identity_of(flagged)
    incomplete = copy.deepcopy(RESULTS_BASE)
    del incomplete['sealed_g0_run']['manifest_sha256']
    with pytest.raises(agg.AggregationError, match='sealed_manifest_sha256'):
        agg.identity_of(incomplete)
    rng = np.random.default_rng(4)
    a, b = make_variant('a', rng), make_variant('b', rng)
    b.identity = {**IDENTITY, 'sealed_manifest_sha256': 'x' * 64, 'tau': .2}
    with pytest.raises(agg.AggregationError, match=r"disagrees between a and b on \['sealed_manifest_sha256', 'tau'\]"):
        agg.check_identities([a, b])
    with pytest.raises(agg.AggregationError, match='Duplicate variant labels'):
        agg.check_identities([a, make_variant('a', rng)])
    with pytest.raises(agg.AggregationError, match='unknown kind'):
        make_variant('k', rng, kind='bogus')
    with pytest.raises(agg.AggregationError, match='TV outside'):
        bad = make_variant('tv', rng)
        bad.tv[0] = 1.5
        agg.Variant(**{**bad.__dict__})


def write_legacy_run(root, experiment, extra, ids, games, rng, verdict_key, status):
    root.mkdir()
    results = copy.deepcopy(RESULTS_BASE)
    results.update({'experiment_id': experiment, **extra, 'comparison': {verdict_key: {'status': status}}})
    (root / 'results.json').write_text(json.dumps(results))
    np.savez_compressed(root / 'decisions.npz', request_id=ids, game_pk=games, tv=rng.uniform(0, .1, len(ids)),
                        top1_same=rng.random(len(ids)) < .9)


def test_loaders_read_legacy_p17_p18_and_seed_stability_layouts(tmp_path):
    rng = np.random.default_rng(5)
    n = 60
    ids, games = np.array([f'r{i}' for i in range(n)]), np.repeat(np.arange(3), 20)
    write_legacy_run(tmp_path / 'P18', 'EXP-P18-001', {'noise_seed': 1}, ids, games, rng, 'verdict', 'WITHIN_SEARCH_NOISE')
    write_legacy_run(tmp_path / 'P17', 'EXP-P17-001', {'mix': {'direct_weight': .5}}, ids[::-1], games[::-1], rng,
                     'stop_rule', 'DIFFERENT_RECOMMENDATIONS')
    (p18,), none, boot = agg.load_run(tmp_path / 'P18')
    assert none is None and boot is None and p18.kind == 'seed' and p18.seed == 1 and p18.samples == 6 and p18.law is None
    assert p18.registered_verdict == 'EXP-P18-001: WITHIN_SEARCH_NOISE (registered, not recomputed)'
    (p17,), _, _ = agg.load_run(tmp_path / 'P17')
    assert p17.kind == 'predictor_replacement' and p17.seed == 0 and p17.label == 'EXP-P17-001-predictor-swap'
    new = tmp_path / 'NEW'
    new.mkdir()
    results = copy.deepcopy(RESULTS_BASE)
    results.update({'experiment_id': 'EXP-P18-00X', 'bootstrap': {'draws': 10000, 'seed': 20260924, 'unit': 'whole game'}, 'variants': {
        'seed2-s6': {'kind': 'seed', 'seed': 2, 'samples': 6, 'seed_basis': 'planning_v2', 'decisions_file': 'decisions-seed2-s6.npz'},
        'seed0-s12': {'kind': 'samples', 'seed': 0, 'samples': 12, 'shares_prefix_draws': True, 'decisions_file': 'decisions-seed0-s12.npz'}}})
    (new / 'results.json').write_text(json.dumps(results))
    sealed_law = laws(rng, n)
    np.savez_compressed(new / 'sealed.npz', request_id=ids, law=sealed_law, top1=sealed_law.argmax(1), mask=np.ones((n, 10), bool))
    for label in ('seed2-s6', 'seed0-s12'):
        law = laws(rng, n)
        np.savez_compressed(new / f'decisions-{label}.npz', request_id=ids, game_pk=games, tv=.5 * np.abs(law - sealed_law).sum(1),
                            top1_same=law.argmax(1) == sealed_law.argmax(1), law=law, top1=law.argmax(1))
    variants, sealed, boot = agg.load_run(new)
    assert boot == {'draws': 10000, 'seed': 20260924}
    assert [v.label for v in variants] == ['seed2-s6', 'seed0-s12'] and variants[1].shares_prefix_draws is True
    assert variants[0].extra == {'seed_basis': 'planning_v2'} and sealed.mask.shape == (n, 10)
    report = agg.aggregate([*variants, p18, p17], sealed, draws=50, seed=0)   # P17 arrives reversed and is realigned
    assert report['alignment']['variants'] == ['seed2-s6', 'seed0-s12', 'EXP-P18-001-seed1', 'EXP-P17-001-predictor-swap']
    assert len(report['registered_verdicts_unchanged']) == 2 and report['support_coverage']['coverage_share'] == pytest.approx(60 / 400)
    broken = tmp_path / 'BROKEN'
    broken.mkdir()
    (broken / 'results.json').write_text(json.dumps({**RESULTS_BASE, 'experiment_id': 'X'}))
    np.savez_compressed(broken / 'decisions.npz', request_id=ids, game_pk=games, tv=np.zeros(n), top1_same=np.ones(n, bool))
    with pytest.raises(agg.AggregationError, match='unrecognised'):
        agg.load_run(broken)
    short = tmp_path / 'SHORT'
    short.mkdir()
    (short / 'results.json').write_text(json.dumps({**RESULTS_BASE, 'experiment_id': 'EXP-P18-001', 'noise_seed': 1}))
    np.savez_compressed(short / 'decisions.npz', request_id=ids, game_pk=games)
    with pytest.raises(agg.AggregationError, match=r"missing arrays \['tv', 'top1_same'\]"):
        agg.load_run(short)


def test_bootstrap_setting_prefers_cli_then_registered_then_default():
    assert agg.resolve_bootstrap([None, None]) == {'draws': 2000, 'seed': 20260924, 'source': 'default'}
    reg = {'draws': 10000, 'seed': 7}
    assert agg.resolve_bootstrap([None, reg, reg]) == {'draws': 10000, 'seed': 7, 'source': 'registered'}
    assert agg.resolve_bootstrap([reg], draws=50) == {'draws': 50, 'seed': 7, 'source': 'cli'}
    with pytest.raises(agg.AggregationError, match='different bootstrap settings'):
        agg.resolve_bootstrap([reg, {'draws': 10000, 'seed': 8}])
    assert agg.registered_bootstrap({'bootstrap': {'draws': 3, 'seed': 4, 'unit': 'whole game'}}) == {'draws': 3, 'seed': 4}
    assert agg.registered_bootstrap({}) is None
    with pytest.raises(agg.AggregationError, match='bootstrap block malformed'):
        agg.registered_bootstrap({'bootstrap': {'draws': 2.5, 'seed': 1}})


def test_cli_writes_report_and_refuses_overwrite(tmp_path, monkeypatch, capsys):
    rng = np.random.default_rng(6)
    ids, games = np.array([f'r{i}' for i in range(40)]), np.repeat(np.arange(2), 20)
    write_legacy_run(tmp_path / 'P18', 'EXP-P18-001', {'noise_seed': 1}, ids, games, rng, 'verdict', 'WITHIN_SEARCH_NOISE')
    out = tmp_path / 'report.json'
    monkeypatch.setattr(sys, 'argv', ['agg', '--run', str(tmp_path / 'P18'), '--output', str(out), '--draws', '20'])
    agg.main()
    report = json.loads(out.read_text())
    assert report['alignment']['decisions'] == 40 and 'STABILITY_AGGREGATE_COMPLETE' in capsys.readouterr().out
    assert report['inputs'] == [str((tmp_path / 'P18').resolve())] and report['fixed_predictor_seed_variability'][0]['seed'] == 1
    assert report['bootstrap'] == {'draws': 20, 'seed': 20260924, 'unit': 'whole game', 'source': 'cli'}
    with pytest.raises(agg.AggregationError, match='Refusing to overwrite'):
        agg.main()
    default = tmp_path / 'default.json'
    monkeypatch.setattr(sys, 'argv', ['agg', '--run', str(tmp_path / 'P18'), '--output', str(default)])
    agg.main()
    assert json.loads(default.read_text())['bootstrap']['source'] == 'default'
