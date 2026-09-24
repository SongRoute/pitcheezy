"""Synthetic T2 family, metadata and calibration tests; no model training."""
from pathlib import Path
import sys
from copy import deepcopy
import numpy as np
import pandas as pd
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'scripts'), str(Path(__file__).parent)]
import score_ml_generalization as scorer
from run_ml_generalization import fold_metadata, _metadata_values
from run_ml_matrix import artifact_hashes
from run_ml_benchmark import dump
from pitchmdp.data import KEY, hash_file
from pitchmdp.matrix_panel import select_panel
from test_matrix_panel import training


def test_metadata_uses_actual_train_and_frozen_selection_thresholds():
    train, source = training()
    panel = select_panel(train, starter_source=source)
    frame = train.iloc[[0, -1]].copy()
    actual = train.loc[train.pitcher.ne(frame.iloc[0].pitcher)].copy()
    actual = actual.groupby('pitcher', group_keys=False).head(1)
    actual['batter'] = 123
    result = fold_metadata(frame, panel, actual)
    assert result.train_pitches.tolist() == [0, 1]
    assert result.train_volume.tolist() == ['zero', 'low']
    assert result.seen_pitcher.tolist() == [False, True]
    assert result.seen_batter.tolist() == [False, False]
    assert result.selection_seen_batter.all()
    assert result.selection_train_pitches.gt(result.train_pitches).all()
    assert result.train_role.iloc[0] == 'unseen'
    assert result[KEY].equals(frame[KEY])


def probabilities(n, truth=.2):
    result = np.full((n, 10), (1-truth)/9)
    result[:, 0] = truth
    return result


def test_june_blends_fit_only_once_and_reused_in_all_regimes(monkeypatch):
    calls = []
    def fake_blend(y, model, baseline, metric):
        calls.append(model.copy())
        return {'model_weight': [.6, .1, .3, .9][len(calls)-1]}
    monkeypatch.setattr(scorer, 'fit_blend', fake_blend)
    baseline = {'blend_y': np.zeros(2, dtype=int), 'blend': probabilities(2, .1),
                'dev_y': np.zeros(3, dtype=int), 'dev': probabilities(3, .1)}
    members = [{'blend': probabilities(2, .2 + i*.02),
                **{r + suffix: probabilities(3, .2 + j*.1 + i*.01)
                   for j, r in enumerate(scorer.REGIMES) for suffix in ('', '_raw')}} for i in range(3)]
    reports, values = scorer.summarize_regimes(members, baseline)
    assert len(calls) == 4
    assert [r['model_weight'] for r in reports['seed_selections']] == [.1, .3, .9]
    for regime in scorer.REGIMES:
        np.testing.assert_allclose(values[regime]['primary'], .6*np.mean([m[regime] for m in members], axis=0)+.4*baseline['dev'])
        for i, w in enumerate((.1, .3, .9)):
            np.testing.assert_allclose(values[regime]['seed_primary'][i], w*members[i][regime]+(1-w)*baseline['dev'])


def test_six_slots_include_missing_and_pitcher_is_adaptation_only():
    comparisons = [scorer.contrast(test) for test in scorer.TESTS]
    paired = {'status': 'measured', 'nll': {'delta': -.01, 'ci95': [-.02, -.001], 'p_less': .004},
              'brier': {'delta': -.002, 'ci95': [-.005, 0]}}
    for i in (0, 4):
        comparisons[i].update(paired=deepcopy(paired), seed_deltas=[-.01, -.02, .01])
    adjusted = scorer.finish_decisions(comparisons)
    assert adjusted == [.024, None, None, None, .024, None]
    assert comparisons[0]['N']['status'] == 'predictive_improvement'
    assert comparisons[4]['N']['status'] == 'history_adaptation_improvement'
    assert comparisons[4]['N']['architecture_improvement_claim'] is False
    assert comparisons[1]['N']['status'] == 'unmeasured'
    with pytest.raises(ValueError, match='six-test'):
        scorer.finish_decisions(comparisons[::-1])


def test_cohort_gate_precedes_primary_bootstrap(monkeypatch):
    monkeypatch.setattr(scorer, 'paired_game_comparison', lambda *a, **k: pytest.fail('Below-gate primary bootstrap'))
    values = {'primary': probabilities(499), 'seed_primary': np.stack([probabilities(499)]*3)}
    result = scorer.contrast(scorer.TESTS[0], np.zeros(499, dtype=int), values, values, np.arange(499))
    assert result['paired'] is None
    assert result['seed_deltas'] is None


def test_family_gate_precedes_any_archive_read(tmp_path):
    prep = {'folds': {'pitcher': {'active': False}, 'batter': {'active': True,
        'cells': ['G1-personal', 'G0-global'], 'units': {'global': {'mode': 'global'}}}}}
    with pytest.raises(ValueError, match='complete family'):
        scorer.load_family(tmp_path, prep, {'candidate': 'G1-personal', 'control': 'G0-global'})


def fixture_family(root):
    train, source = training()
    panel = select_panel(train, starter_source=source)
    frame = train.iloc[:2].copy()
    metadata = fold_metadata(frame, panel, train)
    keys = frame[KEY].to_numpy(np.int64)
    vals = {}
    for split in ('blend', 'dev'):
        vals.update({split: probabilities(2), split+'_raw': probabilities(2), split+'_keys': keys,
                     split+'_y': np.zeros(2, dtype=int), split+'_game_pk': frame.game_pk.to_numpy(),
                     split+'_pitcher': frame.pitcher.to_numpy(), split+'_batter': frame.batter.to_numpy()})
    spec = {'active': True, 'cells': ['G1-personal', 'G0-global'], 'units': {'global': {'mode': 'global'}},
            'samples': {s: {'path': s+'_keys.parquet'} for s in ('blend', 'dev')}}
    prep = {'folds': {'pitcher': {'active': False}, 'batter': spec}}
    config = {'candidate': 'G1-personal', 'control': 'G0-global'}
    dump(root/'preparation.json', prep)
    ph = hash_file(root/'preparation.json')
    directory = root/'batter'; directory.mkdir()
    np.savez(directory/'baseline_predictions.npz', **vals)
    metadata.to_parquet(directory/'dev_metadata.parquet', index=False)
    for split in ('blend', 'dev'): frame[KEY].to_parquet(directory/(split+'_keys.parquet'), index=False)
    for seed in scorer.SEEDS:
        folder = directory/'fits'/f'seed{seed}'/'global'; folder.mkdir(parents=True)
        (folder/'model.pt').write_bytes(b'synthetic checkpoint')
        dump(folder/'fit.json', {'seconds_total': 0})
        dump(folder/'state.json', {'identity': {'preparation_sha256': ph, 'axis': 'batter', 'seed': seed,
            'unit': 'global', 'spec': spec['units']['global']}, 'artifact_hashes': artifact_hashes(folder, ['model.pt', 'fit.json'])})
        deps = {str(folder/name): hash_file(folder/name) for name in ('state.json', 'model.pt', 'fit.json')}
        for cell in spec['cells']:
            member = directory/'members'/cell/f'seed{seed}'; member.mkdir(parents=True)
            payload = {**vals, **{r+suffix: probabilities(2) for r in scorer.REGIMES for suffix in ('', '_raw')},
                       **{r+'_delivery_level': np.zeros(2, dtype=int) for r in scorer.REGIMES}}
            np.savez(member/'predictions.npz', **payload)
            dump(member/'runtime.json', {})
            dump(member/'calibration.json', {})
            dump(member/'prediction_state.json', {'identity': {'preparation_sha256': ph, 'axis': 'batter', 'cell': cell, 'seed': seed},
                'dependencies': deps, 'artifact_hashes': artifact_hashes(member, ['predictions.npz', 'runtime.json', 'calibration.json'])})
    natural = {name: vals['dev_'+name] for name in ('keys', 'y', 'game_pk', 'pitcher', 'batter')}
    natural['frequency'] = probabilities(2)
    for cell in spec['cells']:
        natural.update({cell+'_'+kind: probabilities(2) for kind in ('primary', 'calibrated', 'raw')})
        natural[cell+'_seed_primary'] = np.stack([probabilities(2)]*3)
    np.savez(root/'natural_matchup_predictions.npz', **natural)
    metadata.to_parquet(root/'natural_matchup_metadata.parquet', index=False)
    dump(root/'natural_matchup_audit.json', {})
    return prep, config


def test_all_artifacts_validated_then_wrong_seed_and_dependencies_rejected(tmp_path):
    prep, config = fixture_family(tmp_path)
    datasets, natural, metadata, inputs, costs = scorer.load_family(tmp_path, prep, config)
    assert list(datasets) == ['batter']
    assert len(datasets['batter']['members']['G1-personal']) == 3
    path = tmp_path/'batter/members/G1-personal/seed2/prediction_state.json'
    state = scorer.read_json(path)
    state['identity']['seed'] = 1
    dump(path, state)
    with pytest.raises(ValueError, match='identity or exact checkpoint'):
        scorer.load_family(tmp_path, prep, config)
    state['identity']['seed'] = 2
    state['dependencies'] = {}
    dump(path, state)
    with pytest.raises(ValueError, match='identity or exact checkpoint'):
        scorer.load_family(tmp_path, prep, config)


def test_metadata_mismatches_fail_and_guardrail_family_is_fixed(monkeypatch):
    train, source = training(); panel = select_panel(train, starter_source=source)
    frame = train.iloc[:2]; metadata = fold_metadata(frame, panel, train)
    vals = {name: frame[name].to_numpy() for name in ('pitcher', 'batter', 'game_pk')}
    vals['keys'] = frame[KEY].to_numpy()
    scorer.validate_metadata(metadata, vals, '')
    bad = metadata.copy(); bad['seen_pitcher'] = False
    with pytest.raises(ValueError, match='counts and seen'):
        scorer.validate_metadata(bad, vals, '')
    called = []
    monkeypatch.setattr(scorer, 'guardrails', lambda *a, **k: called.append(k) or {})
    n = 2; p = {'primary': probabilities(n), 'seed_primary': np.stack([probabilities(n)]*3)}
    scorer.contrast(scorer.TESTS[0], np.zeros(n, dtype=int), p, p, np.arange(n), metadata, robustness=True)
    assert called == [{'candidate_family_size': 4, 'draws': 100000}]
