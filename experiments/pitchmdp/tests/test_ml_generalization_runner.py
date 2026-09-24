"""T2 runner contracts on tiny synthetic CPU data; no neural fitting or DEV scoring."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'scripts'), str(Path(__file__).parent)]
from test_matrix_generalization import observations
from pitchmdp.data import hash_file
from pitchmdp.matrix_sharing import SharingContext, SharingPredictor
from pitchmdp.model import outcome_labels
import run_ml_generalization as runner


def config(candidate='G3-cluster'):
    return {'protocol': 'ml_generalization_v1', 'experiment_id': 'EXP-TINY',
        'parent_run': '/frozen/g', 'parent_preparation_sha256': 'a'*64, 'parent_analysis_sha256': 'b'*64,
        'candidate': candidate, 'control': runner.CONTROLS[candidate], 'selection_basis': 'frozen G selection',
        'scope': 'Cpanel', 'seeds': [0, 1, 2], 'axes': ['pitcher', 'batter'], 'regimes': ['Z', 'W', 'O'],
        'prefix_games': 2, 'selector_seed': 20260924, 'draws': 400, 'width': 128,
        'budget': deepcopy(runner.BUDGET), 'device': 'cpu', 'adaptation': 'fixed_prefix_context_only_v1'}


def tiny_fold(axis='pitcher', candidate='G3-cluster', frame=None):
    frame = observations() if frame is None else frame
    parts = {name: frame.loc[frame.split.eq(name)] for name in ('train', 'earlystop', 'temperature', 'blend', 'dev')}
    panel = {'pitcher_ids': [10, 20, 30, 40, 50, 99]}
    selected = {'ids': [10 if axis == 'pitcher' else 100], 'axis': axis}
    return runner.build_fold(config(candidate), frame, axis, selected, parts, panel)


def test_exact_config_and_selected_g1_mapping():
    for candidate in runner.CONTROLS:
        assert runner.config_check(config(candidate))['candidate'] == candidate
    for key, value in [('seeds', [0]), ('draws', 25), ('prefix_games', 3), ('scope', 'MLB'),
                       ('regimes', ['W', 'O']), ('control', 'G0-global'), ('adaptation', 'centroid_update')]:
        changed = config(); changed[key] = value
        with pytest.raises(ValueError): runner.config_check(changed)
    changed = config(); changed['budget']['epochs'] = 1
    with pytest.raises(ValueError): runner.config_check(changed)
    changed = config(); changed['silent_tuning'] = True
    with pytest.raises(ValueError): runner.config_check(changed)


def test_dependency_unions_include_global_and_only_eligible_experts():
    train = pd.DataFrame({'pitcher': np.repeat([1, 2, 3, 4], 500)})
    early = pd.DataFrame({'pitcher': np.repeat([1, 2, 3, 4], [20, 19, 20, 20])})
    clusters = {'pitcher_cluster': {str(pid): pid-1 for pid in range(1, 5)}}
    for candidate, modes in [('G1-personal', {'global', 'personal'}),
                             ('G2-feature', {'global', 'feature'}),
                             ('G3-cluster', {'global', 'feature', 'cluster'}),
                             ('G4-partial', {'global', 'feature', 'cluster', 'personal'})]:
        units, eligibility = runner.plan_units([candidate, runner.CONTROLS[candidate]], train, early, clusters, [1, 2, 3, 4])
        assert {spec['mode'] for spec in units.values()} == modes
        assert 'global' in units
        assert 'personal2' not in units and 'cluster1' not in units
        if eligibility: assert any(item['fallback'] is not None for item in eligibility)


def test_fold_refits_all_aux_on_exclusions_and_preserves_original_truth():
    for axis in runner.AXES:
        fold, parts, aux, report = tiny_fold(axis)
        assert not parts['train'][axis].isin(fold.ids).any()
        assert not parts['temperature'][axis].isin(fold.ids).any()
        assert report['coverage']['eligible_pitches'] == 4
        assert aux['normalizer'].n_train == int(fold.allowed_fit['train'].sum())
        assert aux['delivery'].draws == 400
        assert report['cells'] == (['G0-global'] if axis == 'pitcher' else ['G3-cluster', 'G2-feature'])
        assert (outcome_labels(parts['dev']) >= 0).all()
        changed = observations()
        changed.loc[fold.heldout, 'description'] = 'hit_by_pitch'
        changed.loc[fold.heldout, 'events'] = 'home_run'
        changed.loc[fold.heldout, 'effective_speed'] = 1000.
        _, other_parts, other_aux, other_report = tiny_fold(axis, frame=changed)
        assert aux['normalizer'].report() == other_aux['normalizer'].report()
        assert aux['context'].report() == other_aux['context'].report()
        assert report['clusters'] == other_report['clusters']
        pd.testing.assert_frame_equal(parts['train'], other_parts['train'])
        np.testing.assert_array_equal(aux['delivery'].fallback, other_aux['delivery'].fallback)
        np.testing.assert_array_equal(aux['baseline'].predict(parts['blend']), other_aux['baseline'].predict(other_parts['blend']))


def test_smaller_parent_train_pool_is_rejected_instead_of_extra_aux_data():
    frame = observations()
    parts = {name: frame.loc[frame.split.eq(name)] for name in ('train', 'earlystop', 'temperature', 'blend', 'dev')}
    parts['train'] = parts['train'].loc[~parts['train'].game_pk.eq(6)]
    with pytest.raises(ValueError, match='D100'):
        runner.build_fold(config(), frame, 'pitcher', {'ids': [10]}, parts, {'pitcher_ids': [10, 20]})


def test_prepare_identity_and_external_dependencies_are_immutable(tmp_path):
    expected = {'source': 'frozen'}
    owned = tmp_path / 'aux.pkl'; owned.write_bytes(b'synthetic aux')
    parent = tmp_path / 'parent.json'; parent.write_text('{}')
    runner.dump(tmp_path / 'preparation.json', {'identity': expected,
        'artifact_hashes': {'aux.pkl': hash_file(owned)}, 'external_hashes': {str(parent): hash_file(parent)}})
    runner.verify(tmp_path, expected)
    parent.write_text('{"changed":true}')
    with pytest.raises(ValueError, match='selection dependency'): runner.verify(tmp_path, expected)


def test_profile_gate_requires_matching_parent_and_successful_budget(tmp_path):
    runner.dump(tmp_path / 'preparation.json', {'frozen': True})
    destination = tmp_path / 'pitcher/profile'; destination.mkdir(parents=True)
    runner.dump(destination / 'profile.json', {'resource_gate': False})
    runner.dump(destination / 'state.json', {
        'identity': {'preparation_sha256': hash_file(tmp_path / 'preparation.json'), 'axis': 'pitcher'},
        'artifact_hashes': {'profile.json': hash_file(destination / 'profile.json')}})
    with pytest.raises(ValueError, match='budget'): runner._profile_gate(tmp_path, 'pitcher')
    runner.dump(destination / 'profile.json', {'resource_gate': True})
    runner.dump(destination / 'state.json', {
        'identity': {'preparation_sha256': hash_file(tmp_path / 'preparation.json'), 'axis': 'pitcher'},
        'artifact_hashes': {'profile.json': hash_file(destination / 'profile.json')}})
    runner._profile_gate(tmp_path, 'pitcher')
    runner.dump(tmp_path / 'preparation.json', {'changed': True})
    with pytest.raises(ValueError, match='Matching'): runner._profile_gate(tmp_path, 'pitcher')


def test_prediction_archives_three_regimes_common_calibration_and_dependency_resume(tmp_path, monkeypatch):
    fold, parts, aux, report = tiny_fold()
    prepared = {'folds': {'pitcher': report}, 'identity': {'source_hashes': {}}}
    runner.dump(tmp_path / 'preparation.json', prepared)
    model_path = tmp_path / 'frozen_model.bin'; model_path.write_bytes(b'fixed weights')
    store = fold.reconstruct('Z').history_store(normalizer=aux['normalizer'], type_vocabulary=report['type_vocabulary'])
    context = SharingContext(aux['context'], report['clusters'])
    class Model:
        def logits(self, arrays):
            tokens, valid, context_values = arrays
            result = np.zeros((len(tokens), 10), dtype=np.float64)
            result[:, 0] = valid.sum(1) + context_values[:, 11] / 10.
            return result
    predictor = SharingPredictor('G0-global', Model(), report['clusters'])
    monkeypatch.setattr(runner, '_profile_gate', lambda *args: None)
    monkeypatch.setattr(runner, 'source_hashes', lambda: {})
    monkeypatch.setattr(runner, 'load_fold', lambda *args: (fold, store, context, parts, aux))
    monkeypatch.setattr(runner, '_predictor', lambda *args: (predictor, {str(model_path): hash_file(model_path)}))
    runner.predict(config(), {}, tmp_path, prepared, 'pitcher', 'G0-global', 0)
    destination = tmp_path / 'pitcher/members/G0-global/seed0'
    with np.load(destination / 'predictions.npz') as archive:
        np.testing.assert_array_equal(archive['dev_y'], fold.truth_labels[parts['dev'].index])
        assert archive['Z'].shape == archive['W'].shape == archive['O'].shape == (4, 10)
        assert not np.array_equal(archive['Z'], archive['W'])
        np.testing.assert_array_equal(archive['Z_delivery_level'], archive['O_delivery_level'])
    assert runner.read_json(destination / 'runtime.json')['calibration_shared_across_regimes'] is True
    monkeypatch.setattr(runner, 'load_fold', lambda *args: pytest.fail('Completed prediction must resume without data load'))
    runner.predict(config(), {}, tmp_path, prepared, 'pitcher', 'G0-global', 0)
    model_path.write_bytes(b'changed checkpoint')
    with pytest.raises(ValueError, match='checkpoint changed'):
        runner.predict(config(), {}, tmp_path, prepared, 'pitcher', 'G0-global', 0)
