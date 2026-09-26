"""Synthetic checkpoint provenance and resource gates; no model deserialization."""
from copy import deepcopy
from pathlib import Path
import sys
import numpy as np
import pytest
PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'scripts'), str(Path(__file__).parent)]
from test_matrix_confirmation import registration
from pitchmdp.matrix_confirmation import member_identity, unit_identity, validate_config
from pitchmdp.matrix_data import canonical_hash
from run_ml_benchmark import dump, read_json
from run_ml_matrix import artifact_hashes
import run_ml_confirmation as runner


def fixture_member(root, *, reused=False, seed=3):
    prep = {'units': {'global': {'mode': 'global', 'train_n': 4, 'earlystop_n': 2},
                      'cluster0': {'mode': 'cluster', 'id': 0, 'train_n': 3, 'earlystop_n': 1},
                      'feature': {'mode': 'feature', 'train_n': 4, 'earlystop_n': 2}},
            'cells': ['G2-feature', 'G3-cluster'], 'samples': {'temperature': {'n': 3}},
            'primary_comparisons': [['G3-cluster', 'G2-feature']], 'parent_preparation_sha256': 'a'*64}
    deps = {}
    for unit, spec in prep['units'].items():
        folder = root/'fits'/f'seed{seed}'/unit; folder.mkdir(parents=True)
        (folder/'model.pt').write_bytes(f'fake checkpoint {seed} {unit}'.encode())
        dump(folder/'fit.json', {'report': {'kind': 'flatten_mlp', 'seed': seed,
                    'training_rows': spec['train_n'], 'earlystop_rows': spec['earlystop_n']}})
        identity = ({'preparation_sha256': canonical_hash(prep), 'seed': seed, 'unit': unit, 'spec': spec}
                    if reused else unit_identity(prep, unit, seed))
        dump(folder/'state.json', {'identity': identity, 'artifact_hashes': artifact_hashes(folder, ['model.pt', 'fit.json'])})
        if spec['mode'] in ('global', 'cluster'):
            deps[str(folder/'model.pt')] = runner.hash_file(folder/'model.pt')
    folder = root/'members/G3-cluster'/f'seed{seed}'; folder.mkdir(parents=True)
    (folder/'predictions.npz').write_bytes(b'fake archive not opened')
    dump(folder/'prediction_runtime.json', {})
    dump(folder/'calibration.json', {'cell': 'G3-cluster', 'delivery_calibration_rows': 3, 'delivery_temperature': 1.})
    identity = ({'preparation_sha256': canonical_hash(prep), 'cell': 'G3-cluster', 'seed': seed}
                if reused else member_identity(prep, 'G3-cluster', seed))
    state = {'identity': identity, 'dependencies': deps, 'artifact_hashes': artifact_hashes(folder,
              ['predictions.npz', 'calibration.json', 'prediction_runtime.json'])}
    dump(folder/'prediction_state.json', state)
    return prep, state, folder


@pytest.mark.parametrize('reused,seed', [(False,3), (True,1)])
def test_exact_dependency_union_rejects_omission_extra_and_wrong_unit(tmp_path, reused, seed):
    prep, state, folder = fixture_member(tmp_path, reused=reused, seed=seed)
    hashes = runner.validate_prediction_dependencies(tmp_path, prep, 'G3-cluster', seed, state, reused=reused)
    assert len(hashes) == 6  # global+cluster, each state/model/report
    bad = deepcopy(state); bad['dependencies'].pop(next(iter(bad['dependencies'])))
    with pytest.raises(ValueError, match='exact routed'):
        runner.validate_prediction_dependencies(tmp_path, prep, 'G3-cluster', seed, bad, reused=reused)
    bad = deepcopy(state); extra = tmp_path/'fits'/f'seed{seed}'/'feature/model.pt'
    bad['dependencies'][str(extra)] = runner.hash_file(extra)
    with pytest.raises(ValueError, match='exact routed'):
        runner.validate_prediction_dependencies(tmp_path, prep, 'G3-cluster', seed, bad, reused=reused)
    path = tmp_path/'fits'/f'seed{seed}'/'cluster0/state.json'
    saved = read_json(path); saved['identity']['seed'] = 99; dump(path, saved)
    with pytest.raises(ValueError, match='unit training identity'):
        runner.validate_prediction_dependencies(tmp_path, prep, 'G3-cluster', seed, state, reused=reused)


def test_fit_report_seed_and_calibration_rows_are_checked(tmp_path):
    prep, state, folder = fixture_member(tmp_path)
    fit = tmp_path/'fits/seed3/global'
    report = read_json(fit/'fit.json'); report['report']['seed'] = 0; dump(fit/'fit.json', report)
    saved = read_json(fit/'state.json'); saved['artifact_hashes'] = artifact_hashes(fit, ['model.pt','fit.json']); dump(fit/'state.json', saved)
    with pytest.raises(ValueError, match='fit report differs: seed'):
        runner.validate_prediction_dependencies(tmp_path, prep, 'G3-cluster', 3, state, reused=False)
    report['report']['seed'] = 3; dump(fit/'fit.json', report)
    saved['artifact_hashes'] = artifact_hashes(fit, ['model.pt','fit.json']); dump(fit/'state.json', saved)
    calibration = read_json(folder/'calibration.json'); calibration['delivery_calibration_rows'] = 2
    dump(folder/'calibration.json', calibration)
    state['artifact_hashes'] = artifact_hashes(folder, ['predictions.npz','calibration.json','prediction_runtime.json'])
    with pytest.raises(ValueError, match='May calibration'):
        runner.validate_prediction_dependencies(tmp_path, prep, 'G3-cluster', 3, state, reused=False)


def test_device_and_native_environment_match_original_auto_g():
    config = registration(['G0-global'], [], 'baseline_stability_only')
    original = {'draws': 400, 'individual_tau': 1000, 'cluster_tau': 10000,
                'registration': {'training': {'kind': config['kind'], 'width': 128, **config['budget']}}}
    parent = {'config_sha256': 'parent', 'source_hashes': {}, 'torch': 'frozen', 'pythonpath': 'deps'}
    own = {**parent, 'config_sha256': 'new', 'source_hashes': {'new': 'source'}}
    runner.validate_parent_science(config, original, parent, own)
    with pytest.raises(ValueError, match='native/local environment'):
        runner.validate_parent_science(config, original, parent, {**own, 'pythonpath': 'other'})
    with pytest.raises(ValueError, match='automatic device'):
        runner.validate_parent_science({**config, 'device': 'cpu'}, original, parent, own)


def test_resource_gate_includes_may_calibration_and_all_selected_predictions():
    config = registration(['G0-global'], [], 'baseline_stability_only')
    profile = {'rough_per_seed_fit_projection_seconds': 7000,
               'parent_full_cpanel_prediction_seconds': {'G0-global': [100, 200, 150]}}
    assert runner.resource_gate(config, profile)['aggregate_projection_seconds'] == 14400
    with pytest.raises(ValueError, match='exceeds7200'):
        runner.resource_gate(config, {**profile, 'rough_per_seed_fit_projection_seconds': 7201})
    config['registration']['batch_wall_budget_seconds'] = 14399
    with pytest.raises(ValueError, match='aggregate'):
        runner.resource_gate(config, profile)
    config['registration']['batch_wall_budget_seconds'] = 0
    with pytest.raises(ValueError, match='resource registration'):
        validate_config(config)


def test_prepared_family_order_and_input_context_cannot_drift():
    config = registration(['G0-global', 'G1-personal', 'G2-feature'],
        [['G1-personal','G0-global'], ['G2-feature','G0-global']])
    parent = {'units': {'global': {'mode': 'global'}, 'personal1': {'mode': 'personal'}, 'feature': {'mode': 'feature'}},
              'samples': {'dev': {'n': 2}}, 'features': {'context': 'frozen'}, 'panel': {}, 'clusters': {}}
    prep = {**deepcopy(parent), 'identity': {'config_sha256': canonical_hash(config)},
            **{name: config[name] for name in ('cells','seeds','primary_comparisons','selection_status')}}
    runner.validate_prepared_contract(config, prep, parent)
    bad = deepcopy(prep); bad['primary_comparisons'].reverse()
    with pytest.raises(ValueError, match='ordered family'):
        runner.validate_prepared_contract(config, bad, parent)
    bad = deepcopy(prep); bad['features']['context'] = 'changed'
    with pytest.raises(ValueError, match='common input'):
        runner.validate_prepared_contract(config, bad, parent)


def test_resolver_exposes_complete_identity_hashes_for_scoring(tmp_path):
    prep, state, folder = fixture_member(tmp_path)
    dump(tmp_path/'preparation.json', prep)
    config = {'cells': prep['cells'], 'primary_comparisons': prep['primary_comparisons']}
    result = runner.resolve_member(config, tmp_path, prep, 'G3-cluster', 3)
    assert len(result['hashes']) == 11  # six fit files, preparation, four prediction files
    assert str(tmp_path/'fits/seed3/cluster0/state.json') in result['hashes']
    assert str(tmp_path/'fits/seed3/feature/state.json') not in result['hashes']
    assert result['hashes'][str(folder/'prediction_state.json')] == result['state_sha256']
