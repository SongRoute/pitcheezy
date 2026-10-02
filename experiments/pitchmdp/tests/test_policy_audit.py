"""Synthetic inference and stage-integrity regression tests; no model loading."""
from pathlib import Path
import sys
import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'scripts')]
from pitchmdp.matrix_policy import game_policy_comparisons, paired_policy_comparisons, POLICY_INFERENCE
from pitchmdp.matrix_data import canonical_hash
from pitchmdp.data import hash_file
from run_ml_policy import seal_stage, verify_stage, verify_tuning, dump, read_json


def values(n):
    v = {name: np.full((n, 2), .5 if name == 'P0' else .51) for name in ('P0', 'P1', 'P2', 'P3')}
    return v, {name: np.zeros((n, 2), dtype=bool) for name in v}


@pytest.mark.parametrize('games', [np.array([1]), np.arange(58)%29, np.arange(49)%30])
def test_below_gate_keeps_all_three_null_inference_slots(games):
    v, flags = values(len(games))
    rows = game_policy_comparisons(v, games, truncated=flags, draws=1000)
    assert len(rows) == 3
    for row in rows:
        assert row['game_bootstrap_ci95'] is None
        assert row['worst_case_game_bootstrap_ci95'] is None
        assert row['holm_p'] is row['imputed_holm_p'] is None
        assert row['model_internal_screen'] == 'descriptive_only'
        assert not row['untruncated_pa_improvement_confirmed']


def test_positive_worst_case_point_bound_does_not_confirm_uncertain_tail():
    v, flags = values(100)
    flags['P1'][0] = True
    row = game_policy_comparisons(v, np.repeat(np.arange(50), 2), truncated=flags)[0]
    assert row['model_internal_P_screen']
    assert row['worst_case_mean_delta_lower'] > 0
    assert row['worst_case_game_bootstrap_ci95'][0] < 0
    assert not row['untruncated_pa_improvement_confirmed']
    assert row['model_internal_screen'] == 'tail_assumption_dependent'
    assert row['combined_p_greater'] == max(row['p_greater'], row['worst_case_p_greater'])
    assert row['holm_p'] >= row['imputed_holm_p']


def test_no_truncation_recovers_same_p_and_ci_at_exact_gate():
    v, flags = values(50)
    rows = game_policy_comparisons(v, np.arange(50)%30, truncated=flags)
    row = rows[0]
    assert row['game_bootstrap_ci95'] == row['worst_case_game_bootstrap_ci95']
    assert row['p_greater'] == row['worst_case_p_greater']
    assert row['holm_p'] == row['imputed_holm_p']
    assert row['untruncated_pa_improvement_confirmed']
    assert rows[1]['holm_p'] == 1


def test_missing_bounds_or_invalid_rollouts_cannot_confirm():
    v, flags = values(50)
    row = game_policy_comparisons(v, np.arange(50))[0]
    assert row['model_internal_P_screen']
    assert row['holm_p'] is None
    assert not row['untruncated_pa_improvement_confirmed']
    v['P3'][0, 0] = np.nan
    with pytest.raises(ValueError, match='Finite paired'):
        game_policy_comparisons(v, np.arange(50), truncated=flags)


def fixture_tuning(root):
    dump(root/'preparation.json', {'frozen': True})
    directory = root/'stages/blend-control'; directory.mkdir(parents=True)
    execution = {'tau_grid': [.001, .003, .01, .03], 'evaluation_rollouts': 2}
    dump(directory/'started.json', {'preparation_sha256': hash_file(root/'preparation.json'),
        'execution': execution, 'world': 'control', 'split': 'blend'})
    keys = ['1:1:1', '2:1:1']
    means = {'P2': .5, **{f'P3-tau{tau}': .5 for tau in execution['tau_grid']}}
    for key in keys:
        np.savez(directory/(key.replace(':', '-')+'.npz'),
                 **{name+'_values': np.full(2, value) for name, value in means.items()})
    dump(directory/'runtime.json', {'seconds': 1})
    dump(directory/'results.json', {'execution_sha256': canonical_hash(execution), 'world': 'control',
         'split': 'blend', 'pa_keys': keys, 'supported_selected': 2, 'mean_original_defensive_we': means,
         'selected_tau': .03, 'rl_comparator': 'P3'})
    seal_stage(directory)
    return directory, execution


def test_tuning_manifest_and_larger_tau_tie_are_verified(tmp_path):
    directory, execution = fixture_tuning(tmp_path)
    tuning, dependency = verify_tuning(tmp_path, execution)
    assert tuning['selected_tau'] == .03
    assert read_json(tmp_path/'stages/tuning_freeze.json') == dependency
    assert verify_tuning(tmp_path, execution)[1] == dependency
    (directory/'runtime.json').write_text('{}')
    with pytest.raises(ValueError): verify_tuning(tmp_path, execution)


def test_resealed_incorrect_tau_and_array_means_are_rejected(tmp_path):
    directory, execution = fixture_tuning(tmp_path)
    result = read_json(directory/'results.json')
    result['selected_tau'] = .001
    dump(directory/'results.json', result)
    (directory/'manifest.json').unlink(); seal_stage(directory)
    with pytest.raises(ValueError, match='argmax/tie rule'):
        verify_tuning(tmp_path, execution)
    result['selected_tau'] = .03
    result['mean_original_defensive_we']['P3-tau0.03'] = .6
    dump(directory/'results.json', result)
    (directory/'manifest.json').unlink(); seal_stage(directory)
    with pytest.raises(ValueError, match='archived rollout'):
        verify_tuning(tmp_path, execution)


def test_both_dev_worlds_must_reuse_exact_common_tuning_digest(tmp_path):
    directory, execution = fixture_tuning(tmp_path)
    _, dependency = verify_tuning(tmp_path, execution)
    dev = tmp_path/'stages/dev-control'; dev.mkdir()
    dump(dev/'started.json', {'preparation_sha256': hash_file(tmp_path/'preparation.json')})
    dump(dev/'tuning_dependency.json', {**dependency, 'manifest_sha256': 'changed'})
    seal_stage(dev)
    with pytest.raises(ValueError, match='DEV worlds reference different'):
        verify_tuning(tmp_path, execution)


def test_partial_or_extra_stage_artifacts_are_not_complete(tmp_path):
    directory, execution = fixture_tuning(tmp_path)
    (directory/'unregistered.json').write_text('{}')
    with pytest.raises(ValueError, match='artifact family'):
        verify_stage(tmp_path, 'blend-control')
    (directory/'manifest.json').unlink()
    with pytest.raises(FileNotFoundError): verify_tuning(tmp_path, execution)


def test_rl_comparator_recomputed_from_june_arrays(tmp_path):
    directory, execution = fixture_tuning(tmp_path)
    result = read_json(directory/'results.json'); result['rl_comparator'] = 'P2'
    dump(directory/'results.json', result)
    (directory/'manifest.json').unlink(); seal_stage(directory)
    with pytest.raises(ValueError, match='RL comparator'):
        verify_tuning(tmp_path, execution)


def test_rl_four_comparison_family_uses_identical_robust_gate():
    arrays = {name: np.full((50, 2), value) for name, value in
              [('IQL', .52), ('CQL', .51), ('planner', .5), ('NNBC', .49)]}
    flags = {name: np.zeros_like(p, dtype=bool) for name, p in arrays.items()}
    pairs = [('IQL', 'planner'), ('CQL', 'planner'), ('IQL', 'NNBC'), ('CQL', 'NNBC')]
    rows = paired_policy_comparisons(arrays, np.arange(50), pairs, truncated=flags, draws=1000)
    assert len(rows) == 4
    assert all(row['untruncated_pa_improvement_confirmed'] for row in rows)
    assert all(row['holm_p'] == 4/1001 for row in rows)
    assert all(row['family_size'] == 4 for row in rows)
    with pytest.raises(ValueError, match='Complete registered'):
        paired_policy_comparisons(arrays, np.arange(50), pairs[:2], truncated=flags)
