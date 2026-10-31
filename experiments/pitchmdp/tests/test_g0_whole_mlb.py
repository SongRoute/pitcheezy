"""G0 whole-MLB five-seed inference adapter: synthetic CPU contract checks only.

No real data, parents, checkpoints, fits or DEV scores. Synthetic success is not
an experiment result; the draft config stays unregistered.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'scripts')]
from pitchmdp.data import KEY, hash_file
from pitchmdp.matrix_data import canonical_hash
from pitchmdp.matrix_benchmark import predict_streamed
from pitchmdp.matrix_g0_whole import (ARCHIVE_FIELDS, CELL, FIXED, SEEDS, SOURCE_ARM, check_eligible_population,
    compare_levels, compare_predictions, cpanel_positions, cpanel_subset_alignment, frozen_weights, launch_gate,
    ledger_totals, member_identity, output_guard, project_costs, require_alignment, require_equal_levels,
    require_within_tolerance, validate_config, validate_member_archive, validate_native_environment)
from pitchmdp.matrix_models import MatrixModel
from pitchmdp.matrix_sharing import training_arrays
import run_ml_confirmation as c1
import run_ml_transfer as transfer
import run_ml_g0_whole as runner
from run_ml_benchmark import dump

DRAFT = REPO / 'configs' / 'EXP-P11-001-DRAFT.yaml'
EXPECTED = FIXED['expected_samples']


def draft():
    return json.loads(DRAFT.read_text())


def frozen(**over):
    """A fully pinned copy of the draft, valid for real-mode validation."""
    config = draft()
    for name in ('parent_g', 'parent_c1', 'parent_t4'):
        for key, value in config[name].items():
            if key.endswith('sha256') and value is None:
                config[name][key] = 'a' * 64
    config['budget']['batch_wall_budget_seconds'] = 14400
    config['registration'] = {**config['registration'], 'status': 'frozen (synthetic test only)'}
    for key, value in over.items():
        config[key] = value
    return config


# ---------------------------------------------------------------- configuration

def test_draft_validates_only_as_draft_and_real_mode_refuses_null_pins_and_draft_status():
    config = validate_config(draft(), real=False)
    assert 'DRAFT' in config['registration']['status'] and config['budget']['batch_wall_budget_seconds'] is None
    assert config['new_fits'] == 0 and config['calibration']['refit'] is False and config['held_out_confirmation'] is False
    with pytest.raises(ValueError, match='Null identity pin'):
        validate_config(draft(), real=True)
    pinned = frozen()
    assert validate_config(pinned, real=True) is pinned
    with pytest.raises(ValueError, match='DRAFT'):
        validate_config({**pinned, 'registration': {'status': 'draft copy', 'output': pinned['registration']['output']}}, real=True)
    unbudgeted = deepcopy(pinned); unbudgeted['budget']['batch_wall_budget_seconds'] = None
    with pytest.raises(ValueError, match='frozen before a real command'):
        validate_config(unbudgeted, real=True)
    assert validate_config(unbudgeted, real=False)


def test_fixed_protocol_settings_parents_budget_and_probe_bounds_cannot_change():
    base = frozen()
    for key, value in (('draws', 100), ('seeds', [0, 1, 2]), ('device', 'cpu'), ('cell', 'G3-cluster'), ('width', 64),
                       ('scope', 'Cpanel'), ('new_fits', 2), ('calibration', {**base['calibration'], 'refit': True}),
                       ('expected_samples', {**EXPECTED, 'dev': 1}), ('source_arms', {**base['source_arms'], '3': 'g'})):
        with pytest.raises(ValueError):
            validate_config({**base, key: value}, real=True)
    with pytest.raises(ValueError, match='schema'):
        validate_config({**base, 'extra': 1}, real=True)
    for mutate, match in ((lambda c: c['parent_g'].update(experiment_id='EXP-P4-002'), 'must be EXP-P4-001'),
                          (lambda c: c['parent_c1'].update(run=c['parent_g']['run']), 'distinct'),
                          (lambda c: c['parent_t4'].pop('dev_metadata_sha256'), 'requires exactly'),
                          (lambda c: c['parent_g'].update(preparation_sha256='zz'), 'hex'),
                          (lambda c: c['parent_c1'].update(config_file=''), 'nonempty path'),
                          (lambda c: c['budget'].update(profile_rows=0), 'positive integer'),
                          (lambda c: c['budget'].update(profile_rows=EXPECTED['dev'] + 1), 'cannot exceed'),
                          (lambda c: c['budget'].update(profile_wall_limit_seconds=7201), 'never authorizes'),
                          (lambda c: c['budget'].update(single_member_wall_limit_seconds=7200.5), 'positive integer'),
                          (lambda c: c['budget'].pop('batch_wall_budget_seconds'), 'requires exactly'),
                          (lambda c: c['equivalence_probe'].update(rows=EXPECTED['cpanel_dev'] + 1), 'within the Cpanel'),
                          (lambda c: c['equivalence_probe'].update(atol=0.), 'tolerance'),
                          (lambda c: c['equivalence_probe'].update(atol=.01), 'tolerance'),
                          (lambda c: c.update(registration={}), 'status string')):
        broken = deepcopy(base); mutate(broken)
        with pytest.raises(ValueError, match=match):
            validate_config(broken, real=True)


def test_native_environment_must_match_parent_except_config_and_sources():
    parent = {'config_sha256': 'a', 'source_hashes': {'x': 'b'}, 'python': '/venv/python', 'torch': '2.14', 'pythonpath': '/root',
              'local_config_sha256': 'c'}
    mine = {**parent, 'config_sha256': 'z', 'source_hashes': {'y': 'q'}}
    assert validate_native_environment(parent, mine)
    with pytest.raises(ValueError, match='pythonpath'):
        validate_native_environment(parent, {**mine, 'pythonpath': '/worktree'})
    with pytest.raises(ValueError, match='local_config_sha256'):
        validate_native_environment(parent, {**mine, 'local_config_sha256': 'd'})


# ---------------------------------------------------------------- identity and frozen calibration

def synthetic_prep(tmp_path):
    members = {}
    for seed in SEEDS:
        members[str(seed)] = {'seed': seed, 'source_arm': SOURCE_ARM[seed], 'source_run': str(tmp_path / SOURCE_ARM[seed]),
                              'directory': str(tmp_path / SOURCE_ARM[seed] / f'seed{seed}'),
                              'state_path': str(tmp_path / SOURCE_ARM[seed] / f'seed{seed}' / 'prediction_state.json'),
                              'model_path': str(tmp_path / SOURCE_ARM[seed] / f'seed{seed}' / 'model.pt'),
                              'model_sha256': 'b' * 64, 'prediction_state_sha256': 'c' * 64, 'predictions_sha256': 'd' * 64,
                              'calibration_sha256': 'e' * 64, 'delivery_temperature': 1.05 + seed / 100, 'device': 'cpu',
                              'network': {'kind': 'flatten_mlp', 'n_context': 52, 'n_token': 38, 'length': 6, 'width': 128, 'n_classes': 10},
                              'parameter_count': 141834}
    return {'identity': {'source_hashes': runner.source_hashes()}, 'members': members, 'clusters': {}}


def test_member_identity_binds_preparation_source_arm_checkpoint_and_temperature(tmp_path):
    prep = synthetic_prep(tmp_path)
    identity = member_identity(prep, 3)
    assert identity['source_arm'] == 'c1' and identity['seed'] == 3 and identity['draws'] == 400
    assert identity['preparation_sha256'] == canonical_hash(prep) and identity['model_sha256'] == 'b' * 64
    assert member_identity(prep, 0)['source_arm'] == 'g'
    changed = deepcopy(prep); changed['members']['3']['delivery_temperature'] = 1.
    assert member_identity(changed, 3) != identity
    with pytest.raises(ValueError, match='Unregistered'):
        member_identity(prep, 5)


def c1_report(weights=(.74, .73, .71, .66, .67, .68)):
    return {'selection': {'model_weight': weights[0], 'calibration_objective': 'log_loss'},
            'seeds': [{'blend_selection': {'model_weight': w}} for w in weights[1:]]}


def test_frozen_weights_require_five_valid_june_weights_and_bounded_may_temperatures():
    temperatures = {str(seed): 1.04 + seed / 100 for seed in SEEDS}
    calibration = frozen_weights(c1_report(), temperatures)
    assert calibration['june_ensemble_model_weight'] == .74 and calibration['june_seed_model_weights']['4'] == .68
    assert calibration['may_delivery_temperatures']['3'] == pytest.approx(1.07) and calibration['refit_on_whole_mlb'] is False
    with pytest.raises(ValueError, match='five ordered'):
        frozen_weights(c1_report((.74, .73, .71)), temperatures)
    with pytest.raises(ValueError, match='ensemble'):
        frozen_weights(c1_report((1.2, .73, .71, .66, .67, .68)), temperatures)
    with pytest.raises(ValueError, match='seed2'):
        frozen_weights(c1_report((.74, .73, .71, float('nan'), .67, .68)), temperatures)
    with pytest.raises(ValueError, match='log-loss'):
        frozen_weights({**c1_report(), 'selection': {'model_weight': .7, 'calibration_objective': 'brier_multiclass'}}, temperatures)
    with pytest.raises(ValueError, match='temperature'):
        frozen_weights(c1_report(), {**temperatures, '1': 3.})
    with pytest.raises(ValueError, match='Five ordered May'):
        frozen_weights(c1_report(), {k: v for k, v in temperatures.items() if k != '4'})


# ---------------------------------------------------------------- population alignment

def population(games=6, per_game=5, seed=0):
    rng = np.random.default_rng(seed)
    game_pk = np.repeat(np.arange(games) + 700, per_game).astype(np.int64)
    keys = np.column_stack([game_pk, np.tile(np.arange(per_game) // 2 + 1, games), np.tile(np.arange(per_game) % 2 + 1, games)]).astype(np.int64)
    baseline = {'dev_keys': keys, 'dev_y': rng.integers(0, 10, len(keys)).astype(np.int64), 'dev_game_pk': game_pk,
                'dev_pitcher': rng.choice([10, 20], len(keys)).astype(np.int64)}
    in_cpanel = np.zeros(len(keys), dtype=bool); in_cpanel[[1, 4, 7, 12, 20, 29]] = True
    expected = {'dev': len(keys), 'dev_games': games, 'cpanel_dev': 6, 'cpanel_games': len(np.unique(game_pk[in_cpanel]))}
    return baseline, in_cpanel, expected


def test_eligible_population_checks_keys_labels_metadata_and_registered_counts():
    baseline, _, expected = population()
    out = check_eligible_population(baseline, baseline['dev_keys'], baseline['dev_game_pk'], baseline['dev_pitcher'], expected)
    assert out == {'n': 30, 'games': 6}
    with pytest.raises(ValueError, match='differs from registration'):
        check_eligible_population(baseline, baseline['dev_keys'], baseline['dev_game_pk'], baseline['dev_pitcher'], {**expected, 'dev_games': 7})
    with pytest.raises(ValueError, match='metadata'):
        check_eligible_population(baseline, baseline['dev_keys'], baseline['dev_game_pk'] + 1, baseline['dev_pitcher'], expected)
    repeated = {**baseline, 'dev_keys': baseline['dev_keys'].copy()}; repeated['dev_keys'][1] = repeated['dev_keys'][0]
    with pytest.raises(ValueError, match='Repeated'):
        check_eligible_population(repeated, repeated['dev_keys'], baseline['dev_game_pk'], baseline['dev_pitcher'], expected)
    bad = {**baseline, 'dev_y': baseline['dev_y'].copy()}; bad['dev_y'][0] = 10
    with pytest.raises(ValueError, match='ten-class'):
        check_eligible_population(bad, baseline['dev_keys'], baseline['dev_game_pk'], baseline['dev_pitcher'], expected)
    disagree = {**baseline, 'dev_game_pk': baseline['dev_game_pk'].copy()}; disagree['dev_game_pk'][0] += 1
    with pytest.raises(ValueError, match='disagree'):
        check_eligible_population(disagree, baseline['dev_keys'], disagree['dev_game_pk'], baseline['dev_pitcher'], expected)


def test_cpanel_positions_require_the_ordered_in_cpanel_subset_and_report_complement():
    baseline, in_cpanel, expected = population()
    cpanel_keys = baseline['dev_keys'][in_cpanel]
    out = cpanel_positions(baseline['dev_keys'], in_cpanel, cpanel_keys, expected)
    assert out['positions'].tolist() == [1, 4, 7, 12, 20, 29] and out['overlap_pitches'] == 6
    assert out['complement_pitches'] == 24 and out['complement_games'] == 6 and out['overlap_games'] == expected['cpanel_games']
    with pytest.raises(ValueError, match='ordered in_cpanel subset'):
        cpanel_positions(baseline['dev_keys'], in_cpanel, cpanel_keys[::-1], expected)
    with pytest.raises(ValueError, match='ordered in_cpanel subset'):
        cpanel_positions(baseline['dev_keys'], in_cpanel, cpanel_keys[:5], expected)
    with pytest.raises(ValueError, match='differs from registration'):
        cpanel_positions(baseline['dev_keys'], in_cpanel, cpanel_keys, {**expected, 'cpanel_games': 1})
    with pytest.raises(ValueError, match='align'):
        cpanel_positions(baseline['dev_keys'], in_cpanel[:-1], cpanel_keys, expected)


def test_compare_predictions_and_member_archive_contract():
    rng = np.random.default_rng(1)
    baseline, _, _ = population()
    p = rng.dirichlet(np.ones(10), 30)
    ok = compare_predictions(p, p + 5e-7, atol=1e-6, name='probe')
    assert ok['within_tolerance'] and require_within_tolerance(ok) is ok and ok['rows'] == 30
    bad = compare_predictions(p, p + 1e-5, atol=1e-6, name='probe')
    assert bad['within_tolerance'] is False and bad['maximum_absolute_difference'] == pytest.approx(1e-5)
    with pytest.raises(ValueError, match='no longer reproduces'):
        require_within_tolerance(bad)
    with pytest.raises(ValueError, match='shapes differ'):
        compare_predictions(p, p[:5], atol=1e-6, name='probe')
    values = {'dev': p, 'dev_raw': p, 'dev_delivery_level': rng.integers(-1, 4, 30).astype(np.int64),
              **{k: v for k, v in baseline.items()}}
    assert set(values) == set(ARCHIVE_FIELDS) and validate_member_archive(values, baseline)
    with pytest.raises(ValueError, match='unpaired: y'):
        validate_member_archive({**values, 'dev_y': (baseline['dev_y'] + 1) % 10}, baseline)
    with pytest.raises(ValueError, match='mass'):
        validate_member_archive({**values, 'dev_raw': p * 1.01}, baseline)
    with pytest.raises(ValueError, match='tier'):
        validate_member_archive({**values, 'dev_delivery_level': np.full(30, 4)}, baseline)
    with pytest.raises(ValueError, match='fields'):
        validate_member_archive({k: v for k, v in values.items() if k != 'dev_raw'}, baseline)


# ---------------------------------------------------------------- cost projection and ledger

def test_projection_is_linear_in_rows_and_gates_member_and_batch_budgets():
    measured = {'rows': 256, 'command_overhead_seconds': 3., 'load_seconds': 40., 'model_load_seconds': 2.,
                'probe_seconds': .5, 'inference_seconds': 2.56, 'wall_seconds': 50.}
    budget = {'single_member_wall_limit_seconds': 7200, 'batch_wall_budget_seconds': 20000}
    out = project_costs(measured, n_dev=311721, probe_rows=64, budget=budget, prepare_seconds=10.)
    assert out['seconds_per_row'] == pytest.approx(.01)
    # Verification + data load + checkpoint load overhead, then per-row inference over population and probe rows.
    assert out['member_seconds'] == pytest.approx(45. + .01 * (311721 + 64))
    without_model = project_costs({**measured, 'model_load_seconds': 0.}, n_dev=311721, probe_rows=64, budget=budget, prepare_seconds=10.)
    assert out['member_seconds'] - without_model['member_seconds'] == pytest.approx(2.)
    with pytest.raises(ValueError, match='model_load_seconds'):
        project_costs({k: v for k, v in measured.items() if k != 'model_load_seconds'}, n_dev=311721, probe_rows=64, budget=budget, prepare_seconds=10.)
    assert out['five_member_seconds'] == pytest.approx(5 * out['member_seconds'])
    assert out['family_seconds'] == pytest.approx(10. + 50. + out['five_member_seconds'])
    assert out['member_gate'] and out['batch_gate']
    tight = project_costs(measured, n_dev=311721, probe_rows=64, budget={**budget, 'batch_wall_budget_seconds': 14400}, prepare_seconds=10.)
    assert tight['member_gate'] and tight['batch_gate'] is False
    slow = project_costs({**measured, 'inference_seconds': 25.6}, n_dev=311721, probe_rows=64, budget=budget, prepare_seconds=10.)
    assert slow['member_gate'] is False
    with pytest.raises(ValueError, match='Finite nonnegative'):
        project_costs({**measured, 'load_seconds': float('nan')}, n_dev=311721, probe_rows=64, budget=budget, prepare_seconds=10.)
    with pytest.raises(ValueError, match='Positive'):
        project_costs({**measured, 'rows': 0}, n_dev=311721, probe_rows=64, budget=budget, prepare_seconds=10.)


def entry(event, identifier, stage='predict', seconds=None):
    record = {'event': event, 'id': identifier, 'stage': stage}
    if event == 'end':
        record['seconds'] = seconds
    return record


def test_ledger_reserves_unresolved_starts_and_launch_gate_refuses_overruns():
    caps = {'prepare': 7200, 'profile': 600, 'predict': 7200}
    entries = [entry('start', 'a', 'prepare'), entry('end', 'a', seconds=5.), entry('start', 'b', 'profile'), entry('end', 'b', seconds=30.),
               entry('start', 'c'), entry('end', 'c', seconds=400.), entry('start', 'd')]
    totals = ledger_totals(entries, caps)
    assert totals == {'spent_seconds': 435., 'reserved_seconds': 7200., 'charged_seconds': 7635., 'unresolved': ['d']}
    assert ledger_totals(entries, caps, exclude='d')['unresolved'] == []
    with pytest.raises(RuntimeError, match='Unresolved'):
        launch_gate(totals, projected=1., budget_seconds=20000, stage='predict')
    resolved = ledger_totals(entries[:-1], caps)
    assert launch_gate(resolved, projected=500., budget_seconds=1000, stage='predict')['remaining_seconds'] == 565.
    with pytest.raises(RuntimeError, match='exceeds the remaining batch budget'):
        launch_gate(resolved, projected=566., budget_seconds=1000, stage='predict')
    with pytest.raises(RuntimeError, match='Finite'):
        launch_gate(resolved, projected=float('inf'), budget_seconds=1000, stage='predict')
    with pytest.raises(ValueError, match='finite'):
        ledger_totals([entry('start', 'x'), entry('end', 'x', seconds=float('nan'))], caps)
    with pytest.raises(ValueError, match='Unregistered ledger stage'):
        ledger_totals([entry('start', 'x', 'score')], caps)


def test_runner_ledger_records_completion_and_failure_and_gates_predict(tmp_path):
    config = frozen()
    config['budget']['batch_wall_budget_seconds'] = 100
    output = tmp_path / 'out'
    with runner.ledger_stage(output, config, 'prepare') as active:
        assert active.stage == 'prepare' and active.elapsed() >= 0
    with pytest.raises(RuntimeError, match='boom'):
        with runner.ledger_stage(output, config, 'profile'):
            raise RuntimeError('boom')
    entries = runner.ledger_entries(output)
    assert [e['event'] for e in entries] == ['start', 'end', 'start', 'end']
    assert entries[1]['status'] == 'completed' and entries[3]['status'] == 'failed' and 'boom' in entries[3]['error']
    assert entries[0]['cap_seconds'] == 7200 and entries[2]['cap_seconds'] == 600
    assert runner._stage_seconds(output, config, 'prepare') == entries[1]['seconds']
    with runner.ledger_stage(output, config, 'predict', 0) as active:
        gate = runner.ledger_gate(output, config, active, 50., 'predict')
        assert gate['remaining_seconds'] <= 100 and gate['ledger']['unresolved'] == []
        with pytest.raises(RuntimeError, match='exceeds the remaining batch budget'):
            runner.ledger_gate(output, config, active, 101., 'predict')
    # An unresolved start (crash without an end) blocks the next launch.
    runner._append_ledger(output, {'event': 'start', 'id': 'ghost', 'stage': 'predict', 'seed': 1, 'cap_seconds': 7200})
    with runner.ledger_stage(output, config, 'predict', 2) as active:
        with pytest.raises(RuntimeError, match='Unresolved'):
            runner.ledger_gate(output, config, active, 1., 'predict')
    assert runner.stage_caps(config) == {'prepare': 7200, 'profile': 600, 'predict': 7200}


# ---------------------------------------------------------------- source closure and no parent mutation

def test_source_closure_covers_frozen_c1_transfer_scorer_and_new_files_and_never_writes_parent_members():
    sources = set(runner.SOURCES)
    assert sources >= set(c1.SOURCES) | set(transfer.SOURCES)
    assert {'scripts/score_ml_confirmation.py', 'pitchmdp/matrix_g0_whole.py', 'scripts/run_ml_g0_whole.py',
            'pitchmdp/matrix_sharing.py', 'pitchmdp/matrix_confirmation.py', 'scripts/run_ml_transfer.py'} <= sources
    assert all((PROJECT / name).is_file() for name in sources)
    hashes = runner.source_hashes()
    assert hashes['scripts/run_ml_g0_whole.py'] == hash_file(PROJECT / 'scripts/run_ml_g0_whole.py')
    text = (PROJECT / 'scripts/run_ml_g0_whole.py').read_text()
    assert 'predict_sharing' not in text and 'mlb=True' not in text and 'mlb_prediction' not in text
    assert 'def fit(' not in text and '.fit(' not in text and 'calibrate(' not in text and 'fit_blend' not in text
    assert 'ledger.jsonl' in runner.MUTABLE


def test_identity_includes_new_sources_and_status_in_draft_config_is_unregistered(tmp_path):
    local = tmp_path / 'local.json'; dump(local, {'artifact_root': str(tmp_path)})
    identity = runner.identity(frozen(), local)
    assert identity['source_hashes'] == runner.source_hashes() and identity['local_config_sha256'] == hash_file(local)
    assert 'pythonpath' in identity and 'python' in identity
    assert runner.identity(frozen(experiment_id='EXP-P11-002'), local)['config_sha256'] != identity['config_sha256']
    registration = draft()['registration']
    assert registration['status'].startswith('DRAFT_UNREGISTERED') and registration['scorer'].startswith('not implemented')


# ---------------------------------------------------------------- profile/predict gates on synthetic outputs

def test_require_profile_refuses_missing_stale_or_failed_gates(tmp_path):
    output = tmp_path / 'out'; output.mkdir()
    dump(output / 'preparation.json', {'x': 1})
    with pytest.raises(ValueError, match='must complete before'):
        runner.require_profile(output)
    dest = output / 'profile'; dest.mkdir()
    dump(dest / 'profile.json', {'projection': {'member_gate': True, 'batch_gate': False, 'member_seconds': 1.}})
    dump(dest / 'state.json', {'preparation_sha256': hash_file(output / 'preparation.json'),
                               'artifact_hashes': {'profile.json': hash_file(dest / 'profile.json')}})
    with pytest.raises(ValueError, match='exceeds the registered'):
        runner.require_profile(output)
    dump(dest / 'profile.json', {'projection': {'member_gate': True, 'batch_gate': True, 'member_seconds': 1.}})
    with pytest.raises(ValueError, match='Artifact identity changed'):
        runner.require_profile(output)
    dump(dest / 'state.json', {'preparation_sha256': hash_file(output / 'preparation.json'),
                               'artifact_hashes': {'profile.json': hash_file(dest / 'profile.json')}})
    assert runner.require_profile(output)['member_seconds'] == 1.
    dump(output / 'preparation.json', {'x': 2})
    with pytest.raises(ValueError, match='Matching'):
        runner.require_profile(output)


def test_verify_member_binds_identity_artifacts_and_dependencies(tmp_path):
    prep = synthetic_prep(tmp_path)
    member = prep['members']['0']
    source = Path(member['directory']); source.mkdir(parents=True)
    Path(member['model_path']).write_bytes(b'checkpoint')
    dump(Path(member['state_path']), {'identity': {}})
    dump(source / 'calibration.json', {'delivery_temperature': member['delivery_temperature']})
    np.savez_compressed(source / 'predictions.npz', dev=np.ones((2, 10)) / 10)
    for name in ('model_sha256', 'prediction_state_sha256', 'calibration_sha256', 'predictions_sha256'):
        path = {'model_sha256': member['model_path'], 'prediction_state_sha256': member['state_path'],
                'calibration_sha256': str(source / 'calibration.json'), 'predictions_sha256': str(source / 'predictions.npz')}[name]
        member[name] = hash_file(Path(path))
    output = tmp_path / 'out'; folder = runner.member_dir(output, 0); folder.mkdir(parents=True)
    dump(output / 'preparation.json', prep)
    np.savez_compressed(folder / 'predictions.npz', dev=np.ones((3, 10)) / 10)
    dump(folder / 'prediction_runtime.json', {'seconds': 1.})
    dump(folder / 'source_prediction_state.json', {'identity': {}})
    dump(folder / 'source_calibration.json', {'delivery_temperature': member['delivery_temperature']})
    names = ['predictions.npz', 'prediction_runtime.json', 'source_prediction_state.json', 'source_calibration.json']
    dependencies = {member['model_path']: member['model_sha256'], member['state_path']: member['prediction_state_sha256'],
                    str(source / 'calibration.json'): member['calibration_sha256'], str(source / 'predictions.npz'): member['predictions_sha256'],
                    str(output / 'preparation.json'): hash_file(output / 'preparation.json')}
    state = {'identity': member_identity(prep, 0), 'dependencies': dependencies,
             'artifact_hashes': {name: hash_file(folder / name) for name in names}}
    dump(folder / 'prediction_state.json', state)
    assert runner.verify_member(output, prep, 0)['identity']['source_arm'] == 'g'
    with pytest.raises(ValueError, match='identity differs'):
        runner.verify_member(output, {**prep, 'members': {**prep['members'], '0': {**member, 'delivery_temperature': 1.}}}, 0)
    Path(member['model_path']).write_bytes(b'tampered')
    with pytest.raises(ValueError, match='dependency changed'):
        runner.verify_member(output, prep, 0)
    Path(member['model_path']).write_bytes(b'checkpoint')
    np.savez_compressed(folder / 'predictions.npz', dev=np.ones((4, 10)) / 10)
    with pytest.raises(ValueError, match='Artifact identity changed'):
        runner.verify_member(output, prep, 0)
    dump(folder / 'prediction_state.json', {**state, 'artifact_hashes': {k: v for k, v in state['artifact_hashes'].items() if k != 'source_calibration.json'}})
    with pytest.raises(ValueError, match='artifact family'):
        runner.verify_member(output, prep, 0)
    with pytest.raises(ValueError, match='Preserve existing content'):
        runner._fresh(folder)
    assert sorted(runner._files(output)) == sorted(str(p.relative_to(output)) for p in output.rglob('*') if p.is_file())
    (output / 'ledger.jsonl').write_text('{}\n')
    assert 'ledger.jsonl' not in runner._files(output)


# ---------------------------------------------------------------- numerical path: same wrapper, same integration, sealed probe

class FakeDelivery:
    """Deterministic per-row draws; the same rows always produce the same logits regardless of chunking."""
    draws = 4

    def __init__(self, tier=3):
        self.tier = tier

    def logits(self, model, store, context, rows):
        rows = np.asarray(rows)
        tokens, valid, ctx = [], [], []
        for row in rows:
            rng = np.random.default_rng(int(row) + 1000)
            tokens.append(rng.normal(size=(self.draws, 6, 21)).astype(np.float32))
            valid.append(np.ones((self.draws, 6), dtype=bool))
            ctx.append(rng.normal(size=(self.draws, 59)).astype(np.float32))
        arrays = (np.concatenate(tokens), np.concatenate(valid), np.concatenate(ctx))
        arrays[0][:, -1, -11:] = 0
        return model.logits(arrays).reshape(len(rows), self.draws, 10), np.full(len(rows), self.tier, dtype=np.int64)


def toy_arrays(n, seed):
    rng = np.random.default_rng(seed)
    tokens = rng.normal(size=(n, 6, 21)).astype(np.float32)
    tokens[:, -1, -11:] = 0
    return tokens, np.ones((n, 6), dtype=bool), rng.normal(size=(n, 59)).astype(np.float32)


def tiny_member(tmp_path, seed=0, temperature=1.1):
    train, early = toy_arrays(64, 0), toy_arrays(16, 1)
    model = MatrixModel('flatten_mlp', seed=seed, width=8, device='cpu').fit(
        training_arrays(train), np.arange(64) % 10, training_arrays(early), np.arange(16) % 10,
        epochs=1, patience=1, batch_size=32, learning_rate=.001)
    folder = tmp_path / 'g' / f'seed{seed}'; folder.mkdir(parents=True)
    model.save(folder / 'model.pt')
    member = {'seed': seed, 'source_arm': 'g', 'source_run': str(tmp_path / 'g'), 'directory': str(folder),
              'state_path': str(folder / 'prediction_state.json'), 'model_path': str(folder / 'model.pt'),
              'model_sha256': hash_file(folder / 'model.pt'), 'delivery_temperature': temperature, 'device': 'cpu',
              'network': model.report['network'], 'parameter_count': model.report['parameter_count']}
    return model, member, folder


def test_load_member_reuses_checkpoint_and_temperature_and_refuses_tampering(tmp_path):
    config = {**frozen(), 'device': 'cpu', 'width': 8}
    model, member, folder = tiny_member(tmp_path)
    predictor, device = runner.load_member(config, member, {})
    assert device == 'cpu' and predictor.delivery_temperature == 1.1 and predictor.cell == CELL
    assert predictor.report['calibration_reused_unchanged'] is True and predictor.report['delivery_temperature'] == 1.1
    np.testing.assert_allclose(predictor.global_model.logits(training_arrays(toy_arrays(5, 3))), model.logits(training_arrays(toy_arrays(5, 3))))
    with pytest.raises(ValueError, match='network differs'):
        runner.load_member(config, {**member, 'parameter_count': 1}, {})
    with pytest.raises(ValueError, match='identity differs'):
        runner.load_member(config, {**member, 'seed': 1}, {})
    with pytest.raises(ValueError, match='Device'):
        runner.load_member(config, {**member, 'device': 'mps'}, {})
    (folder / 'model.pt').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='checkpoint changed'):
        runner.load_member(config, member, {})


def test_probe_reproduces_sealed_predictions_through_the_same_path_and_rejects_drift(tmp_path):
    config = {**frozen(), 'device': 'cpu', 'width': 8}
    _, member, folder = tiny_member(tmp_path)
    delivery = FakeDelivery()
    frame = pd.DataFrame({'game_pk': np.repeat([1, 2], 10), 'at_bat_number': np.tile(np.arange(5) + 1, 4) // 1,
                          'pitch_number': np.tile([1, 2], 10)})
    frame['at_bat_number'] = np.repeat(np.arange(10) + 1, 2)
    dev = frame.iloc[:12]
    # Seal predictions the way the original panel path did: same wrapper, same integration, any chunking.
    predictor, _ = runner.load_member(config, member, {})
    sealed_p, sealed_raw, sealed_levels = predict_streamed(predictor, delivery, None, None, dev.index.to_numpy(), chunk_size=5)
    keys = dev[KEY].to_numpy(np.int64)
    np.savez_compressed(folder / 'predictions.npz', dev=sealed_p, dev_raw=sealed_raw, dev_delivery_level=sealed_levels, dev_keys=keys)
    probe = runner.probe_cpanel(predictor, {'delivery': delivery}, None, None, dev, member, rows=7, atol=1e-6)
    assert probe['rows'] == 7 and [r['name'] for r in probe['reports']] == ['probe_calibrated', 'probe_raw', 'probe_delivery_level']
    assert all(r['within_tolerance'] for r in probe['reports'][:2]) and probe['reports'][2] == {'name': 'probe_delivery_level', 'rows': 7, 'mismatches': 0, 'equal': True}
    assert probe['reports'][0]['maximum_absolute_difference'] <= 1e-6
    # Identical probabilities with a different delivery tier must fail: tiers are exact routing decisions.
    with pytest.raises(ValueError, match='delivery tiers differ'):
        runner.probe_cpanel(predictor, {'delivery': FakeDelivery(tier=2)}, None, None, dev, member, rows=7, atol=1e-6)
    # A whole-population pass over a superset reproduces the overlap rows exactly.
    whole_p, whole_raw, _ = predict_streamed(predictor, delivery, None, None, frame.index.to_numpy(), chunk_size=64)
    positions = np.arange(12)
    # Chunk composition changes float results at the 1e-9 level; the registered tolerance, not equality, is the gate.
    subset = compare_predictions(whole_p[positions], sealed_p, atol=1e-6, name='subset')
    assert subset['within_tolerance'] and subset['maximum_absolute_difference'] <= 1e-6
    assert compare_predictions(whole_raw[positions], sealed_raw, atol=1e-6, name='subset_raw')['within_tolerance']
    # Any temperature or checkpoint drift breaks the probe.
    drifted, _ = runner.load_member(config, {**member, 'delivery_temperature': 1.3}, {})
    with pytest.raises(ValueError, match='no longer reproduces'):
        runner.probe_cpanel(drifted, {'delivery': delivery}, None, None, dev, member, rows=7, atol=1e-6)
    np.savez_compressed(folder / 'predictions.npz', dev=sealed_p, dev_raw=sealed_raw, dev_delivery_level=sealed_levels, dev_keys=keys[::-1])
    with pytest.raises(ValueError, match='sealed Cpanel keys'):
        runner.probe_cpanel(predictor, {'delivery': delivery}, None, None, dev, member, rows=7, atol=1e-6)


def test_full_subset_alignment_requires_exact_tiers_even_with_identical_probabilities():
    rng = np.random.default_rng(4)
    baseline, in_cpanel, _ = population()
    positions = np.flatnonzero(in_cpanel)
    p = rng.dirichlet(np.ones(10), 30)
    levels = rng.integers(0, 4, 30).astype(np.int64)
    values = {'dev': p, 'dev_raw': p, 'dev_delivery_level': levels, **baseline}
    sealed = {'dev_keys': baseline['dev_keys'][positions], 'dev_y': baseline['dev_y'][positions], 'dev': p[positions] + 1e-8,
              'dev_raw': p[positions], 'dev_delivery_level': levels[positions].copy()}
    reports = cpanel_subset_alignment(values, sealed, positions, atol=1e-6)
    assert [r['name'] for r in reports] == ['cpanel_subset_calibrated', 'cpanel_subset_raw', 'cpanel_subset_delivery_level']
    assert require_alignment(reports) is reports and reports[2]['equal'] and reports[2]['rows'] == 6
    drifted = dict(sealed); drifted['dev_delivery_level'] = sealed['dev_delivery_level'].copy(); drifted['dev_delivery_level'][2] = (drifted['dev_delivery_level'][2] + 1) % 4
    broken = cpanel_subset_alignment(values, drifted, positions, atol=1e-6)
    assert broken[0]['within_tolerance'] and broken[1]['within_tolerance'] and broken[2] == {'name': 'cpanel_subset_delivery_level', 'rows': 6, 'mismatches': 1, 'equal': False}
    with pytest.raises(ValueError, match='1 delivery tiers differ'):
        require_alignment(broken)
    with pytest.raises(ValueError, match='keys/labels differ'):
        cpanel_subset_alignment(values, {**sealed, 'dev_y': (sealed['dev_y'] + 1) % 10}, positions, atol=1e-6)
    with pytest.raises(ValueError, match='shapes/dtypes'):
        compare_levels(levels[:5].astype(float), levels[:5], name='x')
    with pytest.raises(ValueError, match='no longer reproduces'):
        require_alignment(cpanel_subset_alignment({**values, 'dev_raw': np.roll(p, 1, axis=0)}, sealed, positions, atol=1e-6))
    assert require_equal_levels(compare_levels(levels, levels, name='same'))['mismatches'] == 0


# ---------------------------------------------------------------- output guard: no write before disjointness

def guarded_config(tmp_path, output):
    config = frozen()
    protocol = tmp_path / 'artifacts' / 'runs' / 'ML-MATRIX-20260924'
    for name, experiment in (('parent_g', 'EXP-P4-001'), ('parent_c1', 'EXP-P10-001'), ('parent_t4', 'EXP-P7-003')):
        config[name]['run'] = str(protocol / experiment)
    config['registration']['output'] = str(output)
    return config, protocol


def test_output_guard_requires_registered_output_and_disjointness_from_all_parents(tmp_path):
    config, protocol = guarded_config(tmp_path, tmp_path / 'artifacts' / 'runs' / 'ML-MATRIX-20260924' / 'EXP-P11-001')
    assert output_guard(protocol / 'EXP-P11-001', config) == (protocol / 'EXP-P11-001').resolve()
    with pytest.raises(ValueError, match='differs from the registered output'):
        output_guard(protocol / 'EXP-P11-002', config)
    for offending, name in ((protocol / 'EXP-P4-001', 'parent_g'), (protocol / 'EXP-P10-001' / 'members', 'parent_c1'),
                            (protocol, 'parent_g'), (tmp_path, 'parent_g'), (protocol / 'EXP-P7-003', 'parent_t4')):
        config['registration']['output'] = str(offending)
        with pytest.raises(ValueError, match=f'not disjoint from the sealed {name}'):
            output_guard(offending, config)
    with pytest.raises(ValueError, match='registered output'):
        validate_config({**frozen(), 'registration': {'status': 'frozen'}}, real=True)


def test_main_refuses_parent_outputs_before_any_write_including_ledger(tmp_path, monkeypatch):
    protocol = tmp_path / 'artifacts' / 'runs' / 'ML-MATRIX-20260924'
    sentinels = {}
    for experiment in ('EXP-P4-001', 'EXP-P10-001', 'EXP-P7-003'):
        (protocol / experiment / 'members').mkdir(parents=True)
        sentinel = protocol / experiment / 'members' / 'sealed.json'
        sentinel.write_text('{"sealed": true}')
        sentinels[experiment] = hash_file(sentinel)
    local = tmp_path / 'local.json'; dump(local, {'artifact_root': str(tmp_path / 'artifacts')})
    monkeypatch.setattr(runner, 'validate_native_runtime', lambda: None)
    monkeypatch.setattr(runner, 'prepare', lambda *a, **k: pytest.fail('prepare must not run'))
    before = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob('*'))
    cases = [(protocol / 'EXP-P4-001', 'parent_g'), (protocol / 'EXP-P10-001' / 'members', 'parent_c1'),
             (protocol, 'parent_g'), (protocol / 'EXP-P7-003', 'parent_t4')]
    for offending, name in cases:
        config, _ = guarded_config(tmp_path, offending)
        config_path = tmp_path / 'config.json'; dump(config_path, config)
        before_run = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob('*'))
        monkeypatch.setattr(sys, 'argv', ['run_ml_g0_whole.py', '--config', str(config_path), '--local-config', str(local),
                                          '--output', str(offending), 'prepare'])
        with pytest.raises(ValueError, match=f'not disjoint from the sealed {name}'):
            runner.main()
        assert sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob('*')) == before_run
        assert not list(tmp_path.rglob('ledger.jsonl')) and not list(protocol.rglob('.heavy.lock'))
    for experiment, digest in sentinels.items():
        assert hash_file(protocol / experiment / 'members' / 'sealed.json') == digest
    # An unregistered but disjoint output is refused just as early.
    config, _ = guarded_config(tmp_path, protocol / 'EXP-P11-001')
    dump(tmp_path / 'config.json', config)
    monkeypatch.setattr(sys, 'argv', ['run_ml_g0_whole.py', '--config', str(tmp_path / 'config.json'), '--local-config', str(local),
                                      '--output', str(protocol / 'EXP-P11-002'), 'prepare'])
    with pytest.raises(ValueError, match='differs from the registered output'):
        runner.main()
    assert not list(tmp_path.rglob('ledger.jsonl')) and set(before) <= set(str(p.relative_to(tmp_path)) for p in tmp_path.rglob('*'))
