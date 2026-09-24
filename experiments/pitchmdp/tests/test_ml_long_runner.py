"""Configuration/resource gates only; fake CPU fits, no real data/profiling."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT)); sys.path.insert(0, str(PROJECT/'scripts'))
from pitchmdp.data import hash_file
from pitchmdp.matrix_long_experiment import DEFAULT_BUDGET
import pitchmdp.matrix_long_profile as profile_module
import run_ml_long_history as runner


def config():
    return {'protocol': 'ml_long_history_v2', 'experiment_id': 'EXP-TEST',
            'parent_run': '/frozen/sharing', 'parent_preparation_sha256': 'a'*64,
            'seeds': [0, 1, 2], 'width': 128, 'budget': deepcopy(DEFAULT_BUDGET),
            'draws': 400, 'device': 'cpu', 'profile': deepcopy(profile_module.PROFILE_SPEC)}


def test_strict_config_fixes_capacity_draws_seeds_budget_and_new_profile():
    assert runner.config_check(config())['width'] == 128
    for key, value in [('seeds', [0]), ('width', 64), ('draws', 25), ('parent_preparation_sha256', '')]:
        changed = config(); changed[key] = value
        with pytest.raises(ValueError): runner.config_check(changed)
    changed = config(); changed['budget']['batch_size'] = 1024
    with pytest.raises(ValueError): runner.config_check(changed)
    changed = config(); changed['profile']['measured_train'] = 8192
    with pytest.raises(ValueError): runner.config_check(changed)
    changed = config(); del changed['profile']
    with pytest.raises(ValueError): runner.config_check(changed)
    changed = config(); changed['silent_extra_axis'] = True
    with pytest.raises(ValueError): runner.config_check(changed)


def test_prepare_resume_checks_immutable_artifacts(tmp_path):
    expected = {'config_sha256': 'frozen'}
    path = tmp_path/'features.json'; path.write_text('{}')
    runner.dump(tmp_path/'preparation.json', {'identity': expected,
                 'artifact_hashes': {'features.json': hash_file(path)}})
    assert runner.verify(tmp_path, expected)['identity'] == expected
    path.write_text('{"changed": 1}')
    with pytest.raises(ValueError): runner.verify(tmp_path, expected)


def projected(fit_seconds=10.):
    return profile_module.resource_projection(warmup_seconds=3., measured_fit_seconds=fit_seconds,
        measured_updates=1024, full_train_rows=1252824, full_earlystop_rows=16000,
        temperature_seconds=.1, temperature_rows=640, inference_seconds=.1, blend_rows=1000, dev_rows=2000, load_seconds=2.)


def profile_report(cell, fit_seconds=10.):
    return dict(profile_spec=deepcopy(profile_module.PROFILE_SPEC), long_length=runner.CELLS[cell],
        population_counts={'train': 1252824, 'earlystop': 16000, 'temperature': 640, 'blend': 1000, 'dev': 2000},
        samples={name: {'n': count, 'rows_sha256': name} for name, count in profile_module.LIMITS.items()},
        warmup={'samples': {name: {'n': n, 'rows_sha256': name} for name, n in [('train', 8192), ('evaluation_train', 2048)]},
                'epochs': 1, 'optimizer_updates': 32},
        warmup_seconds=3., measured_fit_seconds=fit_seconds, fit_seconds=fit_seconds, optimizer_updates=1024,
        epochs=4, temperature_seconds=.1, inference_seconds=.1, load_seconds=2., projection=projected(fit_seconds))


def save_profile(root, cell, fit_seconds=10.):
    destination = root/'profiles'/cell; destination.mkdir(parents=True, exist_ok=True)
    runner.dump(destination/'profile.json', profile_report(cell, fit_seconds))
    runner.dump(destination/'manifest.json', {'identity': {'preparation_sha256': hash_file(root/'preparation.json'),
        'cell': cell, 'profile_version': 'f4_resource_profile_v2'},
        'artifact_hashes': {'profile.json': hash_file(destination/'profile.json')}})


def prepare_fixture(root):
    counts = profile_report('F4-H0')['population_counts']
    runner.dump(root/'preparation.json', {'samples': {name: {'n': n} for name, n in counts.items()}})


def family_fixture(root, fit_seconds=10., prior=100.):
    prepare_fixture(root)
    for cell in runner.CELLS: save_profile(root, cell, fit_seconds)
    prep_sha = hash_file(root/'preparation.json')
    entries = []
    for category in ('preparation', 'profiles', 'cold_failed_attempts', 'equivalence'):
        path = root/(category+'-evidence.json'); path.write_text('{}')
        entries.append({'category': category, 'attempt': category+'-1', 'seconds': prior/4,
                        'evidence': [{'path': str(path), 'sha256': hash_file(path)}]})
    ledger = root/'owner-ledger.json'
    runner.dump(ledger, {'protocol': 'ml_long_owner_budget_ledger_v1', 'preparation_sha256': prep_sha,
                        'entries': entries, 'elapsed_seconds_total': prior})
    return {'protocol': 'ml_long_family_budget_v2', 'preparation_sha256': prep_sha,
            'profile_manifest_sha256': {cell: hash_file(root/'profiles'/cell/'manifest.json') for cell in runner.CELLS},
            'owner_budget_ledger': {'path': str(ledger), 'sha256': hash_file(ledger)},
            'prior_seconds': prior, 'family_seconds': 28800}


def test_matching_profile_gate_recomputes_formula_and_requires_affordable_projection(tmp_path):
    prepare_fixture(tmp_path); save_profile(tmp_path, 'F4-128', fit_seconds=60.)
    with pytest.raises(ValueError, match='7200'): runner.verify_profile(tmp_path, 'F4-128')
    save_profile(tmp_path, 'F4-128')
    runner.verify_profile(tmp_path, 'F4-128')
    runner.dump(tmp_path/'preparation.json', {'changed': True})
    with pytest.raises(ValueError, match='matching'): runner.verify_profile(tmp_path, 'F4-128')


def test_projection_charges_cold_once_two_loads_ceil_full_updates_and_bounds_earlystop_ratio():
    args = dict(warmup_seconds=3., measured_fit_seconds=10.24, measured_updates=1024, full_train_rows=65537,
        full_earlystop_rows=2048, temperature_seconds=.2, temperature_rows=160,
        inference_seconds=.3, blend_rows=32, dev_rows=48, load_seconds=2.)
    result = profile_module.resource_projection(**args)
    assert result['full_optimizer_updates'] == 30*257
    assert result['cold_warmup_seconds_added_once'] == 3.
    assert result['data_loads_per_member'] == 2 and result['load_seconds'] == 4.
    assert result['load_fit_temperature_prediction_seconds'] == pytest.approx(87.6)
    with pytest.raises(ValueError, match='early-stop batch ratio'):
        profile_module.resource_projection(**{**args, 'full_earlystop_rows': 4096})
    with pytest.raises(ValueError, match='update-count'):
        profile_module.resource_projection(**{**args, 'measured_updates': 64})


def test_complete_family_budget_includes_all_prior_costs_and_all_three_profiles(tmp_path):
    budget = family_fixture(tmp_path)
    reports = runner.verify_family_budget(tmp_path, budget)
    saved = runner.read_json(tmp_path/'family_budget_projection.json')
    assert saved['nine_member_projection_seconds'] == pytest.approx(3*sum(r['projection']['load_fit_temperature_prediction_seconds'] for r in reports.values()))
    assert saved['total_projected_seconds'] == pytest.approx(saved['nine_member_projection_seconds']+100.)
    assert runner.read_json(tmp_path/'family_budget.json') == budget
    (tmp_path/'profiles/F4-32/manifest.json').unlink()
    with pytest.raises(FileNotFoundError): runner.verify_family_budget(tmp_path, budget)


def test_family_gate_rejects_nine_member_overrun_even_when_individual_members_pass(tmp_path):
    budget = family_fixture(tmp_path, fit_seconds=40.)
    assert all(runner.verify_profile(tmp_path, cell)['projection']['within_limit'] for cell in runner.CELLS)
    with pytest.raises(ValueError, match='nine-member family exceed'):
        runner.verify_family_budget(tmp_path, budget)
    assert not (tmp_path/'family_budget.json').exists()


def test_prior_cost_ledger_cannot_drop_or_fudge_old_attempts(tmp_path):
    budget = family_fixture(tmp_path, prior=20000.)
    with pytest.raises(ValueError, match='nine-member family exceed'): runner.verify_family_budget(tmp_path, budget)
    changed = deepcopy(budget); changed['prior_seconds'] = 1.
    with pytest.raises(ValueError, match='Prior seconds'): runner.verify_family_budget(tmp_path, changed)
    ledger_path = Path(budget['owner_budget_ledger']['path'])
    ledger = runner.read_json(ledger_path); ledger['entries'] = ledger['entries'][:-1]
    runner.dump(ledger_path, ledger); budget['owner_budget_ledger']['sha256'] = hash_file(ledger_path)
    with pytest.raises(ValueError, match='cover preparation'): runner.verify_family_budget(tmp_path, budget)


def test_missing_family_gate_precedes_loading_or_model_work(tmp_path, monkeypatch):
    budget = family_fixture(tmp_path)
    (tmp_path/'profiles/F4-128/manifest.json').unlink()
    monkeypatch.setattr(runner, '_load', lambda *args: pytest.fail('No data/model work before complete family gate'))
    with pytest.raises(FileNotFoundError):
        runner.run_member(config(), {}, tmp_path, {}, 'F4-H0', 0, 'fit', budget)


def test_profile_helper_uses_two_fresh_models_train_only_warmup_and_no_dev_quality(monkeypatch):
    class Batch:
        def __init__(self, n, ids=None, split='train'):
            self.ids = np.arange(n) if ids is None else ids; self.split = split
            self.store = SimpleNamespace(long_length=128)
        def __len__(self): return len(self.ids)
        def subset(self, indices): return Batch(0, self.ids[indices], self.split)
        def frame(self):
            return pd.DataFrame({'game_pk': self.ids+1, 'at_bat_number': 1, 'pitch_number': 1,
                                 'description': 'ball', 'events': None, 'strikes': 0, 'split': self.split})
    class HiddenBatch:
        def __len__(self): return 100
        def frame(self): raise AssertionError('DEV must not be gathered by a profile')
        def subset(self, indices): raise AssertionError('DEV must not be gathered by a profile')
    instances, calls = [], []
    class Model:
        def __init__(self, **kwargs):
            assert kwargs == {'seed': 0, 'width': 128, 'device': 'cpu'}
            self.instance = len(instances); instances.append(self.instance)
            self.device = 'cpu'; self.net = SimpleNamespace(config={'width': 128})
            self.report = {'parameter_count': 500, 'max_expanded_training_rows': 256,
                           'history': 'must not be emitted', 'log_loss': 'must not be emitted'}
        def fit(self, train, y, early, ey, **kwargs):
            if self.instance == 0:
                assert len(train) == 8192 and len(early) == 2048 and early.split == 'train'
                assert set(early.ids) <= set(train.ids)
                assert kwargs == {'epochs': 1, 'patience': 1, 'batch_size': 256, 'learning_rate': .0005}
                self.report.update(epochs_run=1, optimizer_updates=32)
            else:
                assert len(train) == 65536 and len(early) == 2048 and early.split == 'earlystop'
                assert kwargs == {'epochs': 4, 'patience': 4, 'batch_size': 256, 'learning_rate': .0005}
                self.report.update(epochs_run=4, optimizer_updates=1024)
            calls.append((self.instance, train.ids.copy(), early.ids.copy()))
            return self
    class Delivery:
        def __init__(self, delivery): assert delivery.draws == 400
        def calibrate(self, model, temp, y): assert model.instance == 1 and len(temp) == 16
        def predict(self, model, temp):
            assert model.instance == 1
            p = np.full((len(temp), 10), .1)
            return p, p, np.zeros(len(temp), dtype=int)
    monkeypatch.setattr(profile_module, 'LazyMatrixModel', Model)
    monkeypatch.setattr(profile_module, 'LazyJointDelivery', Delivery)
    batches = {'train': Batch(200000), 'earlystop': Batch(5000, split='earlystop'),
               'temperature': Batch(64, split='temperature'), 'blend': HiddenBatch(), 'dev': HiddenBatch()}
    result = profile_module.profile_batches(batches, SimpleNamespace(draws=400), device='cpu', load_seconds=2.)
    assert instances == [0, 1] and len(calls) == 2
    np.testing.assert_array_equal(calls[0][1], np.linspace(0, 199999, 8192, dtype=np.int64))
    np.testing.assert_array_equal(calls[1][1], np.linspace(0, 199999, 65536, dtype=np.int64))
    assert result['samples']['train']['n'] == 65536
    assert result['warmup']['samples']['train']['n'] == 8192
    assert result['optimizer_updates'] == 1024 and result['warmup']['optimizer_updates'] == 32
    assert result['projection']['load_seconds'] == 4.
    assert result['dev_scores_read'] is False and result['dev_features_gathered'] is False
    assert result['profile_weights_reused_as_full_member'] is False
    assert 'log_loss' not in str(result) and 'must not be emitted' not in str(result)
    batches['train'] = Batch(8192)
    with pytest.raises(ValueError, match='do not shrink'):
        profile_module.profile_batches(batches, SimpleNamespace(draws=400), device='cpu')
