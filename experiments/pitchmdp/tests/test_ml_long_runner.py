"""Configuration/profile gates only; no real-data loading or resource profiles."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'scripts'))
from pitchmdp.data import hash_file
from pitchmdp.matrix_long_experiment import DEFAULT_BUDGET
import pitchmdp.matrix_long_profile as profile_module
import run_ml_long_history as runner


def config():
    return {'protocol': 'ml_long_history_v1', 'experiment_id': 'EXP-TEST',
            'parent_run': '/frozen/sharing', 'parent_preparation_sha256': 'a' * 64,
            'seeds': [0, 1, 2], 'width': 128, 'budget': deepcopy(DEFAULT_BUDGET),
            'draws': 400, 'device': 'cpu'}


def test_strict_config_fixes_capacity_draws_seeds_and_budget():
    assert runner.config_check(config())['width'] == 128
    for key, value in [('seeds', [0]), ('width', 64), ('draws', 25), ('parent_preparation_sha256', '')]:
        changed = config(); changed[key] = value
        with pytest.raises(ValueError): runner.config_check(changed)
    changed = config(); changed['budget']['batch_size'] = 1024
    with pytest.raises(ValueError): runner.config_check(changed)
    changed = config(); changed['silent_extra_axis'] = True
    with pytest.raises(ValueError): runner.config_check(changed)


def test_prepare_resume_checks_immutable_artifacts(tmp_path):
    expected = {'config_sha256': 'frozen'}
    path = tmp_path / 'features.json'; path.write_text('{}')
    runner.dump(tmp_path / 'preparation.json', {'identity': expected,
                 'artifact_hashes': {'features.json': hash_file(path)}})
    assert runner.verify(tmp_path, expected)['identity'] == expected
    path.write_text('{"changed": 1}')
    with pytest.raises(ValueError): runner.verify(tmp_path, expected)


def test_matching_profile_gate_requires_affordable_projection(tmp_path):
    runner.dump(tmp_path / 'preparation.json', {'frozen': True})
    destination = tmp_path / 'profiles/F4-128'; destination.mkdir(parents=True)
    identity = {'preparation_sha256': hash_file(tmp_path / 'preparation.json'), 'cell': 'F4-128'}
    runner.dump(destination / 'profile.json', {'projection': {'within_limit': False}})
    runner.dump(destination / 'manifest.json', {'identity': identity,
                 'artifact_hashes': {'profile.json': hash_file(destination / 'profile.json')}})
    with pytest.raises(ValueError, match='7200'): runner.verify_profile(tmp_path, 'F4-128')
    runner.dump(destination / 'profile.json', {'projection': {'within_limit': True}})
    runner.dump(destination / 'manifest.json', {'identity': identity,
                 'artifact_hashes': {'profile.json': hash_file(destination / 'profile.json')}})
    runner.verify_profile(tmp_path, 'F4-128')
    runner.dump(tmp_path / 'preparation.json', {'changed': True})
    with pytest.raises(ValueError, match='matching'): runner.verify_profile(tmp_path, 'F4-128')


def test_profile_helper_never_gathers_dev_and_omits_quality_scores(monkeypatch):
    class Batch:
        def __init__(self, n, ids=None):
            self.ids = np.arange(n) if ids is None else ids
            self.store = SimpleNamespace(long_length=128)
        def __len__(self): return len(self.ids)
        def subset(self, indices): return Batch(0, self.ids[indices])
        def frame(self):
            return pd.DataFrame({'game_pk': self.ids + 1, 'at_bat_number': 1, 'pitch_number': 1,
                                 'description': 'ball', 'events': None, 'strikes': 0})
    class HiddenBatch:
        def __len__(self): return 100
        def frame(self): raise AssertionError('DEV must not be gathered by a profile')
        def subset(self, indices): raise AssertionError('DEV must not be gathered by a profile')
    class Model:
        def __init__(self, **kwargs):
            self.device = 'cpu'
            self.net = SimpleNamespace(config={'width': 128})
            self.report = {'parameter_count': 500, 'max_expanded_training_rows': 256,
                           'history': 'must not be emitted', 'log_loss': 'must not be emitted'}
        def fit(self, train, y, early, ey, **kwargs):
            assert len(train) == 8192 and len(early) == 2048
            assert kwargs == {'epochs': 2, 'patience': 2, 'batch_size': 256, 'learning_rate': .0005}
            return self
    class Delivery:
        def __init__(self, delivery): assert delivery.draws == 400
        def calibrate(self, model, temp, y): assert len(temp) == 16
        def predict(self, model, temp):
            p = np.full((len(temp), 10), .1)
            return p, p, np.zeros(len(temp), dtype=int)
    monkeypatch.setattr(profile_module, 'LazyMatrixModel', Model)
    monkeypatch.setattr(profile_module, 'LazyJointDelivery', Delivery)
    batches = {'train': Batch(20000), 'earlystop': Batch(5000), 'temperature': Batch(64),
               'blend': HiddenBatch(), 'dev': HiddenBatch()}
    result = profile_module.profile_batches(batches, SimpleNamespace(draws=400), device='cpu')
    assert result['samples']['train']['n'] == 8192
    assert result['samples']['earlystop']['n'] == 2048
    assert result['samples']['temperature']['n'] == 16
    assert result['dev_scores_read'] is False and result['dev_features_gathered'] is False
    assert 'log_loss' not in str(result) and 'must not be emitted' not in str(result)
