"""Synthetic CPU contracts for ML2 preparation, identities and streaming inference."""
from copy import deepcopy
import importlib.util
import os
import subprocess
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'scripts'))
from pitchmdp.data import KEY, hash_file
from pitchmdp.matrix_benchmark import (CELLS, NEURAL, LIGHTGBM, validate_config, select_keys,
                                       integrated_probabilities, member_identity, predict_streamed)
from pitchmdp.matrix_data import ordered_key_hash
from pitchmdp.matrix_models import MatrixModel
import run_ml_benchmark as runner


def config():
    return {'protocol': 'ml_architecture_v1', 'experiment_id': 'EXP-TEST',
            'parent_run': '/temporary/parent', 'parent_preparation_sha256': 'a' * 64,
            'seeds': [0, 1, 2], 'data_sample': 'd100', 'history_length': 5,
            'draws': 400, 'device': 'cpu', 'width': 128,
            'neural': deepcopy(NEURAL), 'lightgbm': deepcopy(LIGHTGBM)}


def test_config_locks_representative_configs_and_population():
    assert validate_config(config())['data_sample'] == 'd100'
    for key, bad in [('seeds', [0, 1]), ('history_length', 32), ('draws', 25),
                     ('data_sample', 'd25'), ('parent_preparation_sha256', 'oops')]:
        changed = config()
        changed[key] = bad
        with pytest.raises(ValueError):
            validate_config(changed)
    changed = config()
    changed['lightgbm']['learning_rate'] = .2
    with pytest.raises(ValueError):
        validate_config(changed)


def test_exact_ordered_key_selection_and_member_identity():
    frame = pd.DataFrame({'game_pk': [5, 6, 7], 'at_bat_number': [1] * 3, 'pitch_number': [1] * 3})
    keys = frame.iloc[[2, 0]][KEY]
    record = {'n': 2, 'rows_sha256': ordered_key_hash(keys)}
    selected = select_keys(frame, keys, record)
    assert selected.game_pk.tolist() == [7, 5]
    with pytest.raises(ValueError):
        select_keys(frame.iloc[:2], keys, record)
    with pytest.raises(ValueError):
        select_keys(frame, keys.iloc[::-1], record)
    preparation = {'samples': {'train': record}, 'features': {'history_length': 5}}
    member = member_identity(preparation, 'A0-MLP', 0)
    assert member['train_rows_sha256'] == record['rows_sha256']
    assert member_identity(preparation, 'A0-MLP', 1) != member
    with pytest.raises(ValueError):
        member_identity(preparation, 'A0-MLP', 42)


def test_float64_draw_reduction_and_streamed_equivalence():
    rng = np.random.default_rng(321)
    logits = rng.normal(size=(5, 400, 10)).astype(np.float32)
    p, raw = integrated_probabilities(logits, 1.31)
    assert p.dtype == np.float64
    np.testing.assert_allclose(p.sum(1), 1., rtol=0, atol=1e-14)
    class Model:
        delivery_temperature = 1.31
    class Delivery:
        draws = 400
        def logits(self, model, store, context, rows):
            return logits[rows], np.zeros(len(rows), dtype=int)
    streamed = predict_streamed(Model(), Delivery(), None, None, np.arange(5), chunk_size=2)
    np.testing.assert_array_equal(streamed[0], p)
    np.testing.assert_array_equal(streamed[1], raw)
    with pytest.raises(ValueError):
        integrated_probabilities(logits, 0)
    with pytest.raises(ValueError):
        integrated_probabilities(logits[:, :, :9], 1)


def test_completed_prepare_checks_hashes_without_loading_data(tmp_path):
    expected = {'config_sha256': 'test'}
    payload = tmp_path / 'features.json'
    payload.write_text('{}')
    runner.dump(tmp_path / 'preparation.json', {'identity': expected,
                'artifact_hashes': {'features.json': hash_file(payload)}})
    assert runner.verify(tmp_path, expected)['identity'] == expected
    payload.write_text('{"changed": true}')
    with pytest.raises(ValueError, match='identity'):
        runner.verify(tmp_path, expected)


def test_parent_rejects_changed_registration_before_data_access(tmp_path):
    local = {'artifact_root': str(tmp_path)}
    parent = tmp_path / 'runs/ML-MATRIX-20260924/parent'
    parent.mkdir(parents=True)
    (parent / 'preparation.json').write_text('{}')
    cfg = config()
    cfg['parent_run'] = str(parent)
    with pytest.raises(ValueError, match='registration'):
        runner.parent_preparation(cfg, local, parent.parent / 'benchmark')


@pytest.mark.skipif(importlib.util.find_spec('lightgbm') is None, reason='optional isolated LightGBM dependency absent')
def test_lightgbm_tiny_cpu_fit_roundtrip_and_joint_logits(tmp_path):
    # Isolate native dependencies; setting library search paths after importing
    # torch in this pytest process would be too late on macOS.
    code = """
from pathlib import Path
import sys
import numpy as np
from pitchmdp.matrix_models import MatrixModel
rng = np.random.default_rng(7)
tokens = rng.normal(size=(20, 6, 23)).astype(np.float32)
tokens[:, -1, -11:] = 0.
arrays = tokens, np.ones((20, 6), dtype=bool), rng.normal(size=(20, 4)).astype(np.float32)
labels = np.arange(20) % 10
model = MatrixModel('lightgbm', seed=0, device='cpu',
                    lightgbm_params={'min_data_in_leaf': 1, 'min_data_in_bin': 1, 'num_leaves': 3})
model.fit(arrays, labels, arrays, labels, epochs=2, patience=1, learning_rate=.05)
logits = model.logits(arrays)
assert logits.shape == (20, 10)
np.testing.assert_allclose(model.predict(arrays).sum(1), 1., atol=1e-12)
model.save(Path(sys.argv[1]))
restored = MatrixModel.load(Path(sys.argv[1]), device='cpu')
np.testing.assert_array_equal(logits, restored.logits(arrays))
"""
    env = dict(os.environ)
    env['PYTHONPATH'] = str(PROJECT) + os.pathsep + env.get('PYTHONPATH', '')
    env['DYLD_LIBRARY_PATH'] = str(Path(runner.torch.__file__).resolve().parent / 'lib')
    result = subprocess.run([sys.executable, '-c', code, str(tmp_path / 'tree.pt')],
                            env=env, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(CELLS) == 7
