"""Fail closed on incomplete or unpaired matrix comparisons."""
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from score_ml_matrix import assert_aligned, require_complete, summarize_cell


def sample():
    result = {}
    for part in ('blend', 'dev'):
        result[part + '_keys'] = np.array([[10, 1, 1], [11, 2, 1]])
        result[part + '_game_pk'] = np.array([10, 11])
        result[part + '_pitcher'] = np.array([9, 8])
        result[part + '_y'] = np.array([0, 1])
        result[part] = np.full((2, 10), .1)
        result[part + '_raw'] = result[part].copy()
    return result


def test_metadata_alignment_rejects_reordering_and_wrong_labels():
    baseline = sample()
    assert_aligned(baseline, baseline)
    for name in ('dev_keys', 'dev_y', 'blend_pitcher', 'dev_game_pk'):
        broken = sample()
        broken[name] = broken[name][::-1]
        with pytest.raises(ValueError, match='Unpaired'):
            assert_aligned(broken, baseline)


def test_duplicate_pitch_keys_rejected():
    data = sample()
    data['dev_keys'][1] = data['dev_keys'][0]
    with pytest.raises(ValueError, match='Repeated'):
        assert_aligned(data, data)


def test_family_requires_every_declared_seed(tmp_path):
    for seed in (0, 1):
        path = tmp_path / 'members' / 'A' / f'seed{seed}' / 'prediction_state.json'
        path.parent.mkdir(parents=True)
        path.write_text('{}')
    with pytest.raises(ValueError, match='A/seed2'):
        require_complete(tmp_path, ['A'], [0, 1, 2])
    require_complete(tmp_path, ['A'], [0, 1])


def test_blend_selection_uses_only_calibration_outcomes():
    baseline = sample()
    member = sample()
    member['blend'][:] = .01
    member['blend'][[0, 1], [0, 1]] = .91
    a, p = summarize_cell([member] * 3, baseline)
    baseline['dev_y'] = np.array([5, 6])
    b, q = summarize_cell([member] * 3, baseline)
    assert a['selection'] == b['selection']
    np.testing.assert_array_equal(p['primary'], q['primary'])
    assert a['selection']['model_weight'] == 1
