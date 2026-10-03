"""Synthetic CPU tests for the DRAFT direct-head frontier arms (no real data, no fit on real rows)."""
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pitchmdp import direct_frontier_model as fm
from pitchmdp.archetypes import RELIABILITY_COLUMNS, STYLE_COLUMNS
from pitchmdp.direct_periodic_model import context_map, parameter_count as periodic_count
from pitchmdp.direct_residual_model import parameter_count as residual_count
from pitchmdp.matrix_models import MatrixModel, _flat
from pitchmdp.matrix_sharing import CLUSTER_FEATURES
from pitchmdp.sequence_model import SequenceContext

REPO = Path(__file__).resolve().parents[3]
CONFIG = REPO / 'configs' / 'DRAFT-CUDA-FRONTIER-20261004.json'
CLUSTER_COLUMNS = [*CLUSTER_FEATURES, 'left_share', *(f'type_{t}' for t in 'ABCDEFGHIJKLMNOPQR')]  # 4 + 1 + 18
NAMES = fm.direct_context_names(CLUSTER_COLUMNS)
FIT = dict(epochs=2, patience=5, batch_size=64, learning_rate=.0005)


def arrays(n, seed):
    rng = np.random.default_rng(seed)
    tokens = rng.normal(size=(n, 6, 38)).astype(np.float32)
    tokens[:, -1, :8] = 0.
    tokens[:, -1, -11:] = 0.
    valid = np.ones((n, 6), dtype=bool)
    valid[:, 0] = rng.random(n) > .5
    return (tokens, valid, rng.normal(size=(n, 52)).astype(np.float32)), rng.integers(0, 10, n)


def fitted(model, seed=0):
    (x, y), (e, ey) = arrays(256, seed), arrays(96, seed + 1)
    return model.fit(x, y, e, ey, **FIT)


def test_context_schema_is_the_frozen_layout():
    base = SequenceContext.report(SimpleNamespace(archetypes=SimpleNamespace(report=dict)))['features']
    assert list(fm.GAME_STATE) == base[:11]
    assert fm.BATTER_STYLE == STYLE_COLUMNS and fm.BATTER_RELIABILITY == RELIABILITY_COLUMNS
    assert 'Archetypes(n_clusters=5' in inspect.getsource(SequenceContext.fit)
    assert fm.PITCHER_CLUSTER_FEATURES == CLUSTER_FEATURES and len(NAMES) == 52
    layout = context_map(CLUSTER_COLUMNS)  # periodic arm's independent derivation of the same 52 columns
    assert layout['n_context'] == 52 and layout['raw'][11:] == [32]
    assert [NAMES.index(c) for c in fm.ARMS['no_batter']['drop_context']] == list(range(11, 28))
    assert [NAMES.index(c) for c in fm.ARMS['no_batter_cluster']['drop_context']] == list(range(23, 28))
    with pytest.raises(ValueError):
        fm.direct_context_names(CLUSTER_COLUMNS[1:])


def test_capacity_arms_match_parameter_counts():
    residual, periodic = residual_count(52, 38, 6), periodic_count(52, 38, 6, context_map(CLUSTER_COLUMNS)['continuous'])
    assert (residual, periodic) == (170634, 286738) == (fm.ARMS['width145']['target_parameter_count'],
                                                        fm.ARMS['width203']['target_parameter_count'])
    for arm in ('width145', 'width203'):
        model = fitted(fm.build_frontier(arm, 0, 'cpu'))
        assert model.report['parameter_count'] == fm.flatten_parameter_count(fm.ARMS[arm]['width'])
        assert abs(model.report['parameter_count'] / fm.ARMS[arm]['target_parameter_count'] - 1) < .01
    assert fm.flatten_parameter_count(128) == 141834


@pytest.mark.parametrize('arm', list(fm.ARMS))
def test_every_arm_trains_saves_loads(arm, tmp_path):
    model = fitted(fm.build_frontier(arm, 1, 'cpu', NAMES))
    probe, _ = arrays(40, 9)
    model.save(tmp_path / 'm.pt')
    loaded = fm.load_frontier(arm, tmp_path / 'm.pt', 'cpu')
    np.testing.assert_array_equal(model.logits(probe), loaded.logits(probe))
    assert loaded.report['parameter_count'] == model.report['parameter_count']
    assert model.report['parameter_count'] == fm.flatten_parameter_count(fm.ARMS[arm]['width'])
    other = next(a for a in fm.ARMS if a != arm and fm.ARMS[a]['model'] == fm.ARMS[arm]['model'])
    with pytest.raises(ValueError):
        fm.load_frontier(other, tmp_path / 'm.pt', 'cpu')


def test_empty_mask_and_neutral_regularization_reproduce_control_exactly():
    probe, _ = arrays(40, 9)
    control = fitted(MatrixModel('flatten_mlp', seed=2, width=128, device='cpu'), 3).logits(probe)
    masked = fitted(fm.MaskedFlattenModel(2, (), False, NAMES, 'cpu'), 3).logits(probe)
    neutral = fitted(fm.FlattenRegularizedModel(2, 0., .01, 'cpu'), 3).logits(probe)
    np.testing.assert_array_equal(control, masked)
    np.testing.assert_array_equal(control, neutral)
    assert not np.array_equal(control, fitted(fm.build_frontier('weight_decay', 2, 'cpu'), 3).logits(probe))
    assert not np.array_equal(control, fitted(fm.build_frontier('dropout', 2, 'cpu'), 3).logits(probe))


@pytest.mark.parametrize('arm,zeroed', [('no_batter', 17), ('no_batter_cluster', 5), ('no_history', 5 * 39)])
def test_masks_zero_exactly_the_named_inputs(arm, zeroed):
    model = fm.build_frontier(arm, 0, 'cpu', NAMES)
    raw, _ = arrays(64, 4)
    masked = model.transform(raw)
    before, after = _flat(raw), _flat(masked)
    changed = np.flatnonzero((before != after).any(0))
    history = [j * 38 + c for j in range(5) for c in range(38)] + [228 + j for j in range(5)]
    expected = history if arm == 'no_history' else [234 + i for i in model.drop_index]
    assert changed.tolist() == expected and len(expected) == zeroed and (after[:, changed] == 0).all()
    np.testing.assert_array_equal(masked[0][:, -1], raw[0][:, -1])  # current action untouched
    assert masked[1][:, -1].all()
    if arm != 'no_history':
        np.testing.assert_array_equal(masked[0], raw[0])
        np.testing.assert_array_equal(masked[2][:, :11], raw[2][:, :11])
        np.testing.assert_array_equal(masked[2][:, 28:], raw[2][:, 28:])
    else:
        np.testing.assert_array_equal(masked[2], raw[2])
    assert fm.MaskedFlattenModel.transform(model, masked)[2].tobytes() == masked[2].tobytes()  # idempotent
    model = fitted(model)
    assert model.report['zeroed_flat_inputs'] == zeroed
    assert model.report['first_layer_weights_without_data_gradient'] == zeroed * 256
    shifted = (raw[0], raw[1], raw[2].copy())
    shifted[2][:, model.drop_index] += 7.
    if model.drop_history:
        shifted = (raw[0] + np.where(np.arange(6)[None, :, None] < 5, 3., 0.).astype(np.float32), raw[1], raw[2])
    np.testing.assert_array_equal(model.logits(raw), model.logits(shifted))  # predict applies the same mask


def test_mask_guards():
    with pytest.raises(ValueError):
        fm.build_frontier('no_batter', 0, 'cpu')
    with pytest.raises(ValueError):
        fm.MaskedFlattenModel(0, ['not_a_column'], False, NAMES, 'cpu')
    with pytest.raises(ValueError):
        fm.MaskedFlattenModel(0, ['balls'], False, ['x'] + NAMES[1:], 'cpu')
    model = fm.build_frontier('no_batter', 0, 'cpu', NAMES)
    (t, v, c), _ = arrays(4, 0)
    with pytest.raises(ValueError):
        model.transform((t, v, c[:, :51]))
    with pytest.raises(ValueError):
        fm.build_frontier('width256', 0, 'cpu')


def test_config_guard():
    config = json.loads(CONFIG.read_text())
    assert fm.check_config(config) is config and len(config['arms']) * len(config['seeds']) == config['fits'] == 24
    assert config['execution']['enabled'] is False
    for change in (lambda c: c['arms'][0].update(width=160), lambda c: c['arms'].pop(),
                   lambda c: c['training'].update(epochs=40), lambda c: c.update(seeds=[0, 1]),
                   lambda c: c['arms'][5].update(dropout=.2), lambda c: c['arms'].append(dict(c['arms'][0]))):
        bad = json.loads(CONFIG.read_text())
        change(bad)
        with pytest.raises(ValueError):
            fm.check_config(bad)
