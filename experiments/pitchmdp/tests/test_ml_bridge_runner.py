"""F1 bridge runner contracts on tiny synthetic CPU data; no real data or DEV score."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.special import softmax

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'scripts'), str(Path(__file__).parent)]
from pitchmdp.matrix_bridge import (BASE_FEATURES, BatterMaskedModel, batter_groups, batter_train_volume,
                                    check_context_layout, fit_masked, mask_batter_context, masked_g0_predictor,
                                    masked_training_arrays, network_signature)
from pitchmdp.matrix_models import MatrixModel
from pitchmdp.matrix_sharing import SharingPredictor
from score_ml_matrix import summarize_cell
import run_ml_bridge as runner
from bridge_synthetic import synthetic_family


def valid_config():
    config = json.loads((REPO / 'configs' / 'EXP-P9-001.yaml').read_text())
    config['parent_preparation_sha256'] = 'a' * 64
    config['parent_analysis_sha256'] = 'b' * 64
    return config


def sharing_arrays(n=12, seed=0):
    rng = np.random.default_rng(seed)
    tokens = rng.normal(size=(n, 6, 21)).astype(np.float32)
    tokens[:, -1, -11:] = 0
    valid = np.ones((n, 6), dtype=bool)
    context = rng.normal(size=(n, 59)).astype(np.float32)
    context[:, -2] = rng.integers(0, 4, n)
    context[:, -1] = rng.integers(1, 1000, n)
    return tokens, valid, context


def test_mask_zeroes_exactly_batter_channels_and_copies():
    context = np.random.default_rng(1).normal(size=(5, 52)).astype(np.float32)
    original = context.copy()
    masked = mask_batter_context(context)
    assert np.array_equal(context, original)
    assert (masked[:, 11:28] == 0).all()
    assert np.array_equal(masked[:, :11], original[:, :11])
    assert np.array_equal(masked[:, 28:], original[:, 28:])
    assert masked.shape == (5, 52) and masked.dtype == original.dtype
    for width in (28, 51, 53, 59):
        with pytest.raises(ValueError):
            mask_batter_context(np.zeros((2, width)))


def test_training_arrays_strip_routing_then_mask():
    tokens, valid, context = sharing_arrays()
    t, v, c = masked_training_arrays((tokens, valid, context))
    assert t is tokens and v is valid
    assert c.shape == (12, 52)
    assert np.array_equal(c[:, :11], context[:, :11])
    assert (c[:, 11:28] == 0).all()
    assert np.array_equal(c[:, 28:52], context[:, 28:52])


class RecordingModel:
    kind, seed, report = 'flatten_mlp', 0, {}

    def __init__(self):
        rng = np.random.default_rng(3)
        self.weight = rng.normal(size=(52, 10)).astype(np.float32)
        self.seen = []

    def logits(self, arrays):
        self.seen.append(arrays[2].copy())
        return arrays[2] @ self.weight


def test_masked_predictor_uses_same_g0_wrapper_path_and_never_sees_batter_values():
    arrays = sharing_arrays()
    inner = RecordingModel()
    masked = masked_g0_predictor(inner, {'cluster_counts': {}, 'pitcher_counts': {}})
    result = masked.logits(arrays)
    assert inner.seen[-1].shape == (12, 52) and (inner.seen[-1][:, 11:28] == 0).all()
    # Full G0 wrapper on pre-masked input gives bitwise-identical log probabilities.
    premasked = (arrays[0], arrays[1], arrays[2].copy())
    premasked[2][:, 11:28] = 0
    full = SharingPredictor('G0-global', RecordingModel(), {})
    assert np.array_equal(result, full.logits(premasked))
    expected = np.log(np.clip(softmax((premasked[2][:, :52] @ inner.weight).astype(float), axis=1), 1e-30, 1))
    assert np.array_equal(result, expected)
    # Any change in batter channels is invisible; other channels still matter.
    changed = (arrays[0], arrays[1], arrays[2].copy())
    changed[2][:, 11:28] += 100
    assert np.array_equal(masked.logits(changed), result)
    changed[2][:, 3] += 1
    assert not np.array_equal(masked.logits(changed), result)
    assert masked.report['masked_context_channels'] == [11, 28]
    assert masked.delivery_temperature == 1.


def test_masked_fit_is_invariant_to_batter_values_and_matches_full_shape():
    tokens, valid, context = sharing_arrays(n=96, seed=4)
    labels = np.arange(96) % 10
    other = context.copy()
    other[:, 11:28] = np.random.default_rng(9).normal(size=(96, 17)) * 50
    kwargs = dict(epochs=2, patience=2, batch_size=32, learning_rate=.0005)
    a = fit_masked(MatrixModel('flatten_mlp', seed=1, width=8, device='cpu'),
                   (tokens, valid, context), labels, (tokens[:32], valid[:32], context[:32]), labels[:32], **kwargs)
    b = fit_masked(MatrixModel('flatten_mlp', seed=1, width=8, device='cpu'),
                   (tokens, valid, other), labels, (tokens[:32], valid[:32], other[:32]), labels[:32], **kwargs)
    for key, value in a.net.state_dict().items():
        assert np.array_equal(value.numpy(), b.net.state_dict()[key].numpy()), key
    full = MatrixModel('flatten_mlp', seed=1, width=8, device='cpu').fit(
        (tokens, valid, context[:, :52]), labels, (tokens[:32], valid[:32], context[:32, :52]), labels[:32], **kwargs)
    assert network_signature(a) == network_signature(full)
    assert network_signature(a)['network']['n_context'] == 52
    wrapped = BatterMaskedModel(a)
    assert np.array_equal(wrapped.logits((tokens, valid, context[:, :52])),
                          wrapped.logits((tokens, valid, other[:, :52])))


def test_initialization_shape_is_identical_for_full_and_masked():
    import torch
    from pitchmdp.matrix_models import MatrixNetwork
    torch.manual_seed(0)
    first = MatrixNetwork('flatten_mlp', 52, 21, 6, 128)
    torch.manual_seed(0)
    second = MatrixNetwork('flatten_mlp', 52, 21, 6, 128)
    for key, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[key])


def test_batter_volume_linear_q25_train_only_and_inclusive_boundary():
    train = np.repeat([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    volume = batter_train_volume(train)
    assert volume['q25'] == 2.0 and volume['positive_batters'] == 5
    groups, counts = batter_groups([1, 2, 3, 99], volume['counts'], volume['q25'])
    assert groups.tolist() == ['low', 'low', 'high', 'zero']
    assert counts.tolist() == [1, 2, 3, 0]
    with pytest.raises(ValueError):
        batter_train_volume([])


def test_config_template_matches_code_and_requires_real_hashes():
    template = json.loads((REPO / 'configs' / 'EXP-P9-001.yaml').read_text())
    assert runner.config_check(template)['experiment_id'] == 'EXP-P9-001'
    unassigned = dict(template, parent_preparation_sha256='ROOT_TO_ASSIGN',
                      parent_analysis_sha256='ROOT_TO_ASSIGN')
    with pytest.raises(ValueError, match='SHA256'):
        runner.config_check(unassigned)
    assert runner.config_check(valid_config())['experiment_id'] == 'EXP-P9-001'
    for key, value in [('mask', {'start': 11, 'stop': 29}), ('seeds', [0, 1]), ('draws', 25),
                       ('width', 64), ('parent_cell', 'G2-feature'), ('context_width', 59),
                       ('parent_experiment_id', 'EXP-P4-002'), ('device', 'cuda')]:
        changed = valid_config(); changed[key] = value
        with pytest.raises(ValueError):
            runner.config_check(changed)
    for path, value in [(('budget', 'epochs'), 10), (('limits', 'member_wall_limit_seconds'), 9000),
                        (('robustness', 'family_size'), 96), (('decision', 'delta_nll_max'), -.001),
                        (('bootstrap', 'draws'), 1000), (('profile', 'train_rows'), 8192)]:
        changed = valid_config(); changed[path[0]][path[1]] = value
        with pytest.raises(ValueError):
            runner.config_check(changed)
    changed = valid_config(); changed['silent_tuning'] = True
    with pytest.raises(ValueError):
        runner.config_check(changed)


def test_full_reconstruction_accepts_exact_and_rejects_changed_predictions():
    full, _, baseline = synthetic_family()
    report, values = summarize_cell(full, baseline)
    stored = {name: baseline['dev_' + name] for name in ('keys', 'y', 'game_pk', 'pitcher')}
    stored.update({'G0-global_' + k: v for k, v in values.items()})
    frozen = json.loads(json.dumps(report))
    runner.verify_full_reconstruction(full, baseline, stored, frozen)
    for kind in ('primary', 'calibrated', 'raw', 'seed_primary'):
        altered = {k: v.copy() for k, v in stored.items()}
        altered['G0-global_' + kind].flat[0] += 1e-12
        with pytest.raises(ValueError, match='reconstruction'):
            runner.verify_full_reconstruction(full, baseline, altered, frozen)
    weight = deepcopy(frozen); weight['selection']['model_weight'] += 1e-9
    with pytest.raises(ValueError, match='June'):
        runner.verify_full_reconstruction(full, baseline, stored, weight)
    seed_weight = deepcopy(frozen); seed_weight['seeds'][1]['blend_selection']['model_weight'] = 0.
    with pytest.raises(ValueError, match='June'):
        runner.verify_full_reconstruction(full, baseline, stored, seed_weight)
    tampered = [{k: v.copy() for k, v in m.items()} for m in full]
    tampered[0]['dev'] = tampered[0]['dev'][::-1].copy()
    with pytest.raises(ValueError):
        runner.verify_full_reconstruction(tampered, baseline, stored, frozen)
    with pytest.raises(ValueError):
        runner.verify_full_reconstruction(full[:2], baseline, stored, frozen)


def test_ledger_records_completed_failed_and_charges_killed_attempts(tmp_path):
    with runner.ledger_stage(tmp_path, 'prepare'):
        elapsed, active = runner.active_elapsed(tmp_path)
        assert active is not None and elapsed >= 0
        assert runner.ledger_total(tmp_path, exclude=active) == 0.
    with pytest.raises(RuntimeError):
        with runner.ledger_stage(tmp_path, 'fit', 1):
            raise RuntimeError('interrupted')
    attempts = runner.ledger_attempts(tmp_path)
    assert [a['status'] for a in attempts] == ['completed', 'failed']
    assert attempts[1]['seed'] == 1 and 'interrupted' in attempts[1]['error']
    closed = sum(a['seconds'] for a in attempts)
    assert runner.ledger_total(tmp_path) == pytest.approx(closed)
    # A SIGKILLed command leaves only a start; it is charged until the next start, capped.
    killed = {'event': 'start', 'id': 'k1', 'stage': 'fit', 'seed': 2, 'unix': 1000., 'utc': 'x'}
    runner._append_ledger(tmp_path, killed)
    assert runner.ledger_total(tmp_path) == pytest.approx(closed + 7200)
    runner._append_ledger(tmp_path, {'event': 'start', 'id': 'k2', 'stage': 'profile', 'seed': None,
                                     'unix': 1300., 'utc': 'y'})
    assert runner.ledger_total(tmp_path) == pytest.approx(closed + 300 + 600)
    assert runner.ledger_attempts(tmp_path)[-1]['status'] == 'unterminated'
    with pytest.raises(ValueError):
        with runner.ledger_stage(tmp_path, 'tune'):
            pass


def test_family_budget_counts_running_command_and_projection(tmp_path):
    runner._append_ledger(tmp_path, {'event': 'start', 'id': 'a', 'stage': 'fit', 'seed': 0, 'unix': 0., 'utc': 'x'})
    runner._append_ledger(tmp_path, {'event': 'end', 'id': 'a', 'status': 'completed', 'seconds': 10000., 'utc': 'x'})
    with runner.ledger_stage(tmp_path, 'fit', 1):
        assert runner.check_family_budget(tmp_path, 4000.) >= 10000.
        with pytest.raises(ValueError, match='14400'):
            runner.check_family_budget(tmp_path, 4500.)


def test_profile_gate_and_registered_output_and_input_shape(tmp_path):
    (tmp_path / 'preparation.json').write_text('{}')
    with pytest.raises(ValueError, match='profile'):
        runner._profile_projection(tmp_path)
    config = valid_config()
    runner.check_registered_output(config, Path(config['registration']['output']).resolve())
    with pytest.raises(ValueError, match='registration'):
        runner.check_registered_output(config, tmp_path)
    tokens, valid, context = masked_training_arrays(sharing_arrays())
    network = {'kind': 'flatten_mlp', 'n_context': 52, 'n_token': 21, 'length': 6, 'width': 128, 'n_classes': 10}
    assert runner.check_input_shape((tokens, valid, context), network)['n_context'] == 52
    with pytest.raises(ValueError, match='differs'):
        runner.check_input_shape((tokens, valid, context), {**network, 'n_context': 59})


class _Archetypes:
    def numeric_features(self, frame):
        return np.zeros((len(frame), 17))


class _Base:
    archetypes = _Archetypes()

    def report(self):
        return {'features': list(BASE_FEATURES)}

    def transform(self, frame):
        return np.zeros((len(frame), 28))


class _Sharing:
    clusters = {'columns': ['c%d' % i for i in range(23)]}

    def transform(self, frame):
        return np.zeros((len(frame), 59))


def test_context_layout_check_pins_feature_order_and_widths():
    frame = list(range(4))
    layout = check_context_layout(_Base(), _Sharing(), frame)
    assert layout['batter'] == [11, 28] and layout['pitcher'] == [28, 52] and layout['routing'] == [52, 59]
    reordered = _Base(); reordered.report = lambda: {'features': list(reversed(BASE_FEATURES))}
    with pytest.raises(ValueError, match='order'):
        check_context_layout(reordered, _Sharing(), frame)
    narrow = _Sharing(); narrow.clusters = {'columns': ['c'] * 22}
    with pytest.raises(ValueError, match='widths'):
        check_context_layout(_Base(), narrow, frame)


def test_cost_projection_gates_member_and_family_budgets():
    samples = {k: v for k, v in runner.EXPECTED_SAMPLES.items() if k != 'dev_games'}
    small = {'command_overhead_seconds': 5., 'load_seconds': 10., 'fit_seconds': 2., 'calibration_seconds': .5,
             'inference_seconds': .5, 'wall_seconds': 30.}
    projection = runner.project_costs(small, samples, 100.)
    assert projection['member_gate'] and projection['family_gate']
    assert projection['member_seconds'] == pytest.approx(projection['fit_command_seconds'] + projection['predict_command_seconds'])
    large = {**small, 'fit_seconds': 60.}
    assert not runner.project_costs(large, samples, 100.)['member_gate']
    borderline = {**small, 'fit_seconds': 20.}
    result = runner.project_costs(borderline, samples, 100.)
    assert result['family_seconds'] == pytest.approx(100. + 30. + 3 * result['member_seconds'])
