"""B2 (LightGBM) checks of EXP-BC-CAL-01. Not collected by the default run (file name); executed by
test_behavior_calibration_model.test_lightgbm_b2_paths_run_with_one_openmp_runtime in a fresh
interpreter with DYLD_LIBRARY_PATH=<torch/lib> so torch and LightGBM share one OpenMP runtime."""
import numpy as np
import pytest

from behavior_calibration_synthetic import make_world
from pitchmdp.behavior_calibration import Temperature
from pitchmdp.behavior_calibration_registry import ACTION_CODES, B2_FEATURES, B2_PARAMS
from pitchmdp.behavior_model import BehaviorModel, Calibrated, LGBMBehavior
from pitchmdp.policy_artifacts import IntegrityError
from pitchmdp.rollout_policy import PAState, PastPitch

PARAMS = {**B2_PARAMS, 'min_data_in_leaf': 20}  # synthetic size; the registered 100 is pinned for real runs


@pytest.fixture(scope='module')
def world():
    return make_world(0)


def fit(world, rounds=12):
    train, es = world['tables']['train'], world['tables']['earlystop']
    rows = np.flatnonzero(es.frame.S_in.to_numpy())
    mask = world['b0'].repertoire.mask(es.keys.values['pitcher'][rows])
    model, best, record = LGBMBehavior.train(train.matrix(B2_FEATURES, None), train.frame.label.to_numpy(),
                                             es.matrix(B2_FEATURES, rows), es.frame.label.to_numpy()[rows], mask,
                                             params=PARAMS, rounds=rounds)
    return LGBMBehavior(model, best, world['b0'].repertoire, record), record, rows, mask


def test_b2_round_rule_masking_and_reload(world):
    b2, record, rows, mask = fit(world)
    assert len(record['curve']) == 12 and record['best_round'] == int(np.argmin(record['curve'])) + 1
    assert record['cap_hit'] == (record['best_round'] == 12)
    es = world['tables']['earlystop']
    p = b2.batch(es, rows)
    assert (p[~mask] == 0).all() and np.allclose(p.sum(axis=1), 1, rtol=0, atol=1e-12)
    y = es.frame.label.to_numpy()[rows]
    nll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()
    assert abs(nll - record['curve'][b2.best_round - 1]) < 1e-6  # feval = masked earlystop S_in NLL per round
    again = LGBMBehavior(b2.model_string, b2.best_round, world['b0'].repertoire)
    assert np.array_equal(again.batch(es, rows), p)
    with pytest.raises(IntegrityError):
        LGBMBehavior(b2.model_string, 99, world['b0'].repertoire)


def test_b2_is_deterministic_with_the_registered_settings(world):
    train = world['tables']['train']
    x, y = train.matrix(B2_FEATURES, None), train.frame.label.to_numpy()
    first = LGBMBehavior.train_only(x, y, params=PARAMS, rounds=5)
    assert first == LGBMBehavior.train_only(x, y, params=PARAMS, rounds=5)


def test_b2_protocol_compliance(world):
    b2 = fit(world, rounds=6)[0]
    table = world['tables']['temperature']
    b0 = world['b0']
    for model in (b2, Calibrated(b2, Temperature(0.8), 'B2')):
        assert isinstance(model, BehaviorModel) and tuple(model.actions) == ACTION_CODES
        model.bind(table)
        batch = model.batch(table)
        frame = table.frame
        for i in range(0, len(table), max(1, len(table) // 20)):
            row = frame.iloc[i]
            history = () if row.previous == '<START>' else (PastPitch(str(row.previous), (), 'unknown', 0, 0),)
            state = PAState(int(row.balls), int(row.strikes), str(int(row.pitcher)), row.batter_side, history,
                            f'{int(row.game_pk)}:{int(row.at_bat_number)}:{int(row.pitch_number)}')
            p = model.probabilities(state)
            assert np.allclose(p, batch[i], rtol=0, atol=1e-12)
            assert np.array_equal(model.support(state), b0.support(state)) and (p[~b0.support(state)] == 0).all()
            assert model.fallback(state) == b0.fallback(state)
        assert model.identity()['class'].endswith(('LGBMBehavior', 'Calibrated'))
