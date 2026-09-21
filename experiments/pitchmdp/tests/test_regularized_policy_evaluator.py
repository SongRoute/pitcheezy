"""Synthetic residual-evaluator checks; no source data or checkpoints."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from regularized_policy_evaluator import RegularizedPolicyEvaluator
from pitchmdp.archetypes import STYLE_COLUMNS, RELIABILITY_COLUMNS
from pitchmdp.model import CountBaseline


def july_rows():
    rows = []
    for day in (1, 10, 21, 22, 31):
        for i in range(60):
            fastball = i % 2 == 0
            rows.append({"game_date": f"2025-07-{day:02d}", "game_pk": 1000 + day,
                "at_bat_number": i + 1, "pitch_number": 1,
                "pitch_type": "FF" if fastball else "SL", "balls": i % 4, "strikes": i % 3,
                "stand": "L", "p_throws": "R", "outs_when_up": i % 3, "bases": i % 8,
                "inning": 1 + i % 9, "inning_topbot": "Top", "home_score": i % 4, "away_score": 1,
                "batter": 50 + i, "pitcher": 10,
                "description": "ball" if fastball else "swinging_strike", "events": "",
                **{key: .2 + (i % 5) * .02 for key in STYLE_COLUMNS},
                **{key: .5 for key in RELIABILITY_COLUMNS}})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def model():
    return RegularizedPolicyEvaluator().fit(july_rows())


def test_disjoint_fit_and_calibration_and_fixed_parent(model):
    frame = july_rows()
    expected = CountBaseline().fit(frame.loc[frame.game_date.le("2025-07-21")])
    np.testing.assert_array_equal(model.parent.predict(frame), expected.predict(frame))
    assert model.report["fit_date_max"] == "2025-07-21"
    assert model.report["calibration_date_min"] == "2025-07-22"
    assert not set(model.report["fit_game_ids"]) & set(model.report["calibration_game_ids"])
    assert model.report["count_fit_rows"] == 180
    assert model.report["calibration_rows"] == 120
    assert model.report["calibration_mixed_log_loss"] <= model.report["calibration_count_log_loss"] + 1e-12
    assert 0 <= model.weight <= 1
    assert model.coefficients.shape[1] == 10


def test_context_contract_and_ids_do_not_affect_predictions(model):
    frame = july_rows().head(20)
    actual = model.predict(frame)
    assert actual.shape == (20, 10)
    assert np.isfinite(actual).all() and (actual > 0).all()
    np.testing.assert_allclose(actual.sum(axis=1), 1, atol=1e-12)
    changed = frame.copy()
    changed["batter"], changed["pitcher"] = 999999, 888888
    np.testing.assert_array_equal(model.predict(changed), actual)
    assert not {"batter", "pitcher"} & set(model.report["feature_names"])
    assert set(STYLE_COLUMNS + RELIABILITY_COLUMNS).issubset(model.report["feature_names"])
    for missing in ("inning", "inning_topbot", "home_score", STYLE_COLUMNS[0]):
        with pytest.raises(ValueError, match="Missing evaluator context"):
            model.predict(frame.drop(columns=missing))
    assert model.predict(frame.iloc[:0]).shape == (0, 10)


def test_action_support_and_unseen_type(model):
    frame = july_rows().head(3).copy()
    p, support, origin = model.predict_with_support(frame)
    assert (support > 0).all() and (origin == 1).all()
    frame.loc[frame.index[0], "pitch_type"] = "NEW"
    q, support, origin = model.predict_with_support(frame)
    assert support[0] == 0 and origin[0] == -1
    assert np.isfinite(q).all()
    np.testing.assert_allclose(q.sum(axis=1), 1)
    frame.loc[frame.index[1], "stand"] = "R"
    _, support, origin = model.predict_with_support(frame)
    assert support[1] == 0 and origin[1] == 0


def test_calibration_outcomes_cannot_change_coefficients_or_parent(model):
    frame = july_rows()
    frame.loc[frame.game_date.ge("2025-07-22"), "description"] = "foul"
    changed = RegularizedPolicyEvaluator().fit(frame)
    np.testing.assert_array_equal(changed.coefficients, model.coefficients)
    np.testing.assert_array_equal(changed.mean, model.mean)
    np.testing.assert_array_equal(changed.parent.predict(frame), model.parent.predict(frame))
    assert changed.report["fit_game_ids"] == model.report["fit_game_ids"]


def test_deterministic_cap_preserves_full_fit_count_baseline():
    frame = july_rows()
    first = RegularizedPolicyEvaluator(max_training_rows=50).fit(frame)
    shuffled = RegularizedPolicyEvaluator(max_training_rows=50).fit(frame.sample(frac=1, random_state=9))
    assert first.report["residual_fit_rows"] == 50
    assert first.report["count_fit_rows"] == 180
    np.testing.assert_array_equal(first.coefficients, shuffled.coefficients)
    np.testing.assert_array_equal(first.parent.global_p, shuffled.parent.global_p)


def test_training_date_game_and_context_guards():
    frame = july_rows()
    bad = frame.copy()
    bad.loc[0, "game_date"] = "2025-08-01"
    with pytest.raises(ValueError, match="July 2025"):
        RegularizedPolicyEvaluator().fit(bad)
    with pytest.raises(ValueError, match="both July"):
        RegularizedPolicyEvaluator().fit(frame.loc[frame.game_date.le("2025-07-21")])
    bad = frame.copy()
    bad.loc[bad.game_date.ge("2025-07-22"), "game_pk"] = 1001
    with pytest.raises(ValueError, match="disjoint"):
        RegularizedPolicyEvaluator().fit(bad)
    bad = frame.copy()
    bad.loc[0, "outs_when_up"] = np.nan
    with pytest.raises(ValueError, match="outs_when_up"):
        RegularizedPolicyEvaluator().fit(bad)
    with pytest.raises(ValueError, match="fixes L2"):
        RegularizedPolicyEvaluator(l2=.01)
