"""Synthetic D2 whole-game restriction and past-only history checks."""
import numpy as np
import pandas as pd
import pytest

from pitchmdp.matrix_aux_data import restrict_train_games, validate_config
from scripts.run_ml_aux_data import fit


def config():
    return {"protocol": "ml_aux_data_v1", "experiment_id": "EXP-P6-001",
            "parent_run": "/tmp/parent", "parent_preparation_sha256": "a" * 64,
            "training_games": "d25", "seeds": [0, 1, 2], "history_length": 5,
            "kind": "flatten_mlp", "width": 128, "draws": 400, "sample_seed": 42,
            "device": "auto", "budget": {"epochs": 30, "patience": 5,
            "batch_size": 1024, "learning_rate": .0005}, "registration": {"cell": "D2-25"}}


def observations():
    rows = []
    for game, date, event, desc in [
        (1, "2023-04-01", "single", "hit_into_play"),
        (2, "2024-04-01", "strikeout", "swinging_strike"),
        (3, "2024-04-02", "home_run", "hit_into_play"),
        (4, "2025-05-01", "walk", "ball"),
        (5, "2025-07-01", "single", "hit_into_play"),
    ]:
        rows.append({"game_pk": game, "at_bat_number": 1, "pitch_number": 1,
                     "game_date": pd.Timestamp(date), "game_type": "R", "batter": 7,
                     "pitch_type": "FF", "description": desc, "events": event,
                     "is_pa_terminal": True, "plate_x": .1, "plate_z": 2.4})
    return pd.DataFrame(rows)


def test_config_pins_registered_axes():
    expected = config()
    assert validate_config(expected) == expected
    with pytest.raises(ValueError, match="fixed sample"):
        validate_config({**expected, "draws": 25})
    with pytest.raises(ValueError, match="schema"):
        validate_config({**expected, "unknown_axis": 1})


def test_restricted_history_excludes_omitted_train_but_keeps_both_context_periods():
    frame, report = restrict_train_games(observations(), np.array([2]))
    assert frame.game_pk.tolist() == [1, 2, 4, 5]
    assert report["omitted_train_rows"] == 1
    assert report["selected_train_games"] == 1
    assert report["pretrain_rows"] == 1
    assert report["posttrain_rows"] == 2
    # First May row sees the pre-TRAIN single and selected TRAIN strikeout,
    # but neither the omitted TRAIN homer nor its league contribution.
    may = frame.loc[frame.game_pk.eq(4)].iloc[0]
    assert may.batter_pa_prior == 2
    expected_obp = (1 + 30 * .320) / (2 + 30)
    assert may.batter_obp_prior == pytest.approx(expected_obp)
    assert may.batter_style_isolated_power_reliability > 0
    unrestricted, _ = restrict_train_games(observations(), np.array([2, 3]))
    unrestricted_may = unrestricted.loc[unrestricted.game_pk.eq(4)].iloc[0]
    assert may.batter_style_isolated_power_prior < unrestricted_may.batter_style_isolated_power_prior
    assert frame.loc[frame.game_pk.eq(5), "batter_pa_prior"].iloc[0] == 3


def test_restriction_refuses_partial_or_wrong_game_identity():
    frame = observations()
    with pytest.raises(ValueError, match="subset"):
        restrict_train_games(frame, np.array([4]))
    with pytest.raises(ValueError, match="unique"):
        restrict_train_games(frame, np.array([2, 2]))
    duplicate = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="unique"):
        restrict_train_games(duplicate, np.array([2]))


def test_full_fit_requires_completed_preparation_bound_profile(tmp_path):
    with pytest.raises(ValueError, match="profile"):
        fit(config(), {}, tmp_path, {"samples": {"train": {"rows_sha256": "x"}}}, 0)
