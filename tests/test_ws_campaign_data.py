import json
from pathlib import Path

import numpy as np
import pytest

from experiments.ws_campaign import data


def test_holdout_roles_and_donor_leak_stop():
    spec = {"roles": {"FIT": ["2026-03-01", "2026-03-31"],
                      "SELECT": ["2026-04-01", "2026-04-30"],
                      "CAL": ["2026-05-01", "2026-05-31"],
                      "EVAL": ["2026-06-01", "2026-06-30"]},
            "heldout_pitchers": [7]}
    days = np.array(["2026-03-10", "2026-04-10", "2026-05-10", "2026-06-10", "2026-03-10"])
    pitchers = np.array([1, 1, 1, 7, 7])
    masks = data.role_masks(days, pitchers, spec, heldout=True)
    assert [int(masks[k].sum()) for k in ("FIT", "SELECT", "CAL", "EVAL")] == [1, 1, 1, 1]
    with pytest.raises(data.DataStop, match="heldout donor"):
        data.check_prequential(["2026-03-10"], ["2026-03-09"], [7], [7])
    with pytest.raises(data.DataStop, match="same-day/future"):
        data.check_prequential(["2026-03-10"], ["2026-03-10"], [1], [7])


def test_history_expansion_maps_only_named_columns():
    names = [f"c{i}" for i in range(91)]
    mapped = data.expanded_history_columns(names, [4] * 0 + list(range(36)), [0, 1, 2], ["c4"])
    assert mapped["indices"] == [4, 95, 131, 167]
    with pytest.raises(data.DataStop, match="registered history names"):
        data.expanded_history_columns(names, list(range(36)), [0, 1, 2], ["missing"])


def test_fixed_c_plan_dates_and_hash_selection():
    plan = json.loads((Path(__file__).resolve().parents[1] / "experiments/ws_campaign/C-PLAN.json").read_text())
    assert plan["status"].startswith("SOURCE_ONLY")
    plan.update(source_sha256="a" * 64, exposure_ledger_sha256="b" * 64, codec_source_sha256="c" * 64)
    data.validate_plan(plan)
    rule = plan["windows"]["pitcher_holdout"]["heldout_selection"]
    assert data.select_heldout(range(1, 101), rule) == data.select_heldout(reversed(range(1, 101)), rule)
    with pytest.raises(data.DataStop, match="integer domain"):
        data.select_heldout([1, 2.5, 3], rule)


def test_fit_support_excludes_same_day_other_game():
    support = data.fit_prior_support(
        ["2026-03-25"] * 3 + ["2026-03-26"], [4] * 4,
        [10, 10, 11, 12], [1, 1, 1, 1], [1, 2, 1, 1],
        np.array([True, True, True, False]))
    assert support.tolist() == [0, 1, 0, 3]


def test_heldout_splice_changes_only_owned_eval_features():
    clean = np.zeros((3, 91))
    clean[2, 90] = 1
    enriched = np.full((1, 91), 7.)
    result = data.splice_heldout_own_history(clean, enriched, [False, False, True], [True, False, False])
    assert np.all(result[0] == 0)
    assert result[2, 50] == result[2, 77] == 7
    assert result[2, 78] == 0 and result[2, 90] == 1
    with pytest.raises(data.DataStop, match="similarity abstention"):
        data.splice_heldout_own_history(np.zeros((3, 91)), enriched,
                                        [False, False, True], [True, False, False])
