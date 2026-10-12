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
