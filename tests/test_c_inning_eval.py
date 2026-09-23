import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("c_inning_eval", ROOT / "scripts/c_inning_eval.py")
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)
from pitchmdp.game import GameState, apply_terminal
from pitchmdp.planner import TERMINALS


CONFIG = {"mass_tolerance": 1e-8, "minimum_state_mass": 0.001,
          "maximum_pas": 9, "maximum_model_calls": 12, "maximum_frontier_states": 256,
          "pa_policy_id": "synthetic_fixed_policy"}
PROFILE = {"rates": [0.5] * 6, "reliabilities": [0.5] * 6, "as_of": "2025-01-01"}
HITTER = {"batter_stand": "L", "batter_profile": PROFILE, "known_before_decision": True}


def request(**updates):
    data = {"date": "2025-08-17", "pitcher_id": 657277, "inning": 5, "topbot": "Top",
            "outs": 1, "bases": 0, "home_score": 0, "away_score": 0,
            "balls": 0, "strikes": 0, **HITTER}
    data.update(updates)
    return data


class DummyWE:
    def predict_defense(self, state, defender_home):
        if state.winner is not None:
            return float((state.winner == "home") == defender_home)
        return 0.7 if defender_home else 0.3


class DeterministicAdvancement:
    def distribution(self, state, event):
        return [(1.0, apply_terminal(state, event))]


def event_provider(event):
    return lambda _: {name: float(name == event) for name in TERMINALS}


def evaluate(**kwargs):
    return mod.evaluate_inning(**kwargs, provider_identity="synthetic_terminal_fixture")


def test_two_pas_absorb_at_current_half_end():
    result = evaluate(initial_request=request(), lineup=[HITTER, HITTER],
                                 provider=event_provider("strikeout"), we=DummyWE(),
                                 advancement=DeterministicAdvancement(), config=CONFIG)
    assert result["status"] == "complete"
    assert result["value"] == pytest.approx(0.7)
    assert result["absorbed_mass"] == pytest.approx(1)
    assert result["processed_pas"] == 2


def test_missing_future_hitter_is_unresolved_without_renormalization():
    result = evaluate(initial_request=request(), lineup=[HITTER],
                                 provider=event_provider("strikeout"), we=DummyWE(),
                                 advancement=DeterministicAdvancement(), config=CONFIG)
    assert result["status"] == "bounded"
    assert result["value"] is None
    assert result["value_interval"] == pytest.approx([0, 1])
    assert result["unresolved_reasons"] == {"predecision_lineup_exhausted": 1}


def test_partial_absorption_reports_probability_bound():
    def mixed(_):
        return {name: 0.5 if name in ("out", "single") else 0.0 for name in TERMINALS}

    result = evaluate(initial_request=request(outs=2), lineup=[HITTER],
                                 provider=mixed, we=DummyWE(),
                                 advancement=DeterministicAdvancement(), config=CONFIG)
    assert result["absorbed_mass"] == pytest.approx(0.5)
    assert result["unresolved_mass"] == pytest.approx(0.5)
    assert result["value_interval"] == pytest.approx([0.35, 0.85])


def test_walkoff_uses_initial_away_defender():
    result = evaluate(initial_request=request(inning=9, topbot="Bot", outs=2),
                                 lineup=[HITTER], provider=event_provider("home_run"),
                                 we=DummyWE(), advancement=DeterministicAdvancement(), config=CONFIG)
    assert result["status"] == "complete"
    assert result["initial_defender"] == "away"
    assert result["value"] == 0


def test_extra_inning_abstains():
    result = evaluate(initial_request=request(inning=10, bases=2), lineup=[HITTER],
                                 provider=event_provider("out"), we=DummyWE(),
                                 advancement=DeterministicAdvancement(), config=CONFIG)
    assert result["status"] == "unavailable"
    assert result["reason"] == "extra_inning_initial_state_unsupported"


def test_future_hitter_requires_predecision_order_and_profile():
    unknown = {"batter_stand": "L", "batter_profile": PROFILE}
    with pytest.raises(ValueError, match="predecision order"):
        evaluate(initial_request=request(), lineup=[HITTER, unknown],
                            provider=event_provider("out"), we=DummyWE(),
                            advancement=DeterministicAdvancement(), config=CONFIG)
    contemporary = {"batter_stand": "L", "batter_profile": {**PROFILE, "as_of": "2025-08-17"}}
    with pytest.raises(ValueError, match="profile must predate"):
        evaluate(initial_request=request(batter_profile=contemporary["batter_profile"]), lineup=[contemporary],
                            provider=event_provider("out"), we=DummyWE(),
                            advancement=DeterministicAdvancement(), config=CONFIG)


def test_frozen_provider_basis_uses_fixed_baseline_policy():
    class Engine:
        metadata = {"repertoire_counts": {"657277": {"FF": 3, "CH": 1}}}

        def predict_counts(self, _):
            p = np.zeros((4, 3, 1, 2, 10))
            p[:, :, :, 0, 3] = 1  # FF -> out
            p[:, :, :, 1, 4] = 1  # CH -> single
            return {"probabilities": {"blend": p}, "pitch_types": ["FF", "CH"],
                    "pitcher_id": 657277, "balls": 0, "strikes": 0}

    result = mod.FrozenPAProvider(Engine())(request())
    assert result["out"] == pytest.approx(0.75)
    assert result["single"] == pytest.approx(0.25)
    assert sum(result.values()) == pytest.approx(1)


def test_comparison_requires_eligibility_and_keeps_bounds():
    identity = {"provider_identity": "synthetic", "initial_state_and_count": {"outs": 2},
                "lineup_sha256": "fixture", "policy_id": "fixed", "horizon": "inning_end",
                "initial_defender": "home"}
    keep = {"status": "bounded", "value_interval": [0.3, 0.5],
            "evaluation_identity": identity, "pitcher_id": 1}
    sub = {"status": "bounded", "value_interval": [0.4, 0.6],
           "evaluation_identity": identity, "pitcher_id": 2}
    assert mod.compare(keep, sub, eligibility_verified=False)["value_pp"] is None
    result = mod.compare(keep, sub, eligibility_verified=True)
    assert result["status"] == "bounded"
    assert result["value_interval_pp"] == pytest.approx([-10, 30])
    assert mod.compare(keep, {**sub, "evaluation_identity": {**identity, "lineup_sha256": "other"}},
                       eligibility_verified=True)["reason"] == "mismatched_evaluation_identity_or_pitcher"
