"""CPU-only synthetic service checks; never load or train real checkpoints."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import threading
from http.server import HTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from minimal_pitch_service import Engine, RequestError, evaluate_policy, make_handler, validate_request
from pitchmdp.game import terminal_values
from pitchmdp.planner import solve_pa


META = {"pitchers": {"10": {"p_throws": "R", "pitch_types": ["FF", "SL"]}},
        "profiles": {"20": {"rates": [.76, .47, .085, .225, .165, .43],
                              "reliabilities": [.5] * 6, "as_of": "2025-04-30"}},
        "profile_cutoff": "2025-04-30"}
REQUEST = {"inning": 6, "topbot": "Top", "outs": 1, "bases": 1,
           "home_score": 2, "away_score": 2, "balls": 1, "strikes": 2,
           "pitcher_id": 10, "batter_id": 20, "batter_stand": "L"}


class FakeContext:
    def transform(self, frame):
        assert "batter" not in frame  # Lookup IDs never enter the neural frame.
        result = np.zeros((len(frame), 23), dtype=np.float32)
        result[:, 0] = frame.balls / 3
        result[:, 1] = frame.strikes / 2
        return result


class FakeDelivery:
    draws = 400

    def sample(self, frame):
        values = np.zeros((len(frame), self.draws, 8), dtype=np.float32)
        values[:, :, 0] = frame.pitch_type.eq("SL").to_numpy()[:, None]
        return values, np.full(len(frame), 3)


class FakeModel:
    delivery_temperature = 1.3

    def logits(self, arrays):
        tokens, valid, context = arrays
        assert not valid[:, :-1].any() and valid[:, -1].all()
        assert not tokens[:, :-1].any()
        result = np.zeros((len(tokens), 10), dtype=np.float32)
        result[:, 1] = tokens[:, -1, 0] + context[:, 1]
        result[:, 3] = 1.5 * (1 - tokens[:, -1, 0])
        result[:, 0] = context[:, 0]
        return result


class FakeFrequency:
    def predict(self, frame):
        return np.broadcast_to(np.array([.3, .2, .1, .15, .08, .04, .01, .04, .03, .05]), (len(frame), 10))


class FakeWE:
    def predict_defense(self, state, defender_is_home):
        if state.winner:
            home = float(state.winner == "home")
        else:
            sign = 1 if state.half == "Bot" else -1
            advantage = state.home_score - state.away_score + sign * (.1 * state.bases - .15 * state.outs)
            home = 1 / (1 + np.exp(-advantage))
        return home if defender_is_home else 1 - home


@pytest.fixture
def engine():
    value = Engine.__new__(Engine)
    value.metadata = deepcopy(META)
    value.context, value.delivery, value.baseline = FakeContext(), FakeDelivery(), FakeFrequency()
    value.models = [FakeModel() for _ in range(5)]
    value.weight, value.baseline_temperature = .7, 1.1
    value.we, value.advancement = FakeWE(), None
    value._lock = threading.RLock()
    return value


@pytest.mark.parametrize("key,value", [("balls", 4), ("strikes", -1), ("inning", True),
    ("outs", 3), ("bases", 8), ("home_score", -1), ("away_score", 2.0),
    ("topbot", "bottom"), ("batter_stand", "S"), ("top_k", 0)])
def test_bad_state(key, value):
    with pytest.raises(RequestError):
        validate_request({**REQUEST, key: value}, META)


def test_lookup_and_profile_guards():
    assert validate_request(REQUEST, META)["profile_source"] == "snapshot"
    for request, code in [({**REQUEST, "pitcher_id": 99}, "unsupported_pitcher"),
                          ({**REQUEST, "batter_id": 99}, "unknown_batter"),
                          ({**REQUEST, "date": "2025-04-30"}, "profile_date_conflict"),
                          ({**REQUEST, "inning": 9, "topbot": "Bot", "home_score": 3}, "invalid_state")]:
        with pytest.raises(RequestError) as error:
            validate_request(request, META)
        assert error.value.code == code
    profile = deepcopy(META["profiles"]["20"])
    profile["rates"][4] = 1.2  # ISO is not a Bernoulli rate.
    result = validate_request({**REQUEST, "batter_profile": profile, "date": "2024-01-01"}, META)
    assert result["profile_source"] == "explicit_prior_date_profile"
    for field, values in [("rates", [float("nan")] * 6), ("rates", [1] * 5),
                          ("reliabilities", [-.1] * 6), ("rates", [True] * 6)]:
        with pytest.raises(RequestError):
            validate_request({**REQUEST, "batter_profile": {**profile, field: values}}, META)


def test_default_profile_is_explicitly_flagged():
    meta = deepcopy(META)
    meta["default_profile"] = {"rates": [.5] * 6, "reliabilities": [0] * 6}
    assert validate_request({**REQUEST, "batter_id": 99}, meta)["profile_source"] == "default_profile_zero_reliability"


@pytest.mark.parametrize("outs,bases,legal_dp", [(1, 1, True), (2, 7, False), (0, 0, False), (0, 2, True)])
def test_counts_and_common_legality(engine, outs, bases, legal_dp):
    result = engine.predict_counts({**REQUEST, "outs": outs, "bases": bases})
    assert result["pitch_types"] == ["FF", "SL"]
    assert result["delivery_tiers"] == {"3": 24}
    for p in result["probabilities"].values():
        assert p.shape == (4, 3, 1, 2, 10)
        np.testing.assert_allclose(p.sum(-1), 1, atol=1e-12, rtol=0)
        assert np.isfinite(p).all() and (p >= 0).all()
        assert ((p[..., 9] > 0).all() if legal_dp else (p[..., 9] == 0).all())


@pytest.mark.parametrize("counts", [None, {"SL": 80, "FF": 20}])
def test_recommendation_matches_frozen_solver(engine, counts):
    if counts is not None:
        engine.metadata["repertoire_counts"] = {"10": counts}
    result = engine.predict_counts(REQUEST)
    p = result["probabilities"]["blend"]
    terminal = terminal_values(result["state"], engine.we, engine.advancement)
    weights = None if counts is None else np.array([counts[name] for name in result["pitch_types"]], float)
    if weights is not None:
        weights /= weights.sum()
    plan = solve_pa(p, terminal, [0, 0], baseline_policy=weights)
    response = engine.recommend(REQUEST)
    first = response["recommendations"][0]
    assert first["pitch_type"] == result["pitch_types"][plan.policy[1, 2, 0]]
    assert first["defensive_we"] == pytest.approx(plan.values[1, 2, 0])
    assert response["baseline_policy"] == ("uniform_pitch_types" if counts is None else "train_repertoire_frequency")
    assert response["baseline_defensive_we"] == pytest.approx(plan.baseline_values[1, 2, 0])
    assert first["delta_vs_baseline_policy"] == pytest.approx(plan.values[1, 2, 0] - plan.baseline_values[1, 2, 0])
    assert "delta_vs_uniform_policy" not in first
    assert sum(first["outcome_probabilities"].values()) == pytest.approx(1)
    assert "target_location" not in first and response["latency_ms"] >= 0
    np.testing.assert_allclose(evaluate_policy(p, terminal, plan.policy), plan.values, atol=1e-10)
    fixed_policy = np.zeros((4, 3, 1), dtype=int)
    fixed_values = evaluate_policy(p, terminal, fixed_policy)
    assert (plan.values >= fixed_values - 1e-10).all()
    with pytest.raises(ValueError):
        evaluate_policy(p, terminal, fixed_policy.astype(float))


def test_local_http_contract(engine):
    server = HTTPServer(("127.0.0.1", 0), make_handler(engine))
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    address = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(address + "/health", timeout=5) as response:
            assert json.load(response)["model_loaded"] is True
        with urlopen(address + "/metadata", timeout=5) as response:
            assert json.load(response)["profile_order"][0] == "contact"
        request = Request(address + "/recommend", json.dumps(REQUEST).encode(), {"Content-Type": "application/json"})
        with urlopen(request, timeout=5) as response:
            assert json.load(response)["recommendations"]
        bad = Request(address + "/recommend", b'{"balls":4}', {"Content-Type": "application/json"})
        with pytest.raises(HTTPError) as error:
            urlopen(bad, timeout=5)
        assert error.value.code == 400
        assert json.load(error.value)["error"] == "invalid_state"
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
