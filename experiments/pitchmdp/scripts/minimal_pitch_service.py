"""Local pitch-type MVP using a frozen history-zero continuous five-seed bundle.

Only load trusted, locally built bundles: pickle and torch checkpoints are executable
formats. No training, downloads, or logged current-delivery inputs occur here.
"""
from __future__ import annotations

import argparse
from datetime import date
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import pickle
import sys
import threading
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))
import numpy as np
import pandas as pd
from scipy.special import softmax
import torch

from pitchmdp.archetypes import STYLE_COLUMNS, RELIABILITY_COLUMNS, STYLE_NAMES
from pitchmdp.game import GameState, terminal_values
from pitchmdp.planner import OUTCOMES, solve_pa
from pitchmdp.sequence_model import SequenceModel, SequenceNetwork
from diagnose_sequence_legality import condition_on_legality
from run_sequence_frequency_baselines import temperature_predictions


ASSUMPTIONS = [
    "Pitch types only; no intended location is estimated or recommended.",
    "History-zero predictions hold game state and batter profile fixed within the PA; only count changes.",
    "Current deliveries are marginalized over 400 joint TRAIN vectors, not observed future deliveries.",
    "IDs select repertoire, delivery pools, frequency tables or profile snapshots; no player-ID neural features.",
    "Double-play mass is conditioned away with two outs or empty bases for every predictor.",
    "Full-PA defensive WE uses the frozen continuation and runner-advancement models; baseline is uniform pitch types.",
    "Model probabilities and WE differences are internal estimates, not causal gains or validated live performance.",
    "Regular-season rules include an automatic runner in extra innings; no steals, substitutions or fatigue dynamics.",
]


class RequestError(ValueError):
    def __init__(self, message, code="invalid_request"):
        super().__init__(message)
        self.code = code


def _integer(data, key, minimum, maximum=None):
    value = data.get(key)
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        bounds = f"{minimum}..{maximum}" if maximum is not None else f">={minimum}"
        raise RequestError(f"{key} must be an integer {bounds}", "invalid_state")
    return value


def _iso_date(value, field):
    try:
        if not isinstance(value, str):
            raise ValueError()
        parsed = date.fromisoformat(value)
        if parsed.isoformat() != value:
            raise ValueError()
        return parsed
    except ValueError:
        raise RequestError(f"{field} must be YYYY-MM-DD") from None


def validate_request(request, metadata):
    """Validate before frame construction, so invalid fields cannot silently coerce."""
    if not isinstance(request, dict):
        raise RequestError("Expected a JSON object")
    row = {key: _integer(request, key, lo, hi) for key, lo, hi in (
        ("inning", 1, 99), ("outs", 0, 2), ("bases", 0, 7),
        ("home_score", 0, 999), ("away_score", 0, 999),
        ("balls", 0, 3), ("strikes", 0, 2), ("pitcher_id", 1, None))}
    row["topbot"] = request.get("topbot")
    if row["topbot"] not in ("Top", "Bot"):
        raise RequestError("topbot must be Top or Bot", "invalid_state")
    if row["home_score"] > row["away_score"] and (
        row["topbot"] == "Bot" and row["inning"] >= 9 or row["inning"] >= 10
    ):
        raise RequestError("Game has already ended under supported rules", "invalid_state")
    row["batter_stand"] = request.get("batter_stand")
    if row["batter_stand"] not in ("L", "R"):
        raise RequestError("batter_stand must be L or R")
    pitcher = metadata.get("pitchers", {}).get(str(row["pitcher_id"]))
    if pitcher is None:
        raise RequestError("Unsupported pitcher_id; see /metadata for available pitchers", "unsupported_pitcher")
    row["pitch_types"], row["p_throws"] = pitcher["pitch_types"], pitcher["p_throws"]
    row["top_k"] = _integer({"top_k": request.get("top_k", 3)}, "top_k", 1, 20)
    request_date = _iso_date(request["date"], "date") if "date" in request else None
    if "batter_profile" in request:
        profile, source = request["batter_profile"], "explicit_prior_date_profile"
    else:
        batter = _integer(request, "batter_id", 1)
        profile = metadata.get("profiles", {}).get(str(batter))
        source = "snapshot"
        if profile is None:
            profile, source = metadata.get("default_profile"), "default_profile_zero_reliability"
        if profile is None:
            raise RequestError("Unknown batter_id; supply batter_profile rates and reliabilities", "unknown_batter")
    if not isinstance(profile, dict):
        raise RequestError("batter_profile must contain rates and reliabilities")
    for field in ("rates", "reliabilities"):
        values = profile.get(field)
        if not isinstance(values, (list, tuple)) or len(values) != 6 or any(
            type(v) not in (int, float) or not np.isfinite(v) for v in values
        ):
            raise RequestError(f"Profile {field} must contain six finite numbers in {list(STYLE_NAMES)} order")
    rates, reliability = np.asarray(profile["rates"]), np.asarray(profile["reliabilities"])
    # ISO is extra bases per at-bat, with a mathematical range of 0..3.
    if (rates < 0).any() or (rates > np.array([1, 1, 1, 1, 3, 1])).any():
        raise RequestError("Profile rates must be in [0,1], except isolated_power in [0,3]")
    if (reliability < 0).any() or (reliability > 1).any():
        raise RequestError("Profile reliabilities must be in [0,1]")
    if source == "default_profile_zero_reliability" and (reliability != 0).any():
        raise RequestError("Bundle default profile must have zero reliabilities")
    as_of = profile.get("as_of", metadata.get("profile_cutoff"))
    if source != "explicit_prior_date_profile" and request_date and as_of:
        if request_date <= _iso_date(as_of, "profile as_of"):
            raise RequestError("Requested date must follow the profile snapshot as_of; supply a historical profile", "profile_date_conflict")
    row.update(profile=dict(zip((*STYLE_COLUMNS, *RELIABILITY_COLUMNS), [*rates, *reliability])),
               profile_source=source, profile_as_of=as_of if source != "explicit_prior_date_profile" else profile.get("as_of"))
    row["state"] = GameState(row["inning"], row["topbot"], row["outs"], row["bases"], row["home_score"], row["away_score"])
    return row


def evaluate_policy(probabilities, terminal, policy):
    """Evaluate a count-dependent deterministic policy under a common tensor."""
    p = np.asarray(probabilities)
    actions = np.asarray(policy)
    if actions.shape != p.shape[:3] or not np.issubdtype(actions.dtype, np.integer):
        raise ValueError("policy must contain integer actions with shape (4,3,n_previous)")
    if (actions < 0).any() or (actions >= p.shape[3]).any() or p.shape[2] != 1:
        raise ValueError("Invalid action or non-history-zero policy")
    weights = np.eye(p.shape[3])[actions]
    return solve_pa(p, terminal, [0] * p.shape[3], baseline_policy=weights).baseline_values


class Engine:
    """Load once; serialize all inference against the immutable local bundle."""
    def __init__(self, bundle, device="cpu"):
        self.bundle = Path(bundle).resolve()
        self.metadata = json.loads((self.bundle / "metadata.json").read_text())
        if self.metadata.get("schema_version") != 1:
            raise ValueError("Unsupported bundle schema_version")
        required = ["metadata.json", "preprocessor.pkl", "game_values.pkl", *self.metadata["models"]]
        for rel in required:
            path = (self.bundle / rel).resolve()
            if not path.is_relative_to(self.bundle) or not path.is_file():
                raise ValueError(f"Missing or out-of-bundle artifact: {rel}")
        manifest_path = self.bundle / "bundle_manifest.json"
        if manifest_path.exists():
            hashes = json.loads(manifest_path.read_text())["sha256"]
            for rel in required:
                if rel not in hashes or hashlib.sha256((self.bundle / rel).read_bytes()).hexdigest() != hashes[rel]:
                    raise ValueError(f"Bundle hash mismatch: {rel}")
        with (self.bundle / "preprocessor.pkl").open("rb") as stream:
            preprocessing = pickle.load(stream)
        self.context, self.delivery, self.baseline = (preprocessing[k] for k in ("context", "delivery", "baseline"))
        with (self.bundle / "game_values.pkl").open("rb") as stream:
            game = pickle.load(stream)
        self.we, self.advancement = game["we"], game["advancement"]
        self.weight = float(self.metadata["neural_weight"])
        self.baseline_temperature = float(self.metadata["baseline_temperature"])
        if not 0 <= self.weight <= 1 or not np.isfinite(self.baseline_temperature) or self.baseline_temperature <= 0:
            raise ValueError("Invalid frozen blend calibration")
        if self.context.mode != "continuous" or self.delivery.draws != 400 or len(self.metadata["models"]) != 5:
            raise ValueError("Expected continuous context, delivery400, and five ensemble members")
        for pitcher in self.metadata["pitchers"].values():
            types = pitcher["pitch_types"]
            if pitcher["p_throws"] not in ("L", "R") or not types or len(types) != len(set(types)):
                raise ValueError("Invalid bundle pitcher repertoire")
        torch.set_num_threads(4)
        self.models = []
        for rel in self.metadata["models"]:
            saved = torch.load(self.bundle / rel, map_location="cpu", weights_only=False)
            cfg = saved["network_config"]
            if cfg["kind"] != "flatten_mlp" or cfg["n_context"] != 23 or cfg["n_classes"] != 10 or cfg["length"] != 6:
                raise ValueError("Unexpected member architecture")
            model = SequenceModel(cfg["kind"], saved["seed"], cfg["width"], cfg["n_classes"])
            model.device = device
            model.net = SequenceNetwork(**cfg).to(device)
            model.net.load_state_dict(saved["state_dict"])
            model.net.eval()
            model.temperature = saved["temperature"]
            model.delivery_temperature = saved.get("delivery_temperature", model.temperature)
            if not np.isfinite(model.delivery_temperature) or model.delivery_temperature <= 0:
                raise ValueError("Invalid member delivery temperature")
            self.models.append(model)
        if sorted(model.seed for model in self.models) != list(range(42, 47)):
            raise ValueError("Expected frozen seeds 42 through 46")
        self._lock = threading.RLock()

    def predict_counts(self, request):
        """Return legal tensors shaped (4,3,1,n_types,10); no optimization yet."""
        with self._lock:
            row = validate_request(request, self.metadata)
            records = []
            for balls in range(4):
                for strikes in range(3):
                    for pitch_type in row["pitch_types"]:
                        records.append({"balls": balls, "strikes": strikes, "pitch_type": pitch_type,
                            "inning": row["inning"], "inning_topbot": row["topbot"], "outs_when_up": row["outs"],
                            "bases": row["bases"], "home_score": row["home_score"], "away_score": row["away_score"],
                            "pitcher": row["pitcher_id"], "p_throws": row["p_throws"], "stand": row["batter_stand"],
                            **row["profile"]})
            frame = pd.DataFrame(records)
            draws, tiers = self.delivery.sample(frame)
            n, count, physical = draws.shape
            tokens = np.zeros((n * count, 6, physical), dtype=np.float32)
            tokens[:, -1] = draws.reshape(-1, physical)
            valid = np.zeros((n * count, 6), dtype=bool)
            valid[:, -1] = True
            context = np.repeat(self.context.transform(frame), count, axis=0)
            neural = np.zeros((n, len(OUTCOMES)), dtype=np.float64)
            for model in self.models:
                logits = model.logits((tokens, valid, context)).reshape(n, count, len(OUTCOMES))
                neural += softmax(logits / model.delivery_temperature, axis=-1).mean(axis=1) / len(self.models)
            frequency = temperature_predictions(self.baseline.predict(frame), self.baseline_temperature)
            predictions = {"neural": neural, "frequency": frequency,
                           "blend": self.weight * neural + (1 - self.weight) * frequency}
            impossible = np.full(n, row["outs"] == 2 or row["bases"] == 0)
            shape = (4, 3, 1, len(row["pitch_types"]), len(OUTCOMES))
            row["probabilities"] = {name: condition_on_legality(p, impossible).reshape(shape) for name, p in predictions.items()}
            row["delivery_tiers"] = {str(int(tier)): int((tiers == tier).sum()) for tier in np.unique(tiers)}
            return row

    def recommend(self, request):
        started = time.perf_counter()
        with self._lock:
            result = self.predict_counts(request)
            terminal = terminal_values(result["state"], self.we, self.advancement)
            probabilities = result["probabilities"]["blend"]
            plan = solve_pa(probabilities, terminal, [0] * len(result["pitch_types"]))
            balls, strikes = result["balls"], result["strikes"]
            recommendations = []
            for entry in plan.topk(balls, strikes, 0, result["top_k"]):
                action = entry["action_index"]
                recommendations.append({"pitch_type": result["pitch_types"][action],
                    "defensive_we": entry["value"], "delta_vs_uniform_policy": entry["delta_vs_baseline"],
                    "outcome_probabilities": dict(zip(OUTCOMES, probabilities[balls, strikes, 0, action].tolist()))})
            return {"recommendations": recommendations, "objective": "full_pa_defensive_we",
                "baseline_defensive_we": float(plan.baseline_values[balls, strikes, 0]),
                "probability_kind": "model_internal_next_pitch_outcomes",
                "profile_source": result["profile_source"], "profile_as_of": result["profile_as_of"],
                "neural_weight": self.weight, "delivery_tiers": result["delivery_tiers"],
                "solver": plan.diagnostics, "assumptions": ASSUMPTIONS,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2)}

    def public_metadata(self):
        return {"schema_version": 1, "pitchers": self.metadata["pitchers"],
            "profile_order": list(STYLE_NAMES), "profile_cutoff": self.metadata.get("profile_cutoff"),
            "profile_count": len(self.metadata.get("profiles", {})), "neural_weight": self.weight,
            "history_length": 0, "delivery_draws": self.delivery.draws, "assumptions": ASSUMPTIONS}


def make_handler(engine):
    class Handler(BaseHTTPRequestHandler):
        def _reply(self, status, payload):
            body = json.dumps(payload, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._reply(200, {"status": "ok", "model_loaded": True})
            elif self.path == "/metadata":
                self._reply(200, engine.public_metadata())
            else:
                self._reply(404, {"error": "not_found"})

        def do_POST(self):
            if self.path != "/recommend":
                return self._reply(404, {"error": "not_found"})
            try:
                if self.headers.get("Transfer-Encoding"):
                    raise RequestError("Transfer-Encoding is unsupported")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 65536:
                    raise RequestError("JSON body must be 1..65536 bytes")
                self.connection.settimeout(10)
                request = json.loads(self.rfile.read(length))
                self._reply(200, engine.recommend(request))
            except (RequestError, json.JSONDecodeError, UnicodeError, ValueError) as exc:
                self._reply(400, {"error": getattr(exc, "code", "invalid_request"), "message": str(exc)})
            except TimeoutError:
                self._reply(408, {"error": "request_timeout"})
            except Exception:
                self.log_error("Recommendation failed", exc_info=True)
                self._reply(500, {"error": "inference_failed", "message": "Inspect local service logs"})

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--request", type=Path, help="Read JSON and print one recommendation, without serving")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    engine = Engine(args.bundle, args.device)
    if args.request:
        try:
            print(json.dumps(engine.recommend(json.loads(args.request.read_text())), indent=2, allow_nan=False))
        except RequestError as exc:
            parser.exit(2, json.dumps({"error": exc.code, "message": str(exc)}) + "\n")
        return
    server = HTTPServer(("127.0.0.1", args.port), make_handler(engine))
    print(f"Local pitch service ready: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
