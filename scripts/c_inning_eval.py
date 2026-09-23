"""Bounded, separate current-half-inning evaluator for frozen pitch-service probabilities.

This is a research artifact. It never infers a future lineup or candidate availability.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime
import hashlib
import json
import math
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "experiments/pitchmdp"))
sys.path.insert(0, str(PROJECT / "experiments/pitchmdp/scripts"))

import numpy as np
from pitchmdp.game import GameState
from pitchmdp.planner import TERMINALS, solve_pa


def _state(request: dict) -> GameState:
    return GameState(request["inning"], request["topbot"], request["outs"],
                     request["bases"], request["home_score"], request["away_score"])


def _request_at(initial: dict, state: GameState, hitter: dict, first: bool) -> dict:
    request = {key: initial[key] for key in ("date", "pitcher_id")}
    request.update(inning=state.inning, topbot=state.half, outs=state.outs,
                   bases=state.bases, home_score=state.home_score, away_score=state.away_score,
                   balls=initial["balls"] if first else 0,
                   strikes=initial["strikes"] if first else 0,
                   batter_stand=hitter["batter_stand"], batter_profile=hitter["batter_profile"])
    return request


class FrozenPAProvider:
    """Use the frozen blend and one fixed repertoire policy for every terminal basis."""

    def __init__(self, engine):
        self.engine = engine
        self.calls = 0
        self.first_distribution = None

    def __call__(self, request: dict) -> dict[str, float]:
        row = self.engine.predict_counts(request)
        self.calls += 1
        probabilities = row["probabilities"]["blend"]
        counts = self.engine.metadata.get("repertoire_counts", {}).get(str(row["pitcher_id"]))
        if counts is None:
            raise ValueError("missing frozen TRAIN repertoire counts")
        weights = np.array([counts.get(t, 0) for t in row["pitch_types"]], dtype=float)
        if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError("invalid frozen TRAIN repertoire weights")
        weights /= weights.sum()
        b, s = row["balls"], row["strikes"]
        distribution = {}
        for terminal in TERMINALS:
            basis = {name: float(name == terminal) for name in TERMINALS}
            plan = solve_pa(probabilities, basis, [0] * len(row["pitch_types"]),
                            baseline_policy=weights)
            distribution[terminal] = float(plan.baseline_values[b, s, 0])
        if self.first_distribution is None:
            self.first_distribution = distribution.copy()
        return distribution


def _checked_distribution(distribution: dict, tolerance: float) -> dict[str, float]:
    if set(distribution) != set(TERMINALS):
        raise ValueError("PA terminal distribution has wrong event set")
    values = {key: float(distribution[key]) for key in TERMINALS}
    if any(not math.isfinite(v) or v < -tolerance or v > 1 + tolerance for v in values.values()):
        raise ValueError("invalid PA terminal probability")
    mass = sum(values.values())
    if abs(mass - 1) > tolerance:
        raise ValueError(f"PA terminal probability mass error: {mass - 1}")
    # Floating-point drift only, after an explicit mass test.
    return {key: max(0.0, value) / sum(max(0.0, v) for v in values.values())
            for key, value in values.items()}


def _identity(initial_request: dict, lineup: list[dict], provider, config: dict,
              provider_identity: str | None) -> dict:
    if provider_identity is None:
        engine = getattr(provider, "engine", None)
        bundle = getattr(engine, "bundle", None)
        if bundle is None:
            raise ValueError("plain providers require explicit provider_identity")
        provider_identity = hashlib.sha256((Path(bundle) / "bundle_manifest.json").read_bytes()).hexdigest()
    if not isinstance(provider_identity, str) or not provider_identity:
        raise ValueError("provider_identity must be nonempty")
    state_and_count = {key: initial_request[key] for key in
                       ("date", "inning", "topbot", "outs", "bases", "home_score", "away_score", "balls", "strikes")}
    lineup_sha = hashlib.sha256(json.dumps(lineup, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    config_sha = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return {"provider_identity": provider_identity, "initial_state_and_count": state_and_count,
            "lineup_sha256": lineup_sha, "evaluation_config_sha256": config_sha,
            "policy_id": config["pa_policy_id"],
            "horizon": "inning_end", "initial_defender": "home" if initial_request["topbot"] == "Top" else "away"}


def evaluate_inning(*, initial_request: dict, lineup: list[dict], provider,
                    we, advancement, config: dict, provider_identity: str | None = None) -> dict:
    """Propagate exact PA-event mass; leave unmodeled future mass unresolved."""
    initial = _state(initial_request)
    tolerance = float(config["mass_tolerance"])
    identity = _identity(initial_request, lineup, provider, config, provider_identity)
    if initial.inning >= 10:
        return {"status": "unavailable", "reason": "extra_inning_initial_state_unsupported",
                "horizon": "inning_end", "value": None, "evaluation_identity": identity}
    if not lineup:
        return {"status": "unavailable", "reason": "missing_predecision_lineup",
                "horizon": "inning_end", "value": None, "evaluation_identity": identity}
    if initial_request["batter_stand"] != lineup[0]["batter_stand"] or initial_request["batter_profile"] != lineup[0]["batter_profile"]:
        raise ValueError("initial hitter differs from predecision lineup")
    if any(not isinstance(hitter.get("batter_profile"), dict) or
           hitter.get("batter_stand") not in ("L", "R") for hitter in lineup):
        raise ValueError("lineup requires explicit prior-date profiles and handedness")
    decision_date = date.fromisoformat(initial_request["date"])
    for hitter in lineup:
        as_of = hitter["batter_profile"].get("as_of")
        if not isinstance(as_of, str) or date.fromisoformat(as_of) >= decision_date:
            raise ValueError("each batter profile must predate the decision game")
    if any(hitter.get("known_before_decision") is not True for hitter in lineup[1:]):
        raise ValueError("future lineup slots require predecision order evidence")
    first_half = (initial.inning, initial.half)
    defender_home = initial.defender_is_home
    frontier = {initial: 1.0}
    absorbed_value = absorbed_mass = unresolved_mass = 0.0
    unresolved_reasons = defaultdict(float)
    calls = 0
    processed_pas = 0
    for index in range(config["maximum_pas"]):
        if not frontier:
            break
        next_frontier = defaultdict(float)
        for state, state_mass in sorted(frontier.items(), key=lambda item: (-item[1], repr(item[0]))):
            if index >= len(lineup):
                reason = "predecision_lineup_exhausted"
            elif state_mass < config["minimum_state_mass"]:
                reason = "minimum_state_mass"
            elif calls >= config["maximum_model_calls"]:
                reason = "maximum_model_calls"
            else:
                reason = None
            if reason:
                unresolved_mass += state_mass
                unresolved_reasons[reason] += state_mass
                continue
            request = _request_at(initial_request, state, lineup[index], index == 0)
            distribution = _checked_distribution(provider(request), tolerance)
            calls += 1
            processed_pas += 1
            for event, event_probability in distribution.items():
                if event_probability == 0:
                    continue
                advances = advancement.distribution(state, event)
                if any(not isinstance(next_state, GameState) or not math.isfinite(float(p)) or
                       float(p) < 0 for p, next_state in advances):
                    raise ValueError("invalid frozen advancement probability")
                advance_mass = sum(float(p) for p, _ in advances)
                if abs(advance_mass - 1) > tolerance:
                    raise ValueError("frozen advancement probability mass error")
                for probability, next_state in advances:
                    mass = state_mass * event_probability * probability
                    if next_state.winner is not None or (next_state.inning, next_state.half) != first_half:
                        value = float(we.predict_defense(next_state, defender_home))
                        if not math.isfinite(value) or not 0 <= value <= 1:
                            raise ValueError("frozen WE outside [0,1]")
                        absorbed_value += mass * value
                        absorbed_mass += mass
                    else:
                        next_frontier[next_state] += mass
        if len(next_frontier) > config["maximum_frontier_states"]:
            for mass in next_frontier.values():
                unresolved_mass += mass
                unresolved_reasons["maximum_frontier_states"] += mass
            frontier = {}
            break
        frontier = dict(next_frontier)
    if frontier:
        tail = sum(frontier.values())
        unresolved_mass += tail
        unresolved_reasons["maximum_pas"] += tail
    total_mass = absorbed_mass + unresolved_mass
    if abs(total_mass - 1) > tolerance:
        raise ValueError(f"inning probability mass error: {total_mass - 1}")
    lower = max(0.0, absorbed_value)
    upper = min(1.0, absorbed_value + unresolved_mass)
    complete = unresolved_mass <= tolerance
    return {"status": "complete" if complete else "bounded", "reason": None if complete else "unresolved_horizon_mass",
            "horizon": "inning_end", "initial_defender": "home" if defender_home else "away",
            "evaluation_identity": identity, "pitcher_id": initial_request["pitcher_id"],
            "value": lower if complete else None, "value_interval": [lower, upper],
            "absorbed_mass": absorbed_mass, "unresolved_mass": unresolved_mass,
            "unresolved_reasons": dict(unresolved_reasons), "mass_error": total_mass - 1,
            "processed_pas": processed_pas, "model_calls": calls,
            "lineup_slots_provided": len(lineup)}


def compare(keep: dict, substitute: dict, *, eligibility_verified: bool) -> dict:
    if not eligibility_verified:
        return {"status": "unavailable", "horizon": "inning_end", "value_pp": None,
                "reason": "candidate_eligibility_unverified"}
    if any(result.get("status") not in ("complete", "bounded") for result in (keep, substitute)):
        return {"status": "unavailable", "horizon": "inning_end", "value_pp": None,
                "reason": "inning_evaluation_unavailable"}
    required_identity = {"provider_identity", "initial_state_and_count", "lineup_sha256",
                         "evaluation_config_sha256", "policy_id", "horizon", "initial_defender"}
    if (not isinstance(keep.get("evaluation_identity"), dict) or
            not required_identity.issubset(keep["evaluation_identity"]) or
            keep["evaluation_identity"] != substitute.get("evaluation_identity") or
            keep.get("pitcher_id") == substitute.get("pitcher_id")):
        return {"status": "unavailable", "horizon": "inning_end", "value_pp": None,
                "reason": "mismatched_evaluation_identity_or_pitcher"}
    lo = 100 * (substitute["value_interval"][0] - keep["value_interval"][1])
    hi = 100 * (substitute["value_interval"][1] - keep["value_interval"][0])
    complete = keep["status"] == substitute["status"] == "complete"
    return {"status": "complete" if complete else "bounded", "horizon": "inning_end",
            "value_pp": lo if complete else None, "value_interval_pp": [lo, hi],
            "reason": None if complete else "unresolved_horizon_mass"}


def _write_new_result(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")


def run_c0_example(config_path: Path, output_path: Path) -> None:
    from minimal_pitch_service import Engine

    config = json.loads(config_path.read_text())
    example = json.loads((PROJECT / config["real_case_source"]).read_text())
    request = example["input"]["pitch"]["request"]
    bundle = Path(json.loads((PROJECT / "docs/contracts/model-v1.json").read_text())["bundle_path"])
    engine = Engine(bundle)
    manifest_sha = hashlib.sha256((bundle / "bundle_manifest.json").read_bytes()).hexdigest()
    provider = FrozenPAProvider(engine)
    # The C0 fixture has only its current hitter. All later hitters remain unknown.
    lineup = [{"batter_stand": request["batter_stand"],
               "batter_profile": request["batter_profile"]}]
    result = evaluate_inning(initial_request=request, lineup=lineup, provider=provider,
                             we=engine.we, advancement=engine.advancement, config=config)
    result.update(experiment_id=config["experiment_id"], algorithm_version=config["algorithm_version"],
                  case_kind="real_predecision_state_one_known_hitter_only",
                  pitch_id=example["pitch_id"], game_pk=example["source_game_id"],
                  initial_request=request, model_bundle_manifest_sha256=manifest_sha,
                  baseline_policy_id=config["pa_policy_id"],
                  source_dataset_sha256=example["source_dataset_sha256"],
                  first_pa_terminal_distribution=provider.first_distribution,
                  actual_replacement={"status": "unavailable", "horizon": "inning_end",
                                      "value_pp": None, "reason": "no_verified_eligible_substitute_or_predecision_lineup"})
    _write_new_result(output_path, result)


def run_roster_packet_screen(config_path: Path, packet_path: Path, output_path: Path) -> None:
    """Screen keep arms from a roster packet without asserting substitution eligibility."""
    from minimal_pitch_service import Engine

    config = json.loads(config_path.read_text())
    packet_bytes = packet_path.read_bytes()
    packet = json.loads(packet_bytes)
    bundle = Path(json.loads((PROJECT / "docs/contracts/model-v1.json").read_text())["bundle_path"])
    engine = Engine(bundle)
    decisions = []
    for decision in packet["decisions"]:
        keep_id = decision["keep_pitcher_id"]
        record = {"game_pk": decision["game_pk"],
                  "decision_first_observed_pitch_id": decision["decision_first_observed_pitch_id"],
                  "keep_pitcher_id": keep_id,
                  "current_batter_stand_source": "observed_under_incoming_pitcher",
                  "stand_transfer_to_keep_verified": False,
                  "actual_eligible_replacement_candidates": decision.get("actual_eligible_replacement_candidates"),
                  "replacement": {"status": "unavailable", "horizon": "inning_end", "value_pp": None,
                                  "reason": "candidate_eligibility_and_predecision_lineup_unverified"}}
        if str(keep_id) not in engine.metadata["pitchers"]:
            record.update(keep_evaluation_status="unavailable", reason="unsupported_keep_pitcher")
            decisions.append(record)
            continue
        state = decision["state_at_first_observed_pitch"]
        profile = engine.metadata["profiles"].get(str(state["current_batter_id"]))
        if profile is None:
            record.update(keep_evaluation_status="unavailable", reason="missing_prior_date_batter_profile")
            decisions.append(record)
            continue
        request = {"date": decision["game_date"], "pitcher_id": keep_id,
                   "inning": state["inning"], "topbot": state["half"], "outs": state["outs"],
                   "bases": state["bases"], "home_score": state["home_score"], "away_score": state["away_score"],
                   "balls": state["balls"], "strikes": state["strikes"],
                   "batter_stand": state["current_batter_stand"], "batter_profile": profile}
        provider = FrozenPAProvider(engine)
        evaluated = evaluate_inning(initial_request=request,
                                    lineup=[{"batter_stand": request["batter_stand"], "batter_profile": profile}],
                                    provider=provider, we=engine.we, advancement=engine.advancement,
                                    config=config)
        record.update(keep_evaluation_status=evaluated["status"], keep_evaluation=evaluated,
                      first_pa_terminal_distribution=provider.first_distribution,
                      batter_profile_as_of=profile["as_of"])
        decisions.append(record)
    payload = {"experiment_id": config["experiment_id"], "algorithm_version": config["algorithm_version"],
               "case_kind": "roster_packet_support_screen_only",
               "roster_packet_sha256": hashlib.sha256(packet_bytes).hexdigest(),
               "roster_packet_path": str(packet_path),
               "model_bundle_manifest_sha256": hashlib.sha256((bundle / "bundle_manifest.json").read_bytes()).hexdigest(),
               "decisions": decisions}
    _write_new_result(output_path, payload)


def run_two_out_amendment(amendment_path: Path) -> None:
    """Select the prespecified first supported two-out PA, then value one known hitter."""
    import pandas as pd
    from minimal_pitch_service import Engine

    amendment = json.loads(amendment_path.read_text())
    config = json.loads((PROJECT / amendment["evaluator_config"]).read_text())
    selection_manifest = json.loads((PROJECT / amendment["selection_manifest"]).read_text())
    source = Path(selection_manifest["dataset_paths"]["dev"])
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    if source_sha != selection_manifest["sha256"]["dev_full_games.parquet"]:
        raise ValueError("fixed DEV source SHA mismatch")
    bundle = Path(json.loads((PROJECT / "docs/contracts/model-v1.json").read_text())["bundle_path"])
    engine = Engine(bundle)
    frame = pd.read_parquet(source).sort_values(amendment["selection_order"])
    first_pitch = frame.drop_duplicates(["game_pk", "at_bat_number"], keep="first")
    selected = None
    for row in first_pitch.itertuples(index=False):
        if int(row.outs_when_up) != 2 or str(int(row.pitcher)) not in engine.metadata["pitchers"]:
            continue
        profile = engine.metadata["profiles"].get(str(int(row.batter)))
        game_date = str(pd.Timestamp(row.game_date).date())
        if profile is None or date.fromisoformat(profile["as_of"]) >= date.fromisoformat(game_date):
            continue
        if row.stand not in ("L", "R") or not (0 <= row.balls <= 3 and 0 <= row.strikes <= 2):
            continue
        try:
            GameState(int(row.inning), row.inning_topbot, int(row.outs_when_up), int(row.bases),
                      int(row.home_score), int(row.away_score))
        except ValueError:
            continue
        selected = (row, profile, game_date)
        break
    if selected is None:
        raise ValueError("no prespecified supported two-out PA exists")
    row, profile, game_date = selected
    request = {"date": game_date, "pitcher_id": int(row.pitcher),
               "inning": int(row.inning), "topbot": row.inning_topbot, "outs": int(row.outs_when_up),
               "bases": int(row.bases), "home_score": int(row.home_score), "away_score": int(row.away_score),
               "balls": int(row.balls), "strikes": int(row.strikes),
               "batter_stand": row.stand, "batter_profile": profile}
    provider = FrozenPAProvider(engine)
    result = evaluate_inning(initial_request=request,
                             lineup=[{"batter_stand": row.stand, "batter_profile": profile}],
                             provider=provider, we=engine.we, advancement=engine.advancement,
                             config=config)
    result.update(experiment_id=config["experiment_id"], amendment_id=amendment["amendment_id"],
                  case_kind="prespecified_real_two_out_state_one_known_hitter_only",
                  selected_pitch_id=f"{int(row.game_pk)}:{int(row.at_bat_number)}:{int(row.pitch_number)}",
                  source_dev_sha256=source_sha, first_pa_terminal_distribution=provider.first_distribution,
                  actual_replacement={"status": "unavailable", "horizon": "inning_end",
                                      "value_pp": None, "reason": "no_verified_eligible_substitute_or_predecision_lineup"})
    output = PROJECT / amendment["output"]
    _write_new_result(output, result)


def run_conditional_anchor(amendment_path: Path) -> None:
    """Evaluate only a fixed-lineup keep scenario from a pre-change source anchor."""
    from minimal_pitch_service import Engine

    config = json.loads(amendment_path.read_text())
    anchor_path = Path(config["source_anchor_path"])
    roster_path = Path(config["source_roster_path"])
    anchor_bytes, roster_bytes = anchor_path.read_bytes(), roster_path.read_bytes()
    if hashlib.sha256(anchor_bytes).hexdigest() != config["source_anchor_sha256"] or \
       hashlib.sha256(roster_bytes).hexdigest() != config["source_roster_sha256"]:
        raise ValueError("conditional anchor source SHA mismatch")
    anchor, roster = json.loads(anchor_bytes), json.loads(roster_bytes)
    game = config["game_pk"]
    matches = [row for row in roster["decisions"] if row["game_pk"] == game]
    if len(matches) != 1 or matches[0]["game_date"] != config["game_date"]:
        raise ValueError("official roster game date does not match frozen amendment")
    if (anchor["game_pk"] != game or anchor["keep_pitcher_id"] != config["expected_keep_pitcher_id"] or
            anchor["decision_first_observed_pitch_id"] != matches[0]["decision_first_observed_pitch_id"] or
            anchor["anchor"] != "immediately_before_logged_pitching_substitution_action" or
            anchor["cutoff_index_verified"] is not True or anchor["cutoff_timestamp_verified"] is not True or
            anchor["eligible_replacements"] is not None):
        raise ValueError("pre-change anchor or eligibility evidence mismatch")
    cutoff = datetime.fromisoformat(anchor["anchor_action_start_time_utc"].replace("Z", "+00:00"))
    latest = datetime.fromisoformat(anchor["source_play_end_time_max"].replace("Z", "+00:00"))
    if latest >= cutoff:
        raise ValueError("lineup source play reaches pitching decision")
    slots = sorted(anchor["lineup_as_of"]["ordered_slots"], key=lambda entry: entry["slot"])
    if [item["slot"] for item in slots] != list(range(1, 10)) or len({item["batter_id"] for item in slots}) != 9:
        raise ValueError("predecision batting slots incomplete or duplicated")
    next_slot = anchor["lineup_as_of"]["next_slot_one_based"]
    if not 1 <= next_slot <= 9 or slots[next_slot - 1]["batter_id"] != anchor["lineup_as_of"]["next_batter_id"]:
        raise ValueError("current predecision batting slot mismatch")
    bundle = Path(json.loads((PROJECT / "docs/contracts/model-v1.json").read_text())["bundle_path"])
    engine = Engine(bundle)
    if str(anchor["keep_pitcher_id"]) not in engine.metadata["pitchers"]:
        raise ValueError("frozen model does not support anchored keep pitcher")
    ordered = slots[next_slot - 1:] + slots[:next_slot - 1]
    lineup = []
    default_ids = []
    for item in ordered:
        stand = item["stand_vs_keep"]
        if stand not in ("L", "R") or not item["stand_evidence"]:
            raise ValueError("missing stance evidence against keep pitcher")
        for evidence in item["stand_evidence"]:
            time = datetime.fromisoformat(evidence["play_end_time"].replace("Z", "+00:00"))
            if time >= cutoff or evidence["pitcher_id"] != anchor["keep_pitcher_id"] or \
               evidence["observed_stand"] != stand:
                raise ValueError("stance evidence postdates or conflicts with decision")
        batter_id = item["batter_id"]
        profile = engine.metadata["profiles"].get(str(batter_id))
        profile_source = "frozen_batter_snapshot"
        if profile is None:
            profile = engine.metadata["default_profile"]
            profile_source = "frozen_default_zero_reliability"
            default_ids.append(batter_id)
        if date.fromisoformat(profile["as_of"]) >= date.fromisoformat(config["game_date"]):
            raise ValueError("batter profile is not prior date")
        lineup.append({"batter_id": batter_id, "batter_stand": stand,
                       "batter_profile": profile, "profile_source": profile_source,
                       "known_before_decision": True})
    state = anchor["state_as_of_anchor"]
    first = lineup[0]
    request = {"date": matches[0]["game_date"], "pitcher_id": anchor["keep_pitcher_id"],
               "inning": state["inning"], "topbot": state["half"], "outs": state["outs"],
               "bases": state["bases"], "home_score": state["home_score"], "away_score": state["away_score"],
               "balls": state["balls"], "strikes": state["strikes"],
               "batter_stand": first["batter_stand"], "batter_profile": first["batter_profile"]}
    provider = FrozenPAProvider(engine)
    result = evaluate_inning(initial_request=request, lineup=lineup, provider=provider,
                             we=engine.we, advancement=engine.advancement, config=config)
    result.update(amendment_id=config["amendment_id"], game_pk=game,
                  case_kind="conditional_fixed_prechange_lineup_keep_pitcher",
                  decision_anchor="immediately_before_logged_pitching_substitution_action",
                  source_anchor_sha256=config["source_anchor_sha256"],
                  source_roster_sha256=config["source_roster_sha256"],
                  model_bundle_manifest_sha256=hashlib.sha256((bundle / "bundle_manifest.json").read_bytes()).hexdigest(),
                  first_pa_terminal_distribution=provider.first_distribution,
                  lineup_batter_ids=[h["batter_id"] for h in lineup],
                  lineup_profile_sources=[h["profile_source"] for h in lineup],
                  default_profile_batter_ids=default_ids,
                  scenario_assumption=config["lineup_assumption"],
                  actual_replacement={"status": "unavailable", "horizon": "inning_end",
                                      "value_pp": None, "reason": "actual_eligible_substitutes_unverified"})
    _write_new_result(PROJECT / config["output"], result)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT / "experiments/pitchmdp/configs/EXP-C-INNING-001.json")
    parser.add_argument("--output", type=Path, default=PROJECT / "results/EXP-C-INNING-001/real_state.json")
    parser.add_argument("--roster-packet", type=Path)
    parser.add_argument("--two-out-amendment", type=Path)
    parser.add_argument("--conditional-anchor", type=Path)
    args = parser.parse_args()
    if args.conditional_anchor:
        run_conditional_anchor(args.conditional_anchor)
    elif args.two_out_amendment:
        run_two_out_amendment(args.two_out_amendment)
    elif args.roster_packet:
        run_roster_packet_screen(args.config, args.roster_packet, args.output)
    else:
        run_c0_example(args.config, args.output)
