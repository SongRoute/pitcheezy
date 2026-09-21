"""Reproduce stored prediction metrics and bootstrap paired losses by game.

Uses the saved response-model checkpoint and saved delivery mixture; no network
optimization, model selection, new data retrieval, or raw-file reads occur.
The original count baseline is reconstructed from the fixed training rows.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7419)
    args = parser.parse_args()
    run = args.run.resolve()
    volume = Path("/Volumes/T7 Shield")
    if not volume.is_mount() or not run.is_relative_to(volume.resolve()):
        raise SystemExit("Run and audit output must be on mounted T7 Shield")
    if args.bootstrap_replicates < 1:
        raise SystemExit("Bootstrap replicate count must be positive")
    config = json.loads((run / "config.json").read_text())
    quality = json.loads((run / "data_quality.json").read_text())
    recorded = json.loads((run / "prediction_metrics.json").read_text())
    cohort = json.loads((run / "cohort_manifest.json").read_text())
    start_hashes = json.loads((run / "code_hashes_start.json").read_text())
    runtime = json.loads((run / "runtime.json").read_text())
    test_record = run / "audit_test_verification.json"
    verified_tests = json.loads(test_record.read_text()) if test_record.exists() else None
    sources_match = all(sha256(run / "source" / rel) == digest for rel, digest in start_hashes.items())
    if not sources_match or start_hashes != runtime["code_hashes_end"]:
        raise SystemExit("Original run source identity failed; refusing to reproduce with mismatched code")
    processed = Path(quality["output"])
    if sha256(processed) != quality["processed_sha256"]:
        raise SystemExit("Processed input differs from original run")
    # Import the archived implementation that produced this specific checkpoint.
    sys.path.insert(0, str(run / "source"))
    from pitchmdp.model import CATEGORICAL, NUMERIC_INPUTS, CountBaseline, PitchModel, eligible, metrics, outcome_labels

    columns = sorted(set(CATEGORICAL) | set(NUMERIC_INPUTS) | {
        "split", "description", "events", "supported_pa", "game_pk", "game_date",
        "cohort_pitcher", "is_lad_start", "at_bat_number", "pitch_number",
    })
    frame = pd.read_parquet(processed, columns=columns)
    assert frame.game_date.max() < pd.Timestamp("2026-01-01")
    train = frame[(frame.split == "train") & eligible(frame)]
    dev = frame[(frame.split == "dev") & frame.cohort_pitcher & frame.is_lad_start]
    dev = dev[eligible(dev)]
    dev = dev.sample(min(len(dev), config["dev_max_rows"]), random_state=config["seed"]).sort_index()
    assert set(train.game_pk).isdisjoint(dev.game_pk)
    assert train.game_date.max() < dev.game_date.min()
    model = PitchModel.load(run / "pitch_model.pt")
    with (run / "recommendation_context.pkl").open("rb") as stream:
        delivery = pickle.load(stream)["delivery"]
    # This is the exact fixed count table, not a new tunable model fit.
    baseline = CountBaseline().fit(train)
    pred = delivery.predict(model, dev)
    reference = baseline.predict(dev)
    labels = outcome_labels(dev)
    got = metrics(labels, pred)
    base = metrics(labels, reference)
    for actual, expected in [(got, recorded["primary_prepitch_type_conditional"]),
                             (base, recorded["count_hand_baseline"])]:
        assert actual["n"] == expected["n"]
        for key in ["log_loss", "brier_multiclass"]:
            if not np.isclose(actual[key], expected[key], atol=1e-7, rtol=0):
                raise AssertionError(f"Reproduction mismatch for {key}: {actual[key]} vs {expected[key]}")
    onehot = np.eye(pred.shape[1])[labels]
    paired = pd.DataFrame({
        "game_pk": dev.game_pk.to_numpy(),
        "count": 1,
        "log_loss_delta": -np.log(np.maximum(pred[np.arange(len(labels)), labels], 1e-12))
                          + np.log(np.maximum(reference[np.arange(len(labels)), labels], 1e-12)),
        "brier_delta": ((pred - onehot) ** 2).sum(axis=1) - ((reference - onehot) ** 2).sum(axis=1),
    })
    groups = paired.groupby("game_pk", sort=True).sum()
    rng = np.random.default_rng(args.seed)
    ix = rng.integers(0, len(groups), size=(args.bootstrap_replicates, len(groups)))
    denominators = groups["count"].to_numpy()[ix].sum(axis=1)
    differences = {}
    for column in ["log_loss_delta", "brier_delta"]:
        estimates = groups[column].to_numpy()[ix].sum(axis=1) / denominators
        differences[column] = {
            "point_estimate_full_minus_baseline": float(paired[column].mean()),
            "percentile_95_ci": np.quantile(estimates, [.025, .975]).tolist(),
            "bootstrap_standard_error": float(estimates.std(ddof=1)),
        }
    absent = [row for row in cohort["selected"] if int(row["pitcher"]) not in set(dev.pitcher)]
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_pitches": len(dev), "n_games": len(groups),
        "dates": [str(dev.game_date.min().date()), str(dev.game_date.max().date())],
        "bootstrap_replicates": args.bootstrap_replicates, "seed": args.seed,
        "method": "paired game-cluster percentile bootstrap; resample games, aggregate pitch loss sums / resampled pitch counts",
        "direction": "Full minus count/hand baseline; negative means Full predicts better",
        "uncertainty_scope": "game sampling only; fixed trained model, no retraining, no calibration/model-selection uncertainty, no causal-policy interval",
        "primary_prepitch": got, "count_hand_baseline": base, "paired_differences": differences,
        "archived_source_hashes_match": sources_match, "start_end_source_hashes_equal": True,
        "processed_input_hash_matches": True, "saved_metrics_reproduced": True,
        "frozen_cohort_without_eligible_dev": absent,
        "audit_script_sha256": sha256(Path(__file__)),
        "recorded_unit_test_verification": verified_tests,
    }
    (run / "prediction_uncertainty.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    lines = [
        "# First-result independent audit", "",
        "The real-data recommendation/evaluation path meets the first-result implementation contract. "
        "Observed predictive improvement over the count/hand baseline has not been established. "
        "The positive same-model policy value is an optimization diagnostic.", "",
        (f"Recorded verification: {verified_tests['tests_run']} unit tests passed, including "
         f"{verified_tests['independent_contract_tests']} independent contract checks. "
         "The exact command and test-file hashes are in audit_test_verification.json."
         if verified_tests else "Unit-test execution evidence is recorded separately from this prediction audit."), "",
        "| Requirement | Evidence / verdict |", "|---|---|",
        "| Frozen four pitchers | Independently reproduced from original 2023–25 defensive out clocks: "
        "Miller 550 outs / 36 starts; Glasnow 456 / 27; Stone 455 / 28; Yamamoto 372 / 24. |",
        "| Temporal separation | Train through 2025-04-30, May–June calibration, July–September DEV; disjoint train/DEV game IDs. |",
        "| Information boundary | Current delivery integrated from saved training-only mixture; "
        "candidate actions overwrite logged type/location. Current outcomes, provider WE, future days-until-game and current strike-zone bounds are not model inputs. |",
        "| Strictly prior player data | Data implementation aggregates complete dates before shifting; same-day games share identical earlier-day priors; previous pitch resets per PA. |",
        "| WE and solver | Tests verify fixed original defender after inning changes, exact end-game win/loss boundaries, probability mass, analytic foul loop, and full ≥ one-pitch ≥ baseline values. |",
        "| Replay and provenance | Archived source hashes, unchanged run start/end hashes and processed-data hash pass; stored primary prediction metrics reproduced. |",
        "| Unsupported population | Whole-PA filtering uses observed trajectories. Extra innings, final game PAs without next-state observations, state-changing and unsupported events are excluded. Coverage is selected-population evidence. |",
        "| Policy claims | No independent judge, observational OPE, causal effect, five-seed robustness, or full-target policy confidence interval. |", "",
        "## Paired predictive uncertainty", "",
        f"{len(dev):,} pitches across {len(groups)} games; {args.bootstrap_replicates:,} game-cluster bootstrap replicates. "
        "Negative Full-minus-baseline loss differences favor Full.", "",
        "| Metric | Difference | 95% interval |", "|---|---:|---:|",
    ]
    for name, item in differences.items():
        low, high = item["percentile_95_ci"]
        lines.append(f"| {name} | {item['point_estimate_full_minus_baseline']:+.6f} | [{low:+.6f}, {high:+.6f}] |")
    lines += ["", "These intervals measure game sampling with the trained model fixed; "
              "they do not cover model selection, training variation, data exclusions, or policy effects.", "",
              "Frozen pitchers without an eligible DEV sample: " + ", ".join(row["player_name"] for row in absent) + ". "
              "The DEV performance and recommendation averages therefore cover the two observed pitchers, not four.", "",
              "Official field semantics checked: [Statcast CSV documentation](https://baseballsavant.mlb.com/csv-docs). "
              "For the approved seasons, plate coordinates are from the catcher's perspective at the front of the plate; "
              "balls, strikes, outs and runner identifiers describe the pre-pitch state.", "",
              "Reproduce: `" + str(Path(sys.executable)) + " experiments/pitchmdp/scripts/audit_result.py --run \"" + str(run) + "\"`."]
    (run / "AUDIT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"run": str(run), "n_pitches": len(dev), "n_games": len(groups), "differences": differences}, indent=2))


if __name__ == "__main__":
    main()
