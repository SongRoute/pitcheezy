"""Fixed exploratory subgroup diagnostics of primary blend versus chosen baseline."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import numpy as np
import pandas as pd

from pitchmdp.data import KEY, hash_file
from pitchmdp.model import OUTCOMES, outcome_labels
from run_sequence_pilot import dump
from audit_sequence_robustness import independent_bootstrap, scores

SCOPE = ("Fixed exploratory subgroup descriptions on previously inspected DEV. "
         "All prespecified groups retained; no subgroup/model selection. "
         "Intervals are unadjusted for multiple comparisons and cannot support confirmatory subgroup claims.")


def summarize_subset(model_scores, baseline_scores, games, mask):
    mask = np.asarray(mask, dtype=bool)
    n = int(mask.sum())
    if not n:
        return {"n": 0, "games": 0, "metrics": {}, "intervals": None}
    left = {key: values[mask][None] for key, values in model_scores.items()}
    right = {key: values[mask][None] for key, values in baseline_scores.items()}
    boot = independent_bootstrap(left, right, games[mask], [0], replicates=2000)
    return {"n": n, "games": int(len(np.unique(games[mask]))),
            "metrics": {key: {"blend": float(left[key].mean()), "baseline": float(right[key].mean()),
                              "blend_minus_baseline": float((left[key]-right[key]).mean()),
                              "descriptive_game_bootstrap95": item["game_only_bootstrap95"]}
                        for key, item in boot.items()},
            "interval_scope": "2000 paired whole-game resamples within this group; fitted ensemble/baseline and CAL weights fixed; no multiplicity adjustment"}


def check_partition(group_masks, model_scores, baseline_scores):
    """Hard guard: exhaustive disjoint groups must exactly recover total losses."""
    membership = np.stack(list(group_masks.values()))
    if membership.dtype != bool or not np.all(membership.sum(axis=0) == 1):
        raise ValueError("Subgroups must be mutually exclusive and exhaustive")
    result = {}
    total = membership.shape[1]
    for metric in model_scores:
        overall = float((model_scores[metric]-baseline_scores[metric]).mean())
        weighted = float(sum((model_scores[metric][mask]-baseline_scores[metric][mask]).sum()/total
                             for mask in membership))
        np.testing.assert_allclose(weighted, overall, atol=1e-12, rtol=1e-12)
        for collection in (model_scores, baseline_scores):
            reconstructed = sum(collection[metric][mask].sum()/total for mask in membership)
            np.testing.assert_allclose(reconstructed, collection[metric].mean(), atol=1e-12, rtol=1e-12)
        result[metric] = {"overall_delta": overall, "weighted_group_delta": weighted, "passed": True}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blend", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    local = json.loads((PROJECT/"configs/local.json").read_text())
    root, volume = Path(local["artifact_root"]).resolve(), Path("/Volumes/T7 Shield")
    if not volume.is_mount() or not root.is_relative_to(volume.resolve()) or Path(sys.executable).absolute() != Path(local["python"]).absolute():
        raise SystemExit("Use the configured mounted SSD and existing Python")
    blend = args.blend.resolve()
    if not blend.is_relative_to(root) or not (blend/"runtime.json").exists():
        raise SystemExit("Completed strong-blend SSD run required")
    blend_config = json.loads((blend/"config.json").read_text())
    blend_results = json.loads((blend/"results.json").read_text())
    frequency = Path(blend_config["frequency_run"])
    frequency_config = json.loads((frequency/"config.json").read_text())
    base = Path(blend_config["base_run"])
    cohort = json.loads((base/"cohort_manifest.json").read_text())
    pitcher_ids = cohort["pitcher_ids"]
    if len(pitcher_ids) != 6 or len(set(pitcher_ids)) != 6:
        raise ValueError("This prespecified diagnostic requires the frozen six-pitcher cohort")
    processed = root/"processed/pitches.parquet"
    if hash_file(processed) != frequency_config["processed_sha256"]:
        raise ValueError("Processed source differs from the fixed comparison")
    if hash_file(blend/"predictions.npz") != blend_results["predictions_sha256"]:
        raise ValueError("Saved blend probabilities changed")
    protocol = PROJECT/"docs/FREQUENCY_BLEND_SUBGROUP_PROTOCOL.md"
    paths = [Path(__file__).resolve(), PROJECT/"scripts/audit_sequence_robustness.py", PROJECT/"scripts/run_sequence_pilot.py",
             PROJECT/"pitchmdp/data.py", PROJECT/"pitchmdp/model.py", protocol]
    sources = {str(path.relative_to(PROJECT)): hash_file(path) for path in paths}
    references = [blend/name for name in ("config.json", "results.json", "selection.json", "predictions.npz", "runtime.json")]
    references += [base/"cohort_manifest.json", frequency/"config.json"]
    reference_hashes = {str(path): hash_file(path) for path in references}
    output = (args.output or blend/datetime.now(timezone.utc).strftime("subgroups-%Y%m%dT%H%M%SZ")).resolve()
    if not output.is_relative_to(root) or output == blend or blend.is_relative_to(output):
        raise SystemExit("Use a distinct SSD diagnostic directory")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"blend_run": str(blend), "model_array": "strong_blend_log_loss", "baseline_array": "selected_frequency_baseline",
                "baseline_chosen_on_CAL4000": blend_config["selected_frequency_baseline"],
                "partitions": {"pitcher": pitcher_ids, "strikes": [0, 1, 2], "PA_history": ["first_logged_pitch", "prior_same_PA_pitch"]},
                "PA_history_rule": "processed prev_pitch_type==START marks first logged pitch; otherwise earlier same-PA history exists",
                "outcome_classes": list(OUTCOMES), "probability_regime": "original, before mechanical legality conditioning",
                "scope": SCOPE, "source_hashes": sources, "reference_hashes": reference_hashes,
                "processed_sha256": frequency_config["processed_sha256"]}
    dump(output/"config.json", manifest)
    for rel in sources:
        target = output/"source"/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT/rel, target)
    shutil.copyfile(protocol, output/protocol.name)
    with np.load(blend/"predictions.npz", allow_pickle=False) as saved:
        keys, y, games = saved["pitch_keys"].copy(), saved["y"].copy(), saved["game_pk"].copy()
        model, baseline = saved["strong_blend_log_loss"].copy(), saved["selected_frequency_baseline"].copy()
    columns = [*KEY, "game_date", "pitcher", "player_name", "strikes", "prev_pitch_type", "description", "events"]
    frame = pd.read_parquet(processed, columns=columns)
    if frame.duplicated(KEY).any() or frame[KEY].isna().any().any() or not pd.to_datetime(frame.game_date).dt.year.isin([2023, 2024, 2025]).all():
        raise ValueError("Invalid or forbidden processed rows")
    lookup = pd.MultiIndex.from_arrays(keys.T, names=KEY)
    rows = frame.set_index(KEY).reindex(lookup).reset_index()
    if rows[["game_date", "pitcher", "strikes", "prev_pitch_type"]].isna().any().any():
        raise ValueError("Saved prediction keys lack required subgroup state")
    np.testing.assert_array_equal(rows[KEY].to_numpy(), keys)
    np.testing.assert_array_equal(rows.game_pk.to_numpy(), games)
    np.testing.assert_array_equal(outcome_labels(rows), y)
    if not rows.pitcher.isin(pitcher_ids).all() or not rows.strikes.isin([0, 1, 2]).all():
        raise ValueError("Unexpected cohort pitcher or strike count")
    model_scores, baseline_scores = scores(y, model), scores(y, baseline)
    for key in model_scores:
        np.testing.assert_allclose(model_scores[key].mean(), blend_results["metrics"]["strong_blend_log_loss"][key], atol=1e-12)
        np.testing.assert_allclose(baseline_scores[key].mean(), blend_results["metrics"]["selected_frequency_baseline"][key], atol=1e-12)
    first = rows.prev_pitch_type.eq("START").to_numpy()
    partitions = {"pitcher": {str(pid): rows.pitcher.eq(pid).to_numpy() for pid in pitcher_ids},
                  "strikes": {str(count): rows.strikes.eq(count).to_numpy() for count in [0, 1, 2]},
                  "PA_history": {"first_logged_pitch": first, "prior_same_PA_pitch": ~first}}
    result = {"scope": SCOPE, "n": len(y), "games": int(len(np.unique(games))),
              "overall": summarize_subset(model_scores, baseline_scores, games, np.ones(len(y), bool)),
              "partitions": {}, "partition_checks": {}, "outcome_contributions": [],
              "first_logged_vs_pitch_number1_disagreements": int((first != rows.pitch_number.eq(1).to_numpy()).sum())}
    for name, masks in partitions.items():
        result["partition_checks"][name] = check_partition(masks, model_scores, baseline_scores)
        result["partitions"][name] = {}
        for group, mask in masks.items():
            summary = summarize_subset(model_scores, baseline_scores, games, mask)
            if name == "pitcher" and mask.any():
                summary["player_name"] = str(rows.loc[mask, "player_name"].iloc[0])
            result["partitions"][name][group] = summary
    class_masks = {str(label): y == label for label in range(10)}
    result["partition_checks"]["outcome"] = check_partition(class_masks, model_scores, baseline_scores)
    for label, outcome in enumerate(OUTCOMES):
        mask = y == label
        delta = model_scores["log_loss"][mask]-baseline_scores["log_loss"][mask]
        result["outcome_contributions"].append({"class": label, "outcome": outcome, "n": int(mask.sum()),
            "games": int(len(np.unique(games[mask]))), "rate": float(mask.mean()),
            "mean_log_loss_blend": float(model_scores["log_loss"][mask].mean()) if mask.any() else None,
            "mean_log_loss_baseline": float(baseline_scores["log_loss"][mask].mean()) if mask.any() else None,
            "mean_delta_log_loss": float(delta.mean()) if mask.any() else None,
            "contribution_to_overall_delta": float(delta.sum()/len(y))})
    np.testing.assert_allclose(sum(row["contribution_to_overall_delta"] for row in result["outcome_contributions"]),
                               result["overall"]["metrics"]["log_loss"]["blend_minus_baseline"], atol=1e-12)
    np.savez_compressed(output/"subgroup_rows.npz", pitch_keys=keys, y=y, game_pk=games, pitcher=rows.pitcher.to_numpy(),
                        strikes=rows.strikes.to_numpy(), first_logged_PA_pitch=first,
                        **{"delta_"+metric: model_scores[metric]-baseline_scores[metric] for metric in model_scores})
    result["subgroup_rows_sha256"] = hash_file(output/"subgroup_rows.npz")
    dump(output/"subgroup_results.json", result)
    if any(hash_file(PROJECT/rel) != digest for rel, digest in sources.items()) or any(hash_file(Path(path)) != digest for path, digest in reference_hashes.items()):
        raise RuntimeError("Source or reference changed during CPU diagnostic")
    dump(output/"runtime.json", {"finished_at_utc": datetime.now(timezone.utc).isoformat(), "partition_checks_passed": True,
                                "training": False, "inference": False})
    print(json.dumps({"output": str(output), "overall": result["overall"], "partition_checks_passed": True}, indent=2))


if __name__ == "__main__":
    main()
