"""CPU reblend of fixed context frequencies and saved 100-draw ensemble forecasts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import numpy as np

from pitchmdp.data import hash_file
from pitchmdp.sequence_model import classification_metrics
from run_sequence_frequency_blend import fit_selected_blends, apply_selected_blends, paired_fixed_predictors, archive_arrays
from run_sequence_pilot import dump
from diagnose_sequence_legality import condition_on_legality

SEEDS = [42, 43, 44, 45, 46]


def key_hash(keys):
    return hashlib.sha256(np.asarray(keys, dtype=np.int64).tobytes()).hexdigest()


def verify_partition_keys(temperature_keys, blend_keys, dev_keys):
    """Exact fixed-size, unique and disjoint keyed CAL/DEV partitions."""
    expected = (4000, 12000, 7276)
    groups = []
    for values, n in zip((temperature_keys, blend_keys, dev_keys), expected):
        values = np.asarray(values)
        if values.shape != (n, 3):
            raise ValueError("Expected unchanged4000/12000/7276 three-column pitch keys")
        unique = set(map(tuple, values.tolist()))
        if len(unique) != n:
            raise ValueError("Duplicate pitch keys in an evaluation partition")
        groups.append(unique)
    if any(not groups[i].isdisjoint(groups[j]) for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Temperature, blending and DEV keys must be disjoint")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=Path, required=True, help="Completed 25-draw context-frequency run")
    parser.add_argument("--calibration100", type=Path, required=True, help="Completed compatible 100-draw ensemble archives")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    local = json.loads((PROJECT/"configs/local.json").read_text())
    root, volume = Path(local["artifact_root"]).resolve(), Path("/Volumes/T7 Shield")
    if not volume.is_mount() or not root.is_relative_to(volume.resolve()) or Path(sys.executable).absolute() != Path(local["python"]).absolute():
        raise SystemExit("Use configured mounted SSD and existing Python")
    context, calibration = args.context.resolve(), args.calibration100.resolve()
    for path in (context, calibration):
        if not path.is_relative_to(root) or not (path/"runtime.json").exists():
            raise SystemExit("Wait for both completed input runs: "+str(path))
    old_config = json.loads((context/"config.json").read_text())
    old_results = json.loads((context/"results.json").read_text())
    old_data = json.loads((context/"data.json").read_text())
    original_selection = json.loads((context/"selection.json").read_text())
    config100 = json.loads((calibration/"config.json").read_text())
    results100 = json.loads((calibration/"results.json").read_text())
    if config100["delivery_draws"] != 100 or config100["seeds"] != SEEDS:
        raise ValueError("This protocol requires all five fixed full models with100 delivery draws")
    frequency = Path(old_config["frequency_run"])
    base = json.loads((frequency/"config.json").read_text())["base_run"]
    if config100["base_run"] != base:
        raise ValueError("25 and100-draw comparisons use different base runs")
    chosen = original_selection["baseline"]["chosen"]
    if chosen != "league_context" or old_results["selection"] != original_selection:
        raise ValueError("Keep the original CAL-chosen league-context baseline and temperatures")
    if hash_file(context/"predictions.npz") != old_results["predictions_sha256"]:
        raise ValueError("25-draw probability archive changed")
    for name, result in old_results["baselines"].items():
        if hash_file(Path(result["model_path"])) != result["model_sha256"]:
            raise ValueError("Context-frequency tables changed: "+name)
    if (results100["temperature_row_hash"] != old_data["temperature_rows_hash"] or
            results100["calibration_row_hash"] != old_data["blend_rows_hash"] or
            results100["dev_row_hash"] != old_data["ordered_rows_hash"]["dev"]):
        raise ValueError("100-draw and frozen25-draw CAL/DEV row hashes differ")
    for path, digest in config100["reference_hashes"].items():
        if hash_file(Path(path)) != digest:
            raise ValueError("100-draw frozen input changed: "+path)
    protocol = PROJECT/"docs/CONTEXT_BLEND_100_PROTOCOL.md"
    scripts = ["run_sequence_context_blend100.py", "run_sequence_frequency_blend.py", "run_sequence_frequency_baselines.py",
               "run_sequence_calibration.py", "run_sequence_ablations.py", "run_sequence_pilot.py",
               "audit_sequence_robustness.py", "diagnose_sequence_legality.py"]
    source_paths = [protocol, *[PROJECT/"scripts"/name for name in scripts], *sorted((PROJECT/"pitchmdp").glob("*.py"))]
    sources = {str(path.relative_to(PROJECT)): hash_file(path) for path in source_paths}
    references = [context/name for name in ("config.json", "data.json", "results.json", "selection.json", "runtime.json",
                                           "temperature_predictions.npz", "calibration_predictions.npz", "predictions.npz")]
    references += [calibration/name for name in ("config.json", "results.json", "runtime.json", "calibration_predictions.npz", "predictions.npz")]
    references += [Path(result["model_path"]) for result in old_results["baselines"].values()]
    reference_hashes = {str(path): hash_file(path) for path in references}
    output = (args.output or calibration/datetime.now(timezone.utc).strftime("context-reblend-%Y%m%dT%H%M%SZ")).resolve()
    if not output.is_relative_to(root) or any(output == path or path.is_relative_to(output) for path in (context, calibration)):
        raise SystemExit("Use a distinct SSD output directory")
    output.mkdir(parents=True, exist_ok=False)
    config = {"context25_run": str(context), "calibration100_run": str(calibration), "draws": 100, "seeds": SEEDS,
              "fixed_baseline": chosen, "baseline_temperature": original_selection["baseline"]["candidates"][chosen]["temperature"],
              "primary": "log_loss", "secondary": "brier_multiclass", "baseline_reselected": False, "baseline_refitted": False,
              "source_hashes": sources, "reference_hashes": reference_hashes,
              "scope": "Exploratory integration-count sensitivity on inspected DEV;25 and100draw pools are nonnested; no convergence claim",
              "fitting": "Only two convex ensemble weights on the exact same12000 CAL rows; no training/table fitting/temperature fitting/inference"}
    dump(output/"config.json", config)
    for rel in sources:
        target = output/"source"/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT/rel, target)
    shutil.copyfile(protocol, output/protocol.name)
    print("CONTEXT_REBLEND100_DIR="+str(output), flush=True)
    with np.load(context/"temperature_predictions.npz", allow_pickle=False) as saved:
        temperature_keys = saved["pitch_keys"].copy()
    with np.load(context/"calibration_predictions.npz", allow_pickle=False) as saved:
        cal_keys, cy, cal_games = saved["pitch_keys"].copy(), saved["y"].copy(), saved["game_pk"].copy()
        fixed_baseline_cal = saved["selected_baseline"].copy()
    if key_hash(temperature_keys) != old_data["temperature_rows_hash"] or key_hash(cal_keys) != old_data["blend_rows_hash"]:
        raise ValueError("Frozen CAL archives fail exact key hashes")
    cal100 = archive_arrays(calibration/"calibration_predictions.npz", {"pitch_keys": cal_keys, "y": cy, "game_pk": cal_games})
    if cal100["seed_predictions"].shape != (5, 12000, 10):
        raise ValueError("All five fixed100-draw ensemble members required")
    np.testing.assert_allclose(cal100["ensemble"], cal100["seed_predictions"].mean(axis=0), atol=1e-7, rtol=1e-7)
    fitted = fit_selected_blends(cy, cal100["ensemble"], fixed_baseline_cal)
    selection = {"baseline_selection_preserved": original_selection["baseline"], "fixed_baseline": chosen,
                 "temperature_row_hash": key_hash(temperature_keys), "weight_calibration_row_hash": key_hash(cal_keys),
                 "weights": fitted, "primary": "log_loss", "secondary": "brier_multiclass", "dev_selection": False}
    dump(output/"selection.json", selection)
    # Read DEV probability payloads only after the two CAL weights are persisted.
    with np.load(context/"predictions.npz", allow_pickle=False) as saved:
        dy, keys, games = saved["y"].copy(), saved["pitch_keys"].copy(), saved["game_pk"].copy()
        fixed_baseline_dev = saved[chosen+"__tempered"].copy()
        old_primary, old_secondary = saved["context_blend_log_loss"].copy(), saved["context_blend_brier_multiclass"].copy()
        impossible = saved["impossible_dp"].copy()
    verify_partition_keys(temperature_keys, cal_keys, keys)
    if key_hash(keys) != old_data["ordered_rows_hash"]["dev"] or int(len(np.unique(games))) != old_data["dev_games"]:
        raise ValueError("DEV sample identity differs")
    dev100 = archive_arrays(calibration/"predictions.npz", {"y": dy, "pitch_keys": keys, "game_pk": games})
    np.testing.assert_allclose(dev100["ensemble"], dev100["seed_predictions"].mean(axis=0), atol=1e-7, rtol=1e-7)
    for name, p in (("context_blend_log_loss", old_primary), ("context_blend_brier_multiclass", old_secondary)):
        for metric in ("log_loss", "brier_multiclass"):
            np.testing.assert_allclose(classification_metrics(dy, p)[metric], old_results["blends"][name]["metrics"][metric], atol=1e-12)
    blended = apply_selected_blends(fitted, dev100["ensemble"], fixed_baseline_dev)
    predictions = {name.replace("strong_blend_", "context100_blend_"): p for name, p in blended.items()}
    predictions.update({"ensemble100": dev100["ensemble"], "fixed_context_baseline": fixed_baseline_dev,
                        "context25_primary": old_primary, "context25_secondary": old_secondary})
    if (impossible & (dy == 9)).any():
        raise ValueError("Common legality mask contradicts an observed DP")
    legal = {name: condition_on_legality(p, impossible) for name, p in predictions.items()}
    result = {"scope": config["scope"], "n": len(dy), "games": len(np.unique(games)), "selection": selection,
              "metrics": {name: classification_metrics(dy, p) for name, p in predictions.items()}, "paired": {},
              "posthoc_legal": {"scope": "Identical fixed mechanical DP support constraint; not weight fitting or model selection",
                                "metrics": {name: classification_metrics(dy, p) for name, p in legal.items()}, "paired": {}}}
    refs = ("ensemble100", "fixed_context_baseline", "context25_primary", "context25_secondary")
    for name in predictions:
        result["paired"][name] = {"minus_"+ref: paired_fixed_predictors(dy, predictions[name], predictions[ref], games)
                                  for ref in refs if ref != name}
        result["posthoc_legal"]["paired"][name] = {"minus_"+ref: paired_fixed_predictors(dy, legal[name], legal[ref], games)
                                                   for ref in refs if ref != name}
    np.savez_compressed(output/"predictions.npz", y=dy, pitch_keys=keys, game_pk=games, impossible_dp=impossible,
                        **predictions, **{name+"__legal": p for name, p in legal.items()})
    np.savez_compressed(output/"calibration_predictions.npz", y=cy, pitch_keys=cal_keys, game_pk=cal_games,
                        ensemble100=cal100["ensemble"], fixed_context_baseline=fixed_baseline_cal)
    result["predictions_sha256"] = hash_file(output/"predictions.npz")
    dump(output/"results.json", result)
    if any(hash_file(PROJECT/rel) != digest for rel, digest in sources.items()) or any(hash_file(Path(path)) != digest for path, digest in reference_hashes.items()):
        raise RuntimeError("Frozen source or input artifact changed during CPU reblend")
    dump(output/"runtime.json", {"finished_at_utc": datetime.now(timezone.utc).isoformat(), "no_neural_inference": True,
                                "baseline_and_temperature_unchanged": True, "partition_checks_passed": True})
    print(json.dumps({"output": str(output), "weights": fitted, "metrics": {name: {key: values[key] for key in ("log_loss", "brier_multiclass")}
                       for name, values in result["metrics"].items()}}, indent=2))


if __name__ == "__main__":
    main()
