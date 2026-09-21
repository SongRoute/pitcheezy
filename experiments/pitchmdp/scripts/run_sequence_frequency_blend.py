"""CPU-only blending of saved five-model ensemble and the CAL-chosen type baseline."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import shutil
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import numpy as np
import pandas as pd

from pitchmdp.data import KEY, hash_file
from pitchmdp.model import CountBaseline, outcome_labels
from pitchmdp.sequence_model import classification_metrics
from run_sequence_ablations import reconstruct_samples, rows_hash
from run_sequence_calibration import fit_blend
from run_sequence_frequency_baselines import HierarchicalFrequencyBaseline, PITCHER_KEYS, temperature_predictions
from run_sequence_pilot import dump
from audit_sequence_robustness import scores, independent_bootstrap
from diagnose_sequence_legality import condition_on_legality

OBJECTIVES = ("log_loss", "brier_multiclass")


def restore_frequency_model(payload):
    """Restore portable saved tables without invoking fit or reading labels."""
    model = HierarchicalFrequencyBaseline(payload["report"]["include_pitcher"])
    model.parent = CountBaseline()
    model.parent.global_p = np.asarray(payload["global_probability"], dtype=float)
    model.parent.table = payload["count_hand_table"]
    model.type_table = payload["type_table"]
    if model.include_pitcher:
        model.pitcher_table = payload["pitcher_table"]
    model.report = payload["report"]
    return model


def fit_selected_blends(calibration_y, ensemble_calibration, baseline_calibration):
    """Only CAL arrays are accepted; DEV predictions cannot select weights."""
    return {objective: fit_blend(calibration_y, ensemble_calibration, baseline_calibration, objective)
            for objective in OBJECTIVES}


def apply_selected_blends(fitted, ensemble, baseline):
    ensemble, baseline = np.asarray(ensemble, float), np.asarray(baseline, float)
    if ensemble.shape != baseline.shape or ensemble.ndim != 2:
        raise ValueError("Aligned ensemble and baseline probability matrices required")
    for values in (ensemble, baseline):
        if not np.isfinite(values).all() or (values < 0).any() or not np.allclose(values.sum(1), 1., atol=1e-5):
            raise ValueError("Invalid prediction mass")
    result = {}
    for objective in OBJECTIVES:
        weight = fitted[objective]["model_weight"]
        if not 0 <= weight <= 1 or not np.isclose(fitted[objective]["baseline_weight"], 1-weight):
            raise ValueError("Invalid frozen convex weight")
        result["strong_blend_"+objective] = weight*ensemble+(1-weight)*baseline
    return result


def archive_arrays(path, expected):
    with np.load(path, allow_pickle=False) as saved:
        for key, values in expected.items():
            np.testing.assert_array_equal(saved[key], values, err_msg=f"Archive row identity mismatch: {key}")
        return {key: saved[key].copy() for key in saved.files}


def paired_fixed_predictors(labels, model, reference, games):
    """Game-only intervals: fixed ensemble, baseline choice and CAL blend weights."""
    left, right = scores(labels, model), scores(labels, reference)
    bootstrap = independent_bootstrap({key: value[None] for key, value in left.items()},
                                      {key: value[None] for key, value in right.items()}, games, [0])
    return {"games": int(len(np.unique(games))), "replicates": 2000,
            "scope": "Paired whole-game sampling conditional on fixed fitted ensemble/baseline and CAL-selected weights; excludes seed-training, baseline-selection and CAL-weight-fitting uncertainty",
            "direction": "model minus reference; negative favors model",
            "metrics": {key: {field: value[field] for field in ("mean_model", "mean_reference", "model_minus_reference", "game_only_bootstrap95")}
                        for key, value in bootstrap.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True, help="Completed root ensemble/count-blend run")
    parser.add_argument("--frequency", type=Path, required=True, help="Completed four-frequency-baseline run")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    local = json.loads((PROJECT/"configs/local.json").read_text())
    root, volume = Path(local["artifact_root"]).resolve(), Path("/Volumes/T7 Shield")
    if not volume.is_mount() or not root.is_relative_to(volume.resolve()) or Path(sys.executable).absolute() != Path(local["python"]).absolute():
        raise SystemExit("Mounted configured SSD and existing Python required")
    calibration, frequency = args.calibration.resolve(), args.frequency.resolve()
    if not all(path.is_relative_to(root) for path in (calibration, frequency)):
        raise SystemExit("Input runs must be inside configured SSD")
    for path in (calibration, frequency):
        if not (path/"runtime.json").exists():
            raise SystemExit("Wait for completed CPU/archive inputs: "+str(path))
    ensemble_config = json.loads((calibration/"config.json").read_text())
    frequency_config = json.loads((frequency/"config.json").read_text())
    base = Path(ensemble_config["base_run"])
    if str(base) != frequency_config["base_run"] or ensemble_config["seeds"] != [42, 43, 44, 45, 46]:
        raise ValueError("Ensemble and frequency references differ")
    original_selection = json.loads((frequency/"selection.json").read_text())
    frequency_results = json.loads((frequency/"frequency_results.json").read_text())
    chosen = original_selection["chosen_baseline"]
    minimum = min(original_selection["baselines"], key=lambda key: original_selection["baselines"][key]["calibrated_log_loss"])
    if chosen != minimum or chosen != frequency_results["chosen_by_calibration"] or len(original_selection["baselines"]) != 4:
        raise ValueError("Original 4000-CAL baseline selection is inconsistent")
    for key, result in frequency_results["baselines"].items():
        if hash_file(frequency/(key+".pkl")) != result["model_sha256"]:
            raise ValueError("Saved frequency model changed: "+key)
    if hash_file(frequency/"heldout_predictions.npz") != frequency_results["predictions_sha256"]:
        raise ValueError("Original frequency predictions changed")
    for origin, config in ((calibration, ensemble_config), (frequency, frequency_config)):
        for rel, digest in config["source_hashes"].items():
            if hash_file(origin/"source"/rel) != digest:
                raise ValueError("Frozen source archive changed: "+rel)
    protocol = PROJECT/"docs/FREQUENCY_BLEND_PROTOCOL.md"
    paths = [Path(__file__).resolve(), PROJECT/"scripts/run_sequence_frequency_baselines.py", PROJECT/"scripts/run_sequence_calibration.py",
             PROJECT/"scripts/run_sequence_ablations.py", PROJECT/"scripts/run_sequence_pilot.py", PROJECT/"scripts/audit_sequence_robustness.py",
             PROJECT/"scripts/diagnose_sequence_legality.py", protocol]
    sources = {str(path.relative_to(PROJECT)): hash_file(path) for path in paths}
    original = json.loads((base/"source_hashes.json").read_text())
    for rel, digest in original.items():
        if hash_file(PROJECT/rel) != digest:
            raise ValueError("Original implementation changed: "+rel)
    sources.update(original)
    references = [calibration/name for name in ("config.json", "results.json", "runtime.json", "calibration_predictions.npz", "predictions.npz")]
    references += [frequency/name for name in ("config.json", "selection.json", "data.json", "frequency_results.json", "heldout_predictions.npz", "runtime.json")]
    references += [frequency/(key+".pkl") for key in original_selection["baselines"]]
    references += [base/name for name in ("config.json", "samples.json", "cohort_manifest.json")]
    reference_hashes = {str(path): hash_file(path) for path in references}
    output = (args.output or calibration/datetime.now(timezone.utc).strftime("frequency-blend-%Y%m%dT%H%M%SZ")).resolve()
    if not output.is_relative_to(root) or any(output == path or path.is_relative_to(output) for path in (root, base, calibration, frequency)):
        raise SystemExit("Use a distinct SSD output subdirectory")
    output.mkdir(parents=True, exist_ok=False)
    config = {"calibration_run": str(calibration), "frequency_run": str(frequency), "base_run": str(base),
              "selected_frequency_baseline": chosen, "baseline_selection_rows": "original 4000 CAL, before this blend experiment",
              "blend_weight_rows": "remaining original12000 CAL; disjoint from4000 temperature/variant-selection rows",
              "baseline_temperature": original_selection["baselines"][chosen]["temperature"],
              "primary_objective": "log_loss", "secondary_objective": "brier_multiclass",
              "source_hashes": sources, "reference_hashes": reference_hashes,
              "scope": "Exploratory previously inspected DEV; no neural inference, no refitting baseline tables, no DEV selection",
              "calibration_limit": "All16000CAL already informed neural early stopping; not independent validation"}
    dump(output/"config.json", config)
    for rel in sources:
        destination = output/"source"/rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT/rel, destination)
    shutil.copyfile(protocol, output/protocol.name)
    print("FREQUENCY_BLEND_DIR="+str(output), flush=True)
    processed = root/"processed/pitches.parquet"
    if hash_file(processed) != frequency_config["processed_sha256"]:
        raise ValueError("Processed data changed since frequency fitting")
    columns = list(dict.fromkeys([*KEY, *PITCHER_KEYS, "game_date", "split", "starter_pitcher", "description", "events",
                                 "supported_pa", "plate_x", "plate_z", "outs_when_up", "bases"]))
    frame = pd.read_parquet(processed, columns=columns)
    if not pd.to_datetime(frame.game_date).dt.year.isin([2023, 2024, 2025]).all() or frame.duplicated(KEY).any():
        raise ValueError("Forbidden dates or duplicate keys")
    cfg = json.loads((base/"config.json").read_text())
    cohort = json.loads((base/"cohort_manifest.json").read_text())
    train, cal, dev = reconstruct_samples(frame, cfg, cohort)
    hashes = {name: rows_hash(part) for name, part in (("train", train), ("calibration", cal), ("dev", dev))}
    if hashes != json.loads((base/"samples.json").read_text())["rows_hash"]:
        raise ValueError("Original ordered sample identities changed")
    temperature_rows = cal.sample(min(len(cal), cfg["delivery_calibration_rows"]), random_state=42).sort_index()
    blend_cal = cal.loc[~cal.index.isin(temperature_rows.index)]
    if len(blend_cal) != 12000 or len(temperature_rows) != 4000:
        raise ValueError("Expected the frozen4000/12000 calibration partition")
    frequency_data = json.loads((frequency/"data.json").read_text())
    root_results = json.loads((calibration/"results.json").read_text())
    if rows_hash(temperature_rows) != frequency_data["calibration_rows_hash"] or rows_hash(blend_cal) != root_results["calibration_row_hash"]:
        raise ValueError("Reserved calibration row identities differ")
    cy, dy = outcome_labels(blend_cal), outcome_labels(dev)
    cal_archive = archive_arrays(calibration/"calibration_predictions.npz", {"y": cy, "pitch_keys": blend_cal[KEY].to_numpy(), "game_pk": blend_cal.game_pk.to_numpy()})
    np.testing.assert_allclose(cal_archive["ensemble"], cal_archive["seed_predictions"].mean(axis=0), atol=1e-7, rtol=1e-7)
    if cal_archive["seed_predictions"].shape[0] != 5:
        raise ValueError("All five fixed ensemble seeds required")
    with (frequency/(chosen+".pkl")).open("rb") as stream:
        baseline = restore_frequency_model(pickle.load(stream))
    baseline_cal = temperature_predictions(baseline.predict(blend_cal), config["baseline_temperature"])
    fitted = fit_selected_blends(cy, cal_archive["ensemble"], baseline_cal)
    selection = {"baseline_chosen_on_original4000": chosen, "original4000_selection": original_selection,
                 "blend_calibration_rows": len(blend_cal), "blend_calibration_rows_hash": rows_hash(blend_cal),
                 "temperature_rows_hash": rows_hash(temperature_rows), "fitted_blends": fitted,
                 "primary": "log_loss", "secondary": "brier_multiclass", "dev_selection": False}
    dump(output/"selection.json", selection)
    # DEV predictions are opened only after both CAL-only weights are persisted.
    expected_dev = {"y": dy, "pitch_keys": dev[KEY].to_numpy(), "game_pk": dev.game_pk.to_numpy()}
    dev_archive = archive_arrays(calibration/"predictions.npz", expected_dev)
    frequency_archive = archive_arrays(frequency/"heldout_predictions.npz", expected_dev)
    np.testing.assert_allclose(dev_archive["ensemble"], dev_archive["seed_predictions"].mean(axis=0), atol=1e-7, rtol=1e-7)
    baseline_dev = temperature_predictions(baseline.predict(dev), config["baseline_temperature"])
    np.testing.assert_array_equal(baseline_dev, frequency_archive[chosen+"__tempered"])
    predictions = apply_selected_blends(fitted, dev_archive["ensemble"], baseline_dev)
    predictions.update({"ensemble": dev_archive["ensemble"], "selected_frequency_baseline": baseline_dev,
                        "old_count_baseline": dev_archive["count_hand"],
                        "count_blend_log_loss": dev_archive["blend_log_loss"],
                        "count_blend_brier_multiclass": dev_archive["blend_brier_multiclass"]})
    result = {"scope": config["scope"], "n": len(dev), "games": int(dev.game_pk.nunique()), "selection": selection,
              "metrics": {name: classification_metrics(dy, p) for name, p in predictions.items()}, "paired": {},
              "interval_limit": "Whole-game resampling conditional on fitted ensemble, baseline and CAL weights; selection/training uncertainty omitted"}
    for name in predictions:
        result["paired"][name] = {}
        for reference in ("ensemble", "selected_frequency_baseline", "count_blend_log_loss", "count_blend_brier_multiclass"):
            if reference != name:
                result["paired"][name]["minus_"+reference] = paired_fixed_predictors(dy, predictions[name], predictions[reference], dev.game_pk.to_numpy())
    impossible = dev.outs_when_up.eq(2).to_numpy() | dev.bases.eq(0).to_numpy()
    if (impossible & (dy == 9)).any():
        raise ValueError("Observed DP contradicts the common legality diagnostic")
    legal = {name: condition_on_legality(p, impossible) for name, p in predictions.items()}
    result["posthoc_legal"] = {"role": "Common mechanical DP support conditioning; never used to fit weights or select models",
                               "constrained_rows": int(impossible.sum()),
                               "metrics": {name: classification_metrics(dy, p) for name, p in legal.items()}, "paired": {}}
    for name in legal:
        result["posthoc_legal"]["paired"][name] = {"minus_"+reference: paired_fixed_predictors(dy, legal[name], legal[reference], dev.game_pk.to_numpy())
            for reference in ("ensemble", "selected_frequency_baseline", "count_blend_log_loss", "count_blend_brier_multiclass") if reference != name}
    result["all_frequency_baselines_descriptive"] = {
        "role": "Retain all four fixed frequency baselines; no extra selection, no claim that the CAL-LL choice is universally best",
        "baselines": {}}
    for baseline_name, baseline_result in frequency_results["baselines"].items():
        entry = {"training_rows": baseline_result["fit"]["training_rows"],
                 "tempered": baseline_result["metrics"]["tempered"], "tempered_legal": baseline_result["metrics"]["tempered_legal"],
                 "strong_blend_comparisons": {}}
        for objective in OBJECTIVES:
            name = "strong_blend_"+objective
            entry["strong_blend_comparisons"][name] = {
                "original": paired_fixed_predictors(dy, predictions[name], frequency_archive[baseline_name+"__tempered"], dev.game_pk.to_numpy()),
                "posthoc_legal": paired_fixed_predictors(dy, legal[name], frequency_archive[baseline_name+"__tempered_legal"], dev.game_pk.to_numpy())}
        result["all_frequency_baselines_descriptive"]["baselines"][baseline_name] = entry
    np.savez_compressed(output/"predictions.npz", **expected_dev, **predictions, impossible_dp=impossible,
                        **{name+"__legal": p for name, p in legal.items()})
    np.savez_compressed(output/"calibration_predictions.npz", y=cy, pitch_keys=blend_cal[KEY].to_numpy(), game_pk=blend_cal.game_pk.to_numpy(),
                        ensemble=cal_archive["ensemble"], selected_frequency_baseline=baseline_cal)
    result["predictions_sha256"] = hash_file(output/"predictions.npz")
    dump(output/"results.json", result)
    if any(hash_file(PROJECT/rel) != digest for rel, digest in sources.items()) or any(hash_file(Path(path)) != digest for path, digest in reference_hashes.items()):
        raise RuntimeError("Frozen sources or input artifacts changed during CPU blending")
    dump(output/"runtime.json", {"finished_at_utc": datetime.now(timezone.utc).isoformat(), "neural_training": False, "neural_inference": False})
    print(json.dumps({"output": str(output), "weights": fitted, "metrics": {name: {key: metric[key] for key in ("log_loss", "brier_multiclass")}
                       for name, metric in result["metrics"].items()}}, indent=2))


if __name__ == "__main__":
    main()
