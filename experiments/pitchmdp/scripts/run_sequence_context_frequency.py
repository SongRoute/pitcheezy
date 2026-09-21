"""CPU-only pre-pitch state frequency baselines and CAL-selected ensemble blends."""
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
from pitchmdp.model import eligible, outcome_labels
from pitchmdp.sequence_model import classification_metrics
from run_sequence_frequency_baselines import TYPE_KEYS, HierarchicalFrequencyBaseline, fit_temperature, temperature_predictions
from run_sequence_frequency_blend import (restore_frequency_model, fit_selected_blends, apply_selected_blends,
                                          paired_fixed_predictors, archive_arrays)
from run_sequence_ablations import reconstruct_samples, rows_hash
from run_sequence_pilot import dump
from diagnose_sequence_legality import condition_on_legality

CONTEXT_KEYS = [*TYPE_KEYS, "outs_when_up", "bases"]
PITCHER_CONTEXT_KEYS = ["pitcher", *CONTEXT_KEYS]
CHOICES = ("old_full_type", "league_context", "pitcher_context")


class ContextFrequencyBaseline:
    """Two fixed strength-100 state levels above an immutable type parent."""
    def __init__(self, parent_type_model, include_pitcher=False):
        if parent_type_model.include_pitcher:
            raise ValueError("Context hierarchy requires the league/type parent")
        self.parent_type_model, self.include_pitcher = parent_type_model, include_pitcher

    @staticmethod
    def _validate_state(frame):
        if not frame.outs_when_up.isin([0, 1, 2]).all() or not frame.bases.isin(range(8)).all():
            raise ValueError("Expected legal pre-pitch outs0..2 and base occupancy0..7")

    def fit(self, train):
        if not len(train) or not train.split.eq("train").all() or not pd.to_datetime(train.game_date).between("2023-05-15", "2025-04-30").all():
            raise ValueError("Context-frequency fitting requires only approved TRAIN rows")
        self._validate_state(train)
        labels = outcome_labels(train)
        if (labels < 0).any():
            raise ValueError("Valid outcome labels required")
        work = train[PITCHER_CONTEXT_KEYS].copy()
        work["y"] = labels
        counts = HierarchicalFrequencyBaseline._counts(work, CONTEXT_KEYS)
        parents = self.parent_type_model.predict(counts.index.to_frame(index=False))
        values = (counts.to_numpy()+100*parents)/(counts.sum(axis=1).to_numpy()[:, None]+100)
        self.context_table = pd.DataFrame(values, index=counts.index, columns=range(10))
        if self.include_pitcher:
            counts = HierarchicalFrequencyBaseline._counts(work, PITCHER_CONTEXT_KEYS)
            parents = self._predict_context(counts.index.to_frame(index=False))[0]
            values = (counts.to_numpy()+100*parents)/(counts.sum(axis=1).to_numpy()[:, None]+100)
            self.pitcher_context_table = pd.DataFrame(values, index=counts.index, columns=range(10))
        self.report = {"training_rows": len(train), "include_pitcher": self.include_pitcher,
                       "context_strength": 100, "pitcher_context_strength": 100,
                       "context_keys": CONTEXT_KEYS, "pitcher_context_keys": PITCHER_CONTEXT_KEYS,
                       "context_groups": len(self.context_table),
                       "pitcher_context_groups": len(self.pitcher_context_table) if self.include_pitcher else 0,
                       "parent_report": self.parent_type_model.report, "batter_id_input": False,
                       "fit_date_max": str(pd.to_datetime(train.game_date).max().date())}
        return self

    def _predict_context(self, frame):
        self._validate_state(frame)
        parent, origin = self.parent_type_model.predict_with_origin(frame)
        values = HierarchicalFrequencyBaseline._lookup(self.context_table, frame, CONTEXT_KEYS)
        seen = np.isfinite(values).all(axis=1)
        origin[seen] = 4
        return np.where(seen[:, None], values, parent), origin

    def predict_with_origin(self, frame):
        values, origin = self._predict_context(frame)
        if self.include_pitcher:
            local = HierarchicalFrequencyBaseline._lookup(self.pitcher_context_table, frame, PITCHER_CONTEXT_KEYS)
            seen = np.isfinite(local).all(axis=1)
            values = np.where(seen[:, None], local, values)
            origin[seen] = 5
        return values, origin

    def predict(self, frame):
        return self.predict_with_origin(frame)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frequency", type=Path, required=True)
    parser.add_argument("--blend", type=Path, required=True, help="Completed original type-baseline/ensemble blend")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    local = json.loads((PROJECT/"configs/local.json").read_text())
    root, volume = Path(local["artifact_root"]).resolve(), Path("/Volumes/T7 Shield")
    if not volume.is_mount() or not root.is_relative_to(volume.resolve()) or Path(sys.executable).absolute() != Path(local["python"]).absolute():
        raise SystemExit("Use configured mounted SSD and existing Python")
    frequency, old_blend = args.frequency.resolve(), args.blend.resolve()
    for path in (frequency, old_blend):
        if not path.is_relative_to(root) or not (path/"runtime.json").exists():
            raise SystemExit("Completed original CPU reference runs required")
    frequency_config = json.loads((frequency/"config.json").read_text())
    old_config = json.loads((old_blend/"config.json").read_text())
    calibration, base = Path(old_config["calibration_run"]), Path(old_config["base_run"])
    if str(frequency) != old_config["frequency_run"] or frequency_config["base_run"] != str(base):
        raise ValueError("Reference runs differ")
    original_selection = json.loads((frequency/"selection.json").read_text())
    frequency_results = json.loads((frequency/"frequency_results.json").read_text())
    if original_selection["chosen_baseline"] != "full_train__type":
        raise ValueError("Protocol fixes the original full-TRAIN league/type parent")
    table_path = frequency/"full_train__type.pkl"
    if hash_file(table_path) != frequency_results["baselines"]["full_train__type"]["model_sha256"]:
        raise ValueError("Frozen type-parent tables changed")
    processed = root/"processed/pitches.parquet"
    if hash_file(processed) != frequency_config["processed_sha256"]:
        raise ValueError("Processed data changed")
    protocol = PROJECT/"docs/CONTEXT_FREQUENCY_PROTOCOL.md"
    extra = [Path(__file__).resolve(), protocol, *[PROJECT/"scripts"/name for name in
             ("run_sequence_frequency_baselines.py", "run_sequence_frequency_blend.py", "run_sequence_calibration.py",
              "run_sequence_ablations.py", "run_sequence_pilot.py", "audit_sequence_robustness.py", "diagnose_sequence_legality.py")]]
    original_sources = json.loads((base/"source_hashes.json").read_text())
    for rel, digest in original_sources.items():
        if hash_file(PROJECT/rel) != digest:
            raise ValueError("Original implementation changed: "+rel)
    sources = {**original_sources, **{str(path.relative_to(PROJECT)): hash_file(path) for path in extra}}
    references = [frequency/name for name in ("config.json", "data.json", "selection.json", "frequency_results.json", "full_train__type.pkl", "heldout_predictions.npz")]
    references += [old_blend/name for name in ("config.json", "results.json", "selection.json", "predictions.npz", "runtime.json")]
    references += [calibration/name for name in ("config.json", "results.json", "calibration_predictions.npz", "predictions.npz", "runtime.json")]
    references += [base/name for name in ("config.json", "samples.json", "cohort_manifest.json")]
    ref_hashes = {str(path): hash_file(path) for path in references}
    output = (args.output or old_blend/datetime.now(timezone.utc).strftime("context-frequency-%Y%m%dT%H%M%SZ")).resolve()
    if not output.is_relative_to(root) or any(output == path or path.is_relative_to(output) for path in (frequency, old_blend, base)):
        raise SystemExit("Use a distinct SSD output directory")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"frequency_run": str(frequency), "old_type_blend_run": str(old_blend), "calibration_run": str(calibration),
                "parent": "frozen full_train__type tables", "baseline_candidates": list(CHOICES),
                "context_keys": CONTEXT_KEYS, "pitcher_context_keys": PITCHER_CONTEXT_KEYS,
                "context_strength": 100, "pitcher_strength": 100,
                "baseline_selection": "minimum tempered original4000 CAL log loss; old baseline first in exact ties",
                "blend_fitting": "remaining12000 CAL; primary log loss, secondary Brier; both retained",
                "source_hashes": sources, "reference_hashes": ref_hashes, "processed_sha256": frequency_config["processed_sha256"],
                "scope": "Exploratory game-state frequency fairness diagnostic after prior DEV decomposition; no neural inference/training or DEV selection",
                "common_legality": "separate post-hoc DP mass conditioning on outs2 or empty bases; never used in CAL selection or fitting"}
    dump(output/"config.json", manifest)
    for rel in sources:
        destination = output/"source"/rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT/rel, destination)
    shutil.copyfile(protocol, output/protocol.name)
    print("CONTEXT_FREQUENCY_DIR="+str(output), flush=True)
    columns = list(dict.fromkeys([*KEY, *PITCHER_CONTEXT_KEYS, "game_date", "split", "starter_pitcher", "description", "events", "supported_pa", "plate_x", "plate_z"]))
    frame = pd.read_parquet(processed, columns=columns)
    if frame.duplicated(KEY).any() or frame[KEY].isna().any().any() or not pd.to_datetime(frame.game_date).dt.year.isin([2023, 2024, 2025]).all():
        raise ValueError("Invalid or forbidden processed rows")
    cfg = json.loads((base/"config.json").read_text())
    cohort = json.loads((base/"cohort_manifest.json").read_text())
    train, cal, dev = reconstruct_samples(frame, cfg, cohort)
    hashes = {name: rows_hash(part) for name, part in (("train", train), ("calibration", cal), ("dev", dev))}
    if hashes != json.loads((base/"samples.json").read_text())["rows_hash"]:
        raise ValueError("Ordered original samples differ")
    train_all = frame[frame.split.eq("train") & eligible(frame)]
    frequency_data = json.loads((frequency/"data.json").read_text())
    if rows_hash(train_all) != frequency_data["full_train_rows_hash"] or len(train_all) != frequency_data["full_training_rows"]:
        raise ValueError("Full eligible TRAIN pool changed")
    temperature_rows = cal.sample(min(len(cal), cfg["delivery_calibration_rows"]), random_state=42).sort_index()
    blend_cal = cal.loc[~cal.index.isin(temperature_rows.index)]
    if len(temperature_rows) != 4000 or len(blend_cal) != 12000 or rows_hash(temperature_rows) != frequency_data["calibration_rows_hash"]:
        raise ValueError("Frozen disjoint4000/12000 CAL partition changed")
    cy = outcome_labels(temperature_rows)
    blend_y, dy = outcome_labels(blend_cal), outcome_labels(dev)
    cal_predictions = archive_arrays(calibration/"calibration_predictions.npz", {"y": blend_y, "pitch_keys": blend_cal[KEY].to_numpy(), "game_pk": blend_cal.game_pk.to_numpy()})
    if cal_predictions["seed_predictions"].shape[0] != 5:
        raise ValueError("Fixed five-seed ensemble required")
    np.testing.assert_allclose(cal_predictions["ensemble"], cal_predictions["seed_predictions"].mean(axis=0), atol=1e-7, rtol=1e-7)
    with table_path.open("rb") as stream:
        parent_payload = pickle.load(stream)
    parent = restore_frequency_model(parent_payload)
    if parent.report["training_rows"] != len(train_all):
        raise ValueError("Type parent was fitted on a different TRAIN pool")
    models = {"old_full_type": parent}
    selected = {"old_full_type": original_selection["baselines"]["full_train__type"]}
    old_cal = temperature_predictions(parent.predict(temperature_rows), selected["old_full_type"]["temperature"])
    np.testing.assert_allclose(classification_metrics(cy, old_cal)["log_loss"], selected["old_full_type"]["calibrated_log_loss"], atol=1e-12)
    dump(output/"data.json", {"ordered_rows_hash": hashes, "full_train_rows_hash": rows_hash(train_all),
                             "train_rows": len(train_all), "temperature_rows_hash": rows_hash(temperature_rows),
                             "blend_rows_hash": rows_hash(blend_cal), "dev_rows": len(dev), "dev_games": int(dev.game_pk.nunique())})
    for name, pitcher in (("league_context", False), ("pitcher_context", True)):
        model = ContextFrequencyBaseline(parent, pitcher).fit(train_all)
        models[name] = model
        selected[name] = fit_temperature(model.predict(temperature_rows), cy)
        with (output/(name+".pkl")).open("wb") as stream:
            pickle.dump({"report": model.report, "type_parent": parent_payload, "context_table": model.context_table,
                         "pitcher_context_table": model.pitcher_context_table if pitcher else None}, stream)
        print("CAL4000", name, selected[name], flush=True)
    chosen = min(CHOICES, key=lambda name: selected[name]["calibrated_log_loss"])
    baseline_selection = {"candidates": selected, "chosen": chosen, "criterion": manifest["baseline_selection"]}
    dump(output/"baseline_selection.json", baseline_selection)
    temperature_predictions_archive = {}
    for name, model in models.items():
        p = model.predict(temperature_rows)
        temperature_predictions_archive[name+"__raw"] = p
        temperature_predictions_archive[name+"__tempered"] = temperature_predictions(p, selected[name]["temperature"])
    np.savez_compressed(output/"temperature_predictions.npz", y=cy, pitch_keys=temperature_rows[KEY].to_numpy(),
                        game_pk=temperature_rows.game_pk.to_numpy(), **temperature_predictions_archive)
    baseline_cal = temperature_predictions(models[chosen].predict(blend_cal), selected[chosen]["temperature"])
    weights = fit_selected_blends(blend_y, cal_predictions["ensemble"], baseline_cal)
    selection = {"baseline": baseline_selection, "blend_weights": weights, "primary": "log_loss", "secondary": "brier_multiclass"}
    dump(output/"selection.json", selection)
    # No DEV predictor probabilities are opened or calculated before both CAL decisions.
    expected = {"y": dy, "pitch_keys": dev[KEY].to_numpy(), "game_pk": dev.game_pk.to_numpy()}
    ensemble_archive = archive_arrays(calibration/"predictions.npz", expected)
    old_archive = archive_arrays(old_blend/"predictions.npz", expected)
    frequency_archive = archive_arrays(frequency/"heldout_predictions.npz", expected)
    np.testing.assert_allclose(ensemble_archive["ensemble"], ensemble_archive["seed_predictions"].mean(axis=0), atol=1e-7, rtol=1e-7)
    result = {"scope": manifest["scope"], "n": len(dev), "games": int(dev.game_pk.nunique()), "selection": selection,
              "baselines": {}, "blends": {}, "references": {}, "paired": {}, "posthoc_legal": {"metrics": {}, "paired": {}}}
    predictions = {"ensemble": ensemble_archive["ensemble"], "old_type_blend_log_loss": old_archive["strong_blend_log_loss"],
                   "old_type_blend_brier": old_archive["strong_blend_brier_multiclass"]}
    impossible = dev.outs_when_up.eq(2).to_numpy() | dev.bases.eq(0).to_numpy()
    if (impossible & (dy == 9)).any():
        raise ValueError("Observed DP contradicts the common post-hoc legality rule")
    for name, model in models.items():
        raw, origin = model.predict_with_origin(dev)
        tempered = temperature_predictions(raw, selected[name]["temperature"])
        predictions[name+"__raw"], predictions[name+"__tempered"] = raw, tempered
        if name == "old_full_type":
            np.testing.assert_array_equal(tempered, frequency_archive["full_train__type__tempered"])
        result["baselines"][name] = {"training_rows": len(train_all), "calibration": selected[name], "fit": model.report,
            "model_path": str(table_path if name == "old_full_type" else output/(name+".pkl")),
            "model_sha256": hash_file(table_path if name == "old_full_type" else output/(name+".pkl")),
            "metrics": {"raw": classification_metrics(dy, raw), "tempered": classification_metrics(dy, tempered),
                        "raw_legal": classification_metrics(dy, condition_on_legality(raw, impossible)),
                        "tempered_legal": classification_metrics(dy, condition_on_legality(tempered, impossible))},
            "origin_counts": {str(int(level)): int((origin == level).sum()) for level in np.unique(origin)}}
    blends = apply_selected_blends(weights, predictions["ensemble"], predictions[chosen+"__tempered"])
    for name, p in blends.items():
        key = name.replace("strong_blend_", "context_blend_")
        predictions[key] = p
        result["blends"][key] = {"calibration": weights[name.removeprefix("strong_blend_")], "metrics": classification_metrics(dy, p)}
    for name in ("ensemble", "old_type_blend_log_loss", "old_type_blend_brier"):
        result["references"][name] = classification_metrics(dy, predictions[name])
    legal = {name: condition_on_legality(p, impossible) for name, p in predictions.items()}
    result["posthoc_legal"]["metrics"] = {name: classification_metrics(dy, p) for name, p in legal.items()}
    result["posthoc_legal"]["scope"] = "Common mechanical conditioning, not calibration selection or learned skill"
    comparisons = [name+"__"+regime for name in CHOICES for regime in ("raw", "tempered")]+list(result["blends"])
    references_names = ("old_type_blend_log_loss", "old_full_type__tempered", "ensemble")
    for name in comparisons:
        result["paired"][name] = {"minus_"+reference: paired_fixed_predictors(dy, predictions[name], predictions[reference], dev.game_pk.to_numpy())
                                   for reference in references_names if reference != name}
        result["posthoc_legal"]["paired"][name] = {"minus_"+reference: paired_fixed_predictors(dy, legal[name], legal[reference], dev.game_pk.to_numpy())
                                                    for reference in references_names if reference != name}
    np.savez_compressed(output/"predictions.npz", **expected, **predictions, impossible_dp=impossible,
                        **{name+"__legal": p for name, p in legal.items()})
    np.savez_compressed(output/"calibration_predictions.npz", y=blend_y, pitch_keys=blend_cal[KEY].to_numpy(),
                        game_pk=blend_cal.game_pk.to_numpy(), ensemble=cal_predictions["ensemble"], selected_baseline=baseline_cal)
    result["predictions_sha256"] = hash_file(output/"predictions.npz")
    dump(output/"results.json", result)
    if any(hash_file(PROJECT/rel) != digest for rel, digest in sources.items()) or any(hash_file(Path(path)) != digest for path, digest in ref_hashes.items()):
        raise RuntimeError("Frozen sources or references changed during CPU run")
    dump(output/"runtime.json", {"finished_at_utc": datetime.now(timezone.utc).isoformat(), "neural_training": False, "neural_inference": False})
    print(json.dumps({"output": str(output), "chosen": chosen, "weights": weights,
                      "baselines": {name: {key: entry["metrics"]["tempered"][key] for key in ("log_loss", "brier_multiclass")}
                                    for name, entry in result["baselines"].items()},
                      "blends": {name: {key: entry["metrics"][key] for key in ("log_loss", "brier_multiclass")}
                                 for name, entry in result["blends"].items()}}, indent=2))


if __name__ == "__main__":
    main()
