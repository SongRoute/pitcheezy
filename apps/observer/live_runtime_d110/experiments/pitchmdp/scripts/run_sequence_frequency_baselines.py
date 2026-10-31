"""CPU-only, known-pitch-type frequency baselines with frozen TRAIN/CAL splits."""
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
from scipy.optimize import minimize_scalar
from scipy.special import softmax

from pitchmdp.data import KEY, RAW_ALLOWLIST, hash_file
from pitchmdp.model import CountBaseline, OUTCOMES, eligible, outcome_labels
from pitchmdp.sequence_model import classification_metrics
from run_sequence_ablations import reconstruct_samples, rows_hash
from run_sequence_pilot import dump
from diagnose_sequence_legality import condition_on_legality
from audit_sequence_robustness import independent_bootstrap, scores

COUNT_KEYS = ["balls", "strikes", "stand", "p_throws"]
TYPE_KEYS = ["pitch_type", *COUNT_KEYS]
PITCHER_KEYS = ["pitcher", *TYPE_KEYS]


class HierarchicalFrequencyBaseline:
    """Fixed pseudo-count strengths: global→count50→type50→pitcher100."""
    def __init__(self, include_pitcher=False):
        self.include_pitcher = include_pitcher

    @staticmethod
    def _counts(frame, keys):
        return frame.groupby([*keys, "y"], sort=True, dropna=False).size().unstack("y", fill_value=0).reindex(columns=range(10), fill_value=0)

    @staticmethod
    def _lookup(table, frame, keys):
        return table.reindex(pd.MultiIndex.from_frame(frame[keys])).to_numpy()

    def fit(self, train):
        if not len(train) or not train.split.eq("train").all() or not pd.to_datetime(train.game_date).between("2023-05-15", "2025-04-30").all():
            raise ValueError("Frequency fitting requires only approved TRAIN rows")
        labels = outcome_labels(train)
        if (labels < 0).any():
            raise ValueError("Frequency fitting requires valid outcome labels")
        self.parent = CountBaseline().fit(train)
        work = train[PITCHER_KEYS].copy()
        work["y"] = labels
        counts = self._counts(work, TYPE_KEYS)
        parents = self.parent.predict(counts.index.to_frame(index=False))
        values = (counts.to_numpy()+50*parents)/(counts.sum(axis=1).to_numpy()[:, None]+50)
        self.type_table = pd.DataFrame(values, index=counts.index, columns=range(10))
        if self.include_pitcher:
            counts = self._counts(work, PITCHER_KEYS)
            parents = self._predict_type(counts.index.to_frame(index=False))[0]
            values = (counts.to_numpy()+100*parents)/(counts.sum(axis=1).to_numpy()[:, None]+100)
            self.pitcher_table = pd.DataFrame(values, index=counts.index, columns=range(10))
        self.report = {"training_rows": len(train), "include_pitcher": self.include_pitcher,
                       "count_hand_groups": len(self.parent.table), "type_groups": len(self.type_table),
                       "pitcher_type_groups": len(self.pitcher_table) if self.include_pitcher else 0,
                       "type_strength": 50, "pitcher_strength": 100, "count_strength": 50,
                       "global_prior": "add one pseudo-observation per ten-class outcome",
                       "fit_date_max": str(pd.to_datetime(train.game_date).max().date()), "batter_id_input": False}
        return self

    def _predict_type(self, frame):
        base = self.parent.predict(frame)
        values = self._lookup(self.type_table, frame, TYPE_KEYS)
        available = np.isfinite(values).all(axis=1)
        origin = np.array([int(key in self.parent.table) for key in frame[COUNT_KEYS].itertuples(index=False, name=None)], dtype=np.int8)
        origin[available] = 2
        return np.where(available[:, None], values, base), origin

    def predict_with_origin(self, frame):
        values, origin = self._predict_type(frame)
        if self.include_pitcher:
            local = self._lookup(self.pitcher_table, frame, PITCHER_KEYS)
            available = np.isfinite(local).all(axis=1)
            values = np.where(available[:, None], local, values)
            origin[available] = 3
        return values, origin

    def predict(self, frame):
        return self.predict_with_origin(frame)[0]


def temperature_predictions(probabilities, temperature):
    return softmax(np.log(np.clip(probabilities, 1e-12, 1.))/temperature, axis=-1)


def fit_temperature(calibration_probabilities, labels):
    def loss(temperature):
        p = temperature_predictions(calibration_probabilities, temperature)
        return float(-np.log(np.clip(p[np.arange(len(labels)), labels], 1e-12, 1.)).mean())
    fit = minimize_scalar(loss, bounds=(.5, 2.5), method="bounded")
    return {"temperature": float(fit.x), "calibrated_log_loss": float(fit.fun), "raw_log_loss": loss(1.)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    local = json.loads((PROJECT/"configs/local.json").read_text())
    root, volume = Path(local["artifact_root"]).resolve(), Path("/Volumes/T7 Shield")
    if (not volume.is_mount() or not root.is_relative_to(volume.resolve()) or local["raw_allowlist"] != RAW_ALLOWLIST
            or Path(sys.executable).absolute() != Path(local["python"]).absolute()):
        raise SystemExit("Use approved configuration, mounted SSD and existing Python")
    run = args.run.resolve()
    output = (args.output or run/datetime.now(timezone.utc).strftime("frequency-baselines-%Y%m%dT%H%M%SZ")).resolve()
    if not run.is_relative_to(root) or not output.is_relative_to(root) or output == run or run.is_relative_to(output):
        raise SystemExit("Use a distinct output directory on the configured SSD")
    cfg = json.loads((run/"config.json").read_text())
    samples = json.loads((run/"samples.json").read_text())
    cohort = json.loads((run/"cohort_manifest.json").read_text())
    original_sources = json.loads((run/"source_hashes.json").read_text())
    for rel, digest in original_sources.items():
        if hash_file(PROJECT/rel) != digest or hash_file(run/"source"/rel) != digest:
            raise ValueError(f"Frozen original source changed: {rel}")
    quality_path, processed = root/"reports/data_quality.json", root/"processed/pitches.parquet"
    quality = json.loads(quality_path.read_text())
    archived_quality = json.loads((run/"audit_supplemental/data_quality.json").read_text())
    processed_hash = hash_file(processed)
    if processed_hash != quality["processed_sha256"] or processed_hash != archived_quality["processed_sha256"]:
        raise ValueError("Processed data no longer match the original run")
    protocol = PROJECT/"docs/STRONG_BASELINE_PROTOCOL.md"
    extra = [Path(__file__).resolve(), PROJECT/"scripts/run_sequence_ablations.py",
             PROJECT/"scripts/diagnose_sequence_legality.py", PROJECT/"scripts/audit_sequence_robustness.py", protocol]
    sources = {**original_sources, **{str(path.relative_to(PROJECT)): hash_file(path) for path in extra}}
    references = [quality_path, run/"audit_supplemental/data_quality.json", run/"config.json", run/"samples.json",
                  run/"cohort_manifest.json", run/"heldout_predictions.npz", run/"all_count_results.json"]
    reference_hashes = {str(path): hash_file(path) for path in references}
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"base_run": str(run), "sample_seed": 42, "type_strength": 50, "pitcher_strength": 100,
                "fit_samples": ["matched_neural_train", "full_eligible_train"],
                "structures": ["count_hand_type", "count_hand_type_pitcher"], "batter_id_input": False,
                "temperature": {"bounds": [.5, 2.5], "calibration": "original deterministic4000CAL subset", "report_raw_and_tempered": True},
                "selection": "label smallest tempered CAL log loss among four baselines; retain all four; no DEV selection",
                "posthoc_legality": "DP impossible if outs==2 or bases==0; apply same conditioning to all baseline regimes and both references",
                "source_hashes": sources, "reference_hashes": reference_hashes, "processed_sha256": processed_hash,
                "scope": "Exploratory stronger known-type baselines on inspected DEV; no neural training or inference; no causal/policy claim"}
    dump(output/"config.json", manifest)
    for rel in sources:
        destination = output/"source"/rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT/rel, destination)
    shutil.copyfile(protocol, output/protocol.name)
    print("FREQUENCY_DIR="+str(output), flush=True)
    columns = list(dict.fromkeys([*KEY, *PITCHER_KEYS, "game_date", "split", "starter_pitcher", "description", "events",
                                 "supported_pa", "plate_x", "plate_z", "outs_when_up", "bases"]))
    frame = pd.read_parquet(processed, columns=columns)
    if frame[KEY].isna().any().any() or frame.duplicated(KEY).any() or not pd.to_datetime(frame.game_date).dt.year.isin([2023, 2024, 2025]).all():
        raise ValueError("Invalid or forbidden processed rows")
    train, cal, dev = reconstruct_samples(frame, cfg, cohort)
    actual = {name: rows_hash(part) for name, part in (("train", train), ("calibration", cal), ("dev", dev))}
    if actual != samples["rows_hash"]:
        raise ValueError("Ordered TRAIN/CAL/DEV samples differ from original")
    mixture = cal.sample(min(len(cal), cfg["delivery_calibration_rows"]), random_state=42).sort_index()
    train_all = frame[frame.split.eq("train") & eligible(frame)]
    cy, dy = outcome_labels(mixture), outcome_labels(dev)
    keys, games = dev[KEY].to_numpy(), dev.game_pk.to_numpy()
    impossible = dev.outs_when_up.eq(2).to_numpy() | dev.bases.eq(0).to_numpy()
    if (impossible & (dy == 9)).any():
        raise ValueError("Observed DP contradicts the common legality diagnostic")
    dump(output/"data.json", {"ordered_rows_hash": actual, "calibration_rows_hash": rows_hash(mixture),
                             "matched_training_rows": len(train), "full_training_rows": len(train_all),
                             "calibration_rows": len(mixture), "dev_rows": len(dev), "dev_games": int(dev.game_pk.nunique()),
                             "full_train_rows_hash": rows_hash(train_all), "processed_sha256": processed_hash,
                             "legality_constrained_rows": int(impossible.sum())})
    with np.load(run/"heldout_predictions.npz", allow_pickle=False) as saved:
        for key, value in (("y", dy), ("pitch_keys", keys), ("game_pk", games)):
            np.testing.assert_array_equal(saved[key], value)
        reference_p = {"old_count_hand": saved["count_hand"].copy(), "full_transformer42": saved["transformer"].copy()}
    results = {"scope": manifest["scope"], "n": len(dev), "games": len(np.unique(games)), "baselines": {}, "references": {}}
    predictions, fitted, selection = {}, {}, {"criterion": manifest["selection"], "baselines": {}}
    for sample_name, fit_frame in (("matched", train), ("full_train", train_all)):
        for with_pitcher in (False, True):
            name = sample_name+("__type_pitcher" if with_pitcher else "__type")
            model = HierarchicalFrequencyBaseline(with_pitcher).fit(fit_frame)
            temperature = fit_temperature(model.predict(mixture), cy)
            fitted[name] = model
            selection["baselines"][name] = temperature
            with (output/(name+".pkl")).open("wb") as stream:
                pickle.dump({"report": model.report, "global_probability": model.parent.global_p,
                             "count_hand_table": model.parent.table, "type_table": model.type_table,
                             "pitcher_table": model.pitcher_table if with_pitcher else None}, stream)
            print("CAL", name, temperature, flush=True)
    selection["chosen_baseline"] = min(fitted, key=lambda name: selection["baselines"][name]["calibrated_log_loss"])
    dump(output/"selection.json", selection)
    results["chosen_by_calibration"] = selection["chosen_baseline"]
    for reference, p in reference_p.items():
        constrained = condition_on_legality(p, impossible)
        predictions[reference], predictions[reference+"__legal"] = p, constrained
        results["references"][reference] = {"original": classification_metrics(dy, p), "posthoc_legal": classification_metrics(dy, constrained)}
    original_results = json.loads((run/"all_count_results.json").read_text())
    for name, original in (("old_count_hand", original_results["count_hand_baseline"]),
                           ("full_transformer42", original_results["transformer"]["primary_delivery_integrated"])):
        for metric in ("log_loss", "brier_multiclass"):
            if not np.isclose(results["references"][name]["original"][metric], original[metric], atol=1e-10, rtol=1e-10):
                raise ValueError("Original reference metric replay failed")
    for name, model in fitted.items():
        raw, origin = model.predict_with_origin(dev)
        tempered = temperature_predictions(raw, selection["baselines"][name]["temperature"])
        regimes = {"raw": raw, "tempered": tempered, "raw_legal": condition_on_legality(raw, impossible),
                   "tempered_legal": condition_on_legality(tempered, impossible)}
        entry = {"fit": model.report, "calibration": selection["baselines"][name], "metrics": {}, "paired": {},
                 "origin_counts": {label: int((origin == code).sum()) for code, label in
                                   ((0, "global"), (1, "count_hand"), (2, "league_type"), (3, "pitcher_type"))},
                 "model_sha256": hash_file(output/(name+".pkl"))}
        for regime, p in regimes.items():
            entry["metrics"][regime] = classification_metrics(dy, p)
            predictions[name+"__"+regime] = p
            entry["paired"][regime] = {}
            for refname in reference_p:
                q = predictions[refname+("__legal" if regime.endswith("_legal") else "")]
                left, right = scores(dy, p), scores(dy, q)
                entry["paired"][regime]["minus_"+refname] = independent_bootstrap(
                    {key: value[None] for key, value in left.items()}, {key: value[None] for key, value in right.items()}, games, [42])
        results["baselines"][name] = entry
        print("DEV", name, {regime: {metric: entry["metrics"][regime][metric] for metric in ("log_loss", "brier_multiclass")}
                            for regime in regimes}, flush=True)
    np.savez_compressed(output/"heldout_predictions.npz", y=dy, game_pk=games, pitch_keys=keys, impossible_dp=impossible, **predictions)
    results["predictions_sha256"] = hash_file(output/"heldout_predictions.npz")
    dump(output/"frequency_results.json", results)
    if any(hash_file(PROJECT/rel) != digest for rel, digest in sources.items()) or any(hash_file(Path(path)) != digest for path, digest in reference_hashes.items()):
        raise RuntimeError("Frozen sources or reference artifacts changed during CPU execution")
    dump(output/"runtime.json", {"finished_at_utc": datetime.now(timezone.utc).isoformat(), "neural_training": False, "neural_inference": False})
    print("COMPLETE "+str(output), flush=True)


if __name__ == "__main__":
    main()
