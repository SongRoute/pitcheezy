"""Verify/cache sequence physics and report the compact, conditional token store."""
from pathlib import Path
import argparse
import json
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pitchmdp.data import utc_now, write_json
from pitchmdp.sequence_data import HistoryStore, prepare_frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/local.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    frame = prepare_frame(config)
    store = HistoryStore.from_frame(frame)
    report = {"created_at_utc": utc_now(), "rows": len(frame),
              "data_identity": frame.attrs["sequence_data_identity"],
              "normalizer": store.normalizer.report(),
              "history_length": store.indices.shape[1], "history_scope": "same PA; prior rows only",
              "physical_array_bytes": store.physical.nbytes, "history_index_bytes": store.indices.nbytes,
              "padding_entries": int((store.indices < 0).sum()),
              "current_token": "observed current physics; conditional outcome evaluation, not pre-pitch input"}
    output = Path(config["artifact_root"]) / "reports/sequence_data.json"
    write_json(output, report)
    print(json.dumps({"report": str(output), **report}, indent=2))
