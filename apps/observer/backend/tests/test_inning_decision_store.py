import copy
import hashlib
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from observer_app.inning_decision_store import (
    DecisionConflict, DecisionCorrupt, DecisionNotFound, InningDecisionRepository,
)
from observer_app.store import Store


ROOT = Path(__file__).parents[4]
SOURCE = ROOT / "results/EXP-C-INNING-001/conditional_keep_777063.json"
ANCHOR = Path("/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/C-ROSTER-001/decision_anchor_777063.json")


@pytest.fixture
def sample(tmp_path):
    source, anchor = tmp_path / "source.json", tmp_path / "anchor.json"
    source.write_bytes(SOURCE.read_bytes())
    anchor.write_bytes(ANCHOR.read_bytes())
    return source, anchor


@pytest.fixture
def repository(tmp_path):
    return InningDecisionRepository(Store(tmp_path / "observer.sqlite3"))


def test_import_list_resolve_and_equivalent_utc(repository, sample):
    decision_id = repository.import_result(*sample)
    with repository.store.transaction(write=False) as db:
        created = db.execute("SELECT created_at FROM inning_decisions").fetchone()[0]
    assert repository.import_result(*sample) == decision_id
    with repository.store.transaction(write=False) as db:
        assert db.execute("SELECT created_at FROM inning_decisions").fetchone()[0] == created
    catalog = repository.list_decisions(777063)
    assert catalog["schema_version"] == "inning-decision-v1"
    assert catalog["mode"] == "historical_decision_review"
    assert len(catalog["decisions"]) == 1
    entry = catalog["decisions"][0]
    assert entry["decision_id"] == decision_id
    assert entry["context"]["linkage"]["anchor_time_utc"].endswith(".827000Z")
    assert "result" not in entry
    assert repository.list_decisions(42)["decisions"] == []
    expected = copy.deepcopy(entry["context"])
    expected["linkage"]["anchor_time_utc"] = expected["linkage"]["anchor_time_utc"].replace("Z", "+00:00")
    resolved = repository.resolve(decision_id, 1, expected)
    assert resolved["result"]["linkage"]["anchor_time_utc"].endswith(".827Z")
    assert resolved["result"]["estimate"]["lower"] > 0


def test_malformed_missing_and_mismatched_inputs(repository, sample):
    decision_id = repository.import_result(*sample)
    context = repository.list_decisions(777063)["decisions"][0]["context"]
    with pytest.raises(DecisionNotFound):
        repository.resolve("inning-decision-" + "0" * 64, 1, context)
    for bad in (True, 0, 2**53, "777063"):
        with pytest.raises(ValueError):
            repository.list_decisions(bad)
    for bad in ("bad", "inning-decision-" + "G" * 64, True):
        with pytest.raises(ValueError):
            repository.resolve(bad, 1, context)
    for bad in (True, 0, "1"):
        with pytest.raises(ValueError):
            repository.resolve(decision_id, bad, context)
    with pytest.raises(DecisionConflict):
        repository.resolve(decision_id, 2, context)
    for key, value in (("phase", "other_phase"),):
        changed = copy.deepcopy(context)
        changed[key] = value
        with pytest.raises(DecisionConflict):
            repository.resolve(decision_id, 1, changed)
    for section, key, value in (("linkage", "keep_pitcher_id", 999), ("linkage", "game_pk", 1),
                                ("initial_state", "outs", 2)):
        changed = copy.deepcopy(context)
        changed[section][key] = value
        with pytest.raises((DecisionConflict, ValueError)):
            repository.resolve(decision_id, 1, changed)
    for section, key, value in (("linkage", "anchor_time_utc", "2025-99-22T00:00:00Z"),
                                ("linkage", "official_game_date", "2025-02-30"),
                                ("linkage", "first_observed_pitch_id", "777063:49:2"),
                                ("initial_state", "outs", True)):
        changed = copy.deepcopy(context)
        changed[section][key] = value
        with pytest.raises(ValueError):
            repository.resolve(decision_id, 1, changed)
    changed = copy.deepcopy(context)
    changed["extra"] = 1
    with pytest.raises(ValueError):
        repository.resolve(decision_id, 1, changed)


def test_conflict_rollback_and_immutability(repository, sample):
    decision_id = repository.import_result(*sample)
    source = json.loads(sample[0].read_text())
    source["reason"] = "changed evaluation reason"
    sample[0].write_text(json.dumps(source))
    with pytest.raises(DecisionConflict):
        repository.import_result(*sample)
    with repository.store.transaction(write=False) as db:
        assert db.execute("SELECT count(*) FROM inning_decisions").fetchone()[0] == 1
    with repository.store.transaction() as db:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            db.execute("UPDATE inning_decisions SET result_sha256=? WHERE decision_id=?", ("0" * 64, decision_id))
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            db.execute("DELETE FROM inning_decisions WHERE decision_id=?", (decision_id,))


def test_same_event_with_changed_state_conflicts(repository, sample):
    decision_id = repository.import_result(*sample)
    source = json.loads(sample[0].read_text())
    anchor = json.loads(sample[1].read_text())
    state = source["evaluation_identity"]["initial_state_and_count"]
    state["outs"] = (state["outs"] + 1) % 3
    anchor["state_as_of_anchor"]["outs"] = state["outs"]
    sample[1].write_text(json.dumps(anchor))
    source["source_anchor_sha256"] = hashlib.sha256(sample[1].read_bytes()).hexdigest()
    sample[0].write_text(json.dumps(source))
    with pytest.raises(DecisionConflict):
        repository.import_result(*sample)
    assert repository.list_decisions(777063)["decisions"][0]["decision_id"] == decision_id


def test_failed_conversion_leaves_no_row(repository, sample):
    source = json.loads(sample[0].read_text())
    source["game_pk"] = 1
    sample[0].write_text(json.dumps(source))
    with pytest.raises(ValueError):
        repository.import_result(*sample)
    assert repository.list_decisions(777063)["decisions"] == []


@pytest.mark.parametrize("column,value", [
    ("result_sha256", "0" * 64), ("revision", 2), ("game_id", 42),
    ("context_json", "{}"), ("result_json", "{}"),
])
def test_corrupt_rows_are_never_returned(repository, sample, column, value):
    decision_id = repository.import_result(*sample)
    context = repository.list_decisions(777063)["decisions"][0]["context"]
    with repository.store.transaction() as db:
        db.execute("DROP TRIGGER inning_decisions_no_update")
        db.execute(f"UPDATE inning_decisions SET {column}=? WHERE decision_id=?", (value, decision_id))
    with pytest.raises(DecisionCorrupt):
        repository.resolve(decision_id, 1, context)
    if column != "game_id":
        with pytest.raises(DecisionCorrupt):
            repository.list_decisions(777063)


def test_concurrent_registration_is_one_id_and_one_row(repository, sample):
    def register(_):
        return repository.import_result(*sample)

    with ThreadPoolExecutor(max_workers=4) as pool:
        ids = list(pool.map(register, range(8)))
    assert len(set(ids)) == 1
    assert len(repository.list_decisions(777063)["decisions"]) == 1
