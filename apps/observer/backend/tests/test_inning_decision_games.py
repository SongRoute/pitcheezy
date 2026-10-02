"""The historical game index is derived only from validated decision records."""

import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from observer_app.inning_decision_store import DecisionCorrupt, _read_row
from observer_app.main import create_app
from observer_app.service import ObserverService
from observer_app.store import Store


ROOT = Path(__file__).parents[4]
SOURCE = ROOT / "results/EXP-C-INNING-001/conditional_keep_777063.json"
ANCHOR = Path("/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/C-ROSTER-001/decision_anchor_777063.json")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


@pytest.fixture
def service(tmp_path):
    return ObserverService(None, None, Store(tmp_path / "observer.sqlite3"))


def insert_synthetic_decision(repository, *, game_id=777063, game_date="2025-07-21", action_index=1):
    """Insert a second internally valid record to exercise game grouping."""
    with repository.store.transaction(write=False) as db:
        row = db.execute("SELECT * FROM inning_decisions LIMIT 1").fetchone()
        context, result = _read_row(row)
    context, result = copy.deepcopy(context), copy.deepcopy(result)
    for linkage in (context["linkage"], result["linkage"]):
        linkage["game_pk"] = game_id
        linkage["official_game_date"] = game_date
        linkage["anchor_action_index"] = action_index
        linkage["first_observed_pitch_id"] = f"{game_id}:49:1"
    context["initial_state"]["date"] = game_date
    result["provenance"]["evaluation_identity"]["initial_state_and_count"]["date"] = game_date
    identity = {key: context["linkage"][key] for key in
                ("game_pk", "anchor_kind", "anchor_time_utc", "anchor_action_index", "first_observed_pitch_id")}
    decision_id = "inning-decision-" + hashlib.sha256(canonical(identity).encode()).hexdigest()
    result_json = canonical(result)
    with repository.store.transaction() as db:
        db.execute("""INSERT INTO inning_decisions
            (decision_id, game_id, revision, context_json, result_json, result_sha256, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (decision_id, game_id, 1, canonical(context), result_json,
             hashlib.sha256(result_json.encode()).hexdigest(), row["created_at"]))
    return decision_id


def test_empty_registry_returns_empty_games_without_sessions(service):
    with TestClient(create_app(service)) as client:
        response = client.get("/api/inning-decision-games")
    assert response.status_code == 200
    assert response.json() == {"schema_version": "inning-decision-v1",
                               "mode": "historical_decision_review", "games": []}
    with service.store.transaction(write=False) as db:
        assert db.execute("SELECT count(*) FROM sessions").fetchone()[0] == 0


def test_import_duplicate_and_sorted_game_index(service):
    repo = service.inning_decisions
    decision_id = repo.import_result(SOURCE, ANCHOR)
    assert repo.import_result(SOURCE, ANCHOR) == decision_id
    insert_synthetic_decision(repo, action_index=1)
    insert_synthetic_decision(repo, game_id=777064, game_date="2025-07-20")
    expected = [{"game_id": 777064, "date": "2025-07-20", "decision_count": 1},
                {"game_id": 777063, "date": "2025-07-21", "decision_count": 2}]
    with TestClient(create_app(service)) as client:
        response = client.get("/api/inning-decision-games")
    assert response.status_code == 200
    assert response.json()["games"] == expected
    assert repo.list_games()["games"] == expected
    with service.store.transaction(write=False) as db:
        assert db.execute("SELECT count(*) FROM sessions").fetchone()[0] == 0


@pytest.mark.parametrize("corruption", ["invalid_date", "inconsistent_date"])
def test_corrupt_date_or_conflicting_game_date_returns_503(service, corruption):
    repo = service.inning_decisions
    repo.import_result(SOURCE, ANCHOR)
    if corruption == "invalid_date":
        with service.store.transaction() as db:
            db.execute("DROP TRIGGER inning_decisions_no_update")
            row = db.execute("SELECT decision_id, context_json FROM inning_decisions").fetchone()
            context = json.loads(row["context_json"])
            context["linkage"]["official_game_date"] = "2025-02-30"
            db.execute("UPDATE inning_decisions SET context_json=? WHERE decision_id=?",
                       (canonical(context), row["decision_id"]))
    else:
        insert_synthetic_decision(repo, game_date="2025-07-22")
    with pytest.raises(DecisionCorrupt):
        repo.list_games()
    with TestClient(create_app(service)) as client:
        response = client.get("/api/inning-decision-games")
    assert response.status_code == 503
    assert set(response.json()) == {"detail"}
