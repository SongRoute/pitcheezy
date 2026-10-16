from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json

import pandas as pd
import pytest

from pitcheezy.data import postseason_six as six
from experiments.ws_campaign.source.abc_corpus import PROJECT


def body(game_pk, official_date, *, game_type="L", pitch_number=1):
    row = dict.fromkeys(PROJECT, 0)
    row.update(game_pk=game_pk, at_bat_number=1, pitch_number=pitch_number,
               game_date=official_date, game_type=game_type, home_team="NYY", away_team="SEA",
               pitcher=10, batter=20, pitch_type="FF", description="ball", events="",
               bb_type="", inning_topbot="Top", stand="R", p_throws="R")
    return pd.DataFrame([row], columns=PROJECT).to_csv(index=False).encode()


def test_fixed_six_projected_custody(tmp_path):
    expected = {pk: body(pk, date) for pk, date in six.GAMES}
    seen = []

    def fetch(pk):
        seen.append(pk)
        return expected[pk]

    out = tmp_path / "fresh"
    result = six.run(out, fetch=fetch)
    assert seen == [pk for pk, _ in six.GAMES]
    assert result["status"] == "COMPLETE" and result["projected"]["rows"] == 6
    assert result["projected"]["columns"] == PROJECT
    assert result["raw_total_bytes"] == sum(map(len, expected.values()))
    assert all(r["raw_sha256"] == hashlib.sha256(expected[r["game_pk"]]).hexdigest() for r in result["games"])
    assert json.loads((out / "manifest.json").read_text()) == result
    assert not (out / "STOP.json").exists()
    with pytest.raises(six.CollectionStop, match="fresh"):
        six.run(out, fetch=fetch)


@pytest.mark.parametrize("bad", ["wrong_game", "wrong_date", "wrong_type", "duplicate"])
def test_identity_failure_preserves_raw_and_stop(tmp_path, bad):
    pk, date = six.GAMES[0]
    original = body(pk, date)
    if bad == "wrong_game":
        source = body(pk + 1, date)
    elif bad == "wrong_date":
        source = body(pk, "2026-10-16")
    elif bad == "wrong_type":
        source = body(pk, date, game_type="R")
    else:
        source = original + original.split(b"\n", 1)[1]
    out = tmp_path / "stopped"
    with pytest.raises(six.CollectionStop, match="STOP"):
        six.run(out, fetch=lambda _: source)
    assert (out / "STOP.json").exists()
    assert (out / "raw" / f"game-{pk}.csv").read_bytes() == source
    assert not (out / "postseason_six.parquet").exists()


def test_http_exact_key_redirect_and_body_cap(monkeypatch):
    requests = []
    class Response:
        status = 200
        def getheader(self, name):
            return None
        def read(self, size):
            return b""
    class Connection:
        def __init__(self, host, timeout):
            assert host == six.HOST and timeout == six.TIMEOUT
        def request(self, method, path, headers):
            requests.append((method, path))
        def getresponse(self):
            return Response()
        def close(self):
            pass
    monkeypatch.setattr(six.http.client, "HTTPSConnection", Connection)
    assert six.fetch_game(six.GAMES[0][0]) == b""
    assert requests == [("GET", six.PATH.format(six.GAMES[0][0]))]
    with pytest.raises(six.CollectionStop, match="unregistered"):
        six.fetch_game(1)
    Response.status = 302
    with pytest.raises(six.CollectionStop, match="HTTP status"):
        six.fetch_game(six.GAMES[0][0])


def test_managed_gate_binds_review_and_source_without_network(tmp_path, monkeypatch):
    repo = six.Path(six.__file__).resolve().parents[3]
    names = ("scripts/fetch_data.py", "src/pitcheezy/data/postseason_six.py",
             "src/pitcheezy/data/statcast_fetch.py", "experiments/ws_campaign/source/abc_corpus.py")
    from pybaseball.datasources import statcast as parser
    parser_files = {"csv_parser": six.Path(inspect.getsourcefile(parser)),
                    "postprocessing": six.Path(inspect.getsourcefile(parser.postprocessing))}
    reg = {"schema": "postseason-six-acquisition-v1", "execute": False, "worktree": str(repo),
           "attempt_id": "generated", "author": "sol_replay", "output": str(tmp_path / "unused"),
           "source_pins": {n: six._file_sha(repo / n) for n in names},
           "parser_source_pins": {k: {"path": str(p), "sha256": six._file_sha(p)}
                                  for k, p in parser_files.items()},
           "versions": {"pybaseball": importlib.metadata.version("pybaseball"),
                        "pandas": pd.__version__, "pyarrow": six.pa.__version__},
           "games": [list(x) for x in six.GAMES],
           "source_route": f"https://{six.HOST}" + six.PATH.format("{game_pk}"),
           "max_requests": 6, "max_body_bytes_each": six.MAX_BODY,
           "max_rows_each": six.MAX_ROWS, "max_columns_each": six.MAX_COLUMNS,
           "max_output_bytes": six.MAX_OUTPUT, "wall_seconds": six.MAX_WALL_SECONDS,
           "socket_timeout_seconds": six.TIMEOUT,
           "schedule_projection_sha256": six.SCHEDULE_SHA256}
    reg_path = tmp_path / "reg.json"
    reg_path.write_text(json.dumps(reg))
    review = {"registration_sha256": six._file_sha(reg_path),
              "reviewer": "sol_review", "verdict": "PASS_SOURCE_ONLY"}
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps(review))
    grant = {"approved_execute": True, "authorized_by": "root", "verdict": "PASS",
             "attempt_id": "generated", "registration_sha256": six._file_sha(reg_path),
             "reviewer": "sol_review", "review_path": str(review_path),
             "review_sha256": six._file_sha(review_path)}
    grant_path = tmp_path / "grant.json"
    grant_path.write_text(json.dumps(grant))
    monkeypatch.setattr(six, "run", lambda path: {"output": str(path)})
    assert six.gated(reg_path, grant_path) == {"output": reg["output"]}
    grant["reviewer"] = "sol_replay"
    grant_path.write_text(json.dumps(grant))
    with pytest.raises(six.CollectionStop, match="grant"):
        six.gated(reg_path, grant_path)
    grant["reviewer"] = "sol_review"
    grant_path.write_text(json.dumps(grant))
    reg["source_pins"][names[0]] = "0" * 64
    reg_path.write_text(json.dumps(reg))
    with pytest.raises(six.CollectionStop, match="source pin"):
        six.gated(reg_path, grant_path)
