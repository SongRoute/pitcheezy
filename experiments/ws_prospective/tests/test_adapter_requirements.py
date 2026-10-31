"""v2 adapter requirements (REVIEW-001 requirements_for_adapters), synthetic only.

Runs in a child process (python A) so the pytest process stays free of the
pinned seq_shadow modules:
* input isolation: network blocked; sqlite3.connect allowed only for the store
  (read-only URI) and the adapter's own state DB;
* T_FINAL and the N1 module set loaded in ONE process (co-registered pins);
* CONTEXT_DISAGREES when the runner's slot disagrees with the body;
* determinism: a fresh adapter over the same forwarded bodies gives identical
  vectors; no stdout/stderr output from predict.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(HERE)]

from ws_prospective.runner import sanitize_reason  # noqa: E402

CHILD = r'''
import contextlib, io, json, socket, sqlite3, sys
from pathlib import Path
ROOT, HERE, TMP = map(Path, sys.argv[1:4])
sys.path[:0] = [str(ROOT), str(HERE)]
import test_adapter_t as TT
from ws_prospective import adapter_t
from ws_prospective.context import build_context, PreState, StorePitch
from ws_prospective.runner import Abstain
from ws_prospective.store_source import LiveStoreSource

store = TT.ShaStore(TMP / "store.sqlite3")
state_dir = TMP / "out"
allowed = {str(store.path)}
real_connect = sqlite3.connect
def guarded(target, *a, **k):
    t = str(target)
    path = t[5:].split("?")[0] if t.startswith("file:") else t
    if path in allowed and (not t.startswith("file:") or "mode=ro" in t) or path.startswith(str(state_dir)):
        return real_connect(target, *a, **k)
    raise PermissionError("sqlite access outside store/state: " + t)
sqlite3.connect = guarded
def no_net(*a, **k):
    raise PermissionError("network blocked")
socket.socket.connect = no_net
socket.create_connection = no_net

n1 = json.loads((HERE / "fixtures_s5" / "model_sources.json").read_text())["SEQ001_N1_3SEED_CAL2"]
reg = TT.make_reg()
reg["model_sources"]["SEQ001_N1_3SEED_CAL2"] = n1
# store writes happen in the parent role of this script (fake publisher): use the real connect
bodies = TT.synthetic_bodies()
sqlite3.connect = real_connect
for i, feed in enumerate(bodies):
    store.publish(feed, TT.T0 + 7.0 * i)
    snap = LiveStoreSource(store.path, TT.GAME).poll()
    if i >= 3 and snap.next_state is not None and len(snap.accepted) >= 3:
        break
sqlite3.connect = guarded

out = {}
a1 = adapter_t.TFinalAdapter.build(reg, TT.game_entry(store.path), state_dir / "a1")
import pitcheezy.seq_shadow.adapter, pitcheezy.seq_shadow.calibration  # the N1 module set
import pitcheezy.seq_shadow.model, pitcheezy.seq_shadow.tokens  # noqa: E401
m = len(snap.accepted)
ctx = build_context(TT.GAME, snap.accepted, m + 1, snap.next_state, "p" * 64,
                    source_seq=snap.source_seq, source_sha=snap.source_sha)
buf = io.StringIO()
with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
    p1 = a1.predict(ctx)
out["quiet"] = buf.getvalue() == ""
out["types"] = sorted(p1)
a2 = adapter_t.TFinalAdapter.build(reg, TT.game_entry(store.path), state_dir / "a2")
out["deterministic"] = a2.predict(ctx) == p1
bad = build_context(TT.GAME, snap.accepted[:-1], m, snap.next_state, "p" * 64,
                    source_seq=snap.source_seq, source_sha=snap.source_sha)
try:
    a2.predict(bad); out["disagree"] = "scored"
except Abstain as e:
    out["disagree"] = str(e)
try:
    real_connect  # noqa
    sqlite3.connect(str(TMP / "elsewhere.sqlite3")); out["guard"] = "open"
except PermissionError:
    out["guard"] = "blocked"
print(json.dumps(out))
'''


def test_sanitize_reason_vocabulary():
    assert sanitize_reason("CONTEXT_DISAGREES") == "CONTEXT_DISAGREES"
    assert sanitize_reason("native_prior_quality_unknown") == "native_prior_quality_unknown"
    assert sanitize_reason("seq_predictor_error:ValueError:p=0.31 for FF") == \
        "seq_predictor_error:ValueError"
    assert sanitize_reason("predictor_error:predict:KeyError:'x'") == "predictor_error:predict:KeyError"
    assert sanitize_reason("free text with 0.42") == "UNLISTED_REASON"


def test_t_isolation_coexistence_disagree_determinism(tmp_path):
    import test_adapter_t as TT
    if not TT.HAVE_ARTIFACTS:
        pytest.skip("pinned T artifacts/source absent")
    r = subprocess.run([sys.executable, "-c", CHILD, str(ROOT), str(HERE), str(tmp_path)],
                       capture_output=True, text=True, timeout=600,
                       env={"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PATH": "/usr/bin:/bin"})
    assert r.returncode == 0, r.stderr[-3000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["quiet"] and out["deterministic"] and out["guard"] == "blocked"
    assert out["disagree"] == "CONTEXT_DISAGREES"
    assert len(out["types"]) >= 1
