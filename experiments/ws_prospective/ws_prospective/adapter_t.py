"""T_FINAL_3SEED_TEMP adapter: stored GUMBO body -> accepted capture-native chain.

Replicates the accepted chain of ``scripts/fullseason_capture_native_replay.py``
(pinned root, codex/ws-postseason-six-1017 @7a64583) per forwarded body:

    Runtime(adapter_config, profile_dir, manifest_sha)
      -> captured_native_state.NativeRuntime(runtime, allow_native_experimental=True)
    NativeShadow(state_path, game_pk, game_date, normalizer_sha256)
    per body:  captured_bridge.step(..., candidate_pitch_type='FF') -> events
               captured_native.inspect_tokens(...)                  -> native rows
               state.ingest(event, native_row, received_ns, sha) for each event
               state.observe_receipt(ordinal, sha, received_ns)
    per registered candidate type t (in the bridge's TYPES):
               candidate = step(..., candidate_pitch_type=t)['candidate']
               decision_id += ':' + t   (the bridge id omits the type and
                                         NativeShadow refuses id reuse)
               decision = state.decide(candidate, received_ns)
               result = scorer.predict(decision)

Only bodies the runner forwards (``ctx.source_seq``) are ingested; each stored
body is a full game prefix, so skipped bodies leave no gap in the evidence.
The receipt ordinal is our own gapless counter continuing from
MAX(native_receipts.receipt_seq) of the state DB on restart.  Every forwarded
body gets a new ordinal, also a repeat of an already ingested body (e.g. the
same context after a restart, or after a crash part-way through an ingest):
NativeShadow records unchanged pitches as 'unchanged_observation', fills any
missing ones, and a repeated decision_id returns the original frozen decision.
(Re-ingesting at the *same* ordinal is refused by NativeShadow.ingest.)

The adapter never reads anything but the stored snapshot named by the context,
verified against ``ctx.source_sha``; it never fits or calibrates.
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional

from . import LEAVES
from .adapters import AdapterBase, PinMismatch, pinned_sys_path, sha256_file, verify_pins
from .context import PrePitchContext
from .runner import Abstain
from . import store_source

MODEL_ID = "T_FINAL_3SEED_TEMP"

# Files whose canonical {relpath: sha256} map defines the native normalizer
# identity (NORMALIZER_SHA_SEMANTICS = 'sha256(canonical_json(native_source_pins))'),
# exactly the set of the accepted native-replay / forecast-persistence registrations.
NATIVE_SOURCE_FILES = (
    "src/pitcheezy/seq_shadow/captured_bridge.py",
    "src/pitcheezy/seq_shadow/captured_diagnostic.py",
    "src/pitcheezy/seq_shadow/captured_identity_bip.py",
    "src/pitcheezy/seq_shadow/captured_native.py",
    "src/pitcheezy/seq_shadow/captured_native_identity.py",
    "src/pitcheezy/seq_shadow/captured_native_state.py",
    "src/pitcheezy/seq_shadow/fullseason_stream.py",
    "src/pitcheezy/seq_shadow/ledger.py",
    "src/pitcheezy/seq_shadow/tokens.py",
)
# normalizer_sha256 of the accepted registrations (FORECAST-PERSISTENCE / NATIVE-IDENTITY-GUARD).
ACCEPTED_NORMALIZER_SHA256 = "c140e31b9e38eb1dfa90fbf4d60cf014d24c791900dadf2fef6a7939dc72b3e8"
# Computed with source_pins() after a full synthetic run (build + 16-type scoring) on
# codex-ws-postseason-six-1017 @7a64583: every module file imported from that root.
# Reference value for the v2 registration's model_sources.T_FINAL_3SEED_TEMP.source_pins.
PINNED_SOURCE_PINS_7A64583 = {
    "experiments/ws_campaign/__init__.py":
        "5b9e1534383b9c8a10a027121fdc987c836873dbed45afb742695b548b180fcc",
    "experiments/ws_campaign/models.py":
        "3e51e26a778b1bdb8e9853e31bd17d8ac4f8dce4561e66b53fa389de130df32f",
    "experiments/ws_campaign/source/abc_corpus.py":
        "4f119b44c80908f62d76f932e70538c105ffa18007649e075683b7f1d4e2feb4",
    "experiments/ws_campaign/source/first_small_features.py":
        "9918f8d1cff07a217f05ea75a8d8fbb299cd8c5174b5f46b98a3d5e41e8f6a16",
    "experiments/ws_campaign/source/history_transfer.py":
        "91502c5f318f41ffa3db40d870a1d0469bc8db411ac4541ac35c750a342f269f",
    "experiments/ws_campaign/source/outcome_transfer.py":
        "4253a7374eb55674948c37fa0b5b303500ead37200f5a57426df57911ab58065",
    "experiments/ws_campaign/source/scale_runner.py":
        "e25db72841066e28aa28b25bcd4ed07e7b7c57e199dbf9ab33f947f80ba0a436",
    "experiments/ws_campaign/source/scale_safe.py":
        "1f2cd2e221bcff53c9f2b1b1ee7ceae0476f02bf5559a21e31a8b520f94b5234",
    "experiments/ws_fullseason/audit.py":
        "116edf4848c645d69b20125f52ac43b8a7f46d9a29ccea3d947e373057b3ef23",
    "experiments/ws_fullseason/calibrate.py":
        "6209cc8c6519dc6aa4536ea7796d1ec3ea07a8a9608653568b42211e08757fbc",
    "experiments/ws_fullseason/calibrate_worker.py":
        "9159e167c3011129ee1c36fcaad235ef9ea94cb2357a3cfaeafffb846b8cf323",
    "experiments/ws_fullseason/data.py":
        "d289e6b4551e0cbf62eb90ec065057954af12dd8c06183f6ba8ad3461ae7db8f",
    "experiments/ws_seq/tm002.py":
        "352a160c0e80334e6224d26066cf1cf74858bd791d31b12a50a389aafe75614d",
    "experiments/ws_seq/train_seq.py":
        "1328c4148f5fa9e89728207a06dc839df72c26a7f6ead89a90ae4ecec8c24ac0",
    "src/pitcheezy/__init__.py":
        "f8634c6d6f3ba0a0c13fdb9ad9a1c5c9dce337e6e86f07238f6cf8c51696f44b",
    "src/pitcheezy/seq_shadow/__init__.py":
        "e553a739d5b9c00af52be5a3d3f09760169b923ceed9043d8de821c63b2fa96b",
    "src/pitcheezy/seq_shadow/adapter.py":
        "e0d429d55b77c7720c898f737ab8391df3688d9599b28a34d3f340ee5cee18b3",
    "src/pitcheezy/seq_shadow/calibration.py":
        "031a8458ea7fb6d809b3716dea928208e87010dcea121349409a95ef8a0b004b",
    "src/pitcheezy/seq_shadow/captured_bridge.py":
        "9a620f4443eac33f3e487446b24413f2c8017c7564055e4c11451fe3c5208cec",
    "src/pitcheezy/seq_shadow/captured_diagnostic.py":
        "49afcd709b3608d9e7cf184e7ab5f3bf8e1bfb66313bee1f089998339aa52995",
    "src/pitcheezy/seq_shadow/captured_identity_bip.py":
        "787283a57b0c57023eab1e6278bc8ad853c287581b02d3e598028da9b49cc456",
    "src/pitcheezy/seq_shadow/captured_native.py":
        "e51a80dda7f75f3636a5116fea8fd13d5eb8da407b2b64fa823ab5df9b92f150",
    "src/pitcheezy/seq_shadow/captured_native_identity.py":
        "def298d1278b14b69d6708423fb74c0df22ede4edd679dbe998efa8f1b7e45d2",
    "src/pitcheezy/seq_shadow/captured_native_state.py":
        "8846922c022dd1f5b1cab10ccfed0ee16079f3f8999f43f94726ba34d2038d91",
    "src/pitcheezy/seq_shadow/fullseason.py":
        "25a584e6a31fd72eedccda330a5c8bdc12e7a25ffa5ee69d42f53d1d32c9853e",
    "src/pitcheezy/seq_shadow/fullseason_profiles.py":
        "d628d2ab21a7a63935899e70d72b4808e56054147feb5d18db8f3473071bd663",
    "src/pitcheezy/seq_shadow/fullseason_runtime.py":
        "a125905f747863feee0930600adba5afbc150281b19702c74963fd0332439480",
    "src/pitcheezy/seq_shadow/fullseason_stream.py":
        "9281cf765505324e0d3ef699c170d5e87400a6958ce369406f72fab07af57feb",
    "src/pitcheezy/seq_shadow/ledger.py":
        "49706193f3aca571f3ed0c5b287250042c19fa7b2d0682dc139405963dc8c551",
    "src/pitcheezy/seq_shadow/tokens.py":
        "f804d4ecb8f8dbffc10d6736fb8a49d6223dc7dc6094b20ad8c620f2d6725631",
}
INGEST_TYPE = "FF"  # events do not depend on the candidate type; accepted chain used FF
# Exact messages of the pinned strict ledger (captured_native_state.NativeShadow.ingest /
# fullseason_stream.StreamState.ingest) when the provider re-issues a pitch's identity
# (e.g. a new playId for the same (ab, pitch)).  The append-only ledger refuses it by
# design and every later body of the game fails at the same event.
STRICT_IDENTITY_MESSAGES = frozenset((
    "physical/provider identity reassignment",
    "stale or conflicting native receipt revision",
))
STRICT_IDENTITY_ABSTAIN = "STRICT_IDENTITY_REASSIGNED"

REGISTRATION_KEYS = {
    "source_root": "absolute path of the pinned source checkout (codex-ws-postseason-six-1017 @7a64583)",
    "source_commit": "optional, recorded only",
    "source_pins": "{relpath under source_root: sha256} - must cover every module imported from "
                   "source_root (see source_pins()); verified before import and re-checked after",
    "normalizer_sha256": "optional; if present must equal sha(canonical({f: source_pins[f] for f in "
                         "NATIVE_SOURCE_FILES}))",
    "adapter_config_path": "ADAPTER-CONFIG.json (binds checkpoints/encoder/calibrator/recipe by "
                           "path+bytes+sha; fullseason.load verifies them and its own source_pins)",
    "adapter_config_sha256": "sha256 of adapter_config_path bytes",
    "profile_dir": "profile bundle directory (manifest.json + npz archives)",
    "profile_manifest_sha256": "sha256 of profile_dir/manifest.json (load_bundle verifies archives)",
    "candidate_pitch_types": "optional; defaults to reg['candidate_pitch_types']",
}
REQUIRED_KEYS = ("source_root", "source_pins", "adapter_config_path", "adapter_config_sha256",
                 "profile_dir", "profile_manifest_sha256")


def _under(path, root: Path) -> Optional[str]:
    try:
        p = Path(path).resolve()
    except (OSError, TypeError):
        return None
    try:
        return p.relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return None


def _registered_pins(reg: dict) -> Dict[str, str]:
    """Union of every model_sources block's source_pins as absolute paths."""
    out: Dict[str, str] = {}
    for block in (reg.get("model_sources") or {}).values():
        if not isinstance(block, dict):
            continue
        base = block.get("source_root") or block.get("seq_shadow_src_root") or "/"
        for k, v in (block.get("source_pins") or {}).items():
            out[str((Path(k) if Path(k).is_absolute() else Path(base) / k).resolve())] = v
    return out


def source_pins(root) -> Dict[str, str]:
    """{relpath: sha256} of every loaded module whose file lies under ``root``."""
    out = {}
    for mod in tuple(sys.modules.values()):
        f = getattr(mod, "__file__", None)
        if not f:
            continue
        rel = _under(f, Path(root))
        if rel is not None and Path(f).is_file():
            out[rel] = sha256_file(Path(root) / rel)
    return dict(sorted(out.items()))


def _native_sha(seq_ledger, pins: Dict[str, str]) -> str:
    try:
        native = {f: pins[f] for f in NATIVE_SOURCE_FILES}
    except KeyError as e:
        raise PinMismatch(f"native source file not pinned: {e}") from None
    return seq_ledger.sha(native)


class TFinalAdapter(AdapterBase):
    model_id = MODEL_ID
    leaf_order = LEAVES

    def __init__(self, *, root: Path, pins: Dict[str, str], game_pk: int, game_date: str,
                 store_path: Path, state_path: Path, types: List[str], config: dict,
                 profile_dir: str, manifest_sha: str, normalizer_sha256: Optional[str] = None,
                 co_pins: Optional[Dict[str, str]] = None):
        self.root, self.pins = Path(root), dict(pins)
        # modules pinned by another registered model of the same process (N1 shares this root)
        self.co_pins = {str(Path(k).resolve()): v for k, v in (co_pins or {}).items()}
        self.game_pk, self.game_date = int(game_pk), str(game_date)
        self.store_path, self.state_path = Path(store_path), Path(state_path)
        verify_pins(self.root, self.pins)
        pinned_sys_path([str(self.root), str(self.root / "src")])
        for name in ("pitcheezy", "experiments.ws_campaign", "experiments.ws_seq",
                     "experiments.ws_fullseason"):
            mod = sys.modules.get(name)
            f = getattr(mod, "__file__", None) if mod else None
            # namespace packages have no __file__; their modules are checked by _check_closure
            if mod is not None and f is not None and _under(f, self.root) is None:
                raise PinMismatch(f"module {name} already imported from another tree")
        from pitcheezy.seq_shadow import (captured_bridge, captured_native,  # noqa: E402
                                          captured_native_state)
        from pitcheezy.seq_shadow.fullseason_runtime import Runtime
        from pitcheezy.seq_shadow import ledger as seq_ledger
        from experiments.ws_campaign.source.first_small_features import TYPES
        self._bridge, self._native, self._cns = captured_bridge, captured_native, captured_native_state
        self.normalizer_sha256 = _native_sha(seq_ledger, self.pins)
        if normalizer_sha256 is not None and normalizer_sha256 != self.normalizer_sha256:
            raise PinMismatch("normalizer_sha256 != sha(native source pins)")
        for rel, digest in config.get("source_pins", {}).items():
            if self.pins.get(rel) != digest:
                raise PinMismatch(f"adapter config source pin not in registration pins: {rel}")
        if (config.get("schema") != "ws-fullseason-shadow-v1" or config.get("seeds") != [0, 1, 2]
                or config.get("leaf_order") != list(LEAVES) or config.get("threads") != 1):
            raise PinMismatch("adapter identity changed")
        self.types = [t for t in types if t in TYPES and t != "UNKNOWN"]
        self.skipped_types = [t for t in types if t not in self.types]
        if not self.types:
            raise ValueError("no registered candidate type is in the bridge TYPES")
        self.runtime = Runtime(config, profile_dir, manifest_sha)
        self.scorer = captured_native_state.NativeRuntime(self.runtime, allow_native_experimental=True)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state = captured_native_state.NativeShadow(
            self.state_path, game_pk=self.game_pk, game_date=self.game_date,
            normalizer_sha256=self.normalizer_sha256)
        if self.state.db.execute("PRAGMA journal_mode").fetchone()[0] != "delete":
            self.state.close()
            raise ValueError("native state SQLite must use single-file rollback journal mode")
        self._n_modules = -1
        self._strict_identity_broken: Optional[str] = None  # sticky for this game
        self._check_closure()

    # -- registration ------------------------------------------------------
    @classmethod
    def build(cls, reg: dict, game: dict, out_dir: Path) -> "TFinalAdapter":
        src = reg["model_sources"][MODEL_ID]
        missing = [k for k in REQUIRED_KEYS if k not in src]
        if missing:
            raise KeyError(f"model_sources.{MODEL_ID} missing {missing}")
        if sha256_file(src["adapter_config_path"]) != src["adapter_config_sha256"]:
            raise PinMismatch("adapter config changed")
        config = json.loads(Path(src["adapter_config_path"]).read_text())
        types = list(src.get("candidate_pitch_types") or reg["candidate_pitch_types"])
        gpk = int(game["game_pk"])
        return cls(root=Path(src["source_root"]), pins=src["source_pins"], game_pk=gpk,
                   game_date=str(game["official_date"]), store_path=Path(game["store_path"]),
                   state_path=Path(out_dir) / f"game_{gpk}" / "native-T.sqlite3", types=types,
                   config=config, profile_dir=src["profile_dir"],
                   manifest_sha=src["profile_manifest_sha256"],
                   normalizer_sha256=src.get("normalizer_sha256"),
                   co_pins=_registered_pins(reg))

    def close(self) -> None:
        self.state.close()

    def _check_closure(self) -> None:
        """Every module imported from the pinned root must be pinned (and unchanged)."""
        if len(sys.modules) == self._n_modules:
            return
        loaded = source_pins(self.root)
        bad = sorted(rel for rel, d in loaded.items()
                     if self.pins.get(rel) != d
                     and self.co_pins.get(str((self.root / rel).resolve())) != d)
        if bad:
            raise PinMismatch(f"unpinned or changed modules imported from source root: {bad}")
        self._n_modules = len(sys.modules)

    # -- chain -------------------------------------------------------------
    def _next_ordinal(self) -> int:
        last = self.state.db.execute("SELECT MAX(receipt_seq) FROM native_receipts").fetchone()[0]
        return 1 if last is None else int(last) + 1

    def _ingest(self, feed: dict, sha: str, received_ns: int) -> int:
        seq = self._next_ordinal()
        step = self._bridge.step(feed, game_pk=self.game_pk, game_date=self.game_date,
                                 receipt_seq=seq, source_sha256=sha, candidate_pitch_type=INGEST_TYPE)
        natives = {}
        for row in self._native.inspect_tokens(feed, game_pk=self.game_pk, game_date=self.game_date,
                                               receipt_seq=seq, source_sha256=sha):
            if (row["input_contract"] != self._native.CONTRACT or row["accepted"] is not False
                    or row["statcast_training_equivalent"] is not False
                    or row["receipt_seq"] != seq or row["source_sha256"] != sha):
                raise ValueError("native source row custody or acceptance changed")
            key = tuple(row["key"])
            if key in natives:
                raise ValueError("duplicate native physical key in one body")
            natives[key] = row
        try:
            for event in step["events"]:
                self.state.ingest(event, natives.get(tuple(event["key"])), received_ns=received_ns,
                                  source_sha256=sha)
        except ValueError as e:
            if str(e) not in STRICT_IDENTITY_MESSAGES:
                raise
            # The strict ledger cannot recover within a game (append-only, identity never
            # reassigned), and this ingest is not atomic: NativeShadow.ingest has already
            # recorded this body's receipt on its first event.  Abstain for the rest of the
            # game instead of re-ingesting (and failing) every later body.
            self._check_closure()  # the failed ingest ran pinned code; a pin breach still wins
            self._strict_identity_broken = str(e)
            raise Abstain(STRICT_IDENTITY_ABSTAIN) from None
        self.state.observe_receipt(seq, sha, received_ns)
        return seq

    def predict(self, ctx: PrePitchContext) -> Dict[str, List[float]]:
        if ctx.source_seq is None:
            raise Abstain("NO_SOURCE_BODY")
        if int(ctx.game_pk) != self.game_pk:
            raise ValueError("context game_pk != adapter game")
        if self._strict_identity_broken is not None:  # no read, no bridge step, no ingest
            raise Abstain(STRICT_IDENTITY_ABSTAIN)
        snap = store_source.read_snapshot_raw(self.store_path, ctx.source_seq)
        body = snap["body_bytes"]
        sha = hashlib.sha256(body).hexdigest()
        if sha != ctx.source_sha or snap["sha256"] != ctx.source_sha or snap["game_pk"] != self.game_pk:
            raise ValueError("stored snapshot does not match the context's source sha/game")
        feed = json.loads(body)
        if (not isinstance(feed, dict) or not isinstance(feed.get("gameData"), dict)
                or not isinstance(feed["gameData"].get("game"), dict)
                or not isinstance(feed["gameData"].get("datetime"), dict)
                or feed["gameData"]["game"].get("pk") != self.game_pk
                or feed["gameData"]["datetime"].get("officialDate") != self.game_date):
            raise ValueError("captured body game/officialDate mismatch")
        received_ns = int(snap["fetched"] * 1_000_000_000)
        # Requirement: the history re-derived from the body must be exactly the
        # runner's slot (n-1 pitches, same next key); otherwise never proceed.
        chk = self._bridge.step(feed, game_pk=self.game_pk, game_date=self.game_date,
                                receipt_seq=self._next_ordinal(), source_sha256=sha,
                                candidate_pitch_type=self.types[0])
        if len(chk["events"]) != int(ctx.slot_ordinal) - 1:
            raise Abstain("CONTEXT_DISAGREES")
        c = chk["candidate"]
        if c is not None and (c["query"]["at_bat_number"], c["query"]["pitch_number"]) != (
                int(ctx.state.at_bat_index), int(ctx.state.pitch_number)):
            raise Abstain("CONTEXT_DISAGREES")
        seq = self._ingest(feed, sha, received_ns)
        probs: Dict[str, List[float]] = {}
        reasons: List[str] = []
        for t in self.types:
            step = self._bridge.step(feed, game_pk=self.game_pk, game_date=self.game_date,
                                     receipt_seq=seq, source_sha256=sha, candidate_pitch_type=t)
            cand = step["candidate"]
            if cand is None:
                reasons.append("bridge_" + str(step["abstain_reason"]))
                break  # bridge abstention does not depend on the candidate type
            if (cand["coverage"]["source_sha256"] != sha
                    or cand["pitch_type_basis"] != "hypothetical_candidate"
                    or cand["query"]["pitch_type"] != t):
                raise ValueError("candidate receipt proof or query identity changed")
            cand = dict(cand, decision_id=f"{cand['decision_id']}:{t}")
            decision = self.state.decide(cand, received_ns=received_ns)
            result = self.scorer.predict(decision)
            if result.get("prediction_ready") is not False or result.get("prospective_claim") is not False:
                raise ValueError("experimental result claimed readiness or prospectivity")
            if result["status"] == "experimental_conditional_score":
                p = result["probabilities"]
                if (not isinstance(p, list) or len(p) != 7
                        or any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 or x > 1
                               for x in p)
                        or not math.isclose(sum(p), 1, rel_tol=0, abs_tol=1e-6)):
                    raise ValueError("invalid experimental seven-leaf output")
                if list(result.get("leaf_order") or LEAVES) != list(LEAVES):
                    raise ValueError("leaf order changed")
                probs[t] = [float(x) for x in p]
            else:
                reasons.append(str(result["reason"] or decision.get("reason")))  # e.g. native_prior_quality_unknown
        self._check_closure()
        if not probs:
            raise Abstain(reasons[0] if reasons else "NO_CANDIDATE_TYPE")
        return probs


def state_receipts(state_path: Path) -> List[tuple]:
    """(receipt_seq, source_sha256, received_ns) rows of a native state DB (read-only helper)."""
    con = sqlite3.connect(f"file:{Path(state_path)}?mode=ro", uri=True)
    try:
        return [tuple(r) for r in con.execute(
            "SELECT receipt_seq, source_sha256, received_ns FROM native_receipts ORDER BY receipt_seq")]
    finally:
        con.close()
