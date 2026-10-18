"""Per-game prospective shadow runner.

Loop (one process per game, CPU only, read-only input):

1. poll the read-only source -> accepted pitches 1..m and, if published, the
   pre-pitch state of slot m+1;
2. record first receipts of newly seen pitches (metadata only);
3. for slot n = m+1, if its pre-pitch state is published and no intent exists
   for (slot, model): write the intent, call the frozen forecaster with a
   ``PrePitchContext`` truncated at n-1, poll the source once more and mark
   the forecast PRE_RECEIPT only if pitch n is still absent, else POST_HOC_REPLAY;
4. optionally back-fill slots <= m that have no intent (late start / restart)
   as POST_HOC_REPLAY - recorded for coverage, never part of the primary
   denominator.

The runner never reads labels for metrics, never writes to the source, and
aborts itself (no signals to anyone else) on resource-limit breaches.
"""
from __future__ import annotations

import resource
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence

from .context import PreState, PrePitchContext, StorePitch, build_context, slot_key
from .ledger import Ledger


class Abstain(Exception):
    """Raised by a forecaster that cannot score this context (reason = str)."""


# Registered abstain vocabulary (v2): exact codes, or a registered family prefix
# followed by the accepted pipeline's own reason token.  Free text (exception
# messages) is cut to the family + exception type so no data reaches the ledger.
ABSTAIN_CODES = frozenset((
    "NO_SOURCE_BODY", "SOURCE_BODY_ABSENT", "SOURCE_BODY_SHA_INVALID", "SOURCE_BODY_OTHER_GAME",
    "SOURCE_SHA_MISMATCH", "BODY_CONTAINS_TARGET", "CONTEXT_DISAGREES", "CONTEXT_STATE_MISMATCH",
    "NO_REGISTERED_CANDIDATE_SCORED", "NO_CANDIDATE_TYPE", "MODEL_ABSTAIN"))
ABSTAIN_FAMILIES = ("bridge_", "native_", "bad_schema_or_hash:", "missing_input:",
                    "missing_profile:", "unsupported_player:", "no_supported_action:",
                    "predictor_error:", "seq_predictor_error:")


def sanitize_reason(reason: str) -> str:
    import re
    r = (reason or "MODEL_ABSTAIN").strip()
    if r in ABSTAIN_CODES:
        return r
    for fam in ABSTAIN_FAMILIES:
        if r.startswith(fam):
            if fam in ("predictor_error:", "seq_predictor_error:"):
                return ":".join(r.split(":")[:3 if fam == "predictor_error:" else 2])[:120]
            return re.sub(r"[^A-Za-z0-9_:.\-]", "_", r)[:120]
    return "UNLISTED_REASON"


class Forecaster(Protocol):
    """Frozen model wrapper.

    ``predict`` returns {candidate_pitch_type: probs over leaf_order}; models
    that do not condition on pitch type return {"*": probs}.  All candidates
    are produced before the pitch, so grading the realized type's vector after
    the seal uses no current-pitch information at forecast time.
    """

    model_id: str
    leaf_order: Sequence[str]

    def predict(self, ctx: PrePitchContext) -> Dict[str, Sequence[float]]: ...


@dataclass
class SourceSnapshot:
    accepted: List[StorePitch]
    next_state: Optional[PreState]
    game_final: bool = False
    source_seq: Optional[int] = None
    source_sha: Optional[str] = None
    source_fetched_ns: Optional[int] = None


class Source(Protocol):
    def poll(self) -> SourceSnapshot: ...


@dataclass
class Limits:
    max_rss_mb: float = 1024.0
    max_compute_ms: float = 2000.0
    poll_interval_s: float = 1.0
    record_post_hoc_replays: bool = False  # registered default: off
    store_probabilities: bool = True       # dry run: False -> status kept, vectors never written


class ResourceAbort(RuntimeError):
    pass


def _rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 * 1024) if sys.platform == "darwin" else r / 1024


@dataclass
class GameRunner:
    game_pk: int
    source: Source
    forecasters: Sequence[Forecaster]
    ledger: Ledger
    profile_sha256: str
    limits: Limits = field(default_factory=Limits)

    def _state_matches_slot(self, accepted: Sequence[StorePitch], st: PreState) -> bool:
        if not accepted:
            return st.pitch_number == 1
        last = accepted[-1]
        if st.at_bat_index == last.at_bat_index:
            return st.pitch_number == last.pitch_number + 1
        return st.at_bat_index > last.at_bat_index and st.pitch_number == 1

    @staticmethod
    def _pre_state_of(p: StorePitch) -> PreState:
        """Pre-pitch fields of a stored pitch (used only for post-hoc replays)."""
        return PreState(p.at_bat_index, p.pitch_number, p.pitcher_id, p.batter_id,
                        p.balls_before, p.strikes_before, p.outs_before, p.inning,
                        p.top_inning)

    def _forecast(self, accepted: Sequence[StorePitch], n: int, state: PreState,
                  fc: Forecaster, *, post_hoc: bool, snap: Optional[SourceSnapshot] = None) -> None:
        # post-hoc replays never get the source body: it already contains pitch n
        ctx = build_context(self.game_pk, accepted, n, state, self.profile_sha256,
                            source_seq=None if post_hoc or snap is None else snap.source_seq,
                            source_sha=None if post_hoc or snap is None else snap.source_sha,
                            source_fetched_ns=None if post_hoc or snap is None
                            else snap.source_fetched_ns)
        key = slot_key(ctx.asof_pitch_uid)
        prev = accepted[n - 2] if n >= 2 and len(accepted) >= n - 1 else None
        mid_pa_sub = int(prev is not None and prev.at_bat_index == state.at_bat_index
                         and (prev.pitcher_id != state.pitcher_id or prev.batter_id != state.batter_id))
        iid = self.ledger.write_intent(
            target_pitcher_id=state.pitcher_id, target_batter_id=state.batter_id,
            mid_pa_substitution=mid_pa_sub,
            slot_key=key, slot_ordinal=n, target_at_bat_number=state.at_bat_index,
            target_pitch_number=state.pitch_number, model_id=fc.model_id,
            context_asof_pitch_uid=ctx.asof_pitch_uid,
            context_asof_receipt_wall_ns=ctx.asof_receipt_wall_ns,
            context_sha256=ctx.sha256(), context_source_seq=ctx.source_seq,
            context_source_fetched_ns=ctx.source_fetched_ns)
        if iid is None:  # first write wins (restart-safe)
            return
        clk = self.ledger.clock
        t0 = clk.mono_ns()
        probs, status, reason = None, "OK", None
        try:
            raw = fc.predict(ctx)
            probs = {str(k): [float(x) for x in v] for k, v in dict(raw).items()}
            for v in probs.values():
                if len(v) != len(fc.leaf_order) or abs(sum(v) - 1.0) > 1e-6 or min(v) < 0:
                    raise ValueError("invalid probability vector")
        except Abstain as e:
            probs, status, reason = None, "ABSTAIN", sanitize_reason(str(e))
        except Exception as e:  # noqa: BLE001 - recorded, never fatal for the game
            probs, status, reason = None, "ERROR", type(e).__name__  # no message text (may carry data)
        t1 = clk.mono_ns()
        gen_wall = clk.wall_ns()
        timing = "POST_HOC_REPLAY" if post_hoc else "PRE_RECEIPT"
        if not post_hoc:
            after = self.source.poll()
            if any(p.ordinal >= n for p in after.accepted):
                timing = "POST_HOC_REPLAY"
        self.ledger.write_forecast(
            iid, status=status, leaf_order=fc.leaf_order,
            probs=probs if self.limits.store_probabilities else None, abstain_reason=reason,
            generated_wall_ns=gen_wall, generated_mono_ns=t1, compute_ns=t1 - t0,
            provisional_timing=timing)
        if (t1 - t0) / 1e6 > self.limits.max_compute_ms:
            self.ledger.event("SLOW_FORECAST", model_id=fc.model_id, slot=n,
                              compute_ms=(t1 - t0) / 1e6)

    def step(self) -> SourceSnapshot:
        snap = self.source.poll()
        accepted = sorted(snap.accepted, key=lambda p: p.ordinal)
        for p in accepted:
            self.ledger.record_receipt(p.pitch_uid, p.first_receipt_wall_ns)
        m = len(accepted)
        if [p.ordinal for p in accepted] != list(range(1, m + 1)):
            self.ledger.event("CONTEXT_GAP", ordinals=[p.ordinal for p in accepted][-5:])
            return snap
        if self.limits.record_post_hoc_replays:
            for k in range(1, m + 1):
                for fc in self.forecasters:
                    asof = accepted[k - 2].pitch_uid if k > 1 else None
                    tgt = (accepted[k - 1].at_bat_index, accepted[k - 1].pitch_number)
                    if not self.ledger.has_intent(slot_key(asof), fc.model_id, tgt):
                        self._forecast(accepted, k, self._pre_state_of(accepted[k - 1]), fc,
                                       post_hoc=True)
        st = snap.next_state
        if st is not None and not snap.game_final:
            if self._state_matches_slot(accepted, st):
                for fc in self.forecasters:
                    self._forecast(accepted, m + 1, st, fc, post_hoc=False, snap=snap)
            else:
                self.ledger.event("STATE_MISMATCH", slot=m + 1,
                                  at_bat_index=st.at_bat_index, pitch_number=st.pitch_number)
        rss = _rss_mb()
        if rss > self.limits.max_rss_mb:
            self.ledger.event("RESOURCE_ABORT", rss_mb=rss, limit_mb=self.limits.max_rss_mb)
            raise ResourceAbort(f"RSS {rss:.0f} MB > {self.limits.max_rss_mb} MB")
        return snap

    def run(self, *, max_polls: Optional[int] = None, sleep=time.sleep) -> None:
        self.ledger.event("RUN_START", models=[f.model_id for f in self.forecasters])
        polls = 0
        try:
            while True:
                snap = self.step()
                polls += 1
                if snap.game_final:
                    self.ledger.event("GAME_FINAL", n_accepted=len(snap.accepted))
                    return
                if max_polls is not None and polls >= max_polls:
                    return
                sleep(self.limits.poll_interval_s)
        finally:
            self.ledger.event("RUN_STOP", polls=polls, peak_rss_mb=round(_rss_mb(), 1))
