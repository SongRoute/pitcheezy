"""Pre-pitch context: the only object a forecaster ever receives.

A ``PrePitchContext`` for slot n is built from the accepted pitches with
ordinal < n (as shown in the newest stored snapshot at forecast time, which is
still before pitch n is stored) plus the
pre-pitch game state that the store published for slot n.  It deliberately has
no field for pitch n's identity, result, type, location or velocity, and
``build_context`` truncates the history at the slot, so even a catch-up
(post-hoc) call cannot leak current-pitch information.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence, Tuple

from .ledger import canonical_sha256


@dataclass(frozen=True)
class StorePitch:
    """One accepted pitch as exposed by the read-only source adapter."""

    pitch_uid: str            # stable pitch identity from the store
    ordinal: int              # 1-based game pitch ordinal in acceptance order
    at_bat_index: int
    pitch_number: int         # pitch number within the plate appearance
    pitcher_id: int
    batter_id: int
    pitch_type: Optional[str]  # raw code as first accepted (label candidate)
    balls_before: int
    strikes_before: int
    outs_before: int
    inning: int
    top_inning: bool
    first_receipt_wall_ns: Optional[int]  # store's first receipt of this pitch
    accepted_wall_ns: Optional[int]
    receipt_clock: Optional[str] = None       # 'observer_local' only for live_poll
    provider_event_time: Optional[str] = None  # playEvent startTime (MLB clock)
    extra: Tuple[Tuple[str, object], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class PreState:
    """Pre-pitch state of the *next* slot, as published before the pitch."""

    at_bat_index: int
    pitch_number: int
    pitcher_id: int
    batter_id: int
    balls: int
    strikes: int
    outs: int
    inning: int
    top_inning: bool
    on_1b: Optional[bool] = None
    on_2b: Optional[bool] = None
    on_3b: Optional[bool] = None
    published_wall_ns: Optional[int] = None


@dataclass(frozen=True)
class PastPitch:
    at_bat_index: int
    pitch_number: int
    pitcher_id: int
    batter_id: int
    pitch_type: Optional[str]
    balls_before: int
    strikes_before: int


@dataclass(frozen=True)
class PrePitchContext:
    game_pk: int
    slot_ordinal: int                 # n
    history: Tuple[PastPitch, ...]    # pitches 1..n-1 only
    state: PreState                   # state before pitch n
    profile_sha256: str               # frozen per-game profile
    asof_pitch_uid: Optional[str]     # pitch n-1 uid (None at game start)
    asof_receipt_wall_ns: Optional[int]
    source_seq: Optional[int] = None   # store snapshots.seq the state came from
    source_sha: Optional[str] = None   # snapshots.sha (canonical body hash)
    source_fetched_ns: Optional[int] = None  # snapshots.fetched of that body (as-of time)

    def sha256(self) -> str:
        return canonical_sha256(asdict(self))


def slot_key(asof_pitch_uid: Optional[str]) -> str:
    """A slot is 'the pitch after X'; corrections that insert a missed pitch
    therefore produce a different slot instead of silently re-labelling."""
    return f"after:{asof_pitch_uid}" if asof_pitch_uid else "game_start"


def build_context(game_pk: int, accepted: Sequence[StorePitch], slot_ordinal: int,
                  state: PreState, profile_sha256: str, *, source_seq: Optional[int] = None,
                  source_sha: Optional[str] = None,
                  source_fetched_ns: Optional[int] = None) -> PrePitchContext:
    prior = [p for p in accepted if p.ordinal < slot_ordinal]
    prior.sort(key=lambda p: p.ordinal)
    if [p.ordinal for p in prior] != list(range(1, slot_ordinal)):
        raise ValueError("history has gaps before the slot")
    last = prior[-1] if prior else None
    return PrePitchContext(
        game_pk=int(game_pk), slot_ordinal=int(slot_ordinal),
        history=tuple(PastPitch(p.at_bat_index, p.pitch_number, p.pitcher_id, p.batter_id,
                                p.pitch_type, p.balls_before, p.strikes_before)
                      for p in prior),
        state=state, profile_sha256=profile_sha256,
        asof_pitch_uid=last.pitch_uid if last else None,
        asof_receipt_wall_ns=last.first_receipt_wall_ns if last else None,
        source_seq=source_seq, source_sha=source_sha, source_fetched_ns=source_fetched_ns)
