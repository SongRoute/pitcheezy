"""Registered label rule: GUMBO playEvent -> one of the 7 outcome leaves.

Mirrors the training label (outcome10 -> seven leaves; whiff = swinging
strike / swinging strike blocked; ground ball = hitData.trajectory ==
'ground_ball').  Anything the training fate rule would abstain on returns
None (LABEL_UNMAPPED): automatic balls/strikes, unknown calls, in-play pitches
without a trajectory, and in-play plays whose result is non-core (error,
catcher interference, triple play).  ``_play_event_type`` is attached by the
label loader from the final snapshot of the game.
"""
from __future__ import annotations

from typing import Optional

BALL = {"B", "*B", "I", "P"}          # ball, ball in dirt, intent ball, pitchout
WHIFF = {"S", "W"}                    # swinging strike, swinging strike (blocked)
OTHER_STRIKE = {"C", "M", "T", "O"}   # called, missed bunt, foul tip, foul tip bunt
FOUL = {"F", "R"}                     # foul, foul pitchout
IN_PLAY = {"X", "D", "E"}
NON_CORE_PLAY = {"field_error", "catcher_interf", "triple_play", "error"}
AUTOMATIC = {"VB", "AB", "AC", "VS", "V"}  # automatic ball / strike (clock, ABS)


def type_of(ev: dict) -> Optional[str]:
    return ((ev.get("details") or {}).get("type") or {}).get("code")


def leaf_of(ev: dict) -> Optional[str]:
    d = ev.get("details") or {}
    code = (d.get("call") or {}).get("code") or d.get("code")
    if code is None or code in AUTOMATIC:
        return None
    if code in BALL:
        return "ball"
    if code in WHIFF:
        return "narrow_whiff"
    if code in OTHER_STRIKE:
        return "other_strike"
    if code == "L":  # foul bunt: strike three counts as a strike, else foul
        return "other_strike" if (ev.get("count") or {}).get("strikes") == 3 else "foul"
    if code in FOUL:
        return "foul"
    if code == "H":
        return "hit_by_pitch"
    if code in IN_PLAY:
        if ev.get("_play_event_type") in NON_CORE_PLAY:
            return None
        traj = (ev.get("hitData") or {}).get("trajectory")
        if not traj:
            return None
        return "ground_ball_in_play" if traj == "ground_ball" else "other_in_play"
    return None
