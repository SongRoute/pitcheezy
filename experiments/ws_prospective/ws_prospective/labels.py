"""Registered label rule: GUMBO playEvent -> one of the 7 outcome leaves.

Mirrors the training label (seq_shadow/tokens.py outcome10 -> seven leaves;
whiff = swinging strike / swinging strike blocked; ground ball =
hitData.trajectory == 'ground_ball').  In-play pitches are labelled only when
the play's eventType is in the training WHITELIST (OUTCOMES[4:8] | _OUT | _DP of
tokens.py); every other eventType (fielders_choice, errors, interference,
triple play, unknown/new values) HOLDs -> None (LABEL_UNMAPPED), as do
automatic balls/strikes, unknown calls and in-play pitches without a
trajectory.  ``_play_event_type`` is attached by the label loader from the
newest stored snapshot of the game (same for both label revisions).
"""
from __future__ import annotations

from typing import Optional

BALL = {"B", "*B", "I", "P"}          # ball, ball in dirt, intent ball, pitchout
WHIFF = {"S", "W"}                    # swinging strike, swinging strike (blocked)
OTHER_STRIKE = {"C", "M", "T", "O"}   # called, missed bunt, foul tip, foul tip bunt
FOUL = {"F", "R"}                     # foul, foul pitchout
IN_PLAY = {"X", "D", "E"}
# Training whitelist, verbatim from seq_shadow/tokens.py (OUTCOMES[4:8], _OUT, _DP).
IN_PLAY_HITS = frozenset(("single", "double", "triple", "home_run"))
IN_PLAY_OUTS = frozenset(("field_out", "force_out", "fielders_choice_out", "sac_fly", "sac_bunt"))
IN_PLAY_DP = frozenset(("double_play", "grounded_into_double_play", "sac_fly_double_play",
                        "sac_bunt_double_play"))
IN_PLAY_WHITELIST = IN_PLAY_HITS | IN_PLAY_OUTS | IN_PLAY_DP
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
        if ev.get("_play_event_type") not in IN_PLAY_WHITELIST:
            return None
        traj = (ev.get("hitData") or {}).get("trajectory")
        if not traj:
            return None
        return "ground_ball_in_play" if traj == "ground_ball" else "other_in_play"
    return None
