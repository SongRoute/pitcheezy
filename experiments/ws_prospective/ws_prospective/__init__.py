"""RS-01 prospective shadow evaluation for the 2026 World Series.

Preparation-only package. Nothing here talks to MLB, to the live service, or
to real 2026 rows by itself; the runner consumes a read-only *source* object
and frozen *forecaster* objects that are injected by the caller. Tests use
synthetic sources and fake forecasters only.
"""

# Outcome leaves of the accepted fullseason models (seq_shadow.fullseason.LEAVES).
LEAVES = ("ball", "foul", "hit_by_pitch", "narrow_whiff", "other_strike",
          "ground_ball_in_play", "other_in_play")
