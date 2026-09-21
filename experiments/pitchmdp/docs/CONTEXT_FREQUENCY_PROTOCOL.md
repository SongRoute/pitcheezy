# Fair pre-pitch game-state frequency baselines and blending

Freeze this protocol before target baseline fitting or calibration. The motivation
is a descriptive prior result: double plays contributed substantially to the
primary neural/type blend's gain, while its frequency parent omitted outs/bases.
That observation does not prove a mechanism; this experiment directly tests a
stronger fixed game-state baseline on previously inspected DEV. It is exploratory,
not an independent confirmation, novelty claim or policy-value estimate.

## Fixed three baseline candidates

Keep the existing full-TRAIN league/type baseline (`full_train__type`) and its
already fitted temperature. Restore its portable tables without modifying them.
Fit exactly two new candidates on the same full eligible TRAIN pool, verifying
the original ordered-row hash and 1,252,824-row count:

1. League game-state frequencies keyed by `(pitch_type, balls, strikes, stand,
   p_throws, outs_when_up, bases)`, where outs are 0–2 and base occupancy is 0–7.
   Smooth each child with 100 pseudo-observations from the frozen type/count/hand
   parent. Unseen context keys back off exactly to that parent.
2. Pitcher game-state frequencies prepend `pitcher` to the preceding key and
   smooth with 100 pseudo-observations from the league game-state parent.
   Unseen pitcher/context keys back off exactly to the league context parent.

The original global/count/type levels and strengths remain unchanged. No batter
ID, realized current physics, future state, or outcome-derived predictor is added.
The only added predictors are known pre-pitch outs and base occupancy, plus the
already known pitcher identifier in candidate 2. No smoothing/feature search.
All three baselines are reported, including unfavorable outcomes.

## Two-stage CAL-only selection and blending

Use the original deterministic 4,000 CAL rows to fit each new candidate's scalar
temperature over [0.5, 2.5]. Retain raw and tempered probabilities. Select the
lowest tempered CAL log loss among the old type baseline and the two new models;
declared order old, league-context, pitcher-context breaks exact ties. Persist
baseline selection before any target DEV prediction.

On the disjoint remaining 12,000 original CAL rows, blend the selected tempered
baseline with the saved fixed five-seed Transformer probability ensemble.
Fit primary log-loss and secondary Brier convex weights independently using the
unchanged bounded method with exact endpoints; save both before opening DEV
probability archives. Do not select the better DEV objective. No neural inference
or training occurs. All 16,000 CAL rows previously informed neural early stopping,
so neither selection stage is independent validation.

## Evaluation and support diagnostic

Preserve original DEV pitch keys, outcomes and 87 games / 7,276 pitches. Report
all three baselines raw and tempered, both selected-baseline ensemble blends,
the old primary/secondary type blends, old type baseline and unblended ensemble.
Measure LL and multiclass Brier. Compute 2,000 paired whole-game bootstrap
differences for both raw/tempered baselines and both new blends against the old
primary type blend, old tempered type baseline and unblended ensemble.

Separately apply the same post-hoc legality conditioning to every predictor:
when outs=2 or bases=0, set double-play probability to zero and renormalize.
Refuse contradictory observed labels. Report conditioned scores and paired
comparisons without using this mechanical step in calibration or selection.
Frequency smoothing may still assign some impossible mass even with state keys;
conditioning removes it mechanically and is not newly learned predictive skill.

All intervals are game-only conditional on fitted tables, fixed ensemble, chosen
baseline and CAL weights. They omit training/calibration/selection uncertainty.
Do not suppress a candidate or claim universal superiority from DEV rankings.

## Integrity

Read only the hash-verified existing 2023–2025 processed frame and saved probability
archives. Never open 2026 or new raw seasons. Check exact TRAIN/CAL/DEV ordered
keys, same full TRAIN pool, the disjoint 4,000/12,000 CAL partition, labels and
saved reference replay. Test the fixed child smoothing and unseen-key backoff on
synthetic data. Freeze this protocol/source/config/reference hashes before counts
are fitted. Save portable tables, selections, CAL/DEV predictions, scores and a
completion record in a new SSD directory; originals remain unchanged.
