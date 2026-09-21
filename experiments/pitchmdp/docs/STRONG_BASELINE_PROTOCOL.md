# Known-pitch-type frequency baselines

This protocol is fixed before CPU fitting/calibration of these baselines. DEV was
already inspected in earlier experiments; the comparison is exploratory. No
neural model is trained or run, and no original model or artifact is modified.

## Motivation and fixed information

The original count/hand baseline ignores the known current pitch type, whereas
the neural forecast's delivery distribution conditions on pitch type and pitcher.
Compare stronger frequency models given these same observed pre-pitch identifiers.
No batter identity, current physical realization, batted-ball feature or future
outcome is a predictor. Outcomes are used only to fit frequencies on TRAIN.

## Fixed hierarchy and sample sizes

Fit both structures on each of two frozen samples: the exact neural TRAIN sample
(275,775 rows) and the full eligible TRAIN pool (1,252,824 rows). Verify dynamic
counts and ordered row hashes; report both sample sizes explicitly. The full-pool
variants have a sample-size advantage and cannot isolate model architecture.

Use the unchanged ten-class outcome definition and these fixed levels:

1. Global frequencies with one pseudo-observation per class.
2. `(balls, strikes, stand, p_throws)`: 50 pseudo-observations from global.
3. `(pitch_type, balls, strikes, stand, p_throws)`: 50 from level 2.
4. Optional `(pitcher, pitch_type, balls, strikes, stand, p_throws)`: 100 from level 3.

Each child is `(observed_counts + strength * parent_probability) /
(observed_sample_size + strength)`. Model A ends at level 3; model B ends at level
4. Unseen groups use their parent's probability exactly. No smoothing-strength
search, feature search, batter-ID input or DEV fitting is allowed.

## Calibration and selection label

Use the original deterministic 4,000-row subset of May–June 2025 CAL. For each of
the four baselines, fit one scalar temperature on log probabilities by minimizing
CAL log loss over [0.5, 2.5]. Report untempered and tempered results regardless of
which is better. The selection label is the baseline with smallest **tempered CAL
log loss**, with stable declared order breaking exact ties; do not discard any
baseline or switch the label to a DEV winner. Persist this decision before DEV
prediction. This CAL set was used before and is not an independent validation set.

## Evaluation and common mechanical constraint

Use identical DEV pitch keys, labels and games as the original saved predictions:
7,276 eligible pitches across 87 games. Replay saved original count/hand and full
Transformer42 log loss and Brier directly from their frozen probabilities.

Primary results retain the original probability regime. Separately apply the
same post-hoc legality conditioning to every raw/tempered baseline and both saved
references: when outs=2 or bases=0, set double-play probability to zero and
renormalize the remaining nine classes. Refuse if an observed DP label contradicts
that rule. Do not use this projection to select models or temperatures. It is a
common mechanical correction, not new learned predictive information; improved
log loss under a valid impossible-event constraint is partly guaranteed.

Report log loss, multiclass Brier, all temperatures, hierarchy fallback counts,
and 2,000 paired whole-game bootstrap differences against both fixed references
for each raw/tempered and legality-conditioned regime. Negative model-minus-
reference differences favor that baseline. These intervals condition on fitting
and calibration and omit their uncertainty. No causal or policy benefit follows.

## Reproducibility

Read only the already processed approved 2023–2025 frame, verifying its SHA256
against current and archived original data manifests. Never access 2026 or new
seasons. Freeze source, protocol, config, ordered sample hashes and reference
artifact hashes before fitting. Save frequency tables, calibration selection,
all DEV probabilities and a completion record on the mounted T7 SSD. Do not
install packages or change existing frozen source files.
