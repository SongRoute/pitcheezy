# Game-day delivery adaptation: frozen exploratory protocol

This is a fixed-model diagnostic on previously inspected 2025 DEV, not unseen
confirmation, a novelty claim, a fatigue measurement, or a policy-value estimate.
It is planned before its calibration search or adapted DEV inference is run.

## Fixed inputs

Use the original full Transformer with model seed 42, width 128, its saved
normalizer and batter encoder, and its 25-draw joint TRAIN delivery pools.
Do not fit network weights. Reconstruct and hash-check the original ordered
TRAIN/calibration/DEV rows. Use only the three approved 2023–2025 raw files.
Never access 2026, add seasons, or install packages.

## Definition fixed before evaluation

Adapt standardized effective speed, release spin rate, horizontal movement and
vertical movement: physical channels 0, 1, 4, 5. Keep both spin-axis channels and
both plate-location channels unchanged. Keep all historical tokens, context,
pool memberships, pool draws, pitcher/type/hand/count conditioning unchanged.
The current observed pitch type is the same conditioned action as in the original
evaluation; its physical realization never enters its own candidate shift.

For each row, gather strictly earlier game/PA/pitch keys with the same game,
pitcher and pitch type. Include previous plate appearances and all logged pitches
regardless of outcome eligibility. A contributing delivery must have finite values
for all four adapted raw physical variables. Missing physical values are not
treated as observed imputed-median deliveries. Chronology is reconstructed from
unique game/PA/pitch keys, independent of input row order. Current and future rows
never contribute; each game/pitcher/pitch-type tuple has a separate count.

Estimate a four-channel reference mean from the full original eligible TRAIN pool,
not the subsampled network-training rows. Use the pitcher/type mean if at least
50 complete TRAIN deliveries exist, otherwise league/type mean, otherwise the
global eligible complete TRAIN mean. Hand and count are deliberately excluded
from this reference grouping to avoid thin in-game samples. Original delivery
pools continue to condition on those variables. Record every evaluated row's
reference origin and prior observation count.

For `n` earlier complete observations and prior mean `m`, reference mean `r`, use
`shift = (m - r) * n / (n + k)`. First observations have exactly zero shift.
Evaluate fixed `k = 10, 30, 100`, plus static `k = infinity`. Translate the four
channels of every existing joint candidate vector by that shift. Do not clip
shifts, alter the remaining channels, or select additional variables on DEV.
This preserves each pool's covariance but changes its mean. It does not guarantee
physical causality: in-game selection, measurement changes, opponent mix and
pitch subtype changes can also contribute to apparent mean shifts.

## Calibration and selection

Use exactly the original deterministic 4,000-row mixture-calibration subset
from May–June 2025. For each fixed setting, refit only a scalar mixture temperature
by bounded scalar minimization over [0.5, 2.5], matching the original method.
Select the setting with the smallest calibrated calibration log loss; exact ties
use the declared order static, k10, k30, k100. Save all calibration results and the
selected setting before any adapted DEV evaluation. The criterion has selection
optimism within CAL, and cannot itself establish an effect.

## Reporting

Report all four settings on the unchanged 87-game, 7,276-pitch DEV sample (use
verified dynamic counts if source metadata differ), with calibrated and
uncalibrated log loss and multiclass Brier, temperatures, prior counts, fallback
origins and shift magnitudes. Reproduce original saved static probabilities before
interpreting differences. Report 2,000 paired whole-game bootstrap log-loss
differences against static, holding the fitted checkpoint and CAL selection fixed.
Clearly identify the CAL-selected setting; do not switch to the DEV winner.

Bootstrap intervals omit model-training and hyperparameter-selection uncertainty.
Any apparent improvement remains exploratory until independently validated.
Ensemble or multi-seed follow-up is separate and should be justified after this
fixed single-model diagnostic. Source, protocol, data, ordered rows and reference
artifacts are frozen in the output before inference; originals remain unchanged.

## Prespecified 100-draw numerical sensitivity

The primary experiment keeps the archived 25-draw pools. After its CAL-only choice
of k is persisted, rebuild JointDelivery from the same full eligible TRAIN pool
(including the original normalizer's imputation policy) with 100 draws and seed 42
using the unchanged original implementation.
Compare static against the **same k selected at 25 draws**. Do not search k again
on 100-draw CAL or on DEV. Refit only each candidate's mixture temperature on the
same 4,000 CAL rows; persist both temperatures before 100-draw DEV evaluation.

Report calibrated/uncalibrated DEV metrics and paired whole-game bootstrap
differences at 100 draws, plus the saved pool hash and sampling configuration.
If CAL selected static in the primary experiment, report only static at 100 draws
and explicitly state that no adaptive contrast is available. Avoid duplicate
static predictions. This check probes dependence on a small sampled integration
pool; it does not exhaust numerical uncertainty or create an independent test set.
It is on by default (`--check-draws 100`); explicit `--check-draws 0` disables it
and must be disclosed. No original module or original checkpoint is modified.
