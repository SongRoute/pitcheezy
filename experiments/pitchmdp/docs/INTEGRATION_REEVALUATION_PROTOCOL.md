# Frozen delivery-integration reevaluation

This exploratory protocol is fixed before its target inference. The trigger was
a previously observed change in Transformer-42 DEV log loss from 1.518155 at 25
TRAIN delivery draws to 1.506911 at 100 draws, without changing neural weights.
The same previously inspected DEV is reused; this is not unseen confirmation.

Reevaluate all 30 existing robustness checkpoints: six variants (full Transformer,
flattened MLP, no game context, no batter style, capacity MLP, no clusters) and
seeds 42–46. Do not retrain, change masks, select seeds or alter ordered samples.
Use the identical saved 100-draw TRAIN pool from the completed delivery-adaptation
run for every checkpoint. Refit each mixture temperature on the original 4,000
CAL rows only, using the existing bounded temperature optimizer. Save that fit
before evaluating its 7,276 DEV rows. Retain uncalibrated and calibrated integrated
probabilities; unchanged conditional probabilities are copied from their verified
original archives rather than unnecessarily rerun.

Use the existing matched-seed/game comparison functions for equal-seed mean
losses, never silently average probabilities before scoring that estimand. Keep
the original five architecture/context contrasts and all variants. Report the
100-versus-25 change separately for each variant. These comparisons concern fixed
fitted models and a fixed empirical delivery pool; the bootstrap does not measure
delivery-pool Monte Carlo uncertainty or unseen-sample performance.

For all five full Transformers, also predict the original 12,000 CAL blend rows,
which exclude the 4,000 temperature rows. Form their arithmetic probability
ensemble on CAL and DEV, then refit the previously defined count/hand convex
blends for log loss and Brier on CAL only. Save archives and JSON with the original
calibration-run schema so stronger frequency/context baselines can consume them.
All 16,000 CAL rows influenced early stopping originally, so this split does not
create independent validation. Keep every 25-draw result unchanged.

Finally perform a prespecified 400-draw check for full Transformer seed 42 only.
Fit a fresh static JointDelivery pool on the same full eligible TRAIN data with
seed 42 and 400 draws; refit mixture temperature on the same CAL 4,000, then
evaluate the same DEV rows. Report 25, 100 and 400 side by side with raw and
calibrated scores. Do not choose a draw count from DEV performance. Pools at these
draw counts are not nested, so changes combine integration budget and empirical
draw realization. A single such sequence is a convergence diagnostic, not a
formal estimate of integration error or proof of convergence.

Freeze code, this protocol, configuration, source checkpoints, 100-draw pool and
reference hashes before neural inference. Store new calibrated checkpoint copies
without changing original files; neural tensor hashes must remain identical.
Write results incrementally, store ordered row hashes and predictions, and verify
source/reference identity at completion. Use the existing Python environment and
mounted configured SSD. Only approved 2023–2025 sources are read; no 2026 data.
