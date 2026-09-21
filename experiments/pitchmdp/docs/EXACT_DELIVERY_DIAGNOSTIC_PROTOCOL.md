# Frozen fixed-temperature empirical integration diagnostic

This is the final bounded numerical diagnostic of the current exploratory batch.
It is frozen before inference. It does not retrain, recalibrate, choose a model or
delivery budget using DEV, or create independent validation evidence.

Use the original full Transformer seed-42 neural weights. Freeze one common
mixture temperature to the already saved 400-draw CAL fit from
`integration100-20260921T033810Z/calibration400.json`. Do not infer on any CAL rows
or fit any temperature/weight in this diagnostic. Predict only the original
7,276 DEV rows under four empirical delivery integration regimes:

1. Original saved 25-draw TRAIN pool.
2. Saved 100-draw TRAIN pool used by the completed reevaluation.
3. Saved 400-draw TRAIN pool.
4. Exact uniform summation over every eligible TRAIN delivery in each selected
   finite pool, with the original tier priorities and support thresholds.

The exact pools use priority pitcher/type/hand/count, pitcher/type/hand,
type/hand/count, type/hand; minimum support is 20 for count tiers and 50 otherwise.
Eligibility, normalization, imputation, physical histories, context and ordered
samples remain unchanged. The verified DEV profile contains 1,330,692 exact
candidate pairs and no global fallback; assert these properties against the
saved profiling artifact. Do not sample, truncate or cap the exact branch.

Each prediction uses observed preceding history plus a hypothetical TRAIN current
delivery, never the logged current or future realization. Batch ragged candidate
pairs, retain their logits and segment offsets, and average candidate softmax
probabilities within each query. Use float64 probability aggregation for all four
regimes. Report both raw-temperature-one and fixed-common-temperature probabilities.

Save complete exact candidate membership identities (TRAIN frame positions and
game/PA/pitch keys), query keys, pool tiers/keys, uniform weights and row offsets.
Save exact ragged DEV logits, and all four regimes' raw/fixed-temperature
probabilities. The sampled pools remain the already saved pools, including their
original repeated samples where applicable.

Report DEV log loss/Brier for all regimes, paired game-bootstrap fixed-temperature
differences against exact integration, and per-query probability L1 distance to
the exact finite-pool result (mean and quantiles). Include the raw counterparts
as descriptive diagnostics. Retain the prescribed temperature even if another
regime's score is better. The pools are not nested, and no pool-resampling
uncertainty is estimated.

Exact here means removal of finite Monte Carlo sampling error relative to the
specified eligible-TRAIN empirical pool, subject to floating-point arithmetic.
It does not make that historical pool the true population delivery distribution
and does not resolve control, temporal drift, model or causal-policy errors.

Freeze protocol, code, reference/data identities before target inference. Verify
all neural tensors and source/reference identities at completion. Keep artifacts
on the configured SSD and stop after this diagnostic.
