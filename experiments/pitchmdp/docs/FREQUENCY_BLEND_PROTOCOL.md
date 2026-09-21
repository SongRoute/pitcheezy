# Five-model ensemble blended with a stronger frequency baseline

This CPU-only follow-up consumes saved predictions, not neural checkpoints.
The protocol is fixed before blend-weight fitting. DEV has already been inspected,
so this is exploratory, not unseen confirmation or evidence of policy benefit.
Original runners, checkpoints, probabilities and frozen protocols remain unchanged.

## Baseline choice: original 4,000 CAL rows only

Use the winner already recorded by the four-frequency-baseline experiment using
tempered log loss on its original 4,000 CAL rows. At protocol writing this is
`full_train__type`. Verify that the saved choice is the minimum of the recorded
four CAL scores, and freeze hashes of all four portable frequency tables and
their selection record. Do not choose a different baseline using the remaining
CAL rows or DEV. Restore the selected tables without refitting them. Keep its
existing scalar temperature fitted on those original 4,000 CAL rows.

## Fixed ensemble and separate weight-fitting subset

Use the saved arithmetic probability mean of full Transformers with seeds
42, 43, 44, 45, 46 from the root ensemble/count-blend experiment. Do not refit
models, pick seeds, run neural inference, or replace the ensemble.

Reconstruct and verify the original 16,000 ordered CAL rows and deterministic
4,000-row temperature subset. Fit weights only on the remaining 12,000 rows,
checking exact pitch keys, labels and game IDs against the saved ensemble CAL
archive. These 12,000 rows are disjoint from the temperature/baseline-choice subset,
but all 16,000 previously informed early stopping, so they are not independent
validation data.

## Two prespecified convex blends

For `p = w * ensemble + (1-w) * chosen_tempered_frequency`, constrain `w` to [0,1].
The primary weight minimizes CAL log loss; the secondary minimizes CAL multiclass
Brier. Use the unchanged root `fit_blend` method: bounded scalar optimization plus
both exact endpoints as candidates. Keep both objective-specific results; do not
choose the better DEV objective or discard an endpoint solution. Persist the two
weights and CAL row hashes before opening DEV prediction archives.

## Evaluation

Apply the frozen weights to the unchanged 7,276-pitch / 87-game DEV sample, using
verified dynamic counts and keys. Reproduce the selected frequency baseline's saved
tempered DEV probabilities exactly. Verify that saved ensemble arrays are the mean
of all five saved seed arrays. Report both strong-baseline blends, unblended ensemble,
selected frequency baseline, old count baseline, and both original ensemble/count
blends on identical rows. Preserve the original unconditioned probability regime
as primary. Also report a separate common post-hoc DP-legality diagnostic for
all seven predictors: when outs=2 or bases=0, remove DP mass and renormalize.
Check that no observed DP contradicts this rule. Never use this mechanical
conditioning to fit weights or choose models; some log-loss gain is guaranteed
under a valid impossible-event constraint and is not learned predictive skill.

For each new blend, report log loss and multiclass Brier, with 2,000 paired
whole-game bootstrap differences against the full ensemble, selected frequency
baseline, and both original count blends. Negative model-minus-reference differences
favor the new blend. Intervals condition on fitted models, selected baseline and
CAL weights; they omit fitting/selection uncertainty. Use explicitly game-only
interval metadata, without describing the probability ensemble as a one-seed
model. Also report ensemble/count-blend comparisons against the selected frequency
baseline with the same LL/Brier intervals in both probability regimes.
Retain descriptive metrics and both new blends' paired comparisons against all
four fixed frequency baselines, without reselecting a baseline or blend. The
original CAL-log-loss choice need not be the strongest baseline for Brier.
Report all results regardless
of whether the stronger-baseline blend helps.

## Reproducibility and access

Require completed root ensemble and frequency runs. Hash their configs, source
archives, saved probabilities and selected tables. Read only the already processed
approved 2023–2025 frame, verifying its frozen hash; never access 2026 or new seasons.
Freeze this protocol and new source code before weight fitting. Save calibration
probabilities, selection, DEV probabilities, metrics, paired comparisons and a
completion record in a new SSD directory. No package changes or MPS execution.
