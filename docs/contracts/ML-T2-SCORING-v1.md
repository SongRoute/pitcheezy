# T2 complete-family scoring protocol v1 — registration draft

Implementation prepared; no T2 fit, DEV score or experiment result is claimed here.
Freeze this contract and the exact `registration.t2_scoring` object from
`scripts/score_ml_generalization.py:SCORING` in the T2 configuration **before
preparation**. The scorer rejects a missing or changed object. The parent G
analysis, candidate/control selection rule and diagnostic/promotion status remain
immutable inputs under ML-TRANSFER-EXECUTION-v1.

## Denominators and preprocessing

Natural new matchups reuse the selected candidate/control's frozen G predictions,
including their archived ensemble and seed June weights. Both identities must be
seen in actual parent TRAIN; the matchup must be absent from every supplied
pre-DEV historical, TRAIN, early-stop and CAL source. Only its first DEV game is
eligible. This is known-player matchup generalization, not strict zero-shot.

Pitcher and batter exclusion folds use the previously specified TRAIN hash
selectors, purge entire affected fitting PAs, and refit every auxiliary using the
sanitized fold. Original primitive truth determines evaluation eligibility.
Post-prefix evaluation rows are identical across Z/W/O, restricted to dates
strictly after the held-out entity's second DEV-game date. No scored held-out
outcome updates cumulative statistics. Z masks held-out query H5; W permits
observed earlier current-PA tokens; O additionally permits the fixed first-two-game
exposure outcomes in cumulative context. Pitcher profile/routing remains unknown
in all three regimes; O may therefore be degenerate, which is a reported result,
not grounds to change adaptation or replace players.

Each fold records actual eligible supervised TRAIN pitcher counts, seen pitcher
and batter flags, and observed TRAIN role. Its canonical volume labels use the
frozen original G q25/q75 thresholds (linear quantiles, original tie convention),
with absent pitchers assigned count zero and `zero`. Original selection counts,
roles, volume and seen flags remain separate `selection_*` fields. Natural-matchup
metadata uses actual parent TRAIN and the same frozen thresholds. No DEV statistic
selects or replaces a cohort.

## Complete-family and calibration gate

Before calculating any DEV quality metric, validate every active fold's three
seed fits, exact selected cells and model dependency union, checkpoint/report
hashes, prediction states and arrays. Match original keys, game, pitcher, batter
and truth across all regimes and controls; check both raw and calibrated
10-class probability mass. Delivery tiers must agree across regimes. Inactive
folds remain explicit and retain their hypothesis/bound slots.

There is one allowed-May temperature per model seed, frozen before Z/W/O
prediction. Fit one ensemble June blend and three ordered seed June blends per
cell, then reuse these four weights across Z/W/O. Do not reselect a blend from
held-out targets, fixed exposure labels, or each regime separately. Natural
predictions retain the already frozen parent weights.

## Six primary NLL hypotheses

Use the following exact order, including null slots for inactive or unreportable
cohorts:

1. Natural new matchup: selected candidate minus its frozen control.
2. Held-out batter Z: selected candidate minus control, both refitted on the same exclusion fold.
3. Held-out batter W: candidate minus control.
4. Held-out batter O: candidate minus control.
5. Held-out pitcher common-global W minus Z.
6. Held-out pitcher common-global O minus Z.

A cohort needs at least 30 games and 500 pitches before a primary test or
adaptation claim. Below threshold, report coverage and descriptive metrics,
retain a null p-value slot, and do not calculate primary uncertainty. Resample
whole paired games 10,000 times with seed 20260924, dividing each replicate's
summed losses by its sampled pitch count. Use null-centered one-sided improvement
p-values with plus-one correction and Holm alpha .05 over all six slots. This is
pitch-weighted inference, not an equal-game average.

A primary improvement requires mean delta NLL <= -.003, the paired two-sided
95% NLL interval's upper endpoint < 0, Holm-adjusted p <= .05, Brier interval upper
endpoint <= .001, and negative NLL differences in at least two of the three
matching seed pairs. Brier is the sum of squared probabilities-minus-indicators
over ten classes; NLL clips the chosen probability at 1e-12. Pitcher W/O successes
are **history adaptation improvements**, never architecture improvements.
Within-model batter W−Z/O−Z differences are descriptive, outside the six-test
confirmatory family, with no promotion decision.

## Robustness family

For natural and batter Z/W/O candidate/control comparisons, keep a fixed family
of 4 contrasts × 12 groups × 2 losses = 96 one-sided bounds. Groups are actual
starter/relief, L/R throwing hand, actual TRAIN-volume low/middle/high/zero,
two strikes/less than two, and runners on/bases empty. Each group needs 30 games
and 500 pitches. Use 100,000 paired whole-game bootstrap replicates, seed
20260924, upper quantile 1 − .05/96, and margins ΔNLL <= .010 and ΔBrier <= .002.
Do not renormalize the family when a group or fold is missing. Any measured failure
wins over missingness; otherwise any missing bound leaves robustness unconfirmed.
Pitcher adaptation has the overall Brier guard only; no architecture R claim.

## Interpretation and artifacts

Preserve the G-selected `screen_promoted` versus `diagnostic_only_not_promoted`
status. A T2 result does not retroactively promote a G diagnostic fallback.
This is a conditional follow-up on previously exposed Cpanel DEV, not independent
confirmation. Bootstrap intervals condition on fitted predictions and omit
training, calibration and selection uncertainty. Report cohort coverage, fold
selection/exposure metadata, inactive reasons, metrics, seed directions, model
costs, all six adjusted slots and all 96 robustness slots. The immutable analysis
manifest pins preparation, input predictions/checkpoints, and scoring source
hashes. No policy-effect claim follows from these prediction tests.

## 부록(2026-09-27, D61)

이 부록은 해석 주석이며 위 판정 규칙(재정규화 금지, 측정 실패 우선, 결측 시 unconfirmed)은 바꾸지 않는다. 자연 새 대진과 타자 제외 fold에서는 투수가 항상 TRAIN에 있으므로 robustness의 `volume_zero` 그룹은 설계상 채워질 수 없는 구조적 결측이다.
보고서는 이 경우를 다른 결측과 구분해 `unconfirmed(structural: volume_zero)`로 표기하고, zero-TRAIN 강건성은 온라인 TRAIN0 그룹이 실제로 있는 T4 전체 MLB에서 측정한다.
