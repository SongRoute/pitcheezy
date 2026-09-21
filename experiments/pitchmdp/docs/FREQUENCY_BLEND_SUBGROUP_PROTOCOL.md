# Fixed subgroup diagnostic: primary blend versus selected frequency baseline

Freeze this protocol before calculating subgroup scores. Use only the saved
primary CAL-log-loss blend and its CAL-4,000-selected tempered frequency baseline.
Neither model, baseline, blend weight nor objective is selected again. The outcome
is prediction loss on already inspected DEV, not a policy or causal effect.

## Prespecified exhaustive partitions

- All six pitcher IDs from the original frozen cohort manifest, in saved order.
- Strike count 0, 1, 2 before the pitch.
- First logged pitch in a PA versus a pitch with earlier same-PA history, using
  the processed `prev_pitch_type == START` marker. Record any disagreement with
  `pitch_number == 1`; do not quietly change definitions after viewing results.

Retain every group, including zero-sample or unfavorable groups. Do not search
interactions, thresholds, alternative periods or additional slices. Report rows,
distinct games, model and baseline log loss and multiclass Brier, their difference,
and 2,000 paired whole-game bootstrap 95% intervals within each group. Negative
blend-minus-baseline values favor the blend. These are descriptive, unadjusted
intervals across multiple subgroup comparisons: no confirmatory subgroup claim,
winner selection or multiplicity-adjusted significance is implied. Models, baseline
choice and CAL weights are held fixed; their fitting uncertainty is omitted.

## Outcome decomposition

Report all ten fixed outcome classes, including empty classes, with sample count,
game count, prevalence, mean model/baseline log loss, and mean log-loss difference.
The class's contribution to overall difference is the sum of its per-pitch loss
differences divided by total DEV pitches, equivalently its prevalence times its
mean difference. Contributions must sum exactly to the overall loss difference.
This is a decomposition by observed outcome, not a deployable subgroup predictor
or evidence about conditional calibration for rare events. No outcome-class
screening or extra class-level significance tests are performed.

## Exact integrity checks and access

Read only saved predictions and the already processed approved 2023–2025 frame.
Verify the processed and prediction hashes. Join by exact game/PA/pitch keys,
verify labels/game IDs and reference overall metrics, and reject unmatched keys,
unknown cohort members, invalid strike counts or forbidden seasons. No raw or
2026 data access, neural inference, model fitting or package installation.

For every partition, verify mutual exclusivity, exhaustive coverage, and that
sample-weighted group model loss, baseline loss and difference reproduce overall
values to numerical precision. Freeze source/protocol/input hashes before scoring.
Save all groups, per-row group keys/loss differences, checks and completion record
in a distinct SSD directory. Preserve all original artifacts.
