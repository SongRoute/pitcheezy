# Batter representation and history length — frozen exploratory sweep

2026-09-21. User retains runners/outs and batter information; comparison is representation and history resolution, not removal of mandatory game state. This supersedes the unstarted group-removal proposal. No new model is selected using evaluation performance.

## Fixed experimental population and controls

Reuse the completed temporal-blend-20260921T050400Z folds, TRAIN-only top-six cohorts, exact TRAIN/earlystop/temperature/June-blend/evaluation keys, normalizer, shared400 joint delivery vectors, strong context-frequency predictions and full MLP reference forecasts. Verify pinned source/data/model/array hashes before reuse.2024 evaluation7737pitches/88games;2025 evaluation5048pitches/60games. Retain unavailable selected pitchers. Existing2023–2025 data only, existing Python/packages, mounted SSD, one neural fit/inference at a time. Existing source files and experiment artifacts remain unchanged.

All representations retain the same11 basic variables: ball/strike count, runners, outs, inning, defensive score lead, home/away, batter hand and pitcher hand. The neural model receives neither current observed physical realization at evaluation nor future outcomes. Current candidate physics is integrated over exactly the same TRAIN delivery pool. The frequency blend still retains runners/outs for all variants.

## Two controlled sweeps

1. Batter sweep at five PREVIOUS pitches: hand-only (no individual identity/profile beyond observed hand), categorical individual ID+hand, six continuous prior-date rates plus six reliability values, clusters-only atK=3,5,10,20. Shared full reference is original continuous12+softK5. All choices preserve mandatory game context. A hand-only predictor is a coarse batter representation requested as a control, not removal of runners/outs.
2. History sweep at fixed original continuous12+K5: previous1,2,3,4,5pitches, with5the reused reference. Each means at most that many previous pitches from the SAME plate appearance, plus a separate current candidate token. Keep the original six slots, flattened input dimensions and network parameter count; zero AND mask older positions. Real histories shorter than the window remain padded. This does not evaluate cross-PA history or interactions between representation and window.

Seven new batter variants and four new history variants ×two folds ×five seeds42–46 =110new MLP fits; full reference is reused, not refitted. Model width128, AdamW lr0.0005, batch1024,30epochs maximum, patience5, same conditional early stopping. Each new member receives its own integrated temperature on the original May16–31 target sample; each five-member ensemble receives its own convex mixture with the SAME frequency forecast on the original June sample. No change to dates, cohorts, sampling, integration, baseline, or tuning budget based on results.

## Representation safeguards

ID vocabulary is deterministic and fitted only from the common neural TRAIN sample. Reserve0for unknown; unknown embedding is fixed zero (padding_idx0). Embed categorical identity at16dimensions; never use numeric ID magnitude as a real-valued predictor. Report unseen-ID rates and evaluation scores separately, plus TRAIN exposure strata. ID-specific capacity is disclosed; this is a practical representation comparison, not equal-parameter attribution.

Cluster fitting uses full eligible TRAIN only, same prior-date six style rates/reliabilities and frozen algorithm seed42. K varies only within3,5,10,20. Clusters-only inputs are soft similarities already attenuated toward uniform for sparse history; continuous rates are not silently retained. All handedness channels remain separate. Continuous-only uses the original TRAIN-fitted standardization. Cluster labels have no assumed baseball semantics. Changes in context projection input width slightly change parameter count; report this explicitly.

## Evaluation and multiplicity

Primary score: held-out log loss; secondary Brier. Report every configuration and both years, not only a winner. Each variant minus full reference is a planned primary contrast: positive means full is better. Two families per year: seven batter contrasts and four history contrasts. Use2000shared paired whole-game bootstrap draws, pointwise95% intervals plus simultaneous95% max-absolute-centered-bootstrap intervals across all contrasts in that family. Claims of a representation/window effect use the simultaneous interval and require the same direction in both years. These intervals are conditional on fitted ensembles, temperatures and CAL blend weights; no full training/CAL-estimation uncertainty claim. Seed metrics are descriptive, not independent evaluation units.

Show adjacent K and adjacent history point differences descriptively without extra significance claims. No DEV-based K/window winner or combined optimum. Choosing a deployable optimum later requires a separate, predeclared selection stage; the June sample here is already used for blend fitting. This is an exploratory resolution sweep on historically inspected years, not a new untouched confirmation set and not evidence of real win-probability improvement.

Also report all-pitch known-ID/unseen-ID subgroup metrics and common-history support subsets (at least5previous pitches) descriptively, to distinguish available-information and cohort-composition effects. Zero-subgroup cells remain empty. Results are for original retrospectively eligible PA/pitch rows; eligibility and current-location availability limitations persist. Frozen sources, calibration logits, member predictions, models and final metrics are archived on SSD for independent verification.
