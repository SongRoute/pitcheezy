# Provisional interpretation of predictive and policy follow-ups

Historical 25-draw snapshot. The completed 100-draw, calibrated context-blend and
exact empirical-pool results supersede its preliminary conclusions; see
[FOLLOWUP_RESULTS.md](../FOLLOWUP_RESULTS.md).

This snapshot combines the completed five-seed replication, stronger frequency
baselines, and six-case control sensitivity. Capacity/cluster attribution and the
actual-history/terminal-rollout outputs are not adjudicated here. All predictive
comparisons use the previously inspected 2025 DEV cohort: 7,276 pitches in 87
games. These are exploratory follow-ups, not unseen confirmatory evidence.

The evidence supports a modest Transformer log-loss advantage over the tested
flattened MLP and predictive value from game context and prior batter profiles.
It does not yet establish superiority over a strong simple forecast, reliable
target rankings, or improved real pitching decisions. Those are separate claims.

## Predictive findings and their scope

Lower log loss and multiclass Brier are better. Differences below are the first
named model minus the second. The five-seed estimand averages single-model losses;
it is not an ensemble of predicted probabilities.

| Comparison | Log-loss difference | 95% interval | Interpretation |
|---|---:|---|---|
| Full Transformer minus flattened MLP, five seeds | -0.003601 | [-0.006511, -0.000546] | Crossed seed/game bootstrap favors Transformer on this setup |
| No game context minus full Transformer, five seeds | +0.009432 | [+0.005976, +0.013119] | Removing the game feature block hurts log loss |
| No batter style minus full Transformer, five seeds | +0.005065 | [+0.001965, +0.008220] | Removing the entire prior-profile/style block hurts log loss |
| CAL-chosen full-TRAIN type-frequency baseline minus Transformer seed 42 | -0.001337 | [-0.007663, +0.004849] | Game-bootstrap log-loss difference is unresolved |

The five-seed Transformer-minus-MLP Brier difference is -0.000863 with crossed
interval [-0.001992, +0.000360], so the log-loss conclusion should not be generalized
to a clear advantage on every scoring rule. The game ablation's Brier interval
also crosses zero. The style ablation's Brier difference is +0.001309 with interval
[+0.000040, +0.002512]. These ablations establish effects of whole input blocks;
they do not isolate the five soft cluster memberships from the underlying rates
and reliability features, or separate architectural effects from capacity.

The stronger baseline was chosen by smallest tempered CAL log loss among four
predefined frequency models. Its DEV log loss is 1.516818 versus Transformer-42's
1.518155. Its Brier is 0.729618 versus 0.734177: baseline-minus-Transformer difference
-0.004559, game-bootstrap interval [-0.007232, -0.002074]. The baseline uses count,
pitch type and handedness; it is a relevant known-action forecast comparison.
The full-TRAIN fit uses 1,252,824 rows versus the neural sample's 275,775, so that
particular comparison mixes model and training-pool size. The matched-sample type
baseline still has better Brier (0.729673; difference -0.004503, interval
[-0.007006, -0.002041]), while its log-loss difference remains unresolved.

Thus beating the older count/hand baseline or an MLP is insufficient evidence that
the physical sequence model is the best pre-pitch probability estimator. The
stronger baseline's favorable Brier is relevant because WE is an expectation over
outcome probabilities, but global Brier does not measure WE-weighted error or
counterfactual action ranking directly. Neither metric alone determines policy
quality. These results motivate checking delivery integration, calibration and
action-conditional errors before attributing planning differences to useful
sequence information. They do not identify which component caused the gap.

Sources: `robustness-20260921T023616Z/summary.json` and
`frequency-baselines-20260921T030038Z/{selection,frequency_results}.json` under the
base run `/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/sequence-20260921T020823Z`.

## What the completed recommendation sensitivity establishes

The first sensitivity replay holds each case's six TRAIN-supported actions fixed
and varies control sigma (0.15, 0.30, 0.45 ft), horizon (one or two pitches), and
nonlocation physics (pool mean or a retained TRAIN-pool medoid). The original
mean/sigma-0.30/depth-two values reproduce the first recommendations exactly.

Every one of the six examples changes its top action somewhere among the twelve
scenarios. At depth two, the best action stays fixed across all three sigmas in
one of six mean-physics cases and three of six medoid cases. At sigma 0.30, changing
horizon changes the best action in two of six cases for either physics variant.
Mean versus medoid changes five of 36 matched sigma/horizon decisions; at the
reference sigma/depth it changes one of six. The maximum within-scenario shortfall
of the reference action ranges from 0.0035 to 0.0905 WE percentage points by case.

These are assumption-sensitivity measurements, not empirical regret. A changed
winner can reflect a near tie, so report value differences alongside exact action
stability. The modest numerical scale does not validate a recommendation: these
small differences may still be overwhelmed by model or execution-kernel error.

The mean/medoid comparison also separates two approximations. Averaging sine and
cosine channels can leave the unit circle, whereas a medoid retains one joint
TRAIN nonlocation vector after frozen imputation. A medoid does not recover the
distribution of execution error, intended targets, or guarantee all source
measurements were observed. Both variants fix nonlocation physics through future
counts. Historical pitches within 0.65 ft of a target establish nearby observed
support, not identified intended-target probabilities.

Source: base-run `sequence_policy_sensitivity.json`.

## How the next policy outputs should be read

The pinned actual-history set contains twelve cases selected before Q calculation:
six first chronological two-strike examples and six first PA starts with runners
and at least one out. All six mid-PA examples have two to four observed prior
physical tokens. Their counts are 1-2, 2-2, 2-2, 1-2, 0-2 and 1-2 in saved pitcher
order. Strikeouts are explicit terminal branches immediately; walks can occur
inside two pitches from the two 2-2 examples. These cases test previously missing
count/history/game-state paths. They are not a representative sample of decision
quality, and they do not themselves test a five-token root history.

`run_sequence_midpa.py` still uses bounded depth two with a separate count/hand
continuation. Its displayed WE differences can reflect that model boundary. Keep
them labeled as bounded model-internal values, and preserve their exact case keys,
observed-history masks and candidate support.

`run_sequence_rollout.py` addresses that specific continuation mismatch. It forces
each root action and continues to a PA terminal event using the same Transformer
under a fixed TRAIN-usage action policy. Root selection uses 1,024 trajectories per
action; independent evaluation uses 4,096, with common random numbers across root
actions within a stream. The baseline is the same fixed policy at root and later
pitches. This is a single-root policy improvement experiment under one simulator,
not full-PA policy optimization. Monte Carlo error can be estimated; simulator
misspecification, synthetic-history shift and unidentified control effects remain.

Report the action chosen by the selection stream even if another action wins the
evaluation sample. Use the independently evaluated paired selected-minus-baseline
contrast and its Monte Carlo interval. Do not treat an interval excluding zero as
evidence of actual policy benefit. If survivors remain at the 32-pitch cap, report
their fractions and cancellation-aware [0,1] utility bounds; do not silently use
the zero-imputed contrast as an uncensored value estimate. The shared seed design
across cases also means per-case Monte Carlo estimates are not independent evidence.

## Fields needed in the final research report

| Output | Fields to retain | Meaning |
|---|---|---|
| `midpa_selection.json` | `cases[].root_key`, `observed_history_keys`, `history_valid_mask`, `context_vector`, `selected_actions`; artifact hashes | Selection and actual-history inputs frozen before recommendations |
| `sequence_midpa_mean.json` or medoid counterpart | `actual_history_length`, actual count/game state, `ranked_actions`, `root_strikeout_probability`, `root_walk_probability`, `solver`, `seconds` | Bounded lookahead examples with explicit separate count tail |
| Rollout `config.json` | case IDs, stream sizes/seeds, cap, protocol/source/input hashes | Exact pre-inference execution specification |
| Rollout `results.json` | `selected_action_index`, action definitions, `policy_weights`, `selected_value_bounds`, `baseline_value_bounds`, `paired_zero_imputed_contrast`, `paired_censor_contrast_bounds`, `paired_censor_expanded_mc95_normal`, per-action censor fractions | Independently evaluated single-root contrast with simulation uncertainty and survivor bounds |
| Per-case rollout NPZ files | lower returns, censor flags, terminal indices, trajectory lengths and common uniforms | Recomputable trajectory-level audit record |
| Rollout `runtime.json` | `integrity_ok`, completion count, device/time and end hashes | Confirmation that inputs, source and protocol stayed unchanged through execution |

Rollout completion is signaled by `COMPLETE <directory>` after the integrity check.
The mid-PA runner instead prints its final output path/case count and records
per-case timing; it does not produce the same dedicated runtime/end-identity audit
as rollout. Do not imply that it does. Static inspection and focused CPU tests
found no necessary correctness blocker in either ready runner; target execution
and audit of their saved outputs remain separate steps.

The appropriate current claim is that a reproducible sequence-conditioned
simulator and decision experiments are available, with modest architectural
log-loss evidence and meaningful context effects. Broad predictive superiority,
the incremental value of soft clusters, stable target recommendations, and real
policy improvement remain distinct questions requiring their own evidence.
