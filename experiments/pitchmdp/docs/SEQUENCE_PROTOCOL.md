# Sequence architecture and PA win-value experiment protocol

Frozen design: 2026-09-21, before training the sequence comparison. This protocol
extends the negative pilot recorded in [REVISED_PILOT.md](../REVISED_PILOT.md).
Those results remain valid; the new experiment does not replace or relabel them.
The implementation and run manifest must record any departure from this protocol
before the affected run starts. This is development research, not a pristine
held-out confirmation: the project has already inspected the 2025 DEV period.

## Question and allowed claims

First compare a Takamido-inspired Transformer with a simpler network receiving
the **same ordered information**, then test the added contribution of batter
types, history, response context, and the PA win-expectancy objective. A better
conditional prediction score does not establish a better pre-pitch prediction,
and neither alone establishes a better policy or real-world win rate.

Use the name **Takamido-inspired common-data adaptation**. This is not a faithful
full reproduction of [Takamido & Nakamoto](https://arxiv.org/abs/2606.17345): the
original 2018 data are not loaded; our approved seasons, splits, pitcher cohort,
as-of batter representation, and all-count ten-outcome task differ. Six tokens
mean the last **five historical pitches plus the current candidate pitch**, not
six past pitches. Never describe the previous one-pitch-history prototype or its
scores as the original Takamido model. Never compare a paper's binary score
numerically with our multiclass score as though they measure the same task.

## Data and populations

- Open only the explicitly approved 2023–2025 source files. Do not open 2026 raw
  data, download another season, or load the paper's 2018 data in this experiment.
- Preserve the existing history warm-up before 2023-05-15. Weight fitting uses
  2023-05-15 through 2025-04-30; calibration/model selection uses 2025-05-01 through
  2025-06-30; DEV reporting uses 2025-07-01 through 2025-09-30.
- Fit scalers, imputations, type centers, delivery distributions, and learned
  weights on train only. Calibration data may select the prespecified checkpoint
  and calibration parameters. DEV labels cannot select features, hyperparameters,
  checkpoints, temperatures, seeds, or a preferred reported run.
- Freeze the existing six-pitcher all-team cohort: Logan Webb, Logan Gilbert,
  José Berríos, Luis Castillo, Framber Valdez, and Chris Bassitt. Retain its
  historical selection rule and manifest: the prior pilot required at least
  50 evaluation-period starting innings before ranking training innings. This
  is a retrospective population selected partly using future availability, not
  a prospective cohort. Do not silently recalculate membership for this run.
- Apply the same eligibility rows to every comparison within a task. Report
  games, pitchers, batters, PAs, pitches, missing physical fields, and exclusions
  before and after filtering. Whole-PA eligibility uses observed future paths
  and therefore defines a restricted retrospective population.

## Inputs and information boundaries

Physical tokens contain only the declared pitch physical feature allowlist,
including effective speed and spin axis when available. Record the exact ordered
columns, units, angle encoding, imputation, scaling, and missingness treatment in
the run manifest. Historical tokens are chronological within the same PA,
left padded when fewer than five past pitches exist; the candidate occupies the
last position. Padding is explicitly distinguished from a zero-valued observation.
The first pitch of every PA has no historical tokens. A previous PA, another
game, or any pitch after the current pitch may not enter history.

Context includes the current legal count, handedness, game state, and the
existing strictly pre-date batter profiles/types. Batter and pitcher IDs may
join records and index a train-fitted delivery distribution; IDs must not enter
the response-network input. Same-day doubleheaders cannot see each other's
batter results. No current outcomes, future count, terminal event, next state,
post-pitch score, supplied win expectancy, or current batted-ball measurement
may enter the response encoder. Historical observed physical values are allowed;
the **realized physical values of the current pitch are not known pre-pitch**.

All members of the same architecture comparison receive identical physical
features, masks, context, and eligible rows. The flattening MLP retains the same
ordered six-token vector, including candidate and padding. It is not a
history-removal ablation. A separate memoryless network receives the current
candidate physical token and the same current context only. Preserve legal count
in every history ablation.

## Two tasks and two observation regimes

1. **Two-strike terminal binary diagnostic:** identify the exact subset and label
   mapping for terminal swinging strikeout versus ball in play with two strikes.
   Report subset coverage and exclusions. This is a diagnostic comparison closer
   to the paper's prediction task; it is not the full paper reproduction and it
   cannot supply transitions for all-count PA planning. Foul tips, bunts, and
   ambiguous labels need explicit inclusion rules.
2. **All-count transition task:** preserve the ten shared outcomes (`ball`,
   `strike`, `foul`, `out`, `double_play`, `single`, `double`, `triple`,
   `home_run`, `hbp`, subject to the implementation's recorded order). A two-strike
   bunt foul terminates as a strike; an ordinary two-strike foul repeats the
   count. Outcome probabilities must be finite, nonnegative, and sum to one.

For each applicable task, distinguish **observed-current-physics conditional
diagnostics** from **pre-pitch delivery-integrated predictions**. The former
conditions on the observed current trajectory and is unavailable when choosing a
pitch. The latter integrates a joint physical-delivery distribution fitted using
train data; every architecture uses identical draws and grouping/backoff rules.
Changing the logged current physical values must leave an integrated prediction
unchanged. Historical physical values remain fixed and observable. Report the
delivery assumptions and Monte Carlo draw count; gains may be limited by this
distribution rather than the conditional network.

## Fair fitting and prediction evaluation

Before fitting, record architecture dimensions, parameter counts, initialization
seeds, optimizer, learning rate, weight decay, batch size, maximum epochs/updates,
early-stop rule, sample cap and sample seed, device, and calibration rule. Use the
same train rows, calibration rows, maximum data passes, seeds, and checkpoint
selection criterion for the Transformer and flattening MLP. Report actual compute
and parameter differences rather than claiming exact compute or capacity parity.
Any architecture-specific search must have the same prespecified search budget.
No architecture gets extra tuning after viewing DEV scores.

Primary prediction comparison: Transformer minus flattening MLP on **all-count
delivery-integrated DEV log loss**, with negative values favoring Transformer.
Also report multiclass Brier score, calibration diagnostics, conditional log loss,
memoryless comparison, and the existing simple count/handedness baseline on the
same rows. Binary metrics appear in a separate table. Report uncalibrated and
calibrated scores with the selected calibration parameters; do not choose the
more favorable DEV variant retrospectively.

Use paired **game-cluster bootstrap, 2,000 replicates, 95% interval** for metric
differences: resample whole games with replacement, retaining all shared rows and
their multiplicities, and recompute the pitch-weighted mean difference. Do not
treat pitches as independent. Report game count and absolute difference as well
as the interval. A single training seed is exploratory; it does not estimate
training variance. Additional seeds must be prespecified, with each result
retained, and not counted as independent additional games.

## Added value and PA WE extension

Only the all-count model supports the PA extension. Compare the following with
shared transition/delivery assumptions and identical action support:

- Full vs no batter types, preserving handedness;
- Full vs no history, preserving current count and legal state;
- Full vs no extra response-model game context, preserving the complete legal
  state in transitions and WE evaluation;
- neutral/RE objective vs WE objective, holding response probabilities fixed;
- one-pitch intervention followed by a common baseline vs full PA planning.

Response context and reward are separate mechanisms: removing response context
must not erase bases, outs, score, inning, or the defended team from the game
state. The target is expected defensive WE at PA termination. Keep the defending
team fixed across half-inning transitions, use exact game-ending win/loss
boundaries, and do not count terminal reward twice.

A policy branch must update count, token history, and legal game state using its
own simulated pitch and outcome. It must never reuse the observed future count
or future physical trajectory after changing an action. If a sequence-aware PA
solver is not implemented, label that extension incomplete; a one-pitch solver
with frozen history cannot be presented as the requested full sequence PA model.
State any approximation, finite horizon, rollout continuation, and two-strike
foul treatment. Compare objectives using common simulation randomness where
possible and report approximation sensitivity.

Observed landing location is not a target label. A target-location policy needs
an explicit latent-intent/command assumption. For the current pilot, model
internal expected WE differences and example recommendations are diagnostic
optimization results. They are not independent policy evaluation or causal win
improvements. Independent evaluation and support diagnostics remain necessary
before claiming useful real-world gain. Until then, report whether each proposed
addition improves the common prediction task and what the PA calculation assumes.

## Run record and decision

Save this protocol with the exact source snapshot, configuration, source hashes,
row-key hashes per split/task, model weights, calibration settings, per-row
predictions, paired intervals, coverage, timings, and failures. Preserve negative
and inconclusive results. A useful advance requires evidence of added value over
the common simple baseline; adopting a published architecture or optimizing its
own WE output is not itself success. Any new choices motivated by this DEV run
must be called a subsequent exploratory experiment and require another untouched
period for confirmation.
