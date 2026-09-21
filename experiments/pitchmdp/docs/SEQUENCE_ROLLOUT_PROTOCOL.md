# Frozen sequence-policy rollout protocol

This protocol is written before running the target model on the selected examples.
It defines an exploratory, model-internal, single-root policy improvement study.
It does not estimate actual policy benefit or a fully optimal PA policy.

## Frozen inputs and examples

Use the original seed-42 all-outcome Transformer, its conditional temperature,
frozen encoders, trained WE/runner-advancement models, and the SHA-verified
`midpa_cases.pkl` / `midpa_selection.json` prepared by `run_sequence_midpa.py`.
The saved examples are chosen chronologically by pre-pitch state, beyond existing
eligibility. Do not select examples using outcomes, winners, model Q, or rollout
results. Preserve each example's actual count and up to five observed prior
physical tokens. No current or future logged physical measurement enters inference.

Keep each example's six previously selected TRAIN-supported actions fixed. Use
the existing mean nonlocation physics approximation and independent target
location Gaussian axes with sigma 0.30 ft and nine Gauss-Hermite points. Pools are
chosen at the root count and remain fixed through the PA. This control kernel is
assumed, not identified from intended targets, and omits nonlocation execution
variability. Conditional calibration under this intervention is unverified.

## Continuation and terminal utility

Force each candidate action at the root. At every later pitch, sample an action
from the fixed TRAIN pitch-type-usage policy, uniform over selected targets within
each type. Sample the delivery quadrature point, then sample one of the ten
outcomes from the same conditional Transformer. Update the count and physical
history on every nonterminal delivery, including two-strike fouls. Project illegal
double plays as in existing recommendations. Game context remains fixed until
the PA terminal event; only previously supported PAs are included.

A fourth ball terminates as a walk; a third strike terminates as a strikeout;
other terminal outcome categories terminate directly. Assign the predetermined
expected defensive WE for that event from `game.terminal_values`. This integrates
the fixed advancement/WE model at termination without additional game simulation.
No separate count-frequency model supplies continuation values.

## Monte Carlo streams and root selection

Defaults fixed before inference:

- Selection: 1,024 trajectories per forced root action, seed 142.
- Evaluation: 4,096 trajectories per forced root action, seed 242.
- Maximum simulated pitches per trajectory: 32, including the forced root pitch.

For each case and stream, pre-generate three uniforms for every trajectory index
and pitch step: continuation action, delivery, and outcome. Reuse those uniforms
across all root actions (common random numbers). Stream randomness is independent
between selection and evaluation. Reuse the same seed design for every case;
report each case separately and do not treat case estimates as independent samples.

Select the root action maximizing its selection-stream mean return, assigning
zero temporarily to censored trajectories. This is the conservative lower-bound
selection score; break exact ties by fixed candidate order. Do not choose the
action on evaluation results. Report all per-action evaluation means and ranks as
diagnostics, without replacing the selection-stream choice.

For evaluation trajectory index i, calculate baseline return as the fixed policy
weights times the vector of returns for all forced root actions at i. This
enumerates the root policy mixture while retaining its same fixed continuation.
The paired contrast is the selected action's return minus that mixture return.
Report its Monte Carlo standard error and normal-approximation 95% interval from
paired per-trajectory contrasts. These quantify simulation noise conditional on
one fitted model, case, and assumed kernel; they are not training, game-sampling,
causal, or deployment uncertainty.

## Censoring and audit artifacts

Stop remaining trajectories after 32 pitches and report truncation fractions for
every action and stream. Their unknown terminal WE lies in [0,1]. Save both the
zero-imputed lower return and the censor flag, yielding the upper return by adding
that flag. Do not substitute a count tail or conceal survivors.

Report estimated lower/upper return bounds for actions, selected root policy and
baseline. For the paired contrast, first cancel the selected arm's contribution
shared with the baseline: its coefficient is (1 - its policy weight), and each
other arm's coefficient is minus its policy weight. Bound the remaining unknown
censored returns in [0,1], choosing the lower return for positive coefficients
and upper return for negative coefficients to obtain the contrast lower bound;
reverse those choices for the upper bound. Thus a one-action policy comparison
has exactly zero contrast even when censored. Also report a censor-expanded
interval using the lower contrast mean minus 1.96 times its Monte Carlo SE and
the upper contrast mean plus 1.96 times its Monte Carlo SE. When nothing is
censored, these collapse to the ordinary paired estimate/interval. If censoring
remains material, retain the bounds and avoid claiming a resolved direction of
improvement. This cancellation refinement is frozen before the first target run.

Write configuration, protocol hash, code hashes, checkpoint/encoder/prepared-case
hashes, per-case outputs, and compressed trajectory lower returns, censor flags,
terminal-event indices, trajectory lengths, and common uniforms to the configured
SSD run directory. Runtime CLI overrides are recorded and define a separate
execution of this protocol; they must not be silently presented as the defaults.
