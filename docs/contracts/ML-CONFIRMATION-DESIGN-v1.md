# C1/C2 five-seed execution design — draft v1

Source-only design, 2026-09-24. No candidate selection, preparation, model fit or
DEV score is claimed. Activate a concrete configuration only after reviewing the
upstream P3/G/F4 results under [ML-FOLLOWUP-REVIEW-v1](ML-FOLLOWUP-REVIEW-v1.md).
This draft neither changes existing selection rules nor activates every route.

## Selection and interpretation

C1 compares at most two promising configurations against a fixed strong control,
using five ordered seeds (0–4) on common Cpanel TRAIN/early-stop/May-temperature/
June-blend/DEV keys. C6 architecture scores cannot be subtracted from Cpanel G or
F4 scores to estimate component gains. These are adaptive development comparisons
on exposed DEV; five seeds do not create independent confirmation or five new
game samples. Multiplicity guarantees are local to each registered family.

The frozen G ranking **vetoes measured `R.failed` for promotion**. Preserve that
rule and its archived decisions. An upper confidence bound above a noninferiority
margin is not proof of actual harm; inspect the point estimate, both interval
endpoints and support when deciding whether to register a separate diagnostic.
A diagnostic follow-up cannot retroactively promote a failed G candidate.
Conditional matrix rows need documented activation/nonactivation review rather
than automatic omission.

If no promising component remains after actual review, the preferred closure is
C1 five-seed strong-baseline stability plus fixed-seed L6 diagnostics, without a
superiority claim or forced combination. Record C2 as not applicable with the
actual evidence and remaining limitations. This fallback is not a declaration
that unreviewed rows are completed.

## Practical additive routes

| Route | Common-input comparison and reuse |
|---|---|
| Existing G or F4 contrast | Lowest implementation cost. Extend identical candidate/control members with seeds 3/4. Reuse 0–2 only after exact training, feature, auxiliary, calibration and sample identity checks. |
| P3 backbone plus G sharing | Practical combination. Full=new backbone+selected G; minus-backbone=MLP+selected G; minus-sharing=new backbone+mapped G control; baseline=MLP+mapped G control. Use the same G continuous profile/context and Cpanel CAL throughout. Existing matching MLP G members can be reused; P3 weights have different context and cannot fill the new-backbone G cells. |
| G plus F4 | Requires a new lazy routing/shrinkage adapter. Use matching dual-stream H0 controls with maximum history capacity 128. Adding cluster flags changes context width, generally requiring fresh G2/G3/G4 lazy cells. Reuse existing F4 global members only where their complete contract remains identical. |
| P3-only Cpanel bridge | Keep the original 28-column context for both architecture candidate and A0. Identical conditional-training weights may be referenced, but Cpanel May calibration and June/DEV predictions are new artifacts. Never reuse C6 calibrated predictions or call this the same control as G's continuous-profile baseline. |

F4 is a nonlinear token-encoder/pooling architecture adaptation. Attaching it to
a P3 Transformer/LSTM is a new architecture adaptation requiring its own bridge,
not an existing history-only comparison. Choose the route and exact baseline
before execution.

Implement a new confirmation runner and identities; do not monkeypatch frozen
`SEEDS` constants or alter existing P3/G/F4 source. `MatrixModel` and
`LazyMatrixModel` accept seeds 3/4 directly. Existing F4 `fit_member` identities
reject them, so a new adapter must call the frozen model/delivery APIs directly.
Reference reused artifacts with original manifests; do not rewrite old states.
Distinguish reused conditional weights from reused calibrated members.

## Calibration, inference and ablation

Each seed uses the same allowed-May temperature procedure. Fit a fresh June blend
for the five-seed probability ensemble; maintain the matched per-seed June blends
for seed-direction checks. Current delivery uses the same frozen 400 joint draws.
Do not tune on C1/C2 DEV or choose a favorable seed.

Proposed C1 primary family: two overall NLL contrasts, retaining an unused null
slot if only one candidate is registered; Holm .05. Require ΔNLL <= -.003, paired
whole-game 95% NLL upper < 0, Brier upper <= .001 and improvement in at least four
of five matching seeds. R has 2×12 groups×2 losses = 48 Bonferroni bounds with
100,000 draws, fixed seed 20260924 and the existing 30-game/500-pitch group gates.
If low-group improvement will promote a candidate, register the expanded four-test
overall/low family beforehand; otherwise keep it descriptive.

C2 removes each actually added component from the full configuration, preserving
all other data, calibration and optimizer settings. Compare five-seed ensembles
and paired full-minus-ablation losses; deduplicate identical ablations and shared
fit units. Register all contrasts together. The matrix minimum is descriptive
paired intervals; inferential component-gain claims require a fixed Holm family.
Removing history masks its stream while preserving 128 capacity. A selected G4
sharing package is one component unless its internal levels receive separate,
explicit ablation registration. Do not choose removal order from C2 DEV results.

L6 needs no additional fits: compare fixed seed 0, 0–2 and 0–4 ensembles from one
configuration, each with its own June ensemble weight. No best-seed selection.

## Resource gate and outstanding decisions

Count distinct neural units, not logical cells. Extending matching G contrasts
from three to five seeds costs: G2/G0 four new fits; G3/G2 twelve; G1/G0 up to 24;
G4/G2 up to 34 under the current 11-personal/4-cluster panel. One F4 length versus
H0 costs four new fits; both lengths versus H0 cost six. New combinations require
all five seeds for every genuinely new unit; share identical units explicitly.

Profile synthetic/May-only workloads before opening new quality results. Register
fit, full-May 400-draw calibration, June/DEV prediction, aggregate memory/time and
per-job ceilings (7200 seconds), with one heavy execution owner. Preserve failures;
never silently reduce seeds, history or sample size to fit the budget.

Root must freeze the candidate route, strong baseline, reuse audit, exact C2
component list, N versus N/G family, and aggregate resource budget after upstream
review. No implementation or heavy execution is authorized by this draft alone.
