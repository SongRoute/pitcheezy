# Fixed-context baseline blended with the 100-draw ensemble

Freeze this protocol before calculating new blend weights. This is an integration-
count sensitivity experiment on previously inspected DEV. The TRAIN delivery pools
with 25 and 100 draws are not nested because sampling consumes the fixed RNG stream
differently; a result difference is not a deterministic convergence guarantee.

## Preserve the frequency model and its earlier CAL decisions

Use the completed 25-draw context-frequency experiment's original selected
`league_context` baseline. Preserve its full TRAIN counts, smoothing, original
4,000-CAL baseline choice and scalar temperature. Reuse its exact saved 12,000-CAL
and DEV tempered probabilities; do not rebuild TRAIN data, reselect a baseline,
refit tables, or refit its temperature merely because neural integration changed.
The original primary and secondary 25-draw context blends remain frozen references.

## Inputs and only permitted fitting

Wait for completed compatible 100-draw ensemble archives from the separately frozen
reevaluation. Require full Transformer seeds 42–46, identical original 4,000 CAL
temperature keys, 12,000 reserved CAL blending keys and DEV keys. Verify the saved
ensemble is the arithmetic mean of all five per-seed probability arrays and pin
the 100-draw pool/input hashes provided by that experiment. No checkpoints are
loaded and no model inference, neural training, raw data read or new-season access
occurs in this CPU-only step.

Fit two convex ensemble/context-baseline weights on the same 12,000 CAL rows:
primary minimizes log loss; secondary minimizes multiclass Brier. Use the unchanged
bounded fit with exact endpoints. Persist both weights before opening DEV
prediction payloads. Do not choose an objective or baseline by DEV performance.
These rows were used for earlier neural stopping; they are not independent validation.

## Evaluation and common support diagnostic

Report both new blends, the 100-draw unblended ensemble, the unchanged context
frequency baseline, and the original primary/secondary 25-draw context blends.
Use the same 7,276 pitches and 87 games. Report LL/Brier and 2,000 paired whole-game
bootstrap differences against all four references. Intervals hold fitted networks,
delivery pools, frequency tables, baseline selection and CAL weights fixed; they
exclude their fitting/sampling/selection uncertainty.

Also apply the same saved impossibility mask to all predictors: remove double-play
mass when outs=2 or bases=0 and renormalize. Refuse a contradictory true DP label.
This is a separate common post-hoc support diagnostic, not a learned effect or a
criterion for selecting weights. Report both original and conditioned scores.

Check uniqueness and mutual disjointness of the exact 4,000/12,000/7,276 keyed
partitions, as well as saved row hashes, labels, game IDs and original metric replay.
Freeze new protocol/source/input hashes in a distinct SSD output before fitting,
save selections/probabilities/results/completion, and leave every 25-draw artifact
unchanged. Any improvement remains exploratory; do not claim new confirmation,
universal optimality, numerical convergence or actual policy-value improvement.
