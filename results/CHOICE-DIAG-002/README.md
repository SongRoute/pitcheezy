# CHOICE-DIAG-002 — frozen July recommendation diagnosis

The preregistration is `configs/EXP-CHOICE-DIAG-002.json` and
`docs/contracts/recommendation-diagnosis-v2.md`, committed as `885bb768` before
new July results were computed. `summary.json` is a small repository copy. The
full per-pitch Q values, action support, member comparisons, input/model hashes,
and 151 PA checkpoint hashes are in the isolated SSD `results.json` path named
in `summary.json`.

The six previously selected 2025 DEV games contain 1,690 pitches overall.
The two target pitchers have 160 PAs/597 pitches; existing complete-PA rules
retain 151 PAs/563 pitches. Nine PAs/34 pitches are excluded: five unsupported
PAs, three with a changing within-PA state, and one without a terminal event.
All 151 evaluations were ready, and the observed type has at least one
supported zone for all 563 eligible pitches. Thus 100% observed-type support
is conditional on eligible PAs; 563/597 is the whole-target fraction that was
both eligible and supported. Excluded actual-type mix is recorded separately.

The actual type's best supported zone is action rank 1 in 147/563 pitches.
Median model-internal gap from the best action is 0.0392 percentage points;
324/563 gaps are at most 0.05 pp. Across the five existing trained checkpoint
members, all agree on the top action for 33.6% and top type for 54.9% of
pitches. These are member sensitivity results with fixed delivery samples and
frequency controls, not independent delivery-sampling uncertainty or policy
effect estimates.

Observed and recommended top-type counts differ: ST is 105 observed versus
204 recommended, and SI is 180 versus 109. Total variation is 0.290 for Webb
and 0.177 for Wheeler on their own evaluable pitches. The frozen optimizer
ranks model defensive WE and has no direct pitch-mix constraint. That mechanism
is verified in code, but the causal share of this mechanism in the observed
mix difference is not identified. Differences may also reflect unobserved
intent, command, batter strategy, model error, and the supported action grid.
The seven never-supported pitch-type×zone cells have null ESS/mass, because the
public adapter returns ESS/mass only for retained actions. Their reported
support rate denominator is eligible PAs for the given pitcher; ESS/mass
summaries are conditional on supported PAs. Kernel ESS is not OPE ESS.

The six-game paired bootstrap interval is exploratory on an exposed DEV set;
member spread is not a confidence interval. Existing prediction quality is
reported separately by EXP-A-COMMON-001. No OPE or causal policy value is
identified, and no service model is adopted from this diagnosis.

Run command:

```sh
/Users/song/Projects/pitcheezy/.venv-observer-standalone/bin/python scripts/b_choice_diagnosis.py --config configs/EXP-CHOICE-DIAG-002.json
```

The command requires an absent `results.json` and validates existing PA
checkpoints before reuse. The initial full pass completed all 151 PAs in about
40 seconds, then its first aggregation hit macOS `._` sidecar files on T7.
That unaggregated pass is preserved in `CHOICE-DIAG-002-run.log`. The first
aggregate is preserved as `results.pre-audit.json`; the final audit run
filtered sidecars and validated PA keys, pitch keys, variant set, Q arithmetic,
model identity, and committed input hashes without new inference. Its log is
`CHOICE-DIAG-002-audit.log`. Checkpoints did not embed a producer-code hash at
creation time; the final report explicitly states that provenance limit.
