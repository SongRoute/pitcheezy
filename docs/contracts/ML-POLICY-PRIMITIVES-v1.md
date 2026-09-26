# Pitch-type rollout policy primitives v1

2026-09-24. This is an implementation/API contract, **not a registered or executed real-data policy comparison**. Real training, profiling, inference, calibration and evaluation remain with the single heavy-execution owner. No 2026/final data or service artifacts are used. Outcome/control selection remains pending; these primitives accept frozen callbacks and do not select a candidate.

## Policies and interpretation

- P0: TRAIN-only hierarchical categorical behavior cloning. Counts condition on pitcher, count, batter side and last past pitch type, shrinking to pitcher frequencies. Unknown pitchers fall back to league frequencies/support. Compare P0 action log-loss with the simpler pitcher-frequency baseline on exactly the same support; do not claim neural BC or value improvement from action agreement.
- P1: estimate each supported root action's terminal defensive WE by Monte Carlo under BC continuation (`Q^BC`), choose the maximum, then use BC on every remaining pitch.
- P2: repeat the same one-step rollout improvement at every generated/observed pitch through PA termination or the registered cap. This is a sequence-aware receding rollout approximation for the full PA, **not exact full-PA optimization**, and is not a sparse depth-two tree.
- P3: repeat `pi(a) ∝ BC(a) exp(Q^BC(a)/tau)` on the same support. This maximizes a one-step expected Q minus KL penalty for the supplied Q; **it is not the solution of a KL-regularized Bellman equation**. Report original unpenalized defensive WE in the common evaluator. Tau is in absolute WE units, not percentage points; freeze a modest grid and select only in a policy-tuning period disjoint from DEV.
- P4/P5 are conditional later IQL/CQL comparisons, not implemented or represented as completed by these primitives. Observational OPE and causal effects remain null unless identification is independently established.

## Integration boundary

New code: `experiments/pitchmdp/pitchmdp/rollout_policy.py`. Existing `game.py`, `planner.py`, `sequence_planner.py`, raw/cache/model/runtime sources are unchanged.

`PAState` exposes count, pitcher, batter side, a fixed context identifier and immutable past tokens. `BCRecord(state, action, split)` uses action only as the label. Current realized pitch type, physical measurements and outcome have no BC input slot. The caller must construct legal as-of states and assert records are TRAIN; the primitive rejects any non-TRAIN split. It does not independently audit dataset hashes/dates or detect a falsely labeled past token. The adapter must never insert logged future observations.

`CategoricalBC` learns vocabulary and pitcher support from TRAIN counts only. Positive probabilities occur only on support. Unknown pitcher fallback is explicit via `fallback(state)`; empty known-pitcher support raises an abstention. The runner must report all requested PA starts, unsupported exclusions/fallbacks, unsupported observed action rate and conditional supported-row action NLL; never silently discard unsupported labels.

`DeliveryPool(values, source_split, source_hash, draw_key)` requires exactly 400 joint finite physical vectors and TRAIN provenance, and copies them to immutable storage. The runner verifies actual source hashes, conditioning/fallback tier and draw identities. Searches sample indices from the entire frozen 400-draw pool; the pool itself is not reduced or refitted. Observed plate coordinates remain physical delivery features, never intended target actions.

`JointSimulator(pool, predictor, terminal, budget)` accepts callbacks. `pool(state, action)` selects the frozen action-conditioned pool. `predictor(states, actions, physical_vectors)` returns batched calibrated conditional 10-class probabilities. Each trajectory first draws physical delivery, then its conditional outcome, and appends that **same** action/physical/outcome/pre-pitch-count token to history. `calibrated_conditional` provides per-seed temperature → probability ensemble → frozen frequency-blend; its `neural_weight=1` means neural-only. The three models must see the same physical draw. Sampling an integrated outcome independently of the generated physical vector violates this contract.

`terminal(state, event)` supplies absolute PA-end WE for the initially defending team, for example from frozen TRAIN-fitted `game.terminal_values`/advancement/WE models. Current game state is fixed through a supported PA; unsupported mid-PA runner/game-state changes require exclusion and support reporting. `cutoff(state)` is a common explicit tail approximation, not logged future or claimed PA completion. All utilities must lie in [0,1].

`RolloutImprovement` offers `policy("P0"|"frequency"|"P1"|"P2"|"P3", tau=...)`. `rollouts` is the same evaluator for every policy. Planning RNG is derived from search seed, full legal context/history and support; evaluation RNG comes from a separate externally generated `[trajectory,pitch,3]` array. Reuse the latter across policies for policy/delivery/outcome common random numbers. Never use evaluation random numbers to optimize Q. Exact-state Q caching (default at most 4096 states per frozen planner) saves repeated work without changing decisions. Construct a new planner after any model, pool, support, cap or parameter changes; existing cached values must not survive such mutations.

## Budgets and reporting

Let J be fixed PA starts, R evaluation rollouts/PA, A maximum supported actions, S search rollouts/action, Cs search pitch cap, Ce evaluation pitch cap. Before cache/early-termination savings:

- Each search decision costs at most `A*S*Cs` conditional prediction rows.
- P0 or frequency costs at most `J*R*Ce` rows.
- P1 costs at most `J*R*(Ce + A*S*Cs)` rows.
- Each P2/P3 costs at most `J*R*Ce*(1 + A*S*Cs)` rows.
- Multiply by the number of seed models for network rows. Fixed-context model feature preprocessing may add additional cost and must also be profiled.

For J=32, R=64, A=8, S=8, Cs=Ce=16, one P2 upper bound is 33,587,200 conditional rows / 100,761,600 seed-network rows for three seeds. Thus a 10M-row budget is not guaranteed to fit those settings. The runner must profile legal representative data before candidate outcomes, freeze feasible counts/caps/resource limits, and preserve all failures. `RowBudget` fails before a call would exceed the aggregate conditional-row ceiling; it does not silently reduce samples. Shared budget objects can count search+evaluation and all policies. `search_conditional_rows` and `network_rows` distinguish actual costs; wall time, memory, seed identities and provenance remain runner responsibilities.

`Rollouts` preserves per-trajectory values, truncation flags and pitch counts. Capped paths use the declared cutoff approximation and also expose worst-case [0,1] terminal-WE bounds. `q_values` returns each action's MC standard error and truncation bounds; maximization/selection uncertainty is not captured by those marginal SEs. `paired_summary` accepts one PA's `[rollouts]` or a panel's `[PAstarts,rollouts]`, reports unpenalized paired mean difference, stratified simulation MC SE and worst-case truncation-difference bounds. Multiply WE difference/SE/bounds by 100 for percentage points. Between-game uncertainty and multiplicity require separate frozen game-level paired bootstrap/Holm analysis; simulation MC SE is not a policy-effect confidence interval. Preserve per-PA/game membership and per-policy rollout arrays for that analysis. A zero-width MC error in a deterministic simulator is not certainty about the real world.

The first real-data config still must freeze selected outcome/control/checkpoint hashes, TRAIN BC/WE/advancement scope, policy-tuning/DEV PA manifests, tau grid, support counts, exact pools/draws, seeds, resource ceilings, cap-tail definition and paired analysis. Already exposed ML DEV reused for policy experiments must remain labeled development/model-internal. Product adoption, causal effects and final confirmation cannot follow from this primitive's synthetic tests.

## Verification

Synthetic tests cover TRAIN-only BC and unknown support; coherent delivery/outcome history; count/foul/walk/strikeout transitions; exact400 immutable pools; calibration/integration equivalence; P1 versus P2 timing; P3 normalization/support; RNG determinism/history sensitivity; hard budget failure and explicit truncation bounds. No real-data profile/fit/inference was run for this implementation.

```sh
PYTHONPATH=experiments/pitchmdp python -m pytest -q experiments/pitchmdp/tests/test_rollout_policy.py experiments/pitchmdp/tests/test_sequence_planner.py experiments/pitchmdp/tests/test_game_planner.py
```
