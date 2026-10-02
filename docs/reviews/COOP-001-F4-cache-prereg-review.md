# COOP-001 F4 cache prospective scientific review

- Date: 2026-09-27
- Reviewer model: `gpt-6-astra`
- Status: prospective, read-only source review; no tests, real data loading, backend measurements, scoring or adoption decision.
- Scope: acceptance conditions for the forthcoming observed-context cache audit. This document does not approve an implementation or change an existing sealed registration.

## Mandatory requirements already present in the historical contract

The [optional-cache contract](../contracts/ML-F4-CONTEXT-CACHE-OPTIONAL-v1.md) requires a fresh sibling audit, frozen configuration/environment/source identities, parent preparation/profile evidence, fixed sample selectors, command limits and tolerances. Preserve previous attempts and costs. The [historical registration](../../configs/EXP-P4-002-v3-cache-audit.yaml) explicitly leaves gradient and updated-weight tolerances unregistered; it alone cannot authorize stage 2.

1. Compare exact float32 context bits on the fixed 65,536 TRAIN rows and 2,048 TRAIN evaluation rows in original, repeated, reverse and uneven-chunk orders. Exercise candidate-type and current-physics overrides. Guarded synthetic count/style changes must fail. Preserve independent returned arrays and observed-row identity. Time construction, context transform/lookup and whole lazy gather separately, warming each path and alternating timing order. Compute no DEV metrics.
2. For H0/32/128, compare forward outputs, loss, parameter gradients and optimizer-updated weights using identical seed-zero width-128 initialization and TRAIN minibatch/order on actual MPS. Register tolerances before execution; do not relax them after failure. Cache operations must not change RNG state or weights outside the explicit training step.
3. Measure complete original/cached resource fits with matching sample, initialization and settings. The permitted short profile fixes four epochs/patience four, batch 256 and learning rate .0005; full scientific settings remain 30/5/256/.0005. Include fixed May16×400 delivery calibration/inference, process costs and memory, and numerical/selected-state comparisons. Never reuse profile models in the full experiment.
4. Adoption requires a separate decision and fresh preparation/profile version. Recompute all three projections with cache construction/lookup, two processes/member, full 30-epoch work, and every previous failed/preparation/profile/audit cost. Require three sealed profiles, ≤7,200 seconds/member and ≤28,800 seconds/complete family. No reduction of seeds, data, epochs, batch or draws is authorized.

Resource-only audits must not construct June or DEV context caches; frozen population counts suffice for extrapolation. The cache applies only to immutable observed rows, not simulated policy states.

## Suggested strict prospective numerical registration

These are recommendations for the lead to freeze before any new real execution, not a claim that they are already registered or that every MPS execution satisfies them. All numerical bounds are elementwise absolute maxima with `rtol=0`; require finite values and matching shapes/dtypes. Record per-parameter maxima and exact-equality status. Do not substitute average differences or rounded scores.

| Quantity | Suggested acceptance limit |
|---|---:|
| Context/all gathered inputs, initialization | Bitwise exact |
| Forward logits | `2e-5` |
| Probability | `2e-6` |
| Probability mass error | `1e-6` |
| Loss/objective | `2e-6` |
| Each pre/post-clip gradient | `2e-6` |
| Updated/selected weights | `2e-7` |

The forward/probability/objective limits inherit the [existing backend precedent](../../experiments/pitchmdp/pitchmdp/matrix_long_equivalence.py). Gradient and weight limits are **new strict prospective acceptance choices**, not empirical error bounds or published guarantees. The rationale is that context memoization preserves exact inputs and network arithmetic. Failure is a preserved result, not permission to loosen the same attempt's threshold. A separately justified implementation/registration may use a different defensible acceptance design before execution; this review does not retroactively redefine the historical contract.

For independent May temperature fits, the earlier precedent provides absolute `1e-3` temperature and `2e-5` refitted-probability bounds. Register these explicitly if retained. Comparing both paths with one shared fitted temperature would not test calibration divergence.

## Additional suggested controls and their rationale

The following strengthen the audit beyond the historical contract's explicit detail. They are prospective review suggestions, not newly discovered historical obligations; an independently justified, preregistered implementation may provide equivalent assurance.

- Compare every named gradient before and after clipping and compare optimizer state after the update, not only final weights. Specify numeric limits for floating optimizer-state tensors and exact equality for step counters before executing such checks. The historical requirement names gradients and updated weights, but does not explicitly demand both gradient stages or optimizer-state comparison.
- Mirror the actual [training path](../../experiments/pitchmdp/pitchmdp/matrix_lazy_model.py): normalized float32 sample weights, weighted cross-entropy, AdamW with learning rate .0005 and weight decay .01, and gradient clipping at 5. A surrogate SGD update does not validate this production path.
- Check all gathered arrays, including masks and history/current overrides, rather than context alone. Compare raw float32 bits: numerical equality alone does not distinguish signed-zero representations.
- Snapshot Python and NumPy global RNG, explicit generator states where present, Torch CPU and MPS states. Distinguish initialization's legitimate RNG use from cache operations; restore identical pre-step state for paired updates and compare post-step states.
- Run backend paired checks in both execution orders and preserve every registered comparison. The historical contract explicitly alternates stage-1 timing order; extending both-order completeness to backend/resource comparisons is an additional suggestion. Synchronize MPS before and after timed segments.
- Compare each epoch's selection decision, epoch/update counts, best epoch and selected weights. Production selects with `score < best - 1e-5`; passing a small numerical loss tolerance does not by itself ensure identical selection. The contract already requires selected-state comparison, while decision-by-decision logging is an added control.
- Check frozen delivery vectors/tiers, logits, separately fitted temperatures/objectives and calibrated/raw probabilities. Do not publish resource-profile quality as scientific model performance.

## Cost projection and execution traps

The [existing projection](../../experiments/pitchmdp/pitchmdp/matrix_long_profile.py) accounts for two data loads but has no cache-build term. Reusing it unchanged would omit new costs.

- Charge complete child-process wall time: loading, verification, construction, guarded lookup, synchronization and serialization, not only internal fit timers. Retain stage decomposition to avoid double counting.
- Project a full TRAIN+early-stop+May cache for each fit process and a separate June+DEV cache for each prediction process. Never amortize builds across separately launched seeds or processes. Use frozen population counts without constructing June/DEV audit caches.
- Preserve the conservative 30-epoch update projection and early-stop workload-ratio check. Explicitly add cache construction and other process overhead not already measured. One small-sample linear estimate alone does not establish full-population scaling. Preregister a scaling check or conservative bound; unsupported/nonlinear extrapolation cannot establish feasibility.
- The [cache implementation](../../experiments/pitchmdp/pitchmdp/matrix_observed_context_cache.py) allocates 208 value bytes per cached row and four lookup bytes per **entire source-frame row**. Include deep guarded-frame/index storage, temporary construction chunks and process high-water RSS. MPS allocator snapshots are not peak MPS memory measurements. Exact lookup guards must remain enabled while measuring speed.
- Multiply complete member costs by three seeds for each of the three arms; add the evidence-backed prior-cost ledger. The [existing family validator](../../experiments/pitchmdp/scripts/run_ml_long_history.py) accepts exactly four prior-cost categories. New cache-audit costs need explicit, validated ledger treatment, possibly a versioned schema; they must not disappear because an old schema lacks a new category name.
- A passing projection is not a runtime guarantee. Keep the shared heavy lock, subprocess limits and authoritative cumulative owner ledger. Failed and timed-out attempts still consume budget.

Fatal omissions before execution/adoption include unregistered backend thresholds, a mismatched optimizer, a one-step-only check presented as fitted-state equivalence, missing selection comparison, unsynchronized timing, omitted full-population/second-process cache construction, disabled or uncharged guards, missing prior audit costs, and interpreting audit success as automatic authorization for full training.

The handoff's historical 63,031.9-second forecast exceeds the 28,800-second family cap; this review did not remeasure it. Cache ineffectiveness under the fixed budget is a valid result. Bounded numerical agreement would support only the registered samples, execution path and backend/environment, not universal equivalence.
