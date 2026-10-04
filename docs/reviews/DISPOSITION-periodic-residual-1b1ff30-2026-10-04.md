# Disposition — same-model review of 1b1ff30 (periodic/linear embedding + residual backbone)

Label: `same_model_reviewed; Codex_review_pending`. Author and reviewer are both Opus 5.5 (separate sessions). Not Codex reviewed.
Review: `docs/reviews/OPUS-review-periodic-residual-1b1ff30-2026-10-04.md`, reviewer commit 5871e1d (cherry-picked as f70b8d5). Reviewer verdict: non-blocking.

| Finding | Severity | Disposition |
|---|---|---|
| F1 nested architecture aliased to the module constant | low, real bug | ACCEPTED, reviewer fix `copy.deepcopy` + failing-then-passing test kept (f70b8d5) |
| F2 contrast 1 also changes dropout 0 → .1 | medium | ACCEPTED. Note wording fixed. Optional arm ADOPTED before any fit: `periodic_counts_d10` (exact P30/P31 entry) → 6 arms × seeds 16–18 = 18 fits; contrasts now backbone_bundle, backbone_structure (rate .1 held, position differs), head_dropout, input, embedding_type, embedding; registry checks the exact changed factor set (`ISOLATES`) over 4 factors (input, backbone, dropout_rate, dropout_position) |
| F3 raw column order not pinned | medium | ACCEPTED, reviewer test kept |
| F4 docstring overclaims continuous-map refusal | low | ACCEPTED: docstring fixed; the runner comparison of loaded `context_continuous` with the registered arm (+ same-length tampered archive test) is a required item of the wiring step |
| F5 RNG / init wording | low | ACCEPTED: note now states linear arm path and bit-identical backbone init (reviewer test) |
| F6 no flatten_counts, 2 independent residual contrasts | info | ACCEPTED as wording: note states 4 = 5 + 6 and that no interaction is estimated. flatten_counts arm NOT added (scope; the question is the two parents) |
| F7 timing is fit.json training mean | info | ACCEPTED: wording fixed; budget now 18 × ~200 s ≈ 3,600 worker-s estimate (new arms unmeasured) |
| F8 load-path checks | info | no change |
| F9 LR init not checked against package | info | kept as known limit |

Tests after disposition: `cd experiments/pitchmdp && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /Users/song/Projects/pitcheezy/.venv/bin/python -m pytest tests/test_direct_embed_residual_registry.py tests/test_direct_periodic_residual_model.py tests/test_run_ml_direct_periodic.py tests/test_run_ml_direct_residual.py -q -p no:cacheprovider` → 60 passed.

The registry change (a690350 + this disposition) was not part of the reviewed diff; it goes into the next review together with the DRAFT config and runner/scorer wiring. No fit, server or GPU use so far.
