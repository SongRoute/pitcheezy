# Disposition — Opus review of TabM-mini head f03bd04

Label: `same_model_reviewed; Codex_review_pending`. Author and reviewer are both Claude Opus 5.5, in different sessions. No Codex review has happened yet.
Review: `docs/reviews/OPUS-review-tabm-mini-f03bd04-2026-10-04.md`. It was cherry-picked from `35456f2` as `367266c` and includes the reviewer's test file `tests/test_direct_tabm_model_review.py`.
Author's disposition: cycle-004, starting 2026-10-04 07:53:27 KST. No TabM fit was run and no P30 score was read before this disposition.

| # | Severity | Disposition |
|---|---|---|
| 1 | MEDIUM, blocks registration | **Fixed in the design note.** The family now uses fresh seeds [13, 14, 15]. These do not overlap any seed block already read or reserved. Server configs checked on 2026-10-04: P24/P25/P27/P29 use 0–2, P30 uses 3–7, and 8–12 is reserved for the P30 repeat. The P30 sentence is corrected: P30 used 3–7, so 13–15 means new RNG seeds on the same rows, not new data. The contrasts are labeled "exploratory, selection-exposed". The runner still accepts only 0–2 for non-P30 configs, so the registry's own `check_seeds` is the next unit. |
| 2 | LOW | **Disclosed** in the module docstring and the design note. Each raw column, including one-hot columns, gets its own size-1 `normal` chunk. Behavior is unchanged. |
| 3 | LOW | **Fixed by the reviewer tests R1–R3**, which are now in the branch. |
| 4 | LOW | **Disclosed** in the docstring and note: T is applied to log p̄ (power calibration), not to each member. |
| 5 | LOW | **Disclosed** in the docstring and note: `train_weighted_nll` is the mean member CE. The key is kept so the shared report schema does not change. `architecture['train_loss']='mean_member_ce'` was already in the report. |
| 6 | LOW (efficiency) | **Applied.** `logits()` now reduces each batch on the host after `.cpu()` instead of materializing `[N, k, 10]`. The new test `test_batched_ensemble_logits_equal_reducing_all_member_logits` checks bitwise equality with `ensemble_logits(member_logits(x, bs))` at the same batch size for bs ∈ {1, 7, 4096}, and checks that empty input returns `(0, 10)` float32. Measured caveat: comparing across batch sizes gives a maximum difference of 2.4e-7. That comes from the network forward pass, which also existed before this change, not from the reduction. The k=1 bitwise fit-mirror test still passes. |
| 7 | NIT | **Fixed.** The unused `_validate` import is removed. `k=True` is now refused, and `{'k': True}` was added to the invalid-settings test. |
| 8 | NIT | **Disclosed.** lr .0005 and wd .01 on all parameters including r (package uses .002 / .0003). |

Tests, run on Mac with one thread and focused files only. The interpreter is the main checkout's `.venv`. Imports resolve to this worktree, verified with `pitchmdp.direct_tabm_model.__file__`.
```
$ cd experiments/pitchmdp && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /Users/song/Projects/pitcheezy/.venv/bin/python -m pytest tests/test_direct_tabm_model.py tests/test_direct_tabm_model_review.py -q -p no:cacheprovider
25 passed in 1.63s
$ ... -m pytest tests/test_cuda_dropout.py -k "network or zero_dropout or guard" -q -p no:cacheprovider
9 passed, 23 deselected in 1.32s
```
Not run: the full suite, GPU/CUDA, and anything on the server. The module is still DRAFT: it is not registered and not in a runner.
