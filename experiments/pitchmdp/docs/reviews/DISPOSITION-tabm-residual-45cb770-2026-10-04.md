# Disposition — same-model review of TabM-residual adapter 45cb770

2026-10-04 13:56 KST, delegated Opus PM worker (v3 cycle-004, T32b). Review: `OPUS-review-tabm-residual-45cb770-2026-10-04.md`
(review branch `4e86e13`, cherry-picked here as `fc8b4af`; reviewer fix/tests `8b827b6` → `a98e26a`). Status after this
disposition: `same_model_reviewed; Codex_review_pending`. No wiring, smoke, fit or score of this family exists yet.

| # | finding | action |
|---|---|---|
| 1 | smoke-time stop rule unmeasurable (smoke = 8,192 rows × 2 epochs) | accepted: fixed projection formula in the design (Amendment 1) |
| 2 | cap 60 needs family-scoped training; `tabm` route reads `head_dropout`; factory must check continuous map | accepted: "Wiring requirements" section; carried into the wiring review |
| 3 | contrast vs `tabm_counts` not one-factor | accepted: arm 5 `tabm_counts_d10` (P31 arm unchanged) → 25 fits, budget ≈ 15,000 worker-s; both backbone contrasts labeled "not one-factor" |
| 4 | malformed archives raised TypeError/KeyError | fixed by reviewer R1 (kept) |
| 5 | inherited P31 caveats missing | accepted: "Inherited caveats" section |
| N1 | inherited `backbone()` returns features | applied: override raises NotImplementedError + test |
| N2 | `n_classes` not load-checked | applied: added to `expected` + test; mutation (guard removed) → that test FAILS, restored |

Checks: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/test_direct_tabm_residual_model.py tests/test_direct_tabm_model.py
tests/test_direct_tabm_model_review.py tests/test_direct_periodic_residual_model.py -q -p no:cacheprovider` → 65 passed (2.93 s).
Frozen modules untouched (`git diff 7313366 HEAD --stat` = new files only).

Note for the wiring child already running from 45cb770 (`impl-tabm-residual-wiring-45cb770`): its output targets 4 arms and the
pre-disposition module hash. Its disposition must add arm 5, pin the post-disposition module hash and satisfy "Wiring requirements".
