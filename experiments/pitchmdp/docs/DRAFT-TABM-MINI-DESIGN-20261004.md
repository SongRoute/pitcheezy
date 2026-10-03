# DRAFT — TabM-mini shared-weight ensemble head (eight-axes A4)

Status: **DRAFT design + model module only.** Not registered, not in the CUDA runner,
no GPU run, no score read. Same-model (Opus 5.5) review returned 07:50 KST
(`docs/reviews/OPUS-review-tabm-mini-f03bd04-2026-10-04.md`), dispositions in
`docs/reviews/DISPOSITION-tabm-mini-f03bd04-2026-10-04.md`: `same_model_reviewed;
Codex_review_pending`. Revised 2026-10-04 (cycle-004) before any TabM fit or P30 score was read.
Committed 2026-10-04 07:44 KST (`date`), BEFORE any P30 score was read by this worker
(P30 was 13/50 fits done at 07:41:07 KST server time, unscored).

## Question

With the same input (periodic embedding + workload `counts`), the same data, the same
training loop and the same calibration chain, does replacing the single output MLP with a
TabM-mini k=8 shared-weight ensemble change calibrated direct NLL / Brier / calibration?
Secondary (descriptive): how does one TabM-mini fit compare with the equal-weight
mean of separately trained single networks (saved-probability ensemble, T1 scorer)?

## Primary source

Gorishniy, Kotelnikov, Babenko, TabM, ICLR 2025 (arXiv:2410.24210). Official package
`yandex-research/tabm` `tabm.py` read at commit `28e47ae` (2026-10-04): TabM-mini =
one ElementwiseAffine (k × d_in, no bias) in front of a fully shared MLP, per-member
LinearEnsemble output, scaling init `normal` with one value per feature chunk when
numerical embeddings precede the backbone; mean of member losses for training, mean
of member probabilities for prediction. Package defaults k=32, d_block 512, ReLU,
dropout .1 — **ours differ** (k=8, width 256, GELU, our AdamW/early-stop loop with
lr .0005 and weight decay .01 on every parameter including the scaling r; package
default lr .002, wd .0003). Raw (unembedded) columns including one-hot groups get one
size-1 `normal` scaling chunk per column; the package uses one chunk per categorical
feature and never mixes raw and embedded inputs. The external temperature acts on
log of the member-averaged probability (power calibration of p̄), not on each member.
History key `train_weighted_nll` holds the mean member CE (≥ ensemble NLL).
Not a faithful reproduction; no claim that paper benchmark results transfer.

## Implementation (this commit)

`pitchmdp/direct_tabm_model.py` (additive; frozen periodic/dropout modules untouched).
Tests `tests/test_direct_tabm_model.py` (14, CPU synthetic): frozen module reuse and
init order, chunked scaling init, member shape and shared head, padding invariance,
mean-member loss ≠ loss of the mean, probability averaging, k=1 wrapper reproduces
`PeriodicHeadDropoutModel.fit` bitwise (dropout 0 and .1), determinism, archive
guards, invalid settings.

## Proposed family (fixed now; to be registered only after review + CUDA smoke)

Arms (all on export `49097109`, workload `counts`, continuous map = P30 `CONTINUOUS['counts']`):

1. `control` — flatten MLP, fresh (runner requirement)
2. `periodic_counts` — single periodic network, head dropout 0
3. `periodic_counts_d10` — single periodic network, head dropout .1
4. `tabm_counts` — TabM-mini k=8, head dropout 0
5. `tabm_counts_d10` — TabM-mini k=8, head dropout .1

Seeds: **[13, 14, 15]** → 15 fits. Fresh RNG block disjoint from every earlier read or
reserved block (checked 2026-10-04 on server configs: P24/P25/P27/P29 and all DRAFT CUDA
families = 0–2; P30 = 3–7; P30 optional repeat reserved = 8–12). Seeds 0–2 were rejected
(review finding 1): `control` and `periodic_counts` on 0–2 would re-fit the P27 fits whose
read results selected `counts`, inheriting that selection. The current runner only allows
0–2 for non-P30 configs, so the TabM registry needs its own torch-free `check_seeds`
accepting exactly [13, 14, 15] (next unit). A new RNG seed is not a new evaluation
population: DEV/MLB DEV stay the same, repeatedly exposed rows.
Fixed contrasts — exploratory, selection-exposed (calibrated, same seeds, game-block CI):
- `tabm_counts − periodic_counts` (ensemble head, dropout 0)
- `tabm_counts_d10 − periodic_counts_d10` (ensemble head, dropout .1)
- `periodic_counts_d10 − periodic_counts` (dropout; the same contrast P30 measures on
  seeds 3–7, here re-measured on new RNG seeds 13–15 over the same rows — not new data)
- every arm − `control`
Saved-probability (T1 scorer, after family completion; descriptive only, not a contrast): each `tabm_*` seed vs the
3-seed mean of the matching single arm (compute is NOT matched: 1 TabM fit vs 3 fits;
report member count and measured seconds next to every number).

Budget (to be measured in smoke before registration): TabM's first shared layer runs
on k× rows, so per-fit time is expected above P27's 84 s mean; family cap 36000
worker-seconds, task cap 3600 s, ≤2 attempts per task, GPUs 3/4 only. Abort the
family (no partial comparison) if any task fails twice.

Exposure: DEV/MLB DEV are repeatedly exposed; `counts` was chosen after reading P27.
Results are post-selection exploratory descriptions. No winner/promotion/verdict
fields; G0/D131/D155 unchanged; no 2026 data.

## Next steps

1. Distinct Opus review of this module (same-model) — done, dispositions applied.
2. Registry/runner/scorer integration as new additive files (`direct_tabm_registry.py`,
   DRAFT config), focused tests, second review.
3. CUDA smoke on every arm (GPU 3/4), measured seconds/memory, then registration.
