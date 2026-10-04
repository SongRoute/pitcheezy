# DRAFT — TabM-mini adapter on the periodic_residual backbone (pre-registration)

Status: **DRAFT pre-registration, written before any module code, wiring, smoke or fit of
this family** (2026-10-04, delegated Opus PM worker, v3 cycle-003, task T31). Not
registered; no GPU run; execution disabled. Same-model review pending. A full run needs
Song's go (`결정 필요` raised in v3 cycle-002) after review + CUDA smoke.

**Amendment 1 (2026-10-04 13:56 KST, v3 cycle-004, T32b)** — after the same-model review
`OPUS-review-tabm-residual-45cb770-2026-10-04.md` (findings 1, 2, 3, 5) and BEFORE any
wiring disposition, smoke, fit or score of this family: (a) arm 5 `tabm_counts_d10` added
(25 fits) so the backbone contrast has dropout .1 on both sides; (b) the smoke-time stop
rule is replaced by a fixed projection formula; (c) wiring requirements written down;
(d) inherited P31 caveats listed. Original text kept where unchanged; diff in git.

Evidence already read when this was written (so the design is post-selection): P30–P33
scores, consolidated report `reports/research-p30-p33/REPORT-P30-P33-consolidated-ko.md`
(PM worktree `codex-continuous-pm-1004`, `d39285f`), SAVED-ENSEMBLE-P31/P32/P33 results.
The two single-fit directions with the largest paired Δ there were TabM-mini (P31) and
periodic embedding + residual backbone (P32). That is why this combination is proposed.

## Question

With identical input (periodic PLR-lite embedding + workload `counts`), data, training
loop, epoch cap and calibration chain:

1. Does adding the TabM-mini shared-weight ensemble (k=8 input scaling + per-member output,
   mean member loss, mean member probability) to the periodic_residual backbone change
   calibrated direct NLL / Brier / calibration? (`tabm_residual_counts − periodic_residual_counts`)
2. Under TabM-mini, does replacing the flatten-MLP head by the residual backbone change them?
   Read mainly as `tabm_residual_counts − tabm_counts_d10` (dropout .1 on both sides; still a
   bundle of backbone + member-head input width 256→128 + dropout placement: inside the
   residual blocks vs after the head GELUs). `tabm_residual_counts − tabm_counts` is also
   reported and additionally differs in dropout (.1 vs 0). Neither is one-factor.

Secondary, descriptive only: one TabM fit vs equal-weight means of n separately trained
single networks of the same family (saved probabilities; n fixed below).

## Primary sources

- Gorishniy, Kotelnikov, Babenko, TabM, ICLR 2025, arXiv:2410.24210; official package
  `yandex-research/tabm` `tabm.py` (commit `28e47ae`, read 2026-10-04; local copy
  `pm-jobs/20261004-continuation/continuous-children/sources/tabm_official.py` fc654af6…).
  TabM-mini = one ElementwiseAffine (k × d_in, no bias) before a fully shared backbone,
  per-member output layer, mean of member losses, mean of member probabilities.
- Gorishniy, Rubachev, Babenko 2022, numerical embeddings (arXiv:2203.05556): embeddings
  with MLP and ResNet backbones.
- **Our adaptation, not in the package:** the package's TabM backbone is an MLP; here the
  shared backbone is our frozen ResNet-style `residual_mlp` (input Linear → 2 pre-LN residual
  blocks → LayerNorm → GELU). The k-scaling acts on the backbone input, the per-member
  output replaces only the final Linear(128, 10). Not a faithful reproduction; no claim
  that paper benchmark results transfer.

## Network (fixed now)

```
raw, e  = as PeriodicResidualNetwork (frozen embedding, same columns and padding mask)
x       = cat(raw, e.flatten(1))                                   # [N, D]
x_k     = x[:, None, :] * r                                        # r [k, D], no bias, 'normal', one value per chunk
z       = Linear(D, 128)(x_k)                                      # shared (frozen input layer)
2 x z   = z + Linear(256,128)(Dropout(.1)(GELU(Linear(128,256)(LayerNorm(z)))))   # shared frozen blocks
h       = GELU(LayerNorm(z))                                       # shared frozen output norm/act
member  = einsum('nkd,kdc->nkc', h, W) + b                         # W [k,128,10], b [k,10], U(±1/sqrt(128))
```

Chunks as `direct_tabm_model.feature_chunks` (one per raw scalar, d_embedding per embedded
scalar). The frozen output Linear is created (same RNG order as periodic_residual) and then
discarded, so for one seed every shared parameter equals the periodic_residual
initialisation; r, W, b are drawn afterwards. Training = `DirectTabMModel.fit` (mean member
CE; tested mirror of the frozen loop). Ensemble logits = log mean member softmax; external
temperature acts on that (as P31). k=8, scaling init `normal`, lr .0005, wd .01, AdamW,
batch 1024, patience 5.

## Family (fixed now; registered only after review + CUDA smoke + Song's go)

Arms (export `49097109`, workload `counts`, continuous map = P30 `CONTINUOUS['counts']`):

1. `control` — flatten MLP 128, fresh (runner requirement)
2. `periodic_residual_counts` — EXP-P32-001 arm unchanged
3. `tabm_counts` — EXP-P31-001 arm unchanged (k=8, head dropout 0)
4. `tabm_residual_counts` — new (above)
5. `tabm_counts_d10` — EXP-P31-001 arm unchanged (k=8, head dropout .1) (Amendment 1)

Seeds **[24, 25, 26, 27, 28]** → 25 fits (Amendment 1; was 20). Disjoint from every earlier/reserved block
(0–2 P24/P25/P27/P29, 3–7 P30, 8–12 reserved P30 repeat, 13–15 P31, 16–18 P32, 19–23 P33).
New RNG seeds are not a new evaluation population.

**Epoch cap 60 for every arm (decided now, before any fit of this family).** P31: 3/6 TabM
fits stopped at the 30 cap; all single networks in P31/P32 stopped by patience 5 by epoch
19. There is no learning-rate schedule (cap only truncates), so a 60 cap leaves any run
that stops by patience before epoch 30 unchanged in algorithm. All within-family contrasts
share the cap; numbers are NOT compared with P31/P32 (different cap, seeds, run batch).
Cap hits are reported per fit.

Fixed contrasts (calibrated, same seeds, seed-mean ensemble and single-fit means, game-block
bootstrap; exploratory, post-selection, repeated DEV):
- `tabm_residual_counts − periodic_residual_counts` (TabM-mini on the residual backbone)
- `tabm_residual_counts − tabm_counts_d10` (backbone + width + dropout-placement bundle under
  TabM-mini, dropout .1 both sides; not one-factor)
- `tabm_residual_counts − tabm_counts` (same bundle + dropout .1 vs 0; not one-factor)
- `tabm_counts − control`, `tabm_counts_d10 − control`, `periodic_residual_counts − control`,
  `tabm_residual_counts − control`

Saved-probability grid (to be committed as a file BEFORE any score of this family is read):
for each TabM arm, its 1-fit and 5-fit means vs means of n ∈ {1, 2, 3, 5} fits of
`periodic_residual_counts` and of `control`, all seed subsets fixed in the file; measured
worker-seconds printed next to every row; "compute-matched" only where measured time ratios
support it.

Budget: expected per fit ≈ control 85 s, periodic_residual 220 s (P32), tabm_counts and
tabm_counts_d10 ≤ 750 s each at a 60 cap (P31 ≈ 12.5 s/epoch), tabm_residual unknown
(estimate ≤ 1200 s; projected from smoke by the fixed formula below). Upper estimate
5 × (85 + 220 + 750 + 750 + 1200) = 15,025 ≈ 15,000 worker-s, ≈ 125 min wall on GPU3/4
(Amendment 1; was ≈ 11,300 / 95 min). Family
cap 36000 worker-s, task cap 3600 s, ≤ 2 attempts per task, newsrv GPU3/4 only
(UUID-pinned), max two fits at once. Explicit family deadline set in the registered config
on the day of launch.

Stop: any task failing twice aborts the family (no partial comparison, failures reported);
TabM-residual projected full-fit time > 1800 s → redesign before registration (not silently
shorten). Projection fixed now (Amendment 1; the runner smoke uses 8,192 TRAIN rows and 2
epochs, so it never measures a full fit): projected seconds = (smoke train seconds / smoke
optimizer updates) × 1,224 updates/epoch × 60 + measured non-train smoke seconds, per GPU,
using the slowest of the 5 smoke seeds. No extra arms or seeds after scores are read.

Exposure: DEV/MLB DEV are the same repeatedly exposed ≤2025 rows used for P24–P33; `counts`
and both parents were chosen after reading earlier results. Descriptive only: no winner,
promotion or verdict fields. G0, D131/D155 unchanged. No 2026 data.

## Inherited caveats to state in the result (Amendment 1; from `direct_tabm_model.py`)

- `history.train_weighted_nll` is the mean member CE, so its gap to
  `earlystop_conditional_nll` is not comparable with single-network arms.
- The scaling r and the per-member heads get weight decay .01 (package default .0003).
- External temperature acts on log mean member probability.

## Wiring requirements (Amendment 1; review finding 2)

- Family-scoped training constant `{epochs: 60, ...}` checked by the new registry; the runner
  keeps 30 for every earlier family (no global edit; earlier source hashes unaffected). The
  two P31 `tabm_*` arms in this family run under the same family-scoped cap.
- New arm key `tabm_residual` (not `tabm`): the existing TabM route reads
  `a['head_dropout']`, which this archive does not have.
- The new factory compares k, scaling init and the registered continuous map after load
  (a same-length permuted map would otherwise load silently).
- Module hash pinned AFTER the review disposition (R1 + N1/N2 change it).

## Next steps

1. Additive module `pitchmdp/direct_tabm_residual_model.py` + focused CPU tests (frozen
   modules untouched) → distinct Opus review → disposition.
2. Torch-free registry/factory/DRAFT config/runner dispatch (pattern of P31/P32) → review.
   *Implemented DRAFT (2026-10-04, delegated Opus worker, branch
   `codex/tabm-residual-wiring-1004`, unreviewed; same_model_review pending;
   Codex_review_pending):* `pitchmdp/direct_tabm_residual_registry.py` (4 arms above,
   earlier arms = exact EXP-P31-001 / EXP-P32-001 entries, own seed check exactly
   [24..28], 20 fits, training = cuda_direct_v1 budget with epochs 60, the 5 fixed
   contrasts above in this order), `pitchmdp/direct_tabm_residual_factory.py` (strict
   build/load incl. post-load continuous-map check), `configs/DRAFT-CUDA-TABM-RESIDUAL-20261004.json`
   (DRAFT, execution disabled, placeholder family deadline), runner/scorer dispatch
   (config epoch cap reaches `fit(epochs=...)`; fit.json `epoch_cap` = cap, epochs_run,
   best_epoch, stopped_by; scorer checks it before any prediction read and lists cap
   hits per fit). Arms, seeds, cap and contrasts unchanged. Record:
   `docs/WIRING-TABM-RESIDUAL-20261004.md`.
   *Wiring disposition (T33, v3 cycle-005): wiring above was written from pre-disposition
   `45cb770` with 4 arms; merged onto `0d8a65b` and extended to Amendment 1 (arm 5
   `tabm_counts_d10`, 25 fits, post-disposition module hash). See
   `docs/reviews/DISPOSITION-tabm-residual-wiring-289fd68-2026-10-04.md`.*
3. CUDA smoke of all 25 tasks on GPU3/4 (runner SMOKE: 8,192 TRAIN rows, 2 epochs) → identities pinned → Song's go → registration.
