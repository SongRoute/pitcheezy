# OPUS review — TabM-mini shared-weight ensemble head (f03bd04)

Label: `same_model_review; Codex_review_pending`
Reviewer: Claude Opus 5.5 (same model as the author, distinct session). No other model or Claude process was called.
Target: `a78ba3a..f03bd04` on `codex/tabm-mini-review-1004`:
`experiments/pitchmdp/pitchmdp/direct_tabm_model.py`, `experiments/pitchmdp/tests/test_direct_tabm_model.py`,
`experiments/pitchmdp/docs/DRAFT-TABM-MINI-DESIGN-20261004.md`.
Official reference read locally: `/tmp/tabm_official.py` (yandex-research/tabm main 28e47ae) and `/tmp/tabm_readme.md`.
Frozen modules `direct_periodic_model.py` / `direct_dropout_model.py`: not edited by f03bd04 (diff stat lists only the three new files).

## Disposition

- **Module + tests (f03bd04): non-blocking.** No correctness bug found. The adaptation matches official TabM-mini on every structural point checked; one undisclosed deviation (raw-column chunk granularity) and several disclosure gaps.
- **Proposed family (design note): blocking before registration** on finding 1 (seed reuse). This review does not block the DRAFT module commit.

## Faithfulness to official TabM-mini (checklist)

| Point | Official (28e47ae) | f03bd04 | Disclosed? |
|---|---|---|---|
| Non-shared params | `ElementwiseAffine((k, d_in), bias=False)` in front of shared MLP (`MLPBackboneMiniEnsemble`, l.1297–1350) | `scaling [k, d_in]`, `x[:, None] * r`, no bias | yes |
| Shared blocks | `Linear → act → Dropout`, `n_blocks=2` when num embeddings | frozen `Linear(D,256) GELU Dropout Linear(256,256) GELU Dropout` | yes (GELU, 256) |
| Dropout masks | applied on `[B, k, d]` → independent per member | same (`h` is `[N, k, 256]`) | n/a, matches |
| Scaling init | `normal` if num embeddings else `random-signs`; chunks = `d_features` (l.1632–1660) | `normal`, chunks from `feature_chunks` | partly — see finding 2 |
| Chunk order | official concat: num embeddings, then cat one-hots | raw first, then `embed(...).flatten(1)` (feature-major); chunks list follows that order | order immaterial; alignment verified by new test R2 |
| Output | `LinearEnsemble(d_block, d_out, k)`, weight and bias `U(±d_in^-½)` (l.495–499) | `output_weight [k,256,10]`, `output_bias [k,10]`, `U(±1/√256)` | yes |
| Train loss | README l.226–229: mean of member losses | `member_loss` = mean over k of CE | yes |
| Prediction | README: average probabilities for classification | `log mean_k softmax` | yes |
| Early stop | TabM monitors the ensemble | `self.logits(calibration)` = ensemble | yes; verified by new test R3 |
| Input to members | default "conventional MLP-like" `(B, D)` | same batch for all members | yes (and it is the official default, so not a deviation) |
| k / width / act | k=32, d_block 512, ReLU, dropout .1 | k=8, 256, GELU, dropout {0, .1} | yes |
| Optimizer | README: AdamW lr .002, wd .0003, applied to all params incl. `r` | AdamW lr .0005, wd .01 on all params incl. `r` | only as "our AdamW loop" — finding 8 |

## Findings

### 1. MEDIUM (blocking for registration) — family seeds 0–2 reuse P27's already-read selection seeds
`experiments/pitchmdp/docs/DRAFT-TABM-MINI-DESIGN-20261004.md:46-50`

- The note fixes seeds `[0, 1, 2]` because the runner (`scripts/run_cuda_direct.py:153`) only accepts 0–2 for non-P30 configs. P27 (`configs/DRAFT-CUDA-DATA-COMBO-RUN-20261004.json`, seeds `[0, 1, 2]`) already fit and **read** `control` and `periodic_counts` on those exact seeds and export, and the note itself says `counts` was chosen after reading P27.
- Concrete failure: arm 2 `periodic_counts` (and `control`) on seeds 0–2 are re-fits of the fits whose results selected `counts`. Any seed-specific luck in those fits is kept in the baseline. So the headline contrast `tabm_counts − periodic_counts` is computed against a selection-biased baseline (winner's curse on seed noise). It is not a fresh-seed comparison. P30 moved to fresh seeds 3–7 for exactly this reason.
- The line "`periodic_counts_d10 − periodic_counts` … repeats a P30 contrast on seeds 0–2" is inaccurate. P30 ran seeds 3–7, so on seeds 0–2 the d0 arm repeats P27 fits and the d10 arm is a new fit.
- Suggested change: give the TabM family its own fresh seed block not used by P27/P29/P30 (e.g. `[13, 14, 15]`), with a torch-free `check_seeds` like `direct_workload_dropout_registry.check_seeds`. Or, at minimum, add the following to the note:
  ```diff
  -Seeds: [0, 1, 2] (current runner seed rule for non-P30 families) → 15 fits.
  +Seeds: [0, 1, 2] (current runner seed rule for non-P30 families) → 15 fits.
  +`control` and `periodic_counts` on seeds 0–2 re-fit the P27 fits whose read results selected `counts`;
  +contrasts against them inherit that selection and are NOT fresh-seed evidence.
  ```
  and fix the "repeats a P30 contrast" wording.

### 2. LOW — undisclosed deviation: per-column chunks for raw (one-hot / unembedded numeric) inputs
`experiments/pitchmdp/pitchmdp/direct_tabm_model.py:66-68, 101-102`; docstring `:19-22`

- In the official package, every numerical feature goes through the embedding when `num_embeddings` is set. Each categorical feature is **one** chunk of size = cardinality (`d_features.extend(cat_cardinalities)`), and init is `normal`. Without embeddings the init is `random-signs`. The official package never builds the mixed case.
- Here every raw scalar is its own size-1 chunk with `normal` init. That covers the per-token pitch-type/outcome one-hot channels (5 × 30), the 38 current-token channels (8 physics + 11 outcome channels are always 0, so their 19 scaling columns are dead parameters), the 6 mask bits, and 12 raw context columns on the `counts` shape (game-state ordinals, flags, `left_share`).
- Effect: each one-hot column gets an independent per-member N(0,1) scale. That is a richer per-member categorical re-weighting than the official "one value per categorical feature". It also means some members start with near-zero or negative scale on the mask bits and ordinals. This is not a bug and is plausibly harmless, but it is not in the "NOT a faithful reproduction" list.
- Suggested docstring addition:
  ```diff
  -our AdamW/early-stop loop and ten-class head, no tuning, same mini-batch for all
  -members (the package's default "conventional MLP-like" input mode).
  +our AdamW/early-stop loop and ten-class head, no tuning, same mini-batch for all
  +members (the package's default "conventional MLP-like" input mode).  Raw (unembedded)
  +columns, including one-hot groups, get one size-1 'normal' chunk per column (the
  +package uses one chunk per categorical feature and never mixes raw numeric with
  +embedded features); raw columns precede the embedded block (package: embeddings first).
  ```

### 3. LOW — test adequacy: the k=1 wrapper never exercises `PeriodicTabMNetwork`; the chunk test is self-referential
`experiments/pitchmdp/tests/test_direct_tabm_model.py:124-156, 61-76`

- The k=1 wrapper test does establish the **loop** equivalence: identical `train_weighted_nll` lists, early-stop NLL within 1e-6, same best epoch, bitwise-equal parameters, at dropout 0 and .1. That covers seeding, batching, clipping, AdamW and best-state restore. With k=1, `ensemble_logits` = `log_softmax`, which differs from raw logits only by a per-row shift, so the early-stop score is unchanged.
- It wraps the frozen network, though, so it says nothing about the TabM forward (`direct_tabm_model.py:108-114`). That forward is a hand copy of the frozen `PeriodicDirectNetwork.forward`, and no test compares the two. If the frozen forward changes, the copies would drift silently.
- `test_scaling_init_is_one_normal_value_per_member_and_feature_chunk` takes its chunk boundaries from `feature_chunks` itself. **Demonstrated:** monkeypatching `feature_chunks` to the wrong order (`[d]*n_emb + [1]*n_raw`) leaves both author tests passing (`2 passed`). The new review test R2 fails on it (`1 failed`).
- Also missing: no test shows that early stopping uses the ensemble NLL rather than mean member CE. The k=1 case cannot tell them apart.
- Fixed by reviewer tests R1–R3 (below).

### 4. LOW — calibration semantics: temperature acts on the ensemble (power calibration), not on members
`direct_tabm_model.py:34-36, 164-168`; `pitchmdp/matrix_models.py:218-219`; `pitchmdp/matrix_benchmark.py:72-78`

- `MatrixModel.predict` = `softmax(log p̄ / T)` = `p̄^{1/T} / Σ p̄^{1/T}`. This is well-defined: `ensemble_logits` is finite whenever member logits are finite, because log_softmax and logsumexp are stable, and it is invariant to a per-row shift. It is also a monotone power calibration of the averaged probability. Verified numerically by new test R4.
- `integrated_probabilities` divides by T, softmaxes, then averages over draws. That is consistent: draws are averaged after the ensemble.
- It is not the same as temperature-scaling each member's logits and then averaging. A reader of the score tables could assume the latter. Suggest one line in the docstring and design note: "T is applied to log p̄ (power calibration of the member-averaged probability)."

### 5. LOW — `train_weighted_nll` in history is mean member CE, not the prediction's NLL
`direct_tabm_model.py:203-214`

- By Jensen, mean member CE ≥ ensemble NLL. In this family the train and early-stop curves are on different quantities, unlike every single-network family whose report uses the same key.
- Concrete failure: a reader comparing `train_weighted_nll − earlystop_conditional_nll` across arms reads the TabM gap as larger overfitting than it is.
- Renaming the key would break the shared report schema. Suggest documenting it instead (docstring line plus a `report['train_loss'] = 'mean_member_ce'` field. `architecture['train_loss']` already carries this, so a docstring line may be enough).

### 6. LOW (efficiency) — `logits()` materializes `[N, k, 10]` on host before reducing
`direct_tabm_model.py:154-168`

- Host memory is k× the base model: 320 B/row at k=8 vs 40 B/row. As an **estimate**, a 1,000,000-row early-stop or DEV set would hold about 320 MB, and this happens every epoch.
- Suggested change: reduce per batch on device, keeping `member_logits` for tests and diagnostics:
  ```diff
   def logits(self, arrays, batch_size=4096):
  -    member = self.member_logits(arrays, batch_size)
  -    if not len(member):
  -        return np.empty((0, 10), dtype=np.float32)
  -    return ensemble_logits(torch.as_tensor(member)).numpy()
  +    arrays = periodic._no_current_physics(arrays)
  +    self.net.eval()
  +    outputs = []
  +    with torch.no_grad():
  +        for start in range(0, len(arrays[0]), batch_size):
  +            inputs = [torch.as_tensor(a[start:start + batch_size], device=self.device) for a in arrays]
  +            outputs.append(ensemble_logits(self.net(*inputs)).cpu().numpy())
  +    return np.concatenate(outputs) if outputs else np.empty((0, 10), dtype=np.float32)
  ```
  Not applied: it touches the author's module and is not needed for correctness. Device-side reduction can differ from the CPU path in the last float32 bit.

### 7. NIT — unused import, bool `k`
- `direct_tabm_model.py:56` imports `_validate`, which is unused (the guard calls it).
- `DirectTabMModel(k=True)` passes `int(k) != k` and builds k=1. This is harmless.

### 8. NIT — optimizer numbers not in the deviation list
`direct_tabm_model.py:19-22`, design note `:23-24`

- AdamW weight decay .01 also applies to the scaling `r`, pulling it toward 0 rather than 1. The official default does the same with uniform wd .0003, so ours is about 33× stronger, and lr is .0005 vs .002.
- The note says only "our AdamW/early-stop loop". Suggest stating `lr .0005, wd .01 (package default .002 / .0003), wd applies to r`.

## Correctness checks that passed (no finding)

- **fit mirror:** `diff` of `DirectResidualModel.fit` (`direct_residual_model.py:84-140`) vs `DirectTabMModel.fit` (`:170-229`) shows exactly three changed lines: the guard call (`_no_current_physics` instead of `_validate`; the guard calls `_validate` itself, the same path as `DirectPeriodicModel.fit`), the loss line (`member_loss`), and one comment. Everything else is identical (seed, `set_num_threads(4)`, AdamW, clip 5, `np.array_split` batching, early-stop tolerance 1e-5, patience, best-state restore, report).
- **Early stop** uses `self.logits` (eval mode, ensemble). `logsumexp(log p̄) ≈ 0`, so the score is the ensemble conditional NLL. Confirmed by R3: the history value at `best_epoch` matches the recomputed ensemble NLL within 1e-5, and mean member CE is strictly larger.
- **Zero-current-physics guard** runs in fit (train and early stop), `member_logits`, `logits`, and therefore `predict`.
- **Padding invariance:** `masked_fill` on tokens plus the embedding `keep` mask. Author test sets padded values to 123.
- **Device:** parameters are initialized on CPU and moved with `.to(device)`. Member logits are moved with `.cpu()` and the ensemble is reduced on CPU in float32. `einsum('nkd,kdc->nkc')` and the 3-D `nn.Linear` are device-agnostic. MPS/CUDA were not run (out of scope).
- **save/load:** `network_config` holds every constructor argument. Load rebuilds the model from the archive's architecture, then refuses on format, kind, architecture mismatch, or config/architecture mismatch on width, frequencies, d_embedding, sigma, head_dropout, k and scaling_init. R5 adds a tamper test for each of: config `head_dropout`, config `scaling_init`, architecture `k`, architecture `ensemble`, `kind`.
- **Parameter count** (measured by constructing the networks on CPU, real `counts` shape: n_context 53, n_token 38, length 6, 41 continuous):
  - `d_in` = 854 (206 raw + 81 × 8 embedded)
  - single periodic: 288,802
  - TabM k=8: 313,624 (+24,822 = 8·854 + 8·2560 + 80 − 2570)
  - `parameter_count()` agrees with `net.parameters()`

## Resource estimate (k=8, real shapes)

- **Measured (Mac M4 CPU, 1 thread, batch 1024, real `counts` shape, 3 steps):**
  - single periodic fwd+bwd: 0.012 s/step
  - TabM k=8 fwd+bwd: 0.035 s/step (about 2.9×)
  - The embedding (`[N, 81, 32] → 8`) is computed once and shared, so the k× cost is confined to the head.
- **GPU time: 미측정 (not measured).** The note's "expected above P27's 84 s mean" is plausible but unquantified. The smoke run must measure it.
- **Memory (estimate only, float32):**
  - backbone input `x_k` at batch 1024 = 1024·8·854·4 B ≈ 28 MB
  - hidden activations ≈ 6 × 1024·8·256·4 B ≈ 50 MB
  - total training activations estimated at < 150 MB/step
  - eval batch 4096: `x_k` ≈ 112 MB
  - host member logits 320 B/row (finding 6)
  - This is negligible against a 24 GB A5000, but the GPU is shared, so the smoke run should record peak memory.

## Design-note framing (scientific)

- **Arms and contrasts:** five arms. Each TabM contrast changes one factor, the head (single → TabM-mini k=8), at fixed dropout. The dropout contrast is separate. Fine apart from finding 1.
- **Compute-unmatched saved-probability comparison** is flagged explicitly: 1 TabM fit vs 3 fits, with member count and seconds reported next to every number. Good. Suggest also saying that this T1 comparison is descriptive only and not a contrast.
- **Exposure:** it says DEV/MLB DEV are repeatedly exposed, `counts` was chosen post-P27, results are post-selection exploratory, there is no winner/promotion/verdict field, G0/D131/D155 are unchanged, and no 2026 data is used. Nothing in the note claims unseen validation.
- **One wording risk:** "Fixed contrasts (calibrated, same seeds, game-block CI)" could be read as confirmatory. With finding 1 those seeds are not fresh, so state "exploratory, selection-exposed" next to the contrast list.
- **Timing claim:** the note says it was committed at 07:44 KST before any P30 score was read. This review cannot verify it beyond the commit time. P30 does not feed the TabM arms.

## Reviewer changes (each listed separately)

1. **Added** `experiments/pitchmdp/tests/test_direct_tabm_model_review.py`, 9 tests (5 functions, one parametrized ×5). All pass on f03bd04 unchanged; they close gaps and are not failing-then-fixed. No module code was changed.
   - **R1** `test_unit_scaling_k1_forward_equals_frozen_network`: TabM forward with k=1, r=1 and the frozen output layer copied in equals `PeriodicHeadDropoutNetwork` forward (atol 1e-6, same init seed).
   - **R2** `test_scaling_chunks_align_with_the_features_actually_concatenated`: a forward hook on `head[0]` checks two things. Perturbing one continuous context column changes exactly one 8-wide chunk at `n_raw + (40+i)·8`, and perturbing one raw context column changes exactly one raw position. It catches the chunk-order mutation (demonstrated above).
   - **R3** `test_early_stop_monitors_ensemble_nll_not_mean_member_ce`
   - **R4** `test_temperature_on_log_mean_probability_is_a_valid_power_calibration`
   - **R5** `test_load_refuses_each_tampered_field` (×5)
2. **Added** this review file.

## Tests run (Mac, single-threaded, focused only)

```
$ cd experiments/pitchmdp && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest tests/test_direct_tabm_model.py -q -p no:cacheprovider
..............                                                           [100%]
14 passed in 1.72s

$ ... -m pytest tests/test_direct_tabm_model.py tests/test_direct_tabm_model_review.py -q -p no:cacheprovider
.......................                                                  [100%]
23 passed in 1.72s

$ ... -m pytest tests/test_cuda_dropout.py -k "network or zero_dropout or guard" -q -p no:cacheprovider
.........                                                                [100%]
9 passed, 23 deselected in 1.42s

# mutation probe: feature_chunks monkeypatched to [d]*n_emb + [1]*n_raw (wrong order)
author tests -k "scaling_init or forward": 2 passed      (not caught)
review test  -k chunks_align:              1 failed      (caught)
```

Not run: full suite, GPU/CUDA, ssh, anything production-facing.
