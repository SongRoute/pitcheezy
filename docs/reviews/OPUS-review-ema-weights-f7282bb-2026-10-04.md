# Opus 5.5 review — EMA weight-averaging flatten arms (eee5c64..f7282bb)

Label: `same_model_review; Codex_review_pending`
Reviewer: Opus 5.5, a separate session from the author. No other model was called and no further Claude process was started. The two cited abstracts were read from the arXiv export API with `curl` (no summarizer model).
Target: `eee5c64..f7282bb` in worktree `codex-ema-weights-review-1004` (branch `codex/ema-weights-review-1004`): `direct_ema_model.py`, `direct_ema_registry.py`, `tests/test_direct_ema_model.py`, `docs/DESIGN-EMA-WEIGHTS-20261004.md`.
Scope kept: light CPU, single thread, focused test files only. No server, GPU, ssh, ports or tmux. No result or score file was opened for scores. One `grep` hit showed the runtime line of `docs/RESULT-EXP-P32-001-20261004.md:15` (control 82.5 s, 1448 MiB, best/run epochs 6/11, 6/11, 8/13). It has no score. Only the best-epoch counts are used, in F3.

## Disposition

**Non-blocking** for building the runner/scorer dispatch, the DRAFT config and the CUDA smoke. Before registration (not before smoke), three conditions apply:
- F5: wire a `uses_ema` branch into `config_check`.
- F6: add the CUDA raw-trajectory identity check to the smoke.
- F1 and F3: change the wording in the design note.

- Fit mirror: clean. A mechanical diff of the two `fit` bodies shows only the lines marked `EMA` (output below). Nothing in the EMA path consumes RNG, the train/eval modes match the base arm, and the swap is exception-safe.
- One registry gap fixed (F4): the seeds 19–23 / 20 fits were not pinned, and comparisons were type-loose. This is a code change, listed separately below.
- Five test gaps closed (T1–T5). Three plausible mutants that the original 8 tests missed are now caught. All new tests passed on the current code, so they add coverage and did not expose a bug in the model.

## Findings

### F1 — medium (design/disclosure). The contrast is "EMA weights + EMA-driven stopping", and EMA tends to stop later
- `pitchmdp/direct_ema_model.py:120`: the 1e-5 / patience-5 rule runs on the EMA score. The EMA trails the raw weights (F3), so its early-stop curve is smoother and reaches its minimum later. The saved EMA model has therefore usually seen more optimizer updates than the control's raw best state.
- Synthetic evidence. This is the reviewer's CPU probe on synthetic arrays, not a score. 1024 rows, labels depend on context, patience 2, 12 (lr, decay, seed) settings:
  - The EMA-rule best epoch was later than the raw-rule best epoch in 11/12 settings, never earlier.
  - Epochs run differed in 8/12 settings.
  - Example: `lr .0003, decay .99`, EMA (best 7, run 9) vs raw (best 6, run 8).
- This is consistent with the stated question ("used for early stopping and as the saved model"). The registry docstring says "stopping can differ". So a redesign is **not** needed. The design note (`DESIGN-EMA-WEIGHTS-20261004.md:37`, "차이는 평가·멈춤·저장하는 가중치뿐이다") is literally true but understates that the number of updates can differ.
- Suggested text: append "그래서 멈추는 epoch와 update 수가 기준 팔과 달라질 수 있다(EMA는 보통 늦게 멈춘다). seed별 Δbest_epoch·Δoptimizer_updates를 같이 적는다." The per-arm best epoch and epochs run are already in the 보고 list. Pre-register the paired Δ as well.

### F2 — low. `raw_best_epoch_within_run` uses `min`, not the shared stopping rule
- `direct_ema_model.py:128`: `min(history, key=raw)` returns the raw argmin within the EMA run. That is not the epoch the control rule would pick:
  - it ignores the 1e-5 delta and patience;
  - it is truncated when the EMA run stops before the raw rule would.
- So it cannot be read as "the control's best epoch". The name says `within_run`, so this is not wrong, only easy to misread.
- Zero-cost improvement (proposed, not applied, because it adds report fields to the author's module):
```diff
-        raw_best = min(history, key=lambda h: h['raw_earlystop_conditional_nll'])
+        raw_best = min(history, key=lambda h: h['raw_earlystop_conditional_nll'])
+        rbest, rbest_epoch, rstale, rstop = np.inf, 0, 0, None   # replay the shared rule on the raw score
+        for h in history:
+            if h['raw_earlystop_conditional_nll'] < rbest - 1e-5:
+                rbest, rbest_epoch, rstale = h['raw_earlystop_conditional_nll'], h['epoch'], 0
+            else:
+                rstale += 1
+            if rstale >= patience and rstop is None:
+                rstop = h['epoch']
 ...
+                           raw_rule_best_epoch=rbest_epoch, raw_rule_stop_epoch=rstop,  # None: not reached in this run
```
- With that field, `history[raw_rule_best_epoch-1]['earlystop_conditional_nll']` gives the EMA early-stop NLL at the control's stopping epoch. It is a free, stop-matched diagnostic on early-stop rows. It is still not a DEV evaluation.

### F3 — low (disclosure). The warm-up governs the whole relevant range, so ".999 averages about 0.8 epoch" holds only from about epoch 7
- `ema_decay_at` reaches .999 only at step t = 8,990, about epoch 7.3 at 1,224 updates per epoch. 13,464 / 11 = 1,224 exactly, so the arithmetic in the note is right.
- Effective mean lag of the warm-up EMA (computed, output below):

| epoch | d at the last step | mean lag |
|---|---|---|
| 1 | .9927 | 0.10 epoch |
| 4 | .9982 | 0.40 epoch |
| 6 | .9988 | 0.60 epoch |
| 8 | .9990 | 0.78 epoch |
| 11 | .9990 | 0.82 epoch |

- The weight left on the random initial weights is about 6e-23 after epoch 1, so "init is forgotten quickly" is correct.
- The P32 control best epochs were 6/6/8. Near the relevant epochs the average therefore spans about 0.6–0.8 epoch, and less earlier.
- Suggested text for `DESIGN…:36`: ".999는 warm-up이 끝난 뒤(step 8,990 ≈ 7.3 epoch) 약 0.8 epoch를 평균한다. 그 전에는 평균 지연이 epoch 1에서 0.1, epoch 6에서 0.6 epoch다."

### F4 — medium → **fixed**. Registry did not pin seeds / fits and compared type-loosely
- `direct_ema_registry.py` had no `SEEDS` / `FITS` / `check_seeds`.
- The sibling families pin their seeds in their registries (`direct_embed_residual_registry.SEEDS = [16,17,18]`, `check_seeds`), and `run_cuda_direct.config_check` relies on that. Without the pin, an EMA branch added there would accept any seed list.
- `!=` treats `True == 1` and `1.0 == 1`. So `priority: true` passed `check_arm`, and `seeds: [19.0, …]` would pass a plain equality check.
- Fix applied: `SEEDS = [19..23]`, `FITS = 20`, a `same()` (JSON-exact, copied from the siblings), and `check_seeds` called from `check_family`. All registry equalities now use `same`.
- Verified that the existing configs' `control` (DATA-COMBO, DROPOUT runs) and frontier `dropout` entries are still `same` as `CONTROL` / `FRONTIER['dropout']` (output below).
- Seeds 19–23 are disjoint from the earlier blocks 0–2, 3–7, 8–12, 13–15 and 16–18.

### F5 — low (next step). `check_contrasts` returns `None` without checking the family
- `direct_ema_registry.py` `check_contrasts`: if `comparison.ema_contrasts` is absent, nothing is checked. This is the same pattern as the siblings.
- The runner must therefore add `elif uses_ema(config): er.check_family(config)` in `config_check`. Use the same order discipline as `uses_tabm` / `uses_embed_residual`. Note that the `dropout` arm carries `frontier: True`, so the EMA branch must come before any frontier gate.
- The runner must also call `check_arm` per arm in dispatch.
- If the branch is forgotten, the fallback `seeds != [0,1,2]` raises. That fails closed, but with a misleading message.

### F6 — low (CUDA). "Raw trajectory equals the base arm" is shown on CPU only
- On CUDA the reasoning still holds, but it was not measured:
  - eval-mode `logits`, `deepcopy` and `lerp_` do not advance the Philox generator;
  - ATen `dropout` returns early for `p == 0` (`is_fused_kernel_acceptable` requires `p > 0`), so the `ema` arm's `Dropout(0)` consumes no RNG, as with `MatrixModel`.
- Matmul and cross-entropy backward are deterministic on a single stream, so bit equality is likely, though GPU3 and GPU4 could in principle differ.
- Smoke assertion (free from outputs the smoke already writes): for each seed, over the overlapping epochs,
  - `ema.history[i].train_weighted_nll == control.history[i].train_weighted_nll`;
  - `ema.history[i].raw_earlystop_conditional_nll == control.history[i].earlystop_conditional_nll`;
  - the same for `dropout_ema` vs `dropout`.
- If the values are not equal, change the note's claim to "same on CPU tests; CUDA equal to within X".

### F7 — info. Archive, load and `kind`: no defect
- The formats are distinct: `direct_ema_flatten_v1` / `ema_flatten_mlp` vs `direct_frontier_regularized_v1` / `regularized_flatten_mlp` vs `matrix_model_v1`. Each loader rejects the other two (now tested both ways, T3).
- The inherited `save` writes `self.FORMAT` and `self.KIND`, so the subclass values are used.
- `load_ema` compares width, dropout, weight decay and decay. `ema` and `dropout_ema` differ in dropout, so neither can load as the other. A tampered decay is refused (T3).
- `MatrixModel` branches on `kind` only for `'lightgbm'` (logits, save) and `'melville'` (report text), so `'ema_flatten_mlp'` takes the torch path everywhere.
- `self.kind = self.KIND` in `__init__` is redundant (the parent already sets it via `self.KIND`) but harmless.
- Nit: `ARMS['ema']['base_model'] = 'flatten_mlp'`, but the network is `DropoutFlattenNetwork(p 0)`, archived as `regularized_flatten_mlp`. It is numerically identical on CPU (`test_neutral_base_still_reproduces_control`). The label is about the training procedure, not the module.

### F8 — info. The same history key has a different meaning per arm
- In EMA arms `history[*].earlystop_conditional_nll` is the EMA score. In control and dropout it is the raw score.
- A scorer or summary that compares this key across arms must read `report['weights'] == 'ema'`. Otherwise it should compare `raw_earlystop_conditional_nll` with the base `earlystop_conditional_nll`.
- Nothing reads it yet, so this is a note for the next step.

### F9 — info. Cost estimate
- The note marks the cost "추정, 미측정". Good.
- Because of F1 the EMA arms may run more epochs. Worst case is 30 epochs × about 7.5 s per epoch (82.5 s / 11) × about 1.1 for the extra evaluation, roughly 250 s per fit and roughly 5,000 worker-s for all 20 fits.
- That is still below the 3,600 s task limit and the 36,000 worker-s family limit. Suggest adding this worst case next to the 2,000 worker-s point estimate.

### F10 — info. Sources and exposure wording
- Izmailov et al. (arXiv:1803.05407, UAI 2018): the abstract confirms "better generalization". It describes SWA, an equal-weight average with a cyclical or constant learning rate. That is related evidence, not the same method as EMA. Suggest adding "(SWA, 균등 평균; 우리는 EMA)".
- Morales-Brotons, Vogels, Hendrikx (arXiv:2411.18704, TMLR 2024): the abstract confirms "generalize better … improved … calibration". The note is accurate.
- The exposure wording is adequate: repeated DEV, descriptive only, no verdict, ranking or promotion, no 2026 data, no policy claim.

## Fit-mirror and question checklist (Q1–Q3)
- **Q1, RNG.** The EMA block runs `deepcopy` before the data tensors and before `default_rng`. `update_ema` is a `no_grad` `lerp_`. The extra raw-score and EMA-score passes are eval-mode `logits` with no dropout and no RNG.
- **Q1, mode.** `logits` puts both nets in eval mode. The next epoch calls `self.net.train()` on the raw net, exactly as the base arm does. The shadow net is never trained.
- **Q1, swap.** The swap is restored in `finally`. Removing the `finally` is now caught by T5.
- **Q1, CUDA.** `deepcopy` runs after `.to(device)`, so the shadow lives on the same device. `lerp_` is fp32 with a Python-float weight. `best_state` is a CPU clone with keys identical to `self.net` (deepcopy), so the strict `load_state_dict` works. The no-buffer guard keeps parameters equal to the whole state.
- **Q2, step index.** `update_ema(…, updates, …)` runs before `updates += 1`, so t = 0 for the first update and d₀ = .1 (TF's `num_updates` rule, applied 0-based).
- **Q2, early stopping.** Selecting on EMA weights is part of the treatment. It is fair for the stated question once F1 is disclosed. To separate "EMA weights" from "EMA stopping" on DEV you would need a third saved state (EMA at the raw-rule epoch). That is a new arm or state and is not needed for the question as registered. F2 gives the early-stop-row version at zero cost.
- **Q3.** See F7.

## Test additions (each new test was checked against a mutant)
Mutants were applied to a temporary copy of `direct_ema_model.py`, which was then restored (`git status` clean for that file):

| mutant | original 8 tests | after T1–T5 |
|---|---|---|
| stop on the raw score instead of the EMA score | **0 fail** | 1 fail (T2) |
| swap-back without `finally` | **0 fail** | 1 fail (T5) |
| `load_ema` ignores `ema_decay` | **0 fail** | 1 fail (T3) |
| extra `torch.rand` in the EMA update | 1 fail | 1 fail |
| patience `>` instead of `>=` | **0 fail** | 2 fail (T1, T2) |

- **T1** `test_vanishing_decay_is_the_base_arm_including_stopping`: with decay 1e-9 the shadow equals the raw weights (`lerp_` with weight 1 − 1e-9 rounds to 1.0 in fp32). The per-epoch scores, best epoch, epochs run (< 12, so the patience branch runs) and logits are bit-equal to `FlattenRegularizedModel`. This is the full mirror test, including stopping.
- **T2** `test_stopping_and_best_epoch_follow_the_ema_score`: on synthetic data where the EMA rule and the raw rule disagree (asserted as a precondition), the reported (best_epoch, epochs_run) equal the EMA rule.
- **T3** `test_archive_negatives`: a regularized archive and a MatrixModel archive are both refused by `load_ema`. An EMA archive is refused by `MatrixModel.load`. A tampered decay, network dropout or kind is refused.
- **T4** `test_registry_rejects_extra_keys_and_loose_flags`: rejects an extra arm key, `ema: 1`, priority 2, a missing `frontier` flag, `ema` on the control, an extra arm, reversed contrasts, `priority: true`, wrong seeds, float seeds, wrong fits and `repeat_tag`. The `priority: true` and float-seed cases need F4.
- **T5** `test_shadow_swap_is_restored_when_scoring_fails`: makes the shadow scoring fail and checks that `model.net` is the raw net afterwards (`requires_grad` stays True).
- Author fixture change: `test_registry_guards`'s config gained `'seeds': [19..23], 'fits': 20`, as F4 requires. No assertion was removed.

## Changes made (separately)
1. `pitchmdp/direct_ema_registry.py` (F4): adds `import json`, `SEEDS`, `FITS`, `same`, `check_seeds`. `check_family` now calls `check_seeds`, and all equalities use `same`. Training, factory and model code are untouched.
2. `tests/test_direct_ema_model.py`: T1–T5 added in a marked reviewer section, plus the seeds/fits fixture line in `test_registry_guards`.
3. This review file.
- Frozen references are unchanged: `git diff eee5c64.. -- direct_frontier_model.py direct_frontier_registry.py direct_residual_model.py matrix_models.py` → 0 lines.
- Not applied: F1, F2, F3, F9 and F10 wording/report changes (the author's files), and F5/F6 (next step).

## Commands run (output)
```
$ cd experiments/pitchmdp && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. /Users/song/Projects/pitcheezy/.venv/bin/python -m pytest tests/test_direct_ema_model.py tests/test_direct_frontier_model.py tests/test_cuda_dropout.py -q -p no:cacheprovider
# at f7282bb:      56 passed in 2.69s
# after changes:   61 passed in 2.73s

$ diff <(FlattenRegularizedModel.fit body) <(FlattenEMAModel.fit body)
# only: comment line; +EMA shadow block (buffers guard, deepcopy, params, decay); +update_ema after optimizer.step();
#       raw score + swap/try/finally EMA score; isfinite(raw_score); 2 history keys; best_state from ema_net;
#       raw_best + 4 report keys

$ git diff eee5c64..f7282bb -- <4 frozen files> | wc -l
0

$ python (warm-up mean lag, 1,224 updates/epoch)
1 d_last=0.992701 mean_lag=0.10 epoch init_weight=5.7e-23
4 d_last=0.998165 mean_lag=0.40 epoch
6 d_last=0.998776 mean_lag=0.60 epoch
8 d_last=0.999000 mean_lag=0.78 epoch
11 d_last=0.999000 mean_lag=0.82 epoch
first t with d=.999: 8990

$ python /tmp/ema_probe3.py   # synthetic, (best, run) EMA rule vs raw rule, patience 2
.0003/.99  (7,9) vs (6,8)  x3 seeds;  .0003/.995 (7,9) vs (6,8) x3
.001/.99   (3,5) vs (2,4), (3,5) vs (2,4), (2,4) vs (2,4)
.0001/.98  (20,20) vs (18,20), (19,20) vs (18,20), (19,20) vs (17,19)

$ python -c "same(frontier DRAFT dropout entry, FRONTIER['dropout']); same(control entries, CONTROL)"
True; DATA-COMBO True; DROPOUT True

$ curl export.arxiv.org/api/query?id_list=1803.05407 | 2411.18704   # abstracts as quoted in F10
```
