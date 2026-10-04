# DRAFT — EXP-P36-001 (tentative ID): RealMLP-style robust scale + smooth clip on the residual backbone (pre-registration)

Status: `author_draft; same_model_review_pending; Codex_review_pending`. Design only. Written 2026-10-04 22:5x KST
(`date` 22:52:18 at task registration, v5 cycle-022, task T101) by the delegated Opus 5.5 PM worker, on
`codex/robust-clip-prep-1004` on top of the prepared module `5f1864b`, **before any EXP-P35-001 fit or score exists**
(P35 is not launched; earliest valid launch 2026-10-05 00:00 KST) and **before any wiring, smoke, timing or fit of this
family**. Not registered. No config under `configs/` named EXP-P36-001. No GPU, no training, no inference.
The ID is tentative and is assigned only at registration.

Go reference for a later run: root v5 runbook directive (sha256 `e73f4402…3b96`) "After P35 use complete-family
evidence and actual primary research to choose the next distinct justified hypothesis, within the existing gates."
This draft does not choose; see §7.

## 요약 (Song용, 쉬운 말)

- 지금까지 숫자 입력은 "학습 구간 평균을 빼고 표준편차로 나누는" 처리만 했다. 아주 드문 구종 비율처럼 한쪽으로
  크게 튀는 값(학습 구간에서 표준화 뒤 최대 209)이 그대로 들어간다(T90b 측정).
- RealMLP 논문(NeurIPS 2024)은 "가운데값으로 맞추고, 사분위 폭으로 나누고, ±3 근처에서 부드럽게 누르는" 처리를 기본값으로
  쓴다. 비유하면 시즌 초 3타수 2안타의 .667을 그대로 믿지 않고 눌러서 보는 것.
- 실험: 4팔 × 새 시드 5개(34–38) = 20회. 핵심 비교는 하나: 같은 잔차 모델에서 입력 처리만 바꾼 것.
- 예상 시간: 최악(모든 적합이 60 epoch 상한) 약 18,000 작업초, 실제는 조기 종료로 훨씬 짧을 것으로 보이나 미측정.
  CUDA smoke에서 잰 뒤 등록한다.
- 판정·승자 없음. dev/mlb_dev는 여러 번 본 자료라 "새 검증"이 아니다. 2026 자료는 쓰지 않는다.

## 1. What is already known (one line each; sources)

- Our inputs: TRAIN mean/std standardization only; no robust scale, smooth clip or learned diagonal scaling anywhere
  (coordinator `reports/research-related-work/FEASIBILITY-candidate3-realmlp-preproc-ko.md`, T84, repo grep 0 hits).
- Input tails on TRAIN (coordinator `reports/research-candidate3-tails/REPORT-T90b-train-tails-ko.md`, T90b, 1,252,824
  rows, 81 inputs): past physics |z| > 5 at most 0.103 % per channel; 11 rare pitch-type share columns |z| up to 209.7
  (under the RealMLP range rule these become |r| ≤ 2); batter reliability columns changed by more than 1 by the smooth
  clip in up to 4.94 % of rows. Descriptive; not evidence of an NLL effect.
- P32 (seeds 16–18, consolidated report `reports/research-p30-p33/REPORT-P30-P33-consolidated-ko.md` line 54/57):
  seed-mean Δ vs same-seed control, dev / mlb_dev: `residual_counts` −0.00173 / −0.00231; `periodic_residual_counts`
  −0.00394 / −0.00325; `linear_residual_counts` −0.00268 / −0.00313 (repeated DEV, game-block intervals in that report).
  Periodic and linear embeddings are the only numeric-input treatments tried on this backbone.
- Primary source: Holzmüller, Grinsztajn, Steinwart 2024, arXiv:2407.04491 §3 "Data preprocessing" (quotes checked
  against the arXiv LaTeX in coordinator T89a, `reports/research-related-work/quote-check/`). The paper reports a direct
  comparison against standardization (T89a correction to T84); its setting (their MLP, NTP, tuned defaults, 118 datasets)
  differs from ours, so its effect size is not transferable.
- Overlap: the PLE arms of P35 use TRAIN quantile bins, which also bound tails. This family therefore targets the
  embedding-free residual backbone, where no tail handling exists.

## 2. Question

With identical data, residual backbone, counts workload, loop, epoch cap 60, early stop, calibration chain and seeds,
does applying the fixed RealMLP robust scale + smooth clip (module `direct_robust_clip_model.py` sha256 `a3d6cbce…52fa`, reviewed
`77cc453`, disposition `55938c7`; transform in its docstring) to the 81
scalars that the periodic arm embeds change calibrated direct-head NLL / Brier / ECE on dev and mlb_dev?

Primary contrast (one factor):
- `robust_clip_residual_counts` − `residual_counts` (isolates `input_preprocessing`; same seed ⇒ identical initial
  parameters, verified by module test).

Secondary descriptive contrast (two numeric-input treatments on the same backbone, not one factor in parameters):
- `robust_clip_residual_counts` − `periodic_residual_counts` (isolates `numeric_input_treatment`; the periodic arm adds
  embedding parameters, so this is not compute matched; reported, never used to pick).

Every arm is also reported vs the same-family same-device CUDA `control` (runner requirement).

## 3. Fixed design (no change after any P35 or P36 score)

- Arms (4), all width 128, workload `prior_records_per100` (counts) except control:
  1. `control` — MatrixModel flatten_mlp 128, no workload (exact EXP-P35-001 entry).
  2. `residual_counts` — EXP-P32-001 arm unchanged (model `residual_mlp`, frozen `direct_residual_model.py`).
  3. `periodic_residual_counts` — EXP-P32/P34/P35 arm unchanged (frozen PLR-lite + residual backbone).
  4. `robust_clip_residual_counts` — new additive module `direct_robust_clip_model.py` (pinned sha256 `a3d6cbce…52fa`,
     reviewed successor of `5f1864b`), `context_continuous` = the same 41-column list as `periodic_residual_counts`.
     Transformed columns (fixed): only this periodic / PLE continuous set (81 scalars of the counts map). Raw, i.e. NOT
     transformed: past type/outcome one-hots, game-state ordinals and flags (context 0–10: balls, strikes, outs,
     inning, defensive_score_lead, six flags: defender_is_home, first/second/third_base, batter_left,
     pitcher_left) and `left_share` (context 32). Deviation from the paper's RealMLP-TD,
     which preprocesses "all numerical columns, including the one-hot encoded ones" (arXiv 2407.04491v3 LaTeX :213);
     follows instead its preprocessing ablation convention (:579, one-hot columns untouched). Reason: hold the input
     set equal to the periodic/PLE arms so the one factor vs `residual_counts` is the transform of the same scalars.
     Added 2026-10-04 23:01 KST from same_model_review m1 (`36ce8d1`/`77cc453`, disposition `55938c7`), before any
     P35/P36 score; the design is otherwise unchanged. Note (review a23e77b m4): the transformed set also holds bounded
     proportions (batter archetype memberships, context 23–27, and pitcher pitch-type shares), not only unbounded physics.
- Seeds: exactly [34, 35, 36, 37, 38], fresh for every arm, disjoint from the P24–P35 family seeds (0–33). Seed 34 was
  drawn once as the P35 timing-probe seed (3 TRAIN epochs per probed arm, no artifact kept, no DEV/MLB DEV row touched);
  it never produced a prediction or a score. Any P36 timing probe uses seed 39 (outside 0–38). New RNG seeds are not a
  new evaluation population. No control reuse from earlier families.
- Fits: 20. Attempts ≤ 2 per task. Task cap 3,600 s; family budget 36,000 worker-s (unchanged defaults). Epoch cap 60,
  patience 5, batch 1,024, lr 5e-4, AdamW, uniform loss weighting, hpo_trials 0 — identical to EXP-P35-001 `training`.
- Robust-clip constants fixed: clip 3, median center, IQR scale with range fallback 2/(max−min), else 0; statistics
  from the `train` split argument only (never early-stop, temperature, blend, dev or mlb_dev rows).
- Device: GPU3/GPU4 by UUID only, ≤ 2 parallel, common lock dir; same-device controls as in P35 (queue assigns by task;
  GPU split by arm is disclosed in the report, as in P34 F2).
- Export: same export manifest `49097109…f578` and splits as EXP-P35-001.
- Scoring: `score_cuda_direct.py` descriptive outputs only — NLL, Brier, ECE, matched-seed Δ, game-block bootstrap
  intervals, seed-mean ensemble Δ; all 20 fits, failures and retries listed. No winner/promotion/verdict fields.
- No saved-probability ensemble grid for this family (adding one needs a grid file committed with time and reason
  BEFORE any P36 score exists).

## 4. Cost projection and gates (before registration)

- Upper bound per task from the P35 timing probe (`RESULT-P35-TIMING-PROBE-3b317f2-20261004.md`, slower GPU, 60-epoch
  cap): control 464 s; periodic_residual_counts 1,046 s. `residual_counts` and the robust-clip arm have fewer
  parameters than periodic_residual_counts (no embedding) and the clip is an element-wise op on 81 scalars, so the
  bound 1,046 s per task is assumed for both — **an assumption until the smoke measures it**.
- Worst case: 5 × 464 + 15 × 1,046 = 18,010 worker-s ≤ 36,000. Realistic (early stop as in P32: periodic_residual
  ≈ 225 s, control 63–98 s per fit including setup) ≈ 4,000 worker-s, ≈ 35 min wall on 2 GPUs — not measured for
  this family.
- Gate before registration (replaced 2026-10-04 from same_model_review a23e77b M1, before any P35/P36 score; the
  earlier "smoke s/epoch × 60" gate could not measure anything because a smoke runs 8,192 TRAIN rows × 2 epochs):
  (i) CUDA smoke of all 20 tasks on GPU3/4 (fresh archive/code dir), exit 0 — wiring and identities only, smoke times
  are not a projection; (ii) timing probe by the P35 Amendment-2 method for `residual_counts` and
  `robust_clip_residual_counts` on each of GPU3/GPU4, probe seed 39 (no artifact kept, no DEV/MLB DEV row touched):
  task_projected = startup + prepare (incl. `compute_stats` on the full TRAIN argument, reported separately)
  + 60 × per_epoch (warm full-TRAIN 2-epoch fit after a 1-epoch warm-up, max(wall, report)/2) + predict + 30 s, slower
  GPU per arm; `control` and `periodic_residual_counts` take the P35 probe values 464 s / 1,046 s
  (`RESULT-P35-TIMING-PROBE-3b317f2-20261004.md:30-31`). Gates: every arm ≤ 1,800 s, family = Σ_arms 5 × slowest
  ≤ 36,000 worker-s, probe exit 0; peak device bytes and peak host RSS (`ru_maxrss`) printed per arm. Any gate false →
  no registration, record kept (gates not raised, not silently shortened).

## 5. Stop conditions

- Any test/review blocker in module or wiring unresolved → no smoke.
- Smoke failure on any path → fix in a new commit + re-smoke (≤ 2 smoke attempts), else stop.
- During the run: queue hard stops (task 3,600 s, family 36,000 worker-s, no_new_fit / kill / aggregate deadlines set
  at registration on one KST date as in the P35 pack). Incomplete family ⇒ no comparison, only a failure record.
- Deadlines on ONE KST date, fixed now (review a23e77b m1): no_new_fit_after = L + 5 h, kill_own_after = no_new + 1 h,
  aggregate_by = kill + 30 min (the P35 pack rule). Coordinator deadline 2026-10-05 23:00 KST ⇒ latest launch
  L = 16:30 KST on 2026-10-05. If P35 is not terminal, or the smoke, probe, reviews and registration are not done by
  then, P36 does not launch on 10-05 and waits for a new dated go.

## 6. Exposure and limits

- DEV / mlb_dev are the same repeatedly exposed ≤ 2025 rows used for P24–P35 (258 fits incl. P35 once it runs). This
  family was designed after reading P24–P34 results (post-selection). Descriptive only.
- No 2026 data for fit, statistics, early stop, temperature, selection or this design.
- Not a reproduction of RealMLP (only the preprocessing step on a subset of columns — see §3 arm 4; no diagonal
  scaling layer, no NTP, our backbone/loop).
- A null or positive Δ says nothing about policy value; G0, G15 mix (D145), D131/D155 unchanged.

## 7. Selection rule (fixed now, before any P35 score)

- This family is the default next unit after P35 **only if** (a) the P35 family completes (25/25) and its descriptive
  report is written, and (b) P35 reveals no pipeline/data-contract problem on the shared residual backbone or export.
- The P35 score magnitudes do **not** change this design (no arm, seed, constant or contrast added/removed after P35
  scores). If P35 is incomplete, the P35 retry inside its own attempts takes priority and this family waits.
- The choice itself is recorded at registration by the PM, citing the complete P35 descriptive report; this section
  fixes only the conditions and forbids design changes from P35 magnitudes (review a23e77b n1).
- Contrast B of the feasibility note (learned diagonal scaling) is NOT part of this family (its learning-rate choice is
  unresolved); it would be a separate pre-registration.

## 8. Next steps (in order)

1. Same-model review of the module `5f1864b` (T100b) → disposition `55938c7` (all accepted). DONE.
2. Additive wiring (registry + factory + runner/scorer dispatch + DRAFT config `configs/DRAFT-CUDA-ROBUST-CLIP-20261005.json`
   + CPU tests), branch `codex/robust-clip-wiring-1004` (T101b `de539d5`, merged/re-pinned `a23e77b`). DONE.
3. Distinct same-model review of this pre-registration + wiring (T104b `77bd575`/`204929e`) → disposition (T106b). DONE.
4. After P35 is terminal: CUDA smoke of all 20 tasks on GPU3/4 (gate §4 (i)); timing probe seed 39 (gate §4 (ii)) —
   the probe script is additive new code and needs its own tests + same-model review before it runs.
5. Registration with immutable config, fresh outputs, one-date deadlines (§5); add `source_configs.ple_registered` =
   `configs/EXP-P35-001.json` (+ sha) and a test that `control` / `periodic_residual_counts` equal its entries (review
   a23e77b m5). Launch only after P35 is terminal and GPU3/4 locks are free, L ≤ 16:30 on 2026-10-05.
