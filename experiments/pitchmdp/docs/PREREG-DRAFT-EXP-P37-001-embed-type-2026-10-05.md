# DRAFT — EXP-P37-001 (tentative ID): periodic vs linear numeric embedding at near-equal parameters (pre-registration)

Status: `author_draft; same_model_distinct_review_done (6b66436, accept with fixes); delivery fixes M1/m1/m2/m3/m5 applied 2026-10-05 ~14:40 KST before any smoke (codex/p37-delivery-1005); not registered`. Written 2026-10-05 14:00–14:10 KST (real
clock, `date` 14:00:04 at task start) by the Opus 5.5 next-research author (job `pm-jobs/20261005-next-research/author-001`)
on `codex/next-research-1005` from `f8a9a1b` (the P36 registration/launch lineage; runtime code identical to
`code-07ab288`, checked by `git diff --stat 07ab288 f8a9a1b -- experiments/pitchmdp/pitchmdp experiments/pitchmdp/scripts src`
= empty). Written **after** the P36 descriptive report (`codex/p36-report-1005@ce2861b`) and **before any P37 smoke,
timing or fit**. No GPU, SSH, server write or data fetch in this job. The ID is tentative; it is assigned at registration.

Config (DRAFT, generated): `configs/DRAFT-CUDA-EMBED-TYPE-20261005.json` sha256
`75265fc715f0fec998f609f0316ffe1de486c609677588d0fbd7241fc427ace2` (single-GPU regeneration 2026-10-05 ~14:35 KST by job
`20261005-single-gpu/author-001`; the first draft `1e3a529c…ec7f` allowed GPU3/4 and is superseded, never registered), written by
`experiments/pitchmdp/docs/launch/make_p37_config.py` from `configs/EXP-P36-001.json` (sha256 `b6479afa…b87e`).

## 요약 (Song용, 쉬운 말)

- P36에서 "주기 임베딩(periodic) 팔"이 대조군보다 손실이 가장 낮았다(dev −0.00379, mlb_dev −0.00324, 5/5 시드). 그런데 이 팔은
  ① 사인/코사인 변환, ② 파라미터 수 증가(약 24.5만 vs 잔차 17.1만), ③ 계산 시간 2.33배가 한꺼번에 묶여 있다.
- 이번 실험은 ②를 고정하고 ①만 바꾼다: 같은 잔차 모델·같은 열·같은 임베딩 폭 8에서 사인/코사인 대신 "열마다 1차식"(linear)을
  쓰는 팔과 비교한다. 파라미터 수는 거의 같다(244,898 vs 244,634, 차이 264개 = 0.11 %). 계산량·시간은 같지 않다 → 따로 보고.
- 비유: 같은 타자에게 같은 무게의 배트를 주고 그립만 바꿔 보는 것. 배트 무게(용량) 덕인지 그립(변환) 덕인지 가른다.
- 3팔(대조군 포함, 러너 요구) × 새 시드 5개(39–43) = 15회. **GPU3 1장에서 한 번에 1개씩**(2026-10-05 Song 자원 지시) — P36 실측 기준 학습만 약 40–60분 추정(linear 팔 60 epoch 시간·smoke 시간은 미측정).
- 판정·승자 없음. dev/mlb_dev는 여러 번 본 자료라 "새 검증"이 아니다. 2026 자료는 쓰지 않는다.

## 1. What is already known (sources)

- P36 (`ce2861b`, seeds 34–38, cap 60, repeated DEV): vs same-family control, `periodic_residual_counts` dev −0.00379,
  mlb_dev −0.00324 (5/5 seeds negative both splits); `residual_counts` −0.00167 / −0.00236. Mean fit seconds: control
  91.1, residual 161.1, periodic 212.2 (2.33x control). robust-clip primary contrast not improved (axis closed).
- P35 (`3218d78`): PLE vs periodic dev +0.00060, mlb_dev +0.00035 — no additive benefit from a third embedding type.
- **EXP-P32-001 already ran this exact pair once** (seeds 16–18, epoch cap 30, 6-arm family, `RESULT-EXP-P32-001-20261004.md`):
  `embedding_type` (periodic_residual − linear_residual) dev −0.00126 [−0.00240, −0.00013] 3/3; mlb_dev −0.00012
  [−0.00039, +0.00015] (seeds 0.00000 / −0.00003 / −0.00127). `embedding` (linear_residual − residual) dev −0.00095,
  mlb_dev −0.00082. Params 244,898 / 244,634; mean seconds 216.9 / 196.7 (cap 30). So at cap 30, most of the mlb_dev
  gain over the raw residual came with the linear embedding too; the dev split favoured periodic, mlb_dev did not.
- No ready, unexecuted pre-registration covers this pair at cap 60 / 5 seeds (searched `experiments/pitchmdp/docs`,
  `configs`, `docs` on the P36 lineage, `codex/next-hypothesis-1004`, `codex/periodic-residual-1004`; P37 unused).

## 2. Question and the one fixed contrast

With identical data, residual backbone (width 128 / hidden 256 / 2 blocks / dropout .1), embedded columns (5 past
positions × 8 physics + 41 continuous context = 81 scalars), embedding width 8 (so identical backbone input width),
counts workload, training loop, epoch cap 60, early stop, calibration chain and seeds, does the PLR-lite periodic map
lower calibrated direct-head NLL / Brier vs the per-feature LR map?

Primary and only contrast (candidate − reference; same seeds; seed-mean ensemble and per seed; whole-game bootstrap 10,000):
- `periodic_residual_counts` − `linear_residual_counts`, isolates `embedding_map`.
  **parameter-near-matched** (+264 parameters, the shared `Linear(32, 8)` minus nothing else; re-derived from the frozen
  networks by the tests), **NOT compute-matched**: periodic computes 2 × 16 cos/sin + Linear(32, 8) per scalar per row;
  LR one affine map per scalar per row; backbone identical. Early stopping can also make epochs and optimizer updates differ.
  It is not "parameter-matched" in the exact sense and must not be called so.

Every arm is also reported vs the same-family same-device CUDA `control` (runner requirement), not as a contrast.
The residual_counts arm is deliberately NOT re-run (P36 has it; 2 arms × 5 seeds + required control keeps 15 fits).

## 3. Fixed design (no new model code)

- Arms = the exact earlier registered config entries (JSON-equal to `EXP-P32-001`; control and periodic also equal to
  `EXP-P36-001`): `control` (flatten_mlp 128, no workload), `periodic_residual_counts`, `linear_residual_counts`.
  Frozen modules `direct_periodic_residual_model.py` (LR network included), `direct_embed_residual_factory.py`,
  `direct_residual_model.py`, `matrix_models.py` are untouched; fit semantics unchanged (same loop, same cap 60 path as P34–P36).
- Seeds exactly [39, 40, 41, 42, 43], fixed here before any P37 metric; disjoint from every earlier block 0–38.
- Training = `cuda_direct_v1` budget with epoch cap 60, patience 5, batch 1024, lr 5e-4, AdamW wd .01 (= P34–P36).
  **No width, learning-rate, frequency, sigma or embedding-width choice** is made from any result; there is no HPO.
- New code (wiring only): `pitchmdp/direct_embed_type_registry.py` (torch-free registry; top-level marker
  `embed_type_family: true`, tested FIRST because both embedded arms keep the P32 `embed_residual` flag); runner
  `config_check` / `training_budget` / `uses_epoch_cap` dispatch; scorer contrast key `embed_type_contrasts` plus a
  per-fit cost record (parameters == registered value, fit seconds, optimizer updates, seconds per update, epochs,
  peak device bytes; plus, added before any smoke (review 6b66436 m1, delivery job 2026-10-05), the pure training-loop
  time `train_seconds` = frozen `report.seconds` and `train_seconds_per_epoch` / `train_seconds_per_update`, reported
  separately from `fit_seconds` = fit wall incl. export load/hash). Earlier registered configs P30–P36 route and budget
  exactly as before (tests).

## 4. TRAIN-only pilot / calibration methodology (fixed now; no DEV selection)

- Temperature: the existing chain (G0 `log(clip(softmax)) / T`, bounded [0.5, 2.5]) fitted on the `temperature` split
  (May 16–31 Cpanel) only; early stop on `earlystop`; DEV / mlb_dev are predicted only.
- Pilot = CUDA smoke of all 15 tasks on the first 8,192 TRAIN rows, 2 epochs (runner SMOKE), checked by the existing
  identity checker. Smoke outputs are never scored; their only use is pass/fail of the pipeline.
- Timing gate (before registration): no new timing probe is required because the per-arm cost is bounded by P36 actuals
  (control mean 91.1 s, periodic mean 212.2 s / max 235.2 s at cap 60, same code; P36 ran on GPU3 and GPU4, P37 uses GPU3 only). linear_residual at cap 60 is
  **미측정** (P32 cap 30: 196.7 s mean).
- **TRAIN-smoke timing gate (corrected 2026-10-05 before any smoke, review 6b66436 M1; supersedes "any per-epoch time")**:
  per arm, the **median over the 5 smoke seeds of `report.seconds / report.epochs_run`** (frozen training-loop time only;
  `fit_seconds` = fit wall incl. export load/hash is recorded separately as overhead and never gated). Stop and re-plan
  (no registration) if linear median / periodic median **> 1.5**, or if any of the 15 records is missing, nonfinite,
  nonpositive or has 0 epochs, or the smoke bound-UUID set ≠ {GPU3}. Implemented by `docs/launch/p37_smoke_gate.py`.
  The smoke (8,192 TRAIN rows × 2 epochs) is **not a cap-60 measurement**; it only bounds the relative per-epoch cost.
- **Schedule projection (fixed now, from P36 actuals, not from the smoke)**: per-task bound = 2 × the P36 per-arm max task
  wall (control 100.13 s, periodic 235.20 s; P36 stopped at 10–13 epochs, so 2× covers ~20–26 epochs) and linear =
  1.5 × the periodic bound (the gate maximum) → 5 × (200.26 + 470.40 + 705.60) = **6,881.4 s ≈ 1.91 h serial**.
  Registration/launch is allowed only if launch + 6,881.4 s ≤ 21:00 (true for any launch ≤ 19:00). A fit that runs to
  the cap 60 can exceed this bound; then the 21:00 / 21:45 clock below applies and the family is reported incomplete.
  Nothing from smoke or timing changes arms, seeds or training.

## 5. Outcomes (all reported, no automatic winner)

Per split (dev, mlb_dev), seed-mean ensemble and each seed: ΔNLL and ΔBrier with 95 % game-bootstrap intervals; number of
negative seeds; top-label ECE10 of each arm; each arm vs control. Cost per fit and per arm (median, min, max; per-seed
candidate/reference ratios of fit seconds, optimizer updates, seconds per update); epochs run / best epoch / stopped-by.
All 15 fits run on GPU3 only, so every paired comparison is same-GPU by design; the scorer still reports the observed
per-fit device UUID and any fit not on GPU3 is a protocol deviation to report. Size reference: control single-fit NLL SD (P36: dev 0.00076, mlb_dev 0.00026).

Reading guide fixed now (descriptive, not a decision rule):
- interval below 0 on both splits and ≥ 4/5 negative seeds on both → the periodic map carries signal beyond parameters on
  these rows (still repeated DEV, not promotion);
- intervals covering 0 or splits disagreeing (as in P32 mlb_dev) → these rows do not show the periodic map beating the
  LR map at near-equal parameters. This is **not** evidence that capacity caused the P36 periodic-vs-control gain (a null
  here proves no cause; any "capacity" reading is a cross-family inference, review 6b66436 m5) — reported as not
  distinguished, nothing more;
- interval above 0 → LR at least as good at lower compute; reported, not adopted.
Any later adoption needs a separate gated decision (unseen data / G0 / D155 rules unchanged).

## 6. Evidence limits and sample independence

- DEV / mlb_dev are the same ≤2025 rows exposed in P24–P36 (and EXP-P30-002); the family is designed after reading them
  (post-selection). Fresh seeds are new model RNG only, not new data. The P32 run of the same pair is on the same rows:
  P37 is a repeat with more seeds and a higher cap, NOT an independent replication; never pool P32 and P37.
- The bootstrap resamples games with predictions fixed (no refit / calibration / selection uncertainty). dev and mlb_dev
  are two splits of the same period's data source family, not independent confirmations of each other.
- No 2026 rows (exporter guard + runner check); all sealed outputs, G0 and D155 unchanged; no policy or service claim.

## 7. Budget, deadline and resources (planned 2026-10-05 KST)

- **Hardware (Song 2026-10-05, supersedes the first draft's GPU3/4 plan):** 15 fits, exactly one at a time, on physical
  GPU3 only — UUID `GPU-c334d905-8ec9-0029-b206-46ce3ac49cd9` (config `device.allowed_gpu_uuids` = [GPU3], `max_parallel` 1; registry `check_device`
  refuses anything else; the queue then accepts only `[GPU3]`, and each fit process binds `CUDA_VISIBLE_DEVICES` to
  that UUID). GPU4 `GPU-ae803a24-0d38-791d-ed7c-3f233fcecf99` is released to another project: never allowed, probed or used as fallback; if GPU3 is busy
  the family waits. The queue is started with **exactly one `--gpu` GPU3** and the **COMMON** lock dir
  `/home/sds733/pitcheezy-autonomy/gpu-locks` (never the per-queue default `<queue-dir>/locks`, review 6b66436 m3); the
  smoke bound-UUID set must equal {GPU3}. No other Pitcheezy family queue running. Task timeout 3,600 s, max
  attempts 2, family budget 36,000 worker-seconds (as P36). Scientific design (arms, seeds, cap, data, contrast)
  unchanged by the hardware change.
- Estimate from P36 actuals (not a measurement of P37): control 5 × ~91 s + periodic 5 × ~212 s + linear 5 × (미측정, P32
  ratio suggests ≤ periodic; timing-gate worst case 1.5 × periodic) ≈ 2,500–3,100 worker-seconds ≈ **40–60 min of
  training wall on one GPU** (serial). Smoke (15 tasks, 8,192 rows, 2 epochs) wall time on one GPU is 미측정.
- Clock (all work ends before the 23:00 delegation deadline): register + launch by 19:00; no new fit after 21:00; kill
  own fits after 21:45; aggregate by 22:30. If not registered by 19:00 the family is not launched today and needs a new
  dated plan. **Scoring (review 6b66436 m2)**: only after 15/15, scorer timeout 900 s, started no later than 21:45 so it
  ends ≤ 22:00; if the family is not 15/15 or the scorer cannot end by 22:00, it is not scored and is reported
  **incomplete by 22:30** (< 23:00). An incomplete family is reported as incomplete; rigor is not reduced to fit the clock.

## 8. Gates before registration / launch (in order)

1. Same-model DISTINCT review (a different Opus 5.5 agent, not this author) of this pre-registration, the DRAFT config,
   the registry/runner/scorer diff and tests; disposition recorded. (Not done; request prepared in the job dir.)
2. Root PM acceptance of the reviewed commit.
3. Launch pipeline from the P36 reviewed TEMPLATE (`docs/launch/run-p36-pipeline.TEMPLATE.py`) adapted to P37
   (config sha, code commit, family dir `-smoke` naming) — to be authored and reviewed; not in this job. The P36
   template's "smoke used exactly both pinned GPUs" check must become "exactly {GPU3}" (set equality with the config
   allow-list, which is [GPU3]) — strengthened to the new allow-list, not dropped; queue started with `--gpu` GPU3 only.
4. CUDA smoke 15/15 (one `--gpu` GPU3, common lock) + `p37_smoke_gate.py` exit 0 (identities, UUID set {GPU3}, §4
   TRAIN-smoke timing gate); registration by `make_p37_registered_config.py`; pipeline `run-p37-pipeline.TEMPLATE.py`.
5. Registration commit: status REGISTERED, execution.enabled true, registration block (source commit, smoke, go) — real
   clock, never backdated. Then launch under the GPU3 lock (GPU3 only, 1 fit at a time). Scoring only with the pinned scorer after 15/15.
Forbidden throughout: code-f429318 or any throughput fast loop, v5/v6 activation, 2026 data, promotion.
