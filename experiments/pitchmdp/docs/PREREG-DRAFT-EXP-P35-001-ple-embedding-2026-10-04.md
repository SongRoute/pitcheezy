# DRAFT — EXP-P35-001: piecewise-linear (PLE) numeric embedding in place of the periodic embedding (pre-registration)

Status: `author_draft; same_model_review_pending; Codex_review_pending`. Design only. Written 2026-10-04
(v4 cycle-010) by a delegated Opus 5.5 author worker, at `codex/next-hypothesis-1004` (base `b0593aa`),
**before any module code, wiring, smoke, timing probe or fit of this family**. Not registered. No
config under `configs/`. No GPU, no training, no inference. A run needs the review chain below plus a
`결정 필요` go (new experiment rule, `~/.claude/rules/machines.md` "실험 운영").
Review: `docs/reviews/OPUS-review-p35-prereg-4ccd13f-2026-10-04.md` (same_model_review; Codex_review_pending) —
3 blocking items (B1–B3) DECIDED by the PM disposition `docs/reviews/DISPOSITION-p35-prereg-review-c1ca3b1-2026-10-04.md`
(same_model_reviewed; Codex_review_pending) before any module code; text-only reviewer fixes marked `[review N…]` below.

## 요약 (Song용, 쉬운 말)

- 지금까지(P24–P34, 학습 233회) 해 본 것: 숫자 입력 바꾸기(주기 임베딩), 잔차 구조, 투수 작업량, 드롭아웃, 폭,
  TabM(한 모델 안에 8개 작은 모델), 가중치 평균(EMA), 저장 확률 섞기(같은/다른 구조·설정). 크기는 모두 NLL 0.001~0.004.
- 아직 안 해 본 것 하나를 고름: **숫자 입력을 "구간별 직선"으로 바꾸는 방법(PLE)**. 비유하면, 지금은 구속·회전수 같은
  숫자를 "물결 모양 여러 개"로 풀어서 넣는데(주기 임베딩), PLE는 "구간을 나눠 어느 구간 몇 % 지점인지"로 넣는다.
- 근거: 같은 연구팀 논문 두 편. 2022년 NeurIPS 논문이 이 두 방법을 처음 제안했고, 2025년 ICLR TabM 논문은
  TabM에 **이 구간 방식(PLE)**을 붙인 변형(TabM†)을 따로 표시해 보고했다(인용은 웹 요약 도구로 읽은 것이라 원문
  PDF 재확인 전까지 미확인). 우리는 TabM에 물결 방식만 붙여 봤다.
- 실험: 5팔 × 새 시드 5개(29–33) = 25회. 바꾸는 것은 "물결 → 구간" 하나. 단일 모델과 TabM 모델에서 각각 비교.
- 예상 GPU 시간: P34 실측 기준 약 7,800 작업초(GPU 2개로 약 65분). 오늘 P30-002가 약 2배 느리게 돌고 있어(원인 미측정)
  2배면 약 130분. 등록 전에 시간 측정으로 한 번 더 확인한다.
- 판정·승자 없음. dev/mlb_dev는 이미 여러 번 본 자료라 "새 검증"이 아니다. 2026 자료는 쓰지 않는다.
- 다음: 같은 모델 검토 → 코드(새 파일만) → 검토 → CUDA smoke → 시간 측정 → `결정 필요`로 실행 승인.
- 버린 후보: 이종 앙상블(P27·P30·P34에서 이미 대부분 봄), SWA(EMA와 겹침), 라벨 스무딩·focal(우리 지표 NLL과 충돌).

## 1. What P24–P34 already tested (descriptive, one line each)

All numbers: Δ = arm − same-family same-seed control, calibrated direct head, before June blend / G0
mix, repeated DEV. Sources in brackets. "Exhausted" = do not repeat as a new hypothesis. Pairs "a / b" are
seed-mean-ensemble Δ dev / mlb_dev unless labelled otherwise (format of the consolidated report, its line 31) [review N9].

- **P24** control/counts/block/residual/periodic, seeds 0–2 [`docs/reviews/PM-P24-CUDA-result-2026-10-04.md`]:
  periodic dev −0.00178 [−0.00336, −0.00020]; counts, block, residual intervals include 0 on dev.
- **P25** width145/203, no_batter, no_batter_cluster, no_history, dropout, weight_decay, dropout+wd, seeds 0–2
  [`docs/reviews/PM-P25-CUDA-result-2026-10-04.md`]: dropout .1 dev −0.00115; removing batter info +0.00586;
  width alone no gain. **Exhausted:** width-only capacity, weight-decay-only, batter/history masks.
- **P27** 3 structures × 3 add-ons + data25/50/75 + pitcher-profile ablations, seeds 0–2
  [`docs/reviews/PM-P27-P29-results-2026-10-04.md`]: periodic+counts −0.00321, periodic+block −0.00337 (dev);
  less TRAIN data worse (25 % +0.00690). **Exhausted:** data-fraction curve on this export.
- **SAVED-ENSEMBLE-P27-001** [`reports/SAVED-ENSEMBLE-P27-001-report-ko.md`]: 3 different structures (same seed)
  vs member 3-seed means: mean −0.00049 dev / −0.00053 mlb_dev, intervals both sides; mixed workload inputs
  +0.00066 / +0.00060. **Largely tested:** heterogeneous-structure saved-probability averaging.
- **P29** dropout on control/residual/periodic, seeds 0–2 [same PM-P27-P29 file]: periodic+dropout vs periodic
  dev −0.00047 [−0.00154, +0.00058]; residual output dropout +0.00002. **Exhausted:** head-dropout-only axis.
- **P30-001** periodic × {none, counts, block} × head dropout {0, .1, .2}, seeds 3–7
  [`reports/research-p30-p33/REPORT-P30-P33-consolidated-ko.md`]: best seed-mean Δ dev −0.00273 (periodic_counts_d10),
  mlb_dev −0.00221 (periodic_block_d20); the d20 cells are −0.00263 / −0.00213 (counts) and −0.00246 / −0.00221 (block)
  [review N9: was "about −0.0025 / −0.0022 (d20 cells)"; consolidated lines 41, 42, 45].
- **SAVED-ENSEMBLE-P30-001** [`reports/t50b-saved-ensemble-p30/RESULT-SAVED-ENSEMBLE-P30-001-20261004.md`]:
  hyperparameter-mixed triples ≈ single-setting seed triples (medians within ±0.0002). **Exhausted:**
  hyperparameter ensembles of P30 cells (that report's own "not re-running" line).
- **P31** TabM-mini k=8 (counts, d0/d10), seeds 13–15 [consolidated report]: tabm_counts −0.00429 / −0.00317;
  ~4× control time; 3/6 TabM fits hit the 30 cap. SAVED-ENSEMBLE-P31: one TabM fit lower than two single
  networks on mlb_dev 6/6 (all < 0.003).
- **P32** periodic/linear embedding + residual backbone, seeds 16–18 [consolidated report]: periodic_residual
  −0.00394 / −0.00325, linear_residual −0.00268 / −0.00313, residual_counts −0.00173 / −0.00231.
  SAVED-ENSEMBLE-P32: one periodic_residual fit higher than the two-parent mean on mlb_dev 3/3.
- **P33** EMA of weights × dropout, seeds 19–23 [consolidated report]: seed-mean EMA Δ −0.00024 (dev, CI incl. 0) /
  −0.00042; single-fit −0.00135 / −0.00162; seed-averaging gain smaller for EMA (0.0009–0.0011 vs ~0.0020).
  Report recommends holding further EMA work. **Exhausted for now:** weight averaging (EMA; see SWA in §3).
- **P34** TabM-mini on the periodic residual backbone, seeds 24–28, cap 60
  [`reports/research-p34/REPORT-P34-and-saved-ensembles-ko.md`, `RESULT-EXP-P34-001-descriptive-ko.md`]:
  tabm_residual −0.00433 / −0.00395; ensemble_head (tabm_residual − periodic_residual) −0.00163 / −0.00094;
  0/25 cap hits; GPU split by arm (backbone contrasts 0/5 same-GPU) [`OPUS-review-p34-report-fa9acba-2026-10-04.md` F2].
- **SAVED-ENSEMBLE-P34-001** [same REPORT §2, `reports/t58-saved-ensemble-p34/group-summary.json`]: one TabM-residual
  fit ≈ mean of [periodic_residual + tabm_d10] fits at the median (|median| ≤ 0.00013), but at equal count 2/2
  the TabM-residual pair is lower (mlb_dev −0.00058, 5/5 CI<0) (review F4).
- **P30-002** same 10 P30 arms, seeds 8–12, RUNNING [`reports/research-p30-002/PRE-SCORE-decisions-ko.md`]: score not
  read and not anticipated here; this design does not depend on it.
- Noise scale [consolidated report]: single-fit control SD dev 0.00094 / mlb_dev 0.00037; between-family
  seed-mean control gap dev 0.0015 / mlb_dev 0.00045. Seed averaging alone lowers control NLL by about 0.002.

Axis that is **not** tested: the numeric-embedding *type* beyond periodic (PLR-lite) and linear (LR). Piecewise-linear
encoding/embedding (PLE), the other family of Gorishniy et al. 2022, has never been run in this repo
(`git grep` of pitchmdp code/docs: no PLE module or arm). The only earlier mention is the candidate list of
`docs/reviews/PM-research-frontier-2026-10-03.md:73` (named, not run) [review N9; all 198 local branches grepped].

## 2. Chosen hypothesis

**Question.** With identical data (export `49097109…f578`), embedded scalars (5 past positions × 8 physics
channels + the registered `counts` continuous context map, 41 columns → 81 embedded scalars), raw part, padding
mask, embedding width d = 8 (so the backbone input width is unchanged), backbone, training loop, epoch cap 60 and
calibration chain: does replacing the periodic PLR-lite embedding by a piecewise-linear embedding (PLE, quantile
bins fitted on TRAIN only) change calibrated direct NLL / Brier / ECE

1. on the single-head residual backbone (`ple_residual_counts − periodic_residual_counts`), and
2. under TabM-mini k=8 on the same backbone (`ple_tabm_residual_counts − tabm_residual_counts`)?

Additive: two new arms only; every frozen module, arm and earlier source hash stays untouched.

**Primary sources (read 2026-10-04 through WebFetch page summaries, open access only; every quote and package
fact below is 미확인 until re-read from the PDF / raw source at a pinned commit — review B3).**
- Y. Gorishniy, I. Rubachev, A. Babenko, "On Embeddings for Numerical Features in Tabular Deep Learning",
  NeurIPS 2022, arXiv:2203.05556 (abstract: "two conceptually different approaches … the first one is based on
  a piecewise linear encoding of scalar values, and the second one utilizes periodic activations"). We already
  use the second; this family tests the first.
- Y. Gorishniy, A. Kotelnikov, A. Babenko, "TabM: Advancing Tabular Deep Learning with Parameter-Efficient
  Ensembling", ICLR 2025, arXiv:2410.24210 (v3). §3.4: "TabM_mini† denotes TabM_mini with non-linear feature
  embeddings from (Gorishniy et al., 2022) … we use a slightly modified version of the piecewise-linear
  embeddings"; §A.2: "we use † to mark TabM variants with the piecewise-linear embeddings". Our P31/P34 TabM arms
  use periodic embeddings instead, so the paper's own embedding choice for TabM has not been tried here.
- Official package `rtdl_num_embeddings` (yandex-research/rtdl-num-embeddings, `package/README.md` and
  `rtdl_num_embeddings.py`, read 2026-10-04, commit not pinned yet → pin at module time): `compute_bins(X, n_bins=48)`
  = per-feature quantiles then `unique()`; `PiecewiseLinearEmbeddings(bins, d_embedding, activation, version)`;
  README: recommended `version="B"`, `activation=False`, starting point `d_embedding=12`; version B adds a
  per-feature linear term (`linear0`) and zero-initialises the PLE linear layer, so the module starts as a
  linear embedding. The formula will be written from the documentation; no third-party code copied; the package
  is **not** installed (no installs on newsrv without asking).

**Not faithful (disclosed now).** d_embedding 8 (package start 12) to hold the backbone input width equal to the
periodic arms; n_bins 48 (package default), quantile bins only (no target-aware tree bins: they would read labels
for binning, a second change); no tuning; our ten-class head and flattened contract; residual backbone instead of
the package's MLP. No claim that paper results transfer.

**Why it is one factor (and what still moves together).** Held: embedded columns, mask, d = 8, raw part,
backbone, head, dropout, workload, loop, cap, calibration. Changes together inside "embedding type": the map
(cos/sin of learned frequencies + shared Linear(32, 8) + ReLU → per-feature linear term + zero-init
per-feature Linear(n_bins_j, 8), no activation), the embedding parameter count (PLR-lite ≈ 81×16 + 264;
PLE ≈ Σ_j (T_j × 8 + 8) + 81×16 — exact counts printed by the module, 미측정 now; arithmetic upper bound at all
T_j = 48: periodic_residual_counts 244,898 (= P34 probe table) → ple_residual_counts ≤ 276,386 (+12.9 %),
tabm_residual_counts 260,760 → ple_tabm_residual_counts ≤ 292,248 (+12.1 %); 648 fewer each if Linear_j has no
bias, see review B3/N3) and the TRAIN-fitted bins
(a data-dependent buffer). The activation difference (ReLU vs none) is part of the bundle; it is disclosed,
not separated.

## 3. Rejected alternatives (2–3, with reasons)

1. **Heterogeneous same-family ensemble** (PM next-unit item 5; P34 F4). Mostly already measured:
   SAVED-ENSEMBLE-P27 (3 different structures vs member seed means: −0.0005, intervals both signs),
   SAVED-ENSEMBLE-P30 (hyperparameter mix ≈ seed mix), SAVED-ENSEMBLE-P34 (TabM-residual vs [pr+d10] at 1/2 and 2/2,
   same family, same seeds 24–28). A new family would re-train the P34 arms on fresh seeds and re-score a grid —
   a seed repeat of P34, i.e. a P30-002-type sensitivity question, not a new mechanism; it adds no new module.
   Keep as a possible later replication, not P35.
2. **Stochastic Weight Averaging** (Izmailov, Podoprikhin, Garipov, Vetrov, Wilson, "Averaging Weights Leads to
   Wider Optima and Better Generalization", UAI 2018, arXiv:1803.05407). Same mechanism class as P33 EMA (weight
   averaging), whose seed-mean effect was −0.0002 / −0.0004 and which the consolidated report recommends holding.
   SWA's benefit in the paper relies on a high constant or cyclical learning rate during averaging; our frozen loop
   has no schedule, so a faithful SWA arm changes two factors (schedule + averaging). Low expected information.
3. **Label smoothing or focal loss on the calibrated head** (Müller, Kornblith, Hinton, "When Does Label Smoothing
   Help?", NeurIPS 2019, arXiv:1906.02629; Mukhoti et al., "Calibrating Deep Neural Networks using Focal Loss",
   NeurIPS 2020, arXiv:2002.09437). Both target over-confident large image networks; ours are small and already
   temperature-scaled (seed-mean ECE10 dev 0.0019–0.0080, consolidated report), so calibration headroom is small.
   Both replace the proper log-loss we score with a non-log-loss objective, so a lower NLL would have to come
   *despite* the objective; the external temperature absorbs part of the change, blurring the contrast.
   `docs/ML_EXPERIMENT_DESIGN.md:169` already advises against focal/sampling as a default for probability quality.
- Also not chosen: data-side changes (pitcher ID embedding, L/R augmentation) need a NEW export — heavy Mac work,
  blocked under the protected-mode freeze (CONTINUATION T21 "why"); tuning periodic σ / n_frequencies = HPO on
  repeated DEV.

## 4. Network (fixed now)

```
# PLE embedding (version-B-style, written from the package docs)
bins_j  = unique(quantile(TRAIN values of scalar j, linspace(0, 1, 49)))     # j = 1..81, edges b_0 < … < b_{T_j}
        # physics scalars: TRAIN rows whose past position is valid only; context scalars: all TRAIN rows
enc_j(x)= [ clamp_max((x−b_0)/(b_1−b_0), 1), clamp((x−b_{t−1})/(b_t−b_{t−1}), 0, 1) for t = 2..T_j−1,
            clamp_min((x−b_{T_j−1})/(b_{T_j}−b_{T_j−1}), 0) ]                 # T_j values; T_j = 1 → single linear piece
e_j(x)  = w0_j * x + c0_j + Linear_j(T_j → 8)(enc_j(x))                       # w0, c0 [81, 8]; Linear_j weight zero-init, bias zero
e       = e * keep                                                             # padded past positions → exactly 0 (as periodic)

ple_residual_mlp:        raw, e → PeriodicResidualNetwork backbone (input Linear → 2 pre-LN residual blocks with Dropout .1 inside → LN → GELU → Linear(128,10))
ple_tabm_residual_mlp:   raw, e → PeriodicResidualTabMNetwork backbone (r [k,D] scaling, shared residual backbone, W [k,128,10], b [k,10]),
                         mean member CE, ensemble logits = log mean member softmax
```

- Initialisation order (so the backbone equals its periodic counterpart at init for one seed): construct the
  periodic counterpart exactly as now (`PeriodicResidualNetwork` / the P34 `PeriodicResidualTabMNetwork`, same RNG
  order), then delete its `frequencies` and `embedding` parameters, then draw the PLE parameters last (the P32
  `LinearResidualNetwork` already does exactly this, `direct_periodic_residual_model.py:89-101`). Every
  shared backbone parameter (and r, W, b for TabM) then equals the periodic arm's initial value for the same seed.
  The batch order uses `np.random.default_rng(seed)` and is identical across arms. Dropout masks: the network is
  built on CPU, so the extra PLE draws advance only the CPU generator; on CPU (tests) the masks differ, on CUDA the
  dropout stream is the CUDA generator, which those draws do not advance, so ple_residual and periodic_residual may
  draw identical masks (same shapes) — 미측정, not claimed either way [review N2].
- Init of w0, c0: package `LinearEmbeddings` init (to be read and written into the module docstring at module
  time; 미측정 now). Linear_j: zeros (version B).
- Bins are computed inside a `fit` override from the TRAIN arrays only, **before** `super().fit` builds the
  network (the frozen `network(n_context, n_token, length)` hook sees shapes only, so the model stores bins on
  `self` first). Computed with numpy per column (torch.quantile has an input-size limit; ~100 M values here);
  quantiles in float64, cast to the float32 used in the forward pass, `unique` applied AFTER the cast, then
  strictly increasing float32 edges asserted (two float64-distinct edges can collapse in float32 → zero-width
  bin → division by zero) [review N5].
  Bins are saved in the archive, hashed into `fit.json` (`ple_bins_sha256`, per-feature `T_j`), and never
  recomputed at predict time. DEV / mlb_dev / earlystop / temperature rows never enter binning (test below).
- Degenerate columns (DECIDED before module code, review B1 option 1): a column with exactly one distinct value in its
  binning rows gets T_j = 0 (no PLE term; e_j = w0_j·x + c0_j). The rule is identical in smoke and in the full run.
  Every fit records `ple_degenerate_columns` (names); the full-run count is reported.
- Fail closed: non-finite values or edges, a T_j > 48, or float32 edges that are not strictly increasing → `ValueError`
  at fit. Whether any real column triggers this is 미측정 (checked first in the CPU test on
  a TRAIN slice and in smoke). (Context for the B1 decision above:) the runner smoke uses the FIRST
  8,192 TRAIN rows (`run_cuda_direct.py:123,437`), where rare pitch-type-proportion columns (context 33–50) can be
  constant → this rule would abort smoke for a reason unrelated to the full run; a slice passing implies full TRAIN
  passes, a slice failing says nothing.
- Training = `DirectResidualModel.fit` (single head) / `DirectTabMModel.fit` path used by P34 (TabM), unchanged.
  AdamW lr .0005, wd .01 (also on PLE parameters; disclosed), batch 1024, patience 5, cap 60, grad clip 5.

## 5. Family (fixed now; registered only after review + smoke + timing probe + go)

Arms (export `49097109`, workload `counts`, continuous map = P30 `CONTINUOUS['counts']`, as P34):

| # | name | role | definition |
|---|---|---|---|
| 1 | `control` | runner requirement | flatten MLP 128, no workload (exact earlier entry) |
| 2 | `periodic_residual_counts` | reference | EXP-P32-001 / P34 arm unchanged, old code path |
| 3 | `ple_residual_counts` | **new** | PLE (above) + same residual backbone, single head |
| 4 | `tabm_residual_counts` | reference | EXP-P34-001 arm unchanged, old code path (module sha `6a9daff6…87cf`) |
| 5 | `ple_tabm_residual_counts` | **new** | PLE + P34 TabM-mini residual network, k=8, scaling init `normal` |

**Seeds [29, 30, 31, 32, 33]** → 25 fits. Legality: used blocks are 0–2 (P24/P25/P27/P29), 3–7 (P30-001),
8–12 (P30-002, registered `96784b5`, `repeat_tag repeat2_seeds_8_12`), 13–15 (P31), 16–18 (P32), 19–23 (P33),
24–28 (P34; P34 `seed_rule`). 29–33 appear in no config on any local branch (`git grep` over all
`refs/heads`, configs/experiments). The P30 registry text "every other cuda_direct_v1 config keeps seeds 0-2" is
P30-registry-scoped; P31–P34 each enforced their own exact block. The new registry must check exactly
[29..33], no repeat_tag, and its `seed_rule` must list every earlier block. **Wiring review must confirm no shared
validator still rejects non-0–2 seeds for a new registry.** New RNG seeds are not a new evaluation population.

**Epoch cap 60, patience 5** for every arm (same as P34; P34 0/25 cap hits). Not compared with P31/P32 numbers.

**Contrasts** (candidate − reference; same seeds; seed-mean ensemble and single-fit means; game-block bootstrap):

| name | candidate | reference | isolates | held |
|---|---|---|---|---|
| `ple_residual_counts_vs_periodic_residual_counts` | ple_residual_counts | periodic_residual_counts | `embedding_type_single` | input columns, d 8, workload counts, backbone residual, head single, dropout .1 |
| `ple_tabm_residual_counts_vs_tabm_residual_counts` | ple_tabm_residual_counts | tabm_residual_counts | `embedding_type_tabm` | as above + head tabm_mini k 8 |
| `ple_tabm_residual_counts_vs_ple_residual_counts` | ple_tabm_residual_counts | ple_residual_counts | `ensemble_head_ple` | embedding ple, workload counts, backbone residual |
| `periodic_residual_counts_vs_control` | periodic_residual_counts | control | bundle | — |
| `ple_residual_counts_vs_control` | ple_residual_counts | control | bundle | — |
| `tabm_residual_counts_vs_control` | tabm_residual_counts | control | bundle | — |
| `ple_tabm_residual_counts_vs_control` | ple_tabm_residual_counts | control | bundle | — |

Full candidate list = the 4 non-control arms above; 7 contrasts. No extra arms, seeds or contrasts after any score
is read. The `embedding_type × head` interaction (difference of the first two rows) may be printed descriptively;
no interaction verdict.

Headline rows (fixed now, review N13): the two `embedding_type_*` contrasts, NLL on dev and mlb_dev side by side, are
reported first; all other rows follow in the order above. GPU pairing (review N7): no scheduler change (runner is
frozen); the report states the observed number of same-GPU (candidate, reference, seed) pairs per contrast, as in P34.

**No saved-probability ensemble grid for P35 (decided now).** Both sides of each embedding contrast have the same
head and the same fit count, so a fit-count grid answers no registered question. Changing this requires a grid file
committed with time and reason **before** any P35 score exists (T54 rule).

## 6. Compute projection (from measured task seconds; unknowns marked)

Measured P34 task seconds per arm (queue summary via `reports/research-p34/p34-extract.json` `runtime.arms`, task
time incl. loading/calibration/prediction, 5-s poll grid, review F5): control mean 103.0 (max 120.0),
periodic_residual_counts 217.2 (max 245.2), tabm_residual_counts 510.3 (max 605.5). P34 family 8,225.6 worker-s,
43 % of its Amendment-2 projection 19,118 (config `date_note`).

- New arms: **미측정.** Planning assumption only (not a projection): each PLE arm ≈ its periodic counterpart.
  PLE does a [N, 81, ≤48] clamp + per-feature matmul instead of cos/sin of [N, 81, 16] + Linear(32, 8).
- Planning sum: 5 × (103.0 + 217.2 + 217.2 + 510.3 + 510.3) = **7,790 worker-s** ≈ 65 min wall on 2 GPUs.
- P30-002 tasks today run ~2× slower (PM-reported in the task brief; cause unmeasured; not read by this author). At 2×: 15,580 worker-s ≈ 130 min wall; slowest task ≈ 2 × 605 = 1,211 s. These are early-stopped
  actual task seconds; the gate below is a 60-epoch-cap projection and is NOT comparable to them (review B2: at 2×
  the cap projection already exceeds 1,800 s).
- Limits: family budget 36,000 worker-s; task timeout 3,600 s; ≤ 2 attempts per task; GPU3/4 only (UUID-pinned
  `GPU-c334d905…`, `GPU-ae803a24…`), max 2 simultaneous fits; common per-GPU flock; never concurrent with another
  family's queue (start only after EXP-P30-002 is terminal). Memory: 미측정 for PLE (measured: P24 GPU peak 1.52–1.62 GB
  across its 5 arms, `docs/reviews/PM-P24-CUDA-result-2026-10-04.md`; P34 Amendment-2 probe peak allocated 1.41–1.59 GiB
  per arm, `RESULT-AMENDMENT2-TIMING-PROBE-9de203a` (a5fe718); PLE intermediates add [1024, 81, 48] fp32 = 15.9 MB per
  tensor per training batch, 63.7 MB at the 4,096-row predict batch; full TRAIN must never be encoded at once,
  1,252,824 × 81 × 48 × 4 B ≈ 19.5 GB) [review N6].
- **Timing gate before registration — DECIDED (review B2 option 1): Amendment-2 method below, gates unchanged.** The formula written here was the P34 *Amendment-1 smoke*
  formula, which Amendment 2 itself records as overstating full fits 7–16× (P34 smoke projection 7,119–7,549 s):
  ~~(probe TRAIN seconds / probe optimizer updates) × 1,224 updates/epoch × 60 + measured non-train seconds, per GPU,
  slowest of the probe seeds~~. Amendment-2 method (`AMENDMENT2-TABM-RESIDUAL-TIMING-PROBE-20261004.md`, 9de203a):
  task_projected = startup + prepare + 60 × per_epoch (warm, full TRAIN, 2-epoch fit after a 1-epoch warm-up,
  max(wall, report)/2, incl. per-epoch early-stop pass) + predict + 30 s allowance; slower of GPU3/GPU4 per arm;
  probe seed outside the family; family = Σ_arms 5 × slowest. Comparable P34 baseline under that method: control 469 s,
  periodic_residual_counts 994 s, tabm_residual_counts 975 s → with PLE ≈ periodic, family ≈ 5 × (469 + 2 × 994 +
  2 × 975) = 22,035 worker-s. At today's ~2× slowdown: ≈ 1,988 / 1,950 s per residual task (> 1,800) and ≈ 44,070
  worker-s (> 36,000) → gates false for 4/5 arms and the family. Decision: the Amendment-2 method is the
  only projection method; gates are NOT raised. The probe runs once, after EXP-P30-002 is terminal, on GPU3/4 as found;
  if any gate is false (e.g. at today's ~2×) → no registration, the probe record is kept, and the family is redesigned
  or dropped (not re-probed until it passes). Gates: every arm ≤ 1,800 s, family ≤ 36,000 s, probe complete with exit 0.
  Any gate false → redesign before registration (not silently shortened).

## 7. Stop rule

- Any task failing twice aborts the family; no partial comparison; failures reported.
- Timing gate false → redesign before registration.
- No new fit after the launch-day `no_new_fit_after`; own processes killed at `kill_own_after`; deadlines set at
  registration from the measured probe (as P34).
- No extra arms, seeds, contrasts, epochs or grids after any P35 score is read. A bug found after scoring is
  reported with the affected rows; a rerun is a new labelled family, never a silent replacement.

## 8. Scoring (descriptive only)

- Scorer: `experiments/pitchmdp/scripts/score_cuda_direct.py` (portable numpy) with the new arm keys; unblended
  calibrated direct-head probabilities on `dev` (12,334 pitches / 328 games) and `mlb_dev` (311,721 pitches).
- Per arm and contrast: NLL, Brier, top-label ECE10; seed-mean ensemble and single-fit means; per-seed Δ and
  negative-sign counts; whole-game paired bootstrap 95 % intervals (10,000 resamples, seed 20260924, as earlier
  families); epochs_run, best_epoch, stopped_by, cap hits; task seconds; **same-GPU pairing per contrast** (P34 F2).
- PLE diagnostics: per-feature T_j and the bins hash per fit (bins are TRAIN-only, so they are identical across
  seeds and across both PLE arms — all 10 PLE fits must carry one `ple_bins_sha256`; checked) [review N12].
- Output has no `decision`, `ranking`, `winner`, `promotion` or verdict field (null / absent, extractor fail-closed).
  A descriptive extractor (P30-002 hardening pattern: explicit `need()` checks, all seed dicts verified, mutation
  tests, `-O` safe) is committed **before** any P35 score exists.
- Reference scale stated next to results: control single-fit SD (dev 0.00094, mlb_dev 0.00037) and the between-family
  control gap; the old −0.003 threshold is a G0/Cpanel quantity and is quoted only as context.

## 9. Exposure and data

- `dev` / `mlb_dev` are the same ≤2025 rows read many times across P24–P34 (233 fits, 9 families, plus 6 saved-probability
  grids P27/P30–P34) and by P30-002 now. **This is repeated-DEV exploration, not unseen validation.**
- Post-selection: PLE was chosen after reading P24–P34 (embedding and TabM were the largest single-fit axes; P32
  showed embedding type matters at the 0.001 level). Both references were chosen from earlier results.
- 2026 data excluded: exporter `guard_dates` + per-split year check; runner rechecks the manifest. DEV / mlb_dev are
  only predicted, never used for fit, binning, early stop, temperature or selection.
- G0, G15 mix (D145), D131, D155 unchanged. No policy or service claim.

## 10. New code required (additive) and checks

Base for the module code: a branch containing `codex/p34-register-1004` (a5fe718). The DRAFT's own branch
(b0593aa lineage) has no `pitchmdp/direct_*` module at all, and a5fe718 is not its ancestor [review N8].

1. `pitchmdp/direct_ple_residual_model.py`: `PiecewiseLinearEmbedding` (formula above), `PLEResidualNetwork`,
   `PLETabMResidualNetwork`, `DirectPLEResidualModel` / `DirectPLETabMResidualModel` (fit override computes bins,
   own archive formats incl. bins; load refuses other kind, architecture, k, n_bins, d, bins hash or continuous map).
   Frozen modules (`direct_periodic_model`, `direct_periodic_residual_model`, `direct_tabm_model`,
   `direct_tabm_residual_model`, `direct_residual_model`) not edited; their sha256 checked unchanged in a test.
2. `pitchmdp/direct_ple_registry.py` (5 arms, earlier arms = exact P34 entries, seeds exactly [29..33], 25 fits,
   7 contrasts in this order, training = cuda_direct_v1 budget with epochs 60) and `pitchmdp/direct_ple_factory.py`
   (strict build/load, post-load continuous-map and bins checks).
3. Runner / scorer dispatch on new arm keys `ple_residual`, `ple_tabm_residual` (not `tabm` / `tabm_residual`, whose
   routes read other keys). **The P35 family predicate must be checked FIRST** in `run_cuda_direct.config_check` and
   `training_budget` and in `score_cuda_direct` (both currently first-match on any arm carrying the key
   `tabm_residual`, `run_cuda_direct.py:202,211`, `score_cuda_direct.py:190,443`); arm 4 is the exact P34 entry with
   `tabm_residual: true`, so otherwise a P35 config routes to the P34 check (seeds 24–28 only) [review N1]. The
   runner/scorer files change; earlier families' archived identities are unaffected because each family runs from
   its own `git archive` code dir [review N11]; `MODEL_SOURCES` records the new module sha in every task identity.
4. `configs/DRAFT-CUDA-PLE-RESIDUAL-20261004.json`, execution disabled, placeholder deadlines.
5. CPU tests (focused): encoding values vs a hand-computed example incl. first/last bin clamps and T_j = 1;
   version-B zero init ⇒ PLE output equals the linear term at init; bins depend on TRAIN only (perturbing
   earlystop/temperature/dev arrays leaves bins byte-identical); padded past positions excluded from binning and
   embedded to exactly 0; backbone (and r, W, b) init equal to the periodic counterpart for the same seed; one
   optimizer step runs; archive round-trip reproduces logits; load refusals (kind, k, n_bins, bins hash, permuted
   map); registry seed/arm/contrast checks with mutants; frozen-module hashes unchanged.
6. CUDA smoke of **all 25 tasks** on GPU3/4 (runner SMOKE: 8,192 TRAIN rows, 2 epochs). Note: smoke bins come from
   the smoke TRAIN subset, so smoke verifies wiring and identities only, never bins or scores.
7. Timing probe (§6 formula) on GPU3/4 after smoke.

**Must be reviewed before registration:** (a) this pre-registration (same-model review, then Codex); (b) the module
+ tests (distinct Opus review → disposition → module sha pinned after disposition); (c) registry/factory/runner/
scorer wiring (distinct review → disposition); (d) smoke identities 25/25; (e) timing probe method and gates;
(f) the pre-score extractor; then `결정 필요` go (new experiment: purpose, changed setting, 2 GPUs, projected time).

## Commands run by the author (cwd = worktree; all read-only except the final commit)

- `ls` / `cat` / `sed -n` / `head` on `reports/research-inventory/`, `reports/research-p30-p33/`, `reports/research-p34/`,
  `reports/t50b-saved-ensemble-p30/`, `reports/research-p30-002/PRE-SCORE-decisions-ko.md`, `reports/SAVED-ENSEMBLE-P27-001-report-ko.md`,
  `docs/reviews/PM-P2{4,5}*.md`, `docs/reviews/PM-P27-P29-results-2026-10-04.md`, `CONTINUATION.md`, `docs/decisions.md` → exit 0
  (two `echo ====` separators failed under zsh with exit 1 after the reads had printed; no effect on content).
- `git show codex/p34-register-1004:{configs/EXP-P34-001.json, experiments/pitchmdp/docs/DRAFT-TABM-RESIDUAL-DESIGN-20261004.md,
  experiments/pitchmdp/pitchmdp/direct_periodic_model.py, direct_periodic_residual_model.py, direct_residual_model.py}` → exit 0
  (one first attempt had a broken path, exit 1, retried).
- `git show codex/p30-repeat-draft-1004:configs/EXP-P30-002.json` (seeds, seed_rule, queue, training only; no score) → exit 0.
- `git ls-tree` / `git grep` over all local branches for P3x configs/docs and for seeds 29–33, and for
  SWA / focal / label smoothing / heterogeneous / PLE keywords → exit 0 (no seed-29–33 use; no PLE).
- `python3` read of `reports/research-p34/p34-extract.json` `runtime` (task seconds only) → exit 0.
- WebFetch (open pages): arxiv.org/abs/2410.24210, arxiv.org/html/2410.24210v3, arxiv.org/abs/2203.05556,
  github.com/yandex-research/tabm, github.com/yandex-research/rtdl-num-embeddings, raw `package/README.md`,
  raw `package/rtdl_num_embeddings.py` → fetched.
- No ssh/newsrv, GPU, MPS, training, inference, ports, push, merge, deletion, 2026 data, or P30-002 score.
