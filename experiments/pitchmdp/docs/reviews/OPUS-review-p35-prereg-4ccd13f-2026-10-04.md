# Opus 5.5 review — EXP-P35-001 PLE embedding pre-registration DRAFT (4ccd13f)

Label: `same_model_review; Codex_review_pending`. Reviewer: a distinct Opus 5.5 process (not the author).
Written 2026-10-04 on `codex/p35-prereg-review-1004`. No other model was called, no WebFetch/WebSearch was used,
and no further Claude process was spawned. Read-only except this file and the text-fix commit `9454076`.
No ssh, GPU, port, training, inference, install or 2026 data. No test suite run (there is no new code).

Target: `experiments/pitchmdp/docs/PREREG-DRAFT-EXP-P35-001-ple-embedding-2026-10-04.md` at 4ccd13f
(`git diff b0593aa 4ccd13f` = this one file, 318 lines). Line numbers `D:n` refer to the 4ccd13f version.
Frozen code was read from `codex/p34-register-1004` (a5fe718), because this branch has no `pitchmdp/direct_*`
module (0 files).

## 요약 (Song용)

- 판정: **먼저 고칠 것 있음(fix first)**. 모듈 코드를 쓰기 전에 B1·B3, 시간 측정 스크립트 전에 B2를 정해야 한다.
- 가설 자체는 새것이 맞다(198개 브랜치 어디에도 PLE 코드·팔 없음). §1 숫자 25개 이상 대조해 1개만 틀림(P30 최고 칸 이름, 고침).
- 가장 큰 문제: 시간 기준식이 P34에서 7~16배 과대로 버린 옛 식이다. P34와 같은 방식으로 다시 계산하면 오늘처럼 서버가 2배 느리면 4/5팔과 가족 전체가 기준을 넘는다.

## Overall disposition

**Fix first.** The hypothesis is genuinely new here, the one-factor bundle is honestly disclosed, the backbone-init
plan is achievable, seeds 29–33 are free, and the exposure / no-verdict / no-grid text is complete. Three items must
be settled before code or probe:

- **B1** (before module code): fail-closed binning vs the 8,192-row smoke slice — decide the degenerate-column rule.
- **B2** (before the probe script; the formula is "fixed now"): wrong timing formula; with the right one the family
  fails its own gates at today's ~2× slowdown.
- **B3** (before module code for formula details; before registration for the citations): every §2 quote and package
  fact came through WebFetch summaries; re-read the raw source/PDF at a pinned commit.

`결정 필요` items for the PM are at the end.

## Blocking findings

### B1 — Fail-closed binning will plausibly abort the smoke on the first 8,192 TRAIN rows (decide the rule now)

- Where: D:170-172 (fewer than 2 distinct TRAIN values → `ValueError`), D:293-294 (smoke bins from the smoke subset).
- Evidence: the runner smoke loads the **first** rows, not a random sample: `SMOKE = {'train': 8192, …}`
  (`run_cuda_direct.py:123`) and `arrays[name] = value[:limit]` (`run_cuda_direct.py:437`). The 41-column `counts`
  continuous map is `BASE_PERIODIC` (11..51 without 32) + 52 (`direct_workload_dropout_registry.py:48`,
  `direct_combo_registry.py:49-50`). Context 33–50 are 18 pitcher pitch-type proportions, made per pitcher from a
  TRAIN crosstab and z-scored across pitchers (`matrix_sharing.py:24-35`). For a rare type such as a knuckleball or
  screwball, every row of every pitcher who never throws it carries the same value. A slice of a few days of games
  can therefore hold a single value for that column → smoke aborts for a reason unrelated to the full run.
  The logic is one-way: if a slice has ≥ 2 values, full TRAIN does too, but a failing slice says nothing about full TRAIN.
  Full TRAIN probably has ≥ 2 values for every column (the types come from the TRAIN crosstab), but this is 미측정.
- Failure: smoke 25/25 cannot pass → the rule gets changed after a run has been seen, or smoke gets skipped.
- Suggested text (replace D:170-172; recommended option):
  > Degenerate columns (decided now): a column with exactly one distinct value in its binning rows gets T_j = 0 (no
  > PLE term; e_j = w0_j·x + c0_j, i.e. a learned constant for a constant input, the same information every arm
  > has). The rule is identical in smoke and in the full run. Fits record `ple_degenerate_columns` (names) and the
  > full-run count is reported. Fail closed (`ValueError`) only for non-finite values, T_j > 48, or float32 edges
  > that are not strictly increasing.

  Alternative: keep fail-closed for the full run, and before module review run a read-only CPU count of distinct
  values on full TRAIN for all 81 scalars, with smoke bins taken from that full-TRAIN pass. This costs more Mac/newsrv
  work and still needs a smoke rule.

### B2 — The timing-gate formula is the Amendment-1 smoke formula, and the cost section compares the wrong numbers

- Where: D:234-237 ("Amendment-2 style, formula fixed now"), D:227-229.
- Evidence: the D:235-236 formula is word-for-word the P34 **Amendment-1** formula
  (`DRAFT-TABM-RESIDUAL-DESIGN-20261004.md:120-123`; `EXP-P34-001.json` `queue.stop_rule`). Amendment 2
  (`AMENDMENT2-TABM-RESIDUAL-TIMING-PROBE-20261004.md`, 9de203a) replaced it because that formula "overstates the
  three arms whose full fits exist by about 7–16×". Under it, P34 projected 7,119–7,549 s and failed the gate. The
  Amendment-2 projection is: startup + prepare + 60 × per_epoch (warm 2-epoch full-TRAIN fit after a 1-epoch warm-up,
  max(wall, report)/2, including the per-epoch early-stop pass) + predict + 30 s, taking the slower GPU per arm, with
  one probe seed (1000) outside the family. The draft's "slowest of the probe seeds" matches neither amendment.
- Cost consequence (reviewer arithmetic from `RESULT-AMENDMENT2-TIMING-PROBE-9de203a` in a5fe718: control 469 s,
  periodic_residual_counts 994 s, tabm_residual_counts 975 s), using the draft's own planning assumption PLE ≈ periodic:
  family ≈ 5 × (469 + 2·994 + 2·975) = **22,035 worker-s** at P34 speed. At the ~2× slowdown reported today:
  ≈ 1,988 s (both residual single-head arms) and ≈ 1,950 s (both TabM-residual arms) per task, all > 1,800, and
  ≈ 44,070 worker-s > 36,000. So **4/5 arms and the family fail the gates at 2×**. D:228-229 concludes the gate
  only fails at 3× because it doubles early-stopped *actual* task seconds (max 605 s), which are not what the
  cap-60 gate projects.
- Suggested text: applied in `9454076` (formula struck through, Amendment-2 method written out, comparable numbers
  added, marked OPEN). The decision itself is left to the PM (see 결정 필요 1).

### B3 — The §2 quotes and the package facts that define "version B" depend on WebFetch summaries

- Where: D:86-102 ("verified 2026-10-04"), D:15-16 (Korean summary), D:145-149, D:163.
- Evidence: per the task brief, the author's WebFetch calls were summarized by claude-haiku-4-5. The repo's own policy
  (`reports/claim-evidence/CLAIMS-evidence-ko.md:28` on `codex/continuous-pm-1004`) treats web-summary citations as
  미확인 until re-read from the original. Everything below came through those summaries, and **I could not
  verify any of it** (no web in this review): the arXiv abstract quote; TabM §3.4 / §A.2 quotes and their section
  numbers; README "recommended `version="B"`, `activation=False`, starting point `d_embedding=12`"; `compute_bins`
  default 48 and its `unique()`; "version B adds `linear0` and zero-initialises the PLE linear layer".
- Specific points the module formula depends on:
  1. **Bias of Linear_j.** D:149 says "bias zero" and D:113 counts +8 per feature. My recollection, unverified, is
     that the package builds the PLE linear layer with `bias = (version != 'B')`, so there is **no bias** in B. A
     zero bias is redundant with c0, adds 648 parameters, and is still decayed by AdamW.
  2. **`LinearEmbeddings` init.** This is deferred at D:163. My recollection, unverified, is U(−d^−½, d^−½) for weight
     and bias, the same as the repo's own LR init, which is marked "not checked against the package"
     (`direct_periodic_residual_model.py:99-101`).
  3. **Padded layout for ragged T_j.** The package pads to max bins and keeps the last piece's `clamp_min` on the
     feature's own last bin. The test list should include mixed T_j in one padded batch (see N4).
  4. **Package commit.** D:97 says "commit not pinned yet". The periodic module already cites rtdl-num-embeddings
     commit `a8fc250` (`direct_periodic_model.py:7`). Reuse it if version B exists there; otherwise pin and state the
     difference.
- Suggested action: read the raw `rtdl_num_embeddings.py` and README at a pinned commit (`git clone`/raw `curl` is
  a read, not an install) and the TabM PDF. Record exact quotes with line/page numbers in §2 before writing the
  module. The "verified" wording and the Korean "강한 변형" claim are fixed in `9454076` (marked 미확인; the summary
  now says only that the paper reports a PLE variant, TabM†).

## Non-blocking findings

- **N1 — Runner/scorer dispatch is first-match on the arm key `tabm_residual`.** In `run_cuda_direct.py:196-197,202,211`
  `uses_tabm_residual` is true for any config containing an arm dict with key `tabm_residual`. The same holds in
  `score_cuda_direct.py:190,443`. P35 arm 4 is the exact P34 entry with `tabm_residual: true`, so a P35 config would
  go to the P34 `check_family` (seeds exactly 24–28) and be rejected. The 0–2 fallback validator D:194 worries about
  is `run_cuda_direct.py:226-227`; it is only reached when no family predicate matches. Requirement "P35 predicate
  checked first" added to §10.3 in `9454076`.
- **N2 — Dropout-mask statement (D:161-162) is probably wrong on CUDA.** `fit` seeds with `torch.manual_seed`, builds
  the network on CPU, then calls `.to(device)` (`direct_residual_model.py:103-105`, `direct_tabm_model.py:204-206`).
  The extra w0/c0 draws advance only the CPU generator. CUDA dropout uses the CUDA generator, and the single-head
  shapes are equal, so ple_residual and periodic_residual may draw identical masks. On CPU the masks differ.
  This is PLAUSIBLE, 미측정. Reworded in `9454076` without claiming either way. The comparison to P34 does not
  hold: TabM vs single-head dropout shapes differ ([N,k,256] vs [N,256]), so those masks differ regardless.
- **N3 — Parameter growth should be stated now.** I reproduced periodic_residual_counts = 244,898 exactly from
  n_raw = 5·30 + 38 + 6 + 12 = 206 and D = 206 + 81·8 = 854 (input 109,440 + blocks 132,352 + output 1,546 +
  embedding 1,560). This equals the P34 probe table. Upper bound at all T_j = 48: ple_residual_counts ≤ 276,386
  (+12.9 %) and ple_tabm_residual_counts ≤ 292,248 (+12.1 % vs 260,760). Each is 648 lower without a Linear_j bias.
  Capacity moves together with the embedding. P25 width-only capacity showed no gain (`PM-P25-CUDA-result:31`); that
  is context, not a control. Added to §2 in `9454076`.
- **N4 — Implementation constraint, not applied.** Use one padded weight [81, 48, 8] with a per-feature mask and a
  single einsum, not a Python loop of 81 `nn.Linear` (about 240 extra kernel launches per update; P34 runs at about
  12–13 ms/update). Tests: padded weight entries stay exactly 0 after a step, and a padded batch with mixed T_j
  (1, 2, 48) gets the right clamps. FLOPs are small either way: 1024·81·48·8 ≈ 31.9 M MAC per batch vs about 248 M
  for the single-head backbone.
- **N5 — float32 edge collapse.** If quantiles are computed in float64 and `unique` runs before the cast, distinct
  edges can merge in float32 → zero-width bin → inf/NaN. Text added to §4 in `9454076`.
- **N6 — Memory.** The draft cites only P24 (1.52–1.62 GB). The P34 Amendment-2 probe measured 1.41–1.59 GiB peak
  allocated per arm on the same export. [1024, 81, 48] fp32 = 15.9 MB per tensor (training), 63.7 MB at the 4,096-row
  predict batch. Full TRAIN must never be encoded at once (≈ 19.5 GB). Added to §6 in `9454076`.
- **N7 — GPU pairing, not applied.** P34 F2: GPUs split by arm, so the backbone contrasts had 0/5 same-GPU pairs
  (`REPORT-P34…:29`). Same-GPU CUDA trajectory identity was recorded only for the P33 EMA path (consolidated report,
  "주의할 점"). Suggest pre-declaring either a seed-major order that puts each (candidate, reference, seed) pair on one
  GPU, or a smoke check that one PLE task and one periodic_residual task are bitwise identical across GPU3/GPU4.
- **N8 — Base branch.** This branch has 0 `direct_*` modules, and a5fe718 is not an ancestor of 4ccd13f
  (`git merge-base --is-ancestor` exit 1). The module code must start from a branch containing a5fe718. Added to §10.
- **N9 — §1 accuracy.** One error: D:43-44 "best seed-mean Δ about −0.0025 dev (periodic_counts_d20,
  periodic_block_d20)". The best dev seed-mean is periodic_counts_d10, −0.00273 (consolidated report line 41); the best
  mlb_dev is block_d20, −0.00221. Fixed. Also added: a legend (pairs = seed-mean Δ dev / mlb_dev) and the one prior
  mention of PLE (`docs/reviews/PM-research-frontier-2026-10-03.md:73`, candidate list, not run).
- **N10 — Names.** "DirectTabMResidual" does not exist. The network is `PeriodicResidualTabMNetwork`
  (`direct_tabm_residual_model.py:54`); the model is `DirectTabMResidualModel` (:98). Fixed. D:152 dropout placement
  reworded: dropout sits inside the blocks.
- **N11 — "earlier families' runner source hashes unaffected"** (D:284). The runner/scorer files do change. What stays
  valid is each earlier family's archived identity, because each family runs from its own `git archive` code dir.
  Reworded.
- **N12 — Bins identity.** Bins depend on TRAIN only, so all 10 PLE fits (both arms) must carry one
  `ple_bins_sha256`, not just per-arm equality across seeds. Added to §8.
- **N13 — Headline rows, not applied.** Optionally name the rows reported first now (the two `embedding_type_*`
  contrasts, NLL, dev and mlb_dev side by side) so emphasis cannot be chosen after scores.
- **N14 — Input scaling (Q2): no problem found.** Physics channels are TRAIN mean/std standardized
  (`sequence_data.py:127-150`; P34 config `export.normalization`). Pitcher profile columns are z-scored across
  pitchers (`matrix_sharing.py:33-35`). Support is log1p(count)/10. Batter priors, reliabilities and "five soft
  memberships" are rates/weights in [0, 1] (P34 config `context_layout`). `prior_records_per100` scale was not
  checked. So `w0·x + c0` is O(1)-scaled. One disclosure is missing: the first and last PLE pieces are unbounded, so
  DEV values outside the TRAIN range extrapolate linearly (package behaviour, per the draft's own formula D:147-148).

## Requested checks — results

### 1. Novelty and §1 numbers

- PLE is untested here. I ran `git grep -i -E "piecewise|compute_bins|ple_residual|ple_tabm|ple_embed|ple_bins"` and
  `git grep -w PLE` over all 198 `refs/heads`. Hits: only the P35 draft itself, `PM-research-frontier-2026-10-03.md:73`
  (a candidate list), and the PM's own CONTINUATION/claims notes about this draft. No module, arm or config.
- Spot-checks (all match the draft except one):
  1. P24 periodic dev −0.00178 [−0.00336, −0.00020] — `PM-P24-CUDA-result:34`; counts/block/residual dev intervals
     include 0 — :31-33.
  2. P25 dropout −0.00115 — `PM-P25:36`; no_batter +0.00586 — `PM-P25:33`.
  3. P27 periodic+counts −0.00321, periodic+block −0.00337, data 25 % +0.00690 — `PM-P27-P29:18,19,21`.
  4. P29 −0.00047 [−0.00154, +0.00058] and residual dropout +0.00002 — `PM-P27-P29:27`.
  5. SAVED-ENSEMBLE-P27 −0.00049 / −0.00053, +0.00066 / +0.00060 — `SAVED-ENSEMBLE-P27-001-report-ko.md:30-31,36-37`.
  6. P31 tabm_counts −0.00429 / −0.00317 — consolidated:50; 3/6 at cap, about 4× time — consolidated:7.
  7. P32 −0.00394 / −0.00325, −0.00268 / −0.00313, −0.00173 / −0.00231 — consolidated:54,57,58.
  8. P33 −0.00024 / −0.00042, single-fit −0.00135 / −0.00162 — consolidated:61; 0.0009–0.0011 — consolidated:73.
  9. Noise: SD 0.00094 / 0.00037, gap 0.0015 / 0.00045 — consolidated:27-28; seed averaging about 0.002 — :71.
  10. P34 −0.00433 / −0.00395, ensemble_head −0.00163 / −0.00094, 0/25 cap, backbone 0/5 same GPU —
      `REPORT-P34…:21,25,28,29`; saved |median| ≤ 0.00013 and 2/2 −0.00058 5/5 — :54, :46.
  11. 233 fits / 9 families = 208 (`INVENTORY-P24-P34-ko.md:11`) + 25 (P34).
  12. dev 12,334 / 328 games — `configs/EXP-P9-001.yaml:19`; mlb_dev 311,721 — `PM-P24:36`; bootstrap 10,000 /
      20260924 — `score_cuda_direct.py:83`; `ML_EXPERIMENT_DESIGN.md:169` focal citation correct.
  - **Mismatch:** P30-001 best cell (N9), fixed.

### 2. One-factor validity

- What changes (code-read): the map (periodic `ReLU(Linear(32,8)(cos,sin(2πcx)))`, `direct_periodic_model.py:97-98`
  → linear term + zero-init PLE, no activation), parameter count (N3), the TRAIN-fitted bins buffer, and indirectly
  the global grad-norm clip at 5 (`direct_residual_model.py:120`), whose norm now includes the PLE parameters. AdamW
  wd .01 applies to every parameter (`direct_residual_model.py:106`, `direct_tabm_model.py:207`); this is disclosed
  at D:174. The grad-clip coupling is not disclosed. It belongs to the same bundle; add one phrase if desired.
- Init-order plan: **achievable.** `PeriodicDirectNetwork.__init__` draws frequencies, then the embedding, then the
  discarded flatten head (`direct_periodic_model.py:87-92`). `PeriodicResidualNetwork` then draws input, blocks and
  output (`direct_periodic_residual_model.py:64-74`). The TabM network then draws scaling, W and b
  (`direct_tabm_residual_model.py:59-70`). Deleting `frequencies`/`embedding` and drawing w0/c0 last leaves every
  backbone/r/W/b value equal for one seed. P32 `LinearResidualNetwork` already does exactly this (:89-101).
  `forward` in both residual networks calls `self.embed` (:86; tabm :77), so overriding `embed` is enough.
- Confounds beyond §2: the dropout-mask nuance (N2) and the grad-clip coupling above. Neither is worse than disclosed.

### 3. Leakage / binning

- TRAIN-only: `fit(train, labels, calibration=early-stop, …)` receives early-stop rows as a separate argument
  (`run_cuda_direct.py:677-679`), so binning `train` only is clean. Padded past positions are zeros in the arrays
  (`sequence_data.py:225`) and must be excluded by `valid`, as D:146 says. The current token is never embedded.
- Discrete / near-constant columns: the batter rates/reliabilities/soft memberships are continuous with point masses,
  and the 18 type proportions have heavy point masses (T_j can be 1–2 on full TRAIN; ≥ 2 distinct values is likely but
  미측정). The smoke slice is the real risk (B1). Recommendation: deterministic T_j = 0 rule pre-declared now (B1).

### 4. Seeds 29–33

- I parsed every `seed*` list in 233 unique config blobs (`configs/`, `experiments/pitchmdp/configs/`) across 198
  branches. Seeds used: 0–28 only. 29–33 appear in no config, and no `SEEDS =` constant in pitchmdp code contains them.
  P30-002 is `[8..12]` with `repeat2_seeds_8_12` (`codex/p30-repeat-draft-1004:configs/EXP-P30-002.json`; seeds only,
  no score opened).
- Validators: the 0–2 fallback is at `run_cuda_direct.py:226-227`. `direct_frontier_registry.py:72` and the matrix
  families fix their own seeds and are not on the P35 path. The real hazard is first-match dispatch (N1).

### 5. Compute

- 7,790: 5 × (103.034 + 2·217.169 + 2·510.303) = 7,789.9 ✓ (`p34-extract.json` `runtime.arms`). 8,225.6 / 19,118 =
  43 % ✓. 1,224 updates/epoch = ⌈1,252,824 / 1024⌉ ✓ (TRAIN n from `configs/EXP-P9-001.yaml:19`; the P34 probe result
  states 1,224). The manifest n for export 49097109 was not opened.
- The timing-gate formula is inconsistent with Amendment 2, and the gate fails at 2× (B2). Memory (N6) and
  per-feature matmul cost (N4) are not a concern if the encoding is vectorized.

### 6. Pre-registration completeness

Present: question (D:75-82), baselines (arms 2 and 4, exact earlier entries), 7 contrasts in fixed order, fixed
candidate list / seeds / budget / timeouts, export provenance, cost, stop rule, repeated-DEV and post-selection
statements (§9), no verdict fields with a fail-closed extractor (§8), and no saved-ensemble grid decided pre-score
(D:214-216).

Must fix before code: B1, B2, B3.

Overclaims: "verified" (B3), "강한 변형" (B3), and "Amendment-2 style" (B2). All three are fixed in text. Nothing
implies unseen validation; §9 says the opposite explicitly.

### 7. WebFetch / Haiku summaries

Yes. Every quote in §2 (both papers and the package README/source) came only from WebFetch. They must be re-read from
the PDF and the raw source at a pinned commit before registration. The facts that fix the module formula must be
re-read before module code (B3).

## Text fixes applied (commit `9454076`, DRAFT only)

1. Status block: pointer to this review.
2. Korean summary: "강한 변형" → "변형(TabM†)… 원문 PDF 재확인 전까지 미확인".
3. §1 legend for "a / b" pairs.
4. §1 P30-001 best cells corrected (N9).
5. §1 prior mention of PLE in the 10/3 frontier list.
6. §2 "verified" → "read through WebFetch summaries … 미확인" (B3).
7. §2 parameter-count upper bounds (N3).
8. §4 class names and dropout placement (N10).
9. §4 dropout-mask RNG wording, plus a pointer to the P32 precedent (N2).
10. §4 float32 `unique` after cast (N5).
11. §4 B1 marked OPEN with the smoke-slice evidence (rule not changed).
12. §6 memory numbers (N6).
13. §6 early-stopped vs cap-projection caveat.
14. §6 timing formula struck through, Amendment-2 method and comparable numbers written out, marked OPEN (B2;
    gates unchanged).
15. §8 single bins hash across all 10 PLE fits (N12).
16. §10 base branch (N8).
17. §10.3 dispatch-first requirement and source-hash wording (N1, N11).

No arm, seed, contrast, budget, gate value or stop rule was changed.

## 결정 필요

1. **Timing at 2× (B2).** (1) Keep the Amendment-2 method and the 1,800 s / 36,000 s gates, and run the probe only when
   GPU3/4 are back near P34 speed; at 2× the gate is false → no registration. (2) Before any probe, pre-declare a
   higher per-task gate (e.g. 2,400 s, still under the 3,600 s timeout) and a higher family budget, with the reason.
   (3) Lower the epoch cap — not recommended: the reference arms would no longer be the exact P34 entries.
   **Recommend 1:** it keeps the pre-registered gates honest, and the slowdown cause is unmeasured.
2. **Degenerate columns (B1).** (1) Deterministic T_j = 0 rule, the same in smoke and full run, reported per fit.
   (2) Keep fail-closed, plus a read-only full-TRAIN distinct count before module review and full-TRAIN bins for smoke.
   **Recommend 1:** a constant column carries no information for any arm, and it needs no extra data pass during
   the protected Mac period.

## Commands run (cwd = this worktree)

- `git diff --stat b0593aa 4ccd13f`; `wc -l` DRAFT → exit 0.
- `ls experiments/pitchmdp/pitchmdp | grep direct` → no output (0 modules on this branch); `git ls-tree -r --name-only
  codex/p34-register-1004 | grep pitchmdp/direct` → exit 0.
- `git show codex/p34-register-1004:experiments/pitchmdp/pitchmdp/{direct_periodic_model, direct_periodic_residual_model,
  direct_residual_model, direct_tabm_model, direct_tabm_residual_model, direct_tabm_residual_registry,
  direct_tabm_residual_factory, direct_embed_residual_registry, direct_tabm_registry}.py > /tmp/p35rev/…` → exit 0 each;
  `git merge-base --is-ancestor codex/p34-register-1004 HEAD` → exit 1 (not an ancestor).
- `cat -n` of those files; `git show …:{direct_workload_dropout_registry, direct_dropout_registry, direct_combo_registry,
  direct_frontier_registry, matrix_sharing, sequence_data}.py` excerpts → exit 0.
- `grep -n` over `docs/reviews/PM-P24/P25/P27-P29…`, `reports/research-p30-p33/REPORT…`, `reports/research-p34/{REPORT,
  RESULT}…`, `reports/SAVED-ENSEMBLE-P27-001-report-ko.md`, `reports/t50b-saved-ensemble-p30/RESULT…`,
  `reports/research-inventory/INVENTORY-P24-P34-ko.md`, `docs/ML_EXPERIMENT_DESIGN.md` → exit 0.
- `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 perl -e 'alarm 300; exec @ARGV' python3 -c …` read `p34-extract.json` `runtime`
  and `/tmp/p35rev/EXP-P34-001.json` (export, arms, device, seeds, training, queue) → exit 0.
- `git show codex/p34-register-1004:…/DRAFT-TABM-RESIDUAL-DESIGN-20261004.md`, `git show 9de203a` (stat) and the
  AMENDMENT2 doc, `git show a5fe718:…RESULT-AMENDMENT2-TIMING-PROBE…` → exit 0.
- `git grep` of PLE keywords over all 198 `refs/heads`. The first attempt passed the ref list as one unsplit zsh word and
  printed nothing (invalid). Redone with `${=R}` → exit 0. A broad `ple_` pattern was too noisy (`sample_`), so it was
  narrowed.
- `git ls-tree -r` of `configs/` and `experiments/pitchmdp/configs/` on all refs → 233 unique blobs; bounded python
  (`alarm 300`, 1 thread) parsed every `seed*` list → exit 0, no 29–33.
- `git grep` of seed validators/dispatch in `run_cuda_direct.py`, `score_cuda_direct.py`, `pitchmdp/*registry*.py` →
  exit 0. One attempt with a `*queue*.py` glob failed under zsh ("no matches found", exit 1) and was redone without it.
- `git show codex/p30-repeat-draft-1004:configs/EXP-P30-002.json` → printed seeds/repeat_tag/seed_rule only → exit 0.
- `shasum -a 256` of the 5 frozen modules (tabm_residual = 6a9daff6…87cf ✓ D:185) → exit 0.
- `git commit` of DRAFT text fixes → `9454076`, exit 0.
