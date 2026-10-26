# EXP-P48-001 (DRAFT): adoption-seed confirmation of the TabM-residual direct head against the G15 mix (pre-registration)

Status: `author_draft; independent review round 1 (Codex gpt-6-astra, 64d57aad) = FAIL with four required fixes, all
applied 2026-10-27 (§11); re-review pending; not registered`. First written 2026-10-26 ~23:50 KST, revised 2026-10-27
~00:20 KST by the Opus 5.5 implementation worker on branch `research/direct-head-confirm-1026` (base `fa133ab7` =
`codex/five-gpu-entry-prep-astra-1006`), **before any smoke, fit or score of this family**. No GPU, SSH, `gpu run`,
server write, data fetch or DEV metric in this job; the external SSD was read only for npz key names, dtypes, shapes,
key-order hashes, file sha256 and manifest hash records (no NLL / Brier / ECE computed or printed).

Config (DRAFT): `configs/EXP-P48-001.json` sha256 `d8cb1a7f70815f9721903718b5d34785110de8e8a6f9252c2a0bd77acd7a9179`
(status `DRAFT`, `execution.enabled` false, registration block empty; supersedes the round-0 DRAFT `73dca530…e2ac2`,
never registered). Registry: `experiments/pitchmdp/pitchmdp/direct_adoption_confirm_registry.py`. Stage-A scorer:
`experiments/pitchmdp/scripts/score_cuda_direct.py`. Stage-B scorer:
`experiments/pitchmdp/scripts/score_cuda_mix_vs_g15.py`. Tests: `experiments/pitchmdp/tests/test_cuda_adoption_confirm.py`,
`experiments/pitchmdp/tests/test_score_cuda_mix_vs_g15.py`. Review round 1 (verbatim):
`experiments/pitchmdp/docs/reviews/REVIEW-EXP-P48-001-ASTRA-001.md`.

## 요약 (Song용, 쉬운 말)

- 10/4~10/6 GPU 탐색(P34·P35)에서 `tabm_residual_counts`(잔차 신경망 위에 작은 앙상블 헤드 TabM-mini k=8, 작업량 counts)가
  같은 가족의 단순 직접 헤드(flatten 대조군)보다 손실이 낮게 읽혔다. 그런데 그 비교는 **G0와 섞기 전 헤드끼리** 비교였고,
  지금 연구 기준선인 **G15 혼합**(단순 직접 헤드 + 같은 시드 G0-global을 반반 섞고 6월 빈도 블렌드)과는 한 번도 비교하지 않았다.
- 이번 실험은 G15와 똑같이 다룬다: 채택 시드 0~4로 세 팔(대조군·periodic·tabm)을 GPU에서 다시 학습(15회)하고,
  맥에서 각 헤드를 같은 시드의 동결 G0-global과 섞어 6월 블렌드를 한 뒤 동결 G15와 짝 비교한다(Stage B가 주 비교).
- 섞는 비율은 두 가지로 미리 고정한다: ① G15와 같은 반반(MIX_FIXED) ② 6월 블렌드 행에서만 고른 비율(MIX_FIT, 0.50~0.95).
- **확인 검정은 하나뿐**: 미리 정한 주 후보 **tabm_residual_counts + MIX_FIT**가 Cpanel과 전체 MLB **두 곳 모두**에서
  기존 매트릭스 N 규칙(5시드 중 4시드 개선, ΔNLL ≤ −0.003, 단측 p ≤ .05 등)을 각자 넘을 때만 "확인"이다. 나머지 후보 행은
  탐색 결과로만 적고 "확인"이라 부르지 않는다.
- **파이프라인 비교라는 점**: 이번 헤드는 GPU에서 epoch 상한 60으로, G15의 단순 헤드는 맥(MPS)에서 상한 30으로 학습했다.
  그래서 "구조만의 차이"가 아니다. 같은 단순 헤드·같은 반반 혼합인 control/MIX_FIXED 행이 이 파이프라인 차이를 직접 잰다.
- 1차 독립 검토(Codex Astra)에서 실패 판정을 받아 네 가지를 고쳤다(§11): 전체 MLB G0 구성원 5개를 고정 사슬로 확인,
  바꿔치기 거부 테스트, 다중 비교를 주 후보 하나로 정리, 결론 문구 수정.
- 비유: 2군에서 잘 친 타자를, 1군 주전과 **같은 타순·같은 구장 조건**에 세워 다시 비교하는 것. 다만 같은 경기 기록을
  여러 번 본 상태라 "새 시즌 검증"은 아니다. 통과해도 G15·G0·정책·서비스는 바뀌지 않고, 2027 또는 RS-02식 사전 비교 후보가 될 뿐이다.
- 2026 자료는 쓰지 않는다. 비용은 실측 참고값만 적고 새로 추정하지 않는다(아래 §8).

## 1. What is already known (sources)

- **G15 mix** (`configs/G15-MIX-RESEARCH-FROZEN-v1.json`, sha256 `b4c19988…201e6d`, D145): `G15-direct-g0-mix` = the
  frozen MPS flatten direct head of seed s (EXP-P15-001 seeds 0-2, EXP-P16-001 seeds 3-4) mixed 1/2 with the frozen
  G0-global member of seed s, five-seed mean, June frequency blend; confirmed against G0-global in EXP-P16-001
  (`results/EXP-P16-001.json`, sha256 `9d372da2…778597`) on Cpanel DEV after the blend and whole-MLB DEV blended.
  It is the research prediction baseline; the service, the demo and the validated policy identifier `c13cc989` still use
  G0-global alone. Its flatten heads were fitted on MPS with the epoch cap 30 (`configs/EXP-P16-001.yaml`,
  `unchanged_from_g0.training`).
- **P34/P35 (CUDA, repeated exposed DEV)**: `tabm_residual_counts` vs the same-family flatten control, unblended
  calibrated heads, whole-MLB ΔNLL about −0.0040 to −0.0043 (5/5 seeds) and Cpanel −0.0043 to −0.0061 — figures as given
  in the PM brief of 2026-10-26; the P34/P35 result records are not on this branch and were not re-derived here. These
  are head-vs-flatten-control numbers, never head-vs-G15.
- The CUDA export (`CUDA-EXPORT-20261004`, manifest sha256 `49097109…1f578`, pinned in the config) has the same blend /
  dev / mlb_dev row order as the Mac baselines: the ordered key hash of each split equals the export manifest
  `rows_sha256` for `EXP-P4-001/baseline_predictions.npz` (blend, dev, mlb_dev), `EXP-P11-001/baseline_predictions.npz`
  (blend, dev, mlb_dev), the EXP-P11-001 G0-global seed-0 whole-MLB member and the EXP-P15-001 seed-0 whole-MLB head
  (checked 2026-10-26 on the SSD, keys only). Stage B still aligns by key and fails closed (§5).

## 2. Question

Does the TabM-mini k=8 head on the periodic-residual backbone with workload counts (`tabm_residual_counts`), treated
exactly like the G15 direct head — mixed with the frozen G0-global member of the same seed, then the June frequency
blend — improve on the frozen G15 mix on **both** Cpanel DEV after the blend and whole-MLB DEV blended, under the matrix
N rule, on the adoption seeds 0, 1, 2, 3, 4? This is a pipeline comparison, not an architecture-only comparison: the
CUDA heads of this family train with the epoch cap 60, the G15 flatten heads with the cap 30 on MPS (§6, §9).

## 3. Fixed design (Stage A, CUDA; no new model code)

- **Arms** (the exact earlier registered config entries, JSON-equal to EXP-P34-001 / EXP-P35-001, same flags, same
  frozen code paths): `control` (MatrixModel flatten_mlp 128, no workload; the runner's same-family control; the same
  architecture as the G15 flatten head, here on CUDA with the cap 60), `periodic_residual_counts` (PLR-lite embedding +
  residual backbone, counts workload; `embed_residual` flag), `tabm_residual_counts` (TabM-mini k=8 on that backbone;
  `tabm_residual` flag).
- **Seeds**: exactly `[0, 1, 2, 3, 4]`, no repeat tag; 3 arms x 5 seeds = 15 fits. Head seed s is mixed in Stage B
  with the frozen G0-global member of seed s.
- **Budget**: the cuda_direct_v1 budget with the epoch cap 60 (as P34-P37): patience 5, batch 1024, lr 5e-4, uniform
  loss, no HPO. **Calibration**: per-seed temperature on the May 16-31 Cpanel rows (bounds .5-2.5), unchanged.
  **Export**: the EXP-P35-001 export block copied verbatim (manifest sha, splits, shapes, workload columns, context
  layout); last year 2025.
- **Device**: CUDA on a non-empty subset of the five A5000 UUIDs Song allowed on 2026-10-11, `max_parallel` <= 5, no CPU
  fallback; one fit per GPU under the common per-GPU flock; launched through `gpu run` after `gpu st` (never another
  user's process or GPU).
- **Family marker**: top-level `"adoption_confirm": true`. Because the arms keep their `embed_residual` /
  `tabm_residual` flags, the marker predicate is tested FIRST in `config_check`, `training_budget`, `uses_epoch_cap`
  and the Stage-A scorer; without the marker the P34 TabM-residual route would claim (and refuse) the config. The
  registry also pins the Stage-B rules (`comparison.stage_b.rules` must equal `STAGE_B_RULES`), so the GPU run cannot
  start under a changed Stage-B rule. The registry file is in the runner's source identity list (`MODEL_SOURCES`), so
  every task identity carries its sha256.

## 4. Stage A contrasts (descriptive only)

`score_cuda_direct.py` reports every arm vs the same-family CUDA control on unblended calibrated heads (dev, mlb_dev)
and three fixed pairs, the `direct_tabm_residual_registry` pairs of these arms (same isolates / held vocabulary):
`tabm_residual_counts_vs_control` (tabm_residual_bundle_vs_control), `periodic_residual_counts_vs_control`
(residual_bundle_vs_control), `tabm_residual_counts_vs_periodic_residual_counts` (ensemble_head; one TabM-mini fit with
8 members vs one single-network fit, not compute-matched). All 15 fits must exist first; every fit records the epoch
cap. No decision, ranking or winner from Stage A.

## 5. Stage B reference and inputs (Mac CPU)

- Reference **G15** = `run_ml_direct_head.mixed(flatten head, G0-global, 0.5)` -> `score_ml_matrix.summarize_cell`
  (June blend selected on the Cpanel June rows) -> Cpanel primary; whole MLB blended exactly as
  `run_ml_direct_head_confirm.score_arrays` (EXP-P11-001 frequency baseline, Cpanel June weights for the ensemble and
  per seed).
- Frozen inputs (relative to `--runs-root` `ML-MATRIX-20260924`): Cpanel frequency baseline
  `EXP-P4-001/baseline_predictions.npz`, whole-MLB frequency baseline `EXP-P11-001/baseline_predictions.npz`, G0-global
  Cpanel members `EXP-P4-001/members/G0-global/seed{0,1,2}` and `EXP-P10-001/members/G0-global/seed{3,4}`
  (`predictions.npz`: blend/dev + `_raw`, `_delivery_level`, `_keys`, `_y`, `_game_pk`, `_pitcher`), G0-global
  whole-MLB members `EXP-P11-001/members/G0-global/seed{0..4}` (`predictions.npz`: `dev` + the same suffixes), flatten
  heads `EXP-P15-001/{0,1,2}` and `EXP-P16-001/{3,4}` (`predictions.npz` blend/dev and `mlb_predictions.npz` mlb_dev,
  same suffixes). CUDA archives: `predictions.npz` blend/dev and `mlb_predictions.npz` mlb_dev with the same suffixes.
- **Integrity: one pinned chain per frozen input, uniform** (`STAGE_B_RULES.inputs.anchors`, checked before any frozen
  array is read). The G15 config, the G0 bundle it pins (`configs/G0-RESEARCH-FROZEN-v1.json`) and
  `results/EXP-P16-001.json` are pinned by sha256 in `rules.reference`. Then:
  - frequency baselines: listed in the G0 bundle (`p4_frequency`, `p11_frequency`);
  - G0-global Cpanel member states: listed in the G0 bundle (`seed{s}_prediction_state`), archives via the state;
  - flatten head states (both splits): listed in the G15 config (`seed{s}_prediction_state`,
    `seed{s}_mlb_prediction_state`), archives via the state;
  - whole-MLB G0-global members (not listed in the bundle itself): the **EXP-P11-001 analysis manifest**
    (`EXP-P11-001/analysis/manifest.json`), whose sha256 the G0 bundle pins (`p11_manifest`, `aaf5db0d…ebda9`); its
    `inputs` record every P11 member state and archive (`scripts/score_ml_g0_whole.py`), so each member must pass
    bundle -> manifest sha -> manifest inputs -> member state -> artifact hashes, state AND `predictions.npz` both listed.
  A file listed in more than one pinned map (the P11 manifest also lists the G0 Cpanel members and both baselines) must
  agree with every one; any missing or differing link fails closed. Every member npz is also checked against its
  adjacent `*_state.json` `artifact_hashes` and member model hashes against the frozen member records. The 15 CUDA
  tasks pass `score_cuda_direct.verify_family` (identity = this config, pinned export, arm, seed, CUDA, expected task
  sources; intact artifacts; one CUDA runtime; epoch-cap records). A dry run of the frozen-input checks (npz headers
  only: hashes, listings, identities, array names and shapes; no CUDA family exists yet) passed against the real SSD
  inventory on 2026-10-26 (round 0) and again with the P11 manifest chain on 2026-10-27.
- **Alignment**: every CUDA archive must carry exactly the Mac baseline rows (keys, labels, game and pitcher ids:
  `score_ml_matrix.assert_aligned` semantics) and the no-physics marker; a different row order is aligned by the
  3-column pitch key with a recorded permutation; a missing, extra or repeated key fails closed. Only metadata is read
  in this step.
- **Reproduction gate** (before any candidate probability is read), tolerance 1e-9 against `results/EXP-P16-001.json`:
  the recomputed G15 Cpanel-after-blend NLL (`cpanel_dev.reports["G15-direct-g0-mix"].primary.log_loss`) and
  whole-MLB-blended NLL (`mlb_dev_blended.comparison["G15-direct-g0-mix"].candidate.log_loss`), the G0-global values of
  the same rows (`cpanel_dev.reports["G0-global"].primary.log_loss`,
  `mlb_dev_blended.comparison["G15-direct-g0-mix"].control.log_loss`) and the five per-seed G15 − G0 deltas of both
  splits (`cpanel_dev.comparison["G15-direct-g0-mix"].seed_deltas`,
  `mlb_dev_blended.comparison["G15-direct-g0-mix"].seed_deltas`). The aggregate NLL alone is invariant to swapping
  members between seeds; the per-seed deltas, which the 4/5-seed rule uses, are not. Any difference aborts Stage B with
  nothing scored.

## 6. Stage B candidates, comparisons and decision rule (fixed)

- **Candidates**: each CUDA arm (`control`, `periodic_residual_counts`, `tabm_residual_counts`) under two mix rules,
  then the June frequency blend as usual:
  - `MIX_FIXED`: weight 1/2 for every seed and the ensemble — identical to the G15 rule; isolates the head (within the
    pipeline difference below).
  - `MIX_FIT`: per seed, w on the calibrated June **blend** rows only, minimizing the pitch-weighted NLL of
    w * head + (1 - w) * G0 over the grid 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95 (ties to the
    smaller w); the ensemble weight is chosen the same way on the seed-mean head and seed-mean G0. DEV / mlb_dev rows
    never enter the weight choice.
  - `control` at `MIX_FIXED` (same flatten architecture, same 1/2 mix as G15) measures the **pipeline difference**
    directly: CUDA, epoch cap 60, this family's seeds vs the frozen MPS G15 flatten heads fitted with the cap 30. Both
    control rows are descriptive and get no N decision.
- **Comparisons**: candidate − G15 by `matrix_metrics.paired_game_comparison` (whole-game bootstrap, 10,000 draws, seed
  20260924) on (i) Cpanel DEV after the June blend and (ii) whole-MLB DEV blended; per-seed deltas against the G15
  member of the same seed.
- **N rule** (`matrix_metrics.prediction_decision`, stage `final`): ensemble ΔNLL <= -0.003, paired 95% CI upper < 0,
  one-sided p <= .05, Brier CI upper <= 0.001, at least 4/5 seeds improving. The p used is the row's own unadjusted p
  for the primary and the Holm-adjusted p across the three secondary rows for the secondary rows (below).
- **Multiplicity — exactly one confirmatory test.** The pre-specified primary `tabm_residual_counts` `MIX_FIT` is
  **confirmed** only by intersection-union: it must be a `predictive_improvement` on both splits, each split judged with
  its own unadjusted one-sided test at alpha .05 (no Holm: a single prespecified test). The three other candidate rows
  (`periodic_residual_counts` x `MIX_FIXED` / `MIX_FIT`, `tabm_residual_counts` `MIX_FIXED`) are **secondary /
  exploratory**: per split, Holm across those three rows as a descriptive adjustment; a combined both-split outcome is
  labelled `exploratory` (with a `both_splits_predictive_improvement` flag) and is never "confirmed". Control rows and
  the unblended head rows have no N decision.
- **Descriptive**: unblended calibrated five-seed head (no G0 mix, no June blend) vs the G15 flatten head on Cpanel DEV
  and whole-MLB DEV; the `MIX_FIT` weights with the June-blend NLL of every grid weight; Brier and ECE10 of every row;
  the pipeline-difference row above.
- **Robustness (R)**: not computed — no 12-group robustness helper exists in `score_ml_matrix` or `matrix_metrics`
  (`pitchmdp/matrix_group_metrics.py` has 12 group guardrails, but they need `dev_metadata.parquet` columns and are
  not wired in this pre-registration); the result states this.
- **Outputs**: `<destination>/results.json`, `SUMMARY.md`, `SHA256SUMS` (read-only) with every input sha256, the
  pinned chain of every frozen file (`provenance.frozen_anchors`), the config sha256, git commit, seconds and peak RSS.
  The destination must not exist and the scorer refuses a destination inside a repository `results/` directory (this
  checkout's and the main checkout's). Stage B holds the Mac's common heavy-job lock (`<runs-root>/.heavy.lock`,
  non-blocking) while it loads and scores.

## 7. Exposure and seed disclosure

- No 2026 rows anywhere (exporter guard and per-split year check; runner re-checks the manifest).
- Cpanel DEV and whole-MLB DEV are the same repeatedly exposed <=2025 rows used by P15/P16 (G15) and P24-P37; the G15
  numbers were read before (D145) and this family was designed **after** reading the P34/P35 scores
  (post-selection). Even a Stage-B pass is a repeated-DEV result.
- Seeds 0, 1, 2, 3, 4 are the adoption RNG seeds of the G15 flatten heads and the G0-global members, not fresh seeds:
  seeds 0-2 were already MODEL RNG seeds of other arms in P24/P25/P27/P29 and seeds 3-4 of EXP-P30-001 arms (RNG reuse
  across families on the same export: not new data, not a new evaluation population).
- `MIX_FIT` adds one grid-selected weight per seed on the same June rows that also choose the June blend weight; that
  selection is not part of the bootstrap interval.
- The honest confirmation is prospective (2027 season or an RS-02-style arm).

## 8. Expected cost and stop rules

- Measured references only (no new projection here): on A5000 in P35/P37, `tabm_residual_counts` fit ~627 s at 33
  epochs and `control` ~135 s (figures as given in the PM brief of 2026-10-26, not re-derived here).
  `periodic_residual_counts`: no EXP-P35-001 fit record (fit.json) on this branch's docs — not recorded here. (The
  main-branch handoff `docs/handoffs/RESOURCE-and-rehearsal-2026-10-05.md`, commit `badab031`, notes queue-observed task
  wall seconds 260 and 185 for the first two completed P35 `periodic_residual_counts` tasks, co-scheduled on two GPUs;
  that is not a fit record.) Stage B (Mac CPU) cost is not measured; the EXP-P16-001 scoring it extends recorded 4.24 s
  (`results/EXP-P16-001.json` `seconds`).
- Queue bounds (config): task timeout 3600 s, family budget 36,000 worker-s, 2 attempts per task; the clock fields
  (`date` 2026-10-27, `no_new_fit_after` / `kill_own_after` / `aggregate_by`) are placeholders set at registration.
- Stop rules: any task failing twice aborts the family (no partial comparison, failures reported); a projected
  `tabm_residual_counts` task > 1800 s at the 60 cap (smoke / timing record at registration) -> redesign before
  registration, not silently shortened; a broken frozen chain or a failed reproduction gate stops Stage B with nothing
  scored (investigate the frozen inputs; never relax a pin or the tolerance); no extra arms, seeds, mix rules, grid
  points, contrasts or epochs after any score of this family is read.

## 9. What pass / fail means (fixed before any result)

- **Pass** (`tabm_residual_counts` `MIX_FIT` confirmed on both splits): a candidate for a prospective 2027 /
  RS-02-style arm and a Song decision recorded in `docs/decisions.md`. It never changes G15, G0-global, the policy
  identifier `c13cc989` or the service by itself, and it is not a policy claim.
- **Fail**: not confirmed; the ceiling of this direct-head pipeline on the exposed 2025 DEV is documented (not a ceiling
  of direct heads in general; no further direct-head architecture search on these rows without a new
  pre-registration); G15 stays the research baseline.
- This is a **pipeline comparison**: EXP-P48-001 trains every head on CUDA with the epoch cap 60, while the G15 flatten
  head was trained on MPS with the cap 30 (`configs/EXP-P16-001.yaml`, `unchanged_from_g0.training.epochs`); a pass or
  fail is about the whole pipeline, and the control `MIX_FIXED` row (cap 60 CUDA flatten vs cap 30 MPS flatten, same
  mix) measures that pipeline difference directly (descriptive).
- A both-split pass of any other candidate row is reported as exploratory and is never a confirmation, of itself or of
  the primary.
- Prediction only: no policy, service or promotion claim either way.

## 10. Before registration (not done here)

1. Re-review of this revision (review round 2) by an agent other than the author.
2. CUDA smoke of all 15 tasks (`--smoke`, output ending `-smoke`) on the allowed GPUs; smoke verifies wiring and
   identities only.
3. Registration: status `REGISTERED`, `execution.enabled` true, `registration` filled (source commit, DRAFT sha256
   above, pre-registration sha256, smoke record, timing, independent review, go), queue clock fields set; Song's
   explicit go for the new experiment.
4. Stage A via `gpu run` / the queue; `gpu pull` of the family; then Stage B on the Mac with the registered config.

## 11. Review round 1

Independent review: Codex `gpt-6-astra` on `64d57aad` (read-only), **VERDICT: FAIL**; kept verbatim as
`experiments/pitchmdp/docs/reviews/REVIEW-EXP-P48-001-ASTRA-001.md`. Required fixes and where each was addressed:

1. **Whole-MLB G0 members not anchored** (five EXP-P11-001 members were checked only against their own state files).
   Addressed in `scripts/score_cuda_mix_vs_g15.py` (`pinned_maps`, `anchored`, `frozen_member`, `load_frozen`) and
   `STAGE_B_RULES.inputs.p11_manifest` / `.anchors`: every frozen input now has one explicit pinned chain (G0 bundle
   listing, G15 listing, or the EXP-P11-001 analysis manifest pinned by the G0 bundle), the P11 members need both state
   and archive in the manifest, any other map listing the same file must agree, and all chains pass before any frozen
   array is read; the chain of every file is written to `provenance.frozen_anchors` (§5). In addition (beyond the
   request) the reproduction gate now also compares the five per-seed G15 − G0 deltas of both splits, which catches an
   in-memory seed swap that leaves every aggregate NLL unchanged (the reviewer's probe).
2. **Missing rejection tests**: `tests/test_score_cuda_mix_vs_g15.py` adds a resealed P11 member (archive replaced and
   state updated consistently), a P11 seed-array swap (resealed, and whole members moved across seed directories) —
   both abort before any candidate is read — the in-memory seed swap caught by the per-seed gate, every broken link of
   the manifest chain, and anchor rules that must cover every frozen input.
3. **Combined multiplicity claim**: exactly one confirmatory test, the prespecified primary, by intersection-union of
   the two splits with its own unadjusted one-sided p (alpha .05); the three secondary rows get per-split Holm across
   the three as a descriptive adjustment and their combined outcome is labelled `exploratory`, never "confirmed".
   Updated in `STAGE_B_RULES.multiplicity` / `.confirmation` / `.alpha` (registry and `configs/EXP-P48-001.json`
   `comparison.stage_b.rules`, still equal), the scorer (`test_rows`, `decide`, `confirmation`,
   `secondary_exploratory`), the tests and §6.
4. **Overclaiming wording**: §9 and the scorer limits now say "ceiling of this direct-head pipeline on the exposed 2025
   DEV" and disclose the pipeline comparison (CUDA cap 60 vs the G15 MPS flatten head with the cap 30,
   `configs/EXP-P16-001.yaml`); the control `MIX_FIXED` row is labelled the pipeline-difference row (§2, §6, §9).

Item 4 of the review's own list (smoke / timing, source and hash, review and execution-registration prerequisites)
remains open by design: §10.
