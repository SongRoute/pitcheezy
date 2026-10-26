# EXP-P48-001 (DRAFT): adoption-seed confirmation of the TabM-residual direct head against the G15 mix (pre-registration)

Status: `author_draft; independent_review_pending; not registered`. Written 2026-10-26 ~23:50 KST by the Opus 5.5
implementation worker on branch `research/direct-head-confirm-1026` (base `fa133ab7` =
`codex/five-gpu-entry-prep-astra-1006`), **before any smoke, fit or score of this family**. No GPU, SSH, `gpu run`,
server write, data fetch or DEV metric in this job; the external SSD was read only for npz key names, dtypes, shapes,
key-order hashes and file sha256 (no NLL / Brier / ECE computed or printed).

Config (DRAFT): `configs/EXP-P48-001.json` sha256 `73dca530ea5f93158be7c0100622761dcb69e51edd1d4c7ac5f4372b595e2ac2`
(status `DRAFT`, `execution.enabled` false, registration block empty). Registry:
`experiments/pitchmdp/pitchmdp/direct_adoption_confirm_registry.py`. Stage-A scorer:
`experiments/pitchmdp/scripts/score_cuda_direct.py`. Stage-B scorer:
`experiments/pitchmdp/scripts/score_cuda_mix_vs_g15.py`. Tests: `experiments/pitchmdp/tests/test_cuda_adoption_confirm.py`,
`experiments/pitchmdp/tests/test_score_cuda_mix_vs_g15.py`.

## 요약 (Song용, 쉬운 말)

- 10/4~10/6 GPU 탐색(P34·P35)에서 `tabm_residual_counts`(잔차 신경망 위에 작은 앙상블 헤드 TabM-mini k=8, 작업량 counts)가
  같은 가족의 단순 직접 헤드(flatten 대조군)보다 손실이 낮게 읽혔다. 그런데 그 비교는 **G0와 섞기 전 헤드끼리** 비교였고,
  지금 연구 기준선인 **G15 혼합**(단순 직접 헤드 + 같은 시드 G0-global을 반반 섞고 6월 빈도 블렌드)과는 한 번도 비교하지 않았다.
- 이번 실험은 G15와 똑같이 다룬다: 채택 시드 0~4로 세 팔(대조군·periodic·tabm)을 GPU에서 다시 학습(15회)하고,
  맥에서 각 헤드를 같은 시드의 동결 G0-global과 섞어 6월 블렌드를 한 뒤 동결 G15와 짝 비교한다(Stage B가 주 비교).
- 섞는 비율은 두 가지로 미리 고정한다: ① G15와 같은 반반(MIX_FIXED, 헤드 차이만 봄) ② 6월 블렌드 행에서만 고른 비율
  (MIX_FIT, 0.50~0.95). 미리 정한 주 후보는 **tabm_residual_counts + MIX_FIT** 하나다.
- 통과 기준은 기존 매트릭스 N 규칙(5시드 중 4시드 개선, ΔNLL ≤ −0.003 등)을 Cpanel과 전체 MLB **두 곳 모두**에서 넘는 것.
- 비유: 2군에서 잘 친 타자를, 1군 주전과 **같은 타순·같은 구장 조건**에 세워 다시 비교하는 것. 다만 같은 경기 기록을
  여러 번 본 상태라 "새 시즌 검증"은 아니다. 통과해도 G15·G0·정책·서비스는 바뀌지 않고, 2027 또는 RS-02식 사전 비교 후보가 될 뿐이다.
- 2026 자료는 쓰지 않는다. 비용은 실측 참고값만 적고 새로 추정하지 않는다(아래 §8).

## 1. What is already known (sources)

- **G15 mix** (`configs/G15-MIX-RESEARCH-FROZEN-v1.json`, sha256 `b4c19988…201e6d`, D145): `G15-direct-g0-mix` = the
  frozen MPS flatten direct head of seed s (EXP-P15-001 seeds 0-2, EXP-P16-001 seeds 3-4) mixed 1/2 with the frozen
  G0-global member of seed s, five-seed mean, June frequency blend; confirmed against G0-global in EXP-P16-001
  (`results/EXP-P16-001.json`, sha256 `9d372da2…778597`) on Cpanel DEV after the blend and whole-MLB DEV blended.
  It is the research prediction baseline; the service, the demo and the validated policy identifier `c13cc989` still use
  G0-global alone.
- **P34/P35 (CUDA, repeated exposed DEV)**: `tabm_residual_counts` vs the same-family flatten control, unblended
  calibrated heads, whole-MLB ΔNLL about −0.0040 to −0.0043 (5/5 seeds) and Cpanel −0.0043 to −0.0061 — figures as given
  in the PM brief of 2026-10-26; the P34/P35 result records are not on this branch and were not re-derived here. These
  are head-vs-flatten-control numbers, never head-vs-G15.
- The CUDA export (`CUDA-EXPORT-20261004`, manifest sha256 `49097109…1f578`, pinned in the config) has the same blend /
  dev / mlb_dev row order as the Mac baselines: the ordered key hash of each split equals the export manifest
  `rows_sha256` for `EXP-P4-001/baseline_predictions.npz` (blend, dev, mlb_dev), `EXP-P11-001/baseline_predictions.npz`
  (blend, dev, mlb_dev), the EXP-P11-001 G0-global seed-0 whole-MLB member and the EXP-P15-001 seed-0 whole-MLB head
  (checked 2026-10-26 on the SSD, keys only). Stage B still aligns by key and fails closed (§6).

## 2. Question

Does the TabM-mini k=8 head on the periodic-residual backbone with workload counts (`tabm_residual_counts`), treated
exactly like the G15 direct head — mixed with the frozen G0-global member of the same seed, then the June frequency
blend — improve on the frozen G15 mix on **both** Cpanel DEV after the blend and whole-MLB DEV blended, under the matrix
N rule, on the adoption seeds 0, 1, 2, 3, 4?

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
- Integrity (all before use): the G15 config, the G0 bundle it pins (`configs/G0-RESEARCH-FROZEN-v1.json`) and
  `results/EXP-P16-001.json` against `rules.reference`; frequency baselines and G0-global Cpanel member states against
  the G0 bundle listing; flatten member states against the G15 listing; every member npz against its adjacent
  `*_state.json` `artifact_hashes`; member model hashes against the frozen member records; the 15 CUDA tasks through
  `score_cuda_direct.verify_family` (identity = this config, pinned export, arm, seed, CUDA, expected task sources;
  intact artifacts; one CUDA runtime; epoch-cap records). A dry run of the frozen-input checks (`references` and
  `load_frozen` with npz headers only: hashes, listings, identities, array names and shapes; no CUDA family exists
  yet) passed against the real SSD inventory on 2026-10-26.
- **Alignment**: every CUDA archive must carry exactly the Mac baseline rows (keys, labels, game and pitcher ids:
  `score_ml_matrix.assert_aligned` semantics) and the no-physics marker; a different row order is aligned by the
  3-column pitch key with a recorded permutation; a missing, extra or repeated key fails closed. Only metadata is read
  in this step.
- **Reproduction gate** (before any candidate probability is read): the recomputed G15 Cpanel-after-blend NLL
  (`cpanel_dev.reports["G15-direct-g0-mix"].primary.log_loss`) and whole-MLB-blended NLL
  (`mlb_dev_blended.comparison["G15-direct-g0-mix"].candidate.log_loss`), and the G0-global values of the same rows
  (`cpanel_dev.reports["G0-global"].primary.log_loss`, `mlb_dev_blended.comparison["G15-direct-g0-mix"].control.log_loss`)
  must equal `results/EXP-P16-001.json` within tolerance 1e-9; otherwise Stage B aborts with nothing scored.

## 6. Stage B candidates, comparisons and decision rule (fixed)

- **Candidates**: each CUDA arm (`control`, `periodic_residual_counts`, `tabm_residual_counts`) under two mix rules,
  then the June frequency blend as usual:
  - `MIX_FIXED`: weight 1/2 for every seed and the ensemble — identical to the G15 rule; isolates the head.
  - `MIX_FIT`: per seed, w on the calibrated June **blend** rows only, minimizing the pitch-weighted NLL of
    w * head + (1 - w) * G0 over the grid 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95 (ties to the
    smaller w); the ensemble weight is chosen the same way on the seed-mean head and seed-mean G0. DEV / mlb_dev rows
    never enter the weight choice.
  - `control` at `MIX_FIXED` is the cross-device reproduction / sanity arm (descriptive); both control rows are
    outside the multiplicity family.
- **Comparisons**: candidate − G15 by `matrix_metrics.paired_game_comparison` (whole-game bootstrap, 10,000 draws, seed
  20260924) on (i) Cpanel DEV after the June blend and (ii) whole-MLB DEV blended; per-seed deltas against the G15
  member of the same seed.
- **N rule** (`matrix_metrics.prediction_decision`, stage `final`): ensemble ΔNLL <= -0.003, paired 95% CI upper < 0,
  Holm-adjusted one-sided p <= .05, Brier CI upper <= 0.001, at least 4/5 seeds improving.
- **Multiplicity**: Holm across the four candidate tests (`periodic_residual_counts`, `tabm_residual_counts` x
  `MIX_FIXED`, `MIX_FIT`) separately per split. A candidate is **confirmed** only if it is a `predictive_improvement` on
  both splits (intersection-union, no further adjustment).
- **Primary candidate** (pre-specified): `tabm_residual_counts` `MIX_FIT`. Every other candidate row is secondary;
  control rows and the unblended head rows are descriptive.
- **Descriptive**: unblended calibrated five-seed head (no G0 mix, no June blend) vs the G15 flatten head on Cpanel DEV
  and whole-MLB DEV; the `MIX_FIT` weights with the June-blend NLL of every grid weight; Brier and ECE10 of every row.
- **Robustness (R)**: not computed — no 12-group robustness helper exists in `score_ml_matrix` or `matrix_metrics`
  (`pitchmdp/matrix_group_metrics.py` has 12 group guardrails, but they need `dev_metadata.parquet` columns and are
  not wired in this pre-registration); the result states this.
- **Outputs**: `<destination>/results.json`, `SUMMARY.md`, `SHA256SUMS` (read-only) with every input sha256, the
  config sha256, git commit, seconds and peak RSS. The destination must not exist and the scorer refuses a destination
  inside a repository `results/` directory (this checkout's and the main checkout's). Stage B holds the Mac's common
  heavy-job lock (`<runs-root>/.heavy.lock`, non-blocking) while it loads and scores.

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
  registration, not silently shortened; a failed reproduction gate stops Stage B with nothing scored (investigate the
  frozen inputs; never relax the tolerance); no extra arms, seeds, mix rules, grid points, contrasts or epochs after any
  score of this family is read.

## 9. What pass / fail means (fixed before any result)

- **Pass** (`tabm_residual_counts` `MIX_FIT` confirmed on both splits): a candidate for a prospective 2027 /
  RS-02-style arm and a Song decision recorded in `docs/decisions.md`. It never changes G15, G0-global, the policy
  identifier `c13cc989` or the service by itself, and it is not a policy claim.
- **Fail**: not confirmed; the ceiling of the direct-head line on the exposed 2025 DEV is documented (no further
  direct-head architecture search on these rows without a new pre-registration); G15 stays the research baseline.
- A pass of any other candidate row is secondary and is not a confirmation of the primary candidate. The control
  `MIX_FIXED` row is read only as a cross-device sanity check (its delta vs G15 is descriptive).
- Prediction only: no policy, service or promotion claim either way.

## 10. Before registration (not done here)

1. Independent review of this pre-registration, the config, the registry, the runner/scorer wiring and the Stage-B
   scorer by an agent other than the author.
2. CUDA smoke of all 15 tasks (`--smoke`, output ending `-smoke`) on the allowed GPUs; smoke verifies wiring and
   identities only.
3. Registration: status `REGISTERED`, `execution.enabled` true, `registration` filled (source commit, DRAFT sha256
   above, pre-registration sha256, smoke record, timing, independent review, go), queue clock fields set; Song's
   explicit go for the new experiment.
4. Stage A via `gpu run` / the queue; `gpu pull` of the family; then Stage B on the Mac with the registered config.
