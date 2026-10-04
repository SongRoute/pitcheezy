# REPORT — robust-clip family DRAFT wiring (tentative EXP-P36-001), author only (T101b)

Status: `author_only; same_model_review_pending; Codex_review_pending`. Not reviewed. Written 2026-10-04 by the Opus 5.5
implementer child (v5 cycle-022) on `codex/robust-clip-wiring-1004`, parent `8311fa4`. Mac CPU only, synthetic data
only. No GPU, no CUDA, no ssh, no training on real data, no score or results directory read, no 2026 data. Nothing
registered: there is no `configs/EXP-P36-001.json`.

## 요약 (Song용)

- 새 실험 묶음(4팔 × 시드 34–38 = 20회)을 러너·채점기에 연결했다. 기존 실험 경로는 바뀌지 않았다(기존 테스트 통과).
- 새 테스트 142개 통과. 일부러 넣은 버그 9개 중 8개를 테스트가 잡았다. 못 잡은 1개는 같은 검사가 두 번 들어 있어 남은 쪽이 막은 경우다.
- 설계 변경 없음. GPU 검증은 하지 않았다(금지).

## Files

- `experiments/pitchmdp/pitchmdp/direct_robust_clip_registry.py` (new, torch-free): 4 arms in PREREG §3 order,
  seeds exactly [34..38] (overlap with 0–33 refused), 20 fits, budget = P35 `training` (epochs 60), contrasts primary
  `robust_clip_residual_counts_vs_residual_counts` (`input_preprocessing`) and secondary
  `robust_clip_residual_counts_vs_periodic_residual_counts` (`numeric_input_treatment`, `compute_matched: false`),
  `MODULE_SHA256 = 22841f2c82b75099cc58ac5e78cda6b1fc4a387366ccb0944b7b6f41cbbee3f2` (the only pin; `check_module`
  fails closed), fit.json record helpers.
- `experiments/pitchmdp/pitchmdp/direct_robust_clip_factory.py` (new): builds only the new arm (41-column counts map);
  strict load against the run-recorded `rc_stats_sha256`; module pin checked on build and load.
- `experiments/pitchmdp/scripts/run_cuda_direct.py` (additive): `uses_robust_clip` checked FIRST in `training_budget`
  and `config_check` (needed: `periodic_residual_counts` keeps its `embed_residual` flag, so the P32 family check would
  otherwise catch this config); per-arm check; build/load dispatch; fit.json `rc_stats_sha256`, `rc_rule_counts`,
  `rc_stats_rows` (`train_rows` + per-past-position valid rows), `rc_module_sha256` (also compared with the task
  identity source map); `RC_STATS_ROWS` log line; 3 files added to `MODEL_SOURCES`.
- `experiments/pitchmdp/scripts/score_cuda_direct.py` (additive): `robust_clip_contrasts` accepted only for this
  family, checked first; per-fit record check (re-derived from the fit report, module sha == pin); ONE statistics hash,
  one TRAIN row hash and one record across all 5 robust-clip fits; `robust_clip_contrasts`, `robust_clip_stats`,
  `seed_block` (primary/secondary names, observed same-GPU pairs); Korean table section. `decision`/`ranking` stay
  `None`; no winner/promotion/verdict/selection fields.
- `configs/DRAFT-CUDA-ROBUST-CLIP-20261005.json` (new): DRAFT, `execution.enabled` false; export / calibration /
  device / training identical to the P35 DRAFT; reference arms are the exact P32/P35 entries; purpose quotes the
  PREREG path; exposure/data_rules say P24–P35. **sha256 `09756d791f11ce845fc85ddca96e486762ca15ed5ebf7aa4f0b9957666dd0372`.**
- `experiments/pitchmdp/tests/test_cuda_robust_clip_wiring.py` (new): 142 tests.

Reference arms are built through their old code paths (test spies on the build calls): `residual_counts` →
`direct_combo_model.build_combo` → `DirectResidualModel`; `periodic_residual_counts` →
`direct_embed_residual_factory.build_embed_residual`. Initial parameters for one seed are identical to the same arm
built from `EXP-P32-001.json` / the P35 DRAFT, and `robust_clip_residual_counts` has the same initial parameters as
`residual_counts` for the same seed (zero-learning-rate fit, CPU).

## Commands and results

All from `experiments/pitchmdp`, prefixed `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`,
`/Users/song/Projects/pitcheezy/.venv/bin/python -m pytest <file> -q -p no:cacheprovider`:

- `tests/test_direct_robust_clip_model.py` exit 0, 12 passed
- `tests/test_cuda_ple_wiring.py` exit 0, 130 passed
- `tests/test_cuda_ple_wiring_review.py` exit 0, 8 passed
- `tests/test_cuda_ple_registered.py` exit 1, 4 failed / 1 passed. All 4 are
  `FileNotFoundError: configs/EXP-P35-001.json`. That file is in no commit (`git log --all -- configs/EXP-P35-001.json`
  is empty), so the failure exists before this change and does not depend on it. I did not re-run it on `8311fa4`.
- `tests/test_direct_ple_model.py` exit 0, 27 passed
- `tests/test_cuda_tabm_residual.py` exit 0, 89 passed
- `tests/test_cuda_embed_residual_registered.py` exit 0, 3 passed
- `tests/test_cuda_robust_clip_wiring.py` exit 0, 142 passed

## Mutants (each applied alone, then restored with `git checkout --`; tree clean afterwards)

- M1 config_check robust-clip predicate disabled: killed (76 failed, 12 errors)
- M2 factory load: first half of the recorded-hash condition removed: **survived** (142 passed). This is an equivalent
  mutant: the second half of the same condition (`net.config['rc_stats_sha256'] != recorded`) still refuses, because
  the module loader already ties the config hash to the stored buffers.
- M2b factory load: the whole recorded-hash check removed: killed (1 failed)
- M3 scorer accepts differing statistics hashes: killed (2 failed)
- M4 `check_module` never raises: killed (1 failed)
- M5 scorer contrast uses control as reference: killed (1 failed)
- M6 seed-overlap check removed: killed (1 failed)
- M7 runner records statistics rows from the early-stop split: killed (2 failed)
- M8 runner skips the module-vs-identity check: killed (1 failed)

## Deviations from the pre-registration

None in arms, seeds, fits, constants, contrasts, budgets, training or calibration. Author choices the review should
check:

1. Placeholder deadlines `22:00 / 22:30 / 23:00` on 2026-10-05 (P35 DRAFT used 19:00/19:30/20:00). They follow
   PREREG §5 (aggregate_by ≤ 23:00) and are marked to be re-dated at registration under the P35 one-date rule.
2. Contrast entries carry two extra fields, `role` (primary/secondary) and `compute_matched` (true/false), to carry the
   PREREG labels into the scorer output. Both contrasts change one registry factor, `numeric_input`
   (raw / periodic / robust_clip); the `isolates` labels are the PREREG names.
3. No `*_vs_control` contrast rows (PREREG lists only the two contrasts). Every arm vs control is still reported in
   `splits.*.paired_vs_control`, as for every family.
4. Smoke fits take the statistics from the first 8,192 TRAIN rows (runner `SMOKE`), so smoke hashes differ from the full
   run by design (same as PLE bins).

## Open questions for the reviewer

1. Module pin: if T100b changes `direct_robust_clip_model.py`, `MODULE_SHA256` in the registry must be re-pinned (one
   line). Until then `config_check` of this DRAFT fails closed by design.
2. `MODEL_SOURCES` grew by 3 files. Tasks run with this runner record 3 more source entries. Scoring tasks from an
   earlier runner with this scorer needs `--source-manifest`, as after the P35 wiring.
3. The runner does not recompute the statistics itself to cross-check "TRAIN only". That is shown on synthetic data by
   tests (fit hash == hash of the TRAIN argument, ≠ early-stop or pooled), and `compute_stats` is called exactly once
   per task. A runner-side recompute would cost an extra quantile pass over about 1.25 M rows (time not measured).
   Should it be added?
4. `verify_family` in the scorer still calls `rc.uses_ple(config)` for this family. It is a harmless query that returns
   False. I left the existing line unchanged rather than reordering it.
5. Module (not changed here, for T100b): I saw nothing wrong. One note: a column with `s_j = 0` becomes 0 for every
   row after the transform. That matches the paper's "else 0" rule, but the information in that column is then gone.
   `rc_rule_counts['constant']` shows how many columns this hits per fit.

No GPU or CUDA verification of any kind was done.
