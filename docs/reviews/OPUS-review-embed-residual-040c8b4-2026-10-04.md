# Opus 5.5 review — embedded-residual registry, DRAFT config, runner/scorer dispatch (a690350..040c8b4)

Label: `same_model_review; Codex_review_pending`
Reviewer: Opus 5.5, a separate session from the author. No other model was called and no further Claude process was started.
Target: `1b1ff30..040c8b4` in worktree `codex-embed-residual-review-1004` (branch `codex/embed-residual-review-1004`). The module itself (`direct_periodic_residual_model.py`) was reviewed in 5871e1d. I re-checked it only where the wiring depends on it: `load`, the format/kind check and the deep copy.
Scope kept: light CPU, single thread, focused test files only. No server, GPU, ssh, ports or tmux. No result file or score file was opened. The design note (a review target) quotes P27 numbers in its "질문" section. I saw them while reading the note, and nothing in this review uses them.

## Disposition

**Non-blocking** for the CUDA smoke of all 18 tasks. **Non-blocking** for later registration, as long as F5 (stale status lines in the pinned design note) is fixed in the same edit that writes the registration deadlines. That edit changes the config anyway, so the design-note SHA pin can be updated with it.

- No code defect found in the registry, factory, runner or scorer dispatch.
- Old families: `config_check` result, build/load class and scorer keys/order are unchanged for every earlier config (checks under Q1).
- Four test gaps were closed with new tests (F1–F4). All of them passed on the first run, so they add coverage and did not expose a bug. No production code was changed.
- Two text fixes are proposed (F5, F6) and were not applied, because F5 changes a SHA-pinned file and F6 changes a scorer source.

## Answers to the five questions

### Q1 Old families
- The new `elif uses_embed_residual` branch in `config_check` (`run_cuda_direct.py:179`) is reached only if some arm has the key `embed_residual`. `grep -l embed_residual configs/*.json` finds only the new DRAFT config. The branch comes after `uses_tabm` and before `uses_workload_dropout`. That order is required, because `periodic_counts_d10` keeps its P30 `workload_dropout` flag.
- The per-arm branch, `build_model` and `load_model` gate on the key, or on `is True`, both of which only new arms have. Every old arm goes down the same path as before.
- `score_family` still returns 3, 4 or 5 elements for old families. `embed_residual_contrasts` and its `seed_block` are added only when `embed_residual` is truthy.
- Checked on every `configs/*CUDA*.json` plus EXP-P30-001 and EXP-P31-001 (output below):
  - `config_check` passes as before. DRAFT-CUDA-FRONTIER keeps failing as before, because it is not `cuda_direct_v1`.
  - Seeds `[16,17,18]` are refused.
  - Appending an embed arm is refused.
  - `er.check_contrasts` returns `None`.
- `MODEL_SOURCES` gained 3 files. So `own_hashes()` has 3 more keys, and `verify_task(..., source_sha256)` fails for every task finished before 040c8b4. That is expected and fails closed. Old completed families can be scored only with `--source-manifest` (`task_sources` still accepts a key subset of the grown list) or from their own snapshot. If an old family's queue were restarted from this code, `is_complete` would raise rather than re-run or re-label anything. Nothing can be silently re-labelled, because a task identity binds `experiment_id`, the config digest, the arm spec, seed, export manifest and device. The embed family's baseline arm specs are JSON-identical to the P30/P31 ones, but experiment_id, config digest and seed all differ (see F7).

### Q2 Family exactness
- Baselines are JSON-exact:
  - `control`, `residual_counts` and `periodic_counts` match DRAFT-CUDA-DATA-COMBO-RUN.
  - `control`, `periodic_counts` and `periodic_counts_d10` match EXP-P30-001, EXP-P31-001, DRAFT-CUDA-WORKLOAD-DROPOUT and DRAFT-CUDA-TABM-MINI.
  - `export`, `training` and `data_rules` equal those of P30, P31 and DATA-COMBO-RUN.
- Code paths: `control` → MatrixModel. `residual_counts` → combo → `DirectResidualModel`. `periodic_counts` → combo → `DirectPeriodicModel`. `periodic_counts_d10` → workload_dropout → `PeriodicHeadDropoutModel`. Existing test `test_baseline_arms_keep_old_code_paths...` asserts these.
- Sneaking an arm in or out:
  - An embed arm in any other family is refused by that family's exact `ARM_ORDER` (tabm, workload_dropout), by `uses_embed_residual` → embed `check_family` (plain, combo, dropout and frontier families), or by the 0–2 seed rule.
  - In this family, an old arm that carries an embed or tabm flag is refused by the JSON-exact baseline compare, or by tabm `check_family` taking precedence.
  - A new arm whose flag is removed is refused by `check_arm` (flag must be `True`). With no flag on any arm, the 0–2 seed rule refuses it.
- Seed exclusivity is complete. Seeds 16–18 are accepted only by the embed `check_seeds` (exactly `[16,17,18]`, no `repeat_tag`, `fits == 18`). Every other branch requires 0–2, 3–7, 8–12 with its tag, or 13–15.

### Q3 Loader
- Other embedding: refused by `cls.load` format/kind (tested).
- Another architecture: refused by `cls.__init__` (fixed-architecture compare) and by the `network_kwargs` compare (module tests).
- Same-length, different continuous map: refused by `load_embed_residual` (tested, review F4 of 1b1ff30 closed).
- Different context width: refused (tested).
- `DirectResidualModel` / `DirectPeriodicModel` / `PeriodicHeadDropoutModel` / `MatrixModel` / `DirectTabMModel` archive loaded as an embed arm: refused. Before this review this was tested only at class level for the first two. The runner-level test added in F1 covers all five.
- The other direction, an embed archive loaded as `residual_counts`, `periodic_counts` or `periodic_counts_d10`: refused (tested).

### Q4 Scorer
- `embed_residual_registry.check_contrasts` runs right after `config_check`, before `task_sources`, `export_rows` and `verify_family` (`score_cuda_direct.py:369`). It requires the fixed list exactly and calls `check_family`.
- Sign is candidate − reference: `paired_block(cand, ref)`. The test checks that each contrast delta equals the candidate-vs-control delta minus the reference-vs-control delta.
- `input = embedding_type + embedding` is asserted for NLL. F2 adds Brier and the `bundle = structure + head_dropout` identity.
- Mixing contrast keys is refused both ways:
  - In an embed config, any other key is refused (factorial and dropout by the `is not None` guard; workload_dropout and tabm by their own `check_family`).
  - In any other config, the embed key is refused by embed `check_family` (F3 adds the test, including the scorer path before any read).
- The output has `decision: None` and `ranking: None`, and no winner or verdict field. The table header says "판정 없음".

### Q5 Config
- The queue settings match EXP-P31-001 except `batch`, `order` and `start_after`:
  - date 2026-10-04; no_new_fit_after 19:00; kill_own_after 19:30; aggregate_by 20:00 (KST)
  - task timeout 3600 s; family budget 36000 worker-s; 2 attempts
  - the two GPU UUIDs; max_parallel 2; no CPU fallback
- `start_after` names the common lock dir `/home/sds733/pitcheezy-autonomy/gpu-locks`.
- Status is DRAFT, `execution.enabled: false`, and there is no `registration` block. The design note and TabM source pins match the current files.
- Budget: the design note estimates about 3,600 worker-s for 18 fits against a 36,000 worker-s cap. Time-wise this is not a risk.
- Deadline caveat (F9): the queue's `Deadlines` uses the literal date. A queue-driven smoke must therefore start before 19:00 KST on 2026-10-04. After that, the queue launches nothing (`deadline_no_new_fit`). This fails closed and is not misleading.

## Findings

### F1 — Low, test coverage (TEST ADDED) — embed arms vs baseline/TabM archives not tested at runner level
`experiments/pitchmdp/tests/test_cuda_embed_residual.py` (new `test_embed_arms_refuse_baseline_and_tabm_archives`)
- Before: `test_checkpoint_roundtrip_and_strict_loader` loaded an embed archive under the baseline arms, but not the reverse. The module test covered the reverse only for `DirectResidualModel` and `DirectPeriodicModel`, at class level.
- Scenario guarded: if a later edit relaxed the format check in `DirectEmbeddingResidualModel.load`, a `periodic_counts_d10` or `tabm_counts` archive in a mis-placed task directory could be loaded as `periodic_residual_counts` by `rc.load_model`.
- The test fits and saves `control`, `residual_counts`, `periodic_counts`, `periodic_counts_d10` and `tabm_counts`. It then checks that `rc.load_model` raises `ValueError` for both embed arms. It passes on 040c8b4.

### F2 — Low, test coverage (TEST ADDED) — design-note identity 1 = 2 + 3 and the Brier identities were not asserted
`tests/test_cuda_embed_residual.py::test_scorer_reports_all_contrasts_with_sign`
- The design note states `bundle = structure + head_dropout` exactly (same seeds). Only `input = embedding_type + embedding`, for NLL only, was tested.
- Added both identities, for both `nll` and `brier`, on `dev` and `mlb_dev`. They pass.

### F3 — Low, test coverage (TESTS ADDED) — old-family regression list incomplete; embed contrast key on another family not tested
`tests/test_cuda_embed_residual.py`
- `test_other_families_unchanged_and_refuse_embed_seeds_or_arms` did not include DRAFT-CUDA-DROPOUT-RUN (which uses the `regularization_contrasts` path) or DRAFT-CUDA-FRONTIER-RUN. Both are now in the parametrization.
- The same test now also asserts that `er.check_contrasts` refuses each old config when `embed_residual_contrasts` is added to it.
- New `test_scorer_refuses_embed_contrasts_on_a_registered_other_family_before_any_read`:
  - Input: EXP-P31-001 (REGISTERED) with the embed key added, a nonexistent export, family and manifest SHA.
  - Result: `sc.score` raises "Embedded-residual family must be exactly ..." and no destination is created. So the refusal happens before any export or prediction read.

### F4 — Low, test coverage (TEST ADDED) — `FACTORS` labels were free text except the dropout rate
`experiments/pitchmdp/tests/test_direct_embed_residual_registry.py` (new `test_review_factor_labels_follow_from_the_arm_specs`)
- `check_contrasts` checks the declared changed-factor sets against the hand-written `FACTORS` table. `FACTORS` was tied to the arm specs only through the dropout rate (`test_factor_table_matches_the_networks`).
- Scenario guarded: if `FACTORS['residual_counts']` were mislabelled (for example input `'periodic'`), the `input` contrast would still pass, as long as the `ISOLATES`/`held` text was adjusted to match.
- The new torch-free test derives each label from the fixed spec:
  - input: `embedding`, otherwise periodic if `context_continuous`, otherwise raw
  - backbone: from `model`
  - rate: residual `ARCHITECTURE['dropout']` or `head_dropout`
  - position: blocks / head / none
  - It also asserts `FACTORS` covers exactly the non-control arms.
- It passes.

### F5 — Low, documentation (NOT applied; fix with the registration edit) — design note status lines are stale at 040c8b4
`experiments/pitchmdp/docs/DRAFT-PERIODIC-RESIDUAL-DESIGN-20261004.md:3` and step 3 of "남은 단계"
- Line 3 says "레지스트리·config·러너 연결 … 없음", and step 3 says "러너 미연결". Commit 040c8b4 added the DRAFT config and the runner/scorer dispatch.
- The note's SHA is pinned in `source_configs.design_note`, so changing it requires updating the pin. A reader of the registered family's pinned note would otherwise read that no runner wiring exists.
- Suggested diff (then re-pin the SHA in the config):
  ```diff
  -상태: 설계 + 모듈 + 합성 테스트 + torch 없는 레지스트리(러너 미연결). 레지스트리·config·러너 연결·서버·GPU 실행 없음. 같은 모델(Opus) 검토 전, Codex 검토 전.
  +상태: 설계 + 모듈 + 합성 테스트 + 레지스트리 + DRAFT config + 러너·채점 연결(040c8b4). 서버·GPU 실행 없음. 같은 모델 검토 5871e1d(모듈)·OPUS-review-embed-residual-040c8b4(연결) 완료, Codex 검토 전.
  ```
  and in step 3: replace "작성됨, 러너 미연결. 남은 것: …" with "완료(a690350, 040c8b4; 연결 검토 OPUS-review-embed-residual-040c8b4)".

### F6 — Low, wording (NOT applied) — the table and scorer note say "two independent" only for the residual side
`experiments/pitchmdp/scripts/score_cuda_direct.py:338-341` (Korean table) and `:399-404` (seed_block note)
- The six contrasts are differences among 5 arms. Their rank is 4: `input = embedding_type + embedding`, and also `backbone_bundle = backbone_structure + head_dropout`.
- The table explains only the residual-side identity. A reader could treat the three periodic-input rows as three independent results.
- Suggested text (scorer source change, so it should land before any registered scoring, not after):
  ```diff
  -'맞췄지만 위치(헤드 vs 블록)는 다릅니다. input = embedding_type + embedding(같은 시드)이라 세 잔차 대비 중 독립은 둘입니다.'
  +'맞췄지만 위치(헤드 vs 블록)는 다릅니다. 같은 시드라 backbone_bundle = backbone_structure + head_dropout, '
  +'input = embedding_type + embedding이 정확히 성립해 여섯 대비 중 독립은 넷입니다.'
  ```
  and append `'; backbone_bundle = backbone_structure + head_dropout'` to the seed_block note.

### F7 — Info — `MODEL_SOURCES` growth and old tasks (no change)
See Q1. It is the same pattern as when the TabM files were added. Operationally:
- Score P24–P31 families with their pinned `--source-manifest` or from their own snapshot.
- Never restart an old family's queue from this code. It fails closed at `is_complete`.

### F8 — Info — `load_embed_residual` does not compare the archived seed, `n_token` or `length` (no change)
This matches `load_tabm`. The task identity (`verify_task`) binds the seed. A wrong `n_token`/`length` archive fails on the first `logits` call. `load_model` is not called by `run_task`, so it is a defensive layer only.

### F9 — Info — literal queue date
`configs/DRAFT-CUDA-EMBED-RESIDUAL-20261004.json` `queue.date`
- A smoke via `cuda_direct_queue.py --smoke` launches nothing after 19:00 KST on 2026-10-04.
- If the smoke slips, edit the DRAFT date. This changes the DRAFT digest, which smoke identities record. Registration changes the digest anyway (status/execution), so smoke identities never equal registered ones, as for P31.
- Running smoke tasks directly with `run_cuda_direct.py task --smoke` is not deadline-gated.

### F10 — Info — "read before design" lists differ slightly
- Registry docstring: P24/P27/P29/P31
- Config `purpose`: P24/P27/P29/P30/P31
- Scorer note: "P24-P31"
- Design note seed section: P24/P27/P29/P31

All of them say post-selection. Harmless, but a single list would read better at registration.

## Changes made by this reviewer (tests only, no production code)

1. `experiments/pitchmdp/tests/test_cuda_embed_residual.py`:
   - DROPOUT-RUN and FRONTIER-RUN added to the old-family parametrization, plus a foreign-contrast-key refusal (F3)
   - new `test_scorer_refuses_embed_contrasts_on_a_registered_other_family_before_any_read` (F3)
   - new `test_embed_arms_refuse_baseline_and_tabm_archives` (F1)
   - Brier and `bundle = structure + head_dropout` identities in `test_scorer_reports_all_contrasts_with_sign` (F2)
2. `experiments/pitchmdp/tests/test_direct_embed_residual_registry.py`: new `test_review_factor_labels_follow_from_the_arm_specs` (F4).

All new tests passed on the unchanged 040c8b4 code. They are coverage additions and did not catch a bug.

## Commands run (outputs abridged)

```
$ cd experiments/pitchmdp && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /Users/song/Projects/pitcheezy/.venv/bin/python -m pytest \
    tests/test_cuda_embed_residual.py tests/test_direct_embed_residual_registry.py tests/test_direct_periodic_residual_model.py \
    tests/test_cuda_tabm_mini.py tests/test_cuda_workload_dropout.py -q -p no:cacheprovider
173 passed in 4.22s            # at 040c8b4, before the review tests

$ ... pytest tests/test_cuda_embed_residual.py -q -p no:cacheprovider
53 passed in 2.05s             # 49 + 4 new (2 parametrized old configs, F1, F3 scorer)

$ ... same five files as above
178 passed in 4.73s            # after all review tests (+5)
```

Old-family probe (in-process, torch-free; one line per config):
```
DRAFT-CUDA-DATA-COMBO-RUN     ok  refuses16-18  refuses-embed-arm  uses_embed False  er.contrasts None  ['factorial_contrasts']
DRAFT-CUDA-DIRECT             ok  refuses16-18  refuses-embed-arm  uses_embed False  er.contrasts None  []
DRAFT-CUDA-DROPOUT-RUN        ok  refuses16-18  refuses-embed-arm  uses_embed False  er.contrasts None  ['regularization_contrasts']
DRAFT-CUDA-EMBED-RESIDUAL     ok  ACCEPTED-16-18 refuses-embed-arm(dup)  uses_embed True  er.contrasts = 6 fixed  ['embed_residual_contrasts']
DRAFT-CUDA-FRONTIER           REFUSED (not cuda_direct_v1, unchanged)
DRAFT-CUDA-FRONTIER-RUN       ok  refuses16-18  refuses-embed-arm  uses_embed False  er.contrasts None  []
DRAFT-CUDA-TABM-MINI          ok  refuses16-18  refuses-embed-arm  uses_embed False  er.contrasts None  ['tabm_contrasts']
DRAFT-CUDA-WORKLOAD-DROPOUT   ok  refuses16-18  refuses-embed-arm  uses_embed False  er.contrasts None  ['workload_dropout_contrasts']
EXP-P30-001                   ok  refuses16-18  refuses-embed-arm  uses_embed False  er.contrasts None  ['workload_dropout_contrasts']
EXP-P31-001                   ok  refuses16-18  refuses-embed-arm  uses_embed False  er.contrasts None  ['tabm_contrasts']
```

Baseline JSON-exactness probe: `control`, `residual_counts` and `periodic_counts` are JSON-exact vs DATA-COMBO-RUN. `control`, `periodic_counts` and `periodic_counts_d10` are JSON-exact vs EXP-P30-001, EXP-P31-001, DRAFT-WORKLOAD-DROPOUT and DRAFT-TABM-MINI. `export`, `training` and `data_rules` are equal to all of them.

Config vs DRAFT-CUDA-TABM-MINI: the only differences are arms, seeds (16–18), fits (18), contrasts/scope, experiment_id, purpose, notes, `queue.batch`/`order`/`start_after`, and the two added source pins. Queue deadlines, budget, attempts, device, export, training and calibration are identical. Pins: design note `5ce7490a…` and TabM config `334528a5…` match the files.
