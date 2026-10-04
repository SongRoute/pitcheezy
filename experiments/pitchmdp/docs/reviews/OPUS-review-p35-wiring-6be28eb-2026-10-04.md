# REVIEW — EXP-P35-001 PLE wiring (`c2352f8..6be28eb`), distinct same-model reviewer, 2026-10-04

Label: `same_model_review; Codex_review_pending`. Reviewer: one Opus 5.5 worker. It is not the wiring author (84755)
and not the PM worker who merged and re-pinned. No other model was called. No Claude process was spawned. No
WebFetch/WebSearch, ssh, GPU, port, network, real export, 2026 data or real-data fit was used. **No performance claim.**

Target: `git diff c2352f8 6be28eb`. That is the author commits `5e8b1d0`, `ec75829` and `f24e9b9`, the merge `d64ba72`
of the reviewed module (`d3575019…a175e`), and the PM re-pin `6be28eb`.
Reviewer commits on `codex/p35-wiring-review-1004` (not pushed, not merged):
- `4514a0a` test: two missing tests. They kill reviewer mutants M1 and M2, which had survived.
- `b81d5cb` fix: W2. `ple_effective_parameter_count` is now lifted next to the allocated count and checked by the scorer.
- `27cc5a6` fix: W5. The runner records and prints the rows the PLE bins were fitted on.
- this document.

## 요약 (Song용)

- 한 일: P35 연결 코드를 독립 검토했다. 직접 만든 변이 6개 중 3개가 살아남았다(그중 의미 있는 2개는 테스트를 추가해 잡았다). 모듈 검토에서 넘어온 W2·W5가 비어 있어 작은 수정 2개를 넣었다.
- 결과: 집중 테스트 4개 파일 283 → 290 통과(검토 테스트 포함 최종 320/320, 6개 파일). 인접 파일 8개는 250 통과 + 기존 실패 1개(알려진 것)다. BLOCKER는 없다.
- 다음: 이 검토 처분 → Codex 검토 → **2026-10-05 19:00 KST 전** CUDA smoke 25개(그 뒤면 DRAFT 날짜를 새로 써야 함) → P35 시간 측정 스크립트(새 코드) → 등록 전 DRAFT 문구 4곳 수정.

## Findings

| # | sev | where (at `6be28eb`) | finding | status |
|---|---|---|---|---|
| 1 | MAJOR (smoke) | `scripts/run_cuda_direct.py:744,774-781` | W5 not done: nothing records the per-position valid rows that bins are fitted on | **fixed `27cc5a6`** |
| 2 | MINOR | `pitchmdp/direct_ple_registry.py:217-229`, `scripts/score_cuda_direct.py:169-180` | W2 only half done: effective count stays nested and unchecked | **fixed `b81d5cb`** |
| 3 | MINOR (test) | `scripts/score_cuda_direct.py:553` | Rule "one bins hash BEFORE any probability is read" is untested; mutant M1 survived | **test `4514a0a`** |
| 4 | MINOR (test) | `pitchmdp/direct_ple_registry.py:178` | bundle/one-factor cardinality clause is untested; mutant M2 survived | **test `4514a0a`** |
| 5 | MINOR | `scripts/score_cuda_direct.py:357`, `:591` | Bundle rows output `isolates: bundle, held: {}`. The changed factor set (name-keyed `CHANGED`) is not in the output | open (suggested diff) |
| 6 | MINOR | `scripts/run_cuda_direct.py:778` | Export-layout verification for the PLE record runs AFTER fit + predict. A mismatch wastes one whole fit | open (suggested diff) |
| 7 | NOTE | `pitchmdp/direct_ple_registry.py:43` | `MODULE_SHA256` is enforced by tests and the smoke checklist only, not at run time | open (optional) |
| 8 | NOTE | `scripts/score_cuda_direct.py:187` | Record-equality clause next to the hash check is redundant (M5 survives as an equivalent mutant) | none |
| 9 | NOTE | `tests/test_cuda_ple_wiring.py:488,492` | Cross-arm refusals use `pytest.raises(Exception)` | none |
| 10 | NOTE | config `queue` | The placeholder deadline also binds smoke: smoke must finish launching before 2026-10-05 19:00 KST | operational |
| 11 | NOTE | author report | Report does not mention W2–W5. It predates the module disposition. Otherwise accurate where I checked it | — |

### 1. MAJOR (smoke) — W5 per-position valid counts were missing (fixed `27cc5a6`)

The module review (N-b → W5) and its disposition ("W5 smoke on the TRAIN slice + valid counts") made the smoke print
the per-position valid rows. At `6be28eb` nothing records them. `fit.json` has `ple_n_bins` (T_j) but not how many
rows each past position contributed. With low counts, a column gets few bins silently. Zero rows abort at fit
(`direct_ple_model.py:90-91`), which fails closed, but nothing explains why. This blocks the smoke because the
disposition requires it. The fix adds no new data path.
Fix: `ple_bin_rows(inputs)` in the runner, computed on the exact `train_inputs` passed to `fit` (before `del train`):
```python
def ple_bin_rows(inputs):
    valid = np.asarray(inputs[1], dtype=bool)
    return {'train_rows': int(len(valid)), 'past_valid_rows': [int(n) for n in valid[:, :-1].sum(axis=0)]}
```
For PLE arms it is written to `fit.json` as `ple_bin_rows` and printed as `PLE_BIN_ROWS <arm> <seed> {…}`.
It is descriptive only, and the scorer does not check it. Test
`test_runner_records_the_rows_the_smoke_bins_were_fitted_on` compares it with the rows `compute_bins` uses. Mutant
`valid[:, 1:]` → killed (exit 1).

### 2. MINOR — W2 effective parameter count was half done (fixed `b81d5cb`)

`ple_effective_parameter_count` reached `fit.json` only nested in `report.network` (`matrix_bridge`-style
`network=self.net.config`). It was not next to `parameter_count` (allocated 275,738 / 291,600), was not re-derived,
and was not checked by the scorer. The author's smoke checklist lists only allocated counts.
Fix: `direct_ple_registry.effective_record(network, allocated)` requires int values and
`allocated − effective == 8 × (81 × 48 − ΣT_j)`. This is the module's own padding identity: padded = (F·48 − ΣT_j)·d,
because each feature has exactly T_j live pieces. The runner lifts it to the top level. The scorer re-derives it in
`check_ple_record` and fails closed if `parameter_count` or the field is missing or inconsistent. It lists
`provenance.tasks.<arm>.<seed>.ple_parameters = {allocated, effective}`. The field is arm-specific, so it is kept OUT
of the shared bins record that `check_one_bins_hash` compares across both PLE arms. The author's test helpers
(`network_record`, `with_records`) now carry the module's allocated counts. No assertion was weakened. Three tests
were added (runner value equals `plm.parameter_count(...)`, registry refusals, scorer refusals and provenance).

### 3. MINOR (test) — "bins hash before any probability" was untested (test `4514a0a`)

Mutant M1 moved `check_one_bins_hash(tasks)` after `score_family(...)` (`score_cuda_direct.py:553`). It survived
130/130, because the existing test only checks that no score directory is written.
`test_scorer_checks_one_bins_hash_before_any_probability_is_read` patches `sc.member` to raise and expects the
bins-hash `ValueError` → M1 killed.

### 4. MINOR (test) — bundle cardinality clause was untested (test `4514a0a`)

Mutant M2 removed `(c['isolates'] == 'bundle') != (len(changed) > 1)` (`direct_ple_registry.py:178`). It survived,
because `same(contrasts, CONTRASTS)` and the name-keyed `CHANGED` already pin everything for the shipped list. The
clause guards a future edit that relabels a row in both registry and config. The new parametrized test relabels row 0
`bundle` and row 3 `embedding_type_single`, with `CHANGED` unchanged → M2 killed (2 failed).

**Assessment of the name-keyed `CHANGED` mechanism (item 1).** It is sound:
- `name` is derived from the pair (`pair()`), and all 7 names are distinct.
- The config list must equal `CONTRASTS` JSON-exactly. A mislabelled, reordered, sign-flipped, extra or missing
  contrast is therefore refused before the factor check (12 author edits plus mine).
- The factor check then catches registry drift (`CHANGED` vs `FACTORS`, author mutant test, my M7 killed). With the
  cardinality clause it also catches label/cardinality drift.
- The bundle rows are honestly multi-factor:
  - `periodic|ple_residual_counts_vs_control` change 5 factors (input, workload, backbone, dropout rate, dropout
    position).
  - `tabm|ple_tabm_residual_counts_vs_control` change all 6.
- The three non-bundle rows change exactly one factor each. Each `held` dict is checked on both sides.
- Residual gap: `FACTORS` is the registry's own description of each arm and is not derived from the arm specs. For
  the three earlier arms it equals `trr.FACTORS` (same tuples). For the new arms it is consistent with `ARMS`
  (`base_model`, `embedding`). Acceptable, as in P34.

### 5. MINOR — bundle rows do not show which factors changed (open)

All four control rows carry `isolates: 'bundle'` and `held: {}`, so `ple_contrasts.<split>.<name>` and the Korean
table cannot tell a reader that the TabM bundles also change the head. Suggested additive diff
(`score_cuda_direct.py`, the PLE loop in `score_family`):
```diff
-            embedding_type[split][c['name']] = {'candidate': c['candidate'], 'reference': c['reference'],
-                                                'isolates': c['isolates'], 'held': c['held'],
+            embedding_type[split][c['name']] = {'candidate': c['candidate'], 'reference': c['reference'],
+                                                'isolates': c['isolates'], 'held': c['held'],
+                                                'changed': sorted(ple_registry.CHANGED[c['name']]),
```
It does not block the smoke (the scorer never runs on smoke). Do it before registration or at Codex discretion.

### 6. MINOR — layout verification runs late for PLE arms (open)

At `run_cuda_direct.py:730`, `verified_context_names` runs only when `needs_context_names(arm)` is true, which is false
for every P35 arm. The PLE record calls it at `:778`, after fit and prediction. If the export layout differs from the
config, the task still fails closed (tested), but only after one full fit (≤ 1,800 s by the gate). The smoke uses the
same config and export, so the smoke would catch it first. That is why this is MINOR. Suggested diff:
```diff
-        names = verified_context_names(config, manifest) if needs_context_names(arm) else None
+        names = verified_context_names(config, manifest) if needs_context_names(arm) else None
+        ple_names = verified_context_names(config, manifest) if is_ple_arm(arm) else None  # fail before the fit
 ...
-                                         verified_context_names(config, manifest), arm['workload']))
+                                         ple_names, arm['workload']))
```

### 7. NOTE — module pin is not enforced at run time

`MODULE_SHA256` is used only by tests (`test_module_pin_and_frozen_modules_unchanged`, runner smoke test). At run
time the identity records the actual sha, and the scorer checks it against the operator's source manifest. This is
the same as P34 and fine with the smoke checklist line "source_sha256 … direct_ple_model.py == MODULE_SHA256".
Optional fail-closed line in `run_task` for PLE arms:
`if own_hashes()['pitchmdp/direct_ple_model.py'] != MODULE_SHA256: raise RuntimeError(...)`.
Verified now: the file sha equals the pin `d3575019…a175e`, and the hex appears in no other `*.py`/`*.json` (git grep).

### 8–11. NOTEs

- 8: Mutant M5 (drop the `json.dumps` record-equality clause, `score_cuda_direct.py:187`) survives. The record is a
  deterministic function of the bins (hash ⇒ edges ⇒ T_j ⇒ names, given one config), so the mutant is equivalent in
  practice. No change.
- 9: `test_factory_refuses_other_arm_archives_both_ways` uses `pytest.raises(Exception)`, so a crash would also pass.
  The specific refusals are covered by other tests (`context width`, `continuous map`, `not this PLE arm`,
  `recorded by the run`).
- 10: `cuda_direct_queue.Deadlines` has no smoke exemption (`cuda_direct_queue.py:86-91,316-343`). With
  `queue.date 2026-10-05` and `no_new_fit_after 19:00`, a smoke not launched by 2026-10-05 19:00 KST leaves its tasks
  `not_run`. It then needs a re-dated DRAFT, which means a new config sha. The author's note is correct.
- 11: The author report (`REPORT-p35-wiring-author-2026-10-04.md`) says nothing about W2–W5, because it was written on
  `c953bfd` before the module disposition. Its file sha line and "MODULE_SHA256 `6a1541a3…`" are superseded by
  `6be28eb`. The counts I could check (130 tests in the file, 4-file run, the pre-existing failure) match.

## Checks by item (what I verified myself)

1. **Pre-registration fidelity.**
   - Arms: `control, periodic_residual_counts, ple_residual_counts, tabm_residual_counts, ple_tabm_residual_counts`,
     which is DRAFT §5 order.
   - Seeds are exactly `[29..33]`, `fits` is 25, there is no `repeat_tag`, and the training is the P34 budget with
     epochs 60.
   - The three earlier arms are JSON-exact (`sort_keys`) to `configs/EXP-P34-001.json` (recomputed: True ×3). The
     export, calibration and device blocks equal P34.
   - The 7 contrasts and labels match the DRAFT table, with HEADLINE as the first two rows.
   - The cap is 60 for this family only. `training_budget` returns `reg.TRAINING` only under `uses_ple`. Earlier
     families get the same objects as before (author test over 9 DRAFTs + 5 registered + constructed P30-002).
2. **Dispatch-first.**
   - Every P35 path tests `uses_ple`/`is_ple_arm` first: `config_check` (`:243`), `training_budget` (`:231`), per-arm
     loop (`:274`), `build_model` (`:508`), `load_model` (`:572`), `run_task` cap/record (`:774-779`), scorer
     `score()` (`:522`) and `verify_family` (`:231-234`).
   - The per-arm loop sends the family's `tabm_residual_counts`/`periodic_residual_counts` to their own ARM checks.
     That is intended (exact earlier entries), not a family route.
   - Earlier families: `uses_epoch_cap == uses_tabm_residual` and no `ple_*` fit fields (only the `is_ple_arm`
     branch adds them).
   - Queue: no family predicate (grep). The only other predicate users are `ema_trajectory_check.py:51` and
     `timing_probe_tabm_residual.py:74`, both P33/P34-scoped and unchanged.
   - Shared seed validator: the `[0,1,2]` fallback (`:262`) is never reached by a PLE config, which answers the
     DRAFT §5 question.
   - My mutants M1–M7 below; the author's D1–D9 were not re-run.
3. **Records.**
   - W2: see finding 2.
   - W3: labels map via `verified_context_names(config, manifest)` (runner). The scorer uses
     `config['export']['context_names']`, which equals it by construction because the runner refuses otherwise.
     History is oldest→newest (`past0` = oldest, test `past4.phys7 → past4.plate_z`). `context52` →
     `prior_records_per100`, the counts workload (test, and my mutant M6 "workload first" killed).
   - W5: see finding 1.
4. **Factory strict load.**
   - It refuses: another class (`cls.load` format/kind, then `type(model) is cls`), kind, architecture incl. n_bins/d
     (via `ARCHITECTURE`/`TABM_ARCHITECTURE`) and k/scaling init, a continuous map ≠ `CONTINUOUS['counts']`
     (attribute and `net.config`, same-length test), another context width (my mutant M3 killed), and a bins hash ≠
     the run-recorded hash (both `bins_sha256(model.ple_bins)` and `net.config`).
   - A missing `fit.json` or missing key fails closed (`not recorded_bins_sha256`).
5. **Scorer.**
   - All 25 `state.json` must exist (`verify_family`).
   - Every PLE fit's top-level record is re-derived from its own network report.
   - `check_one_bins_hash` (10 fits, one hash, one record) now runs provably before `member()` (test `4514a0a`).
   - Other contrast keys on a PLE config are refused, and `ple_contrasts` on other families are refused.
   - The output keeps `decision: null` and `ranking: null` as every family does. There is no winner/verdict/promotion
     key (author test asserts no other-family keys). Headline rows are ΔNLL with CIs, with "판정 없음" in the table.
6. **Config.**
   - `status DRAFT`, `execution.enabled false`.
   - GPU pins `GPU-c334d905-8ec9-0029-b206-46ce3ac49cd9`, `GPU-ae803a24-0d38-791d-ed7c-3f233fcecf99`, max_parallel 2,
     no fallback.
   - Export `49097109b4f5…f578`, `last_year` 2025.
   - Queue: 3600 / 36000 / 2 attempts, all priority 1. Placeholder 2026-10-05 19:00/19:30/20:00 with `date_note`.
   - All 14 `source_configs` sha256 recomputed in this tree: 14/14 equal (13 configs + the PREREG DRAFT `82dd5b15…`).
   - Config sha `f0ae4e91…b9b2` is unchanged by my commits.
7. **DRAFT text gaps** — classified below.
8. **Tests.** Runs and mutants below. The author tests are meaningful: they cover config drift (36 edits), the
   contrast list, dispatch with forbidden routes, CPU run_task on synthetic exports including a smoke-slice B1 case,
   factory refusals, and scorer refusals. The gaps were findings 3–4 and the W2/W5 records.

## DRAFT text gaps (item 7) — classification and proposed text (DRAFT not edited)

| gap | class | proposed text |
|---|---|---|
| §10.1 file name `direct_ple_residual_model.py` | blocking-for-registration (text must name the pinned file) | "1. `pitchmdp/direct_ple_model.py` (sha256 `d3575019f971b06ba26060a1fa31e3894a070fd3b4b88191393743cb16ea175e`, reviewed `840c771`, disposition `c2352f8`): `DirectPLEResidualModel` / `DirectPLETabMResidualModel` …" |
| §5 four `bundle` labels | blocking-for-registration (the registry enforces sets the text does not state) | add a column "changed (registry `CHANGED`)": rows 1–2 `input`; row 3 `head`; rows 4–5 `input, workload, backbone, dropout_rate, dropout_position`; rows 6–7 the same + `head`. Also add a sentence: "bundle rows change more than one factor; non-bundle rows exactly one (registry check)." |
| §5 "held" column (input columns, d 8; dropout .1 missing in row 3) | note | "held (registry factors): row 3 also holds dropout .1; input columns and d = 8 are held structurally (identical `context_continuous`, `ARMS` n_bins/d, factory architecture check)." |
| §10.3 stale line numbers | note | replace with function names: "`run_cuda_direct.config_check`, `training_budget`, `build_model`, `load_model`, `run_task`; `score_cuda_direct.score`, `verify_family`". |
| §6 / §10.7 P35 timing probe needs new code | blocking-for-registration (not smoke) | "7. `scripts/timing_probe_ple_residual.py` (new; Amendment-2 method, one probe seed outside 29–33, refuses any config but `DRAFT-CUDA-PLE-RESIDUAL-*`), with its own CPU tests and a distinct review before it runs; `timing_probe_tabm_residual.py` is P34-only and is not reused." |
| pre-existing failing `test_frozen_runner_and_models_unchanged_by_probe` | blocking-for-registration (hygiene; not smoke: the server never runs pytest) | Disposition, not DRAFT text. Re-scope the assertion to the P34 probe's own window (`git diff --name-only ee954a5 <P34 probe run commit>`), or mark it historical with the reason. Either needs its own commit. Do not leave a red test in the registered tree. |
| N-a (i)(ii)(iii) not in §2 | blocking-for-registration (disposition: "before registration") | append to §2(c) deviations: "(i) no row floor: the package requires ≥ 2 rows and n_bins < len(X); ours bins any non-empty column (rows per past position are recorded in `fit.json ple_bin_rows`); (ii) version B: the package draws a uniform init for the PLE linear then zeroes it; ours draws nothing for W, so the post-construction RNG stream differs from a package-built model (outputs at a given state are identical); (iii) the package warns on single-bin features; ours is silent and records T_j in `ple_n_bins`." |
| placeholder deadline binds smoke | conditionally blocking-for-smoke (only if smoke slips past 2026-10-05 19:00 KST) | none; re-date in a new DRAFT commit if needed. |

## Mutation results (reviewer; applied in place by `/tmp/pz-p35-wiring-review/mutate.py`, `git checkout` restore after each)

| # | mutant | `tests/test_cuda_ple_wiring.py` (author) | after reviewer tests |
|---|---|---|---|
| M1 | scorer: `check_one_bins_hash` after `score_family` | **survived** (130 passed, exit 0) | killed (1 failed, exit 1) |
| M2 | registry: bundle cardinality clause removed | **survived** (130 passed, exit 0) | killed (2 failed, exit 1) |
| M3 | factory: context-width check removed | killed (1 failed, exit 1) | — |
| M5 | scorer: record-equality clause removed | survived (equivalent, NOTE 8) | — |
| M6 | registry `column_name`: workload before context names | killed (1 failed, exit 1) | — |
| M7 | registry `CHANGED` of a bundle row widened by `head` | killed (1 failed, exit 1) | — |
| M8 | runner `ple_bin_rows` uses `valid[:, 1:]` (after fix 1) | — | killed (1 failed, exit 1) |

`git status --short` was empty after every restore (M1–M7). M8 was restored by `sed` before commit `27cc5a6`.

## Commands (cwd `experiments/pitchmdp` unless noted; every pytest with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 perl -e 'alarm 600; exec @ARGV' /Users/song/Projects/pitcheezy/.venv/bin/python -m pytest … -q -p no:cacheprovider`)

1. Reads: `git log/diff/grep/status`, `sed -n`, `cat`, `grep` → exit 0. The `grep --include` form failed under zsh
   globbing (exit 1, no effect) and was redone with `git grep`.
2. Root: `/usr/bin/python3` torch-free JSON/sha256 one-shots (config fields, P34 exactness, 14 source_configs) →
   exit 0.
3. `shasum -a 256 pitchmdp/direct_ple_model.py` → `d3575019…a175e` (= pin).
4. At `6be28eb`: `tests/test_cuda_ple_wiring.py tests/test_direct_ple_model.py tests/test_cuda_tabm_residual.py
   tests/test_cuda_direct.py` → **283 passed, exit 0**.
5. Mutants M1–M7 with `-x` on `tests/test_cuda_ple_wiring.py` → exits as in the table.
6. After `4514a0a`: `tests/test_cuda_ple_wiring_review.py` → 3 passed, exit 0. Under M1 → exit 1. Under M2 → exit 1.
7. After `b81d5cb`: the 4 focused files + review file → **290 passed, exit 0**. Then `test_cuda_frontier_run`,
   `test_cuda_tabm_residual_wiring_review`, `test_cuda_tabm_residual_registered`, `test_cuda_ema_wiring_review` →
   53 passed, exit 0.
8. After `27cc5a6`: 4 focused + review + `test_cuda_frontier_run` → **320 passed, exit 0**. Under M8 → exit 1.
9. `test_timing_probe_tabm_residual.py test_cuda_tabm_residual_registered.py test_cuda_ema.py
   test_cuda_embed_residual.py test_cuda_tabm_mini.py test_cuda_workload_dropout.py test_cuda_dropout.py
   test_cuda_data_combo.py` → **1 failed, 250 passed, exit 1**. The failure is only the known pre-existing
   `test_frozen_runner_and_models_unchanged_by_probe`.
10. `git diff --quiet 6be28eb HEAD -- pitchmdp/direct_ple_model.py tests/test_direct_ple_model.py <5 frozen modules>
    ../../configs` → exit 0. The module, its test, the frozen modules and all configs are unchanged.
    Reviewer HEAD file sha: registry `97aadf9f…`, factory `5b9faa25…` (unchanged), runner `8ad44179…`, scorer
    `a812ee63…`.
11. `git add <explicit paths>`, `git commit` ×4 → exit 0. No push, no merge.

## Disposition

**Ready for the CUDA smoke of all 25 tasks** at the reviewer head (`27cc5a6` + this doc), provided that:
- (a) the PM accepts or replaces `b81d5cb`/`27cc5a6`. Without `27cc5a6`, W5 stays open and the smoke would not print
  the valid counts the module disposition requires.
- (b) Codex review is disposed as planned.
- (c) EXP-P30-002 and every other queue is terminal, and GPU3/4 are idle.
- (d) the smoke launches before 2026-10-05 19:00 KST, or a re-dated DRAFT is committed first.

Smoke checklist additions to the author's list:
- `ple_effective_parameter_count` next to the allocated 275,738 / 291,600.
- `PLE_BIN_ROWS` lines (`past_valid_rows`, five values, oldest first, plus `train_rows` = 8,192).
- Same config sha across all 25 identities.

**Before registration:**
1. Write the P35 timing probe script and get it reviewed, then run it with every gate true.
2. Write the pre-score extractor (§8).
3. Make the four DRAFT text changes classified as blocking-for-registration above (§10.1 file and pin, §5 changed
   sets, §2 N-a disclosures, §10.7 probe script).
4. Dispose the pre-existing probe test.
5. Optionally apply findings 5–6.
6. Re-date the deadlines from the measured probe.
7. Get an explicit go from Song.

Not measured here: GPU time and memory, real-TRAIN ΣT_j, valid counts and degenerate columns (smoke/probe will show
them), CUDA dropout-stream equality.
