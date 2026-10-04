# Review: TabM-residual wiring + Amendment 1 disposition (`0d8a65b..6238d6c`)

**Label: `same_model_review; Codex_review_pending`.**
2026-10-04 KST. Reviewer: Opus 5.5 (`claude-opus-5-5`). I did not write the wiring
(`61d678c`/`289fd68`) or the disposition (`6238d6c`). No other model was called and no other
Claude process was started. Worktree `codex-tabm-residual-wiring-review-1004`, branch
`codex/tabm-residual-wiring-review-1004`. Nothing was pushed or merged. I did not use ssh, a
GPU or any port, ran no fit on real data, read no 2026 data and ran no full test suite.
The Mac was in protected mode: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, and every command was
bounded with `perl -e 'alarm 600; exec @ARGV'`.

Scope: `git diff 0d8a65b 6238d6c`, covering the registry, factory, runner and scorer dispatch,
the DRAFT config, `test_cuda_tabm_residual.py`, the design note (pre-registration +
Amendment 1), the WIRING record and the disposition. I also read the frozen early-stop loops
in `matrix_models.py`, `direct_residual_model.py` and `direct_tabm_model.py`, plus the
`DirectEmbeddingResidualModel.fit` wrapper, and checked that `DirectTabMResidualModel` has no
`fit` override of its own.

## Overall disposition

**Ready for a CUDA smoke of all 25 tasks** once the two reviewer fixes below are accepted (or
reverted, if preferred: neither changes what a correct run does). Nothing I found blocks it.
One operational constraint (N5): the placeholder deadline also blocks smoke launches after
19:00 KST today.

Reviewer commits on top of `6238d6c`, each separate:

| commit | kind | change |
|---|---|---|
| `e600108` | test only | `tests/test_cuda_tabm_residual_wiring_review.py`: checks `epoch_record` against the real loop of all 5 arm classes; projection fields present in every arm's smoke `fit.json`; 1,224 constant |
| `da3be8c` | minimal fix + test | `run_cuda_direct.epoch_record` refuses `run − best > patience` (finding B1) |
| `4a712f1` | minimal fix + test | scorer `seed_block.note` discloses `backbone_bundle_d10` (finding N1) |

New sha256 values after the fixes: runner `d8d1fab6…416a` (was `051b134b…`), scorer
`3363e959…6b3b` (was `977b76a4…`). The disposition's "File identities" section is now stale
for these two files. The module `direct_tabm_residual_model.py` is unchanged at `6a9daff6…87cf`.
The registry, factory, config and design note are unchanged (`4fe206ee…`, `e1d6aba8…`,
`5e3ce3ad…`, `31e2aa01…`).

## Checklist results

### 1. Pre-registration fidelity — OK

- `ARM_ORDER` (registry l.73) = control, periodic_residual_counts, tabm_counts,
  tabm_residual_counts, tabm_counts_d10. This is the design note's numbering 1–5, and the
  config follows the same order.
- Seeds are exactly `[24..28]` (JSON-exact, so `24.0` is refused), with no `repeat_tag`,
  `fits` 25 and own `check_seeds`. No earlier registry accepts this seed block (tested).
- Cap 60: `TRAINING` (registry l.53) is the cuda_direct_v1 budget with only `epochs` changed.
  The runner's `training_budget` (l.200) returns it only when `uses_tabm_residual`. Every other
  family still compares against `rc.TRAINING` (epochs 30), and the tests check that P30, P31,
  P32, P33 and every DRAFT config validate unchanged, keep the 30-epoch cap and keep the same
  classes. `fit.json` of other families has no `epoch_cap` key (tested).
- I checked that the cap cannot change any run that stops early. In all frozen loops, `epochs`
  is used only as `range(epochs)`. The RNG permutation per epoch and the stopping state do not
  depend on it, so any run that stops by patience before epoch 30 is algorithmically identical
  under cap 60.
- The 7 contrasts match Amendment 1 in order, candidate, reference, isolates and held factors
  (registry l.101–116). `check_contrasts` recomputes the changed-factor set from `FACTORS` and
  checks it equals `ISOLATES`.
- Bundle labelling: both pairs are named `*_bundle`. The registry docstring, the design note,
  `arm_notes.bundle` and the Korean table all say they are not one-factor. Member-head input
  width (256 → 128) is folded into the `backbone` factor, which is disclosed in text. The
  machine-readable `seed_block.note` mentioned only `backbone_bundle`; fixed in N1.

### 2. Arm 5 and dispatch — OK

- Arm 5 `tabm_counts_d10` is JSON-exact to `configs/EXP-P31-001.json` (checked independently
  below, and tested). The control arm is JSON-exact to both P31 and P32.
- In `config_check`, the family check (l.211) runs before the EMA, TabM, embedded-residual and
  workload checks, through an `if/elif` chain, so the TabM family check never sees this config.
- In the per-arm loop (l.239), only `tabm_residual` arms go through the new `check_arm`.
  `tabm_counts` and `tabm_counts_d10` fall through to `direct_tabm_registry.check_arm`, and
  `periodic_residual_counts` to `direct_embed_residual_registry.check_arm`. `build_model` and
  `load_model` (l.466, l.517) put `tabm_residual is True` first.
- I found no way for an arm to be silently routed to another class:
  - An arm carrying both `tabm_residual` and `tabm` is refused, because the extra key breaks the
    exact match to `ARMS`.
  - `tabm_residual: false` or `1` is refused by `check_arm` or the baseline exact match.
  - Arm 5 builds `DirectTabMModel` with `head_dropout` .1 (tested).
  - A `tabm_residual` arm added to any other family is refused by that family's name list.

### 3. Runner — OK after B1

- The epoch count passed to `fit` is `SMOKE['epochs']` (2) for a smoke run and
  `spec['epochs']` (60) for a full run (l.676). A spy test confirms that `fit(epochs=60)`
  arrives for `control` and the new arm.
- `MODEL_SOURCES` includes the 3 new files, and the identity covers `SELF` plus every model
  source.
- The `epoch_record` stop rule matches the loops (details in B1):
  - `stopped_by = 'patience'` exactly when `run − best ≥ patience`.
  - Best epoch equal to the last epoch at the cap gives `'cap'`.
  - A patience stop that falls exactly on the cap epoch gives `'patience'`. This is correct,
    because the break fires and that fit would have stopped there under any larger cap.
  - The original consistency rule missed one impossible case (B1).

### 4. Scorer — OK

- Order inside `score()`:
  1. Config SHA and `config_check`.
  2. All `check_contrasts`, including family exclusivity (l.443–448).
  3. `export_rows`, which reads labels and game ids from the export, not predictions (l.466).
  4. `verify_family` (l.467), which checks per-fit identity, CUDA and `check_epoch_cap` (l.191).
  5. `score_family`, the only place predictions are read.
- So contrasts are checked before any read, and cap records before any prediction read. Cap
  records are not checked before the export label read; the docstring makes no claim otherwise.
- The scorer refuses all of the following (all tested):
  - a missing cap record;
  - a cap of 30;
  - another family's contrast keys;
  - this family's contrast key on P31, P32 or P33;
  - an incomplete family;
  - a DRAFT config.
- Cap hits are listed per fit in `provenance.tasks.*.epoch_cap` and in the Korean table.
- `decision` and `ranking` are `None`, and there are no winner or verdict fields.

### 5. Timing projection — fields present; 1,224 derivable

- `fit.json` carries `report.seconds`, `report.optimizer_updates` and `seconds_total` for all
  4 classes in the family (`MatrixModel`, `DirectPeriodicResidualModel` through the
  `DirectResidualModel` loop, `DirectTabMModel`, `DirectTabMResidualModel` through the inherited
  `DirectTabMModel.fit`). New test `test_timing_projection_fields_present_for_every_arm_class`
  runs a CPU smoke `run_task` for each of the 5 arms. It also checks
  `optimizer_updates == 2 × ceil(training_rows / 1024)`.
- 1,224 updates per epoch:
  - `ceil(1,252,824 / 1024) = 1224` (1223.46), using D100 TRAIN = 1,252,824 rows from
    `docs/contracts/ML-BRIDGE-RUNNER-v1.md` and `ML-BATTER-BRIDGE-v1.md`.
  - The same number is measured on export `49097109`: P32 control seed 16 had 13,464 updates in
    11 epochs = 1,224 exactly (`configs/EXP-P33-001.json` `decay_basis`;
    `OPUS-review-ema-weights-f7282bb` l.53).
  - I did not read the export manifest's TRAIN count, so the first derivation is inferred from
    committed docs. The second is a direct measurement on this export.
- The bias of the projection formula is discussed in N2.

### 6. Config — OK

- `status` is DRAFT and `execution.enabled` is false.
- The placeholder `queue.date` is 2026-10-04 with deadlines 19:00, 19:30 and 20:00 KST. It
  fails closed (N5).
- GPU UUIDs: GPU-c334d905… and GPU-ae803a24…, the same as EXP-P33-001.
- Budgets: task 3600 s, family 36000 s, 2 attempts.
- All 12 `source_configs` sha256 values are current (checked below).
- There are no verdict-type keys. The only match is the top-level `decision` string, which
  says "DRAFT; … No scientific N/G decision", the same pattern as the other configs.
- The `exposure` text covers repeated exposure, ≤2025 data, post-selection, descriptive only,
  and no 2026 data.

### 7. Tests — meaningful

- 9 mutations: 8 were killed, and 1 survived only because it cannot change behaviour (table
  below).
- One real gap: no test linked `epoch_record` to the actual loop, so the impossible-report case
  went unnoticed. Closed by B1 plus the new tests.

## Findings

### B1 (fixed in `da3be8c`; it would not have broken a correct run, but it is a real hole in a fail-closed check) — `epoch_record` accepts reports the frozen loop cannot produce

`scripts/run_cuda_direct.py:588` (as of `6238d6c`):

```python
if not 1 <= best <= run <= cap or (run < cap and run - best < patience):
```

In all three frozen loops (`matrix_models.py` l.143–170, `direct_residual_model.py` l.109–136,
`direct_tabm_model.py` l.210–237):
- `stale` resets on improvement and otherwise increments.
- The loop breaks as soon as `stale >= patience`.
- At the end, `epochs_run − best_epoch == stale`.

So a stop before the cap always has `run − best == patience`, and a run to the cap has
`run − best ≤ patience`.

The old rule accepted `{'epochs_run': 30, 'best_epoch': 10}` at cap 60 as a patience stop
(the loop would have stopped at epoch 15). It also accepted `{'epochs_run': 60, 'best_epoch': 50}`
as a patience stop "at the cap". The scorer reuses `rc.epoch_record`, so it accepted these too.

Concrete failure: a `fit.json` from a modified loop (for example a larger patience) or a
mislabelled archive passes the scorer's cap check.

Fix (applied):

```diff
-    if not 1 <= best <= run <= cap or (run < cap and run - best < patience):
+    if not 1 <= best <= run <= cap or run - best > patience or (run < cap and run - best != patience):
```

The new tests cover both sides:
- `test_epoch_record_refuses_reports_the_frozen_loop_cannot_produce` has 3 cases; all 3 failed
  before the fix.
- `test_epoch_record_agrees_with_the_real_loop_of_every_arm_class` runs lr-0 fits of all 5
  arms. A flat early-stop loss makes the loop deterministic, which gives a patience stop before
  the cap, a patience stop exactly at the cap, a cap stop with 1 stale epoch, and a cap stop at
  the best epoch. These pass before and after the fix, so the tightening refuses nothing a real
  loop produces.

The record is written only for this family, so other families are unaffected.

### N1 (fixed in `4a712f1`) — machine-readable note omitted the Amendment 1 bundle

`scripts/score_cuda_direct.py` l.502 (at `6238d6c`). `seed_block.note` was written for 4 arms
and disclosed only `backbone_bundle`. The Korean table disclosed both bundles. Fix: the note
now adds "backbone_bundle_d10 holds dropout .1 but changes backbone, member-head input width
(256 -> 128) and dropout placement (head -> blocks) together; … neither is one-factor". Test
`test_score_json_discloses_both_backbone_bundles` failed before the fix and passes after.

### N2 (non-blocking, not changed: the formula is pre-registered) — projection bias runs both ways

Formula: `(report.seconds / report.optimizer_updates) × 1224 × 60 + (seconds_total − report.seconds)`.

Overestimates in the train term:
- Smoke has 16 updates (8192 rows / 1024 × 2 epochs), so CUDA and cuDNN warm-up is spread over
  very few updates.
- The per-epoch early-stop pass covers 2,048 rows per 8 updates in smoke, versus 16,000 rows per
  1,224 updates in a full fit. That is about 256 versus about 13 rows per update.

Underestimate in the non-train term: the smoke loads 8,192 TRAIN rows and predicts 256 rows per
split. A full task loads about 1.25 M TRAIN rows and predicts every temperature, blend, DEV and
MLB DEV row.

Expected net effect: conservative for the 1800 s rule. The non-train term is likely tens of
seconds, but it is unmeasured. Suggestion, without changing the rule: when reporting the
projection, also print `seconds_total − report.seconds` from a full P32 or P31 task already on
the server, labelled as a cross-check, not as part of the rule.

### N3 (non-blocking) — WIRING record still describes 4 arms with no superseded banner

`docs/WIRING-TABM-RESIDUAL-20261004.md` l.13–20 and l.95–99 say 4 arms, `fits` 20 and 5
contrasts. The design note and the disposition point forward, but this file does not. I left
it unedited because it is the implementer's record. Suggested first line under the title:

```diff
+> Superseded in part by Amendment 1 / disposition `6238d6c`: 5 arms (+ `tabm_counts_d10`), 25 fits, 7 contrasts. See `docs/reviews/DISPOSITION-tabm-residual-wiring-289fd68-2026-10-04.md`.
```

### N4 (non-blocking) — one guard can never fire

In `score()` l.444, `tabm is not None` in the TabM-residual branch can never be true without an
earlier raise: `tabm_registry.check_contrasts` already calls the TabM `check_family`, which
refuses this arm list. Mutation M8 removed the guard and nothing changed (89 passed). This is
defensive and harmless; no change needed.

### N5 (non-blocking, operational) — placeholder deadline also blocks the smoke

`cuda_direct_queue.Queue` checks `deadlines.may_launch` for smoke and full runs alike (l.319).
With `queue.date` 2026-10-04 and `no_new_fit_after` 19:00 KST, a queue smoke started after
19:00 today launches nothing, and running smoke tasks are killed at 19:30. Either start and
finish the 25-task smoke before then, or re-date the DRAFT `queue` block. Re-dating changes the
config sha (disposition records `5e3ce3ad…`, and no test pins it). Do not edit the design note
without refreshing `source_configs.design_note`, which is test-pinned.

### N6 (non-blocking) — cap records are checked after the export label read

The cap check runs after `export_rows` (labels and game ids) but before any prediction read.
This matches the existing `train_subset` pattern and the docstring, which says "no read of
predictions". It is fine as is.

## Commands and outputs

From `experiments/pitchmdp`, with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`,
`perl -e 'alarm 600; exec @ARGV' /Users/song/Projects/pitcheezy/.venv/bin/python -m pytest <files> -q -p no:cacheprovider`:

```
tests/test_cuda_tabm_residual.py                         (at 6238d6c)        89 passed in 5.04s
tests/test_cuda_tabm_residual_wiring_review.py           (before B1 fix)     3 failed, 7 passed  (the 3 refusal cases)
  ... -k discloses                                       (before N1 fix)     1 failed
tests/test_cuda_tabm_residual.py + _wiring_review.py     (after both fixes) 100 passed in 5.51s
related set (after B1; 16 files: test_cuda_tabm_residual{,_wiring_review}, test_cuda_tabm_mini{,_disposition,_review},
  test_cuda_tabm_registered, test_cuda_embed_residual{,_registered}, test_direct_embed_residual_registry,
  test_cuda_ema{,_registered,_wiring_review}, test_direct_tabm_residual_model, test_direct_tabm_model{,_review},
  test_direct_periodic_residual_model)                                       336 passed in 10.05s
```

Independent config checks (python3, repo root):

```
source_configs: base_cuda, frontier_run, data_combo_run, dropout_run, workload_dropout_run, tabm_run,
  design_note, embed_residual_run, ema_run, tabm_registered, embed_residual_registered, ema_registered -> all OK
JSON-exact vs EXP-P31-001 / EXP-P32-001: control exact, tabm_counts exact, tabm_counts_d10 exact,
  periodic_residual_counts exact; control == P31 control True
module sha256 6a9daff6c3de13a753357d8615bd9f0d5a7bfe796b350f7f7bd0a5e60c9287cf
verdict-type key scan: only /decision (string "DRAFT; ... No scientific N/G decision ...")
math.ceil(1252824/1024) -> 1224  (1223.4609375)
```

Mutations (each applied by exact single-occurrence replacement, the focused file run, then
`git checkout -- <file>`; `git status` clean after the series):

| # | file | mutation | result |
|---|---|---|---|
| M1 | runner | full fit gets global `TRAINING['epochs']` (30) | 2 failed (`test_registered_config_epoch_cap_60_reaches_fit[control, new]`) |
| M2 | scorer | cap record stored without `check_epoch_cap` | 1 failed (`test_scorer_refuses_missing_or_wrong_epoch_cap_records`) |
| M3 | runner | `stopped_by = 'cap' if run == cap` | 1 failed (`test_epoch_record_marks_cap_and_patience_stops`) |
| M4 | registry | `backbone_bundle_d10` isolates also claims `dropout_rate` | 4 failed (contrast check + 3 scorer tests) |
| M5 | runner | TabM-residual family check removed (falls to the TabM check) | 57 failed, 11 errors |
| M6 | registry | seed 28 → 29 | 57 failed, 11 errors |
| M7 | runner | early-stop consistency clause dropped | 1 failed |
| M8 | scorer | `tabm is not None` guard removed | 89 passed: cannot change behaviour (N4) |
| M9 | runner | `epoch_cap` never written | 7 failed |

## Not checked

- No CUDA execution, so device determinism and the timing projection are unmeasured.
- The export manifest's TRAIN row count was not read (the 1,224 sources are listed in §5).
- Full test suite not run (protected mode).
- The module itself was not re-reviewed beyond confirming its pinned hash and that it does not
  override `fit`.
