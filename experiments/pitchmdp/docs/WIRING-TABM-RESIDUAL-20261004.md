# WIRING — TabM-mini × periodic_residual family (DRAFT, 2026-10-04)

Status: **unreviewed; same_model_review pending; Codex_review_pending.** Delegated Opus 5.5
implementation worker, branch `codex/tabm-residual-wiring-1004`, base `45cb770`. No push, no
merge, no smoke, no registration, no fit on real data, no GPU, no server, no 2026 data.

Binding design: `docs/DRAFT-TABM-RESIDUAL-DESIGN-20261004.md` (arms, seeds, cap and contrasts
unchanged; only its "Next steps" item 2 got an implementation note).

## Files changed

New
- `pitchmdp/direct_tabm_residual_registry.py` — torch-free. 4 arms in order `control`,
  `periodic_residual_counts`, `tabm_counts`, `tabm_residual_counts`; the first three are the
  exact registered entries (built from the TabM / embed-residual registries, tested against the
  committed EXP-P31-001 / EXP-P32-001 arm text, JSON-exact). Own `check_seeds` (exactly
  `[24, 25, 26, 27, 28]`, no repeat_tag, `fits` 20), `check_training` (cuda_direct_v1 budget
  with `epochs` 60, everything else unchanged), `check_family`, `check_contrasts` (exactly the
  5 pre-registered pairs, each verified to change exactly its declared factor set).
- `pitchmdp/direct_tabm_residual_factory.py` — `build_tabm_residual` /
  `load_tabm_residual` via `DirectTabMResidualModel`; load refuses another format/kind (class
  loader), type, `kind`, architecture (backbone + embedding + k + scaling init), a
  continuous map other than the registered one (model attribute and `net.config`, same length
  or not) and another context width (52 + workload).
- `configs/DRAFT-CUDA-TABM-RESIDUAL-20261004.json` — status DRAFT, `execution.enabled` false,
  export pin `49097109…` (export / calibration blocks = EMA DRAFT), workload counts, seeds
  24–28, `training.epochs` 60 (all arms), task cap 3600 s, family cap 36000 worker-s,
  `max_attempts` 2, GPU3/4 UUIDs as EXP-P33-001, `queue.date_note` = PLACEHOLDER family
  deadline to re-date at registration (fails closed after 2026-10-04 19:00 KST until then),
  `queue.stop_rule`, `exposure` and `data_rules` text; no winner/promotion/verdict/ranking keys.
  `source_configs` pins EMA DRAFT, EXP-P31/P32/P33 and the pre-registration (all hashes current).
- `tests/test_cuda_tabm_residual.py` — 79 focused CPU synthetic tests.
- this file.

Modified (additive)
- `scripts/run_cuda_direct.py` — `uses_tabm_residual`, `training_budget` (returns the
  unchanged `TRAINING` object for every other family), family check first (before EMA / TabM /
  embed-residual, because two arms keep those flags), per-arm `tabm_residual` branch,
  `build_model` / `load_model` dispatch, `epoch_record`; the fit epoch count is taken once
  (`SMOKE['epochs']` or `config.training.epochs`) and passed to `fit(epochs=...)`; only this
  family writes `fit.json["epoch_cap"] = {cap, patience, epochs_run, best_epoch, stopped_by}`.
  Three new files appended to `MODEL_SOURCES` (before the frontier pair, as for EMA).
- `scripts/score_cuda_direct.py` — imports the new registry (added to `SCORER_SOURCES`);
  `tabm_residual_contrasts` checked before any export/prediction read; the family is scored
  only with them (any other contrast key refused; the key on any other family refused);
  `verify_family` requires every fit's `epoch_cap` to equal the config cap and its own
  report; output `tabm_residual_contrasts`, `seed_block.epoch_cap`, per-fit
  `provenance.tasks.*.*.epoch_cap`; Korean table section + per-fit cap table.
- `docs/DRAFT-TABM-RESIDUAL-DESIGN-20261004.md` — Next steps item 2 note only.

Not touched: `pitchmdp/direct_tabm_residual_model.py` (sha256 `5cf6ea5f…c166`),
`tests/test_direct_tabm_residual_model.py` (`ddba1dd7…d713`), every frozen model module,
`scripts/cuda_direct_queue.py` (generic, config-driven; verified the placeholder deadline parses).

## Config sha256

`configs/DRAFT-CUDA-TABM-RESIDUAL-20261004.json` =
`1b66971173c0623b4d57f6812adefbe7a8027084d8514ac19eff8960253b006b`
(design-note pin inside = `f4dff969fe300ab1b73aa4ef4de5788adb75e22af9fd61c1aab0f5dad36e5c8d`).

## Tests (Mac, protected recording mode)

From `experiments/pitchmdp`:

```
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 perl -e 'alarm 900; exec @ARGV' \
  /Users/song/Projects/pitcheezy/.venv/bin/python -m pytest tests/test_cuda_tabm_residual.py -q -p no:cacheprovider
→ 79 passed in 4.25s

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 perl -e 'alarm 900; exec @ARGV' \
  /Users/song/Projects/pitcheezy/.venv/bin/python -m pytest tests/test_cuda_tabm_residual.py \
  tests/test_cuda_tabm_mini.py tests/test_cuda_tabm_mini_disposition.py tests/test_cuda_tabm_mini_review.py \
  tests/test_cuda_tabm_registered.py tests/test_cuda_embed_residual.py tests/test_cuda_embed_residual_registered.py \
  tests/test_direct_embed_residual_registry.py tests/test_cuda_ema.py tests/test_cuda_ema_registered.py \
  tests/test_cuda_ema_wiring_review.py tests/test_direct_tabm_residual_model.py -q -p no:cacheprovider
→ 265 passed in 7.42s
```

Full suite not run (instruction). Mutation checks (reverted, tree clean afterwards): runner
passing `TRAINING['epochs']` instead of the config cap → 2 failures
(`test_registered_config_epoch_cap_60_reaches_fit[*]`); disabling the family check → 49
failures + 10 errors.

What the new tests cover: config = DRAFT exact family/seeds/cap/export/device/queue limits,
pins current, no verdict-type keys; earlier arms JSON-equal to committed EXP-P31/P32 entries;
registry constants = torch module; seed block disjoint from 0–23 and refused by the TabM /
embed / EMA registries; 29 drift mutations refused (cap 30/61, k, scaling init, extra keys,
flags, reorder, seeds, fits, device, 2026); every earlier config (base, frontier, combo,
dropout, workload-dropout DRAFT, P30, TabM/embed/EMA DRAFTs, P31/P32/P33) still validates
unchanged with `training_budget(cfg) is TRAINING`, refuses seeds 24–28, cap 60, a
`tabm_residual` arm or the new contrast key; P31/P32/P33 keep their contrasts and dispatch
(MatrixModel / DirectTabMModel / DirectPeriodicResidualModel / FlattenEMAModel /
FlattenRegularizedModel); old-family fit.json has no `epoch_cap`; scorer refuses the new key
on registered P31/P32/P33 before any read; factory round trip, refusals of parent/control
archives, same-length other map, k=4, random-signs, wider context; runner smoke on all 4 arms
(cap record, stopped_by `cap` at the 2-epoch smoke cap), seed refusal (0,3,8,13,16,19,23,29),
DRAFT refuses real fits; registered non-smoke synthetic run passes `epochs=60` to `fit` for
control and the new arm and fit.json records it; `epoch_record` cap/patience cases; scorer
reports all 5 contrasts (delta = difference of vs-control deltas), per-fit cap hits in JSON and
table, refuses missing / wrong-cap records, incomplete family, missing or mixed contrast keys.

## NOT done

- No CUDA smoke, no queue run, no registration, no `REGISTERED` config, no push/merge.
- No real-data fit or timing. `tabm_residual_counts` full-fit seconds at the 60 cap: 미측정.
- Saved-probability grid file (pre-registration: committed BEFORE any score of this family
  is read; seed subsets and n ∈ {1, 2, 3, 5}) is not written; no scorer for it.
- Family deadline is a placeholder (2026-10-04); must be re-dated at registration.

## Open questions / notes for the reviewer

1. **Smoke cannot measure the 60-cap time.** The pre-registration says "smoke measures it
   before registration" and stops if "smoke time per TabM-residual fit > 1800 s at the 60
   cap", but runner smoke is 8192 rows × 2 epochs. Needs a decision: extrapolate from smoke
   seconds per optimizer update × (full TRAIN updates/epoch × 60), or a separately approved
   one-epoch full-data timing probe. I did not add either.
2. Contrast labels for the three vs-control rows (`tabm_bundle_vs_control`,
   `residual_bundle_vs_control`, `tabm_residual_bundle_vs_control`) and the factor table
   (input, workload, backbone, dropout rate/position, head) are my naming; pairs and order are
   exactly the pre-registration. Control dropout is recorded as 0 / none (flatten_mlp).
3. Pre-registration says "seed-mean ensemble and single-fit means". The scorer (unchanged
   helper) reports the ensemble paired delta and every per-seed paired delta; an explicit
   mean of single-fit deltas is not computed.
4. Adding three files to `MODEL_SOURCES` changes `own_hashes()` keys for every family (same
   pattern as every earlier wiring). Re-scoring an old family with the current runner needs
   its pinned `--source-manifest` (already the case since EMA wiring).
5. `epoch_cap` in fit.json is written only for this family so earlier families' task outputs
   stay as before; the scorer requires it for this family.
6. Model module (not edited, read only): no defect found while wiring. Observation only:
   `DirectTabMResidualModel.__init__` goes through `DirectTabMModel.__init__` with head dropout
   0 and then replaces `architecture`; the factory therefore compares the whole architecture
   dict (residual backbone + embedding + ensemble), not `head_dropout`.
