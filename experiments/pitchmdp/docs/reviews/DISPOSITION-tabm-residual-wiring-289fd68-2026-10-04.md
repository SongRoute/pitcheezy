# Disposition — TabM-residual wiring (impl child `289fd68`, base `45cb770`) onto reviewed module `0d8a65b`

2026-10-04 14:02 KST, delegated Opus PM worker (v3 cycle-005, T33). Wiring child
`impl-tabm-residual-wiring-45cb770` returned 13:54:58 KST (exit 0, no permission denials,
model claude-opus-5-5). The wiring was written from the PRE-disposition module (4 arms, 20 fits).
This disposition is author-side integration, NOT a review: status after it is
`unreviewed (wiring + this disposition); module same_model_reviewed; Codex_review_pending`.
A distinct Opus review of the wiring + this disposition + Amendment 1 is the next gate.

## Integration

- `8c68cc4` merge of `codex/tabm-residual-wiring-1004` (`61d678c`, `289fd68`) onto `0d8a65b`.
  One conflict, design note "Next steps": kept the wiring note, added a disposition pointer,
  item 3 now reads 25 tasks.
- Amendment 1 applied to the wiring (this commit):
  - registry: arm 5 `tabm_counts_d10` = exact EXP-P31-001 entry (built from
    `direct_tabm_registry.ARMS`, JSON-equal to `configs/EXP-P31-001.json`), `ARM_ORDER` 5 arms
    (pre-registration numbering), `FITS` 25, factors `('periodic', counts, 'flatten', .1, 'head',
    'tabm_mini')`; 7 contrasts in the pre-registration order:
    1. `tabm_residual_counts − periodic_residual_counts` (`ensemble_head`)
    2. `tabm_residual_counts − tabm_counts_d10` (`backbone_bundle_d10` = backbone + dropout
       position; dropout rate .1 held; not one-factor: member-head input width is inside
       "backbone")
    3. `tabm_residual_counts − tabm_counts` (`backbone_bundle`)
    4. `tabm_counts − control`, 5. `tabm_counts_d10 − control` (new), 6. `periodic_residual_counts −
       control`, 7. `tabm_residual_counts − control`
  - config `DRAFT-CUDA-TABM-RESIDUAL-20261004.json`: arm 5 appended, `fits` 25, contrasts,
    `arm_notes` (arm 5, bundle wording, `module_pin`), queue batch/order text, `stop_rule` with the
    Amendment 1 projection formula, design-note pin refreshed. Still DRAFT, execution disabled,
    placeholder family deadline (re-date at registration).
  - runner/scorer docstrings: five arms / 7 pairs / 25 fits; Korean table explains both backbone
    bundles and says neither is one-factor.
  - tests: 5-arm/25-fit/7-contrast assertions, arm-5 drift refusals (dropout, flag, order, fits 20),
    d10-pair refusals (wrong isolates, wrong held rate, pair dropped), arm 5 records the cap,
    arm 5 dispatches to the old `DirectTabMModel` path with head dropout .1, module hash pin
    `6a9daff6…87cf` (post-disposition).

## Wiring child open questions

| # | question | disposition |
|---|---|---|
| 1 | smoke cannot measure a 60-epoch fit | already fixed by Amendment 1 (13:56, before this wiring disposition): projected = (report.seconds / report.optimizer_updates) × 1,224 × 60 + (seconds_total − report.seconds), slowest of 5 smoke seeds per GPU. `fit.json` has `report.seconds`, `report.optimizer_updates`, `seconds_total` (runner l.707–708; `direct_tabm_model.py` l.240–241). Smoke has only 16 updates per fit, so per-update time includes warm-up; this overestimates (conservative for the 1800 s rule). |
| 2 | contrast names / factor table chosen by implementer | accepted; two Amendment 1 names added (`backbone_bundle_d10`, `tabm_d10_bundle_vs_control`) |
| 3 | "single-fit means" not an explicit scorer field | accepted as is: scorer records every per-seed delta; the report will compute the plain mean of those recorded deltas (no new score code, no other number). Reviewer may object. |
| 4 | three new files in runner `MODEL_SOURCES` change `own_hashes()` keys for every family | accepted (same as EMA wiring); re-scoring an old family needs its pinned `--source-manifest` |
| 5 | module/test untouched by wiring | confirmed: module sha `6a9daff6…87cf` = post-disposition `0d8a65b` |

## Checks (Mac protected mode, threads 1)

From `experiments/pitchmdp`:

```
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 perl -e 'alarm 900; exec @ARGV' \
  /Users/song/Projects/pitcheezy/.venv/bin/python -m pytest tests/test_cuda_tabm_residual.py -q -p no:cacheprovider
→ 89 passed in 5.07s   (before the head-dropout assertion was added; rerun below includes it)

... tests/test_cuda_tabm_residual.py tests/test_cuda_tabm_mini.py tests/test_cuda_tabm_mini_disposition.py
    tests/test_cuda_tabm_mini_review.py tests/test_cuda_tabm_registered.py tests/test_cuda_embed_residual.py
    tests/test_cuda_embed_residual_registered.py tests/test_direct_embed_residual_registry.py tests/test_cuda_ema.py
    tests/test_cuda_ema_registered.py tests/test_cuda_ema_wiring_review.py tests/test_direct_tabm_residual_model.py
    tests/test_direct_tabm_model.py tests/test_direct_tabm_model_review.py tests/test_direct_periodic_residual_model.py
→ 326 passed in 9.27s
```

Mutation: arm-5 factor `dropout_position` 'head' → 'blocks' in the registry → 4 failed / 85 passed
(contrast-factor checks refuse it); file restored from copy, sha identical.

Full suite not run (protected recording mode). No server, GPU, smoke, fit or score.

## File identities after this disposition

- config `5e3ce3ad…6d34` (`configs/DRAFT-CUDA-TABM-RESIDUAL-20261004.json`)
- registry `4fe206ee…f319`, factory `e1d6aba8…f86b` (unchanged from wiring), runner `051b134b…dd3`,
  scorer `977b76a4…019f` (pre-final-commit; the scorer docstring/table only)
- design note `31e2aa01…5ab6`, module `6a9daff6…87cf`

## Next

1. Distinct Opus review of `git diff 0d8a65b <this commit>` (wiring + disposition + Amendment 1).
2. Disposition → CUDA smoke of all 25 tasks on GPU3/4 (fresh archive/code dir; foreground
   single status reads; never a remote wait loop; never create/chmod `p33-launch.json`) →
   projection by the fixed formula → `결정 필요` for the full run (Song's rule for new experiments).
