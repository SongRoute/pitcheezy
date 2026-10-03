# CUDA direct-head frontier arms (2026-10-04, DRAFT, not run)

Status: implementation draft for root review. Synthetic CPU tests only. No SSH,
no real export, no real fit, no prediction or scoring. Parameter counts below are
computed from the frozen network definitions (n_context 52, n_token 38, length 6).

Files (all new; no existing runner, model or config changed):

- `experiments/pitchmdp/pitchmdp/direct_frontier_model.py` — registry `ARMS`,
  `build_frontier(arm, seed, device, context_names=None)`,
  `load_frontier(arm, path, device)`, `check_config(config)`,
  `direct_context_names(cluster_columns)`
- `experiments/pitchmdp/tests/test_direct_frontier_model.py` — 16 synthetic tests
- `configs/DRAFT-CUDA-FRONTIER-20261004.json` — `cuda_frontier_v1`, 8 arms × seeds 0-2 = 24 fits

Question: does the direct head's DEV performance come from capacity, from input
information or from regularization? One disclosed change per arm against the
same-device CUDA control (flatten_mlp 128, 141,834 params, AdamW wd .01, no
dropout, all inputs, 30 epochs / patience 5 / lr 5e-4 / batch 1024, uniform).

## Arms

Capacity (plain `MatrixModel('flatten_mlp', width=w)`, no new class).
flatten params = h² + 298h + 10, h = 2w.

| arm | width | params | target | gap |
|---|---|---|---|---|
| width145 | 145 | 170,530 | residual 170,634 | -0.06 % |
| width203 | 203 | 285,834 | periodic 286,738 | -0.32 % |

Targets are recomputed from `direct_residual_model.parameter_count` and
`direct_periodic_model.parameter_count` in the tests, not copied.

Information (`MaskedFlattenModel`: control network and parameter count; the
same mask applied in `fit` and `logits`, archived with the model):

- `no_batter` — context 11-27 (17): 6 standardized batter style rates, 6
  reliabilities, 5 soft archetype memberships
- `no_batter_cluster` — context 23-27 (5): memberships only. Verified:
  `Archetypes.numeric_features` stacks styles, reliabilities, memberships, so the
  5 cluster columns are the LAST 5 of the 17, not the first 5
- `no_history` — past positions 0-4 marked invalid, so MatrixNetwork zeroes their
  38 channels and mask bit (195 flat inputs). Current token (type/action, zero
  physics, zero outcome) and all 52 context columns unchanged (tested)

The pitcher cluster one-hot is not in the 52 (it is one of the 7 routing
columns `training_arrays` strips), so no pitcher-cluster arm exists here.
Zeroed inputs leave `zeroed × 256` first-layer weights without data gradient
(recorded in `report`: 4,352 / 1,280 / 49,920); the nominal count stays 141,834.
Zero is the TRAIN mean for standardized styles, "no evidence" for
reliabilities, and not the uniform .2 for memberships; any constant removes the
information, which is the arm's only claim.

Regularization (`FlattenRegularizedModel`: control Linear layers with
`Dropout(p)` after each hidden GELU, 141,834 params; fit is a copy of
`DirectResidualModel.fit` with only the AdamW weight decay read from the arm):

| arm | dropout | weight decay |
|---|---|---|
| dropout | .1 | .01 (control) |
| weight_decay | 0 | 1e-4 |
| dropout_weight_decay | .1 | 1e-4 (two-component bundle) |

Separate single-component arms isolate each effect; the third is labeled a bundle.

## Schema derivation (no guessing)

`direct_context_names(clusters['columns'])` returns the 52 ordered names:
`SequenceContext.report()` game-state names (0-10), `archetypes.STYLE_COLUMNS`,
`RELIABILITY_COLUMNS`, `batter_archetype_membership_0..4` (Archetypes(n_clusters=5)),
`pitcher_<clusters['columns']>` (CLUSTER_FEATURES, left_share, type_*), and
`pitcher_log1p_train_count_over10`. Tests check these against the frozen
source and against `direct_periodic_model.context_map` (same 52-wide layout,
left_share at 32). The module does not import pandas, so it runs in the CUDA venv.

Guards: drop names must each match exactly one column; the first 28 names must
equal the frozen layout; context width must equal the archived name count; the
history mask raises if the current token changes. `load_frontier` refuses an
archive of another arm (width, mask or regularization differs).

## Tests (synthetic, CPU)

`pytest tests/test_direct_frontier_model.py` → 16 passed; with
`test_cuda_direct`, `test_run_ml_direct_residual`, `test_run_ml_direct_periodic`
→ 62 passed.

- every arm trains, saves, loads and gives identical logits after reload
- empty mask and (p 0, wd .01) both reproduce `MatrixModel` flatten 128 logits
  exactly for the same seed (so a difference comes only from the arm's change)
- each mask changes exactly the expected flat input positions and nothing else;
  predict applies the same mask (shifting dropped inputs leaves logits unchanged)
- config guard rejects changed width, missing/duplicated arm, budget, seeds, dropout

Current-pitch physics: the synthetic inputs keep it zero and the masks never
touch the current token. The single zero-physics gate for every arm is the
runner's `arm_inputs` check (run_cuda_direct), which the adapted runner must keep.

## For root integration

- Exporter: add `context_names = direct_context_names(clusters['columns'])` to
  the manifest; the runner passes it to `build_frontier` (information arms only).
- Runner: replace `build_model`/`load_model` with `build_frontier`/`load_frontier`
  for these arms; add `pitchmdp/direct_frontier_model.py` to the source hashes.
- Control reuse: reuse the base-batch CUDA control fits only if matrix_models.py
  hash, training loop and budget, export manifest SHA256, device class and
  torch/CUDA runtime are identical; otherwise refit the control here.
- Cost: 24 fits; per-fit time 미측정 (no GPU smoke).

Deferred to a separate task: periodic+residual combination, TabM.

## Integration follow-up (NEW code version, DRAFT, not run)

Frontier arms now run through the existing CUDA runner; a portable scorer runs on
newsrv. First-batch code (edfd4cc) and the immutable export are unchanged.
No SSH, GPU, real data or 2GB re-export was used. Only the small export
`manifest.json` (SHA 49097109…) metadata was read.

- `pitchmdp/direct_frontier_registry.py` (new, no torch): `ARMS`, names, `check_config`
  moved out of `direct_frontier_model.py` and re-exported unchanged. `config_check` runs
  before the GPU is bound, so it must not import torch.
- `scripts/run_cuda_direct.py`: an arm with `"frontier": true` must equal its registry entry
  (+ name, priority, empty workload). Base arms keep the old checks. `build_model` /
  `load_model` dispatch to `build_frontier` / `load_frontier`. `MODEL_SOURCES` now also
  lists both frontier modules, so new-code task source hashes differ from the first batch.
  Information arms get the 52 names only after the export manifest's
  `lineage.context_layout` exactly matches the config's expected layout:
  - groups 0:28 / 28:51 / 51:52
  - base features = 11 names + grouped style6 / reliability6 / membership5
  - Archetypes style/reliability columns and n_clusters 5
  - 23 pitcher-profile columns, 7-column stripped tail

  Names come from source equivalences only. An optional `export.manifest_sha256` pin is enforced.
- `configs/DRAFT-CUDA-FRONTIER-RUN-20261004.json` (sha256 `2b057e8ede5ddb9dd2f880d352fc272fc51fa9e1344c4d7993e55d02bac42fc7`): base CUDA config + fresh
  uniform control + 8 frontier arms = 27 fits (seeds 0-2), DRAFT, execution disabled. Root sets
  the queue date and cutoffs and registers it.
- `scripts/score_cuda_direct.py`: numpy + `matrix_metrics` only. Before any probability is
  read, it requires all of these:
  - the config is pinned and REGISTERED
  - the export manifest is pinned, with hash-checked dev/mlb_dev keys/y/game_pk
  - every arm×seed has a finished non-smoke CUDA identity with the EXPECTED source map
  - one shared CUDA runtime signature

  Each member's keys, labels and game order must equal the export rows. It reports the
  3-seed-mean calibrated direct probabilities vs the same-family control on dev and mlb_dev
  (`prediction_metrics`, `paired_game_comparison` with 10,000 draws, seed 20260924), for the
  ensemble and for each seed. This is before the June blend / G0 mix: not G15, not policy,
  and no decision, ranking or winner. Output is a new read-only directory: `score.json`,
  `table_ko.md`, `SHA256SUMS`. Scorer source hashes are recorded separately from task sources.
- To score the first batch, pass `--source-manifest` with a `cuda_direct_task_sources_v1` file
  that holds the edfd4cc `own_hashes()`: runner + 3 model sources, no frontier modules. Root
  writes it from the local first-batch tree. The default is the current runner's map. An
  artifact's self-declared sources are never trusted.

Tests (synthetic, CPU): `tests/test_cuda_frontier_run.py` 29 new. Focused
(frontier_run + cuda_direct + frontier_model) 82 passed. The direct/cuda/metrics set
173 passed. The full `experiments/pitchmdp/tests` suite: 960 passed, 1 skipped.
