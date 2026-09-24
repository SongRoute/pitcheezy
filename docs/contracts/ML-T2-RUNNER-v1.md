# T2 additive runner, version 1

Status: implemented and tested with synthetic CPU data only. No real T2 fold,
resource profile, neural fit, prediction or score has been executed. The selected
G comparison and its immutable parent hashes must be registered before execution.

This runner implements the exclusion and observation rules in
[ML-T2-T3-PROTOCOL.md](ML-T2-T3-PROTOCOL.md). It is an artifact producer, not a
scorer or model selector. Root orchestration owns candidate selection, scoring,
the complete hypothesis family and the one-heavy-job execution slot.

## Configuration

`scripts/run_ml_generalization.py` requires these exact settings:

- `protocol`: `ml_generalization_v1`; `scope`: `Cpanel`.
- `experiment_id`, `parent_run`, `parent_preparation_sha256`, and
  `parent_analysis_sha256`: frozen G run and its completed panel analysis.
- `candidate`, `control`, `selection_basis`: frozen comparison and rationale.
  Supported mappings are G1-personal→G0-global, G2-feature→G0-global,
  G3-cluster→G2-feature, and G4-partial→G2-feature.
- `seeds`: `[0,1,2]`; `axes`: `["pitcher","batter"]`;
  `regimes`: `["Z","W","O"]`.
- `prefix_games`: 2; `selector_seed`: 20260924; `draws`: 400; `width`: 128.
- `budget`: epochs 30, patience 5, batch_size 1024, learning_rate .0005.
- `device`: `auto`, `cpu` or `mps`.
- `adaptation`: `fixed_prefix_context_only_v1`.
- Optional `registration`: the complete inference family, margins, reporting
  gates and resource/candidate-selection rationale. It cannot override the fixed
  computational settings.

The CLI takes `--config`, `--local-config`, and `--output`, followed by:

1. `prepare`
2. `profile --axis pitcher|batter`
3. `fit --axis pitcher|batter --seed 0|1|2`
4. `predict --axis pitcher|batter --cell CELL --seed 0|1|2`
5. `status`

All mutating/heavy commands acquire the existing ML matrix heavy lock. The caller
must enforce a 600-second profile timeout and 7200-second fit/predict command
timeouts. Each fit command completes the selected fold's dependency union for
one seed. Completed units resume after hash checks; partial output requires
review. Profiling uses at most 8192 TRAIN, 2048 early-stop and 16 temperature rows,
two epochs, and 400 draws; it never evaluates DEV predictions or emits quality
scores. Real fits require the matching successful resource profile.

## Fold preparation and fitting

The primitive loader verifies the same processed/cache bytes and provenance as
the G parent, filters regular-season records and assigns the 2025 temporal fold.
It deliberately avoids computing inherited unsanitized batting histories.
Selection and historical exclusion include the pre-TRAIN 2023 observations.

Each fold freshly fits its normalizer, type vocabulary, frequency baseline,
batter encoder, pitcher profiles/centroids, and delivery pools. Parent auxiliary
pickles and neural checkpoints are not reused. Whole affected PAs are removed
from fitting and calibration. Parent ordered D100/early-stop/CAL samples are
intersected with these exclusions; a reduced TRAIN sample is rejected.

Pitcher fits only the three global members. It uses common fallback and common
calibration, so no architecture contrast is implied for unknown pitcher routing.
O can expose prefix outcomes to legal batter context, but does not assign a new
pitcher cluster, update delivery pools or fit a personal expert.

Batter fits the selected candidate/control dependency union. Global fallback is
always included. Cluster/personal units require at least 500 allowed TRAIN and
20 allowed early-stop pitches; otherwise their fallback remains in the target
denominator. Maximum units per seed are global+feature for G2/G0,
global+personal for G1/G0, global+feature+four clusters for G3/G2, and that union
plus eligible personal models for G4/G2. Matching units are reused within the
fold, never across distinct exclusion folds.

An empty selected cohort or no eligible post-prefix panel observations remains
unmeasured without replacement. Each held-out entity's exposure prefix is drawn
from its first two complete-log DEV games, even when the batter faced pitchers
outside Cpanel. Scored rows are restricted to Cpanel and dates strictly after the
entity's second exposure-game date. These identical rows are used for Z/W/O.

## Artifacts and future scoring

Preparation archives source snapshots, the selected G analysis, sample keys,
exposure keys, fold auxiliary pickles, grouping metadata and baseline predictions.
It records source/environment identities, all artifact hashes and external G
analysis dependencies. The natural-new-matchup audit checks the complete
pre-DEV history, TRAIN and CAL sources and copies the appropriate first-meeting
subset of frozen parent predictions. It performs no new fit or diagnostic score.

Each `axis/members/CELL/seedN/predictions.npz` contains:

- `blend`, `blend_raw`, `blend_delivery_level`, and keyed June metadata;
- `Z`, `W`, `O`, their `_raw` and `_delivery_level` arrays;
- one common `dev_keys`, `dev_y`, `dev_pitcher`, `dev_batter`, `dev_game_pk`.

One allowed-May temperature is reused across all observation regimes. The June
predictions are common, so the later scorer must fit one ensemble June blend per
cell and one June blend per seed, then apply those same weights to Z/W/O. Do not
recalibrate on prefix outcomes or regime-specific DEV predictions.

Prediction manifests pin every contributing unit state, checkpoint and fit report.
Resume checks reject changed dependencies. The scorer must validate every active
fold/cell/seed before opening scores, preserve inactive hypothesis slots, compare
identical original truth keys across regimes, and distinguish the frozen natural
matchup diagnostic from the exclusion-fold retrains. It must not interpret an
unchanged unknown-pitcher routing branch as learned pitcher adaptation.
