# Direct head periodic numerical embeddings — DRAFT implementation (2026-10-03)

Author: Claude Opus 5.5 (implementation worker). Codex PM does the independent review.
Status: DRAFT. No ID registered. No real-data load, fit, predict or score. No newsrv. Nothing pushed.

## What changed (one factor)
80 continuous scalars per row go from raw value to an 8-dim periodic embedding (PLR-lite).
The head is still the EXP-P15-001 `flatten_mlp` with width 128: Linear(n,256) GELU Linear(256,256) GELU Linear(256,10).
Its first Linear widens from 286 to 846 inputs. Budget, early stop, May temperature, June blend, controls (frozen EXP-P15-001 and its fixed 1/2 G0 mix) and the descriptive scorer are those of the residual draft (`9b6ec19`).

## Source and adaptation
- Paper: Gorishniy, Rubachev, Babenko 2022, https://arxiv.org/abs/2203.05556.
- I read the official package at commit `a8fc250` (https://github.com/yandex-research/rtdl-num-embeddings, MIT): `package/rtdl_num_embeddings.py` (`_Periodic`, `PeriodicEmbeddings`) and `package/README.md`. I reimplemented the formula myself and copied no code.
- Formula: `ReLU(Linear(cat(cos v, sin v)))`, where `v = 2π·c·x` and `c ~ TruncNormal(0, σ, ±3σ)`. The frequencies are trainable.
- Fixed before any score:
  - σ = 0.01. This is the package default; the README calls it safe.
  - 16 frequencies and d_embedding = 8. The package defaults are 48 and 24. I made them smaller to limit how much the head grows.
  - `lite`: one shared outer Linear. The README calls `lite=True` "a reasonable starting point".
  - No tuning (hpo_trials 0).

## Input contract (from the code, not assumed)
- **Tokens [N, 6, 38]:**
  - channels 0–7: standardized physics (`effective_speed`, `release_spin_rate`, spin-axis sin/cos, `pfx_x/z`, `plate_x/z`)
  - channels 8–26: pitch-type one-hot (18 TRAIN types + unknown)
  - channels 27–37: outcome one-hot
  - position 5 is the current pitch; its physics is all zero (`direct_arrays`)
- **Context [N, 52]:**
  - 0–10: SequenceContext game state and flags
  - 11–27: batter style priors, reliabilities and memberships
  - 28–50: pitcher profile — 4 physics means, `left_share`, 18 type proportions
  - 51: `log1p(count)/10`
  - The 7 routing columns are already stripped.
- **Embedded:**
  - physics channels 0–7 at past positions 0–4 (40 scalars). Their embedding is multiplied by the mask, so padding gives exactly 0.
  - context 11–31 and 33–51 (40 scalars)
- **Raw:**
  - past type/outcome one-hots
  - the whole current token. Its physics is zero and never embedded, because embed(0) ≠ 0.
  - the mask
  - context 0–10: integer ordinals and binary flags
  - context 32: `left_share`, which is near binary
- **Guards:**
  - `context_map` derives the split from the frozen pitcher-profile columns.
  - The runner refuses a parent whose `clusters.columns` differ from the config.
  - The model refuses nonzero current physics in both fit and logits.

## Parameters
- Candidate: 286,738 = frequencies 1,280 + shared embedding Linear 264 + head 285,194.
- Control: 141,834. The candidate has 2.02× as many parameters. Capacity and representation are confounded; there is no equal-parameter control.

## Files
- `experiments/pitchmdp/pitchmdp/direct_periodic_model.py`: `PeriodicDirectNetwork` and `DirectPeriodicModel`. Fit is inherited read-only from `DirectResidualModel`. The model has its own archive format (`direct_periodic_model_v1`).
- `experiments/pitchmdp/scripts/run_ml_direct_periodic.py`: run/predict/score, smoke, seeds 0,1,2, `--device auto|mps|cpu|cuda`. It mirrors the residual runner.
- `experiments/pitchmdp/tests/test_run_ml_direct_periodic.py`: 11 synthetic CPU tests.
- `configs/DRAFT-DIRECT-PERIODIC-20261003.json`: `execution.enabled false`.

## Tests run
```
cd experiments/pitchmdp && DYLD_LIBRARY_PATH=<venv>/torch/lib <venv>/bin/python -m pytest \
  tests/test_run_ml_direct_periodic.py tests/test_run_ml_direct_residual.py tests/test_run_ml_direct_head.py -q
27 passed in 2.21s   (periodic alone: 11 passed in 2.51s)
```
The tests cover:
- the column map on the registered 52-wide contract, and rejection of a bad layout
- fixed σ, frequency count, d_embedding and parameter counts
- padded embedding = 0 and invariance to padding payload; the current-physics guard in fit and logits
- the inherited loop equals `MatrixModel.fit` bit for bit when given the flatten network
- save/load: identical logits and predictions, and each loader rejects the other formats
- the config check refuses DRAFT and parameter drift
- source/map/device identity
- fit and predict reuse finished steps only after verification; partial fit, partial prediction, a stray output folder and a moved profile layout are all refused
- the spy sees zero current physics and the EXP-P15-001 budget
- an `auto` run that falls back to CPU is marked not comparable to the MPS control; an explicit `mps` run is marked comparable

## Limitations and risks
- `pitcher_profile_columns` in the config holds 18 placeholder `type_*` names. Before registration the root must copy them from the frozen EXP-P4-001 `preparation.json` `clusters.columns`. Until then the runner refuses real data, by design.
- The 28-wide SequenceContext is a code constant (`BASE_CONTEXT`). It is checked only indirectly, through `n_context` = 52.
- If a prediction is interrupted after the archive is written, the rerun rewrites `calibration.json` and only then refuses. This is inherited from the direct and residual predict mirror.
- Cost on MPS (80×16 cos/sin per row) is 미측정. Not tested on MPS or CUDA.
- DEV has been read many times, so any score is descriptive only. No decision, no holdout claim, no 2026 rows.

## Codex PM review follow-up (2026-10-03 12:34 KST)

The PM read the implementation and runner, copied the exact frozen parent profile column names from preparation metadata into the DRAFT config, and fixed partial-prediction refusal to happen before calibration or prediction writes. A regression assertion verifies that every existing file's bytes and modification time remains unchanged when the completion state is absent. Focused periodic tests: 11 passed in 2.72 s. These changes supersede the placeholder-column and partial-calibration-write limitations above. Real-data smoke and full fits are still pending.
