# DRAFT direct-head ensemble search (66 fixed candidates) — implementation report, 2026-10-03

Status: DRAFT, `execution.enabled: false`, no experiment ID (root assigns). Grid fixed before any P20/P21/P22 result.
Nothing was run on real data; no prediction payloads were read. Needs independent root research review before registration.

## What
- Runner `experiments/pitchmdp/scripts/search_direct_head_ensemble.py` (additive; P19 runner and frozen sources untouched).
- Config `configs/DRAFT-HEAD-ENSEMBLE-66-20261003.json`; tests `experiments/pitchmdp/tests/test_search_direct_head_ensemble.py`.
- Six direct heads: frozen P15 uniform, P20 hl180/hl365/hl730, P21 residual, P22 periodic. Seeds 0,1,2.
  22 subsets (6 singletons, 15 pairs, all six), equal-probability mean of the same seed's members, mixed with the same seed's frozen G0 at w = .25/.5/.75 → 66 candidates.
  No temperature grid: each member's existing May calibration. Then the P19 Cpanel path (`p19.cpanel` → `summarize_cell`, three-seed mean, June blend).
- Whole MLB: same members, each candidate's own Cpanel June weight against the frozen frequency baseline (P15/P16/P19 construction). This is labelled as different from the unblended P20–P22 secondary.
- Gate first: three-seed G0, uniform direct and uniform 1/2 mix must reproduce `results/EXP-P15-001.json` (21 checks, tol 1e-9) on Cpanel and whole MLB. The control row `P15-uniform_w0.50` must equal the reproduced mix bit for bit.
- All 66 rows kept. Ranks are descriptive and reuse `p19.rank`/`p19.describe`. No p-values, no winner, no promotion.

## Fail-closed verification (before any array is loaded)
- Per family: preparation and registered_config hash-covered. Config canonical hash equals the sibling config, computed now from git blobs: P20 `f19c69e`, P21 `2c6d9e7`, P22 `69d3a37`. The shared direct-head source hashes equal the frozen P15 control's. The original source snapshot equals the pinned own sources. Recency checks variant/half-life/anchor; residual/periodic check architecture (and input_map) against the registered config. `resolved_device == mps` and `comparable_device_to_control`.
- Per seed: fit, prediction and mlb_prediction states all present, `smoke: false`, same preparation and seed, predictions made by the hashed `model.pt`, artifact hashes intact, `fit.json` report device `mps`. CPU and CUDA are refused.
- `run` additionally requires pinned preparation/state SHA256 for every family. `pins` produces them from state JSON only, after the families finish. The P15 fit_state pin is still null.
- Arrays: Cpanel keys/labels/game/pitcher per component (`assert_aligned`), whole-MLB keys/labels, `(n,10)` finite probabilities summing to 1. Outputs are never overwritten.
- Shared `.heavy.lock` plus `check_location`. Operating windows are enforced by the external supervisor.

## Commands / output
```
cd experiments/pitchmdp
<venv>/bin/python -m pytest tests/test_search_direct_head_ensemble.py tests/test_search_predictor_blend_temperature.py -q
33 passed in 3.34s
<venv>/bin/python scripts/search_direct_head_ensemble.py --config ../../configs/DRAFT-HEAD-ENSEMBLE-66-20261003.json --local-config configs/local.json --output /tmp/hes-<pid>-smoke smoke
ENSEMBLE_SMOKE_DONE {'gate': True, 'candidates': 66}
```

## Cost (estimates, not measured)
Load 18 direct archives + 3 G0 + frequency from T7: P19 loaded a comparable set in 2.8 s. Scoring 69 keys × 3 seeds over 311,721 whole-MLB rows: about 20–90 s (P19 total 21.3 s for 77×5). Peak RSS about 2–4 GB (P19 2.69 GB; whole-MLB direct arrays kept as `mlb_dev` only). `max_wall_seconds` 600.

## Limitations
- All 2025 DEV is repeatedly exposed. 66-way selection optimism. Any later choice is exploratory, not confirmation. Three seeds only.
- Two assumptions come from code and config reading, not from real records: that the P15 control `fit.json` report carries `device` (P20 config cites "mps (fit.json report, seed 0)"), and that the P15 states carry `smoke`. If either is wrong, the run fails closed rather than passing silently.
- The P20 identity has no device field. Its device check relies on per-seed `fit.json`.
- Pins assume P20–P22 run from the pinned sibling commits. If a sibling runner changes, the run fails closed and the config must be re-pinned.
