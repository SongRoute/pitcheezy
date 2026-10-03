# PM predictor search (DRAFT) — weight × temperature, 77 combos — 2026-10-03

Author: Claude Opus 5.5 (delegated by Codex PM). Base commit `0abda69`, branch `codex/predictor-search-1006`.
Status: **implemented and synthetic-tested; not registered and not run on real data.** The PM assigns the ID, registers the config and runs it.

## What it is
- Exploratory search on repeatedly exposed 2025 DEV. This is **not** a fresh confirmation. No candidate is promoted, no p-values are computed, and no confidence claims are made after the 77-way search.
- G0 stays the service/policy model, G15 mix stays the research baseline, and the D131/D155 verdicts are unchanged. Policy revalidation (D155) stays closed.
- No fitting and no inference. The search reuses the sealed, May-calibrated ≤2025 probabilities:
  - direct head: EXP-P15-001 (seeds 0–2) and EXP-P16-001 (seeds 3–4)
  - G0-global: EXP-P4-001 (seeds 0–2), EXP-P10-001 (seeds 3–4), and EXP-P11-001 (whole MLB, seeds 0–4)
  - frequency baseline: EXP-P4-001 / EXP-P11-001 `baseline_predictions.npz` (same sha256)

## Pinned before any candidate score was read
- Config: `configs/DRAFT-PREDSEARCH-WT77-20261003.json` (`status: DRAFT_UNREGISTERED`, `experiment_id: null`, `execution.enabled: false`).
- Grid: direct weight {0, .1, …, 1} × temperature {.8, .9, .95, 1, 1.05, 1.1, 1.2} = 77 combos. Seeds 0–4.
- Stage (the exact EXP-P15-002 / EXP-P16-001 stage):
  1. Per seed s, take `w·direct_s + (1−w)·G0_s` on the May-calibrated blend and dev splits (`run_ml_direct_head.mixed`).
  2. Apply scalar temperature `p^(1/T)` and renormalize. This equals dividing the log-probabilities by T and applying softmax. T = 1 is the identity.
  3. Average the five seeds and apply the June frequency blend. The blend weight is re-selected on the June Cpanel blend rows only (`score_ml_matrix.summarize_cell` → `fit_blend`, log_loss). Per-seed rows use per-seed weights.
  4. Whole MLB: apply the same tempered members to `mlb_dev`, blended with the frozen frequency baseline using the candidate's own Cpanel June weight. This is the EXP-P16-001 co-primary construction.
- The temperature is a second fixed scalar on top of the per-seed May temperature. It is never fitted. DEV is never used to fit or select anything inside a candidate.
- **Reproduction gate (runs first):** (w, T) = (0, 1), (1, 1) and (.5, 1) must reproduce `results/EXP-P16-001.json` (sha256 pinned) within 1e-9 on 21 checks:
  - Cpanel after the June blend: log_loss, Brier and June weight for each of the three arms; seed deltas vs G0 for direct and mix.
  - Whole MLB blended: candidate and control log_loss and Brier, plus seed deltas.
  - If any check fails, only the gate is written and the 77 candidates are never scored.
- Populations: Cpanel DEV (12,334 rows, 328 games) and whole-MLB DEV (311,721 rows, 1,161 games). These counts come from the EXP-P16-001 results header. Within each population all model arms share the same key/label arrays; the two populations have different row counts and are never interchanged.
- Summaries (descriptive):
  - per candidate and population: NLL, Brier, ECE10
  - deltas vs G0 and vs G15, plus per-seed deltas vs the same-seed G0 and G15 and the number of seeds better than each
  - June weights
  - per calendar month (`game_date` from the pinned ≤2025 raw files, columns `game_pk`/`game_date` only)
- Ranking (descriptive): the rank of ΔNLL vs G15 in each population, the mean of the two ranks, Brier ranks, and the four-objective Pareto set (both NLL, both Brier). No winner is declared.
- No bootstrap is run for the 77 candidates.

## Safety / provenance in the runner
- File: `experiments/pitchmdp/scripts/search_predictor_blend_temperature.py`.
- `run` refuses to start unless all of these hold: `status == REGISTERED`, `execution.enabled == true`, `experiment_id` matches `EXP-P\d+-\d{3}`, and the output is named after the ID.
- It refuses an existing output and a non-ML-MATRIX location (`check_location`). It holds the shared `.heavy.lock` (non-blocking).
- Every member `prediction_state.json` sha256 is pinned. Each npz is then verified against its state's `artifact_hashes` (reused `member()`). The frequency npz sha256 and the parent preparation sha256 are also checked.
- Missing seeds stop the run. So does any key/label mismatch (`assert_aligned` plus whole-MLB key checks); rows are never realigned or dropped. Period labels must cover every DEV row. Unknown games and 2026 dates stop the run, and 2026 source files are refused.
- Whole MLB is scored in deterministic row chunks (`chunk_rows` 65,536), one pass for all combos. ECE10 is accumulated from the same ten bins as `prediction_metrics`.
- `max_wall_seconds` (1,800) is checked between chunks; if exceeded, nothing is written.
- Output: `results.json` records git commit, runner and config sha256, input pins, Python/numpy versions, timings and peak RSS. `results/<ID>.json` is copied only if it is absent.
- `smoke` runs the same code on synthetic arrays only. It writes to a path ending `-smoke` and builds its reference with the independent `run_ml_direct_head_confirm.score_arrays` (bootstrap stubbed out).

## Verification done (synthetic only)
- `pytest experiments/pitchmdp/tests/test_search_predictor_blend_temperature.py`: 8 passed. The tests cover:
  - temperature identity, normalization, direction, and equivalence to logits/T
  - grid/seed pins, and refusal of an unregistered full run
  - all 21 gate checks match the independent P16 scorer on synthetic data, and a 1e-6 perturbation withholds candidates
  - chunked streaming equals the dense result and matches `prediction_metrics` log_loss and ECE10
  - missing seed, shuffled rows and short period labels are refused
  - the Pareto set is non-dominated and the output contains no p-values
  - period mapping, unknown game, and refusal of a 2026 file
- CLI smoke on synthetic data: gate passed, 77 candidates, 2.4 s.
- Synthetic run at real size (4,821 / 12,334 / 311,721 rows, 5 seeds): search 19.8 s, total 22.4 s, max RSS 1.59 GB, including the synthetic reference build.

## Runtime / memory estimate for the real run
- The scoring part should be close to the 20 s synthetic measurement at the same shapes.
- Extra steps not measured:
  - load 20 member npz plus 1 frequency npz from T7, about 330 MB of float64 arrays
  - hash the three raw parquet files (≈303 MB) and read two columns from each
- Expected total: a few minutes. Expected memory: below 2 GB (an estimate, not a measurement).
- The real run uses the CPU only (no torch model), on the Mac mini.

## Blockers / open points for the PM
- **ID and registration:** the PM assigns the ID, sets `status`, `experiment_id` and `execution.enabled`, and runs the command in `execution.prefix` with `run --output <runs_root>/<ID>`.
- **DEV seasons are not checked yet:** I have not confirmed that every DEV `game_pk` is 2025. The runner checks this at run time and stops on any 2026 date or unknown game.
- **The Cpanel gate is partly self-referential:** Cpanel log_loss, Brier and June weight come from the same `summarize_cell` that EXP-P16-001 used. The seed-delta and whole-MLB checks do recompute independently: the losses are summed per chunk and compared with P16's stored values.
- **Frozen G15 mix vs. mix-plus-temperature:** T applies after the mix. The frozen G15 at T = 1 is reproduced exactly, but a T ≠ 1 row at w = .5 is a new candidate, not G15.
- **References:** Guo et al. 2017 (scalar temperature; here a fixed grid, not fitted) — https://proceedings.mlr.press/v70/guo17a.html. Lakshminarayanan et al. 2017 is inspiration only: the arithmetic blend of two heterogeneous heads is not a deep-ensemble reproduction — https://proceedings.neurips.cc/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html.

## PM registration / review

2026-10-03: registered EXP-P19-001 under D165, all77 candidates unchanged before scoring, 180-second external time limit. Codex reviewed source, reran 8 tests, added imported source hashes/local-config hash to provenance and nonzero exit on failed reproduction gate. Rank is descriptive only, no selection/promotion decision. Original P16/G15 files remain untouched.
