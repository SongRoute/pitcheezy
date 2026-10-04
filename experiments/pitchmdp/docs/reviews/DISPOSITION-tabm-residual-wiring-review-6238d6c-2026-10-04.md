# Disposition — distinct Opus review of TabM-residual wiring + Amendment 1 (`6238d6c`)

2026-10-04 14:1x KST, v3 coordinator cycle-007 (T35b). Label: wiring + Amendment 1 now
`same_model_reviewed; Codex_review_pending` (reviewer was another Opus 5.5 instance, not an
independent model). No server, GPU or fit in this step.

Review: `docs/reviews/OPUS-review-tabm-residual-wiring-6238d6c-2026-10-04.md` (child job
`continuous-children/review-tabm-residual-wiring-6238d6c`, helper 33778, exit 0, no permission
denials, model claude-opus-5-5, 520 s, final commit `93ebe44`).

## Findings and disposition

| id | finding | disposition |
|---|---|---|
| B1 | `epoch_record` accepted reports the frozen loop cannot produce (`run − best > patience`) | **Accepted.** Checked against `matrix_models.py` l.143–170: `stale` resets on improvement, the loop breaks at `stale >= patience`, so `run − best == stale ≤ patience` and `== patience` before the cap. Fix `4fdeafc` (cherry-pick of `da3be8c`). |
| N1 | scorer `seed_block.note` omitted `backbone_bundle_d10` | **Accepted.** `4a77b03` (cherry-pick of `4a712f1`). |
| tests | `test_cuda_tabm_residual_wiring_review.py` (epoch_record vs real loop of all 5 arm classes, projection fields, 1,224 constant) | **Accepted.** `d1a5b2d` (cherry-pick of `e600108`). |
| N2 | projection formula bias runs both ways (smoke warm-up inflates train term; non-train term underestimated) | **Accepted as a reporting rule, formula unchanged** (pre-registered). The smoke projection report will also print a measured `seconds_total − report.seconds` from a completed full P31/P32 task, labelled cross-check only. |
| N3 | WIRING record still says 4 arms with no banner | **Accepted.** Banner added in this commit. |
| N4 | one scorer guard cannot fire | Kept (defensive, harmless). |
| N5 | DRAFT `queue.date` 2026-10-04 / no new fit after 19:00 KST also blocks the smoke | **Accepted operationally.** The DRAFT config is NOT re-dated (sha `5e3ce3ad…` stays). The 25-task smoke is started and finished before 19:00 KST today, or not at all today. A later full run gets a NEW registered config with its own realistic deadline (never an active config edit). |
| N6 | cap records checked after export label read, before any prediction read | Kept (matches docstring and `train_subset` pattern). |

## Checks (this disposition, Mac protected mode, `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, `perl alarm 900`)

- `pytest tests/test_cuda_tabm_residual.py tests/test_cuda_tabm_residual_wiring_review.py` → 100 passed (5.83 s).
- 16-file neighbour set (reviewer's list) → 337 passed (9.88 s). Reviewer recorded 336 before adding the N1 test; +1 is that test.
- Own mutation: B1 fix reverted in `scripts/run_cuda_direct.py` → `test_cuda_tabm_residual_wiring_review.py` 3 failed / 8 passed; restored with `git checkout`, `git status` clean.

## Identities after disposition

- runner `scripts/run_cuda_direct.py` sha256 `d8d1fab69206a1e5…` (was `051b134b…`), scorer `scripts/score_cuda_direct.py` `3363e959267024a5…` (was `977b76a4…`) — equal to the reviewer's values.
- module `6a9daff6…87cf`, registry `4fe206ee…`, factory `e1d6aba8…`, config `5e3ce3ad…`, design note `31e2aa01…` unchanged. The `DISPOSITION-tabm-residual-wiring-289fd68` "File identities" section is stale for the runner and scorer only; this table supersedes it.

## Next

CUDA smoke of all 25 tasks on GPU3/4 only from a fresh immutable archive of this commit (same
pattern as the EMA smoke, foreground single status reads, no remote wait loop, no
`p33-launch.json`), then the projection report and a `결정 필요` for the full run.
