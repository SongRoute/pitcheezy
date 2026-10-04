# Disposition — same_model_review of robust-clip module 5f1864b (T100b → T102)

- 2026-10-04 23:0x KST, coordinator cycle-023 (T102). Author = coordinator worker (Opus 5.5); reviewer = separate Opus 5.5
  instance (**same_model_review**, not independent, Codex_review_pending).
- Review job: `pm-jobs/20261004-end-to-end/continuous-children/review-robust-clip-5f1864b/` — exit 0, `permission_denials: []`,
  wall 449.9 s, request sha 4ab185a6…f257f8, base 5f1864b, final 039641f, git clean.
- Review doc `OPUS-review-robust-clip-5f1864b-2026-10-04.md` (review commit e2977af → here 36ce8d1).

## Decisions

- M1 (forward-uses-transform untested): ACCEPTED — fix 039641f → here 77cc453.
- M2 (one-factor fit only tested at init): ACCEPTED — identity-transform fit test reproduces `DirectResidualModel.fit`.
- m1 (column-set deviation from RealMLP-TD not disclosed): ACCEPTED (docstring). The prereg DRAFT 8311fa4 §5 already
  fixes `context_continuous` = the 41-column periodic list; the raw-column deviation must be carried into the prereg
  text when it is reviewed (open item for the wiring/prereg review child).
- m2, m3 (logits guard, load hash/architecture checks untested): ACCEPTED.
- m4 (double checkpoint write): ACCEPTED — `_base_report` hook. Read check: frozen
  `DirectResidualModel.fit` calls `self._base_report(...)` (direct_residual_model.py:137) before its only `save`
  (:142) and returns `self` (:143), so the override yields one save with the `rc_*` fields.
- n1–n7: recorded, no change. n4 (extra guarded host copy) and n5 (`index_copy` on CUDA unrun) go to the CUDA smoke list.

## Checks on `codex/robust-clip-prep-1004` @ 77cc453 (module/test bytes = 039641f, `git diff` empty)

- `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_direct_robust_clip_model.py tests/test_direct_ple_model.py`
  → 41 passed (14 + 27), exit 0.
- Reviewer harness copied to `continuous-children/t102-robust-clip-disposition/` with ROOT retargeted to this worktree:
  `mut.py` 24/24 caught, `mut_author.py` 5/5 caught (outputs `mut.out.txt`, `mut_author.out.txt`); tree clean after.
- Module sha256 a3d6cbce…bcf52fa (git blob ddccc42).

## Status

author-checked + same_model_reviewed + dispositioned; Codex_review_pending. NOT selected, NOT registered, no GPU.
The T101b wiring branch was cut from 8311fa4 (pre-fix module) → must be merged onto 77cc453 and its module pins
re-checked before its review.
