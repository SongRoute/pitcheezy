# Disposition — same-model review of EMA wiring b79888f (review commit 7916939, cherry-picked as e652ead)

Status: `same_model_reviewed; Codex_review_pending`. DRAFT config, execution disabled. Review verdict: non-blocking for the 20-task CUDA smoke.

- R1 (F6 checker had no code): ADOPTED as written — `scripts/ema_trajectory_check.py` + `tests/test_cuda_ema_wiring_review.py` cherry-picked. Must be run on the smoke output and its JSON recorded before any registration.
- R2 (scorer wording): FIXED — `score_cuda_direct.py` `seed_block.note` and the Korean table line now say the trajectory equality is CPU-tested with CUDA checked by `ema_trajectory_check`, and that epochs/updates can differ. Same wording fix in the `direct_ema_registry.py` docstring (text only).
- R3 (config pins untested): ADOPTED — reviewer test kills C1–C3.
- R4 (smoke ↔ full identity only via source map): carried to the registration step (registration test: REGISTERED minus {status, decision, execution, queue, registration} == DRAFT minus same; source map equal).
- R5 (queue date 2026-10-04, no new fit after 19:00 KST): smoke runs before 19:00 today; a later full run needs a new dated REGISTERED config (new digest) — at registration.
- R6 (two redundant guards): no change, kept as defence in depth.
- R7 (build/load by arm name): no change, same as TabM / embed-residual; `config_check` precedes every path.

Check after changes: `cd experiments/pitchmdp && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m pytest tests/test_cuda_*.py tests/test_direct_*.py -q -p no:cacheprovider` → see commit message.
