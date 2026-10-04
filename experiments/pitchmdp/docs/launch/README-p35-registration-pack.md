# P35 registration pack — PREPARED, NOT REGISTERED, NOT LAUNCHED (T75, 2026-10-04 ~20:05 KST)

Why: the P35 timing probe (3b317f2, result dc6b94b) passed all gates. The DRAFT `decision` field requires an explicit
go from Song before registration (new-experiment rule), requested in the coordinator worktree
`reports/research-p35/DECISION-REQUEST-p35-go-ko.md`. This pack lets a go turn into a launch without new code.

Files:
- `make_p35_config.py` — builds `configs/EXP-P35-001.json` from DRAFT f0ae4e91… + probe summary 311f0d4f…; changes
  only the pre-score extractor's `REGISTRATION_PATHS`. Dry run (date 2026-10-05, 15:00/16:00/16:30, cycle "dry-run")
  → sha 03361007…2dd45; extractor `config_diff` vs DRAFT: 10 paths, 0 outside REGISTRATION_PATHS; negative control
  `training.epochs` 61 → caught. Dry output is NOT committed (dates must be set on launch day).
- `run-p35-pipeline.TEMPLATE.py` — derived mechanically from the P34 pipeline (`codex-p34-register-1004`
  `docs/launch/run-p34-pipeline.py`): code-8864389 (smoke-verified; `git diff 8864389 3b317f2` over pitchmdp/,
  run_cuda_direct, cuda_direct_queue, score_cuda_direct, configs is empty), probe gates incl. `ple_arms_within`,
  terminal P30-001/P31/P32/P33/P34/P30-002, no P35 output, no queue/probe process, 25/25 smoke identities, scorer into a
  NEW `scores/EXP-P35-001`. `configsha='FILL_AT_REGISTRATION'` fails closed (checked: AssertionError, exit 1).

Steps after go (not before):
1. Choose launch-day deadlines from the probe (family 25,172 worker-s / 2 GPUs ≈ 3.5 h at the cap; P34 actual was
   0.43 × projection) with margin before the coordinator deadline; run `make_p35_config.py`; commit config + a
   registered-config test; refresh the extractor only if needed BEFORE any score.
2. Fill `configsha`, rename to `run-p35-pipeline.py`, commit; archive config to the server read-only.
3. Preflight (GPU3/4 idle, no queue/probe), then `nohup setsid timeout --kill-after=60 <budget> python run-p35-pipeline.py`
   with a Mac-side launch record only (no server launch json; never p33-launch.json; no remote wait loop).

Label: author-checked; distinct Opus review requested (same-model). Codex_review_pending.
