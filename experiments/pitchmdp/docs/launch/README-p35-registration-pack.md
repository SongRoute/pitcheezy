# P35 registration pack — PREPARED, NOT REGISTERED, NOT LAUNCHED (T75, 2026-10-04 ~20:05 KST)

Why: the P35 timing probe (3b317f2, result dc6b94b) passed all gates. The DRAFT `decision` field requires an explicit
go from Song before registration (new-experiment rule), requested in the coordinator worktree
`reports/research-p35/DECISION-REQUEST-p35-go-ko.md`. This pack lets a go turn into a launch without new pipeline code (launch day adds only the generated config and its registered-config test).

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
1. Deadlines follow the FIXED rule in `make_p35_config.py` (review MINOR-1, ed02276; from the budget, never from P34's
   0.43 ratio): launch L, `no_new_fit_after ≥ L+5 h`, `kill_own_after ≥ no_new+1 h`, `aggregate_by ≥ kill+30 min`, one
   KST date, all before the coordinator deadline. Run `make_p35_config.py <summary> configs/EXP-P35-001.json <date>
   <L> <no_new> <kill> <agg> <cycle> <go reference>` (refuses an existing file, malformed/past date, non-HH:MM,
   rule violations, empty go reference). Then run the extractor `config_diff` on the final file (0 paths outside
   `REGISTRATION_PATHS`) and commit config + a registered-config test that also asserts queue timeout/budget/
   attempts/start_after/stop_rule, `protocol`, `seed_rule`, `source_configs` equal the DRAFT (P34 pattern
   `test_cuda_tabm_residual_registered.py`, review NOTE-2) and `registration.go` is non-empty.
2. Fill `configsha`, rename to `run-p35-pipeline.py`, commit; archive config to the server read-only.
3. Preflight (GPU3/4 idle, no queue/probe) in a SEPARATE ssh call, then `nohup setsid timeout --kill-after=60 <budget>
   python run-p35-pipeline.py` with a Mac-side launch record only (no server launch json; never p33-launch.json; no
   remote wait loop). `<budget>` ≥ (aggregate_by − L) + 900 s scorer + 600 s and ≤ the pipeline's 40,000 s queue wait
   (review MINOR-2). Plain `python`, never `-O`/`PYTHONOPTIMIZE` (all checks are asserts, NOTE-1). The launching
   command string must not contain `cuda_direct_queue.py` or `timing_probe_`, and no tail/less on probe logs may be
   open: newsrv procps `pgrep -f` does not exclude ancestors and sees every user's processes (fail-closed false
   block only, MINOR-3).

Label: author-checked; distinct Opus review ed02276 (same-model, 0 blocker/major) dispositioned — see `DISPOSITION-p35-register-pack-review-ed02276.md`. Codex_review_pending.
