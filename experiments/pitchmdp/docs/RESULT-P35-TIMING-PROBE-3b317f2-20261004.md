# P35 PLE timing probe — result (3b317f2, 2026-10-04). Not a registration.

Label: timing measurement + projection only. No model of the family was fit, saved, calibrated or scored; only
TRAIN was loaded (other split files byte-hashed for integrity). Pre-registration:
`AMENDMENT-P35-TIMING-PROBE-20261004.md` as amended pre-run in `3b317f2` (review `7569d2f`, disposition
`docs/reviews/DISPOSITION-p35-timing-probe-review-14072ac-2026-10-04.md`). `same_model_reviewed; Codex_review_pending`.

## Run (attempt 1 of ≤ 2; complete → final, amendment F3)

- Code: `git archive 3b317f2 configs experiments/pitchmdp | gzip -n` sha
  4dc75f80bf2a57a58867a9324a7c5bb455cba141dd875ba44fb47f4591006148 → newsrv `cuda-ple-probe-3b317f2.tar.gz` (same sha,
  a-w) → NEW read-only `code-3b317f2`. On the server: config `DRAFT-CUDA-PLE-RESIDUAL-20261004.json` f0ae4e91…b9b2
  (unedited), probe script 0ac1a098…df017 (= reviewed blob; `git diff 14072ac 3b317f2 -- scripts pitchmdp configs`
  empty), export manifest 49097109…f578.
- Preflight 19:53:12 KST: GPU3/4 2 MiB / 0 %, no compute app on 3/4, no own queue/probe/score process, output absent.
- Command: exactly the amendment command with `<R>` = 3b317f2, foreground under `timeout --kill-after=30 1800`, env
  OMP/MKL/OPENBLAS 2, output `runs/DRAFT-PLE-RESIDUAL-probe-3b317f2-timing`, common lock dir, GPU3/4 UUIDs. Run from a
  Mac background shell holding the ssh session (no server launch record, no wait loop).
- 19:53:43 → 19:58:14 KST, probe exit 0, ssh exit 0, `complete: true`, worker exits 0/0 (GPU3 266.4 s, GPU4 267.3 s).
- Host: nproc 64; load 13.90/14.48/15.25 before, 15.20/15.38/15.48 after (other users' work); GPU0–2 used by other
  users (not touched); GPU3/4 2 MiB before and after.
- Evidence (a-w): `/Users/song/Projects/pitcheezy-pm-jobs/20261004-end-to-end/t75-p35-probe/evidence/`
  summary.json 311f0d4f5dff318320a6da0d8944e6d7b50fdc1a1eb8ee4cd7a95d754fb67ff5, preflight 5609c759…, workers
  852201d9… (GPU3) / dca3338a… (GPU4); server `sha256sum` matched for all 6 files; wrapper log `attempt1-wrapper.log`.

## Measured per epoch (max(wall, report)/2 = wall; warm, 1,224 updates/epoch) and projection

| arm | params | s/epoch GPU3 / GPU4 | binning s inside projection, not credited (GPU3 / GPU4) | task projected (slower GPU) | peak device |
|---|---|---|---|---|---|
| control | 141,834 | 7.00 / 7.15 | – | 464 s | 1.41 GiB |
| periodic_residual_counts | 244,898 | 16.85 / 16.73 | – | 1,046 s | 1.51 GiB |
| ple_residual_counts | 275,738 | 18.50 / 18.34 | 130.3 / 100.6 | 1,149 s | 1.60 GiB |
| tabm_residual_counts | 260,760 | 18.27 / 18.59 | – | 1,150 s | 1.54 GiB |
| ple_tabm_residual_counts | 291,600 | 19.76 / 19.75 | 127.7 / 127.8 | 1,225 s | 1.60 GiB |

Family (5 seeds × slowest per arm, every fit at the 60-epoch cap) = **25,171.5 worker-s**. One TRAIN bins hash on both
GPUs and both PLE arms: e6036f4a…7ada (3,295 bins, 0 degenerate columns). PLE binning measured ≈ 4.3 s per call.

## Gates (amendment §Gates; all must hold)

- complete true, exit codes 0/0 → true
- `ple_arms_within`: 1,149 and 1,225 ≤ 1,800 → true (68 % of the gate at most)
- `every_arm_within`: max 1,225 ≤ 1,800 → true
- `family_within`: 25,171.5 ≤ 36,000 → true
- `marginal_arms`: [] (no arm within 10 % of 1,800)

The probe gate of DRAFT §6 is met. Per F1, the binning seconds above stay inside the projection (not subtracted).

## Context, not part of the gate

- P34 probe (9de203a, load ≈ 28) vs this probe (load ≈ 15), same arms: control 7.21 → 7.15, periodic_residual_counts
  15.98 → 16.85, tabm_residual_counts 15.67 → 18.59 s/epoch (slower GPU). Cause 미측정 (GPU0–2 other-user load differs).
- P34 actual vs its projection (`reports/t75-p35-probe/p34-actual-vs-projected.md` in the coordinator worktree): every
  task ≤ 0.78 × projection, family 8,226 / 19,118 worker-s (early stopping). One family; not a promise for P35.

## What this does NOT do

- No registration. Registration still needs: re-dated deadlines in a NEW `configs/EXP-P35-001.json` that differs from
  the DRAFT only at the extractor's `REGISTRATION_PATHS`, a launch pipeline, and the explicit go requested from Song
  (`reports/research-p35/DECISION-REQUEST-p35-go-ko.md`, coordinator worktree).
- No score, no model artifact, no DEV/MLB DEV row, no 2026 data.
