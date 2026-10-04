# P36 robust-clip timing probe — result (code-07ab288, 2026-10-05). Not a registration.

Label: timing measurement + projection only. No model of the family was fit, saved, calibrated or scored; only TRAIN
was loaded (other split files byte-hashed for integrity). Pre-registration: `AMENDMENT-P36-TIMING-PROBE-20261004.md`
(review 3b305ad, disposition 07ab288). Run by the delegated v5 PM (T119, cycle-040). `same_model_reviewed (probe code);
Codex_review_pending`.

## Precondition (PREREG P36 §7)
P35 `score-p35-state.json` complete 01:10:07 KST, 25/25 rc 0, descriptive report written (coordinator worktree
`codex/continuous-pm-1004` @ 3218d78, `reports/research-p35/result-20261005/README-ko.md`), no shared
backbone/export contract problem observed → §7 met. The P35 score magnitudes did not change this probe or the P36 design.

## Run (attempt 1 of ≤ 2; complete → final)
- Code: server `code-07ab288` (staged T110b: archive 58ee430a…bf2b8, dr-xr-xr-x, 627 files). Probe preflight
  `source_sha256` for 31 files == `git show 07ab288:experiments/pitchmdp/<f>` sha256 (31 match, 0 mismatch; checked on
  the Mac after the run). module_sha256 a3d6cbce…52fa, config 644239f3…470a, export manifest 49097109…f578.
- Separate preflight 01:14:00 KST: GPU3/4 2 MiB / 0 %, no compute app on 3/4, no queue/probe/score process, output absent.
- Command: exactly the T110b README command (`timeout --kill-after=30 1800`, env OMP/MKL/OPENBLAS 2, output
  `runs/DRAFT-ROBUST-CLIP-probe-07ab288-timing`, common lock dir, GPU3/4 UUIDs), foreground on the server, one ssh
  held by a Mac background shell (no server launch record, no wait loop).
- 01:14:10 → 01:15:52 KST; ssh exit 0; `TIMING_PROBE_DONE complete True`; worker exits 0/0.
- Host: nproc 64; load 10.70/11.47/11.05 before, 12.31/11.85/11.24 after (other users); GPU0–2 used by other users
  (not touched).
- Evidence (a-w): `/Users/song/Projects/pitcheezy-pm-jobs/20261004-end-to-end/p36-probe-07ab288-pull/`: summary.json
  0edb45bf299ef442f0b0679a6b0612ea55e771235a03af68eba57911f0fca837, preflight, two worker json/log, wrapper log; `shasum -c` of the server sha256sum list: 6/6 OK.

## Measured per epoch and projection (60-epoch cap, slower GPU)
| arm | s/epoch GPU3 / GPU4 | task projected GPU3 / GPU4 | slowest |
|---|---|---|---|
| residual_counts | 13.97 / 13.53 | 872.5 / 845.6 | 872.5 s |
| robust_clip_residual_counts | 15.17 / 15.86 | 947.5 / 987.9 | 987.9 s |
| control (fixed P35 constant) | – | – | 464 s |
| periodic_residual_counts (fixed P35 constant) | – | – | 1046 s |

robust-clip compute_stats 2.53 / 2.55 s measured (≈76 s inside the projection, not credited). Family projection
16,851.9 worker-s (4 arms × 5 seeds × slowest).

## Gates (amendment; all three)
- every_arm_within (≤ 1800 s): **true**; family_within (≤ 36000): **true**; probe_exit_zero: **true**; marginal arms: none.
- Next (t112 §4 cycle B): smoke 20 from code-07ab288 → checker exit 0 → registration + launch (L ≤ 16:30 KST).
