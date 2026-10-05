# EXP-P37-001 launch record (2026-10-05 KST, review-execute-001)

Exploratory execution only. No verdict, promotion or policy claim. Distinct Opus 5.5 reviewer/executor
(`pm-jobs/20261005-p37-delivery/review-execute-001`); author `author-001` (`2265f44`). Go: Song 2026-10-05 GPU rule
(GPU3 UUID c334d905 only, max 1 fit; GPU4 released) + Root PM dispatch `request-review-execute-001.md`.

## Steps (README-p37-execution-packet.md)
1. Review of `git diff 6b66436 2265f44`: accept with small fixes → `9a08aef` (new AST test for the 15/15 and
   22:00-score branches, prereg m5 parenthetical, `queue.stop_rule` erratum). 77 passed (wiring 53 + launch pack 24).
   Mutants each fail: second `--gpu`, per-queue lock, ratio 2.0, UUID check off, launch_by off, zero epochs,
   projection off, score deadline off, 15/15 check off.
2. Preflight 14:43:08 (server): GPU3 2 MiB 0 %, no compute app; no queue/probe/scorer process; all P37 paths absent;
   `queue-p36-001` `{done: 20}`; 530 G free.
3. `git archive 9a08aef configs experiments/pitchmdp` (tar sha 106f8d1d…aa97) → new `code-9a08aef`, `chmod -R a-w`;
   server sha256 of DRAFT (75265fc7…), runner, queue, scorer, registry, gate, template = commit.
4. Smoke 14:43:24, queue pid 23175, `--smoke`, one `--gpu` GPU3, common lock → `{done: 15}`, family 130.0 s.
   GPU3 10 × 1 s sample during smoke: 0–6 % SM, ≤ 432 MiB.
5. Gate 14:53 `P37_SMOKE_GATE PASS 72e3a6cb…3f08`: train s/epoch median control 0.632, periodic 0.677, linear 0.580;
   linear/periodic **0.858** (≤ 1.5); fit-wall overhead ~1.5–1.7 s; UUIDs {GPU3}; errors []. TRAIN 8,192 × 2 epochs:
   not a cap-60 measurement.
6. Registration 14:53 KST (real clock): `configs/EXP-P37-001.json` sha **77574125…9a5a** (only status/decision/
   execution.enabled/registration differ); prereg sha 455bc1ad…623b; `run-p37-pipeline.py` sha **c13380f7…63c1**
   (TEMPLATE + 5 placeholder lines). Commit 848358a. scp as new files, server sha256 equal, `a-w`.
7. Launch 14:53:50 KST: `nohup setsid timeout --kill-after=60 26160 python run-p37-pipeline.py` (26,160 = 22:00 −
   14:54 + 600). Pipeline pid 27774, queue pid 27849, first fit control-39 pgid 27851 on GPU3 05:53:53Z; GPU3 lock
   owner 27849. GPU3 sample 14:54:15–24: 8–15 % SM, 1,786 MiB.

## Not yet known
Fit times at cap 60, completion time and scores: 미측정 until `score-p37-state.json` = `complete`.
