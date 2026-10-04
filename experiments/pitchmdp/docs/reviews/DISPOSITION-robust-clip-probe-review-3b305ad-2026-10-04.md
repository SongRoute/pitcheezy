# Disposition — same-model review of the P36 robust-clip timing probe (3b305ad), T109

PM worker (Opus 5.5), 2026-10-04 23:47–23:50 KST. Review `OPUS-review-robust-clip-probe-3b305ad-2026-10-04.md`
(same_model_review; Codex_review_pending; child helper 26124, exit 0, denials [], returned 23:46:35 KST). Labels after
this disposition: probe script `same_model_reviewed; Codex_review_pending`. Nothing run on a GPU, nothing registered.

| # | review | disposition |
|---|---|---|
| X1 | pack requires gate key `robust_clip_arms_within` the probe never writes | ACCEPTED as a pack fix (PM had found the same at 23:37 and put it in the pack review request as must-fix). Probe names are the prereg §4 (ii) names; probe unchanged. |
| m1 | mutant R7 survived (rule-count check) | ACCEPTED: test `da750d1` cherry-picked as `52db71d`; review doc as `fd0ea2a`. |
| m2 | output dir created before pin/name checks (inherited) | ACCEPTED as amendment text (refusal before `preflight.json` is not an attempt; empty dir kept; new name). Script blob kept identical to the reviewed `36d84ab` script. |
| m3 | memory per arm only in worker JSON | ACCEPTED as note: the result record reads `worker-*.json` (as P35). |
| m4 | this probe's frozen-window test breaks once the pack / P35-window fix (`1085755`) land on this lineage | ACCEPTED as sequencing: merge the pack and `1085755` only in the commit that records the P36 probe result, re-scoping this test's window end to the archived probe commit (T70c precedent) in that same commit. |
| n1–n6 | notes | Accepted as notes; n3 (464/1,046 measured at host load ≈ 15, 1.72× headroom) recorded for the result doc. |

Run order unchanged: only after EXP-P35-001 is terminal (25/25 + scorer complete), GPU3/4 idle, common locks free; one
foreground run with the review's §7 command (bound 1,830 s); at most 2 attempts.
