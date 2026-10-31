You are an independent, adversarial reviewer (ASTRA-REVIEW style). Review ONLY; do not modify files, do not run anything that writes outside /tmp.

Repository root: this directory (git worktree, branch svc/ui-1026-main, HEAD 32e79c2f). Base: origin/main (61be7a86).
Scope = `git diff origin/main...HEAD` restricted to apps/observer/** and scripts/publisher_replay.py. Three changes to judge:
1. ops/ws_alert.py: new critical condition `ops_armb_off` (publisher running with publisher.armb=false). Test in ops/tests/test_ws_alert.py.
2. web/src/service: merge of ui/notice-hide-1021 (service notice hides received live content, restores on next good answer, never re-exposes a pre-notice recommendation) onto UI-1026 (header .sv-hd-1/.sv-hd-status, .sv-reclist/.sv-rec-cell row, folded at-bat list). Smoke: web/tests/notice-hide-smoke.mjs (passes).
3. web/src/service port of DEMO-WS-GAPS-1028 (asOfIndex record/mix sheets, pasAsOf, settledCorrection/heldReceipt, actualHeld, weKnown) onto UI-1026 (commit 32e79c2f). Known not ported: series words in header, demo PitchScatter.
Context: this build is the exhibition (2026-11-24) replay of a finished game (static v1 files, final=true, next=null) plus a local live-replay for one monitor. Fixed-model recommendations; no new data.

Deliver a markdown report with: (a) P1 = would break the exhibition or show wrong information to a fan; P2 = incorrect but not visible in the exhibition path; P3 = nits. For each: file:line, why, a concrete repro (command or steps), and the minimal fix. (b) Anything in the diff that silently changes behavior of the WS operations line (ws_operate/ws_alert) beyond the new rule. (c) Checks you ran (commands + results). (d) A verdict: PASS / PASS_WITH_NOTES / BLOCK with one sentence. Be concrete; quote code. Do not pad.
