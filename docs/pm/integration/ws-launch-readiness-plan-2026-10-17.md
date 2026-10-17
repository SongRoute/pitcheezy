# World Series Launch Prep Plan — 2026-10-17 (Claude PM)

Goal: be able to make the public/hold call at the 10/22 readiness check. The World Series runs 10/23–31 (US dates; start times TBD). Under the collaboration rules, only breakage fixes are allowed after 10/21, and the contact channel must be set by 10/20.
Basis: branch lineage inventory and checklist investigation (10/17 06:30 KST, read-only).

## Current state in one paragraph
The public app lineage runs as a single line: 9f4b36c → 9f97a81 → 222f46d → 1f0260f → 7d5be82 → 5c9c31a. Its tip, 5c9c31a, has the publisher, the conditional upload, and the public viewer. The research lineage (93a80ef) branched off at 81fbea7 and never touched `apps/observer`, so a trial merge of the two tips conflicts only in `docs/decisions.md`. The research 7-leaf shadow has no code connecting it to the service, so we don't put it into this launch. The public app keeps the default-off `shadow_expectation` slot. Pending service fixes that are not implemented: the "확인 중" (confirming) rule relaxation, the response to lost pitch IDs, and the rule to skip rewriting unchanged files. Not started: real-device E2E, real host connection, and a live-game rehearsal on the publisher path.

## Workstreams (A–D use separate files, so they can run in parallel)

- **A. Integration branch** (prerequisite, mechanical)
  - Create the branch `ws/launch-2026` at 5c9c31a → merge 93a80ef (keep both sides in `docs/decisions.md`).
  - Run the backend synthetic tests, static checks, and public build. Done means: everything passes, or we have a list of failures and their causes. No push.
- **B. Relax "확인 중" + handle lost pitch IDs** (`display_state.py`, `live_store.py`, `service_api.py`)
  - Content (approved by Song 10/12): keep showing already-accepted pitches; apply the game-level gate only when recommendation inputs change; skip it at 0 pitches.
  - Also: do not mark a re-acceptance with identical content as a correction (on 849831, all 257 pitches came through as `corrected=true`).
  - Keep the immutable first-receipt and append-only rules.
  - Done means: registration → implementation → independent review → before/after comparison on the 849831 replay (number of unconfirmed cycles, gated cycles, and exposure of never-accepted content = 0).
- **C. Stop the publisher rewriting unchanged files** (`publisher.py` `_game`/`_put_locked`)
  - Skip writing an in-progress game file when its digest is unchanged. Put it behind a config switch, on by default.
  - Done means: replay tests show the write count and file contents are unchanged; heartbeat/sequence fields keep updating through health.
- **D. Recheck the 3 layout defects plus the status-line mismatch** (`service.css` and others)
  - Confirm with Playwright at 320/375/390/1440 widths; fix only the defects that remain.
- **E. Stage-2 rehearsal (publisher path, one remaining LCS game)** — needs Song's approval
  - Local publish only (no external host). Check before/after of B and C, delay, and recovery.
  - Candidates (as of 10/13; recheck the schedule on the day): 10/18 09:00 KST TB@CLE G5*, 10/19 09:00 LAD@MIL G6*, 10/20 06:00·09:00, 10/21 09:00. *Played only if needed.
- **F. Real host connection and real-device E2E** — needs the teammate and Song
  - Private delivery channel, dedicated target and test key, bearer issue/revoke, single writer, test window, and contact channel (by 10/20).
  - Real-device check: iPhone Safari.
- **Deferred (not for launch):** connecting the research 7-leaf shadow to the service, scoring the 314 newly eligible rows, and a fair baseline.

### Change to A (10/17 08:xx)
The merge itself only conflicted in `docs/decisions.md` (cc0d174). But on the research side, D117–D127 changed `configs/ML-POLICY-MATERIALIZATION-v1.json` and `policy_runtime.py`, and after the merge the live ARM-B recommendation path (`demo_precompute.LivePolicy`) is rejected at startup by the d107 addendum chain and S2 identity checks (`test_demo_precompute` fails). Research is outside the launch scope, so `ws/launch-2026` stays based on 5c9c31a with only B, C and D merged in. The merge commit is kept on `ws/launch-research-merge-1017`. Sorting out the research merge after launch is follow-up work (a separate pinned copy for live use, or another approach).
Separately: web checks on 5c9c31a got worse than the 10/12 record (static 358/21, compact 344/245, studio 8 failures). The cause is unconfirmed (could be environment, e.g. Node v25.9.0) — this has to be checked before the readiness check.

## Order
1. Today: start A, B, C, D in parallel. Send Song's MLB permission email; settle the teammate channel.
2. By 10/18 morning: merge B and C into A, then run the 849831 replay regression → rehearsal E on the first remaining LCS game (if approved).
3. 10/19–20: F (host and real device) and contact channel confirmed → RC readiness check on 10/20.
4. 10/21–22: freeze (breakage fixes only) → public/hold decision on 10/22.

## Song decisions and actions
- Send the MLB permission email and the advisor inquiry (drafts: `docs/pm/service/mlb-permission-request-draft.md`; blanks: name, school, contact, plan if not permitted).
- Teammate: private delivery channel, test window, contact channel.
- Approve rehearsal E.
- 4 pending decisions (10/12 rc handoff):
  - Handling of the one "확인 중" pitch below the game-final body
  - Whether to lock as final when an "in progress" body arrives after the game ended
  - Whether to enable the unsupported-pitcher reference card
  - Common delay: 0s or 15s (recommended)
- Whether to push the remote `rc/ws-2026` (currently e39bb41, the version with delay on).

## Song 결정 — 2026-10-17 ("추천대로 가자")
- 다음 작업: 웹 정적 검사 기존 실패 21건 정리부터.
- B 기본값 3개 유지:
  - 경기 종료 본문 아래 확인 중 1구는 마지막 수락 본문으로 표시하고, 종료 파일도 그 상태로 잠근다.
  - 추천은 현재 타석 투구가 보류·확인 중일 때만 보류하고, 이전 타석만 문제일 때는 낸다.
  - 비어 있던 값이 나중에 채워지는 것은 정정으로 표시하지 않는다.
- 화면: 320 폭의 긴 팀명은 말줄임으로 둔다(전체 이름은 aria-label에 있음).
- 가정(Song 미확인): 폰 화면에 "다음 타자" 줄이 없는 것(10/13 개편)은 의도한 설계로 보고 그대로 둔다.
