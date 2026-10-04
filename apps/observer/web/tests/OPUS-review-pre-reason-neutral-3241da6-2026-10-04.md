# same_model_review; Codex_review_pending — neutral pre-pitch line for actual-dependent refusal reasons (3241da6)

- Reviewer: Opus 5.5 (same model, separate session from the author). No other models called, no sub-processes spawned.
- Target: `git diff c08b330 3241da6` (`apps/observer/web/src/WatchAlong.tsx`, new `tests/pre-reason-neutral-smoke.mjs`).
- Worktree: `codex-pre-reason-neutral-review-1004`. No ssh, GPU, network or port access. Headless Chromium against route mocks only.
- Date: 2026-10-04.

## Verdict

**No blockers. The change does what it says.** Before the reveal, the four reasons decided from the actual row no longer show. After a successful reveal the real reason shows. The neutral line stays through a failed reveal and in the embedded view. The smoke fails on c08b330 (negative control).

Two non-blockers matter beyond the UI:
- **N1:** the raw status codes still reach the browser in the pre-reveal JSON.
- **N2:** the "추천 없음" state itself is still a hint, whatever the text says.

Both need precompute or backend changes. They are recommended below and were not applied.

This review commit adds only test hardening (F-T1, F-T2).

## Commands and real exit codes

The shell ran with `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=UV_THREADPOOL_SIZE=1`. Every command was wrapped in `perl -e 'alarm 300; exec @ARGV'`. Output went to files, so no pipes masked an exit code.

| # | command (cwd `apps/observer/web`) | exit |
|---|---|---|
| 1 | `npx tsc -b` | 0 |
| 2 | `npx vite build --outDir dist-review --emptyOutDir` (3241da6) | 0 |
| 3 | `PZ_WEB_DIST=$PWD/dist-review node tests/pre-reason-neutral-smoke.mjs` (as committed) | 0 (`ok`, 10 checks) |
| 4 | `PZ_WEB_DIST=$PWD/dist-review node tests/watch-spoiler-smoke.mjs` | 0 (`watch spoiler smoke: ok`) |
| 5 | `git checkout c08b330 -- src/WatchAlong.tsx && npx vite build --outDir dist-review-base --emptyOutDir`, then `git checkout 3241da6 -- src/WatchAlong.tsx` (source restored, `git diff HEAD` empty) | 0 |
| 6 | negative control: smoke from #3 with `PZ_WEB_DIST=$PWD/dist-review-base` (source back at 3241da6, so Part 1 passes) | **1** (18 FAIL lines: e.g. `1440 #1: before reveal shows "자동 볼·스트라이크(투구 없음)라 …"`, same for #2–#4 and at 390) |
| 7 | extended smoke (this review, see F-T1/F-T2) on `dist-review` | 0 (`ok`) |
| 8 | extended smoke on `dist-review-base` | **1** (22 FAIL lines, including the new cases: `next pitch did not reset`, `failed reveal dropped the neutral line`, `embedded view shows the outcome reason before reveal`, at both 1440 and 390) |
| 9 | ad-hoc 390×844 screenshot of the neutral card (temp script, deleted) | 0; `scrollWidth` 390. The two-sentence line wraps to two lines with no overflow (`/tmp/pz-review-390-neutral.png`) |
| 10 | read-only count over the 13 local timelines in `/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/DEMO-WS-2026/watch/*.json` (python3, json only) | 0 |

Build output went to `dist-review/` and `dist-review-base/`. Both are untracked and were not added.

## 1. Is the actual-dependent set right and complete?

I read `STATUS_TEXT` (`scripts/demo_precompute.py:336-348`), `PolicyRuntime._evaluate` (`experiments/pitchmdp/pitchmdp/policy_runtime.py:300-364`), `pa_requests` (`experiments/pitchmdp/pitchmdp/policy_requests.py:105-153`) and the watch loop (`scripts/demo_precompute.py:681-713`).

| status → viewer text | decided from | before reveal |
|---|---|---|
| NO_LOGGED_ACTION | the row's logged label is the no-pitch sentinel (automatic ball/strike) | **actual**, hidden ✔ |
| MISSING_ACTION_LABEL | the row's logged `pitch_type` is missing on a real pitch | **actual**, hidden ✔ |
| LOGGING_POSITIVITY | the logged type has no TRAIN logging mass. Normally replaced by `service_candidate` (`:692`); the text survives only when that fallback returns None | **actual**, hidden ✔ |
| NO_DECISION | `k >= len(requests)` and problem ∈ {None, `no_decision`}. In practice this is the terminal placeholder row (`terminal_placeholder`: an event plus a no-pitch description, or no type and no description) | **actual**, hidden ✔ |
| UNKNOWN_PITCHER, PITCHER_HAND, EMPTY_SUPPORT | pitcher id, request hand, frozen TRAIN tables | pre-decision ✔ |
| INCOMPLETE_START | the first row's count | pre-decision ✔ |
| INCONSISTENT_HISTORY | the *previous* pitch's outcome against this count | a past pitch only ✔ |
| NOT_SUBMITTED | ordering, gaps, illegal count, missing identity, invalid history | pre-decision or past, with one narrow exception (below) |
| MID_PA (shown with the *root* text, `:688-690`) | the first refusal earlier in the same PA | past pitch only |

The four-item list is right and complete for the reasons the viewer can see. Notes on the edge cases:

- **MID_PA rooted in an actual-dependent refusal.** `NO_LOGGED_ACTION` and `MISSING_LABEL` are sticky (`policy_runtime.py:313`). Later pitches in that PA therefore show the *root* text, e.g. "자동 볼·스트라이크(투구 없음)라 …". That text describes an earlier pitch, which is already past: the strip auto-fetches earlier PA reveals (`WatchAlong.tsx:279-282`). This is not a spoiler, and the task asked whether it is acceptable: **yes**. Because the UI matches on text, these rows are now *over-hidden* before the reveal. That is harmless. The real data has 6 such rows (849841: 4, 849848: 2). See N3 for the after-reveal wording problem.
- **NOT_SUBMITTED on the closing row** (`policy_requests.py:151`, `closing and gap == len(records)`). This status is assigned only because the row was classified as a terminal placeholder, which is actual-dependent. The text "기록 구조 문제(누락·순서)…" names no outcome, and the case appears in 0 of 3,725 rows. Not material; noted for completeness.
- **The live path** (`demo_precompute.py:806` `pre_pitch`) evaluates with `logged_action=None` and a fresh PA state. It cannot produce NO_LOGGED_ACTION, MISSING_LABEL, LOGGING_POSITIVITY or MID_PA for the current pitch. NO_DECISION only applies to rows that already happened. `Live.tsx`/`Game.tsx` need no change.

## 2. Is text matching robust?

- **Every watch reason is a `STATUS_TEXT` value.** The `STATUS_TEXT.get(root, root)` fallback (`demo_precompute.py:709`) would only return a raw code for an unmapped status. The root is either the ledger status or the last `UNSUPPORTED_[A-Z_]+` token of the MID_PA detail, and every non-FATAL status in `STATUSES` (`policy_runtime.py:47-49`) has an entry. Only `FAILED_INTEGRITY` or `FAILED_RUNTIME` could fall through as a raw code. Those are runtime failures, not facts about the pitch, and a precompute run that hits one is not shippable. No reachable outcome leak.
- **Showcase** embeds the same `WatchAlong` through an iframe with `&embed=1` (`Showcase.tsx:129`), so the same rule applies; this is now verified by the smoke. Showcase itself reads only `pre.status`.
- **No offline bundle** exists for watch-along; `showcase-scenes.json` has no reasons.
- **Smoke regex (F-T2, fixed here).** The committed regex `/'([A-Z_]+)': '([^']+)'/g` ran over the *whole* Python file. It also captured the pitch-call map (`'B': 'ball'`, …, `demo_precompute.py:49-54`). The four values were still captured correctly, and no key collides today. A future dict with one of those keys elsewhere in the file could silently override the value, though. The parse is now scoped to the `STATUS_TEXT = {…}` block.

## 3. UI states

| state | 3241da6 | evidence |
|---|---|---|
| before reveal | neutral line for the 4 texts; the unknown-pitcher text shown as-is | smoke #3/#7, both widths |
| after reveal | the real reason | smoke #3/#7 |
| after → next pitch | resets to the neutral line (the reveal is per-index, `reveals[index]`) | smoke #3/#7 |
| ← back to a revealed pitch | real reason kept | **added** in #7 |
| Space (keyboard) reveal | works the same as the button | **added** in #7 |
| failed reveal (503) | neutral line stays, error shown, no reveal block. Space retry → real reason, error cleared | **added** in #7 |
| embedded `?embed=1` | neutral line before, real reason after Space | **added** in #7 |
| 390 layout | no horizontal overflow; two-line wrap | smoke + screenshot #9 |

On c08b330 all of these fail as expected (#8).

## Findings

### Blockers
None.

### Non-blockers (ranked)

**N1 — the raw status still reaches the browser before the reveal (payload-level spoiler).** `apps/observer/backend/observer_app/watch_along.py:18`
- `PRE_FIELDS` includes `status`, and `timeline()` ships `pre.reason` verbatim for **every** pitch of the game at page load.
- A viewer with DevTools, or any client of `/api/watch/{pk}`, therefore sees future `UNSUPPORTED_NO_LOGGED_ACTION` and `NO_DECISION` rows and their reason text.
- The UI does not use `decision.status` at all; it is only declared in the TS type.
- A casual fan will not see this, so it does not block this UI fix. Still, the correct boundary is the server, since `timeline()`'s own docstring says "everything a viewer may see before any pitch is revealed".
- Recommendation:
  - Drop `status` from `PRE_FIELDS`, or map the four actual-dependent statuses to a neutral `status`.
  - In `timeline()`, replace `pre.reason` for those rows with the neutral text.
  - Add the real reason to the `reveal()` payload (`POST_FIELDS`).
  - The UI then just renders whatever `pre.reason` it gets, and the duplicated text list disappears.
- `tests/test_watch_along.py:20` constructs `status`. Check for consumers before dropping it.

**N2 — "추천 없음" itself is the hint; the neutral line only removes the *which*.**
Counts across the 13 local games (3,725 pitches):

| rows | count |
|---|---|
| unknown pitcher (direct + MID_PA) | 861 (23%) |
| neutral line (NO_LOGGED_ACTION 3 + MID_PA-root 6 + NO_DECISION 3) | 12 (0.3%) |

- Unknown-pitcher refusals cover whole PAs or pitchers. An actual-dependent refusal shows up as a sudden "추천 없음" **mid-PA for a pitcher who had recommendations a pitch earlier**.
- In that context, the neutral line ("이유는 실제 투구를 공개하면 보여 드려요") also confirms that the reason is about *this* pitch. That narrows it to "automatic call or a PA-ending non-pitch row".
- Using the same neutral line for all "추천 없음" rows would **not** remove that contrast, because it comes from the state, not the text. It would also hide an honest pre-decision reason on 861 rows. **Not recommended.**
- **Recommended mitigation (follow-up, precompute plus data regeneration):** give NO_LOGGED_ACTION and MISSING_LABEL rows, and MID_PA rows rooted in them, the pre-pitch candidate law. Do this the same way `service_candidate` already does for LOGGING_POSITIVITY (`demo_precompute.py:350, 692`), i.e. add them to `ACTUAL_DEPENDENT`.
- This matches what the live service shows: `pre_pitch` evaluates with `logged_action=None`, so live viewers get a recommendation before an automatic ball.
- For NO_DECISION placeholder rows, build the request from the row's pre-decision fields in the same way, or keep the neutral line there (3 rows in 13 games).
- Until then, the current neutral line is an acceptable stop-gap.

**N3 — misleading after-reveal text on MID_PA rows rooted in NO_LOGGED_ACTION.** `scripts/demo_precompute.py:688-690,709`
- These rows are real pitches. After the reveal they read "자동 볼·스트라이크(투구 없음)라 추천 대상이 아닙니다." (6 rows: 849841 ×4, 849848 ×2.)
- The text describes the *earlier* pitch. Pre-existing; this change makes it more prominent because it is the "reason" revealed after the neutral line.
- Recommendation: for `status == MID_PA`, prefix the root text with "같은 타석의 앞선 공이 …" or similar. N2's fallback would remove the rows entirely.
- Product wording, so not applied.

**N4 — the text list is duplicated between Python and TS.** `WatchAlong.tsx:53-58`
- The smoke pins equality. Matching on text rather than on `decision.status`/`root_status` is a little fragile but works: today every reason is a `STATUS_TEXT` value.
- N1's server-side fix removes the duplication.

**F-T1 — fixed in this commit (test).** The smoke did not cover Space, ←, a failed reveal plus retry, or embed. It also had no guard for new `STATUS_TEXT` entries. Added all of these. A new `STATUS_TEXT` key must now be classified as pre-decision or actual-dependent, or the smoke fails.

**F-T2 — fixed in this commit (test).** The `STATUS_TEXT` parse is now scoped to the dict (see §2).

## 5. Wording

"이 공은 투구 전 추천 대상이 아니에요. 이유는 실제 투구를 공개하면 보여 드려요."
- Clear, polite Korean for a casual fan.
- Fits the card at 390 in two lines.
- It does not name an outcome. Its only hint is the structural one in N2.
- Small redundancy with the next line ("실제 투구는 아래 버튼으로 그대로 볼 수 있어요."), which is acceptable.
- No wording change applied.

## Disposition

- 3241da6: **accept** (no blockers).
- This review commit: smoke hardening only (F-T1, F-T2). Product code unchanged.
- Follow-ups for the PM / Codex queue, in recommended order:
  1. N2 — pre-pitch law fallback for the actual-dependent refusals in precompute. This also fixes N3 for those rows.
  2. N1 — move the neutralisation to `watch_along.timeline()` and drop `status` from the pre payload.
  3. N3 — wording, if N2 is not done.
- No push, merge or deploy.
