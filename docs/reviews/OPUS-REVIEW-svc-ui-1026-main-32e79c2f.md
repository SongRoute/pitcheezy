# Review: svc/ui-1026-main @ 32e79c2f (base origin/main 61be7a86)

I only read the code; I ran nothing from the project. There is no venv and no node_modules here, so I could not run tsc, pytest or the Playwright smokes. Everything below comes from reading the code, plus two small scripts I wrote: a standalone simulation of `ws_alert.process()` and a CSS cascade diff. The commands are in (c).

## (a) Findings

### P1-1: the merge put the whole pre-10/26 stylesheet back after the UI-1026 layout. It overrides the accepted look of the zone rank squares and the stage.
`apps/observer/web/src/service/service.css:1-715`

On the UI-1026 tip (64581cc), `service.css` was only 3 lines:
```css
/* Preserve cascade order: shared foundations first, current screen layout last. */
@import "./service-base.css";
@import "./service-layout.css";
```
The merge of `ui/notice-hide-1021` (651779f) kept those imports and then also kept the whole origin/main `service.css` (702 lines) below them. Line counts by commit: 64581cc and 2dab213 have 3 lines; e3e3cbe has 703; 651779f, f11de37 and dfdd40a have 706; HEAD has 715. Lines 4-715 of HEAD are origin/main's file plus `.sv-alert p` (line 27) and the DEMO-WS-GAPS block (line 707+). Rules after an `@import` come later in the cascade, so every legacy rule with the same specificity now beats `service-layout.css`. The 10/26 tail that 53aba02 removed on purpose is active again. On line 659:
```css
.sv-rec[data-rank="1"] rect{fill:#1e6a4e;stroke:#1e6a4e;stroke-dasharray:none} ...
.sv-rec[data-rank="2"] rect{fill:#cfe3d8;stroke:#cfe3d8;stroke-dasharray:none} .sv-rec[data-rank="2"] text{fill:#14503a;...}
.sv-rec[data-rank="3"] rect{fill:none;stroke:#1e6a4e;stroke-dasharray:3 2}
```
- `.sv-rec[data-rank="3"] rect` has specificity (0,2,1), the same as layout's `.sv-game .sv-rec rect` (0,2,1), and it comes later. A standalone rank-3 square therefore draws transparent with a **dashed** border. 53aba02 decided "every border solid (no dashed rank 3)".
- Rank 2 keeps layout's fill, because `.sv-game .sv-rec.r2 rect` is (0,3,1). Its stroke and text colour come from the legacy rule, though: the stroke becomes #cfe3d8, the same as a light fill, so the outline disappears.
- Rank 1's stroke becomes #1e6a4e instead of `--sv-rec` #06776f.
- `zone-geometry-check.mjs:86-90` only checks a merged `'1·3'` square and a `'2'` square (fill and dash), so it cannot catch this. The port commit did not list zone-geometry among the tests it ran anyway.

My cascade diff found other unconditional overrides of layout values:
- `.sv-game .sv-stage` gets border `1px solid`, radius 20px, padding 16px and gap 8px. Layout sets 0 / 0 / 14px 12px 4px / 10px.
- `.sv-page.sv-game` gets max-width 1600px. Layout sets none.
- `.sv-game .sv-zrow` gets height 260px. Layout uses `var(--sv-zone-height)`, which is 216px on phones. That adds 44px on every phone. At ≥760px both are 330px.

Repro: open any game whose top-3 has three separate target cells, at any width. The rank-3 square is dashed and unfilled, and the rank-2 square has no dark outline. This shows on every recommendation of the exhibition monitor.

Minimal fix:
- Restore `service.css` to the 3 lines from 64581cc.
- Move `.sv-alert p{word-break:keep-all;overflow-wrap:anywhere}` into `service-layout.css` or right after the imports. `notice-hide-smoke` asserts `wordBreak === 'keep-all'`, and `service-base.css` does not have this rule.
- The DEMO-WS-GAPS block at line 707+ can go too. Its selectors are not used (see P3-1).
- Then re-run zone-geometry-check and ws-launch-layout-check.

### P1-2: in the exhibition default view, the player sheet's 투수 tab shows two different "today" pitch counts
`GameScreen.tsx:745`
```tsx
{playerRole === 'p' && !maskPublished && <MixTable detail={detail} pitcherId={pitcherP?.id ?? null} fold={fold} heading />}
```
The same sheet has an "오늘" row built from `context = latest ? detail.current_context : pitch?.context` (`:574`), which is the as-of `context_before` of the pitch on screen. A finished game (`live === false`) defaults to its **first pitch** (`:391`), so `latest` is false. On the exhibition file, that row and the table disagree.

Data from `tests/fixtures/publish-ws/2026-10-28/v1/game/849803.json`:
- First pitch selected, 투수 tab (Griffin Jax): the row reads "오늘 0구 · 0삼진 · 0자책". Directly below it, "오늘 구종 · 74구" shows the whole-game mix.
- Pitch index 40 selected: the row reads 29구 and the table still reads 74구.
- The 구종 sheet for the same pitcher passes `asOfIndex` (`:735`) and shows 30구.
- The 박스 tab's basis line even tells the fan that the player cell's '오늘 기록' is the value at that point.

Fix: `<MixTable ... heading asOfIndex={asOfIndex} />`. A small leftover: `context_before` does not count the selected pitch, while `upTo` uses `<=` and does, so the counts still differ by 1 when the selected pitch belongs to this pitcher (see P3-3).

### P2-1: a notice followed by any non-notice error shows the pre-notice game again, and can name a pre-notice recommendation
`GameScreen.tsx:276-278, 342`

`noticeHold = error?.notice === true` is derived from the **latest** error. If the next attempt fails for another reason, `setError({..., notice: false})` ends the hold without any good answer. Examples: a network error or 15-second timeout (status 0), a 404, or an HTML fallback page during host recovery, which is parsed as status 0. The full screen then renders `detail` as received before the notice: candidates, zone marks, at-bat list, score, behind a 연결 끊김 chip. `drawn.current` was cleared by the effect at `:343`, so this render records the pre-notice recommendation as "last drawn". When the next good answer for the same slot carries a different list, the change note reads `추천이 갱신됐어요 (이전: <pre-notice pick>)`. That is exactly what the fix promised never to show.

Repro (API bundle): the host answers 503 maintenance, then one poll gets a network error, then the host answers 200 with a different top-3 for the same `pitch_key`. In the static bundle the same happens whenever `pending.current === null`.

`notice-hide-smoke` only goes notice → ok, so it cannot see this. It is not on the exhibition path, because a local static host serves no notices.

Fix: make the hold sticky. Keep it in a ref or state that is set when `notice` is true and cleared only on a successful apply (the `setDetail(...)` paths and the static "sequence unchanged" success at `:243`). Derive `noticeHold` from that instead of from `error`.

### P2-2: in static mode, a notice served only for the game file makes the old content flash, or never hides it
`GameScreen.tsx:232, 242-244`
```ts
if (present) { setLastOk(Date.now()); setError(null); }
...
if (pendingTarget.current === null && check && detailRef.current && appliedSequence.current !== null && check.file_sequence! <= appliedSequence.current) {
  setError(null); setLastOk(Date.now()); return;
}
```
- If `health.json` answers 200 while `game/<pk>.json` answers `data_hold`, every poll clears the notice before the file is read. The old header and score flash on screen until the file read throws the notice again. Recs are masked by `awaitingFile`, but the score and linescore are not.
- If no new sequence is announced, the file is never asked for, so the pre-notice game stays on screen.

This only matters if the host can serve notices per path. `data_hold` sounds like it could be per game, but I could not check, because the host is not in this repo. The smoke makes every `/v1` path answer the notice.

Fix: while the sticky hold from P2-1 is set, skip the two `setError(null)` calls and force the game-file read (treat it like a pending target).

### P2-3 (ops line): when the publisher restarts, `ops_armb_off` sends a false "복구", then stays silent up to 300 s while ARM-B is still off
`ws_alert.py:546`
```python
armb_off = ops_state == 'running' and pub.get('running', True) is not False and pub.get('armb') is False
```
In backoff, prestart, stopping and stopped states the rule is still evaluated, but as inactive, so the engine records a recovery. ws_operate writes `publisher: {'running': False, ...}` without `armb` between segments. Restart backoff is 30/60/120 s (`ws_ops_config.json:33`), shorter than `cooldown_s` 300. If the next segment again starts with ARM-B off, the re-fire is `suppressed` until 300 s after the false recovery.

I simulated this with `process()` from this file:
```
t+  0 running armb off   events=['fire']
t+100 backoff            events=['recover']      <- "[복구] 운영: 추천 정책(ARM-B)이 꺼진 채 발행 중"
t+145 new segment, off   events=['suppressed']   <- log only
t+400                    events=['fire']
```
At game end (stopping/stopped), the same rule also sends a "복구" for ARM-B.

There are no false fires: `armb` only exists in the running-publisher block. A publisher block without a `running` key does fire if `armb` is False; that is intended for status v1.0.

Fix: only emit the condition while a segment runs, so the engine closes it silently ("closed", log only) in between and a new off segment fires at once:
```python
if ops_state == 'running' and pub.get('running', True) is not False and 'armb' in pub:
    conds.append(_cond('ops_armb_off', CRITICAL, pub['armb'] is False, ...))
```
Then add a test step: running/off → backoff → running/off fires again without a recovery.

### P3 (nits)
1. Dead code:
   - `.sv-head-series` and `.sv-mix-title`, `.sv-mix-zone`, `.sv-mix-dot`, `.sv-mix-legend` (`service.css:707-714`) are not referenced by any component. The commit says the header series words and PitchScatter were deliberately not ported.
   - The new exports `U`, `ux`, `uz` in `Zone.tsx:14-18` have no user.
2. The api.ts comment at `:406-407` says the screen shows 미측정 when the win-expectancy slot exists but is unusable. The win sheet actually prints "현재 승리확률 정보 없음" (`GameScreen.tsx`, `sheet === 'win'`). Align the comment or the copy; CLAUDE.md's word is "미측정".
3. The pitch-mix and record sheets include the selected pitch (`upTo` uses `<=`, which is correct for "up to the pitch on screen"). The player sheet's 오늘 row is `context_before`, which excludes it. Once P1-2 is fixed, the sheet shows 29구 above a 30구 table when the selected pitch is this pitcher's. Either add a basis word to the row ("이 공 전") or cut MixTable at `< asOfIndex` there.
4. `asOfWords` returns `"${inning} ${pitch_number}구"`, but `pitch_number` is the count within the plate appearance. "고른 공(3회말 2구)까지" is ambiguous, so add the batter's name. Also, on a finished game the first pitch is selected by default, so a fan who chose nothing reads "고른 공(1회초 1구)까지의 타석이에요" and a single PA marked 타석 진행 중. This is consistent with the design, but the word "고른" is odd there.
5. The corrections line (`LivePanels.tsx:50`) labels `corrected - tagged` as "보류 중이라 표시하지 않은". That difference also counts corrected pitches that are `unconfirmed` or carry a non-held reason. The wording is only exact for `held_last_accepted`.
6. `pasAsOf`: a PA with a numeric `first` but `last` null keeps its file result even when the selected pitch is inside it. The fixture has 0 such PAs, so the exhibition is unaffected. `first`/`last` null with an earlier `at_bat_number` is handled correctly.
7. The notice closes sheets but leaves `pop` set (the status bubble). After the notice the bubble re-opens by itself and takes focus (`StatusPop` focuses on mount). Add `setPop(false)` in the noticeHold effect.

### Checked and fine
- **ws_alert key.** `ops_armb_off` is not in `OPS_ALERTS_COVERED`, and ws_operate has no `armb` problem key (`PROBLEM_SEVERITY`/`ALERT_KEYS`), so there is no duplicate `ops_problem:*`. The family is not game-gated. hold_s=0, and renotify follows the usual 900 s (critical), with dedup per key. The test's dedup and close steps are correct.
- **Notice in api.ts.** `isServiceNotice` is true only for 503 with `error === 'service_notice'` and an object `notice` carrying a string id. Both `getJson` and `getPublished` set it.
- **Static branch rethrow.** `fetchPublishHealth` lets `FetchError` through (`readPublished` → `getPublished`). The new `catch (e) { if (e instanceof FetchError && e.notice) throw e; }` rethrows correctly. The outer catch bypasses the `pending.current !== null` neutral return only for notices.
- **firstLoadRetry.** It keeps retrying while a 503 notice stands. The effect depends on `error` (a new object each failure) and re-arms. The one gap is when the retry lands on a non-forced `load` while another is in flight; that in-flight load then sets `error` itself, and static mode polls anyway. I found no path where retrying stops.
- **No old detail during a notice.** The early return `if (!detail || noticeHold)` happens before any drawing, there are no hooks after it, and the `drawn` reset runs in an effect after that render. When the notice ends normally (notice → ok), the first render either re-records the same file or is masked by `awaitingFile` with an empty signature. Either way no "이전:" note can name a pre-notice recommendation.
- **WatchList.** A notice replaces the list, a later non-notice failure does not un-hide it (it stays sticky there), and a good read clears it.
- **`asOfIndex` semantics.**
  - `latest` (`:437`) and `shownPitch` (`:443`) are defined before `:613`. When not latest, `shownPitch === pitch === selectedPitch`, the pitch on screen.
  - In a live game (local replay monitor) `latest` is always true, so the sheets show everything.
  - `pasAsOf` keeps the PA-ending pitch's PA complete (`last > asOfIndex` is false when they are equal). Fixture indices are 0..366, sorted and contiguous.
- **settledCorrection / heldReceipt.** On the fixture (14 settled corrections, 5 `held_last_accepted`), the tags and sentences follow the intended rules.
- **weKnown on the exhibition file.** That file has no `win_expectancy` slot and `live === false`, so the 승리확률 link is hidden.
- **Preview replay code.** `src/preview/*` loads only through `vite.preview.config.ts` (it rewrites index.html's script). The production `index.html` still loads `src/main.tsx`.

## (b) Silent behaviour changes to the operations line
- `ws_operate.py` is untouched, and `ws_alert.py` changes only by the new rule.
- Side effects of that rule, beyond the false recover/suppress window in P2-3:
  - Any segment deliberately run with `--armb 0` now pages **critical** every 15 minutes in a game window, every 6 hours outside one. Under `render()`/`dispatch(critical=True)` the batch title becomes "긴급". The replay publisher (`scripts/publisher_replay.py`) does not go through ws_operate, so the exhibition is unaffected. Operators should know that a forced-off rehearsal will page.
  - The rule is not game-gated, so it fires whenever the supervisor reports `running` with ARM-B off, even outside a scheduled window.
- Nothing else in the ops line changes.

## (c) What I checked and how
- `git log/diff origin/main...HEAD`, plus `git diff 64581cc HEAD` (UI-1026 tip → HEAD) for GameScreen, LivePanels, Sheets and Zone, and `git show a7f8a15` (the original DEMO-WS-GAPS-1028) to compare semantics.
- `git show <commit>:…/service.css | wc -l` across 64581cc, 2dab213, e3e3cbe, 651779f, f11de37, dfdd40a, 32e79c2 and origin/main, to find where the stylesheet came back. I diffed origin/main `service.css` against HEAD `service.css` lines 4 onward.
- A small Python CSS diff (`scratchpad/review/tools/cssov.py`) between `service-layout.css` and the legacy body of `service.css`. I checked specificity by hand for the `[data-rank]`, `.sv-stage` and `.sv-zrow` rules.
- `python3 -I scratchpad/review/tools/sim_armb.py`, which loads `ws_alert.py` from the worktree and drives `process()` with the default config through running → backoff → running.
- `python3 -I` reads of the 849803 exhibition fixture: receipts (14 corrected, 5 held), `context_before` pitch counts, whole-game vs as-of mix counts, PA first/last nulls (0), no `win_expectancy`, `publish.final: true`, `next: null`.
- Read the ws_operate status writer (`status_body`, `prestart`) for the publisher block shape, and `ws_ops_config.json` for backoff and the ARM-B floor.
- Read `notice-hide-smoke.mjs` and the `static-publish-smoke.mjs` diff, and `zone-geometry-check.mjs:80-90` for what the rank-look check covers.
- Not run: tsc, pytest, any Playwright smoke. Per its own commit message, `ws-gaps-smoke` and `ws-moments-smoke` (the exhibition-path smokes) were not ported, and `static-publish-smoke` passes only sections 1-3. So the exhibition path has no automated check on this UI.

## (d) Verdict
**BLOCK.** A merge artifact brought the pre-10/26 stylesheet back on top of the UI-1026 layout, which visibly reverts the accepted rank-square and stage design on every recommendation. The pitcher sheet also shows contradictory "today" counts in the exhibition's default view. Both are one-line fixes (restore the 3-line `service.css` while keeping `.sv-alert p`; pass `asOfIndex` to MixTable). After them this would be PASS_WITH_NOTES, with P2-1 and P2-3 worth fixing before the World Series ops line relies on them.
