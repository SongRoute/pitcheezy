# Review — /watch navigation without spoilers (f169c14)

Label: **same_model_review; Codex_review_pending**. Reviewer: Opus 5.5 (separate from the author). 2026-10-04.
Target: `840eeee..f169c14` on `codex/watch-spoiler-review-1004` (`WatchAlong.tsx`, `tests/watch-spoiler-smoke.mjs`, author note).
Scope respected: no API/backend/model/support-gate/recommendation change; no server, port, tmux, ssh or GPU touched.

## Disposition

**Non-blocking for root's integration decision.** The change does what audit option 2 asks: the 1초–9말 row, extras only
once reached, the reached-only PA select and the progress without a total all hold in the browser. I found no path where a
stale `furthest` or the select leaks game B or a future PA. I fixed one real but fail-safe bug (finding 1, the shared
link is forgotten after a reload) and closed one test gap (finding 2). Findings 3–6 are suggestions only.

## Findings

1. **Low (fixed here). A pitch reached by opening a shared `#i=N` link was never saved.** `WatchAlong.tsx:217` vs. the old
   effect at `:225–229` (f169c14). On load, `furthest` became `max(first, stored)` in state only. localStorage was written
   only when `index > furthest`, which cannot happen right after a jump. Scenario (probe, 360 and 390 px): open
   `?game=…#i=<last pitch of a 12-inning game>` → the row shows 10초–12말 → tap 1초 → reload → the stored key is `null` and the
   row goes back to 18 buttons. This hides more than it should (no spoiler). It does contradict the author note "a reload
   does not hide them again". Fix applied (change A): one effect raises `furthest` and a second persists `furthest` itself.
   Writing is skipped in embed mode, so the showcase iframe keeps its old behaviour (it never wrote the key there either).
   ```diff
   -  useEffect(() => {
   -    if (!timeline || index <= furthest) return;
   -    setFurthest(index);
   -    try { localStorage.setItem(`pitcheezy.watch.${timeline.game.game_pk}.furthest`, String(index)); } catch { … }
   -  }, [index, furthest, timeline]);
   +  useEffect(() => { if (timeline && index > furthest) setFurthest(index); }, [index, furthest, timeline]);
   +  useEffect(() => {  // also keeps a pitch reached by opening a shared #i=N link, not only by stepping past it
   +    if (!timeline || embedded) return;
   +    try { localStorage.setItem(`pitcheezy.watch.${timeline.game.game_pk}.furthest`, String(furthest)); } catch { … }
   +  }, [furthest, timeline, embedded]);
   ```
   The stored value never goes down, because `furthest ≥ stored` after load.

2. **Low (test gap, fixed here). The smoke did not catch `furthest` shared across games.** In mutant M1 the key drops
   `<game>` (`pitcheezy.watch.furthest`), and the original smoke still passed. Game 2 then lists 51 PAs at its first pitch,
   but game 2 has no option-count check. Extras did not show only because game 1's 101 < 108 (game 2's first 10초 pitch).
   Change B adds that check plus the shared-link case from finding 1.
   - The smoke now also exits cleanly: `try/catch/finally browser.close()`, and the 9말 click has a 3 s timeout that
     records a FAIL.
   - Before, the old build failed with an uncaught 30 s `TimeoutError`. Now it prints 10 FAIL lines and exits 1 in about
     6.6 s.

3. **Low (suggestion). A regular half missing from the data jumps to the game's last pitch.** `WatchAlong.tsx:248`,
   `at.get(label) ?? last`. Scenario (probe game with no 4말 pitches): tap 4말 → `#i=101`, which is the final pitch, and
   the current button is 9말. One tap skips the whole game and lands on the ending. The real 12 datasets have 17–20 halves
   each, and the audit reports no gaps, so this is unlikely. A safer fallback is the first pitch of the next half that
   exists:
   ```ts
   const order = (l: string) => { const n = Number.parseInt(l, 10); return n * 2 + (l.endsWith('말') ? 1 : 0); };
   …map(label => ({ label, first: at.get(label) ?? halves.find(h => order(h.label) > order(label))?.first ?? last }))
   ```
   With this fallback, an unplayed 9말 still goes to `last` and a data gap goes to the next half. Not applied: it is a
   behaviour choice for root.

4. **Low (UX wording). Two different "구째" sit next to each other.** The board shows `${pitch_number}구째`
   (`:307`, the pitch in this at-bat, e.g. "2구째"). The nav now shows `${index+1}구째` (`:315`, e.g. "144구째") one line
   below. Before, the nav said "144/144구", so the two did not clash. Suggest `{index + 1}구` or `경기 {index + 1}번째 공`.

5. **Info. A never-played 9말 (or 6초–9말 of a shortened game) jumps to the last pitch. This is acceptable.** Tapping 9말
   asks to skip past the end of 9초, so landing on the last pitch, before its reveal and highlighted 9초, is the nearest
   honest target. Any behaviour after that tap shows that 9말 did not happen. That is the viewer's own choice, and the
   author note already lists it. Shortened-game probe (ends after 5말): tap 7초 → last pitch, current 5말, as designed.
   Postseason games are not shortened, so this case is rare. If root wants it gentler, a one-line notice ("9말은 열리지
   않았어요") after such a jump would explain the landing.

6. **Info. No other leak found in the row or the select.**
   - Button count is always 18 before extras are reached.
   - `current` uses the label lookup (`:253`), so at the last pitch of a home-lead game it marks 9초, not 9말. Mutant M5
     (the old `reduce`) marked 9말 and was caught.
   - A walk-off 9말 and a played 9말 give the same row.
   - `key={h.label}` is not rendered.
   - The select (`:313`) is safe:
     - `reachedPas` is a prefix of `plateAppearances`, sorted by `first`, so option `i` is the same PA index that
       `onChange` uses.
     - `currentPa` ≤ the index of the PA containing `index` ≤ `max(furthest, index)`, so `value` is always among the
       options. That holds for forward and backward hash jumps and for keyboard steps.
   - Stale `furthest` across games: `gamePk` changes only on a full page load (`<a href>`, no pushState), so state starts
     fresh. Even if it changed in place, the load effect sets `timeline`, `index` and `furthest` in one batched update,
     and the effects return early while `timeline` is null.
   - StrictMode double effects only fetch twice. The effects are idempotent.
   - Remaining leak, out of scope and stated by the author: `/api/watch/{game}` still sends every pitch's `situation`.

## Layout (probe, 12-inning synthetic game at its last pitch, 24 buttons)

- 360 px: document `scrollWidth` 360 (no page scroll). The row scrolls inside itself (`scrollWidth` 1322, client 360).
  Buttons are at least 48×44. The current button (12말) was scrolled into view at x 296–344.
- 390 px: document 390. The row is 1322 wide inside a 390 px client. Buttons at least 48×44. The current button sits at
  x 326–374.
- The smoke itself checks 44 px targets and page scroll at 390 px only. 360 px is covered by this probe, not by the test.

## Changes made by the reviewer (separate from the author's commit)

- A. `apps/observer/web/src/WatchAlong.tsx:225–229`: persist `furthest` whenever it changes (not only when stepping
  past it); skip writing in embed mode (finding 1).
- B. `apps/observer/web/tests/watch-spoiler-smoke.mjs`:
  - adds game 3 (shared link → back → reload keeps 10초/10말) and a game-2 option count of 1;
  - wraps the run in try/catch/finally;
  - gives the 9말 click a bounded wait (finding 2).

## Commands run (Mac, light; temporary `node_modules` symlink to the main checkout, removed before commit)

```
tsc --noEmit -p tsconfig.json                                → exit 0 (f169c14 and with change A)
vite build --outDir /tmp/pz-watch-spoiler-review-dist        → ok (f169c14)
vite build --outDir /tmp/pz-watch-spoiler-review-fix-dist    → ok (with change A)

original smoke, f169c14 build, 5 runs                        → "watch spoiler smoke: ok" ×5
original smoke, 840eeee build (/tmp/pz-reconnect-1004-old-dist):
  FAIL game 1 row at the first pitch is not 1초–9말: 1초,…,9초
  FAIL progress shows the total: 1/102구
  FAIL plate-appearance list shows 51 options at the first pitch
  locator.click: Timeout 30000ms exceeded (uncaught, exit 1, 30.4 s)   ← reproduces the author's "3 FAIL then timeout"

mutants vs original smoke (f169c14 source): M1 shared furthest key → ok (SURVIVED); M2 select unfiltered → FAIL;
  M3 progress total → FAIL; M4 reached by index only → FAIL "going back shrank"; M5 old currentHalf reduce → FAIL
  "current half at the last pitch is 9말"; M6 furthest not persisted → FAIL "forgotten after reload"

extended smoke (change B):
  fix build, 6 runs (1 + 5 repeats)                          → ok ×6
  f169c14 build                                              → FAIL extra innings reached by a shared link forgotten after reload (exit 1)
  840eeee build                                              → 10 FAIL lines, exit 1, 6.6 s, no uncaught timeout
  mutants vs fixed source: M1–M7 (M6 = persist String(0); M7 = extras always shown) → all FAIL (7/7 killed)

showcase-smoke (temp copy, dist from PZ_WEB_DIST, deleted), fix build
  → "showcase smoke: ok (synthetic WAV media + timeline/reveal fixtures, desktop + phone)" (embed still has no navigation)

probe (temp file, deleted): 360/390 layout above; shared-link reload (finding 1: stored key null, row 18 at both widths);
  shortened game 7초 → #i=59 = last, current 5말; data gap 4말 → #i=101 = last, current 9말;
  hash-only goto then immediate row read: 0/20 stale (the smoke's same-document `open()` was not racy in practice)
```

All runs were wrapped in `perl -e 'alarm N'`. No leftover headless browser processes were found afterwards. Not run:
`watch-along-smoke.mjs` (needs a running observer), the full test suite, a real iPhone or Safari (미측정).

## Notes on the test

- `open()` reaches the same game by changing only the hash (a same-document navigation), so the persistence before
  `reload` comes from the `hashchange` → `go()` path. A pure full-load path is now covered by game 3.
- After the `3초` click (`:71`), the option count is read without an explicit wait. React flushes discrete click updates
  synchronously, and mutant M4 was caught, so this is not flaky here. A `waitForFunction` on the hash would make it
  explicit.
