# Opus 5.5 review — phone-width showcase reveal reachability (ed52840..cc919b4)

Label: `same_model_review; Codex_review_pending`

- Reviewer: Claude Opus 5.5, distinct from the author. Same model, so this is not an independent-model review.
- Target: `git diff ed52840 cc919b4` (`apps/observer/web/src/Showcase.tsx`, `apps/observer/web/src/showcase.css`, new `apps/observer/web/tests/showcase-phone-reveal-smoke.mjs`)
- Problem: T44 rehearsal attempt 1 ("1차 시도 실패"). At ≤800 px the comparison iframe stacks below the video, so its fixed reveal bar starts below the fold.
- Reviewer commit: `04f4884` (test only, see F1)
- Environment: Mac mini, headless Chromium (Playwright) emulation with synthetic fixtures only. No iPhone, no Safari, no 8766/8770/8771, no deploy.

## Disposition

**Ready for 8771 preview integration.** None of the findings block it.

- Hard rules (a) to (c) hold. I checked them in code and with the smoke, and mutations M1 to M5 confirm the smoke can catch them failing.
- (d) holds at 390x844 and 390x664 with the synthetic labels.
- F2 is a margin issue: a two-line comparison header pushes the iframe 6 px past the fold at 390x664. The reveal button itself stays fully visible. When the preview runs, check the real 39-scene labels at 390x664.
- (e) is untested (see "iOS Safari: untested").

## Hard-rule verification

- **(a) Scrolling never reveals.**
  - `scrollToSection` only calls `scrollIntoView` (`Showcase.tsx:68-71`). It sets no state and does not touch `take` or `selected`, so the iframe key `${selected}:${take}` cannot change because of a scroll.
  - The smoke checks that the selected pitch's own reveal is never requested (`own(i) === 0`) and `.wa-reveal` count is 0 after:
    - the jump-down button
    - the clip-stop scroll
    - the natural-end scroll
  - It also checks the next scene starts unrevealed. Pass.
- **(b) Clip, scene, abort and retry behaviour unchanged.**
  - The `onTimeUpdate` stop path still pauses and sets `stopAt = null`; it only adds the scroll (`clipFinished`, `Showcase.tsx:72-75`).
  - `onEnded` is new and guarded by `stopAt.current !== null` (`Showcase.tsx:111`).
  - These are untouched: `startClip`, the scene effect (which clears `stopAt` before pause and seek), `retryVideo`, and the timeline `AbortController`.
  - A scene change mid-clip does not scroll: the scene effect sets `stopAt = null` before its `pause()`, and there is no `onPause` handler. M1 shows the smoke catches it if this breaks.
- **(c) Desktop unchanged.**
  - The jump buttons are `display:none` outside the media query (`showcase.css:15`).
  - The JS scroll is gated by `matchMedia('(max-width: 800px)')`, read when the event fires. That is the same string as the CSS breakpoint.
  - The h2 bottom border moved to `.sc-compare-head`, which looks the same.
  - In the 1440x900 screenshot the comparison header is visually identical and there are no jump buttons. M2 and M4 show the smoke catches regressions here.
- **(d) 390 px.**
  - No horizontal overflow, measured at 390, 375 and 320.
  - The header button measures 70x40 px, which meets the ≥40 px rule.
  - Long names wrap the h2 to two lines without overlapping the button (screenshot `geom-390x664-19.png`).
  - The reveal button is fully in view after the scroll at 390x844 and 390x664. See F2 for the margin.
- **(e) iOS Safari.** Not run; see the last section.

## Findings

### F1 — non-blocking, test gap (fixed in `04f4884`)

- **Where:** `tests/showcase-phone-reveal-smoke.mjs`; the scene-change block ends at line 133 of cc919b4.
- **What:** The smoke did not cover manual playback reaching the media end when no clip is running.
- **Failure:** I mutated `onEnded={() => { if (stopAt.current !== null) clipFinished(); }}` to `onEnded={() => clipFinished()}`. That means scrolling whenever a viewer drags the native scrubber to the end. The cc919b4 smoke still passed it (exit 0).
- **Fix:** `04f4884` adds a step that seeks to `duration - .2`, calls `play()` with no clip running, waits for `ended`, and asserts `scrollY` is unchanged.
  - Passes on the cc919b4 build.
  - Fails that mutation at 390x844 and 390x664 (exit 1).

### F2 — non-blocking, layout margin

- **Where:** `showcase.css:18` (`.sc-compare iframe{height:min(700px,calc(100dvh - 64px))}`) together with the new `.sc-compare-head`.
- **What:** The 64 px reserve fits a one-line header (49 px) but not a two-line one (69 px).
- **Measurements** (Chromium, synthetic labels `9회 말 · <name> · 3구째`, after the jump):

  - 390x664, "Batter 2" / "Fernando Tatis Jr.": header 49, iframe bottom 650, reveal button bottom 640 (in view)
  - 390x664, "Pete Crow-Armstrong": header 69, iframe bottom **670 > 664**, button bottom 660 (in view, 4 px left)
  - 375x600, two-line names: iframe bottom 606 > 600, button bottom 596 (in view)
  - 320x568, a 36-character name (three lines): header 87, button bottom **582 > 568 (clipped 14 px)**

- **Failure:** When the label wraps, the bar's bottom padding and the safe-area strip fall below the fold. In the narrowest case, part of the button does too. I have not checked which real batter names appear among the 39 scenes.
- **Suggested fix (pick one):**

  ```css
  /* (a) reserve for a two-line header */
  @media(max-width:800px){ .sc-compare iframe{height:min(700px,calc(100dvh - 84px))} }
  ```

  ```tsx
  // (b) on clip end, align the iframe bottom (the bar) instead of the section top
  const frame = compare.current?.querySelector('iframe');
  (frame ?? compare.current)?.scrollIntoView({ behavior: still ? 'auto' : 'smooth', block: frame ? 'end' : 'start' });
  ```

  Option (a) is simpler. It costs 20 px of iframe height on every phone.

### F3 — non-blocking, UX: clip-end scroll can pull a reader back

- **Where:** `Showcase.tsx:74`
- **What:** If the viewer has already scrolled past the comparison during the 7 s clip (for example to the model or CV evidence cards), the clip end scrolls them back to the comparison.
- **Suggested fix:** Scroll only when the section is still below or partly below the fold:

  ```tsx
  const r = compare.current?.getBoundingClientRect();
  if (r && (r.top > 0 || r.bottom > window.innerHeight) && window.matchMedia('(max-width: 800px)').matches) scrollToSection(compare.current);
  ```

  This keeps the T44 case: the section is below the fold, so `r.top > 0` and the scroll happens.

### F4 — non-blocking, a11y: arrow glyphs in accessible names

- **Where:** `Showcase.tsx:129` and `Showcase.tsx:143`
- **What:** VoiceOver reads "↑ 영상" as roughly "위쪽 화살표, 영상". "영상" alone is also vague next to the section label "경기 영상".
- **Suggested fix:**

  ```tsx
  <button className="sc-jump" aria-label="영상으로 올라가기" …>↑ 영상</button>
  ```

  The visible "영상" is contained in the label, so it meets label-in-name. Update the smoke locator to match. Optionally put `↓` in `<span aria-hidden="true">`.

### F5 — non-blocking, note: focus is intentionally not moved

- **What:** After the clip-end scroll or the jump-down, focus stays on the now off-screen "이 공 다시 보기" or jump button. Space or Enter from a hardware keyboard would replay the clip off screen.
- **Why this is right:** Not moving focus into the iframe is the correct choice. If focus were in the iframe, its Space shortcut would reveal, so moving focus there would make a reveal one keystroke away.
- **Recommendation:** No change. Phone demo use is touch-only.

### F6 — cosmetic

- **Header button height:** 40 px, below the page's 44 px button standard (`.sc-page button{min-height:44px}`). It meets the ≥40 rule.
- **Hint spacing:** At 390x844, the hint paragraph sits close under the full-width jump button (screenshot `top-390x844-8.png`). Optional: `.sc-player>.sc-jump{margin-bottom:4px}`.

### Edge cases checked (no finding)

- **`onEnded` with `stopAt === null`** (the viewer scrubbed manually): no scroll. The guard is correct, and F1 now tests it.
- **Clip end while the comparison shows an error or missing mapping:**
  - `startClip` is disabled when there is no `decision`, so missing mappings cannot start a clip.
  - If a timeline retry fails after a clip started, the clip still scrolls to the comparison and shows the `role="alert"` retry. That is harmless and reveals nothing.
- **`matchMedia` timing:** read at event time. Rotating to landscape (844 px wide) gives the two-column layout and no scroll, which matches the CSS.
- **Reduced motion:** `behavior:'auto'` gives an instant scroll.
- **Media-error retry:** `retryVideo` clears `stopAt`, and no `ended` event follows an error, so no scroll.

## Commands and outputs

All commands ran with `nice -n 10 perl -e 'alarm 600; exec @ARGV' …`, one at a time.

**Setup and build**

- Symlinked `apps/observer/web/node_modules → /Users/song/Projects/pitcheezy/apps/observer/web/node_modules`. Nothing was installed.
- `npx tsc -b` → exit 0
- `npx vite build --outDir /tmp/opus-rv-cc919b4-dist --emptyOutDir` → exit 0 (`index-DZolEaWT.js` 380.24 kB, `index-B7RmCy0h.css` 103.18 kB)

**New smoke on cc919b4**

- `PZ_WEB_DIST=/tmp/opus-rv-cc919b4-dist node tests/showcase-phone-reveal-smoke.mjs /tmp/opus-rv-cc919b4-shots` → exit 0, 40/40 ok
- After `04f4884` → exit 0, 43/43 ok

**Negative control**

- Built ed52840 from `git archive` into `/tmp/opus-rv-ed52840-src`, output `/tmp/opus-rv-ed52840-dist`.
- New smoke → exit 1: `FAIL 390x844: jump-down button visible`, `FAIL 390x844: clip stop brings the reveal bar into view`, then a `TimeoutError` on the reveal click. This matches T44 attempt 1.

**Mutations** (each built to `/tmp/opus-rv-cc919b4-mut{N}`; src restored from a /tmp copy; `git diff cc919b4 -- apps/observer/web/src` is 0 lines afterwards)

- **M1** — `onPause` scrolls on phone (scroll on every pause): exit 1. `FAIL 390x844/390x664: scene change mid-clip does not scroll`
- **M2** — drop the `max-width` guard (scroll on desktop too): exit 1. `FAIL 1440x900: clip stop does not scroll on desktop`, `… natural end does not scroll on desktop`
- **M3** — remove the `onEnded` handler: exit 1. `FAIL 390x844/390x664: natural end brings the reveal bar into view`
- **M4** — remove `.sc-page .sc-jump{display:none}`: exit 1. `FAIL 1440x900: jump buttons hidden on desktop`
- **M5** — `onEnded={() => clipFinished()}`: **exit 0 on the cc919b4 smoke** (F1); exit 1 on the `04f4884` smoke

**Regressions** (`dist` symlinked to `/tmp/opus-rv-cc919b4-dist` for showcase-smoke, then removed with `unlink`)

- `node tests/showcase-smoke.mjs` → exit 0
- `PZ_WEB_DIST=… node tests/watch-spoiler-smoke.mjs` → exit 0
- `PZ_WEB_DIST=… node tests/pre-reason-neutral-smoke.mjs` → exit 0
- `PZ_WEB_DIST=… node tests/watch-setup-smoke.mjs` → exit 0
- `DIST=… CV_M3_RECEIPT=/Users/song/Projects/pitcheezy/results/checks/CV-M3-receipt-recalculation-20261004.json node tests/cv-evidence-smoke.mjs` → exit 0
- `DIST=… MODEL_EVIDENCE_REPO=/Users/song/Projects/pitcheezy node tests/model-evidence-smoke.mjs` → exit 0

**Geometry probes for F2**

- `zz-opus-probe-geom.mjs` and `zz-opus-probe-top.mjs` ran from `tests/` and were then moved to `/tmp/opus-rv-cc919b4-shots/`. They are not committed.
- A long-name variant of the smoke (`Pete Crow-Armstrong` / `Fernando Tatis Jr.`, 9회 말, at 390x664 and 375x600) → exit 0.

**Cleanup**

- `unlink apps/observer/web/node_modules` and `unlink apps/observer/web/dist`. `git status` is clean apart from this review.

## Screenshots (`/tmp/opus-rv-cc919b4-shots/`)

- `phone-reveal-390x844.png`: after the natural-end scroll, the comparison sits at the top. The reveal bar is at about y=750 with the evidence card below it. There is a sliver of the hint text above the section. No clutter.
- `phone-reveal-390x664.png`: the comparison fills the viewport and the reveal bar is fully visible (bottom 640). The header reads "1회 초 · Batter 2 · 5구째" next to the "↑ 영상" button, with clear space between them.
- `phone-reveal-1440x900.png`: the two-column layout is unchanged. There are no jump buttons and the header border is intact.
- `top-390x844-8.png`: the full-width jump button below the three clip controls reads clearly. The hint text is tight under it (F6).
- `geom-390x664-19.png`: the two-line header wraps "3구째" onto line 2 without touching the button. The bar's bottom edge meets the fold (F2).

## iOS Safari: untested

I reasoned about each of these but did not run any of them.

- **Native fullscreen.** `playsInline` plays inline by default, but a viewer can tap the fullscreen control.
  - `pause()` at release+4 s keeps Safari in fullscreen. The `scrollIntoView` happens behind it, and I don't know whether WebKit keeps or restores that scroll position on exit.
  - iOS usually leaves fullscreen when media ends. I don't know whether the `onEnded` scroll then holds.
- **Smooth `scrollIntoView` options.** These need iOS 15.4 or later. Older versions should still jump instantly.
- **`dvh` with collapsing toolbars.** A programmatic scroll may or may not collapse Safari's toolbars. If it does, `100dvh` grows during the smooth scroll and the iframe resizes. The bar is fixed to the iframe bottom, so it should stay inside the viewport, but I haven't checked.
  - If `dvh` is unsupported (before iOS 15.4), the iframe height declarations are invalid. That is already true before this diff.
- **Safe area in the iframe.** `env(safe-area-inset-bottom)` inside the iframe's action bar is probably 0, because insets belong to the top-level page. With Safari's toolbar visible this is irrelevant; with the toolbar collapsed on Face ID iPhones the home indicator may overlap the bar's bottom padding.
- The 390x664 Chromium viewport stands in for "Safari with toolbars". The real visible height depends on the iOS version and the tab bar layout (compact or bottom).
