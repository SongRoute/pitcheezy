# Opus 5.5 review — keep `?media=local` across same-tab in-app navigation (`ed40b66..879b092`)

Status: `same_model_review; Codex_review_pending`. Reviewer: Opus 5.5 (distinct from the author). 2026-10-04 KST.
Branch `codex/media-persist-review-1004`. My commits on top of 879b092: `136d8b0` (test), `03c6f93` (README).

**Verdict: no BLOCKER, no MAJOR.** The code change is correct and small. Default behaviour is unchanged. Blocked storage falls back to the URL. Reveal, abort and scene logic are untouched. The author's test D was weaker than claimed. It still passed with `localStorage` in place of `sessionStorage`, and it still passed with the try/catch removed (mutation runs below). I strengthened it (`136d8b0`). README line 29 had become inaccurate; fixed in `03c6f93`. Safari, iPhone and the MacBook are still unchecked.

## Commands and exit codes

Settings: `OMP_NUM_THREADS=1 UV_THREADPOOL_SIZE=1`, `nice -n 10`, one process at a time. `node_modules` was a temporary symlink to `codex-showcase-1006/apps/observer/web/node_modules`. It was never added and was `unlink`ed at the end. Logs are in `/tmp/pz-media-persist-review-r1/`. There were no requests to mlb.com: every page routes the MLB URL to `abort()` (`seen.mlb` counts the interceptions). No video was downloaded, and no ports 8766/8770/8771 were used.

1. `npx tsc -b` → exit 0.
2. `npx vite build --outDir /tmp/pz-media-persist-review-build --emptyOutDir` → exit 0. The sha256 of `index.html`, `index-BGk569ec.css` and `index-C2cOnYq0.js` **matches the author's `build-SHA256SUMS.txt` byte for byte**.
3. `PZ_WEB_DIST=<2> node tests/showcase-local-media-smoke.mjs` (author's version, 879b092) → exit 0, `decoded {"390":"ready","1440":"ready"}`.
4. `PZ_WEB_DIST=<2> node tests/showcase-smoke.mjs` → exit 0.
5. `PZ_WEB_DIST=<2> node tests/showcase-phone-reveal-smoke.mjs` → exit 0 (390x844, 390x664, 1440x900).
6. Negative control: author's smoke against the served ed40b66 bytes `pitcheezy-pm-jobs/20261004-end-to-end/t64d-build` → exit 1, `AssertionError: back link keeps the opt-in` (actual `/watch`, expected `/watch?media=local`). This is an assertion failure, not a setup error. But it stops at the back-link href, which is cosmetic (see N1). It never reaches the core claim.
7. Mutation M1: the try/catch removed from `readLocalMedia`, built to `/tmp/pz-media-persist-review-mut1` → author's smoke **exit 0** (not caught).
8. Mutation M2: `sessionStorage` → `localStorage`, built to `/tmp/pz-media-persist-review-mut2` → author's smoke **exit 0** (not caught). The author's "new tab" is `browser.newPage()`, which is a new browser context. Even `localStorage` is empty there.
9. After `136d8b0`: the strengthened smoke against <2> → exit 0, `decoded {"390":"ready","1440":"ready"}`.
10. After `136d8b0`: the negative control (ed40b66 bytes) → exit 1, `AssertionError: plain /showcase in the same tab stays local` (actual = MLB URL, intercepted). It now fails on the core claim.
11. After `136d8b0`: M1 → exit 1 (case E: the page crashes on the sessionStorage throw, so the iframe never appears; TimeoutError at `open`). M2 → exit 1, `AssertionError: a new tab starts from the MLB URL`.

The Showcase source did not change after 879b092, so build <2> is still the code under review. Steps 4 and 5 were not rerun after `136d8b0`, which only touches the local-media smoke and the README.

## 1. Default behaviour (no `media=` in a fresh tab)

- `Showcase.tsx:18-26` adds no module-level storage access. Module scope only declares `MEDIA_KEY`. This matters because `main.tsx:9,13` imports Showcase on every route.
- `readLocalMedia` runs as a `useState` initializer (`Showcase.tsx:40`), so it runs on first render only. With `asked === null`, line 22 and line 23 don't run, so nothing is written. Line 24 does one `getItem`, which returns `null`, so the result is `false`.
- Rendered output is then identical to ed40b66: the back link is `'/watch'` (`:115`), the video src is `source.mediaUrl` after the load click (`:123`), and there is no note (`:134`). Case A of the smoke confirms this (MLB src, 0 `/offline-media/` requests, no note).
- StrictMode (`main.tsx:13`) calls initializers twice in dev builds only. The function is idempotent: setting the same value, removing an absent key, and reading are all harmless twice. Production calls it once. There is no ordering issue: the read happens after this render's write, inside the same call.
- One behaviour change is intended: in a tab that opted in earlier, a plain `/showcase` now plays the local copy. That is the feature.

## 2. Opt-out, new tab, reload, indicator

- `?media=local` sets the flag (`:22`). Any other explicit value clears it, including `?media=` empty or `LOCAL` (`:23`). Absent keeps the stored value (`:24`). This matches the comment at `:16-17` and the README's "끄려면 `?media=online`".
- New tab or another origin: sessionStorage is per tab and per origin, so it starts online. `127.0.0.1:8790` and `localhost:8790` are different origins. A same-profile new tab is now tested (D, `136d8b0`).
- **Duplicate tab** (Chrome/Safari "Duplicate Tab") copies sessionStorage, as the HTML spec allows, so it would stay local. This is not tested; noted in the README (`03c6f93`).
- Reload keeps it. Before navigation, the scene effect's `replaceState` (`:60-63`) keeps `media=local` in the URL anyway. After navigation, the URL is `?scene=N`, and sessionStorage carries the flag.
- Acceptable for a demo operator: yes. The opt-in is explicit, scoped to the tab, and clearable. **A visible indicator already exists**: `:134` shows "맥북에 저장한 같은 공식 영상 사본을 씁니다" whenever `localMedia` is true, before the load click, so the operator can tell without reading the URL.
- Residual risk: an operator who opts in once on an online deployment (e.g. 8771 with no `/offline-media/`) gets a 404 in that tab. That shows the existing alert with "공식 영상 따로 열기" (`:136`). Recovery is `?media=online` or a new tab. See N3.

## 3. Storage blocked

- Every `sessionStorage` access, including the property read itself (which throws `SecurityError` when storage or cookies are blocked), is inside the `try` at `:21-25`. `URLSearchParams` parsing is outside the try, and it cannot throw.
- The catch returns `asked === 'local'`, which is the old behaviour. The scope is right. A failing `setItem` (quota) still returns `true` for `?media=local`. A failing `removeItem` returns `false` for `?media=online`.
- Note: modern Safari private mode no longer throws on sessionStorage `setItem`. That was a Safari ≤10 behaviour. The catch is still the right guard for blocked-cookie settings.
- Now tested: case E (`136d8b0`) makes the `window.sessionStorage` getter throw. `?media=local` stays local, a plain `/showcase` falls back to online, and there are 0 page errors. M1 proves E can fail.

## 4. In-app routes into /showcase (report only; WatchAlong.tsx and Game.tsx not edited)

- The only in-app link to `/showcase` is `WatchAlong.tsx:212` (`href="/showcase"`, plain). The stored flag covers it, and D now clicks this exact link.
- `Game.tsx:349,359,422`, `Live.tsx:107,137` and `WatchAlong.tsx:111,333,383` link to `/watch` without the param. That doesn't matter, because the flag lives in storage, not the URL. `App.tsx`, `VideoLab.tsx` and `InningDecisions.tsx` have no link to `/showcase` or `/watch`.
- The embedded `/watch?…&embed=1` iframe (`Showcase.tsx:166`) is same-origin, so it shares the tab's sessionStorage. It doesn't read the flag, and its header back link is hidden in embed mode (`WatchAlong.tsx:347`).
- No route lands on `/showcase` without being covered, as long as storage works. With storage blocked, every in-app return is online. That is the pre-T66 behaviour; the README's fallback advice (browser back, or append the param again) still applies.

## 5. Contracts untouched (Showcase.tsx diff, line by line)

The diff touches only three places: `:16-26` (new comment, key, function), `:40` (the initializer now calls `readLocalMedia`) and `:115` (back-link href). The following are byte-identical to ed40b66:
- the timeline fetch with `AbortController` (`:49-58`)
- the scene effect and `replaceState` (`:60-70`)
- `startClip`/`clipFinished`/`retryVideo` (`:72-101`)
- `copyLink`
- the video element (`:123-131`)
- the reveal iframe keyed on `selected:take` (`:166`)
- `showcase-scenes.json`

No data or API change.

## 6. Tests

See the commands above. The author's D proved less than its comment said:
- it did not use the list's link (it called `page.goto('/showcase')`);
- its "new tab" was a new context, so it passed with `localStorage` (M2);
- nothing exercised the catch (M1);
- the negative control stopped at the cosmetic back-link assertion.

All four are fixed in `136d8b0`:
- the synthetic snapshot gains `/api/watch/games`, so the list renders and the real link is clicked;
- the second tab opens in the same context;
- case E is added;
- the behavioural assertion comes before the href assertion.

## 7. Honesty

- The author's report and CONTINUATION make no Safari or iPhone claim, and the smoke header says "Headless Chromium only".
- The README's T66 line is accurate. The exception was line 29 ("`?media=local`이 없으면 평소처럼 인터넷 영상"), which became false after T66. It now says to check by the on-screen note (`03c6f93`).
- "묶음 v5 이후" refers to a bundle that is not built yet. It is phrased conditionally ("…빌드가 아니면 예전처럼…"), which is fine.
- README line 14 ("`?media=local` 없이 열면 화면 동작은 8771과 같음") describes the v4 dist and stays true for v4. When v5 is cut, it should say "in a fresh tab".

## Findings

| ID | Severity | Finding | Disposition suggestion |
|---|---|---|---|
| m1 | MINOR | Test D did not prove its claims: the per-tab scope (M2 passed), the storage fallback (M1 passed) and the real list link were all untested, and the negative control stopped at a cosmetic href | **Fixed** `136d8b0`; accept |
| m2 | MINOR | README:29 said no `?media=local` in the address means internet video, which is false after T66 in the same tab | **Fixed** `03c6f93` (check by the note); accept |
| N1 | NOTE | The back link's `/watch?media=local` (`Showcase.tsx:115`) does nothing functionally. WatchAlong ignores `media`, and its list link (`WatchAlong.tsx:212`) drops it. Storage does all the work. Harmless, and it only changes local mode | Keep (it shows the operator the mode in the URL), or drop it later; no action for 10-06 |
| N2 | NOTE | Duplicate Tab carries the flag (sessionStorage copy). Not tested in any browser | README caveat added; accept |
| N3 | NOTE | Opting in once on an online deployment leaves that tab trying `/offline-media/` (404 → existing alert + official link) until `?media=online` or a new tab | Accept; the note at `:134` makes the mode visible |
| N4 | NOTE | When bundle v5 is cut, README:14 should be updated to say "in a fresh tab" | Do it with the v5 bundle |
| N5 | NOTE | Safari, iPhone, the MacBook and the real 436 MB file were not checked by this unit or this review | Keep the Safari rehearsal on the 10-06 checklist (README:34) |

Not done: no deploy, push or merge; no edits to WatchAlong.tsx or Game.tsx; no full suites (the `offline/` pytest and the other 7 regressions were not rerun, since 879b092 does not touch Python and those pages' sources are unchanged).
