# Review — offline demo fallback (177ce1b)

`same_model_review; Codex_review_pending`

- Reviewer: Claude Opus 5.5, separate session from the author. No other models called, no sub-agents.
- Target: `git diff 88976ab 177ce1b` → `apps/observer/web/offline/{capture_snapshot.py, serve_offline.py, test_offline_bundle.py, README-ko.md}`
- Date: 2026-10-04 (KST). Worktree `codex-offline-fallback-review-1004`, branch `codex/offline-fallback-review-1004`.

## Disposition

**No blocking findings. OK to keep as the 10/6 fallback.** The built bundle `pitcheezy-offline-demo.tar.gz` (sha256 `f473ab47…9fae`, re-hashed: matches) can be used as it is:

- It covers every request the showcase and its embedded watch frame make for all 39 scenes.
- The API bytes and JS bytes are identical to 8771. The served HTML matches 8771 except for the added banner.
- Path traversal is refused.

I applied 6 small non-blocking fixes (R1–R6 below). They are in this branch only. The tarball still holds the 177ce1b `serve_offline.py` and README. To pick up the fixes, repack only: copy `serve_offline.py` and `README-ko.md` into the bundle and tar again. No recapture and no 8771 contact are needed, because the fixes do not touch `capture_snapshot.py` output. The bundle was not modified by me.

## What I checked (commands, real exit codes)

| # | Check | Result |
|---|---|---|
| C1 | `shasum -a 256` bundle | `f473ab47…9fae` = expected |
| C2 | extract to `/tmp/pzoff-review.ddhV`; `cmp` bundle `offline/{capture_snapshot,serve_offline}.py`, `README-ko.md` vs 177ce1b | identical (tar exit 0) |
| C3 | dist sha256 | `index.html 2db4feab…`, `index-BpZyo5LN.js 4064e103…`, `index-CbbIe769.css 02cbd371…`, `favicon.svg bbfd72e5…`. All 4 match the expected values. The JS contains the strings added in aabd012 (model card) and 7b268a3 (CV card) (`grep -cF` = 1 each) |
| C4 | Coverage, under `/usr/bin/python3` 3.9.6: ran `needed_paths(saved timeline, src/showcase-scenes.json)` and compared with the bundle manifest | exit 0. `missing []`. Needed 176 (137 reveal + 39 summary), all in the manifest. The manifest has exactly 2 extra entries (timeline, games). `scenes_sha256` equals the repo file. 262 decisions, `index == position` for every one. 0 non-contiguous `pa_id`s. I also recomputed `paStart` independently, the way the frontend does (compare to the PA's first pitch, `WatchAlong.tsx:234-238,257`), and coverage still holds |
| C5 | 8771 byte comparison, 5 GETs total (budget used up): `/api/watch/849843`, `/reveal/261`, `/summary?through=261&top=5`, `/showcase`, `/assets/index-BpZyo5LN.js`. Each also fetched from the offline server (8792) | API ×3 and JS: **identical**, all curl exit 0. `/showcase` differs only by the banner: `cmp` after stripping `<div id="pz-offline-note">…</div>` → identical |
| C6 | Offline server from the extracted bundle under `/usr/bin/python3` 3.9.6 on 127.0.0.1:8792 (`alarm 300`); 40 curl probes | see table below |
| C7 | Headless Chromium against 8792 only (`/tmp/pzoff-review.ddhV/flow.mjs`, alarm 300). Opened the "이 공 관전 화면 열기" target `/watch?game=849843#i=7` directly, then: reveal → `→` → reveal → `←←` | node exit 0 (the first run exited 1 because my script tried to press reveal on a pitch that was already revealed; I fixed the script and reran). 0 page errors. Results in F2 |
| C8 | Startup failure modes (on a copy of the extracted bundle) | changed file → `snapshot file changed: …`, exit 1. Missing file → raw `FileNotFoundError` traceback, exit 1. Port in use → raw `OSError: [Errno 48]`, exit 1. Missing dist → `FileNotFoundError`, exit 1. So every case stops the server; two of them show a raw Python error (fixed in R3/R4) |
| C9 | `pytest test_offline_bundle.py` (.venv, OMP 1, alarm 300) | 177ce1b: **9 passed**, exit 0. After my fixes: **11 passed**, exit 0 |
| C10 | `/usr/bin/python3 -m py_compile serve_offline.py capture_snapshot.py` (after fixes) | exit 0. The fixed server also started on 8793 under 3.9.6, loaded 178 answers, and served the same bytes as 177ce1b for timeline, reveal/261 and summary/261 (`cmp` identical) |
| C11 | Servers stopped | `lsof -nP -iTCP:{8792,8793,8794,8795} -sTCP:LISTEN` → all exit 1 (empty) |

C6 probe table (status, type, Cache-Control, banner):

- `/showcase`, `/showcase?scene=38`, `/showcase/`, `/watch`, `/watch?game=849843`, `/api` (no slash), `/../../etc/passwd`, `//etc/passwd`, `/assets/` → 200 html, `no-cache`, banner 1. These all get the SPA index.
- `/watch?game=849843&embed=1`, `…embed=1&…&x=y`, `…embed=%31` → 200 html, no banner. This matches the frontend's `params.get('embed') === '1'`.
- `…embed=1&embed=0` → banner shown, but the frontend treats the page as embedded, so the banner appears inside the frame. `…embed=0&embed=1` → banner shown, page not embedded. The server never leaves the banner off a page the frontend treats as non-embedded. The mismatch only ever shows *more* disclosure.
- `/assets/index-*.js` → `text/javascript`, `/assets/*.css` → `text/css`. Both `public, max-age=31536000, immutable`. A query string on an asset is ignored. `/favicon.svg` → `image/svg+xml`, `no-cache`.
- API answers → `application/json`, `no-store`. HEAD: 200 with correct `Content-Length` (453) and no body. POST → 501.
- Summary: `top=5&through=2` = `through=2` = canonical answer. `top=6` → 404. `through=02` → 404 (conservative). `through=2&through=7` → 200 with the **through=2** answer (F5, fixed in R2).
- `/api/watch/849843?x=1`, `/setup`, `/api/replay/…/outline`, `reveal/99999` → 404 with the MISSING message. `/api/live/…` → 503 NO_LIVE.
- `/assets/../../serve_offline.py`, `/assets/%2e%2e/%2e%2e/offline/manifest.json`, `/%2e%2e/offline/serve_offline.py`, `/x.js`, `/.` → 404 JSON. The path is never percent-decoded, `resolve()` follows symlinks, and the target must be strictly under `dist`.

## Findings

### F1 — non-blocking (honesty, docs) — `serve_offline.py:5`, README l.14
The docstring says "Every API answer is a byte-identical file". `/api/watch/games` is not byte-identical. It is the 8771 list filtered to 849843 and re-serialised (`capture_snapshot.py:99-104`). The filtering itself is honest and disclosed in three places: manifest `note: filtered to game 849843`, the banner "경기 849843 시연 장면만", and the picker shows one game. Only the docstring and README overstated it. **Fixed (R1).**

### F2 — non-blocking (coverage outside the scene flow) — behaviour confirmed in C7
The normal demo flow is: pick a scene in the outer page → embedded frame → reveal → summary → retry. Every request in that flow is saved. In the new-tab non-embedded watch ("이 공 관전 화면 열기", `Showcase.tsx:122`):
- `→` without reveal to i=8 shows the pre-pitch card from the timeline. Pressing reveal works (reveal/8 is saved, because i=8 is in the PA of another scene). The "경기 흐름" panel then shows the 404 text "…시연 장면 39개만 저장돼 있어요" with 다시 시도.
- `←` to i=6 (earlier pitch of the same PA): the reveal comes from the strip, and summary through=6 → 404 text.
- A pitch in no saved PA: reveal → the same 404 text in the alert.

Nothing is synthesised and nothing crashes, but the page is partly filled in. The old README line only said "다른 공으로 넘기면 … 뜸". **README clarified (R6).** Other non-demo links from the picker: `/game?…&replay=0` and `/live?…&replay=0` hit `/api/live/{pk}/state` → 503 "실시간 경기 정보는 볼 수 없어요". That is technically the live-state API, but the user clicked a *replay* link. Optional wording change, not applied: `NO_LIVE = '오프라인 예비 묶음이라 실시간 경기·다시 보기 화면은 쓸 수 없어요.'`.

### F3 — non-blocking (honesty, latent drift) — `serve_offline.py:23`
`MISSING` hard-coded "39". The banner reads from the manifest, but this message did not. If someone recaptures with a different scene list (the README allows "다시 만들기"), the 404 text would state a wrong number. **Fixed (R5):** the count now comes from `manifest['scenes']`, with a test.

### F4 — non-blocking (robustness/safety order) — `serve_offline.py:32-37`
1. `load_snapshot` read `directory / entry['file']` *before* checking that `entry['file']` is the canonical `api/…json` name. A manifest entry `file: ../x` was read first and only rejected afterwards (by sha or name).
2. A missing file produced a raw `FileNotFoundError` traceback.
3. Mutation M9 (removing the name check) survived the 177ce1b tests.

**Fixed (R3):** name check first, explicit `snapshot file missing: …`, and a new test that kills M9.

### F5 — non-blocking (canonicalisation) — `serve_offline.py:45-48`
A repeated summary param (`through=2&through=7`) was answered with the *first* value's saved bytes. It is unclear which value the live API would use, so a saved answer could belong to a different query. The frontend never sends repeats (`WatchSummaryPanel.tsx:35`), so this cannot happen in the demo. **Fixed (R2):** any repeated param → 404 MISSING, with a test.

### F6 — non-blocking (operator UX) — `serve_offline.py:119`
Port 8790 already in use → raw `OSError: [Errno 48] Address already in use`, and the README had no hint. On a borrowed MacBook in a professor's office this is the most likely startup failure. **Fixed (R4):** `127.0.0.1:8790 포트를 열지 못했어요(Address already in use). 다른 번호로: --port 8791`. README step 3 now covers it.

### F7 — non-blocking (README accuracy) — README l.17–18
- "인터넷이 없으면 영상 대신 화면 캡처로 설명" reads as if captures are provided, but the bundle contains none. Reworded.
- The showcase's "아이폰용 관전 링크 복사" copies an `http://127.0.0.1:8790/…` link and shows "아이폰에서 열어 주세요". On the iPhone that link points at the phone itself. The README already said the iPhone can't connect, but now explicitly says not to use that button. **Fixed (R6).**
- Claims I confirmed as accurate: iPhone cannot reach 127.0.0.1; video needs internet (`mediaUrl` = `mlb-cuts-diamond.mlb.com/…mp4`, not bundled); "GET만 약 180번" (177 saved + 1 games fetch = 178).

### F8 — non-blocking, not fixed — `/index.html` served without banner — `serve_offline.py:91`
`/index.html` has a dot, so it takes the static-file branch and is served raw with no banner. `main.tsx` would then render the legacy `<App/>`, because the route is `/index.html`. No demo link leads there. Optional diff: route `path == '/index.html'` through the SPA branch, or 404 it.

### F9 — non-blocking, not fixed — integrity scope
- Only snapshot files are hashed at startup. `dist/` is not, so a changed `dist` would still start, while the README says "8771과 바이트가 같음".
- The manifest is self-certifying: a consistent edit of file and manifest together passes.

Suggestion: in the next repack, record the dist sha256s in the manifest and check them in `load_snapshot`. Write the tarball sha256 (`f473ab47…`) into the hand-off note so it can be checked on the MacBook with `shasum -a 256`.

### F10 — non-blocking, not fixed — `--host` accepts any address — `serve_offline.py:113`
The default is 127.0.0.1, so it is safe as shipped. `--host 0.0.0.0` would expose the server to the LAN (the data is non-secret and read-only; still, the README says "127.0.0.1 전용"). Leave it, or reject non-loopback hosts unless an explicit flag is given.

### F11 — non-blocking, observations (no change)
- **Hidden-until-reveal behaviour is unchanged.** No frontend code was modified. The served HTML/JS/API equal 8771 except for the banner `div` (C5). T37's 1440 px check found the pre- and post-reveal text identical to 8771 for 3 scenes, and my C7 run showed nothing revealed before a click.
- **T37 attempt-1 failure.** At 390 px the reveal button was outside the viewport. This is a Playwright iframe-scroll limitation. The same outer `scrollIntoView({block:'end'})` already exists in `reports/t34-preview/t34-verify-preview.mjs:22`, so it is not a bundle regression.
- **Python 3.9.** Responses are HTTP/1.0 with one connection per request (stdlib default); fine for this load. `ThreadingHTTPServer` with a read-only `answers` dict and a stateless handler is thread-safe.
- **Games list request outside the cap.** `capture_snapshot.py:92` fetches the games list outside `save()`, so it is not counted in `MAX_REQUESTS` and has no explicit status check. `urlopen` raises on 4xx/5xx; acceptable.

## Mutation testing (temp copies in `/tmp/pzoff-review.ddhV/mut`, worktree untouched)

Against 177ce1b tests (9):

| Mutation | Result |
|---|---|
| M1 drop PA-prefix loop in `needed_paths` | 1 failed, exit 1 ✔ |
| M2 inject banner into embedded frames | 1 failed, exit 1 ✔ |
| M3 remove sha/length check | 1 failed, exit 1 ✔ |
| M4 remove `dist in target.parents` check | 1 failed, exit 1 ✔ |
| M5 key = raw query (no canonicalisation) | 1 failed, exit 1 ✔ |
| M6 no summaries in `needed_paths` | 1 failed, exit 1 ✔ |
| M7 banner nowhere | 1 failed, exit 1 ✔ |
| M8 `/api/live` → 404 instead of 503 | 1 failed, exit 1 ✔ |
| M9 remove file-name check | **9 passed, exit 0 ✘ (survived)** |
| M10 accept a query on non-summary API paths | 1 failed, exit 1 ✔ |

After R1–R6 (11 tests): M9 → 1 failed (exit 1) ✔. M11 (allow repeated params) → 1 failed ✔. M12 (hard-code 39 again) → 1 failed ✔. M1 rechecked → 1 failed ✔.

## Fixes applied in this review (each minimal, tests rerun: 11 passed)

- **R1** `serve_offline.py` docstring: states the games-list exception (F1).
- **R2** `api_key`: a repeated summary param → no saved answer (F5). Test `test_repeated_summary_params_have_no_saved_answer`.
- **R3** `load_snapshot`: file-name check before reading, explicit missing-file stop (F4). Test `test_manifest_cannot_point_outside_api_or_lose_a_file`.
- **R4** `main`: readable port-in-use stop with the next port suggested (F6). Checked under 3.9.6 against a busy 8792.
- **R5** MISSING count from `manifest['scenes']` (F3). The test manifest gains `scenes: 2`. On the real bundle the text is unchanged ("39개").
- **R6** `README-ko.md`: port hint, games-list exception, partial pages when stepping in the new tab, don't use the iPhone copy-link button, no captures bundled (F2, F7).

Not done: no push, merge or deploy. The tarball was not rebuilt or modified. 8771 got 5 GETs and nothing else. Ports 8766/8770 and other worktrees were not touched.
