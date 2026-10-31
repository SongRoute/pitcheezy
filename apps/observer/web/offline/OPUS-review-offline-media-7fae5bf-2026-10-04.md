# Opus 5.5 review — offline local video copy (`ab2cbfd..7fae5bf`), 2026-10-04

Status: `same_model_review; Codex_review_pending`. Reviewer: separate Opus 5.5 reviewer (not the author). No other model was called.
Worktree `codex-offline-media-review-1004`, branch `codex/offline-media-review-1004`. Reviewed `7fae5bf`; review fix commits on top: `d3f300f`, `be1a156`, `be7871e`, `b305cd4` (+ this file).

**Verdict: no BLOCKER, no MAJOR.** Default behaviour (no `media=local`) is unchanged. Range serving follows RFC 9110 §14 for the cases Safari and Chromium send. The negative control fails for the right reason. Four MINOR issues: two fixed here, one fixed in tests, one documented in the README (the code change belongs to the PM). Safari, iPhone and the real 436 MB file are still unchecked. The author's report says so too.

## Commands and exit codes (reviewer runs)

Settings: `OMP_NUM_THREADS=1 UV_THREADPOOL_SIZE=1`, `nice -n 10`, one process at a time. pytest ran under `/Users/song/Projects/pitcheezy/.venv/bin/python`. The smoke's `serve_offline.py` ran under system `python3` 3.9.6. `node_modules` was a temporary symlink to `codex-showcase-1006/apps/observer/web/node_modules`, never added, `unlink`ed at the end. Logs are in `/tmp/pz-offline-media-review-r1/`.

1. `offline/`: `pytest -q .` at `7fae5bf` → exit 0, **78 passed** (31 bundle + 38 media + 9 fetch). The author's count is reproduced.
2. repo root: `pytest apps/observer/web/offline -q` → exit 0, 78 passed (works from the root too).
3. `npx tsc -b` → exit 0.
4. `npx vite build --outDir /tmp/pz-offline-media-review-r1/dist` → exit 0. sha256 values match the author's report exactly: `index.html` `6f935e89…63603`, `assets/index-OhLqUnP9.js` `9944ccb7…c1369`, `assets/index-BGk569ec.css` `c624c910…e1db`.
5. `PZ_WEB_DIST=<4> node tests/showcase-local-media-smoke.mjs` → exit 0, `decoded {"390":"ready","1440":"ready"}`, mp4 702,685 bytes.
6. `PZ_WEB_DIST=<4> node tests/showcase-smoke.mjs` → exit 0.
7. `PZ_WEB_DIST=<4> node tests/showcase-phone-reveal-smoke.mjs` → exit 0 (390x844, 390x664, 1440x900).
8. **Negative control (fresh rebuild, separate from the author's):** `git archive e9e9d4d apps/observer/web` → `/tmp/pz-offline-media-review-neg/tree`, then `vite build --outDir /tmp/pz-offline-media-review-neg/dist` → exit 0. The JS has 0 `offline-media` occurrences. Its sha256 matches the author's negative build (`index.html c66db122…a499`, `index-D0KUefQy.js ab05aeda…ec2`). Running the current smoke against it → **exit 1**: `TimeoutError … waiting for getByText('맥북에 저장한 같은 공식 영상 사본으로 재생 중')`. That is the right reason: the old build has no local mode. The direct-HTTP checks and pass A (default) still pass on the old build, as they should.
9. **Extra mutation (reviewer):** `git archive HEAD` → `/tmp/pz-offline-media-review-neg/mut`, with `serve_offline.py` changed to `span = None` (Range ignored). The smoke against dist 4 → **exit 1**: `AssertionError 200 !== 206` at smoke line 76. The 206 check is a real check.
10. Probe script `/tmp/pz-offline-media-review-probe/probe.py` (3 MiB random body, real handler):
    - range across the 1 MiB chunk boundary → 206, correct bytes;
    - full multi-chunk GET → 200, correct bytes;
    - `bytes=-3` → 206;
    - 3 RST disconnects mid-body plus 1 close before reading → server still answers 206, **0 tracebacks** → exit 0.
11. Hash throughput, `file_sha256` on a 64 MiB synthetic file in /tmp: **2761 MiB/s**, warm page cache. For 436 MB that is ≈ **0.15 s** warm. A cold first read of the real file from SSD or T7 is 미측정. It is bounded by disk read speed, not by SHA-256.
12. After the fixes: `offline/`: `pytest -q .` → exit 0, **81 passed** (+1 fetch, +2 media). `showcase-local-media-smoke.mjs` against dist 4 → exit 0. The new tests' own negative checks:
    - `test_stale_sum_line_is_refused_before_download` **fails** with the precheck line removed (`/tmp/pz-offline-media-review-neg/fetchneg`);
    - `test_client_disconnect_mid_body_is_quiet` **fails** on the Traceback assertion when the `except (BrokenPipeError, ConnectionResetError)` clause is mutated away (`/tmp/pz-offline-media-review-neg/servneg`);
    - 5 repeated runs of the two new media tests: all passed.

`Showcase.tsx` was not changed by the review, so dist 4 still matches the reviewed source.

## 1. Default behaviour without `media=local` — Showcase.tsx, line by line

The diff has 3 hunks and nothing else in the file changed.
- `src/Showcase.tsx:15` adds `LOCAL_MEDIA`, a module-level constant. `src/Showcase.tsx:29` adds `localMedia` as a one-time `useState` initializer. It is appended after the existing hooks, so hook order is stable.
- `:112` sets `src={loaded ? (localMedia ? LOCAL_MEDIA : source.mediaUrl) : undefined}`. With `localMedia` false this is the old expression. `onLoadedMetadata`, `onError`, `onTimeUpdate`, `onEnded`, `startClip`, `retryVideo`, `clipFinished` are unchanged. So are the timeline `AbortController` effect (`:38-47`) and the reveal iframe.
- `:123` adds the note, rendered only when `localMedia` is true. `:125` (error link → `source.mediaUrl`) is unchanged.
- `:49-52` uses `replaceState`, which keeps the other params, so `media=local` survives scene changes. The smoke checks this at `tests/showcase-local-media-smoke.mjs:120-121,154-155`.
- No API call or reveal timing changed.

## 2. Range handling (RFC 9110 §14) — `offline/serve_offline.py`

- `byte_range` `:106-118`:
  - `a-b`, `a-`, `-n` → 206. An end past the file is clipped (§14.1.2). A suffix longer than the file → whole file (§14.1.3).
  - `a ≥ size` or `-0` → 416 (§14.1.1 satisfiability). The 416 response (`:182-187`) has `Content-Range: bytes */size` (§15.5.17).
  - `b < a`, a multi-range, another unit or bad syntax → Range ignored, 200 full. §14.2 allows this. Multi-range → 200 is documented in the docstring `:13-15` and the README.
  - The unit match is case-insensitive (correct: range units are case-insensitive).
- 206 headers `:189-196`: `Content-Range bytes a-b/size`, `Content-Length` = span length, the same `ETag`/`Cache-Control` a 200 would send (§15.3.7), `Content-Type: video/mp4`, `Accept-Ranges: bytes`.
- HEAD (`:180`, `:198`): Range is ignored, which is correct since §14.2 defines range handling for GET only. Headers only.
- If-Range `:180`: only an exact strong-ETag match gives a range. A weak tag or HTTP-date → 200 full (§13.1.5 allows this).
- Streaming `:200-208`: 1 MiB chunks, never the whole file. `BrokenPipeError`/`ConnectionResetError` are swallowed. Verified by probe 10 and the new test.
- Safari: its first probe is `bytes=0-1` (tested). That Safari plays from this server is **unchecked** (see NOTE N1).

## 3. sha256 manifest — `load_media` `:79-103`

- Refuses to start when:
  - the sums file is missing;
  - a line is malformed;
  - a name is not `SAFE_NAME` (no `/`, no leading `.`, so no traversal);
  - a name is a duplicate;
  - a listed file is missing or a symlink;
  - a sha256 does not match;
  - the list is empty;
  - an unlisted `*.mp4` is present (case-insensitive).
- Serving: `:213-215` uses an exact dict lookup on the raw path segment. `../`, `%2e%2e`, sub-paths, `.part` and `MEDIA-SHA256SUMS` all → 404 (tested `test_offline_media.py:113-117`).
- Without `--media-dir`, `/offline-media/…` falls through to the old static path → 404 (tested `:125-137`).
- Startup cost: one hash pass. ≈0.15 s for 436 MB at the measured warm-cache throughput; cold-disk time 미측정.

## 4. `fetch_media.py`

- Never overwrites:
  - an existing final file (`:36`);
  - an existing `.part` (exclusive `open(part, 'xb')` `:42`);
  - a final file that appeared during the download (`:58`).
- Rename only after a full download: `Content-Length` check `:54`, `fsync`, then `os.rename` `:60`. On any exception it deletes only its own `.part` (`:61-63`).
- Writes `MEDIA-SHA256SUMS` through a tmp file + `os.replace`. `--dry-run` makes no request and creates no folder (tested).
- The tests use a 127.0.0.1 `http.server` only. The non-mp4 test is refused before connecting. No test touches the internet.

## 5. Findings

- **MINOR M1 — fixed `d3f300f`.**
  - Problem: `fetch_media.py` checked for an existing `MEDIA-SHA256SUMS` line for the name only in `add_sum`, after the ~436 MB download and rename. A stale line (for example a deleted earlier copy) left a new file with no new sum line. A rerun then refused with "already there".
  - Fix: `main` runs the same check (`sum_lines`, `fetch_media.py:67,96`) before any request. Test `test_stale_sum_line_is_refused_before_download` added; it fails without the fix.
- **MINOR M2 — fixed in tests `be1a156`.**
  - Problem: the author's claims "streamed in 1 MiB chunks" and "client disconnects swallowed" were not exercised. The 10,240-byte test body fits in one read and in the socket buffers.
  - Fix: added `test_ranges_span_chunks` (CHUNK monkeypatched to 1000, range across chunks) and `test_client_disconnect_mid_body_is_quiet` (8 MiB body, 3 RST disconnects, server still answers 206, no Traceback). The second one fails when the except clause is removed. No server code changed.
- **MINOR M3 — documented `b305cd4`; code change for PM to decide.**
  - Problem: `?media=local` lives only in the address bar. The in-app links do not carry it: `src/Showcase.tsx:104` "← 경기 목록" → `/watch`, then `src/WatchAlong.tsx:212` "영상과 나란히 비교 ▸" → `/showcase`. A presenter who goes through the list falls back to the internet video, which fails offline.
  - Done here: README line telling the presenter to use browser back or re-add `?media=local`.
  - Not done: propagating the param (e.g. via `sessionStorage`) would touch default-mode links and needs its own review.
  - Suggested disposition: accept the README note for 10-06; consider the code change afterwards.
- **MINOR M4 — fixed `be7871e`.**
  - Problem: in `tests/showcase-local-media-smoke.mjs`, `chromium.launch` ran before `try`. A launch failure left the spawned `serve_offline.py` running.
  - Fix: launch moved inside `try`, `browser?.close()` in `finally`.
- **MINOR M5 — not fixed (UI copy).**
  - Problem: the note at `src/Showcase.tsx:123` says "…사본으로 재생 중" from page load, before anything plays. It stays visible next to the error alert when the local file is unreachable (smoke pass C shows both).
  - Suggestion: wording such as "맥북에 저장한 같은 공식 영상 사본을 씁니다". This is a product-copy choice and the smoke asserts the exact text, so it is left to the PM.
- **NOTE N1 (Safari) — needs a real run.** `BaseHTTPRequestHandler` answers HTTP/1.0, so each range is a new connection. Chromium is fine with this. Safari's media loader against this server is unchecked. The README now says to rehearse once in Safari with the real file before the demo (`b305cd4`). Suggested disposition: rehearsal item for the coordinator, outside protected windows.
- **NOTE N2.** There is a small window between `final.exists()` (`fetch_media.py:58`) and `os.rename` (`:60`): POSIX rename would replace a file created in that window. `os.link` + unlink would close it, but may fail on exFAT (the T7 drive). Only one fetch runs at a time, so this is acceptable as is.
- **NOTE N3.** Size and sha are fixed at startup. Replacing the file while the server runs would give wrong `Content-Length`s; restart after any change. This matches how the snapshot is handled.
- **NOTE N4.** There is no independent expected sha256 for the official file (trust on first download). Suggestion: record the printed sha256 in PM state when the coordinator downloads.
- **NOTE N5.** `LOCAL_MEDIA` (`src/Showcase.tsx:15`) is evaluated in default mode too. It would throw at import if `mediaUrl` ever became relative. Today it is absolute; harmless.
- **NOTE N6.** A disconnect during the header write is not caught. Probe 10 saw no traceback. Even if one happens, `socketserver` logs it and keeps serving.

## 6. Honesty of README and report

- Neither claims Safari, iPhone or real-file playback. Both list them as unchecked (README `offline/README-ko.md` "확인 못 한 것"; report "Not checked / limits").
- README step 3 says "Safari에서 … 열기" as an instruction. The new rehearsal line makes clear this is not a verified path.
- Every number in the author's report I re-ran matches:
  - 78 tests;
  - smoke ok at both widths;
  - all build sha256 values, including the negative-control build;
  - negative control failing at smoke line 117.
- Private copy: `git check-ignore` ignores `offline/media/MEDIA-SHA256SUMS`, `offline/media/x.mp4`, `offline/x.mp4.part` and `web/public/a.mp4`. `git ls-files` shows no tracked `.mp4` or `SHA256SUMS`.

## Rules followed

- No request to mlb.com and no download of the video; `fetch_media.py` was not run (not even `--dry-run`).
- No ssh/newsrv, no GPU/MPS, no ports 8766/8770/8771.
- No push, merge or deploy; no team messages.
- No `rm -rf` or directory deletion; temporary dirs are left in `/tmp/pz-offline-media-review-*`.
- No 2026 data reads.
- No permission denials.
