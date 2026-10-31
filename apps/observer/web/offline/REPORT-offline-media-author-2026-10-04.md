# Offline bundle local video copy — author report (2026-10-04)

Status: `author_self_tested; same_model_review_pending; Codex_review_pending`. Author: Opus 5.5 worker (coordinator v4 cycle-011).
Branch `codex/offline-media-1004`, base `ab2cbfd` (showcase `e9e9d4d` + offline-setup `1d09543`).
Commits: `e1f03dc` feat (Showcase + serve_offline + fetch_media + Python tests + .gitignore + README), `d7e3dae` test (browser smoke; `PZ_WEB_DIST` for showcase-smoke), and this report.

## What changed

- `src/Showcase.tsx`: `?media=local` (exact value) → `<video src="/offline-media/<basename of mediaUrl>">` + visible note "맥북에 저장한 같은 공식 영상 사본으로 재생 중" (`role="note"`). Otherwise the src expression is the same `source.mediaUrl`. Seek/stop, `retryVideo`, reveal, AbortController, API calls untouched; the error link "공식 영상 따로 열기" still points to `source.mediaUrl`. `scene` keeps working with `media=local` (the existing `replaceState` keeps other params).
- `offline/serve_offline.py --media-dir DIR`: requires `DIR/MEDIA-SHA256SUMS` (`<sha256>  <name>` or `<sha256> *<name>`, name `[A-Za-z0-9][A-Za-z0-9._-]*\.mp4`). Start refused on: no sums file, empty list, bad/duplicate/path line, listed file missing or a symlink, sha256 mismatch, any unlisted `*.mp4` (case-insensitive) in DIR. Serves only listed names at `GET/HEAD /offline-media/<name>`, everything else under that prefix → 404 JSON.
  - Range (RFC 9110 §14): one range `a-b`, `a-`, `-n` → 206 + `Content-Range: bytes a-b/size`, end clipped to size−1; unsatisfiable (`a ≥ size`, `-0`) → 416 + `Content-Range: bytes */size`, empty body. **Multi-range, other unit, malformed or `b < a` → ignored, 200 full body** (§14.2 permits ignoring). HEAD ignores Range (range handling is defined for GET only) → 200 headers. `If-Range` honoured only when equal to the strong ETag (`"<sha256>"`), otherwise 200 full. Always `Content-Type: video/mp4`, `Accept-Ranges: bytes`, `Content-Length`. Body streamed in 1 MiB chunks; client disconnects (BrokenPipe/ConnectionReset) are swallowed.
  - Banner line says "영상은 인터넷 또는 맥북 사본(주소 끝 ?media=local)" only when `--media-dir` is given; startup prints a second line with the copy's URL, size and sha256. The first printed URL now uses the bound port (identical unless `--port 0`).
  - Without `--media-dir`: `make_handler(..., media=None)` default; `/offline-media/...` falls through to the old static-file path (404). Existing `test_offline_bundle.py` unchanged and passing.
- `offline/fetch_media.py` (NOT run against the network): reads `mediaUrl` from `showcase-scenes.json`, downloads to `<target>/<name>.part` (exclusive create), checks Content-Length, fsync, refuses if the final file appeared, `os.rename`; on failure deletes only its own `.part`; a pre-existing `.part` or final file → refuse, untouched. Appends `<sha>  <name>` to `MEDIA-SHA256SUMS` via tmp + `os.replace`, refuses a duplicate name. Prints size/sha256/next command. `--dry-run` prints URL + target only (no mkdir, no request). Default target `offline/media/`.
- `.gitignore`: `apps/observer/web/offline/media/`, `apps/observer/web/**/*.mp4`, `apps/observer/web/**/*.mp4.part`.
- `offline/README-ko.md`: fetch (by the coordinator, outside protected recording windows), `--media-dir media`, `/showcase?media=local`, limits.
- `tests/showcase-smoke.mjs`: only `const dist = process.env.PZ_WEB_DIST ?? <old path>` + usage line (same pattern as the phone smoke) so it can run against the /tmp build.

## Commands run (cwd, command → exit, result)

Environment: `OMP_NUM_THREADS=1`, `UV_THREADPOOL_SIZE=1`, `nice -n 10`, one process at a time. Node v25.9.0, Playwright chromium-1243 (headless shell). Python for pytest: `/Users/song/Projects/pitcheezy/.venv/bin/python` (system `python3` 3.9.6 has no pytest; first attempt `python3 -m pytest` → "No module named pytest"). `node_modules` was a temporary symlink to `codex-showcase-1006/apps/observer/web/node_modules`, never added, `unlink`ed at the end.

1. `offline/`: `pytest -q test_offline_bundle.py test_offline_media.py` → exit 1, 68 passed / 1 failed (`bytes=10240-` on a 10240-byte file returned 200 instead of 416: open range past the end was treated as invalid). Fixed in `byte_range` (only an explicit `b < a` is invalid).
2. same → exit 0, **69 passed**.
3. `offline/`: `pytest -q test_fetch_media.py` → exit 0, **9 passed** (after replacing one test URL that would have reached 127.0.0.1:9; it never ran).
4. `apps/observer/web`: `npx tsc -b` → exit 0, no output.
5. `npx vite build --outDir /tmp/pz-offline-media-a1/dist` → exit 0.
6. `PZ_WEB_DIST=/tmp/pz-offline-media-a1/dist node tests/showcase-local-media-smoke.mjs` → exit 0: `showcase local-media smoke: ok {"tmp":"/tmp/pz-offline-media-smoke-xJ0ula","mp4Bytes":702685,"decoded":{"390":"ready","1440":"ready"}}`.
7. Negative control: `git archive e9e9d4d apps/observer/web | tar -x -C /tmp/pz-offline-media-neg1/tree`, symlinked node_modules there, `npx vite build --outDir /tmp/pz-offline-media-neg1/dist` → exit 0 (0 occurrences of `offline-media` in its JS); `PZ_WEB_DIST=/tmp/pz-offline-media-neg1/dist node tests/showcase-local-media-smoke.mjs` → **exit 1 (FAIL as required)**: `locator.waitFor: Timeout 30000ms exceeded … waiting for getByText('맥북에 저장한 같은 공식 영상 사본으로 재생 중')` at line 117 (first local-mode check; direct HTTP checks and pass A "default unchanged" passed on the old build). Run dir `/tmp/pz-offline-media-smoke-obB1n2`.
8. `PZ_WEB_DIST=/tmp/pz-offline-media-a1/dist node tests/showcase-smoke.mjs` → exit 0: `showcase smoke: ok (synthetic WAV media + timeline/reveal fixtures, desktop + phone)`.
9. `PZ_WEB_DIST=/tmp/pz-offline-media-a1/dist node tests/showcase-phone-reveal-smoke.mjs` → exit 0: 47 `ok` lines, `showcase phone reveal smoke: ok (synthetic fixtures; 390x844, 390x664, 1440x900)`.
10. `offline/`: `pytest -q test_offline_bundle.py` alone → exit 0, **31 passed** (existing file, unchanged).
11. `offline/`: `pytest -q .` → exit 0, **78 passed** (31 bundle + 38 media + 9 fetch).
12. `offline/`: `python3 fetch_media.py --dry-run --target /tmp/pz-offline-media-a1/dryrun-media` → exit 0, printed the mediaUrl and target path; target dir not created. No other run of `fetch_media.py`.

Logs: `/tmp/pz-offline-media-pytest{1,2,3}.log`, `/tmp/pz-offline-media-a1/{tsc,build,local-smoke,showcase-smoke,showcase-phone-reveal-smoke,pytest-bundle,pytest-all}.log`, `/tmp/pz-offline-media-neg1/{build,neg-smoke}.log`. After the edit at step 1→2, the only later source change was README wording (docs only); the built dist matches `Showcase.tsx` at `e1f03dc` (sha256 `2075480df9f35bc320f7226db6b80b2870bee163409dc008bd2822041c89c244`).

## Build checked

- dist: `/tmp/pz-offline-media-a1/dist`
  - `index.html` sha256 `6f935e899a1b3710a3a3cab8b76e100440cdf1649c6ef3b500c54cef52363603`
  - `assets/index-OhLqUnP9.js` sha256 `9944ccb767b232d43392e1429414419a18f0090d77b2a9ed62284d6884bc1369`
  - `assets/index-BGk569ec.css` sha256 `c624c910496a1dd0b9c56abb372fdd4e3e6be4a1cf41422e7a50c24ba026e1db`
- negative-control dist (`e9e9d4d`): `/tmp/pz-offline-media-neg1/dist`, `index.html` `c66db122c586d07d6d060a415003a291e2f6b9a1dc0820d278a100dd424ea499`, `assets/index-D0KUefQy.js` `ab05aeda9e1cc2c202c52c53b358a52b66de5e5bcb9e9b1cf0fbdedf5051fec2`.

## What the browser smoke checks

Real `serve_offline.py --port 0 --media-dir` on a synthetic snapshot (timeline + reveal 2/3) and a 702,685-byte synthetic H.264/AAC mp4 (ffmpeg testsrc+sine, 160×90, 66 s, faststart) named like the official file.
- Direct HTTP: `Range: bytes=0-1` → 206 `bytes 0-1/702685`; `/offline-media/unknown.mp4` → 404.
- A (default, `?scene=1`, 1440): no local note; src = MLB URL; the MLB request is intercepted and aborted (never sent); zero `/offline-media/` requests; error link = MLB URL.
- B (`?media=local&scene=1`, 1440 and 390): note visible; no media request before "공식 경기 영상 불러오기"; src = `/offline-media/<name>`; every `/offline-media/` response 206 with `Content-Range … /702685`; zero MLB requests. Headless Chromium decoded the synthetic H.264 at both widths ("ready"), so it also checked: metadata seek to scene 1 release−3 s, clip plays, clip stops after release+4 s. Reveal: no `.wa-reveal` and no reveal request before the click; after the click, reveal 3 only. Switching to scene 0 keeps `media=local`, `scene=0`, local src, and seeks to scene 0 release−3 s. No page errors, no off-origin requests.
- C (`?media=local`, local file aborted): error alert, "영상 다시 불러오기" present, "공식 영상 따로 열기" = MLB URL, zero MLB requests.

## Not checked / limits

- Safari, iPhone: not checked. Real MLB file (436 MB) playback from the local copy: not checked (never downloaded; no request to mlb.com was made by any command here).
- Python tests cover range/416/HEAD/If-Range/traversal/refusals on a 10,240-byte synthetic body; large-file streaming was not measured.
- `--media-dir` refuses a symlinked media file by design (copy the file instead).
- If-Range with an HTTP-date is never matched (→ full 200), which is allowed but means no partial resume by date.
- Rules followed: no ssh/newsrv, no GPU, no ports 8766/8770/8771, no push/merge/deploy, no 2026 data, no directory deletion; temporary dirs left in `/tmp/pz-offline-media-*`.
