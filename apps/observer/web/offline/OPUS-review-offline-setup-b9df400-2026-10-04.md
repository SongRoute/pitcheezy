# Review — offline capture `--with-setup` (f3c9da0..b9df400)

`same_model_review; Codex_review_pending`. Reviewer: Claude Opus 5.5, a different session from the author. Date: 2026-10-04.
Scope: `git diff f3c9da0 b9df400` (`capture_snapshot.py`, `test_offline_bundle.py`). Also read, without changing them: `serve_offline.py`, `README-ko.md`, the two earlier reviews in this folder, `WatchAlong.tsx` and `Game.tsx` on `codex/catcher-integ-1004` (ed52840), and `apps/observer/backend/observer_app/main.py:421-428` plus `setup_estimates.py`.
Nothing was fetched from 8771 or any other port. No server was started except the in-process test servers on 127.0.0.1:0.

## Disposition

**Ready to capture a v3d bundle** with `--all-decisions --with-setup`. No blocking findings.

The code was correct as written. The weakness was the tests: `main()` was never run, so three of my first five mutations survived (cap arithmetic, setup path name, flag turned into a no-op). I added tests that run `main()` against a fake 8771. All six mutations are now caught (commit e50f500). I also added the README line (b546077).

When capturing v3d, check that `setup_rows` in the manifest is a number, and that the copied `dist/` is the ed52840-or-later build. An older build never requests the setup answer, so it draws no mitt.

## Answers to the questions

### 1. Correctness

- **Saved byte for byte: yes.** `save()` writes the `fetch()` body unchanged, and `check_setup` only parses it (`capture_snapshot.py:104-117, 124`). The new test confirms that `load_snapshot` returns exactly the fake server's bytes.
- **Saved under the path the page requests: yes.**
  - The page calls `get(`/watch/${gamePk}/setup`)` with no query string (`WatchAlong.tsx:249`; `get()` adds the `/api` prefix at `:43`).
  - The backend route is `/api/watch/{game_pk}/setup` (`main.py:421`).
  - `serve_offline.api_key` passes through a non-summary path that has no query (`serve_offline.py:65`), so the lookup key is `/api/watch/849843/setup`.
  - `snapshot_file` maps it to `api/watch__849843__setup.json`. That name is distinct from the timeline, reveal and summary files (tested).
- **`load_snapshot` handles the extra file correctly.**
  - The path-escape check (`snapshot_file(path) == file`), the existence check and the sha256/byte checks apply to the new entry like any other.
  - The all-decisions completeness check only requires that every reveal and summary is present. Extra entries are allowed, so the setup file is neither required nor rejected.
- **Request-cap arithmetic is right.**
  - Before: `len(paths) + 1`, where 1 = the timeline.
  - Now: `len(entries) + len(paths)`, where `entries` already holds the timeline and, with the flag, the setup answer. This now matches `save()`'s own guard (`len(entries) >= cap` before each fetch).
  - The unsaved `/api/watch/games` fetch is still left out of the count, as before (earlier review, finding 3).
  - For 849843: 1 + 1 + 524 = 526 saved entries, under the cap of 800.
- **`setup_rows` breaks no reader.**
  - `serve_offline` reads only `game_pk`, `scenes`, `decisions`, `top`, `coverage`, `captured_kst` and `files`. Extra keys and `null` are ignored.
  - The tests build their own manifests.
  - The README did not mention the field before; it does now.

### 2. Honesty

- **Another game's mitt cannot appear.**
  - Capture: `check_setup` refuses a body whose `game_pk` differs (`capture_snapshot.py:66`).
  - Page: it ignores a body with a different `game_pk` and keys each mitt by `game:at_bat:pitch` (`WatchAlong.tsx:250-253, 344-345`).
  - Offline server: a request for another game's setup gets a 404, which the page swallows (tested).
- **No mitt appears before a reveal or for an unsaved pitch.** The page draws a mitt only when `reveal.index === decision.index` (`WatchAlong.tsx:344`). In v1/v2 (scene-only) bundles, a pitch without a saved reveal never shows one.
- **`/game` page: no mitt offline.** `Game.tsx:230` also requests the setup answer, but it draws a mitt only for pitches that came through the replay stream. Offline that stream is 404/503, so nothing is drawn.
- **The one real way the two can differ: a stale copy.**
  - 8771 re-reads `<game>.jsonl` whenever its modification time changes (`setup_estimates.py:46-48`). If the video module's estimates are replaced after the capture (for example a reviewed M3 version), the offline bundle still shows the old mitts.
  - If the setup files are moved or the T7 drive is unplugged, 8771 returns 404 or 503 and draws no mitt, while the bundle still draws the saved ones.
  - The yellow line at the top shows the save time, and the saved body carries `source.sha256`/`commit`, so any drift can be found. The README now says this.

### 3. Tests and mutations

See the commands section below. Base run: 26 passed. After my test commit: 31 passed. All mutations were reverted (`git diff` was empty before each commit).

## Findings

1. **Non-blocking (test gap), fixed in e50f500.**
   - Problem: `test_offline_bundle.py:235-261` (b9df400) tests `check_setup` and serving, but never runs `capture_snapshot.main()`. It also builds the setup path with `setup_path()` on both the write side and the read side.
   - Consequence: these mutations passed all 26 tests:
     - M2: pre-check reverted to `len(paths) + 1 > cap`
     - M3: `setup_path` returns `/setups`. The page would then 404 silently and the mitt would vanish, which is the exact failure this commit is meant to fix.
     - M4: `if args.with_setup:` changed to `if False:`
   - Fix: `run_capture()` runs `main()` with a fake `fetch`. Its site map writes the page's path `/api/watch/{GAME}/setup` out literally. Four new tests:
     - with the flag, the answer is saved byte for byte and `setup_rows == 1`
     - without the flag, setup is never fetched and `setup_rows is None`
     - a 404 or another game's answer stops the capture and no manifest is written (parametrized)
     - a cap equal to timeline + all pitches passes without the flag and stops with `exceed cap` when the flag adds one more request
2. **Non-blocking (docs), fixed in b546077.**
   - Problem: the README did not mention `--with-setup`, so whoever makes the bundle (the operator) would not know the mitt depends on it.
   - Added under "무엇이 같고 무엇이 다른가":
     > - 포수 미트(공을 공개한 뒤 존 그림의 세로선 "미트"): `--with-setup`으로 만든 묶음에만 들어 있음. manifest.json의 `setup_rows`가 숫자면 들어 있고 `null`이면 없음. 없으면 미트 표시만 조용히 빠지고 나머지 화면은 같음. 들어 있어도 저장 시각의 영상 추정치 사본이라, 그 뒤 맥미니 쪽 추정치가 바뀌면 8771과 다를 수 있음.
   - Added to "다시 만들기":
     > 포수 미트까지 담으려면 `--with-setup`도 붙임: GET 1번 추가(v3면 527번), 셋업 자료가 HTTP 200이 아니거나 다른 경기 것이면 저장을 멈춤.
3. **Non-blocking, suggestion, not changed.**
   - Problem: `serve_offline.py:152` prints no setup status at start. The original failure mode (an ed52840 build with a bundle that has no setup answer, so the mitt silently disappears) is only visible by reading `manifest.json`.
   - Suggested diff:
     ```diff
     -    print(f'http://{args.host}:{args.port}/showcase  ({len(answers)} saved answers, {manifest["captured_kst"]})',
     +    mitt = 'no setup' if manifest.get('setup_rows') is None else f'setup rows {manifest["setup_rows"]}'
     +    print(f'http://{args.host}:{args.port}/showcase  ({len(answers)} saved answers, {mitt}, {manifest["captured_kst"]})',
     ```
   - Not applied, because the author kept `serve_offline.py` unchanged in this step.
4. **Non-blocking, pre-existing behaviour, not changed.** `--with-setup` fails closed on everything, but some failures end in a traceback instead of the one-line message:
   - `urllib.request.urlopen` raises `HTTPError` on 4xx/5xx, so a real 404 or 503 from 8771 never reaches the `status != 200` branch (`capture_snapshot.py:108`).
   - A JSON body that is not an object raises `AttributeError` in `check_setup` (`view.get`).

   Either way the capture stops with no manifest, and the server refuses a folder without one. Optional hardening:
   ```diff
   -    view = json.loads(body)
   -    if view.get('game_pk') != game_pk or not isinstance(view.get('pitches'), list):
   +    view = json.loads(body)
   +    if not isinstance(view, dict) or view.get('game_pk') != game_pk or not isinstance(view.get('pitches'), list):
   ```
5. **Info.** `setup_rows` counts every line in the answer, including pitches marked `unavailable` (no mitt drawn). It means "rows saved", not "mitts drawn". The README wording only uses it as present/absent, which is accurate. If a drawn count is wanted, add `setup_with_x = coverage.with_plate_x` from the body.
6. **Info.** The setup answer is fetched before the missing-scenes check and the cap pre-check. A refused capture therefore makes up to 2 GETs (timeline + setup) instead of 1 and leaves partial files without a manifest. That is harmless and matches how the other refusals already behave: the next run needs an empty folder anyway (`capture_snapshot.py:98-99`).

## Commands and outputs

Base test, as asked (b9df400):
```
$ cd apps/observer/web/offline && OMP_NUM_THREADS=1 perl -e 'alarm 300; exec @ARGV' nice -n 10 /Users/song/Projects/pitcheezy/.venv/bin/python -m pytest -q test_offline_bundle.py
..........................                                               [100%]
26 passed in 4.25s
exit=0
```

Mutations on `capture_snapshot.py`. Each was a `sed`/python edit, then the same pytest command, then `git checkout -- capture_snapshot.py`.

| # | Mutation | b9df400 tests (26) | + e50f500 tests (31) |
|---|---|---|---|
| M1 | drop the `game_pk` check in `check_setup` | 1 failed | 2 failed |
| M2 | cap pre-check back to `len(paths) + 1 > cap` | **26 passed (survived)** | 1 failed |
| M3 | `setup_path` → `/api/watch/{game_pk}/setups` | **26 passed (survived)** | 2 failed |
| M4 | `if args.with_setup:` → `if False:` | **26 passed (survived)** | 4 failed |
| M5 | drop the `pitches` list check | 1 failed | (not re-run) |
| M6 | save setup after the reveals (outside the pre-check) | (not run) | 1 failed |

After the test commit:
```
31 passed in 4.13s
```
`git diff --stat` was empty after each restore. Final `git status`: clean apart from this file before it was committed.

## Commits added by this review

- `e50f500` test(observer-web): offline capture main() with fake 8771 — setup path literal, cap counts setup, fail-closed setup
- `b546077` docs(observer-web): offline README — --with-setup, setup_rows, frozen mitt caveat
- this review file (next commit)

No push, no merge, no deploy, no network capture.
