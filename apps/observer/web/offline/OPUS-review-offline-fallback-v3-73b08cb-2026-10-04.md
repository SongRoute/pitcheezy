# Opus 5.5 review — offline fallback `--all-decisions` (bundle v3), 73b08cb

`same_model_review; Codex_review_pending` — reviewer: Claude Opus 5.5 (same model as the author, separate session). No other models were called and no subagents were spawned.

- Target: `git diff 3ed820b 73b08cb` (`apps/observer/web/offline/{capture_snapshot.py, serve_offline.py, test_offline_bundle.py, README-ko.md}`)
- Branch: `codex/offline-fallback-v3-review-1004`. The review fixes are in the commit that adds this file.
- Bundles were read only. The v3 bundle (sha256 `57de17fd…6041`, re-hashed and matching) and the v2 bundle (`6dbce692…0c32`) were extracted into `/tmp/pzrev.2RSS`. Neither bundle was modified or rebuilt.

## Disposition

**Accept with the non-blocking fixes below (applied).** No blocking findings. Results:
- v3 covers every API request the watch page makes for pitch navigation.
- v1/v2 behaviour is byte-identical to 3ed820b.
- All 526 v3 answers are served unchanged. 4 of them were spot-checked against 8771 and matched.
- The new tests catch the mutations tried.

The fixes make the completeness check fail cleanly on bad input instead of crashing, and stop it from trusting an understated pitch count.

## Findings

1. **Non-blocking (robustness), fixed.** In `serve_offline.py:44-48` (73b08cb), the `isinstance(n, int)` check ran after `range(n)`. As a result:
   - `decisions` set to `"262"`, `262.0` or `null` crashed with a `TypeError` traceback instead of the clear refusal. A missing `decisions` key crashed with `KeyError`. In each case the server still did not start (fail-closed), but the error message was unclear.
   - `decisions: true` was accepted, because `bool` is a subclass of `int`. The server loaded 526 answers and the banner said "공 True개 전체".

   Measured on a copy of the real v3 manifest:
   - `'str' object cannot be interpreted as an integer`
   - `float …`
   - `NoneType …`
   - `KeyError: 'decisions'`
   - `True` → LOADED

   Fix: use `manifest.get('decisions')` and check `type(n) is not int or n <= 0` before `range()`. The error message is `all-decisions snapshot has no valid decision count: …`.

2. **Non-blocking (honesty of the claim), fixed.** The check trusted the manifest's `decisions` and never compared it with the saved timeline. A v3 manifest with `decisions: 39` loaded fine and the banner said "공 39개 전체", while the watch page navigates over all `len(timeline.decisions)` pitches. The reverse case is a manifest that matches its files but undercounts the timeline. That bundle would pass the check, yet ←/→ beyond N would 404 under a banner that says "전체".

   Fix: parse the sha-verified `/api/watch/{game}` answer and require `len(decisions) == n` (`serve_offline.py:47-49`). The real v3 bundle passes (262 == 262).

3. **Non-blocking (docs), fixed.** `README-ko.md:24` said "GET 525번". The real count is 526:
   - 525 come from `save()`: the timeline plus 262 reveals plus 262 summaries.
   - 1 is the unsaved `fetch` of `/api/watch/games`, which is later stored filtered.

   The manifest has 526 files. The README now says "GET 526번(… + 경기 목록)" and also mentions the new timeline-count refusal.

4. **Non-blocking (test gap), fixed.** No test asserted the v1/v2 banner wording. Mutation M9 (v2 scope text changed) passed all 15 original tests. I added an assertion to `test_pages_carry_saved_copy_line_except_embedded_frame` (`test_offline_bundle.py:114`): a manifest without `coverage` shows "경기 1 시연 장면만" and never "전체".

5. **Non-blocking, not changed.** The check uses `manifest['top']`, not the frontend's `TOP = 5` (`WatchSummaryPanel.tsx:23`). A capture made with a different `TOP` would carry a matching manifest and files, so it would load, but the page would request `top=5` and get 404s. This can't happen today: `capture_snapshot.TOP` is pinned to 5 and the v3 manifest says 5. A mismatch between the manifest's `top` and the files is now tested (`test_all_decisions_top_mismatch_does_not_start`). Optional hardening: `if top != TOP: raise SystemExit(...)`, importing `TOP` from `capture_snapshot`.

6. **Non-blocking, pre-existing, out of scope.** The offline picker (`/watch`) still shows two links for 849843 (`WatchAlong.tsx:188-190`):
   - "한 화면으로 다시 보기" (`/game?…&replay=0`), which requests `/api/replay/849843/outline` and `/api/watch/849843/setup`
   - "실제 속도로 다시 보기" (`/live?…&replay=0`)

   Offline, these return 404 ("…경기 849843의 공 262개만 저장돼 있어요.") or 503 (`/api/live/…`). The 404 text is accurate and nothing is made up. The README promise is limited to ←/→ on the watch page, which holds. The same links behaved this way in v1 and v2.

7. **Info.** `/usr/bin/python3` (3.9.6) has no `pytest`, so the requested command cannot run there unchanged (exit 1, `No module named pytest`). Nothing was installed. Instead, the same 24 test functions ran under 3.9.6 through a throwaway stub (`/tmp/pzrev.2RSS/stub/pytest.py`, not committed): 24 passed. The server itself ran on 3.9.6 with both real bundles.

## Check 1 — coverage of watch-page requests

Reading the frontend:
- `WatchAlong.tsx:261` requests `reveal/{i}`:
  - from `doReveal` (button, →, space, retry) with `i = index`;
  - from the plate-appearance strip (`:268`) for `i ∈ [paStart, index)`.
- `index` is always clamped to `0..N-1`: by `step` (`:277`), the `#i=N` hash handler (`:280`, `target < length`) and the initial load (`:215`).
- `WatchSummaryPanel` is mounted only when `reveal` exists (`:327`), with `through = index ≥ 0` and `top = 5` (`WatchSummaryPanel.tsx:23,35`).
- `through=-1` exists only as a valid prop value; no caller passes it. `WatchSummaryPanel` has no other user (grep).
- The other requests are `/api/watch/{pk}`, which is saved, and, in the picker, `/api/watch/games` (saved, filtered) and `/api/live/games` (503 by design).
- `Showcase.tsx` embeds the same page with `&embed=1#i=N`.

So `reveal/0..N-1` plus `summary?through=0..N-1&top=5` covers every per-pitch request. Live check on 8792:
- `summary?top=5&through=3` and `summary?through=3` return 200 (canonicalised).
- `through=-1`, `top=4`, `reveal/262` and `/setup` return 404 with the v3 text.

## Check 3 — backward compatibility (v2)

On the v2 snapshot, I compared this branch's server (8793) with the v2 bundle's own 3ed820b `serve_offline.py` (8794; both are byte-identical to 3ed820b). I sent 188 GETs:
- all 178 saved paths;
- the 404, 503 and static paths;
- `/showcase`, `/watch`, and `/watch?embed=1`.

All 188 responses were identical: 0 differences. The banner says "… 경기 849843 시연 장면만 …". The 404 says "시연 장면 39개만 저장돼 있어요."

## Check 4 — honesty / bundle contents

- The v3 manifest declares `coverage: all_decisions` and `decisions: 262`, with 526 files: 262 reveals, 262 summaries, the timeline, and the filtered games list.
- The timeline has 262 decisions with indices exactly 0..261.
- The bundled `capture_snapshot.py` and `serve_offline.py` are byte-identical to 73b08cb.
- All 178 v2 answers are present in v3 with the same sha256.
- 8792 served all 526 answers byte-identical to the manifest sha256.
- Port 8771: 4 read-only GETs (`reveal/261`, `summary?through=261&top=5`, `reveal/130`, `summary?through=130&top=5`). All returned 200 and matched the bundle sha256.
- The banner says "경기 849843 공 262개 전체". The 404 says "경기 849843의 공 262개만 저장돼 있어요." Both are accurate.
- Evidence `t42-offline-check.json` agrees: the walk over 11–21 revealed every pitch and loaded every summary, including pitches not saved in v2. Its `snapshot-manifest.json` is byte-identical to the bundle's.

## Mutations (venv pytest; each restored afterwards)

Against 73b08cb as written (restored with `git checkout -- <file>`):

| # | mutation | result |
|---|---|---|
| M1 | completeness block disabled (`if False:`) | 1 failed (`…missing_an_answer_does_not_start`), exit 1 |
| M2 | banner always "시연 장면만" | 1 failed (`…serves_every_pitch_and_says_so`), exit 1 |
| M3 | `missing_text` always v2 text | 1 failed (same), exit 1 |
| M4 | `all_decision_paths` drops last reveal | 3 failed, exit 1 |
| M5 | `lacking` ignored | 1 failed, exit 1 |
| M6 | `all_decisions()` true without `coverage` | 6 errors (v2 fixture), exit 1 |

Against the fixed code (restored from a /tmp copy, `shasum` checked):

| # | mutation | result |
|---|---|---|
| M7 | type check removed | 6 failed (all bad-count params), exit 1 |
| M8 | timeline cross-check removed | 2 failed (`decisions` 4 and 6), exit 1 |
| M9 | v2 banner wording changed | 1 failed (new assertion; this mutation **survived** the original 15 tests), exit 1 |
| M10 | `type(n) is int` → `isinstance` | 1 failed (`True`), exit 1 |

## Fixes applied (this commit)

1. `serve_offline.py:43-54`: validate `decisions` before `range()` (rejecting bool, str, float, None, a missing key, and values ≤ 0), and require it to equal the saved timeline's decision count. The message for missing answers is unchanged (`incomplete: k of m missing`).
2. `README-ko.md:24`: GET count 525 → 526, plus a mention of the timeline-count refusal.
3. `test_offline_bundle.py`:
   - The v2 banner assertion (`:114`).
   - `write_all_bundle(**manifest_overrides)`.
   - `test_all_decisions_bundle_with_bad_count_does_not_start`, parametrised over `'5'`, `5.0`, `None`, `True`, `0` and `-1`.
   - `test_all_decisions_count_must_match_saved_timeline`, parametrised over 4 and 6.
   - `test_all_decisions_top_mismatch_does_not_start`.

## Commands and exit codes

- `OMP_NUM_THREADS=1 perl -e 'alarm 300; exec @ARGV' /Users/song/Projects/pitcheezy/.venv/bin/python -m pytest test_offline_bundle.py -q -p no:cacheprovider`
  - at 73b08cb: 15 passed, exit 0
  - after fixes: 24 passed, exit 0
- `/usr/bin/python3 -m pytest --version`: exit 1 (`No module named pytest`; Python 3.9.6)
- Stub runner under `/usr/bin/python3` 3.9.6, after fixes: 24 passed, 0 failed, exit 0
- `serve_offline.py` under `/usr/bin/python3` on ports 8792 (v3), 8793 (v2, new code) and 8794 (v2, 3ed820b code), each bounded with `alarm 300`:
  - all three started: 526, 178 and 178 saved answers
  - probe script exit 0
  - processes killed afterwards; `lsof -nP -iTCP:879{2,3,4,5} -sTCP:LISTEN` is empty
- Fixed `load_snapshot` on the real bundles: v3 loads 526 answers, v2 loads 178; exit 0
- 8771: 4 GETs, all 200 and sha-equal; exit 0
- No headless browser was run; the existing t42 evidence covers the UI path.
