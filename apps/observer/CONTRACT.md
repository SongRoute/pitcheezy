# Observer MVP contract v1

Local browser / FastAPI. Frontend calls relative `/api` paths. All labels Korean, MLB abbreviations retained. Historical replay, never label live. GET view returns only revealed actual pitches. Future pitch outcome/type/location/total length remain server-only.

## Routes

- `GET /api/health`: `{status,model_ready,dataset_ready,mode,cv_mode}`
- `GET /api/catalog`: `{games:[{id,date,home_team,away_team,title,plate_appearances:[{id,batter_label,pitcher_label,inning,half}]}], model_version, limitations:[str]}`
- `POST /api/sessions` body `{game_id:int,pa_id:int}` -> view
- `GET /api/sessions/{id}` -> view
- `POST /api/sessions/{id}/advance` body `{revision:int}` -> next view; 409 stale revision
- `POST /api/sessions/{id}/manual-intent` body `{revision:int,zone_id:str}` -> view. Only complete PA. Annotation concerns selected terminal pitch, not next recommendation.
- `GET /api/zones`: `{zones:[{id,label,column,row}],coordinate_frame:"catcher_view"}`; ids `low_left,low_middle,low_right,middle_left,middle_middle,middle_right,high_left,high_middle,high_right`. row0=low,row2=high; column0=left in catcher view. Don't mirror by batter hand.
- `GET /api/runtime`: compact diagnostics for optional details panel (not main product UI).
- `GET /api/inning-decision-games`: dates and record counts for games with verified stored prechange decisions; empty list when none.
- `GET /api/inning-decisions?game_id=<positive integer>`: separately stored historical pre-pitching-change decision contexts; no result values in list.
- `POST /api/inning-decisions/{decision_id}/resolve` body `{revision:1,context}`: returns a conditional inning result only after exact normalized context matching. See `docs/contracts/inning-decision-api-v1.md`.

Errors are JSON `{detail: string}` with 400/404/409/503. Catch and show a retryable error without losing current view. Frontend polls GET session every2s only when analysis.status queued/running. No background auto-advance.

## View

```json
{
  "id":"uuid", "revision":0,"cursor":0,"complete":false,
  "game":{"id":777262,"date":"2025-08-17","home_team":"SF","away_team":"TB","title":"TB @ SF"},
  "plate_appearance":{"id":32,"batter_label":"타자 #123","pitcher_label":"Logan Webb","batter_stand":"R"},
  "state":{"inning":5,"half":"Top","outs":1,"bases":2,"home_score":0,"away_score":1,"balls":0,"strikes":0},
  "recommendation":{
    "id":"sha256", "status":"ready", "mode":"experimental_location_proxy", "model_version":"observer-zone-v1",
    "candidates":[{"pitch_type":"SL","pitch_label":"슬라이더","zone_id":"low_left","zone_label":"낮은 왼쪽","target":{"x":-0.55,"z":1.8},"value":0.35,"delta_pp":0.02,"support":51.2}],
    "baseline_value":0.34,"zone_bounds":{"bottom":1.5,"top":3.5},
    "basis":["주자·아웃·카운트 반영","이전 날짜 타자 성향 반영","목표 구역은 근사 모델의 제안"],
    "reason":null
  },
  "last_pitch":null,
  "history":[],
  "analysis":null,
  "event_analysis":null,
  "summary":null,
  "context_notes":["이 투구 전까지 던진 공 54개"],
  "notices":["기록 재생 · 실제 승률 향상이 검증된 추천은 아닙니다."]
}
```

Unavailable recommendation: same metadata, `status:"unavailable", reason:Korean,candidates:[]`; zone_bounds provided. App can advance even if recommendation unavailable. Recommendation=null at PA completion.

Revealed pitch (history entries and last_pitch): `{id,pitch_number,pitch_type,pitch_label,x,z,speed_mph,description,result_label,pre_state,recommendation,zone_bounds}`; x/z/speed nullable. `recommendation` is the saved **pre-pitch** recommendation for that pitch. The chart must compare last actual against last_pitch.recommendation, not against a newly calculated next-pitch recommendation. On incomplete PA allow explicit chart mode `다음 공 추천` vs `방금 던진 공`; default actual comparison after advancing. Missing coordinates displayed as unavailable, never default (0,0).

Analysis on PA completion: `{id,status,source,selected_pitch_id,selected_pitch_number,selection_reason,cv_status,message,manual_zone_id,comparisons,narrative,version}`.
- status: queued/running/unavailable/complete/failed.
- source: none/manual (no supplied video, no automatic CV claims).
- cv_status: unavailable:no_media.
- selection_reason: 타석의 마지막 공.
- comparisons: null unless manual input; then `{recommended_zone_label,intended_zone_label,actual_zone_label,target_error_zone_units,interpretation}`. Spatial comparison only; do not invent causal WE execution attribution.
- narrative: list of Korean grounded sentences.

Summary: `{headline,result_label,pitch_count,selected_pitch_number,notes:[str]}`. Available only after completion. Worker marks no-media jobs unavailable and retains manual annotation separately. Returning manual annotation raises analysis version; original pre-pitch recommendations immutable.

## PA event result (integration v1)

`event_analysis` is null until the selected terminal pitch is revealed. On completion it is a separate persisted `event-analysis-v1` result, independent of the no-media CV `analysis` job. Its `linkage.pitch_id`, recommendation ID and canonical SHA identify the saved recommendation for **that same terminal pitch**. The recommendation is stored before reveal with a UTC creation timestamp. The source contract and numeric definitions are in `docs/contracts/event-analysis-v1.md`.

For actual records with no linked, validated pre-release intent, the result is normally `partial`: the saved pre-pitch baseline and actual post-PA state are valued by the same frozen defensive WE; `values.total_pp` is signed percentage points for the initial defending team. Strategy, execution, outcome residual and shares stay null, with the entire difference in `components.unallocated_residual_pp`. This is a descriptive model comparison, never a player responsibility or causal effect. `unavailable` carries missing compatible values; `failed` carries a calculation error with all numeric fields null. No-media CV status cannot turn a WE event calculation into success or block it.

The `event_results` table keeps immutable `(session_id,pitch_id,revision)` payloads and validates the saved recommendation ID and canonical SHA before insertion. A late/corrected source creates a higher event input revision; the saved pre-pitch recommendation row does not change. New recommendations save the frozen model SHA, adapter identity, value spec and baseline policy ID. An older or mismatched recommendation baseline is never relabeled with the current evaluator: its reference and total are unavailable. Manual zone notes are spatial user annotations and do not become model intent or change the event result. Old sessions that predate recommendation timestamps have `event_analysis:null` because their original storage time cannot be reconstructed. The UI never renders synthetic `development_only` numeric results as actual contributions.

## Separate conditional inning result (handoff v1)

`inning-result-v1` is a standalone C-to-D research payload defined in `docs/contracts/inning-result-v1.md`. It is persisted in a separate immutable decision table and returned by the context-matched decision API; it is not a session View field or a persisted PA event result. It describes the frozen initial defender's final-game win probability after propagating a fixed keep-pitcher scenario to the current half-inning boundary. Its unresolved-mass bounds are probabilities; the UI formats them as percent, never percentage-point contribution or confidence intervals. Actual replacement value stays null.

D consumes validated JSON through `web/src/inningResult.ts`; the examples live in `results/C-D-INNING-001/`. The `inning-decision-v1` API compares the full decision context before returning its `result`. A future card must explicitly select this separate historical decision, rather than pretend that a PA cursor is at the same point. The later first observed pitch is a reference only. Do not attach this payload to a completed PA merely because its pitch ID matches, add it to event components, or replace `event_analysis.replacement` with its keep scenario. The unavailable example is development-only and is not imported through the file-backed registration path. Malformed payloads fail validation instead of displaying 0%.

The `/inning-decisions` page now provides the independent selector and conditional inning card. The home header links to it. Registered-game discovery is separate from the PA catalog. Selecting a decision reveals its prechange scoreboard; clicking `이닝 전망 보기` resolves the exact selected context. The client checks returned ID/revision/context against that selection and checks result linkage again before rendering. Selection changes and retries clear the prior result and cancel stale requests. Errors and empty records never show numeric defaults. The page does not read/change the PA session cursor or its localStorage bookmark. Team/player display names are not invented when the source supplies only IDs.

## Catcher setup estimates (display only)

`GET /api/watch/{game_pk}/setup` returns `pitcheezy-setup-estimates-v1`: the video module's IntentEstimate v1 lines of one game, read from `PITCHEEZY_INTENT_DIR/<game_pk>.jsonl` and pinned by `<game_pk>.source.json` (repository, commit, sha256). Every line passes `validate_intent_estimate` or the whole game is refused (503); a game without a file is 404.

Each pitch carries `at_bat_number`, `pitch_number`, `status`, `unavailable_reason`, `plate_x_feet` and `x_band` (`left`/`middle`/`right` in the catcher's view like `/api/zones` column 0-2, or `off_left`/`off_right`). Only the horizontal position is served: `display.vertical` is false and no height is returned. Pitches the source video does not show have no entry. The values are unreviewed estimates of the catcher's setup, a proxy and not the pitcher's intent (`is_intent_proxy` true, `accuracy.measured` false); the screen shows the badge and never calls the setup-to-pitch distance a command error.

## Watch-along revealed-prefix summary (display only)

`GET /api/watch/{game_pk}/summary?through=<k>[&top=<n>]` returns `pitcheezy-watch-summary-v1`: line score, chronological home WE curve and notable plays of watch-along pitches `0..k`. It is built only from the precomputed watch dataset (no model is loaded, nothing estimated) and is additive: `GET /api/watch/games` and `GET /api/watch/{game_pk}` are unchanged and still never carry the final score.

- `through` is **required** (no default, so a client cannot spoil the game by omission): `-1` = nothing revealed yet, `0..n-1` = index of the last revealed pitch (`n` = `decisions.length` of the timeline). Postgame = `through=n-1`; there is no separate postgame mode. `top` = 0..20 (default 5) entries in `largest_we_changes`.
- Inputs read for `through=k`: `actual`/`we` of pitches `0..k`, the pre-pitch `situation` of pitch `k+1` (already public in the timeline; it is the score after pitch `k`, the same moment as reveal `k`'s `we.home_after`), and `game.final` **only** when `k=n-1`. Nothing else of pitch `k+1` or later is read (tested by mutating all of it).
- Errors: missing/invalid `through`, `through<-1`, `top` outside 0..20 → 400; `through>=n` or unknown game → 404; malformed dataset or a read pitch without a valid inning/`Top`/`Bot` half, or half-innings out of game order → 503.

```json
{
  "schema": "pitcheezy-watch-summary-v1", "game_pk": 900002, "through": 4, "revealed_pitches": 5, "complete": false,
  "teams": {"away": "Away Club", "home": "Home Club"},
  "score": {"away": 1, "home": 2, "source": "next_pre_pitch_situation", "after_index": 4},
  "innings": [{"inning": 1, "away": {"runs": 1, "status": "complete"}, "home": {"runs": 2, "status": "complete"}}],
  "we_curve": [{"index": 4, "inning": 1, "half": "Bot", "at_bat_number": 4, "pitch_number": 1, "home_before": 0.45, "home_after": 0.7}],
  "events": [{"index": 1, "pa_id": "900002:1", "at_bat_number": 1, "pitch_number": 2, "inning": 1, "half": "Top",
              "pitcher": {"id": 11, "name": "P1"}, "batter": {"id": 21, "name": "B1"},
              "event_label": "홈런", "result_label": "홈런", "play_text": "play 1",
              "runs_scored": {"away": 1, "home": 0}, "score_after": {"away": 1, "home": 0},
              "we": {"kind": "observed_model_change", "home_before": 0.5, "home_after": 0.4, "home_delta": -0.1, "batting_delta": 0.1}}],
  "largest_we_changes": ["<same shape as events, sorted by |home_delta| desc then index>"],
  "notes": {"we": "승리확률 변화는 고정된 모델 추정치의 관측 차이이며, 선수 책임이나 인과 기여도가 아닙니다.", "we_model": "<dataset we.note>", "attribution": null},
  "issues": []
}
```

(Synthetic test game; `we_curve`/`events` truncated to one entry each. Deltas are raw floats in the payload.)

Field meanings and nulls:
- `complete`: true only when `through=n-1` (every pitch of the completed game revealed).
- `score`: total after pitch `through`. `source` = `next_pre_pitch_situation` (pitch `k+1`'s situation), `final_record` (dataset final, only when complete), `pre_pitch_situation` (`through=-1`: pitch 0's situation) or `unavailable` (no pitches, or complete without a final record → `away`/`home` null). A null team score = unknown, never 0. `after_index` is null for `through=-1`.
- `innings`: one row per inning from 1 to the latest inning in the revealed prefix (later innings are not listed, so extra innings are not spoiled; the UI pads to 9 columns with "not reached"). Each half cell `{runs, status}`:
  - `complete`/`in_progress`: half was played (has revealed pitches). `runs` is an integer when every score change in it is known; `null` = played but runs unknown (missing score, decreasing score, or the last pitch's runs without a final record). `0` is a real zero.
  - `not_reached`: not yet reached in the revealed prefix; `runs` null.
  - `not_played`: game complete and the half never happened (e.g. home leading after the top of the last inning); `runs` null. UI shows "X"/"-".
  - `unknown`: a half that must have been played (before the latest revealed half) has no pitches in the dataset; `runs` null.
  A half is `complete` once a later half was reached or the next pre-pitch view is already in a later half (e.g. after the 3rd out).
- Run attribution: runs for pitch `i` = score after `i` − score before `i` (pre-pitch situations; final record for the last pitch). Away runs go to the top half, home runs to the bottom half — this pitch's half, or (rare, a run before a half's first pitch) the next pitch's half. Any other case is reported in `issues`, never folded into an inning.
- `we_curve`: one point per revealed pitch in game order; `home_before`/`home_after` are the dataset's frozen home WE in [0,1] or `null` (missing, NaN/inf, non-number or outside [0,1]). The last pitch's `home_after` is the dataset value (1/0 by final result, null for a tie). Plot `home_before` at each index and the final point's `home_after`; break the line at nulls, never interpolate or draw 50%.
- `events`: revealed pitches that ended a PA (`event_label` non-null) or changed the score (`runs_scored` non-zero), in game order. `runs_scored`/`score_after` team values may be null (unknown). `we.home_delta`=`home_after−home_before`, `batting_delta` is the same change for the batting team; null if either end is null.
- `largest_we_changes`: top `top` revealed pitches by |`home_delta`| (null deltas excluded; ties by earlier index). Pitch-level, not PA-level.
- `we.kind:"observed_model_change"` and `notes.attribution:null`: these are observed differences of a count-independent display WE, not player responsibility, causal contribution or recommendation value. The UI must not label them "기여도".
- `issues`: data-consistency codes (`{team}_score_decreased_after_index_{i}`, `{team}_runs_outside_batting_half_after_index_{i}`, `{team}_line_score_differs_from_score`). Empty in normal data; show a small "기록 불일치" note when non-empty.
- Not provided: hits/errors (H/E), PA-level WE, causal attribution.

UI integration: after each reveal of index `i`, call `summary?through=i` (on load with `#i=N`, `through=N-1` if pitch N is not yet revealed, else `N`); before any reveal use `through=-1`. Drop responses whose `through` differs from the current cursor (stale request). Line score: columns from `innings` padded to 9, R column from `score` (`—` when null). Curve x-axis = `index`, half-inning boundaries from `inning`/`half` changes; tap a point → jump to that pitch. Notable-play list from `events` or `largest_we_changes`, with `inning`+`half` (`초`/`말`), `event_label`, `play_text` and signed `batting_delta` formatted as %p of the displayed model.

Tests: `tests/test_watch_summary.py` (empty game, bounds, prefix advance, final record/not played, unknown last-pitch runs, final-run checks, no future leakage for every prefix, unknown/NaN WE and valid JSON, ordering/limit, missing scores, unknown half and order, run before a half's first pitch, API status codes and unchanged privacy of the timeline/games list).

## UI direction

Warm ivory + dark ink, deep teal accent, orange actual-pitch marker; polished sports editorial, Korean text. Clear scoreboard, large zone, rank3 recommendations, count/base diamonds, pitch timeline, terminal analysis card with optional manual target selector. Responsive layout with useful empty/loading/error states and visible data date. No login/billing/public hosting. Avoid technical jargon in main flow. Show model details only when expanded.

Default backend port8766. Vite proxy `/api` -> localhost8766. Production build served by backend, single launch command. Prefer accessible buttons/labels, keyboard/manual control, honest unavailable states. No framework-specific experimental APIs.

## Selected-time linescore (ui-calm-polish, 2026-10-06, commit 2b32e30)
- `GET /api/service/games/{pk}`: every item of `pitches[]` carries `linescore_before` — the same object shape as the top-level
  `linescore` (`basis`, `innings[]`, `teams.{away,home}.{runs,hits,errors,bb_hbp}`, `current`, `provenance`), as of just BEFORE
  that pitch (the moment of `situation_before`, i.e. the header score and B/S/O shown for a selected pitch). Built server-side
  from events before the pitch only (`game_history.build` `line_before`); `situation_after`, `gap_after`, `line_before`,
  `line_after` stay internal. Additive: older clients ignore it.
- Client: a selected past pitch's 이닝별 기록 uses `linescore_before`; the latest view keeps the top-level `linescore`. A pitch
  without the field shows the 'not ready' text, never the latest line.
- Checks: `backend/tests/test_pitch_linescore_before.py`, `web/tests/boxscore-asof-check.mjs`.
