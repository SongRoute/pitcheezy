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

## UI direction

Warm ivory + dark ink, deep teal accent, orange actual-pitch marker; polished sports editorial, Korean text. Clear scoreboard, large zone, rank3 recommendations, count/base diamonds, pitch timeline, terminal analysis card with optional manual target selector. Responsive layout with useful empty/loading/error states and visible data date. No login/billing/public hosting. Avoid technical jargon in main flow. Show model details only when expanded.

Default backend port8766. Vite proxy `/api` -> localhost8766. Production build served by backend, single launch command. Prefer accessible buttons/labels, keyboard/manual control, honest unavailable states. No framework-specific experimental APIs.
