# 월드시리즈 운영 상태 파일 계약 (`pitcheezy-ws-ops-status-v1`, 초안 2026-10-18)

운영 실행기 `apps/observer/ops/ws_operate.py`의 감독 프로세스가 쓰고, 경고 감시기(별도 작업)가 읽는 JSON 파일이다. 감시기는 이 파일만 읽으면 되고 프로세스 표를 뒤질 필요가 없다. 특히 감시 명령줄에 `rehearse.py supervise`나 `observer_app.publisher` 글자를 넣지 않는다(리허설·실행기의 "다른 발행기" 점검은 파이썬 인자만 보도록 고쳤지만, 확인은 이 파일로 한다).

## 위치·쓰기 방식

- 경로: `<ops_root>/status.json`. `ops_root` 기본값은 `~/Projects/pitcheezy-pm-jobs/ws-ops`(저장소 밖, `ws_ops_config.json` `ops_root`, 환경 변수 `PITCHEEZY_WS_OPS_ROOT`로 바꿀 수 있음). 같은 내용이 `<ops_root>/runs/<KST 날짜>/status.json`에도 있다(날짜별 기록).
- 공용 `status.json`은 감독 lock(`<ops_root>/supervisor.lock`)을 잡은 감독만 쓴다. 동시에 감독은 하나뿐이다.
- 원자적 교체(임시 파일 → `rename`)로 쓴다. 읽는 쪽은 반쯤 쓴 파일을 보지 않는다.
- 쓰는 때: 상태가 바뀔 때마다, 그리고 운영 중 `status_every_s`(10초)마다, 대기(`waiting`) 중에는 `waiting_status_every_s`(60초)마다. 감시기는 `written_epoch`가 **운영 중 60초, 대기 중 180초**보다 오래되면 "감독 정지/멈춤"으로 본다(잠정).
- 비밀 값은 들어가지 않는다: 토큰·업로드 주소는 쓰지 않는다(업로드는 있음/없음과 개수·오류 종류만).

## 필드

맨 위 키는 아래가 전부 항상 있다(값이 없으면 `null`). 모르는 키는 무시한다. 시각 문자열은 UTC ISO 8601.

| 키 | 타입 | 뜻 |
|---|---|---|
| `schema` | str | `pitcheezy-ws-ops-status-v1` |
| `written_at`, `written_epoch` | str, float | 이 파일을 쓴 시각(감독 heartbeat) |
| `state` | str | 아래 상태 중 하나 |
| `state_reason` | str/null | 상태의 이유(예: `publisher_exited:3`, `final_settled`, `deploy_guard:...`, `guard:disk_low`) |
| `kst_date`, `run_dir` | str | 운영 날짜(KST)와 실행 폴더 |
| `games` | list | `{game_pk, label, start_utc, final_seen_at}` 운영 대상 경기 |
| `timeline` | obj/null | `first_pitch`, `publisher_start`, `hard_end`(KST 문자열), `publisher_start_epoch`, `hard_end_epoch`, `all_final_at`, `settle_until`, `stop_rule` |
| `supervisor` | obj | `{pid, lstart, started_at}` |
| `publisher` | obj/null | 돌고 있을 때만: `{pid, lstart, segment, started_at, commit, commit_dirty, armb, fixed_delay_s, upload_enabled, childwatch_pid, fake}` |
| `health` | obj/null | 마지막으로 본 health.json: `{sequence, published_at, observed_at, age_s, files_written, feed_failing_games, unavailable_count, unavailable_codes}`. `age_s` = 지금 − 감독이 새 sequence를 본 시각 |
| `game_checks` | obj | 경기별 `{checked_at, feed_ok, file_sequence, sequence_checked, final}` (health의 `game_checks` 요약 + 감독이 본 final) |
| `feed_ok` | bool/null | `feed_failing_games`가 비고 모든 경기 `feed_ok`가 false가 아니면 true. health를 아직 못 봤으면 null |
| `upload` | obj | `{enabled}` 항상. 켜져 있으면 `pending`, `code`(health.upload), `attempts_by_phase`, `blocked`, `error_classes`(사유별 개수), `last_error_class`, `pending_in_state`, `recovery_blocked` (상태 파일에서 30초마다) |
| `resources` | obj | `{publisher_rss_bytes, available_bytes, free_disk_bytes, run_dir_bytes, sampled_at}` (10초마다; 메모리 = vm_stat free+inactive+speculative) |
| `restarts` | obj | `{count, max, backoff_s, last_exit, next_restart_at, history}`. `last_exit` = `{segment, exit_code, why, method, at}` |
| `protected` | obj | 보호 프로세스(8766–8772, 데모 루프, T7 워커) 비교: `{ok, pinned_gone, listeners_8772}`. 기록만 하고 신호는 보내지 않는다 |
| `alerts` | list[str] | 감독이 본 경고: `memory_low`, `health_stale`, `feed_failing`, `upload_blocked`, `protected_process_changed` |
| `guard` | str/null | 발행기를 멈추게 한 가드(`guard:publisher_rss_over_limit`, `guard:disk_low`, `guard:run_dir_over_limit`) |

### 상태(`state`)

| 값 | 뜻 | 감시기 해석(제안) |
|---|---|---|
| `waiting` | 감독이 시작 시각(첫 투구 − 30분)을 기다림 | 정상 |
| `prestart` | 시작 직전 점검 중 | 정상(수 초) |
| `running` | 발행기 실행 중 | `health.age_s` > 30초면 경고(런북 3절) |
| `backoff` | 발행기가 예상 밖으로 끝나 재시작 대기 | 즉시 알림(공개 화면 갱신 멈춤) |
| `stopping` | 발행기 정상 종료 중 | 정상(최대 40초) |
| `stopped` | 정상 끝(`final_settled`, `operator_stop`, `operator_stop_before_start`, `hard_end`) | `hard_end`면 확인 |
| `refused` | 시작 전 점검 거부(`prestart_refused`, `supervisor_lock_held`) | 즉시 알림 |
| `failed` | 비정상 끝(재시작 한도, 배포 가드, 가드, 감독 오류) | 즉시 알림 |

## 예시(가짜 발행기 시험에서 나온 형태, 값은 예시)

```json
{"schema": "pitcheezy-ws-ops-status-v1", "state": "running", "state_reason": null, "kst_date": "2026-10-24",
 "publisher": {"pid": 51234, "lstart": "Sat Oct 24 08:30:01 2026", "segment": 1, "commit": "<40 hex>",
               "armb": true, "fixed_delay_s": 0, "upload_enabled": false},
 "health": {"sequence": 412, "published_at": "2026-10-24T00:39:10Z", "age_s": 3.2, "feed_failing_games": []},
 "game_checks": {"849805": {"checked_at": "2026-10-24T00:39:10Z", "feed_ok": true, "file_sequence": 409, "final": false}},
 "feed_ok": true, "upload": {"enabled": false},
 "restarts": {"count": 0, "max": 3, "backoff_s": [30, 60, 120], "last_exit": null}, "alerts": []}
```

## 바꿀 때

필드를 지우거나 뜻을 바꾸면 `schema`의 버전을 올린다. 필드 추가는 같은 버전에서 한다. 코드 쪽 키 목록은 `ws_operate.STATUS_KEYS`이고 `tests/test_ws_operate.py::test_status_file_schema`가 확인한다.
