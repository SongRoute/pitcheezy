# 월드시리즈 운영 상태 파일 계약 (`pitcheezy-ws-ops-status-v1`, 판 1.1, 2026-10-18)

운영 실행기 `apps/observer/ops/ws_operate.py`의 감독 프로세스가 쓰고, 경고 감시기(`apps/observer/ops/ws_alert.py`)가 읽는 JSON 파일이다. 감시기는 이 파일만 읽으면 되고 프로세스 표를 뒤질 필요가 없다. 특히 감시 명령줄에 `rehearse.py supervise`나 `observer_app.publisher` 글자를 넣지 않는다.

- 판 1.1(검토 REVIEW-001 B1): 1.0 필드는 모두 그대로 두고 `schema_version`, `updated_at`, `publisher.running`, `problems[]`, `games[]`의 `start`/`status`/`window_start`/`window_end`, `out_dir`, `state_path`를 **더했다**. `schema` 문자열은 그대로 `pitcheezy-ws-ops-status-v1`이다.

## 위치·쓰기 방식

- 경로: `<ops_root>/status.json`. `ops_root` 기본값은 `~/Projects/pitcheezy-pm-jobs/ws-ops`(저장소 밖, `ws_ops_config.json` `ops_root`, 환경 변수 `PITCHEEZY_WS_OPS_ROOT`로 바꿀 수 있음). 같은 내용이 `<ops_root>/runs/<KST 날짜>/status.json`에도 있다(날짜별 기록).
- 공용 `status.json`은 감독 lock(`<ops_root>/supervisor.lock`)을 잡은 감독만 쓴다. 동시에 감독은 하나뿐이다.
- 원자적 교체(임시 파일 → `rename`)로 쓴다. 읽는 쪽은 반쯤 쓴 파일을 보지 않는다.
- 쓰는 때(= heartbeat, 매번 `updated_at` 갱신): 상태가 바뀔 때마다, 운영 중(`prestart`/`running`/`backoff`/`stopping`) `status_every_s`(10초)마다, 대기(`waiting`) 중 `waiting_status_every_s`(60초)마다. `stopped`/`refused`/`failed` 뒤에는 더 쓰지 않는다(감독이 끝났으므로).
- 오래됨 기준(잠정, 2026-10-18 통합에서 감시기 값으로 맞춤): 감시기(`ws_alert_config.json` `ops_stale_running_s`/`ops_stale_other_s`)는 `state`가 `running`/`backoff`일 때 `updated_at`이 90초(heartbeat 10초의 9배), 그 밖의 진행 상태(`waiting`/`prestart`/`stopping`)일 때 300초(대기 heartbeat 60초의 5배)보다 오래되면 "감독 멈춤"(`ops_status_stale`)으로 본다. 끝난 상태(`stopped`/`refused`/`failed`)에서는 오래됨을 따지지 않는다. 처음 적었던 60/180초 안은 쓰지 않는다.
- 비밀 값은 들어가지 않는다: 토큰·업로드 주소는 쓰지 않는다(업로드는 있음/없음과 개수·오류 종류만). `runs/` 폴더는 이슈·저장소에 복사하지 않는다(런타임 설정에 업로드 주소가 있다).

## 이전 문제의 정리(REVIEW-002 D3)

- 새 감독이 시작하면(다음 날이든 같은 날 재시작이든) 직전 status.json에서 `active: true`였던 문제를 `active: false` + `cleared_at` + `cleared_by`로 옮겨 적는다. 그래서 어제의 `refused`/`failed` 문제와 그날 아침 `start_refused`는 새 감독의 첫 heartbeat에서 풀린다. `state`도 새 감독의 값(`waiting`)으로 바뀐다.
- 경기 없는 날의 매일 `start`는 아무것도 쓰지 않으므로 어제의 끝 상태가 그대로 남는다. 어제가 `refused`/`failed`였다면 그 경고는 다음 감독이 뜰 때까지 남는다. 감시기는 `kst_date`가 오늘·어제가 아니면 이 파일을 무시한다.

## 감시기가 읽는 핵심 필드

| 키 | 타입 | 뜻 |
|---|---|---|
| `updated_at` | str (ISO UTC) | 이 파일을 쓴 시각. 모든 heartbeat에서 갱신 |
| `state` | str | 아래 상태 표 중 하나(1.0과 같음) |
| `publisher` | obj | **항상 객체.** `running`(bool). 돌 때: `pid`, `lstart`, `commit`, `commit_dirty`, `segment`, `started_at`, `armb`, `fixed_delay_s`, `upload_enabled`, `childwatch_pid`, `fake`. 안 돌 때: `running: false`, `segment`, `commit`(마지막), `last_exit` |
| `problems` | list | `{key, severity, message, active}` (+ `since`, 해제되면 `cleared_at`). 키마다 한 줄. 조건이 풀리면 줄을 지우지 않고 `active: false`로 바꾼다. 활성 줄이 앞에 온다 |
| `games` | list | `{game_pk, start, status, window_start, window_end}` (+ `label`, `start_utc`, `final_seen_at`). 아래 참고. `upstream_changed_at`은 넣지 않는다(실행기에 상류 출처가 없음, 선택 필드) |
| `out_dir` | str | 그날 발행기 출력 폴더(`<ops_root>/runs/<날짜>/out`). 감시기는 health.json을 여기서 읽는다 |
| `state_path` | str | 그날 발행기 상태 파일(`<out_dir>.state.json`) |

### `problems[].key`와 심각도

| key | severity | 켜지는 때 |
|---|---|---|
| `start_refused` | urgent | `start` 명령 자체가 거부됨(재시도 창을 다 써도 일시적 문제가 남았거나, 영구 문제). **할 일이 없어서 끝난 경우(`no_game` 경기 없는 날, `all_final` 이미 모두 종료, `hard_end_passed` 그날 상한 지남)만 있으면 status.json을 쓰지 않는다**(전날의 정상 종료가 그대로 남고 감시기는 조용하다; REVIEW-003 B3). 이런 키가 진짜 문제와 함께 있으면 urgent로 쓴다. 일정 응답은 모양을 확인한다(REVIEW-004 B4): statsapi 응답은 `dates` 목록이 있는 객체여야 하고(`totalGames`가 있으면 경기 수와 같아야 함), 일정 파일은 `rows` 또는 `dates` 목록이 있어야 한다. 오류 본문·`[]`·`{"rows": null}` 같은 것은 '경기 없음'이 아니라 일정 확인 실패로 보고 재시도한 뒤 urgent로 쓴다. 예정된 WS 경기가 연기·취소돼 경기가 없으면 거부 사유에 `called off: <경기> Postponed`가 붙는다(이때도 쓰지 않음). 감독이 없으므로 `start`가 직접 status.json을 쓴다(state `refused`, `refused_by: "start"`, `supervisor: null`). 다른 감독이 lock을 잡고 있으면 쓰지 않는다. 띄운 감독이 `start_confirm_s`(기본 15초, `ws_ops_config.json`) 안에 `events.jsonl`에 자기 pid의 `supervisor_start`를 남기지 않고 프로세스도 없으면, 메시지 `start refused: supervisor pid … did not record supervisor_start …`로 이 키를 쓴다(start의 `problem_keys`는 `supervisor_not_started`; OPS-02, 2026-10-19). 살아 있지만 늦으면 쓰지 않는다 |
| `prestart_refused` | urgent | 시작 점검이 끝내 통과하지 못해 그날 발행기를 띄우지 않음(state `refused`) |
| `prestart_retrying` | warn | 일시적 문제로 시작 점검을 다시 하는 중(60초 간격, 첫 투구까지) |
| `restart_limit` | urgent | 재시작 한도(3번)를 넘어 그날 운영을 멈춤(state `failed`) |
| `deploy_guard` | urgent | 경기 중 코드 커밋이 바뀌어 재시작을 거부함(state `failed`) |
| `resource_guard` | urgent | 디스크·발행기 RSS·실행 폴더 크기 가드로 발행기를 멈춤(state `failed`) |
| `upload_blocked` | urgent | 상태 파일에 `phase=blocked` 업로드 또는 복구 차단(자동으로 안 풀림) |
| `another_publisher` | urgent | 재시작하려는데 다른 발행기가 돌고 있음(state `failed`) |
| `supervisor_error` | urgent | 감독 내부 오류로 끝남(state `failed`) |
| `publisher_restarting` | warn | 예상 밖 종료 뒤 재시작 대기 중(state `backoff`). 재시작되면 해제 |
| `protected_process_changed` | warn | 보호 프로세스(8766–8772, 데모 루프, T7 워커) 변화. 기록만 하고 신호는 보내지 않음 |
| `memory_low` | warn | 여유 메모리 < 1.5 GiB |
| `health_stale` | warn | 발행기는 도는데 health.json이 30초 넘게 안 바뀜 |
| `feed_failing` | warn | health `feed_failing_games`가 비어 있지 않음 |
| `schedule_fallback` | warn | 시작 점검의 일정 요청이 실패해 start 때 기록한 경기 목록을 씀 |
| `identity_unreadable` | warn | 새 발행기의 ps 신원을 읽지 못해 그 세그먼트에 childwatch가 없음 |
| `mount_lost` | warn | 발행기가 도는 중 `required_mounts`(T7)가 빠짐(OPS-03, 2026-10-19). 발행기는 멈추지도 재시작하지도 않는다. 새 투구마다 추천이 "추천 없음"(`compute_failed`)으로 나가고, 같은 자리에 다시 붙으면 저절로 풀린다. 메시지에 경로는 넣지 않는다(경로는 `resources.mounts_missing`) |
| `mount_restart_needed` | warn | T7이 돌아왔지만 지금 발행기 세그먼트가 T7 없이 시작됐음. 그 프로세스는 추천 모델을 다시 읽지 않으므로 "추천 준비 안 됨"(`recommender_unavailable`)이 계속된다. `stop` 뒤 `start`로 풀린다(런북 4.8) |

`alerts`(1.0, 문자열 목록: `memory_low`, `health_stale`, `feed_failing`, `upload_blocked`, `protected_process_changed`, 1.1 OPS-03부터 `mount_lost`, `mount_restart_needed`)는 그대로 둔다. 같은 키가 `problems`에도 있다.

### `games[]`

- `start`: 예정 첫 투구(ISO UTC).
- `status`: `final`(발행 파일이 final이거나 일정이 Final), `postponed`(일정이 Postponed/Cancelled/Suspended), `live`(예정 시각이 지났고 final 아님 — 시계 기준이지 실제 첫 투구 확인이 아님), `scheduled`(예정 전), `other`(예약값, 지금은 쓰지 않음). 감시기는 `live`로 경기 시간대를 열지만, 이 값만으로 "실제 진행 중"으로 보지 않는다(`file_stuck`은 발행 파일의 `game.status_text`가 `In Progress`이거나 발행기 자신이 live라고 할 때만).
- `window_start`: 발행기 시작 시각(그날 첫 경기 − 30분). `window_end`: 그날 상한(마지막 경기 + 9시간). 모든 경기가 final이 되면 `min(상한, 마지막 final + 15분)`으로 당겨진다. 감시기는 이 창 안에서만 "발행기 멈춤"을 경고하면 된다.

## 나머지 필드(1.0부터, 그대로)

| 키 | 타입 | 뜻 |
|---|---|---|
| `schema`, `schema_version` | str | `pitcheezy-ws-ops-status-v1`, `1.1` |
| `written_at`, `written_epoch` | str, float | `updated_at`과 같은 시각(1.0 이름) |
| `state_reason` | str/null | 상태의 이유(예: `publisher_exited:3`, `final_settled`, `deploy_guard:...`, `guard:disk_low`, `retrying`) |
| `kst_date`, `run_dir` | str | 운영 날짜(KST)와 실행 폴더 |
| `timeline` | obj/null | `first_pitch`, `publisher_start`, `hard_end`(KST 문자열), `publisher_start_epoch`, `hard_end_epoch`, `all_final_at`, `settle_until`, `stop_rule` |
| `supervisor` | obj | `{pid, lstart, started_at}` |
| `health` | obj/null | 마지막으로 본 health.json: `{sequence, published_at, observed_at, age_s, files_written, feed_failing_games, unavailable_count, unavailable_codes}` |
| `game_checks` | obj | 경기별 `{checked_at, feed_ok, file_sequence, sequence_checked, final}` |
| `feed_ok` | bool/null | `feed_failing_games`가 비고 모든 경기 `feed_ok`가 false가 아니면 true. health를 아직 못 봤으면 null |
| `upload` | obj | `{enabled}` 항상. 켜져 있으면 `pending`, `code`, `attempts_by_phase`, `blocked`, `error_classes`, `last_error_class`, `pending_in_state`, `recovery_blocked` |
| `resources` | obj | `{publisher_rss_bytes, available_bytes, free_disk_bytes, run_dir_bytes, mounts_missing, sampled_at}`. `mounts_missing`(OPS-03 추가): 발행기가 도는 동안 빠진 `required_mounts` 경로 목록(발행기 없으면 `[]`) |
| `restarts` | obj | `{count, max, backoff_s, last_exit, next_restart_at, history}` |
| `protected` | obj | `{ok, pinned_gone, listeners_8772}` |
| `alerts` | list[str] | 위 참고 |
| `guard` | str/null | 발행기를 멈추게 한 자원 가드 |

### 상태(`state`)

| 값 | 뜻 | 감시기 해석(제안) |
|---|---|---|
| `waiting` | 시작 시각(첫 투구 − 30분)을 기다림 | 정상 |
| `prestart` | 시작 점검 중(`state_reason: retrying`이면 재확인 대기) | 정상. 재확인은 `prestart_retrying` warn |
| `running` | 발행기 실행 중 | 정상 |
| `backoff` | 예상 밖 종료 뒤 재시작 대기 | `publisher.running: false` + `publisher_restarting` |
| `stopping` | 발행기 정상 종료 중 | 정상(최대 40초) |
| `stopped` | 정상 끝(`final_settled`, `operator_stop`, `operator_stop_before_start`, `hard_end`) | `hard_end`면 확인 |
| `refused` | 시작 거부(`prestart_refused`, `supervisor_lock_held`, 또는 `start` 명령 거부 `start_refused`) | 해당 urgent 문제 |
| `failed` | 비정상 끝(재시작 한도, 배포 가드, 자원 가드, 다른 발행기, 감독 오류) | 해당 urgent 문제 |

## 재시작·정체 기준(설정 값, 잠정)

- 예상 밖 종료 → 30/60/120초 뒤 재시작, 최대 3번(감독 1회 실행 기준). 대기는 그날 상한을 넘지 않는다.
- health 정체 재시작: 마지막 새 sequence와 세그먼트 시작 중 늦은 쪽부터 300초. 그래서 **첫 정체 재시작은 세그먼트 시작 뒤 최소 300초**다(`startup_grace_s` 180초는 확인을 시작하는 시점일 뿐).
- 모든 경기가 final이 된 뒤 발행기가 끝나면 재시작하지 않고 `final_settled`로 정상 종료한다(그때는 배포 가드도 적용하지 않음).
- `start` 명령: 일시적 문제(일정 요청 실패, 디스크, T7, 다른 발행기, 다른 미종료 경기)는 60초마다 최대 30분, 발행기 시작 시각을 넘기지 않고 다시 확인한다. 끝내 거부하면 위 `start_refused`를 쓴다. `--dry-run`은 재시도도 쓰기도 하지 않는다. 재시도 중 Ctrl-C로 끊으면 아무것도 쓰지 않고 끝난다(감독도 없음).
- 시작 점검: 일시적 문제는 첫 투구까지 60초마다 다시 확인(늦게 시작했으면 10분), 그래도 안 되면 `refused` + `prestart_refused`. 일정 요청 실패는 start 때 기록한 경기 목록으로 대신한다.

## 예시(값은 예시)

```json
{"schema": "pitcheezy-ws-ops-status-v1", "schema_version": "1.1", "updated_at": "2026-10-24T00:39:12+00:00",
 "state": "running", "state_reason": null, "kst_date": "2026-10-24",
 "out_dir": "/Users/song/Projects/pitcheezy-pm-jobs/ws-ops/runs/2026-10-24/out",
 "state_path": "/Users/song/Projects/pitcheezy-pm-jobs/ws-ops/runs/2026-10-24/out.state.json",
 "publisher": {"running": true, "pid": 51234, "lstart": "Sat Oct 24 08:30:01 2026", "segment": 1, "commit": "<40 hex>"},
 "games": [{"game_pk": 849805, "start": "2026-10-24T00:00:00+00:00", "status": "live",
            "window_start": "2026-10-23T23:30:00+00:00", "window_end": "2026-10-24T09:00:00+00:00"}],
 "problems": [{"key": "memory_low", "severity": "warn", "message": "available memory ...", "active": false,
               "since": "2026-10-24T00:20:00+00:00", "cleared_at": "2026-10-24T00:21:00+00:00"}],
 "alerts": [], "health": {"sequence": 412, "age_s": 3.2}, "feed_ok": true}
```

## 바꿀 때

필드를 지우거나 뜻을 바꾸면 `schema`의 버전을 올린다. 필드 추가는 같은 `schema`에서 `schema_version`만 올린다. 코드 쪽 키 목록은 `ws_operate.STATUS_KEYS`·`PROBLEM_SEVERITY`이고 `tests/test_ws_operate.py`(`test_status_file_schema`, `test_problems_*`)가 확인한다. 감시기가 넘기지 않는 키(`ws_alert.OPS_ALERTS_COVERED`: `memory_low`, `health_stale`, `feed_failing`, `upload_blocked`)와 실제 감독 → 감시기 흐름은 `apps/observer/ops/tests/test_ws_ops_alert_integration.py`가 확인한다. 함께 띄우는 법: 운영 런북 7.6.
