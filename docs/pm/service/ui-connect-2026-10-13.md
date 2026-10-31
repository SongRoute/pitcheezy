# UI 서비스 연결 — 2026-10-13

사용자 승인: 모바일은 현재 상황·추천·이번 타석을 우선한다. PC 상세는 라이브와 함께 보이는 비모달 패널, 태블릿은 추천·경기 2열과 선수 상세를 사용한다. 분석 용어와 공개 데이터의 뜻을 맞춘다. 별도 로컬 후보 구현이며 공개 배포·운영 교체·모델 선택/승격 승인이 아니다.

기준 브랜치 `codex/ws-ui-connect-1013`, 시작 커밋 `9707770`. `git merge-base --is-ancestor 9f4b36c 9707770` 성공으로 RC `9f4b36c` 포함을 확인했다. 따라서 추가 RC merge는 하지 않았다. 타 차선 P3·프로필 복구 및 기존 프로세스는 수정하지 않는다.

## 공개 계약

기존 `pitcheezy-service-game-v2` 및 정적 publish v1의 호환 가능한 optional 확장이다. `/api/service/games/{game_pk}`와 `v1/game/{game_pk}.json`은 같은 builder를 사용한다. 기존 파일에서 필드가 없거나 null이면 UI는 미제공 상태로 표시한다.

| 필드 | 의미 및 경계 |
|---|---|
| `game.game_type`, `series_description`, `series_game_number`, `games_in_series` | 이미 받은 경기/일정 메타데이터. WS 경기 유형과 시리즈 경기번호. 임의 추정 없음 |
| `game.probable_pitchers.away/home` | 제공된 예상 선발의 `id`, `name` 또는 null. 실제 선발 확정으로 부르지 않음 |
| `current.context.lineups.away/home` | 공개 cutoff까지 적용된 타순 `id/name/order`. 불완전 prefix는 빈 배열. 미래 교체를 적용하지 않음 |
| `current.context.boxscore.away/home.batters` | 동일 prefix의 `id/name/at_bats/hits/on_base`. `on_base`는 안타·볼넷·사구 출루 횟수이며 출루율이 아님 |
| `current.context.boxscore.away/home.pitchers` | 동일 prefix의 `id/name/pitches/strikeouts/earned_runs`. 자책점은 현재 feed의 기록원 판단을 과거 사건에 적용한 재구성이며 당시 확정값을 입증하지 않음 |
| `pitches[].context_before` | 기존 타자/투수/다음 타자 요약만. 전체 명단을 매 투구 반복 전송하지 않음 |
| `pitches[].actual.tracking` | `extension_feet`, `spin_rate_rpm`, `exit_velocity_mph`, `launch_angle_deg`, `distance_feet`; feed에 존재하는 유한 숫자만, 나머지 null |
| `pregame` | 기존 exact-game·경기 전일 이전 검증을 통과한 로컬 AVG/OPS/ERA 캐시를 live 경로에도 연결. 새 수집 없음 |
| `win_expectancy` | ready 캐시를 읽기만 하는 홈팀 기준 경기상태 승률. 카운트 비의존, 기존 frozen C0 기준. 정확한 snapshot/revision·pending pitch·policy identity와 유한 [0,1] 검증. 새 추론/queue 없음 |

`win_expectancy`는 pre-pitch, display `recommended`, `recommendation_shown=true`, 전체 공개 투구 receipts에 미확인 0, ledger `recorded`일 때만 조회한다. 캐시 miss/다른 상태/경기 종료에서는 숫자를 추정하지 않는다. 승률 추이, 투구 가치, 사건 기여도는 새로 공개하지 않는다. 기존 모듈 존재와 live 공개 준비는 구분한다.

미확인 receipt가 하나라도 있으면 현재 `context`는 null이고, 첫 미확인 투구부터 이후의 `context_before`도 null이다. 투구 결과를 숨긴 뒤 같은 결과가 hit/K 누계로 나타나는 우회 노출을 막는다. 앞선 확인 투구의 요약은 유지하고 이후 정상 admission에서 복원한다. 기존 actual/타석 결과 차단은 유지한다.

## 검증 기록

사용 인터프리터: `/Users/song/Projects/pitcheezy/.venv-observer/bin/python`. 모든 명령은 후보 worktree에서 `PYTHONPATH=apps/observer/backend`로 실행했다. 모델 payload·원격 자료·운영 프로세스·GPU는 사용하지 않았다.

1. `-m pytest apps/observer/backend/tests/test_ui_context_contract.py apps/observer/backend/tests/test_service_player_context.py -q`: 초기 합성 **15 passed**.
2. `-m pytest apps/observer/backend/tests/test_ui_context_contract.py apps/observer/backend/tests/test_service_player_context.py apps/observer/backend/tests/test_game_history_state.py apps/observer/backend/tests/test_publisher.py apps/observer/backend/tests/test_rec_pairing.py apps/observer/backend/tests/test_live_win_expectancy.py -q`: 당시 **172 passed**. **합성 전용 검증으로 부르지 않는다.** `test_publisher.py` 헤더의 합성 전용 설명과 달리 아래 3개가 `live_fixture.listed_dir`를 호출하여 레포의 기존 `fixtures/live/*.json` 세 파일을 임시 gzip 사본으로 변환했다. 첫 테스트는 849845 저장 feed를 재생했다. 실자료 읽기 금지 작업 범위를 벗어난 접근으로 PM에 즉시 보고했다. 외부 수집·원본 변경·fit·실제 모델 추론은 없었다. 이후 재실행하지 않았다.
   - `test_the_command_line_publishes_recorded_fixtures_once`
   - `test_a_second_publisher_over_the_same_output_fails_fast`
   - `test_the_command_line_refuses_to_run_without_a_store`
3. WE route/schema 추가 후 `-m pytest apps/observer/backend/tests/test_ui_context_contract.py -q`: 합성 **6 passed**.
4. 독립 검토의 미확인 누계 우회 노출 수정 및 per-pitch payload 축소 후 `-m pytest apps/observer/backend/tests/test_ui_context_contract.py apps/observer/backend/tests/test_service_player_context.py apps/observer/backend/tests/test_display_state.py -q`: 합성 **63 passed**. terminal 미확인 → 누계 차단 → 정정 수용 → 복원 회귀 포함.
5. `apps/observer/scripts/export_publish_contract.py`: 합성 `publish_fixture`로 examples/schema 재생성. 현재 스키마를 통한 새 숫자/구버전 필드 부재/잘못된 타입 검증은 위 신규 테스트에 포함.

6. 독립 재검토에서 과거 타석 미확인 결과도 WE 입력에 포함될 수 있음을 확인하여 callback을 receipt summary 이후로 옮기고 미확인 0 조건을 추가했다. `-m pytest apps/observer/backend/tests/test_ui_context_contract.py apps/observer/backend/tests/test_publisher.py::test_the_committed_examples_and_schemas_are_what_the_publisher_writes_now apps/observer/backend/tests/test_publisher.py::test_every_example_and_every_fixture_body_validates_against_its_schema -q`: 합성 **9 passed**. 이 두 publisher 테스트는 합성 `publish_fixture`만 사용하며 위 3개 기록 fixture 테스트는 실행하지 않았다.

7. 최종 단일 backend suite: `PYTHONPATH=apps/observer/backend /Users/song/Projects/pitcheezy/.venv-observer/bin/python -m pytest apps/observer/backend/tests/test_ui_context_contract.py apps/observer/backend/tests/test_service_player_context.py apps/observer/backend/tests/test_display_state.py apps/observer/backend/tests/test_live_win_expectancy.py apps/observer/backend/tests/test_publisher.py::test_the_committed_examples_and_schemas_are_what_the_publisher_writes_now apps/observer/backend/tests/test_publisher.py::test_every_example_and_every_fixture_body_validates_against_its_schema -q` → **83 passed, 2 warnings in 2.09s**. 앞서 확인한 합성 검증 경로만 실행했으며 기록 fixture CLI 테스트는 제외했다.

## 최종 독립 소스 수용

작성자와 다른 검토자가 최종 후보의 backend와 frontend 소스를 읽고 다음 수정을 확인했다. 검토 범위에서 남은 blocker는 없다.

- 미확인 투구 이후 현재/후속 선수 집계를 숨기고, 전체 공개 투구가 확인되기 전에는 WE helper를 호출하지 않는다.
- WE는 원문 receipt SHA와 모델 입력 fingerprint를 구분한다. 재사용 fingerprint에서 모델이 읽지 않는 `metaData.timeStamp`만 제외하며, 점수·주자·과거 정정 등 다른 변경은 허용하지 않는다.
- 프런트 WE는 pre-pitch·recommended·shown 조건과 연결/공개 파일 검증 상태를 모두 통과해야 표시한다. 모바일에도 승률 상세 진입점이 있다.
- `mergeDetail`은 key/index 대응이 달라지면 전체 재조회를 요청하여 이웃 투구의 조용한 삭제를 방지한다. 선택 상세는 안정된 투구 key를 유지하고 라이브 상태 갱신과 분리한다.
- 실제 투구 overlay는 미확인·좌표 결측을 제외한다. 화면 안 좌표를 이동하지 않으며 화면 밖 표시는 기존 화살표 규칙을 따른다.

Root가 후보 worktree의 `graphify update .` AST 갱신 완료를 보고했다: **10,854 nodes / 27,836 edges**. 독립 검토자는 그래프 갱신을 중복 실행하지 않았다.

기존 pytest 경고 2개는 Starlette/httpx와 AnyIO deprecation이다. 프런트 빌드·브라우저 검사와 commit은 PM의 별도 완료 기록을 따른다. 실제 서비스 데이터 커버리지·배포 및 실기기 검증은 이 검사 결과가 입증하지 않는다.
