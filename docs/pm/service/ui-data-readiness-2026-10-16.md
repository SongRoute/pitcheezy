# Pitcheezy UI 데이터 준비와 후속 작업 가이드

2026-10-16. [사용자 데이터 목록 원문](ui-data-request-2026-10-16.md)을 다음 UI 작업에서 재사용하기 위한 준비 문서다. 기본 경기 상황·투구·타석·라인스코어 경로는 구현되어 있다. 시리즈 집계, 상세 기록, 연구 모델의 225행동 인터페이스는 현재 공개 UI 계약과 연결되지 않은 부분이 있다. 이 문서는 구현 변경이나 새 스키마 확정이 아니다.

## 기준과 읽는 방법

- 감사 기준은 **후보 `codex/ws-integration-1016@222f46d08ff092014b86837a0a0489fe66b2f20c`**다. 문서를 저장한 정본 작업 디렉터리의 HEAD, 운영 배포본과 같다는 뜻이 아니다.
- 후보 위치: `/Users/song/Projects/pitcheezy-worktrees/codex-ws-integration-1016`. 아래 소스 경로·줄번호는 모두 이 커밋 기준이다. `B` = `apps/observer/backend/observer_app/`, `F` = `apps/observer/web/src/service/`.
- `구현 있음`은 builder → 공개 body → 프런트 adapter 경로가 있다는 뜻이다. 실제 경기에서 값이 모두 수신·채워졌다는 뜻은 아니다. nullable 필드의 실제 제공률, 시즌 캐시 존재, 운영 스위치·배포 상태는 별도 확인 대상이다.
- `부분`은 일부 필드만 연결되거나 조건부로 제공되는 상태다. `미연결`은 요청 의미의 공개 경로가 없다는 뜻이며, 내부 연구 함수의 존재 여부와 구분한다. `결정 필요`는 의미·출처·노출 정책을 먼저 정할 항목이다. 실제 채움과 운영 노출은 전 항목 `운영 미검증`이다.
- 원문은 수정 없이 별도 보존했다. 아래 표의 현행 설명과 결정사항이 원문 제안을 자동 확정하거나 과거 계약을 변경하지 않는다.

후속 작업자는 브랜치가 이동해도 다음과 같이 같은 소스를 재현할 수 있다.

```sh
git -C /Users/song/Projects/pitcheezy-worktrees/codex-ws-integration-1016 show 222f46d08ff092014b86837a0a0489fe66b2f20c:apps/observer/backend/observer_app/game_history.py | nl -ba
git -C /Users/song/Projects/pitcheezy-worktrees/codex-ws-integration-1016 show 222f46d08ff092014b86837a0a0489fe66b2f20c:apps/observer/web/src/service/api.ts | nl -ba
```

현재 화면 계약의 출발점은 [후보 service/model.ts](/Users/song/Projects/pitcheezy-worktrees/codex-ws-integration-1016/apps/observer/web/src/service/model.ts:18)와 [후보 service/api.ts](/Users/song/Projects/pitcheezy-worktrees/codex-ws-integration-1016/apps/observer/web/src/service/api.ts:366)다. 구형 `web/src/types.ts`만 보고 준비 여부를 판정하지 않는다.

## 전체 요구사항 대응표

공개 경로는 별도 언급이 없으면 `v1/game/<game_pk>.json` 본문의 경로다. 라이트 표시는 원문을 따른다. `공통`은 화면 공통, `일부`는 원문에서 일부만 라이트, `—`는 원문에 구분 없음이다.

### 0 공통 규칙

| 필드 | 라이트 | 현재 공개 경로 또는 없음 | 준비 상태 | 남은 일 |
|---|---|---|---|---|
| 모든 값 value + status | 공통 | nullable scalar, 객체별 status | 미연결 | 네 가지 공통 상태와 기존 null/상태의 매핑 결정 |
| received_at, source_ts | 공통 | feed.as_of, display_state.feed.received_at 등 객체 단위 | 부분 | 모든 레코드 시각 아님. 원천 사건시각·원천 revision·수신·발행시각 구분 |
| revision과 이전 값 | 공통 | receipt.corrected, receipts.version | 부분 | 내부 원장은 append-only. 개별 revision·이전 값 공개 방식 결정 |
| base_out/count/pitch/loc/action/outcome ID | 공통 | situation의 raw 값, actual.pitch_type, 위치 라벨 | 부분 | 연구 숫자 ID와 현재 서비스 의미를 연결할지 결정 |
| 구속 km/h | 공통 | actual.speed_mph → 클라이언트 ×1.609344 | 구현 있음 | 기존 변환 재사용. 타구 속도·거리 단위도 명시 |
| EXP-ID·seed·spec/data/RE24 버전 | 공통 | rec.policy, publish.contract/sequence 등 일부 | 부분 | 실제 제공 모델에 맞는 provenance 공개 묶음 결정 |

근거: `B/live_store.py:362–431`, `B/display_state.py:198–210,460`, `B/publish_schema.py:199–205`, `F/GameScreen.tsx:49–50`, `F/model.ts:18–77`.

### 1 경기 목록

| 필드 | 라이트 | 현재 공개 경로 또는 없음 | 준비 상태 | 남은 일 |
|---|---|---|---|---|
| 시리즈 팀별 승수 | ✓ | 없음 | 미연결 | 시리즈 범위·경기 결과 집계 기준 |
| 1~7차전·상태·필요 시 | ✓ | games[].series_game_number/games_in_series/status | 부분 | 전체 시리즈 묶음과 조건부 경기 상태 정의 |
| 시작 일시 | ✓ | games[].start/start_kst/date_kst | 구현 있음 | 원천 결측 처리 유지 |
| 최종 점수 | ✓ | games[].away_score/home_score | 구현 있음 | 종료/정정 처리 유지 |
| 진행 이닝·초말·현재 점수 | ✓ | 상세 game/current.situation, 목록 games[] | 부분 | schedule 목록은 inning/half 미제공 가능. 상단 카드 경로 선택 |
| 선발투수 | v3 | game.probable_pitchers, games[].probable_pitchers | 부분 | nullable 전달 존재. 실제 원천 공급 확인 |
| 승·패·세이브 투수 | v3 | 없음 | 미연결 | 원천 필드·공개 projection 추가 여부 |
| 경기 시간·관중 | v3 | 없음 | 미연결 | 정의·원천·결측 처리 |

근거: `B/live_feed.py:361–376`, `B/game_history.py:48–78`, `B/service_api.py:364–395`, `F/api.ts:189–195`.

### 2 라이브 상황

| 필드 | 라이트 | 현재 공개 경로 또는 없음 | 준비 상태 | 남은 일 |
|---|---|---|---|---|
| game_id·이닝·초말·점수 | ✓ | game.game_pk, current.situation | 구현 있음 | 원문 game_id와 명칭 매핑 |
| 볼·스트라이크·아웃·주자 | ✓ | current.situation.balls/strikes/outs/bases/runners | 구현 있음 | 숫자 count/base_out ID는 별도 |
| 타자 id·이름·타석 방향 | ✓ | current.batter.id/name/side | 구현 있음 | 미확인 좌우를 추정하지 않음 |
| 투수 id·이름·투구 손 | ✓ | current.pitcher.id/name/hand | 구현 있음 | pitcher_idx는 연구 번들 전용 매핑 |
| 오늘 투구수 | v3 | current.context.pitcher.pitches | 구현 있음 | prefix 불완전/확인 중이면 null 또는 context 숨김 |
| 오늘 타수·안타 | v3 | current.context.batter.at_bats/hits | 구현 있음 | 공식 최종 박스스코어와 계산 근거 구분 |
| 이번 타석 투구 번호 | ✓ | pitches[].pitch_number, next.pitch_key/display_state.pitch_key | 구현 있음 | pending key와 공식 pitchNumber 구분 |
| 직전 타석 이름·결과 | v3 | pas[].batter/result_text/event_label | 구현 있음 | 목록에서 직전 완료 타석 선택 |
| feed_status·마지막 수신 | ✓ | feed, display_state.feed/state | 부분 | ok/delay/lost 이름 매핑, 발행 지연과 원천 지연 분리 |
| record_status | v3 | pitches[].receipt.status/corrected | 부분 | received/unconfirmed는 저장소 수용 상태. 공식 최종 판정과 동일시 금지 |

근거: `B/live_feed.py:543–615`, `B/service_api.py:535–590`, `B/game_history.py:303–405,493–498`, `F/model.ts:35–47`.

### 3 투구

| 필드 | 라이트 | 현재 공개 경로 또는 없음 | 준비 상태 | 남은 일 |
|---|---|---|---|---|
| 투구 번호·타석 ID | ✓ | pitches[].pitch_number/pa_key/at_bat_number | 구현 있음 | 공식 번호와 내부 pending key의 연결 규칙 유지 |
| 구종 원본 코드 → pitch_id | ✓ | actual.pitch_type/pitch_label만 | 부분 | 연구 9종 매핑과 PO/IN/UN/미지 구종 표시 계약 |
| 판정 → outcome_id | ✓ | actual.call_label/result_label/event | 부분 | 숫자 ID 없음. 원천 call/event와 Statcast 매핑 간 adapter 필요 여부 |
| 판정 전후 카운트 | ✓ | situation_before만 공개 | 부분 | 내부 situation_after는 비공개. 종결구 이후 카운트 의미 포함 결정 |
| 구속 | ✓ | actual.speed_mph | 구현 있음 | null 유지, km/h 프런트 변환 재사용 |
| plate_x, plate_z | ✓ | actual.x/z | 구현 있음 | feet·포수 시점 유지 |
| sz_top/bot, z_norm, loc_id | ✓ | 개별 투구 필드 없음. game-level zone_bounds | 미연결 | 고정 존과 타자별 실제 존, 5×5 정의를 분리 |
| 회전수·익스텐션 | v3 | actual.tracking.spin_rate_rpm/extension_feet | 구현 있음 | 원천 미제공은 null |
| 무브먼트 | v3 | 없음 | 미연결 | 축·단위·좌표계 선정 |
| 타구 속도·발사각 | v3 | tracking.exit_velocity_mph/launch_angle_deg | 구현 있음 | 비인플레이/파울에서 결측 가능 |
| 타구 방향 | v3 | 일부 PA 결과 문장에 원천 방향 반영 | 부분 | 구조화 방향 필드는 없음 |
| xBA | v3 | 없음 | 미연결 | 출처·제공 범위 결정 |

근거: `B/game_history.py:81–87,123–158,418,544–561,600–605`, `B/service_api.py:28,167`, `B/domain.py:12–31`, `F/api.ts:216–240`.

### 4 추천

| 필드 | 라이트 | 현재 공개 경로 또는 없음 | 준비 상태 | 남은 일 |
|---|---|---|---|---|
| 대상 투구 번호·state_id | ✓ | next.pitch_key, pitches[].key; state_id 없음 | 부분 | 연구 state/맥락 축 도입 여부 |
| cluster_id·미지 타자 | ✓ | 라이브 공개 군집 없음 | 미연결 | 연구 손별 default 존재. 현재 라이브 모델에 그대로 이식하지 않음 |
| 상위 3개 action_id/pitch_id/loc_id·순위 | ✓ | next.rec/pitches[].rec.candidates의 rank/pitch_type/zone_id/target | 부분 | ARM-B 구종확률 순위이며 225행동 추천과 다름 |
| 행동별 Q RE24 | v3 | 없음 | 미연결 | 모델·가치 단위·부호와 공개 산출물 결정 |
| 모델 선택 비중 | v3 | candidates[].detail.probability → frontend probability | 부분 | 현행은 ARM-B 구종 확률. 225행동 완화 분포와 동일시 금지 |
| 구종별 25칸 위치 가치·valid | v3 | 없음 | 미연결 | 현재 위치 proxy를 Q지도/유효 마스크로 사용 금지 |
| 기록 시각·계산 상태 | ✓ | rec.recorded_at/status, display_state | 부분 | ready/unsupported/missing·계산대기 표현과 원문 상태 매핑 |
| 추천 이유 문장 | v3 | rec.reason은 미지원/결측 사유 | 미연결 | 모델 설명을 지어내지 않고 missing 유지. 실패 사유와 구분 |
| 실험·시드·spec·lookup_mode | v3 | policy 이름/identity 일부 | 부분 | 모델별 metadata 묶음 검토 |
| 미지원 투수·수신 공백·첫 연결 전 추천 | ✓ | absent/missing 및 pairing 거부 | 구현 있음 | 사후 재계산으로 과거 추천 채우지 않음 |

근거: `B/recommendation_adapter.py:86–127`, `B/live_armb.py:55–80`, `B/rec_pairing.py:1–29`, `B/live_store.py:590–611`, `F/api.ts:196–208`; 연구 fallback `src/pitcheezy/data/batters.py:98–115`, `src/pitcheezy/data/prepare.py:67–72`.

### 5 타석 책임 분해

| 필드 | 라이트 | 현재 공개 경로 또는 없음 | 준비 상태 | 남은 일 |
|---|---|---|---|---|
| V추천·V의도·V실제, 같은 lookup mode | 협의 | 현행 공개 game에 RE24 값 없음 | 미연결 | 모델·단위·의도 출처·lookup 계약 먼저 |
| 의도 위치 | 협의 | 일부 actual.catcher_setup는 수평 CV 근사 | 부분 | 투구 전 의도·실제 도달 위치와 구분. 의도 셀 확정 아님 |
| 실제 ΔRE24·동일 RE24 버전 | 협의 | 없음 | 미연결 | 버전·종결 결과·부호 계약 |
| 전략·투수 실책·타자·운 네 성분 | 협의 | 없음 | 결정 필요 | 식과 식별 가능성 검토; 기존 WE 대비/잔차와 다름 |
| 합 정합성·실패 시 숨김 | 협의 | 요청한 네 성분 검증 경로 없음 | 미연결 | 부호 포함 invariant와 노출 기준 확정 후 구현 |

연구 `src/pitcheezy/decomp/__init__.py:1`은 placeholder다. `B/event_analysis.py:190–208`의 기존 WE %p 대비·잔차는 원문의 RE24 네 성분과 같은 산출물이 아니다. `src/pitcheezy/interfaces/value.py:66`의 lookup은 snap 경로이며 bilinear를 구현된 것으로 취급하지 않는다. 같은 칸이라는 이유만으로 “투수 실책 0”이라는 해석을 붙이지 않는다.

### 6 선수 기록

| 필드 | 라이트 | 현재 공개 경로 또는 없음 | 준비 상태 | 남은 일 |
|---|---|---|---|---|
| 이름·팀·포지션·타투 방향 | ✓ | person 이름/ID/side 또는 hand, 경기 팀, jersey | 부분 | 독립 선수 team/position·타투 양쪽 메타 확장 |
| 타자 정규 PA·AVG·OBP·SLG·OPS·wRC+ | 일부 | pregame.players[id].avg/ops만 | 부분 | PA/OBP/SLG/wRC+ 및 상수·출처 |
| 타자 포스트 기록 | 일부 | 없음 | 미연결 | 정규와 분리된 기간·집계 |
| 투수 정규 IP·ERA·WHIP·FIP·K율·BB율 | 일부 | pregame.players[id].era만 | 부분 | IP/WHIP/FIP/율 분모·상수·출처 |
| 투수 포스트 기록 | 일부 | 없음 | 미연결 | 별도 기간·집계 |
| 기준 날짜·포스트 경기수 | v3 | pregame.as_of, players[id].as_of | 부분 | 포스트 경기수·시즌/단계 메타 |
| 오늘 타자 AB·H·HR·BB·K | ✓ | context.batter.at_bats/hits/on_base | 부분 | HR/BB/K 분리. on_base는 H+BB+HBP |
| 오늘 투수 투구수·IP·H·BB·K·R | ✓ | context.pitcher.pitches/strikeouts/earned_runs | 부분 | IP/H/BB/R 추가. earned_runs는 R이 아님 |

근거: `B/player_metadata.py:26–69`, `B/service_api.py:68,112–126,579–586`, `B/game_history.py:106–112,278–405`, `F/api.ts:354–364`.

### 7 구종

| 필드 | 라이트 | 현재 공개 경로 또는 없음 | 준비 상태 | 남은 일 |
|---|---|---|---|---|
| 시즌 구종 사용률·평균 구속 | — | 없음 | 미연결 | 시즌 전체 집계·기간·결측 분모 |
| 오늘 구종별 투구수·평균 구속 | — | 수신 pitches를 PitchMixSheet에서 집계 | 구현 있음 | 확인된 원본 구종별 관측치라는 표기 유지 |
| 구종별 주요 성과 | 협의 | 없음 | 결정 필요 | 지표 선정 후 자료·표본 계약 |

`F/LivePanels.tsx:20–25`는 미확인 투구를 제외하고 구속 있는 공만 평균한다. “경기 전체를 모두 수신”한 통계라는 뜻이 아니다. `B/profile_reference.py:328–359`의 조건부 표본 count/share 참고 카드는 시즌 전체 사용률·평균 구속의 대체물이 아니다.

### 8 경기 기록

| 필드 | 라이트 | 현재 공개 경로 또는 없음 | 준비 상태 | 남은 일 |
|---|---|---|---|---|
| 문자 타석 결과·투구 목록 | — | pas, pitches | 구현 있음 | 기존 key 기반 연결 재사용 |
| 교체·도루·폭투 등 사건 로그 | — | 내부 상태 갱신에 일부 반영 | 부분 | 별도 non-pitch event 공개 목록·시간 |
| 양 팀 라인업 순서·이름·포지션·오늘 결과 | — | current.context.lineups의 order/id/name | 부분 | 포지션·결과 연결 |
| 타자 박스 AB·H·RBI·BB·K | — | boxscore.batters의 at_bats/hits/on_base | 부분 | RBI/BB/K |
| 투수 박스 IP·투구수·H·K·R | — | boxscore.pitchers의 pitches/strikeouts/earned_runs | 부분 | IP/H/R와 자책점 분리 |
| 이닝별 득점·RHE | — | linescore, pitches[].linescore_before | 구현 있음 | 공개 시점 cutoff·provenance 유지 |

근거: `B/game_history.py:313–327,396–405,435–448,607–611`, `B/service_api.py:165–170,205–243`, `F/LivePanels.tsx:6–18`.

### 9 승리확률

| 필드 | 라이트 | 현재 공개 경로 또는 없음 | 준비 상태 | 남은 일 |
|---|---|---|---|---|
| 현재 팀별 승리확률 | — | win_expectancy.home_probability, 조건부 | 부분 | C0 의미·노출 조건 유지. 원정은 보완확률로 표시 가능 |
| 타석별 추이 | — | 현재 service/public 경로 없음 | 미연결 | 시간·사건 단위·모델 일관성 |
| 경기 전 승리확률 | — | 현재 경로 없음 | 미연결 | 모델/기준 선택 |
| LI·타석별 WPA | — | 없음 | 미연결 | 정의·출처·표본/이벤트 정렬 |
| 크게 움직인 장면 | — | 현재 service/public 경로 없음 | 미연결 | 신뢰할 수 있는 추이 확보 후 파생 |
| 모델명·기준 설명 | — | raw WE source/basis/perspective/identity | 부분 | frontend adapter는 status/home_probability만 유지. 설명 정보 연결 |

근거: `B/live_armb.py:27–30,140–178`, `B/service_api.py:587–590`, `F/api.ts:382`. `B/watch_summary.py:152–162`에는 legacy watch용 투구별 곡선·큰 변화 산출이 있지만, 현재 service/public 화면에 연결된 타석별 WPA 기능이 아니다.

## 공통 계약과 해석에서 먼저 맞출 것

### 연구 225행동과 현재 서비스 추천

연구 스펙은 9구종 × 25위치 = 225행동, outcome 11종이다. 현재 ARM-B 공개 추천은 **구종 정책확률 내림차순 + 실제 투구 분포에 기초한 3×3 위치 근사**다. 위치는 평가된 정책 행동이라고 주장하지 않는다. `zone_id` 문자열을 연구 `loc_id` 숫자로 이름만 바꾸거나 구종 확률을 action 확률로 부르면 의미가 달라진다. 근거: `docs/interface-spec.md:30–34`, `B/domain.py:12–31`, `B/recommendation_adapter.py:94–127`.

연구 상태는 `((count_id × 24 + base_out_id) × K + cluster_id) × C + context_id`다. C=1에서만 이전 v1 ID와 같다. C>1 모델을 사용하면 C·context_kind·context_id까지 함께 고정해야 한다(`docs/interface-spec.md:16–29`, `src/pitcheezy/interfaces/states.py:3`).

연구 Q의 보상은 투수 관점 **−ΔRE24**, V는 **max Q**다(`src/pitcheezy/policy/vi.py:3–7,19–27`). 따라서 원 Q를 오름차순 정렬하여 “예상 실점 낮은 순”으로 표시하면 역순이 된다. 예상 실점으로 변환해 표시하려면 부호·기준점·단위를 별도로 정의해야 한다. 원문과 `docs/interface-spec.md:76`의 네 성분 합식도 이 부호와 함께 재검토할 대상이며 검증된 분해 식으로 인용하지 않는다.

### 값 상태와 수신 및 정정

현재 public contract는 전체 값의 네 상태 래퍼를 요구하지 않는다. 전면 value/status 래핑은 기존 공개 v1 호환성 검토가 필요하며 이 문서에서 적용을 결정하지 않는다. `null`, 추천 status, display_state, receipt가 각각 다른 의미를 가진다. `missing`과 `computing`, 기록 확인 중을 합치지 않도록 다음 작업에서 매핑을 정한다. publish.sequence는 파일 발행 순번이며 투구 revision이 아니다. received는 저장소가 수용한 관측이고 공식 판정이 더는 정정되지 않는다는 보증이 아니다.

내부 pitch ledger는 변경을 append-only로 보존하지만 네트워크 공백·저장소 장애 중의 모든 중간 정정을 보존했다고 증명하지 못한다(`B/live_store.py:413–431`). 전 레코드 시각을 추가하려면 source event time, provider revision stamp, local receipt, publish time을 따로 정의해야 한다. 현재 원천 지연은 45초 임시 기준(`B/display_state.py:40`), 공개 파일 stale/lost는 최소 세 poll 간격을 반영하는 별도 임시 기준(`F/publish.ts:17–41`)이다.

### 당시 기록된 추천

동일 decision의 첫 추천 내용은 바뀌지 않으며 충돌은 추가 행으로 보존한다. 원천 정정으로 입력 state가 바뀌면 같은 pending pitch key라도 새 decision이다(`B/live_store.py:590–611`). 과거 투구와 연결할 때는 그 투구 최초 수신 이전 기록, 저장소 순서, 동일 투타·카운트·주자·점수 등 엄격한 대응 조건을 확인한다(`B/rec_pairing.py:8–29`). 이것은 팬이 그 순간 실제 화면에서 봤다는 증거나 물리적인 투구 발생 전 시각의 독립 증거가 아니다. 공백은 사후 계산한 추천으로 채우지 않는다.

### 선수 캐시와 오늘 기록

시즌 캐시는 `PITCHEEZY_PLAYER_SNAPSHOT_DIR` 등 별도 경로가 있어야 하고 exact game ID, `regular_season_through_previous_day`, 경기 officialDate보다 엄격히 앞선 파일·선수별 as_of를 만족해야 한다. 허용 통계는 AVG·OPS·ERA뿐이다(`B/player_metadata.py:36–69`). 현재는 캐시 생성·실제 파일 존재·일 1회 갱신을 입증한 상태가 아니다. 경기 후 누적 seasonStats를 과거 경기 전 정보로 대체하지 않는다.

오늘 context는 공개 event prefix로 재구성하며 확인 중인 투구가 있으면 숨긴다. 출루는 H+BB+HBP이고 earned_runs는 자책점이다. 과거 자책점 재구성에는 현재 feed의 기록원 판정이 반영되므로 “당시 알려졌던 판정”으로 쓰지 않는다(`B/game_history.py:278–283`, `B/service_api.py:579–586`).

### C0 승리확률

C0는 타석 시작 상태로 학습한, 볼·스트라이크를 반영하지 않는 홈팀 경기상태 승리확률 추정이다. pitch WPA가 아니다. 동일 입력 캐시, pending pitch와 유효한 policy identity, snapshot 검증이 필요하다(`B/live_armb.py:140–178`). 공개 route는 pre_pitch·recommended·추천 표시·확인된 receipts·recorded ledger 조건에서만 연결한다(`B/service_api.py:587–590`). 단순히 raw 숫자가 있다고 항상 표시하지 않는다. 새 추이·LI·WPA·경기 전 확률은 이 조건부 단일 값 기능과 별도로 설계한다.

## 10 결정사항

아래 다음 선택은 후속 검토 항목이며 이 문서에서 승인·확정하지 않는다.

| 항목 | 현행 | 미정 | 다음 선택 |
|---|---|---|---|
| 실시간 원천·지연 | MLB schedule/feed/live 연결 코드, 원천 45초·발행 별도 임시 기준 | 운영 제공률·실측 지연·최종 임계값 | 현행 경로를 기준으로 등록된 점검 계획 작성 |
| 추천 기록 시점·정정 | 동일 decision 불변, 입력 정정은 새 decision, 엄격한 과거 pairing | UI가 새/이전 추천을 어떻게 안내할지 | 기존 원장 의미를 보존한 표현 선택 |
| 미지 타자 군집 | 연구 손별 default fallback 존재, 라이브 cluster 공개 없음 | 사용할 모델과 fallback 정책 노출 | ARM-B 유지 또는 별도 225행동 계약 선택 뒤 결정 |
| wRC+·FIP | 공개 경로 없음 | 출처·리그 상수·기간·버전 | 외부 출처/자체계산 비용·재현성 비교 |
| 승리확률 | 조건부 C0 현재값 존재 | 전체 추이·WPA·LI·경기 전 모델 | 현재값 설명 연결과 추가 연구 기능 분리 |
| 구속 변환 | API mph, 프런트 km/h 기본 | 다른 tracking 단위 표기 | 기존 변환 재사용 |
| 책임 분해 노출 | 요청 RE24 네 성분 미연결 | 의도·식·단위·검증·노출 여부 | 합식/부호/식별 검토 전 협의 중 유지 |
| 값 상태·ID·좌표·버전 | 서비스 계약과 연구 스펙이 다름 | 공통 adapter 또는 별도 계약 | 라이트에 필요한 최소 필드부터 의미 합의 |
| 시리즈 범위 | 날짜별 경기와 일부 series metadata | 승수·필요 시·1~7 전체 범위 | 시리즈 모델과 종료/취소 규칙 정의 |

## 부록 API 묶음과 현행 정적 발행 대응

공개 UI는 정적 파일을 polling한다. 스트림이나 아래 제안별 endpoint가 이미 있다는 뜻이 아니다. 내부 직접 경로는 `/api/service/games`, `/api/service/games/{game_pk}` 등이고 공개 번들은 이를 호출하지 않는다. 파일별 sequence 역행 거부·ETag·오류 시 마지막 값 유지 규칙을 재사용한다(`F/api.ts:40–95`, `F/publish.ts:1–10`).

| 원문 묶음 | 현행 대응 | 상태와 다음 작업 |
|---|---|---|
| series | v1/games/<YYYY-MM-DD>.json, dates.json, manifest.json | 날짜 목록만. 시리즈 전용 집계 미연결 |
| live_snapshot | v1/game/<game_pk>.json의 current/pitches/next | 전체 게임 파일 polling. 별도 투구 스트림 없음 |
| pitch/{n} | 같은 game 파일 pitches[]에서 key로 선택 | 별도 상세 endpoint 없음. raw 숫자 ID·추가 tracking 별도 |
| rec_detail | next.rec 또는 pitches[].rec | 상위 후보만. Q/valid/25칸 지도 없음 |
| player/{id} | game.pregame.players와 current.context | 독립 player endpoint·일 1회 cache 작업 미확정 |
| arsenal/{pitcher} | 수신 pitches의 프런트 집계 | 오늘 관측값. 시즌 집계 없음 |
| game_log | game.pas/pitches/linescore/current.context | 타석 로그 중심. non-pitch 사건목록·완전 박스 보강 필요 |
| wp | game.win_expectancy | 조건부 현재값. 추이 전용 파일 없음 |

`v1/health.json`은 발행/경기 점검 상태, `publish.published_at`과 sequence는 발행 freshness에 사용한다. manifest의 publish_interval_s가 polling 기본값을 조정한다. cadence의 기본 숫자를 실측 지연으로 인용하지 않는다(`F/api.ts:98–145`, `F/publish.ts:17–45`).

## 후속 작업 순서와 완료 조건

1. **UI 최소 계약 합의.** 현재 ARM-B 화면과 연구 225행동 요구를 분리하고, 상태·시각·ID·좌표·단위·버전·Q 부호의 대응표를 확정한다. 완료 조건: 라이트 필수 항목마다 정확한 원천과 null/상태 의미가 하나로 정해지고 미정은 명시된다.
2. **라이트 필수 누락의 작은 구현 계획.** 시리즈 승수/조건부 경기, 개별 존 상하단/정규화, 투구 전후 카운트, 부족한 오늘 기록 중 실제 화면에 필요한 것을 우선순위화한다. 원문 라이트에 포함된 wRC+·WHIP·탈삼진율도 이 단계에서 출처·분모를 정해 제공할지, 미제공 표시를 유지할지 결정한다. 완료 조건: backend builder, public schema, frontend adapter의 변경 범위와 필요한 한계 검사가 작업별로 적혀 있다.
3. **상세 정보 공급 계획.** 시즌 캐시·포스트 구분·wRC+/FIP·박스 R/ER·non-pitch 사건·추가 tracking의 출처와 기간을 정한다. 완료 조건: 각 통계의 분모·as_of·버전·결측 기준·갱신 책임자가 정해진다. 원자료 실행은 기존 등록·독립 검토 게이트를 따른다.
4. **추천 및 승리확률 설명 연결.** 기존 불변 원장·pairing·C0 조건을 유지하며 provenance 설명과 실제 값 누락 표시를 설계한다. 완료 조건: 정책확률/성공확률, 위치근사/평가된 행동, 수신전/물리적 투구전, 현재 WE/WPA를 혼동하지 않는 화면 계약이 검토된다.
5. **승인된 구현 후 경로 검증.** 변경된 최소 범위에 적절한 합성·계약 검사를 붙이고 실제 제공률은 별도 등록된 확인으로 다룬다. 완료 조건: 누락/정정/연결 끊김/이전 파일/미지원 상태에서도 화면이 가짜 0이나 사후 추천을 만들지 않는다. 배포 여부는 별도 결정이다.

## 다음 작업자의 시작 체크리스트

- 양쪽 상황판과 열린 우편, 최신 사용자 지시·서비스 state를 읽고 다른 작성자와 작업 범위를 겹치지 않는다.
- 원문과 이 가이드를 먼저 읽고 감사 커밋과 실제 작업 기준 커밋을 비교한다. 정본 루트의 기존 dirty 파일 및 `docs/product/` 작업을 보존한다.
- 해당 field group의 후보 소스를 `git show`로 재확인하고, 현재 branch에 통합됐는지 별도로 확인한다.
- 기능 경로 존재, 데이터 실제 채움, 운영 노출을 구분한다. 새 자료 열람·fit·운영 변경은 이 문서가 승인하지 않는다.
- 구현 전에 공통 계약 결정표의 관련 항목과 완료 조건을 정한다. 기존 helper/adapter를 재사용하고 미정 수치를 채우지 않는다.
