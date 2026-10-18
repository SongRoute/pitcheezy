# Pitcheezy Observer — 로컬 관전 MVP

과거 경기를 한 공씩 열어 보면서 **다음 구종·목표 구역 → 실제 투구 → 타석 마지막 공의 분석**을 확인하는 야구팬용 앱이다. 실시간 중계 서비스는 아니다.

## 실행

개선판은 연구용 가상환경을 참조하지 않는 독립 런타임으로 실행한다. 현재 설치된 `.venv-observer-standalone`과 T7 Shield의 동결 모델·개선 데이터가 필요하다.

```sh
sh apps/observer/run_standalone.sh
```

관전 화면은 <http://127.0.0.1:8768>, 영상 라벨 보관함과 명시적 추적 실행은 <http://127.0.0.1:8768/video-lab>이다. 12경기·249타석·975구, 선수 이름/날짜 검색, 이전 기록 기반 구속 관측, 후보별 결과 분포 설명을 제공한다. 영상 검토실은 경기 결과가 포함된 별도 연구 공간이다. 저장한 라벨의 기본 추적기는 고정 템플릿이며, 광류 후보는 연구 CLI에서만 평가했다.

개선 데이터·관전 DB·영상 라벨 DB·실험 결과는 SSD의 `runs/observer-improvement-v2`에 보존한다. 기존 연구 `.venv` 의존성은 제거했지만 SSD 모델 번들과 현재 머신의 경로 설정은 여전히 필요하다. 다른 컴퓨터에 배포 가능한 설치 패키지라는 뜻은 아니다. 개선 결과와 채택 판단은 [IMPROVEMENT_LOG.md](IMPROVEMENT_LOG.md)를 참고한다.

아래 명령은 원래 v1 환경을 재현할 때 사용한다.

저장장치 **T7 Shield**를 연결한 상태에서 프로젝트 루트에서 실행한다.

```sh
sh apps/observer/run.sh
```

브라우저에서 <http://127.0.0.1:8766> 접속. 경기·타석 선택 → **타석 관전하기** → 실제 공 공개 → 마지막 공의 목표 구역 입력 순서다. 종료는 실행 터미널에서 `Ctrl-C`. API가 별도 분석 워커도 함께 종료한다. 같은 브라우저에서 다시 접속하면 마지막 관전 세션을 복원한다. 같은 타석을 새로 열면 독립된 관전 세션을 만든다.

포트를 바꾸려면 `PITCHEEZY_OBSERVER_PORT=8767 sh apps/observer/run.sh`. 기본 바인딩은 로컬 주소이며 외부 공개 설정을 포함하지 않는다.

## 지연 라이브(데모, 검증 전 실험 버전)

`GET /api/live/games?date=YYYY-MM-DD`, `GET /api/live/{gamePk}/state`가 MLB Stats API 공개 피드의 현재 경기 상태를 `delay_s`(기본30초) 지연해 투구 전 입력으로 바꾼다. 설정은 `live_config.json`. 추천은 동결 ≤2025 ARM-B 정책(아래 “서비스로 실행”)이며, 예전 번들(투수 6명) 핀은 `python -m observer_app.live_feed` CLI에서만 쓴다. 원본 피드는 `apps/observer/live_snapshots/`(gitignore)에 gzip으로 기록되고, `PITCHEEZY_OBSERVER_LIVE_REPLAY_DIR` 또는 `python -m observer_app.live_feed --replay-dir DIR --game PK`로 오프라인 재생한다. 선택 기능 `PITCHEEZY_OBSERVER_LIVE_STORE=<db>`를 주면 지연 버퍼를 SQLite에도 써서 재시작 뒤 이어서 보여 준다(기본은 꺼짐, `deploy/staging/README.md`).

## 중계 영상과 함께 보기(watch-along, 검증 전 실험 버전)

끝난 경기 하나를 중계 영상과 나란히 한 공씩 넘겨 본다. 실시간이 아니다. 추천은 ML-POLICY-VAL-v1에서 동결한 ≤2025 ARM-B 정책(τ 0.1, samples 6, pitch_cap 12, 식별자 `a6dffaea…`)이고, 위치는 그 투수의 ≤2025 실제 투구 분포(G0 TRAIN 투구 풀)에 관전 앱의 커널 규칙을 적용한 근사다. 레거시 번들(투수 6명)은 쓰지 않는다.

```sh
# 1) 사전 계산(메인 .venv, T7 필요, 약 30초): 공개 GUMBO 최종 피드 -> Statcast 형식 -> 동결 정책
.venv/bin/python scripts/demo_precompute.py precompute --game-pk 849843
# (선택) 열 대응 점검: 2025 정규시즌 30경기 GUMBO vs 원본 Statcast
.venv/bin/python scripts/demo_precompute.py check-mapping --games 30
# 2) 관전 앱 실행 후 http://127.0.0.1:8766/watch?game=849843
sh apps/observer/run.sh
```

- 산출물은 `/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/DEMO-WS-2026/`에만 쓴다: `feeds/`(최종 피드), `games/{gamePk}/run-*/`(Statcast 형식 입력·원장·manifest), `watch/{gamePk}.json`(앱이 읽는 최신 자료), `style/`(2026 타자 성향 스냅샷, ≤2025 행만), `mapping_check/`. `PITCHEEZY_WATCH_DIR`로 읽는 위치를 바꿀 수 있다.
- 요청은 연구 경로 그대로다(`policy_requests.pa_requests` → `bind_components` → `build_runtime`, 완전 식별자가 `a6dffaea…`가 아니면 거부). 원장에서 **실제 공을 읽는 거부**(실제 구종이 그 투수의 TRAIN 목록 밖: `LOGGING_POSITIVITY`와 그로 인한 타석 나머지)는 투구 전 화면에 드러나면 안 되므로 같은 후보 확률을 따로 계산해 보여 주고 `path: service_fallback`으로 표시한다. OPE 원장 상태는 그대로 남긴다.
- API: `GET /api/watch/games`, `GET /api/watch/{gamePk}`(투구 전 정보만), `GET /api/watch/{gamePk}/reveal/{index}`(실제 공·승리확률 변화). 화면: 상황판(이닝·아웃·카운트·점수·주자·투수·타자) → 추천 구종과 대략 위치 → 공개 → 실제 공·결과·승리확률 변화. `#i=N` 링크로 특정 공에 바로 간다.
- 열 대응 한계: `effective_speed`는 피드에 없어 `release_speed`로 대신한다(2025 30경기 기준 정규화 단위 오차 중앙값 0.10, TRAIN 중앙값 대체는 0.73). `pfx_x/pfx_z`는 `coordinates.pfxX/pfxZ`(40ft 정의, 약 1.65배 차이)가 아니라 `breaks.breakHorizontal`(부호 반대)·`breakVerticalInduced`를 12로 나눈 값이 Statcast와 0.1인치 반올림 안에서 같다.
- 브라우저 확인: `PLAYWRIGHT_BROWSERS_PATH=... node apps/observer/web/tests/watch-along-smoke.mjs http://127.0.0.1:8766 849843`

## 서비스로 실행 · 폰에서 보기 (월드시리즈 데모, 검증 전 실험 버전)

한 명령으로 웹 화면 빌드(바뀐 경우만)와 백엔드를 띄운다. 첫 화면 `/watch`에 **지금 경기(지연 중계)**와 **끝난 경기(중계 영상과 함께 보기, 추천 가능 비율 표시)**가 나온다. 예전 과거 기록 관전 앱(`/`)은 T7에 DB를 쓰므로 이 실행에서는 끄고 `/`는 `/watch`로 보낸다(`PITCHEEZY_OBSERVER_LEGACY=1`이면 켠다).

```sh
# 이 맥에서만
sh apps/observer/serve.sh                                    # http://127.0.0.1:8766/watch
# 폰(같은 tailnet)에서도: Tailscale 주소에만 바인딩한다. 0.0.0.0·공인 IP·Funnel은 거부한다.
PITCHEEZY_OBSERVER_HOST=tailscale sh apps/observer/serve.sh
```

### 폰에서 보기

1. 폰에 Tailscale 앱을 설치하고 맥미니와 **같은 tailnet 계정**으로 로그인한다(연결 스위치 켬).
2. 맥미니에서 위 두 번째 명령으로 실행한다(또는 아래 launchd 예시로 자동 시작).
3. 폰 브라우저에서 `http://song-macmini.taila189e6.ts.net:8766/watch` (MagicDNS 이름) 또는 `http://100.108.252.111:8766/watch` (Tailscale IP)를 연다. 홈 화면에 추가해 두면 앱처럼 열린다.
   - 이름·IP 확인: `tailscale status --self`, `tailscale ip -4`. 열리지 않으면 폰의 Tailscale이 켜져 있는지, 맥에서 `curl http://$(tailscale ip -4):8766/api/health`가 되는지 본다.
   - 공개 주소는 만들지 않는다(`tailscale funnel`/`serve` 쓰지 않음). tailnet 밖에서는 접속되지 않는다.

### 화면

- **지연 중계** `/live?game=N`: MLB 공개 피드를 **시청자의 중계 지연만큼** 늦게 보여 준다(D134). 상황판(점수·이닝·아웃·주자·볼카운트·투수·타자) → **다음 공의 추천 구종과 대략 위치**(숫자는 접힘, D49) → **이 투수 오늘**(이 경기에서 던진 구종 비율·평균 구속, 추천이 없는 투수에도 표시) → 공이 들어오면 **방금 던진 공**(실제 구종·결과, 직전 추천과 비교, 승리확률 변화 카드) → 이 타석 투구 순서 띠. 2초마다 자동 갱신, 피드 장애 시 마지막 상황을 유지하고 다시 시도한다.
  - **중계 맞추기**: "내 화면이 N초 늦음"을 ±1·±5초 버튼으로 맞춘다(기기에 기억, 기본 `delay_s` 30초, 최대 `max_delay_s` 120초). 대략 맞춘 뒤 "공이 미트에 들어오는 순간 누르기"로 미세 조정한다(±8초 안의 공만 인정). 서버는 받은 시각이 아니라 **경기 사건 시각**(공의 `endTime`) 기준으로, 그 지연만큼 지난 상황만 보여 준다. 기록이 그보다 늦게 도착하면 도착하는 대로 보여 주고 안내한다.
- **실제 속도로 다시 보기** `/live?game=N&replay=0`: 끝난 경기의 최종 피드를 지연 중계와 **같은 경로**(피드 → 상황 → 동결 정책)에 기록된 투구 시각 간격 그대로 흘려보낸다. `replay`는 시작 투구 번호(회 버튼이 그 회 첫 공 15초 전부터 시작), `t0`는 시작 시각(주소에 고정되어 새로 고침·다른 기기에서도 같은 시계), `speed`는 배속. ±버튼은 재생 시계를 앞뒤로 옮긴다. `live_config.json`의 `replay_lag_s`로 피드 도착 지연을 흉내 낼 수 있다(기본 0).
- **중계 영상과 함께 보기** `/watch?game=N`: 끝난 경기를 한 공씩(추천 → 공개 → 실제). 목록·타임라인에 최종 점수는 없다. 첫 방문에 한 화면 안내("이렇게 보세요", 브라우저 localStorage `pitcheezy.guide.v1`에 닫음 기억, 헤더 "보는 법"으로 다시 열기), 영상 맞추기용 회(1초·1말…) 바로가기, 화면 아래 고정 버튼(← / 실제 투구 공개·다음 공 / 공개 없이 다음), 어두운 모드. "숫자로 보기"는 모델이 그 구종을 고를 확률과 이 투수 평소(기준 모델) 비율이며, 볼·스트라이크·인플레이 확률은 사전 계산 자료에 없다(미측정).
- 두 화면 모두 배지 "검증 전 실험 버전 · 위치는 실제 투구 분포 근사". 승리확률 카드는 사건 기여도(D43) 중 승리확률 변화만 보여 주며, 볼카운트는 반영하지 않는다(C0 WE는 타석 시작 상태 모델).

### 추천이 계산되는 방식(지연 중계)

`/api/live/{gamePk}/state?delay=N`은 받은 피드를 백그라운드 스레드 하나에 넘긴다(요청은 새 상황의 계산을 최대 1초 기다린다. 최근 두 경기의 묶음을 유지한다). 스레드는 `scripts/demo_precompute.LivePolicy`로 **지금까지의 경기 행 + 다음 공 자리표시 행**을 만들고, 사전 계산과 같은 `bind`(S2 부품 식별자, 정책 식별자 `a6dffaea…`가 아니면 거부)로 묶은 뒤 런타임의 투구 전 판정(`_evaluate`, 기록된 행동 없음, 원장 기록 없음)을 돌린다. 묶기(약 5초)는 경기당 한 번이고, 이후 상황은 새로 생긴 행만 묶기와 같은 방식으로 더한다(이미 묶인 행이 바뀌면—예: 투구 전 주자 이동·투수 교체—처음부터 다시 묶는다). 849843 262개 상황 기준 한 상황당 중앙값 0.13초·p95 0.16초(이전 5.4초), 계산당 메모리 증가는 2.5MB→상한 있는 캐시만(tracemalloc 50회 약 0.09MB/회)이며, 가장 최근(지연 전) 피드를 먼저 계산해 두므로 30초 지연 화면에는 보통 바로 나온다. 계산 중이면 "추천 계산 중…", T7이 없거나 정책을 못 읽으면 "추천 모델을 준비하지 못했습니다"를 보여 주고 상황판은 계속 동작한다. 끝난 경기 두 개(849843·849851)를 잘라 만든 29개 상황에서 사전 계산과 상태·후보 확률·구역이 모두 같았다.

### 끝난 경기 추가(사전 계산)

```sh
.venv/bin/python scripts/demo_precompute.py sync --since 2026-09-29   # 새로 끝난 F/D/L/W 경기만, 경기당 약 20–30초
```

한 경기 실패는 나머지를 막지 않고, 동시에 두 번 돌면 뒤의 것은 바로 끝난다(잠금 파일). 산출물은 `DEMO-WS-2026/`에만 쓴다.

### 자동 시작(예시, 설치하지 않음)

`apps/observer/deploy/com.pitcheezy.observer.plist.example`(로그인 시 Tailscale 주소로 서비스 시작, 비정상 종료 시 재시작)과 `com.pitcheezy.demo-sync.plist.example`(30분마다 `sync`). 파일 안 주석대로 `~/Library/LaunchAgents/`에 복사하고 `launchctl bootstrap gui/$(id -u) …`로 켠다.

### 운영 확인

- 상태: `GET /api/health` → `demo.status`(ok/degraded), `storage_mounted`(T7), `watch_games`, `live_policy.state`(loading/ready/unavailable)·`computed`·`last_seconds`.
- 로그: 한 줄에 JSON 하나(`apps/observer/logging.json`). API 요청마다 method·path·status·ms, 지연 중계 계산마다 key·seconds.
- 설정(환경 변수): `PITCHEEZY_OBSERVER_HOST`, `PITCHEEZY_OBSERVER_PORT`, `PITCHEEZY_OBSERVER_PYTHON`, `PITCHEEZY_WATCH_DIR`, `PITCHEEZY_OBSERVER_LIVE_DELAY_S`, `PITCHEEZY_OBSERVER_LIVE_ARMB=0`(지연 중계 추천 끄기), `PITCHEEZY_OBSERVER_LIVE_REPLAY_DIR`.
- 리허설(경기가 없을 때 지연 중계 화면 시험): `.venv/bin/python scripts/demo_precompute.py live-replay --game-pk 849843 --start 40 --count 40` 뒤 `PITCHEEZY_OBSERVER_LIVE_REPLAY_DIR=<출력 폴더> PITCHEEZY_OBSERVER_LIVE_DELAY_S=15 sh apps/observer/serve.sh`, `/live?game=849843`. 10초마다 한 공씩 진행한다.
- 저장소를 켠 실행(RC, `rc/ws-2026`): `PITCHEEZY_OBSERVER_LIVE_STORE=<새 SQLite 경로>`를 주면 폴링한 스냅샷이 `live_store.py`를 거친다(투구 식별·최초 수신 불변·정정 append-only·중복/지연 도착/재시작 복원). 이때 투구 전 추천은 `served_recommendations` 원장에 먼저 기록된 것만 화면에 나가고, 기록하지 못하면 그 추천은 표시하지 않는다(`PITCHEEZY_OBSERVER_REC_LEDGER=0`으로 원장만 끌 수 있다). 수신 시각 `fetched`는 응답을 다 받은 뒤의 시각이고 요청을 보낸 시각은 `requested`로 따로 남는다. 녹화 재생(`PITCHEEZY_OBSERVER_LIVE_REPLAY_DIR`)과 함께 쓰면 저장 행의 source가 `replay_poll`, 시계가 `unverified`로 남아 실제 경기 수신 기록과 섞이지 않는다. 운영 DB와 다른 새 경로를 쓴다. 재생은 upstream 폴링 한 번에 본문 하나씩 나가므로 기본 속도는 폴링 간격(10초, 빠른 시청자 5초)이다. 리허설을 빨리 돌리려면 `PITCHEEZY_OBSERVER_LIVE_REPLAY_POLL_S=<초>`(재생 폴더를 줄 때만 적용, 0 = 요청마다 한 개)를 준다. 실제 MLB 폴링 간격은 이 값으로 바뀌지 않는다.
- 화면 상태(`display_state`, `backend/observer_app/display_state.py`): 실시간 경기의 `/api/service/games/{pk}`와 `/api/live/{pk}/state`는 다음 공 자리마다 상태 하나와 사유 코드를 싣는다. `recommended`(원장에 먼저 기록된 추천), `unconfirmed`(확인 중: 저장 버전의 제공처 순서가 `ordered`가 아니거나, 진행 중 타석·가장 최근 투구의 본문이 투구 원장에서 수용되지 않음. SVC-WS-03 합성 경기와 같은 `admit_body` 규칙), `unsupported`(`unknown_pitcher`/`no_supported_pitch_types`/`at_bat_rejected`), `feed_delayed`(이 사건을 담지 못한 직전 성공 폴링이 사건보다 `live_config.json` `display.feed_lag_delayed_s`초 넘게 뒤였음. **잠정값**. `lag_s`는 그렇게 증명된 최소 지연, `lag_max_s`는 상한. 직전 폴링이 없거나 재생 실행이면 측정 안 함), `disconnected`(폴링 실패. 이미 기록된 추천만 유지하고 새로 계산·기록하지 않음), `ledger_unavailable`(계산했지만 기록하지 못했거나, 이전 표시 이력을 읽지 못했거나, 설정한 저장소를 열지 못해 미표시), `waiting`, `unavailable`, `pregame`, `final`. 같은 다음 공에 이미 다른 추천을 보여 준 뒤 내용이 바뀌면 `correction`(`replaced`/`withdrawn`, 이전 1순위 포함)으로 알린다(조용한 뒤집힘 없음, 원장 기준이라 재시작 뒤에도 유지). 저장소를 켜면 각 투구에 `receipt`가 붙고, 수용되지 않은 본문은 구종·속도·위치·결과를 비우고 `확인 중`으로만 내보낸다. 판정은 서빙 중인 저장 버전 시점 기준이라 지연이 긴 시청자에게 이후 결정이 새지 않는다. `currentPlay`가 `allPlays` 마지막 타석과 타석 번호·투구 목록이 다르면(뒤처진 경우 포함) 확인 중으로 두고, `allPlays`에 없거나 본문이 다른 `currentPlay` 투구는 수용 기록이 없으므로 `/api/live/{pk}/state`에서도 비워서 `확인 중`으로 내보내고, 두 경로 모두 추천을 계산·기록하지 않는다(`prior_pitch_unconfirmed`). `확인 중` 계열 문장은 두 가지뿐이다: "경기 상황을 다시 확인하고 있어요. 확인되면 추천을 보여 드려요." / "방금 던진 공을 다시 확인하고 있어요. 확인되면 추천을 보여 드려요."(`prior_pitch_unconfirmed`). 남은 한계: 볼카운트·점수·타석 결과는 선택된 스냅샷 그대로라 미확인 본문의 영향이 남을 수 있다.
- 확인 중 완화(Song 2026-10-12, 브랜치 `ws/launch-b-1017`): 한 번이라도 투구 원장에 수용된 투구는, 지금 받은 본문이 거부돼도 마지막으로 수용된 본문으로 계속 보인다(`receipt.reason: held_last_accepted`). 한 번도 수용되지 않은 투구만 `확인 중`이다. 다만 진행 중 타석이나 가장 최근 투구가 이 상태면 정책이 받은 본문을 읽으므로 추천은 보류한다. 경기 단위 같은 제공처 시각 충돌은 추천 입력(`live_armb.state_key`: 타석·카운트·주자·아웃·점수·투수/타자와 좌우·최근 투구 본문·모든 사건의 카운트와 주자 이동)이 다를 때만 막고, 날씨·관중·박스스코어만 다르면 막지 않는다. **투구가 0개인 경기에서는 같은 시각 충돌로 막지 않는다**(투수·타자가 달라도 마찬가지, Song 승인). 오래된·순서 불명 시각은 그대로 막는다. `corrected`는 이미 보인 칸(구종·판정·카운트·구속·회전·무브먼트·위치·타구값)이 바뀌거나 사라질 때만 붙고, 제공처가 `endTime`·`typeConfidence`·`breakAngle` 등만 고쳐 다시 보낸 경우와 빈 값이 채워진 경우는 붙지 않는다.
- 지나간 공의 추천(`backend/observer_app/rec_pairing.py`): 실시간 경기의 `/api/service/games/{pk}`는 지나간 공마다 `rec`에 "투구 전에 기록된 추천"을 다음 공 추천과 같은 모양(`provenance: captured_live`, `ledger`, `recorded_at`, `candidates`)으로 싣는다. 원장은 결정만 기록하므로 "그때 화면에 있었다"는 뜻이 아니다. 붙이는 조건: 그 공의 `receipt`가 `received`, 같은 투구 키의 `first` 행(`conflict` 행 제외), 그 행의 스냅샷 `store_seq`가 그 공이 처음 담긴 본문보다 앞이고 기록 시각도 그 본문 수신보다 앞, 그런 행 중 가장 마지막 상태, 같은 타석의 직전 공이 그 스냅샷에 이미 있음, 결정 입력(투수·타자·이닝·초말·아웃·볼·스트라이크·주자·점수)이 실제 공과 같음. 하나라도 어긋나면 `provenance: absent` + "이 공은 기록된 추천이 없어요."이고 `rec.pairing.rule`에 사유 코드가 남는다(화면 비표시). 원장 행에는 결정 입력 `inputs`를 함께 기록하고, 그 전 행은 저장된 스냅샷에서 읽어 확인한다. 원장을 읽지 못하는 동안에는 아무것도 붙이지 않고 `receipts.version`에 표식이 붙어, 열려 있는 화면이 복구 뒤 전체를 다시 받는다. `receipts.paired`는 붙은 공 수. 공수 교대 중 `current`는 `situation`·`batter`·`pitcher`·`pa_key`가 모두 null이고 `finished_half`(방금 끝난 이닝·초말)만 싣는다. 카드 점수는 유지한다.
- 폰 폭 확인: `node apps/observer/web/tests/service-smoke.mjs http://127.0.0.1:8766 [라이브 gamePk|-] [스크린샷 폴더|-] [다시 보기 gamePk]`.
- 실경기 측정(서비스 시간·지원 범위, 모델 평가 아님): `python3 scripts/live_rehearsal.py record --game-pk N --server http://…:8766 --output DIR [--not-before 2026-10-03T16:50:00Z]`가 폰처럼 상태를 조회해 한 줄씩 기록하고, 경기가 끝나면 `report --game-pk N --output DIR --snapshots apps/observer/live_snapshots`가 피드 도착 지연·다음 공 전 여유·추천 가능 비율·응답 시간을 낸다. `simulate --feeds …/DEMO-WS-2026/feeds --lags 5 10 15`는 끝난 경기의 투구 시각으로 조회 간격별 "5초 전 추천" 비율을 계산한다.

## 기능 스위치와 시청 지연(잠정, 기본 꺼짐)

- 스위치는 `live_config.json`의 `features`와 환경 변수로 정한다(`backend/observer_app/features.py`). 웹을 다시 빌드하지 않고 끌 수 있다. 값은 모두 잠정이고, 잘못된 환경 변수 값은 무시하고 경고를 한 번 남긴다.
  - `PITCHEEZY_OBSERVER_VIEWER_DELAY=1`: 팬 화면 시청 지연. **기본 꺼짐.** 공개 서비스는 지연 0 본문 하나를 모두에게 배포하므로, 시청자별 지연은 이 서버에 직접 붙는 경로(폰 시연·내부)에서만 켠다.
  - `PITCHEEZY_OBSERVER_VIEWER_DELAY_DEFAULT_S`(기본 30), `PITCHEEZY_OBSERVER_VIEWER_SYNC_PROMPT`(0 = 첫 화면에서 묻지 않음).
  - `PITCHEEZY_OBSERVER_PROFILE_REFERENCE`, `PITCHEEZY_OBSERVER_PROFILE_SOURCES`, `PITCHEEZY_OBSERVER_STEP5_EXPECTATION`, `PITCHEEZY_OBSERVER_STEP5_LEDGER`, `PITCHEEZY_OBSERVER_STEP5_ALLOW_REPLAYED`: 기본 꺼짐. 값만 전달하고 동작은 각 모듈이 정한다.
- 꺼져 있을 때: `?delay=`는 경기·목록에서 무시되고(지연 0), 응답에는 `features`와 `viewer`(`enabled: false`, `delay_s: 0`)만 더해진다. 그 밖의 본문은 a72f1fa와 같다(`tests/test_viewer_off_golden.py`). a72f1fa에 없던 경로와 값: `GET /api/service/features`(200, 스위치의 공개 부분), `GET /api/service/games/{pk}/tap`(어떤 요청이든 404 "이 실행에서는 중계 맞추기를 쓰지 않아요."), `/api/health`의 `service_extras`(운영자용).
- 켜져 있을 때:
  - `GET /api/service/games/{pk}?delay=<초>`: 중계가 그만큼 늦은 시청자의 시점(가장 새 사건이 그만큼 지난 본문 중 최신)으로 본문 전체를 만든다. 상한은 `max_delay_s`(120). 아직 그 시점의 본문이 없으면 점수·투구 없이 `viewer.buffering: true`와 "내 중계 시점까지 기다리는 중이에요 (약 N초)."를 준다. 더 새 본문으로 대신하지 않는다.
  - `GET /api/service/games?date=&delay=`: 진행 중 경기 카드는 점수·이닝 없이 `score_hidden: true`.
  - `GET /api/service/games/{pk}/tap?ago=<초,초…>&current=<초>&sure=0|1`: 내 TV에서 공이 미트에 들어온 순간을 누른 뒤 지난 시간(기기에서 잰 값). 답은 `status: ok|wait|unclear`와 지연 초뿐이고 경기 내용은 없다. 한 번으로 어느 공인지 가리지 못하면 다음 공에서 한 번 더 누르게 한다.
  - 수신 즉시 기록(`record_at_receipt`): 지연 시청자만 있어도 가장 새 본문의 추천을 같은 게이트를 거쳐 원장에 바로 기록한다. 그 시청자에게는 보여 주지 않는다.
- 참고 카드(A2, `PITCHEEZY_OBSERVER_PROFILE_REFERENCE=1` + `PITCHEEZY_OBSERVER_PROFILE_SOURCES`): 다음 공의 상태가 "추천 없음"이고 사유가 투수 단위(기록 없는 투수, 구종 기록 부족)일 때만 `next.reference`에 `profile_reference` 카드를 싣는다. 상태·사유는 그대로다. 화면은 상태 칸 안 한 줄("참고 · 기록 통계 자주 던진 공 싱커 29% · 포심 28% …"; 좁은 화면은 첫 구종만)을 보이고, 누르면 카드 전체(제목, 구종별 비율, 표본, "추천이나 다음 공 예측이 아닙니다")를 연다. 순위·위치·추천 색은 쓰지 않는다.
  - 공개 발행기: `publish_config.json` `profile_reference`(2026-10-18 Song 결정으로 켬, 경로는 `PITCHEEZY_OBSERVER_PROFILE_SOURCES`로 일일 갱신 포인터 `LATEST.json`을 준다; `PITCHEEZY_OBSERVER_PROFILE_REFERENCE=0`이 이긴다). 카드는 `as_of`(프로필 마감일, 경기일보다 엄격히 앞)와 `stale`(마감일이 경기일보다 2일 넘게 앞, 잠정)을 싣고, stale이면 화면 줄에 "기록 기준 MM/DD"가 붙는다. 공개 계약은 `publish_contract/v1/schema/game.schema.json` `next.reference`, 예시 `examples/game-live-unsupported-reference.json`. 운영은 `docs/pm/service/ws-operations-runbook.md`.
- 던지기 전 예상(B2, `PITCHEEZY_OBSERVER_STEP5_EXPECTATION=1` + `PITCHEEZY_OBSERVER_STEP5_LEDGER`, 재생 원장은 `PITCHEEZY_OBSERVER_STEP5_ALLOW_REPLAYED=1`일 때만): 주요 장면(삼진·홈런·병살 등 잠정 목록)으로 타석을 끝낸, 수신 확인된 지난 공에만 `pitches[i].expectation`을 싣는다. 화면은 그 공의 "자세히" 안 실제 투구 부분에 한 줄로만 보인다. 보이기 전 조건(`shadow_expectation.py`): ① 원장의 실행 기록(`shadow_run`)이 `fan_facing_output: true`라고 명시해야 한다. 지금 있는 849831 재생 원장은 `false`라서 `ALLOW_REPLAYED=1`이어도 아무것도 보이지 않는다(운영자 사유 `run_not_granted_fan_facing`). ② "던지기 전 예상"이라는 말은 실시간 원장에서, 실행기 자체의 순서 대조 행(`provider_time_after_decision_unverified`)이 있을 때만 쓴다. 그 행이 없으면 우리 서버 수신 순서만 아는 것이므로 `ALLOW_REPLAYED=1`일 때에 한해 "던지기 전 정보로 본 예상 … · 2024 모델 · 참고 · 이 공의 기록이 도착하기 전에 계산"으로만 보이고, 아니면 보이지 않는다. ③ 원장이 나중에 연결을 끊으면(격리, 늦음 판정, 경기 뒤 대조에서 분리, 타석 중 다른 선수 관측) 다음 요청부터 줄이 사라진다(요청마다 다시 묻는다). ④ 상한: 헛스윙 30% 미만, 땅볼 **20% 미만**만 보인다(`PITCHEEZY_OBSERVER_STEP5_WHIFF_CAP`, `PITCHEEZY_OBSERVER_STEP5_GROUND_BALL_CAP`; 0.30보다 큰 값은 무시). 땅볼 0.20은 잠정값이다: PM 규칙은 "30% 이상은 보이지 않는다"이고, 20~30% 구간은 보정 표본이 611개로 실험 자체 기준(1,000개)에 못 미쳐 더 엄격하게 잡았다. 소유자 결정이 남아 있다.
- 두 기능의 서버 쪽 연결은 `backend/observer_app/service_extras.py` 한 곳이고, 운영자용 상태는 `/api/health`의 `service_extras`에만 나온다. 꺼져 있으면 필드가 없다.
- `?delay=` 값이 숫자가 아니거나 음수이면 없는 것으로 본다(요청을 거절하지 않는다). 꺼져 있으면 값을 보지도 않는다.
- 꺼진 상태에서 a72f1fa와 달라진 동작 하나(의도한 변경, 장애 경로): 투구 원장을 읽지 못할 때 쓰는 "앞서 수신 확인된 공" 기억을 저장 버전별로 둔다. 예전에는 마지막으로 요청한 시청자의 것으로 덮어써서, 예전 경로(`/api/live/{pk}/state?delay=`)의 지연 시청자가 마지막에 요청한 뒤 투구 원장을 읽지 못하면 최신 시점의 팬 화면이 "확인 중"이 됐다. 지금은 그 팬이 받는 버전 기준으로 판단해 추천을 그대로 보인다(`test_after_a_legacy_delayed_request_an_unreadable_pitch_ledger_is_judged_on_the_fan_viewers_own_version`).
- 시청 지연을 켰을 때의 알려진 한계(시연·내부용인 이유):
  - **새 사건 없이 온 정정은 모든 지연에 동시에 보인다.** 지연 버퍼는 사건 시각마다 본문 하나만 두고 내용을 그 자리에서 바꾼다. 그래서 새 경기 사건 없이 기존 내용을 고치는 본문(예: 투구 26~37초 뒤에 결론이 나는 챌린지 번복. 849831의 365개 버전 중 이런 정정 19건, 번복된 판정 2건)은 지연과 상관없이 바로 반영되고, 지연 시청자는 자기 중계보다 먼저 번복된 판정을 볼 수 있다. 고치려면 사건 시각별 버전 이력이 필요하다(`test_a_correction_without_a_newer_event_waits_for_the_delayed_viewers_own_moment`가 xfail로 재현).
  - **목록의 끝난 경기**: 일정이 종료(또는 점수가 있는 중단·연기·취소)라고 한 경기는, 그 지연으로 받는 본문이 종료가 되었거나 이 서버가 **지연 요청으로** 그 상태를 처음 본 뒤 지연 시간만큼 지날 때까지 점수·상태 문구 없이 진행 중으로 보인다. 그 시계는 지연이 있는 목록 요청이 처음 볼 때 시작하므로(지연 0 요청이나 재시작 전 기록은 세지 않는다), 한참 전에 끝난 경기도 지연 시청자의 첫 목록 요청부터 지연 시간 동안은 점수 없이 보인다. 예정 시작이 12시간보다 전이거나 서버 시계보다 뒤인 경기는 가리지 않는다. 일정이 종료 → 진행 → 종료로 바뀌어도 시계는 처음 값 그대로다.
  - 켜져 있을 때 숫자가 아닌 `?delay=`는 지연 0으로 응답한다(`viewer.delay_s`에 0으로 나온다. 화면은 그런 값을 보내지 않는다).
  - **맞추기 경로는 투구 시각을 알려 줄 수 있다.** 답은 초 단위 숫자뿐이고 경기 내용은 없지만, `ago`를 바꿔 가며 물으면 최근 공이 언제 던져졌는지(또는 최근에 공이 없었는지)를 알 수 있다. 답은 정수 초로만 주고 한 번에 4개까지만 받는다.
  - 연결 상태 값(`feed.ok`, `stale_s`, `display_state.feed.last_ok_at`)은 지연 시청자에게도 현재 기준이다.
- 화면(`/watch/{pk}`): 공개 빌드(`VITE_PUBLISH_BASE` 설정)에서는 지연·맞추기 화면이 아예 없다. 그 외 빌드에서 서버가 켜져 있다고 답할 때만 "중계 맞추기"가 보인다. 맞추기 전에는 실제 투구·결과(투구 카드의 구종·속도·위치, 선택한 공의 실제 투구 줄, 자세히의 실제 투구, 타석 결과)를 "중계 맞추기 후 표시"로 접는다. 점수·볼카운트·아웃·주자·이닝별 기록은 그대로 둔다.

## 공개 발행기 (정적 파일로 내보내기, 검증 전 실험 버전)

팬의 브라우저가 이 맥에 닿지 않도록, 화면이 가져갈 내용을 **요청 없이 미리 파일로 써 두는** 프로그램이다. 계약 초안은 `docs/pm/service/public-publish-contract-draft.md`(아직 커밋 전, 주 작업 폴더에만 있음)이고, 이 절은 그 구현을 설명한다. 기본은 로컬 폴더 발행이며, 선택적으로 HTTP 업로드를 켤 수 있다. 호스팅·업로드 API는 아직 미정이다. 실제 MLB 피드·실제 저장소로 돌리는 것은 별도 승인 단계이며 아직 해 보지 않았다.

```sh
# 저장소에 들어 있는 녹화 픽스처(849845)로 한 번 발행해 보기. 네트워크·T7·모델을 쓰지 않는다.
T=$(mktemp -d); mkdir "$T/empty"
PYTHONPATH=apps/observer/backend:apps/observer/backend/tests .venv-observer/bin/python -c "from live_fixture import listed_dir; listed_dir('$T/replay')"
PYTHONPATH=apps/observer/backend \
PITCHEEZY_OBSERVER_LIVE_REPLAY_DIR="$T/replay" PITCHEEZY_OBSERVER_LIVE_REPLAY_POLL_S=0 \
PITCHEEZY_OBSERVER_LIVE_ARMB=0 PITCHEEZY_OBSERVER_LIVE_STORE="$T/store.sqlite3" \
.venv-observer/bin/python -m observer_app.publisher --out "$T/out" --once --date 2026-09-30 \
  --feed-dir "$T/empty" --watch-dir "$T/empty" --intent-dir "$T/empty" --demo-dir "$T/demo"
find "$T/out" -type f
```

**실시간 저장소(`PITCHEEZY_OBSERVER_LIVE_STORE`)와 추천 장부가 켜져 있어야만 돈다.** 저장소가 없으면 서버는 확인 중인 공을 가리지 못하고 추천을 기록 없이 보여 주는데, 공개 파일에서는 그것을 허용하지 않는다. 둘 중 하나라도 없으면 시작하지 않고, 도는 중에 장부를 끄면(`PITCHEEZY_OBSERVER_REC_LEDGER=0`) 그 주기부터 아무것도 쓰지 않는다. 영수증이 없는 실시간 본문, 장부 기록이 없는 추천이 든 본문은 파일로 나가지 않는다(`not_public`).

`--once`를 빼면 `publish_interval_s`마다 계속 발행한다. 포트를 열지 않고 다른 프로세스를 건드리지 않는다. `--date`(여러 번 가능, KST)를 주지 않으면 오늘 기준 앞뒤 날짜(`live_days_back`/`live_days_ahead`)의 경기를 매 주기 다시 만든다.

### 파일

- `v1/manifest.json` — 계약 버전, 코드 커밋(`commit`, 커밋 안 된 변경이 있으면 `commit_dirty: true`), 발행 시각, 발행 간격, 고정 지연, 매 주기 다시 만드는 날짜.
- `v1/health.json` — 마지막 발행 시각과 순번. 한 주기의 맨 마지막에 쓴다. 이 파일이 움직이지 않으면 발행이 멈춘 것이다. 이번 주기에 만들거나 쓰지 못한 경기는 `unavailable_games`(`game_pk`, `status`, 정해진 `code`: `refused`·`build_failed`·`write_failed`·`setup_failed`·`not_public`·`withdraw_failed`(지워야 할 파일을 지우지 못함)·`archive_unread`(보관 폴더에 들어온 최종 피드를 읽지 못함, 읽힐 때까지 계속 표시))에, 날짜 목록은 `unavailable_dates`(날짜만)에, 피드가 응답하지 않아 마지막 상태로 만든 경기는 `feed_failing_games`에 남는다. 경로나 오류 문장은 싣지 않는다. 한 경기의 실패는 다른 경기와 health 발행을 막지 않는다.
- `v1/dates.json` — `GET /api/service/dates`와 같은 본문.
- `v1/games/<YYYY-MM-DD>.json` — `GET /api/service/games?date=`와 같은 본문.
- `v1/game/<game_pk>.json` — `GET /api/service/games/{pk}`와 같은 본문(항상 전체, `after` 없음).
- `v1/setup/<game_pk>.json` — `GET /api/watch/{pk}/setup`의 본문에서 `source`만 `commit`과 `sha256` 두 개로 줄인 것(`commit`은 해시 모양이 아니면 null, 파일 경로·저장소·브랜치·영상 이름은 내보내지 않는다. 서버 경로의 본문은 그대로다). 끝난 경기이고 `setup_estimates.py` 검사(sha256 고정값, 줄마다 `validate_intent_estimate`)를 통과한 경우에만 있다. 검사에서 떨어지면 지운다.

통합본은 RC의 `features`(목록·경기)와 `viewer`(실시간 경기) 메타데이터도 싣는다. v1에서 이 두 필드는 선택 사항이며 기존 파일도 유효하다. 기본 스위치는 모두 꺼져 있고 공개 고정 지연은 여전히 0초다. 꺼진 시청자 지연 설정의 `default_s: 30`은 공개 적용값이 아니다.

셋업 파일을 뺀 본문은 서버 경로가 쓰는 **같은 함수**(`service_api.ServiceBodies`)가 만든다. 그래서 표시 판정, 확인 중인 공 비우기, 추천 장부 먼저 쓰기가 그대로 적용된다. 발행기가 더하는 것은 모든 파일의 맨 위 키 하나뿐이다.

```json
"publish": {"contract": "v1", "sequence": 42, "published_at": "2026-10-05T10:20:12Z", "final": false}
```

- `sequence`는 발행기 하나에 하나뿐인 번호이고 주기마다 1씩 오른다. 파일은 자신을 마지막으로 쓴 주기의 번호를 갖는다. 재시작해도 이어지고 뒤로 가지 않는다: 시작할 때 상태 파일과 **이미 나가 있는 모든 파일**의 순번 중 가장 큰 값에서 이어 가고, 주기마다 번호를 디스크에 확실히 기록(fsync)한 뒤에 파일을 쓴다.
- `final`은 경기 파일과 셋업 파일에만 있다. 경기가 끝났고 확인 중인 공이 하나도 없으면 `final: true`로 한 번 쓰고 더 쓰지 않는다. 이후에는 `settled_recheck_s`마다 다시 만들어 보고, 내용이 달라졌을 때만(정정) 새 번호로 다시 쓴다. 경기 뒤에 도착한 셋업 파일도 이때 발견한다.
- 파일은 출력 폴더 **바깥** 옆 폴더(`<상위>/.<출력 폴더 이름>.tmp/`, 같은 파일시스템)에 다 쓴 뒤 이름을 바꿔 통째로 교체한다. 반쯤 쓰인 파일은 보이지 않고, 출력 폴더에는 발행 파일만 있다. 그래도 올리는 쪽(업로더)은 점(.)으로 시작하는 파일과 위 여섯 가지 이름이 아닌 파일을 올리지 말아야 한다.
- 끝난 경기를 다시 만들 때 시계에 따라 움직이는 값(맨 위 `feed` 전체, `display_state.feed`의 `ok`·`delayed`·`lag_*`·`last_ok_at`·`failing_*`·`received_at`)은 변화로 치지 않는다. 경기 내용이 같으면 파일도 순번도 그대로다.
- 실시간 피드에서 받은 경기는 종료 표시 뒤 `final_recheck_for_s`가 지나면 `final_slow_recheck_s`마다 한 번만 다시 조회한다(늦은 공식 정정을 찾기 위해). 셋업 파일은 계속 찾고, 보관 폴더에 최종 피드가 들어오면 조회 없이 바로 그것으로 바꿔 쓴다.
- 발행기는 출력 폴더 하나에 하나만 돈다. 출력 폴더 옆 `.<출력 폴더 이름>.lock`을 잡고(상태 파일 위치나 경로 표기가 달라도 같은 폴더면 같은 잠금), 이미 잡혀 있으면 바로 끝난다. 잡고 있던 프로세스가 죽으면 잠금은 풀린다. 출력 폴더의 실제 경로는 시작할 때 한 번만 정하고, 잠금·임시 파일·읽기·쓰기·지우기가 모두 그 경로를 쓴다.
- 출력 폴더 안에 심볼릭 링크가 하나라도 있으면 시작하지 않는다. 도는 중에 생기면 그 경로로는 읽지도 쓰지도 지우지도 않는다(출력 폴더 밖 파일을 건드리지 않기 위해서다).
- 시작할 때 순번을 찾는 검사는 이 계약의 파일 이름에 맞는 파일만, 파일 끝부분만 읽는다. 다른 파일이나 깨진 파일은 무시한다. 상태 파일이 깨져 있으면 경고만 남기고 이어 간다.
- 상태 파일은 출력 폴더 옆 `<출력 폴더>.state.json`에 둔다(올릴 대상이 아니다).

### 발행하지 않는 것

- 시연 세션(`?demo=`), 운영자 API(`/api/service/operator/**`), 시연 시계·다시 보기. 시연 중인 경기는 서버 경로와 똑같이 경기 파일을 쓰지 않고 목록 카드는 가린다. 이미 발행된 경기에 시연이 시작되면 다음 주기에 그 경기 파일과 셋업 파일을 지우고(상태 파일이 그 파일을 모르더라도 지운다), 시연이 끝나면 다시 발행한다. 파일을 쓰기 직전과 직후에도 시연 여부를 다시 확인하고, 주기 도중에 시연이 시작되면 같은 주기 안에 파일을 지우고 목록을 다시 만든다. 시연 중에 목록을 다시 만들지 못하면 옛 점수를 남기지 않고 그 목록 파일을 지운다. 마지막 시연 확인은 `health.json`을 쓰기 바로 전이다. **받아들인 틈(2026-10-11 PM 결정)**: 그 확인 뒤에 시작된 시연은 다음 주기(최대 발행 간격 한 번)에 반영된다. 시연 시계는 다른 프로세스이고 둘 사이에 공유 잠금은 두지 않는다. 시연 가림은 이미 끝난 경기를 시연 관객에게 가리는 용도이며, 공개 전 정보를 지키는 장치가 아니다.
- 서버가 지금 보여 주지 않는 것: 확인 중인 공의 구종·속도·위치·결과, 고정 지연보다 새로운 공, 표시 상태가 추천을 보여 주지 않을 때의 후보, 장부에 기록하지 못한 추천.
- 서버 경로가 거절하는 본문(피드 첫 조회 실패 503 등). 이때는 이전 파일을 그대로 둔다.

### 설정

값은 `apps/observer/publish_config.json`에 있고 코드에 박아 두지 않았다. `publish_interval_s`(기본 10), `settled_recheck_s`(기본 600), `final_recheck_for_s`·`final_slow_recheck_s`(기본 21600), `live_days_back`/`live_days_ahead`(기본 1)는 **재지 않은 잠정값**이며 리허설에서 정한다. `fixed_delay_s`는 모든 시청자에게 같은 지연 하나이고 기본 0초다. `contract_version`(`v1`)이 곧 파일 경로 앞부분이다. 환경 변수 `PITCHEEZY_PUBLISH_OUT_DIR`, `PITCHEEZY_PUBLISH_INTERVAL_S`, `PITCHEEZY_PUBLISH_DELAY_S`, `PITCHEEZY_PUBLISH_STATE`, `PITCHEEZY_PUBLISH_COMMIT`와 명령줄 `--out`/`--interval`/`--delay`가 덮어쓴다. 피드를 실제로 조회하는 간격은 여전히 `live_config.json`(`poll_interval_s`, `fast_poll_s`)이 정한다.

### 선택적 HTTP 업로드 (잠정 API)

업로드 기본값은 OFF다. `upload.base_url`은 전용 HTTPS `/_publish` 경로를 정확히 한 번 포함한다. HTTP는 로컬 합성 시험의 loopback에서만 허용한다. 팬 공개 URL과 인증 업로드 URL을 혼동하지 않는다. 현재 검증 범위는 로컬 합성 계약이며 실제 호스트 연결/배포 승인이 아니다.

토큰은 `token_env`에 환경변수 이름만, 또는 `token_env: null`과 저장소 밖 절대 `token_file`로 지정한다. 두 방식을 동시에 쓰지 않는다. 파트너 형식인 URL-safe 문자 43–128자만 허용한다. 토큰/URL/서버 응답 원문을 health·오류·상태·로그에 남기지 않는다.

전송은 인증 HEAD로 missing/현재본/tombstone과 strong ETag를 읽은 뒤 PUT에 원본 바이트 SHA256을 붙인다. 최초 PUT은 `If-None-Match: *`, 교체·삭제·복원은 관측한 ETag의 `If-Match`를 사용한다. DELETE는 조건부 tombstone이며 삭제와 동일 바이트 복원도 서로 다른 ETag를 갖는다. 200 + 올바른 ETag/JSON 영수증만 성공이다. SHA는 원본 JSON 내용의 무결성이며 모델/스키마/경기 정합성 승인이 아니다. 파일은 UTF-8 JSON object, 최대 5 MiB이며 remote 경로는 partner v1 whitelist로 제한한다. local publisher의 다른 버전 경로 지원과 구분한다.

- `timeout_s`는 소켓 제한(최대 60초)이다. 기본 `background=true`는 로컬 cycle과 네트워크를 분리하며, 소유한 spawn 자식 하나가 각 HTTP 작업을 수행한다. `operation_deadline_s`(기본 15, 1–60초)가 DNS를 포함한 자식 수명을 제한한다. `shutdown_grace_s`(기본 5, 0–30초) 뒤 미완료 전송을 중단·회수하고 pending을 보존한다. 원격 서버가 이미 받은 요청의 후행 완료까지 취소하는 것은 아니다.
- 예전 `retries`/`backoff_s`는 구 설정을 읽기 위해 검증만 한다. mutation을 무조건 재시도하지 않는다. 429는 영속 `not_before`에 Retry-After를 반영하고 이후 cycle에서만 다시 시도한다. 403 등의 확정 거부는 같은 intent를 반복하지 않으며 새 local intent는 다시 준비할 수 있다.
- `--check-remote --config <config>`는 인증 HEAD `v1/manifest.json`만 수행한다. missing 404도 정상 진단이며 쓰기/삭제 성공을 주장하지 않는다. 임의 probe 객체나 경기 파일을 만들지 않는다.
- `--dry-run-upload`는 토큰/네트워크 없이 상대 경로·크기를 기록하지만 **로컬 본문 생성은 실행**한다. 등록된 합성 입력에서만 사용한다.
- 모든 원격 JSON은 `no-store`, ETag 응답은 파트너의 `no-transform`을 요구한다. 이전 live 5초/final 300초 캐시 기대는 폐기했다.

`upload_pending`은 최신 path→action, `upload_intents`는 action/SHA/revision, `upload_attempts`는 실제 전송 직전에 영속한 동일 identity와 expected ETag/create 조건이다. 로컬 파일과 상태를 함께 보존한다. ACK가 늦게 와도 더 최신 intent를 지우지 않는다. 실패한 로컬 삭제는 remote DELETE를 보내지 않고 P3가 최신 PUT으로 바꿀 수 있다. 업로드 OFF 동안도 기존 outbox의 새 쓰기/삭제를 기록한다. 목적지는 URL 대신 hash로 상태에 묶이며 변경 시 시작을 거절한다.

**차단과 재개:** timeout/비정상 200/예상 밖 2xx/5xx/412 또는 재시작 때 남은 inflight는 `upload_attempts[path].phase=blocked`와 고정 reason으로 보존한다. 이전 무조건 전송 형식의 pending은 `legacy_unknown`, 손상/유실 state는 `upload_recovery_blocked`다. 새 local body, 재시작, 같은 SHA의 HEAD만으로 이를 성공/재시도 처리하지 않는다. 로컬 발행은 계속하고 health.upload.pending과 unavailable_games는 부분 실패를 표시한다. `game_checks.file_sequence`는 로컬 파일 기준이므로 원격 적용 증거로 읽지 않는다. health 자체가 실패하면 공개 health도 정지한다. 객체별 CAS는 다중 파일 원자 commit이 아니다.

차단을 없애려고 `upload_pending`/attempt/destination을 지우거나 최신 ETag로 오래된 본문을 보내지 말 것. 우선 업로드를 OFF로 두고 exact state·최신 local SHA/revision·고정 endpoint의 인증 HEAD·동일 출력 lock 소유 및 이전 worker 종료 증거를 모아 별도 대사 검토한다. 이를 자동 승인하는 reset 명령은 없다. 새 목적지는 별도의 출력/상태와 명시적 초기 등록으로 준비한다. 현재 gateway HEAD는 작성자/attempt ID를 제공하지 않는다. 유효한 mutation ACK와 일치하는 최신 intent만 완료 처리하며, ACK를 잃었거나 원격 충돌의 출처가 불명확하면 blocked를 유지한다.


로컬 출력과 원격 prefix에는 발행기 하나만 배정한다. 원격 장애 중 이미 공개된 파일을 즉시 철회할 수 없으므로 읽는 쪽은 health의 부분 실패와 발행 시각 정체를 확인한다. health.upload의 pending=0은 그 시점의 로컬 outbox가 비었다는 뜻이며 원격 공개 GET 검증이나 다중 파일 transaction의 증거가 아니다. 메일박스는 경로 metadata 128개를 넣기 전에 합치고, 가득 차면 durable pending을 남겨 다음 cycle에서 다시 예약한다. blocked 항목은 슬롯을 쓰지 않는다. 전송 중 본문은 최대 5 MiB의 불변 hardlink snapshot 하나이며 기존 전체 outbox/state는 알려진 경로 수에 비례한다. 이는 전체 프로세스 RSS 상한이 아니다. 자식과 coordinator가 모두 종료되기 전에는 출력 flock을 놓지 않는다. 종료를 입증하지 못하면 오류와 함께 lock을 유지한다. `background=false`는 합성 프로토콜 시험용 동기 경로이며 운영의 비차단 보장을 제공하지 않는다.

### 다른 팀에 넘기는 예제와 스키마

`apps/observer/publish_contract/v1/`에 파일 종류별 예제(`examples/`)와 JSON Schema(`schema/`)가 있다. 합성 경기를 실제 발행기로 돌려 만든 것이고 스키마는 그 본문에서 뽑았다. 그래서 스키마는 합성 경기에 나온 값만 안다. 실제 경기에서 다른 타입(예: null)이 나올 수 있으니 리허설 뒤 다시 뽑아야 한다. 읽는 쪽은 모르는 키를 무시해야 한다.

```sh
PYTHONPATH=apps/observer/backend .venv-observer/bin/python apps/observer/scripts/export_publish_contract.py   # 다시 만들기
```

`test_publisher.py`가 커밋된 폴더와 지금 코드가 쓰는 결과가 같은지 확인한다. 업로드는 `publisher.HttpSink`와 `MirrorSink`가 맡으며 본문을 만드는 함수는 그대로다.

### 월드시리즈 운영 실행기 (`ops/ws_operate.py`, 검증 전)

맥미니에서 실제 월드시리즈 경기 날짜 하나(KST)에 발행기를 붙여 돌리는 실행기다. stdlib만 쓴다. 리허설 도구(F-rehearsal `rehearse.py`)의 분리 감독·caffeinate·최소 자식 환경·childwatch·보호 프로세스 기록·자원 가드를 가져왔고, 계획된 개입 대신 재시작 정책과 배포 가드를 넣었다. 설정은 `apps/observer/ops/ws_ops_config.json`(값은 모두 잠정), 상태 파일 계약은 `docs/pm/service/ws-ops-status-contract.md`.

```sh
OPS=apps/observer/ops/ws_operate.py
python3 $OPS plan                                   # 오늘부터 14일 안의 WS 경기와 날짜별 시간표 (일정 요청 1회, 메타데이터만)
python3 $OPS start --date 2026-10-24 --dry-run      # 사전 점검·명령·환경만 출력. 쓰기·실행·신호 없음
PITCHEEZY_OBSERVER_PROFILE_SOURCES=$HOME/Projects/pitcheezy-pm-jobs/profile-refresh-daily/profiles/LATEST.json \
  python3 $OPS start --date 2026-10-24              # 분리된 감독만 뜨고 첫 투구 30분 전까지 잔다
python3 $OPS status                                 # <ops_root>/status.json + 기록된 프로세스가 아직 맞는지
python3 $OPS stop --date 2026-10-24                 # STOP 파일 -> 감독이 자기 발행기만 SIGINT로 정상 종료
python3 $OPS stop --date 2026-10-24 --publisher-only  # 감독이 죽었을 때: pid+시작 시각+명령줄이 맞을 때만 SIGINT
```

- 시간: 시작 = 그날 첫 경기 예정 시각 − `lead_min`(30분). 끝 = 운영 경기가 모두 발행 파일에서 final이 된 뒤 `settle_s`(15분), 또는 마지막 경기 예정 + `hard_end_min`(9시간).
- 시작 직전 다시 확인: 일정(연기·취소·이미 종료, 같은 날 다른 미종료 경기), 메모리·디스크·T7, 다른 발행기·리허설 감독, 코드 HEAD(추적 파일 깨끗, `--expect-commit`).
- 일시적인 문제(일정 요청 실패, 메모리·디스크 하한, T7, 다른 발행기, 다른 미종료 경기)는 첫 투구까지 60초마다 다시 확인하고, 그래도 안 되면 `refused` + 긴급 문제(`prestart_refused`)로 끝낸다. 일정 요청이 실패하면 start 때 기록한 경기 목록(RUN.json)을 쓴다. 코드·배포 가드·업로드 설정 문제는 바로 거부한다.
- 재시작: 예상 밖 종료 또는 health가 300초 멈추면(세그먼트 시작 뒤 최소 300초부터) `backoff_s` 30/60/120초 뒤 최대 3번. 발행기는 언제나 하나(감독 lock + 프로세스 표 + 발행기 출력 lock).
- 배포 가드: 그날 처음 띄운 커밋과 HEAD가 다르면 경기가 final이 아닌 동안 재시작·새 start를 거부한다(B가 결정 ID를 바꾸므로 경기 사이에만 교체).
- 업로드: 기본 꺼짐. `--upload` + 설정 `upload.allowed: true` + 환경의 토큰(`PITCHEEZY_PUBLISH_TOKEN` 또는 토큰 파일 경로 `PITCHEEZY_WS_OPS_TOKEN_FILE`)·주소(`PITCHEEZY_WS_OPS_UPLOAD_BASE_URL`)가 모두 있어야 켠다. 값은 출력·로그에 쓰지 않고 있음/없음만 남긴다. 주소는 실행 폴더의 `publish_config.runtime.json`(권한 0600, 저장소 밖)에만 들어간다.
- 공통 지연: `fixed_delay_s`(지금 0, Song의 0/15초 결정 대기). 실행마다 런타임 설정에 넣고 저장소 파일은 고치지 않는다.
- `runs/` 폴더(실행 기록, 런타임 설정에 업로드 주소 포함)는 이슈·저장소·채팅에 복사하지 않는다. 공유가 필요하면 `status.json`의 필요한 줄만 옮긴다.
- 자동 시작 예시(설치하지 않음): `apps/observer/ops/com.pitcheezy.ws-operate.plist.example`.
- 시험: `PYTHONPATH=apps/observer/backend .venv-observer/bin/python -m pytest apps/observer/backend/tests/test_ws_operate.py -q` (가짜 발행기, MLB·네트워크 없음).
- 경고 감시기와 함께: 감시기 `apps/observer/ops/ws_alert.py --loop`를 `--out` 없이 띄우면 `status.json`의 `out_dir`·`state_path`를 따라간다(런북 `docs/pm/service/ws-operations-runbook.md` 7.6). 통합 시험: `python -m pytest apps/observer/ops/tests/ -q`(실제 감독 + 가짜 발행기 + 감시기 `--once`, 가짜 알림만).

## 현재 구성

| 모듈 | 구현 |
|---|---|
| 경기 상태·재생 | 실제 기록 목록, 한 공 공개, 상태 버전, 중복 진행 거부 |
| 타자·구종 | 경기 전날까지의 연속형 타자 성향, 최근90일 지원 구종 |
| 추천 | 동결된5개 신경망 + 빈도 모델, 구종×9구역의 타석 종료 수비 승률 계산 |
| 비교 화면 | 당시 저장된 추천과 실제 도달 좌표, 포수 시점, 투구 타임라인 |
| 선택적 분석 | 타석 마지막1구 작업 생성, 중복 방지, 임대·재시도, 별도 워커 |
| 의도·해석 | 영상 없음 표시, 수동 목표 메모, 추천·목표·실제 위치의 공간 비교 |
| 간단한 컨텍스트 | 주자·아웃·카운트·점수·홈원정, 투구 전 누적 투구 수 |
| 저장 | SQLite WAL, 불변 사전 추천, 관전 세션·수동 메모·작업 상태 |

타자 MLB ID는 기록 연결과 화면 표시에 사용한다. 신경망의 개별 타자 ID 특성은 사용하지 않는다. 기존 실험에서 채택한 이력 없는 모델을 그대로 재사용하므로, 누적 투구 수 표시는 피로도를 학습한 추천이라는 뜻이 아니다.

## 위치 추천을 읽는 방법

실제 도달 위치와 구속·회전·무브먼트의 상관관계를 보존한 TRAIN 투구400개를 사용한다. 각 목표 구역 중심 주변의 투구에 Gaussian 가중치(`sigma=0.45ft`)를 주고 결과 확률을 계산한다. 모든 카운트에서 유효 표본 수20 이상, 평균 커널 질량0.01 이상인 후보만 타석 계획에 넣는다. 표본들이 모두 목표에서 멀리 떨어진 경우를 유효 표본 수만으로 지원 구역이라 판단하지 않는다. 존 높이는 현재 공의 실제 위치가 아니라 해당 타자의 **이전 날짜** 기록 중앙값이다.

목표를 바꾸면 투수가 그곳에 던진다는 인과 모형이나 실제 제구 오차 모형은 아직 아니다. 화면의 수비 승률 및 기준 대비 차이는 근사 모델 내부 계산이다. 기준 정책 역시 최근 구종 빈도와 관측 위치의 구역별 질량을 결합한 비교용 근사다. 실제 성능 향상을 입증한 지표로 해석하지 않는다. 현재 결과나 미래 투구를 추천 입력으로 전달하지 않는다.

## 저장·환경

- 코드: `apps/observer/backend`, `apps/observer/web`, `apps/observer/scripts`.
- 원본 연구 환경 `.venv`를 보존하고, 웹 환경은 `.venv-observer`에 별도로 설치했다. 새 환경의 `.pth`가 원본 과학 패키지를 읽기 전용으로 참조한다. 따라서 원본 `.venv`도 필요하다.
- 동결 모델: `/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/minimal-pitch-service-v1`.
- 새 데이터·DB·추천 캐시·로그·검증 결과: `/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/observer-mvp-v1`.
- 모델 번들 해시와 캡처된 Python 소스를 검증한다. 완전한 연구 코드 트리에서 필요한 유틸리티를 가져오며 그 소스 해시도 추천 캐시 키에 포함한다.
- 앱 설정은 `config.json`; API와 정보 공개 규약은 `CONTRACT.md`.

프런트엔드를 수정한 경우:

```sh
cd apps/observer/web
npm ci
npm run build
```

### 공개용(정적 파일 읽기) 화면 빌드

공개 화면은 백엔드 API를 부르지 않고, 발행기가 올린 파일만 읽는다([공개 발행 계약 초안](../../docs/pm/service/public-publish-contract-draft.md), 파일 형태는 `apps/observer/publish_contract/v1`).

```sh
cd apps/observer/web
npm ci
npm run build:public        # 결과: dist-public/ (평소 빌드 dist/는 그대로)
node tests/static-publish-smoke.mjs   # npm run build 도 먼저 해 둘 것 (API 방식 비교 검사 포함)
PZ_STATIC_CORE_ONLY=1 node tests/static-publish-smoke.mjs  # 브라우저 없는 어댑터·시간 규칙 검사만
```

- 방식 선택은 빌드할 때 정한다. `VITE_PUBLISH_BASE`가 비어 있으면 지금까지의 API 방식 그대로다. 공개 번들에는 API 호출 코드가 아예 들어가지 않는다(검사가 번들을 직접 뒤져 확인).
- 설정 이름(기본값은 `web/.env.public`, 셸 환경변수가 우선):
  - `VITE_PUBLISH_BASE`: `v1/` 폴더가 있는 주소나 경로. 기본 `.` = index.html 옆.
  - `VITE_PUBLISH_POLL_MS`(10000) · `VITE_PUBLISH_PREGAME_POLL_MS`(10000) · `VITE_PUBLISH_LIST_POLL_MS`(10000) · `VITE_PUBLISH_RETRY_MS`(4000): 경기 중 / 경기 전 / 목록 / 경기 화면의 최초 읽기 실패 뒤 다시 읽는 간격. 목록 읽기 실패는 P, dates·manifest 실패는 max(60초, P) 뒤에 재시도한다.
  - `VITE_PUBLISH_TIMEOUT_MS`(15000): 요청 하나의 제한 시간(본문 끝까지). health → 경기 파일은 직렬이므로 둘 다 시간 제한에 걸리면 한 묶음은 최대 약 30초와 처리 시간이 걸린다. `VITE_PUBLISH_LOST_MS`(30000): 이만큼 못 읽으면 "연결 끊김". `VITE_PUBLISH_TICK_MS`(5000): 시간 표시를 다시 따지는 간격.
  - `VITE_PUBLISH_STALE_S`(30, **임시값·미측정**): 새 발행이 이만큼 없으면 "갱신 지연". 리허설에서 잰 발행 간격으로 다시 정한다.
  - `LOST_MS`와 `STALE_S`는 그 화면이 읽는 간격의 3배보다 짧게는 적용되지 않는다(읽는 간격을 늘려도 매번 경고가 뜨지 않게).
- 읽는 파일: `v1/dates.json`, `v1/manifest.json`(`live_dates`로 오늘 날짜), `v1/games/<날짜>.json`, `v1/game/<game_pk>.json`, `v1/health.json`. 이 다섯 꼴 말고는 요청하지 않는다(주소의 `?date=`는 실제 달력 날짜, `?game=`은 숫자일 때만 쓴다). 경기 파일은 항상 통째로 읽고, 같은 파일에서 순번(`publish.sequence`)이 뒤로 간 응답은 버린다.
- health를 먼저 읽는다. 해당 경기의 유효한 `game_checks[game_pk]`가 있으면 최초 로드 또는 `file_sequence` 증가 때만 경기 파일을 받는다. 전파 지연·실패는 다음 주기에 같은 목표 순번을 다시 확인하며, 목표보다 높은 본문은 허용하고 역행은 버린다. `file_sequence: null`은 철회로 처리한다. 철회 뒤에는 health 실패·키 누락·잘못된 확인값으로 구형 파일 읽기로 돌아가지 않는다. 유효한 비null 확인값과 그 순번 이상의 파일을 받아야 다시 보인다. 종료 파일도 같은 주기로 정정·철회를 확인한다.
- 새 health의 경기 확인값은 `checked_at`(유효한 시각), `sequence_checked`(0 이상 정수), `file_sequence`(확인 순번 이하의 0 이상 정수 또는 null), `feed_ok`(boolean)다. 아직 철회되지 않은 경기의 확인값에 누락·오류가 있으면 매 주기 경기 파일을 읽는다. 해당 경기 키가 있지만 값이 잘못되면 정상 확인으로 표시하지 않는다. `checked_at`은 health의 `publish.published_at` 이하이어야 정상 갱신으로 판단한다. 발행기에서 game_checks를 추가할 때는 모든 확인을 마친 뒤의 시각으로 health를 찍어야 하며, 아직 이 필드를 쓰지 않는 발행기는 구형 파일 규칙을 따른다.
- 새 확인값이 있는 경기의 "갱신 지연": `T=max(VITE_PUBLISH_STALE_S, 3P)`초이며, 서버 추정시각에서 health 발행 시각 또는 `checked_at`까지 T를 넘으면 경고한다. 경기 파일 나이는 사용하지 않는다. `feed_ok:false` 또는 `feed_failing_games`에 이 경기가 있으면 즉시 연결 끊김을 표시한다. 새 본문을 적용하지 못한 동안 추천·실제 투구는 가리고 "확인 중", T 초과 시 "갱신 지연"이다. 경기 전에는 예정 시작 이후부터 시간 경고를 적용하며, 종료 상태이면서 `publish.final:true`인 경기는 시간 경고에서 제외한다.
- 구형 파일(`game_checks`에 해당 경기 없음)의 "갱신 지연" 판정:
  - 진행 중인 경기: health가 오래됐거나, health는 새것인데 이 경기 파일만 뒤처졌을 때.
  - 그 밖의 경기(경기 전·지연·연기·더블헤더 대기): health가 오래됐고 예정 시작 시각도 지났을 때만. health가 새것이면 띄우지 않는다.
  - "지금"은 응답의 `Date`(+`Age`) 헤더로 잡고, 그 뒤 흐른 시간은 화면이 직접 센다. 헤더를 한 번도 못 읽었을 때만 최근에 본 발행 시각으로 대신한다(이때 미래 시각이 찍힌 파일이 하나 섞여도 잠시 뒤 회복). 기기 시계는 쓰지 않는다.
  - 탭으로 돌아오거나 기기가 깨어나면 먼저 다시 읽고, 그 응답이 올 때까지는 "확인 중"으로 둔다.
- 없는 경기 파일: "지금 볼 수 없어요"로 두고 간격을 두 배씩 늘려 다시 묻는다(`VITE_PUBLISH_MISSING_RETRY_MS` 10000 → `VITE_PUBLISH_MISSING_MAX_MS` 60000, **임시값**). 발행기가 따라가는 날짜의 목록 어디에도 그 경기가 없으면 묻기를 멈춘다.
- 지난 공을 골라 고정해 둔 화면도 계속 확인한다(종료 후 정정 포함). manifest가 없을 때는 `VITE_PUBLISH_PINNED_POLL_MS`(10000, **임시값**)를 쓴다. 서버가 그 공을 "확인 중"으로 되돌리면 고정 화면에서도 구종·위치를 지운다.
- 정적 모드의 경기 중·경기 전·고정 투구·목록·health 주기는 manifest의 `publish_interval_s`(숫자 초)를 우선한다. 유한한 양수이면서 브라우저 타이머 범위(1ms–2147483647ms)일 때만 사용하며, 그 밖에는 위 설정으로 대체한다. health는 해당 화면 주기를 따른다. 목록의 P가 바뀌어도 같은 날짜의 카드는 유지하며 진행 중인 읽기/예약은 마친 뒤 다음 타이머부터 새 P를 쓴다.
- manifest(경기·목록 화면)와 dates(목록 화면)는 `max(60초, P)`마다 다시 읽는다. `VITE_PUBLISH_DATES_POLL_MS`는 제거했다. 탭 복귀·온라인 복귀 시에는 즉시 확인한다.
- 올리는 법: `dist-public/` 내용을 `v1/`과 같은 폴더에 둔다. 주소는 `…/?game=<game_pk>`, `…/?date=<날짜>` 꼴이라 주소 다시쓰기 규칙이 필요 없다.
- 호스팅이 지켜 줘야 할 것:
  - `v1/*.json`은 gzip, 내용별 `ETag`, `Cache-Control: no-cache`를 제공한다. `If-None-Match`가 일치하면 본문 없는 304로 답한다. 화면은 URL별 마지막 ETag와 검증된 본문을 보관해 304에서 재파싱하지 않는다. ETag가 없으면 매번 200으로 동작한다. 종료 경기에도 정정이 오므로 무기한 immutable 캐시는 쓰지 않는다.
  - 캐시가 응답을 대신 줄 때는 `Age` 헤더를 붙인다. `Date` 헤더는 지우지 않는다.
  - 화면과 `v1/`을 다른 출처에 두면 `Access-Control-Allow-Origin`, `Access-Control-Allow-Headers: If-None-Match` 및 OPTIONS 처리, `Access-Control-Expose-Headers: ETag, Date, Age`가 필요하다. 동일 출처를 권장한다. ETag가 노출되지 않으면 조건부 헤더 없이 200을 읽으며, Date/Age도 없으면 발행 시각과 경과 시간으로 판단한다(화면을 열기 전에 이미 멈춘 발행은 기준 시간만큼 늦게 알아챈다).
  - 없는 파일은 404로 답한다(index.html을 200으로 대신 주지 않는다). 파일은 통째로 바꿔치기한다.
- 공개 번들에 없는 것: `/api/**` 호출과 옛 경로 대체, 시연 세션(`?demo=`), 운영자 화면, `?after=` 이어받기. 접속 기록·피드백·영상도 넣지 않았다.
- 호스트용 빈 자리 하나: 화면 뒤에 `<div id="pitcheezy-host-slot" data-screen="list|game" data-game-pk="…">`가 비어 있고, 생길 때 `window`에 `pitcheezy:host-slot` 이벤트(`detail.element/screen/gamePk`)가 난다. 피드백·접속 기록은 호스트 페이지가 여기에 붙인다(`web/src/service/HostSlot.tsx`).

백엔드 테스트:

```sh
PYTHONPATH=apps/observer/backend .venv-observer/bin/python -m pytest apps/observer/backend/tests -q
```

데모 자료 구성 또는 기존 자료 해시 확인:

```sh
.venv-observer/bin/python apps/observer/scripts/build_demo.py
```

기존 데모 자료는 자동 덮어쓰지 않는다. 생성 시2023–2025 원본·처리 자료 해시를 검증하고, 지원되는 투수의 날짜순 타석을 선택한다. 원본 자료의 지원 타석 필터를 사용하는 회고적 표본이므로 일반 경기 전체의 성능 평가 세트로 간주하지 않는다.

실행 중인 서버의 전체 데모 기록 회귀 검사(사용자 세션과 별개의 검사 세션 생성):

```sh
.venv-observer/bin/python apps/observer/scripts/validate_replay.py
```

독립된 테스트 브라우저에서 사용자 흐름·화면 확인:

```sh
cd apps/observer/web
PLAYWRIGHT_BROWSERS_PATH='/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/observer-mvp-v1/browser-cache' node tests/observer-smoke.mjs
```

검증 수치와 산출물은 `RESULTS.md`, 후속 결정은 `CHECKPOINTS.md`를 참고한다.

## 기존 v1의 연결 지점과 개선판 상태

v1은 영상 없는 어댑터와 수동 입력을 사용한다. 개선판에는 실제 짧은 영상, 라벨 저장·불러오기, 투구 전 구간 추적과 이미지 평면 보정 검사가 있다. 실제 포수 의도·물리적 좌표·제구 정확도는 검증하지 않았다. 이름, 대결 회차·구속 관측, 위치 독립 평가는 개선판에 반영했다. 운영 PostgreSQL·라이브 공급·외부 배포·사용자의 직접 관전 평가는 후속 과제로 남는다.
