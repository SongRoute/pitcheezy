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
