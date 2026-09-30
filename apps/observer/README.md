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

`GET /api/live/games?date=YYYY-MM-DD`, `GET /api/live/{gamePk}/state`가 MLB Stats API 공개 피드의 현재 경기 상태를 `delay_s`(기본30초) 지연해 투구 전 입력으로 바꾼다. 설정은 `live_config.json`. 선수 특성은 동결 번들 핀만 쓴다: 번들에 없는 투수는 `unsupported_pitcher`, 프로필 없는 타자는 리그 기본값, 존 높이는 ≤2025 리그 중앙값. 원본 피드는 `apps/observer/live_snapshots/`(gitignore)에 gzip으로 기록되고, `PITCHEEZY_OBSERVER_LIVE_REPLAY_DIR` 또는 `python -m observer_app.live_feed --replay-dir DIR --game PK`로 오프라인 재생한다.

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
