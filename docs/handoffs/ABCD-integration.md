# ABCD 첫 통합 실행 — 2026-09-23

후속 CV 제외 연구/교체 준비는 [A·B/A·C v2 인수인계](AB-AC-v2.md)에 기록했다. 아래는 첫 통합 시점의 재현 기록이며 최신 다음 작업은 후속 문서를 따른다.

사용자의 후속 지시로 A가 B/C/D를 직접 관리하며 실행한다. 이전 인수인계의 ‘사용자가 별도 작업을 시작’ 방침은 이번 묶음에 한해 이 지시로 대체한다. 재귀 위임은 하지 않는다.

공통 시작 커밋 `16b30cf`(C0 `5289d90`, 사전 선정 `ccda09a`, A 기준선 `bfea2f6` 포함). B/C/D 각각 별도 worktree/브랜치를 사용한다. 동결 모델·runtime_src·원자료·2026 데이터는 변경/추가 열람하지 않는다.

| 담당 | 이번 완료 목표 | 소유·출력 |
|---|---|---|
| A 주 에이전트 | 공통 계약·평가 유지, B 한 축 비교 검토, A4 실제 사건 입력 근거, 코드 통합·브라우저 확인 | 공통 docs, A scripts/results, 통합 검증 |
| B Sol medium | B1 추천 어댑터/대표 추천 재현, B3 실행 분포 교체 경계, B2 배합 이력 한 축 작은 비교 | `codex/b-recommendation-v1`, recommender/adapter, B 연구·테스트·인수인계 |
| C Sol high | C1 형식 먼저 공유, C2 모형 내부 대비/잔여 계산, C3 교체 입력과 불가 사유 | `codex/c-events-v1`, 새 event_analysis 모듈/계약·테스트·인수인계 |
| D Sol medium | D1 대표 추천·실제 비교, C 형식 이후 D2 사건 카드/상태와 D3 불변 분석 버전 | `codex/d-observer-v1`, service/store/API/web·통합 테스트·인수인계 |

이번 범위는 첫 제품 통합 및 소규모 후보 비교다. 실제 CV 없이 선수별 인과 기여나 실제 제구 품질을 완료했다고 하지 않는다. C의 수치 성분은 순서가 명시된 모델 대비와 결과 잔여이며, 실제 의도가 없으면 실행/작전 성분과 비중은 null이다. 교체의 가용 명단/시점 자료가 없으면 관측된 등판 기록만으로 후보를 만들어내지 않는다.

B 실험 축은 기존 count/좌우 빈도 기준선에 현재 공보다 앞선 같은 타석의 직전 구종 계열만 추가한다. A 고정 TRAIN/DEV·지원 행·평활 기본값을 유지하고 결과를 보기 전에 config를 커밋한다. 4/30 스냅샷을 4월 TRAIN 피처로 쓰지 않는다. 예측 지표와 정책 가치는 구분하고 S1의 triple 0 등 불확실성은 계속 보존한다. 후보의 예측 차이만으로 서비스 모델을 교체하지 않는다.

대규모 학습 없음. B의 작은 빈도 적합만 실행하며 C/D는 기존 모델을 사용한다. 모든 산출물은 SSD의 작업별 신규 경로로 격리한다. 토큰·비용은 도구에 세션별 실측이 없으므로 미측정이다.

진행/최종 커밋, 검증 결과와 남은 의존성은 아래에 기록했다.

## 통합 결과와 버전

**첫 묶음 완료. 실행 코드 기준 `62b423f`**, 현재 루트 브랜치 `claude/intent-contract-0922`. 별도 브랜치의 변경을 핵심 diff 검토 후 순차 cherry-pick했다. 다음 작업은 최신 통합 루트에서 분기한다. 완료한 B/C/D worktree는 깨끗한 상태로 보존했다.

| 묶음 | 원 작업 커밋 → 루트 반영 | 결과 |
|---|---|---|
| A4 사건 근거 | `d4392b1`, `adf49f5` | 115개 실제 사건, 38개 관측 교체 입력. 실제 의도/가용 명단 부재 보존 |
| A3 B 독립 검토 | 사양/코드 `adf49f5`, 결과 `62b423f` | 동일 키·라벨·기준선·엄격한 이전 공 이력 확인, 후보 미채택 |
| B 추천·실험 | `1113e60`→`bfa7bc5`, `b4dcaf4`→`97b32ac`, `2058558`→`34951ad` | 전체 후보 대응/확률/가치 공개, 실행 커널 경계, 작은 이력 비교·저장 복원, 소스 변경 시 캐시 무효화 |
| C 계약·계산 | `7075730`→`8eab29d`, `ef8c9e9`→`bec7228`, `bd3d099`→`45608d6`, `0dd638c`→`f83d4bc` | C1을 먼저 D에 전달; 동일 모델·투구·사전 의도 검증, 부호 있는 모델 대비와 null/잔여, 합성 운영 승격 금지 |
| D 화면·저장 | `b70c31a`→`168159d`, `39470e9`→`426cab8`, `b66258d`→`0e5bbf2` | 대표 추천·그 공의 실제 비교·사건 카드, 저장 추천/분석 버전 불변, 모델 변경/동시 GET 경계 |
| 통합 검증 | `62b423f` | PC/모바일 실제 K/HR 재생, B 검사 캐시 격리 |

공통 계약은 `C0-v1`, 가치 `defense-we-pa-v1`, 공개 추천 `observer-zone-v1`, 사건 `event-analysis-v1`, 기준 정책 `observer-repertoire-kernel-v1`이다. 동결 번들은 `/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/minimal-pitch-service-v1`, manifest SHA256 `43ece920cb60c6c24ea9f1e720a7000b3b83038a3e24ac27d77ef17a2e8f0f1f`. 실제 번들의 학습 가중치는 바꾸지 않았다. 어댑터 소스 변경 때문에 추천 identity/cache key는 갱신된다. 저장 추천이 현재 모델/어댑터와 다르면 새 모델로 기준값을 재해석하지 않고 비교를 unavailable로 둔다.

A S0/S1은 기존 `A-S0S1-001`의 사전 고정 코호트·시간 분할 그대로다. B는 count/좌우 기준선에 직전 구종 family 한 축만 추가했다. S1 563구에서 NLL **1.589641→1.602247**, 후보−기준선 **+.012606 [+.003048,+.025801]**. 작은 개발 평가에서 악화하여 후보를 채택하지 않았다. 서비스 모델 NLL 1.521566은 별도 학습량을 가진 기존 blend 결과다. ECE 일부 개선을 정책 우위로 취급하지 않는다. 실제 목표 개입/OPE·인과 기여는 미식별/미측정으로 유지한다.

C의 전체 차이는 `100 × (타석 종료 후 초기 수비팀 WE − 마지막 공의 저장된 사전 기준 정책 WE)`다. 확률과 %p, PA와 이닝을 분리한다. 실제 의도 없이 작전/실행/타자 몫을 채우지 않으며 전체 차이를 미배분 잔여로 남긴다. 기존 RE24 연구와 혼합하지 않는다. 목표 구종·위치의 유효한 사전 근거가 들어온 뒤의 모델 대비와 실제 제구 성능 검증은 별도다.

## 검증·실제 화면

- 최종 backend/계약/B/A 검사 **150 passed, 2 기존 라이브러리 deprecation warnings**, pytest 보고 1.93초. npm/TypeScript/Vite 빌드 통과.
- 최초 통합 검사의 실패 1건은 테스트가 이전 실행의 시뮬레이션 캐시를 읽어 cold-cache 가정이 깨진 문제였다. 캐시를 임시 경로로 격리하고 환경변수를 복원하도록 수정했다. 웹 검사 최초 1건은 같은 클래스의 접기 요소 2개에 단일 locator를 써서 실패했으며 모든 요소가 접혔는지 검사하도록 수정했다. 제품 실패를 성공으로 숨기지 않고 최초 로그도 보존했다.
- 앱 내 브라우저 제어 도구가 노출되지 않아 기존 Playwright의 별도 headless Chromium을 사용했다. 실제 서버/모델/DB/API에서 동작시켰으며 screenshot을 직접 확인했다.
- 삼진 `776703:3`(2025-08-17, 1440px): 3구 재생, partial **+1.202167 %p**. 홈런 `776710:10`(같은 날짜, 390px): 2구 재생, partial **−22.586870 %p**. 두 사건 모두 작전·실행·결과 성분/비중 null, 미배분 잔여=전체 차이. PC/모바일 가로 넘침·브라우저 console/page error 없음.
- 공개 전 실제 공 없음, 매 공 공개 후 정확히 그 공의 사전 추천과 동일, 기존 공 다시 선택, 수동 목표 메모 후 사건 불변, 새로고침 후 추천/분석/메모 유지 확인. 당시 모델 identity 변경·잘못된 추천/투구 링크·동시 GET 및 분석 revision 충돌은 backend 검사로 확인했다.
- 이 두 화면 사례는 기존 Observer 자료의 K/HR 동작 검사다. S1 자료나 미열람 최종 평가로 부르지 않는다. 기존 replay dataset SHA256 `8ad2b369884265bb5197137499e002a32354930cce036253f92b91671dde479b`, 새 통합 실행 경로로 복사했다.
- 기계 판독 결과·아티팩트 SHA: `results/ABCD-integration-v1.json`. 화면/로그/독립 DB는 `/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/ABCD-integration-v1/`의 `browser-*.png`, `browser-smoke.json`, `pytest-final.log`, `frontend-build.log`, `server.log`, `worker.log`.

## 재현

루트 `/Users/song/Projects/pitcheezy`에서, 승인 SSD와 기존 환경을 사용한다. 기존 모델/데이터를 새로 학습할 필요 없다.

```sh
# 통합 검사 (모델 캐시 출력만 통합 실행 경로 사용)
PYTHONPATH=.:apps/observer/backend \
PITCHEEZY_OBSERVER_RUN='/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/ABCD-integration-v1' \
.venv-observer-standalone/bin/python -m pytest apps/observer/backend/tests \
  tests/test_a_contract_examples.py tests/test_b_recommendation.py tests/test_a_small_eval.py -q

npm --prefix apps/observer/web run build

# 검증 후 종료한 서버를 다시 여는 명령
PITCHEEZY_OBSERVER_RUNTIME=standalone \
PITCHEEZY_OBSERVER_RUN='/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/ABCD-integration-v1' \
PITCHEEZY_OBSERVER_PYTHON='/Users/song/Projects/pitcheezy/.venv-observer-standalone/bin/python' \
PITCHEEZY_OBSERVER_PORT=8769 sh apps/observer/run.sh

# 서버가 떠 있는 동안 별도 터미널. 새 세션 생성; 같은 이름의 QA 로그/그림만 갱신
PLAYWRIGHT_BROWSERS_PATH='/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/observer-mvp-v1/browser-cache' \
node apps/observer/web/tests/observer-smoke.mjs
```

A S0/S1·B 적합/보관·C 독립 smoke 재현은 각 [A](A.md), [B](B.md), [C](C.md), [D](D.md) 인수인계와 고정 config를 따른다. 최초 완료 자료 생성 명령을 불필요하게 재실행하지 않는다.

## 역할·사용량·현재 프로세스·남은 한 단계

B `gpt-6-sol/medium`, C `gpt-6-sol/high`, D `gpt-6-sol/medium`을 실제 모델 인자로 지정했다. 각자 필요한 맥락만 전달하고 수정 범위를 분리했으며 하위 에이전트의 재귀 생성은 금지했다. A 주 에이전트는 Astra 역할(계약·가치·누수/평가·실패 판단·diff/근거 검토·통합)을 수행했다. 주 에이전트 정확한 모델 ID 및 에이전트별 토큰/비용 카운터는 미노출이므로 **미측정**이다. C의 high는 모형 분해·부호·결측·정체성 경계 구현 때문에 사용했다.

별도 worktree 위치는 `/Users/song/Projects/pitcheezy-worktrees/{b-recommendation-v1,c-events-v1,d-observer-v1}`. B/C/D 에이전트 종료, 브랜치 깨끗함 확인. 통합 서버 PID 82776·워커 82780도 검증 후 정상 shutdown 및 프로세스 부재를 확인했다. 남은 학습·서버 프로세스 없음. 동결 모델·runtime_src·원자료는 수정하지 않았고 2026을 추가 열람하지 않았다.

B/C/D는 모두 시작했고 이번 첫 증분을 통합했다. 실제 CV 기반 B4/C4/D의 품질 검증, 후속 입력의 서비스 ingestion, 당시 가용 교체 후보/휴식 자료와 별도 이닝 종료 평가기, 식별 가능한 정책 평가·별도 최종 검증은 남았다. C3은 입력 경계와 불가 사유까지 제공하며 교체 대안 계산 완료가 아니다. 비동기 CV 정정 API가 완성됐다고 하지 않는다.

**사용자 일정 갱신(2026-09-23): CV는 약 일주일 뒤 인수 가능하므로 그전 실행 범위에서 실제 CV 인수·학습·품질 검증을 제외한다.** CV를 기다리는 것을 다음 작업으로 두지 않는다. 기존 계약과 unavailable 동작은 유지한다.

**다음 한 가지: A2의 비영상 선행연구 비교 사양을 실행 가능한 공통 기준선으로 구체화한다.** 현재 서비스와 같은 정보 시점·행동·WE 범위의 비교 가능성을 먼저 검토하고, 충실 재현/변형 및 식별 가능한 평가 범위를 구분한다. 이후 B 후보 악화 원인·다음 한 축 사전 고정, A4/C3의 당시 가용 명단·휴식 자료/이닝 평가, D의 기록 재생·정정 처리·팬 사용 검증을 진행한다. 새 B 후보나 S2 확장은 별도 고정 실험 사양과 판정 후 진행한다. 실제 CV가 도착하면 기존 IntentEstimate의 투구 ID·좌표·릴리스 이전 시점·검토 상태로 인수 검사한다.
