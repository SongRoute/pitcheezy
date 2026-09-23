# A — C0 및 작은 실제 데이터 기준선 완료

후속 ABCD 실행 갱신(2026-09-23): 사용자가 A의 직접 관리 아래 B/C/D 위임 실행을 지시해 첫 통합까지 완료했다. 아래 1~6절은 **초기 A 단독 완료 시점** 기록이다. 당시의 ‘B/C/D 미실행·C 형식 미수신·UI 미변경’은 그 시점에 해당하며, 최신 상태는 7절과 [통합 인수인계](ABCD-integration.md)를 따른다.

2026-09-23. 이번 A 범위인 공통 준비·S0/S1 실행·검증 완료. 정책 성능 우위·실제 CV/기여도 품질 완료와는 구분한다.

## 1. 공통 기준·산출물 버전

| 항목 | 고정 내용 |
|---|---|
| 작업 브랜치 | 기존 `claude/intent-contract-0922` 유지 |
| 시작 상태 | HEAD `8771c06`; 계획 관련 7파일만 미커밋. 프로젝트 학습·서버 프로세스 없음 |
| 계획 기준 | `a82ddf4` — D49/D50 및 A~D 인수인계 관련 변경만 커밋 |
| **B/C/D 공통 기준** | **`5289d900344cefeeeb91cb1c5218fde8ff8863f2`** — 계약·모델·실행 예시·개발 의도 사례 |
| 결과 전 자료/평가 고정 | `ccda09ab02872e79586b34ad983237e300397cfd` |
| 실제 기준선 결과 | `bfea2f6260a00c5bf0b535e67f2473cf938ee391` — 위 두 커밋을 포함하므로 새 B/C/D 작업은 이 커밋에서 시작 가능 |
| 계약·가치 | `C0-v1`, 기존 IntentEstimate v1, `defense-we-pa-v1` |
| 모델 | `observer-zone-v1` / `minimal-pitch-service-v1`, 5 seeds 42–46; 모델 변경 없음 |
| 자료·실험 | `EXP-A-S0S1-001`, 승인 원자료 버전 `d20260911-s2325`, config와 manifest의 SHA256으로 식별 |
| 마지막 인수인계 커밋 | `git log -1 --format=%H -- docs/handoffs/A.md`로 확인. 본 문서의 커밋은 결과 코드 변경 없음 |

- 공통 계약: `docs/contracts/C0-v1.md`; 서비스 API: `apps/observer/CONTRACT.md`.
- 모델 위치·전체 해시·실행 명령: `docs/contracts/model-v1.json`.
- 실제 고정 모델 입출력: `docs/contracts/examples/model-v1.json`.
- 합성 의도 9사례: `docs/contracts/examples/c0-v1.json`.
- 선정/분할/평가/선행연구 비교 사양: `docs/contracts/S0-S1-v1.md`.
- 설정: `configs/EXP-A-S0S1-001.json`.
- 자료 목록·해시: `results/EXP-A-S0S1-001/selection_manifest.json`.
- 기준표·상세 재현·한계: `results/EXP-A-S0S1-001/README.md`; 수치 원본 `results.json`, 무결성 `verification.json`.

## 2. 공통 계약과 자료·평가

초기 수비 팀의 최종 승리 확률 [0,1]을 저장하고 차이는 100×ΔW %p로 표시한다. 현재 타석 종료까지 계획하며 이후는 동결 WE continuation이다. 이닝 종료 교체 평가와 RE24 연구는 별도 범위/표다. 기존 투구 키 `game_pk:at_bat_number:pitch_number`, 포수 시점 좌표, IntentEstimate 및 검토 라벨 분리를 재사용한다. 늦은 도착·정정은 사건 버전만 갱신하고 저장된 사전 추천을 바꾸지 않는다.

TRAIN 순위 상위 2명 Webb(657277)·Wheeler(554430)를 결과 계산 전에 고정했다. 투수별 4월 첫 3경기를 작은 TRAIN, 7월 첫 3경기를 DEV로 선택했다. 날짜·경기 ID 순이며 지원/결과로 경기를 재선정하지 않았다. 선정 경기 전체 행과 타석/투구 이력을 보관하고 점수 계산에서만 지원 마스크를 적용했다.

| 단계 | 보존 자료 | 점수·사용 범위 |
|---|---|---|
| 작은 TRAIN | 4/1~13, 6경기·429타석·1,655구 | 선정 투수 149타석 중 145 지원 타석으로 count/좌우 빈도 기준선 적합 |
| S0 | DEV의 투수별 첫 경기, 7/5·6 | 공통 지원 211구; 저장/복원·추천·평가 연결; 목표 추천은 첫 지원 타석 1개/경기 |
| S1 DEV | 7/5~21, 6경기·439타석·1,690구 | 선정 투수 160타석 중 151타석·563구 공통 예측 평가 |

동결 서비스의 원래 TRAIN은 2023-05-15~2025-04-30, 기존 조기종료/보정은 5~6월이다. 이번 7월은 과거에도 사용된 개발 자료이며 새로운 확인적 평가셋이 아니다. 2026은 기존 P0/P1 OPE 노출 이력을 유지하고 이번 작업에서 추가로 열지 않았다. 최종 평가셋은 학습/튜닝에 쓰지 않았다.

이번 서비스 예측 프로필은 4/30 동결 스냅샷이고 7월에만 사용했다. **B는 이 스냅샷을 더 이른 4월 TRAIN 공에 사용하면 안 된다.** B의 학습 피처는 각 공 이전 데이터로 만들어야 한다. S0 Observer 연결 검사는 4월 자료만의 존 높이·레퍼토리를 써서, 서비스의 전체 과거/최근90일 집계 입력 범위와 구분했다. 실제 현재 공의 물리 피처·결과는 추천에 전달하지 않았다.

로컬 처리자료는 기존 승인 manifest의 SHA256과 일치했다. 클라우드 원자료 재해시는 timeout이어서 기존 원자료 해시 기록을 보존했으며 `raw_rehashed_this_run=false`로 기록했다. 원본·모델을 덮어쓰지 않았다.

## 3. 실제 결과·검증·재현

| S1 예측기 | NLL | Multiclass Brier | Top-label ECE10 |
|---|---:|---:|---:|
| 고정 서비스 blend(구종 주변화) | 1.521566 | 0.726403 | 0.009218 |
| 고정 frequency | 1.555541 | 0.734695 | 0.047247 |
| 작은 TRAIN count/좌우 기준선 | 1.589641 | 0.746602 | 0.018373 |

blend−작은 기준선 NLL −0.068076 [−0.107074, −0.033128], blend−고정 frequency −0.033975 [−0.049475, −0.009452]. 경기 paired bootstrap 95% CI, 1,000회, seed 20260923. 6경기의 노출된 개발 자료·서로 다른 학습량이므로 새 구조 효과나 실세계 성능 우위의 증거로 승격하지 않는다.

- S0 현재 목표 추천 2회 모두 ready. 이 2/2는 연결 검사 범위이며 전체 위치 정책 지원률이 아니다.
- DEV 타석 지원률 151/160. 제외: 종료 공 없음 1, 타석 중 상태 변화 3, 미지원 5. S1 triple 없음, S0 triple/hbp 없음: 해당 클래스 상세 성능 null.
- 구종 정책 내부 WE는 151타석에서 별도 저장. 실제 정책 가치·목표 개입 OPE는 의도/propensity 부재로 **미식별·미측정**. 커널 ESS와 OPE ESS를 구분한다.
- 전체 선정 경기의 홈런 13개·삼진 102개를 `event_cases.json`에 보존했다. 비코호트 선수 사건도 포함하므로 고정 모델 지원 여부를 C가 검사해야 한다. 실제 의도·목표 구종·사인 결정자·당시 가용 불펜/휴식 정보는 없음. 교체 반사실 계산은 아직 불가하다.
- C0 실제 재생 첫 공 `776703:1:1`: standalone 모델 복원·추천 ready, 새 인스턴스의 캐시 저장/복원 동일, actual 필드 변조 불변, 확률→%p 일치.
- S0/S1: 작은 baseline 적합/저장/복원 동일, 독립 **research Engine** 재로드의 neural/frequency/blend 텐서 정확히 일치, 동결 소스 검증 통과. standalone과 research 실행 범위를 혼동하지 않는다.
- Sol 의도 계약 검증 28개와 fixture 6개 통과. 최종 C0/service/runtime/catalog 검사 30개 통과(fixture 6개 포함), 데이터 선정·전체 타석/투수 변경·저장 복원 검사 5개 통과. 핵심 diff 검토에서 발견한 투수 변경 시 부분 타석 필터 위험은 점수 계산 전에 수정했다.
- 출력·선정·코드/config·번들 각 파일 해시 검증 통과. 별도 읽기 전용 검증 명령도 통과했다. 평가 실측 12.077초, 최대 RSS 421,773,312 bytes(macOS). 준비 단계 최대 RSS 2,686,697,472 bytes(처리자료 읽기).
- 실제 CV DB 재확인: 기존 AI 미검토 annotation 5개, pitch_id 연결 0, calibration_corners 0, 추적 4개 모두 abstained. 기존 미완료 상태 유지. 합성 fixture를 실제 라벨로 저장하지 않았다.

프로젝트 루트 `/Users/song/Projects/pitcheezy`에서:

```sh
# 현재 결과의 읽기 전용 검증
.venv-observer-standalone/bin/python scripts/a_verify_run.py

# 모델 예시 재실행(새 출력 파일, A 전용 캐시)
.venv-observer-standalone/bin/python scripts/a_model_smoke.py \
  --run '/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/A-C0-v1' \
  --output '/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/A-C0-v1/model-example-replay.json'

# 최초 생성 환경에서만: 완료 자료는 자동 덮어쓰지 않음
.venv-observer-standalone/bin/python scripts/a_small_eval.py prepare
# 선정 manifest를 커밋한 다음
.venv-observer-standalone/bin/python scripts/a_small_eval.py evaluate

# 계약·재생 경계 검사
PYTHONPATH=.:apps/observer/backend .venv-observer-standalone/bin/python -m pytest \
  tests/test_a_contract_examples.py apps/observer/backend/tests/test_observer_service.py \
  apps/observer/backend/tests/test_standalone_runtime.py apps/observer/backend/tests/test_catalog_context.py -q
# 평가 경계 검사
.venv-observer-standalone/bin/python -m pytest tests/test_a_small_eval.py -q
```

## 4. B/C/D 시작 범위와 남은 의존성

진행 메시지로 C0 커밋 직후 B1/C/D 시작을, 결과 커밋 직후 B 소규모 학습 시작을 각각 알렸다. 별도 작업은 사용자가 시작한다. A가 B/C/D 작업을 생성하거나 전체 구현을 위임하지 않았다.

| 알림 | 지금 시작할 범위 | 남은 의존성 |
|---|---|---|
| **B 추천 어댑터 개발 시작 가능** | C0·현재 모델·입출력 예시로 B1, 후보 확률/행동 대응 공개, 기존 추천 재현, 제구 교체 경계 | 새 모델은 같은 계약/평가를 통과한 뒤 적용 |
| **B 소규모 학습 실험 시작 가능** | 고정 TRAIN/DEV·지원 마스크·기준표로 요소 한 축 비교; TRAIN의 과거 프로필 재구성 | 실제 제구 학습/검증은 실제 CV, 최종 우위는 별도 적합한 평가 |
| **C 사건 기여도 개발 시작 가능** | 같은 수비 WE·단위·PA 범위와 개발 의도 샘플로 C1/C2. **분석 결과 형식을 먼저 D에 전달** | B 새 학습·CV 도착을 기다리지 않음. 실제 C4는 CV 인수; 교체는 당시 가용 후보 자료 필요 |
| **D 기본 관전 화면 개발 시작 가능** | 기존 API·사전 저장 추천·동일 투구 실제 비교로 D1 | **C 결과 형식 아직 미수신**, 사건 카드 D2는 그 형식 필요. D4 실제 연결/품질은 B/C 구현·실제 입력 필요 |

## 5. Astra/Sol 역할과 확인 가능한 사용량

- 주 에이전트는 Astra 역할로 공통 계약·가치·선정/누수 설계, 핵심 코드/근거 검토, 실패 원인 판단 및 시작 판정을 수행했다. 도구가 주 에이전트의 정확한 런타임 모델 ID를 노출하지 않아 Astra라는 모델 ID 자체는 독립 확인하지 못했다.
- `collaboration.spawn_agent`에서 모델 지정 가능함을 확인하고 **`gpt-6-sol`, `medium`**으로 2개를 명시 지정했다: `c0_fixtures`(합성 사례·검사), `s1_path`(기존 자료 점검·작은 평가 구현·실행·집계). 모델 식별 근거는 실제 도구 호출 인자다. high 사용 없음.
- 전체 대화 상속 없이 필요한 목표·파일·제약만 전달했고 수정 범위를 분리했다. 재귀 생성 없음. 모델·학습량·데이터 선택 연구 판단은 주 에이전트가 고정하고, 결과 계산 전 커밋 후 Sol에 실행을 맡겼다.
- 세션/에이전트별 토큰·비용의 확인 가능한 값 없음: **미측정**. 추정 사용량을 실제 사용량으로 보고하지 않는다. 위 시간/RSS만 실제 프로세스 측정값이다.

## 6. 프로세스·로그·남은 것·다음 한 가지

평가와 검증 모두 정상 종료했으며 이 A가 남긴 학습·서버·워커 프로세스 없음. 대규모 계산/전체 재학습을 시작하지 않았다. UI·CV 알고리즘·새 추천 모델 및 동결 서비스 소스를 변경하지 않았다.

SSD 출력 루트 `/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/`:

- `A-C0-v1/`: 추천 캐시, `verification.log`, 기존 CV 상태 `existing_cv_inventory.json`.
- `A-S0S1-001/`: 전체 경기 train/dev parquet, `selected_pitch_keys.json`, baseline pickle, 예측 npz, 내부 WE/목표 추천 진단, 사건 목록, `evaluate.log`, `reverify.log`.
- 큰 자료는 Git에 넣지 않고 manifest·config·수치 보고서·검증 결과만 커밋했다.

남은 것은 B 후보 모델의 같은 조건 평가(A3), C의 결과 형식/사건 계산, D 화면 통합, 실제 CV 라벨·당시 가용 교체 명단, 식별 가능한 정책 평가와 최종 확인이다. 이번 C0/S0/S1 완료를 이 항목들의 완료로 표시하지 않는다.

**다음 한 가지: C가 `defense-we-pa-v1`의 분석 결과 형식과 개발 샘플을 먼저 고정해 D에 전달한다.** B의 소규모 학습과 D1은 동시에 시작할 수 있다.

## 7. 후속 A3/A4·ABCD 직접 관리 완료

사용자 후속 지시로 공통 시작점 `16b30cf`에서 B/C/D를 별도 worktree의 Sol에 직접 위임했다. A는 평가 판단·핵심 diff 검토·통합 검증을 맡았고 재귀 위임은 하지 않았다. 통합 실행 코드 `62b423f`, 세부 커밋·명령·로그·역할은 [ABCD 통합 인수인계](ABCD-integration.md)에 있다.

- A3: `configs/EXP-A-B-REVIEW-001.json`, `scripts/a_review_b.py`, `results/EXP-A-B-REVIEW-001.json`. B의 고정 예측 배열을 독립 검토하여 동일 563구/라벨/경기/기준선과 이전 공만 쓰는 이력을 확인했다. NLL 1.589641→1.602247, 차이 +.012606, 경기 bootstrap 95% [+.003048, +.025801]. 새 후보 미채택, 현 서비스 유지. 2스트라이크 부분집합도 악화하며 해당 절편은 사후 기술 분석이다. OPE/정책 값은 null. 튜닝·재선정·새 평가셋 열람 없음.
- A4: `configs/EXP-A-EVENTS-001.json`, `scripts/a_event_packets.py`, `results/EXP-A-EVENTS-001.json`; SSD `A-events-v1/event_packets.json`. 기존 고정 DEV에서 삼진 102·홈런 13, 관측 투수 교체 38건을 시점·상태·투구 ID·전체 타석 ID와 함께 기록했다. 후속 상태 누락 1건. 실제 의도/당시 가용 명단은 null이며, 나중에 등판한 투수를 당시 후보로 취급하지 않는다. 사건 후 분석 입력이므로 사전 추천 피처에는 사용하지 않는다.
- B 추천 어댑터/실행 분포 경계, C `event-analysis-v1` 결과 형식·계산 경계, D 화면/API/버전 저장까지 반영했다. C 형식 미수신 의존성은 해소됐다. 실제 의도 없는 기여 성분·비중은 미측정이며, 후기 수동 메모를 의도로 승격하지 않는다.
- 150개 통합 검사 통과, 웹 빌드 통과. 실제 2025년 Observer 삼진(PC)·홈런(390px) 타석에서 추천→실제→사건 카드, 사전 추천 불변·새로고침·메모 격리·가로 넘침 없음 확인. 이 두 사례는 화면 연결 검사이며 S1/최종 성능 평가가 아니다.
- 위임 모델 실제 지정: B `gpt-6-sol/medium`, C `gpt-6-sol/high`(복잡한 가치 분해), D `gpt-6-sol/medium`; 전체 대화 상속 없음. 주 에이전트는 Astra 역할, 정확한 모델 ID 미노출. 토큰/비용은 계속 미측정.
- 통합 SSD `ABCD-integration-v1/`에 새 DB·캐시·서버/워커/pytest/브라우저 로그·스크린샷을 보존했다. 검증 서버 PID 82776·워커 82780은 종료 확인, 남은 학습/서버 없음. 동결 모델·원자료·runtime_src·2026 자료 변경/추가 열람 없음.

남은 것은 실제 CV의 동일 투구·릴리스 이전·좌표 검증과 제구/기여 품질, 실제로 가용한 교체 후보 검증, 식별 가능한 정책 효과와 최종 확인이다. **사용자 후속 일정: 실제 CV 인수는 약 일주일 뒤이므로 당장 작업에서 제외한다.** S2 학습 확대는 자동 진행하지 않는다.

## 8. CV 제외 A·B/A·C 후속 실행

공통 count/구종/투수 조건 기준선을 실제 실행하고 첫 B 후보의 악화를 진단했다. 다음 후보 `EXP-B-PAHISTORY-002`는 시간순 TRAIN 내부 선택·alpha=0 허용·첫 공 정확한 부모 복귀로 고정했으며 아직 실행하지 않았다. 고정 6경기의 당시 등록 투수/이전 7일 workload와 결정 전 타순 근거를 준비했고, 동결 WE를 쓰는 별도 반이닝 평가기를 구현·실행했다. 조건부 모형 계산과 실제 교체 효과를 구분한다. 최종 검증·근거·남은 범위·재현은 [A·B/A·C v2 인수인계](AB-AC-v2.md)를 따른다. 다음 단계는 고정한 B002의 TRAIN 검증을 구현·실행하고, 선택 결과를 커밋한 뒤 DEV 실행 여부를 결정하는 것이다.

## 9. B002 실행 완료

위 설계 고정 이후 사용자 지시로 B002를 실행했다. 소스 `4ccd31f` → TRAIN 선택/체크포인트 `fa07bbb` → DEV 1회 순서를 지켰다. TRAIN alpha=.75 선택, DEV NLL +.005142 악화로 미채택했다. 첫 공/미관측 행은 부모와 정확히 같아 의도한 수정은 확인했다. 관련 검사 13개 통과, 실행 프로세스 없음. [B002 실행 인수인계](B002-execution.md)에 산출물·해시·재현·Sol 역할/측정 자원을 기록했다. 다음 한 가지는 C 조건부 이닝 결과 형식을 고정해 D에 전달하는 것이다. 동결 모형·CV/2026/최종 자료 제한은 유지한다.

## 10. C → D 이닝 결과 전달 완료

계약 `1403211`, 변환/소비 코드 `8979588`. 실제 기존 결과를 `inning-result-v1`으로 변환하고 D의 표시 함수까지 연결 검사했다. Python 80개·D fixture·TypeScript 검사 통과. 새 추론/학습/자료 수집 없이 원문을 보존했다. [C→D 인수인계](C-D-inning-v1.md)에 산출물·모델/출처 hash·재현·Astra/Sol 역할과 미측정 사용량을 기록했다. 다음 한 가지는 교체 이벤트 직전 상태를 식별하는 저장/API 연결이다. 실제 교체 가용성/효과, 운영 카드 연결은 아직 미완료다.
