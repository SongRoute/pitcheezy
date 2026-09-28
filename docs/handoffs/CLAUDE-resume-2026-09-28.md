# Claude 재개 인수인계 — D91 이후

2026-09-28 작성. 이 문서는 새 세션의 시작점이다. 기존 긴 인수인계의 과거 ‘다음 작업’을 현재 지시로 오해하지 않도록 현재 상태와 다음 작업만 정리한다. 과학 규약 자체를 변경하거나 실험 실행을 새로 승인하지 않는다.

## 1. 현재 위치와 먼저 확인할 것

- 저장소: `/Users/song/Projects/pitcheezy`
- 통합 브랜치: `codex/ml-matrix-execution`
- 인수인계 작성 직전 검증 완료 HEAD: `ec514f5b49b620e0ca91c260e743a97fe61bf236`. 이 문서를 추가한 커밋은 그 후손이다. 재개 시 현재 HEAD를 확인하고 이 해시로 reset하지 않는다.
- 코드 통합: `d7e5bae`; Opus 작성/독립 검토 기준: `1ecdbb42031874ba69da2911a8baf25431d02bc0`.
- [PR #34](https://github.com/SongRoute/pitcheezy/pull/34): 기존 연구 작업을 누적한 PR. 직전 작업까지 push 완료, merge하지 않음. 새 PR 중복 생성·자동 merge하지 않는다.
- 마지막 작업 COOP-016은 `completed_implementation_only`. 원래 감독기는 종료됐고 실제 실험 큐는 시작하지 않았다. 재개 시 새 프로세스/다른 세션 변경이 생겼는지 확인한다.
- 작성 worktree: `/Users/song/Projects/pitcheezy-worktrees/policy-runtime`, branch `codex/policy-runtime`. 작성 커밋은 이미 통합됐다. 같은 커밋을 다시 cherry-pick하거나 이전 attempt를 재실행하지 않는다.

```sh
cd /Users/song/Projects/pitcheezy
git status --short
git branch --show-current
git log -5 --oneline
```

읽기 순서:

1. [AGENTS.md](../../AGENTS.md), [CLAUDE.md](../../CLAUDE.md), [협업 규칙](../AI_COLLABORATION.md).
2. 이 문서, [최신 세션 상태](../SESSION_HANDOFF.md)의 맨 위 D91, [ML 인수인계](ML-experiments-next-session.md)의 맨 위 D91.
3. [구현 보고서](../reports/ML-policy-runtime-implementation-2026-09-28.md), [독립 검토](../reviews/COOP-016-policy-runtime-implementation-2026-09-28.md), [기계 검증 기록](../../results/ML-policy-runtime-review-v1.json).
4. [정책 준비 계약](../contracts/MLB-2026-POLICY-PREPARATION-v1.md), [설정](../../configs/MLB-2026-POLICY-PREPARATION-v1.json).
5. 2026·실제 검증 자료에 관한 판단이 필요하면 [사용 이력 감사](../reports/MLB-2026-use-history-audit-2026-09-28.md), [평가 초안](../contracts/MLB-2026-OPE-PREREG-DRAFT-v1.md), [준비도](../reports/MLB-2026-evaluation-readiness-2026-09-28.md).

과거 결정은 [decisions.md](../decisions.md)의 D86–D91을 확인한다. 전체 역사와 이미 완료한 실험을 처음부터 재조사할 필요는 없다.

## 2. 연구 상태 — 그대로 유지할 결론

목표는 MLB 정규시즌 선발·불펜 전체의 다양한 상황에서 구종·목표 위치 확률을 제공하는 강건한 모델이다. 예측 성능과 추천 정책 가치는 구분해서 검증한다.

| 항목 | 현재 확정 상태 |
|---|---|
| G0 | 5모델 연구 예측 기준선 유지. 연구 번들 동결과 추천 정책 동결은 서로 다르다. |
| 전체 June 보정 비교 | D86에서 실행/보고 완료. 두 후보가 등록 최소 개선 폭에 미달하여 미선정. 재실행하지 않는다. |
| 정책 준비 M1 | D89 정의/합성 준비 + D91 런타임 구현 완료. M1 전체 완료나 실제 인과 식별 완료가 아니다. |
| 목표 위치 ARM-A | 의도/배정 식별 근거 부족으로 보류. 관측 도착 위치를 의도 목표로 대신하지 않는다. |
| 구종-only ARM-B | 비활성. ARM-A 보류를 이유로 자동 활성화하지 않는다. |
| 2026 평가 | 사전등록 초안·실행 비활성. 과거 OPE 사용이 있어 전체 시즌을 미열람 holdout으로 부를 수 없다. |
| 미열람 확인 자료 | 인증된 모집단 0개, 가용성 unknown. 미사용 기간을 추정하지 않는다. |

## 3. 이미 구현한 것 — 다시 만들지 말 것

| 경로 | 역할 |
|---|---|
| [policy_artifacts.py](../../experiments/pitchmdp/pitchmdp/policy_artifacts.py) | G0 5member factory, TRAIN BC 직렬화/복원, 별도 개입 지원 표, 버전/해시/구조 검증 |
| [policy_runtime.py](../../experiments/pitchmdp/pitchmdp/policy_runtime.py) | `build_runtime` → `submit(DecisionRequest)` → `summary()`, 후보 P3/기준/로깅 확률, 요청 JSONL 원장 |
| [test_policy_runtime.py](../../experiments/pitchmdp/tests/test_policy_runtime.py) | 신규 합성 15검사, factory→BC→P3→원장 연결 경로 |

지켜야 할 의미:

- G0는 각 member의 May temperature 보정 → **5개 확률 평균** → June **공통 앙상블** 가중치로 frequency 혼합. 개별 seed 가중치를 평균해 대체하지 않는다. 기존 3member 동결 코드를 변경하지 않고 추가했다.
- 로깅 법칙은 전체 TRAIN 어휘의 BC 확률이다. 추천 지원 마스크로 재정규화하거나 floor를 추가하지 않는다. 지원을 제한한 기준 정책은 별도로 계산한다.
- TRAIN BC는 canonical JSON, 필수 SHA, 어휘/개수표/행 수/출처 검사, 기존 파일을 덮어쓰지 않는 저장 방식이다. 출처 선언과 export가 관측한 날짜 범위를 구분한다.
- 미지원·실패도 제출 요청 분모에 보존한다. 마스크 밖 로그 행동의 정상적인 ρ=0과 logging positivity 실패는 다르다. 같은 ID/내용 재시도는 중복 집계하지 않고 충돌은 감사 기록 후 중단한다.
- 해시 체인만으로 원장의 끝부분 전체 삭제를 검출할 수 없다. 외부에 보관한 head SHA/행 수가 필요하다. 단일 writer 가정이다.

완료한 검증: Opus 신규15 + 기존 관련 포함54, D89 합성 `all_pass`, 오류 변이8종 검출. Sol과 Root가 각각 신규15를 재현했다. 이 수를 합산하지 않는다. 실패 fixture/수정 이력도 보고서에 보존했다. 합성 검사를 실데이터 성능이나 정책 우위로 표현하지 않는다.

소스 변경/새 우려가 없으면 완료 검사와 변이 검사를 반복하지 않는다. 변경 후 필요한 최소 검사의 재현 명령은 다음과 같다.

```sh
PYTHONPATH=src:experiments/pitchmdp PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q -p no:cacheprovider experiments/pitchmdp/tests/test_policy_runtime.py
```

## 4. 다음 작업 — 순서와 완료 기준

첫 작업 묶음은 **실제 구성요소 연결 명세·식별자 보완·≤2025 검증 등록 준비**다. 코드/기존 메타데이터 조사와 합성 구현·검사를 진행할 수 있다. 실제 payload 로드·BC 생성·실험 실행은 등록 준비와 구분한다.

| 순서 | 구체적인 작업 | 완료 기준 |
|---|---|---|
| 1 | 기존 source/config/manifest에서 frequency 객체, G0 `PolicyInputs`/전처리, member loader, delivery pool, terminal/cutoff, WE의 출처 및 τ·MC samples·pitch cap·seed·budget 설정을 추적 | 구성요소별 생성 코드·필요 입력·기존 pin·누락 정보·검증 방법을 표로 기록. 알 수 없는 파일 역할은 unknown 유지 |
| 2 | 완전한 정책 식별자와 실제 연결 검증 설계·구현 | pool/terminal/cutoff·전처리·member/frequency 로더까지 descriptor/source/artifact pin에 포함. descriptor 선언뿐 아니라 연결된 객체/파일과의 일치를 확인. 다른 구성요소 교체를 합성 검사로 검출 |
| 3 | TRAIN BC/개입 지원 표 생성 계획 작성 | TRAIN 2023-05-15..2025-04-30의 선택 조건·어휘·행 수·날짜·입력/설정/코드 해시·새 출력 경로·재현 명령 명시. 실제 생성 전이므로 가짜 실제 해시/행 수를 채우지 않음 |
| 4 | PA와 시간 출처 규칙 확정 | MID_PA 미지원, 불완전 PA, PA 종료/보상, 투구 전 가용 정보·history 출처 검증, 원 구종 코드 대응을 명시. 추정량/분모를 바꾸는 결정은 계약·결정 이력에 기록 |
| 5 | ≤2025 검증의 자료/비용/실행 계획 준비 | 자료 노출 이력, 개발 검증과 독립 확인 구분, 측정/미측정 비용, 상한·중단·실패·재시도 처리, 단일 큐, 소스/환경/config 식별자와 실행 명령 명시 |
| 6 | 독립 검토·인수인계 갱신 | Opus 작성, Sol 검사, Astra 핵심 과학 검토를 한 번으로 모음. 가능한 실행과 미해결 조건을 분리하여 보고하고 실제 실행 승인/등록 상태를 확인 |

현재 `runtime_sha`는 일부 구성요소의 식별자일 뿐 **완전한 동결 정책 ID가 아니다**. `we_identity` 문자열 하나로 실제 WE 구현이 인증되지 않는다. frequency 역할은 기존 bundle에서 명확히 지정되지 않았다. 실제 member 연결은 `run_ml_g0_whole.load_member`의 kind/seed/width/class/network/parameter/device 검증을 유지해야 한다.

현재 원장의 분모는 **제출된 유효 키 요청/PA**이며 전체 MLB 커버리지를 인증하지 않는다. history 검사는 국소 순서/길이/직전 구종의 일관성이다. 실제 투구 전 관측 가능성이나 원자료 출처 인증은 별도다.

## 5. 실행 경계와 협업

- 현재 완료 범위에는 실제 TRAIN BC 생성, G0 실가중치 연결/검증, 실데이터 fit/추론/OPE, 새 수집이 없다. 2026 행/배열/헤더/모델 payload를 새로 읽는 권한을 이 문서에서 만들지 않는다.
- `policy_frozen=false`, 실데이터 실행 비활성 설정을 유지한다. 실제 실행은 사용자 지시와 등록된 범위·자료핀·비용·검토 조건을 확인한 뒤 진행한다. 기존 승인된 코드/문서/합성 검사는 매번 허락을 다시 묻지 않는다.
- **Opus 메인, Sol 서브, Astra 필수 최소**. Astra는 새 과학 작업 묶음의 핵심 최종 검토 1회로 모은다. Fable 제외. 이번 재개 문서 정리는 새 과학 검토가 아니다.
- Claude가 직접 재개하면 Opus가 작업을 주도한다. Sol/Astra를 호출할 수 있는 협업 환경이 있으면 역할을 지킨다. 사용할 수 없으면 실제 사용 모델/미수행 검토를 솔직히 기록하고 독립 검토용 짧은 패킷을 준비한다. 접근 가능한 척하거나 다른 모델을 Astra/Sol이라고 적지 않는다. 검토가 없어도 독립적으로 가능한 구현/합성 검사는 진행할 수 있지만 과학 검토 완료/실험 release를 주장하지 않는다. 새 변경의 자기검토를 이전 커밋의 독립 PASS로 대신하지 않는다.
- 이전 Codex→Claude 감독기를 Claude가 거꾸로 호출하여 순환 위임하지 않는다. CLI `--dangerously-skip-permissions` 허용은 자료/실험 범위를 확장하지 않는다.
- AGENTS의 Ponytail 기본 모드와 연구 예외 준수. 비교 변형·ablation·설정/시드/로깅·평가 검증을 불필요 코드로 삭제하지 않는다. 기존 코드 일괄 리팩터링 금지.
- 코드 질문은 Graphify query로 탐색하고 코드 변경 후 `graphify update .`를 실행한다. AST만 갱신하며 문서/PDF 의미 분석과 유료 LLM 라벨 갱신은 보류한다.
- 기존 봉인 산출물·config·attempt를 덮어쓰지 않는다. 새 작업은 새 작업대장/출력 경로로 분리한다. 무거운 실행은 공유 `.heavy.lock`과 단일 큐 규약을 따른다.

## 6. 재개에 필요한 외부 기록

COOP-016 작업 디렉터리:

```text
/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/ML-MATRIX-20260924/coordination/20260928-policy-runtime
```

`board.json`이 완료 상태의 출처다. `COOP-016/attempt-001/state.json`, `events.jsonl`, `result.json`과 `opus-validation.json`, `sol-review-final.json`, `sol-independent-repro-final.json`, `astra-focused-review.json`에 원문 증거가 있다. 실제 Opus 모델은 `claude-opus-5-5`, CLI 1회이며 작성 시간 831.075초는 실험 비용이 아니다. SSD가 없으면 저장소의 기계 검증 기록/보고서로 코드·문서 준비를 계속하고 외부 증거 확인은 미완료로 표시한다.

인수인계 종료 때 현재 HEAD·수정 파일·테스트·새 데이터 노출/비용·미해결 항목·다음 한 작업을 기록하고 SESSION_HANDOFF 및 ML 인수인계의 최상단을 갱신한다. 기존 완료된 D91 원문과 원본 검토 기록은 보존한다.
