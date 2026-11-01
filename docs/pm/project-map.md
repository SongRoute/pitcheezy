# 프로젝트 지도

소유: 통합 문서 담당(코드 책임은 각 담당자) · 갱신: 모듈/정본 경로 변경 시 · 갱신일: 2026-11-01(`6e2038fd` 문서 경로 대조).

[시작](../../README.md) · [온보딩](../onboarding.md) · [실제 배치](operations-inventory.md) · [자원 규칙](resources.md). 아래는 논리 구조이며 폴더 이동·삭제 계획이 아니다.

## 코드에서 찾기

| 경로 | 목적·책임 | 변경 경계 |
|---|---|---|
| `src/pitcheezy/` | 공통 연구·모델·인터페이스 구현, 연구 담당 | `interfaces` 스키마 변경은 [계약](../interface-spec.md)·검사를 함께 검토 |
| `experiments/pitchmdp/` | 실험별 runner·비교/ablation·등록, 연구 담당 | 비슷한 runner라도 과학 계약이 다르면 병합하지 않음; frozen source 보존 |
| `configs/`, `results/` | 등록 설정·결과 식별, 연구 기록 담당 | 기존 결과·seed/config를 덮지 않고 새 attempt 사용 |
| `apps/observer/` | backend/web/scripts 및 로컬 재현, 서비스 담당 | [앱 안내](../../apps/observer/README.md)는 역사적 재현 경로 포함; 현재 serving은 receipt 확인 |
| `tests/`, 앱·실험 하위 tests | 각 계약의 회귀 검사, 해당 코드 담당 | 데이터 없는 작은 검사부터; 전체 테스트를 fit/운영 검증으로 대체하지 않음 |
| `scripts/` | 개발·자료·실행 지원, 작업별 지정 담당 | 이름만 보고 실행하지 않고 호출 경로·출력·자원 범위 확인 |
| `data/`, `runs/` | 로컬/외부 연결 자료·실행 산출물, 연구/서비스 소유 분리 | gitignore는 삭제 허가가 아님; `runs/`는 이 Mac에서 symlink |
| `graphify-out/` | 탐색 보조, 통합 tooling | 정본 증거 아님; 그래프 결과와 현재 원문 대조 |

## 문서에서 찾기 — 내용별 정본 하나

| 질문 | 원문 | 갱신 책임/계기 |
|---|---|---|
| 팬 경험·표현과 검증할 예시 | [사용자 경험 방향](../product/experience.md) | 통합 제품; 사용자 합의/검증 결과 |
| 현재 전시 방향·역할·일정 | [10/31 인계 §2·§5](../handoffs/CODEX-resume-2026-10-31.md) | 통합; 최신 사용자 지시와 결정으로 갱신 |
| 오늘 무엇이 끝났고 실행 중인가 | [통합](integration/state.md), [연구](research/state.md), [서비스](service/state.md) | 각 지정 작성자; 새 증거/소유권 인수 |
| 어떻게 착수·검증·통합·복귀하나 | [workflow](workflow.md) | 통합; 절차/수용 기준 변경 |
| 연구/서비스의 넘지 않을 경계 | [contracts](contracts.md) | 양쪽 검토 후 통합 단일 작성 |
| 서버/GitHub/worktree/보존 대장 | [운영 인벤토리](operations-inventory.md) | 자원 소유자 관측 수령 후 통합 |
| 텐서/위치/설계 | [interface-spec](../interface-spec.md), [design](../design.md) | 계약 변경과 코드/검사 동시 검토 |
| 결정/실험 증거 | [decisions](../decisions.md), [experiments](../experiments.md) | 지정 기록자; 승인 결정/봉인 결과 추가 |
| 과거 이유/재현 | `docs/handoffs/`, `docs/reports/`, `SESSION_HANDOFF.md` | 역사 보존; 현재 상태 주장은 state로 연결 |

README→이 지도 또는 운영 안내→원문까지 두 번의 링크로 찾는다. 신규 문서는 변경 이유와 소유자가 독립적인 경우에만 만든다. 동일 상태/수치를 여러 문서에 복사하기보다 원문과 관측 시각을 링크한다. 대규모 rename·코드/자료 이동·worktree prune은 이번 문서 정리 범위에 없다.

[10/6 월드시리즈 계획](integration/world-series-launch-plan-2026-10-06.md)은 당시 계획이며 현재 전시 방향을 대체하지 않는다. 지도 변경 전 내용은 [고정 원문](https://github.com/SongRoute/pitcheezy/blob/6e2038fd0010657e6f6f2d1e49820490f6195da4/docs/pm/project-map.md)에서 확인한다.
