# Pitcheezy 작업 지침 — 정본

이 파일은 Codex와 Claude의 공통 지침이다. `CLAUDE.md`는 이 파일을 import한다. 최신 사용자 지시와 상위 도구 지침이 우선하며, 과거 역할·자료 권한을 현재 작업에 자동 적용하지 않는다.

## 목표와 범위

MLB 투구 추천 관전 서비스, 3인 캡스톤. 현재 전시 목표·역할·일정은 [운영 진입점](docs/pm/overview.md)을 따른다. 저장소가 정본이며 노션은 아카이브다. 전시 모델은 G0 동결이다. 2026 자료를 학습·튜닝·모델 선택에 사용하지 않는다. 과거 별도 연구 권한은 해당 고정 계약의 이력이며 이번 유지보수·전시 작업에 적용하지 않는다.

## 항상 지킬 것

- 숫자를 추정하지 않는다. 검사하지 않은 것은 **미측정**. 실행 성공, 기능 수용, 과학적 우위, 운영 적용을 구분한다. 작업 종료는 **한 일 / 검사 결과 / 미측정·남은 조건 / 결정 필요** 네 줄로 보고한다.
- 기존 구현·비교군·ablation·등록 설정·시드·검증 게이트·봉인 결과를 정리 목적으로 삭제하거나 약화하지 않는다. 원본·실패 시도도 보존한다.
- 실자료 실행은 정확한 입력·역할·as-of·분할·지표·예산을 먼저 등록하고 작성자와 다른 에이전트가 검토한다. 같은 모델 검토를 다른 모델의 독립 검토라고 부르지 않는다. 자료 노출·OPE 경계를 지킨다.
- 맥 기록기·수집 프로세스에 신호를 보내지 않는다. 실자료·GPU·MPS·운영 작업 전 [자원 계약](docs/pm/resources.md)의 소유권·메모리·공통 lock을 확인한다. GPU 허용 상한은 예약이 아니다.
- 한 작업 = 한 브랜치 = 한 PR. 별도 worktree와 단일 작성자/index를 사용한다. 기존 dirty·다른 작업·운영 자원을 보존한다. 커밋은 Conventional Commits, scope는 모듈명이다.
- 작업 전과 쓰기 직전 [작업큐 절차](docs/pm/work-queue.md)로 최신 점유를 확인한다. 활성 소유자의 ACK·명시 인계 없이 재배정하지 않는다. 시간 경과는 해제가 아니다. 재귀 위임은 가능하되 파일·자원·실제 슬롯을 겹치지 않는다.
- 외부 발신·이슈 게시·main 직접 push·설정 변경은 현재 사용자 권한을 확인한다. PR 생성·push·merge의 허가도 각각 현재 요청과 담당 범위로 확인한다. 큐 댓글 금지 시 읽기 조회와 로컬 범위 기록으로 제한하고 선점 완료라고 주장하지 않는다.

## 작업별로 필요한 문서만 읽기

| 작업 | 먼저 확인할 정본 |
|---|---|
| 구현·수정 | [구현 원칙](docs/engineering/implementation.md), 변경 지점과 모든 호출자 |
| 테스트·PR·인계 | [작업 절차](docs/pm/workflow.md), [검사 명령](docs/engineering/verification.md) |
| 서비스 | [서비스 상태](docs/pm/service/state.md), 해당 계약·런북, [전시 인계 §2·§5](docs/handoffs/CODEX-resume-2026-10-31.md) |
| 연구 설계 | [연구 상태](docs/pm/research/state.md), [공통 계약](docs/pm/contracts.md), 해당 등록서 |
| 문서 변경 | [문서 규약](docs/maintenance/documentation.md) |
| 도구 간 인계 | [협업 안내](docs/AI_COLLABORATION.md) |

코드 질문은 `graphify-out/graph.json`이 있으면 `graphify query`부터 쓴다. 관계는 `path`, 개념은 `explain`, 넓은 탐색은 wiki/index를 우선한다. dirty 그래프만으로 건너뛰지 않는다. 그래프는 탐색 보조이며 과학적 증거가 아니다. 코드 변경 뒤 `graphify update .`로 AST만 갱신하고 `.graphifyignore`와 semantic-document 보류를 지킨다.

과거 지침은 [고정 원문 목록](docs/maintenance/instruction-history.md)에 보존한다. 현재 지침과 충돌하는 이력을 복원하지 않는다.
