# Pitcheezy 운영 진입점

상위: [프로젝트 안내](../../README.md) · 작업 지침: [AGENTS](../../AGENTS.md)

현재 목표는 **2026-11-24 전시**다. 11/20 기능 동결, 11/21~23 리허설을 목표로 한다. 모니터 1은 UI-1026 발행기 재생, 모니터 2는 849843 경기 영상, 관람객 폰은 QR로 고정 빌드를 연다. 영상 소스와 최종 호스트 경로는 미확정이다.

방향·역할·일정의 근거는 [10/31 인계 §2·§5](../handoffs/CODEX-resume-2026-10-31.md), 기존 빌드의 사실·명령은 [전시 빌드 기록](service/exhibition-build-2026-10-31.md)이다. 저장된 관측과 새 작업의 완료 주장을 구분한다. 열린 PR은 병합·배포 완료가 아니다.

[11/1 PR 통합 기록](../handoffs/CODEX-integration-2026-11-01.md)에서 PR44~64 병합과 검사 결과를 확인한다. [이전 유지보수 인계](../handoffs/CODEX-maintenance-2026-11-01.md)는 병합 전 기록이다.

## 지금 할 일별 진입

| 필요한 일 | 읽을 문서 |
|---|---|
| 시작·소유권·검토·종료 | [작업 절차](workflow.md), [작업큐](work-queue.md), [시작 프롬프트](kickoff.md) |
| 전시 구현·운영 | [전시 인계](../handoffs/CODEX-resume-2026-10-31.md), [서비스 상태](service/state.md) |
| 연구 후보·설계 | [연구 상태](research/state.md), [실험 행렬](../ML_EXPERIMENT_MATRIX.md), 해당 등록서 |
| 통합·인계 | [통합 상태](integration/state.md), [협업 책임](../AI_COLLABORATION.md) |
| 계약·공유 자원 | [공통 계약](contracts.md), [자원](resources.md) |
| 코드·문서 위치 | [프로젝트 지도](project-map.md), [문서 규약](../maintenance/documentation.md) |

각 state·등록·보고서는 자기 관측 날짜와 고정 소스를 기준으로 읽는다. 더 최근 제목만으로 봉인 계약을 덮지 않는다. 자료 사용·실험·정책 승격은 해당 작업의 권한과 검증 게이트를 충족해야 한다.

## 역할과 이력

Song이 방향을 결정하고 Codex가 배정된 구현·검사·PR을 수행한다. 이번 PR 통합은 11/1 Song의 직접 요청으로 Codex가 수행했다. 독립 검토·검사 게이트는 유지했으며 [이전 관리 세션 안내](../handoffs/OPUS-github-manager-prompt-2026-10-31.md)의 담당 지정은 이번 작업에 한해 대체됐다. 실제 활성 소유권은 작업큐에서 확인하며, 문서만으로 실행 중 세션을 추정하지 않는다.

이전 3PM·모델·슬롯 표는 [고정 원문](../maintenance/instruction-history.md)으로 보존한다. 현재 절차와 함께 반복해서 로딩하지 않는다. 새 작업과 결과는 커밋·검사 증거로 연결하고, 과학 결론은 [결정](../decisions.md)·[실험 원장](../experiments.md)·고정 산출물에서 확인한다.
