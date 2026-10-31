# Codex 유지보수·전시 후속 인계 — 2026-11-01

상위: [운영 안내](../pm/overview.md) · 검사: [통합 검증](../reviews/VALIDATION-maintenance-2026-11-01.md)
전시 산출물과 실측 범위는 [재생·묶음 검증](../pm/service/exhibition-validation-2026-11-01.md)에서 확인한다.

사용자가 조사 기반 계획과 필요한 후속 작업·PR 생성을 승인했다. 원격 main 병합·배포·이슈 발신·설정 변경은 수행하지 않았다. 원래 checkout의 기존 `M runs`를 유지하고 작업별 worktree·브랜치·PR로 분리했다.

## 검토·통합 순서

기존 [PR44](https://github.com/SongRoute/pitcheezy/pull/44)~[PR50](https://github.com/SongRoute/pitcheezy/pull/50)을 먼저 검토한다. 49는48, 50은45에 의존한다. UI 포팅·provider stamp 수정·런북·모델 설계·등록 초안·검토 후속으로 나뉜다. 2026 학습·튜닝·모델 선택과 새 실험은 실행하지 않았다.

| PR | 변경 | 선행 |
|---|---|---|
| [51](https://github.com/SongRoute/pitcheezy/pull/51) | AGENTS 정본, CLAUDE import, 현행 안내·계약 범위 | main |
| [52](https://github.com/SongRoute/pitcheezy/pull/52) | 온보딩·프로젝트 지도 | main |
| [53](https://github.com/SongRoute/pitcheezy/pull/53) | 작업큐·자원 규칙과 과거 기록 분리 | 51 |
| [54](https://github.com/SongRoute/pitcheezy/pull/54) | 분야별 state와 설계·런북 연결 | 52 및47~49 |
| [55](https://github.com/SongRoute/pitcheezy/pull/55) | 모델 설계 주제별 분할 | 48 |
| [56](https://github.com/SongRoute/pitcheezy/pull/56) | 길이·링크·import 검사와 파생 그래프 | 51 |
| [57](https://github.com/SongRoute/pitcheezy/pull/57) | 전시 런북 주제별 분할 | 47 |
| [58](https://github.com/SongRoute/pitcheezy/pull/58) | Zone 내부 함수 공개 범위·주석 정합성 | 50 |
| [59](https://github.com/SongRoute/pitcheezy/pull/59) | 실제 생성 값에 누락된 공개 스키마 타입 | main |

각 PR의 최신 head와 검사를 확인하고 Opus 검토·통합 절차를 따른다. 기반 PR을 squash하거나 병합한 뒤 종속 PR의 base/diff를 확인한다. 로컬 통합 검사는 원격 병합 완료가 아니다.

## 유지보수 규칙

일반 안내는 1,000~2,000자 목표, 상한3,000자, 최소 길이 없음. Markdown 원문 공백·개행을 포함한다. AGENTS가 공통 정본이고 CLAUDE는 `@AGENTS.md`만 import한다. 실제 두 CLI에서 로딩을 확인했지만 모든 도구·장기 효율을 입증한 것은 아니다.

고정 등록·봉인·과거 결정·검증 증거·생성물은 보존한다. 긴 옛 문서를 모두 잘랐다고 주장하지 않는다. 변경한 현행 안내부터 검사하며 원문은 전체 커밋 링크로 추적한다. Markdown이 정본, 링크 그래프와 Graphify는 탐색물이다. 모델·비교군·실험 코드·검증 게이트를 단순화하지 않았다.

영상/VOD와 최종 폰 관전 경로는 사용자 답변이 **미확정**이다. 실제 영상 동기화·QR·Safari·두 모니터/폰 공동 리허설·외부 배포는 후속 조건으로 남는다. 모델 개선은 설계·등록 준비까지이며 결과를 새로 만들지 않았다.
