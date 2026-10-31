# 유지보수·기능 통합 검증 — 2026-11-01

상위: [종료 인계](../handoffs/CODEX-maintenance-2026-11-01.md)

기준 main은 `6e2038fd0010657e6f6f2d1e49820490f6195da4`다. 실행 증거는 로컬 작업 기록 `codex-maintenance-20261101/integration/`에 명령·종료 코드·원본 로그로 보존한다. 별도 에이전트의 검토이며 다른 모델 검토나 원격 병합을 뜻하지 않는다.

## 기능 PR44~50

고정 로컬 통합 커밋 `8e399a0be776c3f57d9b496bcc0ab15dc2d0a212`에서 검사했다. 재현 순서는 위 main에 `89b912c6`, `790f1fbe`, `e59b0650`, `172d73f9`, `2a3d4d23`, `9f7df2c4`, `5e44e69e`를 순차 통합한다. 기존 작업 공간을 덮지 않는다.

웹 디렉터리에서 `npm run build`, `npm run build:public` 성공. 아래는 `node tests/<이름>.mjs` 결과다.

| 검사 | 통과 | 실패·제외 |
|---|---:|---|
| static-publish-smoke | 416 | 실패0 |
| ws-gaps-smoke | 15 | 시리즈 헤더 미포팅2 skip |
| ws-moments-smoke | 40 | 실패0 |
| notice-hide-smoke | 267 | 실패0 |
| ws-launch-layout-check | 360 | 실패0 |
| zone-geometry-check | 397 | 실패0 |
| zone-label-clip-check | 160 | 실패0 |
| ui-review-followups-check | 52 | 실패0 |

루트 `PYTHONPATH=src:.:apps/observer/backend pytest apps/observer/ops/tests -q`: **128 passed**. 전체 루트 테스트를 모두 실행했다고 주장하지 않는다. PR58의 내부 export·주석 변경은 별도 두 빌드와 geometry397 통과로 확인했다.

## 문서 PR51~58

최종 로컬 통합 `e845e76d733ee95fb264adcc6e85eb9d9937d25b`에서 유지보수 변경 **36문서·189상대링크**, 3,000자 상한·정확한 CLAUDE import·diff 검사 PASS. `tests/test_check_docs.py`는 **4 passed**. AGENTS에서 분야별 state를 거쳐 변경36문서에 모두 도달한다. 고정 SHA 원문 참조21개와 변경 링크의 fragment를 별도 대조해 오류0을 확인했다.

main 대비 전체 변경38문서·195링크 검사에는 원본 보존 예외2건이 있다: 독립 검토 보고7,590자와 B2/B3 등록 초안13,822자. 바이트는 기능 통합 소스와 같다. 작업 기록의 `length-exceptions.json`을 `check(root, paths, exceptions)`에 전달해 검증했으며 저장소 예외 설정은 추가하지 않았다. 따라서 기본 CLI의 전체 기준 검사는 이 두 길이를 계속 보고한다.

초기 통합에서 발견한 런북·모델 설계 진입 링크 누락, 공통 계약의 현재 범위 혼선, 파일 끝 빈줄은 수정 후 재검사했다. 실제 Codex·Claude 로딩 확인은 PR51의 별도 근거 문서에 있다. 장기 토큰·속도·성공률 개선은 미측정이다.

전체 그래프의 미존재 대상76건은 변경 범위 밖 **검토 후보**다. 오탐·과거 소스 참조가 포함될 수 있어 실제 깨진 링크76건으로 확정하지 않는다. HTML·축약 reference·외부 HTTP의 완전 검사, 실호스트·실폰·영상 동기화는 미측정이다.

## 후속 공개 계약 수정

[PR59](https://github.com/SongRoute/pitcheezy/pull/59)는 묶음 생성에서 발견한 스키마 누락을 수정한다. 기존 생성 코드에 맞춰 6개 경로의 타입만 보완하고 3개 스키마를 재생성했다. `test_publisher.py`·`test_setup_estimates.py`는 작성자와 독립 검토자가 각각 **99 passed**를 확인했다. 합성 회귀는 위치 없음/미보정/위치 있음·문자열 구장을 생성하고 잘못된 타입을 거부한다. 이전 실패에서 보존한 공개 JSON6개를 읽기 전용 검증한 스키마 오류는61→0이다. 기존 실패를 성공으로 바꾸거나 이 수정으로 전체 모델 재생을 반복하지 않았다.
