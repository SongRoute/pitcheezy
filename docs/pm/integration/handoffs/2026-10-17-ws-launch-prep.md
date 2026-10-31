# 인수인계 — 월드시리즈 출시 준비 (2026-10-17, Claude PM)

계획과 Song 결정: [출시 준비 계획](../ws-launch-readiness-plan-2026-10-17.md). 운영 안내: [WS 운영 런북](../../service/ws-operations-runbook.md). 결정 줄: `docs/decisions.md` 2026-10-17 줄들.

## 한눈에
- 출시 브랜치 `ws/launch-2026` = `5c9c31a` + B·C·D·E, 끝 커밋 `a9d52c4`(이 인수인계 문서 커밋 전 기준). 푸시하지 않았다.
- B(확인 중 규칙)·C(안 바뀐 파일 건너뛰기)는 849831 오프라인 재생으로 전/후 비교를 마쳤고 독립 검토 PASS다. 실경기·실제 호스트로는 아직 돌리지 않았다.
- 연구 합치기는 빼 두었다(아래 "보류").
- 10/22 공개/보류 판단 전에 남은 일: 실경기 리허설 E, 호스트·실기기 F, 연락 채널, Song 화면 점검.

## 오늘 한 일

### 1. 이전 Codex 차선 마감 (10/16 밤 ~ 10/17 04:30 KST)
작업 폴더 `pitcheezy-pm-jobs/20261016-autonomous-profile-stream-005`, 상태 CLOSED. 둘 다 2026 EXPOSED DEVELOPMENT(이미 본 적 있는 개발용 자료) 회고 진단이며 전향·우위·운영 주장이 아니다.
- **투수 식별 보존 진단(채점 없음):** 고정 6경기(849812, 849820, 849813, 849811, 849810, 849818)에서 품질 통과 1,659행 중 기존 1,323행 준비·336행 보류를 재현했다. 결과 라벨을 못 읽은 행에서도 검증된 투수 ID를 남기면 보류가 336 → 22로 줄고 새로 채점 가능한 행이 314개다. 자격만 확인했고 314행 채점은 새 등록이 필요하다. Root 수용 `ROOT-POSTCUTOFF-UNKNOWN-IDENTITY-RESULT-ACCEPTANCE.json`. 실행은 GRANT-v2(편차 기록 `DEVIATION-GRANT-PATH-001.json`).
- **Step5 FF 보조 비교(서술용):** v3 검토 FAIL, v4·v5는 키 불일치로 fail-closed 정지 → 버그 수정·재검토 뒤 v6(attempt-003) 완료. 실제 포심(FF) 560개 중 T 보류 100개를 뺀 짝지은 460행에서 NLL T 1.6147 vs Step5 1.6234, Brier 0.7703 vs 0.7718. 경기 단위 부트스트랩 95% CI(NLL 차이) [−0.0214, +0.0075]로 0을 포함한다(차이가 있다고 말할 수 없음). 6경기 중 4경기 T, 2경기 Step5가 낫다. 학습 시점이 달라(T 10/11, Step5 9/27 프로필) 구조 비교가 아니다. Root 수용 `ROOT-STEP5-SIX-FF-RESULT-ACCEPTANCE.json`.

### 2. 출시 브랜치 `ws/launch-2026` (작업 폴더 `pitcheezy-pm-jobs/20261017-ws-launch-prep`)
합친 순서: `a6d6772`(B) → `ef3b00e`(C) → `f420c83`(테스트 보정) → `383cd31`(D) → `a9d52c4`(E).

- **B 확인 중 규칙** (`a52012e`, `04aa25b`, `c0d3d94`, `75e2518`): 받아들인 공은 마지막 수락 내용 유지, 같은 시각 충돌 정지는 정책 입력이 바뀔 때만, 정정 표시는 보이는 칸이 바뀔 때만, 현재 타석 입력 미확인이면 추천 보류, 종료 본문이 보류 상태를 잠금.
  - 849831 재생 4회(off/on × 전/후), 모두 exit 0, 불변식 유지.
  - 확인 중 사이클 12 → 1, 게이트 사이클 166 → 21, 최종 파일 정정 표시 257 → 3, 받아들이지 않은 내용 노출 0.
  - M5(현재 타석 입력이 확인 안 됐는데 추천 발행): 등록 분석 v2에서는 on 6건으로 "미통과"였다. 원인은 분석 스크립트의 시간 반올림 버그(발행기가 실제로 내보낸 스냅샷이 아닌 바로 앞 스냅샷을 검사)였다. 고친 분석 v3를 새로 등록·검토(REVIEW-003, 조건 R1)한 뒤 기존 실행 폴더에서 분석만 다시 돌려 4회 모두 M5 = 0, R1 충족. v2 판정 기록은 RESULT.md에 그대로 남겼다.
  - 정책 계산 중앙값 0.1225 → 0.1221초, 최대 6.71 → 6.94초, peak RSS on 2.05 → 2.00 GB.
  - 검토: REVIEW-001(조건 C1: 결정 키에 점수 등 정책 입력 추가) → `75e2518`로 닫음(REVIEW-002) → REVIEW-003(분석 v3). 모두 PASS.
  - **배포는 경기 사이에만**(결정 ID가 모두 바뀜, REVIEW-002 N1).
- **C 안 바뀐 파일 건너뛰기** (`2602d8d`, 설정 `skip_unchanged_live` 기본 켬): 849831 재생 전/후, 경기 파일 쓰기 1174 → 359, 경기 파일 바이트 277.8 MB → 104.0 MB, 최종 파일 해시 동일, health는 1177주기 모두 기록. C1–C7 통과, REVIEW-001 PASS.
- **D 출시 화면 결함** (`ed2edb2`): 머리글 겹침, 박스스코어 합계, 정정 표시, "확인 중" 상태 줄. ws-launch-layout-check 178/0, display-state-smoke 1854/0, REVIEW-001 PASS.
- **E 웹 검사 정리** (`0fbbfc7`, `5cdeeff`): 실패 원인이 모두 10/13 승인 개편(`0fe66af`)이었다(개편 직전에는 실제 실패 0). 옛 배치를 가정한 검사를 새 배치로 옮기고, 카드 설명 글 넘침(22건)은 실제 결함이라 말줄임으로 고쳤다.
  - static-publish 358/21 → 385/0, viewer-delay 중단 → 105/0, boxscore-asof 중단 → 6/0, compact 344/244 → 776/4, studio 8 실패+중단 → 238/3.
  - 남은 7건은 모두 390×844에서 투구 띠가 탭 막대 밑으로 3.5–20px 들어가는 한 문제. Song 화면 점검까지 알려진 실패로 둔다. REVIEW-001 PASS.
- **브랜치 끝(`a9d52c4`)에서 전체 백엔드 테스트·웹 검사를 한 번에 다시 돌린 기록: 미확인.** 각 작업 폴더에서는 돌렸지만, 합친 뒤 전체 재실행 기록은 찾지 못했다.

### 3. 보류: 연구 합치기
`93a80ef`(연구)를 합친 커밋은 `ws/launch-research-merge-1017`(`cc0d174`)에 보관했다. 충돌은 `docs/decisions.md`뿐이었지만, 연구 쪽 D117–D127이 `configs/ML-POLICY-MATERIALIZATION-v1.json`과 `policy_runtime.py`를 바꿔 실시간 ARM-B 추천(`demo_precompute.LivePolicy`)이 시작 때 d107 addendum 사슬·S2 식별자 검사에서 거절된다(`test_demo_precompute` 실패). 출시 뒤 실시간용 고정 사본을 따로 두는 식으로 정리한다.

## 남은 일
1. 합친 브랜치 끝에서 전체 백엔드 테스트·웹 빌드·웹 검사 재실행(위 미확인 해소).
2. 공개 화면 e2e(`test_public_viewer_e2e`, `test_public_viewer_gateway_e2e`): C 작업 폴더에 playwright가 없어 못 돌렸다. 웹 node_modules 있는 곳에서 출시 전에 돌린다.
3. 리허설 E(남은 LCS 경기 한 판, 로컬 발행만): B·C 실경기 확인, 지연·복구, 실제 주기·흔들림 측정 → 런북의 잠정 기준(30초, 45초 등)을 고친다. 10/18 이후 일정은 당일 다시 확인.
4. F: 실제 호스트 연결(전용 경로, 시험 키 발급·폐기, 원격 경로당 발행기 하나), iPhone Safari 실기기 확인.
5. 런북 미확인 칸 채우기: 실경기 실행 명령·환경 변수 묶음, 발행기 실행 위치(tmux/launchd), 즉시 내림 스위치 위치·담당, 연락 채널.
6. 작은 후속(출시 필수 아님): REVIEW-002 N2(결정 키가 매번 경기 전체를 직렬화 — 긴 경기 비용 미측정), N3(경기 상태를 키에 추가), `publish.ts` publishLag 설명 낡음, `tests/service-ui-smoke.mjs` 낡은 선택자, 결과 라벨 없을 때 aria-label 빈칸.
7. 남겨 둔 작업트리 정리(PM 몫): `ws-launch-b-replay-before`, `ws-launch-b-replay-after`, `ws-launch-c-1017-before` 등. 지우기 전에 확인.

## Song 결정 대기
1. MLB 허가 메일과 지도교수 문의 보내기(초안 `docs/pm/service/mlb-permission-request-draft.md`, 빈칸: 이름·학교·연락처·불허 시 계획).
2. 팀원 연락 채널(10/20까지)과 비공개 전달 채널·시험 시간.
3. 공통 지연 0초 또는 15초(추천 15초).
4. 화면 점검: 390×844 투구 띠가 탭 막대 밑으로 들어가는 7건을 고칠지(고치려면 경기장 높이 등을 줄여야 해 다른 보장에 영향), 카드 설명 말줄임(320 폭에서 결과가 거의 안 보임)을 그대로 둘지. 폰의 "다음 타자" 줄 없음은 의도로 가정했다(Song 미확인).
5. 리허설 E 승인.
6. (계획에서 이어짐) 지원하지 않는 투수 참고 카드 켤지, 원격 `rc/ws-2026` 푸시 여부.

## 지키는 것
- 실행 중인 서버·기록기·수집 프로세스에는 신호를 보내지 않는다. 메모리는 기록기가 먼저다.
- 실자료 실행은 등록 + 작성자와 다른 에이전트의 독립 검토 뒤에만.
- 2026 자료 역할(EXPOSED DEVELOPMENT 등)·as-of·누수 방지, 봉인 결과·OPE 경계 유지.

## Added 2026-10-18
- Launch branch `ws/launch-2026`: on top of B/C/D/E it now carries J (reference card), the operations launcher and alert watcher (`apps/observer/ops/`), robust daily start (retry, quiet off-days, schedule shape validation), and the WS game list pinned in the watcher. Every piece passed independent review; 110 tests pass.
- Rehearsal: 849816 (10/18) refused at the prestart check because the PM session's pgrep waiter loop was misread as another publisher (no feed polled). Fixed (REGISTRATION-v4, REVIEW-003 PASS); retried on 849817 (10/19 06:08 KST, publisher at 05:53).
- Reference card replay (J): card on 47 / off 0, body identical with the card excluded, 0 audit violations → merged.
- Teammate: host pack (3ea7ad6) verified, Node 304/304, compatibility 4/4. Reply posted on transition-models#18. Our pack v4.2 sent privately by Song. One-key test registration v2 passed review (Fable) → waiting on the teammate's answers to 12 questions and a test window.
- Song decisions: MLB permission not a gate (advised); UI pass on 10/21 (freeze exception); restart up to 3×, start 30 min before; phone alerts via ntfy+iMessage (**Song sets up Tue 10/20 afternoon/evening**); launchd daily start once the WS schedule is confirmed.
- Open: 8772 switch after the rehearsal (Song), T7 move (~21GB, after the rehearsal), WS roster recount, confirmed WS times, pack v4.3 (adds reference card schema).
