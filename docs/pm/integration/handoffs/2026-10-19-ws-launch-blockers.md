# 출시 블로커 6항목 처리 인계 (2026-10-19, Claude PM claude-pm-20261017)

근거: 외부 평가 docs/reviews/external-evaluation-2026-10-19.md §5 S-4, §7-A (평가 문서는 수정하지 않음). 큐: SongRoute/pitcheezy#37 CLAIM BLK-01~06 (6258768043).
증거 폴더(로컬): pitcheezy-pm-jobs/20261019-blockers/

## 상태 한눈에
- BLK-01 원격 백업 — **Song 결정 대기**. ws/launch-2026은 원격에 없음, codex/ml-matrix-execution 97·rc/ws-2026 24커밋 앞섬, origin은 공개 저장소. 추천: 비공개 저장소 새로 만들어 push(push 전 비밀 스캔). 증거 BLK-01-BACKUP.md (fb229ee4…)
- BLK-02 메모리 — **Song 결정 대기(10/20 08:53 전)**. 여유 7.68 GiB(12:10). E2는 `--armb auto`(T-15에 ≥8 GiB면 ARM-B 켬). 08:30 데모 중지 뒤 예상 약 8.5 GiB. 불필요 pitcheezy 서버 7개 합계 약 0.18 GB, 큰 것은 Song 앱(Safari 탭 1.4 GB, Claude 세션들, openclaw). 추천: 서버 7개 내림 + Song 앱 정리. ARM-B 꺼진 채 통과한 E2는 출시 근거로 쓰지 않음. 증거 BLK-02-MEMORY.md (a5af2d19…)
- BLK-03 지연 — **Song 결정 대기**. 외부 평가 수치 검산 일치(849817 p50 19.2/p90 26.2/max 33.5 s, n 260; 849809 130/179). 15초: 15초 이하 지연 시청자 스포일러 0, 실시간 추천 도달 849809 130→123/179, 30초+ 지연 스포일러는 거의 그대로. 추천: 15초 + 1차전 뒤 재검토. 변경 위치 apps/observer/ops/ws_ops_config.json:16. 증거 BLK-03-DELAY.md (fd50fc67…)
- BLK-04 팀원 질문 — **끝남(초안)**, 발신은 Song. 시험 창·즉시 내림 담당·비상 연락 한 메시지. 증거 BLK-04-TEAMMATE-MESSAGE.md (4fa4de7f…). GATE-05 규칙은 Song 결정 완료(docs/decisions.md 10/19 줄).
- BLK-05 공개 문구 — **끝남(제안)**, 적용은 Song 10/21 UI. 한계 문구는 Sheets.tsx:41에 있으나 탭해야 보임. 상시 문구·"중계보다 먼저 보일 수 있음" 고지 없음. 증거 BLK-05-WORDING.md (9f287869…)
- BLK-06 테스트 — **영수증 끝남, 골든은 Song 결정 대기**. 후보 505932e6: backend+ops 19 failed / 1213 passed / 1 xfailed; web static 385/0, viewer-delay 105/0, compact 776/4, studio 238/3, 공개 E2E 5/0. 골든 14건은 win_expectancy만이 아니라 의도된 변경 a52012eb(B 확인 완화)·75e25189(결정 키)까지 435차이. 추천: 현재 트리로 기준 재생성. 증거 BLK-06-TEST-RECEIPT.md (66a46450…), 검토 BLK-06-REVIEW-001.json PASS.

## 같은 날 추가 증거
- CL-13 최종 후보 오프라인 재생(a9f18c1c, 849831): 끔·켬 모두 기준 통과, 결과 검증 CONFIRMED_PASS. 실제 폴링·일정 장애·업로드·감독기는 미검증. pitcheezy-pm-jobs/20261019-cl13-final-replay/RESULT.md
- GATE-05 결정 초안 v2.1 + Song 결정(pitcheezy-pm-jobs/20261019-gate05-draft/).
- 10/22 판단 묶음 v2.1(pitcheezy-pm-jobs/20261019-gate03-readiness/), E2 결과 칸은 10/20 뒤 채움.

## 다음 세션이 할 일
- Song 결정 반영: 백업 push, 서버 내림, fixed_delay_s 변경 커밋, 골든 재생성(별도 브랜치 ws/golden-1019 → 병합 결정), 노출 장부 한 줄(CL-13).
- 10/20 E2 분석: pitcheezy-pm-jobs/20261017-ws-launch-prep/F-rehearsal/E2-ANALYSIS-PREP.md 순서. PRESTART.json에서 ARM-B 결정 확인.

## 갱신 — Song 결정 반영 (2026-10-19 오후, "모두 추천안으로 가자")
- BLK-01 **끝남**: 비공개 저장소 SongRoute/pitcheezy-private 생성, ws/launch-2026·codex/ml-matrix-execution·rc/ws-2026 push(비밀 패턴 검사 0건). 로컬 remote 이름 `private`. 출시 브랜치는 이후 커밋도 계속 이 저장소에 올림(현재 d6ba9a3f).
- BLK-02 **끝남(Mac 쪽)**: pitcheezy 옛 서버 7개(8766×2, 8767, 8771, 8770, pytest 잔재 2)를 PID·시작 시각·포트 확인 뒤 SIGTERM, 모두 종료 확인. 여유 메모리 9.73 GiB(기준 8 GiB). 8772 데모 유지. Song 앱 정리는 Song 몫. E2 PRESTART.json에서 ARM-B 결정 확인 필요.
- BLK-03 **끝남**: fixed_delay_s 15 (ws/launch-2026 d98b2010, decisions.md 줄). 운영 테스트 78 통과, 덮어쓰는 환경 변수 없음. WS 1차전 뒤 재검토. 공개 화면 스포일러 문구(15초)는 10/21 UI.
- BLK-04 **끝남(초안)**: 발신은 Song.
- BLK-05 **끝남(제안)**: 적용은 Song 10/21 UI.
- BLK-06 **끝남**: 골든을 505932e 트리로 재생성(ws/golden-1019 e6108595·11cbe01e → ws/launch-2026 d6ba9a3f 병합). 원인 귀속 0fe66af9 363 / a52012eb 64 / 75e25189 16(+하네스 2). golden 15 passed, backend+ops 5 failed(자료 의존 4·환경 1) / 1227 passed / 1 xfailed. 독립 검토 PASS_WITH_NOTES (BLK-06b-REVIEW-001.json). web은 BLK-06 영수증 값 그대로(compact 776/4·studio 238/3은 알려진 390×844 b2 레이아웃).
- 노출 장부: CL-13 줄을 docs/pm/service/state.md에 추가(미커밋, Song 승인).
