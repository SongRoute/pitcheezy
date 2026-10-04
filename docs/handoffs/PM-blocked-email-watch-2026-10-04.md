# Pitcheezy 막힘 알림 감시 지침

상태: 2026-10-04 **준비됨, 예약 감시 미등록**. Gmail 프로필/발송 도구는 확인했으나 자동화 생성 도구는 이 세션에 노출되지 않았다. 실행 중인 Opus 감독에는 Gmail 도구가 없다. 이 파일의 존재는 알림 활성화를 뜻하지 않는다. 등록되지 않은 예약 ID·메일 ID·전송 성공을 만들지 않는다.

목적: 개발 진행 중의 일상적인 상태 보고 없이, 사용자의 조치 없이는 더 진행할 수 없는 경우에만 본인에게 이메일로 알린다. 발송은10/4 사용자 명시 요청에 근거한다. 지금은 전체 막힘이 확인되지 않아 메일을 보내지 않았다.

## 로컬 예약 작업에 넣을 실행 지침

작업 위치 `/Users/song/Projects/pitcheezy`, 15분 간격, 같은 대화의 맥락과 연결된 Gmail을 사용할 수 있는 로컬 실행. 맥과 앱이 켜져 있고 프로젝트 접근이 가능해야 한다. 웹 예약 작업이 로컬 경로를 읽을 수 있다고 가정하지 않는다. 연결된 Gmail 수신자는 `me`이며 최초 확인된 주소는 로컬 파일에만 있다.

```text
Pitcheezy의 자율 개발이 사용자 조치 없이는 더 진행할 수 없는지 확인한다.

읽을 자료:
- /Users/song/Projects/pitcheezy/docs/handoffs/PM-state.md
- /Users/song/Projects/pitcheezy/docs/handoffs/PM-destination-2026-10-04.md
- /Users/song/Projects/pitcheezy-pm-jobs/20261004-end-to-end/continuous-state/status.json
- /Users/song/Projects/pitcheezy-worktrees/codex-continuous-pm-1004/CONTINUATION.md
- /Users/song/Projects/pitcheezy-pm-jobs/20261004-continuation/next-work-ledger.md
- /Users/song/Projects/pitcheezy-pm-jobs/20261004-end-to-end/blocker-mail-policy.json
PM-state가 새 감독 경로를 가리키면 그것을 사용한다. 오래된 감독의 의도적 전환 종료를 장애로 오인하지 않는다. 상태 JSON과 실제 PID/시작 시각을 함께 확인하고, 마지막 출력이 오래됐다는 이유만으로 멈췄다고 단정하지 않는다.

정상 진행, 단순 리뷰/계산 대기, 독립 작업을 계속할 수 있는 한 항목의 막힘이면 조용히 종료한다. 승인된 범위의 일상 선택은 기존 PM 위임 아래 해결하고 이메일로 재승인을 요구하지 않는다. 운영 안전 중단, 인증/권한/사용량 실패로 감독 종료, 또는 대안을 점검했는데도 사람의 조치 없이 어떤 승인된 일도 진행할 수 없는 상태라면 알림 대상이다. 예약 한도/기간 종료는 새 감독 또는 승인된 인계가 실제로 없는지 확인한 뒤 '자동 실행 종료'로 구분한다.

메일은 연결된 Gmail의 검증된 본인 주소(me)로만 보낸다. 제목은 '[Pitcheezy] 진행 재개에 확인이 필요합니다'. 본문은 KST 기준 시각, 완료/보호 중인 작업, 막힌 이유, 시도한 해결, 사용자에게 필요한 최소 조치1~3개, PM 상태판 경로를 짧게 담는다. 비밀번호/토큰/개인 원문 로그/연구 데이터는 넣지 않는다. 권한 거절을 우회하거나 보호된 운영 서버를 수리하지 않는다.

동일 원인과 동일 실행의 중복 발송을 피한다. 로컬 mail-alert-log.json에 이미 성공한 메시지 ID가 있으면 다시 보내지 않는다. 새 발송은 recipient+coordinator identity+stop/blocker reason의 키를 먼저 pending으로 기록하고, 전송 성공 응답의 메시지 ID를 받은 뒤 sent로 바꾼다. 타임아웃 등 성공 여부가 불명확하면 자동 재전송하지 말고 해당 발송을 조정 필요로 기록한다. 이전 sent 기록이 있는 같은 사건에는 변경이 없으면 조용히 종료한다.

메일 기능이나 로컬 자료 접근이 없으면 전송했다고 말하지 않고 실패를 기록한다. 이 감시는 외부 계정 설정 변경, 모델 교체, 새 비용 지출이나 운영 교체를 허용하지 않는다. 정기 진행 이메일은 보내지 않는다.
```

등록 후 확인할 것: 실제 예약 ID와 다음 실행 시각, 로컬 파일 접근, Gmail 프로필 일치, 첫 정상 확인이 메일 없이 끝났다는 기록. 사용자가 요청한 조건이 발생하기 전 테스트 메일을 임의로 보내지 않는다.
