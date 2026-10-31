# 2026-11-24 전시 런북

상태: 실행 절차 초안. 이 문서 작성으로 배포·설정 변경·실자료 재생을 실행하지 않았다. 기준 코드 main `d04afab1`, 고정 빌드 `fba6b2ff`. 근거는 [10/31 빌드 기록](exhibition-build-2026-10-31.md) §2·§6과 Song이 전달한 10/31 인계 §2·§5다. 기능 동결 11/20, 리허설 11/21~23, 전시 11/24. 11/11 저녁 공동 연결시험은 **제안 일정이며 확정 전**이다.

## 1. 화면·역할·정직 표기

| 자리 | 보여 줄 것 | 담당·확인 |
|---|---|---|
| 모니터 1 | 849843 CHC@SD, 발행기 재생 → v1 → UI-1026 공개 빌드. 추천과 미지원 구간 참고 카드 | Song: 서비스·모델·발행·결정 |
| 모니터 2 | 같은 경기 풀 중계 우선. 확보되지 않으면 MLB 공식 클립 체인 | 현준: 영상·클립·호스트, 사용 가능한 소스 확정 전 |
| 관람객 폰 | QR → pitcheezy.com의 팀원 Cloudflare 호스트에 올린 고정 빌드 | 현준: URL·경로·호스팅 / Song: 전달 빌드 / 강민: QR·발표물 |
| 발표물 | UI-1026 캡처·근거가 붙은 수치·설명 | 강민. 포스터 구조는 유지하고 동결 뒤 한 번에 정정 |

이것은 **완료 경기의 재생 시연**이다. LIVE 표시나 재생 시계는 실제 경기 생중계·실수신 시각을 보증하지 않는다. 전시 G0는 동결하며 연구 후보와 교체하지 않는다. 참고 카드는 추천이 아니다. 모델 성능·승리 개선을 이 시연의 성공 횟수로 주장하지 않는다. 폰의 고정 최종 경기 파일은 모니터 1의 움직이는 시점과 자동 동기화되지 않는다.

## 2. 고정 자산과 실행 전 확인

- 경기 849843: 현지 2026-09-29, KST 9/30. 빌드 기록상 262구, ready 246구, 미지원 16구. 이는 **최종 archive 파일의 집계**다.
- 맥미니 작업 폴더: `~/Projects/pitcheezy-pm-jobs/exhibition-849843-1031/` (아래 `task_assets`). 원본 `out4/v1`, `profiles/LATEST.json`, ZIP, 기존 stamped 재생 폴더를 덮어쓰지 않는다.
- 원본 ZIP: `bundle-849843-fba6b2ff.zip`, 867,682 bytes, SHA-256 `af10293d5134219b7f750d00786366b93b0134c895519b939fd99a356d77a4bc`. 새 코드로 만든 묶음에는 새 커밋·파일 목록·SHA를 기록한다. 이 SHA를 재사용하지 않는다.
- T7 읽기 경로: `/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/DEMO-WS-2026/{feeds,watch,intent}`. 연결 여부와 `849843_final.json.gz` 존재를 확인한다. 누락 시 수집·fit을 시작하지 않고 고정 묶음으로 전환한다.
- 참고 카드 포인터: `profiles/LATEST.json`, 대상 9/29·컷오프 9/28, profile SHA `f64abb8511fa071ebcd3561e7634a76ead8fbaf8e3db4cda2f5f78178016b7f3`. 최신 날짜 프로필로 바꾸지 않는다.
- 실행할 worktree의 커밋·변경 여부, 사용 Python/Node, 시작 투구 순번, 출력 폴더, store 경로, 재생 poll/발행 interval을 회차 기록에 적는다. 다른 작업자의 기록기·GPU·운영 포트·프로세스는 건드리지 않는다.

## 3. 모니터 1 시작

[입력·타임스탬프 준비](exhibition-replay-setup.md) → [발행기·로컬 서버](exhibition-replay-publisher.md) → [화면·속도·종료 확인](exhibition-displays.md) 순서로 진행한다. 셸 변수는 같은 터미널에서 이어 쓰며 별도 터미널에는 실제 회차 경로를 지정한다.

<a id="31-회차-준비와-재생-파일-생성"></a>
<a id="32-타임스탬프-호환성"></a>
입력 준비의 옛 §3.1·§3.2 링크는 [준비 안내](exhibition-replay-setup.md)로 이어진다.

<a id="33-발행기-실행"></a>
발행기 옛 §3.3은 [실행 안내](exhibition-replay-publisher.md)에 있다.

<a id="34-화면-확인속도종료"></a>
화면 옛 §3.4는 [확인 안내](exhibition-displays.md)에 있다.

## 4. 모니터 2와 폰 QR

[영상·QR 확인](exhibition-displays.md)을 따른다. 영상 소스와 최종 관전 URL은 미확정이며 예시 URL로 완료 처리하지 않는다.

## 5. 장애 대응

[증상별 복구](exhibition-rehearsal.md)에서 새 store 재시작과 archive 전환을 확인한다. 기존 원본과 다른 프로세스를 보호한다.

## 6. 리허설 체크리스트·회차 기록

[리허설 체크리스트](exhibition-rehearsal.md)에 실제 결과·미측정·회차 식별자를 남긴다.

분리 전 원문은 [고정 커밋](https://github.com/SongRoute/pitcheezy/blob/172d73f9449a723cc247a142ca871abcad34f043/docs/pm/service/exhibition-runbook-2026-11-24.md)에 보존한다. 이 문서 분리는 재생·배포 실행 결과가 아니다.
