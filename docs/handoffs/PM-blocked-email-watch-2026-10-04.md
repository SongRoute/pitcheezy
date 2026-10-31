# 막힘 알림 메일 감시기 (2026-10-04)

**현재 상태(10/4 16:21 KST): 작동 중 아님.** 16:11 첫 실제 알림에서 Gmail `send_message`가 **Insufficient scope**를 반환했다. 메시지ID·발송 성공 없음. 감시기53945는 `notify_needs_review`로 종료했고 재시도하지 않았다. 읽기 연결 성공은 보내기 권한 확인이 아니었다. Gmail 보내기 권한을 사용자 연결 화면에서 재승인하기 전에는 알림을 보장하지 않는다. 다른 메일 도구·계정·초안 경로로 우회하지 않는다. 독립 개발·연구는 새 감독56856에서 계속한다.

증거: `/Users/song/Projects/pitcheezy-pm-jobs/20261004-end-to-end/email-watch/jobs/38219d9336d6-0-permission_denial/result.json`, 같은 디렉터리 `events.jsonl`, `email-watch/watch-state.json`. 아래 기동·정상 대기는15:59~16:09의 이력이다.

연속 PM 코디네이터가 사람 손이 필요한 상태로 멈췄을 때만 Song에게 Gmail 한 통을 보내는 로컬 감시기.
`blocker-mail-policy.json`의 `notification_instruction`이 가리키는 문서다. Codex PM이 검토·보완 후 10/4 15:59 KST에 실제 기동했다.

- 코드: `/Users/song/Projects/pitcheezy-worktrees/codex-blocked-email-watch-1004/scripts/pm_email_watch.py` (표준 라이브러리 + `ps`; main `.venv`에 psutil 없음)
- 테스트: `tests/test_pm_email_watch.py` (가짜 프로세스 표·가짜 helper, 실제 Claude·메일 없음)
- 상태: `/Users/song/Projects/pitcheezy-pm-jobs/20261004-end-to-end/email-watch/` (git 밖)
  - `watch-state.json`(신원·후보·로그), `events/<key>.json`(알림 기록), `requests/`, `jobs/`(helper 원본 기록), `logs/`
  - 멈춤: `touch .../email-watch/STOP-EMAIL-WATCH`. 마감 `2026-10-05T23:15+09:00`(상한), 단일 인스턴스 flock

## Gmail 도구 확인 (2026-10-04 구현 세션, 이 워크트리에서 도는 Opus 5.5 CLI 세션)
- ToolSearch로 `mcp__claude_ai_Gmail__send_message` 스키마 확인: `to`/`cc`/`bcc`(문자열 배열), `subject`,
  `body`(평문), `htmlBody`, `draftId`, `replyThreadId`, `replyToMessageId`, `attachments`. 반환은 `id`·`threadId`·`labelIds`가 있는 Message.
- 읽기 전용 연결 확인 `mcp__claude_ai_Gmail__list_labels` 성공: 라벨 13개(SENT·INBOX 등 시스템 + 사용자) 반환. 메일 본문은 읽지 않음.
- send/draft는 호출하지 않았다. 코디네이터 cycle-020 init에도 같은 Gmail 도구가 있다. 알림 helper는 이 워크트리를 cwd로 쓰므로 같은 커넥터를 본다.

## 판정
매 60초: pid 파일, 매니페스트 해시, `status.json`, `ps`(pid + 시작 시각 + 명령에 고정된 `pm_continuous_worker.py run --manifest <v3>`).
- 조용히 계속: running / waiting_next_cycle / 긴 주기. 로그가 오래돼도 그 자체로는 알리지 않는다. 모델 호출 없음.
- 조용히 종료: `stop_file`, `signal_SIGTERM/SIGINT`, `agent_reported_done`.
- 교대(`handoff_detected`, 조용히 종료 → 루트 검토): pid 파일 변경, 매니페스트 변경, status가 다른 매니페스트, 새 `pm_continuous_worker.py run` 프로세스 등장.
- 즉시 알림(같은 관측 2회 연속): `health_failed`, `auth`, `model_mismatch`, `permission_denial`, `consecutive_failures`, `agent_reported_blocked`, `coordinator_error*`, 확인된 코디네이터 프로세스가 사라졌는데 status가 `stopped`가 아님(pid 재사용은 시작 시각으로 걸러냄).
- 유예 후 알림(`--grace` 180초 + 교대 없음): `deadline`, `max_cycles`, 그 밖의 정지 사유(핀 불일치, helper_overrun). 메일은 "정해진 실행 한도 도달, 과학적 실패 아님"이라고 쓰게 한다.
- status.json을 못 읽으면(쓰기 경합) 그 관측은 세지도 초기화하지도 않는다.

## 알림 한 번
1. `events/<key>.json`을 배타 생성(fsync)으로 `pending` 기록 → 다시 관측·교대 확인. 바뀌었으면 `cleared_before_send`, helper 호출 없음.
2. 고정 helper(`run_claude_task.py`, SHA256 `03009626…b5f7`)를 한 번: `claude-opus-5-5`, effort medium, 180초, 16턴, `--skip-permissions --disable-hooks`(D68), cwd = 깨끗한 이 워크트리(코디네이터 워크트리 아님).
3. 요청: 로컬 상태 재확인(교대·진행 중이면 `EMAIL_SKIPPED`) → `<로컬 설정의 본인 이메일>`에게만 한 번, 제목 `[Pitcheezy] 진행 재개에 확인이 필요합니다`, 한국어 짧은 본문(KST 시각, 이유·근거, 끝난/보존된 작업, 기록된 해결 시도, 할 일 1~3, PM 상태 경로). 재시도·초안·답장·전달·다른 계정 금지.
4. 판정은 helper 원본 `events.jsonl`에서: send_message 도구 결과의 실제 메시지 id가 있어야 `sent`. 도구 결과 없음·오류·id 없음·수신자/제목 이상·메일 도구 2회 이상 → `needs_review`(`delivery_confirmed: false`). 보내지 않고 실패 → `failed_no_send`. 에이전트 판단 생략 → `skipped_by_agent`.
5. 어떤 결과든 감시기는 끝난다. 재시작해도 같은 코디네이터 신원에 기록이 있으면 `already_handled`로 끝나고 다시 부르지 않는다. 재시작 시 `pending`은 helper 기록으로 마무리하거나 `needs_review`로 표시한다.

## 실행 설정

수신 주소는 소스 상수 대신 필수 `--recipient`로 받는다. root는 git 밖 `blocker-mail-policy.json`의 검증된 본인 주소를 읽어 argv 배열로 전달한다. 셸 문자열 결합이나 자격 증명 추출을 하지 않는다. 관측 전용은 `observe`, 지속 감시는 `run`; 공통 필수 인수는 `--status`, `--pid-file`, `--manifest`, `--worker`, `--helper`, `--helper-sha256`, `--notify-worktree`, `--watch-dir`, `--deadline`, `--recipient`다. 모든 경로는 절대 경로다.

root의 검토 후 실제 실행 argv·소스 해시·PID는 외부 `email-watch-launch.json`에 기록한다. 먼저 `observe`를 실행해 현재 감독의 PID/시작시각/명령/manifest가 일치하고 healthy인지 확인한다. 기본 유예180초+helper상한180초는 감독23:00 종료 뒤 감시기23:15 종료 전에 처리할 수 있다. 수신 주소·모델·helper가 바뀌는 변경은 감시 중에 하지 않는다. 실행 중에는 이 작업트리를 깨끗하게 둔다.

## 한계
- 맥미니가 켜져 있고 잠자지 않아야 한다. 꺼져 있거나 오프라인이면 알림 보장 없음.
- Claude CLI 인증·크레딧과 Gmail 커넥터가 살아 있어야 한다. 코디네이터를 멈춘 원인이 같은 인증 장애(`auth`)면 알림 호출도 실패할 수 있다 → `failed_no_send`로 로컬에만 남는다.
- 같은 코디네이터 신원당 최대 한 번. 첫 알림 뒤 새 문제는 루트가 감시기를 다시 띄워야 본다.
- 실행 중 helper가 메일을 보낸 직후 감시기가 죽으면 재시작 시 helper 기록으로 확인하지만, 도구 결과가 기록되기 전이면 `needs_review`로만 남는다(재전송하지 않음).
- 코디네이터가 다른 pid 파일·다른 매니페스트로 교체되면 감시기는 교대로 보고 끝난다. 새 코디네이터는 새 감시기로 본다.

## 실제 실행 확인

- 소스: 작성62e7b76, Codex 검토·수정1fcfdfc. 수신주소를 소스 상수에서 외부 설정으로 분리했고, 마감 유예를180초로 바꿔23:15감시 종료 전에 알림 helper가 끝날 시간을 확보했다. 관련 회귀 검사 추가,25 passed(1.49s), AST갱신 완료.
- PID53945,10/4 15:59시작, 상태 watching.16:00재확인에서도 healthy:running, 후보없음, 알림 이벤트0/모델호출0/메일0. 단위 검사는 가짜메일로만 했고 실제 발송 시험은 하지 않았다.
- 감독27867의 PID·시작시각·실제 명령·manifest를 관측 전용 검사로 확인한 뒤 연결했다. 현재 감독 v3에 한정하고 자동 교대하면 재연결이 필요하다.
- 실행기록: `/Users/song/Projects/pitcheezy-pm-jobs/20261004-end-to-end/email-watch-launch.json`, 상태: `email-watch/watch-state.json`, 수신 설정: `blocker-mail-policy.json`. 실행기록·설정은 git밖에 있고 개인 수신주소를 포함하는 파일은0600이다.
- 이 감시기는 로컬 조건 감시 프로세스다. 일반 예약 작업 생성 도구가 없어 앱의 예약 기능을 등록했다고 표현하지 않는다. 일반 예약 기능과 실행 조건은 [공식 안내](https://learn.chatgpt.com/docs/automations?surface=app)를 참고한다.
