# 세션 종료 인계 — 출시 후보(rc/ws-2026)·연구 차선 (2026-10-11 ~ 10-12 16:50 KST)

작성: Claude PM 세션(단일 PM, 2026-10-11 Song 지정). 같은 기간의 협업·공개 발행 차선은 [2026-10-12-collab-ws-public-launch.md](2026-10-12-collab-ws-public-launch.md)가 따로 인계한다. 이 문서는 출시 후보 서버, 실경기 시험 운영, 연구(EXP-WS-*), 그림자 실행, 프로필 갱신만 다룬다.

숫자는 모두 아래 증거 경로의 산출물에서 옮겼다. 2026 자료는 EXPOSED DEVELOPMENT이며 채택·우열 주장은 없다.

## 1. 지금 상태 한눈에

- `rc/ws-2026` 로컬 끝 `9f4b36c` (작업 폴더 `pitcheezy-worktrees/rc-ws-2026`, 깨끗함). **원격 `origin/rc/ws-2026`은 여전히 `e39bb41`** (시청자별 지연이 켜진 판). push는 Song 결정 대기.
- `codex/ml-matrix-execution` 로컬 끝 `e25978a` (다른 세션 커밋 포함). 이 세션의 마지막 커밋은 `cdd3c71`.
- `feat/profile-refresh` 끝 `d27cc9d` (예약 작업이 이 커밋에 고정돼 있다. 이 작업 폴더를 옮기거나 더럽히면 예약이 거부한다).
- 돌고 있는 것:
  - 운영 서버 8771(PID 99202), 기록기 8766(PID 94628, 94642): **건드리지 않는다.**
  - 폰 시연 `http://100.108.252.111:8772/watch/849831` (849831 재생, 내부 경로라 시청 지연·참고 카드 켬). 끄기: `sh /Users/song/Projects/pitcheezy-pm-jobs/20261011-rc-features-001/demo/stop_demo.sh`. 시연이 쓰는 화면 빌드는 `3e076fc` 기준이고 백엔드는 작업 폴더의 현재 코드다.
  - launchd `com.pitcheezy.profile-refresh` 매일 21:00 KST (2026-10-12 Song 승인). 운영 메모 `/Users/song/Projects/pitcheezy-pm-jobs/profile-refresh-daily/OPERATIONS.md`.
- 세션 예약(cron)은 세션과 함께 사라졌다. 다음 시험 운영은 새 세션이 직접 예약한다.
- GPU 서버: 실행 중인 잡 없음.

## 2. 이 세션에서 끝낸 것

### 출시 후보
- 세 스위치(`viewer_delay`, `profile_reference`, `step5_expectation`) 기본 꺼짐 (`f9a04f2`, D176). 공개 빌드(`VITE_PUBLISH_BASE`)는 지연 UI를 그리지 않는다.
- 화면 검사 정리: `zone-geometry-check`는 낡은 검사였음(`613e8b3`), 낮은 폰에서 투구 카드의 높은 공 표시 잘림은 실제 결함이라 수정(`7789277`, `3e076fc`), `compact-rev3-fit`·`responsive-fit-smoke` 갱신(`f83019a`).
- **첫 실경기 시험 운영: 849809 (LAD@MIL), 2026-10-12 08:14~12:18 KST, 실제 폴링, 스위치 모두 꺼짐.** 재시작 0, 서버 경고·오류 0, 조회 14,034회 중 실패 0, 실제 폴링 간격 중앙값 5.0초, 피드 지연 중앙값 15.7초(최대 32.9초), 서버 RSS 최대 1.8 GiB.
  - 전체 281구, 결과 전 원장 기록 추천 179, 끝까지 남음 170, 미지원 91, 묶여 도착 4.
  - 지연 0에서 공보다 먼저 뜬 추천 130/179. 늦은 49구 중 46구는 피드 지연(직전 결과 본문 지연 중앙값 19.6초 > 투구 간격 중앙값 16.0초).
- **시험 운영에서 찾아 고친 결함 (`9f4b36c`, `live_store.py` 4곳):** MLB의 경기 종료 본문이 직전 본문보다 1초 이른 시각 표시로 와서 "오래된 본문"으로 버려졌고, 그 결과 지나간 공 280구가 미확정으로 바뀌어 "투구 전 기록 추천" 결합이 1/281이 됐으며 화면이 "경기 종료"로 넘어가지 않았다. 이제 경기 상태가 앞으로 가는 본문은 받는다. 백엔드 702 passed / 1 xfailed / 1 기존 실패(`test_service_context_rev3`, 849819 홈 실책 수). 독립 검토 ACCEPT. 응답 본문 뜻 변경 없음. **실경기에서는 아직 확인하지 못했다.**
- 미지원 91구는 전부 Jacob Misiorowski(694819, MIL 선발, 학습 기간 밖). 명단 조사에 이미 미지원으로 있었다. 참고 카드를 켰다면 94구 중 92구에 카드가 나간다.

### 연구
- EXP-WS-CAL-001, RECENCY-001/002, SEQ-001, SEQ-CAL-001, SEQ-002, STEP5-CAL-001 완료 (`docs/experiments.md`, `docs/pm/research/state.md`). 순서 모델(Transformer, 직전 32구)의 나무 모델 대비 NLL 이득 약 −0.004(CI 0 아래)는 재현되지만 설정 10가지 탐색에서 더 나아지지 않았다. EVAL 구간은 7번 열람됨.
- 순서 모델 그림자 실행기 (`a779fbb`, `src/pitcheezy/seq_shadow/`, `scripts/seq_shadow_runner.py`, `configs/seq_shadow.json`). 모델 = SEQ-001 N1 3-seed + CAL2 보정기. 계산 p50 4.1 ms.
- 그림자 기록 두 경기 (짝지은 비교, 순서 모델 − 2024 모델):
  - 849831 (이미 노출, n=212): NLL −0.0223 [−0.0422, −0.0036]
  - 849809 (경기 전 고정, n=266): NLL +0.0040 [−0.0135, +0.0211], Brier +0.0074 [+0.0008, +0.0139]
  - 방향이 엇갈린다. 경기를 더 쌓아야 한다.

### 운영
- 매일 선수 프로필 갱신 예약 설치. 수동 1회 실행 정상(컷오프 2026-10-10, 849831 포함, sha `d05624c3…3bc7`). **launchd 아래에서 Drive·T7 접근이 되는지는 2026-10-12 21:00 첫 자동 실행에서야 확인된다.**

## 3. Song 결정 대기

시험 운영 뒤 올린 네 가지 (답 없음. 1·2는 "답 없으면 추천안"으로 올렸으나 **아직 구현하지 않았다**):
1. 종료 본문 아래 "확인 중"으로 남는 1구 — 그대로 둔다(추천) / 종료 본문이 투구 정정도 덮게 규칙 변경.
2. 경기 종료 뒤 "진행 중" 본문이 다시 오면 — 종료로 고정하는 한 줄 추가(추천, 검토자 의견) / 관찰.
3. 미지원 투수 참고 카드 — 공개 서비스에서 켠다(추천; 발행 계약에 `next.reference` 추가) / 끈 채 둔다.
4. 공통 지연 값 — 0초 유지 / 15초(추천) / 다음 시험 운영에서 더 재고 결정.

그 밖에:
- `rc/ws-2026` push 시점 (원격은 `e39bb41`).
- 발행기 차선(`svc/ws-publisher`)과 합치는 순서. 이 세션이 겹치는 파일에 넣은 것: `59ad2bc`의 `main.py`(+149/−20)·`service_api.py`(+66/−13), `9f4b36c`의 `live_store.py` 4곳.
- 디스크: 맥 여유 약 20 GiB. `pitcheezy-pm-jobs`의 10/8 폴더 네 개가 약 20 GiB. 삭제는 내용 확인 뒤 Song에게 묻는다.

## 4. 다음 세션이 할 일 (순서대로)

1. 2026-10-12 21:00 프로필 자동 갱신 결과 확인: `tail -n 3 /Users/song/Projects/pitcheezy-pm-jobs/profile-refresh-daily/logs/RESULTS.log`. 실패면 원인(launchd의 Drive·T7 접근)을 본다.
2. **두 번째 실경기 시험 운영**으로 `9f4b36c`를 확인한다. 후보: 849812 LAD@MIL 2026-10-13 06:00 KST, 849820 CLE@TB 2026-10-13 09:00 KST, 849813 CLE@TB 2026-10-14 09:00 KST (일정은 실행 전 다시 확인).
   - 절차: `/Users/song/Projects/pitcheezy-pm-jobs/20261011-rc-features-001/RUNBOOK-849809.md`를 그대로 쓰되 경기 번호와 `--expect-commit`(현재 끝)만 바꾼다. 시작 전에 폰 시연을 내리고, 화면 빌드가 소스보다 오래됐으면 `npm run build`.
   - 확인할 것: 종료 뒤 저장소에 Final 본문이 있고 화면이 "경기 종료"인지, 지나간 공의 추천 결합 수, 종료 뒤 "진행 중"으로 되돌아가는지, playId 없는 본문과 같은 시각 표시 충돌(849809에서 경기 중 28회, 최장 132초) 빈도.
3. 그 경기의 그림자 2회(2024 모델 → 순서 모델). `/Users/song/Projects/pitcheezy-pm-jobs/20261012-shadow-849809/`의 등록·RUN-ORDER를 본떠 **경기 전에** 경기별 설정·등록을 고정하고 독립 검토를 받는다. 프로필은 경기일보다 엄격히 앞선 컷오프의 것으로 경기 전에 고정한다.
4. 결정 1~4에 답이 오면 구현. 2번(종료 고정)과 3번(참고 카드)은 겹치는 파일·발행 계약에 닿으므로 `Contract-Change:` 표시와 발행기 차선 통보가 필요하다.
5. 미분류 화면 검사 4개(`astra-refinement-check`, `boxscore-asof-check`, `live-pinned-stale-fit`, `service-ui-smoke`)를 낡은 검사인지 결함인지 분류한다.
6. 문서: 849809 시험 운영 결과와 두 그림자 결과를 `docs/pm/service/state.md`, `docs/pm/research/state.md`, `docs/experiments.md`에 결과 줄로 옮긴다(노출 장부 줄만 들어가 있다). `9f4b36c`와 순서 모델 그림자 전환(2026-10-12 Song 승인), 프로필 예약 설치를 `docs/decisions.md` D177 이후로 적는다.
7. 아이폰 Safari 실기기 확인은 한 번도 하지 못했다.

## 5. 지켜야 할 것

- PID 94628, 94642, 99202와 포트 8766, 8771, 기록기 DB에 손대지 않는다. 신호 금지.
- push 금지(Song 지시 전까지). `git add -A` 금지. 작업 폴더 `ws-publisher`, `ws-static-web`은 건드리지 않는다.
- `/api/service/**` 응답 본문의 필드 뜻을 바꾸거나 빼면 커밋에 `Contract-Change:` 꼬리말.
- GPU 서버는 `/Users/song/.local/bin/gpu run`으로만 (셸의 `gpu`는 `git push upstream` 별칭이다).
- 실자료 실행 전 등록 + 작성자와 다른 에이전트의 독립 검토. 그림자 실행은 경기당 한 번만 읽는다.
- `ps -o lstart=` 파싱은 `LC_ALL=C LANG=C`.
- 서버는 127.0.0.1 또는 100.108.252.111(내부망)에만 묶는다.

## 6. 증거 경로

- 시험 운영: `/Users/song/Projects/pitcheezy-pm-jobs/20261011-rc-features-001/live/runs/849809-20261011T231222Z-real/` (`analysis/REPORT.md`, `shadow-run-849809-001/SUMMARY.json`, `seq-shadow-run-849809-001/SUMMARY.json`)
- 그림자 등록·순서: `/Users/song/Projects/pitcheezy-pm-jobs/20261012-shadow-849809/`, `/Users/song/Projects/pitcheezy-pm-jobs/20261012-seq-shadow-001/`
- 연구 로그: `/Users/song/Projects/pitcheezy-pm-jobs/20261011-ws-research-e1e3-001/README.md`
- 명단 조사: `/Users/song/Projects/pitcheezy-pm-jobs/20261011-coverage-census-001/CENSUS.md`
- 기능 묶음 검토: `/Users/song/Projects/pitcheezy-pm-jobs/20261011-rc-features-001/review/`
