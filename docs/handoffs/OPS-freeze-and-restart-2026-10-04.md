# 운영 동결과 복구 절차 (디비전시리즈 10/4 측정까지)

작성: 2026-10-02 19:30 KST. 근거: 사용자가 2026-10-02 PM 브리핑의 "오늘 정할 두 가지"에서 고른 동결 안(결정 기록 `decisions.md`에는 아직 줄이 없다), D149(4경기 자동 기록·통과 기준).
아래 명령은 그 시각에 맥미니에서 돌고 있던 tmux 창과 프로세스에서 그대로 옮긴 것이다. 이 문서는 무엇도 실행하지 않았다.

## 1. 동결 (10/4 디비전시리즈 4경기 대조가 끝날 때까지)

실경기는 다시 잴 수 없다. 4경기(849829 02:00, 849828 05:00, 849835 07:30, 849830 09:30 KST 시작)의 기록·대조·요약이 끝날 때까지 다음을 하지 않는다.

- 맥미니 재부팅·로그아웃
- 외장 SSD(`/Volumes/T7 Shield`) 분리
- tmux 세션 `pz` 종료, 아래 표의 창 닫기
- 운영 서버 `:8766` 재시작. 서버는 `/Users/song/Projects/pitcheezy-demo` 체크아웃의 코드로 뜨므로, 그 폴더의 커밋이 바뀐 뒤 다시 띄우면 다른 코드가 올라간다(다시 띄우기 전에 `git -C /Users/song/Projects/pitcheezy-demo log -1`과 `status --short`를 본다)
- 무거운 실험(학습·정책 검증·OPE). 세 경기가 겹칠 때의 메모리는 미측정이다(한 경기 1,377MB, 두 경기 2,351MB만 실측. 두 경기 값은 시험 인스턴스 8769에서 잰 것, LIVE-service-plan §4 측정 표).

끝났는지는 `'/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/DEMO-WS-2026/live_rehearsal/ds/all.done'` 파일과 같은 폴더의 `summary.md`로 확인한다. 요약 루프는 2026-10-05 06:00 KST 뒤 첫 확인(30분 간격) 때 멈춘다. 대조 루프에는 시간 제한이 없어서, 경기가 연기되거나 사전 계산 파일이 안 생기면 그 경기에서 계속 기다리고 `all.done`도 생기지 않는다.

## 2. 지금 돌고 있는 것 (tmux `pz`)

| 창 | 이름 | 하는 일 | 재부팅 뒤 |
|---|---|---|---|
| 4 | demo-serve | 운영 서버 `100.108.252.111:8766` (2026-10-02 17:54 시작) | 안 뜸 |
| 5 | demo-sync | 30분마다 끝난 경기 사전 계산 | 안 뜸 |
| 6 | ui-frame-view | 새 화면 미리보기 `:8770` | 안 뜸 |
| 7 | ds-check | 경기가 끝나면 실시간 대 최종 대조 | 안 뜸 |
| 10~13 | rec-849829·849828·849835·849830 | 경기 10분 전부터 기록 → 보고서 | 안 뜸 |
| 14 | ds-summary | 30분마다 한 장 요약 | 안 뜸 |

launchd·cron에 등록된 것은 없다. 서버는 죽어도 스스로 다시 뜨지 않는다(`serve.sh`가 `exec`로 한 번 띄운다). 서버 로그 `/tmp/pitcheezy-demo-serve.log`는 시작할 때마다 덮어쓴다.

## 3. 재부팅됐거나 창이 죽었을 때 다시 띄우기

먼저 확인:

```sh
ls '/Volumes/T7 Shield/pitcheezy'            # SSD가 붙어 있는가
/usr/local/bin/tailscale ip -4               # 100.108.252.111 이 나오는가
tmux has-session -t pz || tmux new-session -d -s pz
```

각 명령은 tmux 창 하나씩에서 실행한다(`tmux new-window -t pz -n <이름>`). 순서는 서버(①) → 기록기(②) → 사전 계산(④) → 대조(③) → 요약(⑤)이다. 재부팅 뒤에는 맥에 화면 로그인이 돼 있고 Tailscale 앱이 떠 있어야 주소가 나온다. 명령은 줄여 쓰지 않는다: 기록기의 `--server`와 미리보기의 세 번째 인자를 빼면 기본값 `127.0.0.1:8766`의 옛 서버(9/23에 띄운 것)에 붙는다.

**① 운영 서버 (demo-serve)**

```sh
cd /Users/song/Projects/pitcheezy-demo && PITCHEEZY_OBSERVER_PYTHON=/Users/song/Projects/pitcheezy/.venv-observer/bin/python PITCHEEZY_OBSERVER_HOST=tailscale sh apps/observer/serve.sh 2>&1 | tee /tmp/pitcheezy-demo-serve.log
```

확인: `curl -s http://100.108.252.111:8766/api/health` 의 `demo.status`가 `ok`, `demo.live_policy.state`가 `ready`. 막 띄운 직후에는 `loading`이고 모델 준비가 끝나면 `ready`가 된다(걸리는 시간은 미측정). `unavailable`이면 실패다. 맨 위의 `mode`·`model_ready`·`dataset_ready`는 예전 앱 기준 값이라 보지 않는다. 경기 중이라면 `ready`를 본 뒤에 기록기를 띄운다. 서버가 없는 동안의 조회는 전부 오류로 기록된다.

**② 기록기 4개 (rec-\<경기\>)** — 경기 번호와 `--not-before`만 다르다. 시각이 이미 지났으면 바로 조회를 시작한다. `<경기>.done`은 기록기 창의 명령이 끝났다는 뜻일 뿐이다(기록기가 죽거나 7시간 상한에 걸려도 생긴다). 다시 띄울지는 `<경기>_record.log` 마지막 줄의 `finished` 값으로 판단한다.

```sh
cd /Users/song/Projects/pitcheezy-demo && python3 scripts/live_rehearsal.py record --game-pk 849829 --server http://100.108.252.111:8766 --not-before 2026-10-03T16:50:00Z --output '/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/DEMO-WS-2026/live_rehearsal/ds' 2>&1 | tee '/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/DEMO-WS-2026/live_rehearsal/ds/849829_record.log'; python3 scripts/live_rehearsal.py report --game-pk 849829 --output '/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/DEMO-WS-2026/live_rehearsal/ds' --snapshots /Users/song/Projects/pitcheezy-demo/apps/observer/live_snapshots > '/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/DEMO-WS-2026/live_rehearsal/ds/849829_report.log' 2>&1; touch '/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/DEMO-WS-2026/live_rehearsal/ds/849829.done'
```

| 경기 | `--not-before` (UTC) | 시작 (KST) |
|---|---|---|
| 849829 | 2026-10-03T16:50:00Z | 10/4 02:00 |
| 849828 | 2026-10-03T19:50:00Z | 10/4 05:00 |
| 849835 | 2026-10-03T22:20:00Z | 10/4 07:30 |
| 849830 | 2026-10-04T00:20:00Z | 10/4 09:30 |

경기 도중에 다시 띄우면 그 사이 구간은 기록에 없다. `<경기>_polls.jsonl`은 이어 붙으므로(`scripts/live_rehearsal.py`의 `record`가 추가 모드로 연다) 옮기지 않고 그대로 둔다. `tee`가 덮어쓰는 것은 `<경기>_record.log` 하나뿐이니 필요하면 그것만 다른 이름으로 옮긴다.

**③ 대조 (ds-check)** — 다시 돌려도 결과 파일은 임시 파일에 쓴 뒤 바꿔치기로 덮어써서 섞이지 않는다. 다만 매번 모델을 새로 올리므로(시간·메모리 미측정), 다른 경기가 진행 중일 때는 이미 대조가 끝난 경기를 `for` 목록에서 뺀다.

```sh
cd /Users/song/Projects/pitcheezy-demo && for pk in 849829 849828 849835 849830; do until [ -f '/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/DEMO-WS-2026/live_rehearsal/ds'/$pk.done ] && [ -f '/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/DEMO-WS-2026/watch'/$pk.json ]; do sleep 120; done; PYTHONPATH=apps/observer/backend /Users/song/Projects/pitcheezy/.venv/bin/python scripts/demo_precompute.py live-check --game-pk $pk --snapshots apps/observer/live_snapshots > '/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/DEMO-WS-2026/live_rehearsal/ds'/${pk}_live_check.log 2>&1; done; touch '/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/DEMO-WS-2026/live_rehearsal/ds'/all.done
```

대조는 끝난 경기의 사전 계산 파일(`watch/<경기>.json`)을 기다린다. 그 파일은 ④의 사전 계산 루프가 만든다.

**④ 사전 계산 루프 (demo-sync)**

```sh
cd /Users/song/Projects/pitcheezy-demo && while true; do date; /Users/song/Projects/pitcheezy/.venv/bin/python scripts/demo_precompute.py sync --since 2026-09-29; sleep 1800; done 2>&1 | tee -a /tmp/pitcheezy-demo-sync.log
```

**⑤ 요약 (ds-summary)** — `1791147600`은 2026-10-05 06:00 KST다.

```sh
cd /Users/song/Projects/pitcheezy-demo && while [ ! -f '/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/DEMO-WS-2026/live_rehearsal/ds'/all.done ] && [ $(date +%s) -lt 1791147600 ]; do python3 scripts/live_rehearsal.py summary --dir '/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/DEMO-WS-2026/live_rehearsal/ds' --games 849829 849828 849835 849830 --checks '/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/DEMO-WS-2026/live_check' > /dev/null 2>&1; sleep 1800; done; sleep 60; python3 scripts/live_rehearsal.py summary --dir '/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/DEMO-WS-2026/live_rehearsal/ds' --games 849829 849828 849835 849830 --checks '/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/DEMO-WS-2026/live_check' > /dev/null 2>&1
```

**⑥ 새 화면 미리보기 (ui-frame-view)** — 측정과 무관하다. 세 번째 인자를 빼면 `127.0.0.1:8766`의 옛 서버(9/23에 띄운 것)에 붙으므로 반드시 적는다.

```sh
cd /Users/song/Projects/pitcheezy-worktrees/ui-frame/apps/observer/web && node viewer.mjs 100.108.252.111 8770 http://100.108.252.111:8766
```

## 4. 통과 기준 (D149·D153, 결과 열람 전에 고정)

- 재시작과 무관한 조회 오류 0건
- 계산 오류 0건
- 중계가 30초 늦는 시청자 기준, 공 5초 전에 추천이 떠 있는 비율 95% 이상
- 실시간 화면과 최종 기록의 1순위 구종 일치 95% 이상 (처음 본 화면 기준, D153)

재부팅으로 다시 띄운 경기는 "재시작과 무관한 오류"를 따로 세어야 하므로, 다시 띄운 시각을 이 문서 아래에 한 줄 적는다.

## 5. 동결이 풀린 뒤

자동 시작(launchd) 설치, 옛 서버 두 개(`127.0.0.1:8766`·`8767`) 정리, 데모 브랜치 통합은 4경기 결과를 본 뒤 사용자 결정으로 진행한다.
