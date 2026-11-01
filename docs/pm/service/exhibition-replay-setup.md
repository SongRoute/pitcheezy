# 전시 재생 입력 준비

상위: [전시 런북](exhibition-runbook-2026-11-24.md) · 근거: [10/31 빌드](exhibition-build-2026-10-31.md)

### 3.1 회차 준비와 재생 파일 생성

아래는 맥미니에서 운영자가 실행할 명령이다. `cd`의 경로는 검토 완료한 전시 worktree로 채운다. 이 작업의 타임스탬프 수정이 포함된 커밋인지 먼저 확인한다. 기존 `d04afab1`만 쓰면 3.2의 구버전 경로가 필요하다. 설치·전역 설정 변경 없이 기존 환경을 사용한다.

```sh
cd /검토완료한/전시-worktree
task_repo="$PWD"
task_python="$HOME/Projects/pitcheezy/.venv-observer/bin/python"
task_assets="$HOME/Projects/pitcheezy-pm-jobs/exhibition-849843-1031"
task_run=$(mktemp -d "$task_assets/rehearsal-XXXXXX")
task_replay="$task_run/replay"
mkdir -p "$task_run/empty/feed" "$task_run/empty/demo" "$task_run/site"
git rev-parse HEAD
git status --short
(cd apps/observer/web && npm run build:public)
cp -R apps/observer/web/dist-public/. "$task_run/site/"
PYTHONPATH=src:.:apps/observer/backend "$task_python" -B scripts/demo_precompute.py live-replay \
  --game-pk 849843 --start 0 --count 262 --output "$task_replay"
```

기본 시작은 `--start 0`(첫 투구 전), 순번은 **0부터**다. 장면 점프는 새 폴더·새 store로 다시 생성한다. 기존 출력 폴더에 짧은 구간을 덮어쓰면 남은 뒷부분 파일까지 재생될 수 있다. Rolison 참고 카드는 빌드 기록의 5회말 138~145구 구간을 안내 표지로 쓰되, 실제 선수·이닝·타석을 확인해 점프 지점을 리허설 기록에 적는다. `--count`는 양수이며 범위 끝은 262구 이내로 정한다.

### 3.2 타임스탬프 호환성

수정 코드로 만든 파일은 별도 stamp 스크립트 없이 `metaData.timeStamp`가 단조 증가하는 `YYYYMMDD_HHMMSS`다. 공식 경기일+절대 순번으로 만든 **재생용 순서 표시**이며 실수신 시각이 아니다. 재생을 처음으로 되돌리면 반드시 새 store를 사용한다. 과거 회차 store의 더 큰 도장을 유지한 채 되감으면 stale 판정이 정상이다.

구버전 `d04afab1`을 재현할 때만 빌드 §6의 우회 절차를 쓴다. 원본 재생 파일은 보존한다.

```sh
"$task_python" "$task_assets/stamp_snapshots.py" "$task_replay" "$task_run/replay-stamped" 25
task_replay="$task_run/replay-stamped"
```

이 명령의 `25`는 도장 간격이고 화면 속도가 아니다. 수정 코드와 구버전 stamp 우회를 한 회차에서 혼용하지 않는다.


다음: [발행기 실행](exhibition-replay-publisher.md). 위 셸 변수는 같은 터미널에서 이어 사용한다.
