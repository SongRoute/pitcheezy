# scripts/wc8 — WC8-CONFIRM-001 (와일드카드 8경기 단 한 번 확인 시험)

등록: `<private-jobs>/20261122-wc8/PREREG.md`, `PRIOR-USE.md`. 순서는 PREREG §8.

1. `gate_849799.py --out <새 dir>`: 849799에서 v4·R 예측이 부스 파일(9d0dfcc4, 64da605c)과 1e-12 안에서 같은지 확인한다. 라벨은 읽지 않는다.
2. `predict.py --game <pk> --model {v4,bseq,r} --out <dir>`: 8경기 × 3모델. 실행마다 새 프로세스를 띄운다. 라벨은 읽지 않는다.
3. `baselines.py --game <pk> --out <dir>`: 빈도표 q와 투수 최빈 구종. 라벨은 읽지 않는다.
4. `pin.py --pred-dir <dir>`: `PINS.json`을 만든다. 파일 sha·코드 sha·커밋을 남기고, 라벨 없이 정렬을 검사한다.
5. `score.py --pred-dir <dir> --out <result.json>`: 단 한 번 실행한다. 라벨은 여기서 처음 읽는다.

`score.py`가 무엇을 받아들이고 거부하는지는 각 파일의 docstring에 있다.

## §7 화면 % 보정 (구현 안 함)

- `eps-screenrec-v4-1122`의 `scripts/svc_pred/eps_screenrec.py --policy v4` 경로를 쓰면 WC 경기별 v4 watch를 만들 수 있다. 다만 그 스크립트의 `game_entry`는 "849799 또는 E_PS"만 받으므로 WC 경기는 거부된다.
- 이를 열려면 새 가드가 필요하다. WC8 목록 허용, `--registration` 대신 이 등록을 쓰는 조건, R3 포스트시즌 as-of 절단(09-29 경기는 포스트시즌 행 0, 09-30 경기는 09-29 WC 행만 남음)을 검토해야 한다. 그래서 이번 범위에서는 구현하지 않았다.
- 한다면 watch의 1순위 후보 몫(%)과 실제 구종을 같은 단일 채점에서 맞대야 한다. 그러려면 `score.py`에 watch 입력·sha를 `PINS.json`에 추가하는 변경이 먼저 필요하다.
- PREREG §7 규칙상 11/23 12:00까지 watch가 없으면 이 항목은 뺀다.
