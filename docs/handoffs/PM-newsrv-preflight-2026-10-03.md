# newsrv GPU 이전 사전점검 (2026-10-03)

작성: Claude Opus 5.5 (Codex PM 위임). 기준 커밋 `0abda69`. 짝 문서: [예측 연구 전선](../reviews/PM-research-frontier-2026-10-03.md).
범위: 읽기 전용 점검과 실행 설계. 설치·전송·잡 실행·프로세스 조작 0. 서버 사용 가능 예상 10/3 17:00 KST, 이 프로젝트 상한 GPU 2장(D163).

## 1. 확인한 사실 (2026-10-03 12:00:56 KST, `ssh -o BatchMode=yes -o ConnectTimeout=8 newsrv`)

- GPU: RTX A5000 24,564MiB × 5.
  - 0: 2,860MiB 사용·77%, 1: 3,236MiB·86%, 2: 3,236MiB·83% (다른 작업 가동 중)
  - 3: 7,012MiB 사용·0% (메모리 점유, 연산 없음 — 소유자 미확인)
  - 4: 2MiB·0% (이 시점 비어 있음)
  - compute 프로세스 4개(소유자·명령은 보지 않음).
- `~/venv`: Python 3.12.3, torch 2.5.1+cu121, CUDA 12.1, `cuda.is_available()` True, 장치 5개. numpy 2.3.5, pandas 3.0.1, pyarrow 24.0.0, scipy 1.17.1, PyYAML 6.0.3. **sklearn·lightgbm·h5py 없음.**
- CPU 64코어(`nproc`), RAM 503GiB(가용 408GiB). 홈 파일시스템 1.7T 중 325G 남음(81% 사용, 공유 overlay).
- 홈 최상위에 이 프로젝트 폴더 없음(이름만 봄). 맥미니 쪽 `~/.local/bin/gpu` 도구는 있다.

12:00 시점 관측이며 17:00 상태는 미측정이다.

## 2. 실행 전에 막아야 할 문제

1. **`gpu` 이름 충돌.** 맥미니 zsh에서 `gpu`는 `alias gpu='git push upstream'`로 가려져 있다(`command -v gpu` 결과). 그대로 치면 **원격 push가 실행된다.** 반드시 `~/.local/bin/gpu ...` 전체 경로로 부른다.
2. **코드에 CUDA 경로가 없다.** `matrix_models.py:108`, `matrix_lazy_model.py:78`, `model.py:142` 등이 `'mps' if mps else 'cpu'`만 고른다. newsrv에서는 그대로 **CPU 실행**이 된다(64코어라 느리지 않을 수는 있으나 GPU를 쓰지 않음). GPU를 쓰려면 새 runner 옵션(`--device cuda`)이 필요하고, 봉인 runner·소스 해시 계약은 바꾸지 않는다.
3. **장치 혼입.** 봉인 G15 mix 직접 head는 맥미니에서 학습됐다. 후보만 CUDA로 학습하면 장치 차이가 변경점에 섞인다 → 같은 장치에서 기준 직접 head를 다시 학습한 대조 arm을 함께 두거나 탐색은 맥미니에서 한다.
4. **2026 자료 혼입.** 원천 폴더 `data/raw/d20260911-s2325/`에 `holdout_2026/`이 함께 있다. 전송은 허용 목록(`statcast_2023/2024/2025.parquet` + `manifest.json`)만, 폴더 통째 복사 금지. 스냅샷에도 `data/`·`runs/`가 들어가지 않게 `.gpurc` `link`만 쓴다.
5. **절대경로.** `experiments/pitchmdp/configs/local.json`과 실험 config `execution.prefix`/`runs_root`가 `/Users/song/...`, `/Volumes/T7 Shield/...`를 박아 둔다. 서버용 local config를 별도 파일로 만들고(코드 하드코딩 금지) 기존 파일은 고치지 않는다.

## 3. 옮길 것 (크기는 맥미니 `ls`/`du` 실측)

- 원천: `statcast_2023.parquet` 95MB, `statcast_2024.parquet` 103MB, `statcast_2025.parquet` 105MB.
- parent run: `ML-MATRIX-20260924/EXP-P4-001` 785MB(preparation sha256 `d544bf38…`), 기준 직접 head `EXP-P15-001` 161MB, `EXP-P16-001`(크기 미측정).
- 전송 뒤 서버에서 sha256 대조(같은 파일 목록으로 맥미니와 비교)를 먼저 하고 결과를 남긴다.

## 4. 권장 배치

- 환경: 기존 `~/venv` 재사용. 부족한 패키지(sklearn 등)는 이번 후보에 필요 없다. 설치가 필요해지면 먼저 묻는다.
- 위치: 서버 홈 아래 프로젝트 전용 폴더 하나(예: `~/pitcheezy/{data,runs}`) — 생성은 실행 승인 뒤. `.gpurc` 예시(커밋 전 PM 확인):
  ```
  venv ~/venv
  min_free_mb 10000
  link data ~/pitcheezy/data
  link runs ~/pitcheezy/runs
  ```
- 잡 이름: `<실험ID>-s<seed>`, 실행 전 커밋, `~/.local/bin/gpu run --dry-run -n ... -g <번호> -- ...`로 스냅샷 크기·링크·GPU 확인 → 실제 실행 → `gpu wait` 백그라운드 → `gpu pull`로 맥미니 회수(`~/gpu-pull/pitcheezy/<잡>/`).

## 5. GPU 고르기와 유휴 확인 (매 실행 직전)

1. `nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv` 를 1분 간격 3회.
2. 후보 = 3회 모두 메모리 사용 < 500MiB AND 사용률 0%. 메모리만 점유한 GPU(12:00의 3번처럼)는 다른 사람이 잡아 둔 것으로 보고 쓰지 않는다.
3. 후보 중 최대 2장(D163). 없으면 실행하지 않고 보고한다. 남의 프로세스는 조회·중지하지 않는다.
4. `-g auto` 대신 확인한 번호를 명시(`-g 4` 등)해 기록에 남긴다.

## 6. 다음 순서 (실행은 PM 배정 뒤)

1. 맥미니에서 연구 전선 문서 S0–S1(runner 옵션·config·합성 테스트·smoke)로 시간·RSS 실측.
2. 로컬 예상 2시간 초과 후보(E5 FT-Transformer 등)만 newsrv 대상으로 정한다.
3. 17:00 이후 §1 항목 재측정 → `--device cuda` 옵션과 기준 재학습 arm을 포함한 config 커밋 → dry-run → 실행.
4. 10/4 01:40 KST까지 모든 무거운 실행 종료·회수(PM-charter §6, D159).

## 7. 한계

- 서버 관측은 12:00 한 번이다. 다른 사용자 계획·예약은 모른다.
- 서버 디스크는 공유 overlay라 남은 325G가 이 프로젝트 몫으로 보장되지 않는다.
- CUDA와 MPS 사이 수치 재현성은 측정하지 않았다.
