# experiments/ws_seq — EXP-WS-SEQ-002 (E6)

투구 시퀀스 모델(EXP-WS-SEQ-001 N1)을 **미리 등록한 10개 설정 격자**로 넓혀 보는 실험.
**2026 EXPOSED DEVELOPMENT, 연구 모델, 승자·채택 아님.** EVAL은 오늘 이미 여러 번 읽혔다(scale-002, E1, E3, E1b, E4, CAL2).

정본 등록: `/Users/song/Projects/pitcheezy-pm-jobs/20261011-ws-research-e1e3-001/e6-seq2/REGISTRATION.json`
(독립 검토 `REVIEW.json` PASS 전에는 실자료 실행이 막혀 있다.)

## 한 줄 설명
- 포수가 "오늘 이 투수가 앞서 던진 공"을 기억하는 모델에, 그 **앞선 공들**의 존(zone)·구속·무브먼트 구간과
  "이 타자가 오늘 앞서 본 공"(타자 쪽 맥락)을 더해 본다.
- **지금 던지는 공**의 존·구속·무브먼트·결과는 절대 입력에 들어가지 않는다(맥락 인덱스가 자기 자신을 가리키면 STOP, 자가검사에서 0.0 변화 확인).
- 구속·무브먼트 구간 경계는 FIT 행만으로 맞춘다. 타자 쪽 맥락도 같은 경기의 엄격히 이전 투구만 쓴다.

## 격자 (설정 파일 `configs/EXP-WS-SEQ-002.json`이 기계 정본, `.yaml`은 사람용 쌍둥이)
| id | 맥락 길이 | 토큰 | 타자 쪽 | 크기 |
|---|---|---|---|---|
| G01 | 32 | 기본 | - | d64 · 2층 (SEQ-001 N1과 같은 구조, 같은 격자 안 기준점) |
| G02 | 16 | 기본 | - | d64 · 2층 |
| G03 | 64 | 기본 | - | d64 · 2층 |
| G04 | 32 | +존·구속·무브먼트 | - | d64 · 2층 |
| G05 | 64 | +존·구속·무브먼트 | - | d64 · 2층 |
| G06 | 32 | +존·구속·무브먼트 | 16구 | d64 · 2층 |
| G07 | 64 | +존·구속·무브먼트 | 16구 | d64 · 2층 |
| G08 | 32 | +존·구속·무브먼트 | - | d96 · 3층 · ff192 |
| G09 | 64 | +존·구속·무브먼트 | - | d96 · 3층 · ff192 |
| G10 | 64 | +존·구속·무브먼트 | 16구 | d96 · 3층 · ff192 |

학습 방식(AdamW 1e-3, 배치 2048, 최대 30 에폭, SELECT 조기 종료 patience 5, fp32, 머리별 온도 1개)은 SEQ-001과 같다.

## 파일
| 파일 | 어디서 | 하는 일 |
|---|---|---|
| `prepare_inputs2.py` | 맥 (`.venv-observer`) | parquet 투영 → corpus·SEQ-001 패키지와 행 단위 일치 확인 → 키+zone 4열만 추가로 읽기 → FIT 전용 구간 경계 → `inputs2/`(정수 토큰, 경계 JSON, manifest) |
| `train_seq2.py` | A5000 서버 (`gpu run`) 또는 CPU 자가검사 | 설정 1개 × 시드 1개 학습. margin·확률만 쓰고 **EVAL 지표는 계산하지 않는다** |
| `select_config.py` | 맥 (`gpu pull` 뒤) | 시드 0의 SELECT 7-leaf NLL(보정 전)만으로 설정 1개 선택 → `SELECTION.json`. EVAL 지표가 하나라도 있으면 거부 |
| `calibrate_logits.py` | 맥 (다른 작성자, EXP-WS-SEQ-CAL-001) | 선택된 설정의 3개 시드에 대해 CAL로 보정기 선택, EVAL 비교 1회 |

## 종료 코드
- `train_seq2.py`: 0 정상 / 2 사용법·설정(시드 1·2는 `--selected-config-hash` 필요) / 3 import 없음 / 4 예약(쓰지 않음) / 5 입력·코드·설정 sha256 불일치 / 6 등록된 STOP / 7 receipt가 이미 있음
- `select_config.py`: 0 선택 완료 / 1 BLOCKED(검토 없음, EVAL 지표·미등록 파일 존재, receipt 누락) / 4 중단 규칙 / 6 등록된 STOP(receipt 불일치)

## 산출물
- `data/ws-seq/inputs2/`: `seq-inputs2.npz`, `bucket-edges.json`, `manifest.json` (SEQ-001의 `data/ws-seq/inputs/` 세 파일은 복사하지 않고 sha256으로 참조)
- `runs/ws-seq2/EXP-WS-SEQ-002/<Gxx>-s<seed>/`: `margins.npz`, `temperatures.json`, `seven-leaf-evaluation.npy`, `curve.json`, `model-state.pt`, `receipt.json`(마지막에 기록)

## 실행 순서
명령 전체는 job 디렉터리의 `AUTHOR-HANDOFF.md`에 있다. 요약:
(1) 독립 검토 → (2) 맥에서 `prepare_inputs2.py --execute` → (3) `inputs2/`를 서버의 `data/ws-seq/inputs2`로 복사 →
(4) 시드 0 작업 10개(카드 4장씩) → (5) `gpu pull` 후 `select_config.py` → (6) 선택된 설정의 시드 1·2 →
(7) `gpu pull` → (8) 보정 단계 등록·검토 후 `calibrate_logits.py` 1회.

## 주의
- 선택은 SELECT만 본다. 10개 중 최선을 고르므로 선택된 설정의 SELECT 값은 낙관적으로 치우친다.
- CUDA 학습은 비트 단위 재현이 보장되지 않는다(시드·버전·GPU는 receipt에 기록). G01도 SEQ-001 수치와 똑같이 나오지 않을 수 있다.
- 코드나 설정을 고치면 sha256이 바뀌므로 검토와 `prepare_inputs2.py`를 다시 해야 한다(manifest가 코드·설정 해시를 묶는다).
- 등록 파일을 검토 뒤에 고치면 `select_config.py`가 막힌다(receipt가 등록 sha256을 담는다).
