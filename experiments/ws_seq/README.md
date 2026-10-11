# experiments/ws_seq — EXP-WS-SEQ-001 (E4)

투구 시퀀스 소형 신경망 vs prequential 트리(E1b P). **2026 EXPOSED DEVELOPMENT, 연구 모델, 승자·채택 아님.**

정본 등록: `/Users/song/Projects/pitcheezy-pm-jobs/20261011-ws-research-e1e3-001/e4-seq/REGISTRATION.json`
(독립 검토 `REVIEW.json` PASS 전에는 실자료 실행이 막혀 있다.)

## 한 줄 설명
- N0: 199개 사전(pre-pitch) 특징만 보는 MLP.
- N1: N0 + "같은 경기에서 같은 투수가 **이미 던진** 공 최대 32개"를 보는 작은 Transformer.
  야구로 치면 포수가 "이 투수가 오늘 앞서 뭘 던졌고 결과가 어땠는지"를 기억하고 다음 공 결과를 예상하는 것.
  지금 던지는 공의 결과·물리값은 절대 입력에 없다.

## 파일
| 파일 | 어디서 | 하는 일 |
|---|---|---|
| `prepare_inputs.py` | 맥 (`.venv-observer`) | parquet 투영 → corpus와 행 단위 일치 확인 → `inputs/` 패키지(정수 토큰, corpus 부분집합, x-prequential 복사, manifest) |
| `train_seq.py` | A5000 서버 (`gpu run`) 또는 CPU 자가검사 | 학습·조기 종료(SELECT)·온도 보정(CAL)·EVAL 7-leaf 확률. numpy/scipy/torch만 사용 |
| `evaluate_seq.py` | 맥 (`gpu pull` 뒤) | N0/N1(시드별·앙상블) vs H0/H2/P, whole-game bootstrap 10,000회, 층별 비교 |

설정: `configs/EXP-WS-SEQ-001.json`이 기계 정본, `configs/EXP-WS-SEQ-001.yaml`은 사람용 쌍둥이(값 동일해야 함, 자가검사가 확인).

## 종료 코드 (`train_seq.py`)
0 정상 / 2 사용법·설정 / 3 import 없음(설치하지 않는다) / 4 중단 규칙(시드 0·N1, 산출물은 기록됨) /
5 입력·코드·설정 sha256 불일치 / 6 등록된 STOP / 7 receipt가 이미 있음

## 산출물 (`<out>/<arm>-s<seed>/`)
`margins.npz`(`<role>__<head>`), `temperatures.json`, `seven-leaf-evaluation.npy`, `curve.json`, `model-state.pt`, `receipt.json`(마지막에 기록)

## 실행 순서
명령 전체는 job 디렉터리의 `AUTHOR-HANDOFF.md`에 있다. 요약: (1) E1b 완료 후 등록의 `PENDING_E1B` 채움 → (2) 독립 검토 →
(3) 맥에서 `prepare_inputs.py --execute` → (4) `inputs/`를 서버 링크 경로로 복사 → (5) `gpu run` 시드 0 →
(6) 중단 규칙 통과 시 시드 1·2 → (7) `gpu pull` → (8) `evaluate_seq.py`.

## 주의
- CUDA 학습은 비트 단위 재현이 보장되지 않는다(시드·버전·GPU는 receipt에 기록).
- 코드나 설정을 고치면 sha256이 바뀌므로 검토와 `prepare_inputs.py`를 다시 해야 한다(manifest가 코드·설정 해시를 묶는다).
