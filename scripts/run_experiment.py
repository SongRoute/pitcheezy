"""실험 실행 진입점 (CLAUDE.md 실험 절).

사용:
    python scripts/run_experiment.py --config configs/EXP-P0-001.yaml --seed 0

규칙:
    - 설정 파일명 = 실험 ID (EXP-P{phase}-{seq}). 실험은 configs/ 파일로만 정의한다.
    - 산출물(텐서·Q·가중치·체크포인트): $PITCHEEZY_RUNS_DIR/{ID}/{seed}/  (기본 ./runs)
    - 데이터 입력: $PITCHEEZY_DATA_DIR  (기본 ./data).
      두 환경변수는 notebooks/colab_runner.ipynb가 Drive 경로로 설정한다.
    - 요약: results/{ID}.json (커밋 대상). 원천 수치는 W&B run "{ID}/s{seed}".
    - 시드마다 체크포인트를 저장하고 재개할 수 있어야 한다 (Colab 세션 만료 대비).

현재 상태: 뼈대. 인자 파싱만 있고 본 로직은 미구현.
"""

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--config", type=Path, required=True, help="configs/p{phase}/{ID}.yaml"
    )
    parser.add_argument(
        "--seed", type=int, required=True, help="시드. 탐색 {0,1,2} / 채택 판정 {0,1,2,3,4}"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(
        f"실험 러너 미구현 (config={args.config}, seed={args.seed}). "
        "docs/interface-spec.md 동기화 후 구현."
    )


if __name__ == "__main__":
    main()
