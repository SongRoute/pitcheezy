"""결과 한 장(16:9 PDF·PNG). 그림은 results/figures/(make_figures.py, make_structure_figure.py), 숫자는 results/rescore/table1-rescored.json.

    python scripts/make_onepage.py [OUT_STEM]   (default docs/onepage-16x9)
"""
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fonts import rc  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
IMG = REPO / "results/figures"
plt.rcParams.update(rc())
RS = json.loads((REPO / "results/rescore/table1-rescored.json").read_text())
T = {t["key"]: t for t in RS["table"]}
TOP, FREQ = T["TOP1"], T["T_frozen_sept"]
CB = next(c for c in RS["contrasts"] if (c["front"], c["back"]) == ("TOP1", "B-SEQ"))
def sg(v, n=5): return ("-" if v < 0 else "+") + f"{abs(v):.{n}f}"

INK, SUB, BLUE, LINE = "#111827", "#4b5563", "#1f5fae", "#e5e7eb"

fig = plt.figure(figsize=(16, 9))
fig.patch.set_facecolor("white")

fig.text(0.035, 0.94, "Pitcheezy — 다음 공을 먼저 보여 주는 야구 관전 화면", fontsize=24, weight="bold", color=INK)
fig.text(0.035, 0.900, "던지기 전에 다음 공의 구종을 보여 주고, 그 예측을 학습에 쓰지 않은 경기로 검증했고, 실제 경기 기록을 받아 서비스로 돌렸습니다.",
         fontsize=12.5, color=SUB)

def place(path, rect, title=None):
    ax = fig.add_axes(rect)
    ax.imshow(plt.imread(path))
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=12, color=INK, loc="left", weight="bold", pad=4)
    return ax

# 왼쪽: 구조 그림, 아래 검증 표
place(IMG / "fig0-model-structure.png", [0.025, 0.42, 0.50, 0.45])
place(IMG / "table3-design.png", [0.06, 0.07, 0.43, 0.34])

# 오른쪽 위: 숫자 3개
tiles = [
    ("예측 오차(NLL)", f"{TOP['nll']:.5f}", f"빈도표 {FREQ['nll']:.5f}\n2026 포스트시즌 {RS['games']}경기 {RS['rows']:,}구"),
    ("1순위 · 3순위 적중", f"{TOP['top1']:.4f} · {TOP['top3']:.4f}", "1순위 약 2번 중 1번\n3순위 안 약 10번 중 9번"),
    ("문헌식 기준 중 최고(GRU)와 차이", sg(CB['nll']), f"B-SEQ 대비 NLL\n95% 구간 [{sg(CB['ci95'][0])}, {sg(CB['ci95'][1])}]"),
]
x0, w, gap, y, h = 0.545, 0.135, 0.008, 0.70, 0.17
for i, (lab, val, note) in enumerate(tiles):
    x = x0 + i * (w + gap)
    fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.01",
                                      transform=fig.transFigure, facecolor="#f3f6fb", edgecolor=LINE))
    fig.text(x + 0.01, y + h - 0.035, lab, fontsize=11, color=SUB)
    fig.text(x + 0.01, y + 0.075, val, fontsize=21 if i != 1 else 18, weight="bold", color=BLUE)
    fig.text(x + 0.01, y + 0.018, note, fontsize=9, color=SUB, linespacing=1.4)

# 오른쪽 아래: NLL 비교
place(IMG / "fig1-nll.png", [0.545, 0.10, 0.425, 0.56])

# 한계 한 줄
fig.text(0.035, 0.025,
         "한계: 31경기는 모델을 고르며 여러 번 채점했고(와일드카드 8경기 1회 확인 -0.0059, 7/8), 검증한 것은 구종 확률뿐입니다. "
         "추천 효과(-0.12~+0.09%p)는 근거가 없고, 공개 이용자는 없었습니다.",
         fontsize=10.5, color="#9a3412")

out = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "docs/onepage-16x9"; out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out.with_suffix(".pdf"))
fig.savefig(out.with_suffix(".png"), dpi=150)
print(out.with_suffix(".pdf"), out.with_suffix(".png"))
