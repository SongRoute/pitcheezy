"""Figure 0: structure of the final model. Mixing weights and temperature are read from results/final-mix-weights.json.

    python scripts/make_structure_figure.py [OUT_DIR]   (default results/figures)
"""
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fonts import rc  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
out = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "results/figures"; out.mkdir(parents=True, exist_ok=True)
plt.rcParams.update(rc())
MIX = json.loads((REPO / "results/final-mix-weights.json").read_text())
assert MIX["members"] == ["X-GBT", "X-GRU-A3-plus", "X-GRU-A5-b24"], MIX["members"]
W = [round(w, 4) for w in MIX["weights"]]; WL = [f"{w:.2f}" for w in MIX["weights"]]; TL = f"{MIX['temperature']:.2f}"
INK, GRAY, RULE, SOFT, ACC, ACC2 = "#101418", "#5f6873", "#cfd4da", "#eef1f5", "#1f5fae", "#a9c6ec"

fig = plt.figure(figsize=(13.333, 7.5)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 56.25); ax.axis("off")
def card(x, y, w, h, title, lines, fc=SOFT, tc=INK, big=13):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=fc, ec="none"))
    ax.text(x + w / 2, y + h - 2.6, title, fontsize=big, fontweight="bold", color=tc, ha="center", va="center")
    for i, l in enumerate(lines):
        ax.text(x + w / 2, y + h - 6.0 - i * 2.7, l, fontsize=10.2, color=tc if fc == ACC else GRAY, ha="center", va="center")
def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16, lw=1.6, color=GRAY, shrinkA=0, shrinkB=0))

ax.text(4, 52, "구종 예측 모델의 구조", fontsize=24, fontweight="bold", color=INK, va="center")
ax.text(4, 48.2, "던지기 전 정보만으로 다음 공의 구종 9가지 확률을 낸다", fontsize=12.5, color=GRAY, va="center")

# 입력
card(4, 16, 21, 27, "입력 (던지기 전 정보)", ["상황 11칸", "볼 · 스트라이크 · 아웃", "이닝 · 점수차 · 주자", "투수 · 타자 번호", "같은 경기에서 이 투수가", "앞서 던진 공 최대 32개"])
# 세 모델
card(33, 34, 24, 9.5, "LightGBM", ["1개 · 표 특징 114차원"])
card(33, 22, 24, 9.5, "GRU × 5", ["2026 + 2023~2025 자료"])
card(33, 10, 24, 9.5, "GRU × 20", ["2026 자료 · 20번 학습해 평균"])
for y in (38.75, 26.75, 14.75): arrow(25.6, 29.5 if y != 26.75 else 26.75, 32.4, y)
# 가중 평균
card(65, 20, 13, 14, "가중 평균", WL)
for y, ty in ((38.75, 31.5), (26.75, 27), (14.75, 22.5)): arrow(57.6, y, 64.4, ty)
ax.text(71.5, 17.4, f"+ 온도 {TL}로 다듬기", fontsize=9.6, color=GRAY, ha="center", va="center")
# 출력
card(84, 16, 12.5, 27, "출력", ["포심", "싱커", "커터", "슬라이더", "스위퍼", "커브", "체인지업", "스플리터 · 기타"], fc=ACC, tc="white")
arrow(78.6, 27, 83.4, 27)
# 비중 띠
bx, bw = 33, 45
for x0, w, fc, lab, tc in ((0, W[0], RULE, WL[0], INK), (W[0], W[1], ACC2, WL[1], INK), (round(W[0] + W[1], 4), W[2], ACC, WL[2], "white")):
    ax.add_patch(FancyBboxPatch((bx + bw * x0, 3.2), bw * w - .3, 2.6, boxstyle="round,pad=0,rounding_size=.3", fc=fc, ec="none"))
    ax.text(bx + bw * (x0 + w / 2), 4.5, lab, fontsize=9.6, color=tc, ha="center", va="center", fontweight="bold")
ax.text(bx - 1, 4.5, "섞는 비중", fontsize=9.6, color=GRAY, ha="right", va="center")
ax.text(bx + bw + 1, 4.5, "LightGBM · GRU×5 · GRU×20", fontsize=9.4, color=GRAY, ha="left", va="center")
fig.savefig(out / "fig0-model-structure.png", dpi=170, facecolor="white"); print("ok")
