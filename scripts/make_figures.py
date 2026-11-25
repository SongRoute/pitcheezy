"""Figures 1-4 and tables 1-3 (PNG + CSV) for the README.

Numbers come from files, not from this script:
  - table 1, figures 1-3 and the 31-game rows of figure 4 / table 2: results/rescore/table1-rescored.json
    (made by scripts/rescore_table.py from saved probabilities)
  - the wild-card 8-game row: results/wildcard-8games.md (a one-time check, not re-scored here; its inputs are not public)
  - table 3 (design experiments): results/design-experiments.csv (values reported by earlier runs; not re-computed here)

    python scripts/make_figures.py [--rescore results/rescore/table1-rescored.json] [--out results/figures]
"""
import argparse
import csv
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fonts import rc  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--rescore", default=REPO / "results/rescore/table1-rescored.json", type=Path)
ap.add_argument("--design", default=REPO / "results/design-experiments.csv", type=Path)
ap.add_argument("--out", default=REPO / "results/figures", type=Path)
a = ap.parse_args()
out = a.out; out.mkdir(parents=True, exist_ok=True)
plt.rcParams.update(rc(**{"font.size": 10, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.facecolor": "white"}))
INK, GRAY, RULE, ACC = "#111111", "#6b7280", "#d9dcdf", "#1f5fae"

R = json.loads(a.rescore.read_text())
# rounded exactly as the published table (NLL 5 digits, top-k 4 digits)
M = [(t["model"], round(t["nll"], 5), round(t["top1"], 4), round(t["top3"], 4), t["made"]) for t in R["table"]]

def clean(ax):
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(RULE); ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=GRAY, labelsize=9); ax.grid(axis="x", color=RULE, lw=.6); ax.set_axisbelow(True)

def dots(idx, title, xlabel, lo, hi, fmt, fname, better):
    rows = sorted(M, key=lambda r: r[idx], reverse=(idx != 1))
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    y = list(range(len(rows)))[::-1]
    for yi, r in zip(y, rows):
        hero = r[0] == "LightGBM + GRU 앙상블"
        c = ACC if hero else GRAY
        ax.hlines(yi, lo, r[idx], color=RULE, lw=1.2, zorder=1)
        ax.scatter(r[idx], yi, s=70 if hero else 46, color=c, zorder=3)
        ax.annotate(fmt(r[idx]), (r[idx], yi), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK, fontweight="bold" if hero else "normal")
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=9.5, color=INK)
    for lab, r in zip(ax.get_yticklabels(), rows):
        if r[0] == "LightGBM + GRU 앙상블": lab.set_fontweight("bold"); lab.set_color(ACC)
    ax.set_xlim(lo, hi); ax.set_ylim(-.7, len(rows) - .3); clean(ax)
    ax.set_xlabel(f"{xlabel}  ({better})", fontsize=9.5, color=GRAY)
    ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold", color=INK, pad=10)
    fig.text(.01, -.06, "포스트시즌 31경기 8,943구(학습에 안 씀, 여러 번 본 자료). 가로축은 0에서 시작하지 않는다. 잔차 앙상블 1·2는 학습 기간만 다르다.",
             fontsize=8, color=GRAY)
    fig.savefig(out / fname); plt.close(fig)

dots(1, "그림 1. 구조별 예측 오차(NLL)", "예측 오차 NLL", 1.12, 1.32, lambda v: f"{v:.5f}", "fig1-nll.png", "낮을수록 좋음")
dots(2, "그림 2. 구조별 1순위 적중", "1순위 적중", .42, .525, lambda v: f"{v:.4f}", "fig2-top1.png", "높을수록 좋음")
dots(3, "그림 3. 구조별 3순위 안 적중", "3순위 안 적중", .855, .925, lambda v: f"{v:.4f}", "fig3-top3.png", "높을수록 좋음")

# 31-game contrasts from the re-score; the wild-card row is the reported one-time check (results/wildcard-8games.md)
C = R.get("contrasts")
if not C:
    raise SystemExit("table1-rescored.json has no contrasts: run scripts/rescore_table.py without --no-bootstrap")
WILDCARD = ("LightGBM + GRU 앙상블 - GRU (와일드카드 8경기)", -.0059, -.0103, -.0023)
P = [(C[0]["label"], C[0]["nll"], *C[0]["ci95"]), WILDCARD] + [(c["label"].replace(" (31경기)", ""), c["nll"], *c["ci95"]) for c in C[1:]]
P = [(lab.replace("Transformer + MLP 잔차 앙상블", "잔차 앙상블"), round(d, 5), round(lo, 5), round(hi, 5)) for lab, d, lo, hi in P]
def sg(v, n=5): return ("+" if v > 0 else "-") + f"{abs(v):.{n}f}"
fig, ax = plt.subplots(figsize=(7.2, 3.9))
y = list(range(len(P)))[::-1]
for yi, (lab, d, lo, hi) in zip(y, P):
    ns = lo < 0 < hi
    ax.hlines(yi, lo, hi, color=GRAY if ns else INK, lw=2.2 if not ns else 1.4, linestyles="--" if ns else "-", zorder=2)
    ax.scatter(d, yi, s=46, facecolor="white" if ns else INK, edgecolor=GRAY if ns else INK, lw=1.4, zorder=3)
    n = 4 if "와일드카드" in lab else 5
    ax.annotate(f"{sg(d, n)}  [{sg(lo, n)}, {sg(hi, n)}]" + ("  구분 안 됨" if ns else ""), (.052, yi),
                va="center", fontsize=8.5, color=GRAY if ns else INK, annotation_clip=False)
ax.axvline(0, color=GRAY, lw=1, ls=":")
ax.set_yticks(y); ax.set_yticklabels([p[0] for p in P], fontsize=9, color=INK)
ax.set_xlim(-.04, .05); ax.set_ylim(-.7, len(P) - .3); clean(ax)
ax.set_xlabel("예측 오차 차이, 앞 모델 - 뒤 모델  (0보다 왼쪽이면 앞 모델이 나음)", fontsize=9.5, color=GRAY)
ax.set_title("그림 4. 두 모델의 예측 오차 차이와 95% 구간", loc="left", fontsize=11.5, fontweight="bold", color=INK, pad=10)
fig.text(.01, -.06, "구간은 경기 단위 부트스트랩이며 경기 표본의 흔들림만 담는다. 흰 점·점선은 구간이 0을 포함하는 비교. 잔차 앙상블 1·2는 학습 기간만 다르다.",
         fontsize=8, color=GRAY)
fig.savefig(out / "fig4-diff-ci.png"); plt.close(fig)

def table(fname, title, header, rows, widths, note, bold_row=None, h=None):
    n = len(rows) + 1
    fig, ax = plt.subplots(figsize=(7.2, .34 * n)); ax.axis("off")
    t = ax.table(cellText=rows, colLabels=header, colWidths=widths, bbox=[0, 0, 1, 1], cellLoc="left")
    t.auto_set_font_size(False); t.set_fontsize(9)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor(RULE); cell.set_linewidth(.6); cell.PAD = .03
        if r == 0: cell.set_facecolor("#eef0f2"); cell.set_text_props(fontweight="bold", color=INK)
        elif bold_row is not None and r - 1 == bold_row: cell.set_text_props(fontweight="bold", color=ACC)
        if c > 0 and header[c] not in ("그래서", "판정", "결과"): cell._loc = "right"; cell.get_text().set_ha("right")
    ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold", color=INK, pad=8)
    ax.text(0, -.5 / n, note, transform=ax.transAxes, fontsize=8, color=GRAY, va="top")
    fig.savefig(out / fname); plt.close(fig)

h1 = ["구조", "예측 오차(NLL)", "1순위 적중", "3순위 안", "만든 날"]
r1 = [[m[0], f"{m[1]:.5f}", f"{m[2]:.4f}", f"{m[3]:.4f}", m[4]] for m in M]
table("table1-models.png", "표 1. 구조별 성적 (포스트시즌 31경기, 8,943구)", h1, r1, [.50, .16, .13, .11, .10],
      "예측 오차는 낮을수록, 적중은 높을수록 좋다. 31경기는 여러 번 본 자료다. 잔차 앙상블 1·2는 학습 기간만 다르다.", bold_row=0)

h2 = ["비교 (앞 - 뒤)", "차이", "95% 구간", "판정"]
r2 = []
for lab, d, lo, hi in P:
    n = 4 if "와일드카드" in lab else 5
    v = "구분 안 됨" if lo < 0 < hi else ("앞 모델이 나음" if d < 0 else "뒤 모델이 나음")
    r2.append([lab, sg(d, n), f"[{sg(lo, n)}, {sg(hi, n)}]", v])
table("table2-diffs.png", "표 2. 두 모델의 예측 오차 차이", h2, r2, [.52, .10, .22, .16],
      "구간이 0을 포함하면 구분되지 않는다. 와일드카드 8경기는 2,315구, 한 번만 채점. 잔차 앙상블 1·2는 학습 기간만 다르다.")

h3 = ["바꿔 본 것", "결과", "그래서"]
with open(a.design, newline="", encoding="utf-8") as f:
    r3 = [row for row in csv.reader(f)][1:]
table("table3-design.png", "표 3. 설계 실험", h3, r3, [.36, .38, .26],
      "아래 세 줄은 9월 경기 기준이라 31경기 값과 나란히 비교하지 않는다.")

for name, h, rows in (("table1-models.csv", h1, r1), ("table2-diffs.csv", h2, r2), ("table3-design.csv", h3, r3)):
    with open(out / name, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(h); w.writerows(rows)
print("\n".join(sorted(p.name for p in out.iterdir())))
