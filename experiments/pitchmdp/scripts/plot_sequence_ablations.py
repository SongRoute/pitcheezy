"""Plot completed exploratory context ablations using saved scores only."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
import numpy as np

INK, MUTED, GRID = "#202F43", "#63748A", "#E5EBF2"
NAMES = {"no_game_context": "Remove game context", "no_batter_style": "Remove batter style",
         "no_game_or_batter": "Remove both groups"}
COLORS = ["#168576", "#798BA5", "#477ED1", "#A4AFBE"]


def plot(folder):
    results = json.loads((folder/"ablation_results.json").read_text())
    data = json.loads((folder/"data.json").read_text())
    config = json.loads((folder/"config.json").read_text())
    variants = config["variants"]
    if not {"no_game_context", "no_batter_style"}.issubset(results["variants"]) or not set(variants).issubset(results["variants"]):
        raise ValueError("Both requested feature-group ablations must be finished")
    scores = [results["full_transformer"]] + [results["variants"][key]["primary_delivery_integrated"] for key in variants]
    comparisons = [results["variants"][key]["paired_ablation_minus_full"] for key in variants]
    if any(score["n"] != data["dev_rows"] for score in scores) or any(item["games"] != data["dev_games"] for item in comparisons):
        raise ValueError("Scores do not share the saved pitch/game sample")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "figure.facecolor": "#F8FAFD", "axes.facecolor": "#F8FAFD",
                         "savefig.facecolor": "#F8FAFD"})
    fig = plt.figure(figsize=(14, 6.3))
    grid = fig.add_gridspec(1, 2, left=.17, right=.965, bottom=.29, top=.70, wspace=.75)
    fig.text(.045, .935, "What do game context and batter style add?", fontsize=23, weight="bold", color=INK)
    fig.text(.045, .865, f"Exploratory after DEV inspection  |  One training seed  |  "
             f"{data['dev_games']:,} games / {data['dev_rows']:,} eligible pitches", fontsize=11, color=MUTED)
    axes = [fig.add_subplot(grid[0, i]) for i in range(2)]
    values = np.array([score["log_loss"] for score in scores])
    axes[0].barh(np.arange(len(scores)), values, color=COLORS[:len(scores)], height=.5)
    axes[0].set_yticks(np.arange(len(scores)), ["Full Transformer"] + [NAMES[key] for key in variants])
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, values.max()*1.20)
    for y, value in enumerate(values):
        axes[0].text(value+values.max()*.025, y, f"{value:.5f}", va="center", fontsize=10, color=INK)
    axes[0].set_title("A  Calibrated all-count log loss", loc="left", fontsize=13, weight="bold", color=INK, pad=35)
    axes[0].text(0, 1.07, "Current delivery integrated · lower is better", transform=axes[0].transAxes,
                 fontsize=9.5, color=MUTED, va="bottom")
    axes[0].set_xlabel("Same rows, architecture, parameter count and seed", fontsize=9, color=MUTED, labelpad=12)
    intervals = np.array([item["bootstrap95"] for item in comparisons])
    axes[1].axvline(0, color=MUTED, linestyle=(0, (3, 3)), linewidth=1)
    for y, item in enumerate(comparisons):
        value = item["model_minus_reference_log_loss"]
        low, high = item["bootstrap95"]
        axes[1].plot([low, high], [y, y], color=COLORS[y+1], linewidth=3, solid_capstyle="round")
        axes[1].scatter([value], [y], color=COLORS[y+1], s=60, zorder=3)
        axes[1].text(value, y+.2, f"{value:+.4f} [{low:+.4f}, {high:+.4f}]",
                     ha="center", va="top", fontsize=9, color=INK)
    axes[1].set_yticks(np.arange(len(variants)), [NAMES[key] for key in variants])
    axes[1].set_ylim(len(variants)-.35, -.5)
    extent = max(float(np.abs(intervals).max()), .001)
    axes[1].set_xlim(min(0., float(intervals.min()))-extent*.25, max(0., float(intervals.max()))+extent*.45)
    axes[1].xaxis.set_major_formatter(StrMethodFormatter("{x:+.3f}"))
    axes[1].set_title("B  Cost of removing each group", loc="left", fontsize=13, weight="bold", color=INK, pad=35)
    axes[1].text(0, 1.07, "Ablation minus full · 95% game-bootstrap CI", transform=axes[1].transAxes,
                 fontsize=9.5, color=MUTED, va="bottom")
    axes[1].set_xlabel("Positive = full features helped prediction", fontsize=10, color=MUTED, labelpad=12)
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(axis="both", length=0, labelsize=10, colors=MUTED, pad=7)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.grid(axis="x", color=GRID, linewidth=.8)
        ax.set_axisbelow(True)
    fig.text(.045, .165, "Game removal: outs, inning, score, home/away and bases. "
             "Style removal: six rates, six reliability values and five soft memberships.", fontsize=9.2, color=MUTED)
    fig.text(.045, .119, "Both retain balls, strikes, batting/pitching handedness and physical history. "
             "Separate calibration temperatures are fitted without DEV selection.", fontsize=9.2, color=MUTED)
    fig.text(.045, .073, f"{comparisons[0]['replicates']:,} paired game resamples; fitted models held fixed. "
             "Intervals omit training uncertainty. No causal or win-rate improvement claim.", fontsize=9.2, color=MUTED)
    fig.text(.045, .031, "Source: ablation_results.json and matched data.json. "
             "This post-inspection, single-seed comparison needs independent confirmation.", fontsize=9, color=MUTED)
    output = folder/"sequence_ablations.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablations", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(plot(args.ablations))
    except (FileNotFoundError, KeyError, ValueError) as error:
        raise SystemExit(f"Cannot plot completed ablations: {error}") from error
