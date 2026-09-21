"""Plot completed sequence experiments without mixing their evaluation tasks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
import numpy as np


ORDER = ("count_hand_baseline", "current_only", "flatten_mlp", "transformer")
LABELS = {"count_hand_baseline": "Count + hand baseline", "current_only": "Current-only network",
          "flatten_mlp": "Flattened-history MLP", "transformer": "Sequence Transformer"}
COLORS = {"count_hand_baseline": "#A4AFBE", "current_only": "#798BA5",
          "flatten_mlp": "#477ED1", "transformer": "#168576"}
INK, MUTED, GRID = "#202F43", "#63748A", "#E5EBF2"


def _style(ax):
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", which="both", length=0, labelsize=10, colors=MUTED, pad=7)
    ax.grid(axis="x", color=GRID, linewidth=.8)
    ax.set_axisbelow(True)


def _heading(ax, title, subtitle):
    ax.set_title(title, loc="left", fontsize=13, weight="bold", color=INK, pad=35)
    ax.text(0, 1.07, subtitle, transform=ax.transAxes, fontsize=9.4, color=MUTED, va="bottom")


def _metric_bars(ax, metrics, key, title, subtitle):
    values = np.array([metrics[k][key] for k in ORDER])
    positions = np.arange(len(ORDER))
    ax.barh(positions, values, height=.52, color=[COLORS[k] for k in ORDER])
    ax.set_yticks(positions, [LABELS[k] for k in ORDER])
    ax.set_xlim(0, values.max()*1.17)
    ax.invert_yaxis()
    for y, value in zip(positions, values):
        ax.text(value+values.max()*.025, y, f"{value:.4f}", va="center", fontsize=10, color=INK)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    _style(ax)
    _heading(ax, title, subtitle)


def build_figure(all_count: dict, binary: dict, samples: dict):
    """Require the finished result schema; never publish a partial comparison."""
    required = set(ORDER) | {"paired_comparisons"}
    missing = required.difference(all_count)
    if missing:
        raise ValueError(f"All-count results are incomplete: missing {sorted(missing)}")
    metrics = {key: all_count[key] if key == "count_hand_baseline" else
               all_count[key]["primary_delivery_integrated"] for key in ORDER}
    n = int(samples["dev_rows"])
    if any(int(metric["n"]) != n for metric in metrics.values()):
        raise ValueError("All-count models do not share the saved DEV sample")
    comparisons = all_count["paired_comparisons"]
    references = ("flatten_mlp", "current_only", "count_hand_baseline")
    if any(int(comparisons[key]["games"]) != int(samples["dev_games"]) for key in references):
        raise ValueError("Paired comparisons disagree with the saved DEV game count")
    binary_metrics = {key: binary[key]["metrics_conditional_current_physics"]
                      for key in ("flatten_mlp", "transformer")}
    if len({int(metric["n"]) for metric in binary_metrics.values()}) != 1:
        raise ValueError("Binary models do not share an evaluation sample")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "figure.facecolor": "#F8FAFD", "axes.facecolor": "#F8FAFD",
                         "savefig.facecolor": "#F8FAFD"})
    fig = plt.figure(figsize=(14.4, 9.7))
    grid = fig.add_gridspec(2, 2, left=.175, right=.962, bottom=.19, top=.80,
                           hspace=.78, wspace=.62)
    dates = [str(value).split(" ")[0] for value in samples["dev_dates"]]
    fig.text(.05, .948, "Does sequence attention improve held-out prediction?",
             fontsize=22, weight="bold", color=INK)
    fig.text(.05, .906,
             f"Frozen cohort DEV  |  {samples['dev_games']:,} games / {n:,} eligible pitches  |  "
             f"{dates[0]} to {dates[1]}", fontsize=11, color=MUTED)
    _metric_bars(fig.add_subplot(grid[0, 0]), metrics, "log_loss", "A  All-count log loss",
                 "10 outcomes · current physics integrated · lower is better")
    _metric_bars(fig.add_subplot(grid[0, 1]), metrics, "brier_multiclass", "B  All-count Brier score",
                 "Multiclass sum of squared errors · lower is better")

    ax = fig.add_subplot(grid[1, 0])
    means = np.array([comparisons[key]["model_minus_reference_log_loss"] for key in references])
    intervals = np.array([comparisons[key]["bootstrap95"] for key in references])
    ax.axvline(0, color=MUTED, linestyle=(0, (3, 3)), linewidth=1)
    for y, (mean, (low, high)) in enumerate(zip(means, intervals)):
        ax.plot([low, high], [y, y], color=COLORS["transformer"], linewidth=3, solid_capstyle="round")
        ax.scatter([mean], [y], color=COLORS["transformer"], s=55, zorder=3)
        ax.text(mean, y+.23, f"{mean:+.4f} [{low:+.4f}, {high:+.4f}]",
                ha="center", va="top", fontsize=8.5, color=INK)
    ax.set_yticks(np.arange(3), [LABELS[key] for key in references])
    ax.set_ylim(2.65, -.5)
    extent = max(float(np.abs(intervals).max()), .001)*1.55
    ax.set_xlim(-extent, extent)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:+.3f}"))
    ax.set_xlabel("Negative favors Transformer", color=MUTED, fontsize=10, labelpad=9)
    _style(ax)
    _heading(ax, "C  Paired log-loss differences",
             "Transformer minus reference · 95% game-bootstrap CI")

    ax = fig.add_subplot(grid[1, 1])
    for y, key in enumerate(binary_metrics):
        auc = binary_metrics[key]["auc"]
        ax.barh(y, auc-.5, left=.5, height=.4, color=COLORS[key])
        ax.text(auc+.012, y, f"{auc:.4f}", va="center", fontsize=10, color=INK)
    ax.set_yticks([0, 1], [LABELS[key] for key in binary_metrics])
    ax.set_ylim(1.75, -.55)
    ax.set_xlim(.5, 1.)
    ax.set_xticks([.5, .6, .7, .8, .9, 1.])
    ax.set_xlabel("ROC AUC · higher is better · axis starts at 0.50", color=MUTED, fontsize=9, labelpad=9)
    _style(ax)
    binary_n = int(binary_metrics["transformer"]["n"])
    _heading(ax, "D  Separate conditional binary task",
             f"Observed current physics · selected 2-strike terminals · n = {binary_n:,}")
    ax.text(0, -.24, "Swinging strikeout vs fair ball in play.\n"
            "This selected binary task is not all-count forecasting.", transform=ax.transAxes,
            fontsize=9.2, color=MUTED, va="top", linespacing=1.45)

    replicates = comparisons["flatten_mlp"]["replicates"]
    fig.text(.05, .080,
             f"A–C: pre-pitch evaluation integrates TRAIN delivery samples; restricted eligible PAs.  "
             f"C: {replicates:,} paired game resamples, fitted models held fixed.",
             color=MUTED, fontsize=9)
    fig.text(.05, .054,
             "One training seed; intervals omit training uncertainty. Binary and 10-class losses are not comparable. "
             "Predictive scores do not establish a policy or win-rate gain.", color=MUTED, fontsize=9)
    fig.text(.05, .027,
             "Source: saved all_count_results.json, binary_results.json, samples.json  |  "
             "TRAIN ends 2025-04-30; calibration May–June 2025.", color=MUTED, fontsize=8.6)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    paths = [args.run/name for name in ("all_count_results.json", "binary_results.json", "samples.json")]
    try:
        result = build_figure(*(json.loads(path.read_text()) for path in paths))
    except (FileNotFoundError, KeyError, ValueError) as error:
        raise SystemExit(f"Cannot plot completed comparison: {error}") from error
    output = args.run/"sequence_results.png"
    result.savefig(output, dpi=180)
    plt.close(result)
    print(output)


if __name__ == "__main__":
    main()
