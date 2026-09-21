"""Plot saved multi-seed results and print compact progress; no model/data access."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
import numpy as np


INK, MUTED, GRID = "#202F43", "#63748A", "#E5EBF2"
LABELS = {"full_transformer": "Full Transformer", "flatten_mlp": "Flattened MLP",
          "no_game_context": "Remove game context", "no_batter_style": "Remove batter style",
          "capacity_mlp": "Capacity-matched MLP", "no_clusters": "Remove soft clusters"}
COLORS = {"full_transformer": "#168576", "flatten_mlp": "#477ED1", "no_game_context": "#798BA5",
          "no_batter_style": "#9676AC", "capacity_mlp": "#356BAF", "no_clusters": "#C08C40"}
PHASES = {"replication": ("full_transformer", "flatten_mlp", "no_game_context", "no_batter_style"),
          "capacity": ("full_transformer", "capacity_mlp"), "clusters": ("full_transformer", "no_clusters")}
CONTRASTS = {"flatten_mlp": ("full_transformer_minus_flatten_mlp", -1),
             "no_game_context": ("no_game_context_minus_full_transformer", 1),
             "no_batter_style": ("no_batter_style_minus_full_transformer", 1),
             "capacity_mlp": ("full_transformer_minus_capacity_mlp", -1),
             "no_clusters": ("no_clusters_minus_full_transformer", 1)}
TITLES = {"replication": "Model architecture and feature groups",
          "capacity": "Capacity control: flattened MLP width 234",
          "clusters": "Incremental value of five soft memberships"}


def compact_summary(summary):
    """One row per model/contrast, with no class-rate or training-history output."""
    out = {"n": summary["n"], "games": summary["games"], "models": {}, "comparisons": {}}
    for key, entry in summary["variants"].items():
        metrics = entry["metrics"]["delivery_integrated_calibrated"]
        out["models"][key] = {"seeds": entry["completed_seeds"], "log_loss": metrics["log_loss"]["mean"],
                               "brier": metrics["brier_multiclass"]["mean"], "seed_ll": metrics["log_loss"]["per_seed"]}
    for key, entry in summary["comparisons"].items():
        metric = entry["log_loss"]
        out["comparisons"][key] = {"seeds": entry["seeds"], "delta_ll": metric["model_minus_reference"],
                                    "crossed95": metric["crossed_seed_game_bootstrap95"], "complete": entry["complete"]}
    return out


def plot_phase(summary, phase, output):
    variants = [key for key in PHASES[phase] if key in summary["variants"]]
    if len(variants) < 2:
        return None
    expected = summary["expected_seeds"]
    complete = all(key in summary["variants"] and summary["variants"][key]["completed_seeds"] == expected
                   for key in PHASES[phase])
    status = "COMPLETE · 5 fixed seeds" if complete else "PARTIAL · completed seeds only"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "figure.facecolor": "#F8FAFD", "axes.facecolor": "#F8FAFD", "savefig.facecolor": "#F8FAFD"})
    fig = plt.figure(figsize=(14.4, 7.5))
    grid = fig.add_gridspec(1, 2, left=.082, right=.97, bottom=.32, top=.70, wspace=.84)
    left, right = (fig.add_subplot(grid[0, i]) for i in range(2))
    fig.text(.045, .945, TITLES[phase], fontsize=23, weight="bold", color=INK)
    fig.text(.045, .885, f"{status}  |  {summary['games']:,} games / {summary['n']:,} eligible pitches",
             fontsize=11, color=INK if complete else "#A16A1C")
    fig.text(.045, .84, "Previously inspected DEV · exploratory robustness, not unseen confirmation", fontsize=10.5, color=MUTED)
    for offset, key in zip(np.linspace(-.12, .12, len(variants)), variants):
        entry = summary["variants"][key]
        scores = entry["metrics"]["delivery_integrated_calibrated"]["log_loss"]
        seeds = sorted(int(seed) for seed in scores["per_seed"])
        values = [scores["per_seed"][str(seed)] for seed in seeds]
        color = COLORS[key]
        left.plot(np.asarray(seeds)+offset, values, color=color, alpha=.45, linewidth=1, zorder=2)
        left.scatter(np.asarray(seeds)+offset, values, color=color, s=42, zorder=3,
                     label=f"{LABELS[key]}  ({len(seeds)}/5)\nmean {scores['mean']:.5f}")
    left.set_xticks(expected)
    left.set_xlim(min(expected)-.4, max(expected)+.4)
    left.set_xlabel("Model seed · sample, encoder and delivery seed fixed at 42", fontsize=9, color=MUTED, labelpad=10)
    left.set_ylabel("Calibrated log loss · zoomed scale", fontsize=10, color=MUTED, labelpad=10)
    left.yaxis.set_major_locator(MaxNLocator(5))
    left.yaxis.set_major_formatter(StrMethodFormatter("{x:.4f}"))
    left.legend(loc="upper left", bbox_to_anchor=(-.04, -.21), ncol=2, frameon=False,
                fontsize=8.6, labelcolor=INK, columnspacing=1.4, handletextpad=.3)
    left.set_title("A  Individual model scores", loc="left", color=INK, weight="bold", fontsize=13, pad=23)
    rows, extents = [], [0.]
    for key in variants:
        if key == "full_transformer":
            continue
        name, sign = CONTRASTS[key]
        if name not in summary["comparisons"]:
            continue
        entry = summary["comparisons"][name]
        metric = entry["log_loss"]
        value = sign*metric["model_minus_reference"]
        low, high = sorted(sign*np.asarray(metric["crossed_seed_game_bootstrap95"]))
        row = len(rows)
        right.plot([low, high], [row, row], color=COLORS[key], linewidth=3, solid_capstyle="round")
        right.scatter([value], [row], color=COLORS[key], s=55, zorder=3)
        right.text(value, row+.17, f"{value:+.4f} [{low:+.4f}, {high:+.4f}]",
                   ha="center", va="top", fontsize=9, color=INK)
        rows.append(f"{LABELS[key]} − full\n{len(entry['seeds'])}/5 matched seeds")
        extents.extend([low, high])
    right.axvline(0, color=MUTED, linestyle=(0, (3, 3)), linewidth=1)
    extent = max(max(extents)-min(extents), .001)
    right.set_xlim(min(extents)-extent*.28, max(extents)+extent*.4)
    right.set_ylim(max(len(rows)-.3, .7), -.5)
    right.set_yticks(np.arange(len(rows)), rows)
    right.xaxis.set_major_locator(MaxNLocator(5))
    right.xaxis.set_major_formatter(StrMethodFormatter("{x:+.3f}"))
    right.set_xlabel("Positive favors full Transformer", fontsize=10, color=MUTED, labelpad=10)
    right.set_title("B  Paired difference · crossed 95% CI", loc="left", color=INK, weight="bold", fontsize=13, pad=23)
    for axis in (left, right):
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.tick_params(axis="both", length=0, colors=MUTED, pad=7)
        axis.grid(axis="y" if axis is left else "x", color=GRID, linewidth=.8)
        axis.set_axisbelow(True)
    fig.text(.045, .116, "Primary score = mean of individual-model per-pitch losses; probabilities are not ensembled. "
             "Comparisons use only matched completed seed IDs.", fontsize=9, color=MUTED)
    fig.text(.045, .078, "2,000 paired resamples of whole games and model seeds. Same games are shared across models and seeds; "
             "five seeds give limited uncertainty coverage.", fontsize=9, color=MUTED)
    note = {"replication": "Game and style removals keep balls, strikes, handedness and the physical sequence; all context paths retain 28 inputs.",
            "capacity": "Capacity MLP: 278,874 parameters; full Transformer: 278,762. Width fixed before fitting; no DEV hyperparameter search.",
            "clusters": "Cluster removal zeros context channels 23–27; all six continuous style rates and six reliability values remain."}[phase]
    fig.text(.045, .04, note, fontsize=9, color=MUTED)
    destination = output/f"robustness_{phase}.png"
    fig.savefig(destination, dpi=180)
    plt.close(fig)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Existing robustness output directory")
    parser.add_argument("--phase", choices=["all", *PHASES], default="all")
    parser.add_argument("--summary-only", action="store_true", help="Print compact metrics without drawing")
    args = parser.parse_args()
    summary = json.loads((args.output/"summary.json").read_text())
    if args.summary_only:
        print(json.dumps(compact_summary(summary), indent=2))
        return
    for phase in PHASES if args.phase == "all" else [args.phase]:
        result = plot_phase(summary, phase, args.output)
        if result is not None:
            print(result)


if __name__ == "__main__":
    main()
