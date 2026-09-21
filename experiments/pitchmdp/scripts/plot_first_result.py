"""Render observed prediction metrics and model-dependent recommendation figures.

Uses a saved model and the earliest saved real PA. No fitting, external data,
or provider win-expectancy is used. Both PNG outputs remain on the mounted SSD.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import pickle
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
os.environ.setdefault("MPLCONFIGDIR", "/Volumes/T7 Shield/pitcheezy/pitchmdp/cache/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np

from pitchmdp.game import GameState, terminal_values
from pitchmdp.model import PitchModel
from pitchmdp.planner import solve_pa
from pitchmdp.recommend import TARGET_X, TARGET_Z, probability_tensor


PITCH_NAMES = {"FF": "Four-seam", "SI": "Sinker", "FC": "Cutter", "SL": "Slider",
               "ST": "Sweeper", "CU": "Curveball", "KC": "Knuckle curve", "CH": "Changeup",
               "FS": "Splitter", "SV": "Slurve"}
NAVY, GRAY, INK = "#205a78", "#a8b2b9", "#172c3a"


def read_json(path):
    return json.loads(path.read_text())


def draw_target(ax, actions, advantage, pitch_type, norm, best_index):
    values = np.full((len(TARGET_Z), len(TARGET_X)), np.nan)
    for action, value in zip(actions, advantage):
        if action["pitch_type"] == pitch_type:
            values[TARGET_Z.index(action["target_z_ft"]), TARGET_X.index(action["target_x_ft"])] = value
    x_edges = np.array([TARGET_X[0] - .35, *[(a + b) / 2 for a, b in zip(TARGET_X[:-1], TARGET_X[1:])], TARGET_X[-1] + .35])
    z_edges = np.array([TARGET_Z[0] - .4, *[(a + b) / 2 for a, b in zip(TARGET_Z[:-1], TARGET_Z[1:])], TARGET_Z[-1] + .4])
    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad("#eef0f2")
    mesh = ax.pcolormesh(x_edges, z_edges, np.ma.masked_invalid(values), cmap=cmap, norm=norm,
                         edgecolors="white", linewidth=2)
    for iz, z in enumerate(TARGET_Z):
        for ix, x in enumerate(TARGET_X):
            value = values[iz, ix]
            text = "unsupported" if np.isnan(value) else f"{value:+.3f}"
            color = "#79848d" if np.isnan(value) else ("white" if abs(value) > .65 * norm.vmax else INK)
            ax.text(x, z, text, ha="center", va="center", color=color, fontsize=9 if np.isnan(value) else 12,
                    fontweight="normal")
    best = actions[best_index]
    if best["pitch_type"] == pitch_type:
        ax.add_patch(Rectangle((best["target_x_ft"] - .35 + .035, best["target_z_ft"] - .4 + .035),
                               .63, .73, fill=False, edgecolor="#db9800", linewidth=3))
    ax.set_xticks(TARGET_X, [f"{x:+.2f}" for x in TARGET_X])
    ax.set_yticks(TARGET_Z, [f"{z:.2f}" for z in TARGET_Z])
    ax.set_xlabel("Horizontal target (ft, catcher view)")
    ax.set_ylabel("Target height (ft)")
    ax.set_aspect("equal")
    ax.set_title(f"{pitch_type}  |  {PITCH_NAMES.get(pitch_type, pitch_type)}", loc="left", fontweight="bold", pad=12)
    return mesh


def metric_bars(ax, baseline, model, key, label):
    values = [model[key], baseline[key]]
    bars = ax.barh([1, 0], values, color=[NAVY, GRAY], height=.54)
    ax.set_yticks([1, 0], ["Pre-pitch model", "Count + handedness baseline"])
    ax.set_xlim(0, max(values) * 1.21)
    for bar, value in zip(bars, values):
        ax.text(value + max(values) * .025, bar.get_y() + bar.get_height() / 2,
                f"{value:.4f}", va="center", fontsize=11, fontweight="bold")
    ax.set_title(label + "  |  lower is better", loc="left", fontsize=12, fontweight="bold", pad=8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#e6eaed", zorder=0)
    ax.set_axisbelow(True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    local = read_json(PROJECT / "configs/local.json")
    root, run = Path(local["artifact_root"]).resolve(), args.run.resolve()
    if not Path("/Volumes/T7 Shield").is_mount() or not run.is_relative_to(root):
        raise SystemExit("Existing run directory on mounted T7 Shield is required.")
    config = read_json(run / "config.json")
    prediction = read_json(run / "prediction_metrics.json")
    coverage = read_json(run / "coverage.json")
    we_metrics = read_json(run / "we_metrics.json")
    with (run / "recommendation_context.pkl").open("rb") as stream:
        context = pickle.load(stream)
    row = context["rows"].sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"]).iloc[0]
    actions = context["actions"][(int(row.pitcher), str(row.stand))]
    model = PitchModel.load(run / "pitch_model.pt")
    tensor, previous, next_previous, baseline = probability_tensor(model, row, actions, config["control_sigma_ft"])
    state = GameState.from_row(row)
    solution = solve_pa(tensor, terminal_values(state, context["we"], context["advancement"]), next_previous, baseline)
    state_index = (int(row.balls), int(row.strikes), previous.index(str(row.prev_pitch_type)))
    reference = float(solution.baseline_values[state_index])
    advantage = 100 * (solution.q_values[state_index] - reference)
    best = int(np.argmax(advantage))
    pitch_types = sorted({action["pitch_type"] for action in actions},
                         key=lambda pitch: -max(value for action, value in zip(actions, advantage) if action["pitch_type"] == pitch))
    limit = max(float(np.max(np.abs(advantage))), 1e-4)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    names = {int(p["pitcher"]): p["player_name"] for p in read_json(run / "cohort_manifest.json")["selected"]}
    name = names.get(int(row.pitcher), str(int(row.pitcher)))
    if ", " in name:
        last, first = name.split(", ", 1)
        name = first + " " + last
    date = str(row.game_date.date())
    context_label = (f"{name} vs batter {int(row.batter)} ({row.stand})  |  {date}\n"
                     f"{state.half} {state.inning}, {state.outs} outs, bases {state.bases}, "
                     f"home-away {state.home_score}-{state.away_score}, count {int(row.balls)}-{int(row.strikes)}")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
                         "axes.labelcolor": INK, "axes.edgecolor": "#8f9ba4",
                         "xtick.color": INK, "ytick.color": INK,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})

    ncols = min(3, len(pitch_types))
    nrows = math.ceil(len(pitch_types) / ncols)
    figure, axes = plt.subplots(nrows, ncols, figsize=(4.35 * ncols + 1.2, 4.5 * nrows + 2.5), squeeze=False)
    figure.subplots_adjust(left=.07, right=.91, top=.78 if nrows == 1 else .84,
                           bottom=.18 if nrows == 1 else .13, wspace=.32, hspace=.38)
    for ax, pitch_type in zip(axes.flat, pitch_types):
        mesh = draw_target(ax, actions, advantage, pitch_type, norm, best)
    for ax in axes.flat[len(pitch_types):]:
        ax.axis("off")
    cax = figure.add_axes([.935, .25, .017, .43])
    figure.colorbar(mesh, cax=cax, label="Modeled defensive WE advantage (percentage points)")
    figure.suptitle("Target values inside the fitted PA model", x=.07, y=.97, ha="left", fontsize=19, fontweight="bold")
    figure.text(.07, .91 if nrows == 1 else .92, context_label, fontsize=11, ha="left", va="top", linespacing=1.7)
    figure.text(.07, .085 if nrows == 1 else .065,
                f"First chronological saved PA; gold outline = best action. Gaussian control sigma = {config['control_sigma_ft']:.2f} ft.\n"
                "Value: chosen first pitch plus optimal PA continuation. Reference: training type usage, uniform supported targets.\n"
                "Intent is unobserved. Cells pass a training delivery-proximity filter. These values do not establish real policy gains.",
                fontsize=9, va="center", linespacing=1.6)
    figure.savefig(run / "heatmap.png", dpi=180)
    plt.close(figure)

    figure = plt.figure(figsize=(15, 8.2))
    grid = figure.add_gridspec(3, 2, width_ratios=[1.35, 1.1], height_ratios=[1, 1, .65],
                              left=.18, right=.94, top=.75, bottom=.19, wspace=.43, hspace=.8)
    primary, count = prediction["primary_prepitch_type_conditional"], prediction["count_hand_baseline"]
    metric_bars(figure.add_subplot(grid[0, 0]), count, primary, "log_loss", "Pitch-outcome log loss")
    metric_bars(figure.add_subplot(grid[1, 0]), count, primary, "brier_multiclass", "Multiclass Brier score")
    ax = figure.add_subplot(grid[2, 0])
    ax.axis("off")
    ax.text(0, .8, f"Game-winner continuation: {we_metrics['n']:,} PA starts / {we_metrics['games']:,} games\n"
            f"Log loss {we_metrics['log_loss']:.4f}  |  Binary Brier {we_metrics['brier']:.4f}",
            transform=ax.transAxes, va="top", fontsize=10, linespacing=1.7)
    ax = figure.add_subplot(grid[:, 1])
    mesh = draw_target(ax, actions, advantage, pitch_types[0], norm, best)
    figure.colorbar(mesh, ax=ax, location="right", fraction=.05, pad=.06, shrink=.85,
                    label="Model-internal advantage (pp)")
    figure.suptitle("First real-data result  |  PitchMDP", x=.05, y=.96, ha="left", fontsize=22, fontweight="bold")
    figure.text(.05, .905, "Prediction is measured on later observed data. Target values are a separate, model-dependent planning diagnostic.", fontsize=12)
    figure.text(.05, .85, "OBSERVED PREDICTION", fontsize=13, fontweight="bold", color=NAVY)
    figure.text(.05, .805, f"{coverage['dev_dates'][0]} to {coverage['dev_dates'][1]}  |  {primary['n']:,} pitches  |  {coverage['dev_games']} games",
                fontsize=11)
    figure.text(.615, .85, "MODELED TARGET VALUE", fontsize=13, fontweight="bold", color="#8d5815")
    figure.text(.615, .805, f"{name}, {date}  |  batter {int(row.batter)}", fontsize=11)
    figure.text(.05, .105,
                f"Candidate targets assume Gaussian control sigma = {config['control_sigma_ft']:.2f} ft; intended targets are absent from the source.\n"
                "Reference policy: training pitch-type usage and uniform supported target cells. Gold outline: best action for this saved PA.\n"
                "A modeled advantage is not an estimate of a causal win-rate improvement. Temporal DEV was previously used in the broader project.",
                fontsize=10, linespacing=1.7, va="center")
    figure.savefig(run / "first_result.png", dpi=180)
    plt.close(figure)
    print(json.dumps({"first_result": str(run / "first_result.png"), "heatmap": str(run / "heatmap.png"),
                      "game_pk": int(row.game_pk), "at_bat_number": int(row.at_bat_number),
                      "best_action": actions[best], "best_advantage_pp": float(advantage[best])}, indent=2))


if __name__ == "__main__":
    main()
