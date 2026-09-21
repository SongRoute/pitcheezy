"""Plot archived integration sensitivity and CAL-only predictive mixtures."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--integration', type=Path, required=True)
    parser.add_argument('--context25', type=Path, required=True)
    parser.add_argument('--blend100', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path('/Volumes/T7 Shield/pitcheezy/pitchmdp').resolve()
    if not Path('/Volumes/T7 Shield').is_mount() or not args.output.resolve().is_relative_to(root):
        raise SystemExit('Write only under mounted configured SSD')
    paths = [args.integration/'convergence.json', args.integration/'calibration100/results.json',
             args.context25/'results.json', args.blend100/'results.json']
    convergence, calibration, context, blended = [read(path) for path in paths]
    assert read(args.integration/'runtime.json')['integrity_ok']
    metrics = ['log_loss', 'brier_multiclass']
    # Named rows are fixed by the experiment design, never selected on DEV scores.
    rows = [('Count / hand baseline', calibration['count_hand']),
            ('Type + count + game-state baseline', context['baselines']['league_context']['metrics']['tempered']),
            ('Transformer42 / 25 draws', convergence['draws']['25']['delivery_integrated_calibrated']),
            ('Transformer42 / 100 draws', convergence['draws']['100']['delivery_integrated_calibrated']),
            ('Transformer42 / 400 draws', convergence['draws']['400']['delivery_integrated_calibrated']),
            ('Five-Transformer ensemble / 100 draws', calibration['ensemble']),
            ('CAL log-loss blend / 25 draws', context['blends']['context_blend_log_loss']['metrics']),
            ('CAL log-loss blend / 100 draws', blended['metrics']['context100_blend_log_loss']),
            ('CAL Brier blend / 100 draws', blended['metrics']['context100_blend_brier_multiclass'])]
    for _, values in rows:
        assert values['n'] == 7276
        assert all(np.isfinite(values[key]) for key in metrics)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5),
                             gridspec_kw={'height_ratios': [1, 1.65]})
    fig.subplots_adjust(left=.285, right=.97, top=.83, bottom=.18, hspace=.62, wspace=.32)
    colors = ['#64748b', '#64748b', '#3b82a6', '#3b82a6', '#3b82a6', '#8055a0', '#b66b29', '#168576', '#168576']
    for column, metric in enumerate(metrics):
        ax = axes[0, column]
        for name, label, color in [('delivery_integrated_uncalibrated', 'Uncalibrated', '#94a3b8'),
                                    ('delivery_integrated_calibrated', 'CAL temperature', '#256c92')]:
            ax.plot([25, 100, 400], [convergence['draws'][str(draw)][name][metric] for draw in [25, 100, 400]],
                    marker='o', color=color, linewidth=1.8, label=label)
        ax.set_xscale('log', base=4)
        ax.set_xticks([25, 100, 400], ['25', '100', '400'])
        ax.set_xlabel('TRAIN delivery draws; pools are not nested')
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.set_title(('A  ' if column == 0 else 'B  ') + ('Log loss' if column == 0 else 'Brier') + ' sensitivity', loc='left', weight='bold')
        ax.grid(alpha=.16)
        if column == 0:
            ax.legend(frameon=False, fontsize=9)
        ax = axes[1, column]
        values = np.array([item[metric] for _, item in rows])
        y = np.arange(len(rows))
        ax.scatter(values, y, c=colors, s=43, zorder=3)
        span = max(float(np.ptp(values)), .001)
        ax.set_xlim(values.min()-.16*span, values.max()+.32*span)
        for i, value in enumerate(values):
            ax.annotate(f'{value:.6f}', (value, i), xytext=(7, 0), textcoords='offset points', va='center', fontsize=8.5, color=colors[i])
        ax.set_yticks(y, [label for label, _ in rows] if column == 0 else ['']*len(rows))
        ax.set_ylim(len(rows)-.5, -.5)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.tick_params(axis='y', length=0, labelsize=9.3)
        ax.grid(axis='x', alpha=.16)
        ax.set_xlabel('DEV score; lower is better')
        ax.set_title(('C  Log loss' if column == 0 else 'D  Brier') + ' comparison', loc='left', weight='bold', pad=12)
    fig.text(.055, .95, 'Delivery integration and calibrated game-state mixtures', fontsize=22, weight='bold', color='#243348')
    fig.text(.055, .91, '7,276 pitches / 87 games | inspected 2025 DEV | fixed checkpoint weights | no DEV-selected blend weights', fontsize=11, color='#64748b')
    fig.text(.055, .12, 'Top: the same Transformer seed42 at three fixed integration budgets; changes include sampling realization. Not a convergence proof.', fontsize=9.5, color='#64748b')
    fig.text(.055, .086, 'Bottom: fixed descriptive rows, with both CAL blend objectives retained. Points are estimates, without uncertainty bars.', fontsize=9.5, color='#64748b')
    fig.text(.055, .052, 'Forecasting improvements do not establish real policy benefit. Policy examples use their separately archived control assumptions.', fontsize=9.5, color='#64748b')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, facecolor='white')
    plt.close(fig)
    metadata = {'rows': [{'label': label, **{key: values[key] for key in metrics}} for label, values in rows],
                'source_hashes': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
                'plotter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(args.output)


if __name__ == '__main__':
    main()
