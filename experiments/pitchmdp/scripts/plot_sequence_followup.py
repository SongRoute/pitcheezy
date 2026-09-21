"""Plot frozen follow-up DEV metrics without refitting or selecting model seeds."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PROJECT = Path(__file__).resolve().parents[1]
METRICS = ('log_loss', 'brier_multiclass')
STYLES = {
    'fixed': ('#59616b', 's', 'Fixed baseline'),
    'seed_mean': ('#126d91', 'o', 'Mean of five single-model losses'),
    'ensemble': ('#7654a1', 'D', 'Probability ensemble'),
    'blend': ('#b35b22', '^', 'CAL-fitted probability blend'),
}


def read_json(path):
    return json.loads(Path(path).read_text())


def metric_pair(metrics, expected_n, name):
    if metrics.get('n') != expected_n:
        raise ValueError(f'{name}: DEV sample count missing or different from {expected_n}')
    values = {metric: float(metrics[metric]) for metric in METRICS}
    if not all(np.isfinite(v) and v >= 0 for v in values.values()):
        raise ValueError(f'{name}: missing or invalid scoring-rule values')
    return values


def collect_rows(robustness, calibration, frequency, strong_blends=None):
    """Require complete original-outcome scores; never infer missing plot values."""
    seeds = robustness['expected_seeds']
    if len(seeds) != 5 or sorted(seeds) != sorted(calibration['seeds']):
        raise ValueError('Figure requires the same five fixed seeds in robustness and ensemble results')
    n = robustness['n']
    rows = []

    def add(label, kind, metrics, name, **extra):
        rows.append({'label': label, 'kind': kind, 'source_key': name,
                     **metric_pair(metrics, n, name), **extra})

    add('Count + hand\nfixed baseline', 'fixed', calibration['count_hand'], 'calibration.count_hand')
    selected = frequency['chosen_by_calibration']
    fixed = frequency['baselines'][selected]
    training = 'full TRAIN' if selected.startswith('full_train__') else 'matched TRAIN sample'
    frequency_name = 'Type + pitcher frequency' if selected.endswith('type_pitcher') else 'Type frequency'
    add(f'{frequency_name} ({training})\nfixed baseline; CAL-selected', 'fixed', fixed['metrics']['tempered'],
        f'frequency.baselines.{selected}.metrics.tempered')
    # Pre-specified additional descriptive comparator, not a DEV-selected winner.
    # Its two scoring-rule values are always shown together.
    descriptive_key = 'full_train__type_pitcher'
    if descriptive_key != selected:
        add('Type + pitcher frequency (full TRAIN)\ndescriptive fixed baseline', 'fixed',
            frequency['baselines'][descriptive_key]['metrics']['tempered'],
            f'frequency.baselines.{descriptive_key}.metrics.tempered')
    for key, label in [('full_transformer', 'Transformer'), ('flatten_mlp', 'Small flattened MLP'),
                       ('capacity_mlp', 'Capacity-matched MLP')]:
        variant = robustness['variants'][key]
        if not variant.get('complete') or sorted(variant['completed_seeds']) != sorted(seeds):
            raise ValueError(f'{key}: all five planned seeds must be complete before plotting')
        values = variant['metrics']['delivery_integrated_calibrated']
        means, sd = {}, {}
        for metric in METRICS:
            item = values[metric]
            if sorted(map(int, item['per_seed'])) != sorted(seeds):
                raise ValueError(f'{key}: {metric} does not contain the fixed five seeds')
            mean, std = float(item['mean']), float(item['seed_std'])
            if not np.isfinite(mean) or not np.isfinite(std) or min(mean, std) < 0:
                raise ValueError(f'{key}: invalid seed mean/SD')
            if not np.isclose(mean, np.mean(list(item['per_seed'].values())), atol=1e-10, rtol=0):
                raise ValueError(f'{key}: saved mean differs from equal-seed mean')
            means[metric], sd[metric] = mean, std
        rows.append({'label': f'{label}\nmean of five single-model losses', 'kind': 'seed_mean',
                     'source_key': f'robustness.variants.{key}', **means, 'seed_sd': sd})
    add('Transformer ensemble\nmean probabilities across five seeds', 'ensemble', calibration['ensemble'], 'calibration.ensemble')
    for objective, short in [('log_loss', 'log-loss'), ('brier_multiclass', 'Brier')]:
        blend = calibration['blends'][objective]
        weight = float(blend['model_weight'])
        add(f'Ensemble + count\nCAL {short} blend (T weight {weight:.3f})', 'blend', blend['dev_metrics'],
            f'calibration.blends.{objective}.dev_metrics', transformer_weight=weight)
    if strong_blends is not None:
        # Refuse to combine files describing different reference scores/cohorts.
        for name, reference in [('ensemble', calibration['ensemble']), ('old_count_baseline', calibration['count_hand']),
                                ('selected_frequency_baseline', fixed['metrics']['tempered'])]:
            got = metric_pair(strong_blends['metrics'][name], n, f'strong_blends.metrics.{name}')
            if not all(np.isclose(got[m], reference[m], atol=1e-9, rtol=0) for m in METRICS):
                raise ValueError(f'Strong-blend reference {name} differs from the supplied comparison inputs')
        for objective, short in [('log_loss', 'log-loss'), ('brier_multiclass', 'Brier')]:
            fit = strong_blends['selection']['fitted_blends'][objective]
            weight = float(fit['model_weight'])
            key = f'strong_blend_{objective}'
            add(f'Ensemble + type frequency\nCAL {short} blend (T weight {weight:.3f})', 'blend', strong_blends['metrics'][key],
                f'strong_blends.metrics.{key}', transformer_weight=weight)
    return rows


def make_figure(rows, n, games):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.spines.left': False, 'axes.edgecolor': '#a5adb5',
                         'text.color': '#202a35', 'axes.labelcolor': '#202a35',
                         'xtick.color': '#45515f', 'ytick.color': '#202a35'})
    height = max(7.1, .59 * len(rows) + 2.5)
    fig, axes = plt.subplots(1, 2, figsize=(14.4, height), sharey=True)
    fig.subplots_adjust(left=.305, right=.975, bottom=.17, top=.79, wspace=.28)
    y = np.arange(len(rows))
    for ax, metric, title in zip(axes, METRICS, ['Log loss  (lower is better)', 'Multiclass Brier  (lower is better)']):
        values = np.array([r[metric] for r in rows])
        deviation = np.array([r.get('seed_sd', {}).get(metric, 0.) for r in rows])
        lo, hi = float(np.min(values - deviation)), float(np.max(values + deviation))
        span = max(hi - lo, .001)
        ax.set_xlim(lo - .09 * span, hi + .40 * span)
        for i, row in enumerate(rows):
            if i % 2 == 0:
                ax.axhspan(i - .49, i + .49, color='#f3f5f7', zorder=0)
            color, marker, _ = STYLES[row['kind']]
            if 'seed_sd' in row:
                ax.errorbar(row[metric], i, xerr=row['seed_sd'][metric], fmt='none',
                            ecolor=color, elinewidth=1.5, capsize=3, zorder=2)
            ax.scatter(row[metric], i, marker=marker, color=color, s=54, edgecolors='white', linewidths=.7, zorder=3)
            ax.text(.985, i, f'{row[metric]:.6f}', transform=ax.get_yaxis_transform(),
                    ha='right', va='center', fontsize=9.2, color=color)
        ax.set_title(title, fontsize=12, fontweight='bold', loc='left', pad=13)
        ax.set_ylim(len(rows) - .5, -.5)
        ax.set_yticks(y)
        ax.tick_params(axis='y', length=0, pad=13)
        ax.grid(axis='x', color='#dde2e7', linewidth=.7, zorder=0)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
        ax.set_xlabel('DEV score', labelpad=8)
    axes[0].set_yticklabels([r['label'] for r in rows], fontsize=9.8, linespacing=1.35)
    axes[1].tick_params(axis='y', labelleft=False)
    fig.suptitle('Pre-pitch forecasting: architecture, baselines and calibration',
                 x=.045, y=.973, ha='left', fontsize=17, fontweight='bold')
    fig.text(.045, .93, f'Inspected 2025 DEV cohort  |  {n:,} pitches in {games} games  |  fixed comparison rows; no DEV winner selection',
             fontsize=10.5, color='#586676')
    handles = [Line2D([], [], color=color, marker=marker, linestyle='none', markersize=7, label=label)
               for color, marker, label in STYLES.values()]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.04, .902), ncol=2,
               frameon=False, handletextpad=.5, columnspacing=2.2, fontsize=9.7)
    fig.text(.045, .093, 'Bars show ±1 standard deviation across five fitted seeds, not confidence intervals. Other rows show point estimates only.',
             fontsize=9.2, color='#526071')
    fig.text(.045, .063, 'Single-model mean loss differs from loss of averaged probabilities. Both CAL blend objectives appear in both panels.',
             fontsize=9.2, color='#526071')
    fig.text(.045, .033, 'Exploratory forecasting results; these scores do not establish counterfactual target accuracy or real policy benefit.',
             fontsize=9.2, color='#526071')
    return fig


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robustness-summary', required=True, type=Path)
    parser.add_argument('--calibration-results', required=True, type=Path)
    parser.add_argument('--frequency-results', required=True, type=Path)
    parser.add_argument('--strong-blend-results', type=Path, help='Optional completed stronger-frequency blend results.json; no placeholder rows if omitted')
    parser.add_argument('--output', required=True, type=Path, help='PNG under the configured SSD artifact root')
    args = parser.parse_args()
    local = read_json(PROJECT / 'configs/local.json')
    if not Path('/Volumes/T7 Shield').is_mount() or not args.output.resolve().is_relative_to(Path(local['artifact_root']).resolve()):
        raise SystemExit('Output must be under the mounted configured SSD artifact root')
    if args.output.suffix.lower() != '.png':
        raise SystemExit('--output must end in .png')
    robustness, calibration, frequency = [read_json(p) for p in [args.robustness_summary, args.calibration_results, args.frequency_results]]
    strong = read_json(args.strong_blend_results) if args.strong_blend_results else None
    rows = collect_rows(robustness, calibration, frequency, strong)
    fig = make_figure(rows, robustness['n'], robustness['games'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=240, facecolor='white')
    plt.close(fig)
    inputs = {name: value for name, value in vars(args).items() if name != 'output' and value is not None}
    sidecar = {'figure': str(args.output.resolve()), 'cohort_n': robustness['n'], 'cohort_games': robustness['games'],
               'input_sources': {name: {'path': str(path.resolve()), 'sha256': file_hash(path)} for name, path in inputs.items()},
               'script_sha256': file_hash(Path(__file__).resolve()), 'figure_sha256': file_hash(args.output),
               'score_variant': 'Original ten-outcome delivery-integrated calibrated DEV scores; no posthoc legal projection',
               'bar_definition': '±1 SD across five single-model seeds, not uncertainty intervals', 'rows': rows}
    sidecar_path = args.output.with_suffix('.json')
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + '\n')
    print(json.dumps({'figure': str(args.output.resolve()), 'data': str(sidecar_path.resolve()), 'rows': len(rows)}))


if __name__ == '__main__':
    main()
