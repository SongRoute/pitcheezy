"""Render already-scored, hash-verified G results; never compute new tests."""
import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pitchmdp.data import hash_file
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    directory = args.run / 'analysis' / 'panel'
    manifest = json.loads((directory / 'manifest.json').read_text())
    if hash_file(directory / 'results.json') != manifest['results_sha256']:
        raise ValueError('G analysis integrity failed')
    result = json.loads((directory / 'results.json').read_text())
    cells = ['G0-global', 'G1-personal', 'G2-feature', 'G3-cluster', 'G4-partial']
    labels = ['Global', 'Personal + fallback', 'Cluster feature', 'Cluster expert', 'Partial sharing']
    pairs = [('G1-personal', 'G0-global'), ('G2-feature', 'G0-global'),
             ('G3-cluster', 'G2-feature'), ('G4-partial', 'G2-feature')]
    if set(result['reports']) != set(cells) or [(x['candidate'], x['control']) for x in result['comparisons']] != pairs:
        raise ValueError('Complete registered sharing family required')
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), constrained_layout=True,
                             gridspec_kw={'width_ratios': [1, 1.3, 1.7]})
    y = np.arange(5)
    for stage, marker, color, title in [('raw_ensemble', 'x', '#9ba4ae', 'Raw'),
        ('calibrated_ensemble', 's', '#82b4a6', 'Temperature'),
        ('primary', 'o', '#136f63', 'Temperature + blend')]:
        axes[0].scatter([result['reports'][cell][stage]['log_loss'] for cell in cells], y,
                        marker=marker, color=color, label=title, s=35)
    axes[0].set(yticks=y, yticklabels=labels, xlabel='NLL per pitch', title='Same Cpanel and D100')
    axes[0].invert_yaxis()
    axes[0].legend(fontsize=8, loc='lower right')
    for i, comparison in enumerate(result['comparisons']):
        for field, offset, color, label in [('paired', -.10, '#136f63', 'All'),
                                          ('low_paired', .10, '#c28620', 'Low TRAIN volume')]:
            value = comparison[field]
            if value is None:
                continue
            interval = value['nll']['ci95']
            axes[1].plot(interval, [i+offset]*2, color=color, linewidth=2)
            axes[1].scatter([value['nll']['delta']], [i+offset], color=color, s=30,
                            label=label if i == 0 else None)
    axes[1].axvline(0, color='#7b8794', linewidth=1)
    axes[1].axvline(-.003, color='#7b8794', linestyle='--', linewidth=1)
    pair_labels = ['Personal - global', 'Feature - global', 'Expert - feature', 'Partial - feature']
    axes[1].set(yticks=np.arange(4), yticklabels=pair_labels, xlabel='Paired NLL difference, 95% CI',
                title='Overall and low-volume effects')
    axes[1].invert_yaxis()
    axes[1].legend(fontsize=8, loc='lower right')
    group_names = list(result['comparisons'][0]['robustness']['groups'])
    grid = np.array([[0 if row['passed'] is None else 1 if row['passed'] else 2
                      for row in c['robustness']['groups'].values()] for c in result['comparisons']])
    colors = ['#dce1e5', '#82b4a6', '#dc987e']
    axes[2].imshow(grid, cmap=ListedColormap(colors), vmin=0, vmax=2, aspect='auto', interpolation='nearest')
    axes[2].set(yticks=np.arange(4), yticklabels=pair_labels, xticks=np.arange(len(group_names)),
                xticklabels=[s.replace('role_', '').replace('volume_', 'vol. ').replace('_', ' ') for s in group_names],
                title='Group noninferiority: 96 joint bounds')
    axes[2].tick_params(axis='x', labelrotation=65, labelsize=8)
    fig.get_layout_engine().set(rect=(0, .12, 1, .88))
    fig.legend(handles=[Patch(color=c, label=t) for c, t in zip(colors,
        ['Insufficient sample', 'Within both margins', 'Margin not confirmed'])],
        fontsize=8, loc='lower center', bbox_to_anchor=(.77, .01), ncol=3, frameon=False)
    fig.suptitle(f"Pitcher sharing — 2025 Cpanel DEV, {result['n']:,} pitches / {result['games']} games", fontsize=14)
    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.output / 'ML-G-sharing-comparison'
    fig.savefig(stem.with_suffix('.png'), dpi=160)
    fig.savefig(stem.with_suffix('.svg'))
    stem.with_suffix('.provenance.json').write_text(json.dumps({
        'results_path': str(directory / 'results.json'), 'results_sha256': hash_file(directory / 'results.json'),
        'source_sha256': hash_file(Path(__file__)), 'matplotlib': matplotlib.__version__,
        'limits': 'Exposed Cpanel; existing conditional intervals only. N/G family of 8. R family of 96; failed upper bound is not proof of harm. No wholeMLB, policy or independent final claim.'}, indent=2)+'\n')
