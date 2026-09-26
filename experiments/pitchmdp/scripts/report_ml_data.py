"""Render the completed D1 family without fitting or selecting a model."""
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
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    directory = args.run / 'analysis' / 'data'
    manifest = json.loads((directory / 'manifest.json').read_text())
    if hash_file(directory / 'results.json') != manifest['results_sha256']:
        raise ValueError('Completed analysis hash differs')
    result = json.loads((directory / 'results.json').read_text())
    if result['cells'] != ['D1-25', 'D1-50', 'D1-100']:
        raise ValueError('Expected complete D1 family')
    cells = result['cells']
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.3), constrained_layout=True)
    nll = [result['reports'][c]['primary']['log_loss'] for c in cells]
    axes[0].plot([25, 50, 100], nll, marker='o', color='#136f63', linewidth=2)
    for fraction, value in zip([25, 50, 100], nll):
        axes[0].annotate(f'{value:.6f}', (fraction, value), xytext=(0, 10), textcoords='offset points', ha='center')
    axes[0].set(xlabel='TRAIN game fraction (%)', ylabel='NLL per pitch (lower is better)', title='3-seed calibrated ensemble + blend', xticks=[25, 50, 100])
    axes[0].margins(y=.3)
    comparisons = result['primary_comparisons']
    for i, comp in enumerate(comparisons):
        value, interval = comp['paired']['nll']['delta'], comp['paired']['nll']['ci95']
        color = '#136f63' if comp['decision']['status'] == 'predictive_improvement' else '#7b8794'
        axes[1].errorbar(value, i, xerr=[[value-interval[0]], [interval[1]-value]], fmt='o', capsize=4, color=color)
    axes[1].axvline(0, color='#7b8794', linewidth=1)
    axes[1].axvline(-.003, color='#c28620', linestyle='--', linewidth=1, label='Practical threshold (point estimate)')
    axes[1].set(yticks=[0, 1], yticklabels=['D50 minus D25', 'D100 minus D25'],
                xlabel='Paired NLL difference, 95% CI', title='60-game paired bootstrap')
    axes[1].margins(y=.6)
    axes[1].legend(loc='upper left', fontsize=8)
    fit = [sum(m['fit']['seconds_total'] for m in result['costs'][c]) for c in cells]
    predict = [sum(m['prediction']['seconds'] for m in result['costs'][c]) for c in cells]
    positions = np.arange(3)
    axes[2].bar(positions, fit, color='#136f63', label='Load + fit + temperature')
    axes[2].bar(positions, predict, bottom=fit, color='#82b4a6', label='Integrated prediction')
    axes[2].set(xticks=positions, xticklabels=['D25', 'D50', 'D100'], ylabel='Recorded process-stage seconds', title='Total cost of three seeds')
    axes[2].legend(fontsize=8)
    fig.suptitle('ML1 data scale comparison — 2025 C6 DEV, 5,048 pitches / 60 games', fontsize=14)
    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output / 'ML-D1-data-comparison.png', dpi=160)
    fig.savefig(args.output / 'ML-D1-data-comparison.svg')
    (args.output / 'ML-D1-data-comparison.provenance.json').write_text(json.dumps({
        'results_path': str(directory / 'results.json'), 'results_sha256': hash_file(directory / 'results.json'),
        'source_sha256': hash_file(Path(__file__)), 'matplotlib': matplotlib.__version__,
        'limits': 'Previously exposed C6 DEV prediction; no whole-MLB or causal policy claim. Excludes interpreter startup, preparation, profiles and failed attempt cost.'}, indent=2) + '\n')


if __name__ == '__main__':
    main()
