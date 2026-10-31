"""Render a verified complete architecture screen as a standalone figure."""
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


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    directory = a.run / 'analysis'
    manifest = json.loads((directory / 'manifest.json').read_text())
    if hash_file(directory / 'results.json') != manifest['results_sha256']:
        raise ValueError('Architecture analysis integrity failed')
    result = json.loads((directory / 'results.json').read_text())
    cells = ['A0-MLP', 'A1-linear', 'A2-lightgbm', 'A3-lstm', 'A4-gru', 'A5-melville', 'A6-transformer']
    if set(result['reports']) != set(cells) or len(result['primary_comparisons']) != 6:
        raise ValueError('Complete seven-architecture results required')
    labels = ['MLP', 'Linear', 'LightGBM', 'LSTM', 'GRU', 'Melville adaptation', 'Transformer']
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 5.1), constrained_layout=True)
    y = np.arange(len(cells))
    for stage, marker, color, legend in [('raw_ensemble', 'x', '#9ba4ae', 'Raw'),
        ('calibrated_ensemble', 's', '#82b4a6', 'Temperature'), ('primary', 'o', '#136f63', 'Temperature + blend')]:
        values = [result['reports'][cell][stage]['log_loss'] for cell in cells]
        axes[0].scatter(values, y, marker=marker, color=color, label=legend, s=38)
    axes[0].set(yticks=y, yticklabels=labels, xlabel='NLL per pitch', title='Same D100 and enriched H5')
    axes[0].invert_yaxis()
    axes[0].legend(fontsize=8, loc='lower right')
    comparisons = {c['candidate']: c for c in result['primary_comparisons']}
    for i, cell in enumerate(cells[1:]):
        c = comparisons[cell]
        value, interval = c['paired']['nll']['delta'], c['paired']['nll']['ci95']
        color = '#136f63' if c['decision']['status'] == 'predictive_improvement' else '#7b8794'
        axes[1].plot(interval, [i, i], color=color, linewidth=2)
        axes[1].scatter([value], [i], color=color, s=35)
    axes[1].axvline(0, color='#7b8794', linewidth=1)
    axes[1].axvline(-.003, color='#c28620', linestyle='--', label='Practical point threshold')
    axes[1].set(yticks=np.arange(6), yticklabels=labels[1:], xlabel='Candidate minus MLP NLL, 95% CI',
                title='Paired game bootstrap; Holm family of 6')
    axes[1].invert_yaxis()
    axes[1].legend(fontsize=8, loc='lower right')
    fit = [result['cost_summary'][c]['total_load_fit_temperature_seconds'] for c in cells]
    inference = [result['cost_summary'][c]['total_prediction_seconds'] for c in cells]
    axes[2].barh(y, fit, color='#136f63', label='Load + fit + temperature')
    axes[2].barh(y, inference, left=fit, color='#82b4a6', label='400-draw prediction')
    axes[2].set(yticks=y, yticklabels=labels, xlabel='Recorded seconds for three seeds', title='Cost on the same M4 host')
    axes[2].invert_yaxis()
    axes[2].legend(fontsize=8, loc='lower right')
    fig.suptitle(f"ML2 architecture screen — 2025 C6 DEV, {result['n']:,} pitches / {result['games']} games", fontsize=14)
    a.output.mkdir(parents=True, exist_ok=True)
    stem = a.output / 'ML-A-architecture-comparison'
    fig.savefig(stem.with_suffix('.png'), dpi=160)
    fig.savefig(stem.with_suffix('.svg'))
    stem.with_suffix('.provenance.json').write_text(json.dumps({
        'results_path': str(directory / 'results.json'), 'results_sha256': hash_file(directory / 'results.json'),
        'source_sha256': hash_file(Path(__file__)), 'matplotlib': matplotlib.__version__,
        'limits': 'C6 exposed DEV; conditional bootstrap; no wholeMLB/policy/final confirmation. Green requires all registered N gates. Cost excludes preparation/profiles/interpreter startup.'}, indent=2)+'\n')
