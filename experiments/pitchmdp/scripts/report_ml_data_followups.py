"""Plot sealed D2/I1 summaries without loading predictions or running tests."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pitchmdp.data import hash_file
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sealed_result(run):
    path = run / 'analysis' / 'results.json'
    manifest = json.loads(path.with_name('manifest.json').read_text())
    if hash_file(path) != manifest['results_sha256']:
        raise ValueError('Completed result hash differs')
    return json.loads(path.read_text()), {'path': str(path), 'sha256': hash_file(path)}


def interval(ax, item, y, color):
    point, bounds = item['delta'], item['ci95']
    ax.plot(bounds, [y, y], color=color, linewidth=2)
    ax.scatter([point], [y], color=color, s=40, zorder=3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--d2-run', type=Path, required=True)
    parser.add_argument('--i1-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    d2, dp = sealed_result(args.d2_run)
    i1, ip = sealed_result(args.i1_run)
    if (d2['cells'] != ['D2-25', 'D1-25', 'D1-100'] or
        i1['factorial']['cell_order'] != ['MLP25', 'MLP100', 'TF25', 'TF100']):
        raise ValueError('Complete registered families required')
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), constrained_layout=True)
    green, amber, gray = '#136f63', '#c28620', '#7b8794'
    labels = ['Model D25\nAuxiliary D25', 'Model D25\nAuxiliary D100', 'Model D100\nAuxiliary D100']
    values = [d2['reports'][cell]['primary']['log_loss'] for cell in d2['cells']]
    axes[0, 0].scatter(range(3), values, color=green, s=50)
    for x, value in enumerate(values):
        axes[0, 0].annotate(f'{value:.6f}', (x, value), xytext=(0, 10), textcoords='offset points', ha='center')
    axes[0, 0].set(xticks=range(3), xticklabels=labels, ylabel='NLL per pitch', title='D2: result-model data and auxiliary data')
    axes[0, 0].margins(x=.25, y=.25)
    for y, comparison in enumerate(d2['primary_comparisons']):
        interval(axes[0, 1], comparison['paired']['nll'], y, green)
    axes[0, 1].axvline(0, color=gray)
    axes[0, 1].axvline(-.003, color=amber, linestyle='--', label='Practical point threshold')
    axes[0, 1].set(yticks=[0, 1], yticklabels=['D1-100 minus D2-25', 'D1-25 minus D2-25'],
        title='D2: both registered N comparisons pass', xlabel='Paired NLL difference, 95% CI', ylim=(-.6, 1.6))
    axes[0, 1].invert_yaxis()
    axes[0, 1].legend(fontsize=8, loc='lower left')
    for model, color in [('MLP', green), ('TF', amber)]:
        values = [i1['reports'][f'{model}{fraction}']['primary']['log_loss'] for fraction in (25, 100)]
        axes[1, 0].plot([25, 100], values, marker='o', color=color, linewidth=2,
                        label='Transformer' if model == 'TF' else model)
    axes[1, 0].set(xticks=[25, 100], xlabel='Result-model TRAIN game fraction (%)', ylabel='NLL per pitch',
        title='I1: same D100 auxiliaries, enriched H5')
    axes[1, 0].legend(fontsize=9)
    item = i1['factorial']['contrasts']['interaction']['nll']
    interval(axes[1, 1], item, 0, gray)
    axes[1, 1].axvline(0, color=gray)
    for threshold in (-.003, .003):
        axes[1, 1].axvline(threshold, color=amber, linestyle='--', linewidth=1)
    axes[1, 1].set(yticks=[], ylim=(-.7, .7), xlim=(-.0035, .006), xlabel='(TF100 − TF25) − (MLP100 − MLP25)',
        title='I1: inconclusive at the registered practical threshold')
    axes[1, 1].text(.5, .20, f"I = {item['delta']:+.6f}; two-sided p = {item['p_two_sided_centered']:.4f}\n"
        'Positive direction: larger MLP data gain\n|I| < .003 despite CI excluding zero',
        transform=axes[1, 1].transAxes, ha='center', va='center', fontsize=9)
    fig.suptitle('Data follow-ups — exposed C6 DEV: 5,048 pitches / 60 games', fontsize=14)
    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.output / 'ML-D2-I1-data-followups'
    fig.savefig(stem.with_suffix('.png'), dpi=160)
    fig.savefig(stem.with_suffix('.svg'))
    stem.with_suffix('.provenance.json').write_text(json.dumps({'results': [dp, ip],
        'source_sha256': hash_file(Path(__file__)), 'matplotlib': matplotlib.__version__,
        'limits': 'Separate fixed-prediction bootstrap families; D2 changes multiple auxiliary components jointly. '
        'D2 and I1 use different input representations; do not compare their absolute scores as a controlled effect. '
        'No new tests, policy effects or independent confirmation.'}, indent=2)+'\n')


if __name__ == '__main__':
    main()
