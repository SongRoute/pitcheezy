"""Independently audit B's saved predictions without training or tuning a model."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KEY = ['game_pk', 'at_bat_number', 'pitch_number']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(y, p):
    assert p.shape == (len(y), 10) and np.isfinite(p).all() and (p >= 0).all()
    np.testing.assert_allclose(p.sum(1), 1, atol=1e-10)
    loss = -np.log(np.maximum(p[np.arange(len(y)), y], 1e-12))
    bins = np.minimum((p.max(1) * 10).astype(int), 9)
    ece = sum(np.mean(bins == i) * abs(np.mean(p.max(1)[bins == i]) -
              np.mean((p.argmax(1) == y)[bins == i])) for i in range(10) if np.any(bins == i))
    return {'nll': float(loss.mean()), 'brier': float(np.square(p - np.eye(10)[y]).sum(1).mean()),
            'ece10': float(ece)}, loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--b-root', type=Path, default=ROOT)
    args = parser.parse_args()
    config_path = ROOT / 'configs/EXP-A-B-REVIEW-001.json'
    cfg = json.loads(config_path.read_text())
    bconfig_path = args.b_root / 'configs/EXP-B-PAHISTORY-001.json'
    bconfig = json.loads(bconfig_path.read_text())
    bresult = json.loads((args.b_root / 'results/EXP-B-PAHISTORY-001/results.json').read_text())
    bpath = Path(bconfig['artifact_dir']) / 'predictions.npz'
    assert sha(bpath) == bresult['sha256']['predictions']
    assert sha(bconfig_path) == bresult['sha256']['config']
    a = np.load(cfg['a_reference_predictions'])
    b = np.load(bpath)
    for key in ('pitch_keys', 'game_pk', 'y'):
        np.testing.assert_array_equal(a[key], b[key])
    np.testing.assert_array_equal(a['april_count'], b['count_only'])
    manifest = json.loads((ROOT / 'results/EXP-A-S0S1-001/selection_manifest.json').read_text())
    dev = pd.read_parquet(manifest['dataset_paths']['dev']).sort_values(KEY)
    # Independent loop over full PAs: current type becomes prior only AFTER this row.
    expected = {}
    lookup = {pitch: family for family, pitches in bconfig['families'].items() for pitch in pitches}
    for _, rows in dev.groupby(KEY[:2]):
        previous = 'NONE'
        for row in rows.itertuples(index=False):
            expected[(int(row.game_pk), int(row.at_bat_number), int(row.pitch_number))] = previous
            previous = lookup.get(row.pitch_type, bconfig['unknown_family'])
    np.testing.assert_array_equal([expected[tuple(key)] for key in b['pitch_keys']], b['previous_family'])
    scored = dev.set_index(KEY).loc[pd.MultiIndex.from_arrays(b['pitch_keys'].T, names=KEY)]
    y = b['y']
    reports = {}
    for name, mask in [('all', np.ones(len(y), bool)), ('two_strikes', scored.strikes.to_numpy() == 2)]:
        base, loss0 = metrics(y[mask], b['count_only'][mask])
        candidate, loss1 = metrics(y[mask], b['count_plus_prior_family'][mask])
        delta = pd.DataFrame({'game': b['game_pk'][mask], 'delta': loss1 - loss0})
        clusters = delta.groupby('game').delta.agg(['sum', 'count'])
        rng = np.random.default_rng(cfg['bootstrap_seed'])
        draws = rng.integers(0, len(clusters), (cfg['bootstrap_replicates'], len(clusters)))
        values = clusters['sum'].to_numpy()[draws].sum(1) / clusters['count'].to_numpy()[draws].sum(1)
        reports[name] = {'n': int(mask.sum()), 'games': len(clusters), 'baseline': base, 'candidate': candidate,
            'nll_delta': float((loss1-loss0).mean()),
            'game_bootstrap95': np.quantile(values, cfg['interval_quantiles']).tolist(),
            'note': 'descriptive development slice; no candidate selection or tuning'}
    result = {'experiment_id': cfg['experiment_id'], 'checks': {'same_rows_labels_games': True,
        'exact_A_baseline': True, 'strictly_prior_same_PA_history': True, 'finite_normalized_probabilities': True},
        'reports': reports, 'policy_value': None, 'decision': 'do_not_adopt_candidate_keep_frozen_service',
        'reason': 'Prior-family candidate worsens full DEV NLL; calibration screening alone cannot establish policy gain.',
        'config_sha256': sha(config_path), 'b_predictions_sha256': sha(bpath),
        'b_config_sha256': sha(bconfig_path), 'script_sha256': sha(__file__),
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'timing_note': 'A review specification recorded before opening prediction arrays, after B computed the single frozen-config run; no new tuning',
        'bootstrap_seed': cfg['bootstrap_seed'], 'bootstrap_replicates': cfg['bootstrap_replicates']}
    path = ROOT / 'results/EXP-A-B-REVIEW-001.json'
    if path.exists():
        raise ValueError('review already recorded')
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
