"""G0 whole-MLB DEV loss by delivery-pool tier (descriptive diagnostic, no fit, no inference).

Reads only the sealed EXP-P11-001 outputs (previously exposed 2025 DEV). It splits
the stored G0 and frequency predictions by the JointDelivery tier each pitch used:
  -1 global fallback, 0 league type/hand, 1 league type/hand/count,
   2 pitcher type/hand, 3 pitcher type/hand/count.
Nothing here is a registered comparison or a selection result. No 2026 data.

Usage: .venv/bin/python docs/reviews/G0-delivery-tier-diagnostic-2026-10-01.py <EXP-P11-001 dir> <out.json>
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BOOTSTRAP_SEED, DRAWS = 20260924, 10_000
SEEDS = (0, 1, 2, 3, 4)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def nll(prob, y):
    return -np.log(np.clip(prob[np.arange(len(y)), y], 1e-12, 1.))


def brier(prob, y):
    return ((prob - np.eye(prob.shape[1])[y]) ** 2).sum(1)


def main(root, out):
    root = Path(root)
    files = {'analysis': root / 'analysis' / 'predictions.npz', 'baseline': root / 'baseline_predictions.npz',
             'metadata': root / 'dev_metadata.parquet',
             **{f'seed{s}': root / 'members' / 'G0-global' / f'seed{s}' / 'predictions.npz' for s in SEEDS}}
    hashes = {name: sha256(path) for name, path in files.items()}
    a, b = np.load(files['analysis']), np.load(files['baseline'])
    meta = pd.read_parquet(files['metadata'])
    keys, y, game = a['keys'], a['y'], a['game_pk']
    assert np.array_equal(keys, b['mlb_dev_keys']) and np.array_equal(y, b['mlb_dev_y'])
    assert np.array_equal(keys, meta[['game_pk', 'at_bat_number', 'pitch_number']].to_numpy())
    tier = None
    for s in SEEDS:
        member = np.load(files[f'seed{s}'])
        assert np.array_equal(keys, member['dev_keys'])
        level = member['dev_delivery_level']
        assert tier is None or np.array_equal(tier, level)
        tier = level

    loss = {'g0': nll(a['primary'], y), 'neural': nll(a['calibrated'], y), 'frequency': nll(b['mlb_dev'], y)}
    bri = {'g0': brier(a['primary'], y), 'frequency': brier(b['mlb_dev'], y)}
    whole = {k: float(v.mean()) for k, v in loss.items()}
    # Validity: the sealed report gives G0 1.490408196725 and frequency 1.504081016265.
    assert abs(whole['g0'] - 1.490408196725) < 1e-9 and abs(whole['frequency'] - 1.504081016265) < 1e-9

    codes, game_index = np.unique(game, return_inverse=True)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    weights = rng.multinomial(len(codes), np.full(len(codes), 1 / len(codes)), size=DRAWS).astype(np.float64)

    def per_game(values, mask):
        return np.bincount(game_index[mask], weights=values[mask], minlength=len(codes))

    def ratio_draws(values, mask):
        num, den = weights @ per_game(values, mask), weights @ per_game(np.ones(len(y)), mask)
        return num / np.where(den > 0, den, np.nan)

    seen = meta.seen_pitcher.to_numpy()
    groups = {f'tier_{t}': tier == t for t in (-1, 0, 1, 2, 3)}
    groups |= {'league_pool_tiers_le1': tier <= 1, 'pitcher_pool_tiers_ge2': tier >= 2, 'whole': np.ones(len(y), bool),
               'seen_pitcher_league_pool': seen & (tier <= 1), 'seen_pitcher_pitcher_pool': seen & (tier >= 2),
               'unseen_pitcher_league_pool': ~seen & (tier <= 1), 'unseen_pitcher_pitcher_pool': ~seen & (tier >= 2)}
    for volume in ('zero', 'low', 'middle', 'high'):
        chosen = meta.train_volume.eq(volume).to_numpy()
        groups |= {f'volume_{volume}_league_pool': chosen & (tier <= 1), f'volume_{volume}_pitcher_pool': chosen & (tier >= 2)}

    delta, delta_neural = loss['g0'] - loss['frequency'], loss['neural'] - loss['frequency']
    report, draws = {}, {}
    for name, mask in groups.items():
        n = int(mask.sum())
        row = {'n': n, 'share': n / len(y), 'games': int(len(np.unique(game[mask]))),
               'pitchers': int(meta.pitcher[mask].nunique())}
        if n:
            draws[name] = ratio_draws(delta, mask)
            row |= {'nll_g0': float(loss['g0'][mask].mean()), 'nll_neural_before_blend': float(loss['neural'][mask].mean()),
                    'nll_frequency': float(loss['frequency'][mask].mean()),
                    'delta_nll_g0_minus_frequency': float(delta[mask].mean()),
                    'delta_nll_ci95': np.nanpercentile(draws[name], [2.5, 97.5]).tolist(),
                    'delta_nll_neural_minus_frequency': float(delta_neural[mask].mean()),
                    'delta_neural_ci95': np.nanpercentile(ratio_draws(delta_neural, mask), [2.5, 97.5]).tolist(),
                    'delta_brier_g0_minus_frequency': float((bri['g0'] - bri['frequency'])[mask].mean()),
                    'ball_rate': float((y[mask] == 0).mean()), 'strike_rate': float((y[mask] == 1).mean()),
                    'two_strikes_share': float(meta.two_strikes[mask].astype(float).mean()),
                    'relief_share': float(meta.game_role[mask].eq('relief').mean())}
        report[name] = row

    def gap(left, right):
        diff = draws[left] - draws[right]
        return {'contrast': f'{left} minus {right} of (G0 - frequency) NLL',
                'point': report[left]['delta_nll_g0_minus_frequency'] - report[right]['delta_nll_g0_minus_frequency'],
                'ci95': np.nanpercentile(diff, [2.5, 97.5]).tolist()}

    contrasts = {'league_vs_pitcher_pool': gap('league_pool_tiers_le1', 'pitcher_pool_tiers_ge2'),
                 'seen_pitchers_league_vs_pitcher_pool': gap('seen_pitcher_league_pool', 'seen_pitcher_pitcher_pool'),
                 'pitcher_pool_without_count_vs_with_count': gap('tier_2', 'tier_3')}
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    result = {'id': 'G0-delivery-tier-diagnostic', 'status': 'descriptive_diagnostic_only',
              'scope': 'Previously exposed eligible 2025 whole-MLB DEV; sealed EXP-P11-001 outputs; no fit, no inference',
              'source_dir': str(root), 'input_sha256': hashes, 'script_commit_head': commit,
              'bootstrap': {'unit': 'game', 'draws': DRAWS, 'seed': BOOTSTRAP_SEED, 'statistic': 'pitch-weighted mean'},
              'tier_meaning': {'-1': 'global fallback', '0': 'league type/hand', '1': 'league type/hand/count',
                               '2': 'pitcher type/hand', '3': 'pitcher type/hand/count'},
              'whole_nll_reproduced': whole, 'tiers_identical_across_seeds': True, 'groups': report, 'contrasts': contrasts,
              'limits': ['Tiers are not randomly assigned: league-pool pitches come from low-sample pitcher/type/hand cells, '
                         'so gap differences mix the pool effect with pitcher, pitch-type and situation composition.',
                         'Bootstrap intervals condition on the fixed predictions; no training or calibration uncertainty.',
                         'Unregistered post-hoc look at an exposed DEV; not a selection result, not independent confirmation.'],
              'independent_confirmation': None, 'policy_effect': None}
    Path(out).write_text(json.dumps(result, indent=1, ensure_ascii=False) + '\n')
    print(json.dumps({'whole': whole, 'contrasts': contrasts}, indent=1))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
