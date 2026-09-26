"""One frozen April-to-July axis: previous pitch family within the same PA."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'experiments/pitchmdp'), str(ROOT/'experiments/pitchmdp/scripts')]

import numpy as np
import pandas as pd
from pitchmdp.model import CountBaseline, OUTCOMES, outcome_labels
from pitchmdp.data import KEY
from a_small_eval import cohort_pa_rows, legal_baseline, paired_bootstrap, score_metrics

CONFIG_PATH = ROOT/'configs/EXP-B-PAHISTORY-001.json'
REPORT = ROOT/'results/EXP-B-PAHISTORY-001'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def family(name, cfg):
    for label, names in cfg['families'].items():
        if name in names:
            return label
    return cfg['unknown_family']


def add_prior(frame, cfg):
    ordered = frame.sort_values(KEY, kind='stable').copy()
    previous = ordered.groupby(['game_pk', 'at_bat_number'], sort=False).pitch_type.shift(1)
    ordered['previous_family'] = previous.map(lambda name: family(name, cfg) if pd.notna(name) else 'NONE')
    first = ordered.pitch_number.eq(1)
    if not ordered.loc[first, 'previous_family'].eq('NONE').all():
        raise ValueError('Prior pitch crossed a PA boundary')
    return ordered


class FamilyCountBaseline:
    """Same count baseline plus one conditional table, smoothed to its cell."""
    def __init__(self, prior_pseudocount):
        self.base = CountBaseline()
        self.prior_pseudocount = prior_pseudocount
        self.keys = [*CountBaseline.keys, 'previous_family']

    def fit(self, train):
        self.base.fit(train)
        table = train[self.keys].copy()
        table['y'] = outcome_labels(train)
        self.table = {}
        for key, group in table.groupby(self.keys):
            base = self.base.table.get(key[:-1], self.base.global_p)
            counts = np.bincount(group.y, minlength=len(OUTCOMES)) + self.prior_pseudocount*base
            self.table[key] = counts/counts.sum()
        return self

    def predict(self, frame):
        return np.asarray([self.table.get(key, self.base.table.get(key[:-1], self.base.global_p))
                           for key in frame[self.keys].itertuples(index=False, name=None)])


def main():
    cfg = json.loads(CONFIG_PATH.read_text())
    manifest = json.loads((ROOT/cfg['parent_manifest']).read_text())
    if cfg['train_cohort_pitchers'] != [657277, 554430] or manifest['experiment_id'] != cfg['parent_experiment_id']:
        raise ValueError('Frozen A selection mismatch')
    for split in ('train', 'dev'):
        path = Path(manifest['dataset_paths'][split])
        if sha(path) != manifest['sha256'][split+'_full_games.parquet']:
            raise ValueError(f'Parent {split} data hash mismatch')
    if cfg['baseline_global_additive_count'] != 1 or cfg['baseline_cell_global_pseudocount'] != 50:
        raise ValueError('CountBaseline parameters changed')
    artifact = Path(cfg['artifact_dir'])
    if artifact.exists() or REPORT.exists():
        raise ValueError('B experiment already exists; immutable result')
    train_full = pd.read_parquet(manifest['dataset_paths']['train'])
    dev_full = pd.read_parquet(manifest['dataset_paths']['dev'])
    train, train_reasons, train_pas = cohort_pa_rows(train_full, cfg['train_cohort_pitchers'])
    # Match A's exact frozen repertoire mask without loading its predictive model.
    bundle = Path(json.loads((ROOT/'configs/EXP-A-S0S1-001.json').read_text())['bundle_dir'])
    metadata = json.loads((bundle/'metadata.json').read_text())
    dev, dev_reasons, dev_pas = cohort_pa_rows(dev_full, cfg['train_cohort_pitchers'], metadata['pitchers'])
    train, dev = add_prior(train, cfg), add_prior(dev, cfg)
    parent_result = json.loads((ROOT/'results/EXP-A-S0S1-001/results.json').read_text())
    parent = parent_result['support']
    if (train_pas, dev_pas, train_reasons, dev_reasons, len(dev)) != (
            parent['train_cohort_pas'], parent['dev_cohort_pas'], parent['train_excluded_pa_reasons'],
            parent['dev_excluded_pa_reasons'], parent_result['tiers']['S1']['pitches']):
        raise ValueError('A support mask changed')
    keys = dev.sort_values(KEY)[KEY].astype(int).to_numpy()
    a = np.load(Path(manifest['dataset_paths']['dev']).parent/'predictions.npz')
    if not np.array_equal(keys, a['pitch_keys']):
        raise ValueError('B DEV keys differ from A saved predictions')
    y = outcome_labels(dev)
    if not np.array_equal(y, a['y']):
        raise ValueError('B labels differ from A')
    base = CountBaseline().fit(train)
    extended = FamilyCountBaseline(cfg['context_cell_count_baseline_pseudocount']).fit(train)
    p_base = legal_baseline(base.predict(dev), dev)
    p_extended = legal_baseline(extended.predict(dev), dev)
    if not np.array_equal(p_base, a['april_count']):
        raise ValueError('Count-only predictions differ from A baseline')
    games = dev.game_pk.to_numpy(dtype=int)
    s0_ids = [group[0]['game_pk'] for group in manifest['selection']['dev'].values()]
    tiers = {}
    for name, mask in [('S0', np.isin(games, s0_ids)), ('S1', np.ones(len(dev), bool))]:
        tiers[name] = {'games': int(len(np.unique(games[mask]))), 'pitches': int(mask.sum()),
                       'count_only': score_metrics(y[mask], p_base[mask]),
                       'count_plus_prior_family': score_metrics(y[mask], p_extended[mask]),
                       'plus_minus_only': paired_bootstrap(y[mask], p_extended[mask], p_base[mask],
                             games[mask], cfg['bootstrap_seed'], cfg['bootstrap_replicates'])}
    artifact.mkdir(parents=True)
    REPORT.mkdir(parents=True)
    np.savez_compressed(artifact/'predictions.npz', y=y, game_pk=games, pitch_keys=keys,
                        count_only=p_base, count_plus_prior_family=p_extended,
                        previous_family=dev.previous_family.to_numpy(dtype='U16'))
    result = {'experiment_id': cfg['experiment_id'], 'phase': 'evaluated_exploratory_dev',
              'axis': cfg['axis'], 'tiers': tiers,
              'support': {'train_cohort_pas': train_pas, 'train_supported_complete_pas': int(train.groupby(['game_pk','at_bat_number']).ngroups),
                          'dev_cohort_pas': dev_pas, 'dev_supported_complete_pas': int(dev.groupby(['game_pk','at_bat_number']).ngroups),
                          'train_excluded_pa_reasons': train_reasons, 'dev_excluded_pa_reasons': dev_reasons},
              'policy_causal_we': None, 'policy_causal_we_reason': 'No identified target intervention or propensity.',
              'sha256': {'config': sha(CONFIG_PATH), 'script': sha(__file__),
                         'parent_manifest': sha(ROOT/cfg['parent_manifest']),
                         'parent_predictions': sha(Path(manifest['dataset_paths']['dev']).parent/'predictions.npz'),
                         'predictions': sha(artifact/'predictions.npz')},
              'artifact_dir': str(artifact)}
    (artifact/'results.json').write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n')
    (REPORT/'results.json').write_text((artifact/'results.json').read_text())
    print(json.dumps({'tiers': {name: {'count_only_nll': value['count_only']['log_loss'],
              'plus_nll': value['count_plus_prior_family']['log_loss'], 'delta': value['plus_minus_only']}
              for name, value in tiers.items()}, 'artifact_dir': str(artifact)}, indent=2))


if __name__ == '__main__':
    main()
