"""Refit frozen B count tables once for archival restore verification only."""
from __future__ import annotations

import json
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts'), str(ROOT/'experiments/pitchmdp'),
                str(ROOT/'experiments/pitchmdp/scripts')]

import numpy as np
import pandas as pd
from pitchmdp.model import CountBaseline
from a_small_eval import cohort_pa_rows, legal_baseline
from b_pa_history_eval import FamilyCountBaseline, add_prior, sha


def main():
    config_path = ROOT/'configs/EXP-B-PAHISTORY-001.json'
    cfg = json.loads(config_path.read_text())
    report = json.loads((ROOT/'results/EXP-B-PAHISTORY-001/results.json').read_text())
    if report['sha256']['config'] != sha(config_path):
        raise ValueError('Scored configuration changed')
    manifest_path = ROOT/cfg['parent_manifest']
    if report['sha256']['parent_manifest'] != sha(manifest_path):
        raise ValueError('A selection changed')
    parent = json.loads(manifest_path.read_text())
    train_path = Path(parent['dataset_paths']['train'])
    dev_path = Path(parent['dataset_paths']['dev'])
    for split, path in [('train', train_path), ('dev', dev_path)]:
        if sha(path) != parent['sha256'][split+'_full_games.parquet']:
            raise ValueError('Selected rows changed')
    train, _, _ = cohort_pa_rows(pd.read_parquet(train_path), cfg['train_cohort_pitchers'])
    bundle = Path(json.loads((ROOT/'configs/EXP-A-S0S1-001.json').read_text())['bundle_dir'])
    metadata = json.loads((bundle/'metadata.json').read_text())
    dev, _, _ = cohort_pa_rows(pd.read_parquet(dev_path), cfg['train_cohort_pitchers'], metadata['pitchers'])
    train, dev = add_prior(train, cfg), add_prior(dev, cfg)
    archive_path = Path(cfg['artifact_dir'])/'predictions.npz'
    if sha(archive_path) != report['sha256']['predictions']:
        raise ValueError('Scored predictions changed')
    archive = np.load(archive_path)
    models = {'count_only': CountBaseline().fit(train),
              'count_plus_prior_family': FamilyCountBaseline(cfg['context_cell_count_baseline_pseudocount']).fit(train)}
    checkpoint_path = Path(cfg['artifact_dir'])/'fitted_count_tables.pkl'
    if checkpoint_path.exists():
        raise ValueError('Checkpoint already exists; no overwrite')
    with checkpoint_path.open('wb') as stream:
        pickle.dump(models, stream, protocol=pickle.HIGHEST_PROTOCOL)
    with checkpoint_path.open('rb') as stream:
        restored = pickle.load(stream)
    for name, model in restored.items():
        np.testing.assert_array_equal(legal_baseline(model.predict(dev), dev), archive[name])
    companion = {'purpose': 'Deterministic frozen-config refit and restore of already scored B count tables; no new scoring or tuning',
                 'experiment_id': cfg['experiment_id'], 'score_result_sha256': sha(ROOT/'results/EXP-B-PAHISTORY-001/results.json'),
                 'config_sha256': sha(config_path), 'archive_script_sha256': sha(__file__),
                 'train_full_games_sha256': sha(train_path), 'dev_full_games_sha256': sha(dev_path),
                 'scored_predictions_sha256': sha(archive_path), 'checkpoint_sha256': sha(checkpoint_path),
                 'restored_predictions_exactly_match_scored': True,
                 'checkpoint_path': str(checkpoint_path)}
    companion_path = ROOT/'results/EXP-B-PAHISTORY-001/checkpoint_manifest.json'
    if companion_path.exists():
        raise ValueError('Checkpoint manifest already exists')
    companion_path.write_text(json.dumps(companion, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'checkpoint': str(checkpoint_path), 'restore_exact': True,
                      'manifest': str(companion_path)}))


if __name__ == '__main__':
    main()
