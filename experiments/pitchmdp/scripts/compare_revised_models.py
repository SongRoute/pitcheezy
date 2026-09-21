"""Matched, exploratory information ablations for the revised MLB cohort.

No best model is selected using DEV. All fixed variants and paired uncertainty
are reported. This diagnoses prediction only, not real policy improvement.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import numpy as np
import pandas as pd
from pitchmdp.model import PitchModel, DeliveryDistribution, eligible, metrics, outcome_labels
from pitchmdp.archetypes import add_batter_style_history

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run', required=True, type=Path)
args = parser.parse_args()
run = args.run.resolve()
root = Path('/Volumes/T7 Shield/pitcheezy/pitchmdp')
if not Path('/Volumes/T7 Shield').is_mount() or not run.is_relative_to(root):
    raise SystemExit('Mounted configured SSD required')
config = json.loads((run/'config.json').read_text())
cohort = json.loads((run/'cohort_manifest.json').read_text())
if config.get('model_variant') != 'archetype':
    raise SystemExit('Requires revised archetype reference run')
variants = ['full', 'minus_b', 'archetype_no_context', 'archetype_no_history']
contract = {'variants': variants, 'reference': 'archetype', 'seed': config['seed'],
            'interpretation': 'Exploratory matched prediction comparisons, no DEV model selection or causal policy claim.',
            'changes': {'full': 'Batter ID + prior OBP/K/count replaces style representation; same cohort.',
                        'minus_b': 'No batter ID or batter rates/types; retain observed batting side.',
                        'archetype_no_context': 'Remove base/out/inning/score/half from response network only; WE rules unchanged.',
                        'archetype_no_history': 'Remove preceding pitch type; retain current count.'},
            'code_hashes': {str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in list((PROJECT/'pitchmdp').glob('*.py'))+[Path(__file__)]}}
(run/'comparison_contract.json').write_text(json.dumps(contract, indent=2)+'\n')
frame = pd.read_parquet(root/'processed/pitches.parquet')
add_batter_style_history(frame)
mask = eligible(frame)
target = frame.pitcher.isin(cohort['pitcher_ids']) & frame.pitcher.eq(frame.starter_pitcher)
train_all = frame[(frame.split == 'train') & mask]
cal_all = frame[(frame.split == 'calibration') & mask]
train = pd.concat([train_all.sample(min(len(train_all), config['training_random_rows']), random_state=config['seed']),
                   train_all[target.reindex(train_all.index)]]).drop_duplicates(['game_pk','at_bat_number','pitch_number']).sort_index()
cal = cal_all.sample(min(len(cal_all), config['calibration_random_rows']), random_state=config['seed'])
dev = frame[(frame.split == 'dev') & mask & target].sort_index()
delivery = DeliveryDistribution().fit(train_all, config['delivery_draws'], config['seed'])
y = outcome_labels(dev)
reference = PitchModel.load(run/'pitch_model.pt')
ref_p = delivery.predict(reference, dev)
ref_ll = -np.log(np.clip(ref_p[np.arange(len(y)), y], 1e-12, 1))
games, gi = np.unique(dev.game_pk, return_inverse=True)
counts = np.bincount(gi)
rng = np.random.default_rng(42)
bootstrap = rng.integers(0, len(games), size=(2000, len(games)))
output = {'contract': contract, 'reference': metrics(y, ref_p), 'n_games': len(games), 'comparisons': {}}
for variant in variants:
    model = PitchModel(variant=variant, seed=config['seed']).fit(train, cal, epochs=config['epochs'])
    model.save(run/f'comparison_{variant}.pt')
    p = delivery.predict(model, dev)
    ll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))
    delta = ref_ll-ll
    sums = np.bincount(gi, weights=delta)
    draws = sums[bootstrap].sum(1)/counts[bootstrap].sum(1)
    output['comparisons'][variant] = {
        'primary_prepitch': metrics(y, p),
        'retrospective_location_diagnostic': metrics(y, model.predict(dev)),
        'reference_minus_variant_logloss': float(delta.mean()),
        'paired_game_bootstrap95': np.quantile(draws, [.025,.975]).tolist(),
        'direction': 'Negative favors reference archetype model; exploratory intervals not multiplicity corrected.',
        'training': model.training_report}
    (run/'matched_comparisons.json').write_text(json.dumps(output, indent=2)+'\n')
    print(variant, {k: output['comparisons'][variant][k] for k in ['reference_minus_variant_logloss','paired_game_bootstrap95']}, flush=True)
print(f'COMPLETE {run}/matched_comparisons.json', flush=True)
