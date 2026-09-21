"""Reproduce one actual development PA recommendation from a saved first run."""
import argparse
import json
from pathlib import Path
import pickle
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pitchmdp.model import PitchModel
from pitchmdp.recommend import recommend

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run', required=True, type=Path)
parser.add_argument('--pitcher', type=int)
parser.add_argument('--sigma', type=float, default=.3)
parser.add_argument('--balls', type=int, choices=range(4))
parser.add_argument('--strikes', type=int, choices=range(3))
parser.add_argument('--prev-pitch-type')
parser.add_argument('--inning', type=int)
parser.add_argument('--outs', type=int, choices=range(3))
parser.add_argument('--bases', type=int, choices=range(8), help='Bitmask: 1=first, 2=second, 4=third')
parser.add_argument('--home-score', type=int)
parser.add_argument('--away-score', type=int)
args = parser.parse_args()
with (args.run/'recommendation_context.pkl').open('rb') as stream:
    context = pickle.load(stream)
rows = context['rows']
if args.pitcher:
    rows = rows[rows.pitcher == args.pitcher]
if not len(rows):
    raise SystemExit('No saved representative PA for this pitcher.')
row = rows.iloc[0].copy()
overrides = {}
for arg, column in [('balls','balls'), ('strikes','strikes'), ('prev_pitch_type','prev_pitch_type'),
                    ('inning','inning'), ('outs','outs_when_up'), ('bases','bases'),
                    ('home_score','home_score'), ('away_score','away_score')]:
    value = getattr(args, arg)
    if value is not None:
        row[column] = value
        overrides[column] = value
model = PitchModel.load(args.run/'pitch_model.pt')
result = recommend(model, row, context['actions'][(int(row.pitcher), str(row.stand))],
                   context['we'], context['advancement'], args.sigma)
result['hypothetical_context_overrides'] = overrides
print(json.dumps(result, indent=2, ensure_ascii=False))
