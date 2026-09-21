"""Generated prediction runtime: exact selected definitions from a hash-verified capture.
Training methods may remain for class identity; the service does not invoke them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


OUTCOMES = ('ball', 'strike', 'foul', 'out', 'single', 'double', 'triple',
            'home_run', 'hbp', 'double_play')

def outcome_labels(frame: pd.DataFrame) -> np.ndarray:
    d, e = frame.description.fillna(''), frame.events.fillna('')
    labels = np.full(len(frame), -1, dtype=np.int64)
    labels[d.isin(['ball', 'blocked_ball', 'pitchout', 'intent_ball'])] = 0
    labels[d.isin(['called_strike', 'swinging_strike', 'swinging_strike_blocked',
                  'missed_bunt', 'foul_tip', 'bunt_foul_tip'])] = 1
    labels[d.isin(['foul', 'foul_bunt'])] = 2
    labels[(d == 'foul_bunt') & (frame.strikes == 2)] = 1
    labels[d == 'hit_by_pitch'] = 8
    for name in ['single', 'double', 'triple', 'home_run']:
        labels[(d == 'hit_into_play') & (e == name)] = OUTCOMES.index(name)
    labels[(d == 'hit_into_play') & e.isin(
        ['field_out', 'force_out', 'fielders_choice_out', 'sac_fly', 'sac_bunt'])] = 3
    labels[(d == 'hit_into_play') & e.isin(
        ['double_play', 'grounded_into_double_play', 'sac_fly_double_play', 'sac_bunt_double_play'])] = 9
    return labels

class CountBaseline:
    """Training-only smoothed count/hand outcome frequencies, no location oracle."""
    keys = ['balls', 'strikes', 'stand', 'p_throws']
    def fit(self, train):
        t = train[self.keys].copy()
        t['y'] = outcome_labels(train)
        self.global_p = (np.bincount(t.y, minlength=len(OUTCOMES)) + 1).astype(float)
        self.global_p /= self.global_p.sum()
        self.table = {}
        for key, g in t.groupby(self.keys):
            count = np.bincount(g.y, minlength=len(OUTCOMES))+50*self.global_p
            self.table[key] = count/count.sum()
        return self

    def predict(self, frame):
        return np.array([self.table.get(k, self.global_p) for k in frame[self.keys].itertuples(index=False, name=None)])
