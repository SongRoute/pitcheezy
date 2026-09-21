"""Generated prediction runtime: exact selected definitions from a hash-verified capture.
Training methods may remain for class identity; the service does not invoke them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import softmax
from pitchmdp.model import CountBaseline, OUTCOMES, outcome_labels


COUNT_KEYS = ["balls", "strikes", "stand", "p_throws"]

TYPE_KEYS = ["pitch_type", *COUNT_KEYS]

PITCHER_KEYS = ["pitcher", *TYPE_KEYS]

class HierarchicalFrequencyBaseline:
    """Fixed pseudo-count strengths: global→count50→type50→pitcher100."""
    def __init__(self, include_pitcher=False):
        self.include_pitcher = include_pitcher

    @staticmethod
    def _counts(frame, keys):
        return frame.groupby([*keys, "y"], sort=True, dropna=False).size().unstack("y", fill_value=0).reindex(columns=range(10), fill_value=0)

    @staticmethod
    def _lookup(table, frame, keys):
        return table.reindex(pd.MultiIndex.from_frame(frame[keys])).to_numpy()

    def fit(self, train):
        if not len(train) or not train.split.eq("train").all() or not pd.to_datetime(train.game_date).between("2023-05-15", "2025-04-30").all():
            raise ValueError("Frequency fitting requires only approved TRAIN rows")
        labels = outcome_labels(train)
        if (labels < 0).any():
            raise ValueError("Frequency fitting requires valid outcome labels")
        self.parent = CountBaseline().fit(train)
        work = train[PITCHER_KEYS].copy()
        work["y"] = labels
        counts = self._counts(work, TYPE_KEYS)
        parents = self.parent.predict(counts.index.to_frame(index=False))
        values = (counts.to_numpy()+50*parents)/(counts.sum(axis=1).to_numpy()[:, None]+50)
        self.type_table = pd.DataFrame(values, index=counts.index, columns=range(10))
        if self.include_pitcher:
            counts = self._counts(work, PITCHER_KEYS)
            parents = self._predict_type(counts.index.to_frame(index=False))[0]
            values = (counts.to_numpy()+100*parents)/(counts.sum(axis=1).to_numpy()[:, None]+100)
            self.pitcher_table = pd.DataFrame(values, index=counts.index, columns=range(10))
        self.report = {"training_rows": len(train), "include_pitcher": self.include_pitcher,
                       "count_hand_groups": len(self.parent.table), "type_groups": len(self.type_table),
                       "pitcher_type_groups": len(self.pitcher_table) if self.include_pitcher else 0,
                       "type_strength": 50, "pitcher_strength": 100, "count_strength": 50,
                       "global_prior": "add one pseudo-observation per ten-class outcome",
                       "fit_date_max": str(pd.to_datetime(train.game_date).max().date()), "batter_id_input": False}
        return self

    def _predict_type(self, frame):
        base = self.parent.predict(frame)
        values = self._lookup(self.type_table, frame, TYPE_KEYS)
        available = np.isfinite(values).all(axis=1)
        origin = np.array([int(key in self.parent.table) for key in frame[COUNT_KEYS].itertuples(index=False, name=None)], dtype=np.int8)
        origin[available] = 2
        return np.where(available[:, None], values, base), origin

    def predict_with_origin(self, frame):
        values, origin = self._predict_type(frame)
        if self.include_pitcher:
            local = self._lookup(self.pitcher_table, frame, PITCHER_KEYS)
            available = np.isfinite(local).all(axis=1)
            values = np.where(available[:, None], local, values)
            origin[available] = 3
        return values, origin

    def predict(self, frame):
        return self.predict_with_origin(frame)[0]

def temperature_predictions(probabilities, temperature):
    return softmax(np.log(np.clip(probabilities, 1e-12, 1.))/temperature, axis=-1)
