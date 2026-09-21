"""Generated prediction runtime: exact selected definitions from a hash-verified capture.
Training methods may remain for class identity; the service does not invoke them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pitchmdp.model import outcome_labels
from run_sequence_frequency_baselines import TYPE_KEYS, HierarchicalFrequencyBaseline


CONTEXT_KEYS = [*TYPE_KEYS, "outs_when_up", "bases"]

PITCHER_CONTEXT_KEYS = ["pitcher", *CONTEXT_KEYS]

class ContextFrequencyBaseline:
    """Two fixed strength-100 state levels above an immutable type parent."""
    def __init__(self, parent_type_model, include_pitcher=False):
        if parent_type_model.include_pitcher:
            raise ValueError("Context hierarchy requires the league/type parent")
        self.parent_type_model, self.include_pitcher = parent_type_model, include_pitcher

    @staticmethod
    def _validate_state(frame):
        if not frame.outs_when_up.isin([0, 1, 2]).all() or not frame.bases.isin(range(8)).all():
            raise ValueError("Expected legal pre-pitch outs0..2 and base occupancy0..7")

    def fit(self, train):
        if not len(train) or not train.split.eq("train").all() or not pd.to_datetime(train.game_date).between("2023-05-15", "2025-04-30").all():
            raise ValueError("Context-frequency fitting requires only approved TRAIN rows")
        self._validate_state(train)
        labels = outcome_labels(train)
        if (labels < 0).any():
            raise ValueError("Valid outcome labels required")
        work = train[PITCHER_CONTEXT_KEYS].copy()
        work["y"] = labels
        counts = HierarchicalFrequencyBaseline._counts(work, CONTEXT_KEYS)
        parents = self.parent_type_model.predict(counts.index.to_frame(index=False))
        values = (counts.to_numpy()+100*parents)/(counts.sum(axis=1).to_numpy()[:, None]+100)
        self.context_table = pd.DataFrame(values, index=counts.index, columns=range(10))
        if self.include_pitcher:
            counts = HierarchicalFrequencyBaseline._counts(work, PITCHER_CONTEXT_KEYS)
            parents = self._predict_context(counts.index.to_frame(index=False))[0]
            values = (counts.to_numpy()+100*parents)/(counts.sum(axis=1).to_numpy()[:, None]+100)
            self.pitcher_context_table = pd.DataFrame(values, index=counts.index, columns=range(10))
        self.report = {"training_rows": len(train), "include_pitcher": self.include_pitcher,
                       "context_strength": 100, "pitcher_context_strength": 100,
                       "context_keys": CONTEXT_KEYS, "pitcher_context_keys": PITCHER_CONTEXT_KEYS,
                       "context_groups": len(self.context_table),
                       "pitcher_context_groups": len(self.pitcher_context_table) if self.include_pitcher else 0,
                       "parent_report": self.parent_type_model.report, "batter_id_input": False,
                       "fit_date_max": str(pd.to_datetime(train.game_date).max().date())}
        return self

    def _predict_context(self, frame):
        self._validate_state(frame)
        parent, origin = self.parent_type_model.predict_with_origin(frame)
        values = HierarchicalFrequencyBaseline._lookup(self.context_table, frame, CONTEXT_KEYS)
        seen = np.isfinite(values).all(axis=1)
        origin[seen] = 4
        return np.where(seen[:, None], values, parent), origin

    def predict_with_origin(self, frame):
        values, origin = self._predict_context(frame)
        if self.include_pitcher:
            local = HierarchicalFrequencyBaseline._lookup(self.pitcher_context_table, frame, PITCHER_CONTEXT_KEYS)
            seen = np.isfinite(local).all(axis=1)
            values = np.where(seen[:, None], local, values)
            origin[seen] = 5
        return values, origin

    def predict(self, frame):
        return self.predict_with_origin(frame)[0]
