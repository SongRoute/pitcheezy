"""New, disjoint-July empirical evaluator and one bounded calibration candidate."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax
from pitchmdp.model import outcome_labels
from pitchmdp.game import GameState, _EVENT_GROUP
from pitchmdp.archetypes import STYLE_COLUMNS


class JulyEvaluator:
    """All counts/priors fitted only on July, with fixed hand/context/style bins."""
    TIERS = [
        ['balls', 'strikes', 'stand', 'p_throws'],
        ['balls', 'strikes', 'stand', 'p_throws', 'pitch_type'],
        ['balls', 'strikes', 'stand', 'p_throws', 'pitch_type', 'outs_when_up', 'bases'],
        ['balls', 'strikes', 'stand', 'p_throws', 'pitch_type', 'outs_when_up', 'bases', 'contact_bin', 'power_bin'],
        ['balls', 'strikes', 'stand', 'p_throws', 'pitch_type', 'outs_when_up', 'bases', 'contact_bin', 'power_bin', 'pitcher'],
    ]

    @staticmethod
    def decorate(frame):
        frame = frame.copy()
        frame['contact_bin'] = np.searchsorted([.75, .85], frame[STYLE_COLUMNS[0]])
        frame['power_bin'] = np.searchsorted([.10, .20], frame[STYLE_COLUMNS[4]])
        return frame

    def fit(self, frame):
        if not len(frame) or not pd.to_datetime(frame.game_date).between('2025-07-01', '2025-07-31').all():
            raise ValueError('Independent evaluator fitting is restricted to July2025')
        y = outcome_labels(frame)
        if (y < 0).any():
            raise ValueError('Invalid evaluator outcome labels')
        self.global_p = (np.bincount(y, minlength=10)+1).astype(float)
        self.global_p /= self.global_p.sum()
        self.tables, self.counts = [], []
        work = self.decorate(frame)
        work['y'] = y
        for level, keys in enumerate(self.TIERS):
            counts = work.groupby(keys+['y'], observed=True).size().unstack('y', fill_value=0).reindex(columns=range(10), fill_value=0)
            totals = counts.sum(axis=1)
            parent = self._predict(counts.index.to_frame(index=False))[0]
            strength = 50 if level < 2 else 100
            self.tables.append(pd.DataFrame((counts.to_numpy()+strength*parent)/(totals.to_numpy()[:, None]+strength), index=counts.index))
            self.counts.append(totals)
        self.report = {'rows': len(frame), 'games': int(frame.game_pk.nunique()),
                       'date_min': str(frame.game_date.min()), 'date_max': str(frame.game_date.max()),
                       'tiers': self.TIERS, 'strengths': [50, 50, 100, 100, 100],
                       'prior_source': 'July2025 only; fixed numerical pseudocounts; no MVP tables'}
        return self

    def _predict(self, frame):
        p = np.broadcast_to(self.global_p, (len(frame), 10)).copy()
        support, origin = np.zeros(len(frame)), np.full(len(frame), -1)
        for level, (keys, table, counts) in enumerate(zip(self.TIERS, self.tables, self.counts)):
            index = pd.MultiIndex.from_frame(frame[keys])
            values = table.reindex(index).to_numpy()
            available = np.isfinite(values).all(axis=1)
            p[available] = values[available]
            support[available] = counts.reindex(index).to_numpy()[available]
            origin[available] = level
        return p, support, origin

    def predict_with_support(self, frame):
        return self._predict(self.decorate(frame))

    def predict(self, frame):
        return self.predict_with_support(frame)[0]


class OutcomeCalibration:
    """One fixed L2-regularized class-intercept correction; no setting search."""
    def __init__(self, bias=None):
        self.bias = np.zeros(10) if bias is None else np.asarray(bias, float)

    def fit(self, p, y, game_dates):
        if not pd.Series(pd.to_datetime(game_dates)).between('2025-08-01', '2025-08-07').all():
            raise ValueError('Repair fitting is restricted to August1–7')
        p, y = np.asarray(p, float), np.asarray(y, int)
        if p.shape != (len(y), 10) or len(y) < 100:
            raise ValueError('Need at least100 aligned diagnostic predictions')
        logits = np.log(np.clip(p, 1e-30, 1))
        onehot = np.eye(10)[y]
        def objective(bias):
            pred = softmax(logits+bias, axis=1)
            loss = -np.log(np.clip(pred[np.arange(len(y)), y], 1e-30, 1)).mean()+.025*np.dot(bias, bias)
            return loss, (pred-onehot).mean(axis=0)+.05*bias
        result = minimize(objective, np.zeros(10), jac=True, method='L-BFGS-B')
        if not result.success:
            raise RuntimeError(result.message)
        self.bias = result.x
        return self

    def apply(self, p):
        corrected = np.asarray(p, float)*np.exp(self.bias-self.bias.max())
        return corrected/corrected.sum(axis=-1, keepdims=True)


def advancement_metrics(model, frame):
    """Observed non-walkoff, supported PA transitions; never use them for policy input."""
    next_cols = ['next_outs', 'next_bases', 'next_home_score', 'next_away_score', 'next_inning', 'next_half']
    data = frame.loc[frame.supported_pa.fillna(False) & frame.is_pa_terminal.fillna(False)].dropna(subset=next_cols)
    probabilities, support, errors = [], [], []
    for _, row in data.iterrows():
        event = _EVENT_GROUP.get(row.terminal_event)
        if event not in ('out', 'single', 'double', 'triple', 'double_play'):
            continue
        state = GameState.from_row(row)
        if state.half == 'Bot' and state.inning >= 9 and row.next_home_score > row.next_away_score:
            continue
        same = row.next_inning == state.inning and row.next_half == state.half
        flip = (state.half == 'Top' and row.next_inning == state.inning and row.next_half == 'Bot') or (
            state.half == 'Bot' and row.next_inning == state.inning+1 and row.next_half == 'Top')
        if not (same or flip):
            continue
        target = tuple(row[k] for k in ['next_inning', 'next_half', 'next_outs', 'next_bases', 'next_home_score', 'next_away_score'])
        distribution = model.distribution(state, event)
        probability = sum(prob for prob, nxt in distribution if
            (nxt.inning, nxt.half, nxt.outs, nxt.bases, nxt.home_score, nxt.away_score) == target)
        expected_runs = sum(prob*((nxt.home_score-state.home_score) if state.half == 'Bot' else
                                 (nxt.away_score-state.away_score)) for prob, nxt in distribution)
        actual_runs = row.next_home_score-state.home_score if state.half == 'Bot' else row.next_away_score-state.away_score
        probabilities.append(probability)
        support.append(sum(model.counts.get((event, state.bases, state.outs), {}).values()))
        errors.append(abs(expected_runs-actual_runs))
    p = np.asarray(probabilities)
    return {'n': len(p), 'zero_probability_rate': float(np.mean(p == 0)),
            'mean_clipped_nll': float(-np.log(np.clip(p, 1e-12, 1)).mean()),
            'cell_fallback_rate': float(np.mean(np.asarray(support) == 0)),
            'mean_absolute_runs_error': float(np.mean(errors)),
            'scope': 'Exact observed supported non-walkoff next state; NLL clipped at1e-12'}
