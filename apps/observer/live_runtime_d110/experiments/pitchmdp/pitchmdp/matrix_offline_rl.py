"""Strict observed-PA transitions and finite-action NNBC/IQL/CQL primitives.

The WE reward is frozen terminal-event utility, not RE24 or observed causal
policy value. Source import performs no fitting, data loading or inference.
"""
from pathlib import Path
import copy
import json
import math
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from .data import KEY
from .game import GameState, apply_terminal, _EVENT_GROUP
from .model import outcome_labels
from .planner import OUTCOMES
from .matrix_policy import safe_rows, paired_policy_comparisons

METHODS = ('NNBC', 'IQL', 'CQL')
FIXED = dict(width=128, hidden_layers=2, expectile=.7, actor_temperature=.01,
             actor_weight_clip=100., cql_alpha=.01, learning_rate=3e-4,
             polyak=.005, batch_size=1024, gradient_clip=10., gamma=1.)
NEXT = ('next_outs', 'next_bases', 'next_home_score', 'next_away_score', 'next_inning', 'next_half')


def terminal_event(label, balls, strikes):
    if label == 0 and balls == 3: return 'walk'
    if label == 1 and strikes == 2: return 'strikeout'
    return OUTCOMES[label] if label >= 3 and label < 10 else None


def legal_terminal_next(row, event):
    """Strict observed-next-state audit; missing fields are never fabricated."""
    if any(pd.isna(row[k]) for k in NEXT): return 'missing_terminal_next_state'
    try:
        before = GameState.from_row(row)
        after = GameState(int(row.next_inning), str(row.next_half), int(row.next_outs),
                          int(row.next_bases), int(row.next_home_score), int(row.next_away_score))
        if any(float(row[k]) != int(row[k]) for k in NEXT if k != 'next_half'):
            return 'noninteger_terminal_next_state'
    except (ValueError, TypeError): return 'illegal_terminal_next_state'
    same = before.inning == after.inning and before.half == after.half
    flip = ((before.half == 'Top' and after.inning == before.inning and after.half == 'Bot') or
            (before.half == 'Bot' and after.inning == before.inning+1 and after.half == 'Top'))
    if not (same or flip) or (flip and after.outs != 0): return 'illegal_terminal_half_transition'
    out_delta = after.outs-before.outs if same else 3-before.outs
    if not 0 <= out_delta <= 3-before.outs: return 'illegal_terminal_outs'
    if event in ('out', 'strikeout') and out_delta != 1: return 'illegal_terminal_outs'
    if event == 'double_play' and out_delta != 2: return 'illegal_terminal_outs'
    home, away = after.home_score-before.home_score, after.away_score-before.away_score
    if min(home, away) < 0 or (before.half == 'Top' and home != 0) or (before.half == 'Bot' and away != 0):
        return 'illegal_terminal_score'
    if max(home, away) > 4: return 'illegal_terminal_score'
    if ('post_home_score' in row and 'post_away_score' in row and
        (row.post_home_score != after.home_score or row.post_away_score != after.away_score)):
        return 'post_score_next_state_mismatch'
    if event in ('walk', 'hbp', 'strikeout', 'home_run') and after != apply_terminal(before, event):
        return 'deterministic_terminal_mismatch'
    return None


def audit_trajectories(frame, d100_positions, actions, support_for_row, train_roles, *, deadline=None):
    """Audit raw TRAIN PAs; return retained transitions and ALL PA reason rows.

    support_for_row uses only TRAIN-frozen action support and legal pre-pitch
    hand/pitcher metadata. Outcome labels are used for targets/eligibility only.
    """
    train = frame.loc[frame.split.eq('train')]
    if train.empty or not pd.to_datetime(train.game_date).between('2023-05-15', '2025-04-30').all():
        raise ValueError('Only the frozen TRAIN window may form RL transitions')
    if not frame.index.equals(pd.RangeIndex(len(frame))): raise ValueError('Canonical global row positions required')
    if not set(NEXT).issubset(frame.columns): raise ValueError('Terminal observed-next schema missing; do not weaken audit')
    eligible = set(int(x) for x in d100_positions)
    action_map = {a: i for i, a in enumerate(actions)}
    labels = outcome_labels(frame)
    records, audit = [], []
    for (game, pa), group in train.groupby(KEY[:2], sort=False):
        if deadline is not None and len(audit) % 1024 == 0 and time.monotonic() > deadline:
            raise TimeoutError('RL preparation trajectory-audit wall limit exceeded')
        group = group.sort_values('pitch_number')
        first, last = group.iloc[0], group.iloc[-1]
        positions = group.index.to_numpy(dtype=int)
        reasons = []
        in_d100 = all(i in eligible for i in positions)
        if not in_d100: reasons.append('not_complete_d100_pa')
        if (group.supported_pa.isna().any() or not group.supported_pa.all()): reasons.append('source_unsupported_pa')
        if not np.array_equal(group.pitch_number.to_numpy(), np.arange(1, len(group)+1)):
            reasons.append('nonconsecutive_pitch_sequence')
        if first.balls != 0 or first.strikes != 0: reasons.append('not_zero_count_start')
        fixed = ['pitcher', 'batter', 'stand', 'p_throws', 'inning', 'inning_topbot',
                 'outs_when_up', 'bases', 'home_score', 'away_score']
        if group[fixed].isna().any().any() or group[fixed].nunique(dropna=False).gt(1).any():
            reasons.append('mid_pa_context_change')
        flags = group.is_pa_terminal.fillna(False).to_numpy(bool)
        if flags.sum() != 1 or not flags[-1]: reasons.append('terminal_boundary_invalid')
        masks, terminal = [], None
        for j, position in enumerate(positions):
            row, label = group.iloc[j], int(labels[position])
            if label < 0: reasons.append('unmapped_outcome'); continue
            if (row.balls not in range(4) or row.strikes not in range(3) or
                row.pitch_type not in action_map):
                reasons.append('invalid_count_or_action'); continue
            try:
                mask = np.asarray(support_for_row(row), dtype=bool)
                if mask.shape != (len(actions),) or not mask[action_map[row.pitch_type]]:
                    reasons.append('observed_action_outside_common_support')
                masks.append(mask)
            except (ValueError, KeyError, TypeError): reasons.append('support_unavailable')
            event = terminal_event(label, int(row.balls), int(row.strikes))
            if j < len(group)-1:
                nxt = group.iloc[j+1]
                expected_b = row.balls+int(label == 0)
                expected_s = min(2, row.strikes+int(label in (1, 2)))
                if event is not None or nxt.balls != expected_b or nxt.strikes != expected_s:
                    reasons.append('nonterminal_count_or_event_mismatch')
            else:
                terminal = event
                if terminal is None or _EVENT_GROUP.get(row.terminal_event) != terminal:
                    reasons.append('terminal_label_mismatch')
                else:
                    reason = legal_terminal_next(row, terminal)
                    if reason: reasons.append(reason)
        reasons = sorted(set(reasons))
        raw_event = _EVENT_GROUP.get(str(last.terminal_event), 'unknown')
        audit.append(dict(game_pk=int(game), at_bat_number=int(pa), pitcher=int(first.pitcher),
            train_role=train_roles.get(int(first.pitcher), 'unknown'), terminal_class=raw_event,
            pitches=len(group), complete_d100=in_d100, retained=not reasons, reasons='|'.join(reasons)))
        if reasons: continue
        offset = len(records)
        for j, (position, mask) in enumerate(zip(positions, masks)):
            done = j == len(positions)-1
            records.append(dict(position=int(position), next_index=-1 if done else offset+j+1,
                action=action_map[str(group.iloc[j].pitch_type)], done=done,
                initial_position=int(positions[0]), terminal_event=terminal if done else None,
                mask=mask))
    if not audit: raise ValueError('No TRAIN PAs for audit')
    return records, pd.DataFrame(audit)


def terminal_rewards(records, frame, we, advancement):
    """Vectorized frozen WE; expected advancement matches terminal_values."""
    distributions, next_states = {}, set()
    for record in records:
        if not record['done']: continue
        state = GameState.from_row(frame.iloc[record['initial_position']])
        key = (state, record['terminal_event'])
        if key not in distributions:
            distributions[key] = advancement.distribution(*key)
            next_states.update(s for _, s in distributions[key])
    ordered = sorted(next_states, key=repr)
    live = [s for s in ordered if s.winner is None]
    p_home = {s: float(s.winner == 'home') for s in ordered if s.winner is not None}
    for start in range(0, len(live), 50000):
        batch = live[start:start+50000]
        values = we.predict_home(pd.DataFrame([dict(inning=s.inning, inning_topbot=s.half,
            outs_when_up=s.outs, bases=s.bases, home_score=s.home_score, away_score=s.away_score) for s in batch]))
        p_home.update(zip(batch, map(float, values)))
    reward_map = {key: sum(weight*(p_home[nxt] if key[0].defender_is_home else 1-p_home[nxt])
                          for weight, nxt in distribution) for key, distribution in distributions.items()}
    rewards = np.zeros(len(records), dtype=np.float32)
    for i, record in enumerate(records):
        if record['done']:
            key = (GameState.from_row(frame.iloc[record['initial_position']]), record['terminal_event'])
            rewards[i] = reward_map[key]
    if not np.isfinite(rewards).all() or (rewards < 0).any() or (rewards > 1).any():
        raise ValueError('Invalid frozen WE terminal reward')
    return rewards


def flatten_pre_pitch(tokens, valid, context):
    return np.ascontiguousarray(np.column_stack([tokens.reshape(len(tokens), -1), valid, context]), dtype=np.float32)


class LazyFeatures:
    """Observed token bank plus strict previous-row indices; no current token."""
    def __init__(self, directory):
        self.directory = Path(directory)
        for name in ('tokens', 'history', 'context', 'support', 'action', 'next_index', 'reward', 'done'):
            setattr(self, name, np.load(self.directory / (name+'.npy'), mmap_mode='r'))
        self.size = len(self.action)
        if not np.array_equal(self.done, self.next_index < 0):
            raise ValueError('Done flags and next-state indices disagree')
        if (self.next_index >= self.size).any(): raise ValueError('Next transition outside bank')
        self.width = 5*self.tokens.shape[1]+5+self.context.shape[1]
        if not self.size: raise ValueError('No retained transitions; stop before fitting')

    @staticmethod
    def write(directory, store, context, records, rewards):
        directory = Path(directory); directory.mkdir(parents=True, exist_ok=False)
        positions = np.array([r['position'] for r in records], dtype=np.int64)
        lookup = np.full(len(store.frame), -1, dtype=np.int64); lookup[positions] = np.arange(len(positions))
        original = store.indices[positions]
        history = np.where(original >= 0, lookup[np.maximum(original, 0)], -1)
        if ((original >= 0) & (history < 0)).any(): raise ValueError('Complete-PA history was lost during filtering')
        if (history >= np.arange(len(positions))[:, None]).any(): raise ValueError('History must precede current row')
        tokens = np.concatenate([store.physical[positions], store.type_channels[positions], store.outcome_channels[positions]], axis=1)
        encoded = []
        for start in range(0, len(positions), 8192):
            # Last two fields are routing cluster/pitcher IDs, not policy inputs.
            encoded.append(context.transform(safe_rows(store.frame.iloc[positions[start:start+8192]]))[:, :-2])
        payload = dict(tokens=tokens, history=history.astype(np.int32), context=np.concatenate(encoded),
            support=np.stack([r['mask'] for r in records]), action=np.array([r['action'] for r in records], dtype=np.int64),
            next_index=np.array([r['next_index'] for r in records], dtype=np.int64),
            done=np.array([r['done'] for r in records], dtype=bool), reward=rewards)
        for name, value in payload.items(): np.save(directory / (name+'.npy'), value, allow_pickle=False)
        return positions

    def features(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        previous = self.history[indices]
        valid = previous >= 0
        tokens = np.array(self.tokens[np.maximum(previous, 0)], copy=True)
        tokens[~valid] = 0
        return flatten_pre_pitch(tokens, valid, self.context[indices])

    def batch(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        x = self.features(indices)
        nxt = self.next_index[indices]; live = nxt >= 0
        next_x = np.zeros_like(x)
        next_mask = np.zeros((len(indices), self.support.shape[1]), dtype=bool)
        next_x[live] = self.features(nxt[live]); next_mask[live] = self.support[nxt[live]]
        return dict(x=x, action=np.array(self.action[indices]), reward=np.array(self.reward[indices]),
            done=np.array(self.done[indices]), next_x=next_x, mask=np.array(self.support[indices]), next_mask=next_mask)


def state_features(inputs, states):
    tokens, valid, context = inputs.arrays(states, [inputs.types[0]]*len(states), np.zeros((len(states), 8)))
    return flatten_pre_pitch(tokens[:, :-1], valid[:, :-1], context[:, :-2])


def masked_log_probs(logits, mask):
    if logits.shape != mask.shape or not bool(mask.any(dim=1).all()): raise ValueError('Empty or mismatched action support')
    return torch.log_softmax(logits.masked_fill(~mask, -torch.inf), dim=1)


def expectile_loss(error, eta=.7):
    return torch.where(error < 0, 1-eta, eta).mul(error.square()).mean()


def _network(width, outputs):
    return nn.Sequential(nn.Linear(width, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, outputs))


class OfflinePolicy:
    def __init__(self, method, width, n_actions, seed, device=None):
        if method not in METHODS: raise ValueError('Unknown offline method')
        self.method, self.width, self.n_actions, self.seed = method, width, n_actions, int(seed)
        self.device = device or ('mps' if torch.backends.mps.is_available() else 'cpu')
        torch.manual_seed(seed)
        self.networks = {}
        names = ('actor',) if method == 'NNBC' else ('q',) if method == 'CQL' else ('actor', 'q1', 'q2', 'v')
        for name in names: self.networks[name] = _network(width, 1 if name == 'v' else n_actions).to(self.device)
        target_names = () if method == 'NNBC' else ('q',) if method == 'CQL' else ('q1', 'q2')
        self.targets = {name: copy.deepcopy(self.networks[name]).requires_grad_(False) for name in target_names}
        self.optimizers = {name: torch.optim.Adam(net.parameters(), lr=FIXED['learning_rate']) for name, net in self.networks.items()}
        self.updates = 0
        self.inference_rows = 0

    def _update(self, name, loss):
        if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite loss')
        optimizer, net = self.optimizers[name], self.networks[name]
        optimizer.zero_grad(set_to_none=True); loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(net.parameters(), FIXED['gradient_clip'], error_if_nonfinite=True)
        if not bool(torch.isfinite(norm)): raise FloatingPointError('Nonfinite gradient')
        optimizer.step()
        if any(not bool(torch.isfinite(p).all()) for p in net.parameters()): raise FloatingPointError('Nonfinite parameters')

    def train_step(self, batch):
        b = {name: torch.as_tensor(value, device=self.device) for name, value in batch.items()}
        x, action, mask = b['x'], b['action'].long(), b['mask'].bool()
        if any(not bool(torch.isfinite(b[k]).all()) for k in ('x', 'next_x', 'reward')):
            raise FloatingPointError('Nonfinite training batch')
        if not bool(mask.gather(1, action[:, None]).all()): raise ValueError('Logged action outside common support')
        info, rows = {}, torch.arange(len(x), device=self.device)
        if self.method == 'NNBC':
            loss = -masked_log_probs(self.networks['actor'](x), mask)[rows, action].mean()
            self._update('actor', loss); info['actor_loss'] = float(loss.detach().cpu())
        else:
            live = ~b['done'].bool()
            if self.method == 'IQL':
                with torch.no_grad(): qbar = torch.minimum(self.targets['q1'](x), self.targets['q2'](x))[rows, action]
                v_loss = expectile_loss(qbar-self.networks['v'](x).squeeze(1), FIXED['expectile'])
                self._update('v', v_loss); info['value_loss'] = float(v_loss.detach().cpu())
                with torch.no_grad():
                    target = b['reward'].clone()
                    target[live] += self.networks['v'](b['next_x'][live]).squeeze(1)
                for name in ('q1', 'q2'):
                    q = self.networks[name](x)
                    loss = (q[rows, action]-target).square().mean()
                    self._update(name, loss); info[name+'_loss'] = float(loss.detach().cpu())
                with torch.no_grad():
                    advantage = qbar-self.networks['v'](x).squeeze(1)
                    exponent = advantage/FIXED['actor_temperature']
                    weight = torch.exp(exponent.clamp(max=math.log(FIXED['actor_weight_clip'])))
                loss = -(weight*masked_log_probs(self.networks['actor'](x), mask)[rows, action]).mean()
                self._update('actor', loss)
                info.update(actor_loss=float(loss.detach().cpu()), advantage_min=float(advantage.min().cpu()),
                    advantage_max=float(advantage.max().cpu()), weight_max=float(weight.max().cpu()),
                    weight_clip_fraction=float((exponent >= math.log(100.)).float().mean().cpu()))
                for label, value in (('advantage', advantage), ('weight', weight)):
                    quantiles = np.quantile(value.cpu().numpy(), [.1, .5, .9])
                    info.update({label+'_p'+str(q): float(v) for q, v in zip((10, 50, 90), quantiles)})
            else:
                with torch.no_grad():
                    target = b['reward'].clone()
                    if bool(live.any()):
                        next_mask = b['next_mask'][live].bool()
                        if not bool(next_mask.any(dim=1).all()): raise ValueError('Nonterminal next state has no support')
                        best = self.networks['q'](b['next_x'][live]).masked_fill(~next_mask, -torch.inf).argmax(1)
                        target[live] += self.targets['q'](b['next_x'][live]).gather(1, best[:, None]).squeeze(1)
                q = self.networks['q'](x)
                td = .5*(q[rows, action]-target).square().mean()
                conservative = (torch.logsumexp(q.masked_fill(~mask, -torch.inf), dim=1)-q[rows, action]).mean()
                loss = td+FIXED['cql_alpha']*conservative
                self._update('q', loss)
                info.update(td_loss=float(td.detach().cpu()), conservative_loss=float(conservative.detach().cpu()))
            with torch.no_grad():
                for name, target_net in self.targets.items():
                    for target_param, online in zip(target_net.parameters(), self.networks[name].parameters()):
                        target_param.lerp_(online, FIXED['polyak'])
                q_names = ('q',) if self.method == 'CQL' else ('q1', 'q2')
                q_values = torch.cat([self.networks[name](x)[mask] for name in q_names])
                if not bool(torch.isfinite(q_values).all()): raise FloatingPointError('Nonfinite Q')
                info.update(q_min=float(q_values.min().cpu()), q_max=float(q_values.max().cpu()),
                    q_outside_diagnostic_range=int(((q_values < -.25) | (q_values > 1.25)).sum().cpu()),
                    target_min=float(target.min().cpu()), target_max=float(target.max().cpu()))
        self.updates += 1
        return info

    def probabilities(self, x, mask):
        self.inference_rows += len(x)
        with torch.no_grad():
            x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
            mask = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
            net = self.networks['q' if self.method == 'CQL' else 'actor']
            logits = net(x)
            if not bool(torch.isfinite(logits).all()): raise FloatingPointError('Nonfinite policy logits')
            if self.method == 'CQL':
                if not bool(mask.any(dim=1).all()): raise ValueError('No supported action')
                p = torch.nn.functional.one_hot(logits.masked_fill(~mask, -torch.inf).argmax(1), self.n_actions).float()
            else: p = masked_log_probs(logits, mask).exp()
            return p.cpu().numpy().astype(float)

    def save(self, path):
        torch.save(dict(format='matrix_offline_rl_v1', method=self.method, width=self.width,
            n_actions=self.n_actions, seed=self.seed, updates=self.updates, fixed=FIXED,
            networks={k: {n: p.detach().cpu() for n, p in net.state_dict().items()} for k, net in self.networks.items()},
            targets={k: {n: p.detach().cpu() for n, p in net.state_dict().items()} for k, net in self.targets.items()}), path)

    @classmethod
    def load(cls, path, device=None):
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if payload['format'] != 'matrix_offline_rl_v1' or payload['fixed'] != FIXED: raise ValueError('Offline model format/settings changed')
        model = cls(payload['method'], payload['width'], payload['n_actions'], payload['seed'], device)
        for key, net in model.networks.items(): net.load_state_dict(payload['networks'][key])
        for key, net in model.targets.items(): net.load_state_dict(payload['targets'][key])
        model.updates = payload['updates']
        return model


class AmortizedPolicy:
    def __init__(self, inputs, bc, models): self.inputs, self.bc, self.models = inputs, bc, models
    def batch_probabilities(self, states, depth=0):
        x, mask = state_features(self.inputs, states), np.stack([self.bc.support(s) for s in states])
        return self.bc.actions, np.mean([m.probabilities(x, mask) for m in self.models], axis=0)
    def __call__(self, state, depth=0):
        names, p = self.batch_probabilities([state], depth)
        return names, p[0]


def rl_comparisons(values, flags, games, *, draws=10000, seed=20260924):
    pairs = [('IQL', 'planner'), ('CQL', 'planner'), ('IQL', 'NNBC'), ('CQL', 'NNBC')]
    names = ('IQL', 'CQL', 'NNBC', 'planner')
    results = paired_policy_comparisons({k: values[k] for k in names}, games, pairs,
        truncated={k: flags[k] for k in names}, draws=draws, seed=seed)
    for row in results[:2]:
        matched = next(r for r in results if r['candidate'] == row['candidate'] and r['control'] == 'NNBC')
        row['rl_gain_over_both_controls'] = bool(row['untruncated_pa_improvement_confirmed'] and
                                               matched['untruncated_pa_improvement_confirmed'])
    return results
