"""CAL2 (EXP-WS-SEQ-CAL-001, attempt seqcal-001) calibrators of the N1 margins, applied only: nothing is fit here.

Parameters come from the hash-pinned ``result.json`` (``seeds[s].parameters``) after ``terminal.json`` (also pinned)
says ``success: true`` and names the same result sha256 and the same ``cal_selected``. The selected arm of each seed
is READ from the file (never written in code).

Apply formulas are copies of ``experiments/ws_seq/calibrate_logits.py``
(sha256 7568b8b43c74de946d822c9b226406823b1789f1ceb4ee37ba80ca66c8b226e7), cited by line:
  leaves            lines 88-92    [p0, p2, p8, p1*q, p1*(1-q), s*r, s*(1-r)], s = sum p[3,4,5,6,7,9]
  pav_apply         lines 378-382  value[min(searchsorted(right, x, side='left'), n-1)]
  isotonic_apply    lines 390-394  per-leaf lookup, clip to [clip, 1-clip], renormalise rows
  probs_c0          lines 419-422  leaves(softmax(zp/Tp), expit(zw/Tw), expit(zg/Tg))
  apply_arms        lines 446-453  C1 vector scaling, C2 Dirichlet (log_softmax(zp) @ W.T + b), Platt heads, C3
Leaf order: ball, foul, hit_by_pitch, narrow_whiff, other_strike, ground_ball_in_play, other_in_play.
The test compares every arm with ``calibrate_logits.apply_arms`` (imported by path) within 1e-12.
"""
from __future__ import annotations

import json

import numpy as np
from scipy.special import expit, log_softmax, softmax

from . import PinMismatch, read_pinned

BIP = (3, 4, 5, 6, 7, 9)
HEADS = ('parent', 'narrow_whiff', 'ground_ball_per_pitch')


def leaves(p10, q, r, tolerance):
    s = p10[:, list(BIP)].sum(1)
    out = np.column_stack([p10[:, 0], p10[:, 2], p10[:, 8], p10[:, 1]*q, p10[:, 1]*(1-q), s*r, s*(1-r)])
    if not np.isfinite(out).all() or np.abs(out.sum(1)-1).max() > tolerance:
        raise ValueError('STOP seven-leaf coherence')
    return out


def pav_apply(blocks, x):
    i = np.minimum(np.searchsorted(blocks['right'], x, side='left'), len(blocks['right'])-1)
    return blocks['value'][i]


def isotonic_apply(blocks, q, clip):
    m = np.column_stack([pav_apply(blocks[k], q[:, k]) for k in range(7)])
    m = np.clip(m, clip, 1-clip)
    return m/m.sum(1, keepdims=True)


def probs_c0(z, T, tolerance):
    return leaves(softmax(z['parent']/T['parent'], axis=1), expit(z['narrow_whiff']/T['narrow_whiff']),
                  expit(z['ground_ball_per_pitch']/T['ground_ball_per_pitch']), tolerance)


def apply_arm(model, arm, z, tolerance):
    """Seven-leaf rows of one arm for the margins ``z`` ({head: float64 array}) under one seed's parameters."""
    z = {h: np.asarray(z[h], dtype=np.float64) for h in HEADS}
    q0 = probs_c0(z, model['T'], tolerance)
    if arm == 'C0':
        return q0
    if arm == 'C3':
        return isotonic_apply(model['blocks'], q0, model['clip'])
    qw = expit(model['pw']['a']*z['narrow_whiff']+model['pw']['b'])
    qg = expit(model['pg']['a']*z['ground_ball_per_pitch']+model['pg']['b'])
    if arm == 'C1':
        return leaves(softmax(z['parent']*model['vec']['w']+model['vec']['b'], axis=1), qw, qg, tolerance)
    if arm == 'C2':
        return leaves(softmax(log_softmax(z['parent'], axis=1) @ model['dir']['W'].T+model['dir']['b'], axis=1),
                      qw, qg, tolerance)
    raise ValueError('unknown calibration arm: '+str(arm))


def model_from_parameters(parameters, leaf_order):
    """``result.json`` ``seeds[s].parameters`` (calibrate_logits.model_params, lines 499-510) -> arrays. C2 shares
    the C1 Platt heads. Isotonic blocks are taken per leaf in ``leaf_order``."""
    heads = parameters['C1']['heads']
    blocks = []
    for name in leaf_order:
        b = parameters['C3']['per_leaf'][name]
        block = {k: np.asarray(b[k], dtype=np.float64) for k in ('left', 'right', 'value')}
        block['weight'] = np.asarray(b['weight'], dtype=np.int64)
        n = len(block['value'])
        if n < 1 or any(len(block[k]) != n for k in block) or b.get('blocks', n) != n \
                or (np.diff(block['right']) <= 0).any() or not np.isfinite(block['value']).all():
            raise PinMismatch('isotonic block geometry: '+name)
        blocks.append(block)
    model = {'T': {h: float(parameters['C0']['temperatures'][h]) for h in HEADS},
             'vec': {'w': np.asarray(parameters['C1']['parent']['w'], dtype=np.float64),
                     'b': np.asarray(parameters['C1']['parent']['b'], dtype=np.float64)},
             'pw': {k: float(heads['narrow_whiff'][k]) for k in ('a', 'b')},
             'pg': {k: float(heads['ground_ball_per_pitch'][k]) for k in ('a', 'b')},
             'dir': {'W': np.asarray(parameters['C2']['parent']['W'], dtype=np.float64),
                     'b': np.asarray(parameters['C2']['parent']['b'], dtype=np.float64)},
             'blocks': blocks, 'clip': float(parameters['C3']['clip'])}
    if model['vec']['w'].shape != (10,) or model['vec']['b'].shape != (10,) or model['dir']['W'].shape != (10, 10) \
            or model['dir']['b'].shape != (10,) or not 0 < model['clip'] < .5 \
            or not all(np.isfinite(v).all() for v in (model['vec']['w'], model['vec']['b'], model['dir']['W'], model['dir']['b'])):
        raise PinMismatch('calibrator parameter geometry')
    return model


class Calibrators:
    def __init__(self, models, selected, sha256, tolerance, leaf_order):
        self.models, self.selected, self.sha256, self.tolerance = models, selected, sha256, tolerance
        self.leaf_order = list(leaf_order)

    def c0(self, seed, z):
        return apply_arm(self.models[seed], 'C0', z, self.tolerance)

    def selected_arm(self, seed, z):
        return apply_arm(self.models[seed], self.selected[seed], z, self.tolerance)


def load_calibrators(config, seeds, temperatures=None):
    """``Calibrators`` for ``seeds`` from the pinned CAL2 result. ``temperatures``: {seed: {head: T}} from the
    pinned temperatures.json files; when given, the C0 temperatures in the result must equal them."""
    c = config['calibration']
    result_raw = read_pinned(c['result']['path'], c['result']['sha256'])
    terminal = json.loads(read_pinned(c['terminal']['path'], c['terminal']['sha256']))
    result = json.loads(result_raw)
    leaf_order = config['metrics']['leaf_order']
    if terminal.get('success') is not True or terminal.get('result_sha256') != c['result']['sha256']:
        raise PinMismatch('CAL2 terminal.json does not say success for this result')
    if result.get('leaf_order') != list(leaf_order):
        raise PinMismatch('CAL2 leaf order differs from the registered one')
    models, selected = {}, {}
    for s in seeds:
        block = result['seeds'][str(s)]
        arm = block['cal_selected']
        if arm not in c['arms'] or result.get('cal_selected', {}).get(str(s)) != arm \
                or terminal.get('cal_selected', {}).get(str(s)) != arm:
            raise PinMismatch('CAL2 selected arm is not stated consistently for seed '+str(s))
        models[s] = model_from_parameters(block['parameters'], leaf_order)
        selected[s] = arm
        if temperatures is not None and any(
                abs(models[s]['T'][h]-temperatures[s][h]) > c['temperature_match_tolerance'] for h in HEADS):
            raise PinMismatch('CAL2 C0 temperatures differ from temperatures.json for seed '+str(s))
    return Calibrators(models, selected, {'result': c['result']['sha256'], 'terminal': c['terminal']['sha256'],
                                          'code': c['code']['sha256']}, c['row_sum_tolerance'], leaf_order)
