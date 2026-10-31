"""N1 shadow adapter: (17x199 features, context tokens) -> per-seed margins, C0 ensemble, SEL ensemble.

The 199 features. ``features.build_candidates`` of the pinned Step 5 source returns the 17x199 matrix encoded with
the 2024 F1-design encoder AND the 17x91 raw columns. N1 was trained on the same raw 91 columns (same formulas,
order, SETTINGS, NaN -> median) encoded with the scale-002 encoder, whose mean/scale differ on every column. So the
model input is ``encode_for_seq(raw91, scale-002 encoder)`` (the arithmetic of ``features.apply_encoder`` =
``scale_safe.apply_encoder``: non-finite -> stored median, z clipped to [-clip, clip], three hinge blocks, float32).
The F1-encoded matrix is used only for its sha256 (gate V against the Step 5 ledger).

Known residual differences from the training rows (documented, not "fixed"):
  * training looked up the previous pitch among ACCEPTED corpus rows only, so an automatic ball before the pitch
    gave previous_valid = 0, while GUMBO uses the last isPitch event of the plate appearance;
  * training mapped a null previous pitch type to previous_UNKNOWN = 1, while Step 5 abstains
    (``previous_pitch_type_not_yet_classified``);
  * the strict-prior profile pool has one more ball-in-play row with a missing bb_type than the training pool
    (713,022 vs 713,021 rows);
  * float summation order may differ by about one ulp.

Ensembles: C0 = mean over seeds of leaves(softmax(zp/Tp), expit(zw/Tw), expit(zg/Tg)); SEL = mean over seeds of each
seed's own CAL2-selected arm. Leaf order as registered (ball, foul, hit_by_pitch, narrow_whiff, other_strike,
ground_ball_in_play, other_in_play).
"""
from __future__ import annotations

import json
import time

import numpy as np

from . import PinMismatch, read_pinned
from .calibration import HEADS


def load_seq_encoder(config):
    """(encoder dict, sha256) of the pinned scale-002 encoder after a geometry check."""
    spec = config['encoder_seq']
    e = json.loads(read_pinned(spec['path'], spec['sha256']))
    raw, blocks = spec['raw_columns'], spec['knot_blocks']
    try:
        ok = (all(len(e[k]) == raw for k in ('median', 'mean', 'scale')) and len(e['knots']) == blocks
              and all(len(k) == len(e['continuous']) for k in e['knots'])
              and raw+blocks*len(e['continuous']) == config['model']['feature_dimension']
              and e['dimension'] == config['model']['feature_dimension']
              and all(type(i) is int and 0 <= i < raw for i in e['continuous'])
              and all(np.isfinite(np.asarray(e[k], dtype=np.float64)).all() for k in ('median', 'mean', 'scale'))
              and (np.asarray(e['scale'], dtype=np.float64) > 0).all())
    except (KeyError, TypeError, ValueError):
        ok = False
    if not ok:
        raise PinMismatch('scale-002 encoder geometry differs from the 91 -> 199 contract')
    return e | {'z_clip': spec['z_clip']}, spec['sha256']


def encode_for_seq(raw91, enc002):
    """Raw (rows, 91) float64 -> (rows, 199) float32 with the stored statistics of ``enc002`` only. Pure: same
    arithmetic as the pinned ``features.apply_encoder`` (the runner asserts bitwise equality on every decision)."""
    raw91 = np.asarray(raw91, dtype=np.float64)
    median, mean, scale = (np.array(enc002[k]) for k in ('median', 'mean', 'scale'))
    cols = np.array(enc002['continuous'], int)
    clip = enc002.get('z_clip', 8)
    z = np.clip((np.where(np.isfinite(raw91), raw91, median)-mean)/scale, -clip, clip)
    return np.column_stack([z]+[np.maximum(z[:, cols]-k, 0) for k in np.array(enc002['knots'])]).astype(np.float32)


class SeqAdapter:
    def __init__(self, ensemble, calibrators, rows):
        if sorted(calibrators.models) != sorted(ensemble.seeds):
            raise ValueError('calibrator seeds differ from the model seeds')
        self.ensemble, self.calibrators, self.rows = ensemble, calibrators, rows

    def forecast(self, x, tokens):
        """``x``: (rows, 199) float32 scale-002 encoded; ``tokens``: (L,) arrays of ``tokens.build``. Returns
        logits {seed: {head: list}}, per-seed C0 and selected-arm rows, both ensembles (rows x 7 arrays) and the
        forward / calibration wall times in milliseconds."""
        t0 = time.perf_counter()
        xt, tok = self.ensemble.tensors(x, tokens, self.rows)
        z = self.ensemble.forward(xt, tok)
        t1 = time.perf_counter()
        c0, sel = {}, {}
        for seed in self.ensemble.seeds:
            if z[seed]['parent'].shape != (self.rows, 10) or z[seed]['narrow_whiff'].shape != (self.rows,) \
                    or z[seed]['ground_ball_per_pitch'].shape != (self.rows,):
                raise ValueError('margin shape')
            c0[seed] = self.calibrators.c0(seed, z[seed])
            sel[seed] = self.calibrators.selected_arm(seed, z[seed])
        c0_ensemble = np.mean([c0[s] for s in self.ensemble.seeds], axis=0)
        sel_ensemble = np.mean([sel[s] for s in self.ensemble.seeds], axis=0)
        for q in (c0_ensemble, sel_ensemble):
            if q.shape != (self.rows, 7) or not np.isfinite(q).all() or (q < 0).any() \
                    or np.abs(q.sum(1)-1).max() > self.calibrators.tolerance:
                raise ValueError('ensemble seven-leaf coherence')
        t2 = time.perf_counter()
        return {'logits': {str(s): {h: z[s][h].tolist() for h in HEADS} for s in self.ensemble.seeds},
                'c0_by_seed': c0, 'sel_by_seed': sel, 'c0_ensemble': c0_ensemble, 'sel_ensemble': sel_ensemble,
                'arms': {str(s): self.calibrators.selected[s] for s in self.ensemble.seeds},
                'forward_ms': (t1-t0)*1000, 'calibration_ms': (t2-t1)*1000}
