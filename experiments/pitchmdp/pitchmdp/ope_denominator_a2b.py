"""EXP-OPE-RES-02 B2 denominator (design 4.1, 5.5, 9.1 item 4): the frozen A3 B2 behavior model at the ledger
decisions, the rho recomputation, B2 validity, E5 and the reward-free weight diagnostics.

- ``load_frozen_b2``: the A3 freeze-primary manifest (sha256 = the registration pin) and the three frozen B2 files
  (each sha256 = its pin, raw bytes) -> ``Calibrated(LGBMBehavior, calibrator)`` through the A3 ``BehaviorModel``
  protocol (A3 design :233; ``behavior_model`` / ``behavior_calibration`` imported unchanged; lightgbm 4.6.0 is
  the pinned inference environment).
- ``decision_vectors``: pi_b_hat_B2 at every evaluated decision of an E0 PA (the evaluated prefix, as the A3
  s6-weights stage), through the frozen pre-pitch context row of its ``context_key`` (pitcher, side, count and the
  previous type checked against the ledger row), keyed by (pa_id, decision_index).
- ``b2_validity`` and ``b2_decision_stream``: the B2 rule (a decision with pi(logged) > 0 and pi_b_hat_B2(logged)
  = 0, or a nonfinite rho, invalidates every B2 row; counts reported) and the same decision rows with
  ``logging`` := pi_b_hat_B2 and rho = pi / pi_b_hat_B2 recomputed for both policies (target policies, q-hat, the
  L1-R classification and the point set P unchanged; a LOGGING_POSITIVITY row keeps rho = 0 under any
  denominator).
- ``b2_weights``: the A3 ``behavior_weights.weight_diagnostics`` (E5, ESS, rho / W quantiles and tails, KL(B2 || B0),
  weight shift D) on the rehearsal ledger, plus the ESS ratio to B0; ``gint_r3`` reproduces the recorded A3 S6
  B2 diagnostics at relative 1e-9 (G-INT-R3). Rewards are never read here.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from . import behavior_weights as bw
from . import policy_estimator as est
from .behavior_calibration import calibrator_from_identity
from .behavior_model import Calibrated, LGBMBehavior, START, parse_context_key
from .policy_artifacts import IntegrityError, _require

B2_FILES = ('02-fit/b2/model.txt', '02-fit/b2/b2.json', '03-calibrate/calibrators/B2.json')
PROBABILITY_SUM_TOLERANCE = 1e-9  # policy_estimator._probabilities
GINT_R3_RTOL = 1e-9
E5_THRESHOLDS = {'pa': 100, 'game': 30}  # A3 design :197 (MLB-2026 preregistration :519-521)


def _sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def load_frozen_b2(manifest_path, manifest_sha256, artifacts_sha256, repertoire, *, lightgbm_version='4.6.0'):
    """The frozen A3 B2 primary arm: manifest bytes = pin, each file's bytes = its pin and its manifest entry."""
    import lightgbm
    _require(lightgbm.__version__ == lightgbm_version, f'B2 inference needs lightgbm {lightgbm_version} (pinned)')
    manifest_path = Path(manifest_path)
    raw = manifest_path.read_bytes()
    _require(_sha256_bytes(raw) == manifest_sha256, 'A3 freeze-primary manifest differs from its registration pin')
    manifest = json.loads(raw)
    root = manifest_path.parent.parent  # <attempt root>/04-freeze-primary/manifest.json
    _require(set(artifacts_sha256) == set(B2_FILES), 'the three frozen B2 files must be pinned')
    texts = {}
    for rel in B2_FILES:
        entry = (manifest.get('artifacts') or {}).get(rel)
        data = (root / rel).read_bytes()
        _require(entry is not None and entry['sha256'] == artifacts_sha256[rel] == _sha256_bytes(data),
                 f'frozen B2 artifact differs from its pin or manifest: {rel}')
        texts[rel] = data.decode()
    meta = json.loads(texts['02-fit/b2/b2.json'])
    calibrator = json.loads(texts['03-calibrate/calibrators/B2.json'])
    _require(calibrator.get('status') != 'CALIBRATION_FAILED', 'the frozen B2 calibrator failed (A3 record)')
    model = LGBMBehavior(texts['02-fit/b2/model.txt'], meta['best_round'], repertoire, meta.get('record'))
    return Calibrated(model, calibrator_from_identity(calibrator['identity']), 'B2')


# ---------------------------------------------------------------- decisions and vectors
def e0_map(facts):
    """{pa_id: (in_E0, reason)} from the ledger-stage facts (PolicyRuntime.start_population at the first row)."""
    return {pa_id: (bool(info['in_population']), info.get('start_reason')) for pa_id, info in facts.items()}


def evaluated_prefix(rows):
    """The evaluated decisions before the first refusal (the recursion steps that need pi_b_hat)."""
    out = []
    for row in rows:
        if row['status'] not in est.EVALUATED:
            break
        out.append(row)
    return out


def b2_decision_rows(decisions, e0):
    """Every evaluated-prefix decision of an E0 PA, in ledger order."""
    return [row for pa_id, rows in bw.by_pa(decisions) if e0[pa_id][0] for row in evaluated_prefix(rows)]


def decision_vectors(model, table, rows):
    """pi_b_hat_B2 [18] per decision via its frozen pre-pitch context row (A3 s6-weights rule): the row exists and
    agrees with the decision on pitcher, batter side, count and the previous pitch type."""
    if not rows:
        return {}
    keys = np.array([parse_context_key(r['context_key']) for r in rows], dtype=np.int64)
    positions = table.positions(keys)
    _require(not (positions < 0).any(), f'{int((positions < 0).sum())} decisions have no frozen context row')
    frozen = table.frame.iloc[positions]
    previous = np.array([r['history_actions'][-1] if r['history_actions'] else START for r in rows], dtype=object)
    same = ((frozen.pitcher.to_numpy(np.int64) == np.array([int(r['pitcher']) for r in rows]))
            & (frozen.batter_side.to_numpy(object) == np.array([r['batter_side'] for r in rows], dtype=object))
            & (frozen.balls.to_numpy(np.int64) == np.array([int(r['balls']) for r in rows]))
            & (frozen.strikes.to_numpy(np.int64) == np.array([int(r['strikes']) for r in rows]))
            & (frozen.previous.to_numpy(object) == previous))
    _require(same.all(), f'{int((~same).sum())} decisions disagree with their frozen context rows')
    vectors = model.batch(table, positions)
    return {(r['pa_id'], r['decision_index']): vectors[i] for i, r in enumerate(rows)}


RAW_KEY_COLUMNS = ['game_date', 'game_pk', 'at_bat_number', 'pitch_number', 'pitcher']  # behavior_data / workload


def dev_feature_table(frame, raw, *, games, archetypes, b0, hands):
    """The frozen pre-pitch B2 feature table of the given DEV games, rebuilt as the A3 runner builds it
    (``feature_context`` then ``split_table``): ``frame`` = processed regular-season rows (any split labels; the A3
    labels are recomputed from the dates, only the A3 used splits kept, A3 order), ``raw`` = the raw Statcast keys
    (RAW_KEY_COLUMNS) of at least the DEV games (workload), ``b0`` = the registered BC_P (``RegisteredBC``),
    ``hands`` = its hand registry, ``archetypes`` = the frozen G0 archetypes. Style tables use TRAIN / earlystop /
    temperature rows (<= 2025-05-31, the DEV snapshot as of 2025-07-01); the workload and every row feature are
    game-local, so restricting them to the chosen games changes no value (tested against the A3 construction)."""
    import pandas as pd
    from .behavior_calibration_registry import ACTION_CODES, USED_SPLITS
    from .behavior_data import split_of
    from .behavior_features import FeatureContext, KEY, split_table, style_tables, workload_features
    from .policy_requests import volume_edges
    used = frame.assign(split=split_of(frame.game_date))
    used = used.loc[used.split.isin(USED_SPLITS)].sort_values(['game_date', *KEY], kind='stable').reset_index(drop=True)
    _require(not pd.to_datetime(used.game_date).dt.year.ge(2026).any(), 'a 2026 row reached the B2 feature table')
    chosen = set(int(g) for g in games)
    dev = used.loc[used.split.eq('dev') & used.game_pk.isin(chosen)]
    _require(set(dev.game_pk.astype(int)) == chosen, 'a chosen game has no DEV rows')
    train = used.loc[used.split.eq('train')]
    keys = raw.loc[raw.game_pk.isin(chosen), RAW_KEY_COLUMNS].reset_index(drop=True)
    ctx = FeatureContext(vocabulary=ACTION_CODES, archetypes=archetypes, b0=b0.vector,
                         train_volume={int(p): int(n) for p, n in train.groupby('pitcher').size().items()},
                         volume_edges=tuple(volume_edges(train, (.25, .75))), style=style_tables(used),
                         workload=workload_features(dev, keys), hands=hands, repertoire=b0.repertoire)
    return split_table(dev, 'dev', ctx)[0]


# ---------------------------------------------------------------- validity and the B2 stream
def b2_validity(decisions, vectors, e0_ids):
    """The B2 rule: counts of decisions with pi(logged) > 0 and pi_b_hat_B2(logged) = 0, of nonfinite rho and of
    supports differing from B0 (the logged action must stay inside the B0 support R_j); any count > 0 or a
    missing / malformed vector invalidates every B2 row."""
    checked = zero_mass = nonfinite = support = missing = malformed = 0
    for pa_id, rows in bw.by_pa(decisions):
        if pa_id not in e0_ids:
            continue
        for row in evaluated_prefix(rows):
            checked += 1
            vector = vectors.get((pa_id, row['decision_index']))
            if vector is None:
                missing += 1
                continue
            v = np.asarray(vector, dtype=np.float64)
            result = row['result']
            logging = np.asarray(result['logging'], dtype=np.float64)
            if v.shape != logging.shape or not np.isfinite(v).all() or (v < 0).any() \
                    or abs(v.sum() - 1) > PROBABILITY_SUM_TOLERANCE:
                malformed += 1
                continue
            a = result['logged_index']
            pi = [float(result[p][a]) for p in ('candidate', 'reference')]
            if v[a] == 0 and any(p > 0 for p in pi):
                zero_mass += 1
            with np.errstate(divide='ignore', invalid='ignore'):
                rhos = [p / v[a] for p in pi] if v[a] > 0 else [math.nan]
            if not all(math.isfinite(r) for r in rhos):
                nonfinite += 1
            if not np.array_equal(v > 0, logging > 0):
                support += 1
    record = {'decisions_checked': checked, 'zero_mass_with_positive_pi': zero_mass, 'nonfinite_rho': nonfinite,
              'support_differs_from_b0': support, 'missing_vectors': missing, 'malformed_vectors': malformed}
    record['pass'] = checked > 0 and all(v == 0 for k, v in record.items() if k != 'decisions_checked')
    record['rule'] = ('pi(logged) > 0 with pi_b_hat_B2(logged) = 0, a nonfinite rho, a support differing from B0, a '
                      'missing or malformed vector: every B2 row invalid on this ledger')
    return record


def b2_decision_stream(decisions, vectors, e0_ids):
    """A factory of fresh streams of the ledger rows with pi_b_hat_B2 in the evaluated prefix of every E0 PA:
    ``logging`` replaced and rho = pi(a) / pi_b_hat_B2(a) recomputed for both policies (exactly the floats
    policy_estimator recomputes); every other row and field unchanged (a LOGGING_POSITIVITY row keeps its B0
    logging and rho = 0: pi(a) = 0 under any denominator)."""
    def stream():
        for pa_id, rows in bw.by_pa(decisions):
            prefix = len(evaluated_prefix(rows)) if pa_id in e0_ids else 0
            for i, row in enumerate(rows):
                if i >= prefix:
                    yield row
                    continue
                v = np.asarray(vectors[(pa_id, row['decision_index'])], dtype=np.float64)
                result = dict(row['result'])
                a = result['logged_index']
                result['logging'] = v.tolist()
                for policy in ('candidate', 'reference'):
                    result[f'rho_{policy}'] = float(np.asarray(result[policy], dtype=np.float64)[a] / v[a])
                yield {**row, 'result': result}
    return stream


# ---------------------------------------------------------------- weight diagnostics (A3 code reused)
def _request_vectors(decisions, vectors, e0):
    """A3 weight_diagnostics keys its vectors by request_id: map the (pa_id, decision_index) vectors."""
    out = {}
    for row in b2_decision_rows(decisions, e0):
        _require(isinstance(row.get('request_id'), str), 'ledger decision rows need their request_id')
        out[row['request_id']] = vectors[(row['pa_id'], row['decision_index'])]
    return out


def b2_weights(decisions, e0, vectors, *, large, mei, draws, seed, ess_min=E5_THRESHOLDS):
    """Reward-free diagnostics of the B2 denominator on one ledger (A3 ``weight_diagnostics``): E5, ESS, rho / W
    quantiles and tail shares, KL(B2 || B0) and the weight shift D against the B0 recursion; plus the ESS ratio
    B2 / B0 per policy and level. Returns (public diagnostics, B0 diagnostics)."""
    pas = bw.by_pa(decisions)
    common = {'large': large, 'mei': mei, 'ess_min': dict(ess_min), 'draws': draws, 'seed': seed}
    baseline = bw.weight_diagnostics(pas, e0, **common)
    new = bw.weight_diagnostics(pas, e0, new_vectors=_request_vectors(decisions, vectors, e0), baseline=baseline,
                                **common)
    ratio = {}
    for policy in ('candidate', 'reference'):
        b0, b2 = baseline['policies'][policy]['ess'], new['policies'][policy]['ess']
        ratio[policy] = {level: (None if not b0[level] or b2[level] is None else b2[level] / b0[level])
                         for level in ('pa', 'game')}
    out = bw.public(new)
    out['large_weight_threshold'] = large
    return {**out, 'ess_ratio_vs_b0': ratio}, bw.public(baseline)


def _relative_same(value, expected, rtol):
    if isinstance(expected, dict):
        return isinstance(value, dict) and all(_relative_same(value.get(k), v, rtol) for k, v in expected.items())
    if isinstance(expected, list):
        return isinstance(value, list) and len(value) == len(expected) and all(
            _relative_same(a, b, rtol) for a, b in zip(value, expected))
    if isinstance(expected, bool) or not isinstance(expected, (int, float)):
        return value == expected
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    a, b = float(value), float(expected)
    if not (math.isfinite(a) and math.isfinite(b)):
        return a == b
    return abs(a - b) <= rtol * abs(b)


GINT_R3_KEYS = ('e0_pas', 'policies', 'candidate_over_reference_ess', 'E5', 'kl_new_vs_b0', 'weight_shift_D')


def gint_r3(new, recorded, *, rtol=GINT_R3_RTOL):
    """G-INT-R3: the B2 diagnostics recomputed on S6 (allowlisted reader, frozen B2) reproduce the recorded A3
    S6 B2 diagnostics at relative 1e-9 (counts and flags exactly). Returns the record (pass/fail and keys)."""
    compared = [k for k in GINT_R3_KEYS if k in recorded]
    mismatched = [k for k in compared if not _relative_same(new.get(k), recorded[k], rtol)]
    return {'check': 'G-INT-R3', 'result': 'FAIL' if mismatched else 'PASS', 'rtol': rtol, 'compared_keys': compared,
            'mismatched_keys': mismatched}


def require_valid_vectors(record):
    if not record['pass']:
        raise IntegrityError(f'B2 denominator invalid on this ledger: {record}')
    return record
