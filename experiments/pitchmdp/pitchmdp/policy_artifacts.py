"""Frozen policy-runtime artifacts: TRAIN BC JSON, intervention support table, five-member G0.

Additive to ``matrix_policy``/``rollout_policy`` (ML-POLICY-RUNTIME-v1, COOP-016). Nothing
here reads data at import time or chooses a real artifact; callers pass explicit paths
and pins. Loading an artifact re-verifies its bytes and structure, never raw TRAIN rows:
provenance that was not observed by ``export_train_bc`` stays labelled as declared.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

from .data import hash_file
from .matrix_data import canonical_hash
from .matrix_policy import FrozenGEnsemble, fit_bc
from .matrix_sharing import SharingPredictor
from .rollout_policy import CategoricalBC

BC_SCHEMA = 'pitcheezy.train_bc.v1'
SUPPORT_SCHEMA = 'pitcheezy.intervention_support.v1'
HANDS_SCHEMA = 'pitcheezy.pitcher_hands.v1'
STYLE_SCHEMA = 'pitcheezy.batter_style_snapshot.v1'
HAND_RULE = 'train_single_hand_v1'  # matrix_panel hand rule: one observed L/R and no missing hand, else AMBIGUOUS


def normalize_hand(value):
    """A throwing/batting hand code or None (None, NaN, pd.NA and '' are all missing)."""
    if value is None:
        return None
    try:
        if value != value:  # NaN
            return None
    except (TypeError, ValueError):  # pd.NA comparisons
        return None
    text = str(value)
    return None if not text or text in ('nan', '<NA>', 'None') else text


def single_hand(values):
    """HAND_RULE over one pitcher's TRAIN rows: 'L'/'R' when every row has that hand, else 'AMBIGUOUS'."""
    hands = [normalize_hand(v) for v in values]
    return hands[0] if hands and hands[0] in ('L', 'R') and all(h == hands[0] for h in hands) else 'AMBIGUOUS'
TRAIN_WINDOW = ('2023-05-15', '2025-04-30')  # matrix_policy.fit_bc guard
G0_PROTOCOL = 'g0_research_frozen_v1'
G0_SEEDS = (0, 1, 2, 3, 4)
G0_SOURCE_ARMS = ('g', 'g', 'g', 'c1', 'c1')
G0_CELL = 'G0-global'
HEX = frozenset('0123456789abcdef')


class IntegrityError(ValueError):
    """FAILED_INTEGRITY: malformed/tampered input. Stop; never repair or substitute."""


class Unsupported(Exception):
    """Legitimate refusal; the request stays in the denominator with this status."""
    def __init__(self, status, detail=''):
        super().__init__(f'{status}: {detail}' if detail else status)
        self.status = status


def vocabulary_sha256(vocabulary):
    return hashlib.sha256('\n'.join(vocabulary).encode()).hexdigest()


def _require(condition, message):
    if not condition:
        raise IntegrityError(message)


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and set(value) <= HEX


def _count(value):
    return type(value) is int and value > 0


def _date(value):
    try:
        return date.fromisoformat(value) if isinstance(value, str) and len(value) == 10 else None
    except ValueError:
        return None


def _nonterminal(balls, strikes):
    return type(balls) is int and type(strikes) is int and 0 <= balls <= 3 and 0 <= strikes <= 2


# ---------------------------------------------------------------- canonical JSON files

def _canonical_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                       allow_nan=False) + '\n').encode()


def write_exclusive(path, data: bytes):
    """Atomic + exclusive: temp file fsync'd in the same directory, then hard-linked.

    ``os.link`` fails if ``path`` exists, so a frozen artifact is never overwritten and
    a crash never leaves a partial file under the final name.
    """
    path = Path(path)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name + '.')
    try:
        with os.fdopen(handle, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)
    return hash_file(path)


def _envelope(schema, payload):
    return {'schema': schema, 'content_sha256': canonical_hash(payload), 'payload': payload}


def _read_envelope(path, schema, expected_sha256):
    raw = Path(path).read_bytes()
    _require(expected_sha256 is None or _sha(expected_sha256), 'file pin must be lowercase sha256')
    file_sha = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and file_sha != expected_sha256:
        raise IntegrityError(f'{schema}: file sha256 differs from pin')
    try:
        value = json.loads(raw, parse_constant=lambda name: (_ for _ in ()).throw(ValueError(name)))
    except ValueError as error:
        raise IntegrityError(f'{schema}: truncated or malformed JSON') from error
    _require(isinstance(value, dict) and set(value) == {'schema', 'content_sha256', 'payload'},
             f'{schema}: envelope fields differ')
    _require(value['schema'] == schema, f'unsupported schema/version {value["schema"]!r}; expected {schema}')
    _require(_canonical_bytes(value) == raw, f'{schema}: bytes are not the canonical serialization')
    _require(canonical_hash(value['payload']) == value['content_sha256'], f'{schema}: content hash mismatch')
    return value['payload'], file_sha


# ---------------------------------------------------------------- TRAIN BC

def _counts(counter, vocabulary):
    return [[a, counter[a]] for a in vocabulary if counter.get(a, 0)]


def bc_payload(bc: CategoricalBC, provenance: dict):
    """Deterministic, typed serialization of the complete fitted CategoricalBC state."""
    vocabulary = list(bc.actions)
    cells = [{'pitcher': p, 'balls': b, 'strikes': s, 'batter_side': side, 'previous': prev,
              'counts': _counts(counter, vocabulary)}
             for (p, b, s, side, prev), counter in sorted(bc.cells.items())]
    payload = {
        'class': 'rollout_policy.CategoricalBC',
        'parameters': {'prior_strength': bc.prior_strength, 'minimum_action_count': bc.minimum_action_count},
        'vocabulary': vocabulary, 'vocabulary_sha256': vocabulary_sha256(vocabulary),
        'league': _counts(bc.league, vocabulary),
        'pitchers': [{'pitcher': p, 'counts': _counts(bc.pitchers[p], vocabulary)} for p in sorted(bc.pitchers)],
        'cells': cells, 'provenance': provenance,
        'unknown_pitcher_rule': 'CategoricalBC league fallback is flagged; runtime refuses it',
    }
    validate_bc_payload(payload)
    return payload


def validate_bc_payload(payload):
    _require(isinstance(payload, dict) and set(payload) == {'class', 'parameters', 'vocabulary',
             'vocabulary_sha256', 'league', 'pitchers', 'cells', 'provenance', 'unknown_pitcher_rule'},
             'BC payload fields differ')
    _require(payload['class'] == 'rollout_policy.CategoricalBC', 'BC class differs')
    params = payload['parameters']
    _require(isinstance(params, dict) and set(params) == {'prior_strength', 'minimum_action_count'}, 'BC parameters')
    _require(type(params['prior_strength']) in (int, float) and math.isfinite(params['prior_strength'])
             and params['prior_strength'] > 0, 'prior_strength must be finite positive')
    _require(_count(params['minimum_action_count']), 'minimum_action_count must be a positive int')
    vocabulary = payload['vocabulary']
    _require(isinstance(vocabulary, list) and vocabulary and all(isinstance(a, str) and a for a in vocabulary)
             and vocabulary == sorted(set(vocabulary)), 'vocabulary must be sorted unique nonempty strings')
    _require(payload['vocabulary_sha256'] == vocabulary_sha256(vocabulary), 'vocabulary sha256 mismatch')

    def table(rows):
        _require(isinstance(rows, list) and rows, 'count table must be a nonempty list')
        counter = Counter()
        for row in rows:
            _require(isinstance(row, list) and len(row) == 2 and row[0] in vocabulary and _count(row[1]),
                     'count rows need an in-vocabulary action and positive int count')
            _require(row[0] not in counter, 'duplicate action in count table')
            counter[row[0]] = row[1]
        _require([a for a, _ in rows] == [a for a in vocabulary if a in counter], 'count rows out of vocabulary order')
        return counter

    _require(isinstance(payload['pitchers'], list) and isinstance(payload['cells'], list), 'pitcher/cell tables must be lists')
    league = table(payload['league'])
    _require(sorted(league) == vocabulary, 'vocabulary must equal the fitted league actions')
    pitchers = {}
    for row in payload['pitchers']:
        _require(isinstance(row, dict) and set(row) == {'pitcher', 'counts'} and isinstance(row['pitcher'], str)
                 and row['pitcher'], 'pitcher row')
        _require(row['pitcher'] not in pitchers, 'duplicate pitcher')
        pitchers[row['pitcher']] = table(row['counts'])
    _require(list(pitchers) == sorted(pitchers), 'pitchers out of order')
    _require(sum(pitchers.values(), Counter()) == league, 'league counts differ from the pitcher sum')
    cells, by_pitcher = {}, defaultdict(Counter)
    for row in payload['cells']:
        _require(isinstance(row, dict) and set(row) == {'pitcher', 'balls', 'strikes', 'batter_side', 'previous', 'counts'},
                 'cell row fields')
        key = (row['pitcher'], row['balls'], row['strikes'], row['batter_side'], row['previous'])
        _require(row['pitcher'] in pitchers and _nonterminal(row['balls'], row['strikes'])
                 and isinstance(row['batter_side'], str) and row['batter_side']
                 and isinstance(row['previous'], str) and row['previous'], 'illegal cell key')
        _require(key not in cells, 'duplicate cell key')
        cells[key] = table(row['counts'])
        by_pitcher[row['pitcher']] += cells[key]
    _require(list(cells) == sorted(cells), 'cells out of order')
    _require(dict(by_pitcher) == pitchers, 'pitcher counts differ from the cell sum')
    _validate_provenance(payload['provenance'])
    _require(payload['provenance']['train_rows'] == sum(league.values()), 'train_rows differs from the fitted count tables')
    return league, pitchers, cells


PROVENANCE_FIELDS = {'train_start', 'train_end', 'dates', 'train_rows', 'source_ids', 'config_sha256',
                     'code_commit', 'data_version'}


def _validate_provenance(provenance):
    _require(isinstance(provenance, dict) and set(provenance) == PROVENANCE_FIELDS, 'provenance fields differ')
    start, end = _date(provenance['train_start']), _date(provenance['train_end'])
    low, high = (date.fromisoformat(d) for d in TRAIN_WINDOW)
    _require(start is not None and end is not None and low <= start <= end <= high,
             'TRAIN dates must be ISO dates inside the frozen 2023-05-15..2025-04-30 window')
    _require(provenance['dates'] in ('observed_by_export_train_bc', 'declared_unverified'),
             'dates must state whether they were observed from rows or only declared')
    _require(_count(provenance['train_rows']), 'train_rows must be a positive int')
    sources = provenance['source_ids']
    _require(isinstance(sources, dict) and sources and all(isinstance(k, str) and _sha(v) for k, v in sources.items()),
             'source_ids must map names to sha256')
    _require(_sha(provenance['config_sha256']), 'config_sha256 must be sha256')
    _require(isinstance(provenance['code_commit'], str) and len(provenance['code_commit']) == 40
             and set(provenance['code_commit']) <= HEX, 'code_commit must be a 40-hex commit')
    _require(isinstance(provenance['data_version'], str) and provenance['data_version'], 'data_version required')


class TrainBCArtifact:
    """A reloaded TRAIN BC plus its identities. ``bc`` is an ordinary CategoricalBC."""
    def __init__(self, payload, file_sha256):
        league, pitchers, cells = validate_bc_payload(payload)
        params = payload['parameters']
        bc = CategoricalBC(params['prior_strength'], params['minimum_action_count'])
        bc.actions = tuple(payload['vocabulary'])
        bc.league, bc.pitchers = league, defaultdict(Counter, pitchers)
        bc.cells = defaultdict(Counter, cells)
        self.bc, self.payload, self.file_sha256 = bc, payload, file_sha256
        self.sha256 = canonical_hash(payload)
        self.vocabulary, self.vocabulary_sha256 = bc.actions, payload['vocabulary_sha256']
        self.provenance = payload['provenance']

    def identity(self):
        return {'bc_sha256': self.sha256, 'file_sha256': self.file_sha256,
                'vocabulary_sha256': self.vocabulary_sha256, 'train_dates': self.provenance['dates']}


def save_train_bc(bc, path, provenance):
    payload = bc_payload(bc, provenance)
    file_sha = write_exclusive(path, _canonical_bytes(_envelope(BC_SCHEMA, payload)))
    return TrainBCArtifact(payload, file_sha)


def load_train_bc(path, expected_sha256):
    """Pin is mandatory: an unpinned reload could silently swap the frozen BC."""
    _require(_sha(expected_sha256), 'a sha256 pin is required to load a TRAIN BC artifact')
    return TrainBCArtifact(*_read_envelope(path, BC_SCHEMA, expected_sha256))


def export_train_bc(store, train, path, *, source_ids, config_sha256, code_commit, data_version,
                    prior_strength=20., minimum_action_count=1):
    """Future TRAIN export: fit through the existing ``fit_bc`` date/split guard, then save.

    Dates recorded here are the observed min/max of the rows passed in; nothing else about
    the TRAIN population (selection, completeness) is verified.
    """
    bc = fit_bc(store, train, prior_strength, minimum_action_count)
    dates = sorted(str(d)[:10] for d in train.game_date)
    return save_train_bc(bc, path, {'train_start': dates[0], 'train_end': dates[-1],
        'dates': 'observed_by_export_train_bc', 'train_rows': len(train), 'source_ids': dict(source_ids),
        'config_sha256': config_sha256, 'code_commit': code_commit, 'data_version': data_version})


# ---------------------------------------------------------------- intervention support table

def support_table_payload(bc_artifact, rows):
    """rows: iterable of (pitcher, batter_side, mask) from ``PolicyInputs.support`` or equivalent."""
    table = {}
    for pitcher, side, mask in rows:
        mask = np.asarray(mask)
        _require(mask.dtype == np.bool_ and mask.shape == (len(bc_artifact.vocabulary),),
                 'support mask must be a bool vector over the BC vocabulary')
        mask = mask.tolist()
        key = (str(pitcher), str(side))
        _require(table.get(key, mask) == mask, f'support differs within one (pitcher, side) key: {key}')
        table[key] = mask
    payload = {'bc_sha256': bc_artifact.sha256, 'vocabulary_sha256': bc_artifact.vocabulary_sha256,
               'rows': [{'pitcher': p, 'batter_side': s, 'actions': [a for a, ok in zip(bc_artifact.vocabulary, m) if ok]}
                        for (p, s), m in sorted(table.items())]}
    validate_support_payload(payload, bc_artifact)
    return payload


def validate_support_payload(payload, bc_artifact):
    _require(isinstance(payload, dict) and set(payload) == {'bc_sha256', 'vocabulary_sha256', 'rows'}, 'support fields')
    _require(payload['bc_sha256'] == bc_artifact.sha256 and payload['vocabulary_sha256'] == bc_artifact.vocabulary_sha256,
             'support table was built for another BC/vocabulary')
    table = {}
    for row in payload['rows']:
        _require(isinstance(row, dict) and set(row) == {'pitcher', 'batter_side', 'actions'}, 'support row fields')
        key = (row['pitcher'], row['batter_side'])
        _require(key not in table and isinstance(row['actions'], list), 'duplicate support key')
        _require(row['actions'] == [a for a in bc_artifact.vocabulary if a in row['actions']]
                 and len(set(row['actions'])) == len(row['actions']), 'support actions unknown or out of order')
        counts = bc_artifact.bc.pitchers.get(row['pitcher'])
        _require(counts is not None and all(counts[a] >= bc_artifact.bc.minimum_action_count for a in row['actions']),
                 'intervention support must be inside the TRAIN BC support of a known pitcher')
        table[key] = np.array([a in row['actions'] for a in bc_artifact.vocabulary])
        table[key].setflags(write=False)
    _require(list(table) == sorted(table), 'support rows out of order')
    return table


def save_support_table(bc_artifact, rows, path):
    payload = support_table_payload(bc_artifact, rows)
    return payload, write_exclusive(path, _canonical_bytes(_envelope(SUPPORT_SCHEMA, payload)))


def load_support_table(path, expected_sha256, bc_artifact):
    _require(_sha(expected_sha256), 'a sha256 pin is required to load a support table')
    payload, file_sha = _read_envelope(path, SUPPORT_SCHEMA, expected_sha256)
    return validate_support_payload(payload, bc_artifact), canonical_hash(payload)


# ---------------------------------------------------------------- TRAIN pitcher hand registry (D-8)

def hand_registry_payload(bc_artifact, hands, train_rows):
    """``hands``: {pitcher: 'L'|'R'|'AMBIGUOUS'} over exactly the BC pitchers (HAND_RULE)."""
    _require(isinstance(hands, dict) and set(hands) == set(bc_artifact.bc.pitchers), 'hand registry covers the BC pitchers')
    _require(_count(train_rows), 'hand registry needs its positive TRAIN row count')
    payload = {'bc_sha256': bc_artifact.sha256, 'rule': HAND_RULE, 'train_rows': train_rows,
               'rows': [{'pitcher': str(p), 'hand': hands[p]} for p in sorted(hands)]}
    validate_hand_registry(payload, bc_artifact)
    return payload


def validate_hand_registry(payload, bc_artifact):
    _require(isinstance(payload, dict) and set(payload) == {'bc_sha256', 'rule', 'train_rows', 'rows'}
             and _count(payload['train_rows']), 'hand fields')
    _require(payload['bc_sha256'] == bc_artifact.sha256 and payload['rule'] == HAND_RULE,
             'hand registry was built for another BC or rule')
    hands = {}
    for row in payload['rows']:
        _require(isinstance(row, dict) and set(row) == {'pitcher', 'hand'} and row['pitcher'] not in hands
                 and row['hand'] in ('L', 'R', 'AMBIGUOUS'), 'hand registry row')
        hands[row['pitcher']] = row['hand']
    _require(list(hands) == sorted(hands) and set(hands) == set(bc_artifact.bc.pitchers),
             'hand registry must list exactly the BC pitchers in order')
    return hands


def save_hand_registry(bc_artifact, hands, train_rows, path):
    payload = hand_registry_payload(bc_artifact, hands, train_rows)
    return payload, write_exclusive(path, _canonical_bytes(_envelope(HANDS_SCHEMA, payload)))


def load_hand_registry(path, expected_sha256, bc_artifact):
    _require(_sha(expected_sha256), 'a sha256 pin is required to load a hand registry')
    payload, _ = _read_envelope(path, HANDS_SCHEMA, expected_sha256)
    return validate_hand_registry(payload, bc_artifact), canonical_hash(payload)


# ---------------------------------------------------------------- batter style snapshot (D-7)

def save_style_snapshot(snapshot, as_of, source, path):
    """Frozen batter style priors: {batter: HISTORY_COLUMNS values} plus the league row (key 'league',
    reliability 0) for batters unseen before ``as_of`` (exclusive)."""
    columns = list(snapshot.columns)
    rows = [{'batter': str(index), 'values': [float(v) for v in row]} for index, row in zip(snapshot.index, snapshot.to_numpy())]
    _require(all(np.isfinite(r['values']).all() for r in rows) and rows[-1]['batter'] == 'league', 'style snapshot rows')
    payload = {'as_of_exclusive': str(as_of), 'columns': columns, 'source': dict(source), 'rows': rows}
    return payload, write_exclusive(path, _canonical_bytes(_envelope(STYLE_SCHEMA, payload)))


def load_style_snapshot(path, expected_sha256):
    """(DataFrame indexed by batter id string with float32 columns, as_of, content sha256)."""
    import pandas as pd
    _require(_sha(expected_sha256), 'a sha256 pin is required to load a style snapshot')
    payload, _ = _read_envelope(path, STYLE_SCHEMA, expected_sha256)
    frame = pd.DataFrame([r['values'] for r in payload['rows']], index=[r['batter'] for r in payload['rows']],
                         columns=payload['columns']).astype(np.float32)
    return frame, payload['as_of_exclusive'], canonical_hash(payload)


# ---------------------------------------------------------------- five-member frozen G0

def validate_g0_manifest(bundle):
    """Structural checks on a G0-RESEARCH-FROZEN-v1-shaped bundle (no file reads)."""
    _require(isinstance(bundle, dict) and bundle.get('protocol') == G0_PROTOCOL, 'G0 protocol differs')
    _require(bundle.get('research_only') is True and bundle.get('service_promotion') is False,
             'G0 bundle must stay research-only and unpromoted')
    members, files = bundle.get('members'), bundle.get('files')
    _require(isinstance(members, dict) and list(members) == [str(s) for s in G0_SEEDS],
             'exactly five G0 members in seed order 0..4 required')
    _require(isinstance(files, dict), 'G0 file pins required')
    shas = []
    for seed, arm in zip(G0_SEEDS, G0_SOURCE_ARMS):
        member = members[str(seed)]
        _require(isinstance(member, dict) and set(member) == {'source_arm', 'model_sha256', 'delivery_temperature',
                 'june_model_weight'}, f'G0 member fields differ: {seed}')
        _require(member['source_arm'] == arm, f'G0 member source arm differs: {seed}')
        _require(_sha(member['model_sha256']), f'G0 member model sha256: {seed}')
        t = member['delivery_temperature']
        _require(type(t) is float and math.isfinite(t) and .5 <= t <= 2.5, f'G0 May temperature: {seed}')
        w = member['june_model_weight']
        _require(type(w) is float and math.isfinite(w) and 0 <= w <= 1, f'G0 per-seed June weight: {seed}')
        entry = files.get(f'seed{seed}_checkpoint')
        _require(isinstance(entry, dict) and entry.get('sha256') == member['model_sha256'],
                 f'G0 checkpoint pin differs from member model sha256: {seed}')
        shas.append(member['model_sha256'])
    _require(len(set(shas)) == 5, 'G0 member checkpoints must be distinct')
    for role in ('p11_frozen_calibration', 'p4_preparation'):
        _require(isinstance(files.get(role), dict) and _sha(files[role].get('sha256')), f'G0 pin missing: {role}')
    return members


def _pinned_json(paths, files, role):
    _require(role in paths, f'explicit path required for pinned role {role}')
    path = Path(paths[role])
    _require(hash_file(path) == files[role]['sha256'], f'pinned file changed: {role}')
    return json.loads(path.read_text())


class _CheckedMember:
    """Requires the G0 wrapper (``SharingPredictor`` strips routing columns before the global
    network) and that the loaded network is the pinned seed with 10 classes."""
    def __init__(self, seed, model):
        _require(isinstance(model, SharingPredictor) and model.cell == G0_CELL,
                 f'G0 member {seed} is not a {G0_CELL} SharingPredictor')
        net = model.global_model
        _require(getattr(net, 'seed', None) == seed and getattr(net, 'n_classes', None) == 10,
                 f'G0 member {seed}: loaded network seed/classes differ from the manifest slot')
        self.seed, self.model, self.cell = seed, model, G0_CELL

    def logits(self, arrays):
        z = np.asarray(self.model.logits(arrays), dtype=np.float64)
        _require(z.shape == (len(arrays[0]), 10) and np.isfinite(z).all(), f'G0 member {self.seed}: logits not finite [N,10]')
        return z


class _CheckedFrequency:
    """Reject malformed raw frequency rows BEFORE the frozen clip/log/temperature transform."""
    def __init__(self, baseline):
        self.baseline = baseline

    def predict(self, rows):
        raw = np.asarray(self.baseline.predict(rows), dtype=np.float64)
        _require(raw.shape == (len(rows), 10) and np.isfinite(raw).all() and ((raw >= 0) & (raw <= 1)).all()
                 and np.allclose(raw.sum(1), 1, rtol=0, atol=1e-6), 'frequency baseline rows invalid')
        return raw


class FrozenG0Ensemble(FrozenGEnsemble):
    """Five-member G0: per-seed May temperature, mean of five, common June ensemble blend.

    Same ``__call__`` as the three-member screen (``calibrated_conditional``); only the
    member count/identity differs. Per-seed June weights are NOT used for the ensemble.
    """
    def __init__(self, inputs, members, temperatures, baseline, baseline_temperature, ensemble_weight, identity):
        _require(len(members) == 5 and len(temperatures) == 5, 'Frozen G0 requires five members')
        _require(math.isfinite(baseline_temperature) and baseline_temperature > 0, 'baseline temperature')
        _require(math.isfinite(ensemble_weight) and 0 <= ensemble_weight <= 1, 'ensemble weight')
        self.inputs, self.temperatures = inputs, list(temperatures)
        self.models = [_CheckedMember(s, m) for s, m in zip(G0_SEEDS, members)]
        self.baseline, self.baseline_temperature = _CheckedFrequency(baseline), float(baseline_temperature)
        self.neural_weight, self.identity = float(ensemble_weight), identity
        self.frequency_cache, self.actual_neural_network_rows = {}, 0

    def __call__(self, states, actions, physical):
        p = super().__call__(states, actions, physical)
        _require(p.shape == (len(states), 10), 'G0 output shape')
        return p


def load_g0_ensemble(bundle, inputs, paths, *, load_member, load_frequency, frequency_role):
    """Factory for the frozen five-member G0 conditional predictor.

    ``paths`` maps manifest roles to explicit local paths (no defaults, no hard-coded runs).
    Every file is hashed against the bundle pin before use. ``load_member(seed, member, path)``
    must return a G0-global predictor with ``logits(arrays)``; a real binding should keep the
    checks of ``run_ml_g0_whole.load_member``. ``frequency_role`` names the pinned file holding
    the frequency predictor, which the bundle does not state (reported gap).
    """
    members = validate_g0_manifest(bundle)
    files = bundle['files']
    calibration = _pinned_json(paths, files, 'p11_frozen_calibration')
    _require(calibration.get('refit_on_whole_mlb') is False, 'G0 calibration must not be refitted')
    weight = calibration.get('june_ensemble_model_weight')
    _require(type(weight) is float and math.isfinite(weight) and 0 <= weight <= 1, 'June ensemble weight missing/invalid')
    _require(calibration.get('may_delivery_temperatures') == {k: m['delivery_temperature'] for k, m in members.items()},
             'May temperatures differ between calibration and manifest')
    _require(calibration.get('june_seed_model_weights') == {k: m['june_model_weight'] for k, m in members.items()},
             'per-seed June weights differ between calibration and manifest')
    preparation = _pinned_json(paths, files, 'p4_preparation')
    baseline_temperature = (preparation.get('baseline_temperature') or {}).get('temperature')
    _require(type(baseline_temperature) is float and math.isfinite(baseline_temperature) and baseline_temperature > 0,
             'frozen baseline temperature missing from pinned preparation')
    vocabulary = ((preparation.get('features') or {}).get('tokens') or {}).get('type_vocabulary')
    _require(isinstance(vocabulary, list) and tuple(vocabulary) == inputs.types,
             'PolicyInputs token vocabulary differs from the pinned G0 preparation')
    _require(frequency_role in files and frequency_role in paths, 'explicit pinned frequency role required')
    _require(hash_file(Path(paths[frequency_role])) == files[frequency_role]['sha256'], f'pinned file changed: {frequency_role}')
    loaded = []
    for seed in G0_SEEDS:
        role = f'seed{seed}_checkpoint'
        _require(role in paths, f'explicit path required for {role}')
        _require(hash_file(Path(paths[role])) == members[str(seed)]['model_sha256'], f'G0 checkpoint changed: {seed}')
        loaded.append(load_member(seed, members[str(seed)], Path(paths[role])))
    identity = {'protocol': G0_PROTOCOL, 'bundle_sha256': canonical_hash(bundle),
                'member_model_sha256': [members[str(s)]['model_sha256'] for s in G0_SEEDS],
                'may_temperatures': [members[str(s)]['delivery_temperature'] for s in G0_SEEDS],
                'june_ensemble_model_weight': weight, 'baseline_temperature': baseline_temperature,
                'calibration_sha256': files['p11_frozen_calibration']['sha256'],
                'preparation_sha256': files['p4_preparation']['sha256'],
                'frequency_role': frequency_role, 'frequency_sha256': files[frequency_role]['sha256']}
    identity['sha256'] = canonical_hash(identity)
    return FrozenG0Ensemble(inputs, loaded, [members[str(s)]['delivery_temperature'] for s in G0_SEEDS],
                            load_frequency(Path(paths[frequency_role])), baseline_temperature, weight, identity)
