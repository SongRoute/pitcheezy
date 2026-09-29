"""Complete policy identity and actual-connection checks (ML-POLICY-IDENTITY-v1, COOP-017).

Additive to ``policy_artifacts``/``policy_runtime``. ``bind_components`` builds every part of
the P3 candidate simulator itself from explicitly pinned files: the five G0 members through a
pinned member loader, the frequency object, context encoder and exact 400-draw TRAIN delivery
pools from the one pinned G0 auxiliary pickle, and the C0 WE terminal/cutoff. Each connected
object is checked against its pinned descriptor (registered class, the report archived in the
pinned preparation, member records) before use. The identity also pins the source file of every
repository class reachable from the connected objects and the numeric environment, and
``BoundComponents.verify`` re-checks code, in-memory content and simulator wiring later.
Nothing is read at import time and no real artifact is chosen here: callers pass paths and pins.
"""
from __future__ import annotations

import hashlib
import importlib
import inspect
import json
from pathlib import Path
import pickle
import platform
import sys

import numpy as np

from .data import hash_file
from .matrix_data import canonical_hash
from .matrix_policy import SAFE_COLUMNS, FrozenWE, PolicyInputs
from .matrix_sharing import SharingContext
from .policy_artifacts import (G0_SEEDS, IntegrityError, _require, _sha, bc_payload, load_g0_ensemble,
                               validate_g0_manifest, vocabulary_sha256)

CONTRACT = 'ML-POLICY-IDENTITY-v1'
REPO = Path(__file__).resolve().parents[3]
PROJECT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = (REPO / 'experiments', REPO / 'src', REPO / 'scripts')
AUX_KEYS = ('baseline', 'context', 'delivery', 'normalizer')
WE_KEYS = ('advancement', 'we')
DRAWS = 400
# Same lineage files as run_ml_policy.WE_SOURCE_FILES (scripts module, not importable here).
WE_SOURCE_FILES = ('pitchmdp/game.py', 'scripts/build_minimal_pitch_service.py', 'scripts/run_temporal_blend.py')
RUNTIME_MODULES = ('pitchmdp.policy_identity', 'pitchmdp.policy_runtime', 'pitchmdp.policy_artifacts',
                   'pitchmdp.rollout_policy', 'pitchmdp.matrix_policy', 'pitchmdp.matrix_sharing',
                   'pitchmdp.matrix_data', 'pitchmdp.data', 'pitchmdp.planner', 'pitchmdp.game')


def class_name(obj):
    cls = obj if isinstance(obj, type) else type(obj)
    return f'{cls.__module__}.{cls.__qualname__}'


def _pinned_bytes(path, sha256):
    _require(_sha(sha256), 'a lowercase sha256 pin is required')
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != sha256:
        raise IntegrityError(f'pinned file changed: {path}')
    return data


def pinned_json(path, sha256):
    return json.loads(_pinned_bytes(path, sha256))


def pinned_pickle(path, sha256):
    """Unpickle only bytes that match the pin: unpickling runs code, so the pin is the trust boundary."""
    return pickle.loads(_pinned_bytes(path, sha256))


def _plain(value):
    try:
        return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise IntegrityError(f'report is not plain JSON: {error}') from error


def _report(obj):
    report = getattr(obj, 'report', None)
    return _plain(report() if callable(report) else report)


def _repository_file(path):
    path = Path(path).resolve() if path else None
    return path if path and any(path.is_relative_to(root) for root in SOURCE_ROOTS) else None


def _repository_class(obj):
    """A registered class name alone could resolve to code outside the pinned source closure."""
    return _repository_file(getattr(sys.modules.get(type(obj).__module__), '__file__', None)) is not None


def row_sha256(row):
    """Hash of one bound safe-context row (numpy scalars as Python values, NaN kept)."""
    plain = {k: v.item() if isinstance(v, np.generic) else v for k, v in row.items()}
    return hashlib.sha256(json.dumps(plain, sort_keys=True, default=str).encode()).hexdigest()


def callable_identity(function):
    path = _repository_file(inspect.getsourcefile(function))
    _require(path is not None, 'loaders must be repository code with a pinnable source file')
    return {'qualname': f'{function.__module__}.{function.__qualname__}',
            'file': str(path.relative_to(REPO)), 'sha256': hash_file(path)}


def project_sources(objects, modules=RUNTIME_MODULES):
    """sha256 of the repository source closure of the connected objects: the module of every
    class reachable from ``objects`` (whole MRO) or function among them, the fixed runtime call
    path ``modules``, and every repository module those modules import at module level.

    ponytail: containers above 4096 entries are sampled (first 64 values); they hold data
    (pools, context rows), not code. A deeper audit would walk every value.
    """
    names, seen, stack = set(modules), set(), list(objects)
    while stack:
        obj = stack.pop()
        if obj is None or id(obj) in seen or isinstance(obj, (str, bytes, int, float, np.ndarray, np.generic)):
            continue
        seen.add(id(obj))
        if isinstance(obj, dict):
            obj = list(obj.values())
        if isinstance(obj, (list, tuple, set, frozenset)):
            stack.extend(list(obj)[:64] if len(obj) > 4096 else obj)
            continue
        if inspect.ismethod(obj):
            stack.extend((obj.__self__, obj.__func__))
            continue
        if inspect.isfunction(obj):
            names.add(obj.__module__)
            continue
        cls = obj if isinstance(obj, type) else type(obj)
        project = [k.__module__ for k in cls.__mro__ if _repository_file(getattr(sys.modules.get(k.__module__), '__file__', None))]
        names.update(project)
        if project and not isinstance(obj, type) and hasattr(obj, '__dict__'):
            stack.extend(vars(obj).values())
    files, queue, done = set(), list(names), set()
    while queue:  # module-level import closure within the repository
        name = queue.pop()
        if name in done:
            continue
        done.add(name)
        module = sys.modules.get(name) if name else None  # live objects' modules are already imported
        if module is None and name in modules:
            module = importlib.import_module(name)
        path = _repository_file(getattr(module, '__file__', None))
        if path is None:
            continue
        files.add(path)
        for value in vars(module).values():
            if inspect.ismodule(value):
                queue.append(value.__name__)
            elif inspect.isclass(value) or inspect.isfunction(value):
                queue.append(value.__module__)
    return {str(p.relative_to(REPO)): hash_file(p) for p in sorted(files)}


def environment():
    import pandas
    import scipy
    torch = sys.modules.get('torch')
    return {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
            'pandas': pandas.__version__, 'torch': getattr(torch, '__version__', None),
            'system': platform.system(), 'machine': platform.machine()}


class _Digest:
    def __init__(self): self.sha = hashlib.sha256()
    def write(self, data): self.sha.update(data)


def content_digest(value):
    """Streamed sha256 of the pickled in-memory state (no full copy); same-process comparisons only."""
    sink = _Digest()
    pickle.Pickler(sink, protocol=pickle.HIGHEST_PROTOCOL).dump(value)
    return sink.sha.hexdigest()


def _network_state(network):
    """Torch weights as CPU arrays (device-independent bytes); any other network pickles as is."""
    module = getattr(network, 'net', None)
    if hasattr(module, 'state_dict'):
        return [(name, tensor.detach().cpu().numpy()) for name, tensor in module.state_dict().items()]
    return network


# ---------------------------------------------------------------- C0 WE terminal / cutoff

def load_pinned_we(paths, contract_sha256, classes):
    """Explicit-path mirror of ``run_ml_policy.verify_we``: C0 contract, bundle lineage JSON,
    WE source files, then the game_values bytes, before anything is unpickled."""
    contract = pinned_json(paths['we_contract'], contract_sha256)
    _require((contract.get('contract_version'), contract.get('value_spec_version'), contract.get('training_cutoff'))
             == ('C0-v1', 'defense-we-pa-v1', '2025-04-30'), 'compatible frozen C0 WE contract required')
    bundle = contract['bundle_path']
    manifest_sha = contract['sha256'].get(f'{bundle}/bundle_manifest.json')
    source_sha = contract['sha256'].get(f'{bundle}/source_hashes.json')
    manifest = pinned_json(paths['we_bundle_manifest'], manifest_sha)
    source = pinned_json(paths['we_source_hashes'], source_sha)
    for name in WE_SOURCE_FILES:
        _require(hash_file(PROJECT / name) == source.get(name), f'frozen WE source lineage differs: {name}')
    expected = contract['bundle_files']['game_values.pkl']
    _require(manifest.get('sha256', {}).get('game_values.pkl') == expected, 'WE bundle manifest differs from the contract')
    values = pinned_pickle(paths['we_game_values'], expected)
    _require(isinstance(values, dict) and set(WE_KEYS) <= set(values), 'game_values must hold we and advancement')
    for key in WE_KEYS:
        _require(class_name(values[key]) == classes[key], f'WE {key} class differs from its registered class')
        _require(_repository_class(values[key]), f'WE {key} class is not repository code')
    return values, {'contract_sha256': contract_sha256, 'contract_version': 'C0-v1', 'value_spec_version': 'defense-we-pa-v1',
                    'training_cutoff': '2025-04-30', 'bundle_manifest_sha256': manifest_sha,
                    'source_hashes_sha256': source_sha, 'game_values_sha256': expected, 'classes': dict(classes),
                    'terminal': 'game.terminal_values: initial-defender WE after each terminal event',
                    'cutoff': 'WinExpectancy.predict_defense at the initial PA state'}


# ---------------------------------------------------------------- binding

def _is_bound(method, owner, function):
    return getattr(method, '__self__', None) is owner and getattr(method, '__func__', None) is function


class BoundComponents:
    """Built only by ``bind_components``. Holds the connected objects and their identity.

    ponytail: derived caches (pool/support/frequency caches) are rebuilt from the covered
    objects and are not digested; in-process tampering with a cache is out of scope.
    """
    def __init__(self, inputs, g0, we, identity, state, links, we_model):
        self.inputs, self.g0, self.we, self.identity = inputs, g0, we, identity
        self.sha256 = canonical_hash(identity)
        self._state, self._links, self._we_model = state, links, we_model
        self._digest = content_digest(state())

    def defense_we(self, state, defender_is_home):
        """Frozen C0 WE of the verified bundle for the initial defender (the estimator's PA-end reward)."""
        return float(self._we_model.predict_defense(state, defender_is_home))

    def context_sha256(self, state):
        """Hash of the bound context row behind ``state`` (None if unbound): part of the request fingerprint."""
        row = self.inputs.rows.get(state.context_key)
        return None if row is None else row_sha256(row)

    def check_context(self, state, pitcher_hand):
        """Candidate mode, before any data refusal: the request is exactly its bound context row
        (pitcher, batter side and throwing hand); a mismatch is an input defect, never a refusal."""
        row = self.inputs.rows.get(state.context_key)
        _require(row is not None, 'request context is not bound to the policy inputs')
        # PolicyInputs.support would swallow this mismatch as an empty (and cached) support.
        _require(str(int(row['pitcher'])) == state.pitcher and row['stand'] == state.batter_side,
                 'request pitcher/batter side differ from the bound context row')
        row_hand = row.get('p_throws')
        row_hand = None if row_hand is None or (isinstance(row_hand, float) and np.isnan(row_hand)) else str(row_hand)
        _require(row_hand == pitcher_hand, 'request pitcher hand differs from the bound context row')

    def check_support(self, state, mask):
        """Per-request link of the pinned support table to the bound delivery pools."""
        row = self.inputs.rows.get(state.context_key)
        _require(row is not None, 'request context is not bound to the policy inputs')
        _require(str(int(row['pitcher'])) == state.pitcher and row['stand'] == state.batter_side,
                 'request pitcher/batter side differ from the bound context row')
        _require(np.array_equal(self.inputs.support(state), mask),
                 'intervention support table differs from the bound delivery-pool support')

    def verify(self, improvement=None):
        """Re-check identity, pinned code, object links, in-memory content and (optionally) wiring."""
        _require(canonical_hash(self.identity) == self.sha256, 'policy identity changed after binding')
        current = {rel: hash_file(REPO / rel) for rel in self.identity['sources']}
        _require(current == self.identity['sources'], 'policy source changed after binding')
        _require(self._links(), 'bound components were re-linked after binding')
        _require(content_digest(self._state()) == self._digest, 'bound component content changed after binding')
        if improvement is not None:
            simulator = improvement.simulator
            _require(simulator.predictor is self.g0 and _is_bound(simulator.pool, self.inputs, PolicyInputs.pool)
                     and _is_bound(simulator.terminal, self.we, FrozenWE.terminal)
                     and _is_bound(improvement.cutoff, self.we, FrozenWE.cutoff),
                     'candidate simulator is not wired to the bound pool/predictor/terminal/cutoff')
        return True


def bind_components(bundle_path, bundle_sha256, paths, *, bc_artifact, context_rows, member_loader, classes,
                    we_contract_sha256):
    """Connect and verify every candidate-simulator component; return ``BoundComponents``.

    The G0 bundle (``configs/G0-RESEARCH-FROZEN-v1.json``) is itself read only through its file
    pin. ``paths`` maps bundle roles and ``we_contract``/``we_bundle_manifest``/``we_source_hashes``/
    ``we_game_values`` to explicit local files. ``member_loader(config, record, clusters)`` is the
    real G0 loader (``run_ml_g0_whole.load_member``), pinned by its source file. ``classes`` is the
    registered descriptor ``{'aux': {key: class}, 'we': {key: class}}``. ``context_rows`` are the
    evaluation contexts (request inputs, not identity; their values enter each request fingerprint).
    Any failure while binding, including the loader's own checks, is FAILED_INTEGRITY.
    """
    return _integrity(lambda: _bind(pinned_json(bundle_path, bundle_sha256), bundle_sha256, paths, bc_artifact,
                                    context_rows, member_loader, classes, we_contract_sha256))


def bind_policy_inputs(bundle_path, bundle_sha256, paths, *, bc_artifact, context_rows, aux_classes):
    """Only the pinned auxiliary side (encoder, delivery pools, vocabulary): what the TRAIN support
    table needs. Same checks as ``bind_components``; members and WE are not loaded."""
    def bind():
        bundle = pinned_json(bundle_path, bundle_sha256)
        validate_g0_manifest(bundle)
        side = _bind_inputs(bundle, paths, bc_artifact, context_rows, aux_classes)
        return side['inputs'], {'g0_bundle_file_sha256': bundle_sha256, **side['identity']}
    return _integrity(bind)


def _integrity(bind):
    try:
        return bind()
    except IntegrityError:
        raise
    except (KeyError, TypeError, AttributeError, ValueError, IndexError, ImportError, EOFError,
            pickle.UnpicklingError) as error:
        raise IntegrityError(f'policy component binding failed: {type(error).__name__}: {error}') from error


def _bind_inputs(bundle, paths, bc_artifact, context_rows, aux_classes):
    files = bundle['files']
    for role in ('p4_auxiliary', 'p4_preparation'):
        _require(isinstance(files.get(role), dict) and _sha(files[role].get('sha256')), f'G0 pin missing: {role}')
        _require(role in paths, f'explicit path required for pinned role {role}')
    _require(isinstance(aux_classes, dict) and set(aux_classes) == set(AUX_KEYS), 'registered component classes are incomplete')
    aux_sha = files['p4_auxiliary']['sha256']

    # Preprocessing, delivery pools and frequency: one pinned auxiliary pickle checked against
    # the reports archived in the pinned G0 preparation when it was built.
    aux = pinned_pickle(paths['p4_auxiliary'], aux_sha)
    _require(isinstance(aux, dict) and set(aux) == set(AUX_KEYS), 'G0 auxiliary pickle keys differ')
    prep = pinned_json(paths['p4_preparation'], files['p4_preparation']['sha256'])
    features = prep['features']
    scope = features['auxiliary_scope']
    for key in AUX_KEYS:
        _require(class_name(aux[key]) == aux_classes[key], f'auxiliary {key} class differs from its registered class')
        _require(_repository_class(aux[key]), f'auxiliary {key} class is not repository code')
        _require(_report(aux[key]) == scope[key], f'auxiliary {key} differs from the report pinned at preparation')
    delivery = aux['delivery']
    _require(delivery.draws == DRAWS and _plain(delivery.TIERS) == scope['delivery']['tiers']
             and len(delivery.pools) == scope['delivery']['pool_count'], 'delivery pools differ from their pinned descriptor')
    for values in delivery.pools.values():
        _require(isinstance(values, np.ndarray) and values.shape == (DRAWS, 8) and np.isfinite(values).all(),
                 'delivery pool arrays must be finite [400,8]')
        values.setflags(write=False)
    clusters, tokens = prep['clusters'], features['tokens']
    _require(canonical_hash(clusters) == features['context']['pitcher_representation_sha256']
             and features['context']['base'] == scope['context'], 'context encoder differs from the pinned representation')
    _require(tokens['history_length'] == 5 and tokens['token_channels'] == 8 + len(tokens['type_vocabulary']) + 1 + 11,
             'token layout differs from PolicyInputs')
    encoder = SharingContext(aux['context'], clusters)
    _require(canonical_hash(bc_payload(bc_artifact.bc, bc_artifact.provenance)) == bc_artifact.sha256,
             'BC state differs from its pinned identity')
    inputs = PolicyInputs(context_rows, encoder, delivery, tokens['type_vocabulary'], aux_sha, bc_artifact.bc)
    identity = {
        'bc': {'bc_sha256': bc_artifact.sha256, 'vocabulary_sha256': bc_artifact.vocabulary_sha256},
        'preprocessing': {'inputs': class_name(inputs), 'safe_columns': list(SAFE_COLUMNS), 'encoder': class_name(encoder),
                          'encoder_base': aux_classes['context'], 'context_report_sha256': canonical_hash(scope['context']),
                          'clusters_sha256': canonical_hash(clusters), 'tokens_report_sha256': canonical_hash(tokens),
                          'type_vocabulary_sha256': vocabulary_sha256(tokens['type_vocabulary']),
                          'normalizer_report_sha256': canonical_hash(scope['normalizer']),
                          'auxiliary_sha256': aux_sha, 'preparation_sha256': files['p4_preparation']['sha256']},
        'delivery_pool': {'class': aux_classes['delivery'], 'draws': DRAWS, 'tiers': scope['delivery']['tiers'],
                          'pool_count': scope['delivery']['pool_count'], 'report_sha256': canonical_hash(scope['delivery']),
                          'source_hash': aux_sha,
                          'policy_rule': 'an action without an action-specific pool is unsupported; the league fallback pool is never used'}}
    return {'inputs': inputs, 'aux': aux, 'prep': prep, 'scope': scope, 'encoder': encoder, 'clusters': clusters,
            'delivery': delivery, 'identity': identity}


def _bind(bundle, bundle_sha256, paths, bc_artifact, context_rows, member_loader, classes, we_contract_sha256):
    members = validate_g0_manifest(bundle)
    files = bundle['files']
    for role in ('p11_preparation', 'p11_registered_config', 'p11_frozen_calibration'):
        _require(isinstance(files.get(role), dict) and _sha(files[role].get('sha256')), f'G0 pin missing: {role}')
        _require(role in paths, f'explicit path required for pinned role {role}')
    _require(isinstance(classes, dict) and set(classes) == {'aux', 'we'} and set(classes['we']) == set(WE_KEYS),
             'registered component classes are incomplete')
    side = _bind_inputs(bundle, paths, bc_artifact, context_rows, classes['aux'])
    inputs, aux, prep, scope, encoder, clusters, delivery = (side[k] for k in (
        'inputs', 'aux', 'prep', 'scope', 'encoder', 'clusters', 'delivery'))

    # Members: pinned records/config, the pinned loader, and the loaded network checked again.
    p11 = pinned_json(paths['p11_preparation'], files['p11_preparation']['sha256'])
    config = pinned_json(paths['p11_registered_config'], files['p11_registered_config']['sha256'])
    records = p11['members']
    _require(set(records) == {str(s) for s in G0_SEEDS} and canonical_hash(p11['clusters']) == canonical_hash(clusters),
             'G0 member records/clusters differ from the bundle')
    for seed in G0_SEEDS:
        record, member = records[str(seed)], members[str(seed)]
        _require((record['seed'], record['source_arm'], record['model_sha256'], record['delivery_temperature'])
                 == (seed, member['source_arm'], member['model_sha256'], member['delivery_temperature']),
                 f'G0 member record differs from the bundle: {seed}')
    loader = callable_identity(member_loader)

    def load_member(seed, member, path):
        record = {**records[str(seed)], 'model_path': str(path)}
        loaded = member_loader(config, record, p11['clusters'])
        # The loader reopens the checkpoint after the factory's hash: re-hash to close that window.
        _require(hash_file(Path(path)) == record['model_sha256'], f'G0 checkpoint changed during load: {seed}')
        predictor = loaded[0] if isinstance(loaded, tuple) else loaded
        network = getattr(predictor, 'global_model', None)
        report = getattr(network, 'report', None) or {}
        _require(report.get('network') == record['network'] and report.get('parameter_count') == record['parameter_count']
                 and getattr(network, 'device', None) == record['device']
                 and getattr(predictor, 'delivery_temperature', None) == record['delivery_temperature'],
                 f'G0 member {seed}: loaded network differs from its pinned record')
        return predictor

    def load_frequency(path):
        _require(Path(path).resolve() == Path(paths['p4_auxiliary']).resolve(),
                 'frequency object must come from the pinned auxiliary pickle')
        return aux['baseline']

    g0 = load_g0_ensemble(bundle, inputs, paths, load_member=load_member, load_frequency=load_frequency,
                          frequency_role='p4_auxiliary')
    # The factory re-reads its JSON after hashing; compare with the bytes verified here.
    calibration = pinned_json(paths['p11_frozen_calibration'], files['p11_frozen_calibration']['sha256'])
    _require(g0.neural_weight == calibration['june_ensemble_model_weight']
             and g0.temperatures == [calibration['may_delivery_temperatures'][str(s)] for s in G0_SEEDS]
             and g0.baseline_temperature == prep['baseline_temperature']['temperature'],
             'G0 factory calibration differs from the verified calibration/preparation bytes')
    we_values, we_identity = load_pinned_we(paths, we_contract_sha256, classes['we'])
    we = FrozenWE(inputs, we_values)

    identity = {
        'contract': CONTRACT,
        'g0_bundle_file_sha256': bundle_sha256,
        'bc': side['identity']['bc'],
        'predictor': g0.identity,
        'members': {'loader': loader, 'records_sha256': files['p11_preparation']['sha256'],
                    'config_sha256': files['p11_registered_config']['sha256'],
                    'config': {k: config.get(k) for k in ('kind', 'width', 'context_width', 'individual_tau', 'cluster_tau', 'device')},
                    'slots': [{'seed': s, 'network_sha256': canonical_hash(records[str(s)]['network']),
                               'parameter_count': records[str(s)]['parameter_count'], 'device': records[str(s)]['device']}
                              for s in G0_SEEDS]},
        'frequency': {'role': 'p4_auxiliary', 'key': 'baseline', 'class': classes['aux']['baseline'],
                      'report_sha256': canonical_hash(scope['baseline']), 'temperature': g0.baseline_temperature,
                      'transform': 'softmax(log(clip(raw, 1e-12, 1)) / T)'},
        'preprocessing': side['identity']['preprocessing'],
        'delivery_pool': side['identity']['delivery_pool'],
        'we': we_identity,
        'sources': project_sources([aux, we_values, g0, we, encoder, inputs, member_loader]),
        'environment': environment(),
    }

    def state():  # everything inference reads, including the context rows/encodings and class-level TIERS
        return (aux, we_values, clusters, _plain(delivery.TIERS), inputs.types, inputs.type_map, inputs.rows,
                inputs.contexts, canonical_hash(bc_payload(inputs.bc, bc_artifact.provenance)),
                [(m.seed, m.model.cell, m.model.delivery_temperature, _network_state(m.model.global_model)) for m in g0.models],
                (g0.temperatures, g0.neural_weight, g0.baseline_temperature), we.terminal_tables, we.initial)

    def links():  # the objects inference goes through are the verified ones
        return (g0.inputs is inputs and g0.baseline.baseline is aux['baseline'] and inputs.delivery is delivery
                and aux['delivery'] is delivery and inputs.context_encoder is encoder and encoder.base is aux['context'])
    return BoundComponents(inputs, g0, we, identity, state, links, we_values['we'])


# ---------------------------------------------------------------- connection probe

def integrated_predictions(g0, states, actions):
    """Policy-path G0 prediction integrated over the action's exact 400-draw TRAIN pool.

    ``calibrated_conditional`` is linear in the per-draw probabilities, so this equals the
    evaluation-path primary prediction (``predict_streamed`` members, frozen frequency, June
    blend) for a row whose evaluation used the same action-specific pool. A row that needed
    the league fallback pool raises ``ValueError`` here: it is not comparable.
    """
    rows = []
    for state, action in zip(states, actions):
        draws = g0.inputs.pool(state, action, count_access=False).values
        rows.append(g0([state] * len(draws), [action] * len(draws), draws).mean(axis=0))
    return np.stack(rows)


def compare_probe(actual, expected, atol):
    actual, expected = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    _require(actual.shape == expected.shape and actual.ndim == 2 and actual.shape[1] == 10, 'probe shapes differ')
    difference = float(np.abs(actual - expected).max()) if actual.size else None
    return {'rows': len(actual), 'max_abs_difference': difference, 'atol': atol,
            'pass': difference is not None and difference <= atol}


# ---------------------------------------------------------------- TRAIN intervention support table

def hand_registry(train_rows, pitchers):
    """D-8 (HAND_RULE): per BC pitcher, the single TRAIN throwing hand when every TRAIN row of the
    pitcher has the same L/R and none is missing; otherwise 'AMBIGUOUS'. No share threshold."""
    hands = {}
    groups = {str(int(p)): part for p, part in train_rows.groupby('pitcher', sort=True)}
    for pitcher in pitchers:
        part = groups.get(str(pitcher))
        _require(part is not None, f'BC pitcher without TRAIN rows: {pitcher}')
        values = part.p_throws
        observed = set(values.dropna().astype(str))
        single = not values.isna().any() and len(observed) == 1 and observed <= {'L', 'R'}
        hands[str(pitcher)] = next(iter(observed)) if single else 'AMBIGUOUS'
    return hands


def support_templates(train_rows, hands):
    """One safe-context template per single-hand registry pitcher x batter side {L, R} at 0-0.

    Support depends only on the pitcher, hand, side and the pools, so any real TRAIN row of the
    pitcher serves as the rest of the context. AMBIGUOUS pitchers get no template (they are
    refused before the support lookup). Template keys are synthetic and negative so they never
    collide with a real pitch key.
    """
    import pandas as pd
    single = {p: h for p, h in hands.items() if h != 'AMBIGUOUS'}
    base = train_rows.assign(_pitcher=train_rows.pitcher.map(lambda v: str(int(v))))
    base = base.loc[base._pitcher.isin(single)].drop_duplicates('_pitcher').sort_values('_pitcher')
    rows = []
    for index, row in enumerate(base.to_dict('records')):
        _require(row['p_throws'] == single[row['_pitcher']], 'template hand differs from the registry')
        row.pop('_pitcher')
        for side_index, side in enumerate(('L', 'R')):
            rows.append({**row, 'game_pk': -1, 'at_bat_number': -(index + 1), 'pitch_number': side_index + 1,
                         'stand': side, 'balls': 0, 'strikes': 0})
    return pd.DataFrame(rows, columns=train_rows.columns)


def support_rows(inputs, templates):
    """(pitcher, side, mask) rows for ``save_support_table``. Templates carry one hand per pitcher,
    so one (pitcher, side) key can never see two masks; that is asserted, not repaired."""
    from .matrix_policy import context_key
    from .rollout_policy import PAState
    masks = {}
    for row in templates.to_dict('records'):
        pitcher, side = str(int(row['pitcher'])), str(row['stand'])
        _require((pitcher, side) not in masks, f'two templates for one support key: {(pitcher, side)}')
        masks[(pitcher, side)] = inputs.support(PAState(0, 0, pitcher, side, (), context_key(row)))
    return [(p, s, mask) for (p, s), mask in sorted(masks.items())]
