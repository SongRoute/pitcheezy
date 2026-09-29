"""<=2025 policy validation runner (ML-POLICY-VAL-v1, COOP-018).

Stages: census (S0), materialize-bc (S1), bind-probe (S2), profile (S3), v5-denominators (S4),
dr-evaluate (V4). The CLI refuses unless configs/ML-POLICY-MATERIALIZATION-v1.json is registered
and execution-enabled; every stage holds the shared heavy lock, writes into a fresh directory
under the artifact root, records its start/identity, seals a manifest on success and preserves a
failure record otherwise (no silent retry). No row dated 2026 or later can enter. Stage logic is
in pure functions over already-loaded inputs so it can be checked without real data.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'scripts'))

import numpy as np
import pandas as pd

from pitchmdp.data import KEY, hash_file
from pitchmdp.matrix_data import canonical_hash, ordered_key_hash
from pitchmdp.matrix_policy import fit_bc, safe_rows
from pitchmdp.policy_artifacts import (IntegrityError, _require, export_train_bc, load_support_table, load_train_bc,
                                       save_support_table)
from pitchmdp import policy_estimator as est
from pitchmdp import policy_identity as pid
from pitchmdp import policy_requests as preq
from pitchmdp import policy_runtime as prt
from pitchmdp.rollout_policy import BudgetExceeded, PAState, PastPitch, RowBudget

PROTOCOL = 'ml_policy_materialization_v1'
COMMANDS = ('census', 'materialize-bc', 'bind-probe', 'profile', 'v5-denominators', 'dr-evaluate')
LAST_DATE = pd.Timestamp('2025-12-31')
DECISION_VALUES = {
    'D-1': ('BC_P', 'BC_E'), 'D-2': ('all_regular_season_pas',), 'D-3': ('refuse_no_logged_action_sticky',),
    'D-4': ('continue_known_pitcher',), 'D-5': ('conditional_plus_worst_case_bounds',),
    'D-6': ('observed_next_row_state',), 'D-7': ('window_start_snapshot', 'rolling_prior'),
    'D-8': ('refuse', 'intersect')}


# ---------------------------------------------------------------- registration and stage records

def registration(config):
    """Refuse unless the registration is final and real-data execution is enabled."""
    _require(config.get('protocol') == PROTOCOL, 'not an ML-POLICY materialization/validation config')
    _require(config.get('registered') is True and config.get('status') == 'REGISTERED'
             and (config.get('execution') or {}).get('enabled') is True
             and (config.get('execution') or {}).get('real_data_enabled') is True,
             'ML-POLICY-VAL-v1 is not registered and execution-enabled; refusing to run')
    decisions = config.get('decisions') or {}
    for key, allowed in DECISION_VALUES.items():
        _require(decisions.get(key) in allowed, f'registered decision required: {key} in {allowed}')
    return decisions


def environment():
    return {'python': sys.executable, 'python_version': platform.python_version(), **pid.environment()}


def git_state():
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True).strip()
    return {'commit': head, 'dirty': bool(dirty)}


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=1, ensure_ascii=False, allow_nan=False, default=str) + '\n')


@contextmanager
def stage(output, command, identity):
    """Fresh stage directory: started.json, then manifest.json on success or failure-*.json (kept)."""
    output = Path(output)
    _require(not output.exists(), f'stage output exists; preserve it and register a new attempt: {output}')
    output.mkdir(parents=True)
    started = time.perf_counter()
    dump(output / 'started.json', {'command': command, 'started_utc': datetime.now(timezone.utc).isoformat(), **identity})
    try:
        yield output
    except BaseException as error:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        dump(output / f'failure-{stamp}.json', {'command': command, 'error_type': type(error).__name__,
             'error': str(error), 'seconds': time.perf_counter() - started, 'preserve_partial_artifacts': True})
        raise
    names = sorted(str(p.relative_to(output)) for p in output.rglob('*') if p.is_file())
    dump(output / 'manifest.json', {'command': command, 'seconds': time.perf_counter() - started,
         'artifact_sha256': {name: hash_file(output / name) for name in names}})


class Deadline:
    """Wall-clock cap for a stage loop; exceeding it is FAILED_RUNTIME, never a silent truncation."""
    def __init__(self, seconds):
        _require(isinstance(seconds, (int, float)) and seconds > 0, 'registered positive stage cap required')
        self.seconds, self.start = float(seconds), time.monotonic()

    def check(self):
        if time.monotonic() - self.start > self.seconds:
            raise BudgetExceeded(f'stage wall cap {self.seconds}s exceeded; preserve partial stage')

    def elapsed(self):
        return time.monotonic() - self.start


def guard_dates(frame):
    dates = pd.to_datetime(frame.game_date)
    _require(len(frame) and dates.max() <= LAST_DATE, 'rows dated after 2025 cannot enter <=2025 validation')
    return frame


# ---------------------------------------------------------------- S0 census

def run_census(frame, vocabulary):
    return {'census': preq.census(guard_dates(frame), vocabulary), 'label_blind': True}


# ---------------------------------------------------------------- S1 TRAIN BC and support table

def bc_rows(frame, rule, train_keys=None, keys_record=None):
    """TRAIN rows for the BC: BC_P (no post-decision condition) or BC_E (pinned eligible keys)."""
    train = frame.loc[frame.split.eq('train')]
    if rule == 'BC_P':
        legal = train.balls.between(0, 3) & train.strikes.between(0, 2)
        return train.loc[train.pitch_type.notna() & ~train.description.isin(preq.AUTOMATIC) & legal]
    _require(rule == 'BC_E' and train_keys is not None and keys_record is not None, 'BC_E needs pinned TRAIN keys')
    positions = pd.MultiIndex.from_frame(frame[KEY]).get_indexer(pd.MultiIndex.from_frame(train_keys[KEY]))
    _require(len(train_keys) == keys_record['n'] and (positions >= 0).all(), 'pinned BC_E keys absent from the frame')
    selected = frame.iloc[positions]
    _require(ordered_key_hash(selected) == keys_record['rows_sha256'], 'BC_E ordered key identity changed')
    return selected.sort_index()


def logging_mass_report(bc, table):
    """TRAIN-cell distribution of pi_b_hat(M(H)|H): D89's rho_ref == 1 condition, checked not assumed."""
    masses, weights = [], []
    for (pitcher, balls, strikes, side, previous), counter in bc.cells.items():
        history = () if previous == '<START>' else (PastPitch(previous, (0.,) * 8, 'unknown', 0, 0),)
        state = PAState(balls, strikes, pitcher, side, history)
        mask = table.get((pitcher, side))
        p = bc.probabilities(state)
        masses.append(0. if mask is None else float(p[mask & bc.support(state)].sum()))
        weights.append(sum(counter.values()))
    masses, weights = np.asarray(masses), np.asarray(weights, dtype=float)
    order = np.argsort(masses)
    cumulative = np.cumsum(weights[order]) / weights.sum()
    quantile = {str(q): float(masses[order][np.searchsorted(cumulative, q)]) for q in (.01, .05, .25, .5, .75, .95)}
    return {'cells': int(len(masses)), 'decision_weighted_quantiles': quantile,
            'decision_share_mass_equal_one': float(weights[masses >= 1 - 1e-12].sum() / weights.sum()),
            'cell_share_mass_equal_one': float((masses >= 1 - 1e-12).mean()),
            'decision_share_mass_zero': float(weights[masses == 0].sum() / weights.sum())}


def run_materialize(frame, store, output, *, rule, provenance, bind_inputs, two_hand_rule, train_keys=None,
                    keys_record=None, compare_rule=None):
    """Export the registered TRAIN BC(s), the intervention support table and the TRAIN mass report.

    ``bind_inputs(bc_artifact, context_rows)`` is ``policy_identity.bind_policy_inputs`` with pins.
    """
    guard_dates(frame)
    artifacts = {}
    for name in [rule] + ([compare_rule] if compare_rule else []):
        rows = bc_rows(frame, name, train_keys, keys_record)
        artifacts[name] = export_train_bc(store, rows, output / f'bc_{name}.json', **provenance)
    primary = artifacts[rule]
    train = frame.loc[frame.split.eq('train')]
    templates = pid.support_templates(safe_rows(train).assign(pitch_type=train.pitch_type.to_numpy()))
    inputs, inputs_identity = bind_inputs(primary, templates[list(safe_rows(train).columns)])
    rows, conflicts = pid.support_rows(inputs, templates, two_hand_rule=two_hand_rule)
    known = {(str(int(p)), s) for p in primary.bc.pitchers for s in ('L', 'R')}
    rows = [row for row in rows if (row[0], row[1]) in known]
    _, support_file_sha = save_support_table(primary, rows, output / 'support_primary.json')
    table, support_sha = load_support_table(output / 'support_primary.json', support_file_sha, primary)
    report = {'bc': {name: {**art.identity(), 'train_rows': art.provenance['train_rows'],
                            'vocabulary': list(art.vocabulary)} for name, art in artifacts.items()},
              'support_table': {'file_sha256': support_file_sha, 'content_sha256': support_sha, 'rows': len(rows),
                                'two_hand_rule': two_hand_rule, 'hand_conflict_pitchers': conflicts},
              'inputs_identity': inputs_identity,
              'logging_mass_on_mask': logging_mass_report(primary.bc, table)}
    dump(output / 'materialize.json', report)
    return report


# ---------------------------------------------------------------- S2 bind + connection probe

def probe_positions(sealed_keys, levels, frame, count):
    """First ``count`` rows in sealed order whose evaluation used an action-specific pool (level >= 0)."""
    levels = np.asarray(levels)
    _require(all(np.array_equal(levels[0], other) for other in levels[1:]), 'member delivery levels disagree')
    chosen = np.flatnonzero(levels[0] >= 0)[:count]
    _require(len(chosen) == count, 'not enough non-fallback probe rows')
    keys = pd.DataFrame(np.asarray(sealed_keys)[chosen], columns=KEY)
    positions = pd.MultiIndex.from_frame(frame[KEY]).get_indexer(pd.MultiIndex.from_frame(keys))
    _require((positions >= 0).all(), 'probe keys absent from the frame')
    return chosen, positions


def blocks_containing(frame, positions):
    """The PA blocks that contain each position (by the frame's actual PA boundaries)."""
    wanted = set(int(p) for p in positions)
    return [(pa_id, block) for pa_id, block in preq.pa_blocks(frame) if wanted.intersection(block.tolist())]


def run_bind_probe(store, components, sealed, *, count, atol_primary, atol_frequency):
    """Policy path vs sealed evaluation path on registered probe rows (no labels read)."""
    chosen, positions = probe_positions(sealed['keys'], sealed['levels'], store.frame, count)
    requests = {}
    for _, block in blocks_containing(store.frame, positions):
        built, problem = preq.pa_requests(store, block, 'probe')
        _require(problem is None, f'probe PA unsubmittable: {problem}')
        requests.update(zip(block.tolist(), built))
    states = [requests[int(p)].state for p in positions]
    actions = [requests[int(p)].logged_action for p in positions]
    primary = pid.integrated_predictions(components.g0, states, actions)
    frequency_rows = pd.DataFrame([components.inputs.query(s, a) for s, a in zip(states, actions)])
    raw = components.g0.baseline.baseline.predict(frequency_rows)
    report = {'rows': count, 'sealed_indices': chosen.tolist(),
              'primary': pid.compare_probe(primary, sealed['primary'][chosen], atol_primary),
              'frequency_raw': pid.compare_probe(raw, sealed['frequency_raw'][chosen], atol_frequency),
              'verify': components.verify()}
    report['pass'] = report['primary']['pass'] and report['frequency_raw']['pass']
    return report


# ---------------------------------------------------------------- S3/S4/V4 runtime passes

def submit_pas(runtime, store, blocks, deadline, *, outcomes=None, defense_we=None):
    """Submit every request of every PA (refusals stay in the ledger); record unsubmittable PAs."""
    unsubmittable = {}
    for pa_id, positions in blocks:
        deadline.check()
        requests, problem = preq.pa_requests(store, positions, runtime.sha256)
        if problem is not None:
            unsubmittable[pa_id] = problem
            continue
        for request in requests:
            runtime.submit(request)
        if outcomes is not None:
            outcomes[pa_id] = preq.pa_outcome(store.frame, positions, defense_we)
    return unsubmittable


def run_v5(runtime, store, split, deadline):
    """Logging-law/reference denominators on one split: statuses, codes and pi_b_hat(M|H)."""
    blocks = preq.pa_blocks(store.frame, np.flatnonzero(store.frame.split.eq(split).to_numpy()))
    unsubmittable = submit_pas(runtime, store, blocks, deadline)
    rows = runtime.ledger.decisions()
    masses = np.array([r['result']['logging_mass_on_mask'] for r in rows if r['status'] in prt.EVALUATED])
    outside = sum(r['status'] == prt.OUTSIDE_POLICY_SUPPORT for r in rows)
    return {'split': split, 'pas_in_split': len(blocks), 'unsubmittable_pas': len(unsubmittable),
            'unsubmittable_reasons': dict(pd.Series(list(unsubmittable.values()), dtype=object).value_counts()),
            'summary': runtime.summary(),
            'logged_outside_mask_share_of_evaluated': float(outside / len(masses)) if len(masses) else None,
            'logging_mass_on_mask_quantiles': ({str(q): float(np.quantile(masses, q)) for q in (.01, .05, .25, .5)}
                                              if len(masses) else None),
            'seconds': deadline.elapsed(), 'population_value': None}


def run_dr(runtime, store, components, split, deadline, *, draws, seed):
    """Candidate runtime over one split, PA rewards from the verified WE, then the DR estimator."""
    blocks = preq.pa_blocks(store.frame, np.flatnonzero(store.frame.split.eq(split).to_numpy()))
    outcomes = {}
    unsubmittable = submit_pas(runtime, store, blocks, deadline, outcomes=outcomes, defense_we=components.defense_we)
    submitted = {r['pa_id'] for r in runtime.ledger.decisions()}
    result, rows = est.estimate(runtime.ledger.decisions(), {k: v for k, v in outcomes.items() if k in submitted},
                                draws=draws, seed=seed)
    runtime.verify_components()
    result.update(split=split, unsubmittable_pas=len(unsubmittable), pas_in_split=len(blocks),
                  reward_flags=int(sum(bool(o['flags']) for o in outcomes.values())),
                  ledger=runtime.summary(), seconds=deadline.elapsed(),
                  interpretation='exposed_development only; not independent confirmation, not a causal effect')
    return result, rows


# ---------------------------------------------------------------- CLI (real data; gated)

def pinned_npz(path, sha256, names):
    """Hash first, then read only the named arrays (label arrays in the same file stay unread)."""
    _require(hash_file(Path(path)) == sha256, f'pinned archive changed: {path}')
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in names}


def pinned_parquet(path, sha256):
    _require(hash_file(Path(path)) == sha256, f'pinned table changed: {path}')
    return pd.read_parquet(path)


def load_inputs(config, local):
    """Pinned G0 bundle, the verified processed regular-season frame (same loader as G0) and its store."""
    from pitchmdp.matrix_features import MatrixHistoryStore
    from run_ml_benchmark import regular_frame
    reg = config['identity_registration']
    bundle_path = REPO / reg['g0_bundle']['path']
    bundle_sha = reg['g0_bundle']['file_sha256']
    bundle = pid.pinned_json(bundle_path, bundle_sha)
    files = bundle['files']
    paths = {role: Path(entry['path']) for role, entry in files.items()}
    paths.update({role: (REPO / value if not Path(value).is_absolute() else Path(value))
                  for role, value in reg['we_paths'].items()})
    parent = pid.pinned_json(paths['p4_parent_preparation'], files['p4_parent_preparation']['sha256'])
    frame = guard_dates(regular_frame(local, parent))
    prep = pid.pinned_json(paths['p4_preparation'], files['p4_preparation']['sha256'])
    aux = pid.pinned_pickle(paths['p4_auxiliary'], files['p4_auxiliary']['sha256'])
    store = MatrixHistoryStore.from_frame(frame, normalizer=aux['normalizer'], history_length=5,
                                         type_vocabulary=prep['features']['tokens']['type_vocabulary'])
    from run_ml_g0_whole import load_member  # the pinned G0 loader (policy identity records its source)
    return {'bundle_path': bundle_path, 'bundle_sha': bundle_sha, 'files': files, 'paths': paths, 'parent': parent,
            'prep': prep, 'frame': frame, 'store': store, 'member_loader': load_member}


def pa_contexts(store, blocks, snapshot=None, as_of=None):
    rows = store.frame.iloc[np.concatenate([positions for _, positions in blocks])]
    contexts = safe_rows(rows)
    return contexts if snapshot is None else preq.apply_style_snapshot(contexts, snapshot, as_of)


def sealed_probe_arrays(inputs):
    files, paths = inputs['files'], inputs['paths']
    predictions = pinned_npz(paths['p11_predictions'], files['p11_predictions']['sha256'], ('keys', 'primary'))
    frequency = pinned_npz(paths['p11_frequency'], files['p11_frequency']['sha256'], ('mlb_dev_keys', 'mlb_dev_raw'))
    manifest = pid.pinned_json(paths['p11_manifest'], files['p11_manifest']['sha256'])
    levels = []
    for seed in range(5):
        name = str(paths['p11_preparation'].parent / 'members' / 'G0-global' / f'seed{seed}' / 'predictions.npz')
        member = pinned_npz(name, manifest['inputs'][name], ('dev_keys', 'dev_delivery_level'))
        _require(np.array_equal(member['dev_keys'], predictions['keys']), 'member archive keys differ')
        levels.append(member['dev_delivery_level'])
    _require(np.array_equal(frequency['mlb_dev_keys'], predictions['keys']), 'frequency archive keys differ')
    return {'keys': predictions['keys'], 'primary': predictions['primary'], 'frequency_raw': frequency['mlb_dev_raw'],
            'levels': levels}


def dispatch(command, config, config_path, local, output, inputs):
    decisions = registration(config)
    from run_ml_matrix import check_location, heavy_lock
    root = check_location(local, output)
    reg, plan = config['identity_registration'], config['le2025_validation_plan']
    store, frame = inputs['store'], inputs['frame']
    vocabulary = inputs['prep']['features']['tokens']['type_vocabulary']
    identity = {'config_sha256': hash_file(config_path), 'git': git_state(), 'environment': environment(),
                'decisions': decisions, 'g0_bundle_file_sha256': inputs['bundle_sha']}
    caps = plan['stages']

    def bind(blocks, bc_artifact, snapshot=None, as_of=None):
        return pid.bind_components(inputs['bundle_path'], inputs['bundle_sha'], inputs['paths'], bc_artifact=bc_artifact,
                                   context_rows=pa_contexts(store, blocks, snapshot, as_of),
                                   member_loader=inputs['member_loader'],
                                   classes=reg['classes'], we_contract_sha256=reg['we_contract_sha256'])

    def registered(name):
        entry = (config.get('registered_inputs') or {}).get(name)
        _require(isinstance(entry, dict) and entry.get('file_sha256'), f'registered input required: {name}')
        return Path(entry['path']), entry['file_sha256']

    with heavy_lock(root), stage(output, command, identity) as out:
        if command == 'census':
            dump(out / 'census.json', run_census(frame, vocabulary))
        elif command == 'materialize-bc':
            dataset = inputs['parent']['dataset_identity']
            provenance = {'source_ids': {**{s['file']: s['sha256'] for s in dataset['sources']},
                                         'processed_cache': dataset['processed_sha256']},
                          'config_sha256': identity['config_sha256'], 'code_commit': identity['git']['commit'],
                          'data_version': f"processed {dataset['processed_sha256'][:12]}; regular R; assign_fold(2025)"}
            files, paths = inputs['files'], inputs['paths']
            keys = pinned_parquet(paths['p4_train_keys'], files['p4_train_keys']['sha256'])
            rule = decisions['D-1']
            run_materialize(frame, store, out, rule=rule, provenance=provenance, two_hand_rule=decisions['D-8'],
                            bind_inputs=lambda art, rows: pid.bind_policy_inputs(
                                inputs['bundle_path'], inputs['bundle_sha'], paths, bc_artifact=art, context_rows=rows,
                                aux_classes=reg['classes']['aux']),
                            train_keys=keys, keys_record=inputs['prep']['samples']['train'],
                            compare_rule='BC_E' if rule == 'BC_P' else 'BC_P')
        elif command == 'bind-probe':
            sealed = sealed_probe_arrays(inputs)
            probe = plan['stages']['S2_bind_probe']
            _, positions = probe_positions(sealed['keys'], sealed['levels'], frame, probe['rows'])
            blocks = blocks_containing(frame, positions)
            bc_path, bc_sha = registered('bc')
            components = bind(blocks, load_train_bc(bc_path, bc_sha))
            dump(out / 'identity.json', {'sha256': components.sha256, 'identity': components.identity})
            dump(out / 'probe.json', run_bind_probe(store, components, sealed, count=probe['rows'],
                                                   atol_primary=probe['atol_primary'],
                                                   atol_frequency=probe['atol_frequency_raw']))
        else:
            bc_path, bc_sha = registered('bc')
            support_path, support_sha = registered('support')
            if command == 'v5-denominators':
                runtime = prt.build_runtime(bc_path, bc_sha, support_path, support_sha, out / 'ledger.jsonl')
                dump(out / 'v5.json', run_v5(runtime, store, 'dev', Deadline(caps['S4_V5_denominators']['cap_seconds'])))
            else:
                search = reg['search']
                _require(all(search.get(k) is not None for k in ('tau', 'samples', 'pitch_cap', 'planning_seed', 'budget')),
                         'registered search settings (D-9) required')
                split = 'temperature' if command == 'profile' else 'dev'
                blocks = preq.pa_blocks(frame, np.flatnonzero(frame.split.eq(split).to_numpy()))
                if command == 'profile':
                    count = caps['S3_profile']['starts']
                    blocks = sorted(blocks, key=lambda b: canonical_hash([search['planning_seed'], b[0]]))[:count]
                snapshot, as_of = None, None
                if command == 'dr-evaluate' and decisions['D-7'] == 'window_start_snapshot':
                    as_of = plan['profile_as_of']['dev']
                    snapshot = preq.style_snapshot(frame, as_of)
                components = bind(blocks, load_train_bc(bc_path, bc_sha), snapshot, as_of)
                runtime = prt.build_runtime(bc_path, bc_sha, support_path, support_sha, out / 'ledger.jsonl',
                                            components=components, budget=RowBudget(search['budget'], seed_count=5),
                                            tau=search['tau'], samples=search['samples'], pitch_cap=search['pitch_cap'],
                                            seed=search['planning_seed'],
                                            expected_identity_sha256=reg.get('expected_identity_sha256'))
                if command == 'profile':
                    deadline = Deadline(caps['S3_profile']['cap_seconds'])
                    submit_pas(runtime, store, blocks, deadline)
                    runtime.verify_components()
                    dump(out / 'profile.json', {'pas': len(blocks), 'seconds': deadline.elapsed(),
                         'conditional_rows': runtime.improvement.simulator.budget.conditional_rows,
                         'decisions': runtime.summary()['requests'], 'quality_values_read': False})
                else:
                    result, rows = run_dr(runtime, store, components, 'dev', Deadline(caps['V4_dr_evaluate']['cap_seconds']),
                                          draws=plan['bootstrap']['draws'], seed=plan['bootstrap']['seed'])
                    dump(out / 'dr.json', result)
                    pd.DataFrame(rows).to_parquet(out / 'pa_values.parquet', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--local-config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('command', choices=COMMANDS)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    registration(config)  # refuse before any data is touched
    pin = (config['le2025_validation_plan'].get('local_config') or {}).get('sha256')
    _require(pin is not None and hash_file(args.local_config) == pin, 'local config differs from its registered pin')
    local = json.loads(args.local_config.read_text())
    dispatch(args.command, config, args.config, local, args.output.resolve(), load_inputs(config, local))


if __name__ == '__main__':
    main()
