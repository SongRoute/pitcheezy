"""SYNTHETIC-ONLY fixtures for the EXP-OPE-RES-02 (A2b) tests: a synthetic DRAFT config with the registered
structure (never the real draft or configs/EXP-OPE-RES-02.json), a filled REGISTERED variant with synthetic paths,
B2-denominator decision streams over the A2 synthetic ledgers, and small synthetic records. No real data, ledger,
model, T7 path or 2026 file is read.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from pitchmdp import ope_registry_a2b as rb  # noqa: E402
from pitchmdp import policy_estimator as est  # noqa: E402


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def draft_config():
    """A synthetic DRAFT config carrying exactly the fields the A2b registry checks (values from the registry)."""
    roles = {name: {'index': rb.ROLES[name], 'value': rb.ROLE_VALUES[name]} for name in rb.ROLES}
    roles['selection_salt']['name'] = 'a2b_selection_salt'
    return {
        'id': rb.EXPERIMENT_ID, 'protocol': rb.PROTOCOL, 'status': rb.DRAFT, 'registered': False,
        'preregistration_complete': False, 'base_commit': rb.BASE_COMMIT,
        'execution': {'enabled': False, 'reason': 'synthetic draft fixture'},
        'registration': {'target': rb.CONFIG_PATH, 'registered_kst': None, 'review_reference': None,
                         'codex_reconfirmation_reference': None, 'code_review_reference': None,
                         'code_review_phase2_reference': None,
                         'config_digest': {'in_config_digest_field': 'none', 'hash_target': 'see registry',
                                           'nulled_json_paths': list(rb.NULLED_PATHS)}},
        'seeds': {'a2b_base': {'value': rb.A2B_BASE}, 'roles': roles,
                  'world_seeds_derived': {'k_1_to_76': list(rb.WORLD_SEEDS_DERIVED)},
                  'run_index_map_decided_plan': {'S2_prime_2zeta': [1, 10], 'S2_prime_2kappa': [11, 65],
                                                 'descriptive_kappa_prime': [66, 70]},
                  's6_selection_salt_recomputed': {'value': rb.S6_SELECTION_SALT_RECOMPUTED},
                  'planning_seeds_S5X': [0, 1, 2],
                  'bootstrap_seed_by_option': {
                      'G-INT-A2_always': dict(rb.EVALUATOR_SEEDS[rb.MODE_S6], placebo=None, pigeonhole=None),
                      'rehearsal_options_2_3': dict(rb.EVALUATOR_SEEDS[rb.MODE_REHEARSAL], placebo=None,
                                                    pigeonhole=None)}},
        'arms': {'tuples': {'adoption_candidates': list(rb.CANDIDATES), 'count_phase_1_rows': 25},
                 'R4_bootstrap': {'game': {'draws': rb.DRAWS}, 'pitcher': {'draws': rb.DRAWS}}},
        'decision_rules': {'condition_ii_half_width': {'X': {'value': 0.3}},
                           'labels_in_precedence': [{'label': label} for label in rb.LABELS],
                           'selection': {'total_order': {'value': list(rb.TOTAL_ORDER)}}},
        'synthetic_bias_check': {'acceptance': {'B': {'value': 1.26e-05}},
                                 'S2_prime': {'N_prime_decided': {'zeta_2x': 10, 'kappa_2x': 55,
                                                                  'descriptive_kappa_prime': 5}},
                                 'reuse_from_EXP_OPE_RES_01': {'eta_p': rb.ETA_P}},
        'cost_budget': {'hang_guards_seconds': {k: {'value': v} for k, v in rb.HANG_GUARDS.items()},
                        'rss_cap_gib': {'value': 8}, 'free_memory_guard_gib': {'value': 4}},
        'stages': [{'id': s} for s in rb.CONFIG_STAGES],
        'governing_rules': {'not_modified_paths': list(rb.PROTECTED_SOURCES)},
        'identities': {'candidate_policy_sha256': rb.CANDIDATE_POLICY_SHA256,
                       'exp_ope_res_01': {'results_file_sha256': _sha('synthetic A2 results file')},
                       'a3_run': {'config_sha256': _sha('a foreign config (allowed name)')}},
        'a3_record': {'config_sha256': {'value': _sha('A3 config'), 'source': 'synthetic'}},
        'registration_input_pins': {key: None for key in rb.PIN_KEYS},
        'pre_run_gates': [{'id': g, 'status': 'open'} for g in ('G-CLAIM', 'G-REV', 'G-CODE', 'G-REG', 'G-SIGN')],
        'stage_output_seals': {'A2b-L_ledger_sha256': None, 'A2b-S3_public_outputs_sha256': None,
                               'rule': 'sealed in the external manifest'},
    }


def registered_config(root):
    """The synthetic draft filled as a REGISTERED config (synthetic absolute paths under ``root``)."""
    root = Path(root)
    config = draft_config()
    config.update(status=rb.REGISTERED, registered=True, preregistration_complete=True)
    config['execution'] = {'enabled': True, 'reason': 'synthetic'}
    config['registration'].update(registered_kst='2026-11-10T22:00+09:00', review_reference='REVIEW-A2B-SYN',
                                  codex_reconfirmation_reference='CODEX-SYN', code_review_reference='GCODE-SYN')
    for gate in config['pre_run_gates']:
        if gate['id'] in rb.GATES_PASSED:
            gate['status'] = 'PASS'
    h = lambda name: _sha(name)
    chain = [{'path': str(root / 'inputs' / f'chain{i}.json'), 'sha256': h(f'chain{i}')} for i in range(9)]
    config['registration_input_pins'].update({
        'rehearsal_population': rb.REHEARSAL_POPULATION, 'rehearsal_population_allowed': [rb.REHEARSAL_POPULATION],
        'r3_status': rb.R3_STATUS_DECIDED, 'r3_status_allowed': [rb.R3_STATUS_DECIDED],
        's2_prime_runs': {'zeta_2x': 10, 'kappa_2x': 55, 'descriptive_kappa_prime': 5},
        's6_dr_a5_ledger_path': str(root / 'inputs' / 's6' / 'ledger-dev-0.jsonl'),
        's6_dr_a5_ledger_sha256': h('ledger'), 's6_dr_a5_manifest_sha256': h('manifest'),
        's6_runtime_sha256_header': h('runtime'), 'a2_s0_outputs_manifest_sha256': h('a2s0'),
        'a2_s2_outputs_manifest_sha256': h('a2s2'), 'a2_digest_manifest_sha256': h('a2dm'),
        'a2_execution_source_commit': 'a' * 40, 'a3_reader_commit': 'b' * 40, 'execution_source_commit': 'c' * 40,
        'a3_b2_artifacts_sha256': {'02-fit/b2/model.txt': h('model'), '02-fit/b2/b2.json': h('b2'),
                                   '03-calibrate/calibrators/B2.json': h('cal')},
        'a3_freeze_primary_manifest_sha256': h('freeze'),
        'processed_frame_aligned': {'path': str(root / 'inputs' / 'aligned.parquet'), 'sha256': h('aligned'),
                                    'identity_manifest_path': str(root / 'inputs' / 'aligned.identity.json'),
                                    'identity_manifest_sha256': h('identity'), 'source_sha256': h('processed')},
        'train_raw_files_sha256': {name: h(name) for name in ('statcast_2023.parquet', 'statcast_2024.parquet',
                                                              'statcast_2025.parquet')},
        'processed_frame_identity': {'path': str(root / 'inputs' / 'pitches.parquet'), 'sha256': h('processed')},
        'x2_world_predictor_identity': {'kind': 'g0_single_member', 'seed': 3},
        'new_files_sha256': {rel: h(rel) for rel in rb.NEW_FILES},
        'new_files_sha256_scope': 'design 9.1 items 1-7 and the new tests',
        'digest_manifest_path': str(root / 'manifests' / 'a2b-digest.jsonl'),
        'output_root': str(root / 'outputs'),
        'a2_digest_manifest_path': str(root / 'inputs' / 'a2-digest.jsonl'),
        'a2_results_file': {'path': 'results/EXP-OPE-RES-01.json',
                            'sha256': config['identities']['exp_ope_res_01']['results_file_sha256']},
        'a3_freeze_primary_manifest_path': str(root / 'inputs' / 'a3' / '04-freeze-primary' / 'manifest.json'),
        'a3_s6_weights_record': {'path': str(root / 'inputs' / 'a3' / 'weights.json'), 'sha256': h('weights')},
        'a3_weight_settings': {'large_weight_threshold': 10.0, 'mei': 0.001, 'bootstrap_draws': 200,
                               'bootstrap_seed': 11, 'E5_thresholds': {'pa': 100, 'game': 30}},
        'policy_registration_chain': chain,
        'ml_policy_d127_chain': {'config_sha256': chain[0]['sha256'],
                                 'addenda_1_to_8_d127_sha256': [c['sha256'] for c in chain[1:]],
                                 'member_loader_pin': {'file': 'loader.py', 'sha256': h('loader')}},
        'selection': {'selection_s6_game_list_sha256': h('s6 list'), 'a2b_new_game_list_sha256': h('new list'),
                      'month_quota_table': {'2025-07': 50, '2025-08': 52, '2025-09': 48}},
        'runtime_setting': {'selected_tau': 0.003, 'samples': 16, 'pitch_cap': 8, 'row_budget': 9000000,
                            'planning_seed': 0, 'evaluation_role': 'evaluation'},
        'dev_style_snapshot': {'path': str(root / 'inputs' / 'style_dev.json'), 'sha256': h('style'),
                               'as_of_exclusive': '2025-07-01'},
        'g0_bundle_sha256': h('g0'),
        'b2_inference_environment': {'lightgbm': '4.6.0', 'pyproject_sha256': h('pyproject')},
        'smoke_train_subset': {'game_dates': ['2024-06-01', '2024-06-07']},
        'smoke_tab_replications_per_cell': 3,
        'rule': 'synthetic', 'r3_status_rule': 'synthetic'})
    return config


# ---------------------------------------------------------------- B2-denominator streams over A2 synthetic ledgers
def with_request_ids(decisions):
    """The synthetic A2 ledger rows with a request id (the real ledger carries one per decision)."""
    for row in decisions:
        row['request_id'] = f"{row['pa_id']}:{row['decision_index']}"
    return decisions


def evaluator_inputs(seed=5, n_pas=300, n_games=20, n_pitchers=12):
    """A synthetic ledger, facts and surrogate rewards in the shape the A2 tests use (_evaluate)."""
    from ope_a2_fixtures import synthetic_ledger
    from pitchmdp import ope_surrogate as osg
    decisions, facts = synthetic_ledger(seed=seed, n_pas=n_pas, n_games=n_games, n_pitchers=n_pitchers)
    with_request_ids(decisions)
    rng = np.random.default_rng(1)
    sur = {pa: {'R1_a': (float(np.clip(f['reward'] + rng.normal(0, .02), 0, 1)), 'class'),
                'R1_b': (float(np.clip(f['reward'] + rng.normal(0, .01), 0, 1)), 'bip_quality'), 'spray_missing': False}
           for pa, f in facts.items() if f.get('reward') is not None}
    census = {'train_rows': 1000, 'bip_rows': 200, 'bip_missing_ev_or_la': 9, 'bip_unmapped_event': 3, 'bip_used': 188,
              'dp_ineligible_double_play_to_out': 2, 'spray_missing_among_used': 4,
              'mapping_counts': {'field_out': 120, 'single': 50, 'double': 18}}
    common = {'surrogates': sur, 'train_residuals': {'hit': {'FB': {'n': 10, 'mean_residual': -0.01, 'se': 0.005}}},
              'r1_paths': {'rehearsal': osg.path_shares(sur), 'train_census': census}, 'minimum_pitchers': 4,
              'placebos': 4, 'draws': 200, 'minimum_games': 5, 'minimum_pa_starts': 10}
    return decisions, facts, common


def e0_ids_of(facts):
    return {pa for pa, info in facts.items() if info['in_population']}


def b2_stream(decisions, vectors, e0_ids):
    """Test-side reference of the B2 stream (independent of ope_denominator_a2b): logging := the vector and rho
    recomputed in the evaluated prefix of each E0 PA; every other row unchanged."""
    def stream():
        stop = set()
        for row in decisions:
            pa = row['pa_id']
            if pa not in e0_ids or pa in stop or row['status'] not in est.EVALUATED:
                if pa in e0_ids and row['status'] not in est.EVALUATED:
                    stop.add(pa)
                yield row
                continue
            v = np.asarray(vectors[(pa, row['decision_index'])], dtype=np.float64)
            result = dict(row['result'])
            a = result['logged_index']
            result['logging'] = v.tolist()
            for policy in ('candidate', 'reference'):
                result[f'rho_{policy}'] = float(np.asarray(result[policy], dtype=np.float64)[a] / v[a])
            yield {**row, 'result': result}
    return stream


def b2_vectors(decisions, seed=11, scale=0.4, zero_at=None):
    """A synthetic pi_b_hat_B2 vector per evaluated decision: the recorded B0 logging law tilted inside its own
    support (same support as B0, renormalized); ``zero_at``: request keys whose logged action gets no B2 mass."""
    rng = np.random.default_rng(seed)
    out = {}
    for row in decisions:
        result = row.get('result')
        if row['status'] not in est.EVALUATED or not isinstance(result, dict):
            continue
        logging = np.asarray(result['logging'], dtype=np.float64)
        tilt = np.exp(rng.normal(0, scale, len(logging)))
        vector = np.where(logging > 0, logging * tilt, 0.)
        key = (row['pa_id'], row['decision_index'])
        if zero_at is not None and key in zero_at:
            vector[result['logged_index']] = 0.
        out[key] = (vector / vector.sum()).tolist()
    return out


# ---------------------------------------------------------------- a synthetic ML-POLICY registration (dispatch chain)
def a2b_game_frame(n_dev_games=4):
    """``test_policy_validation.game_frame`` with ``n_dev_games`` DEV games (2 = that frame exactly): the S6 rehearsal
    takes its registered 2-game selection, the other DEV games form the A2b new list."""
    import test_policy_validation as tpv
    from pitchmdp.archetypes import add_batter_style_history
    from pitchmdp.sequence_data import PHYSICAL_COLUMNS
    rng, rows = np.random.default_rng(3), []
    for g in range(8):  # TRAIN
        date = (pd.Timestamp('2024-06-01') + pd.Timedelta(days=g)).strftime('%Y-%m-%d')
        for ab in range(1, 7):
            rows += tpv.pa_rows(100 + g, ab, date, 'train', ('short', 'long')[ab % 2],
                                event=('single', 'field_out')[ab % 2])
    for ab in range(1, 4):  # one May game (profile starts)
        rows += tpv.pa_rows(150, ab, '2025-05-20', 'temperature', 'short')
    for g in range(2):  # two June games (tau selection)
        for ab in range(1, 4):
            rows += tpv.pa_rows(160 + g, ab, f'2025-06-1{g}', 'blend', ('short', 'long')[ab % 2])
    for g in range(n_dev_games):  # DEV
        date = (pd.Timestamp('2025-07-01') + pd.Timedelta(days=g)).strftime('%Y-%m-%d')
        for ab, (kind, extra) in enumerate(tpv.DEV_SCRIPT, start=1):
            rows += tpv.pa_rows(200 + g, ab, date, 'dev', kind, **extra)
    frame = pd.DataFrame(rows)
    for c in PHYSICAL_COLUMNS:
        frame[c] = rng.normal(size=len(frame))
    last = ~frame.duplicated(['game_pk', 'at_bat_number'], keep='last')
    frame['is_pa_terminal'] = last & frame.events.notna() & frame.events.ne('truncated_pa')
    for game, part in frame.groupby('game_pk'):  # home wins every game: final result is determined
        frame.loc[part.index[-1], ['post_home_score', 'post_away_score']] = (1, 0)
    return add_batter_style_history(frame)


def ml_policy_world(n_dev_games=4):
    """A SYNTHETIC ML-POLICY registration built through the registered ``run_policy_validation.dispatch`` (the
    ``test_policy_validation.DispatchTests`` chain: census, materialize-bc, style-snapshot, the labelled bind-probe
    stub, v5-denominators, profile, tau-select, v2-world, dr-evaluate on runtime v3 with the L1-R estimator block),
    on ``a2b_game_frame(n_dev_games)`` with the fake pinned G0 / WE files. Returns a namespace (``rpv``, ``policy``,
    ``inputs`` with the store, ``s6`` = the sealed dr-evaluate stage directory, ``root`` - removed by the caller)."""
    import test_policy_validation as tpv
    from pitchmdp import policy_estimator as est
    from pitchmdp import policy_runtime as pr
    from pitchmdp.matrix_data import ordered_key_hash
    rpv = tpv.rpv
    fx = tpv.DispatchTests('test_registered_chain_through_dispatch')
    fx.setUp()
    fx.frame = a2b_game_frame(n_dev_games)
    fx.store = tpv.build_store(fx.frame)
    runs, config = fx.config()
    config['le2025_validation_plan']['stages']['S6_V4']['row_budget'] = 9000000  # the registered S6_V4 value
    config['le2025_validation_plan']['stages']['S6_V4']['estimator'] = {
        'runtime': pr.CONTRACT_V3, 'censoring': 'l1r', 'natural_course_refusals': list(est.H_K_REFUSALS), 'gr_cap': .5,
        'c_deltas': [.008, .0278], 'mei': .001, 'v3_law': 'tempered_alpha_0.5'}
    root = fx.root
    config_path = root / 'config.json'
    config_path.write_text(json.dumps(config))
    bundle_path, bundle_sha = fx.syn.bundle_file()
    keys_path = root / 'train_keys.parquet'
    fx.keys.to_parquet(keys_path, index=False)
    files = {**fx.syn.bundle['files'], 'p4_train_keys': {'path': str(keys_path), 'sha256': tpv.sha(keys_path)}}
    inputs = {'bundle_path': bundle_path, 'bundle_sha': bundle_sha, 'files': files,
              'paths': {**fx.syn.paths, 'p4_train_keys': keys_path},
              'prep': {'features': {'tokens': {'type_vocabulary': tpv.TYPES}},
                       'samples': {'train': {'n': len(fx.keys), 'rows_sha256': ordered_key_hash(fx.keys)}}},
              'parent': {'dataset_identity': {'sources': [{'file': 'synthetic', 'sha256': 'a' * 64}],
                                              'processed_sha256': 'f' * 64}},
              'frame': fx.frame, 'member_loader': tpv.load_member}
    chain = [config_path]

    def load(store):
        return {**inputs, 'store': fx.store if store else None}

    def run(command, name):
        rpv.dispatch(command, rpv.load_registration(chain[0], chain[1:]), {}, runs / name, load)
        return runs / name

    def register(stage, entries):
        path = root / f'addendum-{len(chain)}.json'
        path.write_text(json.dumps({'parent_sha256': rpv.hash_file_bytes(chain[-1].read_bytes()), 'stage': stage,
                                    'registered_inputs': {n: {'path': str(p), 'file_sha256': tpv.sha(p)}
                                                          for n, p in entries.items()}}))
        chain.append(path)
    runs.mkdir(parents=True)
    # a MagicMock, not a function of this module: the bound-components identity hashes the repository source closure
    # reachable from module namespaces (policy_identity.project_sources), so a lambda here would put this file in the
    # closure while patched and make later bindings (outside the patch) differ from the registered bind identity
    with mock.patch('run_ml_matrix.check_location', return_value=root / 'artifact'):
        census = run('census', 'S0')
        register('S0', {'census': census / 'census.json'})
        s1 = run('materialize-bc', 'S1')
        s1b = run('style-snapshot', 'S1b')
        stub = runs / 'S2-stub'  # bind-probe needs sealed G0 archives; its sealed record is a labelled stub
        stub.mkdir()
        bound = fx.syn.bind(tpv.pa.load_train_bc(s1 / 'bc_BC_P.json', tpv.sha(s1 / 'bc_BC_P.json')),
                            rpv.pa_contexts(fx.store, tpv.blocks_of(fx.frame, 'dev')[:1])[0])
        (stub / 'probe.json').write_text(json.dumps({'pass': True, 'synthetic_stub': True}))
        (stub / 'identity.json').write_text(json.dumps({'sha256': bound.sha256}))
        (stub / 'manifest.json').write_text(json.dumps({'command': 'bind-probe', 'artifact_sha256': {
            'probe.json': tpv.sha(stub / 'probe.json'), 'identity.json': tpv.sha(stub / 'identity.json')}}))
        register('S1-S2', {'bc': s1 / 'bc_BC_P.json', 'hands': s1 / 'hands.json', 'support': s1 / 'support_primary.json',
                           'materialize': s1 / 'materialize.json',
                           **{f'style_{s}': s1b / f'style_{s}.json' for s in ('dev', 'temperature', 'blend')},
                           'bind_probe': stub / 'probe.json', 'bind_identity': stub / 'identity.json'})
        v5 = run('v5-denominators', 'S4')
        profile = run('profile', 'S3')
        register('S3-S4', {'profile': profile / 'profile.json', 'v5': v5 / 'v5.json'})
        freeze = run('tau-select', 'S3b')
        register('S3b', {'tau_freeze': freeze / 'tau_freeze.json'})
        v2 = run('v2-world', 'S5')
        register('S5', {'v2': v2 / 'v2.json'})
        s6 = run('dr-evaluate', 'S6')
    policy = rpv.load_registration(chain[0], chain[1:])
    return SimpleNamespace(rpv=rpv, policy=policy, inputs=load(True), root=root, s6=s6, chain=list(chain), fx=fx,
                           frame=fx.frame, store=fx.store, syn=fx.syn, no_pitch=tpv.NO_PITCH, tpv=tpv)


def synthetic_a2_s0(seed=0):
    """A SYNTHETIC sealed-A2-S0 document: the EXP-OPE-RES-01 ``build_s0`` on the A2 synthetic TRAIN frame (toy WE),
    JSON round-tripped as the sealed s0.json. Returns (train frame, document)."""
    import run_ope_a2 as roa
    from ope_a2_fixtures import ToyAdvancement, ToyWE, synthetic_train_frame
    train = synthetic_train_frame(seed)
    s0 = roa.build_s0(train, {'we': ToyWE(), 'advancement': ToyAdvancement()}, min_pitches=100)
    return train, json.loads(json.dumps(s0, default=roa._default))


def read_s6(world):
    """The synthetic S6 ledger through the allowlisted A3 reader: (header, decisions, pins)."""
    from pitchmdp import behavior_weights as bw
    ledger = world.s6 / 'ledger-dev-0.jsonl'
    first = json.loads(ledger.read_text().splitlines()[0])
    pins = {'ledger_sha256': world.tpv.sha(ledger), 'manifest_sha256': world.tpv.sha(world.s6 / 'manifest.json'),
            'runtime_sha256': first['runtime_sha256'], 'candidate_sha256': first['pins']['candidate']['sha256']}
    header, decisions, _ = bw.read_ledger(ledger, manifest_path=world.s6 / 'manifest.json', **pins)
    return header, decisions, pins


# ---------------------------------------------------------------- synthetic public files and phase-2 records
def public_doc(widths=None, s5=None, *, baseline=(1.0e-3, 1.1e-3), flag=None, b2_pass=False):
    """A minimal typed A2b-S3 public file: H_point per public row and bootstrap (default: the baseline widths; rows
    in ``widths`` {row: (game, pitcher)} override, None = invalid), S-5 per row ({row: (game, pitcher)}, default
    PASS), the R5 h flag and the B2 precision flag. The lambda-fixed row is never public (sealed until A2b-UB)."""
    widths, s5 = dict(widths or {}), dict(s5 or {})
    pairs, s5_out = {}, {}
    for row in rb.PUBLIC_ROWS:
        h = baseline if row == rb.BASELINE else widths.get(row, baseline)
        h = (None, None) if h is None else h
        pairs[row] = {'H_point': {'game': h[0], 'pitcher': h[1]},
                      'valid': {'game': h[0] is not None, 'pitcher': h[1] is not None}}
        s5_out[row] = ({'game': 'BASELINE', 'pitcher': 'BASELINE'} if row == rb.BASELINE else
                       dict(zip(('game', 'pitcher'), s5.get(row, ('PASS', 'PASS')))))
    return {'experiment': rb.EXPERIMENT_ID, 'stage': 'A2b-S3', 'r3_status': rb.R3_STATUS_DECIDED, 'pairs': pairs,
            'S5': s5_out, 'R5': {'h': {'flag': flag, 'decision_input': False}},
            'flags': {'B2_PRECISION_PASS_DIAGNOSTIC': b2_pass}}


def a2_tab(outcome='PASS', override=None):
    """Reused EXP-OPE-RES-01 TAB acceptance records {T-A id: {A2 pair: cell}} (override {(id, pair): outcome})."""
    from pitchmdp import ope_redesign_registry as a2
    override = override or {}
    return {spec['id']: {pair: {'outcome': override.get((spec['id'], pair), outcome)} for pair in a2.PAIRS}
            for spec in a2.TAB_ACCEPTANCE}


def a2_nets(outcome='PASS'):
    from pitchmdp import ope_redesign_registry as a2
    return {net: {pair: {'outcome': outcome} for pair in a2.PAIRS} for net in ('S1', 'S4')}


def pooled(cell, outcome='PASS', *, invariant='PASS', invalid_tuple=None):
    """A pooled S-2' cell record in the ope_synthetic_a2b.pool_cell shape."""
    record = {'cell': cell, 'n_prime': rb.N_PRIME[cell], 'components': {c: {'outcome': outcome} for c in ('R1_a', 'R1_b')},
              'tuples': {}, 'invariant': {'outcome': invariant, 'max_abs_gamma_a': 0.0} if invariant else None,
              'decision_input': cell in rb.S2P_DECISION_CELLS}
    for t in ('R1_axR2_0xB0', 'R1_axR2_1xB0', 'R1_bxR2_0xB0', 'R1_bxR2_1xB0'):
        record['tuples'][t] = ({'outcome': 'INCONCLUSIVE', 'invalid_repetition': True} if t == invalid_tuple
                               else {'outcome': outcome})
    return record
