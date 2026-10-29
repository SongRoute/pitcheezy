"""EXP-OPE-1STEP-01 runner (spec v2): one-step dose-response OPE on a sealed decision-ledger stage (side experiment).

    .venv/bin/python experiments/pitchmdp/scripts/run_ope_onestep.py --config CONFIG.json --check
    .venv/bin/python experiments/pitchmdp/scripts/run_ope_onestep.py --config CONFIG.json --run [--max-seconds N]

Design: docs/pm/research/side-onestep-ope-proposal-2026-10-29.md + SPEC DELTA v2; estimator: pitchmdp/ope_onestep.py.

Order (fail closed; IntegrityError = refusal, exit 2):
1. the config is validated (unknown or missing keys refused); the output ``<output_dir>/<check|run>`` must be a NEW
   directory outside the stage directory; with ``source_commit`` the code at HEAD must be that commit's code;
2. every pinned stage file (ledger, pa_values.parquet, dr.json, manifest.json, started.json) is hashed and compared
   with its pin BEFORE any of them is parsed; the stage manifest must list the others with the same sha256; the
   ledger is hashed again while it is streamed and must still equal its pin; the ledger header must exist and pin a
   P3 candidate at the sealed tau whose DR q-hat source is the registered one;
3. PA facts (``--pa-facts-source frame``, the only source): the registered PA-end reward recomputed from the processed
   frame and the frozen C0 WE through the registered loaders (run_policy_validation.load_inputs with store=False,
   policy_identity.load_pinned_we, run_policy_validation.select_games / game_blocks, policy_requests.game_table /
   pa_outcome / pa_manifest) and checked PA by PA against pa_values (policy_estimator.pa_value re-run on the ledger
   rows reproduces the sealed row). No ``derived`` source: fl(fl(G - q) + q) != G in general and PAs censored after
   their decisions carry no terminal residual;
4. one streamed ledger pass (ope_onestep.Scan): row validation, the sealed-candidate identity check, D, the population;
5. prediag.json (no G: ESS, noise ratio, KL, evaluability per family x tau) is written FIRST; ``--check`` then writes
   check.json; ``--run`` computes every G-dependent section and writes results.json and per_tau.csv; SHA256SUMS last.
Every run records wall/CPU seconds, peak RSS, seeds, the git commit and the python/numpy/pandas versions. A hang guard
(``--max-seconds``: SIGALRM plus loop checkpoints) and an optional peak-RSS guard abort with failed.json (exit 3).
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gc
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import resource
import signal
import subprocess
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'scripts'))

import numpy as np
import pandas as pd

from pitchmdp import ope_onestep as ope
from pitchmdp import policy_estimator as est
from pitchmdp import policy_runtime as prt
from pitchmdp.matrix_policy_artifacts import is_appledouble
from pitchmdp.policy_artifacts import IntegrityError, _require

MODES = ('check', 'run')
FIXED_PINS = ('pa_values.parquet', 'dr.json', 'manifest.json', 'started.json')
REQUIRED = ('experiment_id', 'stage_dir', 'ledger', 'pins', 'output_dir', 'tau_grid', 'sealed_tau',
            'identity_tolerance', 'expected_dr_q_source', 'families', 'primary', 'disagreement', 'eligibility',
            'evaluability', 'placebos', 'sensitivities', 'bootstrap', 'pa_facts')
OPTIONAL = ('exposure_label', 'candidate_identity_sha256', 'source_commit', 'outputs', 'notes')
CODE_PATHS = ('experiments/pitchmdp',)
EXIT = {'ok': 0, 'error': 1, 'refused': 2, 'guard': 3}
HEX = set('0123456789abcdef')
BASE_VARIANTS = ('dr', 'ips', 'sndr')


class HangGuardExceeded(BaseException):
    """The registered wall-clock guard fired (the run is aborted with a failure record, never silently retried)."""


class MemoryGuardExceeded(BaseException):
    """The optional peak-RSS guard fired."""


def _sha(value, n=64):
    return isinstance(value, str) and len(value) == n and set(value) <= HEX


def file_sha256(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(chunk), b''):
            digest.update(block)
    return digest.hexdigest()


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _int(value, low=0):
    return type(value) is int and value >= low


def _keys(block, name, required, optional=()):
    _require(isinstance(block, dict), f'{name} must be an object')
    unknown, missing = set(block) - set(required) - set(optional), set(required) - set(block)
    _require(not unknown and not missing, f'{name}: unknown keys {sorted(unknown)} / missing keys {sorted(missing)}')


def _pin(block, name):
    _keys(block, name, ('path', 'sha256'))
    _require(isinstance(block['path'], str) and block['path'] and _sha(block['sha256']), f'{name}: path and sha256 pin')
    return {'path': block['path'], 'sha256': block['sha256']}


def validate_config(config):
    """The registered config as a normalised spec (taus parsed); anything unknown or missing is refused."""
    _keys(config, 'config', REQUIRED, OPTIONAL)
    _require(isinstance(config['experiment_id'], str) and config['experiment_id'], 'experiment_id required')
    for key in ('stage_dir', 'output_dir'):
        _require(isinstance(config[key], str) and Path(config[key]).is_absolute(), f'{key} must be an absolute path')
    _require(isinstance(config['ledger'], str) and config['ledger'] and '/' not in config['ledger']
             and config['ledger'].endswith('.jsonl'), 'ledger must be a .jsonl file name inside the stage directory')
    pins = config['pins']
    _keys(pins, 'pins', (config['ledger'], *FIXED_PINS))
    _require(all(_sha(v) for v in pins.values()), 'every stage pin must be a sha256 hex digest')
    sealed_tau = ope.parse_tau(config['sealed_tau'])
    taus = ope.check_grid([ope.parse_tau(t) for t in config['tau_grid']], sealed_tau)
    tol = config['identity_tolerance']
    _require(_number(tol) and 0 <= tol < 1e-6, 'identity_tolerance must be a number in [0, 1e-6)')
    _require(isinstance(config['expected_dr_q_source'], str) and config['expected_dr_q_source'],
             'expected_dr_q_source required (e.g. "evaluation_seed")')
    families = config['families']
    _require(isinstance(families, list) and families and families[0] == 'temperature'
             and set(families) <= set(ope.MAIN_FAMILIES) and len(set(families)) == len(families),
             f'families: "temperature" first, values of {ope.MAIN_FAMILIES}')
    primary = config['primary']
    _keys(primary, 'primary', ('family', 'tau', 'subset', 'variant'))
    _require(primary['family'] == 'temperature' and primary['subset'] == 'all' and primary['variant'] == 'dr'
             and ope.parse_tau(primary['tau']) == sealed_tau, 'primary: temperature, dr, subset all, the sealed tau')
    dis = config['disagreement']
    _keys(dis, 'disagreement', ('q', 'ties', 'margins', 'confident'))
    margins = dis['margins']
    _require(dis['q'] == ope.Q_PI and dis['ties'] == 'not_in_D' and isinstance(margins, list) and margins
             and margins[0] == 0 and all(m == '1se' or (_number(m) and m >= 0) for m in margins)
             and len(set(map(str, margins))) == len(margins) and dis['confident'] in [ope.margin_key(m) for m in margins],
             'disagreement: q "q_planning", ties "not_in_D", margins [0, ...] (numbers or "1se"), confident in margins')
    elig = config['eligibility']
    _keys(elig, 'eligibility', ('statuses', 'population', 'require_reward'))
    _require(isinstance(elig['statuses'], list) and elig['statuses'] and set(elig['statuses']) <= set(ope.RESULT_STATUSES)
             and len(set(elig['statuses'])) == len(elig['statuses']), f'eligibility.statuses within {ope.RESULT_STATUSES}')
    _require(elig['population'] == 'E0' and elig['require_reward'] is True,
             'eligibility: population "E0" and require_reward true (whole PAs without a reward are excluded)')
    evaluability = config['evaluability']
    _keys(evaluability, 'evaluability', ('ess_ratio_min', 'noise_q90_max', 'game_ess_min', 'pa_ess_min'))
    _require(all(_number(v) and v >= 0 for v in evaluability.values()), 'evaluability thresholds must be numbers >= 0')
    plac = config['placebos']
    _keys(plac, 'placebos', ('perm_draws', 'perm_seed', 'kl_tolerance', 'outcome_shuffle_draws', 'outcome_shuffle_seed',
                             'neg'))
    _require(_int(plac['perm_draws']) and _int(plac['perm_seed']) and _int(plac['outcome_shuffle_draws'])
             and _int(plac['outcome_shuffle_seed']) and isinstance(plac['neg'], bool) and _number(plac['kl_tolerance'])
             and 0 < plac['kl_tolerance'] < 1, 'placebos: nonnegative int draws/seeds, kl_tolerance in (0, 1), bool neg')
    sens = config['sensitivities']
    _keys(sens, 'sensitivities', ('switch_caps', 's_pos', 'cross_replicate'))
    _require(isinstance(sens['switch_caps'], list) and all(_number(c) and c > 0 for c in sens['switch_caps'])
             and len(set(sens['switch_caps'])) == len(sens['switch_caps']) and isinstance(sens['s_pos'], bool)
             and isinstance(sens['cross_replicate'], bool), 'sensitivities: positive switch caps, bool s_pos/cross')
    boot = config['bootstrap']
    _keys(boot, 'bootstrap', ('draws', 'seeds', 'invalid_share_max', 'minimum'))
    _require(_int(boot['draws'], 2), 'bootstrap.draws: int >= 2')
    _keys(boot['seeds'], 'bootstrap.seeds', ope.UNITS, ('trend',))
    _require(all(_int(v) for v in boot['seeds'].values()), 'bootstrap seeds must be nonnegative ints')
    _require(_number(boot['invalid_share_max']) and 0 <= boot['invalid_share_max'] <= 1, 'bootstrap.invalid_share_max')
    _keys(boot['minimum'], 'bootstrap.minimum', ('games', 'pitchers', 'pas'))
    _require(all(_int(v, 1) for v in boot['minimum'].values()), 'bootstrap.minimum: positive ints')
    source = config.get('source_commit')
    _require(source is None or _sha(source, 40), 'source_commit: null or a 40-hex commit')
    identity = config.get('candidate_identity_sha256')
    _require(identity is None or _sha(identity), 'candidate_identity_sha256: null or a sha256')
    outputs = config.get('outputs', {'decisions_parquet': False})
    _keys(outputs, 'outputs', ('decisions_parquet',))
    _require(isinstance(outputs['decisions_parquet'], bool), 'outputs.decisions_parquet: bool')
    facts = config['pa_facts']
    _keys(facts, 'pa_facts', ('source', 'split', 'policy_registration', 'local_config'),
          ('processed_frame_sha256', 'value_tolerance', 'check_selection'))
    _require(facts['source'] == 'frame', 'pa_facts.source: "frame" is the only PA-facts source (no exact "derived" identity)')
    _require(isinstance(facts['split'], str) and facts['split'], 'pa_facts.split required')
    registration = facts['policy_registration']
    _keys(registration, 'pa_facts.policy_registration', ('config',), ('addenda',))
    _require(isinstance(registration.get('addenda', []), list), 'pa_facts.policy_registration.addenda must be a list')
    processed = facts.get('processed_frame_sha256')
    _require(processed is None or _sha(processed), 'pa_facts.processed_frame_sha256: null or a sha256')
    value_tol = facts.get('value_tolerance', 1e-12)
    _require(_number(value_tol) and 0 <= value_tol < 1e-6, 'pa_facts.value_tolerance in [0, 1e-6)')
    _require(isinstance(facts.get('check_selection', True), bool), 'pa_facts.check_selection: bool')
    scan_families = [*families, *(['cross_replicate'] if sens['cross_replicate'] else []), *(['neg'] if plac['neg'] else [])]
    return {
        'experiment_id': config['experiment_id'], 'exposure_label': config.get('exposure_label'),
        'stage_dir': Path(config['stage_dir']), 'ledger': config['ledger'], 'pins': dict(pins),
        'output_dir': Path(config['output_dir']), 'taus': taus, 'sealed_tau': sealed_tau, 'identity_tolerance': float(tol),
        'expected_dr_q_source': config['expected_dr_q_source'], 'main_families': list(families),
        'scan_families': scan_families, 'primary': {**primary, 'tau': sealed_tau}, 'margins': list(margins),
        'confident': dis['confident'], 'statuses': list(elig['statuses']), 'evaluability': dict(evaluability),
        'variants': [*BASE_VARIANTS, *[f'switch:{c:g}' for c in sens['switch_caps']]], 'sensitivities': dict(sens),
        'placebos': {**plac, 'kl_tolerance': float(plac['kl_tolerance'])},
        'bootstrap': {'draws': boot['draws'], 'seeds': dict(boot['seeds'])},
        'invalid_share_max': float(boot['invalid_share_max']), 'minimum': dict(boot['minimum']),
        'source_commit': source, 'candidate_identity_sha256': identity, 'outputs': outputs,
        'pa_facts': {'source': 'frame', 'split': facts['split'],
                     'policy_registration': {'config': _pin(registration['config'], 'policy_registration.config'),
                                             'addenda': [_pin(a, 'policy_registration.addenda[]')
                                                         for a in registration.get('addenda', [])]},
                     'local_config': _pin(facts['local_config'], 'pa_facts.local_config'),
                     'processed_frame_sha256': processed, 'value_tolerance': float(value_tol),
                     'check_selection': facts.get('check_selection', True)}}


# ---------------------------------------------------------------- guards and environment

class Guard:
    """Loop checkpoints for the wall-clock guard and the optional peak-RSS guard (SIGALRM is the backstop)."""
    def __init__(self, max_seconds=None, max_rss_gb=None):
        self.max_seconds, self.max_rss_gb, self.start = max_seconds, max_rss_gb, time.monotonic()

    def checkpoint(self):
        if self.max_seconds is not None and time.monotonic() - self.start > self.max_seconds:
            raise HangGuardExceeded(f'--max-seconds {self.max_seconds} exceeded')
        if self.max_rss_gb is not None and peak_rss_bytes() > self.max_rss_gb * 2 ** 30:
            raise MemoryGuardExceeded(f'peak RSS above --max-rss-gb {self.max_rss_gb}')


def peak_rss_bytes():
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(raw if sys.platform == 'darwin' else raw * 1024)


def _raise_hang(signum, frame):
    raise HangGuardExceeded('SIGALRM hang guard fired')


class Environment:
    """Real dependencies (git, the processed frame, the frozen WE); synthetic tests subclass it. Nothing here runs before
    the config, output-directory and stage-pin checks."""

    def git_state(self, source_commit):
        run = lambda *args: subprocess.check_output(['git', *args], cwd=REPO, text=True).strip()
        state = {'commit': run('rev-parse', 'HEAD'), 'code_paths': list(CODE_PATHS),
                 'code_dirty': bool(run('status', '--porcelain', '--', *CODE_PATHS)), 'source_commit': source_commit,
                 'source_is_ancestor': None, 'code_changed_since_source': None}
        if source_commit:
            state['source_is_ancestor'] = subprocess.call(['git', 'merge-base', '--is-ancestor', source_commit, 'HEAD'],
                                                          cwd=REPO, stderr=subprocess.DEVNULL) == 0
            state['code_changed_since_source'] = [x for x in run('diff', '--name-only', source_commit, 'HEAD', '--',
                                                                 *CODE_PATHS).splitlines() if x]
        return state

    def load_frame_and_we(self, spec, checkpoint):
        """(frame, defense_we, ML-POLICY config, provenance) through the registered loaders. The ML-POLICY chain and the
        local config are verified against their pins before they are parsed, the processed-frame pin (optional, A2
        verify_processed_pin) before any row is decoded."""
        import run_policy_validation as rpv
        from pitchmdp import policy_identity as pid
        links = [spec['policy_registration']['config'], *spec['policy_registration']['addenda']]
        for link in links:
            _require(file_sha256(link['path']) == link['sha256'],
                     f'ML-POLICY registration file differs from its pin: {link["path"]}')
        reg = rpv.load_registration(Path(links[0]['path']), [Path(link['path']) for link in links[1:]])
        _require([link['sha256'] for link in reg['chain']] == [link['sha256'] for link in links],
                 'the loaded ML-POLICY chain differs from its pins')
        config = reg['config']
        _require(config.get('protocol') == rpv.PROTOCOL, 'not an ML-POLICY materialization/validation config')
        raw = Path(spec['local_config']['path']).read_bytes()
        local_sha = hashlib.sha256(raw).hexdigest()
        _require(local_sha == spec['local_config']['sha256'] == config['le2025_validation_plan']['local_config']['sha256'],
                 'local config differs from its pins')
        local = json.loads(raw)
        if spec['processed_frame_sha256'] is not None:
            root = Path(local['artifact_root'])
            quality = json.loads((root / 'reports' / 'data_quality.json').read_text())
            _require(quality.get('processed_sha256') == spec['processed_frame_sha256'],
                     'the processed-frame pin differs from the quality manifest (processed_sha256)')
            _require(file_sha256(root / 'processed' / 'pitches.parquet') == spec['processed_frame_sha256'],
                     'processed frame bytes differ from their pin')
        checkpoint()
        inputs = rpv.load_inputs(config, local, store=False)  # the registered frame: same loader, order and date guard
        checkpoint()
        ident = config['identity_registration']
        values, we_identity = pid.load_pinned_we(inputs['paths'], ident['we_contract_sha256'], ident['classes']['we'])
        we = values['we']

        def defense(state, defender_is_home):  # = policy_identity.BoundComponents.defense_we
            return float(we.predict_defense(state, defender_is_home))
        provenance = {'policy_registration_chain': reg['chain'], 'policy_config_sha256': reg['config_sha256'],
                      'local_config_sha256': local_sha, 'processed_frame_sha256': spec['processed_frame_sha256'],
                      'g0_bundle_file_sha256': inputs['bundle_sha'], 'we_identity': we_identity,
                      'frame_rows': int(len(inputs['frame'])),
                      'loaders': ['run_policy_validation.load_inputs(store=False)', 'policy_identity.load_pinned_we']}
        return inputs['frame'], defense, config, provenance


def frame_pa_facts(env, spec, dr, checkpoint):
    """{pa_id: facts} of the stage's games in the registered block order; ``check_selection`` re-derives the game list
    on this frame (run_policy_validation.select_games with the registered S6 count and selection salt)."""
    import run_policy_validation as rpv
    from pitchmdp import policy_requests as preq
    frame, defense, policy_config, provenance = env.load_frame_and_we(spec, checkpoint)
    games = dr['games']
    if spec['check_selection']:
        n_games = policy_config['le2025_validation_plan']['stages']['S6_V4']['n_games']
        selected = rpv.select_games(frame, spec['split'], n_games, rpv.role_seed(policy_config, 'selection_salt'))
        _require(selected == sorted(games), 'the stage game list differs from the registered selection on this frame')
    blocks = rpv.game_blocks(frame, games, spec['split'])
    games_table = preq.game_table(frame, {int(frame.game_pk.iloc[p[0]]) for _, p in blocks})
    checkpoint()
    facts = ope.frame_facts(frame, blocks, games_table, defense)
    dates = pd.to_datetime(frame['game_date'].iloc[np.concatenate([p for _, p in blocks])]) if blocks else None
    provenance = {**provenance, 'split': spec['split'], 'games': len(games), 'pas': len(facts),
                  'selection_checked': spec['check_selection'],
                  'window': None if dates is None else [str(dates.min().date()), str(dates.max().date())],
                  'reward_rule': 'policy_requests.pa_outcome(structural-end-v1, next_row), frozen C0 defense WE'}
    del frame, blocks, games_table
    gc.collect()
    return facts, provenance


# ---------------------------------------------------------------- stage pins and header

def verify_stage(spec):
    """Every pinned file hashed and compared BEFORE any is parsed; small files are kept as the verified bytes."""
    stage = spec['stage_dir']
    _require(stage.is_dir(), f'stage directory missing: {stage}')
    verified = {}
    for name, pin in spec['pins'].items():
        path = stage / name
        _require(path.is_file() and not is_appledouble(path), f'pinned stage file missing: {name}')
        if name == spec['ledger']:
            _require(file_sha256(path) == pin, f'pinned stage file differs from its pin: {name}')
        else:
            raw = path.read_bytes()
            _require(hashlib.sha256(raw).hexdigest() == pin, f'pinned stage file differs from its pin: {name}')
            verified[name] = raw
    manifest = json.loads(verified['manifest.json'])
    listed = manifest.get('artifact_sha256') if isinstance(manifest, dict) else None
    _require(isinstance(listed, dict), 'stage manifest lists no artifact_sha256')
    for name, pin in spec['pins'].items():
        if name != 'manifest.json':
            _require(listed.get(name) == pin, f'stage manifest does not list {name} with its pinned sha256')
    return verified, manifest


def header_record(header, spec, dr):
    """The ledger header (kind 'header', required): candidate P3 at the sealed tau, its sha256 and DR q-hat source."""
    pins = header.get('pins') or {}
    candidate = pins.get('candidate') or {}
    search, dr_q = candidate.get('search') or {}, candidate.get('dr_q') or {}
    _require(search.get('policy') == 'P3' and _number(search.get('tau')) and float(search['tau']) == spec['sealed_tau'],
             'ledger header candidate is not P3 at the registered sealed tau')
    _require(dr_q.get('source') == spec['expected_dr_q_source'],
             f'ledger header DR q-hat source {dr_q.get("source")!r} is not the registered {spec["expected_dr_q_source"]!r}')
    _require(pins.get('contract') in (prt.CONTRACT, prt.CONTRACT_V3), 'unknown runtime contract in the ledger header')
    _require(_sha(candidate.get('sha256')), 'ledger header pins no candidate sha256')
    _require(dr.get('identity_sha256') == candidate['sha256'], 'dr.json evaluates another candidate than the ledger header')
    if spec['candidate_identity_sha256'] is not None:
        _require(candidate['sha256'] == spec['candidate_identity_sha256'],
                 'ledger header candidate differs from the registered candidate identity')
    return {'runtime_sha256': header.get('runtime_sha256'), 'contract': pins.get('contract'),
            'positivity_record': pins.get('positivity_record', False), 'candidate_sha256': candidate['sha256'],
            'candidate_search': search, 'dr_q': dr_q}


def censoring_of(dr):
    """The censoring rule the sealed pa_values were computed with (dr.json censoring_rule; L1-R defaults only)."""
    rule = dr.get('censoring_rule')
    if not rule:
        return 'worst_case'
    _require(rule.get('rule') == 'L1-R' and rule.get('natural_course_refusals') == list(est.H_K_REFUSALS)
             and rule.get('gap_delta') is None and not rule.get('excluded_pas'),
             'sealed pa_values were not computed with the default L1-R rule')
    return 'l1r'


def environment():
    return {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
            'system': platform.system(), 'machine': platform.machine(), 'executable': sys.executable}


def execute(spec, mode, verified, manifest, env, guard, write):
    """The checked pass, prediag (written first through ``write``) and, for ``run``, every G-dependent section.
    Returns (document, extra files {name: bytes})."""
    ledger_path = spec['stage_dir'] / spec['ledger']
    dr = json.loads(verified['dr.json'])
    started = json.loads(verified['started.json'])
    _require(isinstance(dr.get('games'), list) and dr['games'] and all(type(g) is int for g in dr['games']),
             'dr.json lists no games')
    header = ope.read_header(ledger_path)
    head = header_record(header, spec, dr)
    censoring = censoring_of(dr)
    facts, facts_provenance = frame_pa_facts(env, spec['pa_facts'], dr, guard.checkpoint)
    guard.checkpoint()
    table = pd.read_parquet(io.BytesIO(verified['pa_values.parquet']))
    sealed = ope.sealed_rows(table.to_dict('records'))
    del table
    _require(list(sealed) == list(facts), 'pa_values PAs differ from the frame blocks of the stage games (ids or order)')
    pas = {pa_id: {**facts[pa_id], 'in_population': sealed[pa_id]['in_population'], 'pa_status': sealed[pa_id]['status']}
           for pa_id in sealed}
    scan = ope.Scan(taus=spec['taus'], sealed_tau=spec['sealed_tau'], identity_tolerance=spec['identity_tolerance'],
                    statuses=spec['statuses'], families=spec['scan_families'], margins=spec['margins'],
                    expected_q_source=spec['expected_dr_q_source'], runtime_sha256=head['runtime_sha256'])
    stream = ope.LedgerStream(ledger_path)
    check = ope.FactsCheck(censoring, spec['pa_facts']['value_tolerance'])
    scan.run(stream.decisions(), pas, sealed=sealed, facts_check=check, checkpoint=guard.checkpoint)
    _require(stream.sha256 == spec['pins'][spec['ledger']], 'ledger bytes changed after their pin was verified')
    _require(stream.header == header, 'streamed ledger header differs from the header read first')
    halting = {k: stream.kinds[k] for k in ope.HALTING_KINDS if stream.kinds[k]}
    _require(not halting, f'ledger has halting rows {halting}: a halted ledger is not estimated')
    tab = scan.table()
    records = scan.records(tab)
    _require(records['decision_rows'] == stream.kinds['decision'], 'decision rows were not all scanned')
    del scan
    gc.collect()
    provenance = {'experiment_id': spec['experiment_id'], 'contract': ope.CONTRACT, 'mode': mode,
                  'exposure_label': spec['exposure_label'], 'estimand': ope.ESTIMAND,
                  'tau_grid': [ope.tau_key(t) for t in spec['taus']], 'sealed_tau': ope.tau_key(spec['sealed_tau']),
                  'families': spec['scan_families'],
                  'stage': {'dir': str(spec['stage_dir']), 'pins': spec['pins'], 'manifest_command': manifest.get('command'),
                            'started': {k: started.get(k) for k in ('command', 'started_utc', 'git', 'config_sha256',
                                                                    'registration_chain')},
                            'games': len(dr['games']), 'censoring_rule_of_pa_values': censoring},
                  'ledger': {'rows': stream.rows, 'head_sha256': stream.head, 'kinds': dict(stream.kinds), 'header': head},
                  'pa_facts': {'source': 'frame', 'provenance': facts_provenance, 'crosscheck': check.record()},
                  'identity_check': records['identity_check'], 'q_source': records['q_source'],
                  'vocabulary_size': records['vocabulary_size'],
                  'seeds': {'bootstrap': spec['bootstrap']['seeds'], 'perm_placebo': spec['placebos']['perm_seed'],
                            'outcome_shuffle': spec['placebos']['outcome_shuffle_seed']}}
    counts = {'by_status': records['by_status'], **records['counts'], 'disagreement': records['disagreement']}
    guard.checkpoint()
    diag = ope.prediag(tab, spec['evaluability'])
    write('prediag.json', {'provenance': provenance, 'counts': counts, 'prediag': diag,
                           'note': 'written before any G-dependent number is computed'})
    doc = {'provenance': provenance, 'counts': counts, 'prediag': diag}
    files = {}
    if mode == 'run':
        guard.checkpoint()
        sections, csv_sections = ope.analysis(
            tab, diag, primary=spec['primary'], variants=spec['variants'], main_families=spec['main_families'],
            margins=spec['margins'], bootstrap=spec['bootstrap'], minimum=spec['minimum'],
            invalid_share_max=spec['invalid_share_max'], placebos=spec['placebos'], sensitivities=spec['sensitivities'],
            identity_tolerance=spec['identity_tolerance'], checkpoint=guard.checkpoint)
        doc.update(sections)
        doc['reading'] = ('stage 1 is EXPLORATORY_REANALYSIS on exposed <=2025 DEV: no adoption label; judgment '
                          'statistics (trend, max_t) use evaluable taus only, other rows are descriptive')
        files['per_tau.csv'] = per_tau_csv(ope.per_tau_rows(csv_sections))
        if spec['outputs']['decisions_parquet']:
            files['decisions.parquet'] = decisions_parquet(tab)
    return doc, files


def per_tau_csv(rows):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]) if rows else ['section'], lineterminator='\n')
    writer.writeheader()
    for row in rows:
        writer.writerow({k: '' if v is None else (repr(v) if isinstance(v, float) else v) for k, v in row.items()})
    return buffer.getvalue().encode()


def decisions_parquet(tab):
    """Per observed decision of the temperature family: ids, count, D, status, M, C, D_t and rho' per tau."""
    obs = tab['observed']
    f = tab['families'].index('temperature')
    on = tab['on_mask'][obs][:, None]
    w = (tab['rho'][obs, f, :] - tab['rho_ref'][obs][:, None]) * on
    corr = w * np.where(on, tab['G'][obs][:, None] - tab['qa'][obs, f][:, None], 0.)
    frame = pd.DataFrame({'pa_id': tab['pa_id'][obs].astype(str), 'game': tab['game'][obs],
                          'pitcher': tab['pitcher'][obs].astype(str), 'decision_index': tab['decision_index'][obs],
                          'balls': tab['balls'][obs], 'strikes': tab['strikes'][obs], 'status': tab['status'][obs].astype(str),
                          **{f'D:{ope.margin_key(m)}': tab['D'][obs, i] for i, m in enumerate(tab['margins'])}})
    for j, tau in enumerate(tab['taus']):
        key = ope.tau_key(tau)
        frame[f'M_{key}'] = tab['model'][obs, f, j]
        frame[f'C_{key}'] = corr[:, j]
        frame[f'delta_{key}'] = tab['model'][obs, f, j] + corr[:, j]
        frame[f'rho_{key}'] = tab['rho'][obs, f, j]
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    return buffer.getvalue()


def write_exclusive(path, data):
    with Path(path).open('xb') as stream:
        stream.write(data)


def dump(path, value):
    write_exclusive(path, (json.dumps(ope.jsonable(value), indent=1, ensure_ascii=False, allow_nan=False) + '\n').encode())


def sha256sums(out):
    names = sorted(p.name for p in out.iterdir() if p.is_file() and p.name != 'SHA256SUMS' and not is_appledouble(p))
    write_exclusive(out / 'SHA256SUMS', ''.join(f'{file_sha256(out / n)}  {n}\n' for n in names).encode())


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true', help='pins, counts, identity, facts and prediag; no G-dependent value')
    mode.add_argument('--run', action='store_true', help='the full registered computation')
    parser.add_argument('--max-seconds', type=float, default=None, help='hang guard (wall seconds)')
    parser.add_argument('--max-rss-gb', type=float, default=None, help='abort once peak RSS exceeds this many GiB')
    parser.add_argument('--pa-facts-source', choices=('frame',), default=None,
                        help='PA facts source; must equal the config (frame is the only source)')
    args = parser.parse_args(argv)
    for name in ('max_seconds', 'max_rss_gb'):
        value = getattr(args, name)
        if value is not None and not (math.isfinite(value) and value > 0):
            parser.error(f'--{name.replace("_", "-")} must be positive')
    return args


def _stop_alarm(max_seconds, previous):
    if max_seconds is not None:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, signal.SIG_DFL if previous is None else previous)


def main(argv=None, env=None):
    args = parse(argv)
    env = Environment() if env is None else env
    mode = 'check' if args.check else 'run'
    wall, cpu = time.monotonic(), time.process_time()
    try:
        raw = Path(args.config).read_bytes()
        config = json.loads(raw)
        spec = validate_config(config)
        _require(args.pa_facts_source in (None, spec['pa_facts']['source']), 'PA facts source differs from the config')
        out = spec['output_dir'] / mode
        _require(not out.exists(), f'output exists: {out}; preserve it and register a new output directory')
        _require(not out.resolve().is_relative_to(spec['stage_dir'].resolve()), 'output inside the sealed stage directory')
        git = env.git_state(spec['source_commit'])
        if spec['source_commit'] is not None:
            _require(git['source_is_ancestor'] is True and not git['code_changed_since_source'] and not git['code_dirty'],
                     'code at HEAD differs from the registered source commit (or uncommitted code changes)')
        verified, manifest = verify_stage(spec)
    except (ValueError, OSError) as error:  # IntegrityError is a ValueError
        print(f'REFUSED: {error}', file=sys.stderr)
        return EXIT['refused']
    out.mkdir(parents=True)
    run_info = {'mode': mode, 'argv': list(sys.argv if argv is None else argv), 'config_path': str(args.config),
                'config_sha256': hashlib.sha256(raw).hexdigest(), 'started_utc': datetime.now(timezone.utc).isoformat(),
                'git': git, 'source_commit_enforced': spec['source_commit'] is not None, 'environment': environment(),
                'max_seconds': args.max_seconds, 'max_rss_gb': args.max_rss_gb}
    dump(out / 'started.json', {**run_info, 'config': config})
    guard = Guard(args.max_seconds, args.max_rss_gb)
    previous = None
    if args.max_seconds is not None:
        previous = signal.signal(signal.SIGALRM, _raise_hang)
        signal.setitimer(signal.ITIMER_REAL, float(args.max_seconds) + 5.)

    def write(name, value):
        dump(out / name, {**value, 'run': {**run_info, 'wall_seconds_so_far': time.monotonic() - wall,
                                           'peak_rss_bytes': peak_rss_bytes()}})
    try:
        doc, files = execute(spec, mode, verified, manifest, env, guard, write)
    except BaseException as error:
        _stop_alarm(args.max_seconds, previous)
        code = (EXIT['refused'] if isinstance(error, IntegrityError) else
                EXIT['guard'] if isinstance(error, (HangGuardExceeded, MemoryGuardExceeded)) else EXIT['error'])
        dump(out / 'failed.json', {**run_info, 'error_type': type(error).__name__, 'error': str(error),
                                   'traceback': traceback.format_exc(limit=8), 'wall_seconds': time.monotonic() - wall,
                                   'peak_rss_bytes': peak_rss_bytes(), 'citable': False})
        print(f'{type(error).__name__}: {error}', file=sys.stderr)
        if code == EXIT['error'] and not isinstance(error, Exception):
            raise
        return code
    _stop_alarm(args.max_seconds, previous)
    doc['provenance']['run'] = {**run_info, 'wall_seconds': time.monotonic() - wall,
                                'cpu_seconds': time.process_time() - cpu, 'peak_rss_bytes': peak_rss_bytes(),
                                'finished_utc': datetime.now(timezone.utc).isoformat()}
    dump(out / ('results.json' if mode == 'run' else 'check.json'), doc)
    for name, data in files.items():
        write_exclusive(out / name, data)
    sha256sums(out)
    print(f'{mode.upper()}_OK {out}')
    return EXIT['ok']


if __name__ == '__main__':
    sys.exit(main())
