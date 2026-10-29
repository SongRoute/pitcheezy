"""EXP-OPE-1STEP-01 runner (spec v3): one-step dose-response OPE on a sealed decision-ledger stage (side experiment).

    .venv/bin/python experiments/pitchmdp/scripts/run_ope_onestep.py --config CONFIG.json --check
    .venv/bin/python experiments/pitchmdp/scripts/run_ope_onestep.py --config CONFIG.json --run

Design: docs/pm/research/side-onestep-ope-proposal-2026-10-29.md + SPEC DELTA v2 + SPEC DELTA 2; estimator:
pitchmdp/ope_onestep.py; config template: scripts/run_ope_onestep.config-template.json.

Order (fail closed; IntegrityError = refusal before the output exists (exit 2), study-level INVALID after it):
1. the config is validated (unknown or missing keys refused); the output ``<output_dir>/<check|run>`` must be a NEW
   directory outside the stage directory; with ``source_commit`` the code at HEAD must be that commit's code;
2. every pinned stage file (ledger, pa_values.parquet, dr.json, manifest.json, started.json) is hashed and compared
   with its pin BEFORE any of them is parsed; the stage manifest must list the others with the same sha256; the ledger
   is hashed again while it is streamed; the ledger header must exist and pin a P3 candidate at the sealed tau with
   its sha256 and the registered DR q-hat source;
3. PA facts (``frame``, the only source): every ML-POLICY registration link, the local config, the processed frame and
   the WE contract are verified against their pins before they are read; the registered PA-end reward and WE_start
   are recomputed through the registered loaders (run_policy_validation.load_inputs with store=False,
   policy_identity.load_pinned_we, select_games / game_blocks, policy_requests.game_table / pa_outcome / pa_manifest)
   and the sealed pa_values rows are reproduced PA by PA (policy_estimator.pa_value re-run; columns pa_id, game,
   in_population, status, kind, reason, end_kind, resolution, candidate, reference, delta, l2): phase (a) with the
   reward value masked before prediag, phase (b) with the real reward after prediag.json and before any estimate
   (a second verified ledger pass), plus the aggregate L2 reward of dr.json. No ``derived`` source;
4. one streamed ledger pass (G-INT, D, populations), prediag.json (G-free: gate per family x tau x contrast) FIRST;
   ``--check`` writes check.json; ``--run`` computes every G-dependent section, the registered ``decide`` step
   (results.json ``decision``), per_tau.csv; SHA256SUMS last.
Guards: the registered hang guard (seconds) and peak-RSS guard (GiB via ru_maxrss) from the config (CLI values may only
tighten them); a fired guard leaves failed.json (exit 3); the run refuses to start when the available memory (vm_stat
free + inactive pages x page size) is below the registered ``min_available_gb``. The deterministic row identity and
the tau = inf identity (study-level ESTIMATOR_SUSPECT) run in both modes before prediag.json.
Every run records wall/CPU seconds, peak RSS, seeds, the git commit
and the python/numpy/pandas versions.
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
REQUIRED = ('experiment_id', 'stage_label', 'stage_dir', 'ledger', 'pins', 'output_dir', 'tau_grid', 'sealed_tau',
            'identity_tolerance', 'ratio_rtol', 'expected_dr_q_source', 'candidate_identity_sha256', 'families',
            'primary', 'confirmatory', 'disagreement', 'eligibility', 'gate', 'sample_minima', 'placebos',
            'sensitivities', 'bootstrap', 'decision', 'guards', 'pa_facts')
OPTIONAL = ('exposure_label', 'source_commit', 'outputs', 'notes', 'interpretations', 'recorded_not_read')
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


def _resolve(value):
    """A registered path: absolute, or relative to this checkout (as run_policy_validation resolves registered paths)."""
    return Path(value) if Path(value).is_absolute() else REPO / value


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
    for key in ('identity_tolerance', 'ratio_rtol'):
        _require(_number(config[key]) and 0 <= config[key] < 1e-6, f'{key} must be a number in [0, 1e-6)')
    _require(isinstance(config['expected_dr_q_source'], str) and config['expected_dr_q_source'],
             'expected_dr_q_source required (e.g. "evaluation_seed")')
    identity = config['candidate_identity_sha256']
    _require(_sha(identity), 'candidate_identity_sha256 must be the registered candidate sha256 (null is refused)')
    _require(config['stage_label'] in ope.STAGE_LABELS, f'stage_label must be one of {ope.STAGE_LABELS}')
    families = config['families']
    _require(isinstance(families, list) and families and families[0] == 'temperature'
             and set(families) <= set(ope.MAIN_FAMILIES) and len(set(families)) == len(families),
             f'families: "temperature" first, values of {ope.MAIN_FAMILIES}')
    primary = config['primary']
    _keys(primary, 'primary', ('family', 'tau', 'subset', 'variant'))
    _require(primary['family'] == 'temperature' and primary['subset'] == 'all' and primary['variant'] == 'dr'
             and ope.parse_tau(primary['tau']) == sealed_tau, 'primary: temperature, dr, subset all, the sealed tau')
    _require(config['confirmatory'] == ope.CONFIRMATORY, f'confirmatory family must be {ope.CONFIRMATORY}')
    dis = config['disagreement']
    _keys(dis, 'disagreement', ('q', 'ties', 'margins', 'confident'))
    margins = dis['margins']
    _require(dis['q'] == ope.Q_PI and dis['ties'] == 'not_in_D' and isinstance(margins, list) and margins
             and margins[0] == 0 and '1se' in margins and all(m == '1se' or (_number(m) and m >= 0) for m in margins)
             and len(set(map(str, margins))) == len(margins) and dis['confident'] == '1se',
             'disagreement: q "q_planning", ties "not_in_D", margins [0, ..., "1se"], confident "1se"')
    elig = config['eligibility']
    _keys(elig, 'eligibility', ('statuses', 'population', 'require_reward'))
    _require(isinstance(elig['statuses'], list) and sorted(elig['statuses']) == sorted(ope.RESULT_STATUSES),
             f'eligibility.statuses must be exactly the registered set {list(ope.RESULT_STATUSES)}')
    _require(elig['population'] == 'E0' and elig['require_reward'] is True,
             'eligibility: population "E0" and require_reward true (whole PAs without a reward are excluded)')
    gate = config['gate']
    _keys(gate, 'gate', ('ess_ratio_min', 'max_weight_share', 'games_min', 'pitchers_min', 'noise_q90_max', 'noise_exempt'))
    _require(all(_number(gate[k]) and gate[k] >= 0 for k in ('ess_ratio_min', 'max_weight_share', 'games_min',
                                                               'pitchers_min', 'noise_q90_max')),
             'gate thresholds must be numbers >= 0')
    _require(isinstance(gate['noise_exempt'], list) and all(
        isinstance(e, dict) and set(e) == {'family', 'tau'} and e['family'] in ope.MAIN_FAMILIES
        and ope.parse_tau(e['tau']) in taus for e in gate['noise_exempt']), 'gate.noise_exempt: [{family, tau}]')
    minima = config['sample_minima']
    _keys(minima, 'sample_minima', ('games', 'pas', 'pitchers'))
    _require(all(_int(v, 1) for v in minima.values()), 'sample_minima: positive ints')
    plac = config['placebos']
    _keys(plac, 'placebos', ('perm_draws', 'perm_seed', 'kl_tolerance', 'tau_bracket_factor', 'bisection_iterations',
                             'outcome_shuffle_null_draws', 'outcome_shuffle_null_seed', 'neg'))
    _require(_int(plac['perm_draws']) and _int(plac['perm_seed']) and _int(plac['outcome_shuffle_null_draws'])
             and _int(plac['outcome_shuffle_null_seed']) and _int(plac['bisection_iterations'], 1)
             and isinstance(plac['neg'], bool) and _number(plac['kl_tolerance']) and 0 < plac['kl_tolerance'] < 1
             and _number(plac['tau_bracket_factor']) and plac['tau_bracket_factor'] > 1,
             'placebos: nonnegative int draws/seeds, iterations >= 1, kl_tolerance in (0, 1), bracket factor > 1')
    sens = config['sensitivities']
    _keys(sens, 'sensitivities', ('switch_caps', 's_pos', 'cross_replicate'))
    _require(isinstance(sens['switch_caps'], list) and all(_number(c) and c > 0 for c in sens['switch_caps'])
             and len(set(sens['switch_caps'])) == len(sens['switch_caps']) and sens['s_pos'] is True
             and isinstance(sens['cross_replicate'], bool),
             'sensitivities: positive switch caps, s_pos true (the decision reads it), bool cross_replicate')
    boot = config['bootstrap']
    _keys(boot, 'bootstrap', ('draws', 'seeds', 'invalid_share_max'))
    _require(_int(boot['draws'], 2), 'bootstrap.draws: int >= 2')
    _keys(boot['seeds'], 'bootstrap.seeds', ope.UNITS, ('trend',))
    _require(all(_int(v) for v in boot['seeds'].values()), 'bootstrap seeds must be nonnegative ints')
    _require(_number(boot['invalid_share_max']) and 0 <= boot['invalid_share_max'] <= 1, 'bootstrap.invalid_share_max')
    decision = config['decision']
    _keys(decision, 'decision', ('alpha_one_sided', 'mde_multiplier', 'mde_multiplier_holm_first'))
    _require(_number(decision['alpha_one_sided']) and 0 < decision['alpha_one_sided'] < .5
             and all(_number(decision[k]) and decision[k] > 0 for k in ('mde_multiplier', 'mde_multiplier_holm_first')),
             'decision: alpha in (0, .5), positive MDE multipliers')
    guards = config['guards']
    _keys(guards, 'guards', ('max_seconds', 'max_rss_gb', 'min_available_gb'))
    _require(all(_number(v) and v > 0 for v in guards.values()),
             'guards: positive max_seconds, max_rss_gb (GiB, ru_maxrss) and min_available_gb (vm_stat free + inactive)')
    source = config.get('source_commit')
    _require(_sha(source, 40), 'source_commit must be the registered 40-hex code commit (null is refused)')
    unread = config.get('recorded_not_read', {})
    _require(isinstance(unread, dict) and all(isinstance(v, dict) and _sha(v.get('sha256')) for v in unread.values()),
             'recorded_not_read: {name: {sha256, ...}} (pins recorded in provenance, never read)')
    _require(config.get('interpretations', ope.INTERPRETATIONS) == ope.INTERPRETATIONS,
             'interpretations differ from the ones this code implements (ope_onestep.INTERPRETATIONS)')
    outputs = config.get('outputs', {'decisions_parquet': False})
    _keys(outputs, 'outputs', ('decisions_parquet',))
    _require(isinstance(outputs['decisions_parquet'], bool), 'outputs.decisions_parquet: bool')
    facts = config['pa_facts']
    _keys(facts, 'pa_facts', ('source', 'split', 'policy_registration', 'local_config', 'processed_frame',
                              'we_contract_sha256'), ('value_tolerance', 'check_selection'))
    _require(facts['source'] == 'frame', 'pa_facts.source: "frame" is the only PA-facts source (no exact "derived" identity)')
    _require(isinstance(facts['split'], str) and facts['split'], 'pa_facts.split required')
    registration = facts['policy_registration']
    _keys(registration, 'pa_facts.policy_registration', ('config', 'addenda'))
    _require(isinstance(registration['addenda'], list), 'pa_facts.policy_registration.addenda must be a list')
    _require(_sha(facts['we_contract_sha256']), 'pa_facts.we_contract_sha256 must be a sha256')
    value_tol = facts.get('value_tolerance', 1e-12)
    _require(_number(value_tol) and 0 <= value_tol < 1e-6, 'pa_facts.value_tolerance in [0, 1e-6)')
    _require(facts.get('check_selection', True) is True, 'pa_facts.check_selection must be true (the registered rule)')
    scan_families = [*families, *(['cross_replicate'] if sens['cross_replicate'] else []), *(['neg'] if plac['neg'] else [])]
    return {
        'experiment_id': config['experiment_id'], 'exposure_label': config.get('exposure_label'),
        'stage_label': config['stage_label'], 'recorded_not_read': dict(unread),
        'stage_dir': Path(config['stage_dir']), 'ledger': config['ledger'], 'pins': dict(pins),
        'output_dir': Path(config['output_dir']), 'taus': taus, 'sealed_tau': sealed_tau,
        'identity_tolerance': float(config['identity_tolerance']), 'ratio_rtol': float(config['ratio_rtol']),
        'expected_dr_q_source': config['expected_dr_q_source'], 'candidate_identity_sha256': identity,
        'main_families': list(families), 'scan_families': scan_families, 'margins': list(margins),
        'statuses': list(elig['statuses']), 'gate': {**gate, 'noise_exempt': [dict(e) for e in gate['noise_exempt']]},
        'minima': dict(minima), 'variants': [*BASE_VARIANTS, *[f'switch:{c:g}' for c in sens['switch_caps']]],
        'sensitivities': dict(sens), 'placebos': {**plac, 'kl_tolerance': float(plac['kl_tolerance'])},
        'bootstrap': {'draws': boot['draws'], 'seeds': dict(boot['seeds'])},
        'invalid_share_max': float(boot['invalid_share_max']), 'alpha': float(decision['alpha_one_sided']),
        'mde_multipliers': {'one_sided_alpha': float(decision['mde_multiplier']),
                            'holm_first_step': float(decision['mde_multiplier_holm_first'])},
        'guards': dict(guards), 'source_commit': source,
        'outputs': outputs,
        'pa_facts': {'source': 'frame', 'split': facts['split'],
                     'policy_registration': {'config': _pin(registration['config'], 'policy_registration.config'),
                                             'addenda': [_pin(a, 'policy_registration.addenda[]')
                                                         for a in registration['addenda']]},
                     'local_config': _pin(facts['local_config'], 'pa_facts.local_config'),
                     'processed_frame': _pin(facts['processed_frame'], 'pa_facts.processed_frame'),
                     'we_contract_sha256': facts['we_contract_sha256'], 'value_tolerance': float(value_tol),
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
        state = {'commit': run('rev-parse', 'HEAD'), 'branch': run('rev-parse', '--abbrev-ref', 'HEAD'),
                 'worktree': run('rev-parse', '--show-toplevel'), 'code_paths': list(CODE_PATHS),
                 'code_dirty': bool(run('status', '--porcelain', '--', *CODE_PATHS)), 'source_commit': source_commit,
                 'source_is_ancestor': None, 'code_changed_since_source': None}
        if source_commit:
            state['source_is_ancestor'] = subprocess.call(['git', 'merge-base', '--is-ancestor', source_commit, 'HEAD'],
                                                          cwd=REPO, stderr=subprocess.DEVNULL) == 0
            state['code_changed_since_source'] = [x for x in run('diff', '--name-only', source_commit, 'HEAD', '--',
                                                                 *CODE_PATHS).splitlines() if x]
        return state

    MEMORY_FORMULA = '(Pages free + Pages inactive + Pages speculative) x page size (vm_stat)'

    def available_memory_bytes(self):
        """MEMORY_FORMULA on macOS (the registration's wording); MemAvailable elsewhere."""
        if sys.platform == 'darwin':
            return parse_vm_stat(subprocess.check_output(['vm_stat'], text=True))
        with open('/proc/meminfo') as stream:
            info = {line.split(':')[0]: int(line.split()[1]) for line in stream}
        return info['MemAvailable'] * 1024

    def load_frame_and_we(self, spec, checkpoint):
        """(frame, defense_we, ML-POLICY config, provenance) through the registered loaders. Every ML-POLICY chain link,
        the local config, the processed frame (path, quality manifest and bytes; A2 verify_processed_pin) and the WE
        contract are verified against their pins before they are read or decoded."""
        import run_policy_validation as rpv
        from pitchmdp import policy_identity as pid
        links = [spec['policy_registration']['config'], *spec['policy_registration']['addenda']]
        for link in links:
            _require(file_sha256(_resolve(link['path'])) == link['sha256'],
                     f'ML-POLICY registration file differs from its pin: {link["path"]}')
        reg = rpv.load_registration(_resolve(links[0]['path']), [_resolve(link['path']) for link in links[1:]])
        _require([link['sha256'] for link in reg['chain']] == [link['sha256'] for link in links],
                 'the loaded ML-POLICY chain differs from its pins')
        config = reg['config']
        _require(config.get('protocol') == rpv.PROTOCOL, 'not an ML-POLICY materialization/validation config')
        local_pin = spec['local_config']
        raw = _resolve(local_pin['path']).read_bytes()
        local_sha = hashlib.sha256(raw).hexdigest()
        _require(local_sha == local_pin['sha256'] == config['le2025_validation_plan']['local_config']['sha256'],
                 'local config differs from its pins')
        local = json.loads(raw)
        frame_pin = spec['processed_frame']
        root = Path(local['artifact_root'])
        _require(Path(frame_pin['path']).resolve() == (root / 'processed' / 'pitches.parquet').resolve(),
                 'processed_frame path differs from the registered processed frame of the local config')
        quality = json.loads((root / 'reports' / 'data_quality.json').read_text())
        _require(quality.get('processed_sha256') == frame_pin['sha256'],
                 'the processed-frame pin differs from the quality manifest (processed_sha256)')
        _require(file_sha256(frame_pin['path']) == frame_pin['sha256'], 'processed frame bytes differ from their pin')
        ident = config['identity_registration']
        _require(ident['we_contract_sha256'] == spec['we_contract_sha256'],
                 'the registered WE contract pin differs from the ML-POLICY identity registration')
        checkpoint()
        inputs = rpv.load_inputs(config, local, store=False)  # the registered frame: same loader, order and date guard
        checkpoint()
        values, we_identity = pid.load_pinned_we(inputs['paths'], spec['we_contract_sha256'], ident['classes']['we'])
        we = values['we']

        def defense(state, defender_is_home):  # = policy_identity.BoundComponents.defense_we
            return float(we.predict_defense(state, defender_is_home))
        provenance = {'policy_registration_chain': reg['chain'], 'policy_config_sha256': reg['config_sha256'],
                      'local_config_sha256': local_sha, 'processed_frame': dict(frame_pin),
                      'g0_bundle_file_sha256': inputs['bundle_sha'], 'we_identity': we_identity,
                      'frame_rows': int(len(inputs['frame'])),
                      'loaders': ['run_policy_validation.load_inputs(store=False)', 'policy_identity.load_pinned_we']}
        return inputs['frame'], defense, config, provenance


def parse_vm_stat(text):
    """Bytes of (free + inactive + speculative) pages from ``vm_stat`` output."""
    page = int(text.split('page size of', 1)[1].split('bytes', 1)[0])
    pages = {line.split(':')[0].strip(): int(line.split(':')[1].strip().rstrip('.'))
             for line in text.splitlines()[1:] if ':' in line}
    return (pages['Pages free'] + pages['Pages inactive'] + pages.get('Pages speculative', 0)) * page


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
    table = pd.read_parquet(io.BytesIO(verified['pa_values.parquet']), columns=list(ope.PHASE_A_COLUMNS))
    sealed = ope.sealed_rows(table.to_dict('records'), ope.PHASE_A_COLUMNS)
    del table
    _require(list(sealed) == list(facts), 'pa_values PAs differ from the frame blocks of the stage games (ids or order)')
    pas = {pa_id: {**facts[pa_id], 'in_population': sealed[pa_id]['in_population'], 'pa_status': sealed[pa_id]['status']}
           for pa_id in sealed}
    scan = ope.Scan(taus=spec['taus'], sealed_tau=spec['sealed_tau'], identity_tolerance=spec['identity_tolerance'],
                    ratio_rtol=spec['ratio_rtol'], statuses=spec['statuses'], families=spec['scan_families'],
                    margins=spec['margins'], expected_q_source=spec['expected_dr_q_source'],
                    runtime_sha256=head['runtime_sha256'])
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
                  'interpretations': ope.INTERPRETATIONS, 'stage_label': spec['stage_label'],
                  'recorded_not_read': spec['recorded_not_read'],
                  'identity_check': records['identity_check'], 'row_identity': records['row_identity'],
                  'se_ties': records['se_ties'], 'q_source': records['q_source'],
                  'vocabulary_size': records['vocabulary_size'],
                  'seeds': {'bootstrap': spec['bootstrap']['seeds'], 'perm_placebo': spec['placebos']['perm_seed'],
                            'outcome_shuffle_null': spec['placebos']['outcome_shuffle_null_seed']}}
    counts = {'by_status': records['by_status'], **records['counts'], 'disagreement': records['disagreement']}
    guard.checkpoint()
    diag = ope.prediag(tab, spec['gate'])
    p = spec['placebos']
    setup = ope.perm_setup(tab, draws_k=p['perm_draws'], seed=p['perm_seed'], kl_tol=p['kl_tolerance'],
                           factor=p['tau_bracket_factor'], iterations=p['bisection_iterations'],
                           checkpoint=guard.checkpoint)
    diag['perm_placebo_setup'] = {'tau_p': setup.record, 'row_identity': setup.identity, 'draws': len(setup.perms)}
    write('prediag.json', {'provenance': provenance, 'counts': counts, 'prediag': diag,
                           'note': 'written before any G-dependent number is computed'})
    doc = {'provenance': provenance, 'counts': counts, 'prediag': diag}
    files = {}
    if mode == 'run':
        guard.checkpoint()  # phase (b): the sealed rows reproduced with the real reward, before any estimate
        values = pd.read_parquet(io.BytesIO(verified['pa_values.parquet']), columns=list(ope.PHASE_B_COLUMNS))
        ope.add_value_columns(sealed, ope.sealed_rows(values.to_dict('records'), ope.PHASE_B_COLUMNS))
        del values
        ope.attach_rewards(tab, pas)
        second = ope.LedgerStream(ledger_path)
        ope.reproduce(second.decisions(), pas, sealed, check, guard.checkpoint)
        _require(second.sha256 == spec['pins'][spec['ledger']], 'ledger bytes changed between the two passes')
        provenance['pa_facts']['crosscheck'] = check.record(dr)
        guard.checkpoint()
        rule = {'primary_tau': spec['sealed_tau'], 'variants': spec['variants'], 'main_families': spec['main_families'],
                'margins': spec['margins'], 'bootstrap': spec['bootstrap'], 'invalid_share_max': spec['invalid_share_max'],
                'minima': spec['minima'], 'placebos': spec['placebos'], 'sensitivities': spec['sensitivities'],
                'alpha': spec['alpha'], 'mde_multipliers': spec['mde_multipliers'], 'stage_label': spec['stage_label']}
        sections, csv_sections, inputs = ope.analysis(tab, diag, rule=rule, perm=setup, checkpoint=guard.checkpoint)
        doc.update(sections)
        doc['decision'] = {'inputs': inputs, **decide(inputs), 'interpretations': ope.INTERPRETATIONS}
        doc['reading'] = ('stage 1 is EXPLORATORY_REANALYSIS on exposed <=2025 DEV: the decision is the registered '
                          'stage-2 entry rule, not an adoption label; non-evaluable rows are descriptive')
        files['per_tau.csv'] = per_tau_csv(ope.per_tau_rows(csv_sections))
        if spec['outputs']['decisions_parquet']:
            files['decisions.parquet'] = decisions_parquet(tab)
    return doc, files


def decide(inputs):
    """The runner's registered decide step (SPEC DELTA 2 item 9): ope_onestep.decide on the decision inputs."""
    return ope.decide(inputs)


def per_tau_csv(rows):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]) if rows else ['section'], lineterminator='\n')
    writer.writeheader()
    for row in rows:
        writer.writerow({k: '' if v is None else (repr(v) if isinstance(v, float) else v) for k, v in row.items()})
    return buffer.getvalue().encode()


def decisions_parquet(tab):
    """Per estimation decision of the temperature family: ids, count, D, status, M_dr, M_pi, C, D_t, rho' per tau."""
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
        frame[f'M_dr_{key}'] = tab['model_dr'][obs, f, j]
        frame[f'M_pi_{key}'] = tab['model_pi'][obs, f, j]
        frame[f'C_{key}'] = corr[:, j]
        frame[f'delta_{key}'] = tab['model_dr'][obs, f, j] + corr[:, j]
        frame[f'rho_{key}'] = tab['rho'][obs, f, j]
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    return buffer.getvalue()


def write_exclusive(path, data):
    with Path(path).open('xb') as stream:
        stream.write(data)


def encode(value):
    return (json.dumps(ope.jsonable(value), indent=1, ensure_ascii=False, allow_nan=False) + '\n').encode()


def dump(path, value):
    write_exclusive(path, encode(value))


def sha256sums(out):
    names = sorted(p.name for p in out.iterdir() if p.is_file() and p.name != 'SHA256SUMS' and not is_appledouble(p)
                   and not p.name.endswith('.partial'))
    write_exclusive(out / 'SHA256SUMS', ''.join(f'{file_sha256(out / n)}  {n}\n' for n in names).encode())


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true', help='pins, counts, identity, facts and prediag; no G-dependent value')
    mode.add_argument('--run', action='store_true', help='the full registered computation')
    parser.add_argument('--max-seconds', type=float, default=None,
                        help='tighten the registered hang guard (wall seconds); never loosens it')
    parser.add_argument('--max-rss-gb', type=float, default=None,
                        help='tighten the registered peak-RSS guard (GiB); never loosens it')
    parser.add_argument('--pa-facts-source', choices=('frame',), default=None,
                        help='PA facts source; must equal the config (frame is the only source)')
    args = parser.parse_args(argv)
    for name in ('max_seconds', 'max_rss_gb'):
        value = getattr(args, name)
        if value is not None and not (math.isfinite(value) and value > 0):
            parser.error(f'--{name.replace("_", "-")} must be positive')
    return args


def _stop_alarm(max_seconds, previous):
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
        _require(git['source_is_ancestor'] is True and not git['code_changed_since_source'] and not git['code_dirty'],
                 'code at HEAD differs from the registered source commit (or uncommitted code changes)')
        verified, manifest = verify_stage(spec)
        available = env.available_memory_bytes()
        _require(available >= spec['guards']['min_available_gb'] * 2 ** 30,
                 f'available memory {available / 2 ** 30:.2f} GiB below the registered start condition '
                 f'{spec["guards"]["min_available_gb"]} GiB ({Environment.MEMORY_FORMULA})')
    except (ValueError, OSError) as error:  # IntegrityError is a ValueError
        print(f'REFUSED: {error}', file=sys.stderr)
        return EXIT['refused']
    out.mkdir(parents=True)
    max_seconds = min(spec['guards']['max_seconds'], args.max_seconds or math.inf)
    max_rss_gb = min(spec['guards']['max_rss_gb'], args.max_rss_gb or math.inf)
    run_info = {'mode': mode, 'argv': list(sys.argv if argv is None else argv), 'config_path': str(args.config),
                'config_sha256': hashlib.sha256(raw).hexdigest(), 'started_utc': datetime.now(timezone.utc).isoformat(),
                'git': git, 'source_commit_enforced': True, 'environment': environment(),
                'guards': {'max_seconds': max_seconds, 'max_rss_gb': max_rss_gb, 'registered': spec['guards'],
                           'available_memory_bytes_at_start': available,
                           'available_memory_formula': Environment.MEMORY_FORMULA}}
    dump(out / 'started.json', {**run_info, 'config': config})
    guard = Guard(max_seconds, max_rss_gb)
    previous = signal.signal(signal.SIGALRM, _raise_hang)
    signal.setitimer(signal.ITIMER_REAL, float(max_seconds) + 5.)

    def write(name, value):
        dump(out / name, {**value, 'run': {**run_info, 'wall_seconds_so_far': time.monotonic() - wall,
                                           'peak_rss_bytes': peak_rss_bytes()}})
    writing = False
    try:
        doc, files = execute(spec, mode, verified, manifest, env, guard, write)
        _stop_alarm(max_seconds, previous)
        writing = True
        doc['provenance']['run'] = {**run_info, 'wall_seconds': time.monotonic() - wall,
                                    'cpu_seconds': time.process_time() - cpu, 'peak_rss_bytes': peak_rss_bytes(),
                                    'finished_utc': datetime.now(timezone.utc).isoformat()}
        final = {('results.json' if mode == 'run' else 'check.json'): encode(doc), **files}
        for name, data in final.items():  # complete files first, renamed only when every one is written
            write_exclusive(out / f'{name}.partial', data)
        for name in final:
            (out / f'{name}.partial').rename(out / name)
        sha256sums(out)
    except BaseException as error:
        _stop_alarm(max_seconds, previous)
        code = (EXIT['refused'] if writing or isinstance(error, (IntegrityError, OSError)) else
                EXIT['guard'] if isinstance(error, (HangGuardExceeded, MemoryGuardExceeded)) else EXIT['error'])
        dump(out / 'failed.json', {**run_info, 'error_type': type(error).__name__, 'error': str(error),
                                   'study_label': ('ESTIMATOR_SUSPECT' if isinstance(error, ope.EstimatorSuspect) else
                                                   'INVALID' if isinstance(error, IntegrityError) else None),
                                   'traceback': traceback.format_exc(limit=8), 'wall_seconds': time.monotonic() - wall,
                                   'peak_rss_bytes': peak_rss_bytes(), 'citable': False})
        print(f'{type(error).__name__}: {error}', file=sys.stderr)
        if not isinstance(error, Exception) and not isinstance(error, (HangGuardExceeded, MemoryGuardExceeded)):
            raise
        return code
    print(f'{mode.upper()}_OK {out}')
    return EXIT['ok']


if __name__ == '__main__':
    sys.exit(main())
