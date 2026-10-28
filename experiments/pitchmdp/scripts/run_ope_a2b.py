"""EXP-OPE-RES-02 (track A2b) stage runner: A2b-SM, A2b-S0, A2b-L, A2b-S3, A2b-G, A2b-S2a, A2b-S1, A2b-S4, A2b-UB.

Every real-data command is guarded from argument parsing to its completion receipt with the EXP-OPE-RES-01 runner
machinery (``run_ope_a2``, imported unchanged): the run-window rule (the whole hang guard ends strictly before the
next 06:30 KST and never enters 06:30-13:30 KST; A2b-L with its 14,400 s guard can start only 13:30-02:30), the
SIGALRM hang guard, the RSS cap (8 GiB) and free-memory guard (>= 4 GiB, a watchdog thread through every code path),
the external append-only digest manifest (hash chain, one writer, atomic receipts), attempt-start receipts (two
attempts per stage), fresh stage directories directly under the pinned output root (disjoint from every input),
AppleDouble-free sealing (SHA256SUMS, manifest.json), the common heavy lock and os.nice(10). Preflight refuses
(fail closed) unless the config passes the A2b registry with status REGISTERED, execution enabled, every pin filled
(the D192 single values asserted), the registration references recorded (the Codex re-confirmation included) and
the gates passed; the digest manifest holds the registered config digest (the A2 rule plus the two A2b null paths)
and the G-CODE / G-REG receipts; the protected sources and the pinned new files are byte-identical and HEAD's code
is the registered (or recorded bug-fix) commit.

Phase order: smoke -> s0 -> ledger -> s3-r3 -> s3 -> gate; every phase-2 command (s2a-gint, s2a, s2a-pool, s1) and
decide require the gate record COMMITTED at HEAD byte for byte (G-GATE) - except the terminal decision: a common stage
(SM, S0, L, S3, G) that exhausted its attempts in a deterministic integrity failure is recorded FAILED_INTEGRITY by
decide without any gate record (the A2 rule; A2b-S3:R3 voids the B2 rows only); s2a runs only after the cell's
G-INT-S2EQ record passed, stops at the cost gate (first run x N' > 2 x the cell projection: decision needed, N'
unchanged) and seals each run; s2a-pool pools only after every registered run of the cell is sealed (operator-blind); unblind opens the
sealed file only after the committed decision record. Every data access goes through an ``Environment`` (the real
one by default; tests subclass it with synthetic inputs). The A2b config is never written by this runner.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import threading
import time
import traceback

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'scripts'))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import run_ope_a2 as roa  # noqa: E402  (the EXP-OPE-RES-01 runner utilities, unchanged)
from pitchmdp import ledger_a2b as lg  # noqa: E402
from pitchmdp import ope_decision_a2b as dc  # noqa: E402
from pitchmdp import ope_denominator_a2b as dn  # noqa: E402
from pitchmdp import ope_redesign_registry as a2  # noqa: E402
from pitchmdp import ope_registry_a2b as rb  # noqa: E402
from pitchmdp import ope_sealed as osd  # noqa: E402
from pitchmdp import ope_sealed_a2b as sa  # noqa: E402
from pitchmdp import ope_surrogate as osg  # noqa: E402
from pitchmdp import ope_synthetic_a2 as osx  # noqa: E402
from pitchmdp import ope_synthetic_a2b as sx  # noqa: E402
from pitchmdp.policy_artifacts import IntegrityError, _require  # noqa: E402

COMMANDS = ('check', 'pins', 'select', 'register-digest', 'record-review', 'register-addendum', 'reconcile-attempt',
            'smoke', 's0', 'ledger', 's3-r3', 's3', 'gate', 's2a-gint', 's2a-gint-record', 's2a', 's2a-pool', 's1',
            'decide', 'unblind')
MANIFEST_COMMANDS = ('register-digest', 'record-review', 'register-addendum', 'reconcile-attempt')
STAGE_OF = {'smoke': 'A2b-SM', 's0': 'A2b-S0', 'ledger': 'A2b-L', 's3-r3': 'A2b-S3', 's3': 'A2b-S3', 'gate': 'A2b-G',
            's2a-gint': 'A2b-S2a', 's2a-gint-record': 'A2b-S2a', 's2a': 'A2b-S2a', 's2a-pool': 'A2b-S2a', 's1': 'A2b-S1',
            'decide': 'A2b-S4', 'unblind': 'A2b-UB',
            'select': 'A2b-S4'}  # 'select' (G-REG helper, reward-free) and manifest commands use the 3,600 s guard
PHASE_2 = ('s2a-gint', 's2a-gint-record', 's2a', 's2a-pool', 's1')
R3_STAGE = 'A2b-S3:R3'  # the R3-only stage (G-INT-R3 and the B2 vectors): its unresolved failure voids the B2 rows only
CODE_PATHS = roa.CODE_PATHS
KST = rb.KST
GIB = roa.GIB


class A2bManifest(roa.DigestManifest):
    KINDS = (*roa.DigestManifest.KINDS, 'cost_gate_stop', 'run_public')


def attempt_stage(args):
    """The manifest stage id of one command: S-2' runs, the G-INT-S2EQ check and the pooling are separate attempts."""
    stage = STAGE_OF[args.command]
    if args.command == 's2a':
        return f'A2b-S2a:k{args.k}'
    if args.command == 's2a-gint':
        return f'A2b-S2a:gint:{args.cell}:k{args.k}'
    if args.command == 's2a-gint-record':
        return f'A2b-S2a:gint:{args.cell}'
    if args.command == 's3-r3':
        return R3_STAGE
    if args.command == 's2a-pool':
        return f'A2b-S2a:pool:{args.cell}'
    return stage


# ---------------------------------------------------------------- guards (the A2 CommandGuard with A2b hang guards)
class CommandGuard(roa.CommandGuard):
    def __init__(self, stage_id, env, entered=None):  # noqa: D401 - the A2 guard with the A2b registry values
        self.stage_id, self.env = stage_id, env
        self.seconds = float(rb.guard_seconds(stage_id.split(':')[0]))
        self.reserve = min(a2.FINALIZATION_RESERVE_SECONDS, self.seconds / 10)
        self.cleanup = min(a2.CLEANUP_RESERVE_SECONDS, self.seconds / 40)
        self.work_seconds = self.seconds - self.reserve
        self.seal_seconds = self.seconds - self.cleanup
        self.start, self.t0 = entered if entered is not None else (env.now(), env.monotonic())
        _require(rb.stage_window_ok(self.start, self.seconds),
                 f'{stage_id}: the command would not end strictly before 06:30 KST within its hang guard, or it '
                 f'overlaps the 06:30-13:30 KST protection window; not started')
        self.memory = env.guards()
        self.armed, self.previous, self.watchdog, self.stop, self.violation = False, {}, None, None, None
        self.lock, self.closed, self.late = threading.Lock(), False, None
        self.deferring, self.pending = 0, None


@contextmanager
def command_guard(stage_id, env, entered=None):
    guard = CommandGuard(stage_id, env, entered)
    roa.LAST_GUARD[:] = [guard]
    guard.arm()
    late = None
    try:
        guard.check()
        yield guard
    finally:
        late = guard.disarm()
    if late is not None:
        raise late


# ---------------------------------------------------------------- registration gate (fail closed, before any data)
def require_config_digest(manifest, config):
    entry = manifest.latest('config_digest')
    _require(entry is not None and entry['payload'].get('digest') == rb.config_digest(config),
             'the registered A2b config digest is missing from the digest manifest or differs (G-REG)')
    return entry


def require_reviews(manifest, config, code_commit):
    digest = rb.config_digest(config)
    receipts = {}
    for gate in rb.REQUIRED_RECEIPTS:
        entry = manifest.latest('review_reference', gate=gate)
        _require(entry is not None, f'no {gate} review receipt in the digest manifest; refusing to run')
        r = entry['payload']
        _require(r.get('verdict') in rb.RECEIPT_VERDICTS and (r.get('verdict') == 'PASS' or r.get('notes_closed') is True),
                 f'{gate} receipt is not a closed pass')
        _require(r.get('config_digest') == digest, f'{gate} receipt names another config digest')
        if gate == 'G-CODE':
            _require(r.get('commit') == code_commit, f'the G-CODE receipt commit is not the running code commit {code_commit}')
        receipts[gate] = r
    return receipts


def new_file_hashes(root=REPO):
    return {rel: rb.file_sha256(Path(root) / rel) for rel in rb.NEW_FILES}


def code_pins(config, manifest=None):
    pins = config['registration_input_pins']
    out = {'execution_source_commit': pins['execution_source_commit'], 'new_files_sha256': pins['new_files_sha256']}
    fix = None if manifest is None else manifest.latest('bugfix_addendum')
    if fix is not None:
        out = {'execution_source_commit': fix['payload']['fix_commit'], 'new_files_sha256': fix['payload']['new_files_sha256']}
    return out


def enforce_sources(config, config_path, state, manifest=None):
    rb.verify_protected_sources(REPO)
    pins = code_pins(config, manifest)
    _require(new_file_hashes() == pins['new_files_sha256'], 'new A2b source/test files differ from their registration pins')
    _require(state['ancestor'] and not state['changed'], 'code at HEAD differs from the registered execution source commit')
    _require(not state['dirty'], 'uncommitted changes under the code paths; refusing to run')
    _require(state['config_committed'], 'the registered config is not committed at HEAD byte for byte')
    return state


def preflight(args, env):
    config, _ = rb.load_config(args.config)
    rb.require_executable(config)
    manifest = A2bManifest(config['registration_input_pins']['digest_manifest_path'])
    require_config_digest(manifest, config)
    code = code_pins(config, manifest)
    state = enforce_sources(config, args.config, env.git_state(args.config, code), manifest=manifest)
    require_reviews(manifest, config, code['execution_source_commit'])
    return config, manifest, state


def identity_record(config, git_commit, extra=None):
    return {'experiment': rb.EXPERIMENT_ID, 'config_digest': rb.config_digest(config), 'git_commit': git_commit,
            'seeds': {'a2b_base': rb.A2B_BASE, 'roles': rb.ROLE_VALUES, 'evaluator': rb.EVALUATOR_SEEDS,
                      'world_seeds': {k: rb.world_seed(k) for c in rb.S2P_CELLS for k in rb.RUN_INDEX[c]}},
            'python': sys.executable, 'python_version': platform.python_version(), 'numpy': np.__version__,
            'pandas': pd.__version__, **(extra or {})}


# ---------------------------------------------------------------- the environment (every data access)
@dataclass
class LedgerWorld:
    rpv: object
    policy: dict
    inputs: dict
    s6_games: list
    no_pitch: frozenset
    game_values: dict
    s6_header: dict = None  # the allowlisted S6-dr-a5 ledger header (runtime_sha256, pins)
    s6_replay_rows: list = None  # the allowlisted S6 rows of the first and the last game in ledger order


class Environment(roa.Environment):
    """The real dependencies of an A2b command (the A2 environment's hooks reused with the A2b pins, which keep the
    EXP-OPE-RES-01 forms of the shared pins); tests subclass it with synthetic inputs."""

    def protected_roots(self, config, local, chain):
        pins = config['registration_input_pins']
        roots = super().protected_roots(config, local, chain)
        roots.update({'the A2 digest manifest directory': Path(pins['a2_digest_manifest_path']).parent,
                      'the A3 attempt directory': Path(pins['a3_freeze_primary_manifest_path']).parent.parent,
                      'the A3 S6 weights directory': Path(pins['a3_s6_weights_record']['path']).parent,
                      'the DEV style snapshot directory': Path(pins['dev_style_snapshot']['path']).parent})
        return roots

    def a2_sealed(self, config, stage_id):
        """A sealed EXP-OPE-RES-01 stage output through its pinned digest manifest (seal re-verified)."""
        pins = config['registration_input_pins']
        path = Path(pins['a2_digest_manifest_path'])
        _require(rb.file_sha256(path) == pins['a2_digest_manifest_sha256'], 'the A2 digest manifest differs from its pin')
        out, entry = roa.require_seal(roa.DigestManifest(path), stage_id)
        expected = {'A2-S0': pins['a2_s0_outputs_manifest_sha256'], 'A2-S2': pins['a2_s2_outputs_manifest_sha256']}
        if stage_id in expected:
            _require(entry['payload']['manifest_sha256'] == expected[stage_id], f'{stage_id} seal differs from its pin')
        return out

    def a2_results(self, config):
        """The committed results/EXP-OPE-RES-01.json (the G-INT-A2 / G-INT-EVAL / G-INT-S2EQ reference)."""
        pin = config['registration_input_pins']['a2_results_file']
        path = REPO / pin['path']
        _require(rb.file_sha256(path) == pin['sha256'] and self.committed(path),
                 'the A2 results file differs from its pin or is not committed at HEAD')
        return json.loads(path.read_text()), {'path': pin['path'], 'sha256': pin['sha256']}

    def a3_s6_weights(self, config):
        pin = config['registration_input_pins']['a3_s6_weights_record']
        raw = Path(pin['path']).read_bytes()
        _require(hashlib.sha256(raw).hexdigest() == pin['sha256'], 'the A3 S6 weights record differs from its pin')
        return json.loads(raw)['arms']['B2']

    def b2_model(self, config, repertoire):
        pins = config['registration_input_pins']
        return dn.load_frozen_b2(pins['a3_freeze_primary_manifest_path'], pins['a3_freeze_primary_manifest_sha256'],
                                 pins['a3_b2_artifacts_sha256'], repertoire,
                                 lightgbm_version=pins['b2_inference_environment']['lightgbm'])

    def b2_dev_table(self, config, local, world, games):
        """The frozen pre-pitch B2 feature table of the given DEV games (``ope_denominator_a2b.dev_feature_table``,
        the A3 construction): the ML-POLICY processed regular-season frame, the raw Statcast keys of the games from the
        pinned 2025 raw file (listed columns, DEV window), the pinned G0 auxiliary's frozen archetypes and the
        registered BC_P with its hand registry. Its fidelity to A3 is what G-INT-R3 checks on S6 (relative 1e-9)."""
        from pitchmdp import policy_identity as pid
        from pitchmdp.behavior_model import RegisteredBC
        from pitchmdp.policy_artifacts import load_hand_registry
        rpv, policy, inputs = world.rpv, world.policy, world.inputs
        window = ('2025-07-01', '2025-09-30')
        pins = config['registration_input_pins']
        frames = []
        for name, path in roa.raw_files_for(window, local):
            _require(rb.file_sha256(path) == pins['train_raw_files_sha256'][name], f'raw file {name} differs from its pin')
            plan = roa.parquet_plan(path, dn.RAW_KEY_COLUMNS, window, strict=False)
            frames.append(roa.read_planned(path, plan, dn.RAW_KEY_COLUMNS, window))
        raw = pd.concat(frames, ignore_index=True)
        aux = pid.pinned_pickle(inputs['paths']['p4_auxiliary'], inputs['files']['p4_auxiliary']['sha256'])
        b0 = RegisteredBC.load(*rpv.registered_path(policy, 'bc'))
        hands, _ = load_hand_registry(*rpv.registered_path(policy, 'hands'), b0.artifact)
        return dn.dev_feature_table(inputs['frame'], raw, games=games, archetypes=aux['context'].archetypes, b0=b0,
                                    hands=hands)

    def ledger_world(self, config, local, base, checkpoint):
        """A2b-L: the ML-POLICY registration (pinned chain), the loaded frame and store, the S6-dr-a5 ledger's game
        set, header and the rows of its first and last game (the driver-equivalence replay) through the allowlisted
        A3 reader (allowlisted fields only; dr.json is never opened) and the frozen C0 game values."""
        from pitchmdp import behavior_weights as bw
        pins = config['registration_input_pins']
        rpv, policy = self.policy_registration(config)
        inputs = self.policy_inputs(config, local, rpv, policy)
        checkpoint()
        ledger = Path(pins['s6_dr_a5_ledger_path'])
        header, decisions, _ = bw.read_ledger(ledger, manifest_path=ledger.parent / 'manifest.json',
                                              ledger_sha256=pins['s6_dr_a5_ledger_sha256'],
                                              manifest_sha256=pins['s6_dr_a5_manifest_sha256'],
                                              runtime_sha256=pins['s6_runtime_sha256_header'],
                                              candidate_sha256=config['identities']['candidate_policy_sha256'])
        replay = lg.replay_games(decisions)
        pcfg = policy['config']
        return LedgerWorld(rpv, policy, inputs, lg.ledger_games(decisions),
                           frozenset(pcfg['pa_time_rules']['R3_codes']['no_pitch_descriptions']), self.game_values(base),
                           s6_header=header, s6_replay_rows=[r for r in decisions if lg.game_of(r) in replay])

    def s6_allowlisted(self, config):
        """The S6 decisions through the allowlisted reader (G-INT-R3: reward-free fields only)."""
        from pitchmdp import behavior_weights as bw
        pins = config['registration_input_pins']
        ledger = Path(pins['s6_dr_a5_ledger_path'])
        return bw.read_ledger(ledger, manifest_path=ledger.parent / 'manifest.json',
                              ledger_sha256=pins['s6_dr_a5_ledger_sha256'], manifest_sha256=pins['s6_dr_a5_manifest_sha256'],
                              runtime_sha256=pins['s6_runtime_sha256_header'],
                              candidate_sha256=config['identities']['candidate_policy_sha256'])[1]

    def raw_join(self, config, local, window, checkpoint):
        return roa.raw_rows(config['registration_input_pins'], local, window, checkpoint=checkpoint)

    def repertoire(self, world):
        from pitchmdp.behavior_model import RegisteredBC
        return RegisteredBC.load(*world.rpv.registered_path(world.policy, 'bc')).repertoire

    def bc_vector(self, world):
        from pitchmdp.behavior_model import VectorBC
        from pitchmdp.policy_artifacts import load_train_bc
        return VectorBC.from_artifact(load_train_bc(*world.rpv.registered_path(world.policy, 'bc')))

    def pyproject_sha256(self):
        """sha256 of this checkout's pyproject.toml (the B2 inference environment pin, A3 record)."""
        return rb.file_sha256(REPO / 'pyproject.toml')


# ---------------------------------------------------------------- commands
@dataclass
class Run:
    args: argparse.Namespace
    env: Environment
    config: dict
    base: dict
    local: dict
    manifest: A2bManifest
    check: object
    plan: object = None
    out: Path = None


def gate_committed(run):
    """G-GATE: the sealed A2b-G gate record, committed at HEAD byte for byte (``--gate-record``)."""
    gate_dir, _ = roa.require_seal(run.manifest, 'A2b-G')
    sealed = (gate_dir / 'gate_record.json').read_bytes()
    path = run.args.gate_record
    _require(path is not None and run.env.committed(path) and Path(path).read_bytes() == sealed,
             'G-GATE: the gate record must be committed at HEAD byte for byte before any phase-2 or decision command')
    return json.loads(sealed)


def plan_phase_2(run):
    gate = gate_committed(run)
    command = run.args.command
    if command in ('s2a-gint', 's2a-gint-record', 's2a', 's2a-pool'):
        cell = run.args.cell if command != 's2a' else rb.registered_run(run.args.k)
        _require(cell in gate['work_list']['S2_prime_cells'], f'the gate record did not open the S-2\' cell {cell}')
        if command in ('s2a-gint', 's2a-gint-record'):
            _require(cell in rb.S2P_DECISION_CELLS, 'G-INT-S2EQ runs for the decision cells only')
        if command == 's2a-gint':
            _require(run.args.k in rb.GINT_S2EQ_RUNS[cell], f'run {run.args.k} is not a recorded run of the {cell} family')
        if command == 's2a-gint-record':  # every reproduced run sealed before the record attempt
            for k in rb.GINT_S2EQ_RUNS[cell]:
                roa.require_seal(run.manifest, f'A2b-S2a:gint:{cell}:k{k}')
        if command == 's2a':
            family = cell if cell in rb.S2P_DECISION_CELLS else '2*kappa_p'  # kappa_p' rides on the kappa family
            out, _ = roa.require_seal(run.manifest, f'A2b-S2a:gint:{family}')
            record = json.loads((out / 'gint_s2eq.json').read_text())
            _require(record['result'] == 'PASS', f'G-INT-S2EQ of {family} did not pass; no new-seed run')
            cost_gate(run, cell)
        if command == 's2a-pool':  # operator-blind: every registered run of the cell sealed before the attempt
            for k in rb.RUN_INDEX[cell]:
                roa.require_seal(run.manifest, f'A2b-S2a:k{k}')
        return {'gate': gate, 'cell': cell}
    _require(gate['work_list']['tab']['rerun'], 'the gate record did not require a TAB re-run')
    return {'gate': gate}


def cost_gate(run, cell):
    """S-2' cost gate (design 9.6): once the cell's first run is sealed, its wall time x N' must stay within 2 x the
    cell projection, else the next run is not started (decision needed; N' unchanged; recorded append-only)."""
    first = rb.RUN_INDEX[cell][0]
    if run.args.k == first:
        return None
    publics = [e for e in run.manifest.entries() if e['kind'] == 'run_public' and e['payload'].get('k') == first]
    _require(publics, f'the cell {cell} runs in order: its first run k = {first} is not sealed yet')
    gate = rb.s2p_cost_gate(cell, publics[-1]['payload']['wall_seconds'])
    if not gate['continue']:
        run.manifest.append('cost_gate_stop', {'stage': f'A2b-S2a:k{run.args.k}', 'cell': cell, **gate})
        _require(False, f'S-2\' cost gate: {gate["projected_seconds"]:.0f} s > {gate["limit_seconds"]:.0f} s for '
                        f'{cell}; decision needed before the next run (N\' unchanged)')
    return gate


COMMON_STAGES = ('A2b-SM', 'A2b-S0', 'A2b-L', 'A2b-S3', 'A2b-G')  # an exhausted integrity failure here is terminal


def _closure(stages, universe, upstream):
    """Every stage of ``universe`` downstream of ``stages`` under ``upstream`` (the A2 ``descendants`` rule)."""
    out, frontier = set(), set(stages)
    while frontier:
        frontier = {s for s in universe if s not in out and set(upstream(s)) & (frontier | out)} - out
        out |= frontier
    return out - set(stages)


def phase_2_stages(gate):
    """The phase-2 attempt stages the committed gate record opened, in run order."""
    out = []
    for cell in gate['work_list']['S2_prime_cells']:
        if cell in rb.S2P_DECISION_CELLS:
            out += [f'A2b-S2a:gint:{cell}:k{k}' for k in rb.GINT_S2EQ_RUNS[cell]] + [f'A2b-S2a:gint:{cell}']
        out += [f'A2b-S2a:k{k}' for k in rb.RUN_INDEX[cell]] + [f'A2b-S2a:pool:{cell}']
    if gate['work_list']['tab']['rerun']:
        out.append('A2b-S1')
    return out


def phase_2_upstream(stage):
    """The registered order inside phase 2: a cell's G-INT-S2EQ record needs its reproduced runs, a new-seed run
    the sealed PASS of its family's record (kappa_p' rides on the kappa family), the pooling every run of its cell."""
    parts = stage.split(':')
    if stage.startswith('A2b-S2a:gint:') and len(parts) == 3:
        return tuple(f'{stage}:k{k}' for k in rb.GINT_S2EQ_RUNS[parts[2]])
    if stage.startswith('A2b-S2a:k'):
        cell = rb.registered_run(int(parts[1][1:]))
        return (f'A2b-S2a:gint:{cell if cell in rb.S2P_DECISION_CELLS else "2*kappa_p"}',)
    if stage.startswith('A2b-S2a:pool:'):
        return tuple(f'A2b-S2a:k{k}' for k in rb.RUN_INDEX[parts[2]])
    return ()


def plan_decide(run):
    """Before the A2b-S4 attempt (a refusal consumes no attempt), the A2 rule (``run_ope_a2.plan_decide``, review
    round 2 #16; G-CODE MAJOR-3): the common stages SM, S0, L, S3:R3, S3 and G are read first, and one of SM, S0, L,
    S3, G that exhausted its attempts in a deterministic integrity failure makes the terminal FAILED_INTEGRITY
    decision available at once WITHOUT a gate record (its unsealed descendants BLOCKED, the other unsealed common
    stages NOT_RUN; phase 2 was never opened). The R3-only stage A2b-S3:R3 is never terminal: its exhausted failure
    voids the B2 rows only (``plan_s3``, flag r3_stage_failed_integrity). Otherwise the committed gate record names
    the phase-2 stages; an exhausted phase-2 failure is terminal as well (its unsealed phase-2 descendants BLOCKED),
    else every stage and every opened cell's pooled record must be sealed."""
    status = roa.stage_status(run.manifest, (*COMMON_STAGES[:3], R3_STAGE, *COMMON_STAGES[3:]))
    failed = [s for s in COMMON_STAGES if status[s] == 'FAILED_INTEGRITY']
    if failed:
        blocked = sorted(s for s in _closure(failed, COMMON_STAGES, lambda s: rb.UPSTREAM[s]) if status[s] != 'sealed')
        not_run = sorted(s for s in COMMON_STAGES if status[s] not in ('sealed', 'FAILED_INTEGRITY') and s not in blocked)
        return {'terminal': 'FAILED_INTEGRITY', 'gate': None, 'stage_status': status, 'failed_integrity': failed,
                'blocked': blocked, 'not_run': not_run,
                'phase_2': 'NOT_OPENED: no gate record (a common stage exhausted its attempts in an integrity failure)'}
    gate = gate_committed(run)
    phase_2 = phase_2_stages(gate)
    status.update(roa.stage_status(run.manifest, phase_2))
    failed = [s for s in phase_2 if status[s] == 'FAILED_INTEGRITY']
    if failed:
        blocked = sorted(s for s in _closure(failed, phase_2, phase_2_upstream) if status[s] != 'sealed')
        not_run = sorted(s for s in phase_2 if status[s] not in ('sealed', 'FAILED_INTEGRITY') and s not in blocked)
        return {'terminal': 'FAILED_INTEGRITY', 'gate': gate, 'stage_status': status, 'failed_integrity': failed,
                'blocked': blocked, 'not_run': not_run, 'phase_2': phase_2}
    pending = {s: v for s, v in status.items() if v != 'sealed' and s != R3_STAGE}  # (plan_s3 gates A2b-S3 on R3)
    _require(not pending, f'stages not complete (decision needed): {pending}')
    for cell in gate['work_list']['S2_prime_cells']:
        roa.require_seal(run.manifest, f'A2b-S2a:pool:{cell}')
    return {'terminal': None, 'gate': gate, 'stage_status': status, 'failed_integrity': [], 'phase_2': phase_2}


def plan_unblind(run):
    """Before the A2b-UB attempt: both seals and the decision record committed at HEAD byte for byte, equal to the
    sealed A2b-S4 record (a refusal consumes no attempt)."""
    s3_dir, _ = roa.require_seal(run.manifest, 'A2b-S3')
    s4_dir, _ = roa.require_seal(run.manifest, 'A2b-S4')
    path = run.args.decision_record
    _require(path is not None and run.env.committed(path), 'the decision record must be committed before unblinding')
    _require(rb.file_sha256(path) == rb.file_sha256(s4_dir / 'decision_record.json'),
             'the committed decision record differs from the sealed A2b-S4 record')
    return {'s3': s3_dir, 's4': s4_dir, 'record': Path(path)}


def plan_s3(run):
    """Before the A2b-S3 attempt: the R3-only stage is sealed, or it ended in an unresolved deterministic integrity
    failure (both attempts used) - then every B2 row is invalid and r3_stage_failed_integrity is recorded; otherwise
    refuse (run s3-r3 first)."""
    status = roa.stage_status(run.manifest, (R3_STAGE,))[R3_STAGE]
    if status == 'sealed':
        out, _ = roa.require_seal(run.manifest, R3_STAGE)
        return {'r3': 'sealed', 'r3_dir': out}
    _require(status == 'FAILED_INTEGRITY', f'the R3-only stage {R3_STAGE} is {status}: run s3-r3 first (decision needed)')
    return {'r3': 'FAILED_INTEGRITY', 'r3_dir': None}


PLANS = {'s2a-gint': plan_phase_2, 's2a-gint-record': plan_phase_2, 's2a': plan_phase_2, 's2a-pool': plan_phase_2,
         's1': plan_phase_2, 'decide': plan_decide, 'unblind': plan_unblind, 's3': plan_s3}


def run_command(args, env, entered=None):
    stage_id = attempt_stage(args)
    with command_guard(stage_id, env, entered) as guard:
        config, manifest, state = preflight(args, env)
        guard.check()
        _require(args.output is not None, '--output (a new directory under the pinned output root) is required')
        chain = env.chain_documents(config)
        local = env.local_config(chain[0])
        protected = env.protected_roots(config, local, chain)
        roa.check_manifest_location(manifest, protected)
        output = roa.check_output(args.output, config, protected)
        env.nice()
        with manifest.writer(), env.heavy_lock(local):
            roa.attempt_gate(manifest, stage_id)
            if stage_id != 'A2b-S4':  # the decision reads the stage status itself (terminal label, G-CODE MAJOR-3)
                for upstream in rb.UPSTREAM[stage_id.split(':')[0]]:
                    roa.require_seal(manifest, upstream)
            run = Run(args, env, config, chain[0], local, manifest, guard.check)
            run.plan = PLANS[args.command](run) if args.command in PLANS else None
            guard.check()
            identity = identity_record(config, state.get('head'), {'command': args.command})
            with roa.stage(output, stage_id, identity, manifest=manifest, final_check=guard.final_check,
                           start=guard.start) as out:
                run.out = out
                BODIES[args.command](run)
    return 0


def sealed_json(run, stage_id, name):
    out, _ = roa.require_seal(run.manifest, stage_id)
    return json.loads((out / name).read_text()), out


# ---------------------------------------------------------------- A2b-SM, A2b-S0
def s0_marginals(s0):
    return roa.marginals_from_doc(s0['tab_marginals'])


def body_smoke(run):
    """TRAIN marginals (sealed A2-S0) and synthetic inputs only: the TAB re-run code path at a few replications on
    the smoke stream (outputs discarded), same-seed bit reproducibility, unit costs and the labelled projections."""
    pins = run.config['registration_input_pins']
    s0 = json.loads((run.env.a2_sealed(run.config, 'A2-S0') / 's0.json').read_text())
    marginals = s0_marginals(s0)
    cell = dict(rb.TAB_ACCEPTANCE[0], index=0, role='acceptance', drift=False)
    t = time.perf_counter()
    world = sx.tab_world(marginals, cell)
    build = time.perf_counter() - t
    reps, walls, first = pins['smoke_tab_replications_per_cell'], [], None
    for j in range(reps):
        run.check()
        t = time.perf_counter()
        est = sx.tab_replication(world, 0, j, n_train=marginals.n_train_bip, n_pas=rb.TAB_DATASET_PAS, stream='smoke')[0]
        walls.append(time.perf_counter() - t)
        first = est if first is None else first
    again = sx.tab_replication(world, 0, 0, n_train=marginals.n_train_bip, n_pas=rb.TAB_DATASET_PAS, stream='smoke')[0]
    _require(json.dumps(first, sort_keys=True) == json.dumps(again, sort_keys=True),
             'smoke TAB replication is not bit-reproducible with the same seed (deterministic integrity)')
    per_rep = float(np.mean(walls))
    projections = {  # labelled projections (planning arithmetic); S-2' and S3 units are the registered records
        'A2b-S1': per_rep * sum(c['R'] for c in rb.TAB_ACCEPTANCE),
        'A2b-S2a': rb.S5X_RUN_SECONDS, 'A2b-S3': 1414.0, 'A2b-L': 3808.0, 'A2b-G': 1.0}
    record = {'stage': 'A2b-SM', 'reads': 'sealed A2-S0 TRAIN marginals only (no DEV, no S6)', 'world_build_seconds': build,
              'replication_seconds': walls, 'integrity': {'tab_v2_dp_expectation_equals_truth': 'PASS',
                                                           'same_seed_bit_reproducible': 'PASS'},
              'projection': {'seconds': projections,
                             'within_guard': {s: rb.smoke_gate(v, s) for s, v in projections.items()},
                             'basis': 'A2b-S1 measured per replication x 8,000; the others the registered projections'},
              'peak_rss_bytes': roa.peak_rss_bytes(),
              'never_computed': ['bias, SD or RMSE', 'acceptance cells', 'selected lambda', 'H_point', 'R5 diagnostics']}
    _require(all(record['projection']['within_guard'].values()),
             f'G-SMOKE: a projected stage cost exceeds its hang guard (decision needed): {record["projection"]}')
    roa.dump(run.out / 'smoke_record.json', record)


def kappa_prime_from_s0(s0):
    """The descriptive kappa_p' (design 5.2, review NOTE N2) = the sealed A2-S0 TAB kappa calibration (G-CODE
    MINOR-1): A2-S0 ran ``calibrate_kappa(hit, groups, delta, names)`` once per world with the SAME hit
    probabilities, groups and cross-fitted residuals (run_ope_a2.py:1355-1357), and the TAB world's names are the four
    large groups FB, FC, BR, OS (``TAB_GROUPS``) - exactly the kappa_p' rule; no TRAIN row is read again."""
    _require(tuple(a2.TAB_GROUPS) == rb.KAPPA_PRIME_GROUPS, 'the TAB groups are not the kappa_p\' groups')
    tab = s0['kappa_calibration']['TAB']
    _require(set(tab) == {'kappa_p', 'target', 'signs', 'groups'} and set(tab['signs']) == set(rb.KAPPA_PRIME_GROUPS)
             and set(tab['groups']) <= set(rb.KAPPA_PRIME_GROUPS) and tab['kappa_p'] == s0['kappa_p']['TAB'],
             'the sealed A2-S0 TAB kappa calibration is malformed (deterministic integrity)')
    return {k: tab[k] for k in ('kappa_p', 'target', 'signs', 'groups')}


def body_s0(run):
    """Input pins (sealed A2 and A3 outputs, the committed A2 results file), r3_status = DIAGNOSTIC_ONLY:DECIDED,
    the reused A2 TAB / net records and eta_p, and the descriptive kappa_p' (the sealed A2-S0 TAB kappa calibration:
    the A2 rule over FB, FC, BR, OS; G-CODE MINOR-1 - sealed outputs and pins only, no TRAIN row)."""
    pins = run.config['registration_input_pins']
    _require(pins['r3_status'] == rb.R3_STATUS_DECIDED, 'r3_status must be DIAGNOSTIC_ONLY:DECIDED (D192 2 3)')
    s0_dir = run.env.a2_sealed(run.config, 'A2-S0')
    s0 = json.loads((s0_dir / 's0.json').read_text())
    _require(osg.table_sha256(np.asarray(s0['r1_table']), s0['m']) == s0['r1_table_sha256'], 'R1 table differs from its seal')
    s1 = json.loads((run.env.a2_sealed(run.config, 'A2-S1') / 'tab.json').read_text())
    _require(s1['eta_p'] == rb.ETA_P, 'the sealed A2-S1 eta_p differs from the registered value')
    s5x = json.loads((run.env.a2_sealed(run.config, 'A2-S2') / 's5x.json').read_text())
    roa.require_s5x_consistent(s5x)
    results, reference = run.env.a2_results(run.config)
    run.check()
    prime = kappa_prime_from_s0(s0)
    record = {'stage': 'A2b-S0', 'r3_status': pins['r3_status'], 'rehearsal_population': pins['rehearsal_population'],
              'a2_results_file': reference, 'eta_p': s1['eta_p'],
              'reused': {'tab_acceptance': s1['acceptance'], 'nets': s5x['nets'],
                         's2_recorded': results['S2_s5x_summary']['dose_response_s2_by_run']['runs']},
              'kappa_p_prime': {**prime, 'kappa_2x': 2 * prime['kappa_p'], 'decision_input': False,
                                'source': 'sealed A2-S0 kappa_calibration.TAB (same inputs, groups FB, FC, BR, OS)'},
              's5x_layers': s0['s5x_layers'], 'tab_marginals': s0['tab_marginals']}
    roa.dump(run.out / 's0_record.json', record)


# ---------------------------------------------------------------- A2b-L
class _Deadline:
    def __init__(self, check):
        self.check, self.start = check, time.monotonic()

    def elapsed(self):
        return time.monotonic() - self.start


def body_ledger(run):
    """The registered runtime pins against the assembly (G-CODE MINOR-3), the selection, the driver equivalence
    BEFORE the new ledger (design 9.2, G-CODE MAJOR-4: written first, then its pass required), the reward-free ledger,
    M-10 and the identity checks; no DR estimate, center or reward written."""
    pins = run.config['registration_input_pins']
    world = run.env.ledger_world(run.config, run.local, run.base, run.check)
    rpv, policy, inputs = world.rpv, world.policy, world.inputs
    pre = rpv.prerequisites('dr-evaluate', policy)
    compared_pins = lg.require_runtime_pins(pins, rpv, policy, pre)
    selection = lg.selection(inputs['frame'], rpv.select_games, rpv.straddling_games, s6_ledger_games=world.s6_games,
                             n_games=lg.N_GAMES, expected=pins['selection'])
    blocks = rpv.game_blocks(inputs['frame'], selection['new_games'], 'dev')
    dates = lg.guard_rehearsal_dates(inputs['frame'], blocks)
    run.check()
    (run.out / 'driver_equivalence').mkdir()
    equivalence = lg.driver_equivalence(rpv, policy, inputs, pre=pre, s6_header=world.s6_header,
                                        s6_rows=world.s6_replay_rows, s6_games=world.s6_games,
                                        new_games=selection['new_games'], out=run.out / 'driver_equivalence',
                                        deadline=_Deadline(run.check), no_pitch=world.no_pitch)
    roa.dump(run.out / 'driver_equivalence.json', equivalence)
    lg.require_equivalence(equivalence)
    run.check()
    runtime, components, provenance = lg.candidate(rpv, policy, inputs, blocks, run.out, pre=pre)
    if equivalence['path'] == 'same_request_twice':  # path B compared the new ledger's runtime identity
        _require(equivalence['runtime_sha256'] == runtime.sha256,
                 'the path-B runtimes are not the new ledger runtime (deterministic integrity)')
    expected = rpv.census_pas(pre['census']['dev'], selection['new_games'])
    facts, sealed = lg.submit(rpv, runtime, inputs['store'], blocks, _Deadline(run.check), no_pitch=world.no_pitch,
                              expected_pas=expected)
    decisions = runtime.ledger.decisions()
    identity = lg.identity_checks(lg.ledger_header(run.out / 'ledger-dev-0.jsonl'), runtime, components, pre, decisions,
                                  run.env.bc_vector(world), candidate_sha256=run.config['identities']['candidate_policy_sha256'])
    from pitchmdp import policy_requests as preq
    games = preq.game_table(inputs['store'].frame, set(selection['new_games']))
    outcome = rpv.outcome_function(inputs['store'], components, games, 'structural-end-v1')
    boot = policy['config']['le2025_validation_plan']['bootstrap']
    kwargs = {'draws': boot['draws'], 'seed': rpv.role_seed(policy['config'], 'bootstrap'),
              'invalid_share_max': boot['invalid_share_max'], 'minimum': boot['minimum'], 'censoring': 'l1r'}
    m10 = lg.m10_check(rpv, lg.pair_runtime(rpv, policy, run.out, runtime.pins['provenance']), inputs['store'], blocks,
                       facts, decisions, no_pitch=world.no_pitch, kwargs=kwargs, outcome=outcome)
    roa.dump(run.out / 'selection.json', selection)
    roa.dump(run.out / 'facts.reward_free.json', lg.reward_free_facts(facts))
    roa.dump(run.out / 'ledger_record.json', {'stage': 'A2b-L', 'sealed_counts': sealed, 'e0_count': lg.e0_count(facts),
                                               'pas': len(facts), 'dates': dates, 'identity': identity, 'm10': m10,
                                               'driver_equivalence': equivalence, 'registered_pins_compared': compared_pins,
                                               'runtime_sha256': runtime.sha256, 'summary': runtime.summary(),
                                               'rewards_written': False})
    lg.check_no_reward_outputs(run.out)


# ---------------------------------------------------------------- A2b-S3
def a2_view_of_original(public, sealed):
    """The original A2 evaluator's outputs in the results-file record layout (G-INT-A2)."""
    pub = {k: v for k, v in json.loads(osd.canonical_bytes(public)).items() if k not in sa.PUBLIC_NOT_COMPARABLE}
    unblinded = osd._jsonable({'selection': sealed['selection'],
                               'pairs': {p: {k: sealed['pairs'][p][k] for k in ('bootstrap', 'center', 'shift_center')
                                             if k in sealed['pairs'][p]} for p in a2.PAIRS}})
    return {'public': pub, 'unblinded': unblinded}


def gint_a2_record(public, sealed, results, reference):
    compared, not_comparable, mismatched, max_delta = rb.compare_documents(a2_view_of_original(public, sealed),
                                                                           sa.a2_record(results))
    return {'check': 'G-INT-A2', 'result': 'FAIL' if mismatched else 'PASS', 'max_abs_delta': float(max_delta),
            'tolerance': dict(rb._TOLERANCE), 'compared_keys': compared, 'not_comparable_keys': not_comparable,
            'mismatched_keys': mismatched, 'reference': reference}


def b0_artifacts(world):
    """The registered BC_P artifact, support table and hand registry (A3 e0_from_ledger inputs)."""
    from pitchmdp.policy_artifacts import load_hand_registry, load_support_table, load_train_bc
    rpv, policy = world.rpv, world.policy
    artifact = load_train_bc(*rpv.registered_path(policy, 'bc'))
    support, _ = load_support_table(*rpv.registered_path(policy, 'support'), artifact)
    hands, _ = load_hand_registry(*rpv.registered_path(policy, 'hands'), artifact)
    return artifact.bc, support, hands


def body_s3_r3(run):
    """The R3-only stage: G-INT-R3 (the frozen B2 on the S6 allowlisted fields reproduces the recorded A3 S6 B2
    diagnostics at relative 1e-9) and the B2 vectors, validity and weight diagnostics on the new ledger (reward-free).
    An unresolved failure here voids only the B2 rows (design 9.6)."""
    from pitchmdp import behavior_weights as bw
    pins = run.config['registration_input_pins']
    settings = pins['a3_weight_settings']
    _require(run.env.pyproject_sha256() == pins['b2_inference_environment']['pyproject_sha256'],
             'pyproject.toml differs from the B2 inference environment pin (A3 record; G-CODE MINOR-3)')
    world = run.env.ledger_world(run.config, run.local, run.base, run.check)
    model = run.env.b2_model(run.config, run.env.repertoire(world))
    bc, support, hands = b0_artifacts(world)
    s6_decisions = run.env.s6_allowlisted(run.config)
    s6_e0 = bw.e0_from_ledger(bw.by_pa(s6_decisions), bc, support, hands)
    s6_table = run.env.b2_dev_table(run.config, run.local, world, lg.ledger_games(s6_decisions))
    s6_vectors = dn.decision_vectors(model, s6_table, dn.b2_decision_rows(s6_decisions, s6_e0))
    s6_weights, _ = dn.b2_weights(s6_decisions, s6_e0, s6_vectors, large=settings['large_weight_threshold'],
                                  mei=settings['mei'], draws=settings['bootstrap_draws'], seed=settings['bootstrap_seed'],
                                  ess_min=settings['E5_thresholds'])
    gint_r3 = dn.gint_r3(s6_weights, run.env.a3_s6_weights(run.config))
    roa.dump(run.out / 'gint_r3.json', gint_r3)
    _require(gint_r3['result'] == 'PASS', f'G-INT-R3 failed (R3-only stage): {gint_r3["mismatched_keys"]}')
    run.check()
    ledger_dir, _ = roa.require_seal(run.manifest, 'A2b-L')
    selection = json.loads((ledger_dir / 'selection.json').read_text())
    facts = json.loads((ledger_dir / 'facts.reward_free.json').read_text())
    decisions = list(run.env.ledger_decisions(ledger_dir / 'ledger-dev-0.jsonl')())
    e0 = dn.e0_map(facts)
    e0_ids = {p for p, (inside, _) in e0.items() if inside}
    table = run.env.b2_dev_table(run.config, run.local, world, selection['new_games'])
    rows = dn.b2_decision_rows(decisions, e0)
    vectors = dn.decision_vectors(model, table, rows)
    validity = dn.b2_validity(decisions, vectors, e0_ids)
    weights = None
    if validity['pass']:
        weights, _ = dn.b2_weights(decisions, e0, vectors, large=settings['large_weight_threshold'], mei=settings['mei'],
                                   draws=rb.DRAWS, seed=rb.ROLE_VALUES['game_bootstrap'], ess_min=settings['E5_thresholds'])
    keys = [[r['pa_id'], r['decision_index']] for r in rows]
    np.save(run.out / 'b2_vectors.npy', np.asarray([vectors[(p, i)] for p, i in keys], dtype=np.float64))
    roa.dump(run.out / 'b2_keys.json', keys)
    roa.dump(run.out / 'b2_record.json', {'stage': R3_STAGE, 'gint_r3': gint_r3, 'validity': validity, 'weights': weights,
                                          'rewards_read': False})


def r3_inputs(run, decisions, e0_ids):
    """The B2 inputs of the 25-row evaluation from the sealed R3-only stage, or (an unresolved R3 failure) an invalid
    B2 denominator with r3_stage_failed_integrity."""
    if run.plan['r3'] != 'sealed':
        zero = {'pass': False, 'decisions_checked': 0, 'zero_mass_with_positive_pi': 0, 'nonfinite_rho': 0,
                'support_differs_from_b0': 0, 'r3_stage_failed_integrity': True}
        return {'validity': zero, 'weights': None}, None, {'result': 'FAILED_INTEGRITY'}
    out = run.plan['r3_dir']
    record = json.loads((out / 'b2_record.json').read_text())
    keys = [tuple(k) for k in json.loads((out / 'b2_keys.json').read_text())]
    vectors = dict(zip(keys, np.load(out / 'b2_vectors.npy', allow_pickle=False)))
    stream = dn.b2_decision_stream(decisions, vectors, e0_ids) if record['validity']['pass'] else None
    return {'validity': record['validity'], 'weights': record['weights']}, stream, record['gint_r3']


def body_s3(run):
    """G-INT-A2 (the original A2 evaluator on S6, A2 seeds), G-INT-EVAL (the new evaluator's 12-row S6 mode, A2
    seeds) and the 25 rows on the new ledger (A2b seeds; the B2 rows from the R3-only stage); centers only in
    centers.sealed.json; the integrity records are written before their pass is required."""
    sealed_json(run, 'A2b-S0', 's0_record.json')  # the pins and r3_status were verified there
    a2_s0 = json.loads((run.env.a2_sealed(run.config, 'A2-S0') / 's0.json').read_text())
    results, reference = run.env.a2_results(run.config)
    raw = lambda window: run.env.raw_join(run.config, run.local, window, run.check)
    from pitchmdp.ope_surrogate import TerminalValues
    # 1. G-INT-A2 on the untouched A2 path (its own subdirectory for the facts' reference ledger)
    ledger_path, s6_dr = run.env.s6_record(run.config)
    (run.out / 's6').mkdir()
    data = run.env.rehearsal_inputs(run.config, run.local, run.base, ledger_path, s6_dr, run.out / 's6', run.check)
    _require(data.window[1] <= a2.ALL_STAGES_MAX_DATE, 'a row after 2025 reached the rehearsal')
    references = {'s6_dr_a5_record': osd.recorded_gint_reference(s6_dr)}  # the A2 G-INT values (then the A2 record)
    public_a2, sealed_a2, _, gint = roa.rehearsal(
        data.decisions, data.facts, data.last_rows, raw_join=lambda: raw(data.window), game_values=data.game_values,
        s0=a2_s0, references=references, ess_gate=data.ess_gate, r3_status='NOT_RUN:R3', checkpoint=run.check)
    gint_a2 = gint_a2_record(public_a2, sealed_a2, results, reference)
    roa.dump(run.out / 'gint_a2.json', {k: v for k, v in gint_a2.items()})
    sa.require_gint_pass(gint_a2, 'G-INT-A2')
    del public_a2, sealed_a2
    run.check()
    # 2. G-INT-EVAL: the new evaluator, S6 mode, A2 seeds, the 12 A2 pair rows only
    terminal = TerminalValues(data.game_values['we'], data.game_values['advancement'])
    surrogates = osg.surrogate_rewards(data.facts, roa.surrogate_rows(data.last_rows, raw(data.window)), terminal,
                                       np.asarray(a2_s0['r1_table']))
    common = {'train_residuals': a2_s0.get('train_residuals'), 'ess_gate': data.ess_gate, 'checkpoint': run.check}
    public6, sealed6, _ = sa.evaluate(roa.guarded_stream(data.decisions, run.check), data.facts, mode=rb.MODE_S6,
                                      surrogates=surrogates, gint={'G-INT-A2': gint_a2['result'], 'G-INT': gint},
                                      r1_paths={'rehearsal': osg.path_shares(surrogates), 'train_census': a2_s0['census']},
                                      **common)
    gint_eval = sa.gint_eval_record(public6, sealed6, results, reference=reference)
    roa.dump(run.out / 'gint_eval.json', gint_eval)
    sa.require_gint_pass(gint_eval, 'G-INT-EVAL')
    del public6, sealed6, data
    run.check()
    # 3. the 25 rows on the new ledger (A2b seeds)
    ledger_dir, _ = roa.require_seal(run.manifest, 'A2b-L')
    selection = json.loads((ledger_dir / 'selection.json').read_text())
    written = json.loads((ledger_dir / 'facts.reward_free.json').read_text())
    (run.out / 'new').mkdir()
    new = run.env.rehearsal_inputs(run.config, run.local, run.base, ledger_dir / 'ledger-dev-0.jsonl',
                                   {'games': selection['new_games']}, run.out / 'new', run.check)
    _require({p: bool(i['in_population']) for p, i in new.facts.items()} ==
             {p: bool(i['in_population']) for p, i in written.items()},
             'the recomputed E0 differs from the A2b-L ledger facts (deterministic integrity)')
    new_sur = osg.surrogate_rewards(new.facts, roa.surrogate_rows(new.last_rows, raw(new.window)), terminal,
                                    np.asarray(a2_s0['r1_table']))
    decisions = list(new.decisions())
    e0_ids = {p for p, info in new.facts.items() if info['in_population']}
    b2, stream, gint_r3 = r3_inputs(run, decisions, e0_ids)
    public, sealed, notes = sa.evaluate(
        roa.guarded_stream(lambda: iter(decisions), run.check), new.facts, mode=rb.MODE_REHEARSAL, surrogates=new_sur,
        gint={'G-INT-A2': gint_a2['result'], 'G-INT-R3': gint_r3, 'G-INT-EVAL': gint_eval['result']},
        decisions_b2=stream, b2=b2, r1_paths={'rehearsal': osg.path_shares(new_sur), 'train_census': a2_s0['census']},
        **common)
    public_bytes, sealed_bytes = sa.seal(public, sealed)
    (run.out / 'public.json').write_bytes(public_bytes)
    (run.out / 'centers.sealed.json').write_bytes(sealed_bytes)
    roa.dump(run.out / 'notes.json', notes)


# ---------------------------------------------------------------- A2b-G, phase 2, A2b-S4, A2b-UB
def body_gate(run):
    public_dir, _ = roa.require_seal(run.manifest, 'A2b-S3')
    raw = (public_dir / 'public.json').read_bytes()
    ledger_record, _ = sealed_json(run, 'A2b-L', 'ledger_record.json')
    record = dc.gate_record(json.loads(raw), e0_count=int(ledger_record['e0_count']),
                            r3_status=run.config['registration_input_pins']['r3_status'],
                            public_sha256=hashlib.sha256(raw).hexdigest())
    (run.out / 'gate_record.json').write_bytes(dc.gate_bytes(record))


def s2_world(run):
    s0_record, _ = sealed_json(run, 'A2b-S0', 's0_record.json')
    world = run.env.s5x_inputs(run.config, run.local, run.base, run.out, run.check)
    return s0_record, world


def s2_layers(components, world, s0_record, kappa, zeta, *, prime=False):
    doc = s0_record['s5x_layers']
    signs = list(doc['signs'])
    if prime:  # kappa_p' tilts only the four large groups (its calibration signs; 'other' untilted)
        signs = [s0_record['kappa_p_prime']['signs'].get(g, 0) if g in rb.KAPPA_PRIME_GROUPS else 0
                 for g in doc['groups']]
    return osx.S5XLayers.from_components(components, world.game_values, groups=doc['groups'], x_dist=doc['x_dist'],
                                         class_table=doc['class_table'], signs=signs, signs_prime=doc['signs_prime'],
                                         kappa=kappa, zeta=zeta)


def _v2_law(runtime):
    return (lambda bc: (lambda s: (bc.actions, bc.probabilities(s))))(runtime.bc)


def one_s2_run(run, s0_record, world, k, scheme):
    """One S-2' run on the registered X1-V2 world (147 starts, cap 12, planning seed 0), A2 or A2b seeds."""
    if scheme == 'A2':
        spec = next(r for r in a2.S5X_RUNS if r['k'] == k)
        kappa_label, zeta_label = spec['kappa'], spec['zeta']
        cell = '2*kappa_p' if zeta_label == '0' else '2*zeta_p'
        prime = False
    else:
        cell = rb.registered_run(k)
        kappa_label, zeta_label = rb.S2P_CELL_LEVELS[cell]
        prime = cell == 'descriptive_kappa_prime'
    doc = s0_record['s5x_layers']
    kappa = 2 * s0_record['kappa_p_prime']['kappa_p'] if prime else a2.scale(kappa_label, doc['kappa_p'])
    zeta = a2.scale(zeta_label, doc['zeta_p'])
    runtime, components, starts = world.candidate({'k': k, 'planning_seed': rb.S2P_WORLD['planning_seed']})
    layers = s2_layers(components, world, s0_record, kappa, zeta, prime=prime)
    return sx.run_s2_prime(runtime, components, starts, k=k, scheme=scheme, cell=cell, law=_v2_law(runtime), layers=layers,
                           logs_per_start=rb.S2P_WORLD['logs_per_start'], truth_rollouts=rb.S2P_WORLD['truth_rollouts'],
                           cap=rb.S2P_WORLD['cap'], deadline=run.check)


def body_s2a_gint(run):
    """One recorded EXP-OPE-RES-01 dose-family run reproduced by the new driver in the A2 seed scheme (its own
    attempt and 1,800 s guard; the full synthetic run record is sealed for the record step)."""
    s0_record, world = s2_world(run)
    roa.dump(run.out / 'run.json', one_s2_run(run, s0_record, world, run.args.k, 'A2'))


def body_s2a_gint_record(run):
    """G-INT-S2EQ (public item 14): written first, then its exact form required (a failure is a deterministic
    integrity failure; no new-seed run of the family starts without a sealed PASS)."""
    cell = run.plan['cell']
    results, reference = run.env.a2_results(run.config)
    runs = {}
    for k in rb.GINT_S2EQ_RUNS[cell]:
        out, _ = roa.require_seal(run.manifest, f'A2b-S2a:gint:{cell}:k{k}')
        runs[k] = json.loads((out / 'run.json').read_text())
    sealed = sx.sealed_runs(run.env.a2_sealed(run.config, 'A2-S2'), rb.GINT_S2EQ_RUNS[cell])
    record = sx.gint_s2eq_record([cell], runs, sx.recorded_runs(results), reference=reference, sealed=sealed)
    roa.dump(run.out / 'gint_s2eq.json', record)
    sx.gint_s2eq_scope(record)


def body_s2a(run):
    s0_record, world = s2_world(run)
    t0 = time.perf_counter()
    result = one_s2_run(run, s0_record, world, run.args.k, 'A2b')
    record = sx.run_record(result)
    data, sha = sx.seal_run(record)
    (run.out / 'run.sealed.json').write_bytes(data)
    public = sx.run_public(record, sha, wall_seconds=time.perf_counter() - t0, peak_rss_bytes=roa.peak_rss_bytes())
    roa.dump(run.out / 'run_public.json', public)
    run.manifest.append('run_public', {'stage': f'A2b-S2a:k{run.args.k}', **public})


def body_s2a_pool(run):
    cell = run.plan['cell']
    sealed = {}
    for k in rb.RUN_INDEX[cell]:
        out, _ = roa.require_seal(run.manifest, f'A2b-S2a:k{k}')  # every run sealed first (operator-blind)
        data = (out / 'run.sealed.json').read_bytes()
        sealed[k] = (data, hashlib.sha256(data).hexdigest())
    roa.dump(run.out / 'pooled.json', sx.pool_cell(cell, sealed))


def body_s1(run):
    s0_record, _ = sealed_json(run, 'A2b-S0', 's0_record.json')
    gate = run.plan['gate']
    marginals = roa.marginals_from_doc(s0_record['tab_marginals'])
    cell_dir = run.out / 'cells'
    cell_dir.mkdir()

    def write_cell(cell, record):
        roa.dump(cell_dir / f"{cell['id']}.json", record)
        run.manifest.append('cell_seal', {'stage': 'A2b-S1', 'cell': cell['id'], 'status': 'COMPLETED',
                                          'sha256': rb.file_sha256(cell_dir / f"{cell['id']}.json")})
    cells = sx.tab_rerun(marginals, n_pas=gate['work_list']['tab']['rerun_dataset_pas'], n_train=marginals.n_train_bip,
                         eta_p=s0_record['eta_p'], deadline=run.check, write_cell=write_cell)
    roa.dump(run.out / 'tab_rerun.json', {'acceptance': {cid: rec['pairs'] for cid, rec in cells.items()}, 'cells': cells})


def body_decide(run):
    plan = run.plan
    gate, status = plan['gate'], plan['stage_status']
    r3 = run.config['registration_input_pins']['r3_status']
    if plan['terminal'] == 'FAILED_INTEGRITY':  # design 8.2 label 1 (the A2 terminal record)
        roa.dump(run.out / 'decision_record.json', {
            'experiment': rb.EXPERIMENT_ID, 'label': 'FAILED_INTEGRITY', 'r3_status': r3,
            **{k: plan[k] for k in ('stage_status', 'failed_integrity', 'blocked', 'not_run', 'phase_2')},
            'gate_record_sha256': None if gate is None else dc.gate_sha256(gate)})
        return
    public_dir, _ = roa.require_seal(run.manifest, 'A2b-S3')
    public = json.loads((public_dir / 'public.json').read_text())
    s0_record, _ = sealed_json(run, 'A2b-S0', 's0_record.json')
    tab = s0_record['reused']['tab_acceptance']
    if gate['work_list']['tab']['rerun']:
        rerun, _ = sealed_json(run, 'A2b-S1', 'tab_rerun.json')
        tab = rerun['acceptance']
    s2 = {}
    for cell in gate['work_list']['S2_prime_cells']:
        s2[cell], _ = sealed_json(run, f'A2b-S2a:pool:{cell}', 'pooled.json')
    dc.check_inputs(tab, s0_record['reused']['nets'], s2, gate)
    r3_failed = status[R3_STAGE] == 'FAILED_INTEGRITY'  # B2 rows void only (never terminal)
    record = dc.require_label(dc.decide(public, gate, tab=tab, nets=s0_record['reused']['nets'], s2_prime=s2,
                                        r3_status=r3, r3_stage_failed_integrity=r3_failed))
    record['inputs_sha256'] = {'public': rb.file_sha256(public_dir / 'public.json'), 'gate': dc.gate_sha256(gate)}
    roa.dump(run.out / 'decision_record.json', record)


def body_unblind(run):
    """Open the sealed file after the committed decision record: the selection, every row's sealed record and the
    lambda = 100 row's public-form items, sealed until now (G-CODE MAJOR-1; with the descriptive flag whether they
    equal the R1_b x R2_1 row, i.e. whether lambda = 100 was selected - never a decision input)."""
    s3_dir, s4_dir, record_path = run.plan['s3'], run.plan['s4'], run.plan['record']
    sealed = osd.unblind((s3_dir / 'centers.sealed.json').read_bytes(),
                         json.loads((s3_dir / 'public.json').read_text())['sealed_sha256'],
                         Path(record_path).read_bytes(), rb.file_sha256(s4_dir / 'decision_record.json'), committed=True)
    fixed = sealed.get('lambda_fixed_row_public')
    _require(isinstance(fixed, dict) and fixed.get('row') == rb.LAMBDA_FIXED_ROW,
             'the sealed file lacks the lambda-fixed row items (stage integrity)')
    roa.dump(run.out / 'unblinded.json', {'selection': sealed.get('selection'),
                                          'pairs': {r: (sealed['pairs'].get(r) or {}) for r in rb.ROWS},
                                          'lambda_fixed_row_public': fixed,
                                          'decision_record_sha256': rb.file_sha256(s4_dir / 'decision_record.json')})


BODIES = {'smoke': body_smoke, 's0': body_s0, 'ledger': body_ledger, 's3-r3': body_s3_r3, 's3': body_s3,
          'gate': body_gate, 's2a-gint': body_s2a_gint, 's2a-gint-record': body_s2a_gint_record, 's2a': body_s2a,
          's2a-pool': body_s2a_pool, 's1': body_s1, 'decide': body_decide, 'unblind': body_unblind}


# ---------------------------------------------------------------- manifest commands and the G-REG selection helper
def manifest_command(args, env, entered=None):
    with command_guard('A2b-S4', env, entered) as guard:
        config, _ = rb.load_config(args.config)
        rb.require_executable(config)
        manifest = A2bManifest(config['registration_input_pins']['digest_manifest_path'])
        chain = env.chain_documents(config)
        roa.check_manifest_location(manifest, env.protected_roots(config, env.local_config(chain[0]), chain))
        guard.check()
        with manifest.writer():
            return _manifest_mutation(args, config, manifest)


def _manifest_mutation(args, config, manifest):
    digest = rb.config_digest(config)
    if args.command == 'register-digest':
        _require(args.expect_digest == digest, 'registered config digest differs from the G-CODE reviewed digest')
        _require(manifest.latest('config_digest') is None, 'the config digest is registered once')
        print(json.dumps(manifest.append('config_digest', {'digest': digest, 'config_path': str(args.config)}), indent=1))
        return 0
    require_config_digest(manifest, config)
    if args.command == 'record-review':
        _require(args.gate in ('G-REV', 'G-CODE', 'G-REG') and bool(args.review) and args.verdict in rb.RECEIPT_VERDICTS,
                 '--gate, --review and --verdict are required')
        _require(args.gate != 'G-CODE' or (isinstance(args.commit, str) and len(args.commit) == 40),
                 'a G-CODE receipt names the reviewed code commit (--commit, 40 hex)')
        payload = {'gate': args.gate, 'reference': args.review, 'verdict': args.verdict,
                   'notes_closed': bool(args.notes_closed), 'commit': args.commit, 'config_digest': digest}
        print(json.dumps(manifest.append('review_reference', payload), indent=1))
        return 0
    if args.command == 'reconcile-attempt':
        _require(bool(args.stage) and bool(args.note), '--stage and --note are required')
        start = roa.unfinished_start(manifest, args.stage)
        _require(start is not None, f'{args.stage} has no attempt start without a receipt; nothing to reconcile')
        out = Path(start['payload']['output_dir'])
        if out.is_dir():
            roa.dump(out / 'ABORTED_UNRECORDED.json', {'stage': args.stage, 'status': 'ABORTED_UNRECORDED',
                                                       'citable': False, 'attempt_start_seq': start['seq'], 'note': args.note})
        print(json.dumps(manifest.append('attempt_reconciliation', {
            'stage': args.stage, 'attempt_start_seq': start['seq'], 'output_dir': start['payload']['output_dir'],
            'status': 'ABORTED_UNRECORDED', 'note': args.note, 'counts_as_attempt': True}), indent=1))
        return 0
    _require(bool(args.stage) and args.failing_test and args.review, '--stage, --failing-test and --review required')
    _require(any(e['kind'] == 'stage_failure' and e['payload']['stage'] == args.stage
                 and e['payload']['class'] == 'deterministic_integrity' for e in manifest.entries()),
             'a bug-fix addendum follows a recorded deterministic-integrity failure of that stage')
    _require(not roa.git('status', '--porcelain', '--', *CODE_PATHS), 'commit the fix first (clean code paths)')
    rb.verify_protected_sources(REPO)
    entry = manifest.append('bugfix_addendum', {'stage': args.stage, 'fix_commit': roa.git('rev-parse', 'HEAD'),
                                                'new_files_sha256': new_file_hashes(), 'failing_test': args.failing_test,
                                                'review_reference': args.review})
    print(json.dumps(entry, indent=1))
    return 0


def select_command(args, env, entered=None):
    """G-REG helper (reward-free): the selection record (both lists' sha256 and the month quota table) from the
    registered rule on game_pk / game_date / split, printed for the PM to pin.

    The one command that may run on a DRAFT config (deliberate exception, design 9.6 / G-CODE MINOR-2): the selection
    pins it prints are inputs of the registration itself, so it cannot wait for REGISTERED. It still runs under the
    run window, the hang and memory guards, os.nice(10) and the common heavy lock (the data load - the ML-POLICY frame
    and store, the S6-dr-a5 ledger through the allowlisted reader - happens inside the lock), reads no reward, writes
    no stage directory and appends nothing to the digest manifest."""
    with command_guard('A2b-S4', env, entered) as guard:
        config, _ = rb.load_config(args.config)
        chain = env.chain_documents(config)
        local = env.local_config(chain[0])
        guard.check()
        env.nice()
        with env.heavy_lock(local):
            world = env.ledger_world(config, local, chain[0], guard.check)
            record = lg.selection(world.inputs['frame'], world.rpv.select_games, world.rpv.straddling_games,
                                  s6_ledger_games=world.s6_games, n_games=lg.N_GAMES)
            print(json.dumps({k: record[k] for k in ('selection_s6_game_list_sha256', 'a2b_new_game_list_sha256',
                                                     'month_quota_table', 'straddling_games_excluded', 'salts')},
                             indent=1))
        return 0


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=COMMANDS)
    parser.add_argument('--config', type=Path, default=REPO / rb.CONFIG_PATH)
    parser.add_argument('--local-config', type=Path, default=PROJECT / 'configs' / 'local.json')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--expect-digest')
    parser.add_argument('--decision-record', type=Path)
    parser.add_argument('--gate-record', type=Path)
    parser.add_argument('--cell', choices=rb.S2P_CELLS)
    parser.add_argument('--k', type=int)
    parser.add_argument('--stage')
    parser.add_argument('--failing-test')
    parser.add_argument('--review')
    parser.add_argument('--gate')
    parser.add_argument('--verdict')
    parser.add_argument('--notes-closed', action='store_true')
    parser.add_argument('--commit')
    parser.add_argument('--note')
    args = parser.parse_args(argv)
    if args.command == 's2a':
        _require(type(args.k) is int, 's2a needs --k (a registered run index)')
        rb.registered_run(args.k)
    if args.command in ('s2a-gint', 's2a-gint-record', 's2a-pool'):
        _require(args.cell is not None, f'{args.command} needs --cell')
    if args.command == 's2a-gint':
        _require(type(args.k) is int and args.cell in rb.GINT_S2EQ_RUNS and args.k in rb.GINT_S2EQ_RUNS[args.cell],
                 's2a-gint needs --cell (a decision cell) and --k (one of its recorded A2 runs)')
    return args


def main(argv=None, env=None):
    env = env if env is not None else Environment()
    entered = (env.now(), env.monotonic())
    args = parse(argv)
    env.args = args
    if args.command == 'check':
        config, sha = rb.load_config(args.config)
        print(json.dumps({'status': config['status'], 'config_file_sha256': sha, 'config_digest': rb.config_digest(config),
                          'protected_sources': 'unchanged' if rb.verify_protected_sources(REPO) else None}, indent=1))
        return 0
    if args.command == 'pins':
        print(json.dumps({'new_files_sha256': new_file_hashes(), 'protected_sources': rb.PROTECTED_SOURCES,
                          'registration_input_pins_expected_keys': list(rb.PIN_KEYS),
                          'added_to_the_draft_pin_keys': list(rb.ADDED_PIN_KEYS),
                          'pins_compared_at_run': rb.COMPARED_AT_RUN, 'record_only_pins': rb.RECORD_ONLY_PINS},
                         indent=1))
        return 0
    if args.command == 'select':
        return select_command(args, env, entered)
    if args.command in MANIFEST_COMMANDS:
        return manifest_command(args, env, entered)
    return run_command(args, env, entered)


def cli(argv=None, env=None):
    """The bounded command-line entry of the A2 runner (final report bounded, os._exit)."""
    try:
        code, message = main(argv, env) or 0, None
    except SystemExit as error:
        code, message = roa.exit_code(error.code)
    except BaseException as error:  # noqa: BLE001 - every failure exits nonzero
        code = 3 if isinstance(error, (roa.HangGuardExceeded, roa.MemoryGuardExceeded)) else 1
        message = (f'run_ope_a2b: {type(error).__name__}: {error}\n'
                   f'{traceback.format_exc()[-a2.TRACEBACK_TAIL_CHARS:]}').rstrip('\n')
    os._exit(roa.finish(code, message))


if __name__ == '__main__':
    sys.exit(cli())
