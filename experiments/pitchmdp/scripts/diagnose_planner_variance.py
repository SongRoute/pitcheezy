"""EXP-STAB-02 (track A1): planner variance removal by a Rao-Blackwellized first step.

Pass runner of the registered design (docs/pm/research/track-a/A1-planner-variance-design.md):

  smoke         first registered step: the 3 EXP-STAB-01 smoke games (FH-DP: the first game of the
                subsample order). Cost of every arm (rows, planner and runtime-submit time, process peak RSS,
                r_b) and structural integrity only (shapes, finite Q on the support, laws summing to 1, RB6
                seed-0 rerun identical, hashes). Its runtime ledger lives in a temporary directory removed when
                the smoke ends; the smoke sink persists only allowlisted cost/structure/identity fields.
  M (mandatory) A0-MC6 (seed 0 = the runtime's own planning Q, seed 5), RB6 (seeds 0, 5), REF-RB6 (seed
                100) on the 150 registered DEV games, joined with the EXP-STAB-01 K1/K3/K5 columns;
                reconstruction against first-step evidence captured during the original A0 searches;
                metrics and the RB6 status; the verdict unless RB12 is required.
  C             RB12 (seeds 0, 5) only if smoke r_b <= 2.0 and RB6 is not STABLE; enters the verdict.
  D             BF48 (50-game subsample) and FH-DP (10-game subsample), descriptive only, each only if RB6
                is not (STABLE and COST_OK) and its smoke projection fits its cap.
  confirmation  50 new DEV games, A0 and the passing arm, only after STABLE_WITHIN_COST and Song's
                confirmation, against the appended games pin.

Every stage refuses unless the config is REGISTERED with execute true (``registration_gate``), its digest
equals the external digest manifest entry, the registered commit contains the reviewed sources, the
registered planner path is byte-identical to its pins and the new code to its registration pins. Every
pass writes JSON + SHA256SUMS into <attempt>/<pass folder> outside the repository (never results/),
appends receipts to <attempt>/receipts.jsonl and to the digest manifest, starts only if its cap fits
before 06:00 KST (else it is deferred and the heavy lock released), runs its data preparation and its
finalization under the deadline/cap/RSS watch, and stops on the registered stop rules (aborted.json,
nothing citable, exit 3). No reward, WE result, events kind or next-row state is read; no policy value is
estimated; nothing is registered as a policy identity.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import contextlib
from dataclasses import dataclass
import json
from pathlib import Path
import re
import resource
import subprocess
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'scripts'))
import numpy as np  # noqa: E402
from pitchmdp import planner_variance as pv  # noqa: E402
from pitchmdp import planner_variance_registry as reg  # noqa: E402
from pitchmdp.policy_runtime import EVALUATED, paired_difference_se  # noqa: E402
from pitchmdp.rollout_policy import RowBudget  # noqa: E402

EXPERIMENT = reg.EXPERIMENT_ID
EXIT_OK, EXIT_REFUSED, EXIT_STOPPED, EXIT_DEFERRED = 0, 2, 3, 4
SOURCES = ('scripts/diagnose_planner_variance.py', 'pitchmdp/planner_variance.py', 'pitchmdp/planner_variance_registry.py',
           'scripts/diagnose_policy_seed_ensemble.py', 'scripts/diagnose_policy_predictor_swap.py',
           'scripts/run_policy_validation.py', 'pitchmdp/policy_runtime.py', 'pitchmdp/policy_identity.py',
           'pitchmdp/policy_requests.py', 'pitchmdp/policy_artifacts.py', 'pitchmdp/rollout_policy.py',
           'pitchmdp/planner.py', 'pitchmdp/matrix_policy.py', 'pitchmdp/matrix_models.py',
           'pitchmdp/matrix_sharing.py')
ATTEMPT_NAME = re.compile(r'attempt-\d{3}')
TRY_NAME = re.compile(r'try-\d{2}')  # confirmation retries under the same profile (bug-fix retry rule)
SELECTION_FILE = 'confirmation-selection.json'
KINDS = {'RolloutImprovement': 'MC', 'RaoBlackwellFirstStep': 'RB', 'FrozenHistoryDP': 'DP'}
SEEDS = {'planning_main': reg.MAIN_SEED, 'pair': list(reg.PAIR), 'reference': reg.REFERENCE_SEED,
         'stab01_sets': reg.STAB01_SEEDS, 'bootstrap': {'draws': reg.BOOTSTRAP['draws'], 'seed': reg.BOOTSTRAP['seed']},
         'rule': 'planning_seed(state, support, seed) per decision; FH-DP has no seed'}
FH_DP_BUDGET = sys.maxsize  # no registered FH-DP row budget: bounded by its 7,200 s cap
LIMITS = ['Stability of the planning law only; no policy value, OPE or recommendation-quality estimate',
          'DEV July-September 2025 is exposed development data (no outcome read here)',
          '0.95, 0.02, 4x, the gap bins, budgets and tolerances are judgement values (D186 approved the three '
          'thresholds); costs are Mac mini CPU measurements, not service latency guarantees',
          'peak_rss_bytes_process_lifetime is ru_maxrss: the peak of the whole process, so a later pass in the '
          'same process inherits the peak of an earlier one',
          'RB / BF48 / FH-DP planners are diagnostics, not a registered policy identity (c13cc989 unchanged)']


class IntegrityFailure(ValueError):
    """A registered integrity check failed: stop the pass."""


def require(condition, message):
    if not condition:
        raise IntegrityFailure(message)


class DeadlinePassed(RuntimeError):
    pass


class CapExceeded(RuntimeError):
    pass


class RSSExceeded(RuntimeError):
    pass


def _stab():
    """EXP-STAB-01 runner (registered, unchanged): compare(), stab_games(), and through it swap and rpv."""
    import diagnose_policy_seed_ensemble as stab
    return stab


def peak_rss():
    """ru_maxrss: the peak resident set of the whole process so far (bytes on macOS)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def git_state(repo=REPO):
    """(HEAD commit, dirty) of the repository; missing Git provenance refuses."""
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=repo, capture_output=True, text=True,
                                check=True).stdout.strip()
        dirty = bool(subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=repo,
                                    capture_output=True, text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError) as missing:
        raise reg.RegistrationError(f'Git provenance unavailable: {missing}') from None
    if not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise reg.RegistrationError('Git provenance unavailable: no HEAD commit')
    return commit, dirty


def plain(value):
    """JSON-safe copy: numpy scalars/arrays to Python, non-finite floats to None."""
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return [plain(v) for v in value.tolist()]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(plain(payload), indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def append_line(path, payload):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(plain(payload), sort_keys=True, allow_nan=False) + '\n')


def sha256_file(path):
    return reg.file_sha256(path)


def write_sums(folder):
    folder = Path(folder)
    files = sorted(p for p in folder.rglob('*') if p.is_file() and p.name != 'SHA256SUMS' and not p.name.endswith('.tmp'))
    lines = [f'{sha256_file(p)}  {p.relative_to(folder).as_posix()}' for p in files]
    (folder / 'SHA256SUMS').write_text('\n'.join(lines) + '\n')
    return sha256_file(folder / 'SHA256SUMS')


def verify_sums(folder):
    """Every file of a finished pass folder is exactly the one its SHA256SUMS lists."""
    folder = Path(folder)
    sums = folder / 'SHA256SUMS'
    require(sums.is_file(), f'{folder.name}: SHA256SUMS missing')
    listed = {}
    for line in sums.read_text().splitlines():
        digest, _, name = line.partition('  ')
        listed[name] = digest
    present = {p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file() and p.name != 'SHA256SUMS'}
    require(set(listed) == present, f'{folder.name}: files differ from SHA256SUMS')
    for name, digest in listed.items():
        require(sha256_file(folder / name) == digest, f'{folder.name}/{name}: sha256 differs from SHA256SUMS')
    return sha256_file(sums)


def subsample_order(games):
    """Registered subsample order: the 150 games sorted by canonical_hash([20261027, game_pk])."""
    from pitchmdp.matrix_data import canonical_hash
    return sorted((int(g) for g in games), key=lambda g: canonical_hash([reg.SUBSAMPLES['salt'], g]))


def assert_attempt_path(attempt, repo=REPO):
    """Outputs never go inside the repository (results/ included); attempt folders are attempt-NNN."""
    path = Path(attempt).resolve()
    if path.is_relative_to(Path(repo).resolve()):
        raise reg.RegistrationError(f'output {path} is inside the repository (results/ included): refused')
    if not ATTEMPT_NAME.fullmatch(path.name):
        raise reg.RegistrationError('attempt folder must be named attempt-NNN')
    return path


@dataclass
class Decision:
    game_pk: int
    request_id: str
    state: object
    pitcher_hand: object
    mask: np.ndarray
    reference: np.ndarray
    runtime: dict | None = None


class SystemClock:
    def now(self):
        return reg.utc_now()

    def monotonic(self):
        return time.perf_counter()


class Watch:
    """Deadline, pass cap and sampled-RSS stop rules: before the data preparation, after it, before every PA
    block, and once more immediately before a pass publishes its results."""
    def __init__(self, clock, deadline, cap_seconds, rss):
        self.clock, self.deadline, self.cap, self.rss = clock, deadline, float(cap_seconds), rss
        self.start, self.peak = clock.monotonic(), 0

    def check(self, where, *, sample_rss=False):
        if self.clock.now() >= self.deadline:
            raise DeadlinePassed(f'deadline {self.deadline.isoformat()} passed {where}')
        if self.clock.monotonic() - self.start > self.cap:
            raise CapExceeded(f'pass cap {self.cap:.0f} s exceeded {where}')
        if sample_rss:
            self.peak = max(self.peak, int(self.rss()))
            if self.peak > reg.COST['rss_cap_bytes']:
                raise RSSExceeded(f'process peak RSS {self.peak} B > {reg.COST["rss_cap_bytes"]} B {where}')

    def block(self, index):
        self.check(f'at PA block {index}', sample_rss=index % reg.COST['rss_sample_every_pa_blocks'] == 0)

    def final_check(self):
        self.check('at the final check before publishing', sample_rss=True)

    def seconds(self):
        return self.clock.monotonic() - self.start


def search(planner, state):
    """(q, diagnostics, conditional rows, planner ms) of one planner call on its own row budget."""
    budget = planner.simulator.budget
    before = budget.conditional_rows
    started = time.perf_counter()
    q, diagnostics = planner.q_values(state)
    return q, diagnostics, budget.conditional_rows - before, (time.perf_counter() - started) * 1000.


def checked_law(q, decision, tau, label):
    """Finite Q on the support and a law summing to 1 (every pass); returns the law."""
    q = np.asarray(q, dtype=np.float64)
    require(q.shape == decision.mask.shape, f'{label}: Q shape differs from the vocabulary')
    require(np.isfinite(q[decision.mask]).all(), f'{label}: non-finite planning Q on the support')
    p = pv.law(q, decision.reference, decision.mask, tau)
    require(abs(p.sum() - 1) <= reg.INTEGRITY['law_sum_atol'], f'{label}: law does not sum to 1')
    return p


def vector(values):
    return np.array([np.nan if v is None else v for v in values], dtype=np.float64)


class MemoryLedger:
    """Smoke-only stand-in for ``policy_runtime.Ledger`` with the same interface and the same hash-chained rows
    (canonical header, seq, prev, sha256) kept in process memory: no file is ever opened, so the smoke runtime's
    A0 Q and laws never reach the disk. Offsets are row indices."""
    def __init__(self, path, header):
        from pitchmdp.matrix_data import canonical_hash
        from pitchmdp.policy_artifacts import _canonical_bytes
        self._hash, self.path = canonical_hash, None
        self.count, self.head, self.size, self.offset, self.kinds, self._rows = 0, None, 0, None, Counter(), []
        self.header = json.loads(_canonical_bytes({'kind': 'header', **header}))
        self._append(self.header)

    def _append(self, record):
        body = {**record, 'seq': self.count, 'prev': self.head}
        row = {**body, 'sha256': self._hash(body)}
        self.offset, self.count, self.head = len(self._rows), self.count + 1, row['sha256']
        self._rows.append(row)
        self.kinds[row['kind']] += 1
        return row

    def stream(self):
        return list(enumerate(self._rows))

    def iter_rows(self):
        return iter(list(self._rows))

    def iter_decisions(self):
        return (row for row in self.iter_rows() if row['kind'] == 'decision')

    @property
    def rows(self):
        return list(self._rows)

    def decisions(self):
        return list(self.iter_decisions())

    def read(self, offset):
        return self._rows[offset]


@contextlib.contextmanager
def memory_ledger():
    """Narrowly scoped smoke sink: while the smoke runtime is constructed, ``policy_runtime.Ledger`` resolves to
    ``MemoryLedger`` (frozen source bytes untouched); the module attribute is restored right after, so every other
    runtime keeps the registered file ledger."""
    from pitchmdp import policy_runtime as prt
    original = prt.Ledger
    prt.Ledger = MemoryLedger
    try:
        yield
    finally:
        prt.Ledger = original


# ---------------------------------------------------------------- the registered data path

class RealEnvironment:
    """EXP-STAB-01 skeleton: registration chain, tau freeze, stab_games (sha), swap.bind (S2 identity),
    prt.build_runtime (seed 0, no evaluation seed). Constructed only behind the registration gate."""
    def __init__(self, config, local):
        reg.registration_gate(config)  # fail closed: nothing below may read data on a draft
        reg.verify_unchanged_sources(REPO)  # the registered A0 planner path is byte-identical to its pins
        reg.verify_registered_code(config, REPO)  # the new code is the reviewed, registered code
        self.config, self.local, self.prepared = config, local, None
        self.reference, self.runtime_sha = None, 'replay-without-runtime'

    def prepare(self, scope):
        if self.prepared is not None:
            require(self.prepared == scope, 'one data scope per process')
            return
        import pandas as pd
        from pitchmdp.policy_artifacts import load_style_snapshot, load_support_table, load_train_bc
        from pitchmdp import policy_requests as preq
        from pitchmdp import policy_runtime as prt
        stab = _stab()
        swap, rpv = stab.swap, stab.rpv
        self.preq, self.prt, self.rpv = preq, prt, rpv
        spec = self.config['policy_registration']
        registration = rpv.load_registration(REPO / spec['config'], [REPO / path for path in spec['addenda']])
        require(registration['config_sha256'] == reg.POLICY_REGISTRATION['config_sha256'], 'Policy registration changed')
        policy = registration['config']
        freeze = rpv.registered_json(registration, 'tau_freeze')
        require(freeze['selected_tau'] == reg.TAU and freeze['setting'] == reg.TAU_FREEZE['setting']
                and policy['seeds']['planning_main'] == reg.MAIN_SEED, 'Settings differ from the registered tau freeze')
        bundle = policy['identity_registration']['g0_bundle']
        require(bundle.get('path') == reg.G0_BUNDLE['path'] and bundle.get('file_sha256') == reg.G0_BUNDLE['file_sha256'],
                'G0 bundle differs from its pin')
        require(sha256_file(REPO / reg.STAB01['config']) == reg.STAB01['config_sha256'], 'EXP-STAB-01 config changed')
        inputs = rpv.load_inputs(policy, self.local)
        dataset = inputs['parent'].get('dataset_identity', {})
        require(dataset.get('processed_sha256') == reg.DATA['processed_sha256'] and dataset.get('sources') == reg.DATA['sources'],
                'Processed data or statcast sources differ from the pins (statcast 2023-2025 only)')
        frame, store = inputs['frame'], inputs['store']
        games_block = {k: reg.GAMES[k] for k in ('split', 'exclude', 'n_games', 'salt')}
        games, record = stab.stab_games(frame, policy, {'games': games_block}, False)
        require(record['games_sha256'] == reg.GAMES['games_sha256'], 'Registered decision set differs (games sha256)')
        self.games, self.games_record = [int(g) for g in games], record
        order = subsample_order(self.games)
        self.subsamples = {'BF48': order[:reg.SUBSAMPLES['BF48']], 'FH-DP': order[:reg.SUBSAMPLES['FH-DP']]}
        self.fhdp_smoke = order[0]
        if scope == 'confirmation':
            self.confirmation, self.confirmation_record = self._confirmation_games(frame, policy, rpv)
            scope_games = self.confirmation
        elif scope == 'smoke':
            scope_games = sorted(set(self.games[:reg.GAMES['smoke_games']]) | {self.fhdp_smoke})
        else:
            require(scope == 'registered', f'unknown data scope {scope}')
            scope_games = self.games
        blocks = rpv.game_blocks(frame, scope_games, 'dev')
        bc_path, bc_sha = rpv.registered_path(registration, 'bc')
        support_path, support_sha = rpv.registered_path(registration, 'support')
        artifact = load_train_bc(bc_path, bc_sha)
        snapshot, as_of, content = load_style_snapshot(*rpv.registered_path(registration, 'style_dev'))
        require(content == reg.DATA['style_snapshot_sha256'], 'Style snapshot differs from its pin')
        require(as_of == policy['le2025_validation_plan']['stages']['S1b_style_snapshot']['as_of_exclusive']['dev'],
                'Style as-of differs from registration')
        first = pd.to_datetime(frame.game_date.iloc[np.concatenate([p for _, p in blocks])]).min()
        require(first >= pd.Timestamp(as_of), 'Evaluation blocks start before the style as-of date')
        components = swap.bind(registration, inputs, blocks, artifact, snapshot, as_of)
        require(components.sha256 == reg.COMPONENTS_SHA256, 'Bound components differ from the S2-certified identity')
        support, _ = load_support_table(support_path, support_sha, artifact)
        self.registration, self.policy, self.inputs, self.frame, self.store = registration, policy, inputs, frame, store
        self.no_pitch = frozenset(policy['pa_time_rules']['R3_codes']['no_pitch_descriptions'])
        self.bc, self.support, self.snapshot = (bc_path, bc_sha), (support_path, support_sha), (content, as_of)
        self.components = components
        self.reference = prt.MaskedReference(artifact.bc, support)
        self.prepared = scope

    def _confirmation_games(self, frame, policy, rpv):
        import pandas as pd
        from pitchmdp.matrix_data import canonical_hash
        salt = rpv.role_seed(policy, 'selection_salt')
        stages = policy['le2025_validation_plan']['stages']
        used = set(rpv.select_games(frame, 'dev', stages['S6_V4']['n_games'], salt))
        used |= set(rpv.select_games(frame, 'dev', stages['S5_V2_V3']['n_games'], salt))
        used |= set(self.games)
        rest = frame.loc[~frame.game_pk.isin(used)]
        games = [int(g) for g in rpv.select_games(rest, 'dev', reg.CONFIRMATION['n_games'], reg.CONFIRMATION['salt'])]
        require(len(games) == reg.CONFIRMATION['n_games'] and not set(games) & used,
                'Confirmation games overlap S5/S6/EXP-STAB-01')
        months = pd.to_datetime(frame.loc[frame.game_pk.isin(games) & frame.split.eq('dev'), 'game_date'])
        require(months.min() >= pd.Timestamp(reg.DATA['dev_dates'][0]) and months.max() < pd.Timestamp(reg.DATA['dev_dates'][1]),
                'Confirmation games fall outside July-September 2025')
        return games, {'excluded_games': len(used), 'selected': len(games), 'games_sha256': canonical_hash(sorted(games)),
                       'first_date': str(months.min().date()), 'last_date': str(months.max().date()),
                       'salt': reg.CONFIRMATION['salt']}

    def smoke_games(self):
        return self.games[:reg.GAMES['smoke_games']]

    def blocks(self, games):
        for pa_id, positions in self.rpv.game_blocks(self.frame, list(games), 'dev'):
            yield int(self.frame.game_pk.iloc[positions[0]]), positions

    def requests(self, block):
        built, _, _ = self.preq.pa_requests(self.store, block, self.runtime_sha, self.no_pitch)
        return built

    def build_runtime(self, ledger_path, budget, *, in_memory=False):
        """The registered runtime; ``in_memory`` (smoke only) gives it the MemoryLedger sink instead of a file."""
        with memory_ledger() if in_memory else contextlib.nullcontext():
            runtime = self.prt.build_runtime(
                *self.bc, *self.support, ledger_path, components=self.components, budget=budget, tau=reg.TAU,
                samples=reg.SAMPLES, pitch_cap=reg.PITCH_CAP, seed=reg.MAIN_SEED,
                hand_registry=self.rpv.registered_path(self.registration, 'hands'),
                provenance={'style_snapshot_sha256': self.snapshot[0], 'as_of_exclusive': self.snapshot[1],
                            'experiment': EXPERIMENT})
        require(isinstance(runtime.ledger, MemoryLedger) == in_memory, 'runtime ledger sink differs from the request')
        self.reference, self.runtime_sha = runtime.reference, runtime.sha256
        return runtime

    def planner(self, kind, samples, seed, budget):
        from pitchmdp.rollout_policy import JointSimulator, RolloutImprovement
        c = self.components
        simulator = JointSimulator(c.inputs.pool, c.g0, c.we.terminal, budget)
        if kind == 'MC':
            return RolloutImprovement(self.reference, simulator, c.we.cutoff, samples=samples, pitch_cap=reg.PITCH_CAP,
                                      seed=seed)
        if kind == 'RB':
            return pv.RaoBlackwellFirstStep(self.reference, simulator, c.we.cutoff, samples=samples,
                                            pitch_cap=reg.PITCH_CAP, seed=seed, first_pitch_chunk=reg.FIRST_PITCH_CHUNK)
        if kind == 'DP':
            return pv.FrozenHistoryDP(self.reference, simulator, c.we.cutoff)
        raise ValueError(kind)

    def check_request(self, request, mask):
        self.components.check_context(request.state, request.pitcher_hand)
        self.components.check_support(request.state, mask)

    def verify(self, planners, runtime=None):
        for planner in planners:
            self.components.verify(planner)
        if runtime is not None:
            runtime.verify_components()
        self.components.verify()
        require(self.components.sha256 == reg.COMPONENTS_SHA256, 'S2-certified bind identity changed')

    def stab01(self):
        path = Path(reg.STAB01['decisions_npz'])
        require(sha256_file(path) == reg.STAB01['decisions_npz_sha256'], 'EXP-STAB-01 decisions.npz differs from its pin')
        with np.load(path, allow_pickle=False) as data:
            columns = {k: np.array(data[k]) for k in reg.STAB01_COLUMNS}
        require(all(len(v) == reg.STAB01['decisions_compared'] for v in columns.values()), 'EXP-STAB-01 columns short')
        require(set(int(g) for g in columns['game_pk']) <= set(self.games), 'EXP-STAB-01 games differ')
        return columns

    def data_version(self):
        dataset = self.inputs['parent'].get('dataset_identity', {})
        return {'processed_sha256': dataset.get('processed_sha256'), 'sources': dataset.get('sources'),
                'split': 'dev (2025-07-01..2025-09-30)', 'style_snapshot_sha256': self.snapshot[0],
                'components_sha256': self.components.sha256, 'registration_chain': self.registration['chain']}


def run_identity(config, config_path, local_path, repo=REPO):
    """Identity recorded in every pass: config file sha256 and registered digest, Git commit (required), pins and
    source hashes."""
    commit, dirty = git_state(repo)
    return {'config_sha256': sha256_file(config_path), 'config_digest': reg.config_digest(config),
            'local_config_sha256': sha256_file(local_path), 'git_commit': commit, 'git_dirty': dirty,
            'registered_commit': config['registration']['registered_commit'], 'pins_sha256': reg.pins_digest(),
            'source_hashes': {rel: sha256_file(PROJECT / rel) for rel in SOURCES}}


# ---------------------------------------------------------------- passes

class PassRunner:
    """Runs the registered passes of one attempt folder in order, under the common heavy lock."""
    PASS_FOLDERS = {name: spec['output'] for name, spec in reg.PASSES.items()}
    IDENTITY_KEYS = ('config_sha256', 'config_digest', 'source_hashes', 'pins_sha256')

    def __init__(self, config, env, attempt, *, identity, clock=None, lock=None, rss=peak_rss, log=print, manifest=None):
        reg.registration_gate(config)  # fail closed: no real-data stage on a draft
        self.config, self.env, self.attempt = config, env, Path(attempt)
        self.identity, self.clock, self.rss, self.log = identity, clock or SystemClock(), rss, log
        self.lock = lock or (lambda: contextlib.nullcontext())
        self.manifest = None if manifest is None else Path(manifest)
        self.tau, self.scope, self.aborts = reg.TAU, None, []

    # ---- attempt bookkeeping
    def folder(self, name, arm=None):
        base = self.attempt / self.PASS_FOLDERS[name]
        return base / arm if arm else base

    def receipt(self, name, event, detail=None, digest=None, file=None):
        """Append-only receipt in <attempt>/receipts.jsonl and in the external digest manifest."""
        self.attempt.mkdir(parents=True, exist_ok=True)
        line = {'utc': self.clock.now().isoformat(), 'experiment_id': EXPERIMENT, 'kind': 'receipt',
                'attempt': str(self.attempt), 'pass': name, 'event': event, 'detail': detail, 'file': file,
                'sha256': digest, 'config_sha256': self.identity.get('config_sha256'),
                'config_digest': self.identity.get('config_digest')}
        append_line(self.attempt / 'receipts.jsonl', line)
        if self.manifest is not None:
            append_line(self.manifest, line)

    def receipts(self):
        path = self.attempt / 'receipts.jsonl'
        return reg.read_manifest(path) if path.exists() else []

    def terminal_receipt(self, name, sums_file):
        """The last COMPLETED/ABORTED receipt of this attempt, pass and SHA256SUMS file (None if there is none); it must
        carry this config identity and, with a digest manifest, appear identically in that external manifest."""
        lines = [r for r in self.receipts() if r.get('attempt') == str(self.attempt) and r.get('pass') == name
                 and r.get('file') == sums_file and r.get('event') in ('COMPLETED', 'ABORTED')]
        if not lines:
            return None
        last = lines[-1]
        require(last.get('config_digest') == self.identity.get('config_digest'),
                f'{sums_file}: its receipt was written under another config identity')
        if self.manifest is not None:
            require(last in reg.read_manifest(self.manifest), f'{sums_file}: receipt missing from the digest manifest')
        return last

    def status(self, name, arm=None):
        """('absent' | 'complete' | 'aborted' | 'incomplete', results or the stop record). A folder counts as complete
        or aborted only through its terminal receipt, whose digest must equal the verified SHA256SUMS digest; a folder
        without one (never finished, or interrupted before its receipt) is incomplete."""
        folder = self.folder(name, arm)
        if not folder.exists():
            return 'absent', None
        terminal = self.terminal_receipt(name, self.relative(folder))
        if terminal is None:
            return 'incomplete', None
        digest = verify_sums(folder)
        require(terminal['sha256'] == digest, f'{folder.name}: contents differ from its {terminal["event"]} receipt')
        if terminal['event'] == 'ABORTED':
            require((folder / 'aborted.json').exists(), f'{folder.name}: ABORTED receipt without aborted.json')
            return 'aborted', json.loads((folder / 'aborted.json').read_text())
        require((folder / 'results.json').exists() and not (folder / 'aborted.json').exists(),
                f'{folder.name}: COMPLETED receipt without a clean results.json')
        results = json.loads((folder / 'results.json').read_text())
        same = {k: results['identity'].get(k) for k in self.IDENTITY_KEYS}
        require(same == {k: self.identity.get(k) for k in self.IDENTITY_KEYS},
                f'{folder.name}: written under another config or code; continue only with the same registration')
        return 'complete', results

    def close_incomplete(self, name, arm=None):
        """A pass folder without a terminal receipt (crash or interrupt, even before its started.json) is an abort:
        nothing citable. Its retry window is the recorded start window, or conservatively the current one."""
        folder = self.folder(name, arm)
        try:
            window = json.loads((folder / 'started.json').read_text())['window_id']
        except (OSError, ValueError, KeyError, TypeError):
            window = reg.window_id(self.clock.now())
        write_json(folder / 'aborted.json', {'experiment_id': EXPERIMENT, 'pass': name, 'arm': arm,
                                             'reason': 'process ended without a terminal receipt (crash or interrupt)',
                                             'utc': self.clock.now().isoformat(), 'effect': reg.PASSES[name]['on_stop'],
                                             'window_id': window, 'partial_results_discarded': True, 'citable': False})
        self.receipt(name, 'ABORTED', 'incomplete folder closed', write_sums(folder), self.relative(folder))

    def relative(self, folder):
        return (Path(folder) / 'SHA256SUMS').relative_to(self.attempt).as_posix()

    # ---- the verdict
    def expected_verdict(self):
        """The final verdict payload recomputed from the verified smoke/M/C records (None while not final)."""
        state, record = self.status('smoke')
        if state == 'aborted':
            if record.get('smoke_status') != 'COST_EXCEEDED_AT_SMOKE':
                return None
            results = json.loads((self.folder('smoke') / 'results.json').read_text())
            return {'verdict': reg.verdict(smoke_r_b=results['r_b'], pass_m='not_run')}
        if state != 'complete':
            return None
        smoke = record
        state, mandatory = self.status('M')
        if state == 'aborted':
            return {'verdict': reg.verdict(smoke_r_b=smoke['r_b'], pass_m='aborted')}
        if state != 'complete':
            return None
        rb6, payload, rb12 = self.rb6(mandatory), {'rb6_status': mandatory['rb6_status']}, None
        if reg.rb12_required(smoke['r_b'], rb6['stable']):
            state, results = self.status('C')
            if state == 'aborted':
                rb12 = {'state': 'aborted'}
            elif state == 'complete':
                rb12 = {'state': 'completed', 'stable': results['rb12_status']['stable'],
                        'cost_ok': results['rb12_status']['cost_ok']}
                payload['rb12_status'] = results['rb12_status']
            else:
                return None
        verdict = reg.verdict(smoke_r_b=smoke['r_b'], pass_m='completed', rb6=rb6, rb12=rb12)
        return {'verdict': verdict, **payload} if verdict['final'] else None

    def check_verdict(self):
        """verdict.json (if written) must equal the verdict recomputed from the verified records and the digest of
        its VERDICT receipt; returns it (or None)."""
        path = self.attempt / 'verdict.json'
        if not path.exists():
            return None
        stored = json.loads(path.read_text())
        expected = self.expected_verdict()
        body = {k: v for k, v in stored.items() if k not in ('experiment_id', 'utc')}
        require(expected is not None and body == json.loads(json.dumps(plain(expected))),
                'verdict.json differs from the verdict recomputed from the verified pass records')
        lines = [r for r in self.receipts() if r.get('event') == 'VERDICT' and r.get('attempt') == str(self.attempt)]
        require(len(lines) == 1 and lines[0].get('sha256') == sha256_file(path), 'verdict.json differs from its VERDICT receipt')
        if self.manifest is not None:
            require(lines[0] in reg.read_manifest(self.manifest), 'VERDICT receipt missing from the digest manifest')
        return stored

    def write_verdict(self, payload):
        path = self.attempt / 'verdict.json'
        require(not path.exists(), 'the verdict is written once')
        write_json(path, {'experiment_id': EXPERIMENT, 'utc': self.clock.now().isoformat(), **payload})
        self.receipt('-', 'VERDICT', payload['verdict']['label'], sha256_file(path), 'verdict.json')

    def settle(self):
        """Write verdict.json once the registered verdict is final; an existing one is re-checked instead."""
        if (self.attempt / 'verdict.json').exists():
            self.check_verdict()
            return
        expected = self.expected_verdict()
        if expected is not None:
            self.write_verdict(expected)

    # ---- orchestration
    def confirmation_tries(self):
        base = self.folder('confirmation')
        return sorted(p.name for p in base.iterdir() if p.is_dir() and TRY_NAME.fullmatch(p.name)) if base.exists() else []

    def verify_attempt(self):
        """Every finished pass folder still matches its SHA256SUMS and was written under this registration, and
        verdict.json equals the verdict recomputed from them."""
        for name in reg.PASS_ORDER:
            for arm in (('BF48', 'FH-DP') if name == 'D' else (None,)):
                self.status(name, arm)
        for label in self.confirmation_tries():
            self.status('confirmation', label)
        self.check_verdict()

    @contextlib.contextmanager
    def locked(self, name):
        """The common heavy lock (non-blocking); False when another job holds it (nothing starts)."""
        with contextlib.ExitStack() as stack:
            try:
                stack.enter_context(self.lock())
            except RuntimeError as held:
                self.receipt(name, 'LOCK_HELD', str(held))
                self.log('PLANNER_VARIANCE_NOT_STARTED', str(held))
                yield False
                return
            yield True

    def advance(self, through):
        """Run every pending pass up to ``through`` (smoke, M, C or D) in order; exit code."""
        require(through in reg.PASS_ORDER, f'unknown pass {through}')
        self.scope = 'smoke' if through == 'smoke' else 'registered'
        with self.locked('-') as held:
            if not held:
                return EXIT_STOPPED
            self.verify_attempt()
            self.aborts = []
            for name in reg.PASS_ORDER[:reg.PASS_ORDER.index(through) + 1]:
                code = getattr(self, f'step_{name}')()
                if code is not None:
                    return code
            return EXIT_STOPPED if self.aborts else EXIT_OK  # a stop rule fired in this process (C or a D arm)

    def smoke_record(self):
        state, results = self.status('smoke')
        return results if state == 'complete' else None

    def mandatory_record(self):
        state, results = self.status('M')
        return results if state == 'complete' else None

    def step_smoke(self):
        state, _ = self.status('smoke')
        if state == 'complete':
            return None
        if state == 'incomplete':
            self.close_incomplete('smoke')
            state = 'aborted'
        if state == 'aborted':
            self.settle()
            self.log('PLANNER_VARIANCE_STOPPED', 'smoke stopped: no full run under this attempt')
            return EXIT_STOPPED
        return self.run('smoke', self.pass_smoke, reg.pass_cap_seconds('smoke'))

    def step_M(self):
        smoke = self.smoke_record()
        require(smoke is not None, 'pass M needs a completed smoke')
        state, _ = self.status('M')
        if state == 'complete':
            return None
        if state == 'incomplete':
            self.close_incomplete('M')
            state = 'aborted'
        if state == 'aborted':
            self.settle()
            self.log('PLANNER_VARIANCE_STOPPED', 'pass M aborted: experiment aborted, nothing citable')
            return EXIT_STOPPED
        return self.run('M', self.pass_mandatory, reg.pass_cap_seconds('M'))

    def step_C(self):
        smoke, mandatory = self.smoke_record(), self.mandatory_record()
        require(smoke is not None and mandatory is not None, 'pass C needs completed smoke and pass M')
        if not reg.rb12_required(smoke['r_b'], mandatory['rb6_status']['stable']):
            self.receipt('C', 'NOT_REQUIRED', 'RB12 runs only if smoke r_b <= 2.0 and RB6 is not STABLE')
            self.settle()
            return None
        state, _ = self.status('C')
        if state == 'incomplete':
            self.close_incomplete('C')
            state = 'aborted'
        if state in ('complete', 'aborted'):
            self.settle()
            return None
        return self.run('C', self.pass_rb12, reg.pass_cap_seconds('C', r_b=smoke['r_b']))

    def rb12_settled(self, smoke, mandatory):
        if not reg.rb12_required(smoke['r_b'], mandatory['rb6_status']['stable']):
            return True
        return self.status('C')[0] in ('complete', 'aborted')

    def step_D(self):
        smoke, mandatory = self.smoke_record(), self.mandatory_record()
        require(smoke is not None and mandatory is not None, 'pass D needs completed smoke and pass M')
        require(self.rb12_settled(smoke, mandatory), 'pass D runs after pass C')
        rb6 = mandatory['rb6_status']
        for arm, projected in (('BF48', smoke['projections']['BF48_pass_D_seconds']),
                               ('FH-DP', smoke['projections']['FH-DP_pass_D_seconds'])):
            cap = reg.pass_cap_seconds('D', arm=arm)
            if not reg.descriptive_allowed(rb6['stable'], rb6['cost_ok'], projected, cap):
                self.receipt('D', 'NOT_REQUIRED', f'{arm}: RB6 STABLE and COST_OK, or projection {projected} > cap {cap}')
                continue
            state, _ = self.status('D', arm)
            if state in ('complete', 'aborted'):
                continue
            if state == 'incomplete':
                self.close_incomplete('D', arm)
                continue
            code = self.run('D', self.pass_descriptive, cap, arm=arm)
            if code is not None:
                return code
        return None

    def rb6(self, mandatory):
        return {'stable': mandatory['rb6_status']['stable'], 'cost_ok': mandatory['rb6_status']['cost_ok']}

    def run(self, name, function, cap, *, arm=None):
        """Start a pass if its cap fits the window. started.json, the data preparation, the pass, results.json and
        SHA256SUMS run under the watch and the abort handler; the final deadline/cap/RSS check follows the last write
        of the pass folder, and only then is the terminal receipt (the completion marker) appended. None = continue."""
        now = self.clock.now()
        decision, deadline = reg.start_decision(now, cap)
        if decision == 'DEFER':
            self.receipt(name, 'DEFERRED', f'{arm or ""} {deadline}'.strip())
            self.log('PLANNER_VARIANCE_DEFERRED', name, arm or '', deadline)
            return EXIT_DEFERRED
        out = self.folder(name, arm)
        out.mkdir(parents=True, exist_ok=False)
        watch, window, sums_file = Watch(self.clock, deadline, cap, self.rss), reg.window_id(now), self.relative(out)
        stopped = None
        try:
            write_json(out / 'started.json', {'experiment_id': EXPERIMENT, 'pass': name, 'arm': arm,
                                              'identity': self.identity, 'cap_seconds': cap,
                                              'deadline_utc': deadline.isoformat(), 'window_id': window,
                                              'started_utc': now.isoformat(), 'outcomes_read': False})
            self.receipt(name, 'STARTED', arm)
            watch.check('before the data preparation', sample_rss=True)
            self.env.prepare(self.scope)
            preparation = watch.seconds()
            watch.check('after the data preparation', sample_rss=True)
            results = function(out, watch, arm)
            watch.check('after the pass computation', sample_rss=True)
            results.update({'experiment_id': EXPERIMENT, 'pass': name, 'arm': arm, 'identity': self.identity,
                            'cap_seconds': cap, 'deadline_utc': deadline.isoformat(), 'window_id': window,
                            'seconds': watch.seconds(), 'preparation_seconds': preparation,
                            'peak_rss_bytes_process_lifetime': watch.peak, 'finished_utc': self.clock.now().isoformat(),
                            'outcomes_read': False, 'policy_identity_registered': False, 'seeds': SEEDS,
                            'limits': LIMITS})
            if name == 'smoke':
                problems = reg.smoke_record_problems(results)
                require(not problems, 'the smoke sink refuses the record: ' + '; '.join(problems[:5]))
                if results['mechanical_rule']['status'] != 'FULL_RUN_ALLOWED':
                    stopped = results['mechanical_rule']['status']
            write_json(out / 'results.json', results)
            if stopped:  # unsuccessful smoke: its cost/structure evidence is kept next to the stop record
                write_json(out / 'aborted.json', {'experiment_id': EXPERIMENT, 'pass': name, 'reason': stopped,
                                                  'smoke_status': stopped, 'effect': reg.PASSES['smoke']['on_stop'],
                                                  'utc': self.clock.now().isoformat(), 'window_id': window,
                                                  'citable': False,
                                                  'evidence_kept': 'results.json: cost, structure and identity only'})
            digest = write_sums(out)
            watch.final_check()  # after the last write of the pass folder
            self.receipt(name, 'ABORTED' if stopped else 'COMPLETED', stopped or arm, digest, sums_file)
        except Exception as stop:  # every stop rule: aborted.json, nothing citable
            return self.abort(name, arm, out, watch, window, stop)
        if stopped:
            self.aborts.append((name, arm))
            self.settle()
            self.log('PLANNER_VARIANCE_STOPPED', 'smoke:', stopped)
            return EXIT_STOPPED
        self.log('PLANNER_VARIANCE_PASS_COMPLETE', name, arm or '', round(results['seconds'], 1))
        if name in ('M', 'C'):
            self.settle()
        return None

    def abort(self, name, arm, out, watch, window, stop):
        """Stop record: aborted.json (replacing any partial record), partial arrays removed, the folder resealed and
        its ABORTED receipt appended; then the registered pass effect."""
        write_json(out / 'aborted.json', {'experiment_id': EXPERIMENT, 'pass': name, 'arm': arm,
                                          'reason': f'{type(stop).__name__}: {stop}', 'effect': reg.PASSES[name]['on_stop'],
                                          'utc': self.clock.now().isoformat(), 'seconds': watch.seconds(),
                                          'window_id': window, 'partial_results_discarded': True, 'citable': False})
        for partial in [*out.glob('*.npz'), out / 'results.json']:
            partial.unlink(missing_ok=True)
        self.receipt(name, 'ABORTED', f'{arm or ""} {type(stop).__name__}: {stop}'.strip(), write_sums(out),
                     self.relative(out))
        self.aborts.append((name, arm))
        self.log('PLANNER_VARIANCE_ABORTED', name, arm or '', f'{type(stop).__name__}: {stop}')
        return self.after_abort(name)

    def after_abort(self, name):
        """Registered stop effect: smoke -> no full run, M -> experiment aborted (no verdict label), C -> RB12_ABORTED
        (NOT_STABLE), D -> that arm discarded, confirmation -> CONFIRMATION_ABORTED."""
        if name in ('smoke', 'M', 'C'):
            self.settle()
        return None if name in ('C', 'D') else EXIT_STOPPED

    def budget(self, arm, seed):
        """The registered per arm-seed safety row budget (FH-DP has none: its cap bounds it)."""
        limit = FH_DP_BUDGET if seed is None else reg.ARMS[arm]['row_budget'][str(seed)]
        return RowBudget(limit, seed_count=5)

    def make(self, arm, seed, budget=None):
        """Planner of a registered arm: estimator and samples come from the pinned arm spec."""
        spec = reg.ARMS[arm]
        return self.env.planner(KINDS[spec['estimator']], spec.get('samples'), seed, budget or self.budget(arm, seed))

    # ---- decision streams
    def runtime_decisions(self, games, runtime, witness, budget, watch, statuses):
        """Evaluated decisions of ``games`` through the registered runtime: the seed-0 Q is read from its serialized
        row (``q_planning``, ``q_planning_diff_se``), its first-step evidence and planner time from the witness of
        the same call (no second search)."""
        improvement = runtime.improvement
        for block_index, (game, block) in enumerate(self.env.blocks(games)):
            watch.block(block_index)
            for request in self.env.requests(block):
                rows, before = budget.conditional_rows, dict(improvement.diagnostics)
                started = time.perf_counter()
                row = runtime.submit(request)
                submit_ms = (time.perf_counter() - started) * 1000.
                statuses[row['status']] += 1
                if row['status'] not in EVALUATED:
                    witness.discard()
                    continue
                result, state = row['result'], request.state
                mask = np.asarray(self.env.reference.support(state), dtype=bool)
                reference = np.asarray(self.env.reference.probabilities(state), dtype=np.float64)
                require(np.array_equal(mask, np.asarray(result['mask'], dtype=bool))
                        and np.allclose(reference, result['reference'], rtol=0, atol=reg.INTEGRITY['reference_atol']),
                        'Runtime reference differs')
                q = vector(result['q_planning'])
                evidence = witness.take(state)
                require(np.array_equal(q[mask], evidence['q'][mask]), 'the runtime row differs from its observed search')
                after = improvement.diagnostics
                yield Decision(game, request.request_id, state, request.pitcher_hand, mask, reference, {
                    'q': q, 'diff_se': vector(result['q_planning_diff_se']), 'rows': budget.conditional_rows - rows,
                    'ms': evidence['ms'], 'submit_ms': submit_ms, 'first': evidence['first'],
                    'samples': evidence['samples'],
                    'truncated': after['search_truncated'] - before['search_truncated'],
                    'rollouts': after['search_rollouts'] - before['search_rollouts']})

    def replay_decisions(self, games, expected, watch):
        """Pass-M decisions of ``games`` without the runtime planner: requests rebuilt from the store, kept by the
        pass-M request_id sequence, masked reference recomputed and equal to the stored one."""
        wanted = set(int(g) for g in games)
        index = {str(rid): k for k, rid in enumerate(expected['request_id'])}
        sequence = [k for k, g in enumerate(expected['game_pk']) if int(g) in wanted]
        position = 0
        for block_index, (game, block) in enumerate(self.env.blocks(sorted(wanted))):
            watch.block(block_index)
            for request in self.env.requests(block):
                k = index.get(request.request_id)
                if k is None:
                    continue
                require(position < len(sequence) and sequence[position] == k,
                        'replayed decisions differ from the pass-M sequence')
                position += 1
                state = request.state
                mask = np.asarray(self.env.reference.support(state), dtype=bool)
                reference = np.asarray(self.env.reference.probabilities(state), dtype=np.float64)
                require(np.array_equal(mask, expected['mask'][k])
                        and np.allclose(reference, expected['reference'][k], rtol=0, atol=reg.INTEGRITY['reference_atol']),
                        'Masked reference differs from pass M')
                self.env.check_request(request, mask)
                yield k, Decision(game, request.request_id, state, request.pitcher_hand, mask, reference)
        require(position == len(sequence), 'replayed decisions fall short of the pass-M sequence')

    def load_npz(self, name, arm=None):
        state, _ = self.status(name, arm)
        require(state == 'complete', f'{name} {arm or ""}: completed pass required')
        with np.load(self.folder(name, arm) / 'decisions.npz', allow_pickle=False) as data:
            return {k: np.array(data[k]) for k in data.files}

    def reconstruct(self, planner, diagnostics, a0, decision, samples):
        """Untraced comparison first; only differing paths are traced and receive their own boundary margins."""
        check = pv.reconstruction_compare(diagnostics, a0, decision.mask, samples)
        if check['first_pitch_flips'] or not check['differing_paths']:
            return check
        return pv.classify_paths(check, diagnostics, planner.trace(decision.state)[1])

    def tally(self):
        return pv.ReconstructionTally(reg.INTEGRITY['reconstruction_margin_max'],
                                      reg.INTEGRITY['reconstruction_mismatch_share_max'],
                                      reg.INTEGRITY['first_pitch_flips_max'])

    # ---- smoke
    def pass_smoke(self, out, watch, arm=None):
        """The smoke runtime keeps its ledger rows (A0 Q and laws) in memory (MemoryLedger sink): nothing but the
        allowlisted cost/structure/identity record is ever written by the smoke."""
        env = self.env
        smoke_games, fh_game = env.smoke_games(), env.fhdp_smoke
        budgets = {('A0', 0): self.budget('A0-MC6', 0)}
        runtime = env.build_runtime(None, budgets[('A0', 0)], in_memory=True)
        witness = pv.A0Witness(runtime.improvement)
        planners = {}
        for label, arm_name, seed in (('A0', 'A0-MC6', 5), ('RB6', 'RB6', 0), ('RB6', 'RB6', 5),
                                      ('REF-RB6', 'REF-RB6', 100), ('RB12', 'RB12', 0), ('RB12', 'RB12', 5),
                                      ('BF48', 'BF48', 0), ('BF48', 'BF48', 5)):
            budgets[(label, seed)] = self.budget(arm_name, seed)
            planners[(label, seed)] = self.make(arm_name, seed, budgets[(label, seed)])
        rerun = self.make('RB6', 0)
        dp = self.make('FH-DP', None)
        rows, ms, submit = Counter(), defaultdict(list), []
        checks, failures, statuses, fh_statuses = Counter(), [], Counter(), Counter()
        a0_rows = decisions = fh_decisions = 0

        def structural(label, q, decision):
            try:
                checked_law(q, decision, self.tau, label)
                checks['passed'] += 1
            except (ValueError, IntegrityFailure) as problem:
                failures.append(f'{decision.request_id} {label}: {problem}')

        def run_dp(decision):
            nonlocal fh_decisions
            q, _, used, took = search(dp, decision.state)
            rows[('FH-DP', None)] += used
            ms[('FH-DP', None)].append(took)
            structural('FH-DP', q, decision)
            fh_decisions += 1

        for decision in self.runtime_decisions(smoke_games, runtime, witness, budgets[('A0', 0)], watch, statuses):
            decisions += 1
            a0_rows += decision.runtime['rows']
            ms[('A0', 0)].append(decision.runtime['ms'])
            submit.append(decision.runtime['submit_ms'])
            structural('A0 seed 0 (runtime)', decision.runtime['q'], decision)
            first = None
            for key, planner in planners.items():
                q, _, used, took = search(planner, decision.state)
                rows[key] += used
                ms[key].append(took)
                structural(f'{key[0]} seed {key[1]}', q, decision)
                if key == ('RB6', 0):
                    first = q
            again, _, used, took = search(rerun, decision.state)
            rows[('RB6-rerun', 0)] += used
            ms[('RB6-rerun', 0)].append(took)
            if not np.array_equal(again[decision.mask], first[decision.mask]):
                failures.append(f'{decision.request_id}: RB6 seed 0 rerun not identical')
            checks['rerun_identical'] += 1
            if decision.game_pk == fh_game:
                run_dp(decision)
        rows[('A0', 0)] = a0_rows
        if fh_game not in smoke_games:  # the FH-DP smoke game: runtime decisions plus FH-DP only
            for decision in self.runtime_decisions([fh_game], runtime, witness, budgets[('A0', 0)], watch, fh_statuses):
                run_dp(decision)
        try:
            env.verify([runtime.improvement, *planners.values(), rerun, dp], runtime)
            components_verified = True
        except Exception as problem:  # recorded as a structural failure, never repaired
            failures.append(f'components.verify: {problem}')
            components_verified = False
        require(decisions > 0, 'no smoke decision evaluated')
        r_b = float(np.mean([rows[('RB6', s)] / rows[('A0', s)] for s in reg.PAIR]))
        name = lambda label, seed: label if seed is None else f'{label} seed {seed}'  # noqa: E731
        seconds = {name(*key): sum(values) / 1000. for key, values in ms.items()}
        projections = {
            'BF48_pass_D_seconds': (seconds['BF48 seed 0'] + seconds['BF48 seed 5'])
                                   * reg.SUBSAMPLES['BF48'] / reg.GAMES['smoke_games'],
            'BF48_cap_seconds': reg.pass_cap_seconds('D', arm='BF48'),
            'FH-DP_pass_D_seconds': seconds.get('FH-DP', 0.) * reg.SUBSAMPLES['FH-DP'] / reg.SUBSAMPLES['fhdp_smoke_games'],
            'FH-DP_cap_seconds': reg.pass_cap_seconds('D', arm='FH-DP'),
            'pass_C_cap_seconds': reg.pass_cap_seconds('C', r_b=r_b),
            'basis': 'smoke planner seconds scaled by games (BF48 x 50/3, FH-DP x 10); caps from the registration'}
        structural_ok = not failures and components_verified and fh_decisions > 0
        status = ('STRUCTURAL_FAILURE' if not structural_ok else
                  'COST_EXCEEDED_AT_SMOKE' if r_b > reg.THRESHOLDS['smoke_r_b_at_most'] else 'FULL_RUN_ALLOWED')
        return {'games': {'smoke_games': smoke_games, 'fhdp_smoke_game': fh_game,
                          'registered_games_sha256': reg.GAMES['games_sha256']},
                'requests': sum(statuses.values()), 'statuses': dict(statuses), 'decisions': decisions,
                'fhdp_decisions': fh_decisions, 'fhdp_only_game_statuses': dict(fh_statuses),
                'structural': {'checks_passed': checks['passed'], 'rerun_decisions': checks['rerun_identical'],
                               'failures': failures[:50], 'failure_count': len(failures),
                               'components_verified': components_verified},
                'structural_ok': structural_ok,
                'cost': {'rows_by_arm_seed': {name(*key): int(value) for key, value in rows.items()},
                         'planner_seconds_by_arm_seed': seconds,
                         'planner_ms_per_decision_median': {name(*key): float(np.median(values))
                                                            for key, values in ms.items()},
                         'runtime_submit_seconds': sum(submit) / 1000.,
                         'runtime_submit_ms_per_decision_median': float(np.median(submit))},
                'r_b': r_b, 'r_b_definition': 'mean over seeds 0 and 5 of rows(RB6)/rows(A0) on the smoke decisions',
                'projections': projections,
                'mechanical_rule': {'r_b_at_most': reg.THRESHOLDS['smoke_r_b_at_most'], 'structural_ok': structural_ok,
                                    'status': status,
                                    'rb12_cost_condition_met': r_b <= reg.THRESHOLDS['rb12_smoke_r_b_at_most']},
                'runtime_ledger': {'location': 'process memory (MemoryLedger sink; no file opened)',
                                   'persisted': False, 'rows': runtime.ledger.count, 'head_sha256': runtime.ledger.head},
                'data_version': self.env.data_version(), 'unchanged_sources': reg.UNCHANGED_SHA256}

    # ---- pass M
    def pass_mandatory(self, out, watch, arm=None):
        env, tau = self.env, self.tau
        stab = _stab()
        smoke = self.smoke_record()
        stab01 = env.stab01()
        allowed_k1_mismatches = int(np.floor((1 - reg.INTEGRITY['stab01_k1_match_min']) * len(stab01['request_id'])))
        budgets = {('A0', 0): self.budget('A0-MC6', 0), ('A0', 5): self.budget('A0-MC6', 5),
                   ('RB6', 0): self.budget('RB6', 0), ('RB6', 5): self.budget('RB6', 5),
                   ('REF', 100): self.budget('REF-RB6', reg.REFERENCE_SEED)}
        runtime = env.build_runtime(out / 'ledger-dev-0.jsonl', budgets[('A0', 0)])
        witness0 = pv.A0Witness(runtime.improvement)
        planners = {('A0', 5): self.make('A0-MC6', 5, budgets[('A0', 5)]),
                    ('RB6', 0): self.make('RB6', 0, budgets[('RB6', 0)]),
                    ('RB6', 5): self.make('RB6', 5, budgets[('RB6', 5)]),
                    ('REF', 100): self.make('REF-RB6', reg.REFERENCE_SEED, budgets[('REF', 100)])}
        witness5 = pv.A0Witness(planners[('A0', 5)])
        replay = self.make('RB6', 0)  # fresh seed-0 planner on its own budget (200-decision replay)
        tally, c = self.tally(), Columns()
        statuses, replay_max, k1_mismatches = Counter(), [], 0
        sets = {'A': [0], 'B': [5], 'K': [1]}
        for decision in self.runtime_decisions(env.games, runtime, witness0, budgets[('A0', 0)], watch, statuses):
            k = len(c.request_id)
            require(k < len(stab01['request_id']) and str(stab01['request_id'][k]) == decision.request_id,
                    'request_id sequence differs from EXP-STAB-01')
            q0 = decision.runtime['q']
            results = {key: search(planner, decision.state) for key, planner in planners.items()}
            evidence5 = witness5.take(decision.state)
            a0 = {0: {'q': q0, 'first': decision.runtime['first'], 'samples': decision.runtime['samples']},
                  5: {'q': results[('A0', 5)][0], 'first': evidence5['first'], 'samples': evidence5['samples']}}
            laws = {('A0', 0): checked_law(q0, decision, tau, 'A0 seed 0')}
            for key, (q, _, _, _) in results.items():
                laws[key] = checked_law(q, decision, tau, f'{key[0]} seed {key[1]}')
            if len(replay_max) < reg.REPLAY_DECISIONS:
                again = replay.q_values(decision.state)[0]
                replay_max.append(float(np.abs(again[decision.mask] - results[('RB6', 0)][0][decision.mask]).max()))
                require(replay_max[-1] <= reg.INTEGRITY['replay_atol'], 'Fresh RB6 seed-0 planner does not reproduce')
            tally.add(decision.request_id, {
                f'RB6 seed {s} vs A0 seed {s}': self.reconstruct(planners[('RB6', s)], results[('RB6', s)][1], a0[s],
                                                                 decision, reg.SAMPLES) for s in reg.PAIR})
            k1 = stab.compare({0: q0, 5: results[('A0', 5)][0]}, decision.reference, decision.mask, tau, sets)
            match = (k1['K1_tv'] == float(stab01['K1_tv'][k]) and k1['K1_top1_same'] == bool(stab01['K1_top1_same'][k]))
            k1_mismatches += int(not match)
            require(k1_mismatches <= allowed_k1_mismatches,
                    f'A0 does not reproduce the EXP-STAB-01 K1 columns ({k1_mismatches} mismatches)')
            c.add_decision(decision, laws, results, tau, k1_match=match)
            c.add_stab01(stab01, k)
            c.extra(a0_first_0=a0[0]['first'], a0_samples_0=a0[0]['samples'], a0_first_5=a0[5]['first'],
                    a0_samples_5=a0[5]['samples'], **{f'rb_first_RB6_{s}': results[('RB6', s)][1]['first_outcome'][:, :reg.SAMPLES]
                                                       for s in reg.PAIR})
        require(len(c.request_id) == len(stab01['request_id']), 'decision count differs from EXP-STAB-01')
        require(len(replay_max) == min(reg.REPLAY_DECISIONS, len(c.request_id)), 'Replay count short')
        reconstruction = tally.finish()
        env.verify([runtime.improvement, *planners.values(), replay], runtime)
        arrays = c.arrays()
        np.savez_compressed(out / 'decisions.npz', **arrays)
        summary = summarize_mandatory(arrays, smoke['r_b'])
        rb6 = summary.pop('rb6_status')
        required = reg.rb12_required(smoke['r_b'], rb6['stable'])
        verdict = reg.verdict(smoke_r_b=smoke['r_b'], pass_m='completed',
                              rb6={'stable': rb6['stable'], 'cost_ok': rb6['cost_ok']},
                              rb12={'state': 'pending'} if required else None)
        return {'games': env.games_record, 'requests': sum(statuses.values()), 'statuses': dict(statuses),
                'decisions_compared': len(c.request_id), 'comparison': summary, 'rb6_status': rb6,
                'rb12_required': required, 'verdict': verdict,
                'integrity': {'games_sha256': reg.GAMES['games_sha256'],
                              'stab01_decisions_npz_sha256': reg.STAB01['decisions_npz_sha256'],
                              'request_order_equals_stab01': True,
                              'stab01_k1_matches': len(c.request_id) - k1_mismatches,
                              'stab01_k1_match_rate': 1 - k1_mismatches / len(c.request_id),
                              'reconstruction': reconstruction,
                              'replay': {'decisions': len(replay_max), 'max_abs_q_difference': max(replay_max)},
                              'components_verified': True},
                'data_version': self.env.data_version(),
                'cost': {'conditional_rows_by_arm_seed': {f'{a} seed {s}': b.conditional_rows for (a, s), b in budgets.items()}}}

    # ---- pass C
    def pass_rb12(self, out, watch, arm=None):
        env, tau = self.env, self.tau
        smoke, mandatory = self.smoke_record(), self.mandatory_record()
        expected = self.load_npz('M')
        budgets = {s: self.budget('RB12', s) for s in reg.PAIR}
        planners = {s: self.make('RB12', s, budgets[s]) for s in reg.PAIR}
        replay = self.make('RB12', 0)
        tally, c, replay_max = self.tally(), Columns(), []
        for k, decision in self.replay_decisions(env.games, expected, watch):
            results = {('RB12', s): search(planners[s], decision.state) for s in reg.PAIR}
            laws = {key: checked_law(r[0], decision, tau, f'RB12 seed {key[1]}') for key, r in results.items()}
            if len(replay_max) < reg.REPLAY_DECISIONS:
                again = replay.q_values(decision.state)[0]
                replay_max.append(float(np.abs(again[decision.mask] - results[('RB12', 0)][0][decision.mask]).max()))
                require(replay_max[-1] <= reg.INTEGRITY['replay_atol'], 'Fresh RB12 seed-0 planner does not reproduce')
            a0 = {s: {'q': expected[f'q_A0_{s}'][k], 'first': expected[f'a0_first_{s}'][k],
                      'samples': expected[f'a0_samples_{s}'][k]} for s in reg.PAIR}
            tally.add(decision.request_id, {
                f'RB12 seed {s} first 6 vs A0 seed {s}': self.reconstruct(planners[s], results[('RB12', s)][1], a0[s],
                                                                          decision, reg.SAMPLES) for s in reg.PAIR})
            c.add_decision(decision, laws, results, tau, index=k)
            c.extra(**{f'rb_first_RB12_{s}': results[('RB12', s)][1]['first_outcome'][:, :reg.SAMPLES] for s in reg.PAIR})
        require(len(c.request_id) == len(expected['request_id']), 'pass C decisions differ from pass M')
        require(len(replay_max) == min(reg.REPLAY_DECISIONS, len(c.request_id)), 'Replay count short')
        reconstruction = tally.finish()
        env.verify([*planners.values(), replay])
        arrays = c.arrays()
        np.savez_compressed(out / 'decisions.npz', **arrays)
        summary = summarize_arm(arrays, expected, 'RB12')
        rb12 = {'stable': reg.stable(summary['primary']['top1_agreement_ci95'], summary['primary']['mean_tv_ci95']),
                'cost_ok': reg.cost_ok(summary['r_rows']), 'r_rows': summary['r_rows']}
        verdict = reg.verdict(smoke_r_b=smoke['r_b'], pass_m='completed', rb6=self.rb6(mandatory),
                              rb12={'state': 'completed', 'stable': rb12['stable'], 'cost_ok': rb12['cost_ok']})
        return {'decisions_compared': len(c.request_id), 'comparison': summary, 'rb12_status': rb12,
                'rb6_status': mandatory['rb6_status'], 'verdict': verdict,
                'integrity': {'reconstruction': reconstruction,
                              'replay': {'decisions': len(replay_max), 'max_abs_q_difference': max(replay_max)},
                              'components_verified': True, 'pass_M_sha256sums': verify_sums(self.folder('M'))},
                'data_version': self.env.data_version(),
                'cost': {'conditional_rows_by_seed': {str(s): b.conditional_rows for s, b in budgets.items()}}}

    # ---- pass D
    def pass_descriptive(self, out, watch, arm):
        env, tau = self.env, self.tau
        expected = self.load_npz('M')
        mandatory = self.mandatory_record()
        games = env.subsamples[arm]
        if arm == 'BF48':
            budgets = {s: self.budget('BF48', s) for s in reg.PAIR}
            planners = {s: self.make('BF48', s, budgets[s]) for s in reg.PAIR}
            c = Columns()
            for k, decision in self.replay_decisions(games, expected, watch):
                results = {('BF48', s): search(planners[s], decision.state) for s in reg.PAIR}
                laws = {key: checked_law(r[0], decision, tau, f'BF48 seed {key[1]}') for key, r in results.items()}
                c.add_decision(decision, laws, results, tau, index=k)
            env.verify(list(planners.values()))
            arrays = c.arrays()
            np.savez_compressed(out / 'decisions.npz', **arrays)
            summary = summarize_arm(arrays, expected, 'BF48')
            summary['rb6_minus_bf48'] = rb6_vs_bruteforce(arrays, expected, mandatory)
            return {'games': games, 'decisions': len(c.request_id), 'comparison': summary, 'descriptive_only': True,
                    'data_version': self.env.data_version(),
                    'cost': {'conditional_rows_by_seed': {str(s): b.conditional_rows for s, b in budgets.items()}}}
        require(arm == 'FH-DP', f'unknown pass-D arm {arm}')
        dp = self.make('FH-DP', None)
        columns = defaultdict(list)
        for k, decision in self.replay_decisions(games, expected, watch):
            q, _, used, took = search(dp, decision.state)
            p = checked_law(q, decision, tau, 'FH-DP')
            columns['index'].append(k)
            columns['request_id'].append(decision.request_id)
            columns['game_pk'].append(decision.game_pk)
            columns['q_FH-DP'].append(q)
            columns['rows'].append(used)
            columns['ms'].append(took)
            for s in (0, 5, reg.REFERENCE_SEED):
                label = 'REF_100' if s == reg.REFERENCE_SEED else f'RB6_{s}'
                same, tv = pv.pair(p, pv.law(expected[f'q_{label}'][k], decision.reference, decision.mask, tau))
                columns[f'dp_vs_{label}_top1_same'].append(same)
                columns[f'dp_vs_{label}_tv'].append(tv)
        env.verify([dp])
        arrays = {k: np.array(v) for k, v in columns.items()}
        np.savez_compressed(out / 'decisions.npz', **arrays)
        return {'games': games, 'decisions': len(arrays['request_id']), 'comparison': summarize_fhdp(arrays, expected),
                'descriptive_only': True, 'data_version': self.env.data_version(),
                'cost': {'conditional_rows': int(np.sum(arrays['rows']))}}

    # ---- confirmation slice
    def require_passing_verdict(self):
        stored = self.check_verdict()
        require(stored is not None, 'no final verdict in this attempt')
        label = stored['verdict']['label'] or ''
        require(label.startswith('STABLE_WITHIN_COST:'), f'confirmation runs only after STABLE_WITHIN_COST (got {label})')
        return label.split(':', 1)[1]

    def confirm_select(self):
        """Record the 50-game confirmation list and its sha256 (no planning call), under the same window and
        deadline/cap/RSS watch as a pass; a stopped selection writes nothing and can be repeated."""
        self.scope = 'confirmation'
        with self.locked('confirmation') as held:
            if not held:
                return EXIT_STOPPED
            self.verify_attempt()
            passing = self.require_passing_verdict()
            path = self.attempt / SELECTION_FILE
            require(not path.exists(), 'confirmation selection already recorded')
            cap = reg.pass_cap_seconds('confirmation', r_b=self.smoke_record()['r_b'], arm=passing)
            now = self.clock.now()
            decision, deadline = reg.start_decision(now, cap)
            if decision == 'DEFER':
                self.receipt('confirmation', 'DEFERRED', f'selection: {deadline}')
                return EXIT_DEFERRED
            watch = Watch(self.clock, deadline, cap, self.rss)
            staged = path.with_suffix('.staged')
            try:
                watch.check('before the data preparation', sample_rss=True)
                self.env.prepare(self.scope)
                watch.check('after the data preparation', sample_rss=True)
                write_json(staged, {'experiment_id': EXPERIMENT, 'games': self.env.confirmation,
                                    'record': self.env.confirmation_record, 'utc': self.clock.now().isoformat(),
                                    'window_id': reg.window_id(now), 'identity': self.identity,
                                    'note': 'append games_sha256 to the registration block before the confirmation'})
                watch.final_check()  # after the last write; the selection is published only afterwards
                staged.replace(path)
                self.receipt('confirmation', 'SELECTED', self.env.confirmation_record['games_sha256'],
                             sha256_file(path), SELECTION_FILE)
            except Exception as stop:
                staged.unlink(missing_ok=True)
                self.receipt('confirmation', 'SELECTION_ABORTED', f'{type(stop).__name__}: {stop}')
                self.log('PLANNER_VARIANCE_ABORTED', 'confirmation selection', f'{type(stop).__name__}: {stop}')
                return EXIT_STOPPED
            return EXIT_OK

    def confirm(self, pin):
        """Run the confirmation slice against the appended games pin (Song's go recorded by the PM). A try after an
        aborted one starts only in a later run window."""
        require(isinstance(pin, dict) and pin.get('experiment_id') == EXPERIMENT and isinstance(
            pin.get('confirmation_games_sha256'), str), 'confirmation games pin required')
        self.scope = 'confirmation'
        with self.locked('confirmation') as held:
            if not held:
                return EXIT_STOPPED
            self.verify_attempt()
            passing = self.require_passing_verdict()
            tries, last_aborted = self.confirmation_tries(), None
            for label in tries:
                state, record = self.status('confirmation', label)
                if state == 'complete':  # CONFIRMED or NOT_CONFIRMED is final
                    return EXIT_OK
                if state == 'incomplete':  # interrupted, even before its started.json: recovered as an abort
                    self.close_incomplete('confirmation', label)
                    state, record = self.status('confirmation', label)
                last_aborted = (label, record)
            if last_aborted is not None:
                label, record = last_aborted
                if record.get('window_id', reg.window_id(self.clock.now())) == reg.window_id(self.clock.now()):
                    self.receipt('confirmation', 'DEFERRED', f'{label} aborted in this window: retry in a later one')
                    return EXIT_DEFERRED
            cap = reg.pass_cap_seconds('confirmation', r_b=self.smoke_record()['r_b'], arm=passing)
            self.passing, self.pin = passing, pin['confirmation_games_sha256']
            code = self.run('confirmation', self.pass_confirmation, cap, arm=f'try-{len(tries) + 1:02d}')
            return EXIT_OK if code is None else code

    def pass_confirmation(self, out, watch, arm=None):
        env, tau, passing = self.env, self.tau, self.passing
        require(env.confirmation_record['games_sha256'] == self.pin, 'confirmation games differ from the appended pin')
        budgets = {('A0', 0): self.budget('A0-MC6', 0), ('A0', 5): self.budget('A0-MC6', 5)}
        for s in reg.PAIR:
            budgets[(passing, s)] = self.budget(passing, s)
        runtime = env.build_runtime(out / 'ledger-dev-0.jsonl', budgets[('A0', 0)])
        witness0 = pv.A0Witness(runtime.improvement)
        planners = {('A0', 5): self.make('A0-MC6', 5, budgets[('A0', 5)]),
                    (passing, 0): self.make(passing, 0, budgets[(passing, 0)]),
                    (passing, 5): self.make(passing, 5, budgets[(passing, 5)])}
        witness5 = pv.A0Witness(planners[('A0', 5)])
        replay = self.make(passing, 0)
        tally, c, statuses, replay_max = self.tally(), Columns(), Counter(), []
        for decision in self.runtime_decisions(env.confirmation, runtime, witness0, budgets[('A0', 0)], watch, statuses):
            q0 = decision.runtime['q']
            results = {key: search(planner, decision.state) for key, planner in planners.items()}
            evidence5 = witness5.take(decision.state)
            a0 = {0: {'q': q0, 'first': decision.runtime['first'], 'samples': decision.runtime['samples']},
                  5: {'q': results[('A0', 5)][0], 'first': evidence5['first'], 'samples': evidence5['samples']}}
            laws = {('A0', 0): checked_law(q0, decision, tau, 'A0 seed 0')}
            for key, (q, _, _, _) in results.items():
                laws[key] = checked_law(q, decision, tau, f'{key[0]} seed {key[1]}')
            if len(replay_max) < reg.REPLAY_DECISIONS:
                again = replay.q_values(decision.state)[0]
                replay_max.append(float(np.abs(again[decision.mask] - results[(passing, 0)][0][decision.mask]).max()))
                require(replay_max[-1] <= reg.INTEGRITY['replay_atol'], f'Fresh {passing} seed-0 planner does not reproduce')
            tally.add(decision.request_id, {
                f'{passing} seed {s} vs A0 seed {s}': self.reconstruct(planners[(passing, s)], results[(passing, s)][1],
                                                                       a0[s], decision, reg.SAMPLES) for s in reg.PAIR})
            c.add_decision(decision, laws, results, tau)
            c.extra(a0_first_0=a0[0]['first'], a0_samples_0=a0[0]['samples'], a0_first_5=a0[5]['first'],
                    a0_samples_5=a0[5]['samples'],
                    **{f'rb_first_{passing}_{s}': results[(passing, s)][1]['first_outcome'][:, :reg.SAMPLES] for s in reg.PAIR})
        require(len(c.request_id) > 0, 'no confirmation decision evaluated')
        require(len(replay_max) == min(reg.REPLAY_DECISIONS, len(c.request_id)), 'Replay count short')
        reconstruction = tally.finish()
        env.verify([runtime.improvement, *planners.values(), replay], runtime)
        arrays = c.arrays()
        np.savez_compressed(out / 'decisions.npz', **arrays)
        summary = summarize_confirmation(arrays, passing)
        primary = summary[passing]
        verdict = reg.confirmation_verdict('completed', top1_ci95=primary['top1_agreement_ci95'],
                                           mean_tv_ci95=primary['mean_tv_ci95'], r_rows=summary['r_rows'])
        return {'games': env.confirmation_record, 'games_pin': self.pin, 'passing_arm': passing,
                'requests': sum(statuses.values()), 'statuses': dict(statuses), 'decisions_compared': len(c.request_id),
                'comparison': summary, 'confirmation_verdict': verdict,
                'integrity': {'reconstruction': reconstruction,
                              'replay': {'decisions': len(replay_max), 'max_abs_q_difference': max(replay_max)},
                              'components_verified': True},
                'data_version': self.env.data_version(),
                'cost': {'conditional_rows_by_arm_seed': {f'{a} seed {s}': b.conditional_rows for (a, s), b in budgets.items()}}}


# ---------------------------------------------------------------- per-decision columns and summaries

class Columns:
    """Per-decision arrays saved to decisions.npz (the verifier recomputes the verdict from them)."""
    def __init__(self):
        self.data = defaultdict(list)
        self.request_id = self.data['request_id']

    def extra(self, **columns):
        for key, value in columns.items():
            self.data[key].append(value)

    def add_decision(self, decision, laws, results, tau, *, k1_match=None, index=None):
        d = self.data
        d['request_id'].append(decision.request_id)
        d['game_pk'].append(decision.game_pk)
        d['mask'].append(decision.mask)
        d['reference'].append(decision.reference)
        d['support_size'].append(int(decision.mask.sum()))
        if index is not None:
            d['index'].append(index)
        if k1_match is not None:
            d['k1_match'].append(k1_match)
        if decision.runtime is not None:
            runtime = decision.runtime
            d['q_A0_0'].append(runtime['q'])
            d['rows_A0_0'].append(runtime['rows'])
            d['ms_A0_0'].append(runtime['ms'])
            d['submit_ms_A0_0'].append(runtime['submit_ms'])
            d['truncated_A0_0'].append(runtime['truncated'])
            d['rollouts_A0_0'].append(runtime['rollouts'])
            d['max_diff_se_A0_0'].append(pv.max_finite(runtime['diff_se'], decision.mask))
            d['kl_A0_0'].append(pv.kl_to_reference(laws[('A0', 0)], decision.reference, decision.mask))
        for (arm, seed), (q, diagnostics, rows, ms) in results.items():
            label = 'REF_100' if arm == 'REF' else f'{arm}_{seed}'
            d[f'q_{label}'].append(q)
            d[f'rows_{label}'].append(rows)
            d[f'ms_{label}'].append(ms)
            d[f'kl_{label}'].append(pv.kl_to_reference(laws[(arm, seed)], decision.reference, decision.mask))
            d[f'max_diff_se_{label}'].append(pv.max_finite(paired_difference_se(diagnostics['sample_values'], decision.mask),
                                                           decision.mask))
            if 'terminal_term' in diagnostics:  # RB search
                d[f'truncated_{label}'].append(diagnostics['truncated_weight'])
                for key, value in pv.residual_terms(diagnostics, decision.mask).items():
                    d[f'residual_{key}_{label}'].append(value)
            else:
                d[f'truncated_{label}'].append(diagnostics['truncated'])
            d[f'rollouts_{label}'].append(diagnostics['rollouts'])
        for arm in sorted({arm for arm, _ in laws if arm != 'REF'}):
            if (arm, 0) in laws and (arm, 5) in laws:
                same, tv = pv.pair(laws[(arm, 0)], laws[(arm, 5)])
                d[f'top1_same_{arm}'].append(same)
                d[f'tv_{arm}'].append(tv)
        if ('REF', 100) in laws:
            q_ref = results[('REF', 100)][0]
            g_eff = pv.effective_gap(q_ref, decision.reference, decision.mask, tau)
            g_q = pv.q_gap(q_ref, decision.mask)
            d['g_eff'].append(np.nan if g_eff is None else g_eff)
            d['g_q'].append(np.nan if g_q is None else g_q)
            d['bin_eff'].append(pv.gap_bin(g_eff, reg.GAP_BINS))
            d['bin_q'].append(pv.gap_bin(g_q, reg.GAP_BINS))
            anchor = int(np.argmax(laws[('REF', 100)]))
            d['anchor'].append(anchor)
            for arm in ('A0', 'RB6'):
                if (arm, 5) in results:
                    d[f'v_{arm}_5'].append(pv.paired_difference_variance(results[(arm, 5)][1]['sample_values'],
                                                                         decision.mask, anchor))

    def add_stab01(self, stab01, k):
        for column in reg.STAB01_COLUMNS:
            if column not in ('request_id', 'game_pk'):
                self.data[f'stab01_{column}'].append(stab01[column][k])

    def arrays(self):
        out = {}
        for key, values in self.data.items():
            out[key] = np.array([str(v) for v in values]) if key == 'request_id' else np.array(values)
        return out


def bootstrap(games):
    return pv.GameBootstrap(games, draws=reg.BOOTSTRAP['draws'], seed=reg.BOOTSTRAP['seed'])


def r_rows(arrays, arm, a0_rows):
    """r_rows(X) = mean over seeds 0 and 5 of sum rows(X, seed) / sum rows(A0, seed)."""
    return float(np.mean([np.sum(arrays[f'rows_{arm}_{s}']) / np.sum(a0_rows[s]) for s in reg.PAIR]))


def noise(arrays, labels):
    out = {}
    for label in labels:
        values = arrays[f'max_diff_se_{label}'] / reg.TAU
        values = values[np.isfinite(values)]
        out[label] = {str(q): float(np.quantile(values, q)) if len(values) else None for q in reg.NOISE_QUANTILES}
    return out


def resources(arrays, labels):
    """Planner time per arm-seed (A0 seed 0: its search inside runtime.submit) and, for A0 seed 0, the whole submit."""
    out = {}
    for label in labels:
        if f'ms_{label}' not in arrays:
            continue
        out[label] = {'planner_ms_mean': float(np.mean(arrays[f'ms_{label}'])),
                      'planner_ms_median': float(np.median(arrays[f'ms_{label}'])),
                      'rows': int(np.sum(arrays[f'rows_{label}']))}
        if f'submit_ms_{label}' in arrays:
            out[label]['runtime_submit_ms_mean'] = float(np.mean(arrays[f'submit_ms_{label}']))
    return out


def truncation(arrays, labels):
    return {label: float(np.sum(arrays[f'truncated_{label}']) / np.sum(arrays[f'rollouts_{label}']))
            for label in labels if f'truncated_{label}' in arrays}


def residual(arrays, labels):
    out = {}
    for label in labels:
        if f'residual_var_sample_{label}' not in arrays:
            continue
        var = arrays[f'residual_var_sample_{label}']
        valid = np.isfinite(var) & (var > 0)
        ratio = arrays[f'residual_var_terminal_{label}'][valid] / var[valid]
        cov = arrays[f'residual_cov_{label}'][valid] / var[valid]
        out[label] = {'decisions': int(valid.sum()),
                      'median_terminal_variance_ratio': float(np.median(ratio)) if len(ratio) else None,
                      'median_terminal_continuation_covariance_ratio': float(np.median(cov)) if len(cov) else None,
                      'note': 'descriptive ratio, not an additive share'}
    return out


def kl_summary(arrays, arm, boot, baseline='A0'):
    kl = (arrays[f'kl_{arm}_0'] + arrays[f'kl_{arm}_5']) / 2
    out = {'mean_kl_to_reference': float(kl.mean())}
    if f'kl_{baseline}_0' in arrays and arm != baseline:
        base = (arrays[f'kl_{baseline}_0'] + arrays[f'kl_{baseline}_5']) / 2
        out[f'minus_{baseline}'] = pv.paired_gain(kl, base, boot)
    return out


def q_level(arrays, arm):
    mask = arrays['mask']
    q0, q5 = arrays[f'q_{arm}_0'], arrays[f'q_{arm}_5']
    diff = np.where(mask, np.abs(np.where(mask, q0, 0.) - np.where(mask, q5, 0.)), 0.).max(axis=1)
    return float(diff.mean())


def gap_bins(arrays, sets, boot):
    """Agreement and mean TV by REF-RB6 effective-gap bin and by raw Q-gap bin (descriptive)."""
    out = {}
    for kind in ('eff', 'q'):
        bins = arrays[f'bin_{kind}']
        table = {}
        for k in [-1, *range(len(reg.GAP_BINS))]:
            name = 'support_1' if k == -1 else (f'[{reg.GAP_BINS[k]}, {reg.GAP_BINS[k + 1]})' if k + 1 < len(reg.GAP_BINS)
                                                else f'[{reg.GAP_BINS[k]}, inf)')
            include = bins == k
            row = {'decisions': int(include.sum())}
            if include.any():
                for label, (same, tv) in sets.items():
                    row[label] = {'top1_agreement': boot.subset(np.asarray(same, dtype=float), include),
                                  'mean_tv': boot.subset(tv, include)}
            else:
                row['note'] = 'none'
            table[name] = row
        out['effective_gap' if kind == 'eff' else 'raw_q_gap'] = table
    return out


def summarize_mandatory(a, smoke_r_b):
    """Registered primary and secondary metrics of pass M (design section 5)."""
    boot = bootstrap(a['game_pk'])
    a0_rows = {s: a[f'rows_A0_{s}'] for s in reg.PAIR}
    arms = {'A0-MC6': pv.primary(a['top1_same_A0'], a['tv_A0'], boot),
            'RB6': pv.primary(a['top1_same_RB6'], a['tv_RB6'], boot)}
    rows_ratio = r_rows(a, 'RB6', a0_rows)
    rb6 = {'stable': reg.stable(arms['RB6']['top1_agreement_ci95'], arms['RB6']['mean_tv_ci95']),
           'cost_ok': reg.cost_ok(rows_ratio), 'r_rows': rows_ratio}
    arms['A0-MC6']['stable_descriptive'] = reg.stable(arms['A0-MC6']['top1_agreement_ci95'], arms['A0-MC6']['mean_tv_ci95'])
    h2 = pv.paired_gain(a['top1_same_RB6'], a['top1_same_A0'], boot)
    k_c = 3 if rows_ratio <= 3 else 5 if rows_ratio <= 5 else None
    k_c_gain = None if k_c is None else pv.paired_gain(a['top1_same_RB6'], a[f'stab01_K{k_c}_top1_same'], boot)
    vrf = boot.ratio(a['v_A0_5'], a['v_RB6_5'])  # seed-5 pair, anchor = REF-RB6 top-1, same draws
    with np.errstate(divide='ignore', invalid='ignore'):
        vrf_rep = boot.sums(a['v_A0_5']) / boot.sums(a['v_RB6_5'])
        rows_rep = np.mean([boot.sums(a[f'rows_RB6_{s}']) / boot.sums(a0_rows[s]) for s in reg.PAIR], axis=0)
        efficiency_rep = vrf_rep / rows_rep
    efficiency_rep = efficiency_rep[np.isfinite(efficiency_rep)]
    efficiency = {'value': None if vrf['value'] is None else vrf['value'] / rows_ratio,
                  'ci95': [float(v) for v in np.quantile(efficiency_rep, (.025, .975))] if len(efficiency_rep) else None}
    sets = {'A0-MC6': (a['top1_same_A0'], a['tv_A0']), 'RB6': (a['top1_same_RB6'], a['tv_RB6']),
            'STAB01-K3': (a['stab01_K3_top1_same'], a['stab01_K3_tv']),
            'STAB01-K5': (a['stab01_K5_top1_same'], a['stab01_K5_tv'])}
    flags = {'rb_improves_over_baseline': bool(h2['ci95'][0] > 0),
             'rb_beats_measured_bruteforce': None if k_c_gain is None else bool(k_c_gain['ci95'][0] > 0),
             'efficiency_gt_1': None if efficiency['ci95'] is None else bool(efficiency['ci95'][0] > 1),
             'note': 'descriptive; never a verdict input (H2 is empirical)'}
    labels = ('A0_0', 'A0_5', 'RB6_0', 'RB6_5', 'REF_100')
    return {'decisions': len(a['request_id']), 'games': len(boot.unique), 'arms': arms, 'r_rows': {'RB6': rows_ratio},
            'rb6_status': rb6,
            'stab01_reference': {f'K{k}': pv.primary(a[f'stab01_K{k}_top1_same'], a[f'stab01_K{k}_tv'], boot)
                                 for k in (1, 3, 5)},
            'secondary': {'h2_rb6_minus_a0_top1': h2,
                          'k_c': {'K_c': k_c, 'gain_rb6_minus_K_c': k_c_gain,
                                  'note': 'cheapest measured K in {3, 5} with K >= r_rows; not an equal-cost comparison'},
                          'vrf_seed5': vrf, 'efficiency': efficiency,
                          'gap_bins': gap_bins(a, sets, boot),
                          'kl': {'A0-MC6': kl_summary(a, 'A0', boot), 'RB6': kl_summary(a, 'RB6', boot)},
                          'q_level': {'A0-MC6': q_level(a, 'A0'), 'RB6': q_level(a, 'RB6'),
                                      'noise_ratio_quantiles': noise(a, labels)},
                          'rb_residual': residual(a, ('RB6_0', 'RB6_5', 'REF_100')),
                          'truncation_share': truncation(a, labels), 'resources': resources(a, labels)},
            'secondary_flags': flags, 'smoke_r_b': smoke_r_b}


def summarize_arm(a, expected, arm):
    """Pass C (RB12) or pass D (BF48) metrics; pass-M arrays supply A0 rows, REF bins and the RB6/A0 columns."""
    index = a['index'] if 'index' in a else np.arange(len(a['request_id']))
    require(np.array_equal(np.array([str(r) for r in expected['request_id'][index]]), a['request_id']),
            'decisions differ from pass M')
    boot = bootstrap(a['game_pk'])
    a0_rows = {s: expected[f'rows_A0_{s}'][index] for s in reg.PAIR}
    sets = {arm: (a[f'top1_same_{arm}'], a[f'tv_{arm}']),
            'A0-MC6': (expected['top1_same_A0'][index], expected['tv_A0'][index]),
            'RB6': (expected['top1_same_RB6'][index], expected['tv_RB6'][index])}
    merged = {**a, **{k: expected[k][index] for k in ('bin_eff', 'bin_q')}}
    a0_kl = {f'kl_A0_{s}': expected[f'kl_A0_{s}'][index] for s in reg.PAIR}
    labels = [f'{arm}_{s}' for s in reg.PAIR]
    return {'primary': pv.primary(a[f'top1_same_{arm}'], a[f'tv_{arm}'], boot), 'r_rows': r_rows(a, arm, a0_rows),
            'gap_bins': gap_bins(merged, sets, boot),
            'kl': kl_summary({**a, **a0_kl}, arm, boot), 'q_level': q_level(a, arm),
            'noise_ratio_quantiles': noise(a, labels), 'rb_residual': residual(a, labels),
            'truncation_share': truncation(a, labels), 'resources': resources(a, labels),
            'bootstrap': 'own draw table over these games' if arm == 'BF48' else 'pass-M draw table (same 150 games)'}


def rb6_vs_bruteforce(a, expected, mandatory):
    """Descriptive paired RB6 - BF48 top-1 agreement on the BF48 subsample (its own draw table); the section 5-7
    comparator when 5 < r_rows(RB6) <= 8."""
    index = a['index']
    gain = pv.paired_gain(expected['top1_same_RB6'][index], a['top1_same_BF48'], bootstrap(a['game_pk']))
    rows_ratio = mandatory['rb6_status']['r_rows']
    return {**gain, 'serves_as_k_c_comparator': bool(5 < rows_ratio <= 8), 'rb6_r_rows': rows_ratio}


def summarize_fhdp(a, expected):
    """FH-DP fidelity (descriptive): DP vs RB seeds 0/5/100 next to RB-RB on the same decisions."""
    index = a['index']
    boot = bootstrap(a['game_pk'])
    tau = reg.TAU
    out = {'decisions': len(index), 'games': len(boot.unique)}
    for label in ('RB6_0', 'RB6_5', 'REF_100'):
        out[f'dp_vs_{label}'] = {'top1_agreement': float(np.mean(a[f'dp_vs_{label}_top1_same'])),
                                 'mean_tv': float(np.mean(a[f'dp_vs_{label}_tv']))}
    laws = {label: [pv.law(expected[f'q_{label}'][k], expected['reference'][k], expected['mask'][k], tau) for k in index]
            for label in ('RB6_0', 'RB6_5', 'REF_100')}
    for s, t in (('RB6_0', 'RB6_5'), ('RB6_0', 'REF_100'), ('RB6_5', 'REF_100')):
        pairs = [pv.pair(x, y) for x, y in zip(laws[t], laws[s])]
        same = np.array([p[0] for p in pairs], dtype=float)
        tv = np.array([p[1] for p in pairs])
        out[f'rb_{s}_vs_{t}'] = {
            'top1_agreement': float(same.mean()), 'mean_tv': float(tv.mean()),
            'dp_minus_rb_agreement': pv.paired_gain(a[f'dp_vs_{s}_top1_same'], same, boot),
            'dp_minus_rb_tv': pv.paired_gain(a[f'dp_vs_{s}_tv'], tv, boot)}
    out['rows_per_decision_mean'] = float(np.mean(a['rows']))
    out['planner_ms_per_decision_mean'] = float(np.mean(a['ms']))
    out['note'] = 'FH-DP estimates a different quantity (frozen history, no cap); no ordering is claimed'
    return out


def summarize_confirmation(a, passing):
    boot = bootstrap(a['game_pk'])
    a0_rows = {s: a[f'rows_A0_{s}'] for s in reg.PAIR}
    labels = ['A0_0', 'A0_5', f'{passing}_0', f'{passing}_5']
    return {'A0-MC6': pv.primary(a['top1_same_A0'], a['tv_A0'], boot),
            passing: pv.primary(a[f'top1_same_{passing}'], a[f'tv_{passing}'], boot),
            'r_rows': r_rows(a, passing, a0_rows),
            'h2_minus_a0_top1': pv.paired_gain(a[f'top1_same_{passing}'], a['top1_same_A0'], boot),
            'kl': {'A0-MC6': kl_summary(a, 'A0', boot), passing: kl_summary(a, passing, boot)},
            'q_level': {'A0-MC6': q_level(a, 'A0'), passing: q_level(a, passing),
                        'noise_ratio_quantiles': noise(a, labels)},
            'rb_residual': residual(a, labels[2:]), 'truncation_share': truncation(a, labels),
            'resources': resources(a, labels), 'bootstrap': 'own draw table over the 50 confirmation games'}


# ---------------------------------------------------------------- command line

def memory_free_percent():
    """macOS memory_pressure report (read only, no option): the system-wide free percentage."""
    text = subprocess.run(['memory_pressure'], capture_output=True, text=True, check=True, timeout=60).stdout
    found = re.search(r'System-wide memory free percentage:\s*(\d+)%', text)
    require(found is not None, 'memory_pressure report unreadable')
    return int(found.group(1))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--local-config', type=Path, required=True)
    p.add_argument('--attempt', type=Path, required=True, help='<runs>/ML-MATRIX-20260924/EXP-STAB-02/attempt-NNN')
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--through', choices=reg.PASS_ORDER, help='run every pending pass up to this one')
    group.add_argument('--confirmation-select', action='store_true', help='record the 50 confirmation games (no planning)')
    group.add_argument('--confirmation', type=Path, metavar='PIN_JSON',
                       help='run the confirmation slice against the appended games pin')
    args = p.parse_args(argv)
    try:
        require(args.config.resolve() == (REPO / reg.CONFIG_PATH).resolve(), 'Use the registered config file')
        config = json.loads(args.config.read_text())
        reg.registration_gate(config)  # fail closed before anything else reads data
        reg.verify_unchanged_sources(REPO)
        reg.verify_registered_code(config, REPO)
        reg.verify_registered_commit(config, REPO)  # reviewed sources in the registered commit; Git provenance
        manifest = reg.verify_digest_manifest(config, REPO)  # external config digest pin
        attempt = assert_attempt_path(args.attempt)
        local = json.loads(args.local_config.read_text())
        identity = run_identity(config, args.config.resolve(), args.local_config.resolve())
        from run_ml_matrix import check_location, heavy_lock
        from run_ml_benchmark import validate_native_runtime
        root = check_location(local, attempt)
        require(attempt.parent.name == EXPERIMENT, 'attempt folders live in <runs>/ML-MATRIX-20260924/EXP-STAB-02')
        validate_native_runtime()
        free = memory_free_percent()
        require(free >= reg.COST['memory_pressure_free_min_percent'],
                f'memory_pressure free {free}% < {reg.COST["memory_pressure_free_min_percent"]}%: do not start')
    except (reg.RegistrationError, IntegrityFailure, ValueError, OSError, subprocess.SubprocessError) as refused:
        print('PLANNER_VARIANCE_REFUSED', f'{type(refused).__name__}: {refused}', flush=True)
        return EXIT_REFUSED
    runner = PassRunner(config, RealEnvironment(config, local), attempt, identity=identity,
                        lock=lambda: heavy_lock(root), log=lambda *parts: print(*parts, flush=True), manifest=manifest)
    try:
        if args.through:
            return runner.advance(args.through)
        if args.confirmation_select:
            return runner.confirm_select()
        return runner.confirm(json.loads(args.confirmation.read_text()))
    except (IntegrityFailure, reg.RegistrationError) as stopped:
        print('PLANNER_VARIANCE_STOPPED', f'{type(stopped).__name__}: {stopped}', flush=True)
        return EXIT_STOPPED
    except Exception as unexpected:  # fail closed: traceback, nothing citable
        traceback.print_exc()
        print('PLANNER_VARIANCE_STOPPED', f'{type(unexpected).__name__}: {unexpected}', flush=True)
        return EXIT_STOPPED


if __name__ == '__main__':
    raise SystemExit(main())
