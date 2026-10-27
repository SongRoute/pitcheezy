"""EXP-OPE-RES-01 (track A2) stage runner: A2-SM smoke, A2-S0 ... A2-S4 and unblind.

Every real-data command refuses (fail closed) unless configs/EXP-OPE-RES-01.json passes the registry
check with status REGISTERED, registered = true, preregistration_complete = true, execution enabled
and every registration-time input pin filled; the protected sources and the pinned new files are
byte-identical, HEAD's code is the registered execution source commit and the config is committed;
the external append-only digest manifest (outside the repository results/) holds the registered
config digest and every upstream stage seal; the output is a fresh directory under the pinned output
root (never inside results/); and the stage's hang guard would end before the next 06:30 KST
protection window. Each stage writes JSON outputs, SHA256SUMS and manifest.json (seeds, config
digest, git commit, seconds, peak RSS) and appends its seal to the digest manifest; downstream stages
verify the seals before reading. Deterministic integrity failures stop the stage (no new arm from its
outputs); statistical outcomes are recorded as they are.

Commands that read no data: ``check`` (config check, digest, protected sources) and ``pins`` (sha256
of the new files for the registration pin). Run from the repository root with the project venv.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import signal
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'scripts'))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pitchmdp import ope_decision as odc  # noqa: E402
from pitchmdp import ope_redesign_registry as reg  # noqa: E402
from pitchmdp import ope_sealed as osd  # noqa: E402
from pitchmdp import ope_surrogate as osg  # noqa: E402
from pitchmdp import ope_synthetic_a2 as osx  # noqa: E402
from pitchmdp.game import GameState  # noqa: E402
from pitchmdp.policy_artifacts import IntegrityError, _require  # noqa: E402

COMMANDS = ('check', 'pins', 'register-digest', 'record-review', 'register-addendum', 'smoke', 's0', 's1', 's2', 's3',
            'decide', 'unblind')
STAGE_OF = {'smoke': 'A2-SM', 's0': 'A2-S0', 's1': 'A2-S1', 's2': 'A2-S2', 's3': 'A2-S3', 'decide': 'A2-S4'}
CODE_PATHS = ('experiments', 'src', 'scripts')
GIB = 1024 ** 3


class HangGuardExceeded(BaseException):
    """Infrastructure stop: the registered hang guard elapsed (partial outputs preserved, not citable)."""


class MemoryGuardExceeded(BaseException):
    """Infrastructure stop: RSS cap or free-memory guard (the recorder keeps priority)."""


# ---------------------------------------------------------------- digest manifest (external, append-only)
def _canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


class DigestManifest:
    """Append-only JSONL outside results/: each entry carries the previous entry's sha256 (hash chain);
    entries are never rewritten. Kinds: config_digest, review_reference, smoke_record, stage_seal,
    bugfix_addendum."""
    KINDS = ('config_digest', 'review_reference', 'smoke_record', 'stage_seal', 'cell_seal', 'stage_failure',
             'bugfix_addendum')

    def __init__(self, path):
        self.path = Path(path)
        _require(self.path.is_absolute(), 'digest manifest path must be absolute')
        _require(not self.path.resolve().is_relative_to((REPO / 'results').resolve()),
                 'the digest manifest must live outside the repository results/')

    def entries(self):
        if not self.path.exists():
            return []
        out, prev = [], None
        for seq, line in enumerate(self.path.read_bytes().splitlines()):
            entry = json.loads(line)
            body = {k: v for k, v in entry.items() if k != 'sha256'}
            _require(entry.get('seq') == seq and entry.get('prev') == prev
                     and entry.get('sha256') == hashlib.sha256(_canonical(body).encode()).hexdigest(),
                     f'digest manifest entry {seq} tampered or out of order')
            out.append(entry)
            prev = entry['sha256']
        return out

    def append(self, kind, payload):
        _require(kind in self.KINDS, f'unknown manifest entry kind {kind!r}')
        entries = self.entries()
        body = {'seq': len(entries), 'time_utc': datetime.now(timezone.utc).isoformat(), 'kind': kind,
                'payload': payload, 'prev': entries[-1]['sha256'] if entries else None}
        entry = {**body, 'sha256': hashlib.sha256(_canonical(body).encode()).hexdigest()}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('ab') as stream:
            stream.write((_canonical(entry) + '\n').encode())
            stream.flush()
            os.fsync(stream.fileno())
        return entry

    def latest(self, kind, **match):
        found = [e for e in self.entries() if e['kind'] == kind
                 and all(e['payload'].get(k) == v for k, v in match.items())]
        return found[-1] if found else None


def require_config_digest(manifest, config):
    entry = manifest.latest('config_digest')
    _require(entry is not None and entry['payload'].get('digest') == reg.config_digest(config),
             'the registered config digest is missing from the digest manifest or differs (G-REG)')
    return entry


def require_seal(manifest, stage_id):
    """The latest seal of an upstream stage, re-verified against its output manifest bytes."""
    entry = manifest.latest('stage_seal', stage=stage_id)
    _require(entry is not None, f'upstream stage {stage_id} is not sealed in the digest manifest')
    out = Path(entry['payload']['output_dir'])
    _require(reg.file_sha256(out / 'manifest.json') == entry['payload']['manifest_sha256'],
             f'{stage_id} output manifest differs from its seal (deterministic integrity)')
    listed = json.loads((out / 'manifest.json').read_text())['artifact_sha256']
    for name, sha in listed.items():
        _require(reg.file_sha256(out / name) == sha, f'{stage_id} artifact {name} differs from its seal')
    return out, entry


# ---------------------------------------------------------------- sources, git and output location
def git(*args):
    return subprocess.check_output(['git', *args], cwd=REPO, text=True).strip()


def committed(path):
    path = Path(path).resolve()
    if not path.is_relative_to(REPO):
        return False
    try:
        blob = subprocess.check_output(['git', 'show', f'HEAD:{path.relative_to(REPO).as_posix()}'], cwd=REPO,
                                       stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return False
    return blob == path.read_bytes()


def new_file_hashes(root=REPO):
    return {rel: reg.file_sha256(Path(root) / rel) for rel in reg.NEW_FILES}


def code_pins(config, manifest=None):
    """The registered code identity, replaced by the latest recorded bug-fix addendum (design 6:
    a deterministic-integrity fix is recorded append-only with its commit, test and review)."""
    pins = config['registration_input_pins']
    out = {'execution_source_commit': pins['execution_source_commit'], 'new_files_sha256': pins['new_files_sha256']}
    if manifest is not None:
        fix = manifest.latest('bugfix_addendum')
        if fix is not None:
            out = {'execution_source_commit': fix['payload']['fix_commit'],
                   'new_files_sha256': fix['payload']['new_files_sha256']}
    return out


def enforce_sources(config, config_path, state=None, manifest=None):
    """Protected sources byte-identical to the base; new files equal their pins; HEAD's code equals the
    registered execution source commit (registration commits may follow); config committed."""
    reg.verify_protected_sources(REPO)
    pins = code_pins(config, manifest)
    _require(new_file_hashes() == pins['new_files_sha256'], 'new source/test files differ from their registration pins')
    source = pins['execution_source_commit']
    if state is None:
        state = {'head': git('rev-parse', 'HEAD'),
                 'ancestor': subprocess.call(['git', 'merge-base', '--is-ancestor', source, 'HEAD'], cwd=REPO,
                                             stderr=subprocess.DEVNULL) == 0,
                 'changed': [x for x in git('diff', '--name-only', source, 'HEAD', '--', *CODE_PATHS).splitlines() if x],
                 'dirty': bool(git('status', '--porcelain', '--', *CODE_PATHS)),
                 'config_committed': committed(config_path)}
    _require(state['ancestor'] and not state['changed'], 'code at HEAD differs from the registered execution source commit')
    _require(not state['dirty'], 'uncommitted changes under the code paths; refusing to run')
    _require(state['config_committed'], 'the registered config is not committed at HEAD byte for byte')
    return state


def check_output(output, config):
    root = Path(config['registration_input_pins']['output_root']).resolve()
    output = Path(output).resolve()
    _require(not output.is_relative_to((REPO / 'results').resolve()), 'outputs never go to the repository results/')
    _require(output.parent == root and not output.exists(), f'output must be a new directory directly under {root}')
    return output


# ---------------------------------------------------------------- guards
def peak_rss_bytes():
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(raw if platform.system() == 'Darwin' else raw * 1024)


def free_memory_bytes():
    """Free + inactive + speculative pages (macOS vm_stat) or MemAvailable (Linux); None if unknown."""
    try:
        if platform.system() == 'Darwin':
            text = subprocess.check_output(['vm_stat'], text=True)
            size = int(text.split('page size of')[1].split()[0])
            pages = {k.strip(): int(v.strip().rstrip('.')) for k, v in (
                line.split(':', 1) for line in text.splitlines()[1:] if ':' in line)}
            return size * sum(pages.get(k, 0) for k in ('Pages free', 'Pages inactive', 'Pages speculative'))
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError, subprocess.CalledProcessError):
        return None
    return None


class Guards:
    """Checkpoint guard: RSS cap (8 GiB) and free memory (>= 4 GiB) checked at every call; the hang
    guard itself is a SIGALRM timer armed by ``stage``."""
    def __init__(self, *, rss_cap=reg.RSS_CAP_GIB * GIB, free_min=reg.FREE_MEMORY_GIB * GIB, rss=peak_rss_bytes,
                 free=free_memory_bytes):
        self.rss_cap, self.free_min, self.rss, self.free = rss_cap, free_min, rss, free

    def __call__(self):
        if self.rss() > self.rss_cap:
            raise MemoryGuardExceeded('peak RSS above the registered cap')
        free = self.free()
        if free is not None and free < self.free_min:
            raise MemoryGuardExceeded('free memory below the registered guard; stopping to protect the recorder')


def _raise_hang(signum, frame):
    raise HangGuardExceeded('registered hang guard exceeded; partial outputs preserved, not citable')


def sha256sums(output):
    names = sorted(str(p.relative_to(output)) for p in output.rglob('*') if p.is_file())
    return {name: reg.file_sha256(output / name) for name in names}


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=1, ensure_ascii=False, allow_nan=False, default=_default) + '\n')


def _default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(f'not JSON serializable: {type(value).__name__}')


@contextmanager
def stage(output, stage_id, identity, *, manifest, now=None, guard_seconds=None, alarm=True):
    """Fresh stage directory with the registered hang guard; seals on success, preserves on failure.
    A stage starts only if its hang guard would end outside every 06:30-13:30 KST window."""
    guard_seconds = reg.HANG_GUARDS[stage_id] if guard_seconds is None else guard_seconds
    start = now() if now is not None else datetime.now(timezone.utc)
    _require(reg.stage_window_ok(start, reg.HANG_GUARDS[stage_id]),
             f'{stage_id} would overlap the 06:30-13:30 KST protection window within its hang guard; not started')
    output = Path(output)
    _require(not output.exists(), f'stage output exists; preserve it and use a new attempt directory: {output}')
    output.mkdir(parents=True)
    t0, cpu0 = time.perf_counter(), time.process_time()
    dump(output / 'started.json', {'stage': stage_id, 'started_utc': start.isoformat(), **identity})
    previous = None
    if alarm:
        previous = signal.signal(signal.SIGALRM, _raise_hang)
        signal.setitimer(signal.ITIMER_REAL, float(guard_seconds))
    try:
        yield output
    except BaseException as error:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        kind = ('infrastructure' if isinstance(error, (HangGuardExceeded, MemoryGuardExceeded))
                else 'deterministic_integrity' if isinstance(error, IntegrityError) else 'error')
        failure = output / f'failure-{stamp}.json'
        dump(failure, {'stage': stage_id, 'class': kind, 'error_type': type(error).__name__,
                       'error': str(error), 'seconds': time.perf_counter() - t0,
                       'peak_rss_bytes': peak_rss_bytes(), 'citable': False, 'partial_outputs_preserved': True})
        manifest.append('stage_failure', {'stage': stage_id, 'class': kind, 'output_dir': str(output),
                                          'failure_sha256': reg.file_sha256(failure)})
        raise
    finally:
        if alarm:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
    sums = sha256sums(output)
    (output / 'SHA256SUMS').write_text(''.join(f'{sha}  {name}\n' for name, sha in sums.items()))
    dump(output / 'manifest.json', {'stage': stage_id, 'cost': {
        'wall_seconds': time.perf_counter() - t0, 'cpu_seconds': time.process_time() - cpu0,
        'peak_rss_bytes': peak_rss_bytes()}, 'artifact_sha256': {**sums, 'SHA256SUMS': reg.file_sha256(output / 'SHA256SUMS')},
        **{k: identity[k] for k in ('config_digest', 'git_commit', 'seeds') if k in identity}})
    manifest.append('stage_seal', {'stage': stage_id, 'output_dir': str(output),
                                   'manifest_sha256': reg.file_sha256(output / 'manifest.json')})


def identity_record(config, extra=None):
    return {'experiment': reg.EXPERIMENT_ID, 'config_digest': reg.config_digest(config),
            'git_commit': git('rev-parse', 'HEAD'),
            'seeds': {'game_bootstrap': reg.GAME_BOOTSTRAP_SEED, 'a2_base': reg.A2_BASE, 'roles': reg.ROLE_VALUES,
                      'world_seeds': {r['k']: reg.world_seed(r['k']) for r in reg.S5X_RUNS}},
            'python': sys.executable, 'python_version': platform.python_version(), 'numpy': np.__version__,
            'pandas': pd.__version__, **(extra or {})}


# ================================================================ A2-S0 (pure on TRAIN rows)
TAB_OUTCOME_OF_10 = np.array([0, 1, 2, 4, 4, 4, 4, 4, 3, 4])  # ball, strike, foul, in play x5, hbp, double play


def _bases(frame):
    if 'bases' in frame:
        return frame['bases'].to_numpy(np.int64)
    return sum((frame[c].notna().to_numpy().astype(np.int64) << i) for i, c in enumerate(('on_1b', 'on_2b', 'on_3b')))


def _groups(frame, names):
    return osg.group_of_types(frame['pitch_type'].to_numpy(object), names)


def tab_marginals_from_train(train, game_values, *, cells, classes, index, table, signs, signs_prime, kappa_p, zeta_p,
                             first_half, second_half, seed_role='tab_world',
                             min_pitches=reg.IMPLEMENTATION['tab_world']['archetype_min_train_pitches']):
    """TRAIN marginals of the TAB world (A2-S0): pitch outcome frequencies, logging-law inputs, the
    eight archetypes and 64 situations (fixed seed role draws), x laws and the in-play mix."""
    from pitchmdp.model import outcome_labels
    groups = reg.TAB_GROUPS
    g = _groups(train, groups)
    labels = outcome_labels(train)
    count = train['balls'].to_numpy(np.int64) * 3 + train['strikes'].to_numpy(np.int64)
    ok = (g != None) & (labels >= 0) & (train['balls'].between(0, 3) & train['strikes'].between(0, 2)).to_numpy()  # noqa: E711
    gi = np.array([groups.index(x) if x is not None else -1 for x in g])
    outcome = TAB_OUTCOME_OF_10[np.maximum(labels, 0)]
    freq = np.zeros((osx.N_COUNTS, len(groups), 5))
    np.add.at(freq, (count[ok], gi[ok], outcome[ok]), 1.)
    pooled = freq.sum(0)  # per group over counts: the fallback of an empty count x group cell
    _require((pooled.sum(-1) > 0).all(), 'every TAB pitch group needs TRAIN pitches')
    empty = freq.sum(-1) == 0
    outcome_freq = np.where(empty[..., None], pooled[None] / pooled.sum(-1)[None, :, None],
                            freq / np.maximum(freq.sum(-1, keepdims=True), 1))
    cg = freq.sum(-1)
    count_group = cg / cg.sum(-1, keepdims=True)
    league = cg.sum(0) / cg.sum()
    pitchers = train['pitcher'].to_numpy()
    per = pd.DataFrame({'pitcher': pitchers[ok], 'g': gi[ok]}).groupby(['pitcher', 'g']).size().unstack(fill_value=0)
    per = per.reindex(columns=range(len(groups)), fill_value=0)
    eligible = per.loc[per.sum(axis=1) >= min_pitches]
    _require(len(eligible) >= reg.TAB_ARCHETYPES, 'not enough TRAIN pitchers for the TAB archetypes')
    order = sorted(eligible.index, key=str)
    pick = np.random.default_rng(reg.role_sequence(seed_role, 0, 1)).choice(len(order), reg.TAB_ARCHETYPES, replace=False)
    shares = np.array([eligible.loc[order[i]].to_numpy(float) / eligible.loc[order[i]].sum() for i in pick])
    first = train.sort_values(['game_pk', 'at_bat_number', 'pitch_number']).drop_duplicates(['game_pk', 'at_bat_number'])
    rows = np.random.default_rng(reg.role_sequence(seed_role, 0, 0)).choice(len(first), reg.TAB_SITUATIONS, replace=False)
    situations = [osx.Situation.from_models(GameState.from_row(first.iloc[int(i)]), game_values['we'],
                                            game_values['advancement']) for i in rows]
    x_dist, mix = x_laws(train, cells, index, reg.S5X_GROUPS)
    return {'groups': list(groups), 'outcome_freq': outcome_freq, 'count_group_freq': count_group,
            'league_shares': league, 'archetype_shares': shares, 'x_dist': x_dist[:len(groups)],
            'bip_mix': mix[:len(groups)] / mix[:len(groups)].sum(), 'class_table': table,
            'first_half_table': first_half, 'second_half_table': second_half,
            'situations': [situation_doc(s) for s in situations],
            'signs': [signs[x] for x in groups], 'signs_prime': [signs_prime[x] for x in groups],
            'kappa_p': kappa_p, 'zeta_p': zeta_p, 'n_train_bip': int(len(cells)),
            'outcome_cells_pooled_fallback': int(empty.sum())}


def x_laws(train, cells, index, names):
    """P(x | group, two strikes) [G, 2, 480] and the (group, two strikes, dp) counts of the in-play rows.
    A (group, two strikes) cell without TRAIN in-play rows falls back to the two-strike-pooled law of all
    groups (reported in the census)."""
    g = _groups(train.iloc[index], names)
    gi = np.array([names.index(x) if x is not None else -1 for x in g])
    ts = (train['strikes'].to_numpy()[index] == 2).astype(int)
    keep = gi >= 0
    counts = np.zeros((len(names), 2, reg.N_X))
    np.add.at(counts, (gi[keep], ts[keep], cells[keep] // 2), 1.)
    mix = np.zeros((len(names), 2, 2))
    np.add.at(mix, (gi[keep], ts[keep], cells[keep] % 2), 1.)
    total = counts.sum(-1, keepdims=True)
    pooled = counts.sum(0)  # [2, 480] over groups
    _require((pooled.sum(-1) > 0).all(), 'TRAIN in-play rows are needed with and without two strikes')
    law = np.where(total > 0, counts / np.maximum(total, 1), (pooled / pooled.sum(-1, keepdims=True))[None])
    return law, mix


def situation_doc(s):
    st = s.state
    return {'state': {'inning': st.inning, 'half': st.half, 'outs': st.outs, 'bases': st.bases,
                      'home_score': st.home_score, 'away_score': st.away_score},
            'outcomes': {e: [list(o) for o in s.outcomes[e]] for e in reg.TERMINALS9}}


def situation_from_doc(doc):
    st = doc['state']
    state = GameState(st['inning'], st['half'], st['outs'], st['bases'], st['home_score'], st['away_score'])
    return osx.Situation(state, {e: [tuple(o) for o in rows] for e, rows in doc['outcomes'].items()})


def marginals_from_doc(doc):
    values = {k: doc[k] for k in osx.TabMarginals.FIELDS}
    values['situations'] = [situation_from_doc(s) for s in doc['situations']]
    for k in ('outcome_freq', 'count_group_freq', 'league_shares', 'archetype_shares', 'x_dist', 'bip_mix', 'class_table',
              'first_half_table', 'second_half_table'):
        values[k] = np.asarray(values[k], dtype=np.float64)
    values['groups'] = tuple(values['groups'])
    return osx.TabMarginals(**values)


def zeta_inputs(train, cells, classes, index, game_values, names):
    """Per eligible TRAIN out (runner on third, < 2 outs): actual runs - C0 expectation, the C0 scoring
    mass and run gap (zeta calibration), with the pitch group."""
    rows = train.iloc[index]
    bases, outs = _bases(rows), rows['outs_when_up'].to_numpy()
    sel = (classes == osg.OUT_INDEX) & ((bases & 4) > 0) & (outs < 2)
    part = rows.loc[sel]
    groups = _groups(part, names)
    cache, residual, p_score, run_gap = {}, [], [], []
    for (_, row), b in zip(part.iterrows(), bases[sel]):
        state = GameState(int(row.inning), str(row.inning_topbot), int(row.outs_when_up), int(b), int(row.home_score),
                          int(row.away_score))
        if state not in cache:
            outcomes, p, gap = osg.out_scoring_split(state, game_values['advancement'], game_values['we'])
            cache[state] = (sum(q * r for q, r, _ in outcomes), p, gap)
        expected, p, gap = cache[state]
        bat = (row.post_home_score - row.home_score) if row.inning_topbot == 'Bot' else (row.post_away_score - row.away_score)
        residual.append(float(bat) - expected)
        p_score.append(p)
        run_gap.append(gap)
    return np.array(residual), np.array(p_score), np.array(run_gap), groups


SMOKE_MIN_PITCHES = 50  # the smoke's short TRAIN subset: archetype eligibility only (outputs discarded)


def build_s0(train, game_values, *, smoke=False, seed_role='tab_world', fixed_m=None,
             min_pitches=reg.IMPLEMENTATION['tab_world']['archetype_min_train_pitches']):
    """A2-S0 on guarded TRAIN rows (raw columns already joined). ``smoke``: the same code path with a
    fixed m, no cross-validated log-loss, no delta_g, kappa_p or zeta_p (zero signs), outputs discarded."""
    timings = {}
    t = time.perf_counter()
    osg.guard_train(train['game_date'])
    cells, classes, index, census = osg.train_bip_rows(train)
    folds = (train['game_pk'].to_numpy()[index] % 2).astype(np.int64)
    timings['rows_and_mapping'] = time.perf_counter() - t
    t = time.perf_counter()
    fit = osg.fit_r1b(cells, classes, folds, m=fixed_m if smoke else None)
    timings['table_fit'] = time.perf_counter() - t
    m, table = fit['m'], fit['table']
    dates = pd.to_datetime(train['game_date']).to_numpy()[index]
    median = np.sort(dates)[len(dates) // 2] if len(dates) else None
    halves = [dates <= median, dates > median]
    _require(all(h.any() for h in halves), 'both TRAIN halves need in-play rows (era drift cell)')
    half_tables = [osg.fit_table(osg.class_counts(cells[h], classes[h]), m) for h in halves]
    out = {'census': census, 'm': m, 'cv_log_loss': fit['cv_log_loss'], 'r1_table': table,
           'r1_table_sha256': osg.table_sha256(table, m), 'timings': timings}
    zero = {g: 0 for g in reg.S5X_GROUPS}
    if smoke:
        signs, signs_prime, kappa, zeta = zero, zero, {'S5X': 0.0, 'TAB': 0.0}, {'S5X': 0.0, 'TAB': 0.0}
    else:
        groups = _groups(train.iloc[index], reg.S5X_GROUPS)
        residual = osg.crossfit_hit_residuals(cells, classes, folds, groups, m)
        delta = osg.group_residuals(residual, groups, reg.S5X_GROUPS)
        hit = osg.hit_probability(table, cells)
        z_res, p_score, run_gap, z_groups = zeta_inputs(train, cells, classes, index, game_values, reg.S5X_GROUPS)
        delta_prime = osg.group_residuals(z_res, z_groups, reg.S5X_GROUPS)
        kappa, zeta = {}, {}
        for world, names in (('S5X', reg.S5X_GROUPS), ('TAB', reg.TAB_GROUPS)):
            kappa[world] = osg.calibrate_kappa(hit, groups, delta, names)
            zeta[world] = osg.calibrate_zeta(p_score, run_gap, z_groups, delta_prime, names)
        signs = kappa['S5X']['signs']
        signs_prime = zeta['S5X']['signs']
        out.update(train_residuals={'hit': delta, 'run': delta_prime}, kappa_calibration=kappa, zeta_calibration=zeta)
        kappa = {w: v['kappa_p'] for w, v in kappa.items()}
        zeta = {w: v['zeta_p'] for w, v in zeta.items()}
    out.update(signs=signs, signs_prime=signs_prime, kappa_p=kappa, zeta_p=zeta)
    t = time.perf_counter()
    out['tab_marginals'] = tab_marginals_from_train(
        train, game_values, cells=cells, classes=classes, index=index, table=table, signs=signs,
        signs_prime=signs_prime, kappa_p=kappa['TAB'], zeta_p=zeta['TAB'], first_half=half_tables[0],
        second_half=half_tables[1], seed_role=seed_role, min_pitches=min_pitches)
    x_dist, _ = x_laws(train, cells, index, reg.S5X_GROUPS)
    out['s5x_layers'] = {'groups': list(reg.S5X_GROUPS), 'x_dist': x_dist, 'class_table': table,
                         'signs': [signs[g] for g in reg.S5X_GROUPS], 'signs_prime': [signs_prime[g] for g in reg.S5X_GROUPS],
                         'kappa_p': kappa['S5X'], 'zeta_p': zeta['S5X']}
    timings['marginals'] = time.perf_counter() - t
    return out


# ================================================================ A2-S1 TAB (pure on sealed A2-S0 outputs)
def stage_s1(s0, *, write_cell, deadline=lambda: None, cells=None, replications=None):
    """Acceptance cells first (each written and sealed on completion), then the descriptive cells in
    the registered order; the first step checks every V2 cell's baseline DP expectation = DP truth."""
    marginals = marginals_from_doc(s0['tab_marginals'])
    eta_p = osx.TabWorld.eta_p(marginals)
    cells = reg.tab_cells() if cells is None else cells
    worlds = lambda cell: osx.TabWorld(marginals, law=cell['law'], eta=reg.scale(cell['eta'], eta_p),
                                       kappa=reg.scale(cell['kappa'], marginals.kappa_p),
                                       zeta=reg.scale(cell['zeta'], marginals.zeta_p), drift=cell['drift'])
    integrity = {}
    for cell in cells:
        if cell['law'] == 'V2':
            world = worlds(cell)
            osx.tab_integrity(world)
            integrity[cell['id']] = {'expected_delta_hat_0_minus_truth': world.expected_delta_hat_0 - world.delta_true}
    results = {'eta_p': eta_p, 'first_step_v2_dp_check': integrity, 'cells': {}, 'not_run': []}
    acceptance_done = False
    for cell in cells:
        if cell['role'] != 'acceptance' and not acceptance_done:
            acceptance_done = True
        try:
            record = osx.tab_cell(worlds(cell), cell, n_train=marginals.n_train_bip, replications=replications,
                                  deadline=deadline)
        except HangGuardExceeded:
            if cell['role'] == 'acceptance':
                raise  # before every acceptance cell is sealed: an infrastructure failure
            results['not_run'] = [c['id'] for c in cells if c['index'] >= cell['index']]
            break
        results['cells'][cell['id']] = record
        write_cell(cell, record)
    results['acceptance'] = {c['id']: results['cells'][c['id']]['pairs'] for c in reg.TAB_ACCEPTANCE
                             if c['id'] in results['cells']}
    return results


# ================================================================ A2-S4 decision (pure)
def stage_s4(public, tab, s5x, *, integrity_failures, r3_status):
    odc.check_inputs(public, tab, s5x)
    return odc.decide(public, tab, s5x, integrity_failures=integrity_failures, r3_status=r3_status)


def s5x_summary(runs):
    """S-1/S-4 nets, S-2 cells and the gamma_a invariant from the S5-X run records."""
    nets = osx.nets(runs)
    by_k = {r['run']['k']: r for r in runs}
    s2 = {}
    for comp in ('R1_a', 'R1_b'):
        s2[comp] = {}
        for label, k in (('2*kappa_p', 4), ('2*zeta_p', 5)):
            entry = (by_k.get(k) or {}).get('s2') or {}
            s2[comp][label] = entry.get(comp) or {'outcome': 'INVALID'}
    invariant_runs = [by_k[k]['invariant'] for k in (4, 9, 10) if k in by_k and by_k[k].get('invariant')]
    s2['invariant'] = 'PASS' if invariant_runs and all(r['outcome'] == 'PASS' for r in invariant_runs) else 'INVALID'
    return {'nets': nets, 's2': s2}


# ================================================================ A2-SM smoke (pure parts)
def smoke_tab(s0_subset, *, replications, guards=lambda: None):
    """A2-S1 code path at kappa = zeta = 0, eta = 0, seed role smoke: integrity checks (V2 DP expectation
    = truth, same-seed bit reproducibility) and wall per replication; every estimate is discarded."""
    choice = reg.IMPLEMENTATION['smoke']
    marginals = marginals_from_doc(s0_subset['tab_marginals'])
    t = time.perf_counter()
    world = osx.TabWorld(marginals, law='V2', eta=choice['eta'], kappa=choice['kappa'], zeta=choice['zeta'],
                         seed_role='smoke')
    build = time.perf_counter() - t
    osx.tab_integrity(world)
    fixed = {'m': choice['fixed_m'], 'lambda': choice['fixed_lambda'], 'M': choice['fixed_M']}
    walls = []
    first = None
    for j in range(replications):
        guards()
        t = time.perf_counter()
        estimates, _, _ = osx.tab_replication(world, 0, j, n_train=marginals.n_train_bip, smoke=fixed)
        walls.append(time.perf_counter() - t)
        if j == 0:
            first = estimates
    again, _, _ = osx.tab_replication(world, 0, 0, n_train=marginals.n_train_bip, smoke=fixed)
    reproducible = json.dumps(first, sort_keys=True) == json.dumps(again, sort_keys=True)
    _require(reproducible, 'smoke TAB replication is not bit-reproducible with the same seed (deterministic integrity)')
    rng = np.random.default_rng(reg.role_sequence('smoke', 2, 0))
    data = world.simulate(rng)
    steps, lengths = data['steps'], data['lengths']
    t = time.perf_counter()
    f = osx._transform_array('shrink', choice['fixed_lambda'])
    from pitchmdp.ope_diagnostics import affine_padded
    affine_padded(steps['v_c'], steps['q_a'], f(steps['rho_c']), lengths)
    affine_padded(steps['v_r'], steps['q_a'], f(np.where(np.arange(steps['v_r'].shape[1])[None] < lengths[:, None], 1., 0.)),
                  lengths)
    variant_unit = time.perf_counter() - t
    return {'world_build_seconds': build, 'replication_seconds': walls, 'variant_estimate_unit_seconds': variant_unit,
            'integrity': {'tab_v2_dp_expectation_equals_truth': 'PASS', 'same_seed_bit_reproducible': 'PASS'},
            'replications': replications}


def smoke_projection(smoke_s0_seconds, table_fit_unit, variant_unit, replication_seconds, train_row_ratio):
    """Labelled projection (not a measurement): the excluded selection workload per replication
    (6 table fits + 30 variant estimates at the smoke unit costs) added before extrapolating."""
    per_rep = float(np.mean(replication_seconds)) if replication_seconds else math.nan
    term = 6 * table_fit_unit + 30 * variant_unit
    projected_s1 = (per_rep + term) * sum(c['R'] for c in reg.TAB_ACCEPTANCE)
    projected_s0 = (smoke_s0_seconds + 6 * table_fit_unit) * train_row_ratio
    return {'label': 'projection (not a measurement)', 'projection_term_seconds_per_replication': term,
            'A2-S1_acceptance_seconds': projected_s1, 'A2-S0_seconds': projected_s0,
            'within_guard': {'A2-S1': bool(projected_s1 <= reg.HANG_GUARDS['A2-S1']),
                             'A2-S0': bool(projected_s0 <= reg.HANG_GUARDS['A2-S0'])},
            'rule': 'a projection above its hang guard -> that stage is not started (decision needed); guards unchanged'}


# ================================================================ real-data glue (registered runs only)
def registered_context(config_path):
    """Registry gate (fail closed) + sources + digest manifest. Nothing touches data before this."""
    config, _ = reg.load_config(config_path)
    reg.require_executable(config)
    manifest = DigestManifest(config['registration_input_pins']['digest_manifest_path'])
    require_config_digest(manifest, config)
    state = enforce_sources(config, config_path, manifest=manifest)
    return config, manifest, state


def attempt_gate(manifest, stage_id):
    """Registered attempts (design 6): a sealed stage never reruns; at most 2 attempts per stage; a
    retry after a deterministic-integrity failure needs a bug-fix addendum recorded after it; two
    deterministic failures -> FAILED_INTEGRITY; two infrastructure failures -> decision needed."""
    entries = manifest.entries()
    _require(not any(e['kind'] == 'stage_seal' and e['payload']['stage'] == stage_id for e in entries),
             f'{stage_id} is already sealed; a registered stage is not rerun')
    failures = [e for e in entries if e['kind'] == 'stage_failure' and e['payload']['stage'] == stage_id]
    integrity = [e for e in failures if e['payload']['class'] == 'deterministic_integrity']
    infra = [e for e in failures if e['payload']['class'] != 'deterministic_integrity']
    _require(len(integrity) < 2, f'{stage_id}: two deterministic-integrity failures -> FAILED_INTEGRITY (no further attempt)')
    _require(len(infra) < 2, f'{stage_id}: two infrastructure failures -> decision needed (Song)')
    _require(len(failures) < 2, f'{stage_id}: at most two attempts per stage')
    if integrity:
        last = integrity[-1]['seq']
        _require(any(e['kind'] == 'bugfix_addendum' and e['seq'] > last and e['payload'].get('stage') == stage_id
                     for e in entries), f'{stage_id}: a retry after an integrity failure needs a recorded bug-fix addendum')
    return {'failures': len(failures), 'integrity_failures': len(integrity)}


def smoke_gate(manifest, stage_id):
    """G-SMOKE: A2-S0 and A2-S1 start only if the smoke's projected cost (projection term included)
    fits their hang guard; otherwise the stage is not started (decision needed; guards unchanged)."""
    entry = manifest.latest('smoke_record')
    _require(entry is not None, 'G-SMOKE: no smoke record in the digest manifest')
    _require(entry['payload']['projection']['within_guard'].get(stage_id) is True,
             f'G-SMOKE: the projected cost of {stage_id} exceeds its hang guard; not started (decision needed)')
    return entry


def failed_integrity(manifest, stages=('A2-S0', 'A2-S1', 'A2-S2', 'A2-S3')):
    """Stages that ended unresolved: two deterministic-integrity failures and no seal."""
    entries = manifest.entries()
    out = []
    for stage_id in stages:
        sealed = any(e['kind'] == 'stage_seal' and e['payload']['stage'] == stage_id for e in entries)
        integrity = sum(e['kind'] == 'stage_failure' and e['payload']['stage'] == stage_id
                        and e['payload']['class'] == 'deterministic_integrity' for e in entries)
        if not sealed and integrity >= 2:
            out.append(stage_id)
    return out


def policy_registration(config):
    """The ML-POLICY registration chain recorded by the pinned S6-dr-a5 stage (started.json), with every
    file re-hashed; then run_policy_validation's loaders (imported only here)."""
    import run_policy_validation as rpv
    pins = config['registration_input_pins']
    ledger = Path(pins['s6_dr_a5_ledger_path'])
    stage_dir = ledger.parent
    _require(reg.file_sha256(stage_dir / 'manifest.json') == pins['s6_dr_a5_manifest_sha256'],
             'S6-dr-a5 manifest differs from its registration pin')
    _require(reg.file_sha256(ledger) == pins['s6_dr_a5_ledger_sha256'], 'S6-dr-a5 ledger differs from its pin')
    listed = json.loads((stage_dir / 'manifest.json').read_text())['artifact_sha256']
    for name in ('started.json', 'dr.json', ledger.name):
        _require(reg.file_sha256(stage_dir / name) == listed[name], f'S6-dr-a5 {name} differs from its stage manifest')
    started = json.loads((stage_dir / 'started.json').read_text())
    chain = started['registration_chain']
    resolve = lambda path: Path(path) if Path(path).is_absolute() else REPO / path  # the runner's CLI is run from the root
    for link in chain:
        _require(reg.file_sha256(resolve(link['path'])) == link['sha256'],
                 f'ML-POLICY registration link changed: {link["path"]}')
    policy = rpv.load_registration(resolve(chain[0]['path']), [resolve(link['path']) for link in chain[1:]])
    return rpv, policy, stage_dir, json.loads((stage_dir / 'dr.json').read_text())


def load_game_values(rpv, policy, inputs):
    from pitchmdp import policy_identity as pid
    ident = policy['config']['identity_registration']
    values, _ = pid.load_pinned_we(inputs['paths'], ident['we_contract_sha256'], ident['classes']['we'])
    return values


def raw_quality_columns(local, pins, keys_frame, date_range):
    """hc_x, hc_y, bb_type, launch_speed, launch_angle from the three approved raw files (sha256 pinned),
    only rows dated inside ``date_range``; join keys must be unique."""
    import pyarrow.parquet as pq
    frames = []
    for name in reg.RAW_ALLOWLIST:
        _require('2026' not in name, 'a 2026 path reached the raw join; stopping')
        path = Path(local['raw_root']) / name
        _require(reg.file_sha256(path) == pins['train_raw_files_sha256'][name], f'raw file {name} differs from its pin')
        cols = ['game_date', 'game_pk', 'at_bat_number', 'pitch_number', 'hc_x', 'hc_y', 'bb_type', 'launch_speed',
                'launch_angle']
        table = pq.read_table(path, columns=cols).to_pandas()
        dates = pd.to_datetime(table['game_date']).dt.normalize()
        frames.append(table.loc[(dates >= pd.Timestamp(date_range[0])) & (dates <= pd.Timestamp(date_range[1]))])
    raw = pd.concat(frames, ignore_index=True)
    key = ['game_pk', 'at_bat_number', 'pitch_number']
    _require(not raw.duplicated(key).any(), 'raw join keys are not unique (deterministic integrity)')
    wanted = keys_frame[key].merge(raw, on=key, how='left', validate='one_to_one')
    return wanted


def train_rows(rpv, policy, local, pins, date_range):
    """Guarded TRAIN rows of the processed frame with the raw quality columns joined (TRAIN dates only)."""
    inputs = rpv.load_inputs(policy['config'], local, store=False)
    frame = inputs['frame']
    dates = pd.to_datetime(frame['game_date']).dt.normalize()
    keep = frame['split'].eq('train').to_numpy() & (dates >= pd.Timestamp(date_range[0])).to_numpy() & (
        dates <= pd.Timestamp(date_range[1])).to_numpy()
    train = frame.loc[keep].reset_index(drop=True)
    joined = raw_quality_columns(local, pins, train, date_range)
    for column in ('hc_x', 'hc_y', 'bb_type'):
        train[column] = joined[column].to_numpy()
    for column in ('launch_speed', 'launch_angle'):
        if column not in train:
            train[column] = joined[column].to_numpy()
    osg.guard_train(train['game_date'])
    return inputs, train


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=COMMANDS)
    parser.add_argument('--config', type=Path, default=REPO / reg.CONFIG_PATH)
    parser.add_argument('--local-config', type=Path, default=PROJECT / 'configs' / 'local.json')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--expect-digest')
    parser.add_argument('--decision-record', type=Path)
    parser.add_argument('--stage')
    parser.add_argument('--failing-test')
    parser.add_argument('--review')
    parser.add_argument('--gate')
    args = parser.parse_args(argv)
    if args.command == 'check':
        config, sha = reg.load_config(args.config)
        print(json.dumps({'status': config['status'], 'config_file_sha256': sha, 'config_digest': reg.config_digest(config),
                          'protected_sources': 'unchanged' if reg.verify_protected_sources(REPO) else None}, indent=1))
        return 0
    if args.command == 'pins':
        print(json.dumps({'new_files_sha256': new_file_hashes(), 'protected_sources': reg.PROTECTED_SOURCES}, indent=1))
        return 0
    if args.command == 'register-digest':
        config, _ = reg.load_config(args.config)
        reg.require_executable(config)
        digest = reg.config_digest(config)
        _require(args.expect_digest == digest, 'registered config digest differs from the G-CODE reviewed digest')
        manifest = DigestManifest(config['registration_input_pins']['digest_manifest_path'])
        _require(manifest.latest('config_digest') is None, 'the config digest is registered once')
        print(json.dumps(manifest.append('config_digest', {'digest': digest, 'config_path': str(args.config)}), indent=1))
        return 0
    if args.command == 'register-addendum':
        return register_addendum(args)
    if args.command == 'record-review':
        config, _ = reg.load_config(args.config)
        reg.require_executable(config)
        manifest = DigestManifest(config['registration_input_pins']['digest_manifest_path'])
        require_config_digest(manifest, config)
        _require(args.gate in ('G-REV', 'G-CODE', 'G-REG') and args.review, '--gate (G-REV, G-CODE or G-REG) and --review')
        print(json.dumps(manifest.append('review_reference', {'gate': args.gate, 'reference': args.review}), indent=1))
        return 0
    config, manifest, state = registered_context(args.config)  # fail closed before any data
    return run_registered(args, config, manifest, state)


def register_addendum(args):
    """Append a bug-fix addendum after a deterministic-integrity failure: the fix commit (HEAD, clean
    tree), the new file hashes, the failing test and the different-model review reference."""
    config, _ = reg.load_config(args.config)
    reg.require_executable(config)
    manifest = DigestManifest(config['registration_input_pins']['digest_manifest_path'])
    require_config_digest(manifest, config)
    _require(args.stage in reg.STAGES and args.failing_test and args.review, '--stage, --failing-test and --review required')
    _require(any(e['kind'] == 'stage_failure' and e['payload']['stage'] == args.stage
                 and e['payload']['class'] == 'deterministic_integrity' for e in manifest.entries()),
             'a bug-fix addendum follows a recorded deterministic-integrity failure of that stage')
    _require(not git('status', '--porcelain', '--', *CODE_PATHS), 'commit the fix first (clean code paths)')
    reg.verify_protected_sources(REPO)
    entry = manifest.append('bugfix_addendum', {'stage': args.stage, 'fix_commit': git('rev-parse', 'HEAD'),
                                                'new_files_sha256': new_file_hashes(), 'failing_test': args.failing_test,
                                                'review_reference': args.review})
    print(json.dumps(entry, indent=1))
    return 0


def run_registered(args, config, manifest, state):
    """Registered stages. Each verifies its upstream seals, then runs inside its guarded stage."""
    stage_id = STAGE_OF.get(args.command)
    if args.command == 'unblind':
        return unblind_command(args, config, manifest)
    _require(args.output is not None, '--output (a new directory under the pinned output root) is required')
    output = check_output(args.output, config)
    attempt_gate(manifest, stage_id)
    if args.command != 'decide':
        for upstream in reg.UPSTREAM[stage_id]:
            require_seal(manifest, upstream)
    if stage_id in ('A2-S0', 'A2-S1'):
        smoke_gate(manifest, stage_id)
    local_raw = args.local_config.read_bytes()
    rpv, policy, s6_dir, s6_dr = policy_registration(config)
    _require(hashlib.sha256(local_raw).hexdigest() == policy['config']['le2025_validation_plan']['local_config']['sha256'],
             'local config differs from its ML-POLICY pin')
    local = json.loads(local_raw)
    from run_ml_matrix import heavy_lock  # the common heavy lock (imported only for registered runs)
    root = Path(local['artifact_root']).resolve()
    os.nice(10)
    guards = Guards()
    guards()
    pins = config['registration_input_pins']
    with heavy_lock(root), stage(output, stage_id, identity_record(config, {'command': args.command}),
                                 manifest=manifest) as out:
        if args.command == 'smoke':
            run_smoke(rpv, policy, local, pins, out, guards, manifest)
        elif args.command == 's0':
            inputs, train = train_rows(rpv, policy, local, pins, reg.TRAIN_WINDOW)
            guards()
            s0 = build_s0(train, load_game_values(rpv, policy, inputs))
            dump(out / 's0.json', s0)
        elif args.command == 's1':
            s0_dir, _ = require_seal(manifest, 'A2-S0')
            s0 = json.loads((s0_dir / 's0.json').read_text())
            cell_dir = out / 'cells'
            cell_dir.mkdir()

            def write_cell(cell, record):  # each cell is sealed on completion (acceptance cells first)
                dump(cell_dir / f"{cell['id']}.json", record)
                manifest.append('cell_seal', {'stage': 'A2-S1', 'cell': cell['id'],
                                              'sha256': reg.file_sha256(cell_dir / f"{cell['id']}.json")})
            results = stage_s1(s0, write_cell=write_cell, deadline=guards)
            dump(out / 'tab.json', results)
        elif args.command == 's2':
            run_s2(rpv, policy, local, config, out, guards, manifest)
        elif args.command == 's3':
            run_s3(rpv, policy, local, config, s6_dir, s6_dr, out, guards, manifest)
        elif args.command == 'decide':
            run_decide(config, out, manifest)
    return 0


def run_smoke(rpv, policy, local, pins, out, guards, manifest):
    first, last = pins['smoke_train_subset']['game_dates']
    t = time.perf_counter()
    inputs, train = train_rows(rpv, policy, local, pins, (first, last))
    load = time.perf_counter() - t
    guards()
    t = time.perf_counter()
    s0 = build_s0(train, load_game_values(rpv, policy, inputs), smoke=True, seed_role='smoke',
                  fixed_m=reg.IMPLEMENTATION['smoke']['fixed_m'], min_pitches=SMOKE_MIN_PITCHES)
    s0_seconds = time.perf_counter() - t
    tab = smoke_tab(s0, replications=pins['smoke_tab_replications_per_cell'], guards=guards)
    full_train = int((inputs['frame']['split'] == 'train').sum())
    record = {'stage': 'A2-SM', 'subset': [first, last], 'rows': int(len(train)), 'train_rows': full_train,
              'load_seconds': load, 's0_seconds': s0_seconds, 'census_counts': s0['census'],
              'table_fit_unit_seconds': s0['timings']['table_fit'], 'tab': tab,
              'projection': smoke_projection(s0_seconds, s0['timings']['table_fit'], tab['variant_estimate_unit_seconds'],
                                             tab['replication_seconds'], full_train / max(len(train), 1)),
              'peak_rss_bytes': peak_rss_bytes(),
              'never_computed': ['cross-validated log-loss', 'm selection', 'lambda/M selection', 'delta_g', 'kappa_p',
                                 'zeta_p', 'bias, SD or RMSE', 'acceptance cells', 'H_point', 'R5 diagnostics']}
    dump(out / 'smoke_record.json', record)
    manifest.append('smoke_record', {'record_sha256': reg.file_sha256(out / 'smoke_record.json'),
                                     'projection': record['projection']})


def build_candidate(rpv, policy, inputs, blocks, *, seed, tag, out, budget):
    """The S5 candidate runtime (same construction as run_policy_validation.dispatch ``candidate`` for
    v2-world: dev style snapshot, S2-certified components, registered tau/setting/evaluation seed)."""
    from pitchmdp import policy_identity as pid
    from pitchmdp import policy_runtime as prt
    from pitchmdp.policy_artifacts import load_style_snapshot, load_train_bc
    config = policy['config']
    ident = config['identity_registration']
    stages = config['le2025_validation_plan']['stages']
    pre = rpv.prerequisites('v2-world', policy)
    freeze, setting = pre['freeze'], pre['setting']
    evaluation = rpv.role_seed(config, 'evaluation') if freeze['dr_q']['source'] == 'evaluation_seed' else None
    path, sha = rpv.registered_path(policy, 'style_dev')
    snapshot, as_of, content = load_style_snapshot(path, sha)
    _require(as_of == stages['S1b_style_snapshot']['as_of_exclusive']['dev'], 'style as-of differs from registration')
    frame = inputs['frame']
    first = pd.to_datetime(frame.game_date.iloc[np.concatenate([p for _, p in blocks])]).min()
    _require(first >= pd.Timestamp(as_of), 'evaluation blocks start before the style as-of date')
    contexts, style = rpv.pa_contexts(inputs['store'], blocks, snapshot, as_of)
    bc_path, bc_sha = rpv.registered_path(policy, 'bc')
    support_path, support_sha = rpv.registered_path(policy, 'support')
    components = pid.bind_components(inputs['bundle_path'], inputs['bundle_sha'], inputs['paths'],
                                     bc_artifact=load_train_bc(bc_path, bc_sha), context_rows=contexts,
                                     member_loader=inputs['member_loader'], classes=ident['classes'],
                                     we_contract_sha256=ident['we_contract_sha256'])
    _require(components.sha256 == pre['components_sha256'], 'bound components differ from the S2-certified identity')
    provenance = {'style_snapshot_sha256': content, 'as_of_exclusive': as_of, 'style_report': style}
    from pitchmdp.rollout_policy import RowBudget
    runtime = prt.build_runtime(bc_path, bc_sha, support_path, support_sha, out / f'ledger-s5x-{tag}.jsonl',
                                components=components, budget=RowBudget(int(budget), seed_count=5),
                                tau=freeze['selected_tau'], samples=setting['samples'], pitch_cap=setting['pitch_cap'],
                                seed=seed, evaluation_seed=evaluation, hand_registry=rpv.registered_path(policy, 'hands'),
                                provenance=provenance)
    return runtime, components


def run_s2(rpv, policy, local, config, out, guards, manifest):
    """S5-X: the registered 15 runs in order; X2 needs a registered world-predictor loader."""
    from pitchmdp import policy_requests as preq
    from pitchmdp import policy_semisynthetic as pss
    s0_dir, _ = require_seal(manifest, 'A2-S0')
    s0 = json.loads((s0_dir / 's0.json').read_text())
    pcfg = policy['config']
    spec = pcfg['le2025_validation_plan']['stages']['S5_V2_V3']
    inputs = rpv.load_inputs(pcfg, local, store=True)
    frame = inputs['frame']
    games = rpv.select_games(frame, 'dev', spec['n_games'], rpv.role_seed(pcfg, 'selection_salt'))
    blocks = rpv.game_blocks(frame, games, 'dev')
    no_pitch = frozenset(pcfg['pa_time_rules']['R3_codes']['no_pitch_descriptions'])
    game_values = load_game_values(rpv, policy, inputs)
    layer_doc = s0['s5x_layers']
    runs = []
    x2 = x2_world_predictor(config['registration_input_pins']['x2_world_predictor_identity'])  # refuses up front
    for run in reg.S5X_RUNS:
        guards()
        runtime, components = build_candidate(rpv, policy, inputs, blocks, seed=run['planning_seed'], tag=f"k{run['k']}",
                                              out=out, budget=spec['row_budget_per_run'])
        starts = []
        for _, positions in blocks:
            requests, _, _ = preq.pa_requests(inputs['store'], positions, runtime.sha256, no_pitch)
            if requests and runtime.start_population(requests[0].state, requests[0].pitcher_hand)[0]:
                starts.append((requests[0].state, requests[0].pitcher_hand))
        layers = osx.S5XLayers.from_components(
            components, game_values, groups=layer_doc['groups'], x_dist=layer_doc['x_dist'],
            class_table=layer_doc['class_table'], signs=layer_doc['signs'], signs_prime=layer_doc['signs_prime'],
            kappa=reg.scale(run['kappa'], layer_doc['kappa_p']), zeta=reg.scale(run['zeta'], layer_doc['zeta_p']))
        law = (pss.tempered_law(runtime.bc, reg.S5X_V3_ALPHA) if run['law'] == 'V3'
               else (lambda bc: (lambda s: (bc.actions, bc.probabilities(s))))(runtime.bc))
        predictor = x2(components) if run['setting'] == 'X2' else None
        record = osx.run_s5x(runtime, components, starts, run=run, law=law, layers=layers, world_predictor=predictor,
                             logs_per_start=reg.S5X_LOGS_PER_START, truth_rollouts=reg.S5X_TRUTH_ROLLOUTS,
                             cap=reg.S5X_CAP, gamma=run['setting'] == 'X1',
                             invariant=run['setting'] == 'X1' and run['zeta'] == '0', deadline=guards)
        dump(out / f"run-k{run['k']:02d}.json", record)
        runs.append(record)
    dump(out / 's5x.json', {'runs': runs, **s5x_summary(runs)})


def x2_world_predictor(identity):
    """X2 world predictor (design: 'a frozen <=2025 G0-family member different from the planner's
    predictor, identified at registration'). Supported kind: {'kind': 'g0_single_member', 'seed': s},
    one member of the bound five-member G0 (ope_synthetic_a2.SingleMemberPredictor). Any other
    identity refuses before the stage runs anything."""
    _require(isinstance(identity, dict) and identity.get('kind') == 'g0_single_member'
             and type(identity.get('seed')) is int, f'unsupported X2 world-predictor identity {identity!r}')
    return lambda components: osx.SingleMemberPredictor(components.g0, identity['seed'])


def recompute_facts(rpv, policy, inputs, blocks, games_table, game_values, no_pitch, out):
    """Planner-free PA facts of the S6 selection: requests and structural problems, E0 membership from
    a reference-only runtime (as the S4 denominators stage), PA-end reward from the frozen C0 WE."""
    from pitchmdp import policy_requests as preq
    from pitchmdp import policy_runtime as prt
    runtime = prt.build_runtime(*rpv.registered_path(policy, 'bc'), *rpv.registered_path(policy, 'support'),
                                out / 'facts-reference-ledger.jsonl', hand_registry=rpv.registered_path(policy, 'hands'))
    we = game_values['we']
    defense = lambda state, home: float(we.predict_defense(state, home))  # = BoundComponents.defense_we
    facts, last_rows = {}, {}
    frame = inputs['store'].frame  # as the registered runner: blocks index the store's frame
    for pa_id, positions in blocks:
        requests, problem, index = preq.pa_requests(inputs['store'], positions, runtime.sha256, no_pitch)
        info = {'game': int(frame.game_pk.iloc[positions[0]]), 'problem': problem, 'problem_index': index,
                'positions': positions, 'submitted': len(requests), **preq.pa_manifest(frame, positions)}
        if requests:
            inside, reason = runtime.start_population(requests[0].state, requests[0].pitcher_hand)
        else:
            inside, reason = False, f'unsubmittable:{problem}'
        info.update(in_population=inside, start_reason=reason)
        info.update(preq.pa_outcome(frame, positions, defense, games_table, rule='structural-end-v1',
                                    score_source='next_row'))
        facts[pa_id] = info
        last_rows[pa_id] = frame.iloc[positions[-1]]
    return facts, last_rows


def run_s3(rpv, policy, local, config, s6_dir, s6_dr, out, guards, manifest):
    """The blinded rehearsal on the sealed S6-dr-a5 ledger (read-only), no planner call."""
    from pitchmdp import policy_requests as preq
    from pitchmdp import policy_runtime as prt
    pins = config['registration_input_pins']
    _require(pins['r3_status'] == 'NOT_RUN:R3', 'R3 is enabled: the A3 artifact path must be added before running')
    s0_dir, _ = require_seal(manifest, 'A2-S0')
    s0 = json.loads((s0_dir / 's0.json').read_text())
    _require(osg.table_sha256(np.asarray(s0['r1_table']), s0['m']) == s0['r1_table_sha256'], 'R1 table differs from its seal')
    pcfg = policy['config']
    inputs = rpv.load_inputs(pcfg, local, store=True)
    frame = inputs['frame']
    games = s6_dr['games']
    blocks = rpv.game_blocks(frame, games, 'dev')
    no_pitch = frozenset(pcfg['pa_time_rules']['R3_codes']['no_pitch_descriptions'])
    game_values = load_game_values(rpv, policy, inputs)
    store_frame = inputs['store'].frame
    games_table = preq.game_table(store_frame, {int(store_frame.game_pk.iloc[p[0]]) for _, p in blocks})
    facts, last = recompute_facts(rpv, policy, inputs, blocks, games_table, game_values, no_pitch, out)
    guards()
    dates = pd.to_datetime(frame['game_date'].iloc[np.concatenate([p for _, p in blocks])])
    _require(dates.max() <= pd.Timestamp(reg.ALL_STAGES_MAX_DATE), 'a row after 2025 reached the rehearsal')
    keys = pd.DataFrame([{'game_pk': int(r.game_pk), 'at_bat_number': int(r.at_bat_number),
                          'pitch_number': int(r.pitch_number)} for r in last.values()])
    raw = raw_quality_columns(local, pins, keys, (str(dates.min().date()), str(dates.max().date())))
    rows = {}
    for (pa_id, row), (_, extra) in zip(last.items(), raw.iterrows()):
        record = {**row.to_dict(), 'hc_x': extra['hc_x'], 'hc_y': extra['hc_y']}
        for column in ('launch_speed', 'launch_angle'):
            if column not in record:
                record[column] = extra[column]
        rows[pa_id] = record
    terminal = osg.TerminalValues(game_values['we'], game_values['advancement'])
    surrogates = osg.surrogate_rewards(facts, rows, terminal, np.asarray(s0['r1_table']))
    ledger_path = Path(pins['s6_dr_a5_ledger_path'])
    with ledger_path.open('rb') as stream:
        first = json.loads(stream.readline())
    header = {k: v for k, v in first.items() if k not in ('kind', 'seq', 'prev', 'sha256')}
    ledger = prt.Ledger(ledger_path, header)  # replays and verifies the whole chain; read-only for an existing file
    ess_gate = pcfg['ess_gate']['thresholds']
    public, sealed, _ = osd.rehearsal_evaluate(
        ledger.iter_decisions, facts, surrogates=surrogates, gint_reference=osd.registered_gint_reference(config),
        train_residuals=s0.get('train_residuals'), r1_paths={'rehearsal': osg.path_shares(surrogates),
                                                             'train_census': s0['census']},
        r3_status=pins['r3_status'], ess_gate=ess_gate)
    public_bytes, sealed_bytes = osd.seal(public, sealed)
    (out / 'public.json').write_bytes(public_bytes)
    (out / 'centers.sealed.json').write_bytes(sealed_bytes)
    print(json.dumps({'centers.sealed.json sha256': osd.sha256_bytes(sealed_bytes)}))


def run_decide(config, out, manifest):
    unresolved = failed_integrity(manifest)
    if unresolved:
        dump(out / 'decision_record.json', {'label': 'FAILED_INTEGRITY', 'integrity_failures': unresolved,
                                            'r3_status': config['registration_input_pins']['r3_status']})
        return
    s1_dir, _ = require_seal(manifest, 'A2-S1')
    s2_dir, _ = require_seal(manifest, 'A2-S2')
    s3_dir, _ = require_seal(manifest, 'A2-S3')
    public = json.loads((s3_dir / 'public.json').read_text())
    osd.validate_public(public)
    tab = json.loads((s1_dir / 'tab.json').read_text())
    s5x = json.loads((s2_dir / 's5x.json').read_text())
    record = stage_s4(public, tab, s5x, integrity_failures=(),
                      r3_status=config['registration_input_pins']['r3_status'])
    record['inputs_sha256'] = {'public': reg.file_sha256(s3_dir / 'public.json'), 'tab': reg.file_sha256(s1_dir / 'tab.json'),
                               's5x': reg.file_sha256(s2_dir / 's5x.json')}
    dump(out / 'decision_record.json', record)


def unblind_command(args, config, manifest):
    """Open the sealed file only after the decision record is committed at HEAD byte for byte."""
    s3_dir, _ = require_seal(manifest, 'A2-S3')
    s4_dir, _ = require_seal(manifest, 'A2-S4')
    record_path = args.decision_record
    _require(record_path is not None and committed(record_path), 'the decision record must be committed before unblinding')
    sealed = osd.unblind((s3_dir / 'centers.sealed.json').read_bytes(),
                         json.loads((s3_dir / 'public.json').read_text())['sealed_sha256'],
                         Path(record_path).read_bytes(), reg.file_sha256(s4_dir / 'decision_record.json'),
                         committed=True)
    print(json.dumps({'selection': sealed.get('selection'), 'pairs': {
        p: (sealed['pairs'].get(p) or {}).get('center') for p in reg.PAIRS}}, indent=1, default=_default))
    return 0


if __name__ == '__main__':
    sys.exit(main())
