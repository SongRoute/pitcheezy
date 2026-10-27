"""Small synthetic Statcast-shaped fixtures for the EXP-BC-CAL-01 tests (no real data is read).

``make_frame`` simulates legal pitch-by-pitch PAs (counts follow the recorded outcomes) for the
four used splits; every one of the 18 registered codes appears in TRAIN (so a BC fitted on TRAIN
has the registered vocabulary) and two DEV pitchers throw a code they never threw in TRAIN (S_off).
Rows of the unused windows (2023-03-30..05-14 'pre', June 2025 'june') are generated on request
and written to the synthetic processed parquet only (the A3 readers decode and drop them).

Config fixtures never depend on the committed config's status: ``as_draft`` / ``as_registered``
build explicit DRAFT / REGISTERED forms from it (``A3_CONFIG_PATH`` names another config file,
used by the registered-preflight test to run the preflight suite against a REGISTERED config).
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from pitchmdp import behavior_calibration_registry as reg
from pitchmdp.archetypes import Archetypes, add_batter_style_history
from pitchmdp.behavior_calibration_registry import ACTION_CODES
from pitchmdp.policy_artifacts import TrainBCArtifact, _canonical_bytes, _envelope, bc_payload, single_hand
from pitchmdp.rollout_policy import CategoricalBC, BCRecord, PAState, PastPitch

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
COMMITTED_CONFIG = REPO / 'configs' / 'EXP-BC-CAL-01.json'
NESTED = 'A3_NESTED_PREFLIGHT'  # set for a preflight run started by a test (no nested isolated runs)
SPLIT_DAYS = {'pre': pd.date_range('2023-03-30', '2023-05-14', periods=2),
              'train': pd.date_range('2023-05-15', '2025-04-30', periods=12),
              'earlystop': pd.date_range('2025-05-01', '2025-05-15', periods=3),
              'temperature': pd.date_range('2025-05-16', '2025-05-31', periods=3),
              'june': pd.date_range('2025-06-01', '2025-06-30', periods=2),
              'dev': pd.date_range('2025-07-01', '2025-09-30', periods=5)}
USED = ('train', 'earlystop', 'temperature', 'dev')
PHYSICS = ('release_speed', 'release_spin_rate', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z', 'sz_top', 'sz_bot')


# ---------------------------------------------------------------- config fixtures (status-neutral)

def config_path():
    return Path(os.environ.get('A3_CONFIG_PATH') or COMMITTED_CONFIG)


def load_config():
    """The committed config (or ``A3_CONFIG_PATH``), whatever its registration status."""
    return json.loads(config_path().read_text())


def _put(config, dotted, value):
    node = config
    *parents, last = dotted.split('.')
    for part in parents:
        node = node[part]
    node[last] = value


REGISTRATION_TEXT = ('registration.registered_kst', 'registration.design_review_reference',
                     'registration.code_review_reference')


def as_draft(config):
    """An explicit DRAFT_UNREGISTERED fixture: every flag false and every registration pin null."""
    out = copy.deepcopy(config)
    out.update(status='DRAFT_UNREGISTERED', registered=False, execute=False, real_data_enabled=False)
    for dotted, _ in reg.REGISTRATION_PIN_SCHEMA:
        _put(out, dotted, None)
    out.get('registration', {}).pop('addenda_records', None)
    return out


RUNNER_SHA256 = 'e' * 64  # synthetic runner pin (well-formed hex)


def as_registered(config, *, runner_sha256=RUNNER_SHA256, commit='a' * 40, runner=None):
    """An explicit REGISTERED fixture with well-formed synthetic pins (S6 runtime = the expected
    c13cc989... identity) and a matching config_sha256; ``runner`` overrides runner.* fields."""
    if str(PROJECT / 'scripts') not in sys.path:
        sys.path.insert(0, str(PROJECT / 'scripts'))
    import run_behavior_calibration as runner_module
    out = as_draft(config)
    out.update(status='REGISTERED', registered=True, execute=True, real_data_enabled=True)
    out['registration'].update(registered_kst='2026-10-28T14:00:00+09:00', design_review_reference='design-review-x',
                               code_review_reference='code-review-x')
    out['code'].update(branch='research/track-a-a3-1027', commit=commit, runner_sha256=runner_sha256)
    out['data']['s6_ledger'].update(ledger_sha256='1' * 64, manifest_sha256='2' * 64)
    out['data']['s6_ledger']['runtime_sha256']['value'] = reg.S6['expected_runtime_sha256']
    out['data']['a3_input'].update(rechunked_sha256='3' * 64, identity_manifest_sha256='4' * 64)
    out['runner'].update(runner or {})
    out['code']['config_sha256'] = runner_module.config_content_sha256(out)
    return out


def _repertoires(rng, pitchers):
    out = {}
    for i, p in enumerate(pitchers):
        size = 3 + i % 3
        codes = list(rng.choice(['FF', 'SI', 'SL', 'CH', 'CU', 'FC', 'ST', 'FS', 'KC'], size=size, replace=False))
        weights = rng.dirichlet(np.ones(size) * 2)
        out[p] = (codes, weights)
    return out


def make_frame(seed=0, games=None, pas_per_half=3, innings=3, splits=USED, first_game=700000):
    rng = np.random.default_rng(seed)
    games = games or {'train': 36, 'earlystop': 8, 'temperature': 10, 'dev': 14}
    pitchers = [500000 + i for i in range(14)]
    batters = [600000 + i for i in range(40)]
    hands = {p: ('L' if i % 4 == 0 else 'R') for i, p in enumerate(pitchers)}
    stands = {b: ('L' if i % 3 == 0 else 'R') for i, b in enumerate(batters)}
    repertoire = _repertoires(rng, pitchers)
    utility = pitchers[-1]  # cycles through every registered code in TRAIN
    unknown = 500099  # appears only in DEV (TRAIN0)
    hands[unknown] = 'R'
    repertoire[unknown] = (['FF', 'SL'], np.array([.6, .4]))
    ambiguous = pitchers[3]  # throws with both hands in TRAIN (AMBIGUOUS registry entry)
    new_code = {pitchers[0]: 'SV', pitchers[1]: 'KN'}  # DEV-only codes (S_off)
    rows, game_pk, cycle = [], first_game, 0
    for split in splits:
        days = SPLIT_DAYS[split]
        for g in range(games.get(split, 0)):
            game_pk += 1
            date = days[g % len(days)]
            starters = {'Top': pitchers[(2 * g) % 12], 'Bot': pitchers[(2 * g + 1) % 12]}
            relievers = {'Top': pitchers[(2 * g + 5) % 12], 'Bot': pitchers[(2 * g + 7) % 12]}
            at_bat, home, away = 0, 0, 0
            for inning in range(1, innings + 1):
                for half in ('Top', 'Bot'):
                    outs = 0
                    for k in range(pas_per_half):
                        at_bat += 1
                        pitcher = starters[half] if inning <= 2 else relievers[half]
                        if split == 'train' and g % 9 == 0 and inning == 3 and half == 'Top':
                            pitcher = utility
                        if split == 'dev' and g % 4 == 1 and inning == 3 and half == 'Bot':
                            pitcher = unknown
                        batter = batters[int(rng.integers(len(batters)))]
                        bases = int(rng.integers(8))
                        balls = strikes = 0
                        number = 0
                        while True:
                            number += 1
                            if pitcher == utility:
                                code, cycle = ACTION_CODES[cycle % len(ACTION_CODES)], cycle + 1
                            else:
                                codes, weights = repertoire[pitcher]
                                code = codes[int(rng.choice(len(codes), p=weights))]
                            if split == 'dev' and pitcher in new_code and rng.random() < .15:
                                code = new_code[pitcher]
                            u = rng.random()
                            automatic = rng.random() < .02
                            if automatic:
                                description, outcome = 'automatic_ball', 'ball'
                            elif u < .36:
                                description, outcome = 'ball', 'ball'
                            elif u < .56:
                                description, outcome = 'called_strike', 'strike'
                            elif u < .66:
                                description, outcome = 'swinging_strike', 'strike'
                            elif u < .82:
                                description, outcome = 'foul', 'foul'
                            else:
                                description, outcome = 'hit_into_play', 'play'
                            events, terminal = None, False
                            if outcome == 'ball' and balls == 3:
                                events, terminal = 'walk', True
                            elif outcome == 'strike' and strikes == 2:
                                events, terminal = 'strikeout', True
                            elif outcome == 'play':
                                events, terminal = ('single' if rng.random() < .3 else 'field_out'), True
                            row = {'game_pk': game_pk, 'at_bat_number': at_bat, 'pitch_number': number,
                                   'game_date': date, 'split': split, 'game_type': 'R', 'pitcher': pitcher,
                                   'batter': batter, 'stand': stands[batter],
                                   'p_throws': ('L' if pitcher == ambiguous and split == 'train' and g % 2 else
                                                hands[pitcher]),
                                   'balls': balls, 'strikes': strikes, 'outs_when_up': outs, 'bases': bases,
                                   'inning': inning, 'inning_topbot': half, 'home_score': home, 'away_score': away,
                                   'pitch_type': None if automatic else code, 'description': description,
                                   'events': events, 'is_pa_terminal': terminal,
                                   'launch_angle': float(rng.normal(12, 20)) if outcome == 'play' else np.nan}
                            for c in PHYSICS:
                                row[c] = float(rng.normal())
                            rows.append(row)
                            if terminal:
                                break
                            if outcome == 'ball':
                                balls += 1
                            elif outcome == 'strike':
                                strikes += 1
                            elif outcome == 'foul':
                                strikes = min(2, strikes + 1)
                        if events == 'field_out' or events == 'strikeout':
                            outs = min(outs + 1, 2)
                        if rng.random() < .1:
                            home, away = home + (half == 'Bot'), away + (half == 'Top')
    frame = pd.DataFrame(rows)
    frame['game_date'] = pd.to_datetime(frame.game_date)
    frame = frame.sort_values(['game_date', 'game_pk', 'at_bat_number', 'pitch_number'], kind='stable')
    return frame.reset_index(drop=True)


def bc_states(frame):
    """B0 records of the TRAIN BC_P rows (fit_bc semantics: previous = earlier row of the PA)."""
    from pitchmdp.behavior_features import bc_rule, history_features
    train = frame.loc[frame.split.eq('train')]
    history = history_features(train)
    keep = bc_rule(train)
    records = []
    for (_, row), previous in zip(train.loc[keep].iterrows(), history.previous[keep]):
        hist = () if previous == '<START>' else (PastPitch(previous, (), 'unknown', 0, 0),)
        state = PAState(int(row.balls), int(row.strikes), str(int(row.pitcher)), str(row.stand), hist)
        records.append(BCRecord(state, str(row.pitch_type), 'train'))
    return records


def make_artifact(frame):
    """A reloaded-shape TRAIN BC artifact (B0), its support table and hand registry."""
    records = bc_states(frame)
    bc = CategoricalBC(20., 1).fit(records)
    provenance = {'train_start': '2023-05-15', 'train_end': '2025-04-30', 'dates': 'declared_unverified',
                  'train_rows': len(records), 'source_ids': {'synthetic': 'a' * 64}, 'config_sha256': 'b' * 64,
                  'code_commit': 'c' * 40, 'data_version': 'synthetic'}
    payload = bc_payload(bc, provenance)
    raw = _canonical_bytes(_envelope('pitcheezy.train_bc.v1', payload))
    artifact = TrainBCArtifact(payload, hashlib.sha256(raw).hexdigest())
    train = frame.loc[frame.split.eq('train')]
    hands = {str(int(p)): single_hand(part.p_throws.tolist()) for p, part in train.groupby('pitcher')}
    hands = {p: hands[p] for p in artifact.bc.pitchers}
    support = {}
    for p, hand in hands.items():
        if hand == 'AMBIGUOUS':
            continue
        mask = np.array([artifact.bc.pitchers[p][a] >= 1 for a in artifact.vocabulary])
        for side in ('L', 'R'):
            support[(p, side)] = mask
    return artifact, support, hands, raw


def make_archetypes(frame):
    from pitchmdp.behavior_features import style_input_rows
    inputs = style_input_rows(frame)
    styled = add_batter_style_history(inputs.copy())
    train = styled.loc[frame.loc[styled.index].split.eq('train')].assign(split='train')
    return Archetypes(5, seed=42).fit(train)


def raw_keys(frame):
    return frame[['game_date', 'game_pk', 'at_bat_number', 'pitch_number', 'pitcher']].copy()


def make_world(seed=0):
    """Frame, B0 artifact, feature context and the TRAIN / earlystop / temperature tables."""
    from pitchmdp.behavior_features import FeatureContext, split_table, style_tables, workload_features
    from pitchmdp.behavior_model import RegisteredBC
    from pitchmdp.policy_requests import volume_edges
    frame = make_frame(seed)
    artifact, support, hands, raw = make_artifact(frame)
    b0 = RegisteredBC(artifact)
    train = frame.loc[frame.split.eq('train')]
    ctx = FeatureContext(vocabulary=ACTION_CODES, archetypes=make_archetypes(frame), b0=b0.vector,
                         train_volume={int(p): int(n) for p, n in train.groupby('pitcher').size().items()},
                         volume_edges=tuple(volume_edges(train, (.25, .75))), style=style_tables(frame),
                         workload=workload_features(frame, raw_keys(frame)), hands=hands, repertoire=b0.repertoire)
    oof = {'salt': 'EXP-BC-CAL-01/te', 'k': 5, 'prior_strength': 20., 'minimum_action_count': 1}
    tables = {'train': split_table(train, 'train', ctx, oof=oof)[0]}
    for split in ('earlystop', 'temperature'):
        tables[split] = split_table(frame.loc[frame.split.eq(split)], split, ctx)[0]
    return {'frame': frame, 'artifact': artifact, 'artifact_bytes': raw, 'support': support, 'hands': hands,
            'b0': b0, 'ctx': ctx, 'tables': tables}


OUTCOME_OF = {'ball': 'ball', 'automatic_ball': 'ball', 'called_strike': 'strike', 'swinging_strike': 'strike',
              'foul': 'foul'}


def make_ledger(directory, world, games=6):
    """A synthetic S6-shaped ledger written by the real PolicyRuntime (runtime v3, reference-pair
    candidate) over the first ``games`` DEV games, with its manifest.json and the PA facts that
    policy_estimator.estimate needs (synthetic reward 0.5; never real data)."""
    from pathlib import Path
    from pitchmdp.policy_artifacts import save_hand_registry, save_support_table
    from pitchmdp.policy_requests import logged_label
    from pitchmdp.policy_runtime import DecisionRequest, build_reference_pair_runtime
    import json
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    frame, artifact = world['frame'], world['artifact']
    bc_path = directory / 'bc_BC_P.json'
    bc_path.write_bytes(world['artifact_bytes'])
    _, support_sha = save_support_table(artifact, [(p, s, m) for (p, s), m in sorted(world['support'].items())],
                                        directory / 'support_primary.json')
    _, hands_sha = save_hand_registry(artifact, world['hands'], int(frame.split.eq('train').sum()),
                                      directory / 'hands.json')
    stage = directory / 'S6-synthetic'
    stage.mkdir()
    ledger_path = stage / 'ledger-dev-0.jsonl'
    runtime = build_reference_pair_runtime(bc_path, artifact.file_sha256, directory / 'support_primary.json',
                                           support_sha, ledger_path, hand_registry=(directory / 'hands.json', hands_sha),
                                           positivity_record=True)
    dev = frame.loc[frame.split.eq('dev')]
    chosen = list(dict.fromkeys(dev.game_pk))[:games]
    facts = {}
    for (g, ab), part in dev.loc[dev.game_pk.isin(chosen)].groupby(['game_pk', 'at_bat_number'], sort=False):
        history, first = [], None
        for i, row in enumerate(part.itertuples()):
            label = logged_label(row.pitch_type, row.description)
            state = PAState(int(row.balls), int(row.strikes), str(int(row.pitcher)), str(row.stand), tuple(history),
                            f'{int(g)}:{int(ab)}:{int(row.pitch_number)}')
            first = (state, row.p_throws) if first is None else first
            runtime.submit(DecisionRequest(f'{int(g)}:{int(ab)}:{int(row.pitch_number)}', f'{int(g)}:{int(ab)}', i,
                                           state, label, runtime.sha256, row.p_throws))
            outcome = OUTCOME_OF.get(row.description, 'out' if row.events != 'single' else 'single')
            history.append(PastPitch(label, (0.,) * 8, outcome, int(row.balls), int(row.strikes)))
        member, reason = runtime.start_population(*first)
        facts[f'{int(g)}:{int(ab)}'] = {'game': int(g), 'in_population': member, 'start_reason': reason,
                                        'problem': None, 'problem_index': None, 'reward': 0.5, 'reason': None,
                                        'kind': None}
    manifest = stage / 'manifest.json'
    manifest.write_text(json.dumps({'command': 'synthetic', 'artifact_sha256': {
        'ledger-dev-0.jsonl': hashlib.sha256(ledger_path.read_bytes()).hexdigest()}}))
    return {'ledger': ledger_path, 'manifest': manifest, 'runtime': runtime, 'facts': facts,
            'support_path': directory / 'support_primary.json', 'support_sha256': support_sha,
            'hands_path': directory / 'hands.json', 'hands_sha256': hands_sha, 'bc_path': bc_path,
            'ledger_sha256': hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
            'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest()}


def estimator_ess(ledger):
    """ess_all_e0 of the registered L1-R estimator on the synthetic ledger (the R2 reference)."""
    from pitchmdp.policy_estimator import estimate
    result, _ = estimate(ledger['runtime'].ledger.iter_decisions(), ledger['facts'], draws=10, seed=1,
                         invalid_share_max=1.0, minimum={'games': 1, 'pa_starts': 1}, censoring='l1r')
    return result['ess_all_e0']


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extra_rows(seed=100):
    """Rows of the unused windows (2023-03-30..05-14 and June 2025) with their own game ids."""
    return make_frame(seed, games={'pre': 2, 'june': 4}, splits=('pre', 'june'), first_game=790000)


def processed_order(frame):
    out = frame.copy()
    out['game_date'] = pd.to_datetime(out.game_date).dt.normalize()
    return out.sort_values(['game_date', 'game_pk', 'at_bat_number', 'pitch_number'], kind='stable').reset_index(drop=True)


SPLIT_STARTS = ('2023-05-15', '2025-05-01', '2025-05-16', '2025-06-01', '2025-07-01')


def default_cuts(ordered):
    """Split-aligned row groups: a boundary at every split date (no row group spans one), with TRAIN
    and DEV split once more inside their segments. Empty groups are left out."""
    dates, n = ordered.game_date, len(ordered)
    bounds = [int((dates < pd.Timestamp(day)).sum()) for day in SPLIT_STARTS]
    train_lo, train_hi, dev = bounds[0], bounds[1], bounds[4]
    return sorted({0, *bounds, train_lo + (train_hi - train_lo) // 2, dev + (n - dev) // 2, n})


def straddling_cuts(ordered):
    """An unsafe layout: one row group holds earlystop .. June .. the first DEV games."""
    dates, n = ordered.game_date, len(ordered)
    train_end = int((dates <= pd.Timestamp('2025-04-30')).sum())
    dev_start = int((dates < pd.Timestamp('2025-07-01')).sum())
    return sorted({0, train_end // 2, train_end, dev_start + (n - dev_start) // 2, n})


PROCESSED_COLUMNS = [*reg.FREE_COLUMNS, *reg.GUARDED_COLUMNS, 'release_speed']  # release_speed: never read


def write_parquet(path, ordered, columns, cuts, statistics=True):
    import pyarrow as pa
    import pyarrow.parquet as pq
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = ordered[columns]
    schema = pa.Schema.from_pandas(part, preserve_index=False)
    with pq.ParquetWriter(path, schema, write_statistics=statistics) as writer:
        for a, b in zip(cuts[:-1], cuts[1:]):
            if b > a:
                writer.write_table(pa.Table.from_pandas(part.iloc[a:b], schema=schema, preserve_index=False))
    return path


def write_processed(directory, frame, cuts=None, name='pitches.parquet', statistics=True):
    """A processed-shaped parquet (date-sorted, explicit row groups); returns (path, sha256, cuts).
    ``cuts`` may be a list of positions or a function of the date-ordered frame."""
    ordered = processed_order(frame)
    cuts = default_cuts(ordered) if cuts is None else (cuts(ordered) if callable(cuts) else list(cuts))
    path = write_parquet(Path(directory) / name, ordered, PROCESSED_COLUMNS, cuts, statistics)
    return path, file_sha256(path), cuts


def write_raw(directory, frame):
    """Two raw Statcast-shaped files (keys, date, pitcher and an unread pitch_type column): rows before
    2025 in one row group; 2025 rows split at the TRAIN end. Returns (root, {name: sha256})."""
    ordered = processed_order(frame)
    root = Path(directory) / 'raw'
    columns = ['game_date', 'game_pk', 'at_bat_number', 'pitch_number', 'pitcher', 'pitch_type']
    early = ordered.loc[ordered.game_date.dt.year < 2025].reset_index(drop=True)
    late = ordered.loc[ordered.game_date.dt.year >= 2025].reset_index(drop=True)
    cut = int((late.game_date <= pd.Timestamp('2025-04-30')).sum())
    pins = {}
    for name, part, cuts in (('statcast_synthetic_a.parquet', early, [0, len(early)]),
                             ('statcast_synthetic_b.parquet', late, sorted({0, cut, len(late)}))):
        pins[name] = file_sha256(write_parquet(root / name, part, columns, cuts))
    return root, pins


def prepare_input(directory, frame, cuts=None):
    """The A3 processed input as the registrar makes it: a size-based (not split-aligned) synthetic
    processed parquet, rewritten by scripts/prepare_behavior_calibration_input.py. Returns the record."""
    from datetime import datetime
    if str(PROJECT / 'scripts') not in sys.path:
        sys.path.insert(0, str(PROJECT / 'scripts'))
    import prepare_behavior_calibration_input as tool
    directory = Path(directory)
    source, sha, _ = write_processed(directory / 'source', frame, straddling_cuts if cuts is None else cuts)
    (directory / 'A3').mkdir(parents=True, exist_ok=True)
    record = tool.prepare(source, directory / 'A3' / 'inputs' / 'pitches.split-aligned.parquet', source_sha256=sha,
                          now=lambda: datetime(2026, 10, 27, 15, 0, tzinfo=tool.KST))
    if record['result'] != 'PASS':
        raise AssertionError(f'synthetic A3 input preparation failed: {record["identity"]}')
    return record


class SyntheticSource:
    """Runner data source: the production A3 readers (ProcessedReader / RawKeysReader / A3Inputs) over
    the split-aligned synthetic input written by the preparation tool, raw key files, the synthetic B0
    and ledger (never real files). ``revalidate`` records every per-stage call and runs ``hook`` (a
    test may make it refuse). The acquisitions go to the run's persisted read log, as with the real
    source."""
    def __init__(self, world, ledger, directory, *, extras=None, hook=None):
        frame = world['frame'] if extras is None else pd.concat([world['frame'], extras], ignore_index=True)
        self.prepared = prepare_input(directory, frame)
        self.processed = Path(self.prepared['rewritten']['path'])
        self.processed_sha256 = self.prepared['rewritten']['sha256']
        self.raw_root, self.raw_pins = write_raw(directory, frame)
        self.world, self.ledger, self.hook = world, ledger, hook
        self.opens, self.revalidations, self.inputs = 0, [], None

    def revalidate(self, run, stage):
        self.revalidations.append(stage)
        if self.hook is not None:
            self.hook(run, stage)
        return {'stage': stage, 'registration_gate': 'synthetic source'}

    def open(self, run):
        from pitchmdp.behavior_data import A3Inputs, ProcessedReader, RawKeysReader
        self.opens += 1
        log = run.log
        reader = ProcessedReader(self.processed, sha256=self.processed_sha256, log=log, phase=run.current_stage)
        ledger = None if self.ledger is None else {
            'path': str(self.ledger['ledger']), 'manifest_path': str(self.ledger['manifest']),
            'ledger_sha256': self.ledger['ledger_sha256'], 'manifest_sha256': self.ledger['manifest_sha256'],
            'runtime_sha256': self.ledger['runtime'].sha256}
        identity = {'source': 'synthetic', 'b0': {'file_sha256': self.world['artifact'].file_sha256},
                    'processed_sha256': self.processed_sha256, 'raw_sha256': self.raw_pins}
        self.inputs = A3Inputs(reader=reader, raw_reader=RawKeysReader(self.raw_root, self.raw_pins, log),
                               b0=self.world['b0'], support=self.world['support'], hands=self.world['hands'],
                               archetypes=self.world['ctx'].archetypes, ledger=ledger, identity=identity, log=log)
        return self.inputs


def synthetic_config(config, world, ledger, expected_ess):
    """The materialized config with the synthetic pins (artifact, ledger identity, R2 aggregates) and
    small-data B2 / smoke sizes. Test-only: the real loader re-runs the registration gate."""
    import copy
    out = copy.deepcopy(config)
    out['data']['b0_artifact']['file_sha256'] = world['artifact'].file_sha256
    out['data']['s6_ledger']['runtime_sha256']['expected_value'] = ledger['runtime'].sha256
    out['data']['s6_ledger']['runtime_sha256']['value'] = ledger['runtime'].sha256
    out['ope_weight_diagnostics']['integrity_gates']['R2']['expected'] = expected_ess
    out['candidates']['B2']['params']['min_data_in_leaf'] = 20
    out['candidates']['B2']['rounds']['train_rounds'] = 10
    out['runner']['smoke'] = {**out['runner']['smoke'], 'train_rows': 1500, 'state_checks': 300, 'b2_rounds': 6}
    out['seeds']['bootstrap_draws'] = 300
    return out


def make_context(frame, world):
    """A FeatureContext rebuilt from ``frame`` (style inputs, workload, TRAIN volumes) with the
    frozen B0 / archetypes / hands of ``world`` (used by the perturbation tests)."""
    from pitchmdp.behavior_features import FeatureContext, style_tables, workload_features
    from pitchmdp.policy_requests import volume_edges
    train = frame.loc[frame.split.eq('train')]
    ctx = world['ctx']
    return FeatureContext(vocabulary=ACTION_CODES, archetypes=ctx.archetypes, b0=ctx.b0,
                          train_volume={int(p): int(n) for p, n in train.groupby('pitcher').size().items()},
                          volume_edges=tuple(volume_edges(train, (.25, .75))), style=style_tables(frame),
                          workload=workload_features(frame, raw_keys(frame)), hands=ctx.hands,
                          repertoire=ctx.repertoire)


def isolated_env(extra=None):
    """Environment of a fresh interpreter whose torch and LightGBM share ONE OpenMP runtime
    (DYLD_LIBRARY_PATH=<torch/lib>, the repository rule of run_ml_benchmark.validate_native_runtime;
    the unsafe KMP_DUPLICATE_LIB_OK override is never set). MPLCONFIGDIR points to a fresh writable
    directory (LightGBM imports Matplotlib when present; a read-only home must not break it)."""
    import importlib.util
    import tempfile
    torch_lib = Path(importlib.util.find_spec('torch').origin).resolve().parent / 'lib'
    env = {k: v for k, v in os.environ.items() if k != 'KMP_DUPLICATE_LIB_OK'}
    env['DYLD_LIBRARY_PATH'] = str(torch_lib)
    env['MPLCONFIGDIR'] = tempfile.mkdtemp(prefix='a3-mpl-')
    env['PYTHONPATH'] = os.pathsep.join([str(PROJECT), str(PROJECT / 'tests'), str(PROJECT / 'scripts'),
                                         env.get('PYTHONPATH', '')])
    env.update(extra or {})
    return env


def run_isolated(test_file, *args, timeout=900, env=None):
    """Run a pytest file in a fresh interpreter (``isolated_env``)."""
    import subprocess
    return subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                           str(PROJECT / 'tests' / test_file), *args], cwd=PROJECT, env=isolated_env(env),
                          capture_output=True, text=True, timeout=timeout)


def counts_by_key(records):
    out = defaultdict(Counter)
    for r in records:
        s = r.state
        out[(s.pitcher, s.balls, s.strikes, s.batter_side, s.history[-1].action if s.history else '<START>')][r.action] += 1
    return out
