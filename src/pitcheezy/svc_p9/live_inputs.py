"""W1 of EXP-SVC-P9-001: one GUMBO snapshot -> B_PSR9N6 model inputs for the pitch about to be thrown, or an omission.

Reuses, unchanged, the accepted RS-01 source readers:
  * ``seq_shadow.captured_bridge.step``: the next pitch's pre-pitch query (count, outs, inning, half, score,
    runners, batter side, pitcher hand, pitcher, batter, key). It already abstains on a body whose playIds
    conflict, on a non-pitch action in the current plate appearance and on any unverified prestate.
  * ``seq_shadow.captured_native.inspect_tokens``: per earlier pitch a verified six-field token
    (pitcher, batter, pitch_type, outcome10, pre-count balls, strikes) or a reason it is unknown.

Never raises on feed content: every refusal is ``Omit(reason)`` and the display shows nothing for that pitch.

Model inputs (the research training definition):
  * state11 = first_small_features.raw_features columns 0..10 of the query
      [balls/3, strikes/2, outs/2, inning/9, sign*(home-away)/5 (sign +1 in the top half), top half,
       on_1b, on_2b, on_3b, batter left, pitcher left]
    encoded like first_small_features.encode with the september fold encoder (median fill, (x-mean)/scale,
    clip +-8, float32), pinned by sha256.
  * tokens = ws_seq.train_seq.build_context_index + Data.tokens: slot j is the pitch thrown j+1 pitches EARLIER
    by the same pitcher in the same game, in (at_bat_number, pitch_number) order, at most 32; fields type
    (first_small_features.TYPES index), out (outcome10), balls, strikes (that pitch's pre-pitch count),
    same (same batter as the query), pos (1..32), valid.

Strict rule (plan §4-1, §4-4): if ANY earlier pitch of the game has no verified token, the pitcher of that pitch
is unknown, so the context cannot be proved complete -> Omit('token_unverified:<reason>').

Narrowed (code review P1-2 가): the RS-01 identity reader breaks its matchup chain at any non-pitch action except a bare
mound visit, so one pickoff made every later pitch of the game unverified. The training generator never sees
non-pitch actions (Statcast rows are pitches), so an action that cannot change pitcher, batter or count does not
change the training input. ``strip_preserving_actions`` removes exactly those actions from a COPY of the body before
``inspect_tokens`` (the RS-01 modules are unchanged):
  isPitch False AND isSubstitution False AND (type 'pickoff' or details.eventType in PRESERVING_ACTIONS)
  AND its count (balls, strikes) equals the count before it (the previous pitch's post count, 0-0 at PA start).
Substitutions, automatic balls/strikes (type 'no_pitch': Statcast numbers them as pitches), unknown action types and
any count change stay strict. The query side (captured_bridge) reads the same stripped copy: a removed action can move
runners, outs or score, which the bridge reads live from linescore/currentPlay (and cross-checks with the last pitch's
count), i.e. the Statcast pre-pitch state; a substitution or other strict action in the CURRENT PA still abstains.
Also removed (code review N2 a): substitution events BEFORE the first pitch of a PA (``pa_start_substitutions``), when
the last pitching substitution brings in the play's matchup pitcher; a substitution after the first pitch stays strict.
Both sides are checked against the training generator on the multi-action fixtures (tests/test_svc_p9_live_inputs.py).
Unkeyed pitches (no usable pitchNumber/playId; captured_native rows with key None): the bridge marks such a body
incomplete and the whole snapshot is omitted (stored_prefix_unkeyed_or_conflicting), so they never become tokens.
Token pitch type outside first_small_features.TYPES: omitted (token_type_outside_vocabulary); captured_native already
admits only TYPES codes, so this is a second guard.

W1 versions (``LiveInputs(..., version=...)``; default 'v2' keeps every earlier caller unchanged):
  v2  relaxed W1 of 0638b37c (EXP-SVC-R1-001): absent isSubstitution counts as False; strict ``tokens_for``.
  v3  EXP-SVC-P9-003 (POST-HOC, chosen after the masked census f of v2): v2 + EXCL + NOPITCH via
      ``svc_p9.w1_v3.inspect_tokens`` (RS-01 modules unchanged; see that module). ``tokens_for`` skips rows with
      status ``excluded_unmapped_label`` (a context gap, as in training) and stays strict for every other unverified row.
  v4  EXP-SVC-R4-003 (demo build only): v3 + MISSREPL via ``svc_p9.w1_v4.inspect_tokens`` (a pitching substitution
      without replacedPlayer gets the previous same-half PA's end-view pitcher as outgoing; see that module).

playId reassignment (TFINAL-ERR-01, 849799): the strict StreamState refuses a re-keyed (ab, pitch) forever. This
converter is stateless per snapshot: each body is rebuilt from scratch, so a later self-consistent body gives
correct inputs (flagged ``rebuilt_after_reassignment``); a body that is itself inconsistent is omitted
(the bridge reports ``provider_identity_conflict`` / ``stored_prefix_unkeyed_or_conflicting``).
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
from dataclasses import dataclass, field

import numpy as np

from pitcheezy.seq_shadow import captured_bridge, captured_native
from pitcheezy.svc_p9 import w1_v3, w1_v4

TYPES = ('UNKNOWN', 'CH', 'CS', 'CU', 'EP', 'FA', 'FC', 'FF', 'FO', 'FS', 'KC', 'KN', 'PO', 'SC', 'SI', 'SL', 'ST', 'SV', 'UN')
TYPE_INDEX = {t: i for i, t in enumerate(TYPES)}
CONTEXT = 32
STATE_COLUMNS = ('balls', 'strikes', 'outs', 'inning', 'defensive_lead', 'home_defense',
                 'on_1b', 'on_2b', 'on_3b', 'batter_left', 'pitcher_left')
W1_VERSIONS = ('v2', 'v3', 'v4')
CONFIG = pathlib.Path(__file__).resolve().parents[3] / 'configs/EXP-SVC-P9-001.json'
ENCODER_SHA256, ENCODER_BYTES = '375c1075db3ecc5f11e0bc0847a75f9298b9f28de84dc0610c35acdc676f8e26', 7153  # fixed pins


def encoder_pin(config=CONFIG):
    """Path from the registration (pins.state_encoder.path) or PITCHEEZY_SVC_P9_ENCODER; sha256/bytes are fixed here
    and must equal the registration's."""
    pin = json.loads(pathlib.Path(config).read_text())['candidate']['pins']['state_encoder']
    if (pin['sha256'], pin['bytes']) != (ENCODER_SHA256, ENCODER_BYTES):
        raise EncoderMismatch('registration encoder pin differs from the code pin')
    return {'path': os.environ.get('PITCHEEZY_SVC_P9_ENCODER', pin['path']), 'sha256': ENCODER_SHA256, 'bytes': ENCODER_BYTES}


class EncoderMismatch(ValueError):
    pass


@dataclass(frozen=True)
class Encoder11:
    median: np.ndarray
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def from_json(cls, data: bytes, expect_sha256=ENCODER_SHA256):
        if expect_sha256 is not None and hashlib.sha256(data).hexdigest() != expect_sha256:
            raise EncoderMismatch('state encoder sha256')
        e = json.loads(data)
        if e.get('dimension') != 199 or len(e['mean']) != 91 or len(e['scale']) != 91 or len(e['median']) != 91:
            raise EncoderMismatch('state encoder shape')
        return cls(*(np.asarray(e[k][:11], np.float64) for k in ('median', 'mean', 'scale')))

    @classmethod
    def load(cls, pin=None):
        pin = pin or encoder_pin()
        p = pathlib.Path(pin['path'])
        data = p.read_bytes()
        if len(data) != pin['bytes']:
            raise EncoderMismatch('state encoder bytes')
        return cls.from_json(data, pin['sha256'])

    def encode(self, raw11):
        raw = np.asarray(raw11, np.float64).reshape(-1, 11)
        filled = np.where(np.isfinite(raw), raw, self.median)
        return np.clip((filled - self.mean) / self.scale, -8, 8).astype(np.float32)


@dataclass(frozen=True)
class Omit:
    reason: str
    pitch_key: str | None = None


@dataclass
class ModelInputs:
    pitch_key: str
    pitcher: int
    batter: int
    raw11: np.ndarray
    state11: np.ndarray
    tokens: dict
    flags: tuple = field(default_factory=tuple)

    def batch(self):
        """Arguments of predictor.Bundle.predict for this one pitch."""
        return self.state11.reshape(1, 11), np.array([self.pitcher]), np.array([self.batter]), {k: v[None] for k, v in self.tokens.items()}


def raw11(query):
    """first_small_features.raw_features columns 0..10 for one bridge query (captured_bridge CONTEXT order)."""
    top = query['inning_topbot'] == 'Top'
    lead = (1.0 if top else -1.0) * (query['home_score'] - query['away_score'])
    return np.array([query['balls'] / 3, query['strikes'] / 2, query['outs_when_up'] / 2, query['inning'] / 9, lead / 5, float(top),
                     float(query['on_1b'] is not None), float(query['on_2b'] is not None), float(query['on_3b'] is not None),
                     float(query['stand'] == 'L'), float(query['p_throws'] == 'L')], np.float64)


PRESERVING_ACTIONS = frozenset({
    'pickoff_1b', 'pickoff_2b', 'pickoff_3b', 'pickoff_error_1b', 'pickoff_error_2b', 'pickoff_error_3b',
    'pickoff_caught_stealing_2b', 'pickoff_caught_stealing_3b', 'pickoff_caught_stealing_home',
    'stolen_base_2b', 'stolen_base_3b', 'stolen_base_home', 'caught_stealing_2b', 'caught_stealing_3b',
    'caught_stealing_home', 'wild_pitch', 'passed_ball', 'balk', 'defensive_indiff', 'other_advance', 'error',
    'mound_visit', 'game_advisory',
    'batter_timeout'})  # R1 (2026-11-19): real 849799 feed; no matchup/count change (the count check below still applies)
PRESERVING_TYPES = frozenset({'pickoff', 'stepoff'})  # 'stepoff' added in R1 (2026-11-19), same count check


def _bs(count):
    if not isinstance(count, dict):
        return None
    return count.get('balls'), count.get('strikes')


def preserving(event, before):
    # Real GUMBO bodies OMIT isSubstitution on non-substitution actions (849799/849803/849843 finals: never False, only
    # absent or True); an absent key counts as False, an explicit non-False value (True, None) stays strict. (R1 2026-11-19)
    if not isinstance(event, dict) or event.get('isPitch') is not False or event.get('isSubstitution', False) is not False:
        return False
    kind = (event.get('details') or {}).get('eventType') if isinstance(event.get('details'), dict) else None
    if not (event.get('type') in PRESERVING_TYPES or (event.get('type') == 'action' and kind in PRESERVING_ACTIONS)):
        return False
    return _bs(event.get('count')) == before


SUBSTITUTIONS = frozenset({'pitching_substitution', 'offensive_substitution', 'defensive_substitution', 'defensive_switch'})


def _is_sub(event):
    return (isinstance(event, dict) and event.get('isPitch') is False and event.get('isSubstitution') is True
            and isinstance(event.get('details'), dict) and event['details'].get('eventType') in SUBSTITUTIONS)


def pa_start_substitutions(play):
    """Indexes (positions) of substitution events BEFORE the first pitch of ``play`` that may be removed (code review
    N2 a): every pitch of the PA comes after them, so the play matchup is the pitch-time matchup of every pitch, as in
    the training rows (statcast_rows takes the matchup pitcher when no substitution follows the pitch). Refused (empty)
    when the last pitching substitution's incoming player is not the play's matchup pitcher, or any of them has a count
    other than 0-0."""
    events = play.get('playEvents') if isinstance(play, dict) else None
    if not isinstance(events, list):
        return []
    first = next((k for k, e in enumerate(events) if isinstance(e, dict) and e.get('isPitch') is True), len(events))
    subs = [k for k in range(first) if _is_sub(events[k])]
    if not subs:
        return []
    pitching = [events[k] for k in subs if events[k]['details']['eventType'] == 'pitching_substitution']
    matchup_pitcher = ((play.get('matchup') or {}).get('pitcher') or {}).get('id')
    if pitching and ((pitching[-1].get('player') or {}).get('id') != matchup_pitcher or matchup_pitcher is None):
        return []
    if any(_bs(events[k].get('count')) not in ((0, 0), None) for k in subs):
        return []
    return subs


def strip_preserving_actions(feed):
    """(copy of ``feed`` without matchup- and count-preserving non-pitch actions, number removed)."""
    import copy
    out, removed = copy.deepcopy(feed), 0
    plays = (out.get('liveData') or {}).get('plays') or {}
    seen = []
    for play in [*(plays.get('allPlays') or []), plays.get('currentPlay')]:
        if not isinstance(play, dict) or not isinstance(play.get('playEvents'), list) or any(play is p for p in seen):
            continue
        seen.append(play)
        kept, before = [], (0, 0)
        drop = set(pa_start_substitutions(play))
        for k, event in enumerate(play['playEvents']):
            if k in drop:
                removed += 1
                continue
            if preserving(event, before):
                removed += 1
                continue
            kept.append(event)
            if isinstance(event, dict) and isinstance(event.get('count'), dict):
                before = _bs(event['count'])   # any kept event (pitch or strict action) moves the reference count
        play['playEvents'] = kept
    return out, removed


def tokens_for(rows, query_key, pitcher, batter):
    """Context tokens of the query from inspect_tokens rows, or an Omit reason string."""
    earlier = [r for r in rows if r['key'] is not None and tuple(r['key'])[1:] < tuple(query_key)[1:]
               and r['status'] != w1_v3.EXCLUDED]   # only w1_v3.inspect_tokens (W1 v3) emits this status
    bad = sorted({r['reason'] for r in earlier if r['status'] != 'native_candidate' or r['native_token'] is None})
    if bad:
        return 'token_unverified:' + ','.join(bad)
    mine = sorted((r for r in earlier if r['native_token']['pitcher'] == pitcher), key=lambda r: tuple(r['key'])[1:], reverse=True)[:CONTEXT]
    t = {'valid': np.zeros(CONTEXT, bool), 'type': np.zeros(CONTEXT, np.int64), 'out': np.zeros(CONTEXT, np.int64),
         'balls': np.zeros(CONTEXT, np.int64), 'strikes': np.zeros(CONTEXT, np.int64), 'same': np.zeros(CONTEXT, np.int64),
         'pos': np.arange(1, CONTEXT + 1, dtype=np.int64)}
    for j, r in enumerate(mine):
        tok = r['native_token']
        if tok['pitch_type'] not in TYPE_INDEX:
            return 'token_type_outside_vocabulary'
        t['valid'][j] = True; t['type'][j] = TYPE_INDEX[tok['pitch_type']]; t['out'][j] = tok['outcome10']
        t['balls'][j] = tok['balls']; t['strikes'][j] = tok['strikes']; t['same'][j] = int(tok['batter'] == batter)
    return t


class LiveInputs:
    """Per-game converter. Stateless per snapshot except a playId memory used only to flag a rebuild."""

    def __init__(self, game_pk, game_date, encoder: Encoder11, version='v2'):
        if version not in W1_VERSIONS:
            raise ValueError(f'W1 version {version!r} not in {W1_VERSIONS}')
        self.game_pk, self.game_date, self.encoder, self.version = game_pk, game_date, encoder, version
        self.inspect = {'v3': w1_v3.inspect_tokens, 'v4': w1_v4.inspect_tokens}.get(version, captured_native.inspect_tokens)
        self.seen = {}  # (ab, pitch) -> playId from earlier snapshots
        self.receipt_seq = 0
        self.reassigned = 0

    def convert(self, feed, source_sha256=None):
        self.receipt_seq += 1
        if source_sha256 is None:
            source_sha256 = hashlib.sha256(json.dumps(feed, sort_keys=True, default=str).encode()).hexdigest()
        try:
            stripped, removed = strip_preserving_actions(feed)
            step = captured_bridge.step(stripped, game_pk=self.game_pk, game_date=self.game_date,
                                        receipt_seq=self.receipt_seq, source_sha256=source_sha256)
        except Exception as e:  # noqa: BLE001 - feed content never raises into the service
            return Omit(f'bridge_error:{type(e).__name__}')
        flags = ['preserving_actions_removed'] if removed else []
        current = {tuple(ev['key'])[1:]: ev['event_id'] for ev in step.get('events') or ()}
        if any(k in self.seen and self.seen[k] != pid for k, pid in current.items()):
            self.reassigned += 1
            flags.append('rebuilt_after_reassignment')
        self.seen.update(current)
        cand = step.get('candidate')
        if cand is None:
            return Omit(step.get('abstain_reason') or 'no_candidate')
        q = cand['query']
        key = (q['game_pk'], q['at_bat_number'], q['pitch_number'])
        pitch_key = f'{key[0]}:{key[1]}:{key[2]}'
        try:
            rows = self.inspect(stripped, game_pk=self.game_pk, game_date=self.game_date,
                                                  receipt_seq=self.receipt_seq, source_sha256=source_sha256)
        except Exception as e:  # noqa: BLE001
            return Omit(f'token_error:{type(e).__name__}', pitch_key)
        tokens = tokens_for(rows, key, q['pitcher'], q['batter'])
        if isinstance(tokens, str):
            return Omit(tokens, pitch_key)
        if any(r['status'] == w1_v3.EXCLUDED and r['key'] is not None and tuple(r['key'])[1:] < key[1:] for r in rows):
            flags.append('w1v3_unmapped_bip_excluded_from_context')
        if self.version == 'v4' and any(pa + 1 <= key[1] for pa, *_ in w1_v4.fill_missing_replaced(stripped)[1]):
            flags.append('w1v4_replaced_player_inferred')
        raw = raw11(q)
        if not np.isfinite(raw).all():
            return Omit('state_not_finite', pitch_key)
        return ModelInputs(pitch_key, int(q['pitcher']), int(q['batter']), raw, self.encoder.encode(raw)[0], tokens, tuple(flags))
