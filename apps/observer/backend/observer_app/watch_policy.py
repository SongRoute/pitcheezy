"""A stand-in for the live ARM-B policy in a REPLAY of a completed game (exhibition, display only): every pending
pitch is answered with the ``pre`` recommendation a watch-along dataset (``scripts/demo_precompute.py``,
``pitcheezy-watch-along-v1``) precomputed for that pitch. Nothing is computed: no model binding, no MPS.

Enabled by ``PITCHEEZY_OBSERVER_LIVE_WATCH_POLICY=<watch json>`` (``live_armb._default_factory``); unset, the live
``demo_precompute.LivePolicy`` runs exactly as before.

The result has the shape ``demo_precompute.LivePolicy.__call__`` returns (``main.armb_view`` reads it):

* ``live_rows(feed)`` (the live policy's own pre-pitch rows) is None -> None, as the live policy;
* the pending pitch ``gamePk:at_bat_number:pitch_number`` (``demo_precompute._key`` of the last live row) and every
  earlier pitch of the same at-bat get the watch decision of that key: ``pre`` as stored, ``status`` = the
  decision's ``root_status`` (the refusal of that pitch itself; the live pre-pitch call never carries a sticky
  MID_PA refusal or an actual-pitch refusal) or ``SUPPORTED`` when the stored ``pre`` is ready. One known
  difference from the live policy: after a MID-AT-BAT pitcher change the dataset keeps the ledger's sticky refusal
  (the live call evaluates the new pitcher afresh); that pitch is shown as ``UNSUPPORTED_MID_PA`` with its sentence;
* ``actual`` and ``previous`` come from the live rows through ``demo_precompute.actual_view`` (what the live policy
  shows), ``home_we_now``/``previous.home_we_before`` = the stored ``we.home_before`` of those pitches (the same
  frozen C0 value of the same pre-pitch state);
* fail closed: a key the dataset does not hold, or a decision whose stored situation (pitcher, batter, count, outs,
  inning) differs from the live row, is never shown under this pitch's name: ``pre`` is unsupported with
  ``NO_ROW_REASON`` and ``status`` ``WATCH_NO_ROW`` / ``WATCH_STATE_DIFFERS`` (``display_state`` shows the sentence);
* another game -> None.

``policy_identity`` is the dataset's ``policy.identity_sha256`` (the policy that made the stored candidates).
``counters`` says how each call was answered.
"""
from __future__ import annotations

from collections import Counter
import copy
import json
from pathlib import Path
import sys

from .settings import REPO

SCHEMA = 'pitcheezy-watch-along-v1'
ENV = 'PITCHEEZY_OBSERVER_LIVE_WATCH_POLICY'
MID_PA = 'UNSUPPORTED_MID_PA'
PITCHER_LEVEL = ('UNSUPPORTED_UNKNOWN_PITCHER', 'UNSUPPORTED_PITCHER_HAND', 'UNSUPPORTED_EMPTY_SUPPORT')
READY_STATUS, NO_ROW_STATUS, DIFFERS_STATUS = 'SUPPORTED', 'WATCH_NO_ROW', 'WATCH_STATE_DIFFERS'
NO_ROW_REASON = '이 상황에 미리 계산된 추천이 없어요.'  # <= 60 chars, Korean, no code: display_state shows it as is


class WatchUnusable(ValueError):
    """The watch dataset cannot be the input of this policy."""


def _key(game_pk, at_bat, pitch):
    return f'{int(game_pk)}:{int(at_bat)}:{int(pitch)}'


def load(path):
    """``(game_pk, identity, {pitch key: decision})`` from a watch dataset; refuses anything else."""
    try:
        data = json.loads(Path(path).read_bytes())
    except (OSError, ValueError) as error:
        raise WatchUnusable(f'watch dataset unreadable: {type(error).__name__}') from None
    if not isinstance(data, dict) or data.get('schema') != SCHEMA:
        raise WatchUnusable('not a pitcheezy-watch-along-v1 dataset')
    game_pk = (data.get('game') or {}).get('game_pk')
    identity = (data.get('policy') or {}).get('identity_sha256')
    if type(game_pk) is not int or game_pk < 1:
        raise WatchUnusable('watch dataset has no gamePk')
    if not isinstance(identity, str) or len(identity) != 64 or any(c not in '0123456789abcdef' for c in identity):
        raise WatchUnusable('watch dataset has no 64-hex policy identity')
    decisions = data.get('decisions') or []
    if not decisions or [d.get('index') for d in decisions] != list(range(len(decisions))):
        raise WatchUnusable('watch decisions must be indexed 0..n-1')
    rows = {}
    for d in decisions:
        pre = d.get('pre')
        if (not isinstance(pre, dict) or set(pre) != {'status', 'reason', 'recommendation'}
                or pre['status'] not in ('ready', 'unsupported')
                or (pre['status'] == 'ready') != bool((pre.get('recommendation') or {}).get('candidates'))):
            raise WatchUnusable(f"decision {d.get('index')}: pre is not a ready/unsupported view")
        key = _key(game_pk, d['at_bat_number'], d['pitch_number'])
        if key in rows:
            raise WatchUnusable(f'two decisions for {key}')
        rows[key] = d
    return game_pk, identity, rows


def _situation(row):
    return (int(row['pitcher']), int(row['batter']), int(row['balls']), int(row['strikes']), int(row['outs_when_up']),
            int(row['inning']), row['inning_topbot'])


def _stored(decision):
    s = decision['situation']
    return (int(decision['pitcher']['id']), int(decision['batter']['id']), int(s['balls']), int(s['strikes']),
            int(s['outs']), int(s['inning']), s['half'])


class WatchPolicy:
    """Callable like ``demo_precompute.LivePolicy``: ``policy(feed) -> result``."""

    def __init__(self, path):
        self.path = str(path)
        self.game_pk, self.identity, self.rows = load(path)
        self.counters = Counter()
        self.binds = 0  # nothing is bound
        scripts = str(REPO/'scripts')
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        import demo_precompute as dp  # live_rows / actual_view / _location_settings: the live policy's own code
        self._dp = dp
        self.bounds = dp._location_settings()[1]

    def _pitch(self, row, *, pending):
        key = self._dp._key(row)
        decision = self.rows.get(key)
        if decision is None:
            status, pre = NO_ROW_STATUS, None
        elif _stored(decision) != _situation(row):
            status, pre = DIFFERS_STATUS, None
        else:
            pre = copy.deepcopy(decision['pre'])
            status = READY_STATUS if pre['status'] == 'ready' else decision.get('root_status') or decision['status']
            if pre['status'] != 'ready' and decision['status'] == MID_PA and status in PITCHER_LEVEL and not any(
                    d['pitcher']['id'] == decision['pitcher']['id'] and d['status'] == status for d in self._pa(decision)):
                # the at-bat's pitcher-level refusal belonged to an earlier pitcher (mid-PA change): the dataset kept
                # the ledger's sticky refusal, so say that, not "unknown pitcher" about a pitcher who is not
                status, pre['reason'] = MID_PA, self._dp.STATUS_TEXT[MID_PA]
        if pre is None:
            pre = {'status': 'unsupported', 'reason': NO_ROW_REASON, 'recommendation': None}
        if pending:
            self.counters[status] += 1
        return {'key': key, 'pitch_number': int(row['pitch_number']), 'status': status, 'pre': pre,
                'actual': None if row['placeholder'] else self._dp.actual_view(row, self.bounds)}, decision

    def _pa(self, decision):
        """The stored decisions of ``decision``'s at-bat before it."""
        return [self.rows[k] for k in (_key(self.game_pk, decision['at_bat_number'], n) for n in range(1, decision['pitch_number']))
                if k in self.rows]

    def _we(self, row, decision):
        if decision is None or _stored(decision) != _situation(row):
            return None
        return (decision.get('we') or {}).get('home_before')

    def __call__(self, feed):
        if int(feed['gameData']['game']['pk']) != self.game_pk:
            self.counters['other_game'] += 1
            return None
        rows = self._dp.live_rows(feed)
        if rows is None:
            self.counters['no_rows'] += 1
            return None
        records = rows.to_dict('records')
        last = records[-1]
        same_pa = [r for r in records if int(r['at_bat_number']) == int(last['at_bat_number'])]
        pitches, decisions = [], []
        for i, row in enumerate(same_pa):
            pitch, decision = self._pitch(row, pending=i == len(same_pa) - 1)
            pitches.append(pitch)
            decisions.append(decision)
        home_we_now = self._we(last, decisions[-1])
        previous = records[-2] if len(records) > 1 else None
        if previous is not None:
            before = self._we(previous, self.rows.get(self._dp._key(previous)))
        return {'key': pitches[-1]['key'], 'pitch': pitches[-1], 'pa_pitches': pitches[:-1],
                'previous': None if previous is None else {
                    'key': self._dp._key(previous), 'actual': self._dp.actual_view(previous, self.bounds),
                    'inning': int(previous['inning']), 'half': previous['inning_topbot'],
                    'home_we_before': before},
                'home_we_now': home_we_now, 'policy_identity': self.identity,
                'seconds': {'bind': 0.0, 'total': 0.0}}

    def record(self):
        return {'kind': 'watch', 'path': self.path, 'game_pk': self.game_pk, 'policy_identity': self.identity,
                'decisions': len(self.rows), 'answers': dict(self.counters)}
