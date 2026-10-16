"""One explicit display state for the pending pitch slot of a live game, derived from store and ledger facts.

The fan screen (``/api/service/games/{pk}``) and the legacy live view (``/api/live/{pk}/state``) both carry the
``display_state`` built here, so what a viewer sees is never inferred by the client from missing fields.

States (``state``) and what they mean for the recommendation area:

* ``recommended``       a recommendation is shown (ledger-committed before display when a store is on)
* ``unconfirmed``       확인 중: the data the recommendation would rest on is not admitted (SVC-WS-03); withheld
* ``unsupported``       the frozen policy does not cover this pitcher / pitch types / at-bat; one short reason
* ``feed_delayed``      a recommendation is shown, but the snapshot reached us later than the threshold
* ``disconnected``      the last upstream poll failed; only the last snapshot is shown, nothing new is recommended
* ``ledger_unavailable`` a recommendation was computed but could not be committed, so it is not shown
* ``waiting``           nothing to recommend right now (inning change, result pending, computing, buffering)
* ``unavailable``       this run has no recommender (off, model not loaded, computation failed)
* ``pregame`` / ``final``

The admission rule (``admit_body``) is the SVC-WS-03 selected-body rule; ``synthetic_game`` calls the same
function, so the synthetic walk-through and the real screen cannot drift apart.

Thresholds are PROVISIONAL configuration (``live_config.json`` ``display``), not validated acceptance limits.
"""
from __future__ import annotations

import hashlib
import os
import re
import time

from .live_store import _canonical, view_changed

SCHEMA = 'pitcheezy-display-state-v1'
STATES = ('recommended', 'unconfirmed', 'unsupported', 'feed_delayed', 'disconnected', 'ledger_unavailable',
          'waiting', 'unavailable', 'pregame', 'final')
# Provider-revision order of the served store version -> why it is not fresh input (same table as the research gate).
FRESHNESS = {'ordered': None, 'stale': 'stale_provider_stamp', 'unorderable': 'unorderable_provider_stamp',
             'same_stamp_conflict': 'same_stamp_conflict_provider_stamp'}
# Provisional default: NOT a measured acceptance limit. Background only: one live game (849844, 2026-10-02) had a
# median feed lag of about 19 s; 45 s is a first guess for "noticeably later than usual" and must be revisited.
DEFAULTS = {'feed_lag_delayed_s': 45.0}
RECEIPT_ALL = '__all__'  # receipts marker: the feed has a play without a usable atBatIndex, so no key can be trusted

POLICY_REASONS = {
    'UNSUPPORTED_UNKNOWN_PITCHER': 'unknown_pitcher', 'UNSUPPORTED_PITCHER_HAND': 'unknown_pitcher',
    'UNSUPPORTED_EMPTY_SUPPORT': 'no_supported_pitch_types',
    'UNSUPPORTED_LOGGING_POSITIVITY': 'at_bat_rejected', 'UNSUPPORTED_MID_PA': 'at_bat_rejected',
    'UNSUPPORTED_NO_LOGGED_ACTION': 'at_bat_rejected', 'UNSUPPORTED_MISSING_ACTION_LABEL': 'at_bat_rejected',
    'UNSUPPORTED_INCOMPLETE_START': 'at_bat_rejected', 'UNSUPPORTED_INCONSISTENT_HISTORY': 'at_bat_rejected',
    'NOT_SUBMITTED': 'at_bat_rejected', 'NO_DECISION': 'at_bat_rejected'}

# The 확인 중 family for a first-time fan: what is happening and what comes next, without a word about how the data
# arrives (the reason code keeps the exact cause for operators).
RECHECK_GAME = '경기 상황을 다시 확인하고 있어요. 확인되면 추천을 보여 드려요.'
RECHECK_PITCH = '방금 던진 공을 다시 확인하고 있어요. 확인되면 추천을 보여 드려요.'
# Words that describe the pipeline, not the game: no sentence a fan reads in the 확인 중 family may contain them.
PIPELINE_WORDS = ('자료', '시각', '순서', '격리', '스냅샷', '원장')
# (state, reason) -> (short label, one calm sentence). The frontend renders these as they are.
COPY = {
    ('recommended', 'ledger_recorded'): ('다음 공 추천', None),
    ('recommended', 'ledger_conflict_first_kept'): ('다음 공 추천', None),
    ('recommended', 'shown_unrecorded_no_ledger'): ('다음 공 추천', None),
    ('unconfirmed', 'stale_provider_stamp'): ('확인 중', RECHECK_GAME),
    ('unconfirmed', 'unorderable_provider_stamp'): ('확인 중', RECHECK_GAME),
    ('unconfirmed', 'same_stamp_conflict_provider_stamp'): ('확인 중', RECHECK_GAME),
    ('unconfirmed', 'provider_order_unknown'): ('확인 중', RECHECK_GAME),
    ('unconfirmed', 'prior_pitch_unconfirmed'): ('확인 중', RECHECK_PITCH),
    ('unconfirmed', 'receipts_unreadable'): ('확인 중', RECHECK_GAME),
    ('unsupported', 'unknown_pitcher'): ('추천 없음', '이 투수는 2025년까지 쌓인 기록이 없어 추천하지 않아요.'),
    ('unsupported', 'no_supported_pitch_types'): ('추천 없음', '이 투수가 이 타석 방향에 던진 구종 기록이 부족해 추천하지 않아요.'),
    ('unsupported', 'at_bat_rejected'): ('추천 없음', '이 타석은 앞선 공 기록을 이어 볼 수 없어 추천을 쉬어요.'),
    ('unsupported', 'policy_refused'): ('추천 없음', '이 상황은 추천 대상이 아니에요.'),
    ('unsupported', 'game_type_out_of_scope'): ('추천 없음', '이 경기는 추천 대상이 아니에요.'),
    ('feed_delayed', 'feed_lag_above_threshold'): ('자료 늦음', '경기 자료가 평소보다 늦게 도착했어요. 추천이 실제 경기보다 늦을 수 있어요.'),
    ('ledger_unavailable', 'display_history_unreadable'): ('추천 보류', '앞서 보여 드린 추천 기록을 확인하지 못해 이번 추천은 표시하지 않아요. 경기 상황은 계속 보여 드려요.'),
    ('disconnected', 'poll_failed_last_snapshot_only'): ('연결 끊김', '경기 자료 연결이 잠시 끊겼어요. 마지막으로 받은 내용을 보여 드려요.'),
    ('ledger_unavailable', None): ('추천 보류', '추천 기록을 저장하지 못해 이번 추천은 표시하지 않아요. 경기 상황은 계속 보여 드려요.'),
    ('waiting', 'between_innings'): ('공수 교대', '공수 교대 중이에요.'),
    ('waiting', 'resolving'): ('결과 확인 중', '방금 공의 결과를 기다리고 있어요.'),
    ('waiting', 'computing'): ('계산 중', '추천을 계산하고 있어요.'),
    ('waiting', 'no_pending_pitch'): ('대기 중', '다음 공을 기다리고 있어요.'),
    ('waiting', 'buffering'): ('대기 중', '내 중계 시점까지 기다리고 있어요.'),
    ('unavailable', 'recommender_off'): ('추천 꺼짐', '이 실행에서는 실시간 추천을 제공하지 않아요.'),
    ('unavailable', 'recommender_unavailable'): ('추천 준비 안 됨', '추천 모델을 준비하지 못했어요.'),
    ('unavailable', 'compute_failed'): ('추천 없음', '이 상황의 추천을 계산하지 못했어요.'),
    ('pregame', 'scheduled'): ('경기 전', '경기 시작 전이에요.'),
    ('pregame', 'not_live'): ('진행 중 아님', '지금은 경기가 진행 중이 아니에요.'),
    ('final', 'game_over'): ('경기 종료', '경기가 끝났어요.'),
}
CORRECTION_COPY = {'replaced': '상황이나 기록이 바뀌어 추천이 달라졌어요.',
                   'withdrawn': '조금 전에 보여 드린 추천은 지금 표시하지 않아요.'}


def config(live_config=None):
    """Display thresholds: ``live_config.json`` ``display`` over ``DEFAULTS``; the env var wins. All provisional."""
    if live_config is None:
        from .live_feed import LIVE_CONFIG as live_config
    out = DEFAULTS | {k: v for k, v in (live_config.get('display') or {}).items() if k in DEFAULTS}
    override = os.environ.get('PITCHEEZY_OBSERVER_FEED_LAG_DELAYED_S')
    if override:
        out['feed_lag_delayed_s'] = float(override)
    return out | {'provisional': True}


# ---------- SVC-WS-03 selected-body admission (shared with synthetic_game) ----------
def _sha(value):
    return value if isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value) else None


def body_sha(event):
    """The pitch ledger's hash of a playEvent body (``sha256(_canonical(playEvent))``), or None if unusable."""
    try:
        return hashlib.sha256(_canonical(event)).hexdigest()
    except (TypeError, ValueError):
        return None


def admit_body(event, key, rows):
    """May the displayed pitch body ``event`` be shown as an observed pitch? Returns (status, detail).

    ``rows`` are committed pitch-ledger rows (``LiveSnapshotStore.pitch_events`` events) and ``key`` the official
    ``(at_bat_number, pitch_number)``. The body is ``received`` only if its hash is the key's NEWEST accepted
    (new/correction) row: the ledger dedups repeat quarantine rows whatever the seq, so neither the absence of a
    quarantine row at the selected seq nor an older accepted copy of the body clears it. A body that is only
    quarantined, superseded by a later accepted revision, never accepted, or whose hash is unusable is
    ``unconfirmed`` (fail closed) with the quarantine reason, ``not_accepted`` or ``body_hash_unusable``.
    Known imprecision (Q1): an unselected stale arrival whose per-pitch correction was accepted makes the
    selected (older) body ``unconfirmed/not_accepted``; that is fail-safe, but the reason is inexact."""
    accepted = [r for r in rows if r['outcome'] != 'quarantine' and (r['at_bat_number'], r['pitch_number']) == tuple(key)]
    shown = body_sha(event)
    if shown is None or (accepted and _sha(accepted[-1]['sha']) is None):
        return 'unconfirmed', 'body_hash_unusable'
    if accepted and accepted[-1]['sha'] == shown:
        return 'received', None
    refused = [r for r in rows if r['outcome'] == 'quarantine' and r['sha'] == shown]
    return 'unconfirmed', refused[-1]['reason'] if refused else 'not_accepted'


HELD = 'held_last_accepted'


def _corrected(accepted):
    """Did an accepted revision of this pitch change what a viewer was already shown? Consecutive accepted rows
    are compared on their ``view`` (``live_store.pitch_view``); a row without one (an older caller) counts as a
    change whenever it is a correction, as before."""
    for before, after in zip(accepted, accepted[1:]):
        if after['outcome'] != 'correction':
            continue
        if 'view' not in before or 'view' not in after or view_changed(before['view'], after['view']):
            return True
    return False


def pitch_receipts(feed, rows, *, committed=True):
    """Admission of every pitch body in ``feed`` against the pitch ledger ``rows``:
    ``{(at_bat_number, pitch_number): {'status', 'reason', 'corrected', 'sha', 'held', ...}}``. A pitch without a
    usable official key (``incomplete_identity``) is keyed by its position ``(at_bat_number or None, pitchNumber or
    None)`` and is never ``received``. ``committed=False`` (memory-only entry during a store outage) relabels a body
    the ledger never accepted as ``uncommitted_memory_entry``.

    Held (Song, 2026-10-12): a pitch whose served body is not admitted but whose official key the ledger HAS
    accepted before stays on screen with its last accepted body: ``status`` 'received', ``reason`` ``HELD`` and
    ``held`` = ``{'id', 'sha', 'reason'}`` (the accepted row to show and why the served body was refused). The
    caller must show that row's body (``attach_held`` + ``held_feed``), never the served one. Only a key no body
    of which was ever accepted is ``unconfirmed``. A held pitch is not a decision input (``_blocking_receipt``).

    ``corrected`` = an accepted revision changed a field the viewer had already been shown (``_corrected``);
    provider bookkeeping (end times, refined break terms, ids) re-sent under a later stamp is not a correction.
    ``first_seq``/``first_received`` are the store version and receipt of the first body that held the pitch (its
    ``new`` row), or None."""
    by_key, by_sha = {}, {}
    for row in rows:
        if row['outcome'] == 'quarantine':
            by_sha.setdefault(row['sha'], []).append(row)
        else:
            by_key.setdefault((row['at_bat_number'], row['pitch_number']), []).append(row)
    out, tainted = {}, set()
    live = feed.get('liveData') if isinstance(feed, dict) else None
    plays = (live.get('plays') or {}).get('allPlays') if isinstance(live, dict) else None
    for play in plays if isinstance(plays, list) else []:
        about = play.get('about') if isinstance(play, dict) else None
        index = about.get('atBatIndex') if isinstance(about, dict) else None
        at_bat = index+1 if type(index) is int and index >= 0 else None
        events = play.get('playEvents') if isinstance(play, dict) else None
        for event in events if isinstance(events, list) else []:
            if not isinstance(event, dict) or not event.get('isPitch'):
                continue
            number = event.get('pitchNumber') if type(event.get('pitchNumber')) is int else None
            key, sha = (at_bat, number), body_sha(event)
            mine = by_key.get(key, [])
            status, reason = admit_body(event, key, [*mine, *by_sha.get(sha, [])])
            if status != 'received' and reason == 'not_accepted' and not committed:
                reason = 'uncommitted_memory_entry'
            held = None
            if (status != 'received' and at_bat is not None and number is not None and mine
                    and _sha(mine[-1]['sha']) is not None and type(mine[-1].get('id')) is int):
                held = {'id': mine[-1]['id'], 'sha': mine[-1]['sha'], 'reason': reason}
                status, reason = 'received', HELD
            first = mine[0] if mine and mine[0]['outcome'] == 'new' else None  # the body that first held this pitch
            out[key] = {'status': status, 'reason': reason, 'sha': sha, 'held': held,
                        'corrected': status == 'received' and _corrected(mine),
                        'first_seq': first and first['store_seq'], 'first_received': first and first['received']}
            if number is None or at_bat is None:
                tainted.add(at_bat)
    # A pitch without an official key is listed by position downstream (``pitchNumber or its order``), which can
    # collide with a real key: every pitch of such an at-bat (of the whole feed, if the at-bat itself is unknown)
    # fails closed rather than borrow another pitch's admission (a held one included: its slot is not certain).
    for key, item in out.items():
        if (None in tainted or key[0] in tainted) and item['status'] == 'received':
            item.update(status='unconfirmed', reason='incomplete_identity', corrected=False, held=None)
    if None in tainted:
        out[RECEIPT_ALL] = {'status': 'unconfirmed', 'reason': 'incomplete_identity', 'sha': None, 'corrected': False,
                            'held': None}
    return out


def attach_held(receipts, load):
    """Give every held receipt the accepted body it stands for: ``load(ids) -> {row id: playEvent}`` (a store
    read). A held pitch whose body cannot be loaded falls back to ``unconfirmed`` with the served body's own
    refusal reason, so nothing is ever shown that the ledger did not accept. Returns ``receipts``."""
    wanted = [item['held']['id'] for item in (receipts or {}).values() if item.get('held')]
    if not wanted:
        return receipts
    try:
        bodies = load(wanted) or {}
    except Exception:  # noqa: BLE001 - any read failure: fail closed below
        bodies = {}
    for item in receipts.values():
        held = item.get('held')
        if not held:
            continue
        body = bodies.get(held['id'])
        if isinstance(body, dict) and body_sha(body) == held['sha']:
            held['body'] = body
        else:
            item.update(status='unconfirmed', reason=held['reason'] or 'not_accepted', corrected=False, held=None)
    return receipts


def held_feed(feed, receipts):
    """``feed`` with every held pitch's served body replaced by its last accepted body (``attach_held``), in
    ``allPlays`` and in ``currentPlay``, for building what the viewer is shown. Plays without a held pitch are
    shared, not copied. Returns ``feed`` itself when nothing is held. Never used for a decision: the policy
    reads the served snapshot, and a held pitch among its inputs withholds the recommendation."""
    held = {key: item['held']['body'] for key, item in (receipts or {}).items()
            if key != RECEIPT_ALL and item.get('held') and 'body' in item['held']}
    if not held:
        return feed

    def swap(play):
        if not isinstance(play, dict):
            return play
        at_bat, _ = _pitch_keys(play)
        events = play.get('playEvents')
        if at_bat is None or not isinstance(events, list):
            return play
        new = [held.get((at_bat, e.get('pitchNumber')), e) if isinstance(e, dict) and e.get('isPitch')
               and type(e.get('pitchNumber')) is int else e for e in events]
        return play if all(a is b for a, b in zip(new, events)) else play | {'playEvents': new}
    plays = feed['liveData']['plays']
    out = plays | {'allPlays': [swap(play) for play in plays.get('allPlays') or []]}
    if isinstance(plays.get('currentPlay'), dict):
        out['currentPlay'] = swap(plays['currentPlay'])
    return feed | {'liveData': feed['liveData'] | {'plays': out}}


def fallback_receipts(feed, remembered):
    """Receipts when the pitch ledger cannot be read: a body is ``received`` only if this process already saw
    exactly that body admitted (``remembered``: key -> sha); everything else is unconfirmed."""
    out = pitch_receipts(feed, [], committed=True)
    for key, item in out.items():
        if item['sha'] is not None and remembered.get(key) == item['sha'] and item['reason'] != 'incomplete_identity':
            item.update(status='received', reason=None)
        else:
            item.update(status='unconfirmed', reason='receipts_unreadable')
    return out


def public_receipt(item):
    """A receipt as the screen gets it (no hashes)."""
    return {'status': item['status'], 'reason': item['reason'], 'corrected': bool(item['corrected'])}


def receipts_version(receipts):
    """Changes whenever any pitch's admission, hold or correction mark changes, so a client that only asks for
    newer pitches knows to take the full history again."""
    if receipts is None:
        return None
    marks = sorted((str(k), v['status'], v['reason'] or '', v['corrected']) for k, v in receipts.items()
                   if v['status'] != 'received' or v['corrected'] or v.get('held'))
    return hashlib.sha256(_canonical(marks)).hexdigest()[:12]


# ---------- facts ----------
def pending_key(feed):
    """The policy's name (``gamePk:at_bat_number:pitch_number``) for the pitch about to be thrown, or None."""
    try:
        plays = feed['liveData']['plays']
        done = plays.get('allPlays') or []
        play = plays.get('currentPlay') or (done[-1] if done else None)
        pk = feed['gameData']['game']['pk']
        if play is None:
            return f'{pk}:1:1'
        if play['about'].get('isComplete', True):
            return f"{pk}:{play['about']['atBatIndex']+2}:1"
        numbered = [e for e in play.get('playEvents') or []
                    if e.get('isPitch') or (e.get('type') == 'no_pitch' and ((e.get('details') or {}).get('call') or {}).get('code'))]
        return f"{pk}:{play['about']['atBatIndex']+1}:{len(numbered)+1}"
    except (KeyError, TypeError, AttributeError):
        return None


def summary(recommendation):
    """What a viewer can see of a recommendation: the ordered (pitch type, zone) of its candidates."""
    return [[c.get('pitch_type'), c.get('zone_id')] for c in ((recommendation or {}).get('candidates') or [])[:3]]


def _iso(wall):
    return None if wall is None else time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(wall))


def feed_facts(feed, game_pk, entry, cfg, *, now, viewer_delay_s=0.0):
    """Connection facts of the served snapshot. Polling runs only while someone watches, so ``receipt - event``
    alone may just be time nobody polled. ``lag_s`` is therefore the PROVEN minimum: the last good poll before the
    one that brought this event did not have it yet, so the event reached us at least ``prior poll - event`` late
    (``lag_max_s`` = ``receipt - event`` is the upper bound). Without such a prior poll (first poll, restart) or for
    a replayed recording (its event times are in the past) nothing is measured and nothing is called delayed.
    ``viewer_delay_s`` is the broadcast delay this viewer chose: data is late FOR THEM only when it reached us later
    than their own delay as well, so choosing a delay is never itself reported as late data."""
    game_pk = int(game_pk)
    failing = game_pk in getattr(feed, 'failing', {})
    prior = (entry or {}).get('prior_poll')
    measured = entry is not None and prior is not None and _is_real_poll(feed)
    lag = round(max(0.0, prior-entry['event']), 1) if measured else None
    since = getattr(feed, 'failing_since', {}).get(game_pk)
    has_store, store_error = hasattr(feed, 'store'), getattr(feed, 'store_error', None)
    return {'ok': not failing, 'lag_s': lag, 'lag_max_s': round(entry['fetched']-entry['event'], 1) if measured else None,
            'lag_measured': measured, 'lag_threshold_s': cfg['feed_lag_delayed_s'],
            'delayed': lag is not None and lag > max(cfg['feed_lag_delayed_s'], viewer_delay_s or 0.0),
            'viewer_delay_s': viewer_delay_s or 0.0, 'thresholds_provisional': True,
            'last_ok_at': _iso(getattr(feed, 'polled_ok', {}).get(game_pk)), 'failing_since': _iso(since) if failing else None,
            'failing_for_s': round(now-since, 1) if failing and since is not None else None,
            'received_at': _iso(entry['fetched']) if entry else None,
            'provider_order': (entry or {}).get('provider_order'),
            'same_stamp_scope': _same_stamp_scope(feed, game_pk, entry),
            # 'failed': a store was configured but could not be opened, so nothing can be recorded (fail closed)
            'store': 'on' if has_store else 'failed' if store_error else 'off', 'store_ok': store_error is None,
            'committed': None if not has_store else bool(entry and entry.get('seq') is not None and entry.get('sha') is not None)}


def _same_stamp_scope(feed, game_pk, entry):
    """``LiveSnapshotStore.same_stamp_scope`` of the served version when its provider order is a same-stamp
    conflict: 'extras_only', 'inputs', or None (no conflict, no committed version, or the store could not say,
    which keeps the gate)."""
    if (entry or {}).get('provider_order') != 'same_stamp_conflict' or entry.get('seq') is None:
        return None
    reader = getattr(getattr(feed, 'store', None), 'same_stamp_scope', None)
    try:
        return reader(game_pk, entry['seq']) if reader else None
    except Exception:  # noqa: BLE001 - an unreadable store keeps the same-stamp gate (fail closed)
        return None


def _is_real_poll(feed):
    """Only a ``LiveFeed`` polling a real upstream measures lag: a store-backed feed says so in ``source``; a plain
    one must have a transport that is not a replay (``ReplayTransport`` / the timed replay, which has a position)."""
    from .live_feed import ReplayTransport
    source = getattr(feed, 'source', None)
    if source is not None:
        return source == 'live_poll'
    transport = getattr(feed, 'transport', None)
    return transport is not None and not isinstance(transport, ReplayTransport) and not hasattr(transport, 'position')


def gate(parsed_status, abstract, facts, receipts, feed):
    """The state that is decided before any recommendation is looked up, or None when the recommender may be asked.
    Returns (state, reason, detail). For ``disconnected``, ``detail`` is None only if the last snapshot's inputs
    are themselves admitted: otherwise it names the 확인 중 reason and nothing may stay on screen."""
    if abstract == 'Final':
        return 'final', 'game_over', None
    if parsed_status == 'not_live':
        return 'pregame', 'scheduled' if abstract == 'Preview' else 'not_live', None
    if parsed_status == 'unsupported_game_type':
        return 'unsupported', 'game_type_out_of_scope', None
    blocked = _unconfirmed(facts, receipts, feed) if parsed_status == 'ready' else None
    if not facts['ok']:
        return 'disconnected', 'poll_failed_last_snapshot_only', None if blocked is None else f'unconfirmed:{blocked[0]}'
    if parsed_status in ('between_innings', 'resolving'):
        return 'waiting', parsed_status, None
    if parsed_status != 'ready':
        return 'waiting', 'no_pending_pitch', parsed_status
    return None if blocked is None else ('unconfirmed', *blocked)


def _unconfirmed(facts, receipts, feed):
    """(reason, detail) if the pending decision's inputs are not admitted (SVC-WS-03), else None.

    The game-level same-stamp gate (Song, 2026-10-12) holds only when the conflict touches the decision: it is
    lifted when the store says the conflicting bodies share the decision identity (``same_stamp_scope``
    'extras_only': weather, attendance, boxscore and other extras differ) and whenever the served game has no
    pitch yet. Stale and unorderable stamps keep the gate as before."""
    if facts['store'] == 'on' and facts['committed']:
        order = facts['provider_order']
        stale = FRESHNESS.get(order, 'provider_order_unknown') if isinstance(order, str) else 'provider_order_unknown'
        if order == 'same_stamp_conflict' and (facts.get('same_stamp_scope') == 'extras_only' or pitch_count(feed) == 0):
            stale = None
        if stale:
            return stale, order
    if receipts is not None:
        blocking = _blocking_receipt(feed, receipts)
        if blocking is not None:
            return ('receipts_unreadable' if blocking[1] == 'receipts_unreadable' else 'prior_pitch_unconfirmed',
                    f'{blocking[0][0]}:{blocking[0][1]}/{blocking[1]}')
    return None


def pitch_count(feed):
    """Pitch events in the served body (``allPlays`` and ``currentPlay``); None if the body is unreadable."""
    plays = ((feed.get('liveData') or {}).get('plays') or {}) if isinstance(feed, dict) else None
    if not isinstance(plays, dict):
        return None
    listed = [p for p in plays.get('allPlays') or [] if isinstance(p, dict)]
    current = plays.get('currentPlay')
    if isinstance(current, dict) and current and not any(p is current for p in listed):
        listed.append(current)
    return sum(1 for p in listed for e in p.get('playEvents') or [] if isinstance(e, dict) and e.get('isPitch'))


def _pitch_keys(play):
    """(at-bat number or None, [official pitch numbers]) of a play."""
    index = (play.get('about') or {}).get('atBatIndex')
    at_bat = index+1 if type(index) is int and index >= 0 else None
    return at_bat, [event.get('pitchNumber') if type(event.get('pitchNumber')) is int else None
                    for event in play.get('playEvents') or [] if isinstance(event, dict) and event.get('isPitch')]


def unlisted_current(feed, receipts):
    """Where ``currentPlay`` and the listed plays disagree: ``{(at_bat_number, pitch_number): reason}``.

    The pitch ledger and ``pitch_receipts`` read ``allPlays`` only, while the pending decision and the legacy live
    view read ``currentPlay``. In a real feed ``currentPlay`` is the last ``allPlays`` entry, so the two name the
    same at-bat and the same pitches, each admitted with exactly that body. Anything else fails closed:

    * a ``currentPlay`` pitch that ``allPlays`` does not list (``not_in_pitch_ledger``) or lists with another body
      (``body_differs_from_listed``) was never admitted;
    * a ``currentPlay`` that LAGS the listed plays (an older at-bat, or fewer pitches than the last listed play)
      would have the next pitch decided on a state that is already over: the listed pitches it misses are named
      (``current_play_lags_listed``), or the at-bat itself (``current_play_not_last_listed``).

    Empty without a store (``receipts`` None)."""
    if receipts is None:
        return {}
    plays = ((feed.get('liveData') or {}).get('plays') or {}) if isinstance(feed, dict) else {}
    play = plays.get('currentPlay')
    if not isinstance(play, dict) or not play:
        return {}
    at_bat, numbers = _pitch_keys(play)
    out = {}
    for event in play.get('playEvents') or []:
        if not isinstance(event, dict) or not event.get('isPitch'):
            continue
        key = (at_bat, event.get('pitchNumber') if type(event.get('pitchNumber')) is int else None)
        item = receipts.get(key)
        if item is None:
            out[key] = 'not_in_pitch_ledger'
        elif item.get('sha') != body_sha(event):
            out[key] = 'body_differs_from_listed'
    listed = plays.get('allPlays')
    last = listed[-1] if isinstance(listed, list) and listed and isinstance(listed[-1], dict) else None
    if last is not None and _pitch_keys(last) != (at_bat, numbers):
        last_at_bat, last_numbers = _pitch_keys(last)
        missed = {(last_at_bat, number): 'current_play_lags_listed' for number in last_numbers
                  if last_at_bat != at_bat or number not in numbers}
        out |= {key: reason for key, reason in missed.items() if key not in out}
        if not out:
            out[(at_bat, None)] = 'current_play_not_last_listed'
    return out


def _blocking_receipt(feed, receipts):
    """The first not-admitted pitch among the pending decision's inputs: every pitch of the open at-bat, the
    game's newest pitch, and any disagreement between ``currentPlay`` and the listed plays (``unlisted_current``).
    Returns (key, reason) or None; for a held pitch the reason is why its served body was refused.

    Why a held pitch of the current at-bat withholds the recommendation (2026-10-17): the policy's state for the
    next pitch is the open at-bat's pitch history plus the count and situation (``policy_requests.pa_requests``),
    read from the SERVED snapshot, while the screen shows the held body. On 849831 the dropped bodies also lost
    pitch type and plate location, so a decision computed then would rest on inputs nobody accepted and its ledger
    row (append-only) could not be taken back. Pitches of earlier at-bats are not in that state, so a held or
    unconfirmed pitch there no longer stops the recommendation (receipt seq 310: 91 such pitches, none current)."""
    if receipts is None:
        return None
    if RECEIPT_ALL in receipts:
        return (None, None), receipts[RECEIPT_ALL]['reason']
    plays = feed['liveData']['plays']
    done = plays.get('allPlays') or []
    play = plays.get('currentPlay') or (done[-1] if done else None)
    open_ab = None
    if play is not None and not (play.get('about') or {}).get('isComplete', True):
        index = (play.get('about') or {}).get('atBatIndex')
        open_ab = index+1 if type(index) is int else None
    newest = list(receipts)[-1] if receipts else None
    for key, item in receipts.items():
        # a held pitch is shown with its last accepted body, but the policy reads the served one: not an input
        if (item['status'] != 'received' or item.get('held')) and (key == newest or (open_ab is not None and key[0] == open_ab)):
            return key, item['held']['reason'] if item.get('held') else item['reason']
    return next(iter(unlisted_current(feed, receipts).items()), None)


def correction(history, shown_now):
    """D1: was something else displayed for this pending pitch before? ``history`` is the earlier displays of the
    same pitch key, oldest first, each ``{'summary', 'decided', 'candidates', 'decision_id'}``, excluding the one
    shown now when it can be told apart. ``shown_now`` is the summary on screen now (None = nothing shown).
    ``previous`` is the most recent display that differs from the current one (after A, B, C, A, B it names C: the
    ledger keeps one row per decision, not the order in which repeats were served).
    Returns None or ``{'kind': 'replaced' | 'withdrawn', 'message', 'previous', 'previous_decided'}``."""
    differing = [h for h in history if h['summary'] and h['summary'] != shown_now]
    if not differing:
        return None
    last = differing[-1]
    kind = 'replaced' if shown_now else 'withdrawn'
    top = (last.get('candidates') or [{}])[0]
    return {'kind': kind, 'message': CORRECTION_COPY[kind],
            'previous': {'pitch_type': top.get('pitch_type'), 'pitch_label': top.get('pitch_label'),
                         'zone_id': top.get('zone_id'), 'zone_label': top.get('zone_label')},
            'previous_decided': _iso(last.get('decided'))}


def decide(*, gated, view, facts, history=(), pitch_key=None, recommender=True, held=False):  # noqa: C901
    """The single display state. ``gated`` is ``gate(...)``; ``view`` the raw recommendation view (``armb_view``
    after ``live_ledger.admit``) or None; ``history`` earlier displays for ``pitch_key`` (see ``correction``).
    ``history=None`` means the earlier displays could not be read: nothing is shown then.
    ``recommender=False``: this run has no recommender wired at all. ``held``: the recommendation in ``view`` was
    already displayed for this very decision (ledger row, or this process's memory when no ledger is on); only such
    a one may stay on screen while the feed is disconnected."""
    view = view or None
    ledger = (view or {}).get('ledger') or None
    rec = (view or {}).get('recommendation') if (view or {}).get('status') == 'ready' else None
    shown, detail = False, None
    if gated is not None and gated[0] != 'disconnected':
        state, reason, detail = gated
    elif gated is not None:  # disconnected: only a recommendation the ledger already held for this decision stays
        state, reason, detail = gated
        shown = bool(rec and rec.get('candidates')) and bool(held) and detail is None and history is not None
    elif not recommender:
        state, reason = 'unavailable', 'recommender_off'
    elif view is None:
        state, reason = 'waiting', 'no_pending_pitch'
    elif (ledger or {}).get('status') == 'withheld':
        state, reason, detail = 'ledger_unavailable', ledger.get('why') or 'write_failed', None
    elif view.get('status') == 'computing':
        state, reason = 'waiting', 'computing'
    elif view.get('status') == 'unavailable':
        state, reason = 'unavailable', view.get('code') or 'recommender_unavailable'
    elif view.get('status') == 'ready' and rec and rec.get('candidates'):
        shown = True
        reason = ('shown_unrecorded_no_ledger' if ledger is None else
                  'ledger_conflict_first_kept' if ledger.get('outcome') == 'conflict' else 'ledger_recorded')
        state = 'feed_delayed' if facts['delayed'] else 'recommended'
        if state == 'feed_delayed':
            detail, reason = reason, 'feed_lag_above_threshold'
    else:
        state, reason = 'unsupported', POLICY_REASONS.get(view.get('policy_status'), 'policy_refused')
        detail = view.get('policy_status')
    if shown and (history is None or facts['store'] == 'failed'):
        # D1 cannot be kept without knowing what was displayed before, and a configured store that did not open
        # cannot record anything: withhold rather than show something that may be an unannounced change.
        state, detail, shown = 'ledger_unavailable', reason, False
        reason = 'display_history_unreadable' if history is None else 'store_open_failed'
    history = history or ()
    label, message = COPY.get((state, reason)) or COPY.get((state, None)) or (state, None)
    if state == 'feed_delayed' and facts['lag_s'] is not None:
        behind = '내 중계' if facts.get('viewer_delay_s') else '실제 경기'  # a delayed viewer compares with their broadcast
        message = f"경기 자료가 {int(facts['lag_s'])}초 넘게 늦게 도착했어요. 추천이 {behind}보다 늦을 수 있어요."
    refusal = (view or {}).get('reason')
    if (state == 'unsupported' and reason == 'policy_refused' and isinstance(refusal, str) and len(refusal) <= 60
            and re.search('[가-힣]', refusal) and not re.search('[A-Z_]{4,}', refusal)):
        message = refusal  # the policy's own short Korean sentence; a bare status code is never shown to a fan
    fix = correction(list(history), summary(rec) if shown else None) if pitch_key else None
    return {'schema': SCHEMA, 'state': state, 'reason': reason, 'detail': detail, 'label': label, 'message': message,
            'pitch_key': pitch_key, 'recommendation_shown': shown, 'correction': fix,
            'feed': {k: facts[k] for k in ('ok', 'delayed', 'lag_s', 'lag_max_s', 'lag_measured', 'lag_threshold_s', 'thresholds_provisional',
                                           'last_ok_at', 'failing_since', 'failing_for_s', 'received_at', 'provider_order')},
            'ledger': {'status': 'off' if facts['store'] == 'off' else 'store_unavailable' if facts['store'] == 'failed' else
                       (ledger or {}).get('status') or 'not_applicable',
                       'outcome': (ledger or {}).get('outcome'), 'why': (ledger or {}).get('why'),
                       'store_ok': facts['store_ok']}}


def buffering(facts, wait_s):
    """The state of a viewer whose chosen delay has nothing to show yet (the server only just started to receive the
    game): nothing newer is ever shown instead. ``wait_s`` is how long until their moment is reached."""
    state = decide(gated=('waiting', 'buffering', None), view=None, facts=facts)
    seconds = max(1, int(-(-float(wait_s or 0)//1)))
    return state | {'message': f'내 중계 시점까지 기다리는 중이에요 (약 {seconds}초).'}
