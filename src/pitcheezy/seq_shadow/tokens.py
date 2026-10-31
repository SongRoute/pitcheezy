"""GUMBO ``feed/live`` body -> N1 context tokens: up to L earlier pitches of the SAME pitcher in the SAME game.

Mirror of the training path (``prepare_inputs.sequence_arrays`` + ``train_seq.build_context_index`` +
``train_seq.Data.tokens``): slot j (0-based) is the accepted pitch thrown (j+1) pitches earlier by the same pitcher
in the same game in (at_bat_number, pitch_number) order; fields of THAT earlier pitch:
  type     index of its pitch type in the 19 frozen TYPES (a code outside them -> 0 = UNKNOWN)
  out      its outcome10 label (0..9), ``outcome10`` below
  balls    its own pre-pitch balls (0..3)
  strikes  its own pre-pitch strikes (0..2)
  same     1 when its batter is the batter of the pitch being forecast
  pos      j+1 (1 = the most recent earlier pitch)
A pitch that was not a row of the accepted training corpus never appears as context and later pitches shift down:
no pitch type code, outcome10 = -1 (call code absent/unmapped, automatic ball/strike, in play with no or an unmapped
event such as field_error, fielders_choice, catcher_interf, triple_play), pre-pitch balls > 3 or strikes > 2.

What is read from the body (the ONE body the pre-pitch state came from; nothing later exists in it by construction):
``liveData.plays.allPlays[*]`` and ``currentPlay``: ``about.atBatIndex``, ``matchup.pitcher.id``,
``matchup.batter.id``, ``result.eventType`` (only for the last pitch of a play that is in play), and per play event
``isPitch``, ``pitchNumber``, ``details.type.code``, ``details.call.code``, ``details.isInPlay``, ``count.balls``,
``count.strikes``. Never read: pitchData, hitData, descriptions, times, linescore, anything of the pitch being
forecast. Only pitches with key (atBatIndex+1, pitchNumber) strictly before (state at_bat_number,
state last_pitch_number+1) are candidates.

Pre-pitch count of a pitch = the count after the previous event of its play that carries ``count`` (pitch or not:
automatic ball/strike, pickoff, step-off, action), starting 0-0.

Known limits (documented, not "fixed"):
  * GUMBO gives one matchup per play. After a pitcher change inside a plate appearance every pitch of that play is
    attributed to the play's final pitcher, and ``same`` uses the play's final batter. Step 5 linkage already
    unlinks those plate appearances from physical pitches.
  * ``currentPlay`` is used for its own atBatIndex in place of the ``allPlays`` entry with that index (the state was
    read from ``currentPlay``); in captured bodies the two are the same object.
  * An in-play pitch whose play carries no ``result.eventType`` yet in this body is skipped (training had the final
    Statcast ``events``); it is counted under ``skipped_unmapped_outcome``.
  * The accepted training corpus (abc_corpus.project, read as code only) differs from the rule above in two ways
    this builder does not reproduce: (a) a Statcast row with a null pitch_type was KEPT there as type UNKNOWN (0),
    while the brief of this job says to skip a pitch without a type code; the registered configuration value
    ``tokens.missing_type_policy`` chooses ('skip' = the brief, 'unknown_token' = what the corpus code did);
    (b) an in-play row without a registered bb_type was excluded there, while this builder must not read
    hitData.trajectory and keeps the pitch. Rows excluded there for outs/inning/score/handedness are not checked.
  * Statcast ``description`` is derived from the GUMBO call code with ``shadow_runner.DESCRIPTION``; parity of that
    table with the Statcast export is not verified on paired data.
"""
from __future__ import annotations

import math

import numpy as np

FIELDS = ('type', 'out', 'balls', 'strikes', 'same', 'pos')
COUNT_KEYS = ('n_context', 'eligible_earlier_pitches', 'skipped_unmapped_outcome', 'skipped_count_range',
              'skipped_no_type', 'truncated')

# ---------------------------------------------------------------------------------------------------------------
# VERBATIM COPY of ``outcome10`` and the constants it reads, from
#   /Users/song/Projects/pitcheezy-pm-jobs/20261010-ws-service-006/step5-2026-profile-source-author-001/projector.py
#   sha256 e7b37c9909cb5efd90be144912b20d84ea8b234fb223032fb32c9ae8ff3a7eeb (lines 22-27 constants, 33-34 _null,
#   37-55 outcome10); the sha256 is also in configs/seq_shadow.json (tokens.outcome10_source) and the test compares
#   the syntax tree of this copy with that file.
# ---------------------------------------------------------------------------------------------------------------
OUTCOMES = ('ball', 'strike', 'foul', 'out', 'single', 'double', 'triple', 'home_run', 'hbp', 'double_play')
_BALL = frozenset(('ball', 'blocked_ball', 'pitchout', 'intent_ball'))
_STRIKE = frozenset(('called_strike', 'swinging_strike', 'swinging_strike_blocked', 'missed_bunt', 'foul_tip',
                     'bunt_foul_tip'))
_OUT = frozenset(('field_out', 'force_out', 'fielders_choice_out', 'sac_fly', 'sac_bunt'))
_DP = frozenset(('double_play', 'grounded_into_double_play', 'sac_fly_double_play', 'sac_bunt_double_play'))


def _null(value):
    return value is None or (isinstance(value, float) and math.isnan(value))


def outcome10(description, events, strikes):
    """(label, held_reason). label -1 = no label; the row is HELD, never imputed. Does not read bb_type."""
    d, e = ('' if _null(v) else v for v in (description, events))
    if d in _BALL:
        return 0, None
    if d in _STRIKE or (d == 'foul_bunt' and strikes == 2):
        return 1, None
    if d in ('foul', 'foul_bunt'):
        return 2, None
    if d == 'hit_by_pitch':
        return 8, None
    if d == 'hit_into_play':
        if e in OUTCOMES[4:8]:
            return OUTCOMES.index(e), None
        if e in _OUT:
            return 3, None
        if e in _DP:
            return 9, None
        return -1, 'hit_into_play_events_null' if not e else 'hit_into_play_events_unmapped'
    return -1, 'description_null' if not d else 'description_unmapped'
# ---------------------------------------------------------------------------------------------------------------
# END OF VERBATIM COPY
# ---------------------------------------------------------------------------------------------------------------


def _get(value, *path):
    for key in path:
        value = value.get(key) if isinstance(value, dict) else None
    return value


def _plays(feed):
    """{atBatIndex: play}: ``allPlays`` in order (a repeated index: the later entry), then ``currentPlay`` for its
    own index."""
    plays = _get(feed, 'liveData', 'plays')
    listed = _get(plays, 'allPlays')
    found = {}
    for play in (listed if isinstance(listed, list) else [])+[_get(plays, 'currentPlay')]:
        index = _get(play, 'about', 'atBatIndex')
        if type(index) is int and index >= 0:
            found[index] = play
    return found


MISSING_TYPE_POLICIES = ('skip', 'unknown_token')


def earlier_pitches(feed, state, *, types, description, pitch_number_range, balls_vocab, strikes_vocab,
                    missing_type):
    """(accepted earlier pitches of ``state['pitcher']`` oldest first, skip counts). Each accepted pitch is
    (at_bat_number, pitch_number, type index, outcome10, pre-pitch balls, pre-pitch strikes, batter id).
    ``missing_type``: what an earlier pitch without a type code becomes, 'skip' (dropped, counted) or
    'unknown_token' (kept with type 0 = UNKNOWN); registered in the configuration."""
    if missing_type not in MISSING_TYPE_POLICIES:
        raise ValueError('unknown missing_type policy')
    pitcher, limit = state['pitcher'], (state['at_bat_number'], state['last_pitch_number']+1)
    lookup = {t: i for i, t in enumerate(types)}
    low, high = pitch_number_range
    kept, skipped = [], {'skipped_no_type': 0, 'skipped_unmapped_outcome': 0, 'skipped_count_range': 0}
    for index, play in sorted(_plays(feed).items()):
        events = _get(play, 'playEvents')
        if not isinstance(events, list) or _get(play, 'matchup', 'pitcher', 'id') != pitcher or type(pitcher) is not int:
            continue
        batter = _get(play, 'matchup', 'batter', 'id')
        pitch_events = [e for e in events if isinstance(e, dict) and e.get('isPitch')]
        last_pitch = pitch_events[-1] if pitch_events else None
        balls, strikes = 0, 0
        for event in events:
            if not isinstance(event, dict):
                continue
            number = event.get('pitchNumber')
            if event.get('isPitch') and type(number) is int and low <= number <= high and (index+1, number) < limit:
                code = _get(event, 'details', 'type', 'code')
                call = _get(event, 'details', 'call', 'code')
                final_in_play = event is last_pitch and _get(event, 'details', 'isInPlay') is True
                event_type = _get(play, 'result', 'eventType') if final_in_play else ''
                label, _ = outcome10(description.get(call) if isinstance(call, str) else None,
                                     event_type if isinstance(event_type, str) else '', strikes)
                if not (isinstance(code, str) and code) and missing_type == 'skip':
                    skipped['skipped_no_type'] += 1
                elif label < 0:
                    skipped['skipped_unmapped_outcome'] += 1
                elif type(balls) is not int or type(strikes) is not int or not (0 <= balls < balls_vocab) \
                        or not (0 <= strikes < strikes_vocab):
                    skipped['skipped_count_range'] += 1
                else:
                    kept.append((index+1, number, lookup.get(code, 0) if isinstance(code, str) else 0, label, balls, strikes,
                                 batter))
            if 'count' in event:  # every counted event moves the count, pitch or not
                balls, strikes = _get(event, 'count', 'balls'), _get(event, 'count', 'strikes')
    kept.sort(key=lambda p: (p[0], p[1]))
    return kept, skipped


def build(feed, state, *, length, types, description, pitch_number_range, balls_vocab, strikes_vocab, missing_type):
    """(tokens, counts). ``tokens``: int64 arrays of shape (length,) for ``FIELDS`` plus bool ``valid``; slot 0 is the
    most recent accepted earlier pitch; padding slots are zero with ``valid`` False and ``pos`` = slot+1 (as
    ``Data.tokens``: ``pos`` is arange(1, L+1) on every row; the network replaces padded slots by pad embeddings).
    """
    kept, skipped = earlier_pitches(feed, state, types=types, description=description,
                                    pitch_number_range=pitch_number_range, balls_vocab=balls_vocab,
                                    strikes_vocab=strikes_vocab, missing_type=missing_type)
    recent = kept[::-1][:length]
    tokens = {name: np.zeros(length, dtype=np.int64) for name in FIELDS}
    tokens['valid'] = np.zeros(length, dtype=bool)
    tokens['pos'][:] = np.arange(1, length+1)
    for slot, (_, _, kind, label, balls, strikes, batter) in enumerate(recent):
        tokens['valid'][slot] = True
        tokens['type'][slot], tokens['out'][slot] = kind, label
        tokens['balls'][slot], tokens['strikes'][slot] = balls, strikes
        tokens['same'][slot] = int(batter is not None and batter == state['batter'])
    counts = {'n_context': len(recent), 'eligible_earlier_pitches': len(kept), **skipped,
              'truncated': max(0, len(kept)-length)}
    return tokens, counts


def to_json(tokens, counts):
    """Plain lists for the ledger (stable key order is the caller's canonical JSON)."""
    return {**{name: tokens[name].tolist() for name in FIELDS}, 'valid': [bool(v) for v in tokens['valid']],
            'counts': {k: int(counts[k]) for k in COUNT_KEYS}}
