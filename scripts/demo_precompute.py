"""Watch-along demo precompute (DEMO-WS-2026): frozen <=2025 ARM-B policy on one completed game.

Not an evaluation. One completed (postseason) game is read from the public MLB GUMBO final feed,
mapped to the Statcast-like columns of the processed frame, and every pitch decision is sent through
the SAME request/runtime path as the <=2025 validation runner (``policy_requests.pa_requests`` ->
``policy_runtime.build_runtime`` over ``policy_identity.bind_components`` with the registered pins
of ML-POLICY-VAL-v1). The complete candidate identity must equal the tau-freeze identity
(a6dffaea...) or the run refuses. Batter style priors are the pinned 2026 snapshot rule
(``style_snapshot_2026``: rows dated <= 2025-12-31 only). Nothing is fitted; no 2026 row enters a
feature, pin or model. Outputs go only to a NEW directory under DEMO-WS-2026.

    .venv/bin/python scripts/demo_precompute.py check-mapping --games 12
    .venv/bin/python scripts/demo_precompute.py precompute --game-pk 849843
    .venv/bin/python scripts/demo_precompute.py sync --since 2026-09-29   # every newly completed postseason game

``LivePolicy`` is the delayed-live service view the observer calls for an in-progress game.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import re
import json
from pathlib import Path
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO / 'experiments/pitchmdp'
OBSERVER = REPO / 'apps/observer'
DEMO_ROOT = Path('/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/DEMO-WS-2026')
FEED_URL = 'https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live'
EXPECTED_IDENTITY = 'a6dffaea70976368a9cdc9f1a8bff4267b1c138162ee48aafda00e81e989109c'
REGISTRATION = REPO / 'configs/ML-POLICY-MATERIALIZATION-v1.json'
ADDENDA = tuple(REPO / f'configs/ML-POLICY-MATERIALIZATION-v1.addendum-{i}-d107.json' for i in range(1, 7))
BADGE = '검증 전 실험 버전 · 위치는 실제 투구 분포 근사'
INPUT_LABEL = 'postseason demo input (GUMBO-mapped; display only; not a Statcast holdout or evaluation row)'

# ---------------------------------------------------------------- GUMBO -> Statcast-like rows
# GUMBO call codes (statsapi /api/v1/pitchCodes) -> Statcast ``description``.
DESCRIPTION = {
    'B': 'ball', '*B': 'blocked_ball', 'C': 'called_strike', 'S': 'swinging_strike', 'W': 'swinging_strike_blocked',
    'T': 'foul_tip', 'F': 'foul', 'L': 'foul_bunt', 'M': 'missed_bunt', 'O': 'bunt_foul_tip',
    'X': 'hit_into_play', 'D': 'hit_into_play', 'E': 'hit_into_play', 'Y': 'hit_into_play', 'J': 'hit_into_play',
    'Z': 'hit_into_play', 'H': 'hit_by_pitch', 'P': 'pitchout', 'I': 'intent_ball', 'R': 'foul_pitchout',
    'Q': 'swinging_pitchout', 'V': 'automatic_ball', 'VC': 'automatic_ball', 'VP': 'automatic_ball',
    'VS': 'automatic_ball', 'VB': 'automatic_ball', 'A': 'automatic_strike', 'AC': 'automatic_strike', 'AB': 'automatic_strike'}
BASES = {'1B': 0, '2B': 1, '3B': 2}
# Column provenance (documented in the output manifest). 'faithful' = same quantity, unit-converted;
# 'proxy' = a different quantity standing in; 'missing' = not in the feed.
COLUMN_MAP = {
    'game_pk': 'gameData.game.pk', 'game_date': 'gameData.datetime.officialDate',
    'at_bat_number': 'allPlays[].atBatIndex + 1', 'pitch_number': 'running number of pitch + no_pitch (automatic call) events in the PA',
    'pitcher': 'matchup.pitcher.id', 'batter': 'matchup.batter.id', 'p_throws': 'matchup.pitchHand.code',
    'stand': 'matchup.batSide.code', 'inning': 'about.inning', 'inning_topbot': 'about.isTopInning -> Top/Bot',
    'balls/strikes': 'count of the playEvent before the pitch (pre-pitch)',
    'outs_when_up': 'previous play count.outs in the half + runner outs with playIndex < pitch index',
    'on_1b/on_2b/on_3b, bases': 'runner movements replayed in playIndex order before the pitch',
    'home_score/away_score': 'previous play result scores + runs scored (movement end=score) before the pitch',
    'post_home_score/post_away_score': 'scores after runner movements with playIndex <= pitch index',
    'pitch_type': 'details.type.code (same Statcast code list)',
    'description': 'details.call.code via DESCRIPTION (statsapi pitchCodes)',
    'events': 'result.eventType on the last pitch row of a completed PA',
    'release_speed': 'pitchData.startSpeed (mph)', 'release_extension': 'pitchData.extension (ft)',
    'release_spin_rate': 'pitchData.breaks.spinRate (rpm)', 'spin_axis': 'pitchData.breaks.spinDirection (deg)',
    # coordinates.pfxX/pfxZ are the PITCHf/x 40-ft definition (x1.65 off Statcast); the breaks match to 0.1 in.
    'pfx_x': '-pitchData.breaks.breakHorizontal / 12 (in -> ft, catcher view)',
    'pfx_z': 'pitchData.breaks.breakVerticalInduced / 12 (in -> ft)',
    'plate_x': 'pitchData.coordinates.pX (ft)', 'plate_z': 'pitchData.coordinates.pZ (ft)',
    'effective_speed': 'NOT in GUMBO: proxy = release_speed (startSpeed); see mapping_check.json'}


def fetch_feed(game_pk, timeout=30):
    request = urllib.request.Request(FEED_URL.format(game_pk=int(game_pk)), headers={'User-Agent': 'pitcheezy-demo'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def read_feed(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as handle:
        return json.load(handle)


def _number(value):
    return np.nan if value is None else float(value)


# Statcast batter-terminal event codes (policy_requests.BATTER_EVENTS plus codes Statcast also uses).
BATTER_EVENTS = frozenset({
    'field_out', 'strikeout', 'force_out', 'fielders_choice_out', 'fielders_choice', 'sac_fly', 'sac_bunt',
    'grounded_into_double_play', 'double_play', 'sac_fly_double_play', 'sac_bunt_double_play',
    'strikeout_double_play', 'triple_play', 'single', 'double', 'triple', 'home_run', 'walk', 'intent_walk',
    'hit_by_pitch', 'field_error', 'catcher_interf'})


def _statcast_event(event_type):
    """Statcast writes 'truncated_pa' when a runner event (e.g. caught stealing) ends the PA."""
    return event_type if event_type in BATTER_EVENTS else 'truncated_pa'


def _hand(data, player_id, key):
    return ((data.get('players') or {}).get(f'ID{player_id}') or {}).get(key, {}).get('code')


def statcast_rows(feed, *, effective_speed='release_speed'):
    """Pitch rows of a completed game in Statcast column names (see ``COLUMN_MAP``).

    ``effective_speed``: 'release_speed' (proxy, default) or 'missing' (NaN: the frozen normalizer
    imputes its TRAIN median). Non-pitch events only move runners/outs/scores between rows.
    """
    data = feed['gameData']
    game_pk, date = int(data['game']['pk']), data['datetime']['officialDate']
    rows, half, bases, outs, score = [], None, [None] * 3, 0, {'home': 0, 'away': 0}
    defense_pitcher = {}
    for play in feed['liveData']['plays']['allPlays']:
        about, matchup = play['about'], play['matchup']
        top = bool(about['isTopInning'])
        if (about['inning'], top) != half:
            half, bases, outs = (about['inning'], top), [None] * 3, 0
            if data['game']['type'] == 'R' and about['inning'] >= 10:  # regular-season automatic runner (not in the feed)
                bases[1] = 'automatic_runner'
        batting, fielding = ('away', 'home') if top else ('home', 'away')
        groups = {}
        for runner in play.get('runners', []):
            groups.setdefault(runner['details'].get('playIndex', 10 ** 6), []).append(runner)
        pending = sorted(groups)

        def apply_until(index):
            """Replay runner movements with playIndex < index, one playIndex at a time. The feed lists
            segments by runner, not chronologically, and a pinch runner replaces a runner without a
            movement, so bases are tracked by base: clear every runner's FIRST segment start, then
            place every runner at the end of its LAST segment."""
            nonlocal outs
            while pending and pending[0] < index:
                segments = {}
                for runner in groups[pending.pop(0)]:
                    segments.setdefault(runner['details'].get('runner', {}).get('id'), []).append(runner['movement'])
                    outs += bool(runner['movement'].get('isOut'))
                for chain in segments.values():
                    if chain[0].get('start') in BASES:
                        bases[BASES[chain[0]['start']]] = None
                for runner_id, chain in segments.items():
                    if any(m.get('isOut') for m in chain):
                        continue
                    if chain[-1].get('end') == 'score':
                        score[batting] += 1
                    elif chain[-1].get('end') in BASES:
                        bases[BASES[chain[-1]['end']]] = runner_id
        count = (0, 0)
        # Statcast numbers pitches AND automatic ball/strike calls (GUMBO type 'no_pitch' with a call code).
        pitches = [e for e in play['playEvents']
                   if e.get('isPitch') or (e.get('type') == 'no_pitch' and (e['details'].get('call') or {}).get('code'))]
        numbers = {id(e): k for k, e in enumerate(pitches, 1)}
        subs = [e for e in play['playEvents'] if e['details'].get('eventType') == 'pitching_substitution']
        for event in play['playEvents']:
            if event['details'].get('eventType') == 'pitching_substitution' and event.get('player'):
                defense_pitcher[fielding] = int(event['player']['id'])
            if id(event) not in numbers:
                count = (event.get('count', {}).get('balls', count[0]), event.get('count', {}).get('strikes', count[1]))
                continue
            index = event.get('index', 0)
            apply_until(index)
            if rows and rows[-1].get('_open'):  # Statcast post_* scores: everything up to the next row
                rows[-1].update(post_home_score=score['home'], post_away_score=score['away'], _open=False)
            pre = {'home': score['home'], 'away': score['away']}
            pre_outs, pre_bases = outs, list(bases)
            details, pitch = event['details'], event.get('pitchData') or {}
            coordinates, breaks = pitch.get('coordinates') or {}, pitch.get('breaks') or {}
            code = (details.get('call') or {}).get('code')
            last = event is pitches[-1] and about.get('isComplete', True)
            speed = _number(pitch.get('startSpeed'))
            # A mid-PA pitching change: earlier pitches belong to the previous pitcher of this defense.
            later_sub = any(s.get('index', 0) > index for s in subs)
            pitcher = defense_pitcher.get(fielding) if later_sub else int(matchup['pitcher']['id'])
            pitcher = int(matchup['pitcher']['id']) if pitcher is None else pitcher
            defense_pitcher[fielding] = pitcher
            hand = (matchup['pitchHand']['code'] if pitcher == int(matchup['pitcher']['id'])
                    else _hand(data, pitcher, 'pitchHand'))
            rows.append({
                'game_pk': game_pk, 'game_date': date, 'game_type': data['game']['type'],
                'at_bat_number': int(about['atBatIndex']) + 1, 'pitch_number': numbers[id(event)],
                'pitcher': pitcher, 'batter': int(matchup['batter']['id']),
                'p_throws': hand, 'stand': matchup['batSide']['code'],
                'inning': int(about['inning']), 'inning_topbot': 'Top' if top else 'Bot',
                'balls': int(count[0]), 'strikes': int(count[1]), 'outs_when_up': int(pre_outs),
                'on_1b': pre_bases[0], 'on_2b': pre_bases[1], 'on_3b': pre_bases[2],
                'bases': sum(1 << i for i, runner in enumerate(pre_bases) if runner is not None),
                'home_score': pre['home'], 'away_score': pre['away'],
                'post_home_score': None, 'post_away_score': None, '_open': True,
                'pitch_type': (details.get('type') or {}).get('code'), 'pitch_name': (details.get('type') or {}).get('description'),
                'description': DESCRIPTION.get(code, f'unmapped:{code}'), 'gumbo_call': code,
                'events': _statcast_event(play['result'].get('eventType')) if last else None,
                'gumbo_event': play['result'].get('eventType') if last else None,
                'des': play['result'].get('description') if last else None,
                'release_speed': speed, 'release_extension': _number(pitch.get('extension')),
                'effective_speed': speed if effective_speed == 'release_speed' else np.nan,
                'release_spin_rate': _number(breaks.get('spinRate')), 'spin_axis': _number(breaks.get('spinDirection')),
                'pfx_x': -_number(breaks.get('breakHorizontal')) / 12,
                'pfx_z': _number(breaks.get('breakVerticalInduced')) / 12,
                'plate_x': _number(coordinates.get('pX')), 'plate_z': _number(coordinates.get('pZ')),
                'sz_top': _number(pitch.get('strikeZoneTop')), 'sz_bot': _number(pitch.get('strikeZoneBottom'))})
            count = (event['count']['balls'], event['count']['strikes'])
        apply_until(10 ** 7)
        if rows and rows[-1].get('_open'):
            rows[-1].update(post_home_score=score['home'], post_away_score=score['away'], _open=False)
        defense_pitcher[fielding] = int(matchup['pitcher']['id'])
        outs = min(outs, 3)
    frame = pd.DataFrame(rows).drop(columns='_open', errors='ignore')
    if len(frame):
        frame['game_date'] = pd.to_datetime(frame.game_date)
    return frame


def _plain(value):
    """JSON-safe copy: NaN/inf -> None, numpy scalars -> Python."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path, value):
    path, value = Path(path), _plain(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=1, ensure_ascii=False, allow_nan=False, default=str) + '\n')
    temporary.replace(path)


def _sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _normalizer_report():
    """The frozen G0 physical normalizer report (TRAIN median fill, mean, scale), read from the pinned
    preparation JSON; no pickle is opened for the mapping check."""
    bundle = json.loads((REPO / 'configs/G0-RESEARCH-FROZEN-v1.json').read_text())
    entry = bundle['files']['p4_preparation']
    raw = Path(entry['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != entry['sha256']:
        raise RuntimeError('pinned G0 preparation changed')
    return json.loads(raw)['features']['auxiliary_scope']['normalizer']


def _as_text(series):
    text = lambda v: '<NA>' if pd.isna(v) else str(int(v)) if isinstance(v, float) and float(v).is_integer() else str(v)
    return pd.Series([text(v) for v in series], index=series.index)


def check_mapping(games, output, raw_path):
    """Agreement of ``statcast_rows`` with the raw <=2025 Statcast rows of the same games (regular
    season 2025, already exposed TRAIN/DEV data; read-only, nothing fitted). Games: every k-th
    game_pk of September 2025 in date order."""
    raw = pd.read_parquet(raw_path)
    raw = raw.loc[pd.to_datetime(raw.game_date).dt.strftime('%Y-%m').eq('2025-09')]
    order = raw.drop_duplicates('game_pk').sort_values(['game_date', 'game_pk']).game_pk.tolist()
    chosen = order[::max(1, len(order) // games)][:games]
    feeds = Path(output) / 'feeds'
    feeds.mkdir(parents=True, exist_ok=True)
    mapped = []
    for game in chosen:
        path = feeds / f'{game}.json.gz'
        if not path.exists():
            with gzip.open(path, 'wt') as handle:
                json.dump(fetch_feed(game), handle)
        mapped.append(statcast_rows(read_feed(path)))
    ours = pd.concat(mapped, ignore_index=True)
    theirs = raw.loc[raw.game_pk.isin(chosen)].copy()
    theirs['bases'] = (theirs.on_1b.notna().astype(int) + 2 * theirs.on_2b.notna().astype(int)
                       + 4 * theirs.on_3b.notna().astype(int))
    key = ['game_pk', 'at_bat_number', 'pitch_number']
    merged = ours.merge(theirs, on=key, how='outer', suffixes=('', '_sc'), indicator=True)
    both = merged.loc[merged._merge.eq('both')]
    exact = {}
    for column in ('pitcher', 'batter', 'p_throws', 'stand', 'inning', 'inning_topbot', 'balls', 'strikes',
                   'outs_when_up', 'bases', 'home_score', 'away_score', 'post_home_score', 'post_away_score',
                   'pitch_type', 'description', 'events'):
        a, b = _as_text(both[column]), _as_text(both[f'{column}_sc'])
        same = a.to_numpy() == b.to_numpy()
        exact[column] = {'agree': float(same.mean()), 'disagree_rows': int((~same).sum()),
                         'examples': [{'key': [int(v) for v in k], 'ours': x, 'statcast': y} for k, x, y in
                                      list(zip(both.loc[~same, key].to_numpy(), a[~same], b[~same]))[:8]]}
    report = _normalizer_report()
    scales = dict(zip(report['channels'], report['scale']))
    numeric = {}
    for column in ('release_speed', 'release_spin_rate', 'spin_axis', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z',
                   'release_extension'):
        diff = (both[column] - both[f'{column}_sc']).to_numpy(float)
        if column == 'spin_axis':
            diff = (diff + 180) % 360 - 180
        finite = np.abs(diff[np.isfinite(diff)])
        numeric[column] = {'rows_compared': int(len(finite)), 'missing_ours': int(both[column].isna().sum()),
                           'missing_statcast': int(both[f'{column}_sc'].isna().sum()),
                           'abs_diff_quantiles': ({str(q): float(np.quantile(finite, q)) for q in (.5, .9, .99, 1.)}
                                                  if len(finite) else None),
                           'normalizer_scale': scales.get(column)}
    proxy = (both.release_speed_sc - both.effective_speed_sc).to_numpy(float)
    proxy = proxy[np.isfinite(proxy)]
    fill_gap = (report['fill'][0] - both.effective_speed_sc).to_numpy(float)
    fill_gap = fill_gap[np.isfinite(fill_gap)]
    scale = report['scale'][0]
    quantiles = lambda v, qs=(.5, .9, .99): {str(q): float(np.quantile(v, q)) for q in qs}
    counts = lambda frame, column: {str(k): int(v) for k, v in frame[column].fillna('<NA>').value_counts().items()}
    result = {'games': [int(g) for g in chosen], 'source': str(raw_path),
              'rows': {'ours': int(len(ours)), 'statcast': int(len(theirs)), 'matched': int(len(both)),
                       'only_ours': int(merged._merge.eq('left_only').sum()),
                       'only_statcast': int(merged._merge.eq('right_only').sum()),
                       'only_statcast_descriptions': counts(merged.loc[merged._merge.eq('right_only')], 'description_sc'),
                       'only_ours_descriptions': counts(merged.loc[merged._merge.eq('left_only')], 'description')},
              'exact': exact, 'numeric': numeric,
              'effective_speed': {
                  'release_speed_minus_effective_speed_mph': quantiles(proxy, (.01, .1, .5, .9, .99)),
                  'abs_error_normalized_units_release_speed_proxy': quantiles(np.abs(proxy) / scale),
                  'abs_error_normalized_units_train_median_fill': quantiles(np.abs(fill_gap) / scale),
                  'normalizer': {'fill_mph': report['fill'][0], 'mean': report['mean'][0], 'scale': scale}},
              'column_map': COLUMN_MAP, 'label': 'mapping check on 2025 regular-season games; read-only; nothing fitted'}
    _write_json(Path(output) / 'mapping_check.json', result)
    return result


# ---------------------------------------------------------------- display pieces (pure)

STATUS_TEXT = {  # policy_runtime statuses -> viewer text (Korean)
    'SUPPORTED': None, 'OUTSIDE_POLICY_SUPPORT': None,
    'UNSUPPORTED_UNKNOWN_PITCHER': '2025년까지 학습 기록이 없는 투수라 추천하지 않습니다.',
    'UNSUPPORTED_PITCHER_HAND': '투수의 던지는 손이 학습 기록과 맞지 않아 추천하지 않습니다.',
    'UNSUPPORTED_EMPTY_SUPPORT': '이 투수·타석 방향에서 추천할 수 있는 학습 구종이 없습니다.',
    'UNSUPPORTED_LOGGING_POSITIVITY': '실제 던진 구종이 학습 기간 구종 목록에 없어 이 타석은 계산에서 빠졌습니다.',
    'UNSUPPORTED_MID_PA': '같은 타석의 앞선 공에서 추천이 멈춰 이 타석의 나머지 공은 추천하지 않습니다.',
    'UNSUPPORTED_NO_LOGGED_ACTION': '자동 볼·스트라이크(투구 없음)라 추천 대상이 아닙니다.',
    'UNSUPPORTED_MISSING_ACTION_LABEL': '구종 정보가 없는 투구라 이 타석은 계산에서 빠졌습니다.',
    'UNSUPPORTED_INCOMPLETE_START': '타석 첫 기록이 0-0이 아니라 추천하지 않습니다.',
    'UNSUPPORTED_INCONSISTENT_HISTORY': '앞선 공의 결과로 카운트를 확인할 수 없어 추천하지 않습니다.',
    'NOT_SUBMITTED': '기록 구조 문제(누락·순서)로 추천 요청을 만들지 않았습니다.',
    'NO_DECISION': '투구 없이 타석이 끝난 기록이라 추천 대상이 아닙니다.'}
# The only refusal that reads the current (actual) pitch; everything else is pre-decision.
ACTUAL_DEPENDENT = frozenset({'UNSUPPORTED_LOGGING_POSITIVITY'})


def location_proxy(xz_by_count, balls, strikes, bounds, *, sigma, minimum_draws, minimum_mass):
    """Observer realized-delivery kernel on one type's frozen TRAIN pool draws.

    ``xz_by_count``: [4, 3, draws, 2] plate (x, z) in feet at every pre-pitch count. A zone is
    supported when its Gaussian-kernel ESS >= ``minimum_draws`` and local mass >= ``minimum_mass`` at
    every count (the observer's ``supported_actions`` rule); the proxy is the supported zone with the
    most mass at the current count. Returns None when no zone is supported.
    """
    from observer_app.domain import ZONES, target_point
    from observer_app.recommendation_adapter import ObservedDeliveryKernel
    targets = np.array([[target_point(z['id'], bounds)['x'], target_point(z['id'], bounds)['z']] for z in ZONES])
    xz = np.asarray(xz_by_count, dtype=np.float64).reshape(12, -1, 2)
    _, ess, mass = ObservedDeliveryKernel().weights(xz, targets, sigma)
    ess, mass = ess.reshape(4, 3, -1), mass.reshape(4, 3, -1)
    supported = (ess.min(axis=(0, 1)) >= minimum_draws) & (mass.min(axis=(0, 1)) >= minimum_mass)
    if not supported.any():
        return None
    best = max(np.flatnonzero(supported), key=lambda z: (mass[balls, strikes, z], ZONES[z]['id']))
    zone = ZONES[best]
    return {'zone_id': zone['id'], 'zone_label': zone['label'], 'target': target_point(zone['id'], bounds),
            'kernel_mass': float(mass[balls, strikes, best]), 'kernel_ess': round(float(ess[balls, strikes, best]), 1)}


def recommendation(result, vocabulary, locate, top_k=3):
    """ARM-B candidate law first (type, then approximate zone; D49), numbers under ``detail``."""
    from observer_app.domain import PITCH_LABELS
    candidate, reference = np.asarray(result['candidate'], float), np.asarray(result['reference'], float)
    order = sorted(np.flatnonzero(candidate > 0), key=lambda i: (-candidate[i], vocabulary[i]))[:top_k]
    out = []
    for rank, i in enumerate(order, 1):
        zone = locate(vocabulary[i])
        out.append({'rank': rank, 'pitch_type': vocabulary[i], 'pitch_label': PITCH_LABELS.get(vocabulary[i], vocabulary[i]),
                    'zone_id': zone['zone_id'] if zone else None,
                    'zone_label': zone['zone_label'] if zone else '위치 근사 불가(표본 부족)',
                    'target': zone['target'] if zone else None,
                    'detail': {'probability': float(candidate[i]), 'reference_probability': float(reference[i]),
                               'kernel_mass': zone['kernel_mass'] if zone else None,
                               'kernel_ess': zone['kernel_ess'] if zone else None}})
    law = lambda p: {vocabulary[i]: float(p[i]) for i in np.flatnonzero(p > 0)}
    return {'candidates': out, 'candidate_law': law(candidate), 'reference_law': law(reference)}


def actual_view(row, bounds):
    from observer_app.domain import PITCH_LABELS, RESULT_LABELS, actual_zone
    x, z = row['plate_x'], row['plate_z']
    finite = lambda v: v is not None and not pd.isna(v)
    event = row.get('events') if finite(row.get('events')) else None
    return {'pitch_type': row['pitch_type'] if finite(row['pitch_type']) else None,
            'pitch_label': (PITCH_LABELS.get(row['pitch_type'], row.get('pitch_name') or row['pitch_type'])
                            if finite(row['pitch_type']) else '투구 없음'),
            'description': row['description'], 'result_label': RESULT_LABELS.get(event or row['description'],
                                                                                 event or row['description']),
            'event': event, 'event_label': RESULT_LABELS.get(event, event) if event else None,
            'play_text': row.get('des') if finite(row.get('des')) else None,
            'speed_mph': float(row['release_speed']) if finite(row['release_speed']) else None,
            'x': float(x) if finite(x) else None, 'z': float(z) if finite(z) else None,
            'zone_label': actual_zone(float(x) if finite(x) else None, float(z) if finite(z) else None, bounds)}


def win_expectancy(frame, home_we):
    """Frozen C0 home-team WE (count-independent PA-start model) before each row and after it
    (= before the next row; the final result after the last row). Display only."""
    before = [home_we(row) for row in frame.to_dict('records')]
    last = frame.iloc[-1]
    final = 1.0 if last.post_home_score > last.post_away_score else 0.0 if last.post_home_score < last.post_away_score else None
    after = before[1:] + [final]
    return [(b, a) for b, a in zip(before, after)]


def coverage(decisions):
    statuses = pd.Series([d['status'] for d in decisions])
    real = [d for d in decisions if d['status'] not in ('UNSUPPORTED_NO_LOGGED_ACTION', 'NO_DECISION')]
    ready = [d for d in real if d['pre']['status'] == 'ready']
    by_pitcher = {}
    for d in real:
        entry = by_pitcher.setdefault(str(d['pitcher']['id']), {'name': d['pitcher']['name'], 'pitches': 0, 'ready': 0})
        entry['pitches'] += 1
        entry['ready'] += d['pre']['status'] == 'ready'
    return {'rows': len(decisions), 'pitch_decisions': len(real), 'ready': len(ready),
            'ready_service_fallback': sum(d.get('path') == 'service_fallback' for d in ready),
            'ready_share_of_pitches': round(len(ready) / len(real), 4) if real else None,
            'status_counts': {str(k): int(v) for k, v in statuses.value_counts().items()},
            'actual_dependent_refusals': int(sum(d['status'] in ACTUAL_DEPENDENT for d in decisions)),
            'root_status_counts': {str(k): int(v) for k, v in
                                   pd.Series([d.get('root_status', d['status']) for d in decisions]).value_counts().items()},
            'by_pitcher': by_pitcher}


# ---------------------------------------------------------------- frozen policy (heavy; T7 + main .venv)

def _paths():
    for path in (PROJECT, PROJECT / 'scripts', OBSERVER / 'backend'):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def registration(addenda=ADDENDA, expected=EXPECTED_IDENTITY):
    """Registered pins and frozen settings; refuses unless the sealed tau freeze names ``expected``."""
    _paths()
    import run_policy_validation as rpv
    reg = rpv.load_registration(REGISTRATION, addenda)
    config = reg['config']
    freeze = rpv.registered_json(reg, 'tau_freeze')
    profile = rpv.registered_json(reg, 'profile')
    if freeze.get('status') != 'SELECTED' or freeze.get('final_identity_sha256') != expected:
        raise RuntimeError(f'sealed tau freeze does not name the expected identity {expected[:12]}')
    setting = profile['selection']
    stages = config['le2025_validation_plan']['stages']
    return {'reg': reg, 'rpv': rpv, 'config': config, 'freeze': freeze,
            'tau': freeze['selected_tau'], 'samples': setting['samples'], 'pitch_cap': setting['pitch_cap'],
            'seed': config['seeds']['planning_main'],
            'evaluation_seed': rpv.role_seed(config, 'evaluation') if freeze['dr_q']['source'] == 'evaluation_seed' else None,
            'budget': stages['S6_V4']['row_budget'],
            'components_sha256': rpv.registered_json(reg, 'bind_identity')['sha256'],
            'no_pitch': frozenset(config['pa_time_rules']['R3_codes']['no_pitch_descriptions'])}


def style_snapshot(settings, inputs_loader, cache_dir):
    """The pinned 2026 style snapshot (``style_snapshot_2026``: rows <= 2025-12-31), cached by the
    processed-data sha256 in the demo directory (write-exclusive; reused only through its pin)."""
    from pitchmdp.policy_artifacts import load_style_snapshot, save_style_snapshot
    from pitchmdp import policy_requests as preq
    index_path = Path(cache_dir) / 'style_2026_index.json'
    if index_path.exists():
        index = json.loads(index_path.read_text())
        snapshot, as_of, content = load_style_snapshot(Path(cache_dir) / index['file'], index['file_sha256'])
        return snapshot, as_of, {**index, 'content_sha256': content, 'reused': True}
    frame, processed = inputs_loader()
    snapshot = preq.style_snapshot_2026(frame)
    source = {'processed_sha256': processed, 'rule': 'policy_requests.style_snapshot_2026',
              'rows_before_as_of': int((pd.to_datetime(frame.game_date) < pd.Timestamp(preq.PROFILE_AS_OF_2026)).sum())}
    name = f'style_{preq.PROFILE_AS_OF_2026}.json'
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    _, file_sha = save_style_snapshot(snapshot, preq.PROFILE_AS_OF_2026, source, Path(cache_dir) / name)
    index = {'file': name, 'file_sha256': file_sha, 'as_of_exclusive': preq.PROFILE_AS_OF_2026, 'source': source}
    _write_json(index_path, index)
    snapshot, as_of, content = load_style_snapshot(Path(cache_dir) / name, file_sha)
    return snapshot, as_of, {**index, 'content_sha256': content, 'reused': False}


def load_frame(settings, local):
    """<=2025 processed regular-season frame through the registered loader (only for the snapshot)."""
    inputs = settings['rpv'].load_inputs(settings['config'], local, store=False)
    return inputs['frame'], inputs['parent']['dataset_identity']['processed_sha256']


def demo_frame(rows, snapshot, as_of):
    from pitchmdp import policy_requests as preq
    frame = rows.sort_values(['game_date', 'game_pk', 'at_bat_number', 'pitch_number'], kind='stable').reset_index(drop=True)
    if pd.to_datetime(frame.game_date).dt.year.min() < 2026:
        raise ValueError('the demo input must be a 2026 game (<=2025 rows belong to the research frame)')
    frame, report = preq.apply_style_snapshot(frame, snapshot, as_of)
    return frame, report


def service_candidate(runtime, request):
    """Pre-pitch candidate law for a request whose ledger row was refused only because of the actual
    pitch (LOGGING_POSITIVITY, directly or as the sticky root of MID_PA). Repeats every pre-decision
    check of ``PolicyRuntime._evaluate`` that does not read the logged action; None if one fails."""
    from pitchmdp import policy_runtime as prt
    from pitchmdp.policy_artifacts import normalize_hand
    state = request.state
    runtime.components.check_context(state, request.pitcher_hand)
    if (request.decision_index == 0 and (state.balls, state.strikes) != (0, 0)) or runtime.bc.fallback(state):
        return None
    if state.history:
        past = state.history[-1]
        if prt.next_count(past.balls, past.strikes, past.outcome) != (state.balls, state.strikes):
            return None
    if runtime.hand_registry is not None and runtime.hand_registry[state.pitcher] != normalize_hand(request.pitcher_hand):
        return None
    mask = runtime.reference.support(state)
    runtime.support_check(state, mask)
    if not mask.any() or not runtime.bc.support(state).any():
        return None
    probabilities, _ = runtime.candidate(state)
    return {'candidate': list(map(float, probabilities)), 'reference': list(map(float, runtime.reference.probabilities(state)))}


def _git():
    import subprocess
    run = lambda *a: subprocess.run(['git', *a], cwd=REPO, capture_output=True, text=True).stdout.strip()
    return {'commit': run('rev-parse', 'HEAD'), 'dirty_paths': run('status', '--porcelain', '--', 'scripts', 'experiments',
                                                                     'src', 'apps/observer/backend').splitlines()}


def _check_output_root(output_root):
    """Only the demo directory on T7 (or a path off T7, e.g. a test tmp dir) may be written."""
    root = Path(output_root).resolve()
    t7 = Path('/Volumes/T7 Shield').resolve()
    if root.is_relative_to(t7) and not root.is_relative_to(DEMO_ROOT.resolve()):
        raise RuntimeError(f'refusing to write under T7 outside {DEMO_ROOT}')
    return root


def store_feed(game_pk, feed_path, feeds_dir):
    """The final feed as a pinned demo input (fetched once, then reused by sha256)."""
    feeds_dir.mkdir(parents=True, exist_ok=True)
    target = feeds_dir / f'{int(game_pk)}_final.json.gz'
    if not target.exists():
        feed = read_feed(feed_path) if feed_path else fetch_feed(game_pk)
        with gzip.open(target, 'wt') as handle:
            json.dump(feed, handle)
    feed = read_feed(target)
    if int(feed['gameData']['game']['pk']) != int(game_pk) or feed['gameData']['status'].get('abstractGameState') != 'Final':
        raise RuntimeError('the demo needs the FINAL feed of the requested gamePk')
    return feed, target, _sha_file(target)


def bind(settings, frame, ledger_path, provenance):
    """The frozen components bound over ``frame``'s context rows and the ARM-B runtime over them;
    refuses unless the components are the S2-certified ones and the runtime is ``EXPECTED_IDENTITY``."""
    rpv, reg, config = settings['rpv'], settings['reg'], settings['config']
    from pitchmdp import policy_identity as pid, policy_runtime as prt
    from pitchmdp.matrix_policy import safe_rows
    from pitchmdp.policy_artifacts import load_train_bc
    from pitchmdp.rollout_policy import RowBudget
    from run_ml_g0_whole import load_member
    ident = config['identity_registration']
    bundle_path, bundle_sha = REPO / ident['g0_bundle']['path'], ident['g0_bundle']['file_sha256']
    files = pid.pinned_json(bundle_path, bundle_sha)['files']
    paths = {role: Path(entry['path']) for role, entry in files.items()}
    paths.update({role: (REPO / value if not Path(value).is_absolute() else Path(value)) for role, value in ident['we_paths'].items()})
    aux = pid.pinned_pickle(paths['p4_auxiliary'], files['p4_auxiliary']['sha256'])
    bc_path, bc_sha = rpv.registered_path(reg, 'bc')
    support_path, support_sha = rpv.registered_path(reg, 'support')
    components = pid.bind_components(bundle_path, bundle_sha, paths, bc_artifact=load_train_bc(bc_path, bc_sha),
                                     context_rows=safe_rows(frame), member_loader=load_member, classes=ident['classes'],
                                     we_contract_sha256=ident['we_contract_sha256'])
    if components.sha256 != settings['components_sha256']:
        raise RuntimeError('bound components differ from the S2-certified identity; refusing to run')
    runtime = prt.build_runtime(bc_path, bc_sha, support_path, support_sha, ledger_path,
                                components=components, budget=RowBudget(int(settings['budget']), seed_count=5),
                                tau=settings['tau'], samples=settings['samples'], pitch_cap=settings['pitch_cap'],
                                seed=settings['seed'], evaluation_seed=settings['evaluation_seed'],
                                expected_identity_sha256=EXPECTED_IDENTITY,
                                hand_registry=rpv.registered_path(reg, 'hands'), provenance=provenance)
    return components, runtime, aux


def history_store(frame, aux, settings):
    """Strictly-previous pitch history of ``frame`` with the frozen G0 normalizer and type vocabulary."""
    from pitchmdp import policy_identity as pid
    from pitchmdp.matrix_features import MatrixHistoryStore
    ident = settings['config']['identity_registration']
    files = pid.pinned_json(REPO / ident['g0_bundle']['path'], ident['g0_bundle']['file_sha256'])['files']
    prep = pid.pinned_json(Path(files['p4_preparation']['path']), files['p4_preparation']['sha256'])
    return MatrixHistoryStore.from_frame(frame, normalizer=aux['normalizer'], history_length=5,
                                         type_vocabulary=prep['features']['tokens']['type_vocabulary'])


def _location_settings():
    return (json.loads((OBSERVER / 'config.json').read_text()),
            json.loads((OBSERVER / 'live_config.json').read_text())['zone_bounds'])


def make_locator(components, aux, bounds, observer):
    """``locator(row)(pitch_type)``: the approximate zone from the pitcher's frozen TRAIN delivery pool."""
    from pitchmdp import policy_requests as preq
    from pitchmdp.rollout_policy import PAState
    mean, scale = np.asarray(aux['normalizer'].mean), np.asarray(aux['normalizer'].scale)
    pools = {}

    def locator(row):
        def locate(pitch_type):
            key = (int(row['pitcher']), row['stand'], row['p_throws'], pitch_type)
            if key not in pools:
                context = preq.context_key(row)
                draws = [[components.inputs.pool(PAState(b, s, str(int(row['pitcher'])), row['stand'], (), context),
                                                  pitch_type, count_access=False).values[:, 6:8] * scale[6:8] + mean[6:8]
                          for s in range(3)] for b in range(4)]
                pools[key] = draws
            return location_proxy(pools[key], int(row['balls']), int(row['strikes']), bounds,
                                  sigma=observer['target_sigma_ft'], minimum_draws=observer['minimum_effective_draws'],
                                  minimum_mass=observer['minimum_kernel_mass'])
        return locate
    return locator


def precompute(game_pk, feed_path, output_root, local, *, effective_speed='release_speed', tag='', addenda=ADDENDA):
    """Every row of one completed game through the frozen ARM-B runtime; returns the demo dataset."""
    timing, started = {}, time.perf_counter()
    mark = lambda name, since: timing.__setitem__(name, round(time.perf_counter() - since, 2))
    output_root = _check_output_root(output_root)
    settings = registration(addenda)
    from pitchmdp import policy_requests as preq, policy_runtime as prt
    from pitchmdp.game import GameState
    reg = settings['reg']

    feed, feed_file, feed_sha = store_feed(game_pk, feed_path, output_root / 'feeds')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run_dir = output_root / 'games' / str(int(game_pk)) / f'run-{stamp}{tag}'
    run_dir.mkdir(parents=True, exist_ok=False)
    rows = statcast_rows(feed, effective_speed=effective_speed)
    rows.to_parquet(run_dir / 'input_statcast_like.parquet', index=False)
    t = time.perf_counter()
    snapshot, as_of, style = style_snapshot(settings, lambda: load_frame(settings, local), output_root / 'style')
    mark('style_snapshot_seconds', t)
    frame, style_report = demo_frame(rows, snapshot, as_of)

    t = time.perf_counter()
    provenance = {'demo': 'DEMO-WS-2026 watch-along (display only; not an evaluation)', 'game_pk': int(game_pk),
                  'input_feed_sha256': feed_sha, 'style_snapshot_sha256': style['content_sha256'],
                  'as_of_exclusive': as_of, 'effective_speed_rule': effective_speed}
    components, runtime, aux = bind(settings, frame, run_dir / 'ledger.jsonl', provenance)
    store = history_store(frame, aux, settings)
    mark('bind_seconds', t)
    observer, bounds = _location_settings()
    locator = make_locator(components, aux, bounds, observer)

    home_we = lambda row: components.defense_we(GameState.from_row(row), True)
    we = win_expectancy(frame, home_we)
    players = feed['gameData'].get('players', {})
    name = lambda pid_: (players.get(f'ID{int(pid_)}') or {}).get('fullName')
    records = frame.to_dict('records')
    decisions, decision_seconds = [], []
    t = time.perf_counter()
    for pa_id, positions in preq.pa_blocks(frame):
        requests, problem, _ = preq.pa_requests(store, positions, runtime.sha256, settings['no_pitch'])
        for k, position in enumerate(positions):
            row = records[position]
            if k < len(requests):
                start = time.perf_counter()
                ledger = runtime.submit(requests[k])
                decision_seconds.append(time.perf_counter() - start)
                status, result = ledger['status'], ledger['result']
            else:
                status, result = ('NOT_SUBMITTED' if problem not in (None, 'no_decision') else 'NO_DECISION'), None
            root = status  # a sticky MID_PA refusal is shown with its first refusal in the PA
            if status == 'UNSUPPORTED_MID_PA':
                root = (re.findall(r'UNSUPPORTED_[A-Z_]+', ledger['detail']) or [status])[-1]
            ready, path = status in prt.EVALUATED, 'ledger'
            if not ready and root in ACTUAL_DEPENDENT and k < len(requests):
                # The ledger refusal read the ACTUAL pitch (OPE ratio rule). A pre-pitch viewer must not
                # learn that from a missing recommendation, so the same candidate law is shown.
                result = service_candidate(runtime, requests[k])
                ready, path = result is not None, 'service_fallback'
            before, after = we[position]
            delta = None if after is None or before is None else after - before
            decisions.append({
                'index': len(decisions), 'pa_id': pa_id, 'at_bat_number': int(row['at_bat_number']),
                'pitch_number': int(row['pitch_number']),
                'situation': {'inning': int(row['inning']), 'half': row['inning_topbot'], 'outs': int(row['outs_when_up']),
                              'balls': int(row['balls']), 'strikes': int(row['strikes']), 'bases': int(row['bases']),
                              'home_score': int(row['home_score']), 'away_score': int(row['away_score'])},
                'pitcher': {'id': int(row['pitcher']), 'name': name(row['pitcher']), 'hand': row['p_throws']},
                'batter': {'id': int(row['batter']), 'name': name(row['batter']), 'side': row['stand']},
                'status': status, 'root_status': root, 'path': path,
                'pre': {'status': 'ready' if ready else 'unsupported',
                        'reason': None if ready else STATUS_TEXT.get(root, root),
                        'recommendation': recommendation(result, runtime.vocabulary, locator(row)) if ready else None},
                'actual': actual_view(row, bounds),
                'we': {'home_before': before, 'home_after': after, 'home_delta': delta,
                       'batting_delta': None if delta is None else (-delta if row['inning_topbot'] == 'Top' else delta)}})
    mark('decisions_seconds', t)
    runtime.verify_components()
    summary = runtime.summary()
    timing.update(per_request_mean_seconds=round(float(np.mean(decision_seconds)), 4) if decision_seconds else None,
                  per_request_max_seconds=round(float(np.max(decision_seconds)), 4) if decision_seconds else None)
    mark('total_seconds', started)
    data = feed['gameData']
    last = frame.iloc[-1]
    dataset = {
        'schema': 'pitcheezy-watch-along-v1', 'badge': BADGE,
        'game': {'game_pk': int(game_pk), 'date': data['datetime']['officialDate'], 'game_type': data['game']['type'],
                 'away_team': data['teams']['away']['name'], 'home_team': data['teams']['home']['name'],
                 'final': {'away': int(last.post_away_score), 'home': int(last.post_home_score)}},
        'policy': {'identity_sha256': EXPECTED_IDENTITY, 'components_sha256': components.sha256,
                   'runtime_sha256': runtime.sha256, 'name': 'ARM-B P3 (pitch type only), frozen <=2025 (ML-POLICY-VAL-v1)',
                   'tau': settings['tau'], 'samples': settings['samples'], 'pitch_cap': settings['pitch_cap'],
                   'planning_seed': settings['seed'], 'evaluation_seed': settings['evaluation_seed'],
                   'reference': 'SupportedBC (TRAIN BC masked to the frozen support table)',
                   'style_as_of_exclusive': as_of, 'style_snapshot_sha256': style['content_sha256'],
                   'validated': False, 'note': '검증 전: 2026 성능 평가는 아직 없습니다. 표시용 실험 버전입니다.'},
        'location': {'basis': 'realized_delivery_proxy', 'evaluated': False,
                     'kernel': 'observed-train-delivery-gaussian-v1 on the frozen G0 TRAIN delivery pools',
                     'sigma_ft': observer['target_sigma_ft'], 'minimum_effective_draws': observer['minimum_effective_draws'],
                     'minimum_kernel_mass': observer['minimum_kernel_mass'], 'zone_bounds': bounds,
                     'note': '위치는 이 투수의 실제 투구 분포(2025년까지) 기반 근사이며 평가되지 않았습니다.'},
        'we': {'model': 'frozen C0 WinExpectancy (count-independent, PA-start states; <=2025-04-30)',
               'note': '승리확률 변화는 다음 투구 직전 상태 기준의 표시용 근사입니다.'},
        'coverage': coverage(decisions), 'timing': timing, 'ledger': summary,
        'input': {'label': INPUT_LABEL, 'feed_sha256': feed_sha, 'rows': int(len(frame)),
                  'effective_speed_rule': effective_speed, 'style_report': style_report},
        'generated_utc': stamp, 'git': _git(), 'decisions': decisions}
    dataset['paths'] = {'run_dir': str(run_dir), 'dataset': str(output_root / 'watch' / f'{int(game_pk)}.json')}
    _write_json(run_dir / 'watch.json', dataset)
    _write_json(run_dir / 'manifest.json', {
        'label': INPUT_LABEL, 'column_map': COLUMN_MAP, 'feed': {'file': str(feed_file), 'sha256': feed_sha},
        'input_parquet_sha256': _sha_file(run_dir / 'input_statcast_like.parquet'), 'style': style,
        'registration_chain': reg['chain'], 'policy': dataset['policy'], 'coverage': dataset['coverage'],
        'timing': timing, 'ledger_summary': summary, 'git': dataset['git'],
        'mapping_check': (str(output_root / 'mapping_check/mapping_check.json')
                          if (output_root / 'mapping_check/mapping_check.json').exists() else None)})
    if not tag:  # the observer serves the latest untagged run
        _write_json(output_root / 'watch' / f'{int(game_pk)}.json', dataset)
    return dataset


# ---------------------------------------------------------------- delayed-live service view

POSTSEASON = ('F', 'D', 'L', 'W')
SCHEDULE_URL = ('https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={since}&endDate={until}'
                '&gameType=' + ','.join(POSTSEASON))


def live_rows(feed):
    """Statcast-like rows of an in-progress game plus one placeholder row for the NEXT pitch, or None
    when no pitch is pending (not live, between innings).

    The placeholder is a pitch event appended after every event already in the feed (at the due-up
    batter's new PA when the current one is complete), so ``statcast_rows`` gives it exactly the
    pre-pitch state (count, outs, runners, score, pitcher) a completed game's row would have. Its type,
    physics and outcome are unknown and never read: callers evaluate it with no logged action.
    """
    data, live = feed['gameData'], feed['liveData']
    if data['status'].get('abstractGameState') != 'Live':
        return None
    plays, line = list(live['plays'].get('allPlays') or []), live.get('linescore') or {}
    if plays and not plays[-1]['about'].get('isComplete', True):
        play = plays.pop()
    else:
        if line.get('outs') == 3 or line.get('inningState') in ('Middle', 'End') or 'offense' not in line:
            return None
        batter, pitcher = line['offense']['batter']['id'], line['defense']['pitcher']['id']
        hand = _hand(data, pitcher, 'pitchHand')
        side = _hand(data, batter, 'batSide')
        side = {'L': 'R', 'R': 'L'}.get(hand) if side == 'S' else side  # a switch hitter bats opposite the pitcher
        play = {'about': {'atBatIndex': plays[-1]['about']['atBatIndex'] + 1 if plays else 0,
                          'inning': line['currentInning'], 'isTopInning': line['isTopInning'], 'isComplete': False},
                'matchup': {'pitcher': {'id': pitcher}, 'pitchHand': {'code': hand}, 'batter': {'id': batter},
                            'batSide': {'code': side}},
                'playEvents': [], 'runners': [], 'result': {}}
    events = play.get('playEvents') or []
    placeholder = {'index': max((e.get('index', 0) for e in events), default=-1) + 1, 'isPitch': True,
                   'type': 'pitch', 'details': {'call': {'code': None}, 'type': {'code': None}},
                   'count': {'balls': 0, 'strikes': 0}, 'pitchData': {}}
    plays.append({**play, 'about': {**play['about'], 'isComplete': False}, 'playEvents': [*events, placeholder]})
    rows = statcast_rows({**feed, 'liveData': {**live, 'plays': {**live['plays'], 'allPlays': plays}}})
    rows['placeholder'] = False
    rows.loc[rows.index[-1], 'placeholder'] = True
    rows.loc[rows.index[-1], 'description'] = None
    return rows


def pre_pitch(runtime, request):
    """The runtime's own pre-decision evaluation of one request with no logged action (the pre-pitch
    service call ``DecisionRequest`` documents): the same refusals and candidate law, nothing written to
    the ledger, no sticky refusal from a logged pitch. Returns (status, result or None)."""
    from dataclasses import replace
    from pitchmdp.policy_artifacts import Unsupported
    request = replace(request, logged_action=None)
    history = request.state.history
    pa = {'next': request.decision_index, 'last_logged': history[-1].action if history else None, 'refused': None}
    try:
        return runtime._evaluate(request, pa)
    except Unsupported as refusal:
        return refusal.status, None


def pre_view(status, result, vocabulary, locate):
    from pitchmdp import policy_runtime as prt
    ready = status in prt.EVALUATED and result is not None
    return {'status': 'ready' if ready else 'unsupported', 'reason': None if ready else STATUS_TEXT.get(status, status),
            'recommendation': recommendation(result, vocabulary, locate) if ready else None}


def _key(row):
    return f"{int(row['game_pk'])}:{int(row['at_bat_number'])}:{int(row['pitch_number'])}"


class LivePolicy:
    """Delayed-live service view (DEMO-WS-2026, display only; not an evaluation).

    Each call binds the frozen components over the in-progress game's rows (the same ``bind``, context
    keys and identity refusal as ``precompute``) and evaluates the current PA's pitches and the pending
    one with ``pre_pitch``. The ledger header goes to a temporary directory; nothing is submitted.
    """

    def __init__(self, output_root=DEMO_ROOT, local=None, addenda=ADDENDA):
        self.settings = registration(addenda)
        local = local or json.loads((PROJECT / 'configs/local.json').read_text())
        self.snapshot, self.as_of, self.style = style_snapshot(
            self.settings, lambda: load_frame(self.settings, local), _check_output_root(output_root) / 'style')
        self.observer, self.bounds = _location_settings()

    def __call__(self, feed):
        import tempfile
        from pitchmdp import policy_requests as preq
        from pitchmdp.game import GameState
        rows = live_rows(feed)
        if rows is None:
            return None
        frame, _ = demo_frame(rows, self.snapshot, self.as_of)
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix='pitcheezy-live-') as scratch:
            provenance = {'demo': 'DEMO-WS-2026 delayed-live service view (display only; nothing submitted)',
                          'game_pk': int(frame.game_pk.iloc[0]), 'style_snapshot_sha256': self.style['content_sha256'],
                          'as_of_exclusive': self.as_of}
            components, runtime, aux = bind(self.settings, frame, Path(scratch) / 'ledger.jsonl', provenance)
            bound = time.perf_counter() - started
            store = history_store(frame, aux, self.settings)
            locator = make_locator(components, aux, self.bounds, self.observer)
            positions = list(preq.pa_blocks(frame))[-1][1]
            requests, problem, _ = preq.pa_requests(store, positions, runtime.sha256, self.settings['no_pitch'])
            records = frame.to_dict('records')
            pitches = []
            for k, position in enumerate(positions):
                row = records[position]
                status, result = pre_pitch(runtime, requests[k]) if k < len(requests) else (
                    'NOT_SUBMITTED' if problem not in (None, 'no_decision') else 'NO_DECISION', None)
                pitches.append({'key': _key(row), 'pitch_number': int(row['pitch_number']), 'status': status,
                                'pre': pre_view(status, result, runtime.vocabulary, locator(row)),
                                'actual': None if row['placeholder'] else actual_view(row, self.bounds)})
            home_we = [components.defense_we(GameState.from_row(r), True) for r in records[-2:]]
        previous = records[-2] if len(records) > 1 else None
        return {'key': pitches[-1]['key'], 'pitch': pitches[-1], 'pa_pitches': pitches[:-1],
                'previous': None if previous is None else {
                    'key': _key(previous), 'actual': actual_view(previous, self.bounds),
                    'inning': int(previous['inning']), 'half': previous['inning_topbot'],
                    'home_we_before': home_we[0]},
                'home_we_now': home_we[-1], 'policy_identity': EXPECTED_IDENTITY,
                'seconds': {'bind': round(bound, 2), 'total': round(time.perf_counter() - started, 2)}}


def completed_postseason_games(since, until, fetch=None):
    """gamePks of FINAL postseason games (F/D/L/W) between two dates (inclusive), in schedule order."""
    fetch = fetch or (lambda url: json.load(urllib.request.urlopen(
        urllib.request.Request(url, headers={'User-Agent': 'pitcheezy-demo'}), timeout=30)))
    schedule = fetch(SCHEDULE_URL.format(since=since, until=until))
    return [int(g['gamePk']) for day in schedule.get('dates', []) for g in day['games']
            if g['gameType'] in POSTSEASON and g['status'].get('abstractGameState') == 'Final'
            and g['status'].get('detailedState') in ('Final', 'Game Over', 'Completed Early')]


def sync(since, until, output_root, local, *, fetch=None, run=None):
    """Precompute every completed postseason game that has no watch dataset yet. One game's failure is
    reported and does not stop the others; a second concurrent sync exits at once (file lock)."""
    import fcntl
    root = _check_output_root(output_root)
    root.mkdir(parents=True, exist_ok=True)
    run = run or (lambda pk: precompute(pk, None, root, local))
    with open(root / '.sync.lock', 'w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {'skipped': 'another sync is running'}
        report = {'since': since, 'until': until, 'done': [], 'present': [], 'failed': {}}
        for game_pk in completed_postseason_games(since, until, fetch):
            if (root / 'watch' / f'{game_pk}.json').exists():
                report['present'].append(game_pk)
                continue
            try:
                dataset = run(game_pk)
                report['done'].append({'game_pk': game_pk, 'ready_share': dataset['coverage']['ready_share_of_pitches']})
            except Exception as error:  # noqa: BLE001 - one game's failure must not stop the others
                report['failed'][str(game_pk)] = f'{type(error).__name__}: {error}'
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    commands = parser.add_subparsers(dest='command', required=True)
    local = json.loads((PROJECT / 'configs/local.json').read_text())
    check = commands.add_parser('check-mapping')
    check.add_argument('--games', type=int, default=12)
    check.add_argument('--raw', type=Path, default=Path(local['raw_root']) / 'statcast_2025.parquet')
    check.add_argument('--output', type=Path, default=DEMO_ROOT / 'mapping_check')
    run = commands.add_parser('precompute')
    run.add_argument('--game-pk', type=int, required=True)
    run.add_argument('--feed', type=Path, help='recorded final GUMBO feed (.json/.json.gz); fetched when omitted')
    run.add_argument('--effective-speed', choices=('release_speed', 'missing'), default='release_speed')
    run.add_argument('--output-root', type=Path, default=DEMO_ROOT)
    run.add_argument('--tag', default='')
    synced = commands.add_parser('sync', help='precompute every newly completed postseason game')
    synced.add_argument('--since', default='2026-09-29')
    synced.add_argument('--until', default=datetime.now(timezone.utc).strftime('%Y-%m-%d'))
    synced.add_argument('--output-root', type=Path, default=DEMO_ROOT)
    args = parser.parse_args(argv)
    if args.command == 'check-mapping':
        result = check_mapping(args.games, args.output, args.raw)
        print(json.dumps({k: result[k] for k in ('rows', 'effective_speed')}, indent=1))
        print(json.dumps({k: (v['agree'], v['examples'][:3]) for k, v in result['exact'].items()}, indent=0))
        print(json.dumps({k: v['abs_diff_quantiles'] for k, v in result['numeric'].items()}, indent=0))
    elif args.command == 'sync':
        report = sync(args.since, args.until, args.output_root, local)
        print(json.dumps(report, indent=1))
        sys.exit(1 if report.get('failed') else 0)
    else:
        dataset = precompute(args.game_pk, args.feed, args.output_root, local, effective_speed=args.effective_speed,
                             tag=args.tag)
        print(json.dumps({'dataset': dataset['paths']['dataset'], 'coverage': {
            k: v for k, v in dataset['coverage'].items() if k != 'by_pitcher'}, 'timing': dataset['timing']}, indent=1))


if __name__ == '__main__':
    main()
