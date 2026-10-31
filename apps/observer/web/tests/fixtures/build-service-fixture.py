"""Build tests/fixtures/service-849843.json in the CONTRACT rev1 shape (pitcheezy-service-game-v2) from the first 70
pitches of the real 849843 watch dataset (GET /api/watch/849843 + /reveal/{i} on preview 8771, 2026-10-05 KST).
UI test fixture only; not research data. PA result_text is written by hand from each play_text below (what the backend
contract promises: fielder/direction only when the source names it); one PA keeps result_text null to test the fallback.
  python3 tests/fixtures/build-service-fixture.py /tmp/pz-fe-001
"""
import json, sys
from pathlib import Path

src = Path(sys.argv[1])
timeline = json.loads((src/'timeline.json').read_text())
reveals = [json.loads(l) for l in (src/'reveals.jsonl').read_text().splitlines() if l.strip()]
N = len(reveals)
RESULT_TEXT = {  # hand-written from each play_text (printed by this script); 849843:12 deliberately null → UI falls back to event_label
    '849843:1': '파울팁 삼진', '849843:2': '헛스윙 삼진', '849843:3': '1루수 땅볼 아웃', '849843:4': '중견수 방면 홈런', '849843:5': '투수 번트 땅볼 아웃',
    '849843:6': '좌중간 방면 홈런', '849843:7': '우익수 직선타 아웃', '849843:8': '헛스윙 삼진', '849843:9': '유격수 송구 실책 출루', '849843:10': '3루수 뜬공 아웃',
    '849843:11': '2루수 땅볼 아웃', '849843:12': None, '849843:13': '헛스윙 삼진', '849843:14': '중견수 플라이 아웃', '849843:15': '헛스윙 삼진',
    '849843:16': '좌익수 플라이 아웃', '849843:17': '헛스윙 삼진', '849843:18': '볼넷', '849843:19': '헛스윙 삼진', '849843:20': '우익수 플라이 아웃'}
pitches = []
for d, r in zip(timeline['decisions'][:N], reveals):
    s = d['situation']; pre = d['pre']
    rec = ({'provenance': 'asof_replay', 'status': 'ready', 'reason': None, 'recorded_at': '2026-09-30T08:48:53Z',
            'candidates': pre['recommendation']['candidates'], 'policy': {'name': 'ARM-B P3 frozen <=2025', 'identity': 'a6dffaea'}}
           if pre['status'] == 'ready' else
           {'provenance': 'asof_replay', 'status': 'unsupported', 'reason': pre['reason'], 'recorded_at': None, 'candidates': [], 'policy': None}
           if pre['reason'] else {'provenance': 'absent', 'status': 'missing', 'reason': None, 'recorded_at': None, 'candidates': [], 'policy': None})
    a = r['actual']
    pitches.append({'index': d['index'], 'key': f"{d['pa_id']}:{d['pitch_number']}", 'pa_key': d['pa_id'], 'at_bat_number': d['at_bat_number'],
                    'pitch_number': d['pitch_number'], 'inning': s['inning'], 'half': s['half'],
                    'situation_before': {k: s[k] for k in ('outs', 'balls', 'strikes', 'bases', 'home_score', 'away_score')},
                    'batter': d['batter'], 'pitcher': d['pitcher'], 'rec': rec,
                    'actual': a | {'call_label': a['result_label'], 'catcher_setup': None}})

# finished-game fixture ends at the last complete PA inside the sample (a real final game has no open PA)
last_end = max(i for i, p in enumerate(pitches) if p['actual']['event'])
pitches = pitches[:last_end + 1]; N = len(pitches)

def pas_of(rows, in_progress=None):
    out = []
    for p in rows:
        if not out or out[-1]['pa_key'] != p['pa_key']:
            out.append({'pa_key': p['pa_key'], 'at_bat_number': p['at_bat_number'], 'inning': p['inning'], 'half': p['half'],
                        'batter': p['batter'], 'pitcher': p['pitcher'], 'complete': False, 'event': None, 'event_label': None,
                        'result_text': None, 'play_text': None, 'pitch_count': 0, 'first_index': p['index'], 'last_index': p['index']})
        pa = out[-1]; pa['pitch_count'] += 1; pa['last_index'] = p['index']
        a = p['actual']
        if a['event']:
            pa.update(complete=True, event=a['event'], event_label=a['event_label'], play_text=a['play_text'], result_text=RESULT_TEXT.get(p['pa_key']))
    return out

game = {'game_pk': 849843, 'start': '2026-09-30T02:00:00Z', 'start_kst': '2026-09-30T11:00:00+09:00', 'date_kst': '2026-09-30',
        'official_date': '2026-09-29', 'status': 'final', 'status_text': 'Final', 'live': False, 'away_team': 'Chicago Cubs',
        'home_team': 'San Diego Padres', 'away_abbr': 'CHC', 'home_abbr': 'SD', 'away_score': 0, 'home_score': 8, 'inning': None,
        'half': None, 'game_number': 1, 'double_header': False, 'double_header_code': 'N', 'embargoed': False, 'has_history': True}
base = {'schema': 'pitcheezy-service-game-v2', 'zone_bounds': timeline['location']['zone_bounds']}
final = base | {'source': {'kind': 'archive', 'session_id': None}, 'game': game, 'feed': {'as_of': '2026-09-30T05:10:00Z', 'ok': True, 'stale_s': None},
                'cutoff': {'index': pitches[-1]['index'], 'pitch_key': pitches[-1]['key']}, 'pitches': pitches, 'pas': pas_of(pitches),
                'current': {'phase': 'final', 'situation': None, 'batter': None, 'pitcher': None, 'pa_key': None}, 'next': None}

def live_at(c):
    """Demo-like live state: pitches 0..c revealed, pitch c+1 is next (its archived pre-pitch rec is the NEXT rec)."""
    rows, nxt = pitches[:c+1], pitches[c+1]
    s = nxt['situation_before'] | {'inning': nxt['inning'], 'half': nxt['half']}
    pas = pas_of(rows)
    if pas[-1]['pa_key'] != nxt['pa_key']:
        pas.append({'pa_key': nxt['pa_key'], 'at_bat_number': nxt['at_bat_number'], 'inning': nxt['inning'], 'half': nxt['half'], 'batter': nxt['batter'],
                    'pitcher': nxt['pitcher'], 'complete': False, 'event': None, 'event_label': None, 'result_text': None, 'play_text': None,
                    'pitch_count': 0, 'first_index': None, 'last_index': None})
    return base | {'source': {'kind': 'demo', 'session_id': 'fixture'},
                   'game': game | {'status': 'live', 'status_text': 'In Progress', 'live': True, 'away_score': s['away_score'], 'home_score': s['home_score'],
                                   'inning': nxt['inning'], 'half': nxt['half']},
                   'feed': {'as_of': '2026-10-06T07:00:00Z', 'ok': True, 'stale_s': None}, 'cutoff': {'index': c, 'pitch_key': rows[-1]['key']},
                   'pitches': rows, 'pas': pas, 'current': {'phase': 'pre_pitch', 'situation': s, 'batter': nxt['batter'], 'pitcher': nxt['pitcher'], 'pa_key': nxt['pa_key']},
                   'next': {'pitch_key': None, 'rec': nxt['rec']}}

# live cutoff in the middle of a plate appearance, then three more pitches revealed
c = next(i for i in range(30, N-5) if pitches[i]['pa_key'] == pitches[i+1]['pa_key'] and pitches[i+1]['rec']['status'] == 'ready')
out = {'note': 'UI fixture from real 849843 (first %d pitches) in CONTRACT rev1 shape; synthetic list entries are marked synthetic' % N,
       'final': final, 'live': live_at(c), 'live_more': live_at(c+3), 'live_cut': c}
Path('tests/fixtures/service-849843.json').write_text(json.dumps(out, ensure_ascii=False))
print('cutoff', c, 'pitches', N, 'pas', len(final['pas']), 'unsupported', sum(p['rec']['status'] != 'ready' for p in pitches))
