"""Real-game rehearsal recorder/report and the lead-time arithmetic, on synthetic polls and snapshots."""
from datetime import datetime, timedelta, timezone
import gzip
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import live_rehearsal as lr  # noqa: E402

BASE = datetime(2026, 10, 2, 0, 0, 0, tzinfo=timezone.utc)


def feed(pitches):
    """A feed whose k-th pitch starts 20 s after the previous one and takes 4 s."""
    text = lambda seconds: (BASE + timedelta(seconds=seconds)).isoformat().replace('+00:00', 'Z')
    events = [{'index': k, 'isPitch': True, 'startTime': text(20 * k), 'endTime': text(20 * k + 4)} for k in range(pitches)]
    return {'liveData': {'plays': {'allPlays': [{'about': {'atBatIndex': 0}, 'playEvents': events}]}}}


def test_record_polls_until_the_game_that_was_live_is_over(tmp_path):
    lines = iter([{'status': 'not_live', 'game_status': 'Pre-Game'}, {'status': 'buffering'}, {'status': 'ready'},
                  {'http': 503, 'error': 'http'}, {'status': 'ready'}]
                 + [{'status': 'not_live', 'game_status': 'Final'}] * 5)
    sleeps = []
    result = lr.record('http://x', 7, tmp_path, poll=lambda server, game_pk: next(lines), sleep=sleeps.append, clock=lambda: 0.0)
    assert result['polls'] == 8 and result['seen_live'] and result['finished']
    assert sleeps == [30.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]  # idle before the game; one failed poll does not end it
    assert len((tmp_path / '7_polls.jsonl').read_text().splitlines()) == 8


def test_report_measures_publication_lag_margin_and_coverage(tmp_path):
    snapshots, base = tmp_path / 'snapshots', BASE.timestamp()
    snapshots.mkdir()
    for pitches, fetched in ((1, 2), (2, 24 + 9), (3, 44 + 6)):  # pitch 2 seen 9 s after it ended, pitch 3 after 6 s
        path = snapshots / f'7_{pitches}.json.gz'
        with gzip.open(path, 'wt') as handle:
            json.dump(feed(pitches), handle)
        os.utime(path, (base + fetched, base + fetched))
    polls = [{'status': 'ready', 'ms': 5.0, 'rec_status': 'computing'},
             {'status': 'ready', 'ms': 7.0, 'rec_status': 'ready', 'rec_key': '7:1:2', 'rec_seconds': {'total': .15}},
             {'status': 'ready', 'ms': 9.0, 'rec_status': 'unsupported', 'rec_key': '7:1:3', 'rec_seconds': {'total': .1}},
             {'http': None, 'ms': 3.0, 'error': 'URLError'}]
    (tmp_path / '7_polls.jsonl').write_text('\n'.join(json.dumps(p) for p in polls) + '\n')
    result = lr.report(7, tmp_path, snapshots)
    assert result['pitches_first_seen_live'] == 2 and result['publication_lag_s']['median'] == 7.5
    assert result['next_pitch_margin_s']['n'] == 1 and result['next_pitch_margin_s']['median'] == 7.0  # 40 - 33
    assert result['share_with_read_time_by_broadcast_delay']['0'] == 1.0
    assert result['polls']['errors'] == 1 and result['polls']['status'] == {'None': 1, 'ready': 3}
    assert (result['recommendation']['decisions_ready'], result['recommendation']['decisions_unsupported']) == (1, 1)
    assert json.loads((tmp_path / '7_report.json').read_text()) == result


def test_lead_share_needs_the_broadcast_delay_to_cover_lag_and_poll():
    feeds = [feed(11)]  # ten gaps: each next pitch starts 16 s after the previous one ended
    kwargs = dict(client_s=0.0, compute_s=0.0, delays=(0, 10))
    assert lr.lead_shares(feeds, lag_s=0, poll_s=0, **kwargs)['share_by_broadcast_delay'] == {'0': 1.0, '10': 1.0}
    late = lr.lead_shares(feeds, lag_s=12, poll_s=0, **kwargs)  # 16 - 12 = 4 s of lead: under the 5 s reading time
    assert late['pairs'] == 10 and late['share_by_broadcast_delay'] == {'0': 0.0, '10': 1.0}
    half = lr.lead_shares(feeds, lag_s=6, poll_s=10, **kwargs)  # on time only when the poll phase is under 5 s
    assert half['share_by_broadcast_delay']['0'] == 0.5
