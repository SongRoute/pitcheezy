"""Synthetic tests for apps/observer/ops/ws_schedule_sync.py. Fake schedule bodies only; no network, no launchd."""
import json
import plistlib
import shutil
import sys
from pathlib import Path

import pytest

OPS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OPS))
import ws_schedule_sync as ss  # noqa: E402

PKS = ss.KNOWN_PKS
# fake confirmed times (UTC). 849801 at 15:30Z = 00:30 KST next day (date rollover).
TIMES = {849805: '2026-10-24T00:08:00Z', 849802: '2026-10-25T00:08:00Z', 849804: '2026-10-26T23:08:00Z',
         849803: '2026-10-27T23:08:00Z', 849799: '2026-10-28T23:08:00Z', 849801: '2026-10-30T15:30:00Z',
         849800: '2026-11-02T01:08:00Z'}


def game(pk, when, gtype='W', detailed='Scheduled', tbd=False, if_nec='N', number=None):
    return {'gamePk': pk, 'gameType': gtype, 'gameDate': when, 'ifNecessary': if_nec, 'seriesGameNumber': number,
            'status': {'detailedState': detailed, 'abstractGameState': 'Preview', 'startTimeTBD': tbd}}


def body(games):
    return {'totalGames': len(games), 'dates': [{'date': g['gameDate'][:10], 'games': [g]} for g in games]}


def confirmed(**over):
    games = []
    for i, pk in enumerate(PKS, 1):
        g = game(pk, TIMES[pk], number=i, if_nec='Y' if i >= 5 else 'N')
        g.update(over.get(str(pk), {}))
        games.append(g)
    return games


@pytest.fixture
def env(tmp_path):
    alert = tmp_path/'ws_alert_config.json'
    shutil.copy(OPS/'ws_alert_config.json', alert)
    out = tmp_path/'plists'

    def run(games, *extra, template=None):
        src = tmp_path/'schedule.json'
        src.write_text(json.dumps(body(games) if isinstance(games, list) else games))
        argv = ['--from-file', str(src), '--out-dir', str(out), '--alert-config', str(alert), *extra]
        if template:
            argv += ['--template', str(template)]
        return ss.main(argv)
    return alert, out, run


def test_dry_run_diff_plists_and_no_write(env, capsys):
    alert, out, run = env
    before = alert.read_bytes()
    assert run(confirmed()) == 0
    text = capsys.readouterr().out
    assert alert.read_bytes() == before  # dry run: untouched
    assert '849805 G1: 2026-10-24T00:00:00Z -> 2026-10-24T00:08:00Z' in text
    assert '2026-10-24 09:08 KST' in text
    assert 'launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.pitcheezy.ws-operate.849805.plist' in text
    assert 'launchctl bootout gui/$(id -u)/com.pitcheezy.ws-operate.849805' in text
    assert sorted(p.name for p in out.iterdir()) == sorted(f'com.pitcheezy.ws-operate.{pk}.plist' for pk in PKS)
    p = plistlib.loads((out/'com.pitcheezy.ws-operate.849805.plist').read_bytes())
    # 00:08Z = 09:08 KST; job at first pitch - (lead_min 30 + start_retry_window_s 1800 s) = 08:08 KST
    assert p['StartCalendarInterval'] == {'Month': 10, 'Day': 24, 'Hour': 8, 'Minute': 8}
    assert p['Label'] == 'com.pitcheezy.ws-operate.849805'
    assert '--date 2026-10-24' in p['ProgramArguments'][-1] and '$(' not in p['ProgramArguments'][-1]
    assert p['RunAtLoad'] is False and p['AbandonProcessGroup'] is True
    tpl = plistlib.loads((OPS/'com.pitcheezy.ws-operate.plist.example').read_bytes())
    assert p['EnvironmentVariables'] == tpl['EnvironmentVariables']


def test_kst_date_rollover(env):
    alert, out, run = env
    assert run(confirmed()) == 0
    # 849801 at 2026-10-30T15:30Z = 2026-10-31 00:30 KST: the launcher's date is 10-31, the job runs 10-30 23:30 KST
    p = plistlib.loads((out/'com.pitcheezy.ws-operate.849801.plist').read_bytes())
    assert '--date 2026-10-31' in p['ProgramArguments'][-1]
    assert p['StartCalendarInterval'] == {'Month': 10, 'Day': 30, 'Hour': 23, 'Minute': 30}
    # 849804 at 2026-10-26T23:08Z = 2026-10-27 08:08 KST -> job 07:08 KST on 10-27
    p = plistlib.loads((out/'com.pitcheezy.ws-operate.849804.plist').read_bytes())
    assert '--date 2026-10-27' in p['ProgramArguments'][-1]
    assert p['StartCalendarInterval'] == {'Month': 10, 'Day': 27, 'Hour': 7, 'Minute': 8}


def test_write_touches_only_starts(env):
    alert, out, run = env
    before_text = alert.read_text()
    assert run(confirmed(), '--write') == 0
    after_text = alert.read_text()
    before, after = json.loads(before_text), json.loads(after_text)
    assert {e['game_pk']: e['start'] for e in after['schedule']} == TIMES
    before.pop('schedule'), after.pop('schedule')
    assert before == after
    changed = [(a, b) for a, b in zip(before_text.splitlines(), after_text.splitlines()) if a != b]
    assert len(before_text.splitlines()) == len(after_text.splitlines()) and len(changed) == 7
    assert all('"game_pk"' in a and a.replace(a[a.index('"start"'):], '') == b.replace(b[b.index('"start"'):], '')
               for a, b in changed)


def test_if_necessary_flagged_still_gets_plist(env, capsys):
    alert, out, run = env
    assert run(confirmed()) == 0
    text = capsys.readouterr().out
    for pk in (849799, 849801, 849800):
        assert f'{pk} if_necessary' in text.split('flagged: ')[1]
        assert (out/f'com.pitcheezy.ws-operate.{pk}.plist').exists()


def test_postponed_flagged_no_plist_old_start_kept(env, capsys):
    alert, out, run = env
    games = confirmed(**{'849803': {'status': {'detailedState': 'Postponed', 'startTimeTBD': False}}})
    assert run(games, '--write') == 0
    assert 'postponed' in capsys.readouterr().out
    assert not (out/'com.pitcheezy.ws-operate.849803.plist').exists()
    starts = {e['game_pk']: e['start'] for e in json.loads(alert.read_text())['schedule']}
    assert starts[849803] == '2026-10-28T00:00:00Z' and starts[849805] == TIMES[849805]


def test_postponed_with_rescheduled_row_uses_new_row(env):
    alert, out, run = env
    games = confirmed()
    games.append(game(849803, '2026-10-29T23:08:00Z', detailed='Postponed'))
    assert run(games) == 0
    p = plistlib.loads((out/'com.pitcheezy.ws-operate.849803.plist').read_bytes())
    assert '--date 2026-10-28' in p['ProgramArguments'][-1]


def test_non_w_rows_ignored(env, capsys):
    alert, out, run = env
    assert run(confirmed() + [game(778001, '2026-10-20T00:08:00Z', gtype='L')]) == 0
    assert 'ignored non-W rows: 1' in capsys.readouterr().out


@pytest.mark.parametrize('case', ['unknown_pk', 'midnight', 'tbd', 'missing', 'two_live', 'not_w', 'bad_body'])
def test_refusals_write_nothing(env, case, capsys):
    alert, out, run = env
    before = alert.read_bytes()
    games = confirmed()
    if case == 'unknown_pk':
        games.append(game(849999, '2026-11-03T00:08:00Z'))
    elif case == 'midnight':
        games[2]['gameDate'] = '2026-10-27T00:00:00Z'
    elif case == 'tbd':
        games[3]['status']['startTimeTBD'] = True
    elif case == 'missing':
        games.pop()
    elif case == 'two_live':
        games.append(game(849805, '2026-10-24T01:08:00Z'))
    elif case == 'not_w':
        games[0]['gameType'] = 'L'
    elif case == 'bad_body':
        games = {'messageNumber': 10, 'message': 'error'}
    assert run(games, '--write') == 2
    assert alert.read_bytes() == before
    assert not out.exists()
    assert 'REFUSED' in capsys.readouterr().err


def test_placeholder_schedule_refused(env, capsys):
    """The schedule as saved on 10-17/18: every time the 00:00:00Z placeholder."""
    alert, out, run = env
    games = [game(pk, f'{TIMES[pk][:10]}T00:00:00Z') for pk in PKS]
    assert run(games) == 2
    cap = capsys.readouterr()
    assert 'midnight_placeholder' in cap.out and 'time not confirmed' in cap.err
    assert not out.exists()


def test_template_missing_key_refused(env, tmp_path, capsys):
    alert, out, run = env
    tpl = plistlib.loads((OPS/'com.pitcheezy.ws-operate.plist.example').read_bytes())
    del tpl['StartCalendarInterval']
    bad = tmp_path/'bad.plist'
    bad.write_bytes(plistlib.dumps(tpl))
    assert run(confirmed(), template=bad) == 2
    assert 'missing keys' in capsys.readouterr().err


def test_template_without_date_expression_refused(env, tmp_path, capsys):
    alert, out, run = env
    tpl = plistlib.loads((OPS/'com.pitcheezy.ws-operate.plist.example').read_bytes())
    tpl['ProgramArguments'][-1] = 'exec python3 ws_operate.py start --date 2026-10-24'
    bad = tmp_path/'bad.plist'
    bad.write_bytes(plistlib.dumps(tpl))
    assert run(confirmed(), template=bad) == 2


def test_out_dir_in_launchagents_or_repo_refused(env, tmp_path, capsys):
    alert, out, run = env
    src = tmp_path/'s.json'
    src.write_text(json.dumps(body(confirmed())))
    for bad in (Path.home()/'Library'/'LaunchAgents', OPS/'tmp-plists'):
        assert ss.main(['--from-file', str(src), '--out-dir', str(bad), '--alert-config', str(alert)]) == 2
    assert not (OPS/'tmp-plists').exists()


def test_ws_operate_path_override(env, tmp_path):
    alert, out, run = env
    assert run(confirmed(), '--ws-operate', '/x/launch/apps/observer/ops/ws_operate.py') == 0
    p = plistlib.loads((out/'com.pitcheezy.ws-operate.849805.plist').read_bytes())
    assert '/x/launch/apps/observer/ops/ws_operate.py start --date 2026-10-24' in p['ProgramArguments'][-1]


def test_fetch_is_one_request_with_ua(monkeypatch, env):
    alert, out, run = env
    calls = []

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(body(confirmed())).encode()

    def fake_urlopen(req, timeout):
        calls.append((req.full_url, req.get_header('User-agent'), timeout))
        return Resp()
    monkeypatch.setattr(ss.urllib.request, 'urlopen', fake_urlopen)
    assert ss.main(['--fetch', '--out-dir', str(out), '--alert-config', str(alert)]) == 0
    assert len(calls) == 1
    url, ua, timeout = calls[0]
    assert 'gamePks=849805,849802,849804,849803,849799,849801,849800' in url
    assert ua == 'pitcheezy-ws-ops-schedule-check' and timeout == 10
