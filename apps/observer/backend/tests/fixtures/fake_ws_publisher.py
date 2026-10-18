"""Synthetic stand-in for ``python -m observer_app.publisher`` in the ws_operate tests (no MLB, no network).

Accepts the publisher's arguments, holds the same kind of output lock, writes v1/health.json every cycle and
v1/game/<pk>.json with the publish block last, and exits 0 on SIGINT. Its behaviour comes from
<run_dir>/fake_control.json (the launcher gives children a minimal environment, so no env switches):
  games: [pk, ...]          game files to write (default [990001])
  interval: s               cycle length (default 0.1)
  crash_first: k            the first k starts exit with exit_code after exit_after cycles
  exit_after, exit_code     (default 3 cycles, code 3)
  hang_after: n             stop writing health after n cycles (keeps running)
  final_after: n            game files say final after n cycles (default never)
  upload_blocked: true      state file carries a blocked upload attempt
Every start/end is appended to <run_dir>/fake_events.jsonl.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time


def write(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.'+path.name+'.tmp')
    tmp.write_text(json.dumps(body))
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser()
    for name in ('--out', '--date', '--config', '--feed-dir', '--watch-dir', '--intent-dir', '--demo-dir'):
        parser.add_argument(name)
    args = parser.parse_args()
    out = Path(args.out)
    run_dir = out.parent
    control = json.loads((run_dir/'fake_control.json').read_text()) if (run_dir/'fake_control.json').exists() else {}
    config = json.loads(Path(args.config).read_text())
    events = run_dir/'fake_events.jsonl'
    counter = run_dir/'fake_starts'
    n = int(counter.read_text()) + 1 if counter.exists() else 1
    counter.write_text(str(n))
    lock = open(run_dir/f'.{out.name}.lock', 'a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print('another publisher is already running on this output (lock is held)')
        return 1
    with open(events, 'a') as handle:
        handle.write(json.dumps({'pid': os.getpid(), 'start': time.time(), 'n': n,
                                 'token_seen': bool(os.environ.get('PITCHEEZY_PUBLISH_TOKEN')),
                                 'delay': config.get('fixed_delay_s'), 'upload': config['upload']['enabled']})+'\n')
    state_path = Path(f'{out}.state.json')
    sequence = (json.loads(state_path.read_text()).get('sequence', 0) if state_path.exists() else 0)
    games = control.get('games', [990001])
    cycles, code = 0, 0
    try:
        while True:
            cycles += 1
            if control.get('crash_first', 0) >= n and cycles > control.get('exit_after', 3):
                code = control.get('exit_code', 3)
                break
            if control.get('hang_after') is None or cycles <= control['hang_after']:
                sequence += 1
                final = control.get('final_after') is not None and cycles >= control['final_after']
                for pk in games:
                    write(out/f'v1/game/{pk}.json', {'game_pk': pk, 'pitches': [], 'publish': {'sequence': sequence, 'final': final}})
                health = {'schema': 'pitcheezy-publish-health-v1', 'sequence': sequence,
                          'published_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'files_written': 2,
                          'unavailable_games': [], 'feed_failing_games': [],
                          'game_checks': {str(pk): {'checked_at': 'x', 'sequence_checked': sequence,
                                                    'file_sequence': sequence, 'feed_ok': True} for pk in games}}
                if config['upload']['enabled']:
                    health['upload'] = {'code': 'upload_failed' if control.get('upload_blocked') else None,
                                        'pending': 1 if control.get('upload_blocked') else 0, 'dry_run': False}
                write(out/'v1/health.json', health)
                state = {'sequence': sequence}
                if control.get('upload_blocked'):
                    state['upload_attempts'] = {'v1/health.json': {'phase': 'blocked', 'reason': 'conflict_412'}}
                    state['upload_pending'] = {'v1/health.json': 'put'}
                write(state_path, state)
            time.sleep(control.get('interval', 0.1))
    except KeyboardInterrupt:
        code = 0
    with open(events, 'a') as handle:
        handle.write(json.dumps({'pid': os.getpid(), 'end': time.time(), 'n': n, 'code': code})+'\n')
    return code


if __name__ == '__main__':
    sys.exit(main())
