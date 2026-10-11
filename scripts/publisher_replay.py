"""Offline rehearsal: replay ONE finished capture through the public publisher and measure what it writes.

    python -B scripts/publisher_replay.py --job-dir <job> --run-name run-001 --game 849831 \
        --capture-db <job>/input/capture-849831.sqlite3 --capture-sha256 <sha> --commit <git sha> \
        --config <job>/replay_config.json --policy armb

The capture copy is read through ``capture_to_replay``'s reader (read-only, immutable; the original capture is
never opened here: the database must sit inside the job folder). A simulated clock starts at the capture's first
receipt and moves one publish interval per cycle; at simulated time ``t`` the publisher is answered with the
newest stored version the capture had received by ``t`` (``observer_app.capture_replay``): no look-ahead.

Everything the run writes is under ``<job>/<run-name>/``: the publisher's OWN new store and ledger
(``publisher-store.sqlite3``), its output (``out/``), state, temporary files, the policy's scratch folder and style
cache copy, and the results (``CYCLES.jsonl``, ``SUMMARY.json``, ``SUMMARY.md``). No snapshot recording, no port,
no network, no other process. An interpreter audit hook refuses and records any write outside that folder, any
new process and any socket (``observer_app.publish_audit.FsAudit``); it also lists the files read under T7.

The run stops at the first broken invariant, outside write, memory or time cap; what was written until then
stays. Exit codes: 0 completed, 2 refused before start (nothing written), 3 stopped by a stop condition.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True  # before any other import: no __pycache__ outside the job folder

import argparse  # noqa: E402
from collections import Counter  # noqa: E402
from contextlib import closing  # noqa: E402
import gzip  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402
import re  # noqa: E402
import resource  # noqa: E402
import subprocess  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
for _path in (REPO/'scripts', REPO/'apps/observer/backend'):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
import capture_to_replay as c2r  # noqa: E402  (stdlib only: the capture reader and the schedule item)

T7 = Path('/Volumes/T7 Shield')
STYLE_DIR = T7/'pitcheezy/pitchmdp/runs/DEMO-WS-2026/style'
CLEARED_ENV = ('PITCHEEZY_OBSERVER_LIVE_STORE', 'PITCHEEZY_OBSERVER_LIVE_REPLAY_DIR', 'PITCHEEZY_OBSERVER_LIVE_REPLAY_POLL_S',
               'PITCHEEZY_OBSERVER_LIVE_REPLAY_PACE', 'PITCHEEZY_OBSERVER_LIVE_DELAY_S', 'PITCHEEZY_OBSERVER_REC_LEDGER',
               'PITCHEEZY_OBSERVER_FEED_LAG_DELAYED_S', 'PITCHEEZY_PUBLISH_OUT_DIR', 'PITCHEEZY_PUBLISH_STATE',
               'PITCHEEZY_PUBLISH_INTERVAL_S', 'PITCHEEZY_PUBLISH_DELAY_S', 'PITCHEEZY_PUBLISH_COMMIT', 'PITCHEEZY_FEED_DIR',
               'PITCHEEZY_WATCH_DIR', 'PITCHEEZY_INTENT_DIR', 'PITCHEEZY_DEMO_DIR', 'PITCHEEZY_PLAYER_SNAPSHOT_DIR')
CONFIG_KEYS = ('publish_interval_s', 'fixed_delay_s', 'tail_cycles', 'max_rss_bytes', 'min_free_bytes', 'max_seconds',
               'settle_cap_s', 'max_body_bytes', 'policy_load_cap_s')


class Refused(Exception):
    """Not started: nothing was written."""


class Stop(Exception):
    """A registered stop condition: the run ends here and keeps what it has."""

    def __init__(self, reason, detail=None):
        super().__init__(reason)
        self.reason, self.detail = reason, detail


def sha256_file(path):
    return c2r.file_sha256(path)


def capture_rows(db_path, game_pk, max_body_bytes):
    """(rows, info) of a capture store: ``rows`` = ``[(fetched, load)]`` in receipt (``seq``) order for
    ``capture_replay.CaptureReplayTransport``; ``info`` gets ``final_fetched`` once a loaded body is Final.
    Bodies stay in the database and are read one at a time; each is checked against its stored sha256."""
    db = c2r.open_source(db_path)
    meta = db.execute('SELECT seq, fetched, sha FROM snapshots WHERE game_pk=? ORDER BY seq', (int(game_pk),)).fetchall()
    if not meta:
        db.close()
        raise Refused(f'the capture holds no stored version of game {game_pk}')
    info = {'versions': len(meta), 'first_fetched': meta[0][1], 'last_fetched': meta[-1][1], 'final_fetched': None,
            'final_seq': None, 'loads': 0, 'db': db}
    lock = threading.Lock()

    def loader(seq, fetched, sha):
        def load():
            with lock:
                blob = db.execute('SELECT body FROM snapshots WHERE seq=?', (seq,)).fetchone()[0]
            raw = c2r.inflate(blob, max_body_bytes)
            if raw is None or hashlib.sha256(raw).hexdigest() != sha:
                raise ValueError(f'stored version {seq} does not match its sha256 or exceeds the body limit')
            body = json.loads(raw)
            info['loads'] += 1
            if info['final_fetched'] is None and ((body.get('gameData') or {}).get('status') or {}).get('abstractGameState') == 'Final':
                info['final_fetched'], info['final_seq'] = fetched, seq
            return body
        return load
    return [(fetched, loader(seq, fetched, sha)) for seq, fetched, sha in meta], info


def free_memory_bytes():
    """free + inactive + speculative pages (``vm_stat``), or None if it cannot be read. Called before the audit."""
    try:
        out = subprocess.run(['/usr/bin/vm_stat'], capture_output=True, text=True, timeout=10, env={'LC_ALL': 'C'}).stdout
        page = int(re.search(r'page size of (\d+) bytes', out)[1])
        pages = sum(int(re.search(rf'^{name}:\s+(\d+)\.', out, re.M)[1]) for name in ('Pages free', 'Pages inactive', 'Pages speculative'))
        return page*pages
    except Exception:  # noqa: BLE001
        return None


def head_commit(repo=REPO):
    """HEAD's commit sha from the git files alone (no git process). None if it cannot be read."""
    try:
        dot = Path(repo)/'.git'
        gitdir = dot if dot.is_dir() else (Path(repo)/dot.read_text().split('gitdir:', 1)[1].strip()).resolve()
        head = (gitdir/'HEAD').read_text().strip()
        if not head.startswith('ref:'):
            return head
        ref = head.split(None, 1)[1]
        common = gitdir/(gitdir/'commondir').read_text().strip() if (gitdir/'commondir').is_file() else gitdir
        for base in (gitdir, common.resolve()):
            if (base/ref).is_file():
                return (base/ref).read_text().strip()
        for line in (common.resolve()/'packed-refs').read_text().splitlines():
            if line.endswith(' '+ref):
                return line.split()[0]
    except (OSError, IndexError):
        pass
    return None


def tracked_changes(repo=REPO):
    """Paths with uncommitted changes to tracked files, or None if git cannot say. Read-only: optional locks are
    off, so the index is not refreshed. Called before the audit."""
    try:
        done = subprocess.run(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=no'], capture_output=True,
                              text=True, timeout=30, env=os.environ | {'GIT_OPTIONAL_LOCKS': '0'}, check=True)
        return [line[3:] for line in done.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        return None


def spread(values):
    values = sorted(values)
    if not values:
        return {'n': 0}
    pick = lambda q: values[min(len(values)-1, int(q*len(values)))]
    return {'n': len(values), 'min': values[0], 'p50': pick(.5), 'p90': pick(.9), 'p99': pick(.99), 'max': values[-1],
            'mean': sum(values)/len(values)}


def general(pointer):
    """A JSON pointer with array indices and id-like keys folded, so one line stands for every element."""
    return re.sub(r'/\d+(?::\d+)*(?=/|$)', '/*', pointer)


def armb_idle(armb):
    return not (armb.pending or armb.running is not None or armb.loading)


def run(*, rows, info, run_dir, game_pk, commit, limits, policy_factory=None, fs_audit=None, schemas=None,
        publish_config=None, perf=time.perf_counter, rss=lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        log=lambda text: print(text, file=sys.stderr, flush=True)):
    """Replay ``rows`` through a publisher that lives entirely in ``run_dir``. Returns the summary (also written)."""
    from observer_app import publish_audit, publish_schema, publisher
    from observer_app.capture_replay import CaptureReplayTransport, SimClock
    from observer_app.live_armb import LiveArmB
    from observer_app.live_store import LiveSnapshotStore, PersistentLiveFeed

    run_dir = Path(run_dir).resolve()
    out, started = run_dir/'out', perf()
    for folder in ('empty', 'demo', 'tmp'):
        (run_dir/folder).mkdir(parents=True, exist_ok=True)
    clock = SimClock(rows[0][0])
    transport = CaptureReplayTransport(rows, clock, game_pk=game_pk, schedule_game=c2r._schedule_game)
    store = LiveSnapshotStore(run_dir/'publisher-store.sqlite3')  # the publisher's own: never the capture
    feed = PersistentLiveFeed(transport, store=store, source='replay_poll', delay_s=0, clock=clock, wall=clock, sleep=lambda _: None)
    os.environ['PITCHEEZY_OBSERVER_LIVE_ARMB'] = '1' if policy_factory else '0'
    armb = LiveArmB(policy_factory) if policy_factory else LiveArmB(lambda: None)
    config = publisher.load_config(publish_config, env={}, out_dir=str(out), state_path=str(run_dir/'publisher.state.json'),
                                   publish_interval_s=limits['publish_interval_s'], fixed_delay_s=limits['fixed_delay_s'])
    app, pub = publisher.build(config, live_feed=feed, live_policy=armb, feed_dir=run_dir/'empty', watch_dir=run_dir/'empty',
                               intent_dir=run_dir/'empty', demo_dir=run_dir/'demo', demo_wall=clock, service_wall=clock,
                               wall=clock, commit=(commit, None))
    ledger_ids = lambda: {row['decision_id'] for row in store.served_recommendations(game_pk) if row['outcome'] == 'first'}
    audit = publish_audit.PublishAudit(contract=config['contract_version'], ledger_ids=ledger_ids,
                                       forbidden=[str(run_dir), str(run_dir.parent), str(Path.home()), str(REPO)])
    game_path = pub.layout['game'].format(game_pk=game_pk)
    interval = config['publish_interval_s']
    cycles, shas, schema_problems, kinds_seen = [], {}, Counter(), Counter()
    bytes_by_kind = {}
    states, codes, recs = Counter(), Counter(), Counter()
    final_published, stop, policy = None, None, {'requested': bool(policy_factory)}
    over_interval = 0

    def guard(stage):
        if fs_audit is not None and fs_audit.violations:
            raise Stop('write_process_or_network_outside_the_job_folder', fs_audit.violations[:20])
        if audit.violations:
            raise Stop('invariant', audit.violations[:50])
        if rss() > limits['max_rss_bytes']:
            raise Stop('rss_cap', {'stage': stage, 'rss': rss()})
        if perf()-started > limits['max_seconds']:
            raise Stop('runtime_cap', {'stage': stage})

    def one_cycle(label):
        nonlocal final_published, over_interval
        begun = perf()
        health = pub.cycle()
        wall_s = perf()-begun
        waited = perf()
        while not armb_idle(armb) and perf()-waited < limits['settle_cap_s']:
            time.sleep(.005)
        settle_s = perf()-waited
        if not armb_idle(armb):
            raise Stop('policy_did_not_settle', {'cycle': len(cycles)+1})
        over_interval += settle_s+wall_s > interval
        written, sizes = [], {}
        present = {}
        for path in sorted(out.rglob('*')):
            if path.is_file():
                present[path.relative_to(out).as_posix()] = path
        for name in set(shas)-set(present):
            audit.gone(name)
            del shas[name]
        for name, path in present.items():
            data = path.read_bytes()
            sha = hashlib.sha256(data).hexdigest()
            if shas.get(name) == sha:
                continue
            shas[name] = sha
            written.append(name)
            kind = publish_audit.kind_of(name, config['contract_version'])
            body = audit.check(name, data, sha)
            packed = len(gzip.compress(data, compresslevel=6))
            entry = sizes.setdefault(kind or 'unknown', {'files': 0, 'raw': 0, 'gzip': 0})
            entry['files'] += 1
            entry['raw'] += len(data)
            entry['gzip'] += packed
            total = bytes_by_kind.setdefault(kind or 'unknown', {'writes': 0, 'raw_total': 0, 'gzip_total': 0, 'raw_max': 0, 'gzip_max': 0,
                                                                'raw_last': 0, 'gzip_last': 0})
            total['writes'] += 1
            total['raw_total'] += len(data)
            total['gzip_total'] += packed
            total['raw_max'], total['gzip_max'] = max(total['raw_max'], len(data)), max(total['gzip_max'], packed)
            total['raw_last'], total['gzip_last'] = len(data), packed
            if body is not None and kind is not None:
                kinds_seen[kind] += 1
                if schemas is not None:
                    for problem in publish_schema.problems(body, schemas[kind]):
                        schema_problems[kind, general(problem['pointer']), problem['keyword'], problem['seen']] += 1
        facts = None
        if game_path in present:
            facts = publish_audit.game_facts(json.loads(present[game_path].read_bytes()))
            states[f"{facts['state']}|{facts['reason']}"] += 1
            recs[facts['recommendation'] if facts['recommendation'] == 'published' else f"none|{facts['state']}|{facts['reason']}"] += 1
            if facts['final'] is True and final_published is None:
                final_published = {'simulated': clock(), 'cycle': len(cycles)+1, 'sequence': health['sequence']}
        for item in health['unavailable_games']:
            codes[item['code']] += 1
        row = {'cycle': len(cycles)+1, 'label': label, 'simulated': round(clock(), 3), 'capture_index': transport.index(),
               'sequence': health['sequence'], 'wall_s': round(wall_s, 6), 'settle_s': round(settle_s, 6),
               'files_written': len(written), 'written': written, 'bytes': sizes, 'game': facts,
               'health': {'unavailable_games': [item['code'] for item in health['unavailable_games']],
                          'unavailable_dates': len(health['unavailable_dates']), 'feed_failing_games': len(health['feed_failing_games'])},
               'rss': rss()}
        cycles.append(row)
        with open(run_dir/'CYCLES.jsonl', 'a') as handle:
            handle.write(json.dumps(row, ensure_ascii=False)+'\n')
        guard(f'cycle {row["cycle"]}')
        return row

    try:
        guard('start')
        if policy_factory:
            armb.start()
            waited = perf()
            while armb.loading and perf()-waited < limits['policy_load_cap_s']:
                time.sleep(.05)
                guard('policy load')
            policy |= {'load_s': round(perf()-waited, 3), 'status': armb.status(), 'rss_after_load': rss()}
            if armb.policy is None:
                raise Stop('policy_unavailable', {'state': armb.status()['state']})
        while not transport.exhausted():
            one_cycle('replay')
            clock.advance(interval)
        for _ in range(limits['tail_cycles']):  # the capture is over: the last body keeps being the answer
            one_cycle('tail')
            clock.advance(interval)
        if config['settled_recheck_s'] is not None:  # one settled recheck: a final file must keep its bytes
            clock.advance(config['settled_recheck_s'])
            one_cycle('settled_recheck')
        status = 'COMPLETED'
    except Stop as stopped:
        status, stop = 'STOPPED', {'reason': stopped.reason, 'detail': stopped.detail}
    finally:
        pub.close()
    held = {name: not any(v['invariant'] == name for v in audit.violations) for name in publish_audit.INVARIANTS}
    held['nothing_outside_the_job_folder'] = None if fs_audit is None else not fs_audit.violations
    final_wait = None
    if final_published is not None and info.get('final_fetched') is not None:
        final_wait = round(final_published['simulated']-info['final_fetched'], 3)
    summary = {
        'schema': 'pitcheezy-publisher-replay-summary-v1', 'status': status, 'stop': stop, 'game_pk': game_pk, 'commit': commit,
        'data_role': 'SERVICE_EXPOSED_DEV', 'claim': 'one captured game replayed offline through the publisher; descriptive only',
        'config': {key: limits[key] for key in CONFIG_KEYS} | {'settled_recheck_s': config['settled_recheck_s'],
                                                              'contract_version': config['contract_version']},
        'capture': {'versions': info['versions'], 'first_fetched': info['first_fetched'], 'last_fetched': info['last_fetched'],
                    'span_s': round(info['last_fetched']-info['first_fetched'], 3), 'bodies_loaded': info['loads'],
                    'versions_never_served': info['versions']-(transport.highest+1) if status == 'COMPLETED' else None,
                    'upstream_requests_answered': transport.requests},
        'cycles': {'n': len(cycles), 'by_label': dict(Counter(row['label'] for row in cycles)),
                   'wall_s': spread([row['wall_s'] for row in cycles]), 'settle_s': spread([row['settle_s'] for row in cycles]),
                   'files_written': spread([row['files_written'] for row in cycles]),
                   'cycles_where_compute_exceeded_the_publish_interval': over_interval,
                   'last_sequence': cycles[-1]['sequence'] if cycles else None},
        'bytes_by_kind': bytes_by_kind,
        'game_file': {'display_states': dict(states), 'recommendation_by_cycle': dict(recs),
                      'unconfirmed_pitches_per_cycle': spread([row['game']['unconfirmed'] for row in cycles if row['game']]),
                      'cycles_with_an_unconfirmed_pitch': sum(1 for row in cycles if row['game'] and row['game']['unconfirmed']),
                      'pitches_in_last_file': next((row['game']['pitches'] for row in reversed(cycles) if row['game']), None),
                      'paired_past_recommendations_in_last_file': next((row['game']['paired_past_recommendations'] for row in reversed(cycles) if row['game']), None)},
        'health_codes': dict(codes),
        'ledger': {'rows': len(store.served_recommendations(game_pk)), 'first_rows': len(ledger_ids())},
        'final': {'feed_final_first_served_version_fetched': info.get('final_fetched'), 'publish_final': final_published,
                  'simulated_seconds_from_feed_final_to_publish_final': final_wait,
                  'note': 'simulated capture clock; resolution is one publish interval. 미측정 if either side is null.'},
        'schema_validation': {'validated': schemas is not None, 'kinds_seen': dict(kinds_seen),
                              'kinds_never_seen': sorted(set(schemas or {})-set(kinds_seen)),
                              'problems': [{'kind': kind, 'pointer': pointer, 'keyword': keyword, 'seen': seen, 'count': count}
                                           for (kind, pointer, keyword, seen), count in sorted(schema_problems.items())]},
        'invariants': {'held': held, 'violations': audit.violations[:200], 'violation_count': len(audit.violations),
                       'files_checked': audit.checked},
        'fs_audit': None if fs_audit is None else {'violations': fs_audit.violations[:200], 'violation_count': len(fs_audit.violations),
                                                   'files_read_under_watch': dict(sorted(fs_audit.read.items()))},
        'policy': policy | {'final_status': armb.status(), 'binds': getattr(armb.policy, 'binds', None)},
        'memory': {'peak_rss_bytes': rss(), 'source': 'resource.getrusage(RUSAGE_SELF).ru_maxrss (process peak, bytes on macOS)'},
        'wall_seconds_total': round(perf()-started, 3),
        'not_measured': ['live polling against the real upstream', 'hosting and upload', 'any other game',
                         'provider versions between the capture polls', 'stale or duplicate responses (the capture stores none)',
                         'end-to-end delay to a viewer'],
    }
    (run_dir/'SUMMARY.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1)+'\n')
    (run_dir/'SUMMARY.md').write_text(summary_text(summary))
    log(f"{status}: {len(cycles)} cycles, summary at {run_dir/'SUMMARY.json'}")
    return summary


def summary_text(s):
    """Short Korean summary. Every number is copied from the summary; a missing one is written 미측정."""
    show = lambda value: '미측정' if value is None else value
    lines = [f"# 발행기 오프라인 재생 요약 — 경기 {s['game_pk']}", '',
             f"- 상태: **{s['status']}**" + (f" (중단 사유: {s['stop']['reason']})" if s['stop'] else ''),
             f"- 코드 커밋: `{s['commit']}` / 자료 역할: {s['data_role']}",
             f"- 주기 수: {s['cycles']['n']} ({s['cycles']['by_label']}), 마지막 순번: {show(s['cycles']['last_sequence'])}",
             f"- 주기 실제 소요(초): 중앙값 {show(s['cycles']['wall_s'].get('p50'))}, p99 {show(s['cycles']['wall_s'].get('p99'))}, 최대 {show(s['cycles']['wall_s'].get('max'))}",
             f"- 최대 RSS(바이트): {s['memory']['peak_rss_bytes']}",
             f"- 종료 표시까지(재생 시계 초): {show(s['final']['simulated_seconds_from_feed_final_to_publish_final'])}",
             f"- 추천 원장 행: {s['ledger']['rows']}", '', '## 불변 조건', '']
    lines += [f"- {name}: {'지켜짐' if ok else '미측정' if ok is None else '**깨짐**'}" for name, ok in s['invariants']['held'].items()]
    lines += ['', '## 파일 크기(바이트, 종류별 최대 raw / gzip)', '']
    lines += [f"- {kind}: {v['raw_max']} / {v['gzip_max']} (쓴 횟수 {v['writes']})" for kind, v in sorted(s['bytes_by_kind'].items())]
    lines += ['', '## 스키마', '',
              f"- 검증함: {s['schema_validation']['validated']}, 본 종류: {s['schema_validation']['kinds_seen']}, 못 본 종류: {s['schema_validation']['kinds_never_seen']}",
              f"- 어긋난 자리 {len(s['schema_validation']['problems'])}곳 (자리·본 타입은 SUMMARY.json)", '',
              '## 말하지 않는 것', ''] + [f'- {item}' for item in s['not_measured']]
    return '\n'.join(lines)+'\n'


def armb_factory(run_dir, style_dir=STYLE_DIR):
    """The frozen ARM-B live policy with every write redirected into the run folder: the pinned 2026 style
    snapshot is COPIED from T7 into ``<run>/policy-root/style`` (so the policy reads its cache there and never
    writes under T7), and the per-game scratch ledger goes to ``<run>/tmp``. T7 is only read."""
    def factory():
        import tempfile
        import demo_precompute
        from observer_app.live_feed import LIVE_CONFIG
        tempfile.tempdir = str(Path(run_dir)/'tmp')
        target = Path(run_dir)/'policy-root/style'
        target.mkdir(parents=True, exist_ok=True)
        index = json.loads((Path(style_dir)/'style_2026_index.json').read_text())
        for name in ('style_2026_index.json', index['file']):
            (target/name).write_bytes((Path(style_dir)/name).read_bytes())
        return demo_precompute.LivePolicy(output_root=Path(run_dir)/'policy-root', max_games=LIVE_CONFIG.get('max_bound_games', 2))
    return factory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--job-dir', required=True)
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--game', type=int, required=True)
    parser.add_argument('--capture-db', required=True)
    parser.add_argument('--capture-sha256', required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--config', required=True, help='replay_config.json (limits and pace)')
    parser.add_argument('--policy', choices=('armb', 'off'), required=True)
    parser.add_argument('--allow-dirty', action='store_true', help='synthetic tests only')
    args = parser.parse_args(argv)
    job, db = Path(args.job_dir).resolve(), Path(args.capture_db).resolve()
    run_dir = job/args.run_name
    try:
        limits = json.loads(Path(args.config).read_text())
        missing = [key for key in CONFIG_KEYS if key not in limits]
        if missing:
            raise Refused(f'replay config lacks {missing}')
        if limits['fixed_delay_s'] != 0:
            raise Refused('this rehearsal is registered for fixed delay 0 only')
        if not job.is_dir() or re.fullmatch(r'[A-Za-z0-9._-]+', args.run_name) is None:
            raise Refused('the job folder must exist and the run name must be a plain name')
        if run_dir.exists():
            raise Refused(f'{run_dir} already exists: one run, one new folder')
        if not db.is_relative_to(job) or not db.is_file():
            raise Refused('the capture database must be a COPY inside the job folder')
        if any(Path(str(db)+suffix).exists() and Path(str(db)+suffix).stat().st_size for suffix in ('-wal',)):
            raise Refused('the capture copy has a non-empty WAL')
        before = sha256_file(db)
        if before != args.capture_sha256:
            raise Refused('the capture copy does not have the registered sha256')
        if head_commit() != args.commit:
            raise Refused(f'the code is not at the registered commit ({head_commit()})')
        changed = tracked_changes()
        if changed is None or (changed and not args.allow_dirty):
            raise Refused(f'tracked files differ from the commit: {changed}')
        free = free_memory_bytes()
        if free is None or free < limits['min_free_bytes']:
            raise Refused(f'free memory {free} is below the registered minimum {limits["min_free_bytes"]}')
        if args.policy == 'armb' and not (T7.is_mount() and (STYLE_DIR/'style_2026_index.json').is_file()):
            raise Refused('the policy needs the T7 volume (read-only use) and its pinned style snapshot')
    except Refused as refused:
        print(f'REFUSED: {refused}', file=sys.stderr)
        return 2
    for name in CLEARED_ENV:
        os.environ.pop(name, None)
    run_dir.mkdir()
    os.environ['TMPDIR'] = str(run_dir/'tmp')
    from observer_app import publish_audit
    fs_audit = publish_audit.FsAudit([run_dir], watch=[T7])
    sys.addaudithook(fs_audit.event)  # from here on: no write outside the run folder, no process, no socket

    def watchdog():  # a hard stop of THIS process only, if the main loop cannot get to its own checks
        started = time.monotonic()
        while True:
            time.sleep(.5)
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if peak > limits['max_rss_bytes'] or time.monotonic()-started > limits['max_seconds']+60:
                (run_dir/'ABORTED.json').write_text(json.dumps({'peak_rss_bytes': peak, 'elapsed_s': round(time.monotonic()-started, 1)}))
                os._exit(87)
    threading.Thread(target=watchdog, name='replay-watchdog', daemon=True).start()
    rows, info = capture_rows(db, args.game, limits['max_body_bytes'])
    contract = REPO/'apps/observer/publish_contract/v1/schema'
    schemas = {path.name.split('.')[0]: json.loads(path.read_text()) for path in sorted(contract.glob('*.schema.json'))}
    (run_dir/'RUN.json').write_text(json.dumps({
        'argv': sys.argv if argv is None else list(argv), 'commit': args.commit, 'capture_sha256_before': before,
        'free_memory_bytes_at_start': free, 'limits': limits, 'policy': args.policy, 'python': sys.version,
        'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}, indent=1)+'\n')
    summary = run(rows=rows, info=info, run_dir=run_dir, game_pk=args.game, commit=args.commit, limits=limits,
                  policy_factory=armb_factory(run_dir) if args.policy == 'armb' else None, fs_audit=fs_audit, schemas=schemas)
    with closing(info['db']):
        pass
    after = sha256_file(db)
    summary['capture_copy'] = {'sha256_before': before, 'sha256_after': after, 'unchanged': before == after}
    (run_dir/'SUMMARY.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1)+'\n')
    return 0 if summary['status'] == 'COMPLETED' and before == after else 3


if __name__ == '__main__':
    sys.exit(main())
