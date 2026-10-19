#!/usr/bin/env python3
"""One-command test receipt for a WS final launch candidate (CL-10p).

Runs, on the checked-out commit and in this order, the same steps as the manual BLK-06 / BLK-06b receipts:

  web-build          npm run build                              (apps/observer/web)
  web-build-public   npm run build:public                       (apps/observer/web)
  web-static         node tests/static-publish-smoke.mjs <shots>
  web-viewer-delay   node tests/viewer-delay-smoke.mjs <shots>
  web-compact        node tests/compact-rev3-fit.mjs <shots>
  web-studio         node tests/studio-layout-check.mjs <shots>
  backend-ops        PYTHONPATH=apps/observer/backend <python> -m pytest apps/observer/backend/tests
                     apps/observer/ops/tests -q -p no:cacheprovider -rfE
  public-e2e         the same pytest on test_public_viewer_e2e.py + test_public_viewer_gateway_e2e.py

and writes ``receipt.md`` + ``receipt.json`` + one log per step to ``--out`` (must be outside the repo).

* Refuses a dirty tree, and refuses unless ``apps/observer/web/node_modules`` is a symlink (its target is recorded).
* The public candidate is explicit: ``--candidate <json>`` sets OBSERVER_E2E_CANDIDATE for the pytest steps;
  without it the test's DEFAULT_CANDIDATE is used. Inherited OBSERVER_E2E_CANDIDATE / PUBLIC_E2E_CANDIDATE are
  dropped so a stray shell variable cannot change the candidate silently.
* ``--known-failures <json>`` labels expected failures; they stay listed in the receipt (never hidden), they just do
  not make the exit code nonzero. Format: ``[{"step": "web-compact", "match": "<regex>", "reason": "...",
  "max": 4, "flaky": false}]`` (``max`` optional: matches beyond it count as unexpected; ``step`` must be a step
  id). Rules that matched nothing are listed too. When a pytest step's only failures match rules marked
  ``"flaky": true`` those tests are rerun alone and the rerun is attached (``--no-flaky-rerun`` skips it).
* pytest exit codes other than 0/1 (interrupted, internal error, usage error, no tests) are always unexpected.
* ``--steps`` runs a subset; the receipt then records the subset and its verdict starts with ``PARTIAL_``.
* Exit 0: every failure is known. Exit 1: an unexpected failure (or a step that failed without a parsable failure
  name). Exit 2: refused before running anything.

No network of its own; the steps use loopback / route-mocked pages only (see each test). It starts no publisher,
inspects no other process and sends no signals.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

KST = dt.timezone(dt.timedelta(hours=9))
WEB = 'apps/observer/web'
TOLERATED_UNTRACKED = {f'{WEB}/node_modules'}
DROPPED_ENV = ('OBSERVER_E2E_CANDIDATE', 'PUBLIC_E2E_CANDIDATE')
SCHEMA = 'pitcheezy-final-candidate-receipt-v1'


@dataclass
class Step:
    id: str
    kind: str            # 'build' | 'node' | 'pytest'
    cwd: str             # relative to repo root
    argv: list[str]
    env: dict[str, str] = field(default_factory=dict)


def steps(python: str, shots: Path) -> list[Step]:
    pytest = [python, '-m', 'pytest']
    tail = ['-q', '-p', 'no:cacheprovider', '-rfE']
    py_env = {'PYTHONPATH': 'apps/observer/backend'}
    node = lambda name, stem: Step(name, 'node', WEB, ['node', f'tests/{stem}.mjs', str(shots / stem)])  # noqa: E731
    return [
        Step('web-build', 'build', WEB, ['npm', 'run', 'build']),
        Step('web-build-public', 'build', WEB, ['npm', 'run', 'build:public']),
        node('web-static', 'static-publish-smoke'),
        node('web-viewer-delay', 'viewer-delay-smoke'),
        node('web-compact', 'compact-rev3-fit'),
        node('web-studio', 'studio-layout-check'),
        Step('backend-ops', 'pytest', '.',
             pytest + ['apps/observer/backend/tests', 'apps/observer/ops/tests'] + tail, py_env),
        Step('public-e2e', 'pytest', '.',
             pytest + ['apps/observer/backend/tests/test_public_viewer_e2e.py',
                       'apps/observer/backend/tests/test_public_viewer_gateway_e2e.py'] + tail, py_env),
    ]


STEP_IDS = tuple(s.id for s in steps('python', Path('shots')))


# ---------------------------------------------------------------- parsers

_PYTEST_COUNT = re.compile(r'(\d+) (passed|failed|errors?|skipped|xfailed|xpassed|deselected|warnings?)\b')
_PYTEST_SUMMARY = re.compile(r'^=*\s*((?:\d+ \w+(?:, )?)+) in [\d.]+s')
_PYTEST_FAILED = re.compile(r'^(FAILED|ERROR) (.+?)(?: - .*)?$')


def parse_pytest(text: str) -> dict:
    counts: dict[str, int] = {}
    for line in text.splitlines():
        m = _PYTEST_SUMMARY.match(line.strip())
        if m:  # keep the last summary line
            counts = {}
            for n, word in _PYTEST_COUNT.findall(m.group(1)):
                key = {'error': 'errors', 'warning': 'warnings'}.get(word, word)
                counts[key] = counts.get(key, 0) + int(n)
    failures = []
    for line in text.splitlines():
        m = _PYTEST_FAILED.match(line.rstrip())
        if m and m.group(2).startswith(('apps/', 'tests/', 'scripts/')):
            failures.append(m.group(2).strip())
    return {'summary_found': bool(counts),
            'passed': counts.get('passed', 0),
            'failed': counts.get('failed', 0) + counts.get('errors', 0),
            'skipped': counts.get('skipped', 0),
            'xfailed': counts.get('xfailed', 0),
            'xpassed': counts.get('xpassed', 0),
            'failures': failures}


_NODE_OK = re.compile(r'^ok\s+\S')
_NODE_FAIL = re.compile(r'^FAIL\s+(.*?)(?:\s+[\[{].*)?\s*$')
_NODE_SUMMARIES = [
    (re.compile(r'^(\d+) passed, (\d+) failed\s*$'), lambda m: (int(m[1]), int(m[2]))),   # static-publish-smoke
    (re.compile(r'^(\d+) ok, (\d+) FAIL\s*$'), lambda m: (int(m[1]), int(m[2]))),         # studio-layout-check
    (re.compile(r'^(\d+) FAIL\s*$'), lambda m: (None, int(m[1]))),                         # compact-rev3-fit
    (re.compile(r'^(\d+) FAILED\s*$'), lambda m: (None, int(m[1]))),                       # viewer-delay-smoke
    (re.compile(r'^all (?:ok|passed)\s*$'), lambda m: (None, 0)),                          # compact / viewer-delay
]


def parse_node(text: str) -> dict:
    lines = text.splitlines()
    ok_lines = sum(1 for line in lines if _NODE_OK.match(line))
    failures = [m.group(1) for line in lines if (m := _NODE_FAIL.match(line))]
    passed = failed = None
    found = False
    for line in lines:
        for rx, get in _NODE_SUMMARIES:
            m = rx.match(line.strip())
            if m:  # keep the last summary line
                found = True
                passed, failed = get(m)
    if passed is None:
        passed = ok_lines
    if failed is None:
        failed = len(failures)
    notes = []
    if failed != len(failures):
        notes.append(f'summary says {failed} failed but {len(failures)} FAIL lines were printed')
    if ok_lines and ok_lines != passed:
        notes.append(f'summary says {passed} passed but {ok_lines} ok lines were printed')
    return {'summary_found': found, 'passed': passed, 'failed': failed, 'skipped': 0,
            'failures': failures, 'notes': notes}


def parse(kind: str, text: str) -> dict:
    if kind == 'pytest':
        return parse_pytest(text)
    if kind == 'node':
        return parse_node(text)
    return {'summary_found': True, 'passed': 0, 'failed': 0, 'skipped': 0, 'failures': []}


# ---------------------------------------------------------------- known failures

def load_known(path: Path | None) -> list[dict]:
    if path is None:
        return []
    rules = json.loads(path.read_text())
    if not isinstance(rules, list):
        raise ValueError('known-failures file must be a JSON list')
    for i, rule in enumerate(rules):
        if not isinstance(rule, dict) or not {'step', 'match', 'reason'} <= rule.keys():
            raise ValueError(f'known-failures entry {i} needs step, match and reason')
        extra = set(rule) - {'step', 'match', 'reason', 'max', 'flaky'}
        if extra:
            raise ValueError(f'known-failures entry {i} has unknown keys {sorted(extra)}')
        if rule['step'] not in STEP_IDS:
            raise ValueError(f'known-failures entry {i}: unknown step {rule["step"]!r} (one of {", ".join(STEP_IDS)})')
        if not isinstance(rule['match'], str) or not rule['match'].strip() or not isinstance(rule['reason'], str) \
                or not rule['reason'].strip():
            raise ValueError(f'known-failures entry {i}: match and reason must be non-empty strings')
        if 'max' in rule and (type(rule['max']) is not int or rule['max'] < 1):
            raise ValueError(f'known-failures entry {i}: max must be an integer >= 1')
        if 'flaky' in rule and type(rule['flaky']) is not bool:
            raise ValueError(f'known-failures entry {i}: flaky must be true or false')
        re.compile(rule['match'])
    return rules


def classify(step_id: str, result: dict, exit_code: int, known: list[dict]) -> dict:
    """Split failures into known / unexpected. A nonzero exit with no parsed failure is itself unexpected."""
    rules = [r for r in known if r['step'] == step_id]
    used = [0] * len(rules)
    labelled, unexpected = [], []
    for name in result['failures']:
        hit = next((i for i, r in enumerate(rules) if re.search(r['match'], name)
                    and used[i] < r.get('max', float('inf'))), None)
        if hit is None:
            over = any(re.search(r['match'], name) for r in rules)
            unexpected.append({'name': name, 'reason': 'matches a known rule beyond its max' if over else None})
        else:
            used[hit] += 1
            labelled.append({'name': name, 'reason': rules[hit]['reason'], 'flaky': rules[hit].get('flaky', False)})
    if result['failed'] > len(result['failures']):
        unexpected.append({'name': f'{result["failed"] - len(result["failures"])} failure(s) without a parsable name',
                           'reason': None})
    if exit_code != 0 and not result['failures'] and not result['failed']:
        unexpected.append({'name': f'exit code {exit_code} without a parsed failure', 'reason': None})
    if not result.get('summary_found', True):
        unexpected.append({'name': 'no summary line in the output', 'reason': None})
    if result.get('kind') == 'pytest' and exit_code not in (0, 1):
        unexpected.append({'name': f'pytest exit code {exit_code} (interrupted / internal / usage error / no tests)',
                           'reason': None})
    unseen = [r for i, r in enumerate(rules) if used[i] == 0]
    return {'known': labelled, 'unexpected': unexpected,
            'known_not_seen': [{'match': r['match'], 'reason': r['reason']} for r in unseen]}


# ---------------------------------------------------------------- repo checks

def git(repo: Path, *args: str) -> str:
    return subprocess.run(['git', '-C', str(repo), *args], check=True, capture_output=True, text=True).stdout


def dirty_entries(repo: Path) -> list[str]:
    """Porcelain entries that make the tree dirty; a symlinked web node_modules is the one tolerated untracked entry."""
    bad = []
    for line in git(repo, 'status', '--porcelain', '--untracked-files=all').splitlines():
        status, path = line[:2], line[3:]
        if status == '??' and path.rstrip('/') in TOLERATED_UNTRACKED and (repo / path.rstrip('/')).is_symlink():
            continue
        bad.append(line)
    return bad


def is_inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_members(directory: Path) -> dict:
    if not directory.is_dir():
        return {}
    return {str(p.relative_to(directory)): {'bytes': p.stat().st_size, 'sha256': sha256(p)}
            for p in sorted(directory.rglob('*')) if p.is_file()}


def now() -> dt.datetime:
    return dt.datetime.now(KST)


def tool_version(argv: list[str]) -> str:
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return 'unavailable'


# ---------------------------------------------------------------- run

def run_step(step: Step, repo: Path, log_dir: Path, base_env: dict[str, str]) -> dict:
    env = {**base_env, **step.env}
    log = log_dir / f'{step.id}.log'
    start = now()
    with log.open('w') as handle:
        handle.write(f'start {start.isoformat(timespec="seconds")}\ncwd {step.cwd}\ncmd {" ".join(step.argv)}\n')
        handle.flush()
        proc = subprocess.run(step.argv, cwd=repo / step.cwd, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, errors='replace')
        handle.write(proc.stdout)
        end = now()
        handle.write(f'\nexit {proc.returncode}\nend {end.isoformat(timespec="seconds")}\n')
    result = parse(step.kind, proc.stdout)
    return {'id': step.id, 'kind': step.kind, 'cwd': step.cwd, 'command': step.argv,
            'env_set': step.env, 'start': start.isoformat(timespec='seconds'),
            'end': end.isoformat(timespec='seconds'), 'seconds': round((end - start).total_seconds(), 1),
            'exit_code': proc.returncode, 'log': log.name, **result}


def flaky_rerun(step: Step, result: dict, repo: Path, log_dir: Path, base_env: dict[str, str]) -> dict | None:
    """When a pytest step's only failures are rules marked flaky, rerun exactly those tests alone (runbook: confirm a
    flaky failure by a solo rerun). The original failure stays listed as known; a failing rerun is unexpected."""
    if step.kind != 'pytest' or result['unexpected'] or not result['known']:
        return None
    if not all(item['flaky'] for item in result['known']):
        return None
    ids = [item['name'] for item in result['known']]
    rerun = Step(f'{step.id}-flaky-rerun', 'pytest', step.cwd,
                 step.argv[:3] + ids + ['-q', '-p', 'no:cacheprovider', '-rfE'], step.env)
    out = run_step(rerun, repo, log_dir, base_env)
    if out['exit_code'] != 0 or out['failed'] or out['passed'] != len(ids):
        result['unexpected'].append({'name': f'flaky rerun alone did not pass ({out["passed"]} passed, '
                                             f'{out["failed"]} failed, exit {out["exit_code"]})', 'reason': None})
    return {k: out[k] for k in ('command', 'start', 'end', 'exit_code', 'passed', 'failed', 'failures', 'log')}


def markdown(receipt: dict) -> str:
    out = [f'# 최종 후보 테스트 영수증 — {receipt["commit"][:8]}', '',
           f'- 커밋: `{receipt["commit"]}` ({receipt["branch"]})',
           f'- 작업트리: `{receipt["repo"]}`',
           f'- 시각(KST): {receipt["start"]} – {receipt["end"]}',
           f'- 공개 후보: {receipt["candidate"]["source"]}'
           + (f' (`{receipt["candidate"]["path"]}`, sha256 `{receipt["candidate"]["sha256"]}`)'
              if receipt['candidate']['path'] else ''),
           f'- web node_modules: symlink → `{receipt["node_modules_link"]}`',
           f'- 실행 단계: {"전체" if not receipt["partial"] else "일부(PARTIAL) " + ", ".join(receipt["steps_requested"])}',
           f'- 알려진 실패 목록: {receipt["known_failures_file"] or "없음"}'
           + (f' (sha256 `{receipt["known_failures_sha256"]}`)' if receipt['known_failures_sha256'] else ''),
           f'- 도구: python `{receipt["tools"]["python"]}`, node `{receipt["tools"]["node"]}`, '
           f'npm `{receipt["tools"]["npm"]}`',
           f'- **판정: {receipt["verdict"]}** (예상 밖 실패 {receipt["totals"]["unexpected"]}건, '
           f'알려진 실패 {receipt["totals"]["known"]}건)', '', '## 단계', '']
    for s in receipt['steps']:
        counts = (f'{s["passed"]} passed / {s["failed"]} failed / {s["skipped"]} skipped'
                  + (f' / {s["xfailed"]} xfailed' if s.get('xfailed') else ''))
        out.append(f'- **{s["id"]}** — exit {s["exit_code"]}, {counts}, {s["seconds"]}s '
                   f'({s["start"][11:19]}–{s["end"][11:19]}), 로그 `logs/{s["log"]}`')
        out.append(f'  - 명령: `{"".join(f"{k}={v} " for k, v in s["env_set"].items())}{" ".join(s["command"])}`'
                   f' (cwd `{s["cwd"]}`)')
        for item in s['unexpected']:
            out.append(f'  - 예상 밖: `{item["name"]}`')
        for item in s['known']:
            out.append(f'  - 알려진 실패{" (불안정)" if item.get("flaky") else ""}: `{item["name"]}` — {item["reason"]}')
        if s.get('flaky_rerun'):
            r = s['flaky_rerun']
            out.append(f'  - 불안정 테스트 단독 재실행: exit {r["exit_code"]}, {r["passed"]} passed / {r["failed"]} failed, '
                       f'로그 `logs/{r["log"]}`')
        for item in s['known_not_seen']:
            out.append(f'  - 알려진 실패 중 이번에 안 나옴: `{item["match"]}` — {item["reason"]}')
        for note in s.get('notes', []):
            out.append(f'  - 참고: {note}')
    out += ['', '## dist-public 파일', '']
    for name, member in receipt['dist_public'].items():
        out.append(f'- `{name}` {member["bytes"]} B `{member["sha256"]}`')
    out += ['', '이 영수증은 `scripts/final_candidate_receipt.py`가 만들었다. 네트워크는 loopback/route-mock만, '
            'MLB 접속·발행기 실행·프로세스 신호 없음.', '']
    return '\n'.join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', required=True, type=Path, help='receipt directory, outside the repo')
    ap.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--candidate', type=Path, help='candidate JSON for OBSERVER_E2E_CANDIDATE (default: DEFAULT_CANDIDATE)')
    ap.add_argument('--known-failures', type=Path, help='JSON list of {step, match, reason}')
    ap.add_argument('--python', help='python for pytest (default: <main checkout>/.venv-observer/bin/python)')
    ap.add_argument('--steps', help='comma-separated subset of step ids (default: all); the receipt says PARTIAL')
    ap.add_argument('--no-flaky-rerun', action='store_true',
                    help='do not rerun alone the tests matched by rules marked "flaky"')
    args = ap.parse_args(argv)

    repo = args.repo.resolve()
    out = args.out.resolve()
    if is_inside(out, repo):
        print(f'REFUSED: --out {out} is inside the repo {repo}', file=sys.stderr)
        return 2
    dirty = dirty_entries(repo)
    if dirty:
        print('REFUSED: dirty tree (commit or remove these first):\n  ' + '\n  '.join(dirty), file=sys.stderr)
        return 2
    node_modules = repo / WEB / 'node_modules'
    if not node_modules.is_symlink() or not node_modules.exists():
        print(f'REFUSED: {WEB}/node_modules must be a symlink to an existing install (no npm install here)',
              file=sys.stderr)
        return 2
    node_modules_link = str(node_modules.resolve())
    if out.exists() and any(out.iterdir()):
        print(f'REFUSED: --out {out} is not empty', file=sys.stderr)
        return 2
    try:
        known = load_known(args.known_failures)
    except (OSError, ValueError, re.error) as exc:
        print(f'REFUSED: known-failures file: {exc}', file=sys.stderr)
        return 2
    candidate = {'source': 'DEFAULT_CANDIDATE (test_public_viewer_e2e.py)', 'path': None, 'sha256': None}
    if args.candidate:
        if not args.candidate.is_file():
            print(f'REFUSED: candidate {args.candidate} not found', file=sys.stderr)
            return 2
        candidate = {'source': 'OBSERVER_E2E_CANDIDATE', 'path': str(args.candidate.resolve()),
                     'sha256': sha256(args.candidate)}
    common = Path(git(repo, 'rev-parse', '--path-format=absolute', '--git-common-dir').strip())
    python = args.python or str(common.parent / '.venv-observer' / 'bin' / 'python')
    plan = steps(python, out / 'shots')
    if args.steps:
        wanted = args.steps.split(',')
        unknown = set(wanted) - {s.id for s in plan}
        if unknown:
            print(f'REFUSED: unknown steps {sorted(unknown)}', file=sys.stderr)
            return 2
        plan = [s for s in plan if s.id in wanted]

    commit = git(repo, 'rev-parse', 'HEAD').strip()
    branch = git(repo, 'rev-parse', '--abbrev-ref', 'HEAD').strip()
    base_env = {k: v for k, v in os.environ.items() if k not in DROPPED_ENV}
    if candidate['path']:
        base_env['OBSERVER_E2E_CANDIDATE'] = candidate['path']
    (out / 'logs').mkdir(parents=True, exist_ok=True)
    start = now()
    results = []
    for step in plan:
        print(f'[{now():%H:%M:%S}] {step.id}: {" ".join(step.argv)}', flush=True)
        result = run_step(step, repo, out / 'logs', base_env)
        result.update(classify(step.id, result, result['exit_code'], known))
        if not args.no_flaky_rerun:
            rerun = flaky_rerun(step, result, repo, out / 'logs', base_env)
            if rerun:
                result['flaky_rerun'] = rerun
        print(f'[{now():%H:%M:%S}] {step.id}: exit {result["exit_code"]}, {result["passed"]} passed, '
              f'{result["failed"]} failed, unexpected {len(result["unexpected"])}', flush=True)
        results.append(result)
    end = now()
    if git(repo, 'rev-parse', 'HEAD').strip() != commit or dirty_entries(repo):
        results.append({'id': 'tree-check', 'kind': 'check', 'cwd': '.', 'command': ['git', 'status'], 'env_set': {},
                        'start': end.isoformat(timespec='seconds'), 'end': end.isoformat(timespec='seconds'),
                        'seconds': 0, 'exit_code': 1, 'log': '-', 'passed': 0, 'failed': 1, 'skipped': 0,
                        'failures': ['tree changed during the run'], 'known': [], 'known_not_seen': [],
                        'unexpected': [{'name': 'HEAD moved or tracked/untracked files changed during the run',
                                        'reason': None}]})
    unexpected = sum(len(r['unexpected']) for r in results)
    receipt = {
        'schema': SCHEMA, 'commit': commit, 'branch': branch, 'repo': str(repo),
        'start': start.isoformat(timespec='seconds'), 'end': end.isoformat(timespec='seconds'),
        'candidate': candidate, 'node_modules_link': node_modules_link,
        'partial': bool(args.steps), 'steps_requested': [s.id for s in plan],
        'known_failures_file': str(args.known_failures.resolve()) if args.known_failures else None,
        'known_failures_sha256': sha256(args.known_failures) if args.known_failures else None,
        'tools': {'python': tool_version([python, '--version']), 'node': tool_version(['node', '--version']),
                  'npm': tool_version(['npm', '--version'])},
        'steps': results,
        'dist_public': build_members(repo / WEB / 'dist-public'),
        'totals': {'unexpected': unexpected, 'known': sum(len(r['known']) for r in results)},
        'verdict': 'PASS_ONLY_KNOWN_FAILURES' if unexpected == 0 else 'FAIL_UNEXPECTED',
    }
    if unexpected == 0 and receipt['totals']['known'] == 0:
        receipt['verdict'] = 'PASS'
    if receipt['partial']:
        receipt['verdict'] = 'PARTIAL_' + receipt['verdict']
    (out / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    (out / 'receipt.md').write_text(markdown(receipt))
    print(f'{receipt["verdict"]}: unexpected {unexpected}, known {receipt["totals"]["known"]} -> {out}/receipt.md')
    return 0 if unexpected == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
