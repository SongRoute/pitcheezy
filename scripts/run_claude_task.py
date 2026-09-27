#!/usr/bin/env python3
"""Run one bounded Claude CLI task with durable input, events and completion state."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def run(args):
    worktree = args.worktree.resolve(strict=True)
    request = args.request.read_bytes()
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=worktree, text=True).strip()
    job = args.job_dir.resolve()
    job.mkdir(parents=True, exist_ok=False)
    (job / 'request.md').write_bytes(request)
    command = [args.claude_bin, '-p', '--model', args.model, '--effort', args.effort,
               '--output-format', 'stream-json', '--verbose', '--max-turns', str(args.max_turns)]
    if args.skip_permissions:
        command.append('--dangerously-skip-permissions')
    if args.disable_hooks:
        command.extend(['--settings', json.dumps({'disableAllHooks': True})])
    if args.resume:
        command.extend(['--resume', args.resume])
    state = {'status': 'starting', 'started_utc': utc(), 'worktree': str(worktree),
             'base_commit': head, 'request_sha256': hashlib.sha256(request).hexdigest(),
             'command': command, 'timeout_seconds': args.timeout_seconds,
             'supervisor_pid': os.getpid(), 'review_status': 'pending'}
    write_json(job / 'state.json', state)
    started = time.monotonic()
    process = None
    interrupted = False
    previous_handlers = {}

    def stop(_signum, _frame):
        nonlocal interrupted
        interrupted = True
        raise KeyboardInterrupt

    for sig in (signal.SIGTERM, signal.SIGINT):
        previous_handlers[sig] = signal.signal(sig, stop)
    try:
        with (job / 'request.md').open('rb') as stdin, (job / 'events.jsonl').open('wb') as stdout, (job / 'stderr.log').open('wb') as stderr:
            process = subprocess.Popen(command, cwd=worktree, stdin=stdin, stdout=stdout,
                                       stderr=stderr, start_new_session=True)
            state.update(status='running', child_pid=process.pid)
            write_json(job / 'state.json', state)
            try:
                exit_code = process.wait(timeout=args.timeout_seconds)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                state['status'] = 'interrupted' if interrupted else 'timed_out'
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    exit_code = process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    exit_code = process.wait()
        result = None
        sessions = []
        for line in (job / 'events.jsonl').open():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get('session_id') and event['session_id'] not in sessions:
                sessions.append(event['session_id'])
            if event.get('type') == 'result':
                result = event
        state.update(exit_code=exit_code, session_ids=sessions)
        if result is not None:
            write_json(job / 'result.json', result)
            state['result_subtype'] = result.get('subtype')
            state['permission_denials'] = result.get('permission_denials', [])
        if state['status'] == 'running':
            success = exit_code == 0 and result is not None and not result.get('is_error', False)
            success = success and result.get('subtype') == 'success' and not result.get('permission_denials')
            state['status'] = 'returned_for_review' if success else 'failed'
        state['final_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=worktree, text=True).strip()
        state['git_status'] = subprocess.check_output(['git', 'status', '--porcelain=v1'], cwd=worktree, text=True)
    except Exception as exc:
        state.update(status='supervisor_failed', error=f'{type(exc).__name__}: {exc}')
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        state.update(finished_utc=utc(), wall_seconds=round(time.monotonic() - started, 3))
        write_json(job / 'state.json', state)
    print(json.dumps({'job_dir': str(job), 'status': state['status'], 'session_ids': state.get('session_ids', [])}))
    return 0 if state['status'] == 'returned_for_review' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--worktree', type=Path, required=True)
    parser.add_argument('--job-dir', type=Path, required=True)
    parser.add_argument('--model', default='sonnet')
    parser.add_argument('--effort', choices=['low', 'medium', 'high', 'xhigh', 'max'], default='high')
    parser.add_argument('--max-turns', type=int, default=60)
    parser.add_argument('--timeout-seconds', type=float, default=1800)
    parser.add_argument('--resume')
    parser.add_argument('--claude-bin', default='claude')
    parser.add_argument('--skip-permissions', action='store_true', help='User-authorized Claude permission bypass, recorded in state.json.')
    parser.add_argument('--disable-hooks', action='store_true', help='Disable hooks for this CLI call only; run required checks explicitly.')
    arguments = parser.parse_args()
    if arguments.max_turns <= 0 or arguments.timeout_seconds <= 0:
        parser.error('turn and time limits must be positive')
    raise SystemExit(run(arguments))
