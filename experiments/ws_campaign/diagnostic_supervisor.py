"""Bounded Linux owned-child monitor with inherited GPU lock lifetime."""
import json
import os
from pathlib import Path
import subprocess
import time

from experiments.ws_campaign.runtime import sha


def supervise(command, out, wall, rss_cap, grant_sha, lock_fd, output_cap=None):
    # The child retains this same flock open-file-description even if parent dies.
    started = time.monotonic()
    child = subprocess.Popen(command, pass_fds=(lock_fd,))
    reason = None; peak = 0; reaped = False
    try:
        while child.poll() is None:
            if output_cap is not None and sum(p.stat().st_size for p in out.iterdir() if p.is_file()) > output_cap:
                reason = 'output-bytes'; break
            if time.monotonic()-started >= wall:
                reason = 'wall'; break
            try:
                status = Path(f'/proc/{child.pid}/status').read_text()
                value = next((s.split()[1] for s in status.splitlines() if s.startswith('VmRSS:')), None)
                if value is None:
                    if child.poll() is not None: break
                    try: child.wait(timeout=.25); break
                    except subprocess.TimeoutExpired: reason = 'rss-unreadable'; break
                peak = max(peak,int(value)*1024)
                if peak > rss_cap: reason = 'rss'; break
            except (OSError,ValueError):
                if child.poll() is not None: break
                reason = 'rss-unreadable'; break
            time.sleep(.1)
    finally:
        if child.poll() is None:
            child.terminate()
            try: child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                try: child.wait(timeout=5)
                except subprocess.TimeoutExpired: reason = 'unreaped-child-lock-retained'
        reaped = child.poll() is not None
    out.mkdir(parents=True, exist_ok=True)
    if output_cap is not None and sum(p.stat().st_size for p in out.iterdir() if p.is_file()) > output_cap:
        reason = 'output-bytes'
    receipt = out/'receipt.json'
    try: completed = receipt.is_file() and json.loads(receipt.read_text()).get('status') == 'COMPLETED'
    except (OSError,ValueError): completed = False
    if child.returncode == 0 and reason is None and not completed: reason = 'missing-completed-receipt'
    result = {'status':'COMPLETED' if child.returncode == 0 and reason is None else 'INCOMPLETE',
              'returncode':child.returncode,'reason':reason,'peak_worker_rss_bytes':peak,
              'elapsed_seconds':time.monotonic()-started,'grant_sha256':grant_sha,
              'receipt_sha256':sha(receipt) if receipt.exists() else None,
              'child_reaped':reaped,'child_pid':child.pid,'inherited_lock_fd':lock_fd}
    (out/'SUPERVISOR.json').write_text(json.dumps(result,indent=2)+'\n')
    return 0 if result['status'] == 'COMPLETED' else 87
