"""One bounded local quality child, reusing accepted metadata supervisor helpers."""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.ws_fullseason.audit import gates, sha


def supervise(regpath, grantpath, worker="audit.py"):
    reg, _ = gates(regpath, grantpath)
    worker_path = ROOT / 'scripts' / worker if worker in ('fullseason_shadow_parity.py', 'fullseason_profile_export.py', 'fullseason_captured_replay.py') else Path(__file__).with_name(worker)
    if worker not in ('audit.py', 'prepare_worker.py', 'calibrate_worker.py', 'final_worker.py', 'fullseason_shadow_parity.py', 'fullseason_profile_export.py', 'fullseason_captured_replay.py') or str(worker_path) not in reg['source_pins']:
        raise ValueError('unpinned worker')
    spec = importlib.util.spec_from_file_location('accepted_metadata_supervisor', reg['supervisor_helper'])
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    bounds = reg['bounds']; out = Path(reg['output']); fence = Path(reg['fence'])
    if out.exists() or fence.exists():
        raise ValueError('fresh output and absent fence required')
    child = None; identity = None; reaped = True; reason = None; peak = 0
    with Path(reg['heavy_lock']).open('rb') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        free = helper.free_bytes()
        if free is None or free < bounds['free_memory_bytes']:
            raise ValueError('memory headroom')
        if shutil.disk_usage(out.parent).free < bounds['free_disk_bytes']:
            raise ValueError('disk headroom')
        out.mkdir()
        with fence.open('x') as f:
            json.dump({'registration_sha256': sha(regpath), 'state': 'PREPARING'}, f)
        start = time.monotonic()
        try:
            env = dict(os.environ, PITCHEEZY_FULLSEASON_LOCK_FD=str(lock.fileno()),
                       OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
            command = [sys.executable, '-I', str(worker_path),
                       '--registration', str(regpath), '--grant', str(grantpath)]
            with (out / 'worker.log').open('x') as log:
                child = subprocess.Popen(command, env=env, pass_fds=(lock.fileno(),), stdout=log, stderr=log)
                seen = helper.observation(child.pid)
                if seen is None:
                    if child.poll() is None:
                        reaped = False
                        raise RuntimeError('unknown child identity; fence retained')
                    raise RuntimeError('child exited before identity')
                identity = seen[0]; reaped = False
                helper.atomic(fence, {'state': 'RUNNING', 'child_pid': child.pid, 'child_lstart': identity,
                                      'registration_sha256': sha(regpath)})
                while child.poll() is None:
                    rss = helper.owned_observation(child, identity); free = helper.free_bytes()
                    if rss is None:
                        if child.poll() is not None: break
                        reason = 'rss-unobservable'; break
                    peak = max(peak, rss)
                    if peak > bounds['rss_bytes']: reason = 'rss'; break
                    if free is None or free < bounds['free_memory_bytes']: reason = 'memory-headroom'; break
                    if time.monotonic() - start >= bounds['wall_seconds']: reason = 'wall'; break
                    if sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) > bounds['output_bytes']: reason = 'output'; break
                    if shutil.disk_usage(out).free < bounds['free_disk_bytes']: reason = 'disk'; break
                    time.sleep(1)
        except Exception as exc:
            reason = str(exc)
        finally:
            if child is not None:
                if child.poll() is not None:
                    child.wait(); reaped = True
                elif identity is not None:
                    reaped = helper.terminate_owned(child, identity)
            size = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
            if size > bounds['output_bytes']: reason = 'output'
            pending = out / 'receipt.pending.json'
            ok = child is not None and child.returncode == 0 and reaped and reason is None and pending.is_file()
            if ok:
                pending.rename(out / 'receipt.json')
            record = {'status': 'COMPLETED' if ok else 'STOP', 'reason': reason, 'child_reaped': reaped,
                      'child_exit': child.returncode if child else None, 'peak_sampled_rss_bytes': peak,
                      'elapsed_seconds': time.monotonic() - start, 'registration_sha256': sha(regpath),
                      'grant_sha256': sha(grantpath), 'receipt_sha256': sha(out/'receipt.json') if ok else None}
            helper.atomic(out / 'SUPERVISOR.json', record)
            if reaped: fence.unlink()
    return 0 if ok else 87


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--registration', required=True); p.add_argument('--grant', required=True)
    p.add_argument('--worker', choices=['audit.py', 'prepare_worker.py', 'calibrate_worker.py', 'final_worker.py', 'fullseason_shadow_parity.py', 'fullseason_profile_export.py', 'fullseason_captured_replay.py'], default='audit.py')
    a = p.parse_args(); sys.exit(supervise(Path(a.registration), Path(a.grant), a.worker))
