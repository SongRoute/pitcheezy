"""`python -m observer_app.publisher` runs publisher.py as __main__, while the background upload worker imports
`.publisher` by name. Both must be the same module: otherwise the worker raises a different UploadError class, the
`except UploadError` in MirrorSink._transfer never matches, and an upload attempt stays 'inflight' for good
(seen 2026-11-15: game file never re-sent for 6 minutes in a real-pace rehearsal)."""
import os
import subprocess
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]

PROBE = """
import runpy, sys
sys.argv = ['publisher', '--help']
try:
    runpy.run_module('observer_app.publisher', run_name='__main__', alter_sys=True)
except SystemExit:
    pass
from observer_app import upload_worker  # noqa: F401
from observer_app.publisher import UploadError, MirrorSink
print(UploadError.__module__, MirrorSink.__module__)
"""


def test_worker_sees_the_running_main_module():
    env = {k: v for k, v in os.environ.items() if not k.startswith('PITCHEEZY_')} | {'PYTHONPATH': str(BACKEND)}
    out = subprocess.run([sys.executable, '-B', '-c', PROBE], cwd=BACKEND, env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.split()[-2:] == ['__main__', '__main__'], out.stdout
