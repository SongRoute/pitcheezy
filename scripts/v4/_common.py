"""Shared bits of the v4 run wrappers (public rewrite of the private originals; behaviour kept, paths made relative).

Originals pinned absolute paths for the worktree, the interpreter and a cross-job heavy lock on the external disk.
Here: repo root = two folders up, interpreter = the one running this script, lock = $PITCHEEZY_HEAVY_LOCK (optional).
"""
from __future__ import annotations

import contextlib
import datetime
import fcntl
import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
PY = sys.executable


def stamp(*parts):
    print(datetime.datetime.now().isoformat(), *parts, flush=True)


@contextlib.contextmanager
def heavy_lock():
    path = os.environ.get('PITCHEEZY_HEAVY_LOCK')
    if not path:
        yield
        return
    with open(path, 'a+') as lk:
        try:
            fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            sys.exit('heavy lock held: not started')
        yield
