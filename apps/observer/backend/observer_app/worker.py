"""Durable no-media worker; it never fabricates a CV result or alters manual input."""
import argparse
import signal
import time

from .settings import CONFIG, database_path
from .store import Store
from .cv import CVRequest, NoMediaAdapter


def run_once(store, *, now=None, adapter=None):
    job = store.claim(CONFIG['cv_lease_seconds'], CONFIG['cv_max_attempts'], now=now)
    if job is None:
        return False
    result = (adapter or NoMediaAdapter()).analyze(CVRequest(pitch_id=job['pitch_id']))
    store.finish(job['id'], job['lease_token'], status=result.status, now=now,
                 message=result.message, cv_status=result.cv_status)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database')
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    store = Store(args.database or database_path())
    if args.once:
        run_once(store)
        return
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while not stopping:
        if not run_once(store):
            time.sleep(.5)


if __name__ == '__main__':
    main()
