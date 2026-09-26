"""Measure saved replay reads without advancing a pitch or modifying a DB."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from urllib.request import urlopen


def fetch(url, timeout):
    start = time.perf_counter()
    with urlopen(url, timeout=timeout) as response:
        payload = response.read()
        status = response.status
    return {'elapsed_ms': 1000 * (time.perf_counter() - start),
            'status': status, 'bytes': len(payload), 'payload': json.loads(payload)}


def identity(view):
    # Public runtime timings/job polling may legitimately vary. Check the saved
    # recommendation and event payloads and the replay's cursor/revision.
    value = {k: view.get(k) for k in ('id', 'cursor', 'revision', 'recommendation',
                                      'history', 'event_analysis')}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if cfg['base_url'] != 'http://127.0.0.1:8767':
        raise ValueError('Only the isolated preview server is authorized here')
    output = Path(cfg['output'])
    if output.exists():
        raise ValueError('Output already exists; use a new registered run')
    health = fetch(cfg['base_url'] + '/api/health', cfg['timeout_seconds'])
    url = cfg['base_url'] + '/api/sessions/' + cfg['session_id']
    warmup = fetch(url, cfg['timeout_seconds'])
    expected = identity(warmup['payload'])
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=cfg['concurrency']) as pool:
        responses = list(pool.map(lambda _: fetch(url, cfg['timeout_seconds']), range(cfg['requests'])))
    elapsed = time.perf_counter() - started
    times = sorted(r['elapsed_ms'] for r in responses)
    # Nearest-rank quantiles, declared explicitly for the small local sample.
    from math import ceil
    quantile = lambda q: times[max(0, ceil(q * len(times)) - 1)]
    result = {'run_id': cfg['run_id'], 'checked_at': datetime.now(timezone.utc).isoformat(),
              'config_sha256': hashlib.sha256(args.config.read_bytes()).hexdigest(),
              'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope': cfg['scope'], 'health': health['payload'],
              'requests': cfg['requests'], 'concurrency': cfg['concurrency'],
              'warmup_excluded': 1, 'elapsed_seconds': elapsed,
              'quantile_method': 'nearest_rank', 'latency_ms': {'p50': quantile(.5), 'p95': quantile(.95), 'max': max(times)},
              'status_counts': {str(s): sum(r['status'] == s for r in responses) for s in {r['status'] for r in responses}},
              'saved_payload_identity_unchanged': all(identity(r['payload']) == expected for r in responses),
              'live_feed_latency_ms': None, 'next_pitch_timely_fraction': None,
              'request_ms': [r['elapsed_ms'] for r in responses]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'request_ms'}, ensure_ascii=False, indent=2))
    if not result['saved_payload_identity_unchanged'] or any(r['status'] != 200 for r in responses):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
