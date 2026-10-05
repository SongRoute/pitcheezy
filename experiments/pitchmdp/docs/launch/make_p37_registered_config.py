"""Build configs/EXP-P37-001.json from DRAFT 75265fc7 + the PASSING P37 smoke-gate summary (deterministic: same inputs ->
same bytes; the registration time is an argument, checked against the real clock, never backdated).

  python make_p37_registered_config.py <gate summary.json> <out.json> <registered HH:MM KST> <source commit> <go reference>

Rules (pre-registration sections 4, 7, 8): date 2026-10-05 KST (queue.date); registered HH:MM <= launch_by 19:00 and
within 15 min of the real clock (not in the future); registered + family projection <= no_new_fit_after 21:00; the
gate summary passes (recomputed: ratio <= 1.5, 15 tasks, bound UUIDs == [GPU3], no GPU4, config sha == DRAFT). Only
REGISTRATION_PATHS change; arms, seeds, training, device, queue clock and comparison stay byte-for-value the DRAFT.
"""
import hashlib
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(REPO / 'experiments/pitchmdp'), str(REPO / 'experiments/pitchmdp/docs/launch')]
import p37_smoke_gate as gate  # noqa: E402
from pitchmdp import direct_embed_type_registry as et  # noqa: E402

DRAFT = REPO / 'configs' / 'DRAFT-CUDA-EMBED-TYPE-20261005.json'
DRAFT_SHA = gate.DRAFT_SHA
PREREG = 'experiments/pitchmdp/docs/PREREG-DRAFT-EXP-P37-001-embed-type-2026-10-05.md'
REGISTRATION_PATHS = ('status', 'decision', 'execution.enabled', 'registration')
KST = timezone(timedelta(hours=9))
CLOCK_SKEW = timedelta(minutes=15)


def at(day, hhmm):
    return datetime.strptime(day + ' ' + hhmm, '%Y-%m-%d %H:%M').replace(tzinfo=KST)


def check_gate(summary):
    assert summary.get('protocol') == gate.PROTOCOL, 'gate protocol differs'
    assert summary.get('pass') is True and summary.get('errors') == [], 'smoke gate did not pass'
    assert summary.get('config_sha256') == DRAFT_SHA and summary.get('export_manifest_sha256') == gate.MANIFEST_SHA
    assert summary.get('tasks') == et.FITS, 'smoke gate tasks != 15'
    assert summary.get('bound_uuids') == [et.RETAINED_GPU], 'smoke UUID set must be exactly {GPU3}'
    medians = summary['train_seconds_per_epoch_median']
    ratio = medians['linear_residual_counts'] / medians['periodic_residual_counts']
    assert abs(ratio - summary['linear_over_periodic']) < 1e-9 and ratio <= gate.RATIO_MAX, 'timing ratio gate fails'
    assert summary['family_projection_seconds'] == gate.FAMILY_PROJECTION_SECONDS, 'projection constant differs'
    return ratio


def build(summary_path, registered, commit, go, now):
    assert re.fullmatch(r'[0-9a-f]{7,40}', commit), 'source commit must be 7-40 lowercase hex'
    assert re.fullmatch(r'\d{2}:\d{2}', registered) and go.strip(), 'registered HH:MM and a go reference required'
    raw = DRAFT.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == DRAFT_SHA, 'DRAFT config differs from its reviewed sha'
    config = json.loads(raw)
    q = config['queue']
    when = at(q['date'], registered)
    assert when <= at(q['date'], q['launch_by']), 'registration after launch_by (19:00): not launched today'
    assert when - CLOCK_SKEW <= now <= when + CLOCK_SKEW and when <= now + timedelta(minutes=1), \
        'registered time must be the real clock (no backdating, no future)'
    assert when + timedelta(seconds=gate.FAMILY_PROJECTION_SECONDS) <= at(q['date'], q['no_new_fit_after']), \
        'projection does not finish before no_new_fit_after (21:00)'
    summary_raw = Path(summary_path).read_bytes()
    ratio = check_gate(json.loads(summary_raw))
    prereg = REPO / PREREG
    config.update(status='REGISTERED',
                  decision='None. REGISTERED for exploratory descriptive execution only (go: registration.go); no '
                           'winner, promotion or policy claim.',
                  execution={**config['execution'], 'enabled': True},
                  registration={'draft_sha256': DRAFT_SHA, 'registered_kst': q['date'] + ' ' + registered,
                                'source_commit': commit,
                                'preregistration': {'path': PREREG,
                                                    'sha256': hashlib.sha256(prereg.read_bytes()).hexdigest()},
                                'smoke': {'gate_summary_sha256': hashlib.sha256(summary_raw).hexdigest(),
                                          'linear_over_periodic_train_seconds_per_epoch': ratio,
                                          'family_projection_seconds': gate.FAMILY_PROJECTION_SECONDS},
                                'go': go})
    return config


def main(argv):
    summary, out, registered, commit, go = argv
    out = Path(out)
    assert not out.exists(), 'output already exists'
    text = json.dumps(build(summary, registered, commit, go, datetime.now(KST)), indent=2, ensure_ascii=False) + '\n'
    with out.open('x') as handle:
        handle.write(text)
    print(out, hashlib.sha256(text.encode()).hexdigest())


if __name__ == '__main__':
    main(sys.argv[1:])
