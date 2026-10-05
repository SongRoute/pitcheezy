"""Build configs/EXP-P38-001.json from DRAFT 8518f754 + the v1 smoke-gate summary (validity) + the PASSING v2 TRAIN-only
fixed-step resource measurement (deterministic: same inputs -> same bytes; the registration time is an argument, checked
against the real clock, never backdated).

  python make_p38_registered_config.py <v1 gate summary.json> <v2 measure.json> <out.json> <registered HH:MM KST> \
      <source commit> <go reference>

Resource rule revision (p38-recovery-001, 2026-10-05, written AFTER observing the v1 result): the v1 smoke gate REFUSED
on its resource ratio alone (1.2528 > 1.25, errors []). Its 10/10 smoke VALIDITY checks are still required here (errors
== [], 10 tasks, {GPU3}, DRAFT/manifest sha), but the resource ratio now comes from the ONE v2 measurement
(p38_resource_measure.py, full TRAIN, fixed epochs, same 1.25), which must pass and share the smoke's source map. The v1
refusal and its ratio are recorded unchanged in registration.smoke; this is not the original pre-registration.

Rules (pre-registration sections 4, 6): queue.date of the DRAFT; registered HH:MM <= launch_by and within 15 min of the
real clock (not in the future); registered + family projection (5,292 s) <= no_new_fit_after; the gate summary passes
(recomputed: candidate/reference <= 1.25, 10 tasks, bound UUIDs == [GPU3], no GPU4, config sha == DRAFT). Only
REGISTRATION_PATHS change (execution.enabled and execution.executable true); arms, seeds, training, device, queue clock
and comparison stay byte-for-value the DRAFT.
"""
import hashlib
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(REPO / 'experiments/pitchmdp'), str(REPO / 'experiments/pitchmdp/docs/launch')]
import p38_smoke_gate as gate  # noqa: E402
from pitchmdp import direct_pitcher_id_registry as pid  # noqa: E402

DRAFT = REPO / 'configs' / 'DRAFT-CUDA-PITCHER-ID-20261005.json'
DRAFT_SHA = gate.DRAFT_SHA
RESOURCE_PROTOCOL = 'p38_train_fixed_step_resource_v2'
RESOURCE_SPEC = 'experiments/pitchmdp/docs/launch/p38-resource-v2-001.json'
PREREG = 'experiments/pitchmdp/docs/PREREG-DRAFT-EXP-P38-001-pitcher-id-2026-10-05.md'
REGISTRATION_PATHS = ('status', 'decision', 'execution.enabled', 'execution.executable', 'execution.blocked_until',
                      'registration')
KST = timezone(timedelta(hours=9))
CLOCK_SKEW = timedelta(minutes=15)


def at(day, hhmm):
    return datetime.strptime(day + ' ' + hhmm, '%Y-%m-%d %H:%M').replace(tzinfo=KST)


def check_gate(summary):
    """v1 smoke VALIDITY (every check except the resource ratio, which v2 replaces). Returns the v1 ratio, unchanged."""
    assert summary.get('protocol') == gate.PROTOCOL, 'gate protocol differs'
    assert summary.get('errors') == [], 'smoke gate has validity errors'
    assert summary.get('config_sha256') == DRAFT_SHA and summary.get('export_manifest_sha256') == gate.MANIFEST_SHA
    assert summary.get('tasks') == pid.FITS, 'smoke gate tasks != 10'
    assert summary.get('bound_uuids') == [pid.RETAINED_GPU], 'smoke UUID set must be exactly {GPU3}'
    medians = summary['train_seconds_per_epoch_median']
    ratio = medians[pid.CANDIDATE] / medians[pid.REFERENCE]
    assert abs(ratio - summary['candidate_over_reference']) < 1e-9, 'v1 ratio does not recompute'
    assert summary.get('pass') is (ratio <= pid.RESOURCE_RATIO_MAX), 'v1 pass flag must reflect only its ratio'
    assert summary['family_projection_seconds'] == pid.FAMILY_PROJECTION_SECONDS, 'projection constant differs'
    return ratio


def check_resource(measure, smoke):
    """The ONE v2 fixed-step measurement passed, on the DRAFT/manifest and the smoke's exact source map."""
    decision, spec = measure.get('decision') or {}, (measure.get('spec') or {})
    assert measure.get('protocol') == RESOURCE_PROTOCOL and decision.get('protocol') == RESOURCE_PROTOCOL
    assert decision.get('pass') is True and decision.get('errors') == [], 'v2 resource measurement did not pass'
    assert decision.get('ratio_max') == pid.RESOURCE_RATIO_MAX == 1.25, 'threshold changed'
    medians = decision['train_seconds_per_epoch_median']
    ratio = medians[pid.CANDIDATE] / medians[pid.REFERENCE]
    assert abs(ratio - decision['candidate_over_reference']) < 1e-9 and ratio <= pid.RESOURCE_RATIO_MAX, \
        'v2 resource ratio fails'
    assert decision.get('bound_uuids') == [pid.RETAINED_GPU] and decision.get('tasks') == 4
    pins = spec['pins']
    assert pins['config_sha256'] == DRAFT_SHA and pins['export_manifest_sha256'] == gate.MANIFEST_SHA
    assert pins['source_sha256'] == smoke['source_sha256'], 'v2 measured a different source map than the smoke'
    return ratio


def build(summary_path, measure_path, registered, commit, go, now):
    assert re.fullmatch(r'[0-9a-f]{7,40}', commit), 'source commit must be 7-40 lowercase hex'
    assert re.fullmatch(r'\d{2}:\d{2}', registered) and go.strip(), 'registered HH:MM and a go reference required'
    raw = DRAFT.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == DRAFT_SHA, 'DRAFT config differs from its reviewed sha'
    config = json.loads(raw)
    q = config['queue']
    when = at(q['date'], registered)
    assert when <= at(q['date'], q['launch_by']), 'registration after launch_by: re-date the DRAFT (make_p38_config --clock)'
    assert when - CLOCK_SKEW <= now <= when + CLOCK_SKEW and when <= now + timedelta(minutes=1), \
        'registered time must be the real clock (no backdating, no future)'
    assert when + timedelta(seconds=pid.FAMILY_PROJECTION_SECONDS) <= at(q['date'], q['no_new_fit_after']), \
        'projection does not finish before no_new_fit_after'
    summary_raw, measure_raw = Path(summary_path).read_bytes(), Path(measure_path).read_bytes()
    smoke = json.loads(summary_raw)
    v1_ratio = check_gate(smoke)
    measure = json.loads(measure_raw)
    measure['spec'] = json.loads((REPO / RESOURCE_SPEC).read_text())
    assert hashlib.sha256((REPO / RESOURCE_SPEC).read_bytes()).hexdigest() == measure_spec_sha(measure_raw), \
        'v2 measurement used a different spec than this commit'
    ratio = check_resource(measure, smoke)
    prereg = REPO / PREREG
    config.update(status='REGISTERED',
                  decision='None. REGISTERED for exploratory descriptive execution only (go: registration.go); no '
                           'winner, promotion or policy claim.',
                  execution={'enabled': True, 'executable': True},
                  registration={'draft_sha256': DRAFT_SHA, 'registered_kst': q['date'] + ' ' + registered,
                                'source_commit': commit,
                                'preregistration': {'path': PREREG,
                                                    'sha256': hashlib.sha256(prereg.read_bytes()).hexdigest()},
                                'smoke': {'gate_summary_sha256': hashlib.sha256(summary_raw).hexdigest(),
                                          'v1_verdict': 'PASS' if smoke['pass'] else 'REFUSED (resource ratio only)',
                                          'v1_candidate_over_reference_train_seconds_per_epoch': v1_ratio,
                                          'family_projection_seconds': pid.FAMILY_PROJECTION_SECONDS},
                                'resource': {'protocol': RESOURCE_PROTOCOL,
                                             'measurement_sha256': hashlib.sha256(measure_raw).hexdigest(),
                                             'spec': RESOURCE_SPEC,
                                             'candidate_over_reference_train_seconds_per_epoch': ratio,
                                             'ratio_max': pid.RESOURCE_RATIO_MAX,
                                             'method_revised_after_observing_v1': True,
                                             'note': 'resource rule only; not the original pre-registration, not an '
                                                     'independent validation, not a science result'},
                                'go': go})
    return config


def measure_spec_sha(measure_raw):
    return json.loads(measure_raw)['spec_sha256']


def main(argv):
    summary, measure, out, registered, commit, go = argv
    out = Path(out)
    assert not out.exists(), 'output already exists'
    text = json.dumps(build(summary, measure, registered, commit, go, datetime.now(KST)), indent=2, ensure_ascii=False) + '\n'
    with out.open('x') as handle:
        handle.write(text)
    print(out, hashlib.sha256(text.encode()).hexdigest())


if __name__ == '__main__':
    main(sys.argv[1:])
