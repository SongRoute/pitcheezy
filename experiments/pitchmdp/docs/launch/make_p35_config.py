"""PREPARED, NOT RUN AS REGISTRATION: build configs/EXP-P35-001.json from DRAFT f0ae4e91 + the P35 timing-probe summary.

Usage (only after Song's explicit go; DRAFT `decision` requires it):
  python make_p35_config.py <probe summary.json> <out.json> <launch date YYYY-MM-DD> <no_new HH:MM> <kill HH:MM> <aggregate HH:MM> <cycle id>
Changes only the extractor REGISTRATION_PATHS (experiment_id, status, decision, execution.enabled, registration,
queue.date/date_note/batch/no_new_fit_after/kill_own_after/aggregate_by). Key order preserved.
"""
import hashlib, json, sys
from pathlib import Path
DRAFT = Path(__file__).resolve().parents[4] / 'configs' / 'DRAFT-CUDA-PLE-RESIDUAL-20261004.json'
DRAFT_SHA = 'f0ae4e9116d3efe90d119fcc84616172bde183a981206caacc4804d68a31b9b2'
PROBE_SHA = '311f0d4f5dff318320a6da0d8944e6d7b50fdc1a1eb8ee4cd7a95d754fb67ff5'
summary_path, out_path, date, no_new, kill, agg, cycle = sys.argv[1:8]
raw = DRAFT.read_bytes()
assert hashlib.sha256(raw).hexdigest() == DRAFT_SHA
d = json.loads(raw)
summary_raw = Path(summary_path).read_bytes()
assert hashlib.sha256(summary_raw).hexdigest() == PROBE_SHA, 'not the 3b317f2 probe summary'
s = json.loads(summary_raw)
g = s['gate']
assert s['complete'] is True and set(s['exit_codes'].values()) == {0}
assert g['ple_arms_within'] is True and g['every_arm_within'] is True and g['family_within'] is True
assert s['preflight']['config_sha256'] == DRAFT_SHA
assert no_new < kill < agg, 'deadlines must be ordered'
out = {}
for k, v in d.items():
    out[k] = v
    if k == 'experiment_id':
        out[k] = 'EXP-P35-001'
    elif k == 'status':
        out[k] = 'REGISTERED'
    elif k == 'decision':
        out[k] = ("Song's explicit go (new-experiment rule, recorded in the coordinator ledger) plus the pinned runbook v4 "
                  "delegation authorize exploratory execution after the P35 timing gate. same_model_reviewed; "
                  "Codex_review_pending. No scientific N/G decision, promotion or selection; G0, G15 mix (D145), "
                  "D131/D155 unchanged.")
    elif k == 'queue':
        q = dict(v)
        q['date'] = date
        q['date_note'] = ('Launch-day deadlines set at registration from the P35 timing-probe projection (family '
                          f"{s['family_worker_seconds_projected']:.0f} worker-s over 2 GPUs); unchanged task timeout, "
                          'family budget and attempts.')
        q['batch'] = ('P35 PLE embedding 5 arms x fresh seeds 29-33, epoch cap 60; registered by the delegated Opus PM '
                      'worker after 25/25 CUDA smoke identities (8864389) and the timing probe (3b317f2) passed all gates.')
        q['no_new_fit_after'], q['kill_own_after'], q['aggregate_by'] = no_new, kill, agg
        out[k] = q
    elif k == 'execution':
        out[k] = {**v, 'enabled': True}
out['registration'] = {
    'source_commit': '8864389',
    'draft_path': 'configs/DRAFT-CUDA-PLE-RESIDUAL-20261004.json',
    'draft_sha256': DRAFT_SHA,
    'smoke': 'queue-ple-smoke-001 25/25 done from code-8864389 (record SMOKE-PLE-8864389-20261004.md, 0978558)',
    'timing_probe': {
        'commit': '3b317f2',
        'preregistration': 'experiments/pitchmdp/docs/AMENDMENT-P35-TIMING-PROBE-20261004.md',
        'result': 'experiments/pitchmdp/docs/RESULT-P35-TIMING-PROBE-3b317f2-20261004.md (dc6b94b)',
        'summary': 'runs/DRAFT-PLE-RESIDUAL-probe-3b317f2-timing/summary.json',
        'summary_sha256': PROBE_SHA,
        'slowest_task_seconds_projected': s['slowest_task_seconds_projected'],
        'family_worker_seconds_projected': s['family_worker_seconds_projected'],
        'gates': {'complete': True, 'exit_code': 0, 'ple_arms_le_1800': True, 'every_arm_task_le_1800': True,
                  'family_le_36000': True},
    },
    'registered_by': f'delegated Opus 5.5 PM worker, {cycle} (not root, not Codex)',
    'review': ('Module, wiring, pre-score extractor and timing probe: distinct Opus reviews (same-model) with '
               'dispositions c2352f8 / fe7ee46+8864389 / fadcb6f / 3b317f2. Family code and results: '
               'same_model_reviewed; Codex_review_pending.'),
    'exposure': 'Repeated DEV, post-selection, descriptive only; no 2026 data; no unseen validation.',
}
text = json.dumps(out, indent=2, ensure_ascii=False) + '\n'
Path(out_path).write_text(text)
print(hashlib.sha256(text.encode()).hexdigest())
