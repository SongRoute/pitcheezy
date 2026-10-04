"""PREPARED, NOT RUN AS REGISTRATION (T107, author_only): build configs/EXP-P36-001.json from DRAFT 644239f3 + the P36
timing-probe summary + the registered configs/EXP-P35-001.json. Mirrors make_p35_config.py (reviewed ed02276/859f315).

Usage (only after P35 is terminal with its descriptive report, the P36 smoke 20/20 and the reviewed P36 probe, and an
explicit go; DRAFT `decision` requires it):
  python make_p36_config.py <probe summary.json> <EXP-P35-001.json> <out.json> <launch date YYYY-MM-DD> <launch HH:MM>
      <no_new HH:MM> <kill HH:MM> <aggregate HH:MM> <archive commit> <cycle id> <go reference>
Deadline rule (pre-registration section 5, review a23e77b m1; the P35 pack rule): with launch time L,
no_new_fit_after >= L + 5 h, kill_own_after >= no_new + 1 h, aggregate_by >= kill + 30 min, all on the launch date
(KST), aggregate_by <= 23:00 (coordinator deadline), and L <= 16:30 when the date is 2026-10-05. <archive commit> is the
smoke-verified commit whose `git archive` is the fresh server code dir (code-<commit>); <go reference> points at the go
(ledger path + line or commit + KST time) and is stored as registration.go.
Changes only REGISTRATION_PATHS below (the P35 set + source_configs.ple_registered, review a23e77b m5). Key order
preserved; never budgets, arms, seeds, epochs, stop rules, training, comparison or any other DRAFT text.
"""
import hashlib, json, re, sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
DRAFT = Path(__file__).resolve().parents[4] / 'configs' / 'DRAFT-CUDA-ROBUST-CLIP-20261005.json'
DRAFT_SHA = '644239f3b2aaa4410391211ab1e4d972f13e1a6b025a026b058f0dff319c470a'
P35_DRAFT_SHA = 'f0ae4e9116d3efe90d119fcc84616172bde183a981206caacc4804d68a31b9b2'
MANIFEST_SHA = '49097109b4f522a5a94d1557e00c52d88a8089c725977fe263c592214fb1f578'
# Probe-summary keys read here, derived from the P35 probe summary schema (scripts/timing_probe_ple_residual.py
# run()/summarize()). The P36 probe (scripts/timing_probe_robust_clip.py, another branch, not seen by this author)
# copies that schema; CHECK THIS TABLE AGAINST THE ACTUAL P36 PROBE SUMMARY AT DISPOSITION. A missing key fails closed.
PROBE_KEYS = {
    'complete': 'complete',                                  # bool, must be True
    'exit_codes': 'exit_codes',                              # {gpu uuid: int}, both GPUs, all 0
    'config_sha256': ('preflight', 'config_sha256'),         # == DRAFT_SHA
    'manifest_sha256': ('preflight', 'export_manifest_sha256'),  # == MANIFEST_SHA
    'gpus': ('preflight', 'gpus'),                           # == DRAFT device.allowed_gpu_uuids (as a set)
    'slowest': 'slowest_task_seconds_projected',             # {arm: s}, must hold PROBED_ARMS, every value <= 1800
    'family': 'family_worker_seconds_projected',             # number <= 36000
    'gate': 'gate',                                          # dict; every GATE_KEYS entry present and True,
}                                                            # and every other boolean in it True as well
GATE_KEYS = ('robust_clip_arms_within', 'every_arm_within', 'family_within')  # P35: ple_arms_within -> P36 name
PROBED_ARMS = ('residual_counts', 'robust_clip_residual_counts')  # pre-registration section 4 (ii)
TASK_GATE_SECONDS, FAMILY_GATE_SECONDS = 1800, 36000
# Paths this generator may change (the P35 extractor set + source_configs.ple_registered). Nothing else differs.
REGISTRATION_PATHS = ('experiment_id', 'status', 'decision', 'execution.enabled', 'registration', 'queue.date',
                      'queue.date_note', 'queue.batch', 'queue.no_new_fit_after', 'queue.kill_own_after',
                      'queue.aggregate_by', 'source_configs.ple_registered')
P35_SHARED_ARMS = ('control', 'periodic_residual_counts')  # pre-registration section 8 item 5 / review a23e77b m5


def probe_value(summary, key):
    path = PROBE_KEYS[key]
    value = summary
    for part in (path,) if isinstance(path, str) else path:
        assert isinstance(value, dict) and part in value, f'probe summary key missing: {path!r}'
        value = value[part]
    return value


def same(a, b):
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


assert len(sys.argv) == 12, 'usage: see module docstring (11 arguments)'
summary_path, p35_path, out_path, date, launch, no_new, kill, agg, commit, cycle, go = sys.argv[1:12]
KST = timezone(timedelta(hours=9))
day = datetime.strptime(date, '%Y-%m-%d').date()
assert re.fullmatch(r'\d{4}-\d{2}-\d{2}', date), f'not YYYY-MM-DD: {date!r}'
for hhmm in (launch, no_new, kill, agg):
    assert re.fullmatch(r'([01]\d|2[0-3]):[0-5]\d', hhmm), f'not HH:MM: {hhmm!r}'
assert go.strip() and cycle.strip(), 'cycle id and go reference must be non-empty'
assert re.fullmatch(r'[0-9a-f]{7,40}', commit), f'archive commit must be 7-40 lowercase hex: {commit!r}'
minutes = {k: int(v[:2]) * 60 + int(v[3:]) for k, v in (('L', launch), ('no_new', no_new), ('kill', kill), ('agg', agg))}
assert minutes['no_new'] >= minutes['L'] + 300, 'no_new_fit_after must be >= launch + 5 h'
assert minutes['kill'] >= minutes['no_new'] + 60, 'kill_own_after must be >= no_new_fit_after + 1 h'
assert minutes['agg'] >= minutes['kill'] + 30, 'aggregate_by must be >= kill_own_after + 30 min'
assert date != '2026-10-05' or minutes['L'] <= 16 * 60 + 30, 'latest launch on 2026-10-05 is 16:30 (prereg section 5)'
assert minutes['agg'] <= 23 * 60, 'aggregate_by must be <= 23:00 (coordinator deadline)'
assert day >= datetime.now(KST).date(), 'launch date is in the past (KST)'
assert not Path(out_path).exists(), f'output exists: {out_path}'
raw = DRAFT.read_bytes()
assert hashlib.sha256(raw).hexdigest() == DRAFT_SHA, 'DRAFT differs from the reviewed 644239f3'
d = json.loads(raw)
# Registered P35 config: control / periodic_residual_counts must equal the DRAFT entries (review a23e77b m5).
p35_raw = Path(p35_path).read_bytes()
assert Path(p35_path).name == 'EXP-P35-001.json', 'second argument must be the registered configs/EXP-P35-001.json'
p35 = json.loads(p35_raw)
assert p35.get('experiment_id') == 'EXP-P35-001' and p35.get('status') == 'REGISTERED', \
    'not the registered EXP-P35-001 config'
assert p35.get('registration', {}).get('draft_sha256') == P35_DRAFT_SHA, 'EXP-P35-001 is not built from DRAFT f0ae4e91'
p35_arms, d_arms = {a['name']: a for a in p35['arms']}, {a['name']: a for a in d['arms']}
for name in P35_SHARED_ARMS:
    assert name in p35_arms and same(p35_arms[name], d_arms[name]), f'{name}: EXP-P35-001 entry differs from the DRAFT'
# P36 timing probe summary (pre-registration section 4 (ii)); every gate must be true, fail closed on a missing key.
summary_raw = Path(summary_path).read_bytes()
s = json.loads(summary_raw)
assert probe_value(s, 'complete') is True, 'timing probe incomplete'
codes = probe_value(s, 'exit_codes')
assert isinstance(codes, dict) and len(codes) == 2 and all(type(c) is int and c == 0 for c in codes.values()), \
    'probe exit codes must be 0 on both GPUs'
assert probe_value(s, 'config_sha256') == DRAFT_SHA, 'probe ran on another config'
assert probe_value(s, 'manifest_sha256') == MANIFEST_SHA, 'probe ran on another export'
assert sorted(probe_value(s, 'gpus')) == sorted(d['device']['allowed_gpu_uuids']) == sorted(codes), 'probe GPUs differ'
gate = probe_value(s, 'gate')
assert isinstance(gate, dict), 'probe gate is not a dict'
for key in GATE_KEYS:
    assert key in gate, f'probe gate key missing: {key}'
    assert gate[key] is True, f'probe gate false: {key}'
assert all(v is True for v in gate.values() if isinstance(v, bool)), 'a probe gate boolean is false'
slowest, family = probe_value(s, 'slowest'), probe_value(s, 'family')
assert isinstance(slowest, dict) and all(a in slowest for a in PROBED_ARMS), 'probe did not project the probed arms'
assert all(isinstance(v, (int, float)) and 0 < v <= TASK_GATE_SECONDS for v in slowest.values()), 'task gate'
assert isinstance(family, (int, float)) and 0 < family <= FAMILY_GATE_SECONDS, 'family gate'
out = {}
for k, v in d.items():
    out[k] = v
    if k == 'experiment_id':
        out[k] = 'EXP-P36-001'
    elif k == 'status':
        out[k] = 'REGISTERED'
    elif k == 'decision':
        out[k] = ("Explicit go recorded in registration.go (new-experiment rule) authorizes exploratory execution after "
                  "the P36 CUDA smoke (20/20) and the P36 timing gate; chosen under pre-registration section 7 (P35 "
                  "complete 25/25 with its descriptive report). same_model_reviewed; Codex_review_pending. No "
                  "scientific N/G decision, promotion or selection; G0, G15 mix (D145), D131/D155 unchanged.")
    elif k == 'source_configs':
        out[k] = {**v, 'ple_registered': {'path': 'configs/EXP-P35-001.json',
                                          'sha256': hashlib.sha256(p35_raw).hexdigest()}}
    elif k == 'queue':
        q = dict(v)
        q['date'] = date
        q['date_note'] = (f'Launch-day deadlines from the fixed rule (launch {launch} KST; no_new >= L+5 h, kill >= '
                          'no_new+1 h, aggregate >= kill+30 min, aggregate <= 23:00; pre-registration section 5, '
                          f"review a23e77b m1). P36 timing-probe projection: family {family:.0f} worker-s over 2 GPUs; "
                          'unchanged task timeout, family budget and attempts.')
        q['batch'] = (f'P36 robust-clip 4 arms x fresh seeds 34-38, epoch cap 60; registered by the delegated Opus PM '
                      f'worker after 20/20 CUDA smoke identities ({commit}) and the P36 timing probe passed all gates.')
        q['no_new_fit_after'], q['kill_own_after'], q['aggregate_by'] = no_new, kill, agg
        out[k] = q
    elif k == 'execution':
        out[k] = {**v, 'enabled': True}
out['registration'] = {
    'source_commit': commit,
    'draft_path': 'configs/DRAFT-CUDA-ROBUST-CLIP-20261005.json',
    'draft_sha256': DRAFT_SHA,
    'preregistration': 'experiments/pitchmdp/docs/PREREG-DRAFT-EXP-P36-001-robust-clip-2026-10-04.md',
    'timing_probe': {
        'summary': summary_path,
        'summary_sha256': hashlib.sha256(summary_raw).hexdigest(),
        'slowest_task_seconds_projected': slowest,
        'family_worker_seconds_projected': family,
        'gates': {'complete': True, 'exit_code': 0, **{k: True for k in GATE_KEYS}},
    },
    'registered_by': f'delegated Opus 5.5 PM worker, {cycle} (not root, not Codex)',
    'go': go,
    'review': ('Module, wiring + pre-registration, registration pack and timing probe: distinct Opus reviews '
               '(same-model) with dispositions. Family code and results: same_model_reviewed; Codex_review_pending.'),
    'exposure': 'Repeated DEV, post-selection, descriptive only; no 2026 data; no unseen validation.',
}
text = json.dumps(out, indent=2, ensure_ascii=False) + '\n'
with open(out_path, 'x', encoding='utf-8') as f:  # never overwrite (P35 review MINOR-6)
    f.write(text)
print(hashlib.sha256(text.encode()).hexdigest())
