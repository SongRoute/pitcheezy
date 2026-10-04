"""Torch-free checks of the PREPARED P36 registration pack (T107, author_only): make_p36_config.py dry-runs into tmp from
a synthetic P36 probe summary and an EXP-P35-001 config built by make_p35_config.py into tmp (synthetic P35-schema probe
summary; only its pinned summary sha is swapped in memory). Nothing under configs/ is written; nothing is launched."""
import ast
import copy
import hashlib
import json
from datetime import date as Date, datetime, timedelta, timezone
from pathlib import Path
import subprocess
import sys

import pytest

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
LAUNCH = PROJECT / 'docs' / 'launch'
sys.path.insert(0, str(PROJECT / 'scripts'))
import run_cuda_direct as rc  # noqa: E402

DRAFT = REPO / 'configs' / 'DRAFT-CUDA-ROBUST-CLIP-20261005.json'
DRAFT_SHA = '644239f3b2aaa4410391211ab1e4d972f13e1a6b025a026b058f0dff319c470a'
P35_DRAFT_SHA = 'f0ae4e9116d3efe90d119fcc84616172bde183a981206caacc4804d68a31b9b2'
P35_PROBE_SHA = '311f0d4f5dff318320a6da0d8944e6d7b50fdc1a1eb8ee4cd7a95d754fb67ff5'
MANIFEST = '49097109b4f522a5a94d1557e00c52d88a8089c725977fe263c592214fb1f578'
GPUS = ['GPU-c334d905-8ec9-0029-b206-46ce3ac49cd9', 'GPU-ae803a24-0d38-791d-ed7c-3f233fcecf99']
MAKE36 = LAUNCH / 'make_p36_config.py'
MAKE35 = LAUNCH / 'make_p35_config.py'
TEMPLATE = LAUNCH / 'run-p36-pipeline.TEMPLATE.py'
TEMPLATE35 = LAUNCH / 'run-p35-pipeline.TEMPLATE.py'
README = LAUNCH / 'README-p36-registration-pack.md'
KST = timezone(timedelta(hours=9))
TODAY = datetime.now(KST).date()
# The launch day of the prepared pack; once it is past, a later date keeps the happy path runnable.
DATE = '2026-10-05' if TODAY <= Date(2026, 10, 5) else (TODAY + timedelta(days=1)).isoformat()
COMMIT = 'abc1234'
EXEC = ('import json, sys\npath, patches = sys.argv[1], json.loads(sys.argv[2])\nsrc = open(path).read()\n'
        'for old, new in patches:\n'
        '    assert src.count(old) == 1, "patch target not unique"\n'
        '    src = src.replace(old, new)\n'
        'sys.argv = [path] + sys.argv[3:]\n'
        'exec(compile(src, path, "exec"), {"__file__": path, "__name__": "__main__"})\n')


def constant(path, name):
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(getattr(t, 'id', None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(script, args, patches=()):
    """Plain python (no -O: every check is an assert), optional in-memory constant swaps, never edits the script."""
    return subprocess.run([sys.executable, '-c', EXEC, str(script), json.dumps(list(patches)), *map(str, args)],
                          capture_output=True, text=True, timeout=120)


def p35_summary():
    slowest = {'control': 464.0, 'periodic_residual_counts': 1046.0, 'ple_residual_counts': 1149.0,
               'tabm_residual_counts': 1150.0, 'ple_tabm_residual_counts': 1225.0}
    return {'protocol': 'synthetic', 'exit_codes': {g: 0 for g in GPUS}, 'complete': True,
            'preflight': {'config_sha256': P35_DRAFT_SHA, 'export_manifest_sha256': MANIFEST, 'gpus': GPUS},
            'slowest_task_seconds_projected': slowest, 'family_worker_seconds_projected': 5 * sum(slowest.values()),
            'marginal_arms': [], 'gate': {'task_seconds_max': 1800, 'family_seconds_max': 36000, 'ple_arms_within': True,
                                          'every_arm_within': True, 'family_within': True}}


# The P36 probe summary exactly in the schema of scripts/timing_probe_robust_clip.py @ 3b305ad (another branch, not
# imported): run() :446-449 (protocol, exit_codes, complete, written_utc, preflight, host_post, + summarize()),
# summarize() :289-303, preflight() :366-375. Values synthetic; the dict construction is copied.
PROBE_PROTOCOL = 'robust_clip_timing_probe_v1'
PROBE_MODULE_SHA = 'a3d6cbcef8350586a23d1a6d35213e36ba8e0ccf21288288464f8ce7dcbf52fa'
PROBE_CONFIG_ARMS = ['control', 'residual_counts', 'periodic_residual_counts', 'robust_clip_residual_counts']
PROBE_ARMS = ['residual_counts', 'robust_clip_residual_counts']
PROBE_FIXED = {'control': 464., 'periodic_residual_counts': 1046.}


def p36_summary(measured=None, seeds_per_arm=5):
    measured = measured or {'residual_counts': 900.0, 'robust_clip_residual_counts': 950.0}
    exit_codes = {g: 0 for g in GPUS}
    per_arm = {n: [{'gpu': g, 'task_seconds_projected': measured[n] - i, 'measured_seconds_per_epoch': 10.0}
                   for i, g in enumerate(GPUS)] for n in PROBE_ARMS}
    slowest = {n: PROBE_FIXED[n] if n in PROBE_FIXED else measured[n] for n in PROBE_CONFIG_ARMS}
    family = sum(seeds_per_arm * v for v in slowest.values())
    identity = {'protocol': PROBE_PROTOCOL, 'config_sha256': DRAFT_SHA, 'export_manifest_sha256': MANIFEST,
                'module_sha256': PROBE_MODULE_SHA, 'source_sha256': {}, 'row_counts': {}, 'fixed': {},
                'gpus': list(GPUS), 'lock_dir': '/locks', 'python': '/venv/bin/python', 'checked_utc': 'x',
                'host_pre': {}}
    summary = {'protocol': PROBE_PROTOCOL, 'exit_codes': exit_codes, 'complete': True, 'written_utc': 'x',
               'preflight': identity, 'host_post': {}}
    summary.update({'per_arm': per_arm, 'slowest_task_seconds_projected': slowest,
                    'family_worker_seconds_projected': family, 'probed_task_seconds_projected': dict(measured),
                    'fixed_task_seconds': {'arms': dict(PROBE_FIXED), 'source': 'x', 'measured_here': False},
                    'marginal_arms': [],
                    'gate': {'task_seconds_max': 1800, 'family_seconds_max': 36000,
                             'every_arm_within': all(v <= 1800 for v in slowest.values()),
                             'family_within': family <= 36000,
                             'probe_exit_zero': bool(exit_codes) and all(c == 0 for c in exit_codes.values())},
                    'label': 'synthetic'})
    return summary


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))
    return path


@pytest.fixture(scope='module')
def pack(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('p36pack')
    s35 = write(tmp / 'p35-summary.json', p35_summary())
    p35 = tmp / 'configs' / 'EXP-P35-001.json'
    p35.parent.mkdir()
    done = run(MAKE35, [s35, p35, DATE, '09:00', '14:00', '15:00', '15:30', 'dry-p35', 'DRY RUN P35 (T107 test)'],
               patches=[(P35_PROBE_SHA, sha(s35))])
    assert done.returncode == 0, done.stderr
    s36 = write(tmp / 'p36-summary.json', p36_summary())
    out = tmp / 'out' / 'EXP-P36-001.json'
    out.parent.mkdir()
    args = [s36, p35, out, DATE, '09:00', '14:00', '15:00', '15:30', COMMIT, 'dry-run-t107', 'DRY RUN ONLY (T107)']
    # The fixture P35 config pins the synthetic P35 summary; swap the generator's real-P35 pin in memory to match it.
    patches = [(P35_PROBE_SHA, sha(s35))]
    done = run(MAKE36, args, patches)
    assert done.returncode == 0, done.stderr
    return {'tmp': tmp, 'p35': p35, 's36': s36, 'out': out, 'args': args, 'stdout': done.stdout, 'patches': patches}


def diff_paths(draft, registered):
    paths = set()
    for key in draft.keys() | registered.keys():
        a, b = draft.get(key), registered.get(key)
        if key in ('queue', 'execution', 'source_configs') and isinstance(a, dict) and isinstance(b, dict):
            paths |= {f'{key}.{k}' for k in a.keys() | b.keys() if a.get(k) != b.get(k)}
        elif a != b:
            paths.add(key)
    return paths


# ------------------------------------------------------------------ happy path

def test_dry_run_differs_from_the_draft_only_in_registration_paths(pack):
    draft, out = json.loads(DRAFT.read_text()), json.loads(pack['out'].read_text())
    assert sha(DRAFT) == DRAFT_SHA
    assert pack['stdout'].strip() == sha(pack['out'])
    allowed = set(constant(MAKE36, 'REGISTRATION_PATHS'))
    changed = diff_paths(draft, out)
    assert changed <= allowed, changed - allowed
    must = allowed - ({'queue.date'} if DATE == draft['queue']['date'] else set())
    assert changed == must
    assert list(out)[:len(draft)] == list(draft) and list(out)[len(draft):] == ['registration']
    assert list(out['source_configs'])[:len(draft['source_configs'])] == list(draft['source_configs'])


def test_budgets_rules_and_text_equal_the_draft(pack):
    draft, out = json.loads(DRAFT.read_text()), json.loads(pack['out'].read_text())
    for key in ('timezone', 'task_timeout_seconds', 'family_budget_seconds', 'max_attempts', 'order', 'start_after',
                'stop_rule'):
        assert out['queue'][key] == draft['queue'][key], key
    for key in ('protocol', 'seed_rule', 'arms', 'seeds', 'fits', 'training', 'calibration', 'export', 'device',
                'comparison', 'arm_notes', 'data_rules', 'purpose', 'control_reuse', 'exposure'):
        assert out[key] == draft[key], key
    assert {k: v for k, v in out['source_configs'].items() if k != 'ple_registered'} == draft['source_configs']
    assert out['execution'] == {**draft['execution'], 'enabled': True}
    assert out['experiment_id'] == 'EXP-P36-001' and out['status'] == 'REGISTERED'
    q = out['queue']
    assert (q['no_new_fit_after'], q['kill_own_after'], q['aggregate_by'], q['date']) == ('14:00', '15:00', '15:30', DATE)
    assert '(launch 09:00 KST;' in q['date_note']


def test_registration_record(pack):
    out = json.loads(pack['out'].read_text())
    assert out['source_configs']['ple_registered'] == {'path': 'configs/EXP-P35-001.json', 'sha256': sha(pack['p35'])}
    reg = out['registration']
    assert reg['go'].strip() == 'DRY RUN ONLY (T107)' and reg['source_commit'] == COMMIT
    assert reg['draft_sha256'] == DRAFT_SHA and reg['draft_path'] == 'configs/DRAFT-CUDA-ROBUST-CLIP-20261005.json'
    assert reg['timing_probe']['summary_sha256'] == sha(pack['s36'])
    assert reg['timing_probe']['gates'] == {'complete': True, 'exit_code': 0, 'every_arm_within': True,
                                            'family_within': True, 'probe_exit_zero': True}
    assert 'Codex_review_pending' in reg['review'] and 'dry-run-t107' in reg['registered_by']
    assert out['decision'].count('No scientific N/G decision, promotion or selection') == 1


def test_registered_dry_run_passes_the_runner_checks_and_the_draft_does_not(pack):
    out, draft = json.loads(pack['out'].read_text()), json.loads(DRAFT.read_text())
    assert rc.config_check(copy.deepcopy(out))['fits'] == 20
    rc.require_executable(out, smoke=False)
    with pytest.raises(ValueError):
        rc.require_executable(draft, smoke=False)
    assert rc.config_digest(out) != rc.config_digest(draft)


def test_p35_shared_arms_match_the_draft_in_the_generated_p35_config(pack):
    p35, draft = json.loads(pack['p35'].read_text()), json.loads(DRAFT.read_text())
    arms35, arms = {a['name']: a for a in p35['arms']}, {a['name']: a for a in draft['arms']}
    assert constant(MAKE36, 'P35_SHARED_ARMS') == ('control', 'periodic_residual_counts')
    for name in ('control', 'periodic_residual_counts'):
        assert arms35[name] == arms[name]


# ------------------------------------------------------------------ refusals

def refused(pack, tmp_path, message, args=None, patch=('', '')):
    args = list(args if args is not None else pack['args'])
    if args and len(args) == 11 and args[2] == pack['out']:
        args[2] = tmp_path / 'never.json'
    done = run(MAKE36, args, pack['patches'] + ([patch] if patch[0] else []))
    assert done.returncode != 0
    assert message in done.stderr, done.stderr[-800:]
    assert not (tmp_path / 'never.json').exists()


def with_args(pack, **changes):
    names = ['summary', 'p35', 'out', 'date', 'L', 'no_new', 'kill', 'agg', 'commit', 'cycle', 'go']
    args = dict(zip(names, pack['args']))
    args.update(changes)
    return [args[n] for n in names]


@pytest.mark.parametrize('changes, message', [
    ({'date': 'garbage'}, 'does not match format'),
    ({'date': '2026-10-5'}, 'not YYYY-MM-DD'),
    ({'date': '2026-10-03'}, 'launch date is in the past'),
    ({'L': '9:00'}, 'not HH:MM'),
    ({'agg': '15:60'}, 'not HH:MM'),
    ({'no_new': '13:59'}, 'no_new_fit_after must be >= launch + 5 h'),
    ({'kill': '14:59'}, 'kill_own_after must be >= no_new_fit_after + 1 h'),
    ({'agg': '15:29'}, 'aggregate_by must be >= kill_own_after + 30 min'),
    ({'date': '2026-10-05', 'L': '16:35', 'no_new': '21:35', 'kill': '22:35', 'agg': '23:05'},
     'latest launch on 2026-10-05 is 16:30'),
    ({'L': '16:30', 'no_new': '21:30', 'kill': '22:30', 'agg': '23:05'}, 'aggregate_by must be <= 23:00'),
    ({'go': '   '}, 'cycle id and go reference must be non-empty'),
    ({'cycle': ''}, 'cycle id and go reference must be non-empty'),
    ({'commit': 'XYZ'}, 'archive commit must be 7-40 lowercase hex'),
])
def test_argv_and_deadline_refusals(pack, tmp_path, changes, message):
    refused(pack, tmp_path, message, with_args(pack, **changes))


def test_wrong_argument_count_refused(pack, tmp_path):
    refused(pack, tmp_path, '11 arguments', pack['args'][:10])
    refused(pack, tmp_path, '11 arguments', pack['args'] + ['extra'])


def test_existing_output_refused(pack, tmp_path):
    before = pack['out'].read_bytes()
    done = run(MAKE36, pack['args'], pack['patches'])
    assert done.returncode != 0 and 'output exists' in done.stderr
    assert pack['out'].read_bytes() == before


def test_draft_sha_mismatch_refused(pack, tmp_path):
    refused(pack, tmp_path, 'DRAFT differs from the reviewed 644239f3', patch=(DRAFT_SHA, '0' * 64))


def p35_variant(pack, tmp_path, mutate, name='EXP-P35-001.json'):
    value = json.loads(pack['p35'].read_text())
    mutate(value)
    return write(tmp_path / 'p35' / name, value)


def arm(config, name):
    return next(a for a in config['arms'] if a['name'] == name)


@pytest.mark.parametrize('mutate, message', [
    (lambda c: arm(c, 'control').update(width=256), 'control: EXP-P35-001 entry differs from the DRAFT'),
    (lambda c: arm(c, 'periodic_residual_counts')['context_continuous'].pop(),
     'periodic_residual_counts: EXP-P35-001 entry differs from the DRAFT'),
    (lambda c: c.update(arms=[a for a in c['arms'] if a['name'] != 'control']),
     'control: EXP-P35-001 entry differs from the DRAFT'),
    (lambda c: c.update(status='DRAFT'), 'not the registered EXP-P35-001 config'),
    (lambda c: c.update(experiment_id='EXP-P34-001'), 'not the registered EXP-P35-001 config'),
    (lambda c: c['registration'].update(draft_sha256='0' * 64), 'not built from DRAFT f0ae4e91'),
])
def test_p35_config_mismatch_refused(pack, tmp_path, mutate, message):
    refused(pack, tmp_path, message, with_args(pack, p35=p35_variant(pack, tmp_path, mutate)))


def test_dry_run_p35_config_refused_by_the_unpatched_generator(pack, tmp_path):
    """Review T109: the tmp make_p35_config.py dry run (synthetic summary) is not the real P35 registration."""
    out = tmp_path / 'never.json'
    done = run(MAKE36, with_args(pack, out=out))
    assert done.returncode != 0 and 'not the code-8864389 / probe 311f0d4f registration' in done.stderr
    assert not out.exists()
    assert constant(MAKE36, 'P35_PROBE_SHA') == constant(MAKE35, 'PROBE_SHA') == P35_PROBE_SHA


@pytest.mark.parametrize('mutate', [
    lambda c: c['registration'].update(source_commit='abc1234'),
    lambda c: c['registration']['timing_probe'].pop('summary_sha256'),
])
def test_p35_registration_identity_mismatch_refused(pack, tmp_path, mutate):
    refused(pack, tmp_path, 'not the code-8864389 / probe 311f0d4f registration',
            with_args(pack, p35=p35_variant(pack, tmp_path, mutate)))


def test_p35_config_wrong_file_name_refused(pack, tmp_path):
    path = p35_variant(pack, tmp_path, lambda c: None, name='DRAFT-CUDA-PLE-RESIDUAL-20261004.json')
    refused(pack, tmp_path, 'must be the registered configs/EXP-P35-001.json', with_args(pack, p35=path))


def drop(*path):
    def mutate(s):
        for part in path[:-1]:
            s = s[part]
        del s[path[-1]]
    return mutate


def setter(value, *path):
    def mutate(s):
        for part in path[:-1]:
            s = s[part]
        s[path[-1]] = value
    return mutate


@pytest.mark.parametrize('mutate, message', [
    (setter(False, 'complete'), 'timing probe incomplete'),
    (drop('complete'), "probe summary key missing: 'complete'"),
    (setter(1, 'exit_codes', GPUS[1]), 'probe exit codes must be 0 on both GPUs'),
    (setter(None, 'exit_codes', GPUS[0]), 'probe exit codes must be 0 on both GPUs'),
    (setter({GPUS[0]: 0}, 'exit_codes'), 'probe exit codes must be 0 on both GPUs'),
    (setter('0' * 64, 'preflight', 'config_sha256'), 'probe ran on another config'),
    (setter('0' * 64, 'preflight', 'export_manifest_sha256'), 'probe ran on another export'),
    (drop('preflight'), "probe summary key missing: ('preflight', 'config_sha256')"),
    (setter([GPUS[0], 'GPU-other'], 'preflight', 'gpus'), 'probe GPUs differ'),
    (drop('gate', 'probe_exit_zero'), 'probe gate key missing: probe_exit_zero'),
    (drop('gate', 'every_arm_within'), 'probe gate key missing: every_arm_within'),
    (drop('gate', 'family_within'), 'probe gate key missing: family_within'),
    (setter(False, 'gate', 'every_arm_within'), 'probe gate false: every_arm_within'),
    (setter(False, 'gate', 'probe_exit_zero'), 'probe gate false: probe_exit_zero'),
    (drop('gate'), "probe summary key missing: 'gate'"),
    (setter(False, 'gate', 'family_within'), 'probe gate false: family_within'),
    (setter(1, 'gate', 'probe_exit_zero'), 'probe gate false: probe_exit_zero'),
    (setter(False, 'gate', 'some_new_gate'), 'a probe gate boolean is false'),
    (drop('slowest_task_seconds_projected', 'robust_clip_residual_counts'), 'probe did not project the probed arms'),
    (setter(1801.0, 'slowest_task_seconds_projected', 'residual_counts'), 'task gate'),
    (setter(36001.0, 'family_worker_seconds_projected'), 'family gate'),
    (drop('family_worker_seconds_projected'), "probe summary key missing: 'family_worker_seconds_projected'"),
    (setter('ple_residual_timing_probe_v1', 'protocol'), 'not a P36 robust-clip timing-probe summary'),
    (drop('protocol'), "probe summary key missing: 'protocol'"),
    (setter('0' * 64, 'preflight', 'module_sha256'), 'probe ran another robust-clip module'),
    (drop('slowest_task_seconds_projected', 'control'), 'probe projection arms differ from the DRAFT arms'),
    (setter(1.0, 'slowest_task_seconds_projected', 'tabm_residual_counts'),
     'probe projection arms differ from the DRAFT arms'),
    (setter(400.0, 'slowest_task_seconds_projected', 'control'), 'not the P35 probe constants'),
    (setter(True, 'slowest_task_seconds_projected', 'residual_counts'), 'task gate'),
    (setter(10000.0, 'family_worker_seconds_projected'), 'family projection is not seeds x sum'),
])
def test_probe_gate_refusals(pack, tmp_path, mutate, message):
    summary = p36_summary()
    mutate(summary)
    refused(pack, tmp_path, message, with_args(pack, summary=write(tmp_path / 'summary.json', summary)))


def test_probe_constants_match_the_probe_branch_values():
    """Pins in the generator equal the probe 3b305ad literals (:62, :66, :71) as copied into this test file."""
    assert constant(MAKE36, 'PROBE_PROTOCOL') == PROBE_PROTOCOL
    assert constant(MAKE36, 'MODULE_SHA') == PROBE_MODULE_SHA
    assert constant(MAKE36, 'FIXED_TASK_SECONDS') == PROBE_FIXED
    from pitchmdp import direct_robust_clip_registry as reg
    assert reg.MODULE_SHA256 == PROBE_MODULE_SHA


def test_gate_keys_are_the_probe_schema_booleans():
    """Review T109 BLOCKER: every GATE_KEYS entry is a boolean of the real probe gate (3b305ad :297-300)."""
    gate = p36_summary()['gate']
    assert set(gate) == {'task_seconds_max', 'family_seconds_max', 'every_arm_within', 'family_within',
                         'probe_exit_zero'}
    for path in (MAKE36, TEMPLATE):
        keys = constant(path, 'GATE_KEYS')
        assert keys == ('every_arm_within', 'family_within', 'probe_exit_zero')
        assert all(gate[k] is True for k in keys)


def test_probe_schema_failing_gate_refused(pack, tmp_path):
    """A real-schema summary whose probed arm projects above 1,800 s carries every_arm_within False."""
    summary = p36_summary({'residual_counts': 900.0, 'robust_clip_residual_counts': 1850.0})
    assert summary['gate']['every_arm_within'] is False and summary['gate']['family_within'] is True
    refused(pack, tmp_path, 'probe gate false: every_arm_within',
            with_args(pack, summary=write(tmp_path / 'summary.json', summary)))


def test_old_guessed_gate_name_is_refused(pack, tmp_path):
    """Mutant: the pre-review GATE_KEYS (guessed robust_clip_arms_within) refuses a passing real-schema summary, and
    the fixed generator refuses a summary carrying the guessed name instead of probe_exit_zero."""
    old = "GATE_KEYS = ('robust_clip_arms_within', 'every_arm_within', 'family_within')"
    fixed = "GATE_KEYS = ('every_arm_within', 'family_within', 'probe_exit_zero')"
    refused(pack, tmp_path, 'probe gate key missing: robust_clip_arms_within', patch=(fixed, old))
    summary = p36_summary()
    summary['gate']['robust_clip_arms_within'] = summary['gate'].pop('probe_exit_zero')
    refused(pack, tmp_path, 'probe gate key missing: probe_exit_zero',
            with_args(pack, summary=write(tmp_path / 'summary.json', summary)))


# ------------------------------------------------------------------ pipeline template

def test_template_fails_closed_on_the_placeholder(tmp_path):
    text = TEMPLATE.read_text()
    assert "configsha='FILL_AT_REGISTRATION'\nassert len(configsha)==64,'template: configsha not filled'" in text
    done = subprocess.run([sys.executable, str(TEMPLATE)], cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert done.returncode == 1 and 'AssertionError: template: configsha not filled' in done.stderr
    assert list(tmp_path.iterdir()) == []


def test_template_other_placeholders_fail_closed(tmp_path):
    text = TEMPLATE.read_text().replace("configsha='FILL_AT_REGISTRATION'", "configsha='" + '0' * 64 + "'", 1)
    script = tmp_path / 'filled.py'
    script.write_text(text)
    done = subprocess.run([sys.executable, str(script)], cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert done.returncode == 1 and 'template: placeholder not filled' in done.stderr
    assert sorted(p.name for p in tmp_path.iterdir()) == ['filled.py']


def test_template_has_no_launch_record_wait_loop_or_self_launch():
    text = TEMPLATE.read_text()
    for word in ('p33-launch', 'launch.json', '-O', 'PYTHONOPTIMIZE', 'nohup', 'setsid', 'ssh', 'while ', 'sleep',
                 'run-p36-pipeline', 'EXP-P35-001.json\').write', 'rmtree', 'unlink'):
        assert word not in text, word
    lines = [line for line in text.splitlines() if 'cuda_direct_queue.py' in line]
    assert len(lines) == 2 and lines[0].startswith(' return [python,str(code/\'scripts/cuda_direct_queue.py\')')
    assert "'pgrep','-f','cuda_direct_queue.py|timing_probe_|score_cuda_direct'" in lines[1]
    compile(text, str(TEMPLATE), 'exec')


def test_template_gates_and_pins_match_the_generator():
    assert constant(TEMPLATE, 'GATE_KEYS') == constant(MAKE36, 'GATE_KEYS')
    assert constant(TEMPLATE, 'draftsha') == constant(MAKE36, 'DRAFT_SHA') == DRAFT_SHA
    assert constant(TEMPLATE, 'manifest') == constant(MAKE36, 'MANIFEST_SHA') == MANIFEST
    from pitchmdp import direct_robust_clip_registry as reg
    assert constant(TEMPLATE, 'modulesha') == reg.MODULE_SHA256
    text = TEMPLATE.read_text()
    assert "('p34-001',25),('p30-002',50),('p35-001',25)" in text and "{'done':20}" in text
    assert "'scores/EXP-P36-001'" in text and 'scores/EXP-P35-001' not in text


class _Clock(datetime):
    fixed = None

    @classmethod
    def now(cls, tz=None):
        return cls.fixed.astimezone(tz)


def one_date_guard(now_kst, queue):
    """Execute the template's two one-date guard lines (review T109 MINOR-4) with a fixed clock."""
    import datetime as real
    import types
    lines = [line.strip() for line in TEMPLATE.read_text().splitlines() if line.endswith('# one-date guard (T109)')]
    assert len(lines) == 2
    _Clock.fixed = now_kst
    clock = types.SimpleNamespace(datetime=_Clock, timezone=real.timezone, timedelta=real.timedelta)
    exec(compile('\n'.join(lines), str(TEMPLATE), 'exec'), {'datetime': clock, 'new': {'queue': queue}})


@pytest.mark.parametrize('now, ok', [
    ('2026-10-05 08:55', True), ('2026-10-05 09:00', True), ('2026-10-05 09:01', False),
    ('2026-10-04 23:50', False), ('2026-10-06 08:00', False),
])
def test_template_refuses_a_launch_later_than_no_new_minus_5h_or_off_date(now, ok):
    queue = {'date': '2026-10-05', 'no_new_fit_after': '14:00'}
    when = datetime.strptime(now, '%Y-%m-%d %H:%M').replace(tzinfo=KST)
    if ok:
        one_date_guard(when, queue)
    else:
        with pytest.raises(AssertionError, match='actual launch is not on queue.date'):
            one_date_guard(when, queue)


def test_template_checks_probe_identity():
    text = TEMPLATE.read_text()
    assert ("p['protocol']=='robust_clip_timing_probe_v1' and p['preflight']['module_sha256']==modulesha and "
            "sorted(p['preflight']['gpus'])==sorted(old['device']['allowed_gpu_uuids'])") in text
    assert text.index('# one-date guard (T109)') < text.index(" execute('p36-full-001'")


def function_source(path, name):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(path.read_text(), node)


@pytest.mark.parametrize('name', ['save', 'interrupted', 'execute', 'queue'])
def test_template_process_control_is_the_p35_code(name):
    assert function_source(TEMPLATE, name) == function_source(TEMPLATE35, name)


def test_readme_launch_command_is_plain_and_bounded():
    text = README.read_text()
    launch = [line for line in text.splitlines() if 'nohup setsid timeout --kill-after=60' in line]
    assert len(launch) == 1
    for word in ('cuda_direct_queue.py', 'timing_probe_', ' -O', 'PYTHONOPTIMIZE', 'p33-launch'):
        assert word not in launch[0], word
    assert '/home/sds733/venv/bin/python run-p36-pipeline.py' in launch[0]
    assert 'NOT REGISTERED' in text and 'author_only' in text



def test_the_p35_fixture_passes_the_p35_registered_config_tests(pack, monkeypatch):
    """The tmp EXP-P35-001 (make_p35_config.py dry run) is a valid registered P35 config under its own test file,
    except the probe-summary pin test (the fixture used a synthetic summary; its sha is recorded instead)."""
    import test_cuda_ple_registered as p35test
    monkeypatch.setattr(p35test, 'REGISTERED', pack['p35'])
    pin = json.loads(pack['p35'].read_text())['registration']['timing_probe']['summary_sha256']
    assert pin == sha(pack['tmp'] / 'p35-summary.json') != P35_PROBE_SHA
    p35test.test_only_identity_status_prose_and_launch_day_deadlines_differ()
    p35test.test_registered_family_passes_frozen_checks_and_allows_real_fits()
    p35test.test_draft_still_refuses_real_fits()
    if DATE == '2026-10-05':  # that test pins the P35 launch day
        p35test.test_deadlines_follow_the_fixed_rule_on_one_date()
