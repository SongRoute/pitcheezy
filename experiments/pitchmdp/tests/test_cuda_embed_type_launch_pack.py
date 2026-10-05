"""CPU checks of the P37 launch pack (author_only): TRAIN-only smoke gate, deterministic registration helper and the
pipeline TEMPLATE (GPU3 only, one --gpu, common lock, deadlines). Nothing is launched; outputs go to tmp only."""
import ast
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
import subprocess
import sys

import pytest

PROJECT = Path(__file__).resolve().parents[1]
LAUNCH = PROJECT / 'docs' / 'launch'
sys.path[:0] = [str(PROJECT), str(PROJECT / 'scripts'), str(LAUNCH)]
import make_p37_registered_config as reg  # noqa: E402
import p37_smoke_gate as gate  # noqa: E402
import run_cuda_direct as rc  # noqa: E402
from pitchmdp import direct_embed_type_registry as et  # noqa: E402

GPU3, GPU4 = et.RETAINED_GPU, et.RELEASED_GPUS[0]
TEMPLATE = LAUNCH / 'run-p37-pipeline.TEMPLATE.py'
KST = timezone(timedelta(hours=9))


def fits(per_epoch=None, uuid=GPU3, **patch):
    per_epoch = per_epoch or {'control': 1., 'periodic_residual_counts': 2., 'linear_residual_counts': 2.5}
    out = {}
    for arm in et.ARM_ORDER:
        for seed in et.SEEDS:
            fit = {'report': {'seconds': 2 * per_epoch[arm], 'epochs_run': 2}, 'fit_seconds': 2 * per_epoch[arm] + 30.,
                   'runtime': {'bound_uuid': uuid}, 'dev_used_for_fit_or_selection': False}
            out.setdefault(arm, {})[seed] = fit
    for (arm, seed), value in patch.get('fix', {}).items():
        out[arm][seed]['report'].update(value)
    return out


def evaluate(f):
    return gate.evaluate(f, et.ARM_ORDER, et.SEEDS, [GPU3], et.RELEASED_GPUS)


def test_gate_passes_on_train_seconds_per_epoch_and_separates_wall_overhead():
    r = evaluate(fits())
    assert r['pass'] and r['errors'] == [] and r['tasks'] == 15 and r['bound_uuids'] == [GPU3]
    assert r['linear_over_periodic'] == pytest.approx(1.25)
    assert r['fit_wall_overhead_seconds_median'] == {a: 30. for a in et.ARM_ORDER}


@pytest.mark.parametrize('bad', [{'seconds': float('nan')}, {'seconds': 0.}, {'seconds': None}, {'epochs_run': 0},
                                 {'epochs_run': None}, {'seconds': float('inf')}])
def test_gate_refuses_missing_nonfinite_or_zero_epoch_records(bad):
    r = evaluate(fits(fix={('linear_residual_counts', 41): bad}))
    assert not r['pass'] and r['errors'] and r['train_seconds_per_epoch_median']['linear_residual_counts'] is None


def test_gate_ratio_threshold_and_missing_fit():
    assert evaluate(fits({'control': 1., 'periodic_residual_counts': 2., 'linear_residual_counts': 3.}))['pass']
    assert not evaluate(fits({'control': 1., 'periodic_residual_counts': 2., 'linear_residual_counts': 3.01}))['pass']
    f = fits()
    del f['control'][43]
    assert not evaluate(f)['pass']


@pytest.mark.parametrize('uuid', [GPU4, 'GPU-other'])
def test_gate_requires_smoke_uuid_set_exactly_gpu3(uuid):
    f = fits()
    f['periodic_residual_counts'][40]['runtime']['bound_uuid'] = uuid
    r = evaluate(f)
    assert not r['pass'] and any('UUID' in e or 'released' in e for e in r['errors'])


def test_projection_constants_fit_the_clock():
    assert gate.FAMILY_PROJECTION_SECONDS == pytest.approx(5 * (200.26 + 470.40 + 705.60))
    q = json.loads(reg.DRAFT.read_text())['queue']
    end = reg.at(q['date'], q['launch_by']) + timedelta(seconds=gate.FAMILY_PROJECTION_SECONDS)
    assert end <= reg.at(q['date'], q['no_new_fit_after'])  # launch at 19:00 still projects before 21:00
    assert reg.at(q['date'], q['kill_own_after']) + timedelta(seconds=gate.SCORER_TIMEOUT_SECONDS) <= reg.at(q['date'], '22:00')


def summary(tmp_path, **over):
    record = {**evaluate(fits()), 'config_sha256': gate.DRAFT_SHA, 'export_manifest_sha256': gate.MANIFEST_SHA, **over}
    path = tmp_path / 'gate.json'
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    return path


def now(hhmm):
    return reg.at('2026-10-05', hhmm)


def test_registration_is_deterministic_executable_and_changes_only_registration_paths(tmp_path):
    s = summary(tmp_path)
    one = reg.build(s, '18:50', 'abc1234', 'root go ref', now('18:52'))
    two = reg.build(s, '18:50', 'abc1234', 'root go ref', now('18:55'))
    assert json.dumps(one) == json.dumps(two)
    draft = json.loads(reg.DRAFT.read_text())
    assert {k for k in set(one) | set(draft) if one.get(k) != draft.get(k)} == {'status', 'decision', 'execution', 'registration'}
    assert one['execution'] == {**draft['execution'], 'enabled': True} and one['queue'] == draft['queue']
    rc.config_check(one)
    rc.require_executable(one, False)
    assert one['registration']['smoke']['gate_summary_sha256'] == __import__('hashlib').sha256(s.read_bytes()).hexdigest()
    assert len(one['registration']['preregistration']['sha256']) == 64


@pytest.mark.parametrize('registered,clock,match', [
    ('19:05', '19:05', 'launch_by'), ('18:00', '18:40', 'real clock'), ('18:50', '18:30', 'real clock')])
def test_registration_refuses_late_backdated_or_future_times(tmp_path, registered, clock, match):
    with pytest.raises(AssertionError, match=match):
        reg.build(summary(tmp_path), registered, 'abc1234', 'go', now(clock))


@pytest.mark.parametrize('over', [{'pass': False}, {'bound_uuids': [GPU3, GPU4]}, {'tasks': 14},
                                  {'config_sha256': '0' * 64}, {'linear_over_periodic': 1.0}])
def test_registration_refuses_a_non_passing_gate(tmp_path, over):
    with pytest.raises(AssertionError):
        reg.build(summary(tmp_path, **over), '18:50', 'abc1234', 'go', now('18:51'))


# ------------------------------------------------------------------ pipeline template

def template_functions(*names):
    tree = ast.parse(TEMPLATE.read_text())
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    env = {'datetime': __import__('datetime'), 'SCORE_SECONDS': 900, 'SCORE_END': '22:00', 'GPU3': GPU3,
           'LOCKS': '/home/sds733/pitcheezy-autonomy/gpu-locks', 'python': 'py', 'code': Path('/c'), 'b': Path('/b'),
           'manifest': 'm'}
    exec(compile(ast.Module(keep, []), str(TEMPLATE), 'exec'), env)
    return env


def test_template_placeholders_fail_closed():
    out = subprocess.run([sys.executable, str(TEMPLATE)], capture_output=True, text=True, cwd='/tmp')
    assert out.returncode == 1 and 'configsha not filled' in out.stderr


def test_template_gpu3_only_common_lock_and_shared_functions():
    text = TEMPLATE.read_text()
    assert GPU4 not in text and text.count("'--gpu'") == 3  # command token + count + index in the one-GPU assert
    assert "LOCKS='/home/sds733/pitcheezy-autonomy/gpu-locks'" in text and "'--lock-dir',LOCKS" in text
    cmd = template_functions('queue')['queue']('cfg', 'sha', 'out', 'q')
    assert cmd.count('--gpu') == 1 and cmd[cmd.index('--gpu') + 1] == GPU3
    assert cmd[cmd.index('--lock-dir') + 1] == '/home/sds733/pitcheezy-autonomy/gpu-locks' and '--smoke' not in cmd
    p36 = ast.parse((LAUNCH / 'run-p36-pipeline.TEMPLATE.py').read_text())
    p37 = ast.parse(text)
    for name in ('save', 'interrupted', 'execute'):
        get = lambda t: next(ast.dump(n) for n in t.body if isinstance(n, ast.FunctionDef) and n.name == name)  # noqa: E731
        assert get(p36) == get(p37), name


def test_template_clock_bounds():
    clock = template_functions('clock')['clock']
    q = json.loads(reg.DRAFT.read_text())['queue']
    timeout, score_by = clock(q, now('18:55'), gate.FAMILY_PROJECTION_SECONDS)
    assert timeout == int((now('21:45') - now('18:55')).total_seconds()) + 300 and score_by == now('21:45')
    for late in ('19:01', '20:00'):
        with pytest.raises(AssertionError, match='launch_by'):
            clock(q, now(late), gate.FAMILY_PROJECTION_SECONDS)
    with pytest.raises(AssertionError, match='no_new_fit_after'):
        clock(q, now('18:55'), 3 * 3600)
    with pytest.raises(AssertionError, match='queue.date'):
        clock(q, now('18:55') + timedelta(days=1), gate.FAMILY_PROJECTION_SECONDS)
    assert math.isclose(gate.FAMILY_PROJECTION_SECONDS / 3600, 1.9115, abs_tol=1e-3)


def test_template_scores_only_after_15_of_15_and_before_the_score_deadline():
    """Launch review (codex/p37-launch-1005): the two 'incomplete' branches precede the only scorer call."""
    tree = ast.parse(TEMPLATE.read_text())
    branch = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                  and ast.unparse(n.test) == "counts != {'done': 15}")
    assert "save('incomplete'" in ast.unparse(branch.body[0])
    late = branch.orelse[0]
    assert isinstance(late, ast.If) and ast.unparse(late.test) == 'datetime.datetime.now(kst) > score_by'
    assert "save('incomplete'" in ast.unparse(late.body[0])
    scored = ast.unparse(ast.Module(late.orelse, []))
    assert "execute('score-p37-001', score, SCORE_SECONDS)" in scored and "save('complete'" in scored
    assert ast.unparse(tree).count('score_cuda_direct.py') == 1
