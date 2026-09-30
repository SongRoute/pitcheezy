"""Re-certify the <=2025 policy identity after a registered code change (D122/D124 procedure).

Runs S2 bind-probe -> S3b tau-select -> S5 v2-world -> S6 dr-evaluate as new attempts, reusing the
identity-independent sealed census, S1/S1b, S3 profile and S4 outputs. Each addendum is written,
committed and pushed before the stage that needs it (the runner refuses uncommitted registration).
Stops at the first failure. Usage: .venv/bin/python scripts/recertify_chain.py <tag> <attempt-suffix>
"""
import hashlib, json, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = Path('/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/ML-MATRIX-20260924')
V = ROOT / 'ML-POLICY-VAL-v1'
CFG = 'configs/ML-POLICY-MATERIALIZATION-v1.json'
REUSE = {1: ('census', {'census': 'S0-census-a2/census.json'}),
         2: ('materialize-bc+style-snapshot', {'bc': 'S1-materialize-a3/bc_BC_P.json', 'support': 'S1-materialize-a3/support_primary.json',
             'hands': 'S1-materialize-a3/hands.json', 'materialize': 'S1-materialize-a3/materialize.json',
             'style_dev': 'S1b-style-a3/style_dev.json', 'style_temperature': 'S1b-style-a3/style_temperature.json',
             'style_blend': 'S1b-style-a3/style_blend.json'}),
         4: ('profile', {'profile': 'S3-profile-a2/profile.json'}),
         6: ('v5-denominators', {'v5': 'S4-v5-a1/v5.json'})}
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sh(*args, **kw):
    print('+', ' '.join(map(str, args)), flush=True)
    subprocess.run(args, cwd=REPO, check=True, **kw)


def addendum(tag, i, stage, inputs, note, expected=None):
    parent = REPO / (CFG if i == 1 else f'configs/ML-POLICY-MATERIALIZATION-v1.addendum-{i - 1}-{tag}.json')
    a = {'parent_sha256': sha(parent), 'stage': stage,
         'registered_inputs': {k: {'path': str(V / v), 'file_sha256': sha(V / v)} for k, v in inputs.items()}, 'note': note}
    if expected:
        a['expected_identity_sha256'] = expected
    path = f'configs/ML-POLICY-MATERIALIZATION-v1.addendum-{i}-{tag}.json'
    (REPO / path).write_text(json.dumps(a, indent=2, ensure_ascii=False) + '\n')
    sh('git', 'add', path)
    sh('git', 'commit', '-q', '-m', f'docs(policy-validation): addendum {i} {tag}\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>')
    sh('git', 'push', '-q')
    return path


def run(command, name, adds):
    log = ROOT / 'coordination' / f'{name}.console.log'
    argv = ['.venv/bin/python', 'experiments/pitchmdp/scripts/run_policy_validation.py', '--config', CFG]
    for a in adds:
        argv += ['--addendum', a]
    argv += ['--local-config', 'experiments/pitchmdp/configs/local.json', '--output', str(V / name), command]
    env = {'PYTHONPATH': f'{ROOT}/deps:experiments/pitchmdp', 'PATH': '/usr/bin:/bin:/usr/local/bin'}
    with open(log, 'w') as out:
        code = subprocess.run(argv, cwd=REPO, env=env, stdout=out, stderr=subprocess.STDOUT).returncode
    print(f'{name}: exit {code}', flush=True)
    if code:
        sys.exit(f'{name} failed; see {log}')


def main(tag, suffix):
    adds = []
    for i in (1, 2):
        adds.append(addendum(tag, i, REUSE[i][0], REUSE[i][1], f'{tag}: re-pinned identity-independent sealed outputs.'))
    run('bind-probe', f'S2-bind-probe-{suffix}', adds)
    adds.append(addendum(tag, 3, 'bind-probe', {'bind_probe': f'S2-bind-probe-{suffix}/probe.json',
                                               'bind_identity': f'S2-bind-probe-{suffix}/identity.json'}, f'{tag}: S2 re-certified.'))
    adds.append(addendum(tag, 4, REUSE[4][0], REUSE[4][1], f'{tag}: profile re-pinned (identity-independent).'))
    run('tau-select', f'S3b-tau-{suffix}', adds)
    freeze = json.loads((V / f'S3b-tau-{suffix}/tau_freeze.json').read_text())
    if freeze['status'] != 'SELECTED' or freeze['selected_tau'] != 0.1:
        sys.exit(f'S3b changed: {freeze["status"]} {freeze["selected_tau"]}')
    adds.append(addendum(tag, 5, 'tau-select', {'tau_freeze': f'S3b-tau-{suffix}/tau_freeze.json'}, f'{tag}: tau re-frozen.'))
    adds.append(addendum(tag, 6, REUSE[6][0], REUSE[6][1], f'{tag}: S4 re-pinned (identity-independent).'))
    run('v2-world', f'S5-v2-{suffix}', adds)
    adds.append(addendum(tag, 7, 'v2-world', {'v2': f'S5-v2-{suffix}/v2.json'}, f'{tag}: V2 re-certified; expected identity pinned.',
                         expected=freeze['final_identity_sha256']))
    run('dr-evaluate', f'S6-dr-{suffix}', adds)
    print('RECERTIFIED', freeze['final_identity_sha256'], flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:3])
