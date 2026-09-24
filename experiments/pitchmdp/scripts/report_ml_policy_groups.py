"""Postprocess sealed policy/RL archives into a separate descriptive group report.

No model loading, source-frame loading, fitting, selection or inferential tests.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import numpy as np
import pandas as pd
from pitchmdp.matrix_policy_groups import p0_groups, rollout_groups, POLICY_PAIRS, RL_PAIRS


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): value.update(block)
    return value.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


class Reader:
    def __init__(self): self.inputs = {}
    def pin(self, path, expected=None):
        path = Path(path).resolve(); observed = digest(path)
        if expected is not None and observed != expected: raise ValueError('Archived input hash differs: '+str(path))
        if str(path) in self.inputs and self.inputs[str(path)] != observed: raise ValueError('Input changed during report')
        self.inputs[str(path)] = observed
        return path
    def json(self, path): return json.loads(self.pin(path).read_text())
    def arrays(self, path):
        with np.load(self.pin(path), allow_pickle=False) as archive: return {k: archive[k].copy() for k in archive.files}
    def table(self, path): return pd.read_parquet(self.pin(path))
    def prepared_file(self, root, prep, name):
        return self.pin(root/name, prep['artifact_hashes'][name])
    def sealed(self, directory):
        manifest = self.json(directory/'manifest.json')
        actual = {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file() and p.name != 'manifest.json'}
        if actual != set(manifest['artifact_hashes']): raise ValueError('Sealed stage file family differs')
        for name, expected in manifest['artifact_hashes'].items():
            path = (directory/name).resolve()
            if not path.is_relative_to(directory.resolve()): raise ValueError('Invalid sealed artifact path')
            self.pin(path, expected)
        return manifest, self.json(directory/'started.json')
    def recheck(self):
        for path, expected in self.inputs.items():
            if digest(path) != expected: raise ValueError('Input changed during subgroup report')


def policy_stage(reader, root, stage):
    directory = root/'stages'/stage
    manifest, started = reader.sealed(directory)
    expected = digest(root/'preparation.json')
    if manifest['stage'] != stage or manifest['preparation_sha256'] != expected or started['preparation_sha256'] != expected:
        raise ValueError('Policy sealed stage preparation differs')
    return directory, started, reader.json(directory/'results.json')


def load_rollouts(reader, directory, result, names):
    keys = result['pa_keys']
    # Validation occurs before paths are formed; no arbitrary archive names.
    for key in keys:
        try: parts = key.split(':'); integers = [int(x) for x in parts]
        except (ValueError, AttributeError) as exc: raise ValueError('Invalid archived PA key') from exc
        if len(parts) != 3 or ':'.join(map(str, integers)) != key or any(v < 0 for v in integers):
            raise ValueError('Invalid archived PA key')
    if not keys or len(set(keys)) != len(keys): raise ValueError('Empty/duplicate archived PA family')
    if {p.name for p in directory.glob('*.npz')} != {key.replace(':', '-')+'.npz' for key in keys}:
        raise ValueError('Archived PA file family differs')
    values, flags = {n: [] for n in names}, {n: [] for n in names}
    for key in keys:
        archive = reader.arrays(directory/(key.replace(':', '-')+'.npz'))
        actual = {n[:-7] for n in archive if n.endswith('_values')}
        if actual != set(names): raise ValueError('Archived policy family differs')
        for name in names:
            values[name].append(archive[name+'_values']); flags[name].append(archive[name+'_truncated'])
    return {n: np.stack(v) for n,v in values.items()}, {n: np.stack(v) for n,v in flags.items()}


def report(policy_root, stage, output, rl_root=None):
    policy_root, output = Path(policy_root).resolve(), Path(output).resolve()
    rl_root = None if rl_root is None else Path(rl_root).resolve()
    if output.exists(): raise ValueError('Preserve prior or interrupted subgroup report')
    for root in [policy_root] + ([] if rl_root is None else [rl_root]):
        if output.is_relative_to(root) or root.is_relative_to(output): raise ValueError('Report must be outside immutable input run trees')
    if stage not in ('p0', 'dev-control', 'dev-candidate') or (stage == 'p0' and rl_root is not None):
        raise ValueError('Registered P0 or DEV policy stage required')
    reader = Reader(); prep = reader.json(policy_root/'preparation.json')
    parent = reader.json(reader.prepared_file(policy_root, prep, 'parent_preparation.json'))
    panel = parent['panel']
    directory, started, result = policy_stage(reader, policy_root, stage)
    reports = {}
    if stage == 'p0':
        for split in ('blend', 'dev'):
            table = reader.table(reader.prepared_file(policy_root, prep, 'p0_'+split+'.parquet'))
            reports[split] = p0_groups(table, reader.arrays(directory/(split+'.npz')), prep['bc_actions'], panel)
    else:
        execution = started['execution']; world = stage.removeprefix('dev-')
        if (started['world'] != world or started['split'] != 'dev' or result['world'] != world
                or result['split'] != 'dev' or result['execution_sha256'] != canonical(execution)
                or execution['preparation_sha256'] != digest(policy_root/'preparation.json')):
            raise ValueError('Policy DEV stage execution differs')
        requests = reader.table(reader.prepared_file(policy_root, prep, 'dev_requests.parquet'))
        values, flags = load_rollouts(reader, directory, result, ['P0','P1','P2','P3'])
        limit = execution['requested_starts']['dev']
        reports['policy'] = rollout_groups(requests, result['pa_keys'], result['game_ids'], values, flags, POLICY_PAIRS, limit, panel)
        counts = reports['policy']['groups']['overall']
        for saved, computed in [('requested_pa_starts','requested_pa_starts'), ('selected_pa_starts','execution_selected'), ('supported_selected','supported_selected')]:
            if result[saved] != counts[computed]: raise ValueError('Policy saved denominator differs')
        if rl_root is not None:
            rp = reader.json(rl_root/'preparation.json')
            if Path(rp['parent_policy_run']).resolve() != policy_root: raise ValueError('RL policy parent differs')
            reader.pin(policy_root/'preparation.json', rp['external_hashes'][str(policy_root/'preparation.json')])
            final = reader.json(rl_root/'final_execution.json')
            if final['preparation_sha256'] != digest(rl_root/'preparation.json'): raise ValueError('RL execution preparation differs')
            rd = rl_root/'evaluation'/world
            _, rs = reader.sealed(rd); rr = reader.json(rd/'results.json')
            if (rs['execution_sha256'] != canonical(final) or rr['execution_sha256'] != canonical(final)
                or rs['policy_execution_sha256'] != canonical(execution) or rr['parent_policy_execution_sha256'] != canonical(execution)
                or rr['world'] != world or rr['comparator'] != rp['comparator'] or rp['comparator'] not in ('P2','P3')
                or rr['tuning_dependency'] != rp['tuning_dependency'] or rr['pa_keys'] != result['pa_keys']
                or rr['game_ids'] != result['game_ids']): raise ValueError('RL/common-policy family identity differs')
            reader.pin(directory/'manifest.json', rr['dependencies'][str(directory/'manifest.json')])
            methods = ['NNBC','IQL','CQL']
            names = methods + [f'{m}-seed{s}' for m in methods for s in (0,1,2)] + ['planner']
            rv, rf = load_rollouts(reader, rd, rr, names)
            if not np.array_equal(rv['planner'], values[rp['comparator']]) or not np.array_equal(rf['planner'], flags[rp['comparator']]):
                raise ValueError('RL archived planner differs from common policy comparator')
            reports['rl'] = rollout_groups(requests, rr['pa_keys'], rr['game_ids'], rv, rf, RL_PAIRS, limit, panel)
            for name in ('requested_pa_starts', 'selected_pa_starts', 'supported_selected', 'selected_unsupported_reasons'):
                if rr[name] != result[name]: raise ValueError('RL/request denominator differs')
    sources = [Path(__file__), PROJECT/'pitchmdp/matrix_policy_groups.py']
    for path in sources: reader.pin(path)
    reader.recheck()
    output.mkdir(parents=True)
    payload = {'protocol': 'policy_subgroups_descriptive_v1', 'policy_run': str(policy_root), 'stage': stage,
        'rl_run': None if rl_root is None else str(rl_root), 'reports': reports,
        'created_utc': datetime.now(timezone.utc).isoformat(), 'input_hashes': reader.inputs,
        'inference': None, 'promotion': None,
        'provenance_scope': 'Sealed upstream stage and consumed preparation metadata checked; no model inference/deserialization, fitting, selection or new inferential tests'}
    (output/'results.json').write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    (output/'manifest.json').write_text(json.dumps({'results_sha256': digest(output/'results.json'), 'inputs': reader.inputs}, indent=2)+'\n')
    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy-run', type=Path, required=True)
    parser.add_argument('--stage', choices=['p0','dev-control','dev-candidate'], required=True)
    parser.add_argument('--rl-run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report(args.policy_run, args.stage, args.output, args.rl_run)
