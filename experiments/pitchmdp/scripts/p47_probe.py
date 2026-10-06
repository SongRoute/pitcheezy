"""P47 TRAIN/earlystop-only pre-registration probe; serial GPU3 under the common lock."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu3_two_slot as two
import run_cuda_direct as runner
from pitchmdp import direct_pitcher_id_registry as pid

PROTOCOL = 'p47_train_earlystop_probe_v1'
LOCK_DIR = '/home/sds733/pitcheezy-autonomy/gpu-locks'
CODE = ('p47_probe.py', 'p47_probe_child.py', 'gpu3_two_slot.py', 'gpu3_bench_child.py',
        'cuda_direct_queue.py', 'docs/launch/make_p47_probe_spec.py')  # lock provider + spec generator


def tasks():
    return ([{'id': f'{a}-{s}', 'arm': a, 'seed': s, 'full': False}
             for s in pid.P47_SEEDS for a in pid.P47_ARM_ORDER]
            + [{'id': f'{pid.PITCHER_BATTER_ZERO_UNK}-{pid.P47_SEEDS[0]}-full',
                'arm': pid.PITCHER_BATTER_ZERO_UNK, 'seed': pid.P47_SEEDS[0], 'full': True}])


def code_hashes():
    scripts = Path(__file__).resolve().parent
    return {name: runner.sha256_file(scripts.parent / name if '/' in name else scripts / name) for name in CODE}


def gpu3_apps():
    output = subprocess.run(['nvidia-smi', f'--id={pid.RETAINED_GPU}', '--query-compute-apps=gpu_uuid,pid',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True,
                            timeout=20).stdout
    rows = [(u.strip(), int(p)) for u, p in (line.split(',') for line in output.splitlines() if line.strip())]
    if any(u != pid.RETAINED_GPU for u, _ in rows):
        raise RuntimeError('GPU3-scoped query returned another UUID')
    return rows


def check_spec(spec):
    if spec.get('protocol') != PROTOCOL or spec.get('gpu_uuid') != pid.RETAINED_GPU \
            or spec.get('concurrent_slots') != 1 or spec.get('lock_dir') != LOCK_DIR:
        raise ValueError('P47 GPU3 serial contract differs')
    if spec.get('tasks') != tasks() or spec.get('workload') != {'train_rows': 8192, 'earlystop_rows': 2048,
                                                                 'epochs': 2}:
        raise ValueError('P47 16-mode plus full-TRAIN task plan differs')
    if spec['budget']['task_timeout_seconds'] > 600 or spec['budget']['total_wall_cap_seconds'] > 3600:
        raise ValueError('Probe cap exceeds reviewed ceiling')
    if spec['pins']['probe_code_sha256'] != code_hashes() or spec['pins']['source_sha256'] != runner.own_hashes():
        raise ValueError('Probe or model source pin differs')
    if spec['pins']['export_manifest_sha256'] != spec['config_export_manifest_sha256']:
        raise ValueError('Manifest/config pin differs')
    if spec['pins']['seed_id_ledger_review'] in (None, '') or spec['pins']['independent_review'] in (None, ''):
        raise ValueError('Coordination and source review evidence required before preflight')
    if not spec.get('absent_before') or any(not isinstance(p, str) or not p for p in spec['absent_before']):
        raise ValueError('Registration/run absence paths required')
    return spec


def assess(spec, run):
    errors, results, uuids, vocabs, payloads = [], {}, set(), {'pitcher': set(), 'batter': set()}, set()
    if run.get('completed') != 17 or run.get('max_live') != 1 or run.get('foreign_gpu_apps'):
        errors.append('17/17 serial GPU3 probe tasks required, no foreign GPU apps')
    if run.get('sampler_errors'):
        errors.append('GPU3 sampler failed')
    for item in run.get('tasks', []):
        task, result = item['task'], item.get('result')
        if item.get('returncode') != 0 or item.get('timed_out') or item.get('oom') or result is None:
            errors.append(f'{task["id"]}: failed or missing')
            continue
        if task['id'] in results:
            errors.append(f'{task["id"]}: duplicate result')
        results[task['id']] = result
        rt = result.get('runtime') or {}
        uuids.add(rt.get('bound_uuid'))
        if (result.get('protocol'), result.get('arm'), result.get('seed'), result.get('full_train')) != \
                (PROTOCOL, task['arm'], task['seed'], task['full']):
            errors.append(f'{task["id"]}: identity differs')
        if result.get('splits_loaded') != ['train', 'earlystop'] or result.get('train_only') is not True \
                or result.get('dev_used_for_fit_or_selection') is not False:
            errors.append(f'{task["id"]}: exposure boundary differs')
        if result.get('epochs_run') != 2 or result.get('source_sha256') != spec['pins']['source_sha256'] \
                or result.get('export_manifest_sha256') != spec['pins']['export_manifest_sha256'] \
                or result.get('sidecar_sha256') != spec['pins']['sidecar_sha256']:
            errors.append(f'{task["id"]}: fit/source/data pin differs')
        wanted = result.get('selected_rows_expected') or {}
        if result.get('training_rows') != wanted.get('train') or result.get('earlystop_rows') != wanted.get('earlystop') \
                or (task['full'] and (not isinstance(wanted.get('train'), int) or wanted['train'] <= 8192)):
            errors.append(f'{task["id"]}: selected TRAIN/earlystop row count differs')
        if rt.get('cuda_visible_devices') != pid.RETAINED_GPU or item.get('cuda_visible_devices') != pid.RETAINED_GPU:
            errors.append(f'{task["id"]}: GPU binding differs')
        if any(not result.get(k) for k in ('train_inputs_sha256', 'earlystop_inputs_sha256', 'train_keys_sha256',
                                           'earlystop_keys_sha256', 'state_dict_sha256')):
            errors.append(f'{task["id"]}: array/model hash missing')
        payload = result.get('verified_payload_sha256')
        if not isinstance(payload, dict) or set(payload) != {'train', 'earlystop'} \
                or any(not payload[s].get('export') or not payload[s].get('batter') for s in ('train', 'earlystop')):
            errors.append(f'{task["id"]}: selected payload hashes missing')
        else:
            payloads.add(json.dumps(payload, sort_keys=True))
        if not isinstance(result.get('fit_seconds'), (int, float)) or not math.isfinite(result['fit_seconds']) \
                or result['fit_seconds'] <= 0 or not isinstance(result.get('peak_device_bytes'), int):
            errors.append(f'{task["id"]}: time/memory missing')
        block = result.get('id_record')
        if task['arm'] == pid.REFERENCE:
            if block is not None:
                errors.append(f'{task["id"]}: reference has ID record')
        elif not isinstance(block, dict) or block.get('id_columns') != ['pitcher', 'batter'] \
                or block.get('index_source') != 'identity' or block.get('shuffle_seed') is not None:
            errors.append(f'{task["id"]}: two-ID mode differs')
        else:
            wanted_drop = 0 if task['arm'] == pid.PITCHER_BATTER else pid.ID_DROPOUT
            if block.get('id_dropout', 0) != wanted_drop or block.get('freeze_unk', False) != \
                    (task['arm'] == pid.PITCHER_BATTER_ZERO_UNK):
                errors.append(f'{task["id"]}: dropout/zero mode differs')
            if task['arm'] == pid.PITCHER_BATTER_ZERO_UNK and block.get('unk_rows_exact_zero') != \
                    {'pitcher': True, 'batter': True}:
                errors.append(f'{task["id"]}: frozen UNK row moved')
            for column in vocabs:
                record = (block.get('vocabs') or {}).get(column) or {}
                n = record.get('n_vocab')
                if not isinstance(n, int) or n < 0 or int(record.get('train_max_date') or 99999999) > 20250430:
                    errors.append(f'{task["id"]}: {column} TRAIN vocab invalid')
                vocabs[column].add((task['full'], n, record.get('ids_sha256')))
                if task['full'] and n != {'pitcher': 920, 'batter': 706}[column]:
                    errors.append(f'{task["id"]}: {column} full-TRAIN vocab differs')
    if set(results) != {t['id'] for t in tasks()} or uuids != {pid.RETAINED_GPU}:
        errors.append('task set or bound GPU UUID differs')
    if len(payloads) != 1:
        errors.append('TRAIN/earlystop payload hashes differ across tasks')
    for column, values in vocabs.items():
        if len({v for v in values if not v[0]}) != 1 or len({v for v in values if v[0]}) != 1:
            errors.append(f'{column}: TRAIN vocabulary differs among same-workload tasks')
    return {'protocol': PROTOCOL, 'valid': not errors, 'errors': errors, 'tasks': len(results),
            'bound_uuids': sorted(uuids), 'config_sha256': spec['pins']['config_sha256'],
            'export_manifest_sha256': spec['pins']['export_manifest_sha256'],
            'sidecar_sha256': spec['pins']['sidecar_sha256'], 'source_sha256': spec['pins']['source_sha256'],
            'probe_code_sha256': spec['pins']['probe_code_sha256'],
            'verified_payload_sha256': json.loads(next(iter(payloads))) if len(payloads) == 1 else None,
            'preflight_seconds_measured': run.get('wall_seconds'),
            'fit_projection_seconds': pid.P47_FAMILY_PROJECTION_SECONDS,
            'results': {key: {k: r.get(k) for k in ('fit_seconds', 'total_seconds', 'peak_device_bytes',
                                                    'train_inputs_sha256', 'earlystop_inputs_sha256',
                                                    'train_keys_sha256', 'earlystop_keys_sha256', 'state_dict_sha256')}
                        for key, r in results.items()}}


def run_probe(spec_path, out):
    spec_path, out = Path(spec_path).resolve(), Path(out)
    spec = check_spec(json.loads(spec_path.read_text()))
    if str(out) != spec['out'] or out.exists():
        raise ValueError('Probe output must be the new predeclared path')
    for path in spec['absent_before']:
        if Path(path).exists():
            raise FileExistsError(f'{path}: P47 already registered or launched')
    if runner.sha256_file(spec['config']) != spec['pins']['config_sha256']:
        raise ValueError('Draft config hash differs')
    config = runner.config_check(runner.read_json(spec['config']))
    if config['status'] != 'DRAFT' or config[pid.DESIGN_KEY] != pid.P47_DESIGN \
            or config['export']['manifest_sha256'] != spec['pins']['export_manifest_sha256'] \
            or config[pid.BATTER_SIDECAR_KEY]['sidecar_sha256'] != spec['pins']['sidecar_sha256']:
        raise ValueError('P47 draft/data pins differ')
    sup = two.TwoSlotSupervisor(LOCK_DIR, pid.RETAINED_GPU)
    sup.acquire()
    try:
        start = time.perf_counter()
        pre = two.preflight(spec, apps=gpu3_apps)
        out.mkdir(parents=True, exist_ok=False)
        two.write_new_json(out / 'spec-copy.json', {'sha256': runner.sha256_file(spec_path), 'spec': spec})
        sampler = two.Sampler(pid.RETAINED_GPU, apps=gpu3_apps)
        sampler.start()
        try:
            run = sup.run(tasks(), lambda task, folder: [spec['python'], str(Path(__file__).with_name('p47_probe_child.py')),
                           '--spec', str(spec_path), '--arm', task['arm'], '--seed', str(task['seed']),
                           *(['--full'] if task['full'] else []), '--result', str(Path(folder) / (task['id'] + '.json'))],
                           out / 'tasks', 1, task_timeout=spec['budget']['task_timeout_seconds'],
                           total_timeout=spec['budget']['total_wall_cap_seconds'])
        finally:
            sampler.halt.set()
            sampler.join(5)
        run['wall_seconds'] = time.perf_counter() - start
        two.attach_results(run, out / 'tasks', sampler.app_rows)
        run['preflight'] = pre
        run['gpu3_peak_used_bytes'] = max((s['used'] for s in sampler.samples), default=None)
        run['sampler_errors'] = sampler.errors
        two.write_new_json(out / 'run.json', run)
        summary = assess(spec, run)
        summary['spec_sha256'] = runner.sha256_file(spec_path)
        summary['run_sha256'] = runner.sha256_file(out / 'run.json')
        two.write_new_json(out / 'check.json', summary)
        return summary
    except BaseException as error:
        if out.exists() and not (out / 'check.json').exists():
            two.write_new_json(out / 'check.json', {'protocol': PROTOCOL, 'valid': False,
                                                   'errors': [f'probe aborted: {error!r}']})
        raise
    finally:
        sup.release()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('spec', type=Path)
    p.add_argument('out', type=Path)
    args = p.parse_args()
    two.install_signal_handlers()
    result = run_probe(args.spec, args.out)
    print('P47_PROBE', 'VALID' if result['valid'] else 'REFUSED', json.dumps(result['errors'][:5]))
    return 0 if result['valid'] else 3


if __name__ == '__main__':
    sys.exit(main())
