"""Two preregistered offline current-action screens from frozen Q checkpoints."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.b_choice_metrics import concentration


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def choose_best(q):
    """Unrounded descending Q, original action order for exact ties."""
    return int(np.argmax(q))


def choose_near_value(q, actions, repertoire, epsilon_pp):
    best = float(np.max(q))
    eligible = [i for i, value in enumerate(q) if 100*(best-value) <= epsilon_pp]
    return min(eligible, key=lambda i: (-repertoire.get(actions[i]['pitch_type'], 0), -q[i], i))


def choose_minimax(q_members, q_reference):
    """Compare within-member regrets; member absolute WE offsets cancel."""
    regret_pp = 100*(np.max(q_members, axis=1, keepdims=True)-q_members)
    worst = np.max(regret_pp, axis=0)
    return min(range(len(worst)), key=lambda i: (worst[i], -q_reference[i], i))


def interval_by_game(rows, field, seed, repeats):
    games = sorted({r['game_pk'] for r in rows})
    by_game = {game: np.asarray([r[field] for r in rows if r['game_pk'] == game], float) for game in games}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        draw = rng.choice(games, size=len(games), replace=True)
        values.append(float(np.concatenate([by_game[game] for game in draw]).mean()))
    return {'mean': float(np.mean([r[field] for r in rows])),
            'percentile_95': np.quantile(values, [.025, .975]).tolist(),
            'cluster': 'game_pk', 'games': len(games), 'repeats': repeats, 'seed': seed,
            'status': 'exploratory_exposed_DEV_six_games'}


def load_inputs(cfg, cfg_path):
    committed = subprocess.run(['git', 'show', f'HEAD:{cfg_path.resolve().relative_to(ROOT)}'],
                               cwd=ROOT, capture_output=True, check=True).stdout
    if cfg_path.read_bytes() != committed:
        raise ValueError('Candidate config differs from committed preregistration')
    cdir = Path(cfg['checkpoint_dir'])
    diagnosis_path = cdir.parent/'results.json'
    diagnosis = json.loads(diagnosis_path.read_text())
    if diagnosis['experiment_id'] != cfg['parent_diagnosis']:
        raise ValueError('Parent diagnosis ID mismatch')
    checkpoints = []
    if len(diagnosis['checkpoint_sha256']) != 151:
        raise ValueError('Expected 151 frozen diagnosis checkpoints')
    for name, expected in sorted(diagnosis['checkpoint_sha256'].items()):
        path = cdir/name
        if sha(path) != expected:
            raise ValueError(f'Diagnosis checkpoint hash mismatch: {name}')
        checkpoints.append(json.loads(path.read_text()))
    if sum(len(item['evaluations'][0]['pitches']) for item in checkpoints) != 563:
        raise ValueError('Expected 563 aligned pitch states')
    return diagnosis_path, diagnosis, checkpoints


def compare_pitch(item, index, epsilon_pp):
    evaluations = item['evaluations']
    expected = ['ensemble'] + [f'member_seed_{s}' for s in range(42, 47)]
    if [e['variant'] for e in evaluations] != expected:
        raise ValueError('Member identity/order changed')
    actions = evaluations[0]['actions']
    if any(e['actions'] != actions for e in evaluations):
        raise ValueError('Action support/order differs across members')
    records = [e['pitches'][index] for e in evaluations]
    key = records[0]['pitch_key']
    if any(r['pitch_key'] != key for r in records):
        raise ValueError('Pitch key differs across members')
    q = np.asarray([r['q_values'] for r in records], float)
    if q.shape != (6, len(actions)) or len(actions) == 0 or not np.isfinite(q).all():
        raise ValueError('Invalid full-action Q arrays')
    if any(r['q_by_action'] != records[0]['q_by_action'] for r in records):
        # Values should differ, but the action keys must agree.
        if any(set(r['q_by_action']) != set(records[0]['q_by_action']) for r in records):
            raise ValueError('Action/Q keys differ across members')
    repertoire = item['repertoire_counts']
    frozen = choose_best(q[0])
    candidates = {'near_value_repertoire': choose_near_value(q[0], actions, repertoire, epsilon_pp),
                  'minimax_member_regret': choose_minimax(q[1:], q[0])}
    prefix = {'pitch_key': key, 'game_pk': key[0], 'at_bat_number': key[1],
              'pitcher': records[0]['pitcher'], 'balls': records[0]['balls'],
              'strikes': records[0]['strikes'], 'actual_type': records[0]['actual_type'],
              'supported_actions': len(actions), 'support_unchanged': True,
              'frozen_index': frozen, 'frozen_action': actions[frozen]}
    out = []
    for name, index_selected in candidates.items():
        member_regret = 100*(q[1:].max(axis=1)-q[1:, index_selected])
        row = prefix | {'candidate': name, 'chosen_index': index_selected,
                        'chosen_action': actions[index_selected],
                        'changed_action': index_selected != frozen,
                        'changed_type': actions[index_selected]['pitch_type'] != actions[frozen]['pitch_type'],
                        'top_type': actions[index_selected]['pitch_type'],
                        'ensemble_delta_pp': float(100*(q[0, index_selected]-q[0, frozen])),
                        'worst_member_regret_pp': float(member_regret.max()),
                        'member_delta_pp': {f'seed{s}': float(100*(q[i, index_selected]-q[i, frozen]))
                                            for i, s in enumerate(range(42, 47), start=1)}}
        heldout = []
        for held in range(5):
            remaining = np.delete(q[1:], held, axis=0)
            reference_q = remaining.mean(axis=0)
            reference_action = choose_best(reference_q)
            choice = (choose_near_value(reference_q, actions, repertoire, epsilon_pp)
                      if name == 'near_value_repertoire' else choose_minimax(remaining, reference_q))
            heldout.append({'held_seed': 42+held, 'chosen_index': choice,
                            'chosen_action': actions[choice], 'reference_index': reference_action,
                            'heldout_delta_pp': float(100*(q[held+1, choice]-q[held+1, reference_action]))})
        row['heldout'] = heldout
        row['heldout_action_unanimous'] = len({x['chosen_index'] for x in heldout}) == 1
        row['heldout_type_unanimous'] = len({x['chosen_action']['pitch_type'] for x in heldout}) == 1
        out.append(row)
    return out


def summarize(rows, cfg):
    result = {}
    for name in [v['id'] for v in cfg['candidates']]:
        part = [r for r in rows if r['candidate'] == name]
        if len(part) != 563 or len({tuple(r['pitch_key']) for r in part}) != 563:
            raise ValueError('Candidate rows must cover the same 563 distinct pitches')
        counts = Counter(r['top_type'] for r in part)
        intervals = {'frozen_ensemble': interval_by_game(part, 'ensemble_delta_pp',
                     cfg['bootstrap_seed'], cfg['bootstrap_replicates'])}
        for seed in range(42, 47):
            field = f'seed{seed}_delta_pp'
            for row in part:
                row[field] = row['member_delta_pp'][f'seed{seed}']
            intervals[f'seed{seed}'] = interval_by_game(part, field,
                cfg['bootstrap_seed'], cfg['bootstrap_replicates'])
        heldout = [h | {'game_pk': r['game_pk']} for r in part for h in r['heldout']]
        heldout_by_seed = {f'seed{seed}': interval_by_game(
            [r | {'delta': h['heldout_delta_pp']} for r in part for h in r['heldout'] if h['held_seed'] == seed],
            'delta', cfg['bootstrap_seed'], cfg['bootstrap_replicates']) for seed in range(42, 47)}
        types = [r | {'top_type': r['top_type']} for r in part]
        result[name] = {'pitches': len(part),
            'changed_action_count': sum(r['changed_action'] for r in part),
            'changed_type_count': sum(r['changed_type'] for r in part),
            'actual_type_match_count_descriptive': sum(r['actual_type'] == r['top_type'] for r in part),
            'top_type_counts': dict(sorted(counts.items())),
            'type_concentration_by_pitcher_count': concentration(types),
            'support_unchanged': all(r['support_unchanged'] for r in part),
            'mean_worst_member_regret_pp': float(np.mean([r['worst_member_regret_pp'] for r in part])),
            'judge_current_action_delta_pp': intervals,
            'heldout_member_delta_vs_four_member_mean_control_pp': heldout_by_seed,
            'heldout_mean_delta_pp': float(np.mean([h['heldout_delta_pp'] for h in heldout])),
            'heldout_action_unanimity_rate': float(np.mean([r['heldout_action_unanimous'] for r in part])),
            'heldout_type_unanimity_rate': float(np.mean([r['heldout_type_unanimous'] for r in part])),
            'decision': ('reject_internal_ensemble_worsening' if intervals['frozen_ensemble']['mean'] < 0
                         else 'defer_no_identified_policy_evaluation')}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if cfg['candidate_limit'] != 2 or len(cfg['candidates']) != 2:
        raise ValueError('Exactly two frozen candidates required')
    dest = Path(cfg['artifact_dir'])
    if dest.exists():
        raise ValueError('Candidate artifact directory already exists')
    diagnosis_path, diagnosis, checkpoints = load_inputs(cfg, args.config)
    rows = [row for item in checkpoints for i in range(len(item['evaluations'][0]['pitches']))
            for row in compare_pitch(item, i, cfg['candidates'][0]['epsilon_pp'])]
    result = {'experiment_id': cfg['experiment_id'], 'parent_diagnosis': cfg['parent_diagnosis'],
              'scope': 'same current pre-pitch action, each frozen judge retains own optimized future PA continuation',
              'policy_ope': None, 'causal_policy_value': None,
              'candidates': summarize(rows, cfg), 'per_pitch': rows,
              'sha256': {'config': sha(args.config), 'diagnosis_result': sha(diagnosis_path),
                         'diagnosis_config': diagnosis['sha256']['config'],
                         'script': sha(__file__), 'checkpoints': diagnosis['checkpoint_sha256']},
              'frozen_member_artifacts': diagnosis['member_artifacts']}
    dest.mkdir(parents=True)
    (dest/'results.json').write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n')
    print(json.dumps({'result': str(dest/'results.json'),
                      'decisions': {k:v['decision'] for k,v in result['candidates'].items()}}))


if __name__ == '__main__':
    main()
