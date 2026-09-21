"""Two-stage bounded, disjoint-fit evaluation of the frozen pitch CLI MVP."""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import pickle
import shutil
import sys
import time
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import numpy as np
import pandas as pd
from pitchmdp.archetypes import add_batter_style_history
from pitchmdp.data import KEY, hash_file
from pitchmdp.sequence_data import prepare_frame
from pitchmdp.model import eligible, outcome_labels, CountBaseline
from pitchmdp.sequence_model import classification_metrics
from pitchmdp.game import WinExpectancy, EmpiricalAdvancement, terminal_values
from pitchmdp.planner import solve_pa
from minimal_pitch_service import Engine
from build_minimal_pitch_service import request_for, re_terminals
from diagnose_sequence_legality import condition_on_legality
from policy_evaluation_models import JulyEvaluator, OutcomeCalibration, advancement_metrics
from run_sequence_pilot import dump


PERIODS = {'judge': ('2025-07-01', '2025-07-31'), 'diagnosis': ('2025-08-01', '2025-08-07'),
           'guard': ('2025-08-08', '2025-08-15'), 'evaluation': ('2025-08-16', '2025-09-30')}


def partition(frame):
    parts = {name: frame.loc[pd.to_datetime(frame.game_date).between(*period)].copy() for name, period in PERIODS.items()}
    sets = [set(part.game_pk) for part in parts.values()]
    if any(not len(part) for part in parts.values()) or any(a & b for i, a in enumerate(sets) for b in sets[i+1:]):
        raise ValueError('Empty period or overlapping games')
    return parts


def legal(p, frame):
    p = np.asarray(p, float)
    return condition_on_legality(p/p.sum(axis=-1, keepdims=True), (frame.outs_when_up.eq(2) | frame.bases.eq(0)).to_numpy())


def select_policy_cases(frame):
    starts = frame.loc[frame.pitch_number.eq(1) & frame.balls.eq(0) & frame.strikes.eq(0)]
    return starts.sort_values(['game_date', *KEY]).groupby(['pitcher', 'game_pk'], sort=False).head(5)


def source_manifest():
    paths = [Path(__file__).resolve(), PROJECT/'scripts/policy_evaluation_models.py', PROJECT/'docs/SERVICE_VALIDATION_PROTOCOL.md']
    return {str(path.relative_to(PROJECT)): hash_file(path) for path in paths}


def verify_sources(output):
    for name, expected in json.loads((output/'source_hashes.json').read_text()).items():
        if hash_file(PROJECT/name) != expected or hash_file(output/'source'/name) != expected:
            raise ValueError('Stage source changed: '+name)


def verify_files(output, manifest):
    for name, expected in json.loads((output/manifest).read_text()).items():
        if hash_file(output/name) != expected:
            raise ValueError('Stage artifact changed: '+name)


def file_manifest(output, destination):
    dump(output/destination, {str(path.relative_to(output)): hash_file(path) for path in output.rglob('*')
                             if path.is_file() and path.name != destination})


def diagnose(root, output):
    if output.exists():
        raise ValueError('Use a new run directory')
    engine = Engine(root/'runs/minimal-pitch-service-v1')
    # Pin the implementation behind the frozen weights as well as the weight bundle.
    base = engine.bundle
    for rel, digest in json.loads((base/'source_hashes.json').read_text()).items():
        if hash_file(PROJECT/rel) != digest:
            raise ValueError('Frozen MVP source changed: '+rel)
    local = json.loads((PROJECT/'configs/local.json').read_text())
    frame = add_batter_style_history(prepare_frame(local))
    history = root/'runs/history-batter-validation-v1'
    if frame.attrs['sequence_data_identity'] != json.loads((history/'data_identity.json').read_text()):
        raise ValueError('Frozen input identity differs')
    parts = partition(frame)
    original_fit_games = set(frame.loc[pd.to_datetime(frame.game_date).le('2025-06-30'), 'game_pk'])
    if original_fit_games & set(parts['judge'].game_pk):
        raise ValueError('Evaluator and original fit/calibration games overlap')
    output.mkdir(parents=True)
    sources = source_manifest()
    dump(output/'source_hashes.json', sources)
    for rel in sources:
        target = output/'source'/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT/rel, target)
    dump(output/'config.json', {'periods': PERIODS, 'base_bundle': str(base), 'sources': sources,
         'data_identity': frame.attrs['sequence_data_identity'],
         'created_utc': datetime.now(timezone.utc).isoformat(),
         'base_manifest_sha256': hash_file(base/'bundle_manifest.json'),
         'scope': 'Evaluator fitting games disjoint from policy fitting/calibration. Previously explored historical dates; not unseen confirmation or causal OPE.'})
    cohort = set(map(int, engine.metadata['pitchers']))
    splits = {}
    for name, part in parts.items():
        supported = part.loc[eligible(part)]
        target = supported.loc[supported.pitcher.isin(cohort) & supported.pitcher.eq(supported.starter_pitcher)]
        splits[name] = {'league_rows': len(part), 'league_games': int(part.game_pk.nunique()),
                        'eligible_rows': len(supported), 'cohort_rows': len(target), 'cohort_games': int(target.game_pk.nunique())}
        part[KEY+['game_date']].to_parquet(output/(name+'_all_keys.parquet'), index=False)
        if name != 'judge':
            target.to_parquet(output/(name+'.parquet'), index=False)
        if name == 'evaluation':
            WinExpectancy.labeled_states(part).to_parquet(output/'evaluation_we_states.parquet', index=False)
            part.loc[part.is_pa_terminal.fillna(False)].to_parquet(output/'evaluation_terminals.parquet', index=False)
    dump(output/'splits.json', splits)
    train = parts['judge'].assign(split='train')
    fit_rows = train.loc[eligible(train)]
    print('FIT_JULY_EVALUATOR', len(fit_rows), flush=True)
    judge = JulyEvaluator().fit(fit_rows)
    count = CountBaseline().fit(fit_rows)
    we, advancement = WinExpectancy().fit(train), EmpiricalAdvancement().fit(train)
    constant = float(we.labeled_states(train).final_home_win.mean())
    with (output/'judge.pkl').open('wb') as stream:
        pickle.dump({'outcomes': judge, 'count': count, 'we': we, 'advancement': advancement, 'constant': constant}, stream)
    dump(output/'judge_fit.json', {'outcomes': judge.report, 'we': we.training_report, 'advancement': advancement.report})
    with np.load(history/'2025/predictions.npz') as archive:
        mapping = {tuple(key): i for i, key in enumerate(archive['pitch_keys'])}
        raw_p, archived_y = archive['continuous_h0'], archive['y']
        for name in ('diagnosis', 'guard', 'evaluation'):
            selected = pd.read_parquet(output/(name+'.parquet'))
            indices = [mapping[tuple(key)] for key in selected[KEY].to_numpy()]
            y = outcome_labels(selected)
            np.testing.assert_array_equal(y, archived_y[indices])
            np.savez_compressed(output/(name+'_predictions.npz'), p=legal(raw_p[indices], selected), y=y,
                                game_pk=selected.game_pk.to_numpy(), pitch_keys=selected[KEY].to_numpy())
    diagnostics = {}
    for name in ('diagnosis', 'guard'):
        part = parts[name]
        target = pd.read_parquet(output/(name+'.parquet'))
        with np.load(output/(name+'_predictions.npz')) as saved:
            y, p = saved['y'], saved['p']
        observed, predicted = np.bincount(y, minlength=10)/len(y), p.mean(0)
        league = part.loc[eligible(part)]
        ly = outcome_labels(league)
        labeled = we.labeled_states(part)
        diagnostics[name] = {'mvp_outcomes': classification_metrics(y, p),
            'class_calibration_gap': (predicted-observed).tolist(),
            'max_absolute_class_gap': float(np.max(np.abs(predicted-observed))),
            'judge_outcomes': classification_metrics(ly, legal(judge.predict(league), league)),
            'judge_count_baseline': classification_metrics(ly, legal(count.predict(league), league)),
            'judge_we': we.evaluate(part), 'mvp_we': engine.we.evaluate(part),
            'constant_we_brier': float(np.mean((labeled.final_home_win.to_numpy(float)-constant)**2)),
            'judge_advancement': advancement_metrics(advancement, part),
            'mvp_advancement': advancement_metrics(engine.advancement, part)}
    diagnostics['repair_eligible'] = bool(diagnostics['diagnosis']['max_absolute_class_gap'] >= .01 and
                                         diagnostics['diagnosis']['mvp_outcomes']['n'] >= 100)
    diagnostics['repair_rule'] = 'Only one fixed class-bias calibration if eligible; fit diagnosis, accept on guard LL decrease and Brier nonincrease.'
    dump(output/'diagnostics.json', diagnostics)
    file_manifest(output, 'diagnostic_hashes.json')
    print('DIAGNOSIS_COMPLETE', output, flush=True)


def grid_from_data(data):
    rows = []
    state = data['state']
    for balls in range(4):
        for strikes in range(3):
            for action in data['pitch_types']:
                rows.append({'balls': balls, 'strikes': strikes, 'pitch_type': action, 'pitcher': data['pitcher_id'],
                    'stand': data['batter_stand'], 'p_throws': data['p_throws'], 'outs_when_up': state.outs,
                    'bases': state.bases, **data['profile']})
    return pd.DataFrame(rows)


def compare_policies(p, judge_p, original_we, original_re, judge_we, usage):
    nxt = [0]*p.shape[3]
    we_plan = solve_pa(p, original_we, nxt, usage)
    re_plan = solve_pa(p, original_re, nxt, usage)
    reference = solve_pa(judge_p, judge_we, nxt, usage)
    one_action = int(we_plan.myopic_q_values[0, 0, 0].argmax())
    result = {'frequency': float(reference.baseline_values[0, 0, 0]),
              'we_one': float(reference.myopic_q_values[0, 0, 0, one_action])}
    for name, plan in [('we_full', we_plan), ('re_full', re_plan)]:
        evaluated = solve_pa(judge_p, judge_we, nxt, np.eye(p.shape[3])[plan.policy])
        result[name] = float(evaluated.baseline_values[0, 0, 0])
    return result


def summarize(cases):
    games, inverse = np.unique([case['game_pk'] for case in cases], return_inverse=True)
    counts = np.bincount(inverse)
    draws = np.random.default_rng(42).integers(0, len(games), (2000, len(games)))
    result = {'cases': len(cases), 'games': len(games), 'comparisons': {}}
    for name in ('original', 'selected'):
        result['comparisons'][name] = {}
        for right in ('frequency', 'we_one', 're_full'):
            delta = 100*np.asarray([case[name]['we_full']-case[name][right] for case in cases])
            boot = np.bincount(inverse, weights=delta)[draws].sum(1)/counts[draws].sum(1)
            alpha = .025 if right == 're_full' else .0125
            result['comparisons'][name]['we_full_minus_'+right] = {'we_percentage_points': float(delta.mean()),
                'game_interval': np.quantile(boot, [alpha, 1-alpha]).tolist(), 'coverage': 1-2*alpha}
    delta = 100*np.asarray([case['selected']['we_full']-case['original']['we_full'] for case in cases])
    boot = np.bincount(inverse, weights=delta)[draws].sum(1)/counts[draws].sum(1)
    result['selected_minus_original'] = {'we_percentage_points': float(delta.mean()), 'game95': np.quantile(boot, [.025, .975]).tolist()}
    return result


def finalize(root, output, repair):
    verify_sources(output)
    verify_files(output, 'diagnostic_hashes.json')
    if (output/'results.json').exists():
        raise ValueError('Final evaluation already completed')
    engine = Engine(root/'runs/minimal-pitch-service-v1')
    config = json.loads((output/'config.json').read_text())
    if hash_file(engine.bundle/'bundle_manifest.json') != config['base_manifest_sha256']:
        raise ValueError('Base bundle changed')
    diagnostics = json.loads((output/'diagnostics.json').read_text())
    candidate = OutcomeCalibration()
    accepted = False
    repair_result = {'requested': repair, 'eligible': diagnostics['repair_eligible']}
    if repair == 'class_bias':
        if not diagnostics['repair_eligible']:
            raise ValueError('No diagnosed calibration weakness meets fixed trigger')
        with np.load(output/'diagnosis_predictions.npz') as data:
            candidate.fit(data['p'], data['y'], pd.read_parquet(output/'diagnosis.parquet').game_date)
        with np.load(output/'guard_predictions.npz') as data:
            before = classification_metrics(data['y'], data['p'])
            after = classification_metrics(data['y'], candidate.apply(data['p']))
        accepted = after['log_loss'] < before['log_loss'] and after['brier_multiclass'] <= before['brier_multiclass']
        repair_result.update(guard_original=before, guard_candidate=after, bias=candidate.bias.tolist())
    from refresh_pitch_service import metadata_hash
    calibration = {'accepted': accepted, 'bias': candidate.bias.tolist(), 'available_from': '2025-08-16',
                   'original_metadata_sha256': metadata_hash(engine.metadata)}
    dump(output/'calibration.json', calibration)
    repair_result['accepted'] = accepted
    # The selection is written before reading final labels or evaluating final policies.
    dump(output/'selection.json', repair_result)
    with (output/'judge.pkl').open('rb') as stream:
        judges = pickle.load(stream)
    final = pd.read_parquet(output/'evaluation.parquet')
    states = pd.read_parquet(output/'evaluation_we_states.parquet')
    terminals = pd.read_parquet(output/'evaluation_terminals.parquet')
    with np.load(output/'evaluation_predictions.npz') as saved:
        y, original = saved['y'], saved['p']
        selected_p = candidate.apply(original) if accepted else original
        prediction = {'original': classification_metrics(y, original), 'selected': classification_metrics(y, selected_p),
                      'candidate_even_if_rejected': classification_metrics(y, candidate.apply(original))}
    judge_p = legal(judges['outcomes'].predict(final), final)
    evaluator_metrics = {'outcomes': classification_metrics(y, judge_p),
        'count': classification_metrics(y, legal(judges['count'].predict(final), final)),
        'we': judges['we'].evaluate(states), 'advancement': advancement_metrics(judges['advancement'], terminals)}
    with (engine.bundle/'game_values.pkl').open('rb') as stream:
        re_table = pickle.load(stream)['run_expectancy']
    cases_frame = select_policy_cases(final)
    if not len(cases_frame):
        raise ValueError('No final supported cases')
    cases_frame[KEY+['pitcher', 'game_date']].to_parquet(output/'policy_case_keys.parquet', index=False)
    cases = []
    all_judge_support = []
    for _, row in cases_frame.iterrows():
        request = request_for(row)
        request['date'] = str(pd.Timestamp(row.game_date).date())
        request['batter_profile']['as_of'] = str((pd.Timestamp(row.game_date)-pd.Timedelta(days=1)).date())
        data = engine.predict_counts(request)
        grid = grid_from_data(data)
        prediction_p, support, origin = judges['outcomes'].predict_with_support(grid)
        p = data['probabilities']['blend']
        evaluator = legal(prediction_p, grid).reshape(p.shape)
        all_judge_support.extend(support.tolist())
        usage = np.array([engine.metadata['repertoire_counts'][str(data['pitcher_id'])][a] for a in data['pitch_types']], float)
        usage /= usage.sum()
        source_we = terminal_values(data['state'], engine.we, engine.advancement)
        source_re = re_terminals(data['state'], engine.advancement, re_table)
        judge_we = terminal_values(data['state'], judges['we'], judges['advancement'])
        old = compare_policies(p, evaluator, source_we, source_re, judge_we, usage)
        new = compare_policies(candidate.apply(p), evaluator, source_we, source_re, judge_we, usage) if accepted else old
        cases.append({'game_pk': int(row.game_pk), 'pitcher': int(row.pitcher), 'request': request,
                      'original': old, 'selected': new})
        if len(cases) == 1:
            dump(output/'example_request.json', request)
        if len(cases) % 25 == 0:
            print('POLICY_CASE', len(cases), '/', len(cases_frame), flush=True)
    guard = diagnostics['guard']
    gate = guard['judge_outcomes']['log_loss'] < guard['judge_count_baseline']['log_loss'] and guard['judge_we']['brier'] < guard['constant_we_brier']
    result = {'prediction': prediction, 'repair': repair_result, 'independent_evaluator': evaluator_metrics,
        'minimal_evaluator_gate_passed': bool(gate), 'policy': summarize(cases), 'cases': cases,
        'judge_deepest_cell_support': {'median': float(np.median(all_judge_support)),
                                     'less_than20_fraction': float(np.mean(np.asarray(all_judge_support) < 20))},
        'scope': 'Disjoint evaluator fitting games, shared historical profile inputs and model assumptions. '
                 'Exploratory previously seen period; support-filtered PA population. No causal policy or real win gain proof.'}
    dump(output/'results.json', result)
    # Freeze final selected calibration separately; prior stage files remain byte-for-byte intact.
    for rel in ('scripts/refresh_pitch_service.py',):
        target = output/'source'/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT/rel, target)
    file_manifest(output, 'completion_hashes.json')
    print('FINAL_COMPLETE', output, 'accepted_repair', accepted, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['diagnose', 'finalize'])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--repair', choices=['none', 'class_bias'], default='none')
    args = parser.parse_args()
    local = json.loads((PROJECT/'configs/local.json').read_text())
    root = Path(local['artifact_root']).resolve()
    output = (args.output or root/'runs/service-validation-v1').resolve()
    if not Path('/Volumes/T7 Shield').is_mount() or not root.is_relative_to('/Volumes/T7 Shield'):
        raise ValueError('Approved mounted SSD required')
    if not output.is_relative_to(root/'runs') or output == root/'runs' or Path(sys.executable).absolute() != Path(local['python']).absolute():
        raise ValueError('Use existing Python and a new approved run directory')
    started = time.perf_counter()
    if args.stage == 'diagnose':
        diagnose(root, output)
    else:
        finalize(root, output, args.repair)
    print('SECONDS', round(time.perf_counter()-started, 2), flush=True)


if __name__ == '__main__':
    main()
