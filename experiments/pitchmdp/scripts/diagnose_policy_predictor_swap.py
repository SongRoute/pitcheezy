"""EXP-P17-001: does swapping the planner's predictor (G0 -> confirmed G15 mix) change the P3 law?

Additive diagnostic (D145 option 1, stages A and B).  No outcome, reward, WE result or events
kind is read: the inputs are the same pre-decision fields the sealed S3b tau-select stage read.

  A. connection probe : on the registered S2 probe rows the policy-path direct head and the
                        policy-path integrated mix must equal the sealed evaluation archives.
  B. law comparison   : on the sealed S3b June sample the planning Q is recomputed with the mix
                        predictor for every decision the sealed G0 ledger evaluated (same states,
                        same planning seed, same reference law and tau).  The G0 side is the
                        sealed ledger itself; a replay of its first decisions with the bound G0
                        must reproduce the recorded planning Q.

The mix predictor is used only here.  Nothing is registered as a policy identity, no ledger of
the validation chain is written, and the validated identity is not touched.

``--smoke`` limits the games and the replay to check the wiring; it writes to an output ending
``-smoke`` and neither computes the comparison nor writes the repository result.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'scripts'))
import numpy as np
from pitchmdp.data import hash_file
from pitchmdp.matrix_data import canonical_hash
from pitchmdp.matrix_models import MatrixModel
from pitchmdp.matrix_sharing import SharingPredictor
from pitchmdp.policy_artifacts import load_support_table, load_train_bc
from pitchmdp import policy_identity as pid
from pitchmdp import policy_requests as preq
from pitchmdp import policy_runtime as prt
from pitchmdp.rollout_policy import JointSimulator, RolloutImprovement, RowBudget, calibrated_conditional, kl_policy
from run_ml_benchmark import read_json, dump, validate_native_runtime
from run_ml_matrix import check_location, heavy_lock
from run_ml_delivery_pool import git_commit, peak
import run_policy_validation as rpv

EXPERIMENT = 'EXP-P17-001'
CELL = 'G0-global'
SOURCES = ('scripts/diagnose_policy_predictor_swap.py', 'scripts/run_policy_validation.py', 'pitchmdp/policy_runtime.py',
           'pitchmdp/policy_identity.py', 'pitchmdp/policy_requests.py', 'pitchmdp/policy_artifacts.py',
           'pitchmdp/rollout_policy.py', 'pitchmdp/matrix_policy.py', 'pitchmdp/matrix_models.py',
           'pitchmdp/matrix_sharing.py')
SMOKE = {'games': 3, 'replay_decisions': 20}
SAME, DIFFERENT = 'SAME_RECOMMENDATIONS', 'DIFFERENT_RECOMMENDATIONS'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def config_check(config, smoke=False):
    require(config['protocol'] == 'policy_predictor_swap_diagnostic_v1' and config['experiment_id'] == EXPERIMENT,
            'Diagnostic settings differ from registered protocol')
    rule = config['stop_rule']
    require(set(rule) >= {'mean_tv_below', 'top1_agreement_at_least'} and 0 < rule['mean_tv_below'] < 1
            and 0 < rule['top1_agreement_at_least'] <= 1, 'Registered stop rule required')
    require(config['mix']['direct_weight'] == .5, 'The mix weight is fixed at one half')
    for key in ('policy_registration', 'sealed_g0_run', 'mix_bundle', 'settings', 'probe', 'replay'):
        require(config.get(key), 'Registered input missing: ' + key)
    require(smoke or config['execution']['enabled'] is True, 'Execution not enabled in the registered config')
    return config


class MixPredictor:
    """Conditional predictor of the confirmed mix for the planner's simulator.

    Per sampled delivery: one half the direct head (current-pitch physics forced to zero, so it
    does not depend on the delivery) plus one half the G0 members given that delivery; per-seed
    May temperatures, five-seed mean, no frequency blend (the mix's June model weight is 1).
    Averaged over the action's 400 draws this is the evaluated mix by linearity.
    """
    def __init__(self, g0, direct_models, direct_temperatures, weight):
        require(len(direct_models) == len(direct_temperatures) == len(g0.models) == 5, 'Five members per component')
        self.g0, self.inputs = g0, g0.inputs
        self.direct, self.direct_temperatures, self.weight = list(direct_models), list(direct_temperatures), float(weight)

    def components(self, states, actions, physical):
        tokens, valid, context = self.inputs.arrays(states, actions, physical)
        zeroed = tokens.copy()
        zeroed[:, -1, :8] = 0
        neutral = np.full((len(states), 10), .1)  # unused at neural_weight 1; only its shape is checked
        g0 = calibrated_conditional(np.stack([m.logits((tokens, valid, context)) for m in self.g0.models]),
                                    self.g0.temperatures, neutral, 1.)
        direct = calibrated_conditional(np.stack([m.logits((zeroed, valid, context)) for m in self.direct]),
                                        self.direct_temperatures, neutral, 1.)
        return direct, g0

    def __call__(self, states, actions, physical):
        direct, g0 = self.components(states, actions, physical)
        return self.weight * direct + (1 - self.weight) * g0


def load_direct_members(bundle, clusters):
    """Direct-head checkpoints pinned by the frozen mix inventory (hash before load)."""
    models, temperatures = [], []
    for seed in range(5):
        entry, member = bundle['files'][f'seed{seed}_checkpoint'], bundle['members'][str(seed)]
        require(hash_file(Path(entry['path'])) == entry['sha256'] == member['model_sha256'], f'Direct checkpoint changed: {seed}')
        models.append(SharingPredictor(CELL, MatrixModel.load(Path(entry['path'])), clusters))
        temperatures.append(float(member['delivery_temperature']))
    return models, temperatures


def law(q, reference, mask, tau):
    """The P3 law exactly as policy_tau recomputes it from a recorded planning Q."""
    return kl_policy(np.where(mask, q, -np.inf), reference, mask, tau)


def decision_metrics(p_mix, p_g0, reference, mask):
    def kl(p):
        with np.errstate(divide='ignore', invalid='ignore'):
            return float(np.sum(p[mask] * np.log(p[mask] / reference[mask]), where=p[mask] > 0))
    return {'tv': .5 * float(np.abs(p_mix - p_g0).sum()), 'top1_same': bool(np.argmax(p_mix) == np.argmax(p_g0)),
            'kl_mix_reference': kl(p_mix), 'kl_g0_reference': kl(p_g0)}


def summarize(rows, games, strikes, rule, *, draws, seed):
    """Point summaries, a whole-game bootstrap for the two registered quantities, and the stop rule."""
    tv = np.array([r['tv'] for r in rows])
    same = np.array([r['top1_same'] for r in rows], dtype=float)
    games, strikes = np.asarray(games), np.asarray(strikes)
    unique, index = np.unique(games, return_inverse=True)
    sums = np.stack([np.bincount(index, weights=v, minlength=len(unique)) for v in (tv, same, np.ones(len(tv)))])
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(unique), (draws, len(unique)))
    boot = sums[:, picks].sum(axis=2)
    interval = lambda k: [float(v) for v in np.quantile(boot[k] / boot[2], (.025, .975))]
    two = strikes == 2
    part = lambda m: {'decisions': int(m.sum()), 'mean_tv': float(tv[m].mean()), 'top1_agreement': float(same[m].mean())}
    result = {'decisions': len(tv), 'games': len(unique), 'mean_tv': float(tv.mean()), 'mean_tv_ci95': interval(0),
              'top1_agreement': float(same.mean()), 'top1_agreement_ci95': interval(1),
              'tv_quantiles': {str(q): float(np.quantile(tv, q)) for q in (.5, .9, .99, 1.)},
              'share_tv_above_0.05': float((tv > .05).mean()),
              'mean_kl_mix_reference': float(np.mean([r['kl_mix_reference'] for r in rows])),
              'mean_kl_g0_reference': float(np.mean([r['kl_g0_reference'] for r in rows])),
              'by_count': {'two_strikes': part(two), 'other': part(~two)}}
    passed = result['mean_tv'] < rule['mean_tv_below'] and result['top1_agreement'] >= rule['top1_agreement_at_least']
    result['stop_rule'] = {**rule, 'status': SAME if passed else DIFFERENT, 'basis': 'point values (registered)'}
    return result


def identity(config, local_path):
    return {'config_sha256': hash_file(REPO / 'configs' / (EXPERIMENT + '.yaml')),
            'local_config_sha256': hash_file(local_path), 'git_commit': git_commit(),
            'source_hashes': {rel: hash_file(PROJECT / rel) for rel in SOURCES}}


def bind(reg, inputs, blocks, artifact, snapshot=None, as_of=None):
    ident = reg['config']['identity_registration']
    contexts, _ = rpv.pa_contexts(inputs['store'], blocks, snapshot, as_of)
    components = pid.bind_components(inputs['bundle_path'], inputs['bundle_sha'], inputs['paths'], bc_artifact=artifact,
                                     context_rows=contexts, member_loader=inputs['member_loader'], classes=ident['classes'],
                                     we_contract_sha256=ident['we_contract_sha256'])
    require(components.sha256 == rpv.registered_json(reg, 'bind_identity')['sha256'],
            'Bound components differ from the S2-certified identity')
    return components


def probe(config, reg, inputs, bundle, artifact, direct, temperatures, no_pitch):
    """Stage A: policy path vs sealed evaluation archives on the registered S2 probe rows (no labels)."""
    spec, store = config['probe'], inputs['store']
    sealed = rpv.sealed_probe_arrays(inputs)
    chosen, positions = rpv.probe_positions(sealed['keys'], sealed['levels'], store.frame, spec['rows'])
    blocks = rpv.blocks_containing(store.frame, positions)
    components = bind(reg, inputs, blocks, artifact)  # rolling priors, as the S2 stage
    requests = {}
    for _, block in blocks:
        built, problem, _ = preq.pa_requests(store, block, 'probe', no_pitch)
        offsets = [int(p - block[0]) for p in positions if block[0] <= p <= block[-1]]
        require(len(built) > max(offsets), f'Probe PA unsubmittable at or before a probe row: {problem}')
        requests.update(zip(block.tolist(), built))
    states = [requests[int(p)].state for p in positions]
    actions = [requests[int(p)].logged_action for p in positions]
    mix = MixPredictor(components.g0, direct, temperatures, config['mix']['direct_weight'])
    archived_direct, archived_g0 = [], []
    manifest = pid.pinned_json(inputs['paths']['p11_manifest'], inputs['files']['p11_manifest']['sha256'])
    for seed in range(5):
        state = bundle['files'][f'seed{seed}_mlb_prediction_state']
        archive = Path(state['path']).parent / 'mlb_predictions.npz'
        require(hash_file(Path(state['path'])) == state['sha256']
                and hash_file(archive) == read_json(Path(state['path']))['artifact_hashes']['mlb_predictions.npz'],
                f'Direct whole-MLB archive changed: {seed}')
        with np.load(archive, allow_pickle=False) as data:
            require(np.array_equal(data['mlb_dev_keys'], sealed['keys']), 'Direct archive keys differ from the sealed probe keys')
            archived_direct.append(data['mlb_dev'][chosen])
        member = inputs['paths']['p11_preparation'].parent / 'members' / CELL / f'seed{seed}' / 'predictions.npz'
        g0_member = rpv.pinned_npz(str(member), manifest['inputs'][str(member)], ('dev_keys', 'dev'))
        require(np.array_equal(g0_member['dev_keys'], sealed['keys']), 'G0 member archive keys differ')
        archived_g0.append(g0_member['dev'][chosen])
    expected_direct, expected_g0 = np.mean(archived_direct, 0), np.mean(archived_g0, 0)
    weight = config['mix']['direct_weight']
    zeros = np.zeros((len(states), 8), dtype=np.float32)
    direct_now, _ = mix.components(states, actions, zeros)
    report = {'rows': spec['rows'], 'sealed_indices': chosen.tolist(),
              'g0_primary': pid.compare_probe(pid.integrated_predictions(components.g0, states, actions),
                                              sealed['primary'][chosen], spec['atol']),
              'direct': pid.compare_probe(direct_now, expected_direct, spec['atol']),
              'mix_integrated': pid.compare_probe(pid.integrated_predictions(mix, states, actions),
                                                  weight * expected_direct + (1 - weight) * expected_g0, spec['atol']),
              'labels_read': False}
    report['pass'] = all(report[k]['pass'] for k in ('g0_primary', 'direct', 'mix_integrated'))
    return report


def sealed_ledger(config):
    spec = config['sealed_g0_run']
    root = Path(spec['path'])
    manifest = read_json(root / 'manifest.json')
    require(hash_file(root / 'manifest.json') == spec['manifest_sha256'] and manifest['command'] == 'tau-select',
            'Sealed G0 tau-select run changed')
    require(hash_file(root / spec['ledger']) == manifest['artifact_sha256'][spec['ledger']], 'Sealed G0 ledger changed')
    header, rows = None, {}
    with (root / spec['ledger']).open() as stream:
        for line in stream:
            record = json.loads(line)
            if record['kind'] == 'header':
                header = record
            elif record['kind'] == 'decision':
                rows[record['request_id']] = record
    require(header is not None and len(rows) == spec['requests'], 'Sealed ledger row count differs from registration')
    return root, header, rows


def run(config, local, local_path, output, smoke):
    started = time.perf_counter()
    reg = rpv.load_registration(REPO / config['policy_registration']['config'],
                                [REPO / path for path in config['policy_registration']['addenda']])
    require(reg['config_sha256'] == config['policy_registration']['config_sha256'], 'Policy registration config changed')
    policy, settings = reg['config'], config['settings']
    freeze = rpv.registered_json(reg, 'tau_freeze')
    root, header, ledger = sealed_ledger(config)
    require(freeze['final_identity_sha256'] == config['sealed_g0_run']['final_identity_sha256']
            and hash_file(root / 'tau_freeze.json') == rpv.registered_path(reg, 'tau_freeze')[1],
            'Sealed run is not the registered tau freeze')
    require(freeze['selected_tau'] == settings['tau'] and freeze['setting'] == {'samples': settings['samples'],
            'pitch_cap': settings['pitch_cap']} and policy['seeds']['planning_main'] == settings['seed'],
            'Diagnostic settings differ from the sealed G0 run')
    bundle_path = REPO / config['mix_bundle']['path']
    require(hash_file(bundle_path) == config['mix_bundle']['sha256'], 'Frozen mix inventory changed')
    bundle = read_json(bundle_path)
    require(bundle['definition']['direct_weight'] == config['mix']['direct_weight']
            and bundle['definition']['june_frequency_blend_model_weight'] == 1.0
            and bundle['g0_component']['bundle_sha256'] == policy['identity_registration']['g0_bundle']['file_sha256'],
            'Frozen mix differs from the registered definition')
    output.mkdir(parents=True, exist_ok=False)
    dump(output / 'started.json', {'identity': identity(config, local_path), 'registration_chain': reg['chain'],
                                   'started_utc': datetime.now(timezone.utc).isoformat(), 'smoke': smoke,
                                   'outcomes_read': False})
    inputs = rpv.load_inputs(policy, local)
    frame, store = inputs['frame'], inputs['store']
    no_pitch = frozenset(policy['pa_time_rules']['R3_codes']['no_pitch_descriptions'])
    artifact = load_train_bc(*rpv.registered_path(reg, 'bc'))
    direct, temperatures = load_direct_members(bundle, inputs['prep']['clusters'])
    report = probe(config, reg, inputs, bundle, artifact, direct, temperatures, no_pitch)
    dump(output / 'probe.json', report)
    require(report['pass'], 'Connection probe failed; nothing else is computed')
    print('PREDICTOR_SWAP_PROBE_PASS', {k: report[k]['max_abs_difference'] for k in ('g0_primary', 'direct', 'mix_integrated')}, flush=True)

    spec = policy['le2025_validation_plan']['stages']['S3b_tau_select']
    games = rpv.select_games(frame, 'blend', spec['n_games'], rpv.role_seed(policy, 'selection_salt'))
    require(games == freeze['games'], 'Selected June games differ from the sealed run')
    if smoke:
        games = games[:SMOKE['games']]
    blocks = rpv.game_blocks(frame, games, 'blend')
    snapshot, as_of, _ = rpv.load_style_snapshot(*rpv.registered_path(reg, 'style_blend'))
    components = bind(reg, inputs, blocks, artifact, snapshot, as_of)
    support, _ = load_support_table(*rpv.registered_path(reg, 'support'), artifact)
    reference = prt.MaskedReference(artifact.bc, support)
    search = dict(samples=settings['samples'], pitch_cap=settings['pitch_cap'], seed=settings['seed'])
    we, budget = components.we, int(config['row_budget'])
    mix = MixPredictor(components.g0, direct, temperatures, config['mix']['direct_weight'])
    planner_mix = RolloutImprovement(reference, JointSimulator(components.inputs.pool, mix, we.terminal,
                                     RowBudget(budget, seed_count=10)), we.cutoff, **search)
    planner_g0 = RolloutImprovement(reference, JointSimulator(components.inputs.pool, components.g0, we.terminal,
                                    RowBudget(budget, seed_count=5)), we.cutoff, **search)
    replay_limit = SMOKE['replay_decisions'] if smoke else config['replay']['decisions']
    replay, rows, row_games, row_strikes, request_ids, seen = [], [], [], [], [], 0
    tau = settings['tau']
    for pa_id, positions in blocks:
        requests, _, _ = preq.pa_requests(store, positions, header['runtime_sha256'], no_pitch)
        for request in requests:
            sealed = ledger.get(request.request_id)
            # The runtime records hash(request fingerprint, bound context row hash): state, history physics,
            # logged label and the context row are all the sealed ones or this refuses.
            fingerprint = canonical_hash({'request': request.fingerprint(),
                                          'context_sha256': components.context_sha256(request.state)})
            require(sealed is not None and sealed['fingerprint'] == fingerprint,
                    f'Request differs from the sealed ledger: {request.request_id}')
            seen += 1
            if sealed['status'] not in prt.EVALUATED:
                continue
            state, result = request.state, sealed['result']
            mask, ref = reference.support(state), reference.probabilities(state)
            require(np.array_equal(mask, np.asarray(result['mask'], dtype=bool))
                    and np.allclose(ref, result['reference'], rtol=0, atol=1e-12)
                    and components.context_sha256(state) == sealed['context_sha256'],
                    f'Reference law or context differs from the sealed ledger: {request.request_id}')
            q_sealed = np.array([np.nan if v is None else v for v in result['q_planning']], dtype=np.float64)
            if len(replay) < replay_limit:
                q_again, _ = planner_g0.q_values(state)
                replay.append(float(np.abs(q_again[mask] - q_sealed[mask]).max()))
                require(replay[-1] <= config['replay']['atol'], f'G0 replay differs from the sealed planning Q: {request.request_id}')
            q_mix, _ = planner_mix.q_values(state)
            rows.append({**decision_metrics(law(q_mix, ref, mask, tau), law(q_sealed, ref, mask, tau), ref, mask),
                         'max_abs_q_difference': float(np.abs(q_mix[mask] - q_sealed[mask]).max())})
            row_games.append(int(frame.game_pk.iloc[positions[0]]))
            row_strikes.append(int(state.strikes))
            request_ids.append(request.request_id)
    components.verify()
    require(smoke or seen == len(ledger), 'Not every sealed request was regenerated')
    require(len(replay) == replay_limit, 'Replay did not reach the registered decision count')
    evaluated = sum(r['status'] in prt.EVALUATED for r in ledger.values())
    require(smoke or len(rows) == evaluated == config['sealed_g0_run']['evaluated_decisions'], 'Evaluated decision count differs')
    require(all(np.isfinite(list(r.values())).all() for r in rows), 'Non-finite comparison value')
    summary = None
    if not smoke:  # the comparison is opened only by the full registered run
        np.savez_compressed(output / 'decisions.npz', request_id=np.array(request_ids), game_pk=np.array(row_games),
                            strikes=np.array(row_strikes), **{k: np.array([r[k] for r in rows]) for k in rows[0]})
        summary = summarize(rows, row_games, row_strikes, config['stop_rule'], draws=config['bootstrap']['draws'],
                            seed=config['bootstrap']['seed'])
    result = {'experiment_id': EXPERIMENT, 'smoke': smoke, 'settings': settings, 'mix': config['mix'],
              'sealed_g0_run': config['sealed_g0_run'], 'mix_bundle': config['mix_bundle'], 'probe': report,
              'replay': {'decisions': len(replay), 'max_abs_q_difference': max(replay), 'atol': config['replay']['atol']},
              'requests_regenerated': seen, 'decisions_compared': len(rows), 'comparison': summary,
              'mean_max_abs_q_difference': None if smoke else float(np.mean([r['max_abs_q_difference'] for r in rows])),
              'cost': {'seconds': time.perf_counter() - started, 'peak_rss_bytes': peak(),
                       'mix_conditional_rows': planner_mix.simulator.budget.conditional_rows,
                       'g0_replay_conditional_rows': planner_g0.simulator.budget.conditional_rows},
              'git_commit': git_commit(), 'outcomes_read': False, 'policy_identity_registered': False,
              'limits': ['Model-internal diagnostic of the planning law; not a policy value estimate',
                         'June 2025 rows already used for tau design (exposed); no outcomes are read here',
                         'The stop-rule thresholds are judgement values without a measured basis (registered before the run)',
                         'Same-model review only; the independent review gate is not passed'],
              'finished_utc': datetime.now(timezone.utc).isoformat()}
    dump(output / 'results.json', result)
    if not smoke:
        repo_result = REPO / 'results' / (EXPERIMENT + '.json')
        require(not repo_result.exists(), 'Preserve the completed repository result')
        shutil.copyfile(output / 'results.json', repo_result)
    print('PREDICTOR_SWAP_COMPLETE', {'decisions': len(rows), 'smoke': smoke, 'seconds': round(result['cost']['seconds'], 1)},
          '' if smoke else {k: summary[k] for k in ('mean_tv', 'top1_agreement')}, '' if smoke else summary['stop_rule']['status'],
          flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--local-config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--smoke', action='store_true', help='three games and a short replay; output must end -smoke')
    args = p.parse_args()
    config, local = config_check(read_json(args.config), args.smoke), read_json(args.local_config)
    output = args.output.resolve()
    require(args.smoke == output.name.endswith('-smoke'), 'Smoke runs write only to an output ending -smoke, and only smoke runs may')
    require(args.smoke or output.name == EXPERIMENT, 'Output directory must be named after the experiment')
    require(args.config.resolve() == (REPO / 'configs' / (EXPERIMENT + '.yaml')).resolve(), 'Use the registered config file')
    root = check_location(local, output)
    validate_native_runtime()
    with heavy_lock(root):
        run(config, local, args.local_config.resolve(), output, args.smoke)


if __name__ == '__main__':
    main()
