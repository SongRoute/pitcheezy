"""Sequence-model SHADOW runner (job 20261012-seq-shadow-001): replay one captured game's stored ``feed/live`` bodies
in receipt order through the accepted Step 5 pipeline (pinned source, PLACEHOLDER heads) and, for every first
decision for which Step 5 built its 17x199 matrix, record the SEQ-001 N1 three-seed forecast (C0 ensemble and the
CAL2 selected-arm ensemble) in a separate append-only ledger. Nothing is shown to a fan; nothing is adopted.

What the Step 5 pipeline is used for here: decision timing, attempt keys, transient/abstain classification,
per-action support and ``reconcile_physical_pitches``. Its heads are ``shadow_runner.synthetic_heads`` in BOTH
modes (this interpreter has no lightgbm), so every ``seven_leaf`` this run writes into ``pitch_attempts`` /
``shadow_forecasts`` is a PLACEHOLDER AND MUST NOT BE READ. The frozen-2024 comparator numbers are read from the
Step 5 real ledger (read-only, sha256-pinned) and joined by attempt key.

Leakage rule. The forecast for physical pitch (PA, k) is computed from ONE body, the stored version whose last
received pitch of the open PA is k-1, immediately after ``live_decide`` returned ``appended`` for it and before the
next body is loaded: the pre-pitch state and the raw 91 columns from ``features`` (pinned), the context tokens from
earlier pitches in the same body (``pitcheezy.seq_shadow.tokens``). Realized type/outcome are appended to
``shadow_realized`` from the bodies that later show the pitch and joined only in the summary.

Gate V. Every attempt scored here and in the Step 5 ledger must carry the same ``vectors_sha256`` (identical state,
profile and raw columns) and the same ``written_at_replay_index``; a mismatch stops before any metric.

Exit codes: 0 completed; 2 refused (nothing written); 3 stopped by a cap (partial ledger + summary); 4 stopped by an
integrity gate (gate V or a vector hash that differs from the pipeline record; ledger + summary without metrics);
87 RSS watchdog (pinned guard).

  python -B scripts/seq_shadow_runner.py --replay-dir DIR --out-dir DIR --game PK --grant GRANT.json
  python -B scripts/seq_shadow_runner.py --synthetic --profile-dir FIXTURE --replay-dir FIXTURE --out-dir DIR
        --config TEMP.json --step5-ledger FIXTURE.sqlite3          (tests only)
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS',
              'MKL_NUM_THREADS'):
    os.environ[_name] = '1'  # before any numeric import

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
if str(REPO/'src') not in sys.path:
    sys.path.insert(0, str(REPO/'src'))

try:
    import pitcheezy.seq_shadow as SS  # noqa: E402
    from pitcheezy.seq_shadow import ledger as LED  # noqa: E402
    from pitcheezy.seq_shadow import summary as SUM  # noqa: E402
    from pitcheezy.seq_shadow import tokens as TOK  # noqa: E402
except ImportError as _exc:  # an interpreter without the registered packages: a refusal, nothing was read or written
    if __name__ != '__main__':
        raise
    print(json.dumps({'refused': f'import_failed_under_this_interpreter:{_exc}'}))
    sys.exit(2)

LEDGER_NAME = 'seq-shadow-ledger.sqlite3'
RC_OK, RC_REFUSED, RC_CAP, RC_GATE = 0, 2, 3, 4
PLACEHOLDER_NOTE = ('Step 5 heads in this run are shadow_runner.synthetic_heads (placeholder, no lightgbm in this '
                    'interpreter): every seven_leaf in pitch_attempts / shadow_forecasts of THIS ledger is a '
                    'placeholder and MUST NOT be read; the 2024 comparator is read from the pinned Step 5 ledger')


class Integrity(Exception):
    """An identity the run depends on does not hold: stop in order, write the summary without metrics."""


def load_seq(config):
    """The hash-checked N1 ensemble, CAL2 calibrators and scale-002 encoder. Raises ``PinMismatch`` (a ValueError)
    before any model byte is deserialized when a registered sha256 differs."""
    from pitcheezy.seq_shadow import adapter as AD
    from pitcheezy.seq_shadow import calibration as CAL
    from pitcheezy.seq_shadow import model as MOD
    encoder, encoder_sha = AD.load_seq_encoder(config)
    # calibrator files are hashed before the weights are deserialized; their temperatures are compared afterwards
    calibrators = CAL.load_calibrators(config, [int(s) for s in config['model']['seeds']])
    ensemble = MOD.load_ensemble(config)
    calibrators = CAL.load_calibrators(config, ensemble.seeds, ensemble.temperatures)
    return SimpleNamespace(adapter=AD.SeqAdapter(ensemble, calibrators, config['model']['candidate_rows']),
                           ensemble=ensemble, calibrators=calibrators, encoder=encoder, encoder_sha256=encoder_sha,
                           encode=AD.encode_for_seq)


def check_review(grant, config, own):
    """Problems of the independent review the grant names (empty = usable)."""
    path, bad = grant.get('review_path'), []
    if not isinstance(path, str) or not isinstance(grant.get('review_sha256'), str):
        return ['review_path_or_review_sha256']
    try:
        raw = Path(path).read_bytes()
    except OSError:
        return ['review_file_unreadable']
    if hashlib.sha256(raw).hexdigest() != grant['review_sha256']:
        bad.append('review_sha256')
    try:
        review = json.loads(raw)
    except ValueError:
        return bad+['review_not_json']
    if not isinstance(review, dict) or review.get('verdict') != 'ACCEPT':
        bad.append('review_verdict_is_not_ACCEPT')
    reviewer = review.get('reviewer') if isinstance(review, dict) else None
    if not isinstance(reviewer, str) or not reviewer.strip() or reviewer.strip() == config['author'].strip():
        bad.append('reviewer_absent_or_the_author')
    if isinstance(review, dict) and 'source_sha256' in review and review['source_sha256'] != own:
        bad.append('review_names_other_source_bytes')
    return bad


def check_grant(grant, *, replay, out, game_pk, config, step5_config, step5_shadow_sha256, executable):
    """Problems of a real-run grant (empty list = usable). The grant is the lead's file; this runner never writes
    one. Every hash it names must equal the registered configuration and the bytes on disk."""
    if not isinstance(grant, dict):
        return ['grant_not_an_object']
    own = SS.own_source_sha256()
    modules = {name: sha for name, sha in step5_shadow_sha256.items() if name.endswith('.py')}
    want = {'grant': config['grant_name'], 'replay_dir': str(replay), 'output_dir': str(out), 'game_pk': game_pk,
            'profile_sha256': step5_config['profile_sha256'],
            'source_sha256': {'own': own, 'step5_shadow': modules},
            'model_sha256': {s: v['sha256'] for s, v in config['model']['state'].items()},
            'calibrator_sha256': config['calibration']['result']['sha256'],
            'step5_ledger_sha256': config['step5_ledger']['sha256']}
    bad = sorted(k for k, v in want.items() if grant.get(k) != v)
    for key in ('allow_concurrent_with_protected_capture', 'acquire_heavy_lock'):
        if type(grant.get(key)) is not bool:
            bad.append(key+'_must_be_true_or_false')
    if not isinstance(grant.get('attempt_id'), str) or not grant['attempt_id']:
        bad.append('attempt_id')
    if type(game_pk) is not int:
        bad.append('a_real_run_names_its_game_with_--game')
    if executable != config['runtime_python']:
        bad.append('interpreter_is_not_the_registered_runtime')
    return bad+check_review(grant, config, own)


def check_step5_run(step5_summary, *, game, manifest_sha, profile_sha):
    """The Step 5 run the comparator comes from must be a completed run of the same game, replay manifest and
    profile bytes. Returns the problems."""
    run = step5_summary.get('run') if isinstance(step5_summary, dict) else None
    if not isinstance(run, dict):
        return ['step5_summary_has_no_run_block']
    return [name for name, ok in (
        ('step5_run_not_COMPLETED', step5_summary.get('status') == 'COMPLETED'),
        ('step5_run_is_another_game', run.get('game_pk') == game),
        ('step5_run_used_another_replay_manifest', run.get('replay_manifest_sha256') == manifest_sha),
        ('step5_run_used_another_profile', (run.get('profile') or {}).get('profile_sha256') == profile_sha)) if not ok]


def run_seq_shadow(replay_dir, out_dir, *, config, config_sha256, step5_config, mods, src, profile, heads, guard, mode,
                   seq, step5_ledger, game_pk=None, max_bodies=None, grant=None):
    """Replay -> seq shadow ledger -> SUMMARY.json. ``out_dir`` must not exist. Returns (rc, summary).
    ``step5_ledger``: {'path', 'sha256', 'summary' (parsed SUMMARY.json or None)}."""
    S, RUN, Refused = mods.S, mods.RUN, mods.G.Refused
    P, F, L = src.P, src.F, src.L
    replay, out = Path(replay_dir).resolve(), Path(out_dir).resolve()
    game, entries, manifest_sha, manifest = RUN.read_replay(replay, game_pk)
    if mode == 'synthetic' and not (manifest and manifest.get('synthetic_fixture') is True):
        raise Refused('synthetic_mode_needs_a_replay_manifest_marked_synthetic_fixture')
    if mode == 'actual' and (manifest is None or manifest.get('synthetic_fixture')):
        raise Refused('actual_mode_needs_a_capture_to_replay_manifest_that_is_not_a_fixture')
    if len(entries) > config['caps']['max_bodies']:
        raise Refused('replay_has_more_bodies_than_the_cap')
    bodies_in_replay = len(entries)
    if max_bodies is not None:
        entries = entries[:max_bodies]
    try:
        encoder, encoder_sha = F.load_encoder(step5_config['encoder']['path'], step5_config['encoder']['sha256'])
        registry = P.load_registry(step5_config['models']['path'], step5_config['models']['sha256'], encoder_sha)
        if S.file_sha256(step5_config['limits']['path']) != step5_config['limits']['sha256']:
            raise ValueError('step5_limits_differ_from_the_pin')
        limits = P.load_limits(step5_config['limits']['path'])
    except (ValueError, OSError) as exc:
        raise Refused(f'step5_registered_input:{type(exc).__name__}:{exc}') from None
    provenance = profile.provenance()
    if step5_ledger.get('summary') is not None:
        bad = check_step5_run(step5_ledger['summary'], game=game, manifest_sha=manifest_sha,
                              profile_sha=provenance.get('profile_sha256'))
        if bad:
            raise Refused('step5_comparator_run_is_not_the_same_replay:'+','.join(bad))
    elif mode == 'actual':
        raise Refused('actual_mode_needs_the_step5_summary')
    if list(P.LEAVES) != list(config['metrics']['leaf_order']) or len(F.ACTIONS) != config['model']['candidate_rows']:
        raise Refused('leaf_order_or_action_count_differs_from_the_registration')
    import numpy as np
    import torch
    architecture = seq.ensemble.architecture['context']
    token_kw = {'length': seq.ensemble.length, 'types': F.TYPES, 'description': RUN.DESCRIPTION,
                'pitch_number_range': config['tokens']['pitch_number_range'],
                'balls_vocab': architecture['balls_vocab'], 'strikes_vocab': architecture['strikes_vocab'],
                'missing_type': config['tokens']['missing_type_policy']}
    if len(F.TYPES) != architecture['pitch_type_vocab']:
        raise Refused('pitch_type_vocabulary_differs_from_the_19_frozen_types')
    own = SS.own_source_sha256()
    out.mkdir(parents=False, exist_ok=False)  # one run, one directory; a second run refuses here
    guard.start()
    ledger = L.DecisionLedger(out/LEDGER_NAME)
    anchor = 'shadow-replay:'+(manifest_sha or 'no-manifest')[:16]
    rule = step5_config['prior_poll_rule']
    source = {'own': own, 'step5_shadow': mods.sha256, 'pinned': src.pins}
    run_info = {'schema': 'seq-shadow-run-v1', 'mode': mode, 'game_pk': game, 'replay_dir': str(replay),
                'replay_manifest_sha256': manifest_sha, 'bodies': len(entries), 'bodies_in_replay': bodies_in_replay,
                'profile': provenance, 'source': source, 'config_sha256': config_sha256,
                'step5_encoder_sha256': encoder_sha, 'step5_registry_sha256': registry['registry_sha256'],
                'step5_heads': {'kind': heads['kind'], 'note': PLACEHOLDER_NOTE},
                'seq_model': {'arm': config['model']['arm'], 'seeds': seq.ensemble.seeds, 'sha256': seq.ensemble.sha256,
                              'cal_selected': {str(s): a for s, a in seq.calibrators.selected.items()},
                              'calibrator_sha256': seq.calibrators.sha256, 'encoder_sha256': seq.encoder_sha256,
                              'context_length': seq.ensemble.length, 'load_ms': seq.ensemble.load_ms,
                              'missing_type_policy': config['tokens']['missing_type_policy'],
                              'outcome10_source_sha256': config['tokens']['outcome10_source']['sha256']},
                'step5_ledger': {'path': str(step5_ledger['path']), 'sha256': step5_ledger['sha256'],
                                 'summary_sha256': step5_ledger.get('summary_sha256')},
                'limits': limits, 'prior_poll_rule': rule, 'clock': RUN.CLOCK, 'clock_anchor_id': anchor,
                'grant': grant, 'interpreter': sys.executable,
                'versions': {'python': sys.version.split()[0], 'torch': torch.__version__, 'numpy': np.__version__,
                             'scipy': __import__('scipy').__version__},
                'torch_num_threads': torch.get_num_threads(), 'fan_facing_output': False,
                'claims': config['claims']}
    tables = RUN.ShadowTables(ledger, run_info)
    seq_tables = LED.SeqTables(ledger.db, run_info, config_sha256=config_sha256, source_sha256=source,
                               model_sha256=seq.ensemble.sha256, calibrator_sha256=seq.calibrators.sha256['result'])
    prefixer = RUN.Prefixer(src, game, anchor, rule, limits)
    try:
        return _replay(mods, src, replay, out, entries, config, guard, prefixer, tables, seq_tables, ledger, profile,
                       dict(encoder=encoder, encoder_sha256=encoder_sha, registry=registry, heads=heads, limits=limits),
                       game, run_info, seq, token_kw, step5_ledger)
    except BaseException:  # never leave the writer lock or the hard stops behind an error
        try:
            ledger.close()
        except Exception:
            pass
        guard.finish()
        raise


def _seq_first_decision(mods, src, seq, seq_tables, record, index, entry, feed, state, profile, encoder, token_kw):
    """The seq_forecasts row of the attempt whose first decision was just appended. Returns timings or None."""
    import numpy as np
    F, RUN = src.F, mods.RUN
    context = None if state is None else {k: (list(v) if isinstance(v, tuple) else v) for k, v in sorted(state.items())}
    context_sha = None if context is None else RUN._sha(context)
    if not record.get('vectors_sha256'):  # Step 5 built no matrix: abstain with its reason, no model call
        seq_tables.append_abstain(record, index, entry.get('file'), context_sha, f"{record['outcome']}:{record['reason']}")
        return None
    t0 = time.perf_counter()
    x_f1, raw = F.build_candidates(state, profile, encoder)  # the same call the pipeline made for this body
    if hashlib.sha256(np.ascontiguousarray(x_f1).tobytes()).hexdigest() != record['vectors_sha256']:
        raise Integrity('vectors_sha256_differs_from_the_pipeline_record:'+record['attempt_key'])
    x_seq = F.apply_encoder(raw, seq.encoder)
    if x_seq.dtype != np.float32 or not np.array_equal(x_seq, seq.encode(raw, seq.encoder)):
        raise Integrity('scale-002_encoding_differs_between_features.apply_encoder_and_encode_for_seq')
    t1 = time.perf_counter()
    timings = {'features_ms': (t1-t0)*1000}
    try:
        tok, counts = TOK.build(feed, state, **token_kw)
        t2 = time.perf_counter()
        forecast = seq.adapter.forecast(x_seq, tok)
    except Exception as exc:  # counted as an abstention with the exact reason, never a crash and never a guess
        seq_tables.append_abstain(record, index, entry.get('file'), context_sha,
                                  f'seq_predictor_error:{type(exc).__name__}:{exc}', timings)
        return None
    timings |= {'tokens_ms': (t2-t1)*1000, 'forward_ms': forecast['forward_ms'],
                'calibration_ms': forecast['calibration_ms']}
    timings['compute_ms'] = timings['tokens_ms']+timings['forward_ms']+timings['calibration_ms']
    seq_tables.append_scored(record, index, entry.get('file'), context_sha,
                             x_seq_sha256=hashlib.sha256(np.ascontiguousarray(x_seq).tobytes()).hexdigest(),
                             tokens=TOK.to_json(tok, counts), forecast=forecast, timings=timings)
    return timings


def _replay(mods, src, replay, out, entries, config, guard, prefixer, tables, seq_tables, ledger, profile, model, game,
            run_info, seq, token_kw, step5_ledger):
    S, RUN, Stop = mods.S, mods.RUN, mods.G.Stop
    P, F, A, mode, rule = src.P, src.F, src.A, run_info['mode'], run_info['prior_poll_rule']
    statuses, lat, bodies = {}, {'all': [], 'scored': [], 'abstain': [], 'load': []}, {'unreadable': 0, 'not_json_object': 0}
    final_feed, final_ref, stop, integrity, processed = None, None, None, None, 0
    try:
        for index, entry in enumerate(entries):
            guard.check()
            t0 = time.perf_counter()
            raw = RUN.load_body(replay/str(entry['file']), config['caps']['max_body_bytes'])
            feed, raw_sha = None, None
            if raw is None:
                bodies['unreadable'] += 1
            else:
                raw_sha = hashlib.sha256(raw).hexdigest()
                try:
                    feed = json.loads(raw)
                except ValueError:
                    feed = None
                if not isinstance(feed, dict):
                    feed = None
                    bodies['not_json_object'] += 1
            del raw
            prefix, decision, state = prefixer.build(index, entry, feed, raw_sha)
            t1 = time.perf_counter()
            status, record, _ = A.live_decide(P, prefix, decision, profile=profile, ledger=ledger, roster_first_seen=None,
                                              **model)
            ms = (time.perf_counter()-t1)*1000
            lat['load'].append((t1-t0)*1000)
            lat['all'].append(ms)
            statuses[status] = statuses.get(status, 0)+1
            if status == 'appended':
                lat['scored' if record['outcome'] == 'scored' else 'abstain'].append(ms)
            tables.sync_forecasts(index, entry, record if status == 'appended' else None, state, ms)
            if status == 'appended':  # the forecast of this body, before the next body is loaded
                _seq_first_decision(mods, src, seq, seq_tables, record, index, entry, feed, state, profile,
                                    model['encoder'], token_kw)
            seq_tables.sync_terminalized(index)
            # Realized side, after the forecast row of this body is committed. Only an identity-verified body counts.
            if feed is not None and feed.get('gamePk') == game and prefix['feed_sha256'] == hashlib.sha256(F.canonical(feed)).hexdigest():
                tables.observe_realized(index, prefix['feed_sha256'], game, feed)
                final_feed, final_ref = feed, {'replay_index': index, 'file': entry['file'], 'feed_sha256': prefix['feed_sha256']}
            processed += 1
            del feed, prefix
    except Stop as exc:
        stop = str(exc)
    except Integrity as exc:
        integrity = str(exc)
    recon = None
    if stop is None and integrity is None:
        P.close_game(ledger, game, 'replay_end')
        tables.sync_forecasts(len(entries), {}, None, None, None)
        seq_tables.sync_terminalized(len(entries))
        if final_feed is not None:
            recon = P.reconcile_physical_pitches(ledger, game, final_feed, evidence={
                'source': 'last identity-verified body of the replay folder'} | final_ref)
    extra = {'status': 'COMPLETED', 'stop': stop, 'mode': mode,
             'run': run_info, 'bodies': {'in_replay': len(entries), 'processed': processed} | bodies,
             'decide_status_counts': dict(sorted(statuses.items())), 'final_body': final_ref,
             'replay_clock': {'rule': rule, 'state_gap_descriptive': prefixer.gaps,
                              'uncertainty': list(RUN.CLOCK_UNCERTAINTY)}}
    complete = stop is None and integrity is None
    summary = RUN.summarize(src, ledger, game, profile, recon, final_feed if complete else None, None, lat, extra)
    summary.pop('metrics', None)  # computed from the placeholder heads: never reported
    summary |= {'schema': 'seq-shadow-summary-v1', 'claims': config['claims'],
                'step5_placeholder_note': PLACEHOLDER_NOTE,
                'step5_blocks_reused': 'denominators, ledger_counts, reconciliation, per_pitcher, '
                                       'coverage_vs_current_serving_path and latency.decide_* are computed by the '
                                       'pinned shadow_runner.summarize; its metrics block is removed'}
    db = ledger.db
    seq_rows = SUM.read_seq_forecasts(db)
    physical, realized, first_seen = SUM.physical_and_realized(db)
    summary['denominators'] |= SUM.seq_denominators(seq_rows) | {
        'seq_first_decision_written_not_before_its_pitch_was_first_shown': SUM.order_violations(seq_rows, first_seen)}
    summary['context'] = SUM.context_statistics(seq_rows)
    summary['latency'] |= {'seq': SUM.latency(seq_rows, RUN._quantiles), 'seq_model_load_ms': seq.ensemble.load_ms}
    gate, metrics = None, None
    if complete:
        step5_rows = SUM.read_step5_forecasts(step5_ledger['path'])
        if S.file_sha256(step5_ledger['path']) != step5_ledger['sha256']:
            integrity = 'step5_ledger_bytes_changed_while_it_was_read'
        else:
            gate = SUM.gate_v(seq_rows, step5_rows)
            if gate['passed']:
                a_rows, b_rows, counts = SUM.populations(physical, realized, seq_rows, step5_rows, list(F.ACTIONS),
                                                         list(P.LEAVES))
                metrics = SUM.metrics(a_rows, b_rows, counts, config=config, leaves=list(P.LEAVES),
                                      calibration=RUN._calibration)
                metrics['realized_leaf_mapping'] = RUN.LEAF_PARITY
            else:
                integrity = 'gate_V_failed'
    summary |= {'gate_V': gate, 'metrics': metrics, 'integrity_stop': integrity}
    summary['status'] = 'STOPPED_BY_CAP' if stop is not None else 'STOPPED_BY_INTEGRITY_GATE' if integrity else 'COMPLETED'
    ledger.close()
    summary['guard'] = guard.finish()
    summary['ledger_file'] = {'path': str(out/LEDGER_NAME), 'sha256': S.file_sha256(out/LEDGER_NAME)}
    with open(out/'SUMMARY.json', 'x') as handle:
        json.dump(summary, handle, indent=1, ensure_ascii=False)
    return (RC_CAP if stop is not None else RC_GATE if integrity else RC_OK), summary


def _prepare(args, prepared):
    """Every check and every hash-pinned load of a run; writes nothing. Fills ``prepared`` (``Refused`` class as
    soon as the pinned guard is imported, then the arguments of ``run_seq_shadow``)."""
    config = SS.load_config(args.config)
    config_sha = SS.file_sha256(args.config)
    replay, out = Path(args.replay_dir).resolve(), Path(args.out_dir).resolve()
    forbidden = [w for w in config['forbidden_path_parts'] if w in replay.parts or w in out.parts]
    if forbidden:
        raise ValueError('path_inside_a_protected_directory:'+','.join(forbidden))
    if out.exists():
        raise ValueError('output_dir_exists')
    if not replay.is_dir():
        raise ValueError('replay_dir_absent')
    if args.max_bodies is not None and args.max_bodies < 1:
        raise ValueError('max_bodies_must_be_positive')
    mods = SS.load_step5_shadow(config)  # three sha256-pinned modules + the pinned Step 5 shadow config
    S, G, RUN = mods.S, mods.G, mods.RUN
    prepared['Refused'] = Refused = G.Refused
    step5_config = S.load_config()
    grant = None
    if args.synthetic:
        if args.grant or not args.profile_dir or not args.step5_ledger:
            raise Refused('synthetic_mode_takes_a_fixture_profile_dir_and_step5_ledger_and_no_grant')
        guard = G.Guard(config, allow_concurrent=False, acquire_heavy_lock=False, stop_marker=None)
        ledger_path = Path(args.step5_ledger).resolve()
        summary_path = ledger_path.parent/'SUMMARY.json'
    else:
        if not args.grant or args.profile_dir or args.step5_ledger or args.max_bodies is not None:
            raise Refused('a_real_run_needs_a_grant_and_uses_the_registered_profile_ledger_and_every_body')
        if Path(args.config).resolve() != SS.CONFIG_PATH.resolve():
            raise Refused('a_real_run_uses_the_registered_config_file')
        grant = json.loads(Path(args.grant).read_text())
        bad = check_grant(grant, replay=replay, out=out, game_pk=args.game, config=config, step5_config=step5_config,
                          step5_shadow_sha256=mods.sha256, executable=sys.executable)
        if bad:
            raise Refused('grant_not_bound_to_this_run:'+','.join(bad))
        grant = grant | {'grant_file_sha256': S.file_sha256(args.grant)}
        guard = G.Guard(config, allow_concurrent=grant['allow_concurrent_with_protected_capture'],
                        acquire_heavy_lock=grant['acquire_heavy_lock'], stop_marker=str(out)+'.RSS-STOP.json')
        ledger_path, summary_path = Path(config['step5_ledger']['path']), Path(config['step5_ledger']['summary_path'])
    if not ledger_path.is_file():
        raise Refused('step5_ledger_absent')
    ledger_sha = S.file_sha256(ledger_path)
    step5_summary, summary_sha = None, None
    if summary_path.is_file():
        raw = summary_path.read_bytes()
        summary_sha, step5_summary = hashlib.sha256(raw).hexdigest(), json.loads(raw)
    if not args.synthetic and (ledger_sha != config['step5_ledger']['sha256']
                               or summary_sha != config['step5_ledger']['summary_sha256']):
        raise Refused('step5_ledger_or_summary_differs_from_the_pin')
    guard.preflight()
    src = S.load(step5_config)
    profile = src.A.load_committed(src.F, args.profile_dir if args.synthetic else step5_config['profile_dir'])
    registry = src.P.load_registry(step5_config['models']['path'], step5_config['models']['sha256'],
                                   step5_config['encoder']['sha256'])
    heads = RUN.synthetic_heads(src.P, registry)  # placeholder in BOTH modes, see the module docstring
    if not args.synthetic:
        RUN.check_live_profile(profile, step5_config)
    seq = load_seq(config)
    prepared['run'] = dict(
        replay_dir=replay, out_dir=out, config=config, config_sha256=config_sha, step5_config=step5_config, mods=mods,
        src=src, profile=profile, heads=heads, guard=guard, mode='synthetic' if args.synthetic else 'actual', seq=seq,
        step5_ledger={'path': ledger_path, 'sha256': ledger_sha, 'summary': step5_summary, 'summary_sha256': summary_sha},
        game_pk=args.game, max_bodies=args.max_bodies, grant=grant)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Sequence-model shadow replay runner (no fan-facing output)')
    parser.add_argument('--replay-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--game', type=int)
    parser.add_argument('--grant', help='lead grant JSON; required unless --synthetic')
    parser.add_argument('--config', default=str(SS.CONFIG_PATH))
    parser.add_argument('--synthetic', action='store_true', help='TESTS ONLY: fixture replay, profile, weights, calibrators')
    parser.add_argument('--profile-dir', help='--synthetic only: fixture profile attempt directory')
    parser.add_argument('--step5-ledger', help='--synthetic only: fixture Step 5 shadow ledger (overrides the config)')
    parser.add_argument('--max-bodies', type=int)
    args = parser.parse_args(argv)
    prepared = {}

    def refuse(reason):
        print(json.dumps({'refused': reason}, ensure_ascii=False))
        return RC_REFUSED
    try:  # everything up to here reads and checks only: any failure is a refusal with nothing written
        _prepare(args, prepared)
    except (ValueError, PermissionError, OSError) as exc:  # PinMismatch is a ValueError
        return refuse(f'{type(exc).__name__}: {exc}')
    except Exception as exc:
        if 'Refused' in prepared and isinstance(exc, prepared['Refused']):
            return refuse(str(exc))
        raise
    try:
        rc, summary = run_seq_shadow(**prepared['run'])
    except FileExistsError as exc:  # the output directory appeared since the check: still nothing written by this run
        return refuse(f'FileExistsError: {exc}')
    except Exception as exc:  # a Refused is raised only before the output directory is made; anything else is a crash
        if isinstance(exc, prepared['Refused']):
            return refuse(str(exc))
        raise
    print(json.dumps({k: summary.get(k) for k in ('status', 'stop', 'integrity_stop', 'mode', 'game_pk', 'bodies',
                                                   'decide_status_counts', 'denominators', 'gate_V', 'context',
                                                   'latency', 'guard')}, indent=1, ensure_ascii=False))
    return rc


if __name__ == '__main__':
    sys.exit(main())
