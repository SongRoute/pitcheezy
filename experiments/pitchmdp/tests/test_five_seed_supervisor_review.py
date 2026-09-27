"""Synthetic fail-closed checks for the shared five-seed supervisor; no ML data or fits."""
import json
import time

import pytest

from pitchmdp.matrix_five_seed_extension import (arm_obligation, five_seed_decision,
    launch_gate, ledger_totals, member_wall_seconds)
import supervise_ml_five_seed_extension as supervisor
from run_ml_benchmark import dump, read_json
from test_five_seed_extension import bundle_fixture, complete


def test_completion_requires_successful_ledger_end_and_unchanged_artifact(tmp_path):
    fx = bundle_fixture(tmp_path)
    path = supervisor.completion_paths(fx.config, read_json(fx.local))[('c1-prepare', None)]
    path.parent.mkdir(parents=True)
    path.write_text('{}')
    with pytest.raises(RuntimeError, match='without a successful ledger job'):
        supervisor.completed_steps(fx.config, read_json(fx.local))
    complete(fx, ('c1-prepare', None))
    assert ('c1-prepare', None) in supervisor.completed_steps(fx.config, read_json(fx.local))
    path.write_text('{"changed": true}')
    with pytest.raises(RuntimeError, match='changed artifact'):
        supervisor.completed_steps(fx.config, read_json(fx.local))


def test_ledger_rejects_negative_nonfinite_and_foreign_bundle_walls(tmp_path):
    fx = bundle_fixture(tmp_path)
    complete(fx, ('c1-prepare', None))
    end = next((fx.ledger / 'ends').glob('*.json'))
    original = read_json(end)
    for invalid in (-1., float('nan'), float('inf')):
        end.write_text(json.dumps({**original, 'elapsed_seconds': invalid, 'within_cap': False}))
        with pytest.raises(ValueError, match='Invalid ledger end'):
            supervisor.ledger_jobs(fx.ledger)
    dump(end, original)
    start = next((fx.ledger / 'jobs').glob('*.json'))
    dump(start, {**read_json(start), 'bundle_sha256': '0' * 64})
    with pytest.raises(ValueError, match='bundle identity'):
        supervisor.ledger_jobs(fx.ledger, supervisor.runner.canonical_hash(fx.config))


def test_preflight_refusal_and_deadline_charge_durable_wall(tmp_path, monkeypatch):
    fx = bundle_fixture(tmp_path)
    def reject(_bundle):
        raise ValueError('synthetic source mismatch')
    monkeypatch.setattr(supervisor.runner, 'verify_bundle_pins', reject)
    outcome, code = supervisor.supervise(fx.bundle, fx.local, 'c1-prepare', None)
    assert (outcome, code) == ('refused', supervisor.EXIT_REFUSED)
    job = supervisor.ledger_jobs(fx.ledger)[0]
    assert job['ended'] and job['elapsed_seconds'] > 0 and job['cap_seconds'] == 7200
    assert 'synthetic source mismatch' in read_json(fx.ledger / 'ends' / (job['job_id'] + '.json'))['note']

    fx2 = bundle_fixture(tmp_path / 'deadline', grace=.1)
    def delay(_bundle):
        time.sleep(.4)
    monkeypatch.setattr(supervisor.runner, 'verify_bundle_pins', delay)
    monkeypatch.setenv(supervisor.INHERITED_START_ENV, str(time.monotonic() - 7199.7))
    outcome, code = supervisor.supervise(fx2.bundle, fx2.local, 'c1-prepare', None)
    assert (outcome, code) == ('timeout', supervisor.EXIT_TIMEOUT)
    job = supervisor.ledger_jobs(fx2.ledger)[0]
    assert job['ended'] and job['outcome'] == 'timeout' and job['elapsed_seconds'] >= 7199.7
    start = read_json(fx2.ledger / 'jobs' / (job['job_id'] + '.json'))
    assert start['accounting_start_source'] == 'outer_queue'


def test_member_attempts_and_obligations_fail_closed():
    jobs = [{'arm': 'c1', 'stage': stage, 'seed': 3, 'ended': True, 'elapsed_seconds': seconds,
             'cap_seconds': 7200, 'job_id': stage} for stage, seconds in [('fit', 5000.), ('predict', 1000.)]]
    assert member_wall_seconds(jobs, 'c1', 3) == 6000.
    totals = ledger_totals(jobs, 'c1')
    with pytest.raises(RuntimeError, match='Finite nonnegative remaining obligation'):
        launch_gate(totals, 7200, 'fit', projected_command=100., remaining_obligation=float('nan'))
    projection = {'prepare_seconds': 1., 'fit_command_seconds': float('inf'),
                  'predict_command_seconds': 1., 'score_seconds': 1.}
    with pytest.raises(ValueError, match='Finite nonnegative projection'):
        arm_obligation(projection, set(), 'c1')


def test_scientific_inference_rejects_nonfinite_and_keeps_registered_statuses():
    measured = {'status': 'measured', 'nll': {'delta': float('nan'), 'ci95': [-.01, -.001], 'p_less': .01},
                'brier': {'delta': 0., 'ci95': [-.001, .001]}}
    with pytest.raises(ValueError, match='Finite ordered nll'):
        five_seed_decision(measured, [-.001] * 5)
    assert five_seed_decision({'status': 'insufficient_games'}, [-.001] * 5)['status'] == 'inconclusive'
