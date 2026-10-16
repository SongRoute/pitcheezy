"""Generated-only native forecast persistence; no model or capture payload."""
import copy
import sqlite3

import pytest

from pitcheezy.seq_shadow import fullseason
from pitcheezy.seq_shadow.captured_native_forecast import NativeForecast
from pitcheezy.seq_shadow.captured_native_state import NativeRuntime
from pitcheezy.seq_shadow.ledger import sha
from test_captured_native_state import SHA, source, state

MANIFEST = 'b' * 64
REGISTRATION = '5' * 64


def config():
    return {'leaf_order': list(fullseason.LEAVES),
            'state': {seed: {'sha256': char * 64}
                      for seed, char in (('0', 'c'), ('1', 'd'), ('2', 'e'))},
            'encoder': {'sha256': 'f' * 64}, 'calibrator': {'sha256': '1' * 64},
            'recipe': {'sha256': '2' * 64}}


class FakeRuntime:
    def __init__(self, cfg, *, result=None, callback=None):
        self.identity = {'adapter_config_sha256': sha(cfg),
                         'profile_manifest_sha256': MANIFEST}
        self.bundle_sha256 = sha(self.identity)
        self.calls = 0
        self.result = result
        self.callback = callback

    def predict(self, snapshot):
        self.calls += 1
        if self.callback:
            self.callback()
        if isinstance(self.result, Exception):
            raise self.result
        return {'status': 'scored', 'stream_evidence_kind': 'capture_native_experimental',
                'prospective_claim': False, 'call_sha256': sha({
                    'bundle': self.bundle_sha256, 'snapshot': snapshot['snapshot_sha256']}),
                'probabilities': self.result if self.result is not None else [1 / 7] * 7}


def setup(tmp_path, *, known=True, result=None, callback=None):
    bridge, native = source()
    tmp_path.mkdir(parents=True, exist_ok=True)
    shadow_path = forecast_path = tmp_path / 'shadow.sqlite'
    shadow = state(shadow_path)
    shadow.ingest(bridge['events'][0], native if known else None,
                  received_ns=100, source_sha256=SHA)
    cfg = config()
    fake = FakeRuntime(cfg, result=result, callback=callback)
    ledger = NativeForecast(shadow, NativeRuntime(fake, allow_native_experimental=True),
                            cfg, MANIFEST, REGISTRATION)
    return bridge['candidate'], shadow, ledger, fake, shadow_path, forecast_path


def test_scored_exact_vector_is_immutable_and_separate_after_later_receipt(tmp_path):
    values = [0.25, 0.25, 0.2, 0.1, 0.1, 0.05, 0.05]
    candidate, shadow, ledger, fake, shadow_path, forecast_path = setup(tmp_path, result=values)
    first = ledger.forecast(candidate, received_ns=101)
    assert first['status'] == 'experimental_conditional_score'
    assert first['probabilities'] == values and first['forecast_sha256']
    assert type(first['score_compute_ns']) is int and first['score_compute_ns'] >= 0
    assert first['local_pre_observation_only'] and not first['prediction_ready']
    assert fake.calls == 1
    assert ledger.forecast(candidate, received_ns=101) == first and fake.calls == 1
    changed = copy.deepcopy(candidate)
    changed['query']['balls'] = 2
    with pytest.raises(ValueError, match='identity or artifact binding reused'):
        ledger.forecast(changed, received_ns=101)
    shadow.observe_receipt(2, '9' * 64, 102)
    assert ledger.forecast(candidate, received_ns=103) == first and fake.calls == 1
    shadow.close()
    assert shadow_path.exists()
    with sqlite3.connect(forecast_path) as db:
        assert db.execute('SELECT COUNT(*) FROM native_forecast_results').fetchone()[0] == 1
        with pytest.raises(sqlite3.DatabaseError, match='append-only'):
            db.execute('DELETE FROM native_forecast_results')
    reopened = state(shadow_path)
    cfg = config()
    fresh_fake = FakeRuntime(cfg, result=[1.0, 0, 0, 0, 0, 0, 0])
    resumed = NativeForecast(reopened, NativeRuntime(fresh_fake, allow_native_experimental=True),
                             cfg, MANIFEST, REGISTRATION)
    assert resumed.forecast(candidate, received_ns=104) == first and fresh_fake.calls == 0
    reopened.close()


def test_abstain_is_terminal_without_probabilities_or_model_call(tmp_path):
    candidate, shadow, ledger, fake, _, _ = setup(tmp_path, known=False)
    result = ledger.forecast(candidate, received_ns=101)
    assert result['status'] == 'experimental_abstain'
    assert result['reason'] == 'native_prior_quality_unknown'
    assert result['probabilities'] is None and fake.calls == 0
    assert result['score_compute_ns'] is None
    shadow.close()


def test_inference_exception_leaves_nonbackfillable_intent(tmp_path):
    candidate, shadow, ledger, fake, shadow_path, forecast_path = setup(
        tmp_path, result=RuntimeError('generated inference failure'))
    with pytest.raises(RuntimeError, match='generated inference failure'):
        ledger.forecast(candidate, received_ns=101)
    assert fake.calls == 1
    with pytest.raises(ValueError, match='incomplete forecast intent'):
        ledger.forecast(candidate, received_ns=101)
    shadow.close()
    reopened = state(shadow_path)
    cfg = config()
    other = FakeRuntime(cfg)
    resumed = NativeForecast(reopened, NativeRuntime(other, allow_native_experimental=True),
                             cfg, MANIFEST, REGISTRATION)
    with pytest.raises(ValueError, match='incomplete forecast intent'):
        resumed.forecast(candidate, received_ns=102)
    assert other.calls == 0
    reopened.close()


def test_existing_native_decision_or_changed_receipt_cannot_backfill(tmp_path):
    bridge, native = source()
    shadow = state(tmp_path/'shadow.sqlite')
    shadow.ingest(bridge['events'][0], native, received_ns=100, source_sha256=SHA)
    shadow.decide(bridge['candidate'], received_ns=101)
    cfg = config()
    fake = FakeRuntime(cfg)
    with pytest.raises(ValueError, match='prior native decisions'):
        NativeForecast(shadow, NativeRuntime(fake, allow_native_experimental=True), cfg,
                       MANIFEST, REGISTRATION)
    shadow.close()


def test_decision_without_intent_after_ledger_open_fails_closed(tmp_path):
    candidate, shadow, ledger, fake, _, db_path = setup(tmp_path)
    shadow.decide(candidate, received_ns=101)
    with pytest.raises(ValueError, match='native decision exists without forecast intent'):
        ledger.forecast(candidate, received_ns=101)
    assert fake.calls == 0
    with sqlite3.connect(db_path) as db:
        assert db.execute('SELECT COUNT(*) FROM native_forecast_intents').fetchone()[0] == 0
    shadow.close()


def test_receipt_advance_during_inference_leaves_intent_incomplete(tmp_path):
    candidate, shadow, ledger, fake, _, forecast_path = setup(tmp_path)
    fake.callback = lambda: shadow.observe_receipt(2, '9'*64, 102)
    with pytest.raises(ValueError, match='receipt, target or artifact changed'):
        ledger.forecast(candidate, received_ns=101)
    with sqlite3.connect(forecast_path) as db:
        assert db.execute('SELECT COUNT(*) FROM native_forecast_results').fetchone()[0] == 0
    shadow.close()


def test_target_observed_during_inference_blocks_score_without_clock_advance(tmp_path):
    candidate, shadow, ledger, fake, _, db_path = setup(tmp_path)
    event = {'event_id': 'target-p2', 'key': (99, 1, 2), 'revision': 1,
             'row': {'status': 'unknown', 'reason': 'statcast_acceptance_unverified',
                     'accepted': None}}
    fake.callback = lambda: shadow.strict.ingest(event['event_id'], event['key'], 2,
                                                event['row'], received_ns=101)
    with pytest.raises(ValueError, match='receipt, target or artifact changed'):
        ledger.forecast(candidate, received_ns=101)
    with sqlite3.connect(db_path) as db:
        assert db.execute('SELECT COUNT(*) FROM native_forecast_intents').fetchone()[0] == 1
        assert db.execute('SELECT COUNT(*) FROM native_forecast_results').fetchone()[0] == 0
    shadow.close()


def test_loaded_binding_and_fixed_ff_are_checked_before_reuse(tmp_path):
    candidate, shadow, ledger, fake, _, _ = setup(tmp_path)
    scored = ledger.forecast(candidate, received_ns=101)
    fake.identity['profile_manifest_sha256'] = '4' * 64
    with pytest.raises(ValueError, match='loaded runtime identity'):
        ledger.forecast(candidate, received_ns=101)
    fake.identity['profile_manifest_sha256'] = MANIFEST
    assert ledger.forecast(candidate, received_ns=101) == scored
    not_ff = copy.deepcopy(candidate)
    not_ff['query']['pitch_type'] = 'SL'
    with pytest.raises(ValueError, match='exact forecast decision request'):
        ledger.forecast(not_ff, received_ns=101)
    shadow.close()


@pytest.mark.parametrize('values', [[float('nan')] * 7, [-0.1, 1.1, 0, 0, 0, 0, 0],
                                     [0.5] * 7, [1.0] * 6])
def test_malformed_probability_vector_is_incomplete_not_persisted(tmp_path, values):
    candidate, shadow, ledger, _, _, forecast_path = setup(tmp_path, result=values)
    with pytest.raises(ValueError, match='invalid experimental seven-leaf score'):
        ledger.forecast(candidate, received_ns=101)
    with sqlite3.connect(forecast_path) as db:
        assert db.execute('SELECT COUNT(*) FROM native_forecast_intents').fetchone()[0] == 1
        assert db.execute('SELECT COUNT(*) FROM native_forecast_results').fetchone()[0] == 0
    shadow.close()
