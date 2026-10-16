"""Generated native sidecar replay; strict accepted semantics stay untouched."""
import copy
import hashlib
import json

import pytest

from pitcheezy.seq_shadow.captured_bridge import step
from pitcheezy.seq_shadow.captured_native import inspect_tokens
from pitcheezy.seq_shadow.captured_native_state import CONTRACT, NATIVE_KIND, NativeRuntime, NativeShadow
from pitcheezy.seq_shadow.fullseason_stream import StreamState
from pitcheezy.seq_shadow import fullseason_profiles as profiles
from pitcheezy.seq_shadow.fullseason_runtime import Runtime
from test_captured_bridge import feed
from test_seq_shadow_fullseason import generated_config
from test_fullseason_profiles import donors, bundle

SHA = 'a' * 64


def source():
    body = feed()
    body['liveData']['plays']['currentPlay']['playEvents'][0]['details']['type'] = {'code': 'FF'}
    bridge = step(body, game_pk=99, game_date='2026-10-11', receipt_seq=1,
                  source_sha256=SHA, candidate_pitch_type='FF')
    native = inspect_tokens(body, game_pk=99, game_date='2026-10-11',
                            receipt_seq=1, source_sha256=SHA)[0]
    return bridge, native


def state(path):
    return NativeShadow(path, game_pk=99, game_date='2026-10-11', normalizer_sha256=SHA)


def test_native_ready_is_distinct_from_strict_unknown_and_restart_immutable(tmp_path):
    bridge, native = source()
    path = tmp_path/'shadow.sqlite'
    s = state(path)
    intake = s.ingest(bridge['events'][0], native, received_ns=100, source_sha256=SHA)
    assert intake['strict_status'] == 'new' and intake['native_status'] == 'native_candidate'
    decision = s.decide(bridge['candidate'], received_ns=101)
    assert decision['strict_status'] == 'abstain' and decision['status'] == 'native_ready'
    assert decision['context_counts'] == {'prior_known_same_pitcher_total': 1,
        'prior_unknown_relevant': 0, 'selected_known_tokens': 1,
        'previous_pitch_native_state': 'known'}
    assert not decision['prediction_ready'] and not decision['prospective_claim']
    snapshot = decision['native_snapshot']
    assert snapshot['status'] == 'ready' and snapshot['identity']['evidence_kind'] == NATIVE_KIND
    assert snapshot['input_contract'] == CONTRACT and not snapshot['statcast_training_equivalent']
    assert snapshot['tokens']['valid'][0][0]
    s.close()
    reopened = state(path)
    assert reopened.decide(bridge['candidate'], received_ns=200) == decision
    reopened.close()


def test_missing_native_token_abstains_and_original_decision_never_backfills(tmp_path):
    bridge, native = source()
    s = state(tmp_path/'shadow.sqlite')
    s.ingest(bridge['events'][0], None, received_ns=100, source_sha256=SHA)
    first = s.decide(bridge['candidate'], received_ns=101)
    assert first['status'] == 'native_abstain' and first['reason'] == 'native_prior_quality_unknown'
    assert first['context_counts']['prior_unknown_relevant'] == 1
    correction = copy.deepcopy(bridge['events'][0]); correction['revision'] = 2
    corrected = dict(native, receipt_seq=2)
    s.ingest(correction, corrected, received_ns=102, source_sha256=SHA)
    assert s.decide(bridge['candidate'], received_ns=103) == first
    next_request = dict(bridge['candidate'], decision_id=bridge['candidate']['decision_id']+':later')
    later = s.decide(next_request, received_ns=104)
    assert later['status'] == 'native_ready'
    s.close()


def test_mixed_contract_and_receipt_clock_regression_rejected(tmp_path):
    bridge, native = source()
    s = state(tmp_path/'shadow.sqlite')
    altered = dict(native, input_contract='STRICT_STATCAST')
    with pytest.raises(ValueError, match='mixed or invalid'):
        s.ingest(bridge['events'][0], altered, received_ns=100, source_sha256=SHA)
    with pytest.raises(ValueError, match='native unknown must not carry'):
        s.ingest(bridge['events'][0], dict(native, status='native_unknown',
                 reason='missing_field', native_token={'malformed': True}),
                 received_ns=100, source_sha256=SHA)
    s.ingest(bridge['events'][0], native, received_ns=100, source_sha256=SHA)
    correction = copy.deepcopy(bridge['events'][0]); correction['revision'] = 2
    with pytest.raises(ValueError, match='receipt clock regressed'):
        s.ingest(correction, dict(native, receipt_seq=2), received_ns=99, source_sha256=SHA)
    s.close()


def test_strict_accepted_ledger_cannot_be_reopened_as_native(tmp_path):
    path = tmp_path/'shadow.sqlite'
    strict = StreamState(path, game_pk=99, game_date='2026-10-11',
                         normalizer_sha256=SHA, evidence_kind='captured')
    strict.ingest('p1', (99, 1, 1), 1, {'status': 'accepted', 'reason': 'generated',
        'accepted': {'pitcher': 10, 'batter': 20, 'pitch_type': 'FF', 'outcome10': 0,
                     'balls': 0, 'strikes': 0}}, received_ns=100)
    strict.close()
    with pytest.raises(ValueError, match='cannot mix'):
        state(path)


def test_native_unknown_floor_and_same_pa_predecessor(tmp_path):
    bridge, native = source()
    s = state(tmp_path/'shadow.sqlite')
    base = bridge['events'][0]
    for pa in range(1, 34):
        event = dict(base, event_id=f'p{pa}', key=(99, pa, 1), revision=pa)
        candidate = None if pa == 1 else dict(native, key=(99, pa, 1), provider_id=f'p{pa}', receipt_seq=pa)
        s.ingest(event, candidate, received_ns=pa, source_sha256=SHA)
    query = dict(bridge['candidate']['query'], at_bat_number=34, pitch_number=1)
    ask = dict(bridge['candidate'], decision_id='after-32-native-candidates', query=query,
               coverage=dict(bridge['candidate']['coverage'], before_key=[99, 34, 1]))
    eligible = s.decide(ask, received_ns=34)
    assert eligible['status'] == 'native_ready'
    assert eligible['context_counts']['prior_known_same_pitcher_total'] == 32
    assert eligible['context_counts']['prior_unknown_relevant'] == 0
    predecessor = dict(base, event_id='p34', key=(99, 34, 1), revision=34)
    s.ingest(predecessor, None, received_ns=35, source_sha256=SHA)
    query2 = dict(query, pitch_number=2)
    ask2 = dict(ask, decision_id='same-pa-unknown-predecessor', query=query2,
                coverage=dict(ask['coverage'], before_key=[99, 34, 2]))
    outcome = s.decide(ask2, received_ns=36)
    assert outcome['status'] == 'native_abstain'
    assert outcome['reason'] == 'native_prior_quality_unknown'
    assert outcome['context_counts']['previous_pitch_native_state'] == 'unknown'
    s.close()


def test_observed_pitch_query_rejected(tmp_path):
    bridge, native = source()
    s = state(tmp_path/'shadow.sqlite')
    s.ingest(bridge['events'][0], native, received_ns=100, source_sha256=SHA)
    observed = copy.deepcopy(bridge['candidate'])
    observed['query']['pitch_number'] = 1
    observed['coverage']['before_key'] = [99, 1, 1]
    with pytest.raises(ValueError, match='already observed'):
        s.decide(observed, received_ns=101)
    observed['pitch_type_basis'] = 'observed_actual'
    with pytest.raises(ValueError, match='hypothetical'):
        s.decide(observed, received_ns=101)
    s.close()


def test_changed_receipt_sha_same_native_fields_dedups_semantically(tmp_path):
    bridge, native = source()
    path = tmp_path/'shadow.sqlite'
    s = state(path)
    first = s.ingest(bridge['events'][0], native, received_ns=100, source_sha256=SHA)
    later_event = dict(bridge['events'][0], revision=2)
    later_native = dict(native, receipt_seq=2, source_sha256='b'*64)
    repeated = s.ingest(later_event, later_native, received_ns=101, source_sha256='b'*64)
    assert repeated['strict_status'] == 'unchanged_observation'
    assert repeated['event_seq'] == first['event_seq']
    assert s.db.execute('SELECT COUNT(*) FROM events').fetchone()[0] == 1
    stored = json.loads(s.db.execute('SELECT body FROM native_evidence').fetchone()['body'])
    assert stored['source_sha256'] == SHA and stored['receipt_seq'] == 1
    with pytest.raises(ValueError, match='native receipt ordinal reused'):
        s.ingest(dict(later_event, revision=1), dict(later_native, receipt_seq=1),
                 received_ns=101, source_sha256='b'*64)
    with pytest.raises(ValueError, match='receipt clock regressed'):
        s.observe_receipt(3, 'c'*64, 100)
    s.close()
    reopened = state(path)
    assert reopened.ingest(dict(later_event, revision=3),
                           dict(later_native, receipt_seq=3), received_ns=102,
                           source_sha256='b'*64)['strict_status'] == 'unchanged_observation'
    reopened.close()


def test_native_runtime_requires_explicit_opt_in_and_labels_score(tmp_path):
    bridge, native = source()
    s = state(tmp_path/'shadow.sqlite')
    s.ingest(bridge['events'][0], native, received_ns=100, source_sha256=SHA)
    decision = s.decide(bridge['candidate'], received_ns=101)
    class Stub:
        def predict(self, snapshot):
            assert snapshot['identity']['evidence_kind'] == NATIVE_KIND
            return {'status': 'scored', 'stream_evidence_kind': NATIVE_KIND,
                    'prospective_claim': False, 'call_sha256': SHA,
                    'probabilities': [1/7]*7}
    with pytest.raises(ValueError, match='opt-in'):
        NativeRuntime(Stub())
    wrapper = NativeRuntime(Stub(), allow_native_experimental=True)
    score = wrapper.predict(decision)
    assert score['status'] == 'experimental_conditional_score'
    assert not score['prediction_ready'] and not score['prospective_claim']
    with pytest.raises(ValueError, match='native contract'):
        wrapper.predict(s.strict.decide(**bridge['candidate'], received_ns=102))
    s.close()


def test_generated_native_snapshot_composes_with_seven_leaf_runtime(tmp_path):
    bridge, native = source()
    s = state(tmp_path/'shadow.sqlite')
    s.ingest(bridge['events'][0], native, received_ns=100, source_sha256=SHA)
    decision = s.decide(bridge['candidate'], received_ns=101)
    config, _ = generated_config(tmp_path)
    g, M = donors()
    directory = tmp_path/'profiles'
    profiles.save_bundle(bundle(g, M), directory, {'generated': True})
    manifest_sha = hashlib.sha256((directory/'manifest.json').read_bytes()).hexdigest()
    runtime = NativeRuntime(Runtime(config, directory, manifest_sha), allow_native_experimental=True)
    scored = runtime.predict(decision)
    assert scored['status'] == 'experimental_conditional_score'
    assert len(scored['probabilities']) == 7
    assert abs(sum(scored['probabilities'])-1) < 1e-12
    assert not scored['prediction_ready']
    s.close()


def test_prior_decision_survives_later_observed_target_key_and_restart(tmp_path):
    bridge, native = source()
    path = tmp_path/'shadow.sqlite'
    s = state(path)
    s.ingest(bridge['events'][0], native, received_ns=100, source_sha256=SHA)
    frozen = s.decide(bridge['candidate'], received_ns=101)
    observed_target = dict(bridge['events'][0], event_id='p2', key=(99, 1, 2), revision=2)
    s.ingest(observed_target, None, received_ns=102, source_sha256='b'*64)
    s.close()
    restarted = state(path)
    assert restarted.decide(bridge['candidate'], received_ns=103) == frozen
    restarted.close()


def test_receipt_without_changed_pitch_is_still_bound_and_clocked(tmp_path):
    bridge, native = source()
    s = state(tmp_path/'shadow.sqlite')
    s.observe_receipt(1, SHA, 100)
    s.observe_receipt(2, 'b'*64, 101)
    assert s.db.execute('SELECT COUNT(*) FROM native_receipts').fetchone()[0] == 2
    event = dict(bridge['events'][0], revision=2)
    row = dict(native, receipt_seq=2, source_sha256='b'*64)
    assert s.ingest(event, row, received_ns=101, source_sha256='b'*64)['strict_status'] == 'new'
    with pytest.raises(ValueError, match='receipt clock regressed'):
        s.observe_receipt(3, 'c'*64, 100)
    with pytest.raises(ValueError, match='current stored receipt'):
        s.decide(bridge['candidate'], received_ns=102)
    s.close()
