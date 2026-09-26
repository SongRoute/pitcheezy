"""Audit mechanics only: synthetic CPU weights, no real fitting/data/MPS work."""
from copy import deepcopy
import numpy as np
import pytest

from pitchmdp.matrix_long_equivalence import (TOLERANCES, LIMITS, FIXED_TEMPERATURES,
    differences, objective, timed_logits, AuditedDelivery, audit_batches)
from test_lazy_delivery_reuse import batch, untrained, FrozenSyntheticPool
from audit_ml_long_equivalence import config_check, seal, sealed, summarize, CELLS, verify_profile_evidence, dump
from pitchmdp.data import hash_file


def config():
    return dict(protocol='ml_long_equivalence_v1', original_f4_run='/frozen/f4',
        original_preparation_sha256='a'*64,
        original_profile_evidence={cell: {'kind': 'caller_failure_ledger', 'path': '/original/'+cell, 'sha256': 'b'*64} for cell in CELLS},
        device='mps', cells=list(CELLS), tolerances=deepcopy(TOLERANCES), limits=deepcopy(LIMITS),
        fixed_temperatures=list(FIXED_TEMPERATURES), cell_seconds=1200)


def test_exact_registration_rejects_relaxed_tolerances_sample_shrink_cpu_or_incomplete_family():
    assert config_check(config())['limits']['temperature'] == 16
    for key, value in [('device', 'cpu'), ('cells', ['F4-H0']), ('cell_seconds', 9999)]:
        changed = config(); changed[key] = value
        with pytest.raises(ValueError): config_check(changed)
    changed = config(); changed['tolerances']['logits'] *= 2
    with pytest.raises(ValueError): config_check(changed)
    changed = config(); changed['limits']['temperature'] = 8
    with pytest.raises(ValueError): config_check(changed)
    changed = config(); changed['extra_tuning'] = True
    with pytest.raises(ValueError): config_check(changed)


def test_temperature_comparison_preserves_float32_objective_and_detects_difference():
    logits = np.random.default_rng(4).normal(size=(16, 400, 10)).astype(np.float32)
    labels = np.arange(16) % 10
    before = logits.copy()
    identical = differences(logits, logits.copy(), labels)
    assert identical['equivalent'] and all(identical['tolerance_pass'].values())
    assert max(identical['max_absolute_differences'].values()) <= 1e-6
    np.testing.assert_array_equal(logits, before)
    changed = logits.copy(); changed[:, :, 0] += .05
    result = differences(logits, changed, labels)
    assert not result['equivalent'] and not result['tolerance_pass']['logits']
    with pytest.raises(ValueError, match='float32'): objective(logits.astype(float), labels, 1.)


def test_runtime_rows_pools_tiers_rng_weights_and_order_are_audited_on_same_cpu_model():
    query = batch(128); model = untrained(query)
    pool = FrozenSyntheticPool()
    old, tiers, old_cost = timed_logits(model, pool, query, False, pitch_chunk=3, model_batch_size=257)
    new, other, new_cost = timed_logits(model, pool, query, True, pitch_chunk=3, model_batch_size=257)
    np.testing.assert_allclose(old, new, rtol=0, atol=TOLERANCES['logits'])
    np.testing.assert_array_equal(tiers, other)
    assert old_cost['network_rows']['long_token_rows'] == len(query)*400*128
    assert new_cost['network_rows']['long_token_rows'] == len(query)*128
    assert old_cost['network_rows']['h5_rows'] == new_cost['network_rows']['h5_rows'] == len(query)*400
    for key in ('pool_vectors_sha256', 'fallback_tiers_sha256', 'weights_sha256', 'rng_state_unchanged'):
        assert old_cost[key] == new_cost[key]
    order = np.array([4, 1, 3, 0, 2])
    reordered, level, _ = timed_logits(model, pool, query.subset(order), True, pitch_chunk=2, model_batch_size=37)
    np.testing.assert_allclose(reordered, old[order], rtol=0, atol=TOLERANCES['logits'])
    np.testing.assert_array_equal(level, tiers[order])


def test_pool_auditor_rejects_changed_chunk_vectors_and_fallback_metadata():
    query = batch(0)
    class ChangingPool(FrozenSyntheticPool):
        calls = 0
        def sample(self, frame):
            p, levels = super().sample(frame)
            if self.calls: p[0, 0, 0] += .01
            self.calls += 1
            return p, levels
    checked = AuditedDelivery(ChangingPool(), query)
    with pytest.raises(ValueError, match='Pool vectors or fallback'):
        checked.sample(query.frame())


def test_no_dev_batches_silent_sample_reduction_or_resealed_extra_artifacts(tmp_path):
    with pytest.raises(ValueError, match='Only TRAIN'):
        audit_batches({'dev': object()}, FrozenSyntheticPool(), tmp_path, device='cpu')
    q = batch(32)
    with pytest.raises(ValueError, match='silently shrink'):
        audit_batches({name: q for name in LIMITS}, FrozenSyntheticPool(), tmp_path, device='cpu')
    sealed_dir = tmp_path/'sealed'; sealed_dir.mkdir(); (sealed_dir/'result').write_text('fixed')
    expected = {'cell': 'F4-32'}; seal(sealed_dir, expected); sealed(sealed_dir, expected)
    (sealed_dir/'extra').write_text('changed')
    with pytest.raises(ValueError): sealed(sealed_dir, expected)


def test_three_complete_cell_gate_rejects_missing_arm_before_summary(tmp_path, monkeypatch):
    monkeypatch.setattr('audit_ml_long_equivalence.verify_registration', lambda *args: 'frozen')
    with pytest.raises(FileNotFoundError): summarize(tmp_path, {})
    assert not (tmp_path/'summary').exists()


def test_original_profile_evidence_binds_each_arm_preparation_and_source(tmp_path):
    parent = tmp_path/'original'; parent.mkdir()
    prep = {'identity': {'source_hashes': {'source': 'frozen'}}}
    cfg = config(); cfg['original_f4_run'] = str(parent)
    for cell in CELLS:
        directory = parent/'profiles'/cell; directory.mkdir(parents=True)
        dump(directory/'profile.json', {'long_length': CELLS[cell], 'dev_scores_read': False})
        dump(directory/'manifest.json', {'identity': {'preparation_sha256': 'a'*64, 'cell': cell},
            'artifact_hashes': {'profile.json': hash_file(directory/'profile.json')}})
        cfg['original_profile_evidence'][cell] = {'kind': 'completed_profile_manifest',
            'path': str(directory/'manifest.json'), 'sha256': hash_file(directory/'manifest.json')}
    verify_profile_evidence(cfg, parent, prep)
    copied = deepcopy(cfg); copied['original_profile_evidence']['F4-32'] = deepcopy(copied['original_profile_evidence']['F4-H0'])
    with pytest.raises(ValueError, match='Distinct'): config_check(copied)
    swapped = deepcopy(cfg)
    swapped['original_profile_evidence']['F4-32'], swapped['original_profile_evidence']['F4-H0'] = (
        swapped['original_profile_evidence']['F4-H0'], swapped['original_profile_evidence']['F4-32'])
    with pytest.raises(ValueError, match='arm/preparation'): verify_profile_evidence(swapped, parent, prep)
    wrong = deepcopy(cfg); wrong['original_preparation_sha256'] = 'c'*64
    with pytest.raises(ValueError, match='arm/preparation'): verify_profile_evidence(wrong, parent, prep)
    for cell in CELLS:
        path = tmp_path/(cell+'-timeout.json')
        ledger = dict(protocol='ml_long_profile_caller_outcome_v1', original_f4_run=str(parent),
            preparation_sha256='a'*64, cell=cell, command='profile', outcome='timeout',
            elapsed_seconds=600.1, source_hashes=prep['identity']['source_hashes'])
        dump(path, ledger)
        cfg['original_profile_evidence'][cell] = {'kind': 'caller_failure_ledger', 'path': str(path), 'sha256': hash_file(path)}
    verify_profile_evidence(cfg, parent, prep)
    # Even rehashed failure evidence must match the original source identity.
    ledger['source_hashes'] = {'source': 'different'}; dump(path, ledger)
    cfg['original_profile_evidence'][cell]['sha256'] = hash_file(path)
    with pytest.raises(ValueError, match='source/elapsed'): verify_profile_evidence(cfg, parent, prep)
