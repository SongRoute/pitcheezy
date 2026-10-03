"""Review gap tests (OPUS-review-tabm-integration-805aadc): exact JSON types and the Korean table disclosures."""
import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_cuda_tabm_mini as tm  # noqa: E402  shared config / export / scorer helpers
from test_cuda_tabm_mini import export, few_draws  # noqa: E402,F401  pytest fixtures

rc, tf = tm.rc, tm.tf


@pytest.mark.parametrize('edit', [
    lambda c: c['arms'][3].update(priority=True),                    # True == 1 in Python, not in JSON
    lambda c: c['arms'][0].update(priority=True),                    # control "exact earlier entry"
    lambda c: c['arms'][3].update(k=8.0),
    lambda c: c['arms'][3].update(head_dropout=False),
    lambda c: c['arms'][1].update(width=128.0),                      # single arm no longer byte-equal to P30
    lambda c: c.update(seeds=[13.0, 14, 15]),
    lambda c: c.update(fits=15.0),
])
def test_config_check_refuses_value_equal_but_type_drifted_entries(edit):
    cfg = copy.deepcopy(tm.load())
    edit(cfg)
    with pytest.raises(ValueError):
        rc.config_check(cfg)


def test_korean_table_scopes_compute_note_and_states_seed_exposure(export, tmp_path, few_draws):  # noqa: F811
    path, digest, splits = export
    cfg = tm.load(registered=True)
    tf.run_score(tmp_path, cfg, export, tf.family(tmp_path, cfg, digest, splits, same_as_control=()))
    table = (tmp_path / 'score' / 'table_ko.md').read_text()
    assert '연산량은 같지 않습니다' in table
    assert 'head_dropout 행은 단일 신경망끼리' in table             # the dropout row is not TabM vs single
    assert '시드 13, 14, 15' in table and '새 데이터가 아닙니다' in table and '사후 선택' in table


@pytest.mark.parametrize('single', ['periodic_counts', 'periodic_counts_d10'])
def test_tabm_arm_refuses_a_single_network_archive(tmp_path, single):
    cfg = tm.load()
    spec = rc.arm_spec(cfg, single)
    tm.small_fit(rc.build_model(spec, 3, 'cpu'), spec).save(tmp_path / 'single.pt')
    for name in ('tabm_counts', 'tabm_counts_d10'):
        with pytest.raises(ValueError):
            rc.load_model(rc.arm_spec(cfg, name), tmp_path / 'single.pt', 'cpu')
