import json
from pathlib import Path

import pytest

from scripts.c_candidate_evidence import BUNDLE, SSD, build


def sources():
    return (SSD / 'replacement_packets.json', SSD / 'lineup_supplement_v2.json',
            SSD / 'decision_anchor_777063.json', BUNDLE / 'metadata.json',
            BUNDLE / 'bundle_manifest.json')


def test_fixed_source_audit_preserves_availability_and_model_limits():
    result = build(*sources())
    assert result['counts'] == {
        'games': 6, 'screened_replacement_entries': 65,
        'evidenced_current_pitchers': 6, 'verified_decision_available_replacements': 0,
        'conditional_roster_screen_entries': 65, 'supported_keep_games': 1,
        'supported_replacement_entries': 1,
        'jointly_supported_keep_and_replacement_games': 0, 'valid_inning_comparisons': 0,
    }
    assert all(c['model_comparison']['value_pp'] is None for c in result['cases'])
    assert all(r['manager_available_at_decision'] is None and not r['inning_evaluation_ready']
               for c in result['cases'] for r in c['candidates'])
    assert result['comparison_spec']['pa_values_may_be_combined'] is False


def test_rejects_promoted_manager_availability(tmp_path: Path, monkeypatch):
    packet, *other = sources()
    mutated = json.loads(packet.read_text())
    mutated['decisions'][0]['replacement_screen'][0]['manager_available_at_decision'] = True
    replacement = tmp_path / 'packet.json'
    replacement.write_text(json.dumps(mutated))
    lineup, anchor, metadata, manifest = other
    import hashlib
    new_sha = hashlib.sha256(replacement.read_bytes()).hexdigest()
    l = json.loads(lineup.read_text()); l['original_packet_sha256'] = new_sha
    a = json.loads(anchor.read_text()); a['original_packet_sha256'] = new_sha
    new_lineup = tmp_path / 'lineup.json'; new_lineup.write_text(json.dumps(l))
    new_anchor = tmp_path / 'anchor.json'; new_anchor.write_text(json.dumps(a))
    from scripts import c_candidate_evidence as audit
    monkeypatch.setattr(audit, 'PINNED_SHA256', {
        'roster': new_sha,
        'lineup': hashlib.sha256(new_lineup.read_bytes()).hexdigest(),
        'anchor': hashlib.sha256(new_anchor.read_bytes()).hexdigest(),
        'bundle': hashlib.sha256(manifest.read_bytes()).hexdigest(),
    })
    with pytest.raises(ValueError, match='candidate availability promoted'):
        build(replacement, new_lineup, new_anchor, metadata, manifest)


def test_rejects_future_workload_even_with_updated_source_hash(tmp_path: Path, monkeypatch):
    packet, lineup, anchor, metadata, manifest = sources()
    mutated = json.loads(packet.read_text())
    item = next(r for r in mutated['decisions'][0]['replacement_screen'] if r['prior_7d_appearances'])
    item['prior_7d_appearances'][0]['official_date'] = '2025-07-22'
    replacement = tmp_path / 'packet.json'
    replacement.write_text(json.dumps(mutated))
    import hashlib
    new_sha = hashlib.sha256(replacement.read_bytes()).hexdigest()
    l = json.loads(lineup.read_text()); l['original_packet_sha256'] = new_sha
    a = json.loads(anchor.read_text()); a['original_packet_sha256'] = new_sha
    new_lineup = tmp_path / 'lineup.json'; new_lineup.write_text(json.dumps(l))
    new_anchor = tmp_path / 'anchor.json'; new_anchor.write_text(json.dumps(a))
    from scripts import c_candidate_evidence as audit
    monkeypatch.setattr(audit, 'PINNED_SHA256', {
        'roster': new_sha,
        'lineup': hashlib.sha256(new_lineup.read_bytes()).hexdigest(),
        'anchor': hashlib.sha256(new_anchor.read_bytes()).hexdigest(),
        'bundle': hashlib.sha256(manifest.read_bytes()).hexdigest(),
    })
    with pytest.raises(ValueError, match='future workload'):
        build(replacement, new_lineup, new_anchor, metadata, manifest)
