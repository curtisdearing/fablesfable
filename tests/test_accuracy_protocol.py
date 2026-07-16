"""Frozen accuracy-policy invariants."""
import json
from pathlib import Path


PROTOCOL = Path(__file__).parents[1] / "analysis" / "accuracy_protocol.json"


def test_accuracy_protocol_has_fail_closed_evidence_gates():
    protocol = json.loads(PROTOCOL.read_text())
    assert protocol["schema_version"] == 1
    assert protocol["truth_windows"]["prospective_final_judge"].startswith("2026")
    assert protocol["matched_control"]["minimum_exposed_n"] >= 100
    assert protocol["matched_control"]["minimum_control_n"] >= 100
    assert protocol["matched_control"]["season_forward_replication_required"] is True
    assert protocol["narrative_factors"]["maximum_live_effect"] <= 0.03
    assert "profit" in protocol["synthetic_lines"]["forbidden_claims"]
    assert protocol["forward_clv"]["minimum_resolved"] == 150
    assert protocol["acceptance"]["expected_delta_required_before_run"] is True
    assert protocol["acceptance"]["shared_core_changes"].startswith("serialize")
