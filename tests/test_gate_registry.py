"""Gate registry: verdicts collected faithfully, dashboard wiring, fail-safe."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nflvalue import gate_registry
from nflvalue.gate_registry import collect


REQUIRED_KEYS = {"name", "scope", "verdict", "numbers", "source", "date"}
ALLOWED_VERDICTS = {"shipped", "rejected", "retained", "research_only"}


def test_collect_entries_are_well_formed():
    entries = collect()
    assert entries, "registry should never be empty (static entries exist)"
    for e in entries:
        assert REQUIRED_KEYS <= set(e), e
        assert e["verdict"] in ALLOWED_VERDICTS, e


def test_collect_is_sorted_newest_first():
    dates = [e["date"] for e in collect()]
    assert dates == sorted(dates, reverse=True)


def test_registry_matches_fair_value_book():
    """A gate-FAILED market must never appear as 'shipped'."""
    path = os.path.join(gate_registry.BOOK_DIR, "fair_value.json")
    if not os.path.exists(path):
        return
    with open(path) as fh:
        book = json.load(fh)
    entries = {e["name"]: e for e in collect()}
    for market, res in book["markets"].items():
        e = entries[f"Fair-value market blend ({market})"]
        expected = "shipped" if res["gate"]["passed"] else "rejected"
        assert e["verdict"] == expected


def test_registry_survives_missing_books(tmp_path, monkeypatch):
    monkeypatch.setattr(gate_registry, "BOOK_DIR", str(tmp_path))
    entries = collect()                     # only static entries remain
    assert entries
    assert all(e["source"].startswith(("docs/", "commit")) for e in entries)


def test_registry_survives_corrupt_book(tmp_path, monkeypatch):
    (tmp_path / "fair_value.json").write_text("{broken")
    monkeypatch.setattr(gate_registry, "BOOK_DIR", str(tmp_path))
    assert collect()                        # no exception, static entries intact


def test_dashboard_payload_carries_registry(tmp_path):
    from nflvalue import dashboard
    out = tmp_path / "dash.html"
    dashboard.write_dashboard({"mode": "demo"}, str(out))
    html = out.read_text()
    start = html.index("__DATA_JSON__") if "__DATA_JSON__" in html else None
    assert start is None, "placeholder must be substituted"
    assert "gate_registry" in html
    assert "Measured gates" in html
