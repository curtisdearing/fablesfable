"""Club-site injury reports reach the run through the real context consumer chain.

``pipeline_weekly._run_context_doc`` (shared by the Wednesday and T-90 runs) reads the
week's ``club_reports`` registry from the committed context file, re-fetches each report
under this run's clock, ``factor_integration.load_context`` turns it into records, and
``persist_run`` stores them under the issuing run id.  Network is replaced by a fake that
serves a shape-only scoreboard and the captured packers.com Week-3 extract.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import pipeline_weekly as pw
from nflvalue import db as dbmod
from nflvalue import factor_integration as fimod
from nflvalue.sources import live_factor_context as lfc

FIX = Path(__file__).parent / "fixtures" / "club_report_packers_2026w3.html"
URL = ("https://www.packers.com/news/"
       "packers-rule-out-four-list-two-questionable-vs-falcons-week-3-injury-report-2026")
GID = "2026_03_ATL_GB"


def _board(week):
    ev = {"id": "401872948", "date": "2026-09-25T00:15Z", "season": {"year": 2026, "type": 2},
          "competitions": [{"competitors": [
              {"homeAway": "home", "team": {"abbreviation": "GB", "id": "9"}},
              {"homeAway": "away", "team": {"abbreviation": "ATL", "id": "1"}}]}]}
    return {"season": {"year": 2026, "type": 2}, "week": {"number": week}, "events": [ev]}


@pytest.fixture
def live_run(monkeypatch, tmp_path):
    calls = []

    def http(url, timeout=20.0):
        calls.append(url)
        if "scoreboard" in url and "week=3" in url:
            return 200, {}, json.dumps(_board(3)).encode()
        if url == URL:
            return 200, {}, FIX.read_bytes()
        return 404, {}, b""
    monkeypatch.setattr(lfc, "default_http", http)
    ctx = tmp_path / "2026-w03.json"
    stale = {"story_id": f"club_status:{GID}:GB:old_capture", "source_tier": "team_official",
             "game_id": GID, "claim": "stale capture from an earlier run"}
    ctx.write_text(json.dumps({"schema": "factor_context/1", "season": 2026, "week": 3,
                               "club_reports": {GID: URL}, "news": [stale], "records": []}))
    monkeypatch.setattr(fimod, "context_path", lambda s, w: str(ctx))
    return calls


def _records(doc, as_of):
    return fimod.load_context(2026, 3, [GID], as_of, doc=doc)["records"]


def test_run_context_doc_fetches_registered_club_report(live_run):
    doc, label, meta = pw._run_context_doc({}, 2026, 3, "live", None, None)
    assert URL in live_run, "the registered club report was not fetched by the run"
    assert meta["club_reports"] == [GID] and doc["routes"][f"club_report:{GID}"] == "ok"
    ids = {i["story_id"] for i in doc["news"]}
    assert f"club_status:{GID}:GB:old_capture" not in ids, "earlier capture inherited as curated"
    assert f"club_status:{GID}:GB:jayden_reed" in ids


def test_club_statuses_and_estimates_persist_under_the_run_id(live_run, tmp_path):
    doc, _, _ = pw._run_context_doc({}, 2026, 3, "live", None, None)
    as_of = doc["captured_at"]  # the run stamps its decision clock after the fetch
    recs = _records(doc, as_of)
    early = {r["factor_id"]: r for r in _records(doc, "2026-09-23T19:00:00Z")}
    assert early[f"news:club_status:{GID}:GB:jayden_reed"]["status"] == "unavailable_unverified", \
        "a report captured after an earlier decision clock must not read as known then"
    conn = dbmod.connect(str(tmp_path / "isolated.db"))
    fimod.persist_run(conn, 2026, 3, "wed", {"run_id": "test-run-1", "as_of": as_of}, recs, [GID])
    stored = {r["factor_id"]: r for r in fimod.load_context_records(conn, 2026, 3)[("test-run-1", GID)]}
    reed = stored[f"news:club_status:{GID}:GB:jayden_reed"]
    assert reed["verified"] is True and reed["status"] == "context_only"
    est = stored[f"news:club_practice:{GID}:GB:jayden_reed"]
    assert est["verified"] is False, "an estimated practice day is not a confirmed observation"
    # no designation != healthy: Penix has a practice record and NO game-status record
    assert f"news:club_status:{GID}:ATL:michael_penix" not in stored
    assert f"news:club_practice:{GID}:ATL:michael_penix" in stored
    # context never changes a number
    assert all(r["status"] != "numeric_applied" and not r.get("consumed") for r in stored.values())


def test_without_a_registry_entry_no_club_route_runs(live_run, tmp_path, monkeypatch):
    ctx = tmp_path / "bare.json"
    ctx.write_text(json.dumps({"schema": "factor_context/1", "season": 2026, "week": 3,
                               "news": [], "records": []}))
    monkeypatch.setattr(fimod, "context_path", lambda s, w: str(ctx))
    doc, _, meta = pw._run_context_doc({}, 2026, 3, "live", None, None)
    assert meta["club_reports"] == [] and URL not in live_run
    assert not [k for k in doc["routes"] if k.startswith("club_report:")]
