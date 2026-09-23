"""Unknown availability through the real consumers: stamps, synthesis, cards, panels.

Unknown is neither confirmed healthy nor OUT.  A run-level "report received" does not
make every player's availability established.
"""
import json

import pandas as pd

from nflvalue import factor_integration as fimod
from nflvalue import pick_cards as pc
from nflvalue import synthesis as synmod
from nflvalue.sources import availability as avmod

AS_OF = "2026-09-23T02:00:00Z"


def _players():
    return pd.DataFrame([
        {"player_id": "A1", "player_name": "Alpha One", "team": "ATL"},
        {"player_id": "A2", "player_name": "Alpha Two", "team": "ATL"},
        {"player_id": "G1", "player_name": "Gee One", "team": "GB"},
    ])


def _resolve(rows, fetched=AS_OF, **kw):
    return avmod.resolve_statuses(_players(), rows, injuries_fetched_at=fetched, **kw)


def test_six_availability_situations_are_distinct():
    rows = [{"team": "ATL", "name": "Alpha One", "status_raw": "Questionable",
             "status": "RISK", "date": "2026-09-22T20:00Z"},
            {"team": "GB", "name": "Somebody Else", "status_raw": "Out", "status": "OUT"}]
    st = _resolve(rows)["statuses"]
    assert st["A1"]["evidence_kind"] == "game_status_feed"          # feed game status
    assert st["A2"]["evidence_kind"] == "not_listed_on_received_feed"  # complete team, unlisted
    assert st["A2"]["status"] == "OK" and st["A2"]["eligibility"] == "eligible"
    missing = _resolve(rows, fetched=None)["statuses"]                 # report missing/failed
    assert all(s["status"] == "UNKNOWN" and s["eligibility"] == "degraded" for s in missing.values())
    other = _resolve([{"team": "GB", "name": "Alpha Two", "status_raw": "Out", "status": "OUT"}])
    assert other["statuses"]["A2"]["availability_state"] == "identity_other_team_only"   # failed match
    assert other["statuses"]["A2"]["status"] == "UNKNOWN"
    t90 = avmod.resolve_statuses(_players(), rows, clock="t90", injuries_fetched_at=AS_OF,
                                 inactive_rows=[{"name": "Alpha Two", "team": "ATL", "active": False}])
    assert t90["statuses"]["A2"]["evidence_kind"] == "confirmed_inactive_event_roster"
    assert t90["statuses"]["A2"]["status"] == "OUT"


def test_stale_feed_designation_is_disclosed_not_silently_this_week():
    rows = [{"team": "ATL", "name": "Alpha One", "status_raw": "Out", "status": "OUT",
             "date": "2026-09-10T12:00Z"}]
    st = _resolve(rows, prior_kickoff={"ATL": "2026-09-14T17:00:00Z"})["statuses"]["A1"]
    assert st["designation_predates_previous_game"] is True


def _cands():
    return pd.DataFrame([{"player_id": p, "team": t, "market": "receiving_yards", "mean": 40.0}
                         for p, t in (("A1", "ATL"), ("A2", "ATL"), ("G1", "GB"))])


def test_teammate_absence_stages_are_not_neutral_when_a_teammate_is_unknown():
    statuses = {"A1": {"status": "UNKNOWN", "eligibility": "degraded",
                       "availability_state": "identity_ambiguous"},
                "A2": {"status": "OK", "eligibility": "eligible", "availability_state": "not_listed"},
                "G1": {"status": "OK", "eligibility": "eligible", "availability_state": "not_listed"}}
    ran = {s: True for s in fimod.STAGES}
    stamps = fimod.build_stamps(_cands(), ran, {}, availability=statuses)
    a2 = stamps[("A2", "receiving_yards")]
    assert a2["stages"]["realloc_volume"]["state"] == "not_evaluated"
    assert "1 teammate" in a2["stages"]["realloc_volume"]["reason"]
    assert stamps[("G1", "receiving_yards")]["stages"]["realloc_volume"]["state"] == "no_change"
    assert stamps[("A1", "receiving_yards")]["availability"]["eligibility"] == "degraded"


def test_player_missing_from_resolver_is_unknown_not_healthy():
    stamps = fimod.build_stamps(_cands(), {s: True for s in fimod.STAGES}, {}, availability={})
    assert stamps[("G1", "receiving_yards")]["availability"]["status"] == "UNKNOWN"


def test_synthesis_never_maps_unknown_or_missing_status_to_ok():
    def player(report):
        return {"player_id": "A1", "name": "Alpha One", "pos": "WR", "team": "ATL",
                "model_projection": {"market": "receiving_yards", "mean": 50, "sd": 20,
                                     "line": 45.5, "p_over": 0.6, "p_under": 0.4},
                "availability": {"report_status": report, "source": "x", "timestamp": AS_OF}}
    for report in ("UNKNOWN", None, ""):
        inp = synmod.build_input(as_of=AS_OF, week=3, game_id="g", matchup="ATL@GB",
                                 data_freshness={"injuries_updated": AS_OF, "lines_updated": AS_OF},
                                 players=[player(report)])
        out = synmod.synthesize(inp)
        p = out["players"][0]
        assert p["status"] == "UNKNOWN", report
        assert p["confidence"] == "low"


def _lean(avail):
    return {"name": "Alpha One", "player_id": "A1", "game_id": "2026_03_ATL_GB",
            "market": "receiving_yards", "side": "over", "line": 45.5, "line_source": "odds_api",
            "price": 1.87, "quote_book": "draftkings", "quote_ts": "2026-09-24T18:30:00Z",
            "as_of": "2026-09-24T18:35:00Z", "mean": 52.0, "sd": 26.0, "p_side": 0.58,
            "status": "active", "_quote_verified": True,
            "stage_json": json.dumps({"team": "ATL", "stages": {}, "availability": avail}
                                     if avail is not None else {"team": "ATL", "stages": {}})}


def test_card_with_unknown_or_unrecorded_availability_is_not_executable():
    import datetime as dt
    now = dt.datetime(2026, 9, 24, 20, 0, tzinfo=dt.timezone.utc)
    ok = pc.build_card(_lean({"status": "OK", "eligibility": "eligible",
                              "availability_state": "not_listed"}), now)
    assert ok["status"] == "watch"
    unk = pc.build_card(_lean({"status": "UNKNOWN", "eligibility": "degraded",
                               "availability_state": "team_not_in_report"}), now)
    assert unk["status"] == "research" and unk["quote"] is None
    assert "not established" in " ".join(unk["status_reasons"])
    none = pc.build_card(_lean(None), now)
    assert none["status"] == "research"


def test_card_availability_record_labels_follow_the_evidence():
    lean = _lean(None)
    stamps = json.loads(lean["stage_json"])
    missing = fimod.availability_record(lean, stamps, AS_OF)
    assert missing["status"] == "unavailable_unverified"
    stamps["availability"] = {"status": "OK", "eligibility": "eligible",
                              "availability_state": "not_listed",
                              "evidence_kind": "not_listed_on_received_feed", "timestamp": AS_OF}
    rec = fimod.availability_record(lean, stamps, AS_OF)
    assert rec["status"] == "considered_no_change" and "Not a medical clearance" in rec["observation"]
    stamps["availability"] = {"status": "UNKNOWN", "eligibility": "degraded",
                              "availability_state": "report_missing"}
    rec = fimod.availability_record(lean, stamps, AS_OF)
    assert rec["status"] == "unavailable_unverified" and "no injury report" in rec["observation"]
