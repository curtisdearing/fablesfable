"""Unknown is not healthy: missing feeds, absent teams and failed identity
resolve to UNKNOWN/degraded -- never OK -- and never to OUT."""
import pandas as pd

from nflvalue.sources import availability as av

TS = "2026-09-22T22:00:00Z"


def _row(name, team, raw, espn_id=None):
    return {"team": team, "name": name, "espn_id": espn_id, "position": "WR",
            "status_raw": raw, "status": av.normalize_status(raw), "date": TS, "comment": ""}


def _players(*rows):
    return pd.DataFrame([{"player_id": pid, "player_name": n, "team": t} for pid, n, t in rows])


P = _players(("a", "Drake London", "ATL"), ("b", "Jayden Reed", "GB"))


def test_failed_injuries_fetch_is_unknown_not_ok_and_no_invented_clock():
    # pipeline_weekly's failure path: rows [] and injuries_fetched_at None
    res = av.resolve_statuses(P, [], clock="wed", injuries_fetched_at=None)
    for st in res["statuses"].values():
        assert st["status"] == "UNKNOWN" and st["status"] != "OUT"
        assert st["availability_state"] == "report_missing"
        assert st["eligibility"] == "degraded"
        assert st["timestamp"] is None
    assert res["report_state"] == "missing" and not av.report_evaluated(res)


def test_empty_feed_with_clock_is_not_a_clean_bill():
    res = av.resolve_statuses(P, [], clock="wed", injuries_fetched_at=TS)
    assert {s["status"] for s in res["statuses"].values()} == {"UNKNOWN"}
    assert res["report_state"] == "empty" and not av.report_evaluated(res)


def test_not_listed_on_received_team_report_is_eligible_not_cleared():
    res = av.resolve_statuses(P, [_row("Other Guy", "ATL", "Questionable"),
                                  _row("Someone Else", "GB", "Out")],
                              clock="wed", injuries_fetched_at=TS)
    st = res["statuses"]["a"]
    assert (st["status"], st["availability_state"], st["eligibility"]) == ("OK", "not_listed", "eligible")
    assert "not listed" in st["source"] and "no injury" not in st["source"]
    assert av.report_evaluated(res)


def test_team_absent_from_report_is_unknown():
    res = av.resolve_statuses(P, [_row("Other Guy", "ATL", "Questionable")],
                              clock="wed", injuries_fetched_at=TS)
    assert res["statuses"]["b"]["status"] == "UNKNOWN"
    assert res["statuses"]["b"]["availability_state"] == "team_not_in_report"
    # an explicit complete-coverage list restores not_listed
    res2 = av.resolve_statuses(P, [_row("Other Guy", "ATL", "Questionable")], reported_teams={"ATL", "GB"},
                               clock="wed", injuries_fetched_at=TS)
    assert res2["statuses"]["b"]["availability_state"] == "not_listed"


def test_cross_team_name_only_match_no_longer_imports_another_players_status():
    # base code: unique name on ANOTHER team -> matched "name_only" -> OUT copied over
    res = av.resolve_statuses(_players(("x", "Jayden Reed", "GB")),
                              [_row("Jayden Reed", "NO", "Out"), _row("Filler", "GB", "Questionable")],
                              clock="wed", injuries_fetched_at=TS)
    st = res["statuses"]["x"]
    assert st["status"] == "UNKNOWN" and st["availability_state"] == "identity_other_team_only"
    assert res["unmatched_espn_rows"][0]["team"] == "NO"          # still visible


def test_duplicate_name_team_rows_are_ambiguous_not_last_write():
    res = av.resolve_statuses(_players(("x", "Jayden Reed", "GB")),
                              [_row("Jayden Reed", "GB", "Out"), _row("Jayden Reed", "GB", "Active")],
                              clock="wed", injuries_fetched_at=TS)
    assert res["statuses"]["x"]["status"] == "UNKNOWN"
    assert res["statuses"]["x"]["availability_state"] == "identity_ambiguous"


def test_unrecognized_designation_keeps_raw_and_is_unknown():
    res = av.resolve_statuses(_players(("x", "Jayden Reed", "GB")), [_row("Jayden Reed", "GB", "Limited")],
                              clock="wed", injuries_fetched_at=TS)
    st = res["statuses"]["x"]
    assert (st["status"], st["status_raw"], st["availability_state"]) == ("UNKNOWN", "Limited", "listed")


def test_espn_id_link_is_authoritative_when_provided():
    players = pd.DataFrame([{"player_id": "x", "player_name": "J. Reed", "team": "GB", "espn_id": "4362"}])
    res = av.resolve_statuses(players, [_row("Jayden Reed", "GB", "Questionable", espn_id="4362")],
                              clock="wed", injuries_fetched_at=TS)
    assert res["statuses"]["x"]["matched_by"] == "espn_id" and res["statuses"]["x"]["status"] == "RISK"


def test_t90_inactive_same_name_on_opponent_does_not_bench_player():
    ina = [{"name": "Jayden Reed", "team": "ATL", "active": False},
           {"name": "Jayden Reed", "team": "GB", "active": True}]
    res = av.resolve_statuses(_players(("x", "Jayden Reed", "GB")), [_row("Filler", "GB", "Out")],
                              inactive_rows=ina, clock="t90", injuries_fetched_at=TS, inactives_fetched_at=TS)
    # base code keyed inactives by name only -> last write (ATL inactive) benched the GB player
    assert res["statuses"]["x"]["status"] == "OK"
    assert res["statuses"]["x"]["availability_state"] == "not_listed"


def test_t90_confirmed_active_resolves_missing_report_unknown():
    res = av.resolve_statuses(_players(("x", "Jayden Reed", "GB")), [], inactive_rows=[
        {"name": "Jayden Reed", "team": "GB", "active": True}], clock="t90",
        injuries_fetched_at=None, inactives_fetched_at=TS)
    assert res["statuses"]["x"]["status"] == "OK" and res["statuses"]["x"]["source"] == "espn_event_roster"


def test_compat_listed_statuses_unchanged():
    res = av.resolve_statuses(_players(("o", "A One", "GB"), ("r", "B Two", "GB"), ("k", "C Three", "GB")),
                              [_row("A One", "GB", "Out"), _row("B Two", "GB", "Questionable"),
                               _row("C Three", "GB", "Active")], clock="wed", injuries_fetched_at=TS)
    got = {k: (v["status"], v["eligibility"], v["timestamp"]) for k, v in res["statuses"].items()}
    assert got == {"o": ("OUT", "ineligible", TS), "r": ("RISK", "eligible", TS), "k": ("OK", "eligible", TS)}
    assert res["summary"]["status"] == {"OUT": 1, "RISK": 1, "OK": 1}


# --------------------------------------------------------------------------- #
# ESPN's pregame 404 "No roster found" is the source answering "not created
# yet" (captured 2026-09-23T22:34Z for event 401872948, both competitors; the
# completed W2 event 401872933 returned 200).  It must read as UNPOPULATED --
# never as inactives data, and not as a fetch defect -- while any other 404 or
# error still fails closed.  The bodies below are shape-only fixtures.
# --------------------------------------------------------------------------- #
import io  # noqa: E402
import json  # noqa: E402
import urllib.error  # noqa: E402

import pytest  # noqa: E402

from nflvalue.sources import availability as _av  # noqa: E402

_SUMMARY = {"header": {"competitions": [{"competitors": [
    {"team": {"id": "9", "abbreviation": "GB"}},
    {"team": {"id": "1", "abbreviation": "ATL"}}]}]}}


def _http_404(url, message):
    body = json.dumps({"error": {"message": message, "code": 404}}).encode()
    return urllib.error.HTTPError(url, 404, "Not Found", {}, io.BytesIO(body))


def _fake_fetch(monkeypatch, roster_error):
    def fake(url, params=None):
        if url.endswith("/summary"):
            return _SUMMARY
        team = url.rstrip("/").split("/")[-2]
        raise roster_error(url, team)
    monkeypatch.setattr(_av, "get_json", fake)


def test_pregame_no_roster_404_is_unpopulated_not_a_fetch_failure(monkeypatch):
    _fake_fetch(monkeypatch, lambda url, team: _http_404(
        url, f"No roster found for team {team} and competition 401872948"))
    res = _av.fetch_event_rosters("401872948")
    assert res["populated"] is False
    assert res["rows"] == [], "an absent roster must never yield inactives rows"
    assert res["fetched_at"], "the source answered, so the fetch is stamped"
    assert "No roster found for team 9 and competition 401872948" in res["reason"]
    assert "GB" in res["reason"] and "ATL" in res["reason"]


@pytest.mark.parametrize("message", [
    "No roster found for team 9 and competition 401872999",   # another event
    "No roster found for team 12 and competition 401872948",  # another team
    "Not Found",
])
def test_other_404s_still_fail_closed(monkeypatch, message):
    _fake_fetch(monkeypatch, lambda url, team: _http_404(url, message))
    with pytest.raises(urllib.error.HTTPError):
        _av.fetch_event_rosters("401872948")


def test_server_errors_still_fail_closed(monkeypatch):
    _fake_fetch(monkeypatch, lambda url, team: urllib.error.HTTPError(
        url, 503, "Unavailable", {}, io.BytesIO(b"")))
    with pytest.raises(urllib.error.HTTPError):
        _av.fetch_event_rosters("401872948")
