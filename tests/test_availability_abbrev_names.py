"""Candidate names are nflverse abbreviations ('D.London', 'Bi.Robinson');
ESPN injury/event-roster rows carry full names. An exact-name lookup matched
none of them, so a real OUT/Doubtful listing resolved to OK (2026-09-22 run:
GB WR Jayden Reed, ESPN 'Doubtful', stayed an OK candidate as 'J.Reed')."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nflvalue.sources import availability as av  # noqa: E402

TS = "2026-09-22T22:00:00Z"


def _row(name, team, status, raw):
    return {"team": team, "name": name, "espn_id": "1", "position": "WR",
            "status_raw": raw, "status": status, "date": TS, "comment": ""}


def _players(*rows):
    return pd.DataFrame([{"player_id": pid, "player_name": n, "team": t} for pid, n, t in rows])


def test_abbreviated_candidate_matches_full_espn_name():
    res = av.resolve_statuses(_players(("p1", "D.London", "ATL"), ("p2", "J.Reed", "GB")),
                              [_row("Drake London", "ATL", "OUT", "Out"),
                               _row("Jayden Reed", "GB", "RISK", "Doubtful")],
                              clock="wed", injuries_fetched_at=TS)
    assert res["statuses"]["p1"]["status"] == "OUT"
    assert res["statuses"]["p2"]["status"] == "RISK"
    assert res["unmatched_espn_rows"] == []


def test_two_letter_prefix_disambiguates_and_ambiguity_never_matches():
    rows = [_row("Brian Robinson Jr.", "ATL", "OUT", "Out")]
    res = av.resolve_statuses(_players(("bi", "Bi.Robinson", "ATL"), ("br", "Br.Robinson", "ATL")),
                              rows, clock="wed", injuries_fetched_at=TS)
    assert res["statuses"]["bi"]["status"] == "OK"
    assert res["statuses"]["br"]["status"] == "OUT"
    amb = av.resolve_statuses(_players(("j", "J.Smith", "GB")),
                              [_row("Jonnu Smith", "GB", "OUT", "Out"),
                               _row("Jaire Smith", "GB", "OK", "Active")],
                              clock="wed", injuries_fetched_at=TS)
    assert amb["statuses"]["j"]["status"] == "UNKNOWN"     # ambiguous: no guess, no OK
    assert amb["statuses"]["j"]["availability_state"] == "identity_ambiguous"
    assert av.resolve_statuses(_players(("x", "D.London", "GB")),
                               [_row("Drake London", "ATL", "OUT", "Out")], reported_teams={"ATL", "GB"},
                               clock="wed", injuries_fetched_at=TS)["statuses"]["x"]["status"] == "OK"


def test_t90_inactive_matches_abbreviated_candidate():
    res = av.resolve_statuses(_players(("p1", "D.London", "ATL")), [],
                              inactive_rows=[{"name": "Drake London", "team": "ATL", "active": False}],
                              clock="t90", injuries_fetched_at=TS, inactives_fetched_at=TS)
    assert res["statuses"]["p1"]["status"] == "OUT"
