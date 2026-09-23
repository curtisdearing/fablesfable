"""Live slate-wide factor context: identity, clocks, practice vs status, degradation.

All payloads are small synthetic fixtures served by an injected ``http``; no socket.
"""

import copy
import json

import pytest

from nflvalue import factor_evidence as fe
from nflvalue import factor_integration as fi
from nflvalue.sources import live_factor_context as lfc

SITE, CORE = lfc.SITE, lfc.CORE
CAP = "2026-09-23T03:00:00Z"


def _event(eid, date, away, home, aid, hid, season=2026):
    return {"id": eid, "date": date, "season": {"year": season, "type": 2},
            "competitions": [{"competitors": [
                {"homeAway": "home", "team": {"abbreviation": home, "id": hid}},
                {"homeAway": "away", "team": {"abbreviation": away, "id": aid}}]}]}


def board(week=3, season=2026, events=None):
    ev = events if events is not None else [
        _event("1", "2026-09-25T00:15Z", "ATL", "GB", "1", "9"),
        _event("2", "2026-09-27T17:00Z", "LAC", "BUF", "24", "2"),
        _event("3", "2026-09-27T20:25Z", "LAR", "WSH", "14", "28")]
    return {"season": {"year": season, "type": 2}, "week": {"number": week}, "events": ev}


PREV = board(2, events=[_event("11", "2026-09-20T17:00Z", "GB", "BUF", "9", "2"),
                        _event("12", "2026-09-20T17:00Z", "ATL", "LAC", "1", "24"),
                        _event("13", "2026-09-21T00:20Z", "WSH", "LAR", "28", "14")])


def inj(eid, name, pos, team, status, date):
    # the feed's own "id" is the injury note id; the athlete id lives in the profile link
    return {"id": f"-9{eid}", "status": status, "date": date,
            "athlete": {"displayName": name, "position": {"abbreviation": pos},
                        "team": {"abbreviation": team},
                        "links": [{"href": f"https://www.espn.com/nfl/player/_/id/{eid}/x"}]}}


INJ = {"injuries": [{"displayName": "x", "injuries": [
    inj("100", "Old Q", "T", "BUF", "Questionable", "2026-09-18T20:00Z"),   # before W2 kickoff
    inj("101", "New Out", "CB", "BUF", "Out", "2026-09-22T18:00Z"),         # after W2 kickoff
    inj("102", "Healthy Guy", "LB", "LAC", "Active", "2026-09-22T18:00Z"),
    inj("103", "Held QB", "QB", "LAC", "Injured Reserve", "2026-09-01T18:00Z"),
    inj("104", "Bye Team", "G", "KC", "Out", "2026-09-22T18:00Z"),          # not on slate
    inj("105", "Future Note", "DE", "GB", "Questionable", "2026-09-24T18:00Z"),  # after capture
]}]}

DEPTH = {"items": [{"name": "3WR 1TE", "positions": {"qb": {"athletes": [
    {"rank": 2, "athlete": {"$ref": f"{CORE}/athletes/222?lang=en"}},
    {"rank": 1, "athlete": {"$ref": f"{CORE}/athletes/111?lang=en"}}]}}}]}

OFFICIAL_W2 = "<title>Official NFL Injury Report for Players - Week 2 of the 2026 Season</title>"


def official_page(week=3):
    row = ("<tr><td><a>{n}</a></td><td>{p}</td><td>{i}</td><td>{pr}</td><td>{g}</td></tr>")
    rows = (row.format(n="Zed Tackle", p="T", i="Knee", pr="Did Not Participate In Practice", g="")
            + row.format(n="Cee Back", p="CB", i="Ankle", pr="Limited Participation in Practice",
                         g="Questionable"))
    return (f"<title>Official NFL Injury Report for Players - Week {week} of the 2026 Season</title>"
            f'<div class="d3-o-section-sub-title"><span>Packers</span></div><table><thead><tr>'
            f"<th>Player</th></tr></thead><tbody>{rows}</tbody></table>")


class FakeHTTP:
    def __init__(self, routes, fail=()):
        self.routes, self.fail, self.calls = routes, tuple(fail), []

    def __call__(self, url):
        self.calls.append(url)
        if any(f in url for f in self.fail):
            raise OSError("network down")
        for key, val in self.routes.items():
            if key in url:
                body = val if isinstance(val, str) else json.dumps(val)
                return 200, {"Date": "Wed, 23 Sep 2026 03:00:00 GMT"}, body.encode()
        return 404, {}, b""


def routes(**over):
    r = {"week=3&": board(), "week=2&": PREV, "nfl.com": OFFICIAL_W2,
         "/injuries": INJ, "/depthcharts": DEPTH}
    r.update(over)
    return r


def build(http, **kw):
    return lfc.build_live_context(2026, 3, captured_at=CAP, http=http, **kw)


def by_story(doc):
    return {i["story_id"]: i for i in doc["news"]}


# ---- identity -------------------------------------------------------------- #

def test_scoreboard_for_another_week_or_season_is_rejected():
    with pytest.raises(lfc.IdentityMismatch):
        lfc.slate_from_scoreboard(board(week=2), 2026, 3)
    with pytest.raises(lfc.IdentityMismatch):
        lfc.slate_from_scoreboard(board(season=2025), 2026, 3)
    stray = board(events=[_event("9", "2025-09-25T00:15Z", "ATL", "GB", "1", "9", season=2025)])
    with pytest.raises(lfc.IdentityMismatch):
        lfc.slate_from_scoreboard(stray, 2026, 3)


def test_slate_uses_nflverse_codes():
    ids = [g["game_id"] for g in lfc.slate_from_scoreboard(board(), 2026, 3)]
    assert ids == ["2026_03_ATL_GB", "2026_03_LAC_BUF", "2026_03_LA_WAS"]


def test_official_page_serving_prior_week_is_rejected_not_parsed():
    with pytest.raises(lfc.IdentityMismatch):
        lfc.parse_official_report(official_page(week=2), 2026, 3)
    doc = build(FakeHTTP(routes()))
    assert doc["routes"]["official_report"].startswith("failed:")
    assert not [i for i in doc["news"] if i["source_tier"] == "league_official"]


def test_cross_game_identity_links_only_unique_same_team_ids():
    m = [{"espn_id": "101", "team": "BUF", "gsis_id": "00-1"},
         {"espn_id": "103", "team": "KC", "gsis_id": "00-2"},       # other team
         {"espn_id": "105", "team": "GB", "gsis_id": "00-3"},
         {"espn_id": "105", "team": "GB", "gsis_id": "00-4"}]       # ambiguous
    assert lfc.link_player("101", "BUF", m)[0] == "00-1"
    rows = lfc.parse_espn_injuries(INJ)
    assert rows[1]["espn_id"] == "101" and rows[1]["note_id"] == "-9101"   # note id is not identity
    eid, _, note = lfc.link_player("103", "LAC", m)
    assert eid == "espn:103" and "not found" in note
    eid, _, note = lfc.link_player("105", "GB", m)
    assert eid == "espn:105" and "ambiguous" in note
    doc = build(FakeHTTP(routes()), id_map=m)
    assert all(i["team"] in i["game_id"].split("_")[2:] or i["team"] == "LA"
               for i in doc["news"])
    assert not [i for i in doc["news"] if i["team"] == "KC"]   # bye team never attached
    assert by_story(doc)["espn_inj:2026_03_LAC_BUF:101"]["entity_id"] == "00-1"


# ---- clocks ---------------------------------------------------------------- #

def test_prior_week_designation_is_superseded_same_source_clock():
    doc = build(FakeHTTP(routes()))
    old = by_story(doc)["espn_inj:2026_03_LAC_BUF:100"]
    assert old["claim_key"] == "prior_game_status"
    assert old["expires_at"] == "2026-09-20T17:00:00Z"
    assert old["published_at"] == "2026-09-18T20:00:00Z"
    rec = fe.assess_news([old], "2026-09-23T04:00:00Z")[0]
    assert rec["status"] == "unavailable_unverified" and "superseded" in rec["reason_not_applied"]
    new = by_story(doc)["espn_inj:2026_03_LAC_BUF:101"]
    assert new["claim_key"] == "game_status_report" and new["claim_kind"] == "report"
    rec = fe.assess_news([new], "2026-09-23T04:00:00Z")[0]
    assert rec["verified"] is False and rec["status"] == "context_only"


def test_feed_note_after_capture_is_dropped_and_later_capture_not_known_earlier():
    doc = build(FakeHTTP(routes()))
    assert "espn_inj:2026_03_ATL_GB:105" not in by_story(doc)
    new = by_story(doc)["espn_inj:2026_03_LAC_BUF:101"]
    rec = fe.assess_news([new], "2026-09-22T19:00:00Z")[0]   # run before capture
    assert rec["cutoff_ok"] is False and rec["status"] == "unavailable_unverified"


# ---- practice vs game status ------------------------------------------------ #

def test_practice_dnp_is_not_a_game_status():
    doc = build(FakeHTTP(routes(**{"nfl.com": official_page(3)})))
    assert doc["routes"]["official_report"] == "ok"
    off = [i for i in doc["news"] if i["source_tier"] == "league_official"]
    zed = [i for i in off if "Zed" in i["claim"]]
    assert [i["claim_key"] for i in zed] == ["practice"] and zed[0]["claim_value"] == "DNP"
    cee = {i["claim_key"]: i["claim_value"] for i in off if "Cee" in i["claim"]}
    assert cee == {"practice": "LP", "game_status": "Questionable"}
    assert doc["coverage"]["2026_03_ATL_GB"]["ol_injury"]["state"] == "official"


# ---- unknown is not healthy / coverage ----------------------------------------- #

def test_active_rows_are_not_health_claims_and_empty_is_not_complete():
    doc = build(FakeHTTP(routes()))
    assert not [i for i in doc["news"] if "Healthy Guy" in i["claim"]]
    cov = doc["coverage"]
    assert set(cov) == {"2026_03_ATL_GB", "2026_03_LAC_BUF", "2026_03_LA_WAS"}
    for gid, row in cov.items():
        assert set(row) == set(lfc.CATEGORIES)
        for cat, c in row.items():
            assert c["state"] in ("official", "feed_report", "not_obtained")
            if c["state"] != "official":
                assert c["reason"]
                assert any(r["factor_id"] == f"coverage:{cat}:{gid}" for r in doc["records"])
    assert cov["2026_03_LA_WAS"]["ol_injury"]["state"] == "not_obtained"
    assert "not evidence of health" in cov["2026_03_LA_WAS"]["ol_injury"]["reason"]


def test_document_loads_through_factor_integration_for_every_game(tmp_path):
    doc = build(FakeHTTP(routes()))
    p = tmp_path / "ctx.json"
    p.write_text(json.dumps(doc))
    out = fi.load_context(2026, 3, [g["game_id"] for g in doc["games"]], "2026-09-23T04:00:00Z",
                          path=str(p))
    assert set(out["games_with_context"]) == {g["game_id"] for g in doc["games"]}
    for r in out["records"]:
        assert r["status"] != "numeric_applied"
    unknown = [r for r in out["records"] if r["factor_id"].startswith("coverage:")]
    assert unknown and all(r["status"] == "unavailable_unverified" for r in unknown)


def test_qb_depth_chart_is_observed_not_verified_starter():
    doc = build(FakeHTTP(routes()), id_map=[{"espn_id": "111", "team": "GB", "gsis_id": "00-9"}])
    r = next(r for r in doc["records"] if r["factor_id"] == "qb_depth:GB:2026_03_ATL_GB")
    assert r["value"] == "00-9" and r["verified"] is False
    assert fe.normalize_record({**r, "as_of": "2026-09-23T04:00:00Z"})["status"] == "unavailable_unverified"


def test_partial_coverage_one_team_depth_chart_fails():
    doc = build(FakeHTTP(routes(), fail=("/teams/28/depthcharts",)))
    assert doc["routes"]["espn_depthchart_qb"] == "5/6 teams"
    assert doc["coverage"]["2026_03_LA_WAS"]["qb_news"]["state"] == "feed_report"
    assert not any(r["factor_id"] == "qb_depth:WAS:2026_03_LA_WAS" for r in doc["records"])


# ---- network failure degradation ------------------------------------------------ #

def test_network_failure_degrades_to_fallback_then_unknown_with_bounded_calls():
    http = FakeHTTP(routes(), fail=("nfl.com", "/injuries", "/summary", "/depthcharts"))
    doc = build(http)
    r = doc["routes"]
    assert r["official_report"].startswith("failed") and r["espn_injuries"].startswith("failed")
    assert r["espn_summary_fallback"] == "0/3 events"
    assert doc["news"] == []
    assert all(c["state"] == "not_obtained" for row in doc["coverage"].values() for c in row.values())
    # each route attempted once: 2 boards + 1 official + 1 feed + 3 summaries + 6 depth charts
    assert len(http.calls) == 13


def test_request_budget_caps_calls():
    http = FakeHTTP(routes())
    doc = build(http, max_requests=4)
    assert len(http.calls) == 4
    assert doc["routes"]["espn_depthchart_qb"] == "0/6 teams"


def test_slate_failure_raises_instead_of_inventing_games():
    with pytest.raises(lfc.RouteFailed):
        build(FakeHTTP(routes(), fail=("week=3&",)))


# ---- curated preservation -------------------------------------------------------- #

def test_curated_items_kept_verbatim_and_other_weeks_ignored():
    cur = {"season": 2026, "week": 3,
           "news": [{"story_id": "atl_penix", "game_id": "2026_03_ATL_GB", "category": "qb_news",
                     "claim": "x", "source_tier": "team_official"}],
           "records": [{"factor_id": "qb_starter:GB:2026_03_ATL_GB", "game_id": "2026_03_ATL_GB",
                        "category": "qb_news"}]}
    snap = copy.deepcopy(cur)
    doc = build(FakeHTTP(routes()), curated=cur)
    assert cur == snap
    assert doc["news"][0] == cur["news"][0] and doc["records"][0] == cur["records"][0]
    assert doc["curated_games_kept"] == ["2026_03_ATL_GB"]
    doc2 = build(FakeHTTP(routes()), curated={**cur, "week": 2})
    assert doc2["curated_games_kept"] == []


# --------------------------------------------------------------------------- #
# Team-official (club-site) report: game status vs practice ESTIMATE, identity
# checks.  Shape-only fixture modelled on the club CMS layout; not real rows.
# --------------------------------------------------------------------------- #
CLUB_URL = "https://www.packers.com/news/club-report-w3"


def club_page(week=3, published="2026-09-23T02:00:00Z", second_team="Atlanta Falcons"):
    def table(team, head, rows):
        th = "".join(f"<th>{h}</th>" for h in head)
        trs = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
        return f"<h3>{team}</h3><table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>"
    pub = f'<script type="application/ld+json">{{"datePublished": "{published}"}}</script>' if published else ""
    return (f"<title>Packers list two questionable vs. Falcons | Week {week} Injury Report</title>{pub}"
            + table("Green Bay Packers", ["Player", "Injury", "*Monday", "*Tuesday", "Game Status"], [
                ["Wide Out, WR", "Neck", "Did Not Participate", "Did Not Participate", "Out"],
                ["Big Guard, G", "Hand", "Did Not Participate", "Limited Participation", "--"]])
            + "<p>* The Packers held a walkthrough; participation reports are an estimation.</p>"
            + table(second_team, ["Player", "Injury", "*Monday", "Tuesday", "Game Status"], [
                ["Nickel Back, CB", "Achilles", "Full Participation", "Full Participation", "Questionable"]]))


def test_club_report_separates_game_status_from_estimated_practice():
    got = lfc.parse_club_report(club_page(), 2026, 3, home="GB", away="ATL")
    assert got["published_at"] == "2026-09-23T02:00:00Z" and got["teams"] == ["ATL", "GB"]
    rows = {r["name"]: r for r in got["rows"]}
    assert rows["Wide Out"]["team"] == "GB" and rows["Wide Out"]["game_status"] == "Out"
    assert rows["Big Guard"]["game_status"] is None, "'--' is no designation, not healthy"
    assert rows["Big Guard"]["practice"][-1] == {"day": "tuesday", "status": "LP", "estimated": True}
    assert rows["Nickel Back"]["team"] == "ATL"
    assert rows["Nickel Back"]["practice"][-1]["estimated"] is False


@pytest.mark.parametrize("page, why", [
    (club_page(week=2), "Week 3"),
    (club_page(published=None), "datePublished"),
    (club_page(second_team="Chicago Bears"), "not ATL@GB"),
])
def test_club_report_identity_is_strict(page, why):
    with pytest.raises(lfc.IdentityMismatch, match=why):
        lfc.parse_club_report(page, 2026, 3, home="GB", away="ATL")


def test_club_report_route_feeds_official_items_and_stays_context():
    doc = build(FakeHTTP(routes(**{"club-report-w3": club_page()})),
                club_reports={"2026_03_ATL_GB": CLUB_URL, "2026_03_KC_DEN": CLUB_URL})
    assert doc["routes"]["club_report:2026_03_ATL_GB"] == "ok"
    assert doc["routes"]["club_report:2026_03_KC_DEN"].startswith("failed: 2026_03_KC_DEN is not on")
    news = {i["story_id"]: i for i in doc["news"] if i["story_id"].startswith("club_")}
    out = news["club_status:2026_03_ATL_GB:GB:wide_out"]
    assert (out["source_tier"], out["claim_kind"], out["claim_value"]) == ("team_official", "confirmed", "Out")
    assert out["published_at"] == "2026-09-23T02:00:00Z" and out["fetched_at"] == CAP
    est = news["club_practice:2026_03_ATL_GB:GB:big_guard"]
    assert est["claim_kind"] == "report" and "estimated" in est["claim"]
    assert "club_status:2026_03_ATL_GB:GB:big_guard" not in news, "no designation -> no status claim"
    assert doc["coverage"]["2026_03_ATL_GB"]["ol_injury"]["state"] == "official"
    recs = fe.assess_news(list(news.values()), CAP)
    assert recs and all(r["status"] != "numeric_applied" for r in recs), "news never self-promotes"
