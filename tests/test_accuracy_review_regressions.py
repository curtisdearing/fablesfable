"""Regression gates from the 2026-09-08 accuracy-patch review
(reports/claude_accuracy_review.md). Each test names the defect it pins.

Executed RED against the unreviewed patch (81d3b64 + uncommitted diff), then
GREEN after the repair; see the report for the log paths.
"""
from __future__ import annotations

import datetime as dt
import io
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pipeline_weekly as pw  # noqa: E402
from nflvalue import db as dbmod  # noqa: E402
from nflvalue import prop_decision as pdm  # noqa: E402
from nflvalue import report as rptmod  # noqa: E402
from nflvalue.composite import score_candidate  # noqa: E402
from nflvalue.projection import p_over as dist_p_over  # noqa: E402
from nflvalue.shortlist import rank_game  # noqa: E402
from analysis.prop_probability_grade import grade_rows  # noqa: E402
from tests.test_pipeline_weekly import _fresh_feeds, _roster, env  # noqa: E402,F401
from tests.test_report_phase2 import SEASON, WEEK, synthetic_inputs  # noqa: E402

NOW = dt.datetime(2026, 9, 9, 16, 0, tzinfo=dt.timezone.utc)
STAMP = "2026-09-09T12:00:00Z"
GAME_ID = f"{SEASON}_09_AAA_BBB"


def _stamp_now():
    from nflvalue.freshness import stamp_now
    return stamp_now()


# =========================================================================== #
# A. Active-roster validity
# =========================================================================== #
def test_live_run_has_a_real_roster_source_not_only_injection(env, monkeypatch):  # noqa: F811
    """P0: the patch read the roster from ``inject`` only, so every real run
    (no injection) failed 'active roster data missing'. The production path
    must call a roster SOURCE; here it is stubbed at the source boundary."""
    from nflvalue.sources import active_roster as armod
    calls = []
    stamp = _stamp_now()          # the asset's Last-Modified predates the run

    def fake_fetch(season, week=None, http=None):
        calls.append(season)
        return _roster(stamp)
    monkeypatch.setattr(armod, "fetch_active_roster", fake_fetch)
    feeds = _fresh_feeds(stamp)
    feeds.pop("active_roster")                         # NO injection
    res = pw.run_week(SEASON, WEEK, mode="live", inputs=synthetic_inputs(), inject_feeds=feeds)
    assert calls == [SEASON], "the roster source was never consulted"
    assert res["publish"] is True, res["publish_reasons"]
    assert res["roster_gate"]["publish"] is True


def test_roster_source_failure_fails_closed_with_diagnostic(env, monkeypatch):  # noqa: F811
    from nflvalue.sources import active_roster as armod

    def boom(season, week=None, http=None):
        raise RuntimeError("HTTP 503")
    monkeypatch.setattr(armod, "fetch_active_roster", boom)
    feeds = _fresh_feeds(_stamp_now())
    feeds.pop("active_roster")
    res = pw.run_week(SEASON, WEEK, mode="live", inputs=synthetic_inputs(), inject_feeds=feeds)
    assert res["publish"] is False
    assert any("active_roster" in r or "active roster" in r for r in res["publish_reasons"])
    assert res["games"] == [] or all(not g["leans"] for g in res["games"])


def test_roster_source_parses_the_nflverse_asset_and_keeps_its_own_timestamp():
    """The snapshot clock is the asset's Last-Modified, not our fetch time."""
    from nflvalue.sources import active_roster as armod
    frame = pd.DataFrame([
        {"season": 2026, "week": 1, "team": "SEA", "status": "ACT", "gsis_id": "00-1",
         "full_name": "A Active", "position": "WR"},
        {"season": 2026, "week": 1, "team": "SEA", "status": "DEV", "gsis_id": "00-2",
         "full_name": "B Squad", "position": "RB"},
        {"season": 2026, "week": 1, "team": "NE", "status": "RES", "gsis_id": None,
         "full_name": "No Id", "position": "TE"},
    ])
    buf = io.BytesIO()
    frame.to_parquet(buf, index=False)

    def http(url):
        assert "roster_weekly_2026.parquet" in url
        return buf.getvalue(), {"Last-Modified": "Mon, 07 Sep 2026 13:10:26 GMT"}
    snap = armod.fetch_active_roster(2026, http=http)
    assert snap["snapshot_at"] == "2026-09-07T13:10:26Z"
    assert snap["week"] == 1 and snap["n_rows"] == 2 and snap["n_unidentified"] == 1
    assert {r["player_id"] for r in snap["rows"]} == {"00-1", "00-2"}


@pytest.mark.parametrize("roster,reason_fragment", [
    (None, "missing"),
    ({"rows": [], "snapshot_at": STAMP, "season": 2026, "week": 1}, "no rows"),
    ({"rows": [{"player_id": "x", "team": "AAA", "status": "ACT", "week": 1}],
      "snapshot_at": None, "fetched_at": None, "season": 2026, "week": 1}, "timestamp"),
    ({"rows": [{"player_id": "x", "team": "AAA", "status": "ACT", "week": 1}],
      "snapshot_at": "2026-09-01T00:00:00Z", "season": 2026, "week": 1}, "stale"),
    ({"rows": [{"player_id": "x", "team": "AAA", "status": "ACT", "week": 1}],
      "snapshot_at": "2026-09-12T00:00:00Z", "season": 2026, "week": 1}, "future"),
    ({"rows": [{"player_id": "x", "team": "AAA", "status": "ACT", "week": 1}],
      "snapshot_at": STAMP, "season": 2025, "week": 1}, "season 2025"),
    ({"rows": [{"player_id": "x", "team": "AAA", "status": "ACT", "week": 3}],
      "snapshot_at": STAMP, "season": 2026, "week": 3}, "covers week 3"),
    ({"rows": [{"player_id": "x", "team": "AAA", "status": "ACT", "week": 1}],
      "snapshot_at": STAMP, "season": 2026, "week": 1}, "slate team(s) BBB"),
])
def test_roster_snapshot_validation_fails_closed(roster, reason_fragment):
    g = pdm.validate_roster_snapshot(roster, season=2026, week=1,
                                     slate_teams={"AAA", "BBB"}, now=NOW)
    assert g["publish"] is False
    assert reason_fragment in g["reason"]


def test_roster_snapshot_from_previous_week_label_is_accepted_when_fresh():
    roster = {"rows": [{"player_id": "x", "team": "AAA", "status": "ACT", "week": 1},
                       {"player_id": "y", "team": "BBB", "status": "ACT", "week": 1}],
              "snapshot_at": STAMP, "season": 2026, "week": 1}
    g = pdm.validate_roster_snapshot(roster, season=2026, week=2,
                                     slate_teams={"AAA", "BBB"}, now=NOW)
    assert g["publish"] is True and g["week"] == 1


def _cands(rows):
    base = {"market": "receiving_yards", "mean": 60.0, "sd": 20.0, "line": 52.5,
            "dist": "gamma", "p_over": 0.6, "p_under": 0.4, "game_id": "G"}
    return pd.DataFrame([{**base, **r} for r in rows])


ROSTER_ROWS = [
    {"player_id": "ACT_A", "name": "Act A", "team": "AAA", "status": "ACT", "week": 1},
    {"player_id": "TRADED", "name": "Tra Ded", "team": "CCC", "status": "ACT", "week": 1},
    {"player_id": "TRADED", "name": "Tra Ded", "team": "AAA", "status": "TRD", "week": 1},
    {"player_id": "RETIRED", "name": "Re Tired", "team": "AAA", "status": "RET", "week": 1},
    {"player_id": "CUT", "name": "Cut Guy", "team": "AAA", "status": "CUT", "week": 1},
    {"player_id": "IR", "name": "On Ir", "team": "AAA", "status": "RES", "week": 1},
    {"player_id": "PSQ", "name": "Practice Squad", "team": "AAA", "status": "DEV", "week": 1},
    {"player_id": "RETURNER", "name": "Back From Ir", "team": "AAA", "status": "ACT", "week": 1},
    {"player_id": "DUP", "name": "Dup Licate", "team": "AAA", "status": "ACT", "week": 1},
    {"player_id": "DUP", "name": "Dup Licate", "team": "BBB", "status": "ACT", "week": 1},
    {"player_id": "WEIRD", "name": "Odd Status", "team": "AAA", "status": "ZZZ", "week": 1},
]


def test_roster_eligibility_classes_are_separate_and_explicit():
    """Traded, retired, released, reserve, practice squad, ambiguous, unknown
    status and unmatched ids are each excluded WITH their own reason; an IR
    returner with no recent play is kept (membership, not recency)."""
    cands = _cands([
        {"player_id": "ACT_A", "name": "Act A", "team": "AAA"},
        {"player_id": "TRADED", "name": "Tra Ded", "team": "AAA"},   # carry-forward on old club
        {"player_id": "RETIRED", "name": "Re Tired", "team": "AAA"},
        {"player_id": "CUT", "name": "Cut Guy", "team": "AAA"},
        {"player_id": "IR", "name": "On Ir", "team": "AAA"},
        {"player_id": "PSQ", "name": "Practice Squad", "team": "AAA"},
        {"player_id": "RETURNER", "name": "Back From Ir", "team": "AAA", "roll_games": 0},
        {"player_id": "DUP", "name": "Dup Licate", "team": "AAA"},
        {"player_id": "WEIRD", "name": "Odd Status", "team": "AAA"},
        {"player_id": "GHOST", "name": "Not On Roster", "team": "AAA"},
    ])
    roster = {"rows": ROSTER_ROWS, "snapshot_at": STAMP, "season": 2026, "week": 1}
    kept, diag = pdm.apply_roster_eligibility(cands, roster)
    assert set(kept["player_id"]) == {"ACT_A", "RETURNER"}
    reasons = {e["player_id"]: e["eligibility"] for e in diag["excluded"]}
    assert reasons == {"TRADED": "team_changed", "RETIRED": "retired", "CUT": "released",
                       "IR": "reserve", "PSQ": "practice_squad", "DUP": "ambiguous",
                       "WEIRD": "unknown", "GHOST": "unknown"}
    assert diag["n_in"] == 10 and diag["n_kept"] == 2 and diag["n_excluded"] == 8
    traded = next(e for e in diag["excluded"] if e["player_id"] == "TRADED")
    assert traded["roster_team"] == "CCC"


def test_practice_squad_elevation_counts_only_when_event_roster_says_active():
    cands = _cands([{"player_id": "PSQ", "name": "Practice Squad", "team": "AAA"}])
    roster = {"rows": ROSTER_ROWS, "snapshot_at": STAMP, "season": 2026, "week": 1}
    kept_wed, _ = pdm.apply_roster_eligibility(cands, roster)
    assert kept_wed.empty
    from nflvalue.sources.availability import normalize_name
    kept_t90, _ = pdm.apply_roster_eligibility(
        cands, roster, t90_active_names={normalize_name("Practice Squad")})
    assert kept_t90["player_id"].tolist() == ["PSQ"]


def test_t90_applies_the_roster_gate_and_scopes_its_persist_to_the_game(env):  # noqa: F811
    """Two defects: (1) the T-90 path must run the same eligibility gate;
    (2) persist_leans for a T-90 game must not delete OTHER games' T-90 leans
    (pre-existing: only the last T-90 game of a week survived)."""
    now = _stamp_now()
    pw.run_week(SEASON, WEEK, mode="live", inputs=synthetic_inputs(),
                inject_feeds=_fresh_feeds(now))
    feeds = _fresh_feeds(now)
    feeds["inactive_rows"] = [{"name": "Bravo Quarterback", "active": True, "team": "BBB"}]
    feeds["inactives_fetched_at"] = now
    # WR_A is now on another club in the roster snapshot
    feeds["active_roster"] = _roster(now)
    for r in feeds["active_roster"]["rows"]:
        if r["player_id"] == "WR_A":
            r["team"] = "ZZZ"
    res = pw.run_t90(SEASON, WEEK, GAME_ID, mode="live", inputs=synthetic_inputs(),
                     inject_feeds=feeds)
    assert res["publish"] is True
    pids = {ln["player_id"] for g in res["games"] for ln in g["leans"]}
    assert "WR_A" not in pids
    assert res["roster_eligibility"]["by_reason"].get("team_changed") == 1

    conn = dbmod.connect()
    # a T-90 lean for ANOTHER game must survive this game's persist
    dbmod.upsert(conn, "leans", [{
        "season": SEASON, "week": WEEK, "clock": "t90", "game_id": "OTHER_GAME",
        "player_id": "X", "name": "Other", "market": "receptions", "side": "over",
        "line": 4.5, "line_source": "odds_api", "status": "active", "as_of": now,
        "created_at": now}], ["season", "week", "clock", "game_id", "player_id", "market"])
    rptmod.persist_leans(conn, SEASON, WEEK, "t90", res["games"], now, game_ids=[GAME_ID])
    left = dbmod.query_df(conn, "SELECT game_id FROM leans WHERE clock='t90'")
    assert "OTHER_GAME" in set(left["game_id"])
    conn.close()


# =========================================================================== #
# B. Probabilities, ranking, market approval
# =========================================================================== #
def _priced(**over):
    c = {"player_id": "P1", "name": "P One", "market": "receiving_yards", "mean": 60.0,
         "sd": 20.0, "line": 52.5, "dist": "gamma",
         "components": {"opp_factor": 1.0, "game_script": 1.0}, "low_confidence": False,
         "prices": {"over": 1.95, "under": 1.95, "n_books": 3, "consensus_p_over": 0.50,
                    "book": "a/b"}}
    c.update(over)
    if "p_over" not in over:
        p = round(dist_p_over(c["mean"], c["sd"], c["line"], c["dist"]), 4)
        c["p_over"], c["p_under"] = p, round(1 - p, 4)
    return c


def test_perturbing_the_ordinal_ranker_cannot_move_probability_ev_or_kelly():
    a = score_candidate(_priced(ml_score=10.0, ml_p_over=0.10))
    b = score_candidate(_priced(ml_score=99.0, ml_p_over=0.99))
    for k in ("model_prob", "market_prob", "ev_best_price", "kelly_fraction", "edge_raw"):
        assert a["components"][k] == b["components"][k]
    assert a["edge"] == b["edge"] and a["side"] == b["side"]


def test_ranker_score_is_for_the_published_side_not_its_own_side():
    """A candidate the classifier is 80% sure goes UNDER must not rank first
    as an OVER because 80 > 60."""
    fav = _priced(player_id="FAV", ml_p_over=0.60)     # ranker agrees with OVER
    contra = _priced(player_id="CON", ml_p_over=0.20)  # ranker: 80% UNDER
    g = rank_game([fav, contra])
    assert [ln["player_id"] for ln in g["leans"]] == ["FAV", "CON"]
    assert all(ln["side"] == "over" for ln in g["leans"])
    con = next(ln for ln in g["leans"] if ln["player_id"] == "CON")
    assert con["ml_score"] == pytest.approx(20.0)


def test_stored_probability_that_drifts_from_the_distribution_is_a_killcheck():
    s = score_candidate(_priced(p_over=0.95, p_under=0.05))
    assert s["market_state"] == "PROBABILITY_KILLCHECK"
    assert s["edge"] is None and s["components"]["ev_best_price"] is None
    assert s["components"]["kelly_fraction"] is None
    assert s["components"]["probability_coherent"] is False


def test_missing_distribution_picks_side_from_the_projection_and_never_acts():
    c = _priced(p_over=0.7, p_under=0.3)
    del c["dist"]
    s = score_candidate(c)
    assert s["side"] == "over"          # mean 60 > line 52.5
    assert s["edge"] is None and s["components"]["model_prob"] is None
    c["mean"] = 40.0
    assert score_candidate(c)["side"] == "under"


@pytest.mark.parametrize("bad", [{"over": 1.0, "under": 1.95}, {"over": 0.0, "under": 1.95},
                                 {"over": float("nan"), "under": 1.95},
                                 {"over": "abc", "under": 1.95}, {"over": None, "under": 1.95},
                                 {"over": 1.95, "under": float("inf")}])
def test_invalid_prices_are_not_actionable(bad):
    prices = {"n_books": 3, "consensus_p_over": 0.5, **bad}
    s = score_candidate(_priced(prices=prices))
    assert s["market_state"] == "INVALID_PRICE"
    assert s["edge"] is None and s["components"]["ev_best_price"] is None
    assert s["components"]["kelly_fraction"] is None


@pytest.mark.parametrize("n_books", [None, 0, 1, -3, float("nan"), "2", "many", True, 1.5])
def test_malformed_or_single_book_counts_never_emit_action_fields(n_books):
    s = score_candidate(_priced(prices={"over": 1.95, "under": 1.95, "n_books": n_books,
                                        "consensus_p_over": 0.5}))
    assert s["market_state"] == "ONE_BOOK_CONTEXT_ONLY"
    assert s["edge"] is None and s["components"]["ev_best_price"] is None
    assert s["components"]["kelly_fraction"] is None


def test_integer_line_on_a_count_market_prices_the_push():
    c = _priced(market="receptions", mean=5.4, sd=2.0, line=5.0, dist="negbinom")
    probs = pdm.side_probabilities(c)
    assert probs["p_push"] > 0
    assert probs["p_over"] + probs["p_under"] + probs["p_push"] == pytest.approx(1.0, abs=1e-9)
    s = score_candidate(c)
    assert s["components"]["p_push"] == pytest.approx(probs["p_push"], abs=1e-4)
    # model_prob is conditional on no push, so EV/Kelly are per unit at risk
    cond = probs["p_over"] / (1 - probs["p_push"])
    assert s["components"]["model_prob"] == pytest.approx(cond, abs=1e-4)
    half = pdm.side_probabilities({**c, "line": 5.5})
    assert half["p_push"] == 0.0


def test_yes_only_td_market_needs_two_books_and_a_valid_yes_price():
    from nflvalue.projection import p_over as po
    p = round(po(0.9, 0.95, 0.5, "poisson"), 4)
    td = _priced(market="anytime_td", mean=0.9, sd=0.95, line=0.5, dist="poisson",
                 p_over=p, p_under=round(1 - p, 4),
                 prices={"over": 2.3, "under": None, "n_books": 1})
    assert score_candidate(td)["market_state"] == "ONE_BOOK_CONTEXT_ONLY"
    td["prices"]["n_books"] = 2
    s = score_candidate(td)
    assert s["market_state"] == "REAL_MARKET" and s["side"] == "over"
    td["prices"]["over"] = 1.0
    assert score_candidate(td)["market_state"] == "INVALID_PRICE"


def test_persisted_decision_carries_the_gate_and_the_distribution_probability(env):  # noqa: F811
    now = _stamp_now()
    res = pw.run_week(SEASON, WEEK, mode="live", inputs=synthetic_inputs(),
                      inject_feeds=_fresh_feeds(now))
    conn = dbmod.connect()
    leans = dbmod.query_df(conn, "SELECT * FROM leans WHERE clock='wed'")
    assert "market_state" in leans.columns and "n_books" in leans.columns
    by_key = {(ln["player_id"], ln["market"]): ln for g in res["games"] for ln in g["leans"]}
    for row in leans.itertuples(index=False):
        lean = by_key[(row.player_id, row.market)]
        assert row.market_state == lean["market_state"]
        assert row.p_side == pytest.approx(lean["components"]["model_prob"], abs=1e-6)
        # the stored p_side is the distribution's side probability, never the
        # ranker's, and matches the published p_over/p_under
        pub = lean["p_over"] if lean["side"] == "over" else lean["p_under"]
        assert row.p_side == pytest.approx(pub, abs=1e-3)
    conn.close()


# =========================================================================== #
# E. Grade implementation
# =========================================================================== #
def _row(i, **kw):
    r = {"p_side": 0.6, "hit": int(i % 3 != 0), "line_source": "odds_api",
         "market_state": "REAL_MARKET", "season": 2026, "week": 1 + (i % 8),
         "game_id": f"g{i % 4}", "player_id": f"p{i}", "market": "receptions",
         "side": "over", "clock": "wed", "status": "active"}
    r.update(kw)
    return r


def test_grade_reports_denominators_and_rejects_bad_rows_explicitly():
    rows = [_row(i) for i in range(20)]
    rows += [_row(100, p_side=1.7), _row(101, hit=2), _row(102, p_side="x"),
             _row(103, status="voided"),
             _row(104, market_state="ONE_BOOK_CONTEXT_ONLY"),
             _row(105, market_state=None),
             _row(106, line_source="synthetic_trailing_mean")]
    # the T-90 re-rank of a Wednesday decision is the same decision
    rows.append({**_row(0), "clock": "t90"})
    rep = grade_rows(rows, bootstrap_n=50, seed=1)
    cov = rep["coverage"]
    assert cov["rows_received"] == len(rows)
    assert cov["graded_real_line"] == 20
    assert cov["invalid"] == 3 and cov["voided"] == 1
    assert cov["real_line_context_only"] == 1
    assert cov["real_line_ungated_pre_gate_rows"] == 1
    assert cov["synthetic_or_reference"] == 1
    assert cov["superseded_duplicate_clock"] == 1
    assert rep["evidence_kind"] == "real_line_insufficient"
    assert len(rep["invalid_rows"]) == 3


def test_grade_bootstrap_is_clustered_by_season_week():
    rows = [_row(i) for i in range(40)]                    # 8 weeks
    rep = grade_rows(rows, bootstrap_n=200, seed=3)
    assert rep["real_line"]["brier_95_interval"]["method"] == "cluster bootstrap by season-week"
    assert rep["real_line"]["brier_95_interval"]["n_clusters"] == 8
    thin = [_row(i, week=1) for i in range(40)]            # one cluster
    rep2 = grade_rows(thin, bootstrap_n=200, seed=3)
    assert rep2["uncertainty"]["brier_95_interval"] == [None, None]
    assert "unavailable" in rep2["real_line"]["brier_95_interval"]["method"]


def test_grade_never_promotes_synthetic_or_thin_records():
    rep = grade_rows([_row(i, line_source="synthetic_trailing_mean") for i in range(300)],
                     bootstrap_n=20, seed=1)
    assert rep["evidence_kind"] == "synthetic_research_only"
    rep = grade_rows([_row(i) for i in range(99)], bootstrap_n=20, seed=1)
    assert rep["evidence_kind"] == "real_line_insufficient"
    rep = grade_rows([_row(i) for i in range(100)], bootstrap_n=20, seed=1)
    assert rep["evidence_kind"] == "prospective_real_line"
    assert "no ROI" in rep["claim"] or "no profit" in rep["claim"].lower()
    assert grade_rows([], bootstrap_n=5)["evidence_kind"] == "no_evidence"


def test_clv_entries_exclude_context_only_rows(env):  # noqa: F811
    """A one-book lean is not a CLV entry; legacy NULL rows stay entries."""
    from nflvalue import clv as clvmod
    conn = dbmod.connect()
    now = _stamp_now()
    base = {"season": SEASON, "week": WEEK, "clock": "wed", "game_id": GAME_ID,
            "name": "n", "market": "receptions", "side": "over", "line": 4.5,
            "line_source": "odds_api", "status": "active", "as_of": now, "created_at": now}
    dbmod.upsert(conn, "leans", [
        {**base, "player_id": "REAL", "market_state": "REAL_MARKET"},
        {**base, "player_id": "ONEBOOK", "market_state": "ONE_BOOK_CONTEXT_ONLY"},
        {**base, "player_id": "LEGACY", "market_state": None},
    ], ["season", "week", "clock", "game_id", "player_id", "market"])
    seen = []
    import nflvalue.clv as _c
    orig = _c.snapshot_prob

    def spy(conn_, game_id, market, player_id, side, at_or_before_ts=None):
        seen.append(player_id)
        return None
    _c.snapshot_prob = spy
    try:
        clvmod.log_close_for_week(conn, SEASON, WEEK, {GAME_ID: "2026-09-13T17:00:00Z"})
    finally:
        _c.snapshot_prob = orig
    assert set(seen) == {"REAL", "LEGACY"}
    conn.close()


# =========================================================================== #
# C. UI labels consume the gate, not line_source alone
# =========================================================================== #
def test_ui_labels_name_the_gate_state_for_a_priced_but_unactionable_lean():
    from nflvalue.document import render_drop
    from nflvalue.notify import _game_embed
    from nflvalue.report import _fmt_edge, gate_label
    one_book = {"name": "X", "pos": "WR", "team": "A", "market": "receiving_yards",
                "side": "over", "line": 61.5, "line_source": "odds_api", "mean": 70.0,
                "edge": None, "no_market": True, "market_state": "ONE_BOOK_CONTEXT_ONLY",
                "composite": 40.0, "reason": "r", "player_id": "X"}
    assert _fmt_edge(one_book) == "`one_book_context_only`"
    assert gate_label(one_book) == "one_book_context_only"
    assert gate_label({**one_book, "market_state": "NO_MARKET"}) == "no_market"
    payload = {"season": 2026, "week": 1, "clock": "wed", "as_of": "t", "publish": True,
               "games": [{"game_id": "G", "matchup": "A @ B", "screened_n": 3,
                          "leans": [one_book]}], "contexts": {}}
    assert "one_book_context_only" in render_drop(payload)
    assert "one_book_context_only" in _game_embed(payload["games"][0], None)["fields"][0]["value"]
    import inspect
    from nflvalue import dashboard
    assert "l.market_state" in inspect.getsource(dashboard)   # the JS reads the gate
