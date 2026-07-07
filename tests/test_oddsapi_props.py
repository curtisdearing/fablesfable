"""Block A guardrails: the credit budget can NEVER be exceeded (simulated
month), pulls rotate + degrade to no_market, snapshots are idempotent,
book names match conservatively."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nflvalue import db as dbmod  # noqa: E402
from nflvalue.sources import oddsapi as oapi  # noqa: E402
from nflvalue.sources import oddsapi_props as oap  # noqa: E402
from nflvalue.sources import _http  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture()
def conn(tmp_path):
    c = dbmod.connect(str(tmp_path / "t.db"))
    yield c
    c.close()


@pytest.fixture(scope="module")
def payload():
    return json.loads((FIXTURES / "oddsapi_event_props_synthetic.json").read_text())["payload"]


def _cfg(**kw):
    cfg = {"odds_api_key": "test", "regions": "us", "max_prop_games_per_run": 4,
           "odds_budget": {"monthly_credits": 500, "reserve": 50},
           "prop_markets_internal": ["receiving_yards", "receptions", "rushing_yards",
                                     "passing_yards", "anytime_td"]}
    cfg.update(kw)
    return cfg


# --------------------------------------------------------------------------- #
# The budget hard stop
# --------------------------------------------------------------------------- #
def test_budget_never_exceeded_over_a_simulated_month(conn, payload):
    """Hammer pull_week_props far past the budget; the ledger must stop at the
    ceiling (500-50=450) no matter how wide the per-run game cap is opened.

    Phase 7.3 GAP #2: entries now ALSO self-limit via the reservation rule
    (entries may spend at most half the remaining budget, so closes are never
    crowded out) -- so total entry pulls stop well short of 450/cost_per_event
    even with ``max_prop_games_per_run`` wide open. That's the point: budget
    is no longer the ONLY brake. The ceiling itself must still never be
    crossed, which this test keeps proving."""
    cfg = _cfg(max_prop_games_per_run=100)           # cap wide open: budget/reservation are the brakes
    calls = {"n": 0}

    def fake_fetch(url, params=None):
        calls["n"] += 1
        return json.loads(json.dumps(payload))

    cost_per_event = 5.0                             # 5 markets x 1 region
    total_pulled = 0
    for week in range(1, 60):                        # way more weeks than a month holds
        event_map = {f"2023_{week:02d}_G{i}": f"evt{week}_{i}" for i in range(16)}
        res = oap.pull_week_props(cfg, event_map, conn=conn, fetch=fake_fetch,
                                  ts=f"2023-11-{week:02d}T12:00:00Z")
        total_pulled += len(res["pulled"])
        if res["skipped_budget"] or (not res["pulled"] and res["skipped_cap"]):
            break

    budget = oap.CreditBudget(conn, 500, 50)         # fresh instance reads the ledger
    assert budget.used <= 450.0                      # the ceiling is NEVER crossed
    assert budget.used == pytest.approx(total_pulled * cost_per_event)
    assert calls["n"] == total_pulled                # skipped games were never fetched
    # the reservation rule holds entries back HARDER than the raw ceiling --
    # total spend stays comfortably below what the ceiling alone would allow
    assert total_pulled < int(450 // cost_per_event)

    # and even a FORCED overspend past whatever remains raises rather than
    # spends (the reservation rule can stall entries with slack still on the
    # books -- below the ceiling but above a single event's cost -- so force
    # an amount that's definitely past the true ceiling, not just one event)
    with pytest.raises(oap.BudgetExceeded):
        budget.spend(budget.remaining + cost_per_event)


# --------------------------------------------------------------------------- #
# Phase 7.3 GAP #2: coupled budget reservation -- entries may never spend
# more than half the remaining monthly budget, so a close pull later in the
# week can always be afforded.
# --------------------------------------------------------------------------- #
def test_entry_pulls_reserve_half_the_budget_for_closes(conn, payload):
    cfg = _cfg(max_prop_games_per_run=1000)          # config cap wide open
    event_map = {f"2023_10_G{i}": f"evt{i}" for i in range(200)}   # far more games than affordable

    def fake_fetch(url, params=None):
        return json.loads(json.dumps(payload))

    res = oap.pull_week_props(cfg, event_map, conn=conn, fetch=fake_fetch,
                              ts="2023-11-08T12:00:00Z")
    budget = oap.CreditBudget(conn, 500, 50)
    cost_per_event = 5.0
    # the cap is computed against the FULL 450 ceiling (nothing spent yet,
    # since this is the first pull on a fresh conn/month)
    pre = oap.CreditBudget(conn, 500, 50, month=budget.month)
    pre.used = 0.0
    expected_cap = oap.entry_event_cap(pre, cost_per_event)
    assert res["entry_cap_reserved"] == expected_cap
    assert len(res["pulled"]) == expected_cap
    # at least half the ceiling remains untouched -- available for closes
    assert budget.remaining >= 450.0 / 2 - cost_per_event


def test_entry_event_cap_formula():
    class _FakeBudget:
        remaining = 450.0
    # floor((450/4.3) / (2*5)) = floor(104.651.../10) = 10
    assert oap.entry_event_cap(_FakeBudget(), 5.0) == 10
    _FakeBudget.remaining = 0.0
    assert oap.entry_event_cap(_FakeBudget(), 5.0) == 0


def test_budget_ledger_persists_across_instances(conn):
    b1 = oap.CreditBudget(conn, 500, 50)
    b1.spend(20.0)
    b2 = oap.CreditBudget(conn, 500, 50)
    assert b2.used == 20.0
    assert b2.remaining == 430.0


def test_budget_trusts_api_reported_usage(conn):
    b = oap.CreditBudget(conn, 500, 50)
    b.spend(5.0, headers={"x-requests-used": "37"})
    assert b.used == 37.0                            # API accounting wins over estimate


# --------------------------------------------------------------------------- #
# Rotation + cap degrade to no_market (never an error)
# --------------------------------------------------------------------------- #
def test_rotation_prefers_least_recently_pulled(conn, payload):
    cfg = _cfg(max_prop_games_per_run=1)
    fetch = lambda url, params=None: json.loads(json.dumps(payload))  # noqa: E731
    event_map = {"2023_10_A_B": "e1", "2023_10_C_D": "e2"}

    r1 = oap.pull_week_props(cfg, event_map, conn=conn, fetch=fetch, ts="2023-11-08T10:00:00Z")
    assert r1["pulled"] == ["2023_10_A_B"]           # alphabetical on first contact
    assert r1["skipped_cap"] == ["2023_10_C_D"]
    r2 = oap.pull_week_props(cfg, event_map, conn=conn, fetch=fetch, ts="2023-11-08T11:00:00Z")
    assert r2["pulled"] == ["2023_10_C_D"]           # never-pulled game jumps the queue


# --------------------------------------------------------------------------- #
# Parse + match + frame
# --------------------------------------------------------------------------- #
def test_parse_and_idempotent_upsert(conn, payload):
    rows = oap.parse_event_props(payload, ts="2023-11-08T10:00:00Z")
    for r in rows:
        r["game_id"] = "2023_10_CLE_BAL"
    assert {r["market"] for r in rows} == {"receiving_yards", "receptions",
                                           "anytime_td", "passing_yards"}
    td = [r for r in rows if r["market"] == "anytime_td"][0]
    assert td["side"] == "over" and td["point"] == 0.5   # Yes -> over @ 0.5

    n1 = dbmod.upsert(conn, "lines", rows, ["ts", "game_id", "book", "market", "player_name", "side", "point"])
    dbmod.upsert(conn, "lines", rows, ["ts", "game_id", "book", "market", "player_name", "side", "point"])
    count = dbmod.query_df(conn, "SELECT COUNT(*) AS n FROM lines").iloc[0]["n"]
    assert n1 == count                               # idempotent snapshot


def test_match_player_ids_conservative(payload):
    rows = oap.parse_event_props(payload, ts="t")
    candidates = pd.DataFrame([
        {"player_id": "00-A1", "name": "M.Andrews"},     # abbreviated, as player_week has it
        {"player_id": "00-C1", "name": "A.Cooper"},
        {"player_id": "00-L1", "name": "L.Jackson"},
    ])
    rows = oap.match_player_ids(rows, candidates)
    by_name = {}
    for r in rows:
        by_name.setdefault(r["player_name"], set()).add(r["player_id"])
    assert by_name["Mark Andrews"] == {"00-A1"}
    assert by_name["Amari Cooper"] == {"00-C1"}
    assert by_name["Unknown Practice Squad Guy"] == {None}   # kept, never guessed


def test_to_prop_lines_frame_cross_book_consensus(payload):
    rows = oap.parse_event_props(payload, ts="t")
    for r in rows:
        r["game_id"] = "G"
    candidates = pd.DataFrame([{"player_id": "00-A1", "name": "M.Andrews"},
                               {"player_id": "00-L1", "name": "L.Jackson"}])
    rows = oap.match_player_ids(rows, candidates)
    frame = oap.to_prop_lines_frame(rows)
    rec = frame[(frame["market"] == "receiving_yards") & (frame["player_id"] == "00-A1")].iloc[0]
    # DK quotes 52.5, FD quotes 53.5 -- tie on book count resolves
    # deterministically to the alphabetically-first book's point
    assert rec["point"] == 52.5 and rec["n_books"] == 1
    assert 0.4 < rec["consensus_p_over"] < 0.6
    td = frame[frame["market"] == "anytime_td"].iloc[0]
    assert pd.isna(td["under_price"]) and td["book"] == "draftkings"  # yes-only TD ok


def test_consensus_and_best_price_across_books():
    """Three books at the same point: consensus is de-vigged + sharp-weighted;
    best price per side is line-shopped with the book named; a soft book's
    fat vig cannot pull fair value."""
    rows = []
    for book, over, under in (("pinnacle", 1.90, 1.94),
                              ("draftkings", 1.85, 1.95),
                              ("softie", 2.05, 1.70)):   # off-market over price
        for side, price in (("over", over), ("under", under)):
            rows.append({"ts": "t", "game_id": "G", "book": book,
                         "market": "receiving_yards", "player_id": "00-A1",
                         "player_name": "Mark Andrews", "side": side,
                         "point": 52.5, "price": price})
    frame = oap.to_prop_lines_frame(rows)
    r = frame.iloc[0]
    assert r["n_books"] == 3
    assert r["over_price"] == 2.05 and "softie" in r["book"]      # line-shopped
    from nflvalue.oddsmath import devig_multiplicative
    pinn = devig_multiplicative([1.90, 1.94])[0]
    assert abs(r["consensus_p_over"] - pinn) < 0.02               # sharp-anchored

    # the composite consumes consensus for edge, best price for EV
    from nflvalue.composite import score_candidate
    cand = {"player_id": "00-A1", "market": "receiving_yards", "mean": 60.0,
            "sd": 20.0, "line": 52.5, "p_over": 0.60, "p_under": 0.40,
            "components": {"opp_factor": 1.0, "game_script": 1.0},
            "low_confidence": False,
            "prices": {"over": r["over_price"], "under": r["under_price"],
                       "book": r["book"], "consensus_p_over": r["consensus_p_over"],
                       "n_books": r["n_books"]}}
    s = score_candidate(cand)
    assert s["edge"] == pytest.approx(0.60 - r["consensus_p_over"], abs=1e-4)
    assert s["components"]["ev_best_price"] == pytest.approx(0.60 * 2.05 - 1, abs=1e-3)
    assert s["components"]["n_books"] == 3


def test_matchup_includes_epa_dimension():
    from nflvalue.composite import score_candidate
    base = {"player_id": "P", "market": "receiving_yards", "mean": 70.0, "sd": 25.0,
            "line": 65.5, "p_over": 0.58, "p_under": 0.42,
            "components": {"opp_factor": 1.0, "game_script": 1.0},
            "low_confidence": False, "prices": None}
    soft = score_candidate({**base, "opp_epa_factor": 1.12})   # bleeds EPA -> over-friendly
    hard = score_candidate({**base, "opp_epa_factor": 0.88})
    none = score_candidate(base)
    assert soft["matchup"] > none["matchup"] > hard["matchup"]


def test_american_to_decimal_guards_zero():
    """American odds of 0 are not a valid price; degrade to 1.0 rather than
    raising ZeroDivisionError. Valid prices are unchanged."""
    from nflvalue import oddsmath
    assert oddsmath.american_to_decimal(0) == 1.0
    assert oddsmath.american_to_decimal(150) == pytest.approx(2.5)
    assert oddsmath.american_to_decimal(-200) == pytest.approx(1.5)


def test_consensus_n_books_counts_only_contributing_books():
    """A book whose prices fail the da<=1.0/db<=1.0 filter is skipped and must
    NOT be counted in n_books. Only books that contribute are counted."""
    from nflvalue import oddsmath
    r = oddsmath.consensus_two_way({
        "pinnacle": (1.90, 1.94),
        "draftkings": (1.85, 1.95),
        "broken": (1.0, 0.0),          # invalid price -> skipped by the filter
    })
    assert r["n_books"] == 2            # not 3


# --------------------------------------------------------------------------- #
# Network-IO hardening: bad feeds degrade LOUDLY, never crash the pipeline
# --------------------------------------------------------------------------- #
def _cfg_game():
    return {"odds_api_key": "test", "regions": "us"}


def test_fetch_game_odds_degrades_on_feed_failure(monkeypatch, capsys):
    """A transient feed blip must degrade to an empty slate, not raise."""
    def boom(url, params=None, timeout=15.0):
        raise _http.HttpJsonError("simulated transient blip")

    monkeypatch.setattr(oapi, "get_json", boom)
    out = oapi.fetch_game_odds(_cfg_game())
    assert out == []
    assert "game odds fetch failed" in capsys.readouterr().out   # degraded LOUDLY


def test_fetch_game_odds_missing_price_skips_book(monkeypatch):
    """A book omitting `price` on an outcome must be skipped mid-parse, not
    KeyError. Valid books in the same payload still parse exactly."""
    payload = [{
        "id": "evt1", "commence_time": "2023-11-08T18:00:00Z",
        "home_team": "BAL", "away_team": "CLE",
        "bookmakers": [
            {"key": "brokenbook", "markets": [
                {"key": "h2h", "outcomes": [
                    {"name": "BAL"},                       # NO price -> must skip
                    {"name": "CLE", "price": 2.5}]},
                {"key": "spreads", "outcomes": [
                    {"name": "BAL", "point": -3.5},        # NO price
                    {"name": "CLE", "point": 3.5, "price": 1.9}]},
            ]},
            {"key": "goodbook", "markets": [
                {"key": "h2h", "outcomes": [
                    {"name": "BAL", "price": 1.5},
                    {"name": "CLE", "price": 2.6}]},
                {"key": "totals", "outcomes": [
                    {"name": "Over", "point": 44.5, "price": 1.91},
                    {"name": "Under", "point": 44.5, "price": 1.91}]},
            ]},
        ],
    }]
    monkeypatch.setattr(oapi, "get_json",
                        lambda url, params=None, timeout=15.0: payload)
    out = oapi.fetch_game_odds(_cfg_game())          # must NOT raise KeyError
    assert len(out) == 1
    books = out[0]["books"]
    # broken book missing a price on both sides drops out of h2h/spreads
    assert "brokenbook" not in books["h2h"]
    assert "brokenbook" not in books["spreads"]
    # the good book parses exactly as before
    assert books["h2h"]["goodbook"] == {"home": 1.5, "away": 2.6}
    assert books["totals"]["goodbook"]["over"]["price"] == 1.91


def test_fetch_game_odds_bad_shape_degrades(monkeypatch):
    """A non-list body (e.g. an API error object) degrades to []."""
    monkeypatch.setattr(oapi, "get_json",
                        lambda url, params=None, timeout=15.0: {"message": "bad key"})
    assert oapi.fetch_game_odds(_cfg_game()) == []


def test_fetch_scores_degrades_and_skips_malformed(monkeypatch):
    """Feed failure -> {}; and a malformed score entry is skipped, not indexed."""
    monkeypatch.setattr(oapi, "get_json",
                        lambda url, params=None, timeout=15.0: (_ for _ in ()).throw(
                            _http.HttpJsonError("blip")))
    assert oapi.fetch_scores(_cfg_game()) == {}

    payload = [
        {"id": "g1", "completed": True, "home_team": "BAL", "away_team": "CLE",
         "scores": [{"name": "BAL", "score": "20"}, {"name": "CLE"}]},   # missing score
        {"id": "g2", "completed": True, "home_team": "KC", "away_team": "DEN",
         "scores": [{"name": "KC", "score": "27"}, {"name": "DEN", "score": "13"}]},
    ]
    monkeypatch.setattr(oapi, "get_json",
                        lambda url, params=None, timeout=15.0: payload)
    res = oapi.fetch_scores(_cfg_game())             # must NOT KeyError on g1
    assert "g1" not in res                            # incomplete scores skipped
    assert res["g2"]["home_score"] == 27.0 and res["g2"]["away_score"] == 13.0


# --------------------------------------------------------------------------- #
# Surgical spend (opt-in): trim markets to the convicted ones, cover more games
# --------------------------------------------------------------------------- #
def test_surgical_markets_keeps_only_convicted_pairs():
    cands = pd.DataFrame([
        {"game_id": "G_A", "market": "receiving_yards", "p_over": 0.70},  # convicted -> keep
        {"game_id": "G_A", "market": "receptions", "p_over": 0.52},       # coinflip  -> drop
        {"game_id": "G_A", "market": "rushing_yards", "p_over": None},    # no prob   -> drop
        {"game_id": "G_B", "market": "passing_yards", "p_over": 0.50},    # coinflip  -> game drops
    ])
    cfg = _cfg(surgical_spend={"enabled": True, "min_conviction": 0.06})
    assert oap.surgical_markets(cands, cfg) == {"G_A": ["receiving_yards"]}


def test_surgical_pull_reserves_closes_and_never_exceeds_budget(conn, payload):
    cfg = _cfg(surgical_spend={"enabled": True, "min_conviction": 0.06})   # 5-market full basis
    cands = pd.DataFrame([{"game_id": f"2023_01_G{i}", "market": "receiving_yards",
                           "p_over": 0.75} for i in range(120)])
    event_map = {f"2023_01_G{i}": f"evt{i}" for i in range(120)}

    def fake_fetch(url, params=None):
        assert params["markets"] == "player_reception_yds"   # only the convicted market
        return json.loads(json.dumps(payload))

    res = oap.pull_week_props(cfg, event_map, conn=conn, fetch=fake_fetch,
                              candidates=cands, ts="2023-11-08T10:00:00Z")
    assert res["surgical"] is True
    budget = oap.CreditBudget(conn, 500, 50)
    assert budget.used <= 450.0                              # ceiling never crossed
    # entries + reserved full-cost closes fit inside the starting budget
    assert res["credits_spent"] + res["close_budget_reserved"] <= 450.0 + 1e-9
    # trimming 5 markets -> 1 covers far MORE games than a full pull's entry half
    # (450/2 / 5 = 45 games) could ever reach
    assert len(res["pulled"]) > 45


def test_surgical_off_by_default_even_with_candidates(conn, payload):
    cfg = _cfg()                                             # no surgical_spend key
    cands = pd.DataFrame([{"game_id": "2023_01_G0", "market": "receiving_yards", "p_over": 0.9}])
    res = oap.pull_week_props(cfg, {"2023_01_G0": "evt0"}, conn=conn,
                              fetch=lambda u, p=None: json.loads(json.dumps(payload)),
                              candidates=cands, ts="2023-11-08T10:00:00Z")
    assert "surgical" not in res                             # full rotating path taken
