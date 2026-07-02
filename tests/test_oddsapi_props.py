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
from nflvalue.sources import oddsapi_props as oap  # noqa: E402

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
    ceiling (500-50=450) and skipped games must be reported, not fetched."""
    cfg = _cfg(max_prop_games_per_run=100)          # cap wide open: budget is the only brake
    calls = {"n": 0}

    def fake_fetch(url, params=None):
        calls["n"] += 1
        return json.loads(json.dumps(payload))

    cost_per_event = 5.0                             # 5 markets x 1 region
    total_pulled = 0
    for week in range(1, 30):                        # way more weeks than a month holds
        event_map = {f"2023_{week:02d}_G{i}": f"evt{week}_{i}" for i in range(16)}
        res = oap.pull_week_props(cfg, event_map, conn=conn, fetch=fake_fetch,
                                  ts=f"2023-11-{week:02d}T12:00:00Z")
        total_pulled += len(res["pulled"])
        if res["skipped_budget"]:
            break

    budget = oap.CreditBudget(conn, 500, 50)         # fresh instance reads the ledger
    assert budget.used <= 450.0
    assert budget.used == pytest.approx(total_pulled * cost_per_event)
    assert calls["n"] == total_pulled                # skipped games were never fetched
    assert total_pulled == int(450 // cost_per_event)

    # and even a FORCED overspend raises rather than spends
    with pytest.raises(oap.BudgetExceeded):
        budget.spend(cost_per_event)


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

    n1 = dbmod.upsert(conn, "lines", rows, ["ts", "game_id", "book", "market", "player_name", "side"])
    dbmod.upsert(conn, "lines", rows, ["ts", "game_id", "book", "market", "player_name", "side"])
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
