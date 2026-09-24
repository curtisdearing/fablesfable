"""Credit accounting must report what the PROVIDER billed, and which games
actually came back with quotes -- not the planned per-event upper bound.

2026-09-23 (run 35840645236): quota preflight 341 used / 159 remaining; the
run log said "pulled 14 game(s), 70 credits spent", but the provider's
x-requests-used went 341 -> 375 (34 credits) and only 10 of the 14 games had
a single quote row. The Odds API bills an event call per market RETURNED
(6 games x 5 markets + 4 anytime-TD-only games x 1 = 34), and an event with
no bookmakers yet costs nothing. So the log over-stated spending by 36 credits
and counted four empty responses (ARI_SF, MIN_TB, SEA_WAS, TEN_NYG) as priced.

Payloads here are SYNTHETIC unit-test fixtures shaped like v4 responses; the
provider header arithmetic mirrors the production numbers above.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nflvalue import db as dbmod  # noqa: E402
from nflvalue.sources import oddsapi_props as oap  # noqa: E402

MARKETS = ["player_reception_yds", "player_receptions", "player_rush_yds",
           "player_pass_yds", "player_anytime_td"]


def _cfg():
    return {"odds_api_key": "test", "regions": "us", "max_prop_games_per_run": 16,
            "odds_budget": {"monthly_credits": 500, "reserve": 50},
            "prop_markets_internal": ["receiving_yards", "receptions", "rushing_yards",
                                      "passing_yards", "anytime_td"]}


def _payload(markets):
    return {"bookmakers": [{"key": "draftkings", "markets": [
        {"key": m, "outcomes": [{"name": "Over", "description": "Some Player",
                                 "price": 1.9, "point": 0.5 if m.endswith("td") else 42.5},
                                {"name": "Under", "description": "Some Player",
                                 "price": 1.9, "point": 0.5 if m.endswith("td") else 42.5}]}
        for m in markets]}]} if markets else {"bookmakers": []}


class _Provider:
    """Bills an event call per market returned, like the real provider."""

    def __init__(self, used, remaining, by_event):
        self.used, self.total = used, used + remaining
        self.by_event = by_event
        self.calls = []

    def quota(self, url, params):
        return [], {"x-requests-used": str(self.used),
                    "x-requests-remaining": str(self.total - self.used),
                    "x-requests-last": "0"}

    def fetch(self, url, params):
        event = url.rsplit("/", 2)[-2]
        self.calls.append(event)
        markets = self.by_event[event]
        self.used += len(markets)
        body = _payload(markets)
        body["_headers"] = {"x-requests-used": str(self.used),
                            "x-requests-remaining": str(self.total - self.used),
                            "x-requests-last": str(len(markets))}
        return body


@pytest.fixture()
def conn(tmp_path):
    c = dbmod.connect(str(tmp_path / "t.db"))
    yield c
    c.close()


def test_pull_reports_provider_billed_credits_and_empty_games(conn, capsys):
    prov = _Provider(341, 159, {"e_full": MARKETS, "e_td": ["player_anytime_td"],
                                "e_empty": []})
    budget = oap.CreditBudget(conn, 500, 50, month="2026-09")
    res = oap.pull_week_props(_cfg(), {"full": "e_full", "td": "e_td", "empty": "e_empty"},
                              conn=conn, fetch=prov.fetch, quota_fetch=prov.quota,
                              budget=budget, reserve_close=True)
    out = capsys.readouterr().out
    assert sorted(prov.calls) == ["e_empty", "e_full", "e_td"]
    assert res["credits_spent"] == 6.0          # provider: 341 -> 347
    assert res["credits_billed_measured"] == 6.0 and res["credits_estimated"] == 0.0
    assert res["account_usage_delta"] == 6.0
    assert res["credits_planned"] == 15.0       # 3 calls x 5, the gate's upper bound
    assert budget.used == 347.0
    assert sorted(res["priced"]) == ["full", "td"]
    assert res["empty"] == ["empty"]
    assert sorted(res["pulled"]) == ["empty", "full", "td"]   # every call answered
    # the log states the billed figure and the empty games, never "15 credits spent"
    assert "15 credits spent" not in out
    assert "provider billed 6" in out and "no quotes: empty" in out
    # the close holds still fit under the hard ceiling after billing
    assert budget.used + res["close_reserved"] <= budget.ceiling


def test_pull_without_provider_headers_falls_back_to_planned_cost(conn):
    budget = oap.CreditBudget(conn, 500, 50, month="2026-09")

    def fetch(url, params):
        return _payload(MARKETS)

    res = oap.pull_week_props(_cfg(), {"g": "e"}, conn=conn, fetch=fetch, budget=budget)
    assert res["credits_spent"] == res["credits_planned"] == 5.0
    assert res["priced"] == ["g"] and res["empty"] == []
    # an estimate is never labelled measured provider billing
    assert res["credits_billed_measured"] == 0.0 and res["credits_estimated"] == 5.0
    assert res["account_usage_delta"] is None
    text = oap.billing_text(res)
    assert "provider billed" not in text and "ESTIMATED" in text


def test_resnap_reports_billed_credits_and_empty_games(conn, capsys):
    prov = _Provider(375, 125, {"e_full": MARKETS, "e_empty": []})
    res = oap.resnap_lines(_cfg(), {"full": "e_full", "empty": "e_empty"}, conn=conn,
                           fetch=prov.fetch, quota_fetch=prov.quota)
    assert res["credits_spent"] == 5.0 and res["credits_planned"] == 10.0
    assert res["priced"] == ["full"] and res["empty"] == ["empty"]
    assert res["budget_remaining"] == 450 - 380


class _SharedKeyProvider(_Provider):
    """Another consumer of the same key spends ``other`` credits between calls."""

    def __init__(self, used, remaining, by_event, other):
        super().__init__(used, remaining, by_event)
        self.other = other

    def fetch(self, url, params):
        self.used += self.other
        return super().fetch(url, params)


def test_shared_key_usage_is_reported_as_account_delta_not_our_billing(conn, capsys):
    prov = _SharedKeyProvider(341, 159, {"e_full": MARKETS, "e_td": ["player_anytime_td"]},
                              other=3)
    budget = oap.CreditBudget(conn, 500, 50, month="2026-09")
    res = oap.pull_week_props(_cfg(), {"full": "e_full", "td": "e_td"}, conn=conn,
                              fetch=prov.fetch, quota_fetch=prov.quota, budget=budget)
    assert res["credits_billed_measured"] == 6.0     # our two requests: 5 + 1
    assert res["account_usage_delta"] == 12.0        # + 2 x 3 by the other consumer
    assert budget.used == 353.0                      # the gate adopts the account figure
    out = capsys.readouterr().out
    assert "provider billed 6" in out and "account usage +12" in out


def test_lower_or_reset_header_never_loosens_the_gate(conn):
    budget = oap.CreditBudget(conn, 500, 50, month="2026-09")
    budget.reconcile(375, 125)
    # stale / reset header: provider says 10 used, this call billed 5
    budget.spend(5, headers={"x-requests-used": "10", "x-requests-remaining": "490",
                             "x-requests-last": "5"})
    assert budget.used == 380.0
    # lower header without x-requests-last: counted at the per-event bound
    budget.spend(5, headers={"x-requests-used": "12"})
    assert budget.used == 385.0
    # unparseable header: counted at the bound
    budget.spend(5, headers={"x-requests-used": "nan", "x-requests-last": "1"})
    assert budget.used == 390.0
    assert budget.ceiling == 450.0                   # ceiling and reserve unchanged


def test_reset_header_mid_pull_reports_negative_account_delta(conn, capsys):
    prov = _Provider(375, 125, {"e1": MARKETS})
    budget = oap.CreditBudget(conn, 500, 50, month="2026-09")

    def fetch(url, params):
        body = prov.fetch(url, params)
        body["_headers"] = {"x-requests-used": "5", "x-requests-remaining": "495",
                            "x-requests-last": "5"}       # provider-side reset
        return body

    res = oap.pull_week_props(_cfg(), {"g": "e1"}, conn=conn, fetch=fetch,
                              quota_fetch=prov.quota, budget=budget)
    assert res["credits_billed_measured"] == 5.0
    assert res["account_usage_delta"] == -370.0
    assert budget.used == 380.0                      # conservative: 375 + this call's 5
