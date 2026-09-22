"""Quote identity and the first metered request (2026-09-22 ATL@GB run).

* Bijan Robinson (00-0038542, 'Bi.Robinson') and Brian Robinson Jr.
  (00-0037746, 'Br.Robinson') are both ATL RBs. The first-initial fallback
  keyed both as 'b robinson', so neither quote joined -- and with only one of
  them in the pool it would have joined the OTHER player's quote.
* The credit ledger learned the provider's count only from a metered
  response, so a stale ledger could authorize the first paid request. The
  quota headers used below are the ones the free events listing returned
  that evening (used 336, remaining 164, last 0).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nflvalue import db as dbmod  # noqa: E402
from nflvalue.sources import oddsapi_props as oap  # noqa: E402

BI, BR, JR = "00-0038542", "00-0037746", "00-0035831"
ROSTER = [{"player_id": BI, "name": "Bijan Robinson", "team": "ATL"},
          {"player_id": BR, "name": "Brian Robinson", "team": "ATL"},
          {"player_id": JR, "name": "Josh Robinson", "team": "GB"}]


def _cands(*ids):
    names = {BI: ("Bi.Robinson", "ATL"), BR: ("Br.Robinson", "ATL"), JR: ("J.Robinson", "GB")}
    return pd.DataFrame([{"player_id": i, "name": names[i][0], "team": names[i][1]} for i in ids])


def _rows(*names, game="2026_03_ATL_GB"):
    return [{"player_name": n, "game_id": game} for n in names]


def _ids(rows):
    return {r["player_name"]: r["player_id"] for r in rows}


GT = {"2026_03_ATL_GB": {"ATL", "GB"}, "2026_03_X_Y": {"X", "Y"}}


@pytest.mark.parametrize("roster", [ROSTER, None])
def test_bijan_and_brian_join_their_own_ids(roster):
    got = _ids(oap.match_player_ids(_rows("Bijan Robinson", "Brian Robinson Jr."),
                                    _cands(BI, BR, JR), roster_rows=roster, game_teams=GT))
    assert got == {"Bijan Robinson": BI, "Brian Robinson Jr.": BR}


@pytest.mark.parametrize("roster", [ROSTER, None])
def test_absent_namesake_never_takes_the_quote(roster):
    # only Bijan survives the gates: Brian's quote must stay unmatched
    got = _ids(oap.match_player_ids(_rows("Bijan Robinson", "Brian Robinson Jr."),
                                    _cands(BI), roster_rows=roster, game_teams=GT))
    assert got == {"Bijan Robinson": BI, "Brian Robinson Jr.": None}
    got = _ids(oap.match_player_ids(_rows("Bijan Robinson", "Brian Robinson Jr."),
                                    _cands(BR), roster_rows=roster, game_teams=GT))
    assert got == {"Bijan Robinson": None, "Brian Robinson Jr.": BR}


def test_ambiguity_is_final_and_other_games_are_out_of_scope():
    two_b = pd.DataFrame([{"player_id": "p1", "name": "B.Robinson", "team": "ATL"},
                          {"player_id": "p2", "name": "B.Robinson", "team": "GB"}])
    assert _ids(oap.match_player_ids(_rows("Bijan Robinson"), two_b, game_teams=GT)) == {
        "Bijan Robinson": None}
    # the same quote on an unrelated game cannot join an ATL/GB candidate
    assert _ids(oap.match_player_ids(_rows("Bijan Robinson", game="2026_03_X_Y"),
                                     _cands(BI, BR), roster_rows=ROSTER, game_teams=GT)) == {
        "Bijan Robinson": None}


def test_matching_is_deterministic_under_row_order():
    a = _ids(oap.match_player_ids(_rows("Brian Robinson Jr.", "Bijan Robinson"),
                                  _cands(BR, BI, JR), roster_rows=ROSTER, game_teams=GT))
    b = _ids(oap.match_player_ids(_rows("Bijan Robinson", "Brian Robinson Jr."),
                                  _cands(JR, BI, BR), roster_rows=list(reversed(ROSTER)),
                                  game_teams=GT))
    assert a == b == {"Bijan Robinson": BI, "Brian Robinson Jr.": BR}


# --------------------------------------------------------------------------- #
@pytest.fixture()
def conn(tmp_path):
    c = dbmod.connect(str(tmp_path / "t.db"))
    yield c
    c.close()


def _cfg():
    return {"odds_api_key": "test", "regions": "us", "max_prop_games_per_run": 4,
            "odds_budget": {"monthly_credits": 500, "reserve": 50},
            "prop_markets_internal": ["receiving_yards", "receptions", "rushing_yards",
                                      "passing_yards", "anytime_td"]}


def _metered_spy():
    calls = []

    def fetch(url, params):
        calls.append(url)
        return {"bookmakers": []}
    return calls, fetch


def test_stale_ledger_cannot_authorize_the_first_request(conn):
    budget = oap.CreditBudget(conn, 500, 50, month="2026-09")
    budget.spend(165)                                  # stale local count
    calls, fetch = _metered_spy()
    provider = {"x-requests-used": "448", "x-requests-remaining": "52", "x-requests-last": "0"}
    res = oap.pull_week_props(_cfg(), {"g1": "ev1"}, conn=conn, fetch=fetch, budget=budget,
                              quota_fetch=lambda url, params: ([], provider))
    assert res["quota_preflight"]["ok"] is True
    assert calls == [] and res["skipped_budget"] == ["g1"]      # 448 + 5 > 450
    assert budget.used == 448


def test_provider_remaining_caps_the_configured_month(conn):
    budget = oap.CreditBudget(conn, 500, 50, month="2026-09")
    calls, fetch = _metered_spy()
    # the provider plan is smaller than config claims: 336 used + 64 left
    provider = {"x-requests-used": "336", "x-requests-remaining": "64", "x-requests-last": "0"}
    res = oap.pull_week_props(_cfg(), {"g1": "ev1"}, conn=conn, fetch=fetch, budget=budget,
                              quota_fetch=lambda url, params: ([], provider))
    assert budget.ceiling == 350 and calls == ["%s/sports/%s/events/ev1/odds" % (oap.BASE, oap.SPORT)]
    assert res["pulled"] == ["g1"]


@pytest.mark.parametrize("headers", [
    {}, {"x-requests-used": "336"}, {"x-requests-used": "x", "x-requests-remaining": "164"},
    {"x-requests-used": "336", "x-requests-remaining": "164", "x-requests-last": "5"}])
def test_missing_or_ambiguous_quota_spends_nothing(conn, headers):
    calls, fetch = _metered_spy()
    res = oap.pull_week_props(_cfg(), {"g1": "ev1"}, conn=conn, fetch=fetch,
                              quota_fetch=lambda url, params: ([], headers))
    assert calls == [] and res["pulled"] == [] and res["quota_preflight"]["ok"] is False


def test_preflight_failure_is_a_refusal(conn):
    def boom(url, params):
        raise OSError("network down")
    calls, fetch = _metered_spy()
    res = oap.resnap_lines(_cfg(), {"g1": "ev1"}, conn=conn, fetch=fetch, quota_fetch=boom)
    assert calls == [] and res["pulled"] == [] and res["quota_preflight"]["ok"] is False
