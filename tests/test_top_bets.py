import json
from nflvalue import top_bets


def _game(edge_ats=3.0, edge_tot=1.0, p_home=0.72, settled=True, ats="W", tot="L", su=True):
    return {"home": "AAA", "away": "BBB", "settled": settled,
            "ats_pick": {"side": "home", "team": "AAA", "line": -3.0, "edge": edge_ats},
            "total_pick": {"side": "over", "line": 44.0, "edge": edge_tot},
            "su_pick": "AAA", "p_home_win": p_home,
            "ats_result": ats, "total_result": tot, "su_correct": su}


def _weekly(n_weeks=3, wins=True):
    weeks = []
    for w in range(1, n_weeks + 1):
        games = [_game(su=wins, ats="W" if wins else "L", tot="W" if wins else "L")
                 for _ in range(30)]
        weeks.append({"week": w, "label": f"Week {w}", "games": games})
    return {"weeks": weeks}


def test_best_tier_requires_67_band():
    out = top_bets.build_top_bets(_weekly(wins=False))  # all losses -> 0% bands
    for wk in out["weeks"]:
        for g in wk["games"]:
            assert all(b["tier"] != "best" for b in g["bets"])


def test_fail_closed_fewer_than_five():
    out = top_bets.build_top_bets(_weekly(wins=False))
    # 0%-accuracy bands: nothing qualifies at all -> games emit no tiers, never padded
    assert all(not g["bets"] for wk in out["weeks"] for g in wk["games"])


def test_tiers_and_rank_order_when_bands_qualify():
    out = top_bets.build_top_bets(_weekly(wins=True))  # all wins -> 100% bands
    g = out["weeks"][0]["games"][0]
    assert g["bets"], "qualifying bands must emit bets"
    ranks = [b["rank"] for b in g["bets"]]
    assert ranks == sorted(ranks) and ranks[0] == 1
    assert len(g["bets"]) <= top_bets.ALL_MAX_RANK
    assert sum(b["tier"] == "best" for b in g["bets"]) <= top_bets.BEST_MAX_RANK
    for b in g["bets"]:
        if b["tier"] == "best":
            assert b["band_accuracy"] >= top_bets.BEST_ACC
        else:
            assert b["band_accuracy"] > top_bets.VALUE_ACC and b["edge"] > 0


def test_min_band_n_guard():
    weekly = _weekly(n_weeks=1, wins=True)
    weekly["weeks"][0]["games"] = weekly["weeks"][0]["games"][:3]  # n<20 per band
    out = top_bets.build_top_bets(weekly)
    assert all(not g["bets"] for wk in out["weeks"] for g in wk["games"])


def test_deterministic():
    a = json.dumps(top_bets.build_top_bets(_weekly()))
    b = json.dumps(top_bets.build_top_bets(_weekly()))
    assert a == b
