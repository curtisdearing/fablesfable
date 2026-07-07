"""Tests for the observation-quality tagging primitive (nflvalue/game_context).

Runs on synthetic in-memory frames only -- no parquet/scipy/sklearn -- so it is
CI-cheap and self-contained. Covers: the injury-shortened arms (early-exit + a
leak-safe snap collapse), records-to-date leak-safety, and the meaningless-game
proxy checked against its own documented cut formula.
"""
import numpy as np
import pandas as pd

from nflvalue import game_context as gc


# --------------------------------------------------------------------------- #
# Injury-shortened
# --------------------------------------------------------------------------- #
def _pbp_with_early_exit():
    rows = []
    # P1: 3 H1 targets (qtr 1-2), ZERO H2 -> early exit, while AAA plays H2
    for q in (1, 1, 2):
        rows.append({"season": 2023, "week": 5, "posteam": "AAA", "qtr": q,
                     "receiver_player_id": "P1", "rusher_player_id": np.nan,
                     "passer_player_id": "QB"})
    # P2: usage in BOTH halves -> not an exit
    for q in (1, 3):
        rows.append({"season": 2023, "week": 5, "posteam": "AAA", "qtr": q,
                     "receiver_player_id": "P2", "rusher_player_id": np.nan,
                     "passer_player_id": "QB"})
    # AAA runs >=10 H2 plays (so the team-kept-playing gate passes)
    for _ in range(12):
        rows.append({"season": 2023, "week": 5, "posteam": "AAA", "qtr": 3,
                     "receiver_player_id": np.nan, "rusher_player_id": np.nan,
                     "passer_player_id": "QB"})
    return pd.DataFrame(rows)


def test_injury_early_exit_flags_the_truncated_player_only():
    inj = gc.injury_shortened_weeks(_pbp_with_early_exit())
    flagged = set(inj[inj["injury_shortened"] == 1.0]["player_id"])
    assert "P1" in flagged
    assert "P2" not in flagged
    assert inj[inj["player_id"] == "P1"]["reason"].iloc[0] == "early_exit"


def test_injury_snap_collapse_is_leak_safe():
    # P3 steady ~0.9 for weeks 1-4, collapses to 0.20 in week 5 -> flagged wk5.
    # P4 steady 0.8 throughout -> never flagged. Weeks 1-3 for P3 lack the
    # >=3 prior-game history, so they must NOT flag (leak-safe baseline).
    snaps = []
    for wk, pct in [(1, 0.9), (2, 0.9), (3, 0.9), (4, 0.9), (5, 0.20)]:
        snaps.append({"season": 2023, "week": wk, "player_id": "P3", "offense_pct": pct})
    for wk in range(1, 6):
        snaps.append({"season": 2023, "week": wk, "player_id": "P4", "offense_pct": 0.8})
    snaps = pd.DataFrame(snaps)
    empty_pbp = pd.DataFrame(columns=["season", "week", "posteam", "qtr",
                                      "receiver_player_id", "rusher_player_id", "passer_player_id"])
    inj = gc.injury_shortened_weeks(empty_pbp, snap_counts=snaps)
    flagged = {(r.week, r.player_id) for r in inj[inj["injury_shortened"] == 1.0].itertuples()}
    assert (5, "P3") in flagged
    assert not any(pid == "P4" for _, pid in flagged)
    assert not any(pid == "P3" and wk < 5 for wk, pid in flagged)  # no early history -> no flag


# --------------------------------------------------------------------------- #
# Records to date
# --------------------------------------------------------------------------- #
def _mini_schedule():
    # 3 weeks, teams X and Y play each other each week; X wins wk1, Y wins wk2,
    # wk3 not yet played (scores NaN).
    return pd.DataFrame([
        {"season": 2023, "week": 1, "game_type": "REG", "home_team": "X", "away_team": "Y",
         "home_score": 24, "away_score": 17},
        {"season": 2023, "week": 2, "game_type": "REG", "home_team": "Y", "away_team": "X",
         "home_score": 30, "away_score": 20},
        {"season": 2023, "week": 3, "game_type": "REG", "home_team": "X", "away_team": "Y",
         "home_score": np.nan, "away_score": np.nan},
    ])


def test_records_to_date_are_strictly_prior():
    rec = gc.records_to_date(_mini_schedule()).set_index(["team", "week"])
    # week 1: nobody has played -> 0-0
    assert rec.loc[("X", 1), "wins"] == 0 and rec.loc[("X", 1), "losses"] == 0
    # week 2: only week 1 counted -> X 1-0, Y 0-1
    assert rec.loc[("X", 2), "wins"] == 1 and rec.loc[("Y", 2), "losses"] == 1
    # week 3: weeks 1-2 counted -> each 1-1
    assert rec.loc[("X", 3), "wins"] == 1 and rec.loc[("X", 3), "losses"] == 1
    assert rec.loc[("Y", 3), "wins"] == 1 and rec.loc[("Y", 3), "losses"] == 1
    # games_left never negative
    assert (rec["games_left"] >= 0).all()


# --------------------------------------------------------------------------- #
# Meaningless-game proxy: assert it matches its own documented cut formula
# --------------------------------------------------------------------------- #
def _conf_schedule(seasons_weeks=8, n_teams=8):
    """Deterministic AFC-only mini-season: fixed pairings, lower index always
    wins -> a real spread of records. Enough teams (>=8) that the 7-seed cut is
    a genuine threshold."""
    teams = [t for t, c in gc.CONFERENCE.items() if c == "AFC"][:n_teams]
    rows = []
    for wk in range(1, seasons_weeks + 1):
        # rotate pairings so win totals spread out
        order = teams[wk % len(teams):] + teams[:wk % len(teams)]
        for i in range(0, len(order) - 1, 2):
            a, b = order[i], order[i + 1]
            hi, lo = (a, b) if teams.index(a) < teams.index(b) else (b, a)
            rows.append({"season": 2023, "week": wk, "game_type": "REG",
                         "home_team": hi, "away_team": lo,
                         "home_score": 27, "away_score": 13})  # home (lower idx) wins
    return pd.DataFrame(rows)


def test_meaningless_matches_the_cut_formula():
    sched = _conf_schedule()
    week_min, margin = 6, 0.0
    got = gc.meaningless_game_flags(sched, week_min=week_min, clear_margin=margin)
    # nothing before week_min
    assert (got["week"] >= week_min).all()
    # recompute expectation independently from records_to_date + the documented
    # rule and assert the function agrees row-for-row
    rec = gc.records_to_date(sched)
    rec = rec[rec["week"] >= week_min].copy()
    rec["conf"] = rec["team"].map(gc.CONFERENCE)
    exp = {}
    for (s, w, conf), grp in rec.groupby(["season", "week", "conf"]):
        ws = grp["wins"].sort_values(ascending=False).to_numpy()
        cut = ws[gc.PLAYOFF_SEEDS - 1] if len(ws) >= gc.PLAYOFF_SEEDS else (ws[-1] if len(ws) else 0)
        for r in grp.itertuples(index=False):
            clinched = (r.wins - cut) > (r.games_left + margin)
            elim = (cut - r.wins) > (r.games_left + margin)
            exp[(s, w, r.team)] = 1.0 if (clinched or elim) else 0.0
    for r in got.itertuples(index=False):
        assert exp[(r.season, r.week, r.team)] == r.meaningless
    # and the proxy must actually FIRE somewhere in a lopsided league
    assert got["meaningless"].sum() > 0


def test_tag_player_weeks_adds_both_columns_and_tolerates_missing_inputs():
    pw = pd.DataFrame([{"season": 2023, "week": 5, "player_id": "P1", "team": "AAA"},
                       {"season": 2023, "week": 5, "player_id": "P2", "team": "AAA"}])
    # no inputs -> columns exist, all zero
    out = gc.tag_player_weeks(pw)
    assert set(["injury_shortened", "game_meaningless"]) <= set(out.columns)
    assert out["injury_shortened"].sum() == 0 and out["game_meaningless"].sum() == 0
    # with pbp -> P1 flagged
    out2 = gc.tag_player_weeks(pw, pbp=_pbp_with_early_exit())
    assert out2.set_index("player_id").loc["P1", "injury_shortened"] == 1.0
