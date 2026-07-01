"""Phase 2 end-to-end: candidate enumeration determinism, eligibility gates,
report smoke, lean persistence idempotency. Runs on small synthetic inputs
(no parquet rebuild) so the suite stays fast."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nflvalue import db as dbmod  # noqa: E402
from nflvalue import report as rpt  # noqa: E402
from nflvalue.candidates import WeekInputs, enumerate_candidates  # noqa: E402

SEASON, WEEK = 2023, 9


def _pw_row(week, pid, name, team, defteam, role, roll_games=8, **kw):
    base = dict(
        season=SEASON, week=week, player_id=pid, player_name=name, team=team,
        defteam=defteam, role=role, position_source="roster",
        targets=0.0, receptions=0.0, rec_yards=0.0, air_yards_sum=0.0, yac_sum=0.0,
        carries=0.0, rush_yards=0.0, pass_attempts=0.0, completions=0.0, pass_yards=0.0,
        pass_tds=0.0, rush_tds=0.0, rec_tds=0.0,
        team_pass_att=32.0, team_rush_att=26.0, team_plays=58.0,
        roll_games=float(roll_games), roll_targets=0.0, roll_target_share=0.0,
        roll_air_yards=0.0, roll_adot=0.0, roll_carries=0.0, roll_carry_share=0.0,
        roll_pass_attempts=0.0, roll_completions=0.0,
        roll_ypt=0.0, roll_catch_rate=0.0, roll_ypc=0.0, roll_ypa=0.0,
        roll_pass_td_rate=0.0, roll_rush_td_rate=0.02, roll_rec_td_rate=0.03,
    )
    base.update(kw)
    return base


def synthetic_inputs():
    """Two teams (AAA @ BBB in week 9), weeks 1-9, four skill players."""
    rows = []
    for wk in range(1, 10):
        opp = ("BBB", "AAA")
        # AAA: a WR with steady 8-target/68-yard games, an RB, a cold-start WR, a scrub
        rows.append(_pw_row(wk, "WR_A", "Alpha Wideout", "AAA", "BBB", "WR",
                            targets=8.0, receptions=5.0, rec_yards=68.0,
                            roll_targets=8.0, roll_target_share=0.25, roll_ypt=8.4,
                            roll_catch_rate=0.64))
        rows.append(_pw_row(wk, "RB_A", "Alpha Back", "AAA", "BBB", "RB",
                            carries=16.0, rush_yards=72.0,
                            roll_carries=16.0, roll_carry_share=0.62, roll_ypc=4.4))
        rows.append(_pw_row(wk, "WR_COLD", "Cold Start", "AAA", "BBB", "WR", roll_games=2,
                            targets=6.0, receptions=4.0, rec_yards=51.0,
                            roll_targets=6.0, roll_target_share=0.19, roll_ypt=8.0,
                            roll_catch_rate=0.6))
        rows.append(_pw_row(wk, "WR_SCRUB", "Bench Scrub", "AAA", "BBB", "WR",
                            targets=1.0, receptions=1.0, rec_yards=7.0,
                            roll_targets=1.0, roll_target_share=0.03, roll_ypt=6.0,
                            roll_catch_rate=0.5))
        # BBB: a QB
        rows.append(_pw_row(wk, "QB_B", "Bravo Quarterback", "BBB", "AAA", "QB",
                            pass_attempts=34.0, completions=22.0, pass_yards=245.0,
                            roll_pass_attempts=34.0, roll_completions=22.0, roll_ypa=7.2))
    pw = pd.DataFrame(rows)

    opd_rows = []
    for wk in range(1, 10):
        for defteam, other in (("AAA", "BBB"), ("BBB", "AAA")):
            for role, col in (("QB", "roll_ypa_allowed_factor"), ("WR", "roll_ypt_allowed_factor"),
                              ("TE", "roll_ypt_allowed_factor"), ("RB", "roll_ypc_allowed_factor")):
                r = dict(season=SEASON, week=wk, defteam=defteam, role=role,
                         targets_allowed=0.0, rec_yards_allowed=0.0, carries_allowed=0.0,
                         rush_yards_allowed=0.0, pass_yards_allowed=0.0,
                         epa_allowed_sum=0.0, plays_faced=30.0, roll_games=wk - 1,
                         roll_ypt_allowed_factor=None, roll_ypc_allowed_factor=None,
                         roll_ypa_allowed_factor=None, roll_epa_allowed_factor=1.0)
                r[col] = 1.10 if defteam == "BBB" else 0.95   # BBB defense soft, AAA tough
                opd_rows.append(r)
    opd = pd.DataFrame(opd_rows)

    tw = pd.DataFrame([
        dict(season=SEASON, week=wk, team=t, roll_team_pass_att=32.0, roll_team_rush_att=26.0)
        for wk in range(1, 10) for t in ("AAA", "BBB")
    ])

    schedules = pd.DataFrame([dict(
        game_id=f"{SEASON}_09_AAA_BBB", season=SEASON, week=WEEK, game_type="REG",
        gameday="2023-11-05", gametime="13:00", home_team="BBB", away_team="AAA",
        spread_line=3.0, total_line=44.5,
    )])
    return WeekInputs(pw=pw, opd=opd, tw=tw, schedules=schedules)


@pytest.fixture()
def inputs():
    return synthetic_inputs()


def test_enumeration_deterministic(inputs):
    a = enumerate_candidates(SEASON, WEEK, inputs=inputs)
    b = enumerate_candidates(SEASON, WEEK, inputs=inputs)
    pd.testing.assert_frame_equal(a, b)


def test_eligibility_gates(inputs):
    df = enumerate_candidates(SEASON, WEEK, inputs=inputs)
    pids = set(df["player_id"])
    assert "WR_A" in pids and "RB_A" in pids and "QB_B" in pids
    assert "WR_COLD" not in pids     # cold-start gate (roll_games=2 < 3)
    assert "WR_SCRUB" not in pids    # min-usage floor (1 trailing target)


def test_synthetic_lines_tagged_and_no_market(inputs):
    df = enumerate_candidates(SEASON, WEEK, inputs=inputs)
    assert (df["line_source"] == "synthetic_trailing_mean").all()
    assert df["line"].notna().all()
    # half-point lines can never push
    non_td = df[df["market"] != "anytime_td"]
    assert ((non_td["line"] % 1) == 0.5).all()


def test_real_prop_line_overrides_synthetic(inputs):
    prop_lines = pd.DataFrame([dict(
        game_id=f"{SEASON}_09_AAA_BBB", market="receiving_yards", player_id="WR_A",
        point=61.5, over_price=1.87, under_price=1.95, book="bookx")])
    df = enumerate_candidates(SEASON, WEEK, inputs=inputs, prop_lines=prop_lines)
    row = df[(df["player_id"] == "WR_A") & (df["market"] == "receiving_yards")].iloc[0]
    assert row["line_source"] == "odds_api"
    assert row["line"] == 61.5
    assert row["prices"]["book"] == "bookx"


def test_generate_report_smoke_and_screen_count(inputs, tmp_path, monkeypatch):
    monkeypatch.setattr(rpt, "REPORTS_DIR", str(tmp_path))
    monkeypatch.setattr(rpt, "WEEKLY_PROPS_JSON", str(tmp_path / "weekly_props.json"))
    res = rpt.generate(SEASON, WEEK, inputs=inputs, mode="historical",
                       write_files=True, persist=False)
    md = res["markdown"]
    assert "1-800-GAMBLER" in md
    assert "Leans, not locks" in md
    assert "not part of the composite score" in md
    assert "AAA @ BBB" in md
    g = res["games"][0]
    assert g["screened"] == f"{len(g['leans'])} of {g['screened_n']}"
    n_expected = res["n_candidates"]
    assert g["screened_n"] == n_expected            # single game: N == all screened
    assert (tmp_path / f"props_week_{SEASON}_{WEEK}.md").exists()
    assert "historical run — live injury/news feeds not applicable" in md


def test_generate_respects_publish_gate(inputs, tmp_path, monkeypatch):
    monkeypatch.setattr(rpt, "REPORTS_DIR", str(tmp_path))
    monkeypatch.setattr(rpt, "WEEKLY_PROPS_JSON", str(tmp_path / "weekly_props.json"))
    res = rpt.generate(SEASON, WEEK, inputs=inputs, publish=False,
                       publish_reasons=["injuries feed stale (52.0h old)"],
                       write_files=True, persist=False)
    assert "NOT PUBLISHED" in res["markdown"]
    assert "injuries feed stale" in res["markdown"]


def test_persist_leans_idempotent(inputs, tmp_path):
    conn = dbmod.connect(str(tmp_path / "t.db"))
    df = enumerate_candidates(SEASON, WEEK, inputs=inputs)
    from nflvalue.shortlist import shortlist_week
    games = shortlist_week(df)
    n1 = rpt.persist_leans(conn, SEASON, WEEK, "wed", games, as_of="2023-11-08T12:00:00Z")
    n2 = rpt.persist_leans(conn, SEASON, WEEK, "wed", games, as_of="2023-11-08T12:00:00Z")
    count = dbmod.query_df(conn, "SELECT COUNT(*) AS n FROM leans").iloc[0]["n"]
    assert n1 == n2 == count                        # INSERT OR REPLACE: no dupes
    conn.close()
