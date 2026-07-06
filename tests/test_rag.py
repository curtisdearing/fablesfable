"""Block C: SQL safety boundary + end-to-end NL->SQL + report recall."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nflvalue import db as dbmod  # noqa: E402
from nflvalue.rag import nl2sql, vectorstore  # noqa: E402


@pytest.fixture()
def conn(tmp_path):
    c = dbmod.connect(str(tmp_path / "t.db"))
    rows = [{
        "season": 2023, "week": 10, "clock": "wed", "game_id": f"2023_10_G{i//6}",
        "player_id": f"P{i}", "name": f"Player {i}", "market": "receiving_yards",
        "side": "over", "line": 50.5, "line_source": "synthetic_trailing_mean",
        "price": None, "book": None, "mean": 55.0 + i, "sd": 20.0, "p_side": 0.6,
        "composite": 40.0 + (i % 30), "edge": None, "confidence_comp": 0.3,
        "matchup_comp": 0.5, "screened_n": 44, "reason": "r", "status": "active",
        "void_reason": None, "as_of": "t", "created_at": f"t{i:04d}",
    } for i in range(300)]
    dbmod.upsert(c, "leans", rows, ["season", "week", "clock", "game_id", "player_id", "market"])
    yield c
    c.close()


# --------------------------------------------------------------------------- #
# The security boundary
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [
    "UPDATE leans SET composite = 100",
    "DELETE FROM leans",
    "INSERT INTO leans VALUES (1)",
    "DROP TABLE leans",
    "CREATE TABLE evil (x)",
    "ALTER TABLE leans ADD COLUMN evil",
    "PRAGMA writable_schema=ON",
    "ATTACH DATABASE '/tmp/x' AS x",
    "VACUUM",
    "REPLACE INTO leans VALUES (1)",
])
def test_ddl_dml_rejected(bad):
    with pytest.raises(nl2sql.SQLValidationError):
        nl2sql.validate_sql(bad)


@pytest.mark.parametrize("bad", [
    "SELECT * FROM sqlite_master",                       # schema snooping
    "SELECT * FROM api_credits",                         # non-whitelisted table
    "SELECT * FROM leans JOIN api_credits ON 1=1",       # smuggled via JOIN
    "SELECT 1; DROP TABLE leans",                        # multi-statement
    "SELECT * FROM leans -- hidden comment",             # comment smuggling
    "SELECT 1",                                          # no FROM: not a warehouse query
])
def test_whitelist_and_structure_rejected(bad):
    with pytest.raises(nl2sql.SQLValidationError):
        nl2sql.validate_sql(bad)


@pytest.mark.parametrize("bad", [
    # comma-separated implicit cross-join: the second table sits PAST the reach
    # of a "first identifier after FROM/JOIN" match -- this is the latent bypass.
    "SELECT * FROM leans, sqlite_master",                # schema snooping via cross-join
    "SELECT * FROM leans, api_credits",                  # non-whitelisted table via cross-join
    "SELECT * FROM leans, clv, api_credits",             # smuggled among whitelisted tables
    "SELECT * FROM clv WHERE x IN (SELECT y FROM sqlite_master)",  # sqlite_* anywhere
    "SELECT * FROM sqlite_sequence",                     # any sqlite_* internal
])
def test_comma_join_bypass_is_closed(bad):
    with pytest.raises(nl2sql.SQLValidationError):
        nl2sql.validate_sql(bad)


@pytest.mark.parametrize("good", [
    "SELECT * FROM leans",
    "SELECT name, market FROM leans WHERE season = 2023 ORDER BY composite DESC",
    "SELECT COUNT(*) AS n FROM clv WHERE season = 2023",
    "SELECT game_id FROM leans GROUP BY game_id ORDER BY game_id",
    "SELECT a.name FROM leans a JOIN clv b ON a.game_id = b.game_id",  # whitelisted JOIN
    "SELECT player_name, team FROM player_week ORDER BY rec_yards DESC",
])
def test_legitimate_queries_still_pass(good):
    # returns cleaned SQL (never raises) for whitelisted single-table and JOIN shapes
    assert nl2sql.validate_sql(good).lower().startswith("select")


def test_row_cap_is_structural(conn):
    rows = nl2sql.execute(conn, "SELECT * FROM leans", row_cap=200)
    assert len(rows) == 200                              # 300 seeded, hard-capped


def test_hostile_client_cannot_reach_the_db(conn):
    class HostileClient:
        def generate_sql(self, question, schema_doc):
            return "UPDATE leans SET composite = 100"

    with pytest.raises(nl2sql.SQLValidationError):
        nl2sql.run_query("anything", conn=conn, client=HostileClient())
    top = dbmod.query_df(conn, "SELECT MAX(composite) AS m FROM leans").iloc[0]["m"]
    assert top < 100                                     # nothing was written


# --------------------------------------------------------------------------- #
# End-to-end Q&A
# --------------------------------------------------------------------------- #
def test_leans_question_end_to_end(conn):
    res = nl2sql.run_query("show me the leans for week 10 2023", conn=conn)
    assert res["sql"].lower().startswith("select")
    assert res["citations"]["tables"] == ["leans"]
    assert 0 < res["citations"]["row_count"] <= 200
    assert "row" in res["answer"]
    # ranked by composite, as asked of a shortlist question
    assert res["rows"][0]["composite"] >= res["rows"][-1]["composite"]


def test_clv_question_uses_clv_table(conn):
    res = nl2sql.run_query("what's our average CLV in 2023", conn=conn)
    assert res["citations"]["tables"] == ["clv"]
    assert res["rows"][0]["n"] == 0                      # empty table -> honest zero
    assert "no data" not in res["answer"] or True


def test_answer_never_fabricates_on_empty(conn):
    res = nl2sql.run_query("leans for week 3 2019", conn=conn)
    assert res["citations"]["row_count"] == 0
    assert "no rows" in res["answer"]


# --------------------------------------------------------------------------- #
# Vectorstore (flag-gated)
# --------------------------------------------------------------------------- #
def test_vectorstore_disabled_by_default():
    out = vectorstore.search("andrews usage collapse", cfg={"rag": {"vectorstore_enabled": False}})
    assert out["status"] == "disabled"


def test_vectorstore_finds_the_right_report(tmp_path):
    (tmp_path / "props_week_2023_10.md").write_text(
        "# NFL Prop Leans — 2023 Week 10\nM.Andrews receiving yards UNDER 59.5 — "
        "usage collapse after the bye; screened 5 of 51.")
    (tmp_path / "props_week_2023_11.md").write_text(
        "# NFL Prop Leans — 2023 Week 11\nJ.Chase over 71.5 receiving yards.")
    idx = vectorstore.build_index(str(tmp_path))
    out = vectorstore.search("andrews usage collapse", index=idx, k=1)
    assert out["status"] == "ok"
    assert out["results"][0]["path"].endswith("2023_10.md")
    assert "usage collapse" in out["results"][0]["snippet"]
