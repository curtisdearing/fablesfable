"""Issued-pick ledger + grading: append-only records of the exact displayed card, graded
against a REAL official box (ESPN, 2026 week 2 WAS@DAL, fetched 2026-09-22T01:20Z).

Card/lean rows below are unit-test inputs, not picks anyone issued; the box and
play-by-play rows are real and unmodified (trimmed to the fields read)."""
import csv
import datetime as dt
import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nflvalue import db as dbmod  # noqa: E402
from nflvalue import issued_grading as ig  # noqa: E402
from nflvalue import issued_ledger as il  # noqa: E402
from nflvalue import pick_cards as pc  # noqa: E402
from nflvalue import prop_learning as pl  # noqa: E402
from nflvalue import settlement as st  # noqa: E402

BOX = Path(__file__).resolve().parent / "fixtures" / "espn_box_2026wk2"
BOX_CAPTURED = "2026-09-22T01:20:00Z"
GAME = "2026_02_WAS_DAL"          # kickoff 2026-09-20T20:25Z per the box header
NOW = dt.datetime(2026, 9, 20, 18, 0, tzinfo=dt.timezone.utc)
AVAIL = json.dumps({"availability": {"status": "OK", "eligibility": "eligible",
                                     "availability_state": "not_listed"}})


def _lean(**kw):
    r = {"season": 2026, "week": 2, "clock": "wed", "game_id": GAME, "player_id": "00-0033077",
         "name": "Dak Prescott", "market": "pass_attempts", "side": "under", "line": 32.5,
         "line_source": "odds_api", "price": 1.87, "book": "draftkings", "mean": 31.0, "sd": 5.0,
         "p_side": 0.6, "composite": 50.0, "status": "active", "as_of": "2026-09-20T17:30:00Z",
         "created_at": "2026-09-20T17:31:00Z", "quote_book": "draftkings", "quote_ts": "2026-09-20T17:00:00Z",
         "run_id": "gha:1-1", "code_sha": "abc", "forecast_version": "ff-football-only-v1",
         "selection_source": "football_selection_score", "stage_json": AVAIL}
    r.update(kw)
    return r


def _db(tmp_path, leans):
    conn = dbmod.connect(str(tmp_path / "s.db"))
    dbmod.upsert(conn, "leans", leans, ["season", "week", "clock", "game_id", "player_id", "market"])
    for l in leans:
        dbmod.upsert(conn, "lines", [{"ts": l["quote_ts"], "game_id": GAME, "book": l["quote_book"],
                                      "market": l["market"], "player_id": "", "player_name": l["name"],
                                      "side": l["side"], "point": l["line"], "price": l["price"]}],
                     ["ts", "game_id", "book", "market", "player_name", "side"])
        conn.execute("INSERT OR REPLACE INTO run_receipts VALUES (?,?,?,?,?,?,?)",
                     (l["run_id"], 2026, 2, "wed", l["as_of"], json.dumps({"run_id": l["run_id"], "publish": True}),
                      l["as_of"]))
    conn.commit()
    return conn


def _record(conn, when="2026-09-20T18:00:00Z", now=NOW):
    cards = pc.week_cards(conn, 2026, 2, now=now)["cards"]
    return il.record_cards(conn, 2026, 2, cards, recorded_at=when), cards


def _games():
    return ig.load_boxes([str(BOX / "401872944.json")], BOX_CAPTURED)


# ------------------------------------------------------------------ ledger --
def test_displayed_card_is_recorded_with_exact_text_provenance_and_clocks(tmp_path):
    conn = _db(tmp_path, [_lean()])
    written, cards = _record(conn)
    assert len(written) == 1 and cards[0]["status"] == "watch"
    rec = il.load(conn)[0]
    assert rec["display_html"] == pc.render_cards_html([cards[0]])       # the exact displayed text
    assert (rec["run_id"], rec["code_sha"], rec["forecast_version"]) == ("gha:1-1", "abc", "ff-football-only-v1")
    assert (rec["quote_book"], rec["quote_price"], rec["quote_ts"]) == ("draftkings", 1.87, "2026-09-20T17:00:00Z")
    assert rec["decision_ts"] == "2026-09-20T17:30:00Z" and rec["pick_class"] == "watch"
    assert rec["tier"] == "primary" and rec["revision"] == 1 and rec["supersedes"] is None
    assert rec["dist"] is None          # recorder is not the issuing commit: no reconstructed family


def test_ledger_is_append_only_and_revisions_are_new_records(tmp_path):
    conn = _db(tmp_path, [_lean()])
    _record(conn)
    assert _record(conn, when="2026-09-20T18:05:00Z")[0] == []          # identical content: no-op
    first = il.load(conn)[0]
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        conn.execute("UPDATE issued_picks SET line=99 WHERE record_id=?", (first["record_id"],))
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        conn.execute("DELETE FROM issued_picks")
    # the run re-prices: persist_leans replaces the lean, the ledger keeps both decisions
    dbmod.upsert(conn, "leans", [_lean(line=31.5, price=1.91, quote_ts="2026-09-20T17:40:00Z",
                                       as_of="2026-09-20T17:45:00Z")],
                 ["season", "week", "clock", "game_id", "player_id", "market"])
    conn.execute("INSERT INTO lines VALUES ('2026-09-20T17:40:00Z',?,'draftkings','pass_attempts','',"
                 "'Dak Prescott','under',31.5,1.91)", (GAME,))
    _record(conn, when="2026-09-20T18:10:00Z")
    recs = il.load(conn)
    assert [r["revision"] for r in recs] == [1, 2]
    assert recs[1]["supersedes"] == first["record_id"] and recs[0] == first


def test_write_week_cards_records_automatically_and_export_roundtrips(tmp_path):
    conn = _db(tmp_path, [_lean()])
    payload = pc.write_week_cards(conn, 2026, 2, out_dir=str(tmp_path / "rep"))
    assert payload["ledger_written"] == 1
    exported = (tmp_path / "rep" / "issued_picks_2026_wk2.json").read_text()
    assert exported == il.export(il.load(conn, 2026, 2))                 # deterministic
    assert json.loads(exported)["records"] == il.load(conn, 2026, 2)     # lossless roundtrip


def test_public_site_export_ids_match_the_run_ledger(tmp_path):
    conn = _db(tmp_path, [_lean()])
    _record(conn)
    cards = pc.week_cards(conn, 2026, 2, now=NOW)["cards"]
    pub = il.publication_records(cards, 2026, 2, "fresh", "2026-09-20T18:30:00Z")
    assert [p["record_id"] for p in pub] == [r["record_id"] for r in il.load(conn)]
    # a published hub.json round-trips through JSON and grades like the ledger row
    from scripts import grade_issued_picks as gip
    hub = {"season": 2026, "week": 2, "label": "fresh", "generated_at": "2026-09-20T18:30:00Z",
           "cards": json.loads(json.dumps(cards, default=str))}
    (tmp_path / "hub.json").write_text(json.dumps(hub))
    assert gip.main(["--hub", str(tmp_path / "hub.json"), "--box-dir", str(BOX), "--box-captured-at",
                     BOX_CAPTURED, "--out", str(tmp_path / "oh")]) == 0
    row = json.loads((tmp_path / "oh" / "issued_grades.json").read_text())["rows"][0]
    assert row["record_id"] == pub[0]["record_id"] and row["surface"] == "public_site:fresh"
    assert row["settlement"] == "win"


def test_analyst_override_needs_exact_text_and_is_its_own_tier(tmp_path):
    conn = dbmod.connect(str(tmp_path / "a.db"))
    card = {"game_id": GAME, "player_id": "00-0033077", "player": "Dak Prescott", "market": "pass_attempts",
            "side": "under", "line": 32.5, "run_as_of": "2026-09-20T17:30:00Z",
            "quote": {"book": "draftkings", "price_decimal": 1.87, "captured_at": "2026-09-20T17:00:00Z"}}
    with pytest.raises(ValueError):
        il.record_analyst_pick(conn, 2026, 2, card, "  ")
    il.record_analyst_pick(conn, 2026, 2, card, "Prescott under 32.5 attempts (DK -115)",
                           recorded_at="2026-09-20T17:35:00Z")
    rec = il.load(conn)[0]
    assert rec["tier"] == "analyst_override" and rec["display_html"].startswith("Prescott under 32.5")


def test_no_records_means_nothing_to_grade(tmp_path):
    from scripts import grade_issued_picks as gip
    dbmod.connect(str(tmp_path / "e.db")).close()
    rc = gip.main(["--db", str(tmp_path / "e.db"), "--box-dir", str(BOX),
                   "--box-captured-at", BOX_CAPTURED, "--out", str(tmp_path / "o")])
    assert rc == 3 and not (tmp_path / "o").exists()


# ------------------------------------------------ settlement vs real box --
def test_real_box_pass_attempts_exclude_sacks_row_level():
    g = _games()["WAS@DAL"]
    assert g["completed"] and g["kickoff"] == "2026-09-20T20:25Z"
    dak, how = ig.identify({"player_name": "Dak Prescott"}, g)
    assert how == "full_name_unique_in_game"
    assert ig.box_actual(dak, "pass_attempts") == 31.0          # official 26/31, sacked 2 times
    assert ig.box_actual(dak, "passing_yards") == 279.0
    pbp = pd.read_csv(BOX / "pbp_2026wk2_DAL_WAS_pass_plays.csv")
    off = st.official_pass_attempts(pbp).set_index("player_id")[st.OFFICIAL_PASS_ATTEMPTS_COL]
    incl = pbp.groupby("passer_player_id")["pass_attempt"].sum()
    names = pbp.drop_duplicates("passer_player_id").set_index("passer_player_name")["passer_player_id"]
    for short, full in (("D.Prescott", "Dak Prescott"), ("J.Daniels", "Jayden Daniels"),
                        ("M.Mariota", "Marcus Mariota")):
        ath, _ = ig.identify({"player_name": full}, g)
        assert off[names[short]] == ig.box_actual(ath, "pass_attempts"), short
    assert incl[names["D.Prescott"]] == 33                      # the sack-inclusive count is wrong


def test_grade_week_settles_pass_attempts_on_official_count_only(tmp_path):
    pbp = pd.read_csv(BOX / "pbp_2026wk2_DAL_WAS_pass_plays.csv")
    dak = pbp[pbp.passer_player_name == "D.Prescott"].passer_player_id.iloc[0]
    pw = pd.DataFrame([{"season": 2026, "week": 2, "player_id": dak, "pass_attempts": 33.0,
                        "rush_tds": 0.0, "rec_tds": 0.0}])
    leans = pd.DataFrame([{**_lean(player_id=dak, line=32.5, side="under"), "proj_components": None}])
    conn = dbmod.connect(str(tmp_path / "g.db"))
    out = pl.grade_week(conn, 2026, 2, st.with_official_pass_attempts(pw, pbp), leans=leans)
    assert out.iloc[0]["actual"] == 31.0 and out.iloc[0]["settlement"] == st.WIN
    unresolved = pl.grade_week(conn, 2026, 2, pw, leans=leans)          # no official column: never settle on 33
    assert unresolved.iloc[0]["settlement"] == st.UNRESOLVED


def test_official_pass_attempts_refuses_without_sack_column():
    with pytest.raises(ValueError, match="sack"):
        st.official_pass_attempts(pd.DataFrame({"season": [2026], "week": [2], "pass_attempt": [1],
                                                "passer_player_id": ["x"]}))


# --------------------------------------------------------------- grading --
def _rec(**kw):
    card = {"game_id": GAME, "player_id": "00-0033077", "player": "Dak Prescott", "market": "pass_attempts",
            "side": "under", "line": 32.5, "status": "watch", "run_as_of": "2026-09-20T17:30:00Z",
            "model_p_side": 0.6, "mean": 31.0, "sd": 5.0, "clock": "wed",
            "quote": {"book": "draftkings", "price_decimal": 1.87, "captured_at": "2026-09-20T17:00:00Z"},
            "provenance": {"run_id": "gha:1-1", "code_sha": "abc"}}
    tier = kw.pop("tier", "primary")
    recorded = kw.pop("recorded_at", "2026-09-20T18:00:00Z")
    dist = kw.pop("dist", "negbinom")
    card.update(kw)
    r = il.build_record(card, 2026, 2, tier=tier, dist=dist, display_html="unit-test card (not issued)")
    r.update(record_id=il._content_id(r), pick_key=il.pick_key(2026, 2, card, tier), revision=1,
             supersedes=None, recorded_at=recorded)
    return r


def test_grade_uses_last_pre_kick_revision_and_excludes_post_kick():
    a = _rec(line=33.5)
    b = {**_rec(line=32.5), "revision": 2, "pick_key": a["pick_key"]}
    late = {**_rec(line=30.5, recorded_at="2026-09-20T21:00:00Z"), "revision": 3, "pick_key": a["pick_key"]}
    res = ig.grade([a, b, late], _games())
    assert len(res["rows"]) == 1
    row = res["rows"][0]
    assert row["line"] == 32.5 and row["actual"] == 31.0 and row["settlement"] == st.WIN
    assert row["n_revisions_pre_kick"] == 2 and row["slate_tag"] == "sunday"
    assert row["brier"] == pytest.approx(0.16) and row["point_error"] == 0.0
    assert row["interval_lo"] < 31 < row["interval_hi"] and row["covered"] == 1
    assert {e["excluded"] for e in res["excluded"]} == {"superseded by a later pre-kick revision",
                                                         "post-kick or unclocked record"}


def test_push_unresolved_tiers_and_single_game_claims():
    push = _rec(line=31.0, side="over")
    ghost = _rec(player_id="x", player="Nobody Here", market="receptions", line=2.5)
    exp = _rec(tier="experimental", line=30.5, side="over")
    res = ig.grade([push, ghost, exp], _games())
    by = {(r["tier"], r["player_name"]): r for r in res["rows"]}
    assert by[("primary", "Dak Prescott")]["settlement"] == st.PUSH
    assert by[("primary", "Dak Prescott")].get("brier") is None
    assert by[("primary", "Nobody Here")]["settlement"] == st.UNRESOLVED
    assert "not knowable" in by[("primary", "Nobody Here")]["detail"]
    grp = res["groups"]
    assert set(grp) == {f"{t}|watch|run_reports|{s}" for t in ("experimental", "primary")
                        for s in ("sunday", "all_slates")}
    assert grp["primary|watch|run_reports|all_slates"]["n_records"] == 2
    assert grp["primary|watch|run_reports|all_slates"]["n_games"] == 1
    assert grp["primary|watch|run_reports|all_slates"]["claims"].startswith("none")
    assert len(res["book_rules_unverified"]) == len(st.BOOK_RULES_UNVERIFIED) >= 3
    assert res["refit"].startswith("none")


def test_thursday_is_its_own_slate_tag():
    assert ig.slate_tag("2026-09-25T00:15Z") == "thursday"      # 8:15 pm ET Thursday 9/24
    assert ig.slate_tag("2026-09-20T17:00Z") == "sunday"


def test_clv_requires_a_pre_kick_close_at_the_issued_line():
    r = _rec()
    key = (GAME, "00-0033077", "pass_attempts", "under")
    ok = ig.clv_row(r, {key: {"close_ts": "2026-09-20T20:00:00Z", "close_point": 32.5, "close_prob": 0.56}},
                    "2026-09-20T20:25Z")
    assert ok["clv_status"] == "valid" and ok["clv_prob_vs_entry_breakeven"] == pytest.approx(0.56 - 1 / 1.87)
    late = ig.clv_row(r, {key: {"close_ts": "2026-09-20T20:30:00Z", "close_point": 32.5, "close_prob": 0.5}},
                      "2026-09-20T20:25Z")
    assert late["clv_status"].startswith("invalid")
    moved = ig.clv_row(r, {key: {"close_ts": "2026-09-20T20:00:00Z", "close_point": 31.5, "close_prob": 0.5}},
                       "2026-09-20T20:25Z")
    assert moved["clv_status"].startswith("invalid")


def test_stat_correction_is_reported_beside_the_original():
    r = _rec()
    prior = [{"record_id": r["record_id"], "actual": 30.0, "actuals_captured_at": "2026-09-21T01:00:00Z"}]
    res = ig.grade([r], _games(), prior_rows=prior)
    assert res["stat_corrections"] == [{"record_id": r["record_id"], "prior_actual": 30.0,
                                        "prior_captured_at": "2026-09-21T01:00:00Z", "actual": 31.0,
                                        "captured_at": BOX_CAPTURED}]


def test_script_replays_saved_state_deterministically_db_and_export_agree(tmp_path):
    from scripts import grade_issued_picks as gip
    conn = _db(tmp_path, [_lean()])
    _record(conn)                                   # recorded pre-kick (a wall-clock write today is post-kick)
    (tmp_path / "rep").mkdir()
    (tmp_path / "rep" / "issued_picks_2026_wk2.json").write_text(il.export(il.load(conn, 2026, 2)))
    conn.close()
    args = ["--box-dir", str(BOX), "--box-captured-at", BOX_CAPTURED, "--season", "2026", "--week", "2"]
    assert gip.main(["--db", str(tmp_path / "s.db"), *args, "--out", str(tmp_path / "o1")]) == 0
    assert gip.main(["--db", str(tmp_path / "s.db"), *args, "--out", str(tmp_path / "o2")]) == 0
    assert gip.main(["--export", str(tmp_path / "rep" / "issued_picks_2026_wk2.json"), *args,
                     "--out", str(tmp_path / "o3")]) == 0
    one = (tmp_path / "o1" / "issued_grades.json").read_text()
    assert one == (tmp_path / "o2" / "issued_grades.json").read_text() == \
        (tmp_path / "o3" / "issued_grades.json").read_text()
    rows = list(csv.DictReader(open(tmp_path / "o1" / "issued_grades_rows.csv")))
    assert len(rows) == 1 and rows[0]["settlement"] == "win" and rows[0]["actual"] == "31.0"
    # the same card recorded after kickoff is never graded as a pre-kick decision
    conn = sqlite3.connect(tmp_path / "s.db")
    late = il.load(conn)[0]
    res = ig.grade([{**late, "recorded_at": "2026-09-20T20:26:00Z"}], _games())
    assert res["rows"] == [] and res["excluded"][0]["excluded"] == "post-kick or unclocked record"


def test_migration_7_is_additive_on_the_production_fixture(tmp_path):
    import shutil
    src = Path(__file__).resolve().parent / "fixtures" / "nfl_props_state_2026wk1_user_version1.db"
    if not src.exists():
        pytest.skip("populated production-state fixture not present")
    shutil.copy(src, tmp_path / "p.db")
    raw = sqlite3.connect(tmp_path / "p.db")
    before = raw.execute("SELECT COUNT(*) FROM leans").fetchone()[0]
    raw.close()
    conn = dbmod.connect(str(tmp_path / "p.db"))
    assert dbmod.user_version(conn) == 7 and conn.execute("SELECT COUNT(*) FROM leans").fetchone()[0] == before
    assert conn.execute("SELECT COUNT(*) FROM issued_picks").fetchone()[0] == 0   # nothing back-filled as issued
