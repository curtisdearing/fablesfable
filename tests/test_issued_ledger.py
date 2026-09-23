"""Issued-pick ledger + grading, against a REAL official box (ESPN event 401872944, 2026 week 2
WSH@DAL final, fetched 2026-09-22T01:20Z; trimmed, source sha256 inside the fixture).

Card/lean rows here are unit-test inputs, not picks anyone was given; the box and the
play-by-play rows are real and unmodified apart from trimming to the fields read."""
import copy
import csv
import datetime as dt
import hashlib
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
BOX_FILE = BOX / "401872944.json"
BOX_CAPTURED = "2026-09-22T01:20:00Z"
GAME = "2026_02_WAS_DAL"          # kickoff 2026-09-20T20:25Z per the box header
DAK = "00-0033077"
AVAIL = json.dumps({"availability": {"status": "OK", "eligibility": "eligible",
                                     "availability_state": "not_listed"}})


def T(hhmm):
    return dt.datetime.fromisoformat(f"2026-09-20T{hhmm}:00+00:00")


def Z(hhmm):
    return f"2026-09-20T{hhmm}:00Z"


def _lean(**kw):
    r = {"season": 2026, "week": 2, "clock": "wed", "game_id": GAME, "player_id": DAK,
         "name": "Dak Prescott", "market": "pass_attempts", "side": "under", "line": 32.5,
         "line_source": "odds_api", "price": 1.87, "book": "draftkings", "mean": 31.0, "sd": 5.0,
         "p_side": 0.6, "composite": 50.0, "status": "active", "as_of": Z("10:30"),
         "created_at": Z("10:31"), "quote_book": "draftkings", "quote_ts": Z("10:00"),
         "run_id": "gha:1-1", "code_sha": "abc", "forecast_version": "ff-football-only-v1",
         "selection_source": "football_selection_score", "stage_json": AVAIL}
    r.update(kw)
    return r


def _put(conn, lean):
    dbmod.upsert(conn, "leans", [lean], ["season", "week", "clock", "game_id", "player_id", "market"])
    dbmod.upsert(conn, "lines", [{"ts": lean["quote_ts"], "game_id": GAME, "book": lean["quote_book"],
                                  "market": lean["market"], "player_id": "", "player_name": lean["name"],
                                  "side": lean["side"], "point": lean["line"], "price": lean["price"]}],
                 ["ts", "game_id", "book", "market", "player_name", "side"])
    conn.execute("INSERT OR REPLACE INTO run_receipts VALUES (?,?,?,?,?,?,?)",
                 (lean["run_id"], 2026, 2, "wed", lean["as_of"],
                  json.dumps({"run_id": lean["run_id"], "publish": True}), lean["as_of"]))
    conn.commit()


def _db(tmp_path, lean=None):
    conn = dbmod.connect(str(tmp_path / "s.db"))
    _put(conn, lean or _lean())
    return conn


def _gen(conn, at):
    """A run renders the cards at ``at`` and the ledger records them (stage generated)."""
    cards = pc.week_cards(conn, 2026, 2, now=T(at))["cards"]
    return il.record_cards(conn, 2026, 2, cards, recorded_at=Z(at)), cards


def _publish(tmp_path, conn, cards, published, recorded, label="fresh", name="site"):
    """Save a page payload + its manifest the way build_public_site does, then record it."""
    d = tmp_path / name
    (d / "api").mkdir(parents=True, exist_ok=True)
    hub = {"season": 2026, "week": 2, "label": label, "generated_at": Z(published),
           "cards": json.loads(json.dumps(cards, default=str))}
    raw = json.dumps(hub, indent=2).encode()
    (d / "api" / "hub.json").write_bytes(raw)
    (d / "publication.json").write_text(json.dumps({"season": 2026, "week": 2, "label": label,
                                                    "published_at": Z(published),
                                                    "files": {"api/hub.json": hashlib.sha256(raw).hexdigest()}}))
    if conn is not None:
        il.record_publication(conn, str(d / "api" / "hub.json"), str(d / "publication.json"),
                              recorded_at=Z(recorded))
    return d


def _boxes(*paths):
    return ig.load_boxes([str(p) for p in (paths or [BOX_FILE])], BOX_CAPTURED)


@pytest.fixture
def actionable(monkeypatch):
    """No market is validated in production; the recommendation path needs one for the test."""
    monkeypatch.setattr(pc, "VALIDATED_MARKETS", frozenset({"pass_attempts"}))


# ------------------------------------------------------------ ledger + stages --
def test_generated_record_keeps_exact_text_and_is_not_proof_of_delivery(tmp_path, actionable):
    conn = _db(tmp_path)
    written, cards = _gen(conn, "11:00")
    assert len(written) == 1 and cards[0]["status"] == "actionable"
    rec = il.load(conn)[0]
    assert rec["display_html"] == pc.render_cards_html([cards[0]])
    assert (rec["quote_book"], rec["quote_price"], rec["quote_ts"]) == ("draftkings", 1.87, Z("10:00"))
    assert rec["decision_ts"] == Z("10:30") and rec["pick_class"] == "recommendation" and rec["dist"] is None
    assert [e["stage"] for e in rec["events"]] == ["generated"]
    res = ig.grade(il.load(conn), _boxes())
    assert res["counts"]["recommendations_given"] == 0 and res["counts"]["generated_not_shown"] == 1


def test_ledger_and_events_are_append_only(tmp_path):
    conn = _db(tmp_path)
    _gen(conn, "11:00")
    assert _gen(conn, "11:05")[0] == []                               # identical content: no new record
    assert len(il.load(conn)[0]["events"]) == 1                       # and no duplicate generated event
    for sql in ("UPDATE issued_picks SET line=99", "DELETE FROM issued_picks",
                "UPDATE issued_pick_events SET stage='delivered'", "DELETE FROM issued_pick_events"):
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            conn.execute(sql)


def test_stale_rerender_to_pass_never_erases_the_given_recommendation(tmp_path, actionable):
    conn = _db(tmp_path)
    _, cards = _gen(conn, "11:00")
    _publish(tmp_path, conn, cards, "11:05", "11:06")
    _, later = _gen(conn, "17:00")                                     # quote now 7 h old -> pass
    assert later[0]["status"] == "pass"
    recs = il.load(conn)
    assert [r["pick_class"] for r in recs] == ["recommendation", "not_a_pick"]
    res = ig.grade(recs, _boxes())
    given = res["sections"]["recommendations_given"]["rows"]
    assert len(given) == 1 and given[0]["record_id"] == recs[0]["record_id"]
    assert given[0]["evidence_stage"] == "published" and given[0]["settlement"] == st.WIN
    assert given[0]["later_records"][0]["change"].startswith("status_changed_only")
    snap = res["sections"]["latest_pre_kick_snapshot"]["rows"]
    assert [r["record_id"] for r in snap] == [recs[1]["record_id"]]   # separate analysis only


def test_line_change_is_a_new_given_pick_that_revises_and_keeps_the_first(tmp_path, actionable):
    conn = _db(tmp_path)
    _, c1 = _gen(conn, "11:00")
    _publish(tmp_path, conn, c1, "11:05", "11:06", name="p1")
    _put(conn, _lean(line=31.5, side="over", price=1.91, quote_ts=Z("12:00"), as_of=Z("12:10")))
    _, c2 = _gen(conn, "12:20")
    _publish(tmp_path, conn, c2, "12:25", "12:26", name="p2")
    res = ig.grade(il.load(conn), _boxes())
    given = res["sections"]["recommendations_given"]["rows"]
    assert [(r["side"], r["line"], r["settlement"]) for r in given] == [("under", 32.5, st.WIN),
                                                                        ("over", 31.5, st.LOSS)]
    assert given[0]["later_records"][0]["change"] == "side_or_line_changed"
    assert given[1]["revises"] == given[0]["record_id"]


def test_cross_surface_duplicate_is_one_decision(tmp_path, actionable):
    conn = _db(tmp_path)
    _, cards = _gen(conn, "11:00")
    _publish(tmp_path, conn, cards, "11:05", "11:06", name="a")
    _publish(tmp_path, conn, cards, "11:30", "11:31", name="b", label="replay")
    recs = il.load(conn)
    assert len(recs) == 1 and sorted(e["stage"] for e in recs[0]["events"]) == ["generated", "published",
                                                                                 "published"]
    # a re-render with different display but the same side/line/quote is the same decision
    twin = {**recs[0], "record_id": "x" * 64, "display_html": "other text", "events": recs[0]["events"]}
    res = ig.grade([recs[0], twin], _boxes())
    rows = res["sections"]["recommendations_given"]["rows"]
    assert len(rows) == 1 and rows[0]["same_decision_records"] == ["x" * 64]


def test_watch_list_is_not_a_recommendation(tmp_path):
    conn = _db(tmp_path)
    _, cards = _gen(conn, "11:00")
    assert cards[0]["status"] == "watch"
    _publish(tmp_path, conn, cards, "11:05", "11:06")
    res = ig.grade(il.load(conn), _boxes())
    assert res["counts"]["recommendations_given"] == 0 and res["counts"]["watch_published"] == 1


def test_publication_recorded_after_kickoff_is_retrospective(tmp_path, actionable):
    conn = _db(tmp_path)
    _, cards = _gen(conn, "11:00")
    conn2 = dbmod.connect(str(tmp_path / "archive.db"))               # an archive recorded later
    _publish(tmp_path, conn2, cards, "11:05", "21:00")
    res = ig.grade(il.load(conn2), _boxes())
    assert res["counts"]["recommendations_given"] == 0 and res["counts"]["retrospective"] == 1


def test_hub_reconstruction_needs_its_manifest_and_is_never_prospective(tmp_path, actionable):
    from scripts import grade_issued_picks as gip
    conn = _db(tmp_path)
    _, cards = _gen(conn, "11:00")
    d = _publish(tmp_path, None, cards, "11:05", None)
    args = ["--box-dir", str(BOX), "--box-captured-at", BOX_CAPTURED]
    assert gip.main(["--hub", str(d / "api/hub.json"), *args, "--out", str(tmp_path / "o0")]) == 5
    (d / "api/hub.json").write_text((d / "api/hub.json").read_text() + " ")   # payload no longer the published one
    assert gip.main(["--hub", str(d / "api/hub.json"), "--publication", str(d / "publication.json"), *args,
                     "--out", str(tmp_path / "o1")]) == 5
    d = _publish(tmp_path, None, cards, "11:05", None, name="ok")
    assert gip.main(["--hub", str(d / "api/hub.json"), "--publication", str(d / "publication.json"), *args,
                     "--out", str(tmp_path / "o2")]) == 0
    res = json.loads((tmp_path / "o2" / "issued_grades.json").read_text())
    assert res["counts"]["recommendations_given"] == 0 and res["counts"]["retrospective"] == 1
    assert res["sections"]["retrospective"]["rows"][0]["record_id"] == il.load(conn)[0]["record_id"]


def test_delivered_pick_needs_exact_text_and_message_id(tmp_path):
    from scripts import record_issued_pick as rip
    conn = dbmod.connect(str(tmp_path / "a.db"))
    card = {"game_id": GAME, "player_id": DAK, "player": "Dak Prescott", "market": "pass_attempts",
            "side": "under", "line": 32.5, "run_as_of": Z("10:30"), "model_p_side": 0.6,
            "quote": {"book": "draftkings", "price_decimal": 1.87, "captured_at": Z("10:00")}}
    with pytest.raises(ValueError, match="exact text"):
        il.record_delivered(conn, 2026, 2, card, " ", "m1", Z("11:00"), "chat")
    with pytest.raises(ValueError, match="message id"):
        il.record_delivered(conn, 2026, 2, card, "Prescott under 32.5", "", Z("11:00"), "chat")
    conn.close()
    (tmp_path / "card.json").write_text(json.dumps(card))
    (tmp_path / "msg.txt").write_text("Prescott under 32.5 pass attempts (DK -115)")
    base = ["--db", str(tmp_path / "a.db"), "delivered", "--season", "2026", "--week", "2", "--card",
            str(tmp_path / "card.json"), "--text-file", str(tmp_path / "msg.txt"), "--channel", "chat",
            "--delivered-at", Z("11:00"), "--message-id", "msg-123"]
    assert rip.main(base) == 0
    conn = sqlite3.connect(tmp_path / "a.db")
    recs = il.load(conn)
    assert recs[0]["tier"] == "analyst_override" and recs[0]["display_html"].startswith("Prescott under")
    assert json.loads(recs[0]["events"][0]["evidence_json"])["message_id"] == "msg-123"
    # recorded at wall clock (today, after this 2026-09-20 game): kept, but retrospective
    res = ig.grade(recs, _boxes())
    assert res["counts"]["recommendations_given"] == 0 and res["counts"]["retrospective"] == 1
    c2 = dbmod.connect(str(tmp_path / "b.db"))
    il.record_delivered(c2, 2026, 2, card, "Prescott under 32.5", "msg-9", Z("11:00"), "chat",
                        recorded_at=Z("11:01"))
    given = ig.grade(il.load(c2), _boxes())["sections"]["recommendations_given"]["rows"]
    assert len(given) == 1 and given[0]["evidence_stage"] == "delivered" and given[0]["settlement"] == st.WIN


def test_no_records_means_nothing_to_grade(tmp_path):
    from scripts import grade_issued_picks as gip
    dbmod.connect(str(tmp_path / "e.db")).close()
    rc = gip.main(["--db", str(tmp_path / "e.db"), "--box-dir", str(BOX),
                   "--box-captured-at", BOX_CAPTURED, "--out", str(tmp_path / "o")])
    assert rc == 3 and not (tmp_path / "o").exists()
    assert gip.main(["--db", str(tmp_path / "e.db"), "--box-dir", str(BOX), "--box-captured-at",
                     "2026-09-22 01:20", "--out", str(tmp_path / "o")]) == 4


# ----------------------------------------------------------- game identity --
def _variant(tmp_path, name, mutate):
    gp = json.loads(BOX_FILE.read_text())
    mutate(gp["gamepackageJSON"])
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps(gp))
    return p


def test_box_binds_only_to_its_exact_canonical_game():
    b = _boxes()
    assert list(b["games"]) == [GAME] and b["rejected"] == []
    g = b["games"][GAME]
    assert (g["season"], g["week"], g["season_type"], g["espn_event"]) == (2026, 2, 2, "401872944")
    assert ig._game_for({"game_id": "2025_02_WAS_DAL"}, b["games"]) is None      # wrong year, same matchup
    assert ig._game_for({"game_id": "2026_03_WAS_DAL"}, b["games"]) is None      # wrong week
    assert ig._game_for({"game_id": "2026_02_DAL_WAS"}, b["games"]) is None      # home/away swapped
    assert ig._game_for({"game_id": GAME, "season": 2026, "week": 3}, b["games"]) is None
    assert ig._game_for({"game_id": "2025_03_ATL_GB"},
                        {"2026_03_ATL_GB": {"season": 2026, "week": 3}}) is None  # parent's counterexample


@pytest.mark.parametrize("name,mutate,reason", [
    ("post", lambda g: g["header"]["season"].update(type=3), "not the regular season"),
    ("noweek", lambda g: g["header"].pop("week"), "lacks season year / week"),
    ("live", lambda g: g["header"]["competitions"][0]["status"]["type"].update(
        name="STATUS_IN_PROGRESS", completed=False), "not an official final"),
])
def test_unusable_box_files_are_rejected(tmp_path, name, mutate, reason):
    b = _boxes(_variant(tmp_path, name, mutate))
    assert b["games"] == {} and reason in b["rejected"][0]["reason"]


def test_duplicate_box_files_for_one_game_reject_both(tmp_path):
    other = _variant(tmp_path, "dup", lambda g: g["boxscore"]["players"][0]["statistics"][0]["athletes"][0]
                     .update(stats=["27/32", "280", "8.8", "4", "0", "2-13", "89.8", "143.8"]))
    b = _boxes(BOX_FILE, other)
    assert b["games"] == {} and len(b["rejected"]) == 2
    assert all("duplicate box files" in r["reason"] for r in b["rejected"])


def test_capture_before_kickoff_is_not_a_final_box():
    b = ig.load_boxes([str(BOX_FILE)], "2026-09-20T20:00:00Z")
    assert b["games"] == {} and "capture clock" in b["rejected"][0]["reason"]


# ------------------------------------------------ settlement vs real box --
def test_real_box_pass_attempts_exclude_sacks_row_level():
    g = _boxes()["games"][GAME]
    dak, how = ig.identify({"player_name": "Dak Prescott"}, g)
    assert how == "full_name_unique_in_game"
    assert ig.box_actual(dak, "pass_attempts") == (31.0, None)        # official 26/31, sacked twice
    assert ig.box_actual(dak, "passing_yards") == (279.0, None)
    pbp = pd.read_csv(BOX / "pbp_2026wk2_DAL_WAS_pass_plays.csv")
    off = st.official_pass_attempts(pbp).set_index("player_id")[st.OFFICIAL_PASS_ATTEMPTS_COL]
    incl = pbp.groupby("passer_player_id")["pass_attempt"].sum()
    ids = pbp.drop_duplicates("passer_player_id").set_index("passer_player_name")["passer_player_id"]
    for short, full in (("D.Prescott", "Dak Prescott"), ("J.Daniels", "Jayden Daniels"),
                        ("M.Mariota", "Marcus Mariota")):
        ath, _ = ig.identify({"player_name": full}, g)
        assert off[ids[short]] == ig.box_actual(ath, "pass_attempts")[0], short
    assert incl[ids["D.Prescott"]] == 33                               # the sack-inclusive count is wrong


def test_box_absence_missing_key_and_non_numeric_are_unresolved_not_zero():
    g = _boxes()["games"][GAME]
    dak, _ = ig.identify({"player_name": "Dak Prescott"}, g)
    assert ig.box_actual(dak, "receptions")[0] is None                # not under receiving: zero unverified
    broken = copy.deepcopy(dak)
    broken["cats"]["passing"].pop("passingYards")
    broken["cats"]["passing"]["completions/passingAttempts"] = "--"
    assert ig.box_actual(broken, "passing_yards")[0] is None
    assert ig.box_actual(broken, "pass_attempts")[0] is None


def test_invalid_side_and_probability_fail_closed():
    assert st.settle("pass_attempts", "yes", 32.5, 31.0, True).settlement == st.UNRESOLVED
    assert st.settle("anytime_td", "under", 0.5, 1.0, True).settlement == st.UNRESOLVED
    r = {"record_id": "r", "pick_key": "k", "season": 2026, "week": 2, "game_id": GAME,
         "player_name": "Dak Prescott", "market": "pass_attempts", "side": "under", "line": 32.5,
         "model_p_side": 1.4, "mean": 31.0, "sd": 5.0}
    g = ig.grade_record(r, _boxes()["games"])
    assert g["settlement"] == st.WIN and "brier" not in g and g["probability_status"].startswith("invalid")


def test_official_attempts_cover_only_seen_passers_and_exclude_two_point_tries():
    base = {"season": 2026, "week": 2, "pass_attempt": 1, "down": 1.0, "sack": 0.0}
    pbp = pd.DataFrame([{**base, "passer_player_id": "A"}, {**base, "passer_player_id": "A", "down": None},
                        {**base, "passer_player_id": "S", "sack": 1.0}])
    off = st.official_pass_attempts(pbp).set_index("player_id")[st.OFFICIAL_PASS_ATTEMPTS_COL]
    assert off.to_dict() == {"A": 1.0, "S": 0.0}                      # 2-pt try dropped; sacked-only = verified 0
    both = pbp.assign(two_point_attempt=[0, 0, 0])                    # explicit column wins over `down`
    assert st.official_pass_attempts(both).set_index("player_id").loc["A"].iloc[-1] == 2.0
    with pytest.raises(ValueError, match="two-point"):
        st.official_pass_attempts(pbp.drop(columns=["down"]))
    pw = pd.DataFrame([{"season": 2026, "week": 2, "player_id": p, "pass_attempts": 1.0} for p in ("A", "S", "M")])
    out = st.with_official_pass_attempts(pw, pbp).set_index("player_id")[st.OFFICIAL_PASS_ATTEMPTS_COL]
    assert out["A"] == 1.0 and out["S"] == 0.0 and pd.isna(out["M"])   # uncovered -> missing, never 0


def test_grade_week_settles_pass_attempts_on_official_count_only(tmp_path):
    pbp = pd.read_csv(BOX / "pbp_2026wk2_DAL_WAS_pass_plays.csv")
    dak = pbp[pbp.passer_player_name == "D.Prescott"].passer_player_id.iloc[0]
    pw = pd.DataFrame([{"season": 2026, "week": 2, "player_id": p, "pass_attempts": 33.0,
                        "rush_tds": 0.0, "rec_tds": 0.0} for p in (dak, "NOT_IN_PBP")])
    leans = pd.DataFrame([{**_lean(player_id=p), "proj_components": None} for p in (dak, "NOT_IN_PBP")])
    conn = dbmod.connect(str(tmp_path / "g.db"))
    out = pl.grade_week(conn, 2026, 2, st.with_official_pass_attempts(pw, pbp), leans=leans).set_index("player_id")
    assert out.loc[dak, "actual"] == 31.0 and out.loc[dak, "settlement"] == st.WIN
    assert out.loc["NOT_IN_PBP", "settlement"] == st.UNRESOLVED
    assert pl.grade_week(conn, 2026, 2, pw, leans=leans).iloc[0]["settlement"] == st.UNRESOLVED


# --------------------------------------------------------------- grading --
def test_slate_tag_uses_new_york_time_across_dst():
    assert ig.slate_tag("2026-09-25T00:15Z") == "thursday"          # EDT
    assert ig.slate_tag("2026-12-04T04:30Z") == "thursday"          # EST: 23:30 Thu (fixed UTC-4 said Friday)


def test_clv_requires_a_pre_kick_close_at_the_issued_line():
    r = {"game_id": GAME, "player_id": DAK, "market": "pass_attempts", "side": "under", "line": 32.5,
         "quote_price": 1.87}
    key = (GAME, DAK, "pass_attempts", "under")
    ok = ig.clv_row(r, {key: {"close_ts": Z("20:00"), "close_point": 32.5, "close_prob": 0.56}}, "2026-09-20T20:25Z")
    assert ok["clv_status"] == "valid" and ok["clv_prob_vs_entry_breakeven"] == pytest.approx(0.56 - 1 / 1.87)
    assert ig.clv_row(r, {key: {"close_ts": Z("20:30"), "close_point": 32.5, "close_prob": 0.5}},
                      "2026-09-20T20:25Z")["clv_status"].startswith("invalid")
    assert ig.clv_row(r, {key: {"close_ts": Z("20:00"), "close_point": 31.5, "close_prob": 0.5}},
                      "2026-09-20T20:25Z")["clv_status"].startswith("invalid")


def test_single_game_makes_no_claims_and_stat_corrections_sit_beside_originals(tmp_path, actionable):
    conn = _db(tmp_path)
    _, cards = _gen(conn, "11:00")
    _publish(tmp_path, conn, cards, "11:05", "11:06")
    recs = il.load(conn)
    prior = [{"record_id": recs[0]["record_id"], "actual": 30.0, "actuals_captured_at": "2026-09-21T01:00:00Z"}]
    res = ig.grade(recs, _boxes(), prior_rows=prior)
    grp = res["sections"]["recommendations_given"]["groups"]["primary|all_slates"]
    assert grp["n_games"] == 1 and grp["claims"].startswith("none")
    assert res["stat_corrections"] == [{"record_id": recs[0]["record_id"], "prior_actual": 30.0,
                                        "prior_captured_at": "2026-09-21T01:00:00Z", "actual": 31.0,
                                        "captured_at": BOX_CAPTURED}]
    assert res["refit"].startswith("none") and len(res["book_rules_unverified"]) >= 3


def test_script_replays_saved_state_deterministically_db_and_export_agree(tmp_path, actionable):
    from scripts import grade_issued_picks as gip
    conn = _db(tmp_path)
    _, cards = _gen(conn, "11:00")
    _publish(tmp_path, conn, cards, "11:05", "11:06")
    (tmp_path / "ledger.json").write_text(il.export(il.load(conn, 2026, 2)))
    assert json.loads((tmp_path / "ledger.json").read_text())["records"] == il.load(conn, 2026, 2)
    conn.close()
    args = ["--box-dir", str(BOX), "--box-captured-at", BOX_CAPTURED, "--season", "2026", "--week", "2"]
    for i, src in enumerate((["--db", str(tmp_path / "s.db")], ["--db", str(tmp_path / "s.db")],
                             ["--export", str(tmp_path / "ledger.json")])):
        assert gip.main([*src, *args, "--out", str(tmp_path / f"o{i}")]) == 0
    outs = [(tmp_path / f"o{i}" / "issued_grades.json").read_text() for i in range(3)]
    assert outs[0] == outs[1] == outs[2]
    rows = list(csv.DictReader(open(tmp_path / "o0" / "issued_grades_rows.csv")))
    given = [r for r in rows if r["section"] == "recommendations_given"]
    assert len(given) == 1 and given[0]["settlement"] == "win" and given[0]["actual"] == "31.0"


def test_write_week_cards_records_generated_stage_and_exports(tmp_path):
    conn = _db(tmp_path)
    payload = pc.write_week_cards(conn, 2026, 2, out_dir=str(tmp_path / "rep"))
    assert payload["ledger_written"] == 1
    assert (tmp_path / "rep" / "issued_picks_2026_wk2.json").read_text() == il.export(il.load(conn, 2026, 2))
    assert [e["stage"] for e in il.load(conn)[0]["events"]] == ["generated"]


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
    assert conn.execute("SELECT COUNT(*) FROM issued_picks").fetchone()[0] == 0     # nothing backfilled
    assert conn.execute("SELECT COUNT(*) FROM issued_pick_events").fetchone()[0] == 0
