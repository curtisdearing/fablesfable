"""Assistant pick delivery (scripts/pick_delivery.py): prepare -> record real evidence -> grade.

REHEARSAL ONLY. Cards, message ids and the Hermes state db here are synthetic software-test
inputs, not picks anyone was given and not betting or model-validation evidence. The box is
the real official ESPN final for 2026 week 2 WSH@DAL (event 401872944, kickoff 2026-09-20T20:25Z,
fetched 2026-09-22T01:20Z), shared with tests/test_issued_ledger.py."""
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nflvalue import db as dbmod  # noqa: E402
from nflvalue import issued_grading as ig  # noqa: E402
from nflvalue import issued_ledger as il  # noqa: E402
from nflvalue import pick_cards as pc  # noqa: E402
from scripts import pick_delivery as pd  # noqa: E402

BOX = Path(__file__).resolve().parent / "fixtures" / "espn_box_2026wk2"
BOX_CAPTURED = "2026-09-22T01:20:00Z"
GAME = "2026_02_WAS_DAL"
KICK = "2026-09-20T20:25:00Z"
DAK = "00-0033077"
AVAIL = json.dumps({"availability": {"status": "OK", "eligibility": "eligible",
                                     "availability_state": "not_listed"}})


def Z(hhmm, day=20):
    return f"2026-09-{day}T{hhmm}:00Z"


def _row(**kw):
    r = {"game_id": GAME, "player_id": DAK, "name": "Dak Prescott", "market": "pass_attempts",
         "side": "under", "line": 32.5, "line_source": "odds_api", "price": 1.87, "mean": 31.0, "sd": 5.0,
         "p_side": 0.6, "status": "active", "as_of": Z("10:30"), "quote_book": "draftkings",
         "quote_ts": Z("10:00"), "run_id": "gha:1-1", "code_sha": "abc", "forecast_version": "ff-football-only-v1",
         "selection_source": "football_selection_score", "stage_json": AVAIL, "_quote_verified": True,
         "_run_publish": True}
    r.update(kw)
    return r


def _cards(tmp_path, rows, name="cards.json", season=2026, week=2, at=Z("11:00")):
    """The run renders its cards at ``at`` (its own clock), as write_week_cards does."""
    cards = [pc.build_card(r, pd._ts(at)) for r in rows]
    p = tmp_path / name
    p.write_text(json.dumps({"season": season, "week": week, "label": "fresh", "cards": cards}, default=str))
    return p


def _kick(tmp_path, kicks=None):
    p = tmp_path / "kickoffs.json"
    p.write_text(json.dumps(kicks if kicks is not None else {GAME: KICK}))
    return p


def _prepare(tmp_path, rows, now=Z("11:00"), out="prep", kicks=None, name="cards.json", rendered=None):
    cards = _cards(tmp_path, rows, name, at=rendered or now)
    rc = pd.main(["--now", now, "prepare", "--cards", str(cards), "--season", "2026",
                  "--week", "2", "--kickoffs", str(_kick(tmp_path, kicks)), "--out", str(tmp_path / out)])
    assert rc == 0
    return json.loads((tmp_path / out / "manifest.json").read_text()), tmp_path / out / "manifest.json"


def _record(tmp_path, manifest, item, now=Z("11:10"), *extra):
    return pd.main(["--now", now, "record", "--db", str(tmp_path / "ledger.db"), "--manifest", str(manifest),
                    "--item", item, *extra])


def _platform(mid="plat-1", at=Z("11:05")):
    return ["--platform-message-id", mid, "--channel", "telegram", "--delivered-at", at]


def _grade(tmp_path):
    conn = sqlite3.connect(tmp_path / "ledger.db")
    try:
        return ig.grade(il.load(conn), ig.load_boxes([str(BOX / "401872944.json")], BOX_CAPTURED))
    finally:
        conn.close()


@pytest.fixture
def actionable(monkeypatch):
    """No market is validated in production; the recommendation path needs one for the test."""
    monkeypatch.setattr(pc, "VALIDATED_MARKETS", frozenset({"pass_attempts"}))


# ---------------------------------------------------------------- prepare --
def test_prepare_separates_pick_watch_and_no_recommendation(tmp_path, actionable):
    rows = [_row(),                                                             # actionable
            _row(market="passing_yards", line=250.5, mean=240.0, sd=60.0),      # watch (unvalidated market)
            _row(player_id="00-1", name="No Price", line_source="synthetic"),   # research: no offered price
            _row(player_id="00-2", name="Unknown Avail", stage_json=json.dumps(
                {"availability": {"eligibility": "degraded", "availability_state": "unknown"}})),
            _row(player_id="00-3", name="Voided", status="voided", void_reason="inactive")]
    m, _ = _prepare(tmp_path, rows)
    assert m["status"] == pd.GENERATED_ONLY and m["counts"] == {"recommendation": 1, "watch": 1,
                                                                "no_recommendation": 3}
    rec, watch = m["items"]
    assert rec["pick_class"] == "recommendation" and rec["text"].startswith("PICK: Dak Prescott pass_attempts UNDER 32.5")
    assert watch["pick_class"] == "watch" and watch["text"].startswith("WATCH ONLY (not a recommendation)")
    reasons = {w["player"]: w["reason"] for w in m["withheld"]}
    assert "no offered price" in reasons["No Price"] and "availability not established" in reasons["Unknown Avail"]
    assert "voided: inactive" in reasons["Voided"]
    msg = (tmp_path / "prep" / "message.txt").read_text()
    assert rec["text"] in msg and watch["text"] in msg and "No Price" not in msg
    assert m["book_rules_unverified"] and not (tmp_path / "ledger.db").exists()   # nothing recorded


def test_prepare_withholds_stale_price_missing_or_passed_kickoff(tmp_path, actionable):
    m, _ = _prepare(tmp_path, [_row()], now=Z("16:30"), rendered=Z("11:00"))
    assert not m["items"] and "h old at preparation" in m["withheld"][0]["reason"]
    m, _ = _prepare(tmp_path, [_row()], kicks={}, out="p2")
    assert "no official zoned kickoff" in m["withheld"][0]["reason"]
    m, _ = _prepare(tmp_path, [_row(quote_ts=Z("20:00"), as_of=Z("20:10"))], now=Z("20:30"), out="p3")
    assert "kickoff has passed" in m["withheld"][0]["reason"]
    assert (tmp_path / "p3" / "message.txt").read_text().startswith("No pick for 2026 week 2")


def test_prepare_is_deterministic_and_refuses_wrong_week(tmp_path, actionable):
    _prepare(tmp_path, [_row()], out="a")
    _prepare(tmp_path, [_row()], out="b")
    assert (tmp_path / "a" / "manifest.json").read_bytes() == (tmp_path / "b" / "manifest.json").read_bytes()
    wrong = _cards(tmp_path, [_row()], name="w.json", week=3)
    assert pd.main(["--now", Z("11:00"), "prepare", "--cards", str(wrong), "--season", "2026", "--week", "2",
                    "--kickoffs", str(_kick(tmp_path)), "--out", str(tmp_path / "w")]) == 2


# ----------------------------------------------------------------- record --
def test_no_evidence_records_nothing(tmp_path, actionable):
    m, man = _prepare(tmp_path, [_row()])
    assert _record(tmp_path, man, m["items"][0]["item_id"]) == 2
    assert _record(tmp_path, man, m["items"][0]["item_id"], Z("11:10"), "--platform-message-id", "x") == 2
    assert not (tmp_path / "ledger.db").exists()


def test_recorded_delivery_grades_exact_text_card_price_line(tmp_path, actionable, capsys):
    m, man = _prepare(tmp_path, [_row(), _row(market="passing_yards", line=250.5, mean=240.0, sd=60.0)])
    rec, watch = m["items"]
    assert _record(tmp_path, man, rec["item_id"], Z("11:10"), *_platform()) == 0
    assert _record(tmp_path, man, watch["item_id"], Z("11:10"), *_platform("plat-2")) == 0
    # the same delivery recorded again is a no-op; the same text re-sent is one record, two events
    assert _record(tmp_path, man, rec["item_id"], Z("11:10"), *_platform()) == 0
    assert "already recorded (no-op)" in capsys.readouterr().out
    assert _record(tmp_path, man, rec["item_id"], Z("11:20"), *_platform("plat-3", Z("11:15"))) == 0
    conn = sqlite3.connect(tmp_path / "ledger.db")
    recs = {r["pick_class"]: r for r in il.load(conn)}
    conn.close()
    r = recs["recommendation"]
    assert r["display_html"] == rec["text"] and r["tier"] == "primary"
    assert (r["line"], r["side"], r["quote_book"], r["quote_price"], r["quote_ts"]) == \
        (32.5, "under", "draftkings", 1.87, Z("10:00"))
    assert sorted(json.loads(e["evidence_json"])["message_id"] for e in r["events"]) == ["plat-1", "plat-3"]
    res = _grade(tmp_path)
    given = res["sections"]["recommendations_given"]["rows"]
    assert len(given) == 1 and given[0]["settlement"] == "win" and given[0]["actual"] == 31.0   # official, sacks excluded
    assert given[0]["evidence_stage"] == "delivered" and given[0]["slate_tag"] == "sunday"
    assert res["counts"]["watch_published"] == 1                    # watch is never a recommendation


def test_revised_line_is_a_new_revision_and_both_given_picks_are_kept(tmp_path, actionable):
    m1, man1 = _prepare(tmp_path, [_row()], out="v1")
    assert _record(tmp_path, man1, m1["items"][0]["item_id"], Z("11:10"), *_platform()) == 0
    m2, man2 = _prepare(tmp_path, [_row(line=33.5, quote_ts=Z("12:00"), as_of=Z("12:05"))], now=Z("12:10"),
                        out="v2", name="c2.json")
    assert _record(tmp_path, man2, m2["items"][0]["item_id"], Z("12:20"), *_platform("plat-9", Z("12:15"))) == 0
    conn = sqlite3.connect(tmp_path / "ledger.db")
    recs = il.load(conn)
    conn.close()
    assert [r["revision"] for r in recs] == [1, 2] and recs[1]["supersedes"] == recs[0]["record_id"]
    given = _grade(tmp_path)["sections"]["recommendations_given"]["rows"]
    assert [g["line"] for g in given] == [32.5, 33.5] and given[1]["revises"] == recs[0]["record_id"]
    assert given[0]["later_records"][0]["change"] == "side_or_line_changed"


def test_delivery_clock_guards(tmp_path, actionable):
    m, man = _prepare(tmp_path, [_row()])
    item = m["items"][0]["item_id"]
    assert _record(tmp_path, man, item, Z("11:10"), *_platform(at=Z("10:59"))) == 2       # before prepared
    assert _record(tmp_path, man, item, Z("11:10"), *_platform(at=Z("11:30"))) == 2       # future
    assert _record(tmp_path, man, item, Z("21:00"), *_platform(at=Z("20:30"))) == 2       # after kickoff
    # quote captured 10:00: pregame delivery at 16:30 is past the card's 6 h quote life
    assert _record(tmp_path, man, item, Z("17:00"), *_platform(at=Z("16:30"))) == 2
    assert _record(tmp_path, man, item, Z("17:00"), *_platform(at=Z("16:30")), "--retrospective") == 2
    assert _record(tmp_path, man, item, Z("21:00"), *_platform(at=Z("20:30")), "--retrospective") == 0
    res = _grade(tmp_path)
    assert res["counts"]["recommendations_given"] == 0 and res["counts"]["retrospective"] == 1


def test_tampered_manifest_text_or_unknown_item_is_refused(tmp_path, actionable):
    m, man = _prepare(tmp_path, [_row()])
    assert _record(tmp_path, man, "0" * 16, Z("11:10"), *_platform()) == 2
    m["items"][0]["text"] = m["items"][0]["text"].replace("32.5", "30.5")
    man.write_text(json.dumps(m))
    assert _record(tmp_path, man, m["items"][0]["item_id"], Z("11:10"), *_platform()) == 2


def _state_db(tmp_path, rows):
    """A Hermes state db with the columns the helper reads (synthetic rows)."""
    p = tmp_path / "state.db"
    c = sqlite3.connect(p)
    c.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, source TEXT NOT NULL, started_at REAL)")
    c.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT, "
              "content TEXT, timestamp REAL, platform_message_id TEXT)")
    c.execute("INSERT INTO sessions VALUES ('sess-a', 'desktop', 0)")
    c.executemany("INSERT INTO messages (session_id, role, content, timestamp, platform_message_id) "
                  "VALUES (?,?,?,?,?)", rows)
    c.commit()
    c.close()
    return p


def test_hermes_local_message_row_is_evidence_only_with_exact_text(tmp_path, actionable):
    m, man = _prepare(tmp_path, [_row()])
    it = m["items"][0]
    at = pd._ts(Z("11:05")).timestamp()
    db = _state_db(tmp_path, [("sess-a", "user", it["text"], at, None),
                              ("sess-a", "assistant", "Tonight: " + it["text"].replace("32.5", "31.5"), at, None),
                              ("sess-a", "assistant", "Here is tonight's pick.\n\n" + it["text"], at, None)])
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    args = ["--hermes-state-db", str(db), "--hermes-message"]
    assert _record(tmp_path, man, it["item_id"], Z("11:10"), *args, "sess-a:1") == 2       # user message
    assert _record(tmp_path, man, it["item_id"], Z("11:10"), *args, "sess-a:2") == 2       # different text
    assert _record(tmp_path, man, it["item_id"], Z("11:10"), *args, "sess-a:9") == 2       # no such row
    assert _record(tmp_path, man, it["item_id"], Z("11:10"), *args, "sess-a:3") == 0
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before                            # read-only
    conn = sqlite3.connect(tmp_path / "ledger.db")
    ev = il.load(conn)[0]["events"][0]
    conn.close()
    assert json.loads(ev["evidence_json"]) == {"message_id": "hermes-local:desktop:sess-a:3",
                                               "channel": "hermes-desktop"}
    assert ev["event_ts"] == Z("11:05")                          # the row's own clock, not a supplied one
    assert _grade(tmp_path)["counts"]["recommendations_given"] == 1


def test_hermes_platform_id_is_kept_when_the_store_has_one(tmp_path, actionable):
    m, man = _prepare(tmp_path, [_row()])
    it = m["items"][0]
    db = _state_db(tmp_path, [("sess-a", "assistant", it["text"], pd._ts(Z("11:05")).timestamp(), "tg-777")])
    ev = pd.hermes_evidence(str(db), "sess-a:1", it["text"])
    assert ev["message_id"] == "hermes-local:desktop:sess-a:1;platform_message_id=tg-777"
    assert ev["id_kind"] == "platform+local_row"


# ------------------------------------------------------------------ grade --
def test_grade_receipt_pins_inputs_and_lists_pending_settlement(tmp_path, actionable):
    rows = [_row(), _row(player_id="00-9", name="Not In Box", market="rushing_yards", line=40.5, mean=30.0,
                         sd=15.0)]
    m, man = _prepare(tmp_path, rows)
    for it, mid in zip(m["items"], ("p1", "p2")):
        assert _record(tmp_path, man, it["item_id"], Z("11:10"), *_platform(mid)) == 0
    before = hashlib.sha256((tmp_path / "ledger.db").read_bytes()).hexdigest()
    common = ["grade", "--db", str(tmp_path / "ledger.db"), "--season", "2026", "--week", "2",
              "--box-dir", str(BOX), "--out", str(tmp_path / "g")]
    assert pd.main(["--now", "2026-09-22T02:00:00Z", *common, "--box-captured-at", "2026-09-23T00:00:00Z"]) == 2
    assert pd.main(["--now", "2026-09-22T02:00:00Z", *common[:-4], "--box-dir", str(tmp_path / "none"),
                    "--out", str(tmp_path / "g"), "--box-captured-at", BOX_CAPTURED]) == 2
    assert pd.main(["--now", "2026-09-22T02:00:00Z", *common, "--box-captured-at", BOX_CAPTURED]) == 0
    rc = json.loads((tmp_path / "g" / "grade_receipt.json").read_text())
    assert rc["grader_exit"] == 0 and rc["db_sha256"] == before and rc["refit"].startswith("none")
    assert rc["box_files"]["401872944.json"] == hashlib.sha256((BOX / "401872944.json").read_bytes()).hexdigest()
    assert [(p["player"], p["sections"]) for p in rc["pending_settlement"]] == [("Not In Box", ["watch_published"])]
    assert "no automatic void" in rc["pending_settlement"][0]["status"]
    assert rc["counts"]["recommendations_given"] == 1 and rc["counts"]["watch_published"] == 1
    assert rc["book_rules_unverified"]
    assert hashlib.sha256((tmp_path / "ledger.db").read_bytes()).hexdigest() == before
    conn = dbmod.connect(str(tmp_path / "ledger.db"))
    assert conn.execute("SELECT COUNT(*) FROM model_adjustments").fetchone()[0] == 0
    conn.close()
