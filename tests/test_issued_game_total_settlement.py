"""Real-data acceptance: the final 2026 week-4 Monday card (ATL at NO, ESPN event 401872979).

The fixture is the ESPN final summary captured after the game (2026-10-06T03:41:50Z), gzip-frozen
byte for byte; its sha256 is checked against the capture's own source receipt. The delivered
selections were captured after the game from the delivered card, so every record here is a
``retrospective_import`` -- never a pregame automation receipt.

* Shough over 258.5 passing yards, FanDuel -113: 286 yards -> WIN
* game under 47.5, DraftKings -102: 45 + 24 = 69 points -> LOSS
* Juwan Johnson over 46.5 receiving yards: an explicit PASS (59 yards) -> never in a W/L record

The total is a game-level market. Before this fix the grader had no definition for it and left
the delivered under ``unresolved`` forever: a losing issued call that never reached the record.
"""

import gzip
import hashlib
import json
import sqlite3
from pathlib import Path

from nflvalue import db as dbmod
from nflvalue import issued_grading as ig
from nflvalue import issued_ledger as il

FIX = Path(__file__).parent / "fixtures" / "mnf_2026_w4_atl_at_no"
CAPTURED = "2026-10-06T03:41:50Z"
GAME = "2026_04_ATL_NO"
KICK = "2026-10-06T00:15:00Z"


def _box_file(tmp_path):
    raw = gzip.decompress((FIX / "espn-summary.json.gz").read_bytes())
    receipt = json.loads((FIX / "source-receipt.json").read_text())
    assert hashlib.sha256(raw).hexdigest() == receipt["sha256"], "fixture is not the captured final box"
    p = tmp_path / "401872979.json"
    p.write_bytes(raw)
    return p


def _card(player_id, player, market, side, line, book, american, decimal, p_side, mean, status="research"):
    return {"game_id": GAME, "player_id": player_id, "player": player, "market": market, "side": side,
            "line": line, "status": status, "clock": "chat", "run_as_of": "2026-10-05T22:30:00Z",
            "quote": {"book": book, "price_american": american, "price_decimal": decimal,
                      "captured_at": "2026-10-05T22:20:00Z"},
            "model_p_side": p_side, "mean": mean, "sd": None, "breakeven": round(1 / decimal, 4),
            "rationale": "delivered final card (research lean; not model-approved)", "countercase": "",
            "invalidation": [], "status_reasons": ["analyst-issued research lean"],
            "provenance": {"run_id": "unknown: chat card", "code_sha": "unknown", "forecast_version": "unknown",
                           "selection_source": "analyst"}}


SHOUGH = _card("00-0039152", "Tyler Shough", "passing_yards", "over", 258.5, "fanduel", -113, 1 + 100 / 113,
               0.5749455726414532, 276.869)
TOTAL = _card(None, "Game total", "game_total", "under", 47.5, "draftkings", -102, 1 + 100 / 102, 0.6476, 42.7)
JOHNSON = _card("00-0035656", "Juwan Johnson", "receiving_yards", "over", 46.5, "fanduel", -113, 1 + 100 / 113,
                0.503233017477728, 54.687, status="pass")


def _ledger(tmp_path):
    conn = dbmod.connect(str(tmp_path / "ledger.db"))
    for card, text in ((SHOUGH, "Shough over 258.5 passing yards, FanDuel -113"),
                       (TOTAL, "Game under 47.5, DraftKings -102")):
        il.record_delivered(conn, 2026, 4, card, text, message_id="final-card-2026-10-05",
                            delivered_at="2026-10-05T23:00:00Z", channel="chat", kickoff=KICK,
                            retrospective=True, recorded_at="2026-10-06T04:00:00Z")
    il.record_cards(conn, 2026, 4, [JOHNSON], surface="delivered_card_pass", recorded_at="2026-10-06T04:00:00Z")
    return conn


def test_real_final_box_binds_to_the_canonical_game_with_both_scores(tmp_path):
    boxes = ig.load_boxes([str(_box_file(tmp_path))], CAPTURED)
    assert boxes["rejected"] == []
    g = boxes["games"][GAME]
    assert (g["away"], g["home"], g["espn_event"], g["final"]) == ("ATL", "NO", "401872979", True)
    assert (g["away_score"], g["home_score"]) == (45.0, 24.0)


def test_delivered_game_total_under_is_graded_a_loss_not_left_unresolved(tmp_path):
    conn = _ledger(tmp_path)
    res = ig.grade(il.load(conn), ig.load_boxes([str(_box_file(tmp_path))], CAPTURED))
    rows = {r["market"]: r for r in res["sections"]["delivered_historical_import"]["rows"]}
    total = rows["game_total"]
    assert (total["settlement"], total["hit"], total["actual"]) == ("loss", 0, 69.0)
    assert total["identity"] == "game_level_market"
    shough = rows["passing_yards"]
    assert (shough["settlement"], shough["hit"], shough["actual"]) == ("win", 1, 286.0)
    # the delivered selections are a postgame capture: never prospective, never "given before kickoff"
    assert res["counts"]["recommendations_given"] == 0 and res["counts"]["retrospective"] == 0
    # delivered before kickoff per the source, imported after it: its own section, recommendation kept
    assert res["counts"]["delivered_historical_import"] == 2
    assert {r["delivery_evidence_kind"] for r in rows.values()} == {"retrospective_import"}
    # research leans delivered as picks: the policy would have blocked them, so they are graded
    # as analyst-override policy violations -- kept in the record, never counted as approved
    assert {(r["policy_class"], r["tier"]) for r in rows.values()} == {("violation", "analyst_override")}
    # the explicit PASS is a recorded card, never part of any W/L section
    assert res["counts"]["not_a_pick_records"] == 1
    assert all(r["market"] != "receiving_yards" for s in res["sections"].values() for r in s["rows"])


def test_pass_actual_is_available_but_never_counted(tmp_path):
    g = ig.load_boxes([str(_box_file(tmp_path))], CAPTURED)["games"][GAME]
    rec = il.load(_ledger(tmp_path))
    johnson = next(r for r in rec if r["market"] == "receiving_yards")
    ath, how = ig.identify(johnson, g)
    assert how == "full_name_unique_in_game"
    assert ig.box_actual(ath, "receiving_yards") == (59.0, None)
    assert johnson["pick_class"] == "not_a_pick"


def test_game_total_without_a_final_score_stays_unresolved(tmp_path):
    raw = json.loads(gzip.decompress((FIX / "espn-summary.json.gz").read_bytes()))
    for t in raw["header"]["competitions"][0]["competitors"]:
        t.pop("score", None)
    p = tmp_path / "noscore.json"
    p.write_text(json.dumps(raw))
    conn = _ledger(tmp_path)
    res = ig.grade(il.load(conn), ig.load_boxes([str(p)], CAPTURED))
    total = next(r for r in res["sections"]["delivered_historical_import"]["rows"] if r["market"] == "game_total")
    assert total["settlement"] == "unresolved" and total["actual"] is None
