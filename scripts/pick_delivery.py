"""Assistant pick delivery: prepare the exact text, record real delivery evidence, grade it.

    # 1. prepare: exact user-facing text + card manifest from a run's cards. Writes files only;
    #    the result is GENERATED, NOT DELIVERED.
    python scripts/pick_delivery.py prepare --cards reports/pick_cards_latest.json \
        --season 2026 --week 3 --kickoffs kickoffs.json --out <dir>
    # 2. record: after the text was actually sent, with evidence of that message
    python scripts/pick_delivery.py record --db data/nfl_props.db --manifest <dir>/manifest.json \
        --item <item_id> --platform-message-id <id> --channel telegram --delivered-at 2026-09-24T22:05:00Z
    python scripts/pick_delivery.py record --db data/nfl_props.db --manifest <dir>/manifest.json \
        --item <item_id> --hermes-message <session_id>:<message_row> --hermes-state-db <profile>/state.db
    # 3. grade: the ledger against saved official finals (read-only; refits nothing)
    python scripts/pick_delivery.py grade --db data/nfl_props.db --season 2026 --week 3 \
        --box-dir <dir of ESPN final box JSON> --box-captured-at 2026-09-25T04:30:00Z --out <dir>

``prepare`` reads the cards payload (``pick_cards.write_week_cards`` output or a site's
``api/hub.json``) and the official kickoffs (JSON ``{game_id: zoned ISO kickoff}``).
Only an ``actionable`` card is a recommendation and only a ``watch`` card is a watch
item. Each also needs one executable quote that is fresh at preparation time and
captured before kickoff, plus a kickoff that is still ahead. Everything else is
withheld as "no recommendation" with its reason. That includes research cards: no
live price, or availability/inactives not established. Nothing is filled in.

``record`` writes the ledger's ``delivered`` stage (``issued_ledger.record_delivered``)
for one manifest item. It needs one of two kinds of evidence:

* a platform message id, supplied as given, with its channel and clock, or
* a local Hermes message row, read read-only from the profile's ``state.db``. The row
  must be an assistant message whose content contains the item's exact text; its own
  timestamp is the delivery clock. The id is recorded as ``hermes-local:...``, never
  as a platform id (the store's ``platform_message_id`` is recorded only when present).

There is no automatic chat capture. Without evidence nothing is recorded. A delivery
clock at or after kickoff is refused unless ``--retrospective`` is given, and the
grader then keeps that record out of the pregame section.

``grade`` runs ``scripts/grade_issued_picks.py`` and writes ``grade_receipt.json``: the
argv, the input sha256s, the counts and the rows still pending settlement. UNRESOLVED
rows are pending, never an automatic void. Sportsbook rules are listed as unverified.
Nothing here refits a model or writes model adjustments.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import json
import os
import sqlite3
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from nflvalue import issued_ledger as il  # noqa: E402
from nflvalue import settlement as st  # noqa: E402
from nflvalue.pick_cards import STALE_QUOTE_HOURS  # noqa: E402

SCHEMA = "fablesfable.pick_delivery_manifest.v1"
CLASS_OF = {"actionable": "recommendation", "watch": "watch"}
GENERATED_ONLY = "generated_only_not_delivered"


def _ts(s):
    if not s:
        return None
    try:
        t = dt.datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else None


def _iso(t: dt.datetime) -> str:
    return t.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _sha(b) -> str:
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode()).hexdigest()


# ------------------------------------------------------------------ prepare --
def withheld_reason(card, kickoff, now):
    """None when the card may be given as its class; otherwise why it is no recommendation."""
    status = card.get("status")
    if status not in CLASS_OF:
        why = "; ".join(card.get("status_reasons") or []) or "no reason recorded"
        return f"card status {status!r} is not a pick ({why})"
    kick = _ts(kickoff)
    if kick is None:
        return "no official zoned kickoff supplied for this game: cannot confirm pregame"
    if now >= kick:
        return "kickoff has passed: a pick prepared now would be retrospective"
    q = card.get("quote") or {}
    price = q.get("price_decimal")
    if not q.get("book") or not isinstance(price, (int, float)) or price <= 1.0:
        return "no live executable price on the card"
    cap = _ts(q.get("captured_at"))
    if cap is None:
        return "quote capture clock missing or not zoned"
    if cap >= kick:
        return "quote captured at or after kickoff"
    age_h = (now - cap).total_seconds() / 3600
    if age_h < 0:
        return "quote capture clock is in the future"
    if age_h > STALE_QUOTE_HOURS:
        return f"quote is {age_h:.1f} h old at preparation (> {STALE_QUOTE_HOURS:.0f} h): price not live"
    for k in ("game_id", "player_id", "player", "market", "side", "line", "run_as_of"):
        if card.get(k) in (None, ""):
            return f"card lacks {k}"
    decided = _ts(card.get("run_as_of"))
    if decided is None or decided > now:
        return "decision clock missing, unzoned or in the future"
    return None


def pick_text(card, pick_class, kickoff) -> str:
    q = card["quote"]
    p = card.get("model_p_side")
    prob = f"{p:.1%}" if isinstance(p, (int, float)) and 0 <= p <= 1 else "n/a"
    be = card.get("breakeven")
    head = "PICK" if pick_class == "recommendation" else "WATCH ONLY (not a recommendation)"
    return (f"{head}: {card['player']} {card['market']} {str(card['side']).upper()} {card['line']:g} "
            f"at {q['book']} {q.get('price_american') or ''} ({q['price_decimal']}), quote captured "
            f"{q['captured_at']}. Game {card['game_id']}, kickoff {kickoff}. Model P({card['side']}) {prob}"
            f"{f', breakeven {be:.1%}' if isinstance(be, (int, float)) else ''}; model probability is not "
            f"validated at offered lines. Do not act if the player is ruled out or the line/price has moved; "
            f"sportsbook settlement rules (inactive, stat corrections) are not verified.")


def prepare(cards_path, season, week, kickoffs, now):
    raw = open(cards_path, "rb").read()
    payload = json.loads(raw)
    if (payload.get("season"), payload.get("week")) != (season, week):
        raise ValueError(f"cards are for season/week {payload.get('season')}/{payload.get('week')}, "
                         f"not {season}/{week}")
    items, withheld = [], []
    for card in payload.get("cards") or []:
        kickoff = kickoffs.get(card.get("game_id"))
        why = withheld_reason(card, kickoff, now)
        ident = {k: card.get(k) for k in ("game_id", "player_id", "player", "market", "side", "line", "status")}
        if why:
            withheld.append({**ident, "classification": "no_recommendation", "reason": why})
            continue
        pick_class = CLASS_OF[card["status"]]
        text = pick_text(card, pick_class, kickoff)
        body = {"season": season, "week": week, "pick_class": pick_class, "kickoff": kickoff,
                "text": text, "card": card}
        items.append({"item_id": _sha(_canon(body))[:16], **body, "text_sha256": _sha(text)})
    items.sort(key=lambda i: (i["pick_class"] != "recommendation", i["kickoff"], i["card"]["game_id"],
                              i["card"]["player"], i["card"]["market"]))
    manifest = {"schema": SCHEMA, "status": GENERATED_ONLY,
                "status_note": ("prepared text only; nothing here proves the user received it. Record a "
                                "delivery with `record` and real message evidence."),
                "season": season, "week": week, "prepared_at": _iso(now),
                "source_cards": os.path.abspath(cards_path), "source_cards_sha256": _sha(raw),
                "source_label": payload.get("label"), "kickoffs": kickoffs,
                "counts": {"recommendation": sum(i["pick_class"] == "recommendation" for i in items),
                           "watch": sum(i["pick_class"] == "watch" for i in items),
                           "no_recommendation": len(withheld)},
                "items": items, "withheld": withheld,
                "book_rules_unverified": list(st.BOOK_RULES_UNVERIFIED)}
    return manifest


def message_text(manifest) -> str:
    if not manifest["items"]:
        return (f"No pick for {manifest['season']} week {manifest['week']} as of {manifest['prepared_at']}: "
                f"no card has a live executable price, a pregame kickoff and a pick status "
                f"({manifest['counts']['no_recommendation']} withheld).\n")
    return "\n\n".join(i["text"] for i in manifest["items"]) + "\n"


# ------------------------------------------------------------------- record --
def load_item(manifest_path, item_id):
    m = json.load(open(manifest_path))
    if m.get("schema") != SCHEMA:
        raise ValueError("not a pick-delivery manifest")
    hits = [i for i in m["items"] if i["item_id"] == item_id]
    if len(hits) != 1:
        raise ValueError(f"item {item_id!r} is not a deliverable item of this manifest")
    it = hits[0]
    if _sha(it["text"]) != it["text_sha256"]:
        raise ValueError("item text does not match its recorded sha256")
    if it["pick_class"] not in il.ISSUED_CLASSES:
        raise ValueError(f"pick_class {it['pick_class']!r} cannot be delivered")
    return m, it


def hermes_evidence(state_db, ref, text):
    """Read ONE referenced message row read-only; the exact text must be in its content."""
    try:
        session_id, row = ref.rsplit(":", 1)
        row = int(row)
    except ValueError:
        raise ValueError("--hermes-message must be <session_id>:<message row id>")
    conn = sqlite3.connect(f"file:{os.path.abspath(state_db)}?mode=ro", uri=True)
    try:
        got = conn.execute("SELECT m.role, m.content, m.timestamp, m.platform_message_id, s.source "
                           "FROM messages m JOIN sessions s ON s.id = m.session_id "
                           "WHERE m.session_id = ? AND m.id = ?", (session_id, row)).fetchone()
    finally:
        conn.close()
    if got is None:
        raise ValueError("referenced message row not found in the state db")
    role, content, ts, platform_id, source = got
    if role != "assistant":
        raise ValueError(f"referenced message is a {role!r} message, not the assistant's")
    if not content or text not in content:
        raise ValueError("the referenced message does not contain the item's exact text")
    delivered = _iso(dt.datetime.fromtimestamp(float(ts), dt.timezone.utc))
    mid = f"hermes-local:{source}:{session_id}:{row}"
    if platform_id:
        mid += f";platform_message_id={platform_id}"
    return {"message_id": mid, "channel": f"hermes-{source}", "delivered_at": delivered,
            "id_kind": "platform+local_row" if platform_id else "local_row_only (no platform id stored)",
            "message_content_sha256": _sha(content)}


def record(db_path, manifest_path, item_id, evidence, now, retrospective=False):
    from nflvalue import db as dbmod
    m, it = load_item(manifest_path, item_id)
    when = _ts(evidence["delivered_at"])
    if when is None:
        raise ValueError("delivery clock missing or not zoned")
    if when > now:
        raise ValueError("delivery clock is in the future")
    if when < _ts(m["prepared_at"]):
        raise ValueError("delivery clock precedes the text's preparation")
    after_kick = when >= _ts(it["kickoff"])
    # Freshness at preparation does not extend the card's quote: a pregame delivery after the
    # quote aged past STALE_QUOTE_HOURS was not given at a live price (no --retrospective escape).
    cap = _ts((it["card"].get("quote") or {}).get("captured_at"))
    if cap is None:
        raise ValueError("card quote capture clock missing or not zoned")
    if not after_kick and (when - cap).total_seconds() / 3600 > STALE_QUOTE_HOURS:
        raise ValueError(f"card quote expired before delivery (> {STALE_QUOTE_HOURS:.0f} h after capture): "
                         "prepare again from a fresh quote")
    if after_kick and not retrospective:
        raise ValueError("delivered at/after kickoff: pass --retrospective to record it as retrospective")
    conn = dbmod.connect(os.path.abspath(db_path))
    try:
        n0 = conn.execute("SELECT COUNT(*) FROM issued_pick_events").fetchone()[0]
        rec = il.record_delivered(conn, m["season"], m["week"], it["card"], it["text"], evidence["message_id"],
                                  _iso(when), evidence["channel"], pick_class=it["pick_class"], tier="primary",
                                  recorded_at=_iso(now))
        new = conn.execute("SELECT COUNT(*) FROM issued_pick_events").fetchone()[0] > n0
        revision = conn.execute("SELECT revision, supersedes FROM issued_picks WHERE record_id=?",
                                (rec["record_id"],)).fetchone()
    finally:
        conn.close()
    return {"item_id": item_id, "record_id": rec["record_id"], "pick_class": it["pick_class"],
            "revision": revision[0], "supersedes": revision[1], "new_event": new,
            "retrospective": after_kick, **evidence}


# -------------------------------------------------------------------- grade --
def grade(argv_out, db, season, week, box_dir, captured_at, out, now, id_map=None):
    from scripts import grade_issued_picks as gip
    cap = _ts(captured_at)
    if cap is None:
        raise ValueError("--box-captured-at must be a zoned ISO clock")
    if cap > now:
        raise ValueError("--box-captured-at is in the future: boxes cannot have been captured yet")
    boxes = sorted(glob.glob(os.path.join(box_dir, "*.json")))
    if not boxes:
        raise ValueError("no saved box files: nothing is graded without captured official finals")
    args = ["--db", os.path.abspath(db), "--season", str(season), "--week", str(week), "--box-dir", box_dir,
            "--box-captured-at", captured_at, "--out", out] + (["--id-map", id_map] if id_map else [])
    rc = gip.main(args)
    receipt = {"argv": argv_out, "grader_args": args, "grader_exit": rc, "graded_at": _iso(now),
               "db_sha256": _sha(open(db, "rb").read()),
               "box_files": {os.path.basename(p): _sha(open(p, "rb").read()) for p in boxes},
               "box_captured_at": captured_at, "refit": "none: grading never fits or adjusts a model"}
    res_path = os.path.join(out, "issued_grades.json")
    if rc == 0:
        res = json.load(open(res_path))
        receipt["counts"] = res["counts"]
        receipt["rejected_boxes"] = res["boxes"]["rejected"]
        pending = {}
        for name, sec in sorted(res["sections"].items()):
            if name == "latest_pre_kick_snapshot":          # a separate analysis, not a given pick
                continue
            for r in sec["rows"]:
                if r["settlement"] == st.UNRESOLVED:
                    pending.setdefault(r["record_id"], {
                        "record_id": r["record_id"], "player": r["player_name"], "market": r["market"],
                        "side": r["side"], "line": r["line"], "detail": r["detail"], "sections": [],
                        "status": "pending: unresolved, no automatic void; sportsbook settlement rule unverified"
                    })["sections"].append(name)
        receipt["pending_settlement"] = sorted(pending.values(), key=lambda p: p["record_id"])
        receipt["book_rules_unverified"] = res["book_rules_unverified"]
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "grade_receipt.json"), "w") as f:
        json.dump(receipt, f, indent=1, sort_keys=True)
        f.write("\n")
    return rc, receipt


# --------------------------------------------------------------------- main --
def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser()
    ap.add_argument("--now", help="zoned clock to act at (tests/rehearsal); default: wall clock")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--cards", required=True)
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--week", type=int, required=True)
    p.add_argument("--kickoffs", required=True, help="JSON {game_id: zoned ISO kickoff} from an official source")
    p.add_argument("--out", required=True)
    r = sub.add_parser("record")
    r.add_argument("--db", required=True)
    r.add_argument("--manifest", required=True)
    r.add_argument("--item", required=True)
    r.add_argument("--platform-message-id")
    r.add_argument("--channel")
    r.add_argument("--delivered-at")
    r.add_argument("--hermes-message", help="<session_id>:<message row id> in the Hermes state db")
    r.add_argument("--hermes-state-db")
    r.add_argument("--retrospective", action="store_true")
    g = sub.add_parser("grade")
    g.add_argument("--db", required=True)
    g.add_argument("--season", type=int, required=True)
    g.add_argument("--week", type=int, required=True)
    g.add_argument("--box-dir", required=True)
    g.add_argument("--box-captured-at", required=True)
    g.add_argument("--id-map")
    g.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    now = _ts(a.now) if a.now else dt.datetime.now(dt.timezone.utc)
    if now is None:
        print("[delivery] --now must be a zoned ISO clock")
        return 4
    try:
        if a.cmd == "prepare":
            kick = json.load(open(a.kickoffs))
            m = prepare(a.cards, a.season, a.week, kick, now)
            os.makedirs(a.out, exist_ok=True)
            with open(os.path.join(a.out, "manifest.json"), "w") as f:
                json.dump(m, f, indent=1, sort_keys=True, default=str)
                f.write("\n")
            with open(os.path.join(a.out, "message.txt"), "w") as f:
                f.write(message_text(m))
            print(f"[delivery] {GENERATED_ONLY}: {m['counts']}; items "
                  f"{[i['item_id'] for i in m['items']]}; text in {os.path.join(a.out, 'message.txt')}")
            return 0
        if a.cmd == "record":
            if a.hermes_message:
                if a.platform_message_id or a.delivered_at or not a.hermes_state_db:
                    print("[delivery] --hermes-message needs --hermes-state-db and takes its clock from the row")
                    return 2
                _, it = load_item(a.manifest, a.item)
                ev = hermes_evidence(a.hermes_state_db, a.hermes_message, it["text"])
            elif a.platform_message_id and a.channel and a.delivered_at:
                ev = {"message_id": a.platform_message_id, "channel": a.channel, "delivered_at": a.delivered_at,
                      "id_kind": "platform (as supplied)"}
            else:
                print("[delivery] no delivery evidence: nothing recorded (the manifest stays generated-only)")
                return 2
            out = record(a.db, a.manifest, a.item, ev, now, retrospective=a.retrospective)
            print(f"[delivery] {'recorded' if out['new_event'] else 'already recorded (no-op)'}: "
                  f"{json.dumps(out, sort_keys=True)}")
            return 0
        rc, receipt = grade(argv, a.db, a.season, a.week, a.box_dir, a.box_captured_at, a.out, now, a.id_map)
        print(f"[delivery] grader exit {rc}; pending settlement rows: {len(receipt.get('pending_settlement', []))}")
        return rc
    except ValueError as exc:
        print(f"[delivery] refused: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
