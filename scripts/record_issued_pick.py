"""Record, at issuance, what the user was actually given or shown. Writes the ledger only.

    # the assistant gave a pick in chat: exact text + source message id (no automatic capture exists)
    python scripts/record_issued_pick.py --db data/nfl_props.db delivered --season 2026 --week 3 \
        --card card.json --text-file message.txt --message-id <id> --channel chat \
        --delivered-at 2026-09-24T21:05:00Z [--watch]
    # a page was published: only with the live-readback receipt website.yml wrote after deploying it
    python scripts/record_issued_pick.py --db data/nfl_props.db published \
        --hub r/hub.json --publication r/publication.json --receipt r/publication_receipt.json

    # a whole delivered card (recommendations, game totals and explicit PASSes) from a
    # public-safe input: selection lines + the private source's id and sha256, never its text
    python scripts/record_issued_pick.py --db data/nfl_props.db delivered-card --input card-input.json

``card.json`` holds the pick as given: game_id, player_id, player, market, side, line,
run_as_of (decision clock), quote {book, price_decimal, captured_at}, model_p_side,
mean, sd, provenance. Nothing is filled in from later data. Recording after kickoff is
kept as retrospective evidence, never graded as a prospective pick.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from nflvalue import db as dbmod  # noqa: E402
from nflvalue import issued_ledger as il  # noqa: E402

REQUIRED = ("game_id", "player_id", "player", "market", "side", "line", "run_as_of")
CARD_SCHEMA = "fablesfable.delivered_card.v1"
CAPTURES = ("historical_postgame", "live_pregame")
ROLES = ("recommendation", "watch", "pass")


def record_card_input(conn, doc: dict, now: str | None = None) -> list:
    """Record every selection of one delivered card from a public-safe input document.

    The ledger keeps the public selection line plus the source id and the sha256 of the private
    original; the original text is never stored. ``historical_postgame`` imports are recorded as
    retrospective (never a prospective pick, whatever the original clock says); ``live_pregame``
    is refused once kickoff has passed. Every role is recorded, losses and PASSes included, and a
    research lean stays a research lean: authorization is the delivery policy's, never implied."""
    import datetime as dt
    from nflvalue import delivery_policy as dp
    if doc.get("schema") != CARD_SCHEMA:
        raise ValueError(f"input schema must be {CARD_SCHEMA}")
    src, capture = doc.get("source") or {}, doc.get("capture")
    if capture not in CAPTURES:
        raise ValueError(f"capture must be one of {CAPTURES}")
    digest = str(src.get("content_sha256") or "")
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError("source.content_sha256 must be the 64-hex sha256 of the original message")
    issued, kick = dp._zoned(src.get("issued_at")), dp._zoned(doc.get("kickoff"))
    if issued is None or kick is None or not src.get("source_id"):
        raise ValueError("source.source_id, a zoned source.issued_at and a zoned kickoff are required")
    now_t = dp._zoned(now) if now else dt.datetime.now(dt.timezone.utc)
    if capture == "live_pregame" and not (issued < kick and now_t < kick):
        raise ValueError("live_pregame capture after kickoff: record it as historical_postgame")
    stamp = now_t.isoformat().replace("+00:00", "Z")
    out = []
    for sel in doc.get("selections") or []:
        role = sel.get("role")
        if role not in ROLES or not sel.get("text"):
            raise ValueError(f"each selection needs role in {ROLES} and its public selection text")
        price = sel.get("american")
        dec = (1 + (price / 100 if price > 0 else 100 / -price)) if isinstance(price, (int, float)) else None
        card = {"game_id": doc["game_id"], "player_id": sel.get("player_id"), "player": sel.get("player"),
                "market": sel["market"], "side": sel["side"], "line": sel["line"],
                "status": "pass" if role == "pass" else sel.get("status", "research"), "clock": "analyst_card",
                "run_as_of": src["issued_at"],
                "quote": {"book": sel.get("book"), "price_american": price, "price_decimal": dec,
                          "captured_at": sel.get("quote_captured_at")},
                "model_p_side": sel.get("model_p_side"), "mean": sel.get("mean"), "sd": sel.get("sd"),
                "breakeven": round(1 / dec, 4) if dec else None,
                "rationale": sel.get("status_text") or "analyst-issued card", "countercase": "", "invalidation": [],
                "status_reasons": [sel.get("status_text") or role],
                "provenance": {"run_id": "unknown: analyst card", "code_sha": "unknown",
                               "forecast_version": "unknown", "selection_source": "analyst",
                               "source_id": src["source_id"], "source_content_sha256": digest,
                               "source_issued_at": src["issued_at"], "capture": capture}}
        text = f"{sel['text']} [source {src['source_id']}; sha256 {digest}]"
        if role == "pass":
            recs = il.record_cards(conn, doc["season"], doc["week"], [card], surface="delivered_card_pass",
                                   recorded_at=stamp)
            out += [{"role": role, "record_id": r["record_id"], "pick_class": r["pick_class"]} for r in recs]
        else:
            rec = il.record_delivered(conn, doc["season"], doc["week"], card, text, src["source_id"],
                                      src["issued_at"], doc.get("channel") or "chat",
                                      pick_class="watch" if role == "watch" else "recommendation",
                                      kickoff=doc["kickoff"], retrospective=capture == "historical_postgame",
                                      recorded_at=stamp)
            out.append({"role": role, "record_id": rec["record_id"], "pick_class": rec["pick_class"],
                        "tier": rec["tier"], **rec["policy"]})
    conn.commit()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("delivered")
    d.add_argument("--season", type=int, required=True)
    d.add_argument("--week", type=int, required=True)
    d.add_argument("--card", required=True)
    d.add_argument("--text-file", required=True)
    d.add_argument("--message-id", required=True)
    d.add_argument("--channel", required=True)
    d.add_argument("--delivered-at", required=True)
    d.add_argument("--watch", action="store_true", help="given as a watch item, not a recommendation")
    d.add_argument("--tier", choices=il.TIERS, help="explicit tier; omitted = derived from the policy and recorded")
    d.add_argument("--exception-by", help="who authorized sending a non-approved pick")
    d.add_argument("--exception-reason", help="why; stored verbatim, never relabels the forecast as approved")
    d.add_argument("--kickoff", help="zoned ISO kickoff: lets the ledger label the evidence live vs retrospective")
    d.add_argument("--retrospective", action="store_true", help="this is an after-the-fact import of a sent pick")
    c = sub.add_parser("delivered-card", help=f"record a whole card from a {CARD_SCHEMA} input")
    c.add_argument("--input", required=True)
    c.add_argument("--now", help="ledger clock override (tests); default: wall clock")
    p = sub.add_parser("published")
    p.add_argument("--hub", required=True)
    p.add_argument("--publication", required=True)
    p.add_argument("--receipt", required=True, help="pages readback receipt (scripts/publication_receipt.py)")
    a = ap.parse_args(argv)
    conn = dbmod.connect(os.path.abspath(a.db))
    try:
        if a.cmd == "delivered-card":
            for r in record_card_input(conn, json.load(open(a.input)), now=a.now):
                print(f"[record] {r['role']}: {r['record_id']} ({r['pick_class']}"
                      + (f", {r['tier']}, policy {r['decision']}, violation={r['violation']}, "
                         f"{r['delivery_evidence_kind']}" if r["role"] != "pass" else "") + ")")
        elif a.cmd == "delivered":
            card = json.load(open(a.card))
            missing = [k for k in REQUIRED if card.get(k) in (None, "")]
            if missing:
                print(f"[record] card lacks {missing}: not recorded")
                return 2
            from nflvalue import delivery_policy as dp
            if bool(a.exception_by) != bool(a.exception_reason):
                print("[record] an exception needs both --exception-by and --exception-reason: not recorded")
                return 2
            exc = ({"by": a.exception_by, "reason": a.exception_reason, "clock": a.delivered_at}
                   if a.exception_by else None)
            pick_class = "watch" if a.watch else "recommendation"
            preview = dp.authorize(card, exc, pick_class=pick_class)
            tier = a.tier or ("primary" if preview["decision"] == "approved" else "analyst_override")
            violation = pick_class == "recommendation" and preview["decision"] == "blocked"
            print(f"[record] pick_class={pick_class} tier={tier}{'' if a.tier else ' (derived)'} "
                  f"policy_decision={preview['decision']} policy_violation={violation}; "
                  f"approval: {preview['approval_status']}")
            if violation:
                print("[record] recording a SENT pick the policy would have blocked: kept and flagged, never promoted")
            rec = il.record_delivered(conn, a.season, a.week, card, open(a.text_file).read(), a.message_id,
                                      a.delivered_at, a.channel, pick_class=pick_class, tier=a.tier,
                                      exception=exc, kickoff=a.kickoff, retrospective=a.retrospective)
            print(f"[record] delivered {rec['record_id']} ({rec['pick_class']}, {rec['tier']}, "
                  f"{rec['policy']['delivery_evidence_kind']})")
        else:
            try:
                n = il.record_publication(conn, a.hub, a.publication, json.load(open(a.receipt)))
            except ValueError as exc:
                print(f"[record] publication not verified: {exc}")
                return 5
            print(f"[record] published events written: {n}")
    except ValueError as exc:
        print(f"[record] refused: {exc}")
        return 2
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
