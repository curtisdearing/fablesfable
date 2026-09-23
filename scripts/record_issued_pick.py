"""Record, at issuance, what the user was actually given or shown. Writes the ledger only.

    # the assistant gave a pick in chat: exact text + source message id (no automatic capture exists)
    python scripts/record_issued_pick.py --db data/nfl_props.db delivered --season 2026 --week 3 \
        --card card.json --text-file message.txt --message-id <id> --channel chat \
        --delivered-at 2026-09-24T21:05:00Z [--watch]
    # a page was published: its saved hub.json verified against its publication.json
    python scripts/record_issued_pick.py --db data/nfl_props.db published \
        --hub site/api/hub.json --publication site/publication.json

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
    p = sub.add_parser("published")
    p.add_argument("--hub", required=True)
    p.add_argument("--publication", required=True)
    a = ap.parse_args(argv)
    conn = dbmod.connect(os.path.abspath(a.db))
    try:
        if a.cmd == "delivered":
            card = json.load(open(a.card))
            missing = [k for k in REQUIRED if card.get(k) in (None, "")]
            if missing:
                print(f"[record] card lacks {missing}: not recorded")
                return 2
            rec = il.record_delivered(conn, a.season, a.week, card, open(a.text_file).read(), a.message_id,
                                      a.delivered_at, a.channel, pick_class="watch" if a.watch else "recommendation")
            print(f"[record] delivered {rec['record_id']} ({rec['pick_class']})")
        else:
            try:
                n = il.record_publication(conn, a.hub, a.publication)
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
