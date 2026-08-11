#!/usr/bin/env python3
"""Build the slim machine-readable feed the Dearing Football hub consumes.

The Pages dashboard (dashboard.html) stays the full product. This script
distills the already-produced artifacts into one small JSON
(``_site/api/hub.json``) so the hub at dearing-wedding.com can render a
native overview without downloading the multi-megabyte weekly archives
(weekly.json alone is ~1.9 MB; the hub feed stays ~100 KB).

Honesty rules carried over from the dashboard:
  * Numbers are copied verbatim from the pipeline artifacts — nothing is
    recomputed here, so the feed cannot disagree with the dashboard.
  * ``mode`` and ``disclaimer`` ride along; consumers must render them.
  * Missing inputs produce a feed with those sections null, never a guess
    (and never a failed deploy: the hub degrades to the full dashboard).

Usage: python scripts/build_hub_feed.py [--out _site/api/hub.json]
"""

from __future__ import annotations

import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BET_FIELDS = [
    "type", "home_team", "away_team", "commence_time", "market", "outcome",
    "point", "best_book", "price_american", "price_decimal",
    "p_model", "p_consensus", "ev", "edge", "stake_units", "player",
]

GAME_FIELDS = [
    "home", "away", "market_spread_home", "market_total",
    "proj_margin", "proj_total", "p_home_win", "su_pick",
    "ats_pick", "total_pick", "settled", "home_score", "away_score",
    "su_correct", "ats_result", "total_result",
]


def _load(name):
    path = os.path.join(ROOT, "data", name)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001 — a corrupt artifact must not kill Pages
        return None


def _slim(row, fields):
    return {k: row.get(k) for k in fields if row.get(k) is not None}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(ROOT, "_site", "api", "hub.json"))
    args = ap.parse_args()

    latest = _load("latest.json")
    weekly = _load("weekly.json")
    top_bets = _load("top_bets.json")

    feed = {"schema_version": 1, "project": "fablesfable"}

    if latest:
        feed["generated_at"] = latest.get("generated_at")
        feed["mode"] = latest.get("mode")
        feed["disclaimer"] = latest.get("disclaimer")
        feed["summary"] = latest.get("summary")
        feed["metrics"] = {
            k: latest.get("metrics", {}).get(k)
            for k in ("graded_total", "settled_total", "win_rate", "roi_all", "brier")
        }
        feed["value_bets"] = [
            _slim(b, BET_FIELDS) for b in (latest.get("value_bets") or [])[:10]]
        feed["value_props"] = [
            _slim(b, BET_FIELDS) for b in (latest.get("value_props") or [])[:10]]
    else:
        feed["value_bets"] = feed["value_props"] = None

    if weekly and weekly.get("weeks"):
        wk = weekly["weeks"][-1]
        feed["week"] = {
            "label": wk.get("label"),
            "season": wk.get("season"),
            "record_to_date": wk.get("record_to_date"),
            "games": [_slim(g, GAME_FIELDS) for g in (wk.get("games") or [])],
        }
    else:
        feed["week"] = None

    if top_bets and top_bets.get("weeks"):
        twk = top_bets["weeks"][-1]
        feed["top_bets"] = {
            "label": twk.get("label"),
            "meta": {k: (top_bets.get("meta") or {}).get(k)
                     for k in ("best_rule", "value_rule", "fail_closed")},
            "games": [
                {"home": g.get("home"), "away": g.get("away"),
                 "settled": g.get("settled"), "bets": g.get("bets")}
                for g in (twk.get("games") or []) if g.get("bets")
            ],
        }
    else:
        feed["top_bets"] = None

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(feed, fh, separators=(",", ":"))
    os.replace(tmp, args.out)
    print(f"[hub-feed] wrote {args.out} ({os.path.getsize(args.out)} bytes; "
          f"sections: " + ", ".join(k for k in ("value_bets", "value_props", "week", "top_bets")
                                    if feed.get(k)) + ")")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
