#!/usr/bin/env python3
"""Fail-loud top-N projection diff for one-lever model reviews.

Supports fablesfable ``weekly_props.json`` and tailstail ``fantasy_latest.json``
shapes, plus a plain list of rows. This is a churn alarm, not an accuracy test.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def _rows(payload):
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        raise ValueError("projection payload must be an object or list")
    if isinstance(payload.get("players"), list):
        return payload["players"]
    if isinstance(payload.get("predictions"), list):
        return payload["predictions"]
    if isinstance(payload.get("games"), list):
        out = []
        for game in payload["games"]:
            for lean in game.get("leans") or []:
                row = dict(lean)
                row.setdefault("game_id", game.get("game_id"))
                out.append(row)
        return out
    raise ValueError("expected players, predictions, games[].leans, or a row list")


def _score(row):
    for key in ("composite", "projection_mean", "mean", "fantasy_points"):
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value), key
    summary = row.get("summary") or {}
    value = summary.get("mean")
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value), "summary.mean"
    raise ValueError(f"row has no finite ranking score: {row.get('player_id')}")


def _identity(row):
    player = row.get("player_id") or row.get("id") or row.get("name")
    if not player:
        raise ValueError("row has no player identity")
    market = row.get("market")
    game = row.get("game_id")
    return "|".join(str(v) for v in (game, player, market) if v is not None)


def _rank(payload, top_n):
    ranked = []
    for row in _rows(payload):
        score, score_field = _score(row)
        ranked.append({"id": _identity(row), "score": score,
                       "score_field": score_field, "side": row.get("side")})
    ranked.sort(key=lambda x: (-x["score"], x["id"]))
    if not ranked:
        raise ValueError("projection payload has no rankable rows")
    return ranked[:top_n]


def compare(baseline, candidate, top_n=10):
    before = _rank(baseline, top_n)
    after = _rank(candidate, top_n)
    bmap = {row["id"]: (rank, row) for rank, row in enumerate(before, 1)}
    amap = {row["id"]: (rank, row) for rank, row in enumerate(after, 1)}
    overlap = sorted(set(bmap) & set(amap))
    denom = min(top_n, len(before), len(after))
    return {
        "schema_version": 1,
        "purpose": "sanity alarm only; not evidence of accuracy",
        "top_n": top_n,
        "baseline_count": len(before),
        "candidate_count": len(after),
        "overlap_count": len(overlap),
        "overlap_rate": round(len(overlap) / denom, 4) if denom else 0.0,
        "added": sorted(set(amap) - set(bmap)),
        "removed": sorted(set(bmap) - set(amap)),
        "side_flips": [key for key in overlap
                       if bmap[key][1]["side"] != amap[key][1]["side"]],
        "rank_changes": [
            {"id": key, "before": bmap[key][0], "after": amap[key][0],
             "delta": bmap[key][0] - amap[key][0],
             "score_delta": round(amap[key][1]["score"] - bmap[key][1]["score"], 6)}
            for key in overlap if bmap[key][0] != amap[key][0]
            or bmap[key][1]["score"] != amap[key][1]["score"]
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--min-overlap", type=float, default=0.50)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.top < 1 or not 0 <= args.min_overlap <= 1:
        parser.error("--top must be positive and --min-overlap must be in [0,1]")
    with open(args.baseline) as fh:
        baseline = json.load(fh)
    with open(args.candidate) as fh:
        candidate = json.load(fh)
    report = compare(baseline, candidate, args.top)
    report["minimum_overlap"] = args.min_overlap
    report["pass"] = report["overlap_rate"] >= args.min_overlap
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(rendered + "\n")
    print(rendered)
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    sys.exit(main())
