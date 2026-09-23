"""Live acquisition probe: slate-wide factor context + participation receipt, to a scratch dir.

Run by hand (never at import).  Writes only under ``--out``; the committed context file is
updated only when ``--publish-context`` names it explicitly.  Free endpoints, generic
user agent, no odds calls::

    python -m analysis.live_coverage_probe --season 2026 --week 3 --out <scratch> \
        [--curated data/factor_context/2026-w03.json] [--publish-context <path>]

Outputs in ``--out``: ``context.json`` (factor_context/1 + coverage), ``participation.json``
(snap-count receipt, route blocker), ``matrix.md`` (per-game coverage), ``raw/`` payloads
with sha256 in ``manifest.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from nflvalue.sources import live_factor_context as lfc  # noqa: E402
from nflvalue.sources import participation_evidence as pe  # noqa: E402
from nflvalue.sources.availability import canonical_abbr  # noqa: E402

PLAYERS_URL = "https://github.com/nflverse/nflverse-data/releases/download/players/players.parquet"
RELEASE_API = "https://api.github.com/repos/nflverse/nflverse-data/releases/tags/{tag}"


class RecordingHTTP:
    """Wraps the generic-UA fetcher and keeps every raw body with its sha256."""

    def __init__(self, out: str):
        self.dir, self.manifest = os.path.join(out, "raw"), []
        os.makedirs(self.dir, exist_ok=True)

    def __call__(self, url: str):
        status, headers, body = lfc.default_http(url, timeout=60)
        name = f"{len(self.manifest):03d}_" + "".join(ch if ch.isalnum() else "_" for ch in url)[-90:]
        with open(os.path.join(self.dir, name), "wb") as f:
            f.write(body)
        self.manifest.append({"url": url, "file": f"raw/{name}", "status": status,
                              "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body),
                              "date": headers.get("Date"), "last_modified": headers.get("Last-Modified")})
        return status, headers, body


def id_map_from_players(players: pd.DataFrame) -> List[Dict]:
    p = players.dropna(subset=["espn_id", "gsis_id", "latest_team"])
    return [{"espn_id": str(int(float(r.espn_id))) if str(r.espn_id).replace(".0", "").isdigit()
             else str(r.espn_id), "team": canonical_abbr(r.latest_team), "gsis_id": r.gsis_id}
            for r in p.itertuples(index=False)]


def matrix_md(doc: Dict) -> str:
    lines = ["| game | " + " | ".join(lfc.CATEGORIES) + " |", "|---" * (len(lfc.CATEGORIES) + 1) + "|"]
    for gid, row in doc["coverage"].items():
        lines.append(f"| {gid} | " + " | ".join(f"{row[c]['state']} ({row[c]['n_items']})"
                                                 for c in lfc.CATEGORIES) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--curated")
    ap.add_argument("--publish-context")
    ap.add_argument("--max-requests", type=int, default=48)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    http = RecordingHTTP(a.out)

    _, h, body = http(PLAYERS_URL)
    players = pd.read_parquet(io.BytesIO(body))
    curated = json.load(open(a.curated)) if a.curated else None
    doc = lfc.build_live_context(a.season, a.week, http=http, id_map=id_map_from_players(players),
                                 curated=curated, max_requests=a.max_requests)

    snap_url = pe.SNAP_URL.format(season=a.season)
    part: Dict = {}
    try:
        _, sh, sb = http(snap_url)
        sched = {}
        for wk in range(1, a.week):
            try:
                _, _, bb = http(f"{lfc.SITE}/scoreboard?seasontype=2&week={wk}&dates={a.season}")
                sched[wk] = [g["game_id"] for g in lfc.slate_from_scoreboard(json.loads(bb), a.season, wk)]
            except Exception as exc:  # noqa: BLE001
                part.setdefault("schedule_errors", []).append(f"week {wk}: {exc}")
        loaded = pe.load_snap_counts(pd.read_parquet(io.BytesIO(sb)), season=a.season,
                                     target_week=a.week, players=players, schedule_games=sched,
                                     source={"url": snap_url, "fetched_at": doc["captured_at"],
                                             "last_modified": sh.get("Last-Modified"),
                                             "sha256_bytes": hashlib.sha256(sb).hexdigest()})
        part["snap_counts"] = loaded["receipt"]
    except Exception as exc:  # noqa: BLE001
        part["snap_counts"] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    try:
        _, _, rb = http(RELEASE_API.format(tag="pbp_participation"))
        assets = [x["name"] for x in json.loads(rb).get("assets", [])]
    except Exception as exc:  # noqa: BLE001
        assets, part["participation_release_error"] = [], str(exc)
    part["routes"] = pe.route_availability(a.season, published_assets=assets)
    part["expected_workload"] = pe.expected_workload("*")

    def dump(name, obj):
        with open(os.path.join(a.out, name), "w") as f:
            json.dump(obj, f, indent=1, default=str)
    dump("context.json", doc)
    dump("participation.json", part)
    dump("manifest.json", http.manifest)
    with open(os.path.join(a.out, "matrix.md"), "w") as f:
        f.write(matrix_md(doc))
    if a.publish_context:
        pub = {k: v for k, v in doc.items() if k != "request_log"}
        with open(a.publish_context, "w") as f:
            json.dump(pub, f, indent=1)
            f.write("\n")
    print(json.dumps({"games": len(doc["games"]), "routes": doc["routes"],
                      "news": len(doc["news"]), "records": len(doc["records"]),
                      "snap_weeks": (part.get("snap_counts") or {}).get("weeks"),
                      "routes_available": part["routes"]["available"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
