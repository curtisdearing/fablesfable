"""Grade the issued-pick ledger against official ESPN final box scores. Offline; no odds requests.

    python scripts/grade_issued_picks.py --db data/nfl_props.db --season 2026 --week 3 \
        --box-dir <dir of ESPN box JSON> --box-captured-at 2026-09-25T04:00:00Z --out <dir>
    python scripts/grade_issued_picks.py --export reports/issued_picks_2026_wk3.json ...   # a ledger export
    python scripts/grade_issued_picks.py --hub site/api/hub.json --publication site/publication.json ...

Reads the ledger read-only and writes ``issued_grades.json`` (sections, row-level grades,
rejected box files, summaries) and ``issued_grades_rows.csv`` to --out; output is
deterministic for identical inputs. The default section is ``recommendations_given``
(published or delivered before kickoff); watch-list, generated-only, retrospective and
the latest-snapshot analysis are separate sections. ``--hub`` needs the page's own
``publication.json`` (sha256 of hub.json must match) and is always retrospective: a page
rebuilt at grading time is evidence of what was shown, not of when. ``--prior`` reports
stat corrections beside the original values. Exit 3 when there are no records, 4 when
--box-captured-at is not a zoned clock, 5 when a page fails publication verification.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sqlite3
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from nflvalue import issued_grading as ig  # noqa: E402
from nflvalue import issued_ledger as il  # noqa: E402


def load_records(a):
    if a.hub:
        if len(a.publication or []) != len(a.hub):
            raise ValueError("--hub needs one --publication manifest per page")
        recs = {}
        for h, m in zip(a.hub, a.publication):
            for r in il.publication_records(h, m):
                if r["record_id"] in recs:
                    recs[r["record_id"]]["events"].extend(r["events"])
                else:
                    recs[r["record_id"]] = r
        recs = list(recs.values())
    elif a.export:
        recs = []
        for p in a.export:
            recs.extend(json.load(open(p))["records"])
    else:
        conn = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
        try:
            if not conn.execute("SELECT name FROM sqlite_master WHERE name='issued_pick_events'").fetchone():
                return []
            recs = il.load(conn, a.season, a.week)
        finally:
            conn.close()
    return [r for r in recs if (a.season is None or r["season"] == a.season)
            and (a.week is None or r["week"] == a.week)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--db")
    src.add_argument("--export", nargs="+")
    src.add_argument("--hub", nargs="+")
    ap.add_argument("--publication", nargs="+", help="publication.json for each --hub page")
    ap.add_argument("--season", type=int)
    ap.add_argument("--week", type=int)
    ap.add_argument("--box-dir", required=True)
    ap.add_argument("--box-captured-at", required=True, help="zoned UTC clock the box files were fetched")
    ap.add_argument("--id-map", help="JSON {ledger player_id: ESPN athlete id}")
    ap.add_argument("--prior", help="an earlier issued_grades.json for stat-correction comparison")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if ig._ts(a.box_captured_at) is None:
        print("[grade] --box-captured-at must be a zoned ISO clock (e.g. 2026-09-22T01:20:00Z)")
        return 4
    try:
        records = load_records(a)
    except ValueError as exc:
        print(f"[grade] publication not verified: {exc}")
        return 5
    if not records:
        print("[grade] no issued-pick records in the ledger for this selection: nothing to grade")
        return 3
    boxes = ig.load_boxes(glob.glob(os.path.join(a.box_dir, "*.json")), a.box_captured_at)
    id_map = json.load(open(a.id_map)) if a.id_map else None
    prior = None
    if a.prior:
        prior = [r for s in json.load(open(a.prior))["sections"].values() for r in s["rows"]]
    res = ig.grade(records, boxes, id_map=id_map, prior_rows=prior)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "issued_grades.json"), "w") as f:
        json.dump(res, f, indent=1, sort_keys=True, default=str)
        f.write("\n")
    rows = [r for s in res["sections"].values() for r in s["rows"]]
    cols = sorted({k for r in rows for k in r})
    with open(os.path.join(a.out, "issued_grades_rows.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows({k: (json.dumps(v, default=str) if isinstance(v, (list, dict)) else v)
                     for k, v in r.items()} for r in rows)
    print(f"[grade] counts {res['counts']}; box files rejected: {len(res['boxes']['rejected'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
