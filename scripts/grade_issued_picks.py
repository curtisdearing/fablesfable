"""Grade the issued-pick ledger against official ESPN final box scores. Offline; no odds requests.

    python scripts/grade_issued_picks.py --db data/nfl_props.db --season 2026 --week 3 \
        --box-dir <dir of ESPN box JSON> --box-captured-at 2026-09-25T04:00:00Z --out <dir>
    python scripts/grade_issued_picks.py --export reports/issued_picks_2026_wk3.json ...  # a run's export
    python scripts/grade_issued_picks.py --hub published-site/api/hub.json ...        # a published page

Reads the ledger read-only (``issued_picks`` table or an export), writes
``issued_grades.json`` (row-level grades, exclusions, grouped summaries) and
``issued_grades_rows.csv`` to --out. ``--hub`` rebuilds ledger-form records from the
cards a published site displayed (same record ids as the run's ledger; recorded at
the page's generation clock). Output is deterministic for identical
inputs. ``--prior`` compares with an earlier grades file and reports stat
corrections beside the original values. Exit 3 when the ledger holds no records.
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
        recs = []
        for p in a.hub:
            h = json.load(open(p))
            recs.extend(il.publication_records(h["cards"], h["season"], h["week"], h["label"],
                                               h["generated_at"]))
        # one record per displayed content; revisions ordered by the pages' generation clocks
        uniq = {}
        for r in sorted(recs, key=lambda r: str(r["recorded_at"])):
            uniq.setdefault(r["record_id"], r)
        seen = {}
        for r in uniq.values():
            seen[r["pick_key"]] = r["revision"] = seen.get(r["pick_key"], 0) + 1
        recs = list(uniq.values())
    elif a.export:
        recs = []
        for p in a.export:
            recs.extend(json.load(open(p))["records"])
    else:
        conn = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
        try:
            if not conn.execute("SELECT name FROM sqlite_master WHERE name='issued_picks'").fetchone():
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
    ap.add_argument("--season", type=int)
    ap.add_argument("--week", type=int)
    ap.add_argument("--box-dir", required=True)
    ap.add_argument("--box-captured-at", required=True, help="UTC clock the box files were fetched")
    ap.add_argument("--id-map", help="JSON {ledger player_id: ESPN athlete id}")
    ap.add_argument("--prior", help="an earlier issued_grades.json for stat-correction comparison")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    records = load_records(a)
    if not records:
        print("[grade] no issued-pick records in the ledger for this selection: nothing to grade")
        return 3
    games = ig.load_boxes(glob.glob(os.path.join(a.box_dir, "*.json")), a.box_captured_at)
    id_map = json.load(open(a.id_map)) if a.id_map else None
    prior = json.load(open(a.prior))["rows"] if a.prior else None
    res = ig.grade(records, games, id_map=id_map, prior_rows=prior)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "issued_grades.json"), "w") as f:
        json.dump(res, f, indent=1, sort_keys=True, default=str)
        f.write("\n")
    cols = sorted({k for r in res["rows"] for k in r})
    with open(os.path.join(a.out, "issued_grades_rows.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(res["rows"])
    print(f"[grade] {len(res['rows'])} decisions graded, {len(res['excluded'])} excluded; "
          f"groups: {sorted(res['groups'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
