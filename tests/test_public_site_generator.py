"""scripts/build_public_site.py: current week replaces the top-level pages, fails closed,
and its output passes the website workflow's own static checker."""

import importlib.util
import json
import os
import re
import sqlite3
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_public_site", ROOT / "scripts" / "build_public_site.py")
bps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bps)

LEAN_COLS = ("season", "week", "clock", "game_id", "player_id", "name", "market", "side", "line",
             "line_source", "price", "book", "mean", "sd", "p_side", "composite", "status", "void_reason",
             "as_of", "created_at", "quote_book", "quote_ts", "run_id", "code_sha", "forecast_version",
             "ranker_sha256", "selection_source", "stage_json")


def _db(tmp_path, leans, provenance=True):
    path = tmp_path / "nfl_props.db"
    conn = sqlite3.connect(path)
    cols = LEAN_COLS if provenance else LEAN_COLS[:20]
    conn.execute(f"CREATE TABLE leans ({', '.join(cols)})")
    conn.execute("CREATE TABLE lines (ts, game_id, book, market, player_id, player_name, side, point, price)")
    conn.execute("CREATE TABLE run_receipts (run_id, season, week, clock, as_of, receipt_json, created_at)")
    conn.execute("INSERT INTO run_receipts VALUES ('local:replay', 2026, 3, 'wed', '2026-09-22T22:35:00Z', "
                 "'{\"run_id\": \"local:replay\", \"publish\": true, \"publish_reasons\": []}', "
                 "'2026-09-22T22:35:05Z')")
    for r in leans:
        conn.execute(f"INSERT INTO leans VALUES ({','.join('?' * len(cols))})", [r.get(c) for c in cols])
    conn.execute("INSERT INTO lines VALUES ('2026-09-22T22:19:33Z','2026_03_ATL_GB','draftkings',"
                 "'receptions','','Drake London','over',5.5,1.87)")
    conn.commit()
    conn.close()
    return str(path)


def _lean(**kw):
    r = dict(season=2026, week=3, clock="wed", game_id="2026_03_ATL_GB", player_id="p1", name="D.London",
             market="receptions", side="over", line=5.5, line_source="odds_api", price=1.87,
             book="draftkings/fanduel", mean=6.1, sd=2.4, p_side=0.55, composite=61.0, status="active",
             as_of="2026-09-22T22:35:00Z", created_at="2026-09-22T22:35:05Z", quote_book="draftkings",
             quote_ts="2026-09-22T22:19:33Z", run_id="local:replay", code_sha="c0de" * 10,
             forecast_version="ff-football-only-v1", ranker_sha256="f" * 64, selection_source="ml_gbdt",
             stage_json='{"team": "ATL", "stages": {}, "availability": {"status": "OK", '
                        '"eligibility": "eligible", "availability_state": "not_listed"}}')
    r.update(kw)
    return r


def _archive(tmp_path):
    a = tmp_path / "archive"
    (a / "reports" / "2026").mkdir(parents=True)
    (a / "games").mkdir()
    (a / "reports/2026/week-1.html").write_text("<html>week 1</html>")
    (a / "games/2026_01_ATL_PIT.html").write_text("<html>game</html>")
    (a / "history.html").write_text("<html>history</html>")
    (a / "index.html").write_text("<html>OLD WEEK 1 INDEX</html>")
    (a / "notes.txt").write_text("not public")
    return str(a)


def _checker():
    wf = (ROOT / ".github/workflows/website.yml").read_text()
    m = re.search(r"          python3 - <<'PY'\n(?P<c>.*?)\n          PY", wf, re.DOTALL)
    return textwrap.dedent(m.group("c"))


def _run_checker(site_parent):
    cwd = os.getcwd()
    try:
        os.chdir(site_parent)
        exec(compile(_checker(), "checker", "exec"), {"__name__": "__main__"})
    finally:
        os.chdir(cwd)


def test_builds_current_week_top_level_and_passes_the_workflow_checker(tmp_path):
    db = _db(tmp_path, [_lean(), _lean(player_id="p2", name="B.Robinson", market="rushing_yards",
                                       line_source="synthetic_trailing_mean", price=None, quote_book=None,
                                       quote_ts=None)])
    out = tmp_path / "site" / "published-site"
    archive = _archive(tmp_path)
    rc = bps.main(["--db", db, "--season", "2026", "--week", "3", "--archive", archive,
                   "--out", str(out), "--label", "replay", "--now", "2026-09-23T00:00:00Z"])
    assert rc == 0
    idx = (out / "index.html").read_text()
    assert "2026 week 3" in idx and "REPLAY" in idx and "OLD WEEK 1 INDEX" not in idx
    assert "local:replay" in idx and "2026-09-22T22:19:33Z" in idx and "No card is a recommended wager" in idx
    m = json.loads((out / "publication.json").read_text())
    assert (m["season"], m["week"], m["label"], m["approved_bets"]) == (2026, 3, "replay", 0)
    assert m["runs"][0]["code_sha"] == "c0de" * 10 and m["source_as_of"] == "2026-09-22T22:19:33Z"
    assert "reports/2026/week-1.html" in m["files"] and "games/2026_01_ATL_PIT.html" in m["files"]
    assert "notes.txt" not in m["files"] and "reports/2026/week-3.html" in m["files"]
    hub = json.loads((out / "api/hub.json").read_text())
    statuses = {c["player"]: c["status"] for c in hub["cards"]}
    assert statuses == {"D.London": "watch", "B.Robinson": "research"}
    assert bps.main(["--db", db, "--season", "2026", "--week", "3", "--archive", archive,
                     "--out", str(tmp_path / "late"), "--label", "replay",
                     "--now", "2026-09-23T06:00:00Z"]) == 0
    late = json.loads((tmp_path / "late/api/hub.json").read_text())
    assert {c["player"]: c["status"] for c in late["cards"]}["D.London"] == "pass"  # >6 h old
    _run_checker(out.parent)


@pytest.mark.parametrize("leans,provenance,code", [
    ([], True, 3),
    ([_lean(week=2)], True, 3),
    ([_lean(run_id=None)], True, 4),
    ([_lean(forecast_version="legacy")], True, 4),
    ([_lean()], False, 4),
])
def test_fails_closed(tmp_path, leans, provenance, code):
    db = _db(tmp_path, leans, provenance=provenance)
    out = tmp_path / "out"
    assert bps.main(["--db", db, "--season", "2026", "--week", "3", "--archive", _archive(tmp_path),
                     "--out", str(out), "--label", "fresh"]) == code
    assert not (out / "publication.json").exists()


def test_missing_db_or_archive_is_an_error(tmp_path):
    assert bps.main(["--db", str(tmp_path / "none.db"), "--season", "2026", "--week", "3",
                     "--out", str(tmp_path / "o"), "--label", "fresh"]) == 1
    db = _db(tmp_path, [_lean()])
    assert bps.main(["--db", db, "--season", "2026", "--week", "3", "--archive", str(tmp_path / "nope"),
                     "--out", str(tmp_path / "o"), "--label", "fresh"]) == 1


def test_production_runs_generate_and_one_publisher_deploys():
    live = (ROOT / ".github/workflows/live-weekly.yml").read_text()
    site = (ROOT / ".github/workflows/website.yml").read_text()
    build = live.index("Build public site from this run")
    assert live.index("Publish successful production state") < build
    step = live[build:live.index("Upload public site for the website publisher")]
    assert 'contains(fromJSON(\'["wed","t90"]\'), steps.select.outputs.job)' in step
    assert "scripts/build_public_site.py --current" in step and "--label fresh" in step
    assert "rc -eq 3" in step and 'exit "$rc"' in step
    assert "name: public-site" in live
    assert re.search(r"^    if: \$\{\{ false \}\}$", live.split("  deploy-dashboard:", 1)[1], re.M)
    assert live.count("actions/deploy-pages") == 1  # only inside the disabled job
    assert 'workflows: ["Live weekly model loop"]' in site and "types: [completed]" in site
    assert "github.event.workflow_run.conclusion == 'success'" in site
    assert "head_branch == 'main'" in site and "-n public-site" in site
    assert "cand > live" in site  # never replaces a newer publication with an older one
    assert site.count("actions/deploy-pages@v4") == 1
    assert "group: football-website-publication" in site and "cancel-in-progress: false" in site
