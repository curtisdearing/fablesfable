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


def test_generated_pages_keep_mobile_scroll_bridge(tmp_path):
    """The next scheduled rebuild must not pin the mobile header again."""
    db = _db(tmp_path, [_lean()])
    out = tmp_path / "site" / "published-site"
    assert bps.main(["--db", db, "--season", "2026", "--week", "3",
                     "--archive", _archive(tmp_path), "--out", str(out),
                     "--label", "replay", "--now", "2026-09-23T00:00:00Z"]) == 0
    for relative in ("index.html", "best-bets.html", "reports/latest.html", "reports/2026/week-3.html"):
        document = (out / relative).read_text()
        assert "dearing-hub:hello" in document, relative
        assert "dearing-hub:scroll" in document, relative
        assert "window.parent" in document, relative
        assert "name=\"viewport\"" in document, relative
    _run_checker(out.parent)


def test_refresh_preserves_same_week_analyst_reading_experience(tmp_path):
    """A T90 refresh must update data, not replace the curated UI with a data dump."""
    import hashlib
    archive = Path(_archive(tmp_path))
    paths = ('index.html', 'best-bets.html', 'reports/latest.html', 'reports/2026/week-3.html')
    reading = '<html><body><a href="#g-ATL-GB">ATL at GB</a><section data-game="ATL-GB" id="g-ATL-GB"><details><summary>Research</summary>Original quote 2026-09-22</details></section></body></html>'
    for name in paths:
        (archive / name).write_text(reading)
    analyst = {'date': '2026-09-27', 'model_approved': False}
    manifest = {'season': 2026, 'week': 3, 'analyst_card': analyst,
                'quote_clocks': {'latest': '2026-09-22T00:00:00Z'},
                'files': {name: hashlib.sha256(reading.encode()).hexdigest() for name in paths}}
    (archive / 'publication.json').write_text(json.dumps(manifest))
    db = _db(tmp_path, [_lean()])
    out = tmp_path / 'site' / 'published-site'
    assert bps.main(['--db', db, '--season', '2026', '--week', '3', '--archive', str(archive),
                     '--out', str(out), '--label', 'fresh', '--now', '2026-09-23T00:00:00Z']) == 0
    for name in paths:
        text = (out / name).read_text()
        assert 'data-game="ATL-GB"' in text, name
        assert '<details>' in text and 'Original quote 2026-09-22' in text
        assert 'model-cards.html' in text
    assert 'D.London' in (out / 'model-cards.html').read_text()
    hub = json.loads((out / 'api/hub.json').read_text())
    assert hub['cards'][0]['player'] == 'D.London'
    assert hub['analyst_card'] == analyst
    assert hub['quote_clocks']['latest'] == '2026-09-22T22:19:33Z'
    saved = json.loads((out / 'publication.json').read_text())
    assert saved['reading_experience']['quote_clocks'] == manifest['quote_clocks']
    _run_checker(out.parent)
    # Rebuilding an already repaired publication keeps analyst provenance and
    # inserts only one notice, while production data remains separate.
    second = tmp_path / 'second'
    bps.build(hub, 'fresh', str(out), str(second), '2026-09-23T01:00:00Z')
    assert (second / 'index.html').read_text().count('<!-- model-refresh-link -->') == 1
    assert json.loads((second / 'publication.json').read_text())['reading_experience'] == saved['reading_experience']
    # A subsequent week must not inherit this Sunday's curated picks.
    manifest['week'] = 2
    (archive / 'publication.json').write_text(json.dumps(manifest))
    assert bps.main(['--db', db, '--season', '2026', '--week', '3', '--archive', str(archive),
                     '--out', str(out), '--label', 'fresh']) == 0
    assert 'Original quote 2026-09-22' not in (out / 'index.html').read_text()
    assert 'analyst_card' not in json.loads((out / 'api/hub.json').read_text())


def test_generated_fallback_has_game_links_and_closed_evidence(tmp_path):
    from html.parser import HTMLParser
    class Tags(HTMLParser):
        def __init__(self):
            super().__init__()
            self.tags = []
        def handle_starttag(self, tag, attrs):
            self.tags.append((tag, dict(attrs)))
    db = _db(tmp_path, [_lean(), _lean(player_id='p2', name='B.Robinson')])
    out = tmp_path / 'site' / 'published-site'
    assert bps.main(['--db', db, '--season', '2026', '--week', '3', '--archive', _archive(tmp_path),
                     '--out', str(out), '--label', 'fresh', '--now', '2026-09-23T00:00:00Z']) == 0
    for name in ('index.html', 'best-bets.html', 'model-cards.html', 'reports/latest.html', 'reports/2026/week-3.html'):
        doc = (out / name).read_text()
        p = Tags()
        p.feed(doc)
        assert any(t == 'a' and a.get('href') == '#g-2026_03_ATL_GB' for t, a in p.tags), name
        cards = [a for t, a in p.tags if t == 'details' and a.get('class') == 'model-card']
        if name != 'best-bets.html':  # stale fixture quotes may leave the watch list empty
            assert len(cards) == 2 and all('open' not in a for a in cards), name
        # Every relative report navigation link resolves inside the published root.
        for t, a in p.tags:
            href = a.get('href', '')
            if t == 'a' and href and not href.startswith(('#', 'https:', 'http:')):
                assert (out / name).parent.joinpath(href).resolve().is_file(), (name, href)
    _run_checker(out.parent)


def test_workflow_rejects_expanded_report_even_with_valid_hashes(tmp_path):
    import hashlib
    db = _db(tmp_path, [_lean()])
    out = tmp_path / 'published-site'
    assert bps.main(['--db', db, '--season', '2026', '--week', '3', '--archive', _archive(tmp_path),
                     '--out', str(out), '--label', 'fresh']) == 0
    bad = '<html><body>Plain expanded report overwrote the dashboard</body></html>'
    (out / 'index.html').write_text(bad)
    manifest = json.loads((out / 'publication.json').read_text())
    manifest['files']['index.html'] = hashlib.sha256(bad.encode()).hexdigest()
    (out / 'publication.json').write_text(json.dumps(manifest))
    with pytest.raises(AssertionError, match='interactive'):
        _run_checker(out.parent)


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
