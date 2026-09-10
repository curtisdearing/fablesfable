#!/usr/bin/env python3
"""Assemble the GitHub Pages payload (``_site``) for the live weekly loop.

This used to be inline shell in ``.github/workflows/live-weekly.yml``, which
meant the only way to test a change to it was to deploy it.  Everything the
Pages step does now lives here so ``tests/test_pages_workflow_contract.py``
can build a whole site from a temporary fixture tree, offline.

What ships
----------
``_site/index.html``                    the dashboard (the product)
``_site/api/hub.json``                  the slim feed dearing-wedding.com reads
``_site/reports/latest.html``           this week's drop, or a visible notice
``_site/reports/{season}/week-{n}.html`` the same document, permanently addressed
``_site/reports/index.json``            what was published, and why/why not
``_site/games/{game_id}.html``          one page per matchup (odds, injuries
                                        with report dates, travel/body clock,
                                        the model's drivers) -- from the
                                        ``game_pages`` block of data/latest.json
``_site/games/index.json``              which game pages were written

Why the season and week come from the payload
---------------------------------------------
``drops/`` is a directory of HTML documents whose names look sortable, and
this repository *ships a sample* (``drops/props_week_2023_9.html``).  Any
"publish the newest drop" shortcut therefore has a live failure mode: a
deploy-only run, or a Wednesday run that never got as far as writing a
document, would quietly publish a 2023 Week 9 card as this week's.

So the identity of "the current week" is taken from the payload the pipeline
generates (``data/weekly_props.json``: season, week, clock, as_of), and the
document that payload points to must
  * exist,
  * still be fresh (``--max-age-hours``), and
  * *name the same season and week in its own body*.

If any of those fails, ``reports/latest.html`` is **overwritten** with a
notice that says so.  Overwritten rather than skipped, so a previous build's
card can never survive as the current one, and no versioned path is written
at all.  The run stays green — a missing weekly document must not take the
dashboard offline — unless ``--strict-report`` is passed.

Usage: python scripts/prepare_pages.py [--root .] [--strict-report]
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: A weekly document older than this is not "current" any more.  Ten days
#: covers a full Wednesday-to-Wednesday cycle plus slack for a postponed
#: game; anything older is a previous week wearing this week's clothes.
DEFAULT_MAX_AGE_HOURS = 240.0


class Fatal(Exception):
    """The deploy must go red: the dashboard or the hub feed is broken."""


def log(message: str) -> None:
    print(f"[pages] {message}", flush=True)


# --------------------------------------------------------------------------- #
# payload
# --------------------------------------------------------------------------- #
def parse_timestamp(value):
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        stamp = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=dt.timezone.utc)


def load_payload(path: Path):
    """(payload, reason) — reason is set when the payload cannot be trusted."""
    if not path.exists():
        return None, ("missing_payload",
                      f"no generated payload at {path.name}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return None, ("unreadable_payload", f"{path}: {exc}")
    if not isinstance(payload, dict):
        return None, ("unreadable_payload", f"{path}: payload is not an object")
    try:
        season = int(payload["season"])
        week = int(payload["week"])
    except (KeyError, TypeError, ValueError):
        return None, ("payload_missing_season_or_week",
                      f"{path} does not name a season and week")
    return {"season": season, "week": week,
            "clock": str(payload.get("clock") or "wed"),
            "as_of": payload.get("as_of")}, None


def drop_filename(season: int, week: int, clock: str) -> str:
    """Mirrors nflvalue.document.write_drop."""
    return f"props_week_{season}_{week}{'_t90' if clock == 't90' else ''}.html"


def names_the_week(document: str, season: int, week: int) -> bool:
    """The document must claim the same identity the payload does — the
    filename alone is a naming convention, not evidence."""
    return re.search(rf"\b{season}\s+Week\s+{week}\b", document) is not None


def resolve_report(payload, drops_dir: Path, now: dt.datetime, max_age_hours: float,
                   root: Path):
    """(path, document, reason) — exactly one of path/reason is set.

    Reasons are rendered into a public page, so paths in them are made
    repository-relative: the runner's absolute layout is nobody's business.
    """
    def rel(path: Path) -> str:
        try:
            return os.path.relpath(path, root).replace(os.sep, "/")
        except ValueError:
            return path.name

    season, week, clock = payload["season"], payload["week"], payload["clock"]
    as_of = parse_timestamp(payload.get("as_of"))
    if as_of is None:
        return None, None, ("payload_missing_as_of",
                            f"{season} week {week}: payload carries no usable as_of "
                            "timestamp, so it cannot be shown to be current")
    age_hours = (now - as_of).total_seconds() / 3600.0
    if max_age_hours > 0 and age_hours > max_age_hours:
        return None, None, ("stale_payload",
                            f"{season} week {week}: the generated payload is stale "
                            f"({age_hours:.1f}h old, limit {max_age_hours:.0f}h)")
    path = drops_dir / drop_filename(season, week, clock)
    if not path.is_file():
        return None, None, ("missing_report",
                            f"{season} week {week} ({clock}): no document at "
                            f"{rel(path)}")
    try:
        document = path.read_text(encoding="utf-8")
    except OSError as exc:
        return None, None, ("unreadable_report", f"{rel(path)}: {exc}")
    if not names_the_week(document, season, week):
        return None, None, ("report_identity_mismatch",
                            f"{rel(path)} does not name {season} week {week}; it disagrees "
                            "with the generated payload and will not be published")
    return path, document, None


# --------------------------------------------------------------------------- #
# site
# --------------------------------------------------------------------------- #
_NOTICE_CSS = (
    "body{font:15px/1.55 -apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;"
    "color:#141a24;max-width:680px;margin:48px auto;padding:0 20px;background:#fff}"
    "h1{font-size:22px;margin:0 0 10px}"
    ".warn{background:#fbf7ee;border-left:4px solid #b8860b;padding:12px 16px;margin:14px 0}"
    ".sub{color:#5a6472;font-size:13px}"
)


def notice_html(reason: str, detail: str, payload, built_at: str) -> str:
    week_label = ("unknown" if not payload
                  else f"{payload['season']} Week {payload['week']}")
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>No current weekly report</title>"
        f"<style>{_NOTICE_CSS}</style></head><body>"
        "<h1>No current weekly report</h1>"
        "<div class='warn'><b>Nothing is being shown here on purpose.</b> The most "
        "recent weekly document could not be confirmed as current, and publishing a "
        "previous week in its place would be worse than publishing nothing.</div>"
        f"<p>Expected week: <b>{html.escape(week_label)}</b><br>"
        f"Reason: <code>{html.escape(reason)}</code><br>"
        f"<span class='sub'>{html.escape(detail)}</span></p>"
        "<p class='sub'>The dashboard on this site is still current for whatever data "
        "it reports. This page will fill in on the next successful weekly run. "
        f"Built {html.escape(built_at)}.</p>"
        "</body></html>")


def build_hub_feed(root: Path, site: Path, script: Path) -> None:
    if not script.is_file():
        raise Fatal(f"hub feed builder not found at {script}")
    out = site / "api" / "hub.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run([sys.executable, str(script), "--out", str(out)],
                            cwd=str(root), capture_output=True, text=True)
    for line in (result.stdout or "").splitlines():
        log(line)
    if result.returncode != 0:
        raise Fatal(f"hub feed build failed (rc={result.returncode}): "
                    f"{(result.stderr or '').strip()}")
    if not out.is_file():
        raise Fatal(f"hub feed build reported success but {out} does not exist")


def build_game_pages(root: Path, site: Path, latest_path: Path,
                     payload) -> dict:
    """Write ``_site/games/*.html`` from the ``game_pages`` block of
    ``data/latest.json``.  Pages from another season/week than the payload
    names are NOT written (a stale page under a current board would be the
    same lie the report guard exists to prevent).  Never fatal: a board with
    no game pages is a board; a broken page builder is logged."""
    result = {"written": 0, "skipped_stale": 0, "reason": None}
    if not latest_path.is_file():
        result["reason"] = "missing_latest"
        log("game pages: no data/latest.json -> none written")
        return result
    try:
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
        pages = latest.get("game_pages") or []
    except (OSError, ValueError) as exc:
        result["reason"] = f"unreadable_latest: {exc}"
        log(f"WARN game pages: {result['reason']}")
        return result
    if payload is not None:
        current = [p for p in pages
                   if str(p.get("season")) == str(payload["season"])
                   and str(p.get("week")) == str(payload["week"])]
        result["skipped_stale"] = len(pages) - len(current)
        pages = current
    if not pages:
        result["reason"] = latest.get("game_pages_error") or "no_pages_in_payload"
        log(f"game pages: none to write ({result['reason']})")
        return result
    # the renderer is code, not state: it comes from this script's repository,
    # not from ``--root`` (a fixture tree carries payloads, not packages)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        from nflvalue import game_pages as gpmod
        written = gpmod.write_site_pages(pages, str(site))
    except Exception as exc:  # noqa: BLE001 -- the board must still ship
        result["reason"] = f"render_failed: {type(exc).__name__}: {exc}"
        log(f"WARN game pages: {result['reason']}")
        return result
    result["written"] = len(written)
    log(f"games/: {len(written)} page(s) written"
        + (f", {result['skipped_stale']} stale skipped" if result["skipped_stale"] else ""))
    return result


def build(root: Path, site: Path, payload_path: Path, drops_dir: Path,
          dashboard: Path, hub_script: Path, now: dt.datetime,
          max_age_hours: float, latest_path: Path = None) -> dict:
    site.mkdir(parents=True, exist_ok=True)

    if not dashboard.is_file():
        raise Fatal(f"dashboard not found at {dashboard}; refusing to publish an "
                    "empty site")
    shutil.copyfile(dashboard, site / "index.html")
    log(f"index.html <- {dashboard.name}")

    build_hub_feed(root, site, hub_script)

    reports = site / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    built_at = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    payload, reason = load_payload(payload_path)
    if payload is not None:
        log(f"payload: {payload['season']} week {payload['week']} "
            f"({payload['clock']}, as_of {payload.get('as_of')})")
        source, _document, reason = resolve_report(
            payload, drops_dir, now, max_age_hours, root)
    else:
        source = None

    manifest = {
        "schema_version": 1,
        "built_at": built_at,
        "published": False,
        "reason": None,
        "season": payload["season"] if payload else None,
        "week": payload["week"] if payload else None,
        "clock": payload["clock"] if payload else None,
        "as_of": payload.get("as_of") if payload else None,
        "source": None,
        "paths": {"latest": "reports/latest.html", "versioned": None},
    }

    if reason is None and payload is not None:
        season, week = payload["season"], payload["week"]
        versioned = reports / str(season) / f"week-{week}.html"
        versioned.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, versioned)
        shutil.copyfile(source, reports / "latest.html")
        manifest.update(published=True,
                        source=os.path.relpath(source, root).replace(os.sep, "/"))
        manifest["paths"]["versioned"] = f"reports/{season}/week-{week}.html"
        log(f"reports/latest.html and {manifest['paths']['versioned']} <- "
            f"{manifest['source']}")
    else:
        code, detail = reason
        manifest["reason"] = code
        # Overwrite, never skip: a previous build's card must not survive here.
        (reports / "latest.html").write_text(
            notice_html(code, detail, payload, built_at), encoding="utf-8")
        log(f"WARN no current weekly report published ({code}): {detail}")
        log("WARN reports/latest.html now carries a visible notice; no versioned "
            "report was written")

    manifest["game_pages"] = build_game_pages(
        root, site, latest_path or (root / "data" / "latest.json"), payload)

    (reports / "index.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                        encoding="utf-8")
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Assemble _site for GitHub Pages.")
    ap.add_argument("--root", type=Path, default=ROOT,
                    help="repository root (default: this script's repository)")
    ap.add_argument("--site", type=Path, default=None, help="default: {root}/_site")
    ap.add_argument("--payload", type=Path, default=None,
                    help="generated weekly payload (default: {root}/data/weekly_props.json)")
    ap.add_argument("--drops-dir", type=Path, default=None, help="default: {root}/drops")
    ap.add_argument("--dashboard", type=Path, default=None,
                    help="default: {root}/dashboard.html")
    ap.add_argument("--hub-feed", type=Path, default=None,
                    help="default: {root}/scripts/build_hub_feed.py")
    ap.add_argument("--latest", type=Path, default=None,
                    help="dashboard payload carrying game_pages (default: {root}/data/latest.json)")
    ap.add_argument("--max-age-hours", type=float, default=DEFAULT_MAX_AGE_HOURS,
                    help="payloads older than this are not current (0 disables)")
    ap.add_argument("--now", default=None, help="ISO-8601 clock override, for tests")
    ap.add_argument("--strict-report", action="store_true",
                    help="fail the build when no current weekly report is publishable")
    args = ap.parse_args(argv)

    root = args.root.resolve()
    site = (args.site or root / "_site").resolve()
    payload_path = (args.payload or root / "data" / "weekly_props.json").resolve()
    drops_dir = (args.drops_dir or root / "drops").resolve()
    dashboard = (args.dashboard or root / "dashboard.html").resolve()
    hub_script = (args.hub_feed or root / "scripts" / "build_hub_feed.py").resolve()
    latest_path = (args.latest or root / "data" / "latest.json").resolve()

    now = parse_timestamp(args.now) if args.now else dt.datetime.now(dt.timezone.utc)
    if now is None:
        print(f"[pages] ERROR --now is not a timestamp: {args.now!r}", file=sys.stderr)
        return 2

    try:
        manifest = build(root, site, payload_path, drops_dir, dashboard, hub_script,
                         now, args.max_age_hours, latest_path=latest_path)
    except Fatal as exc:
        print(f"[pages] ERROR {exc}", file=sys.stderr)
        return 2

    if not manifest["published"] and args.strict_report:
        print(f"[pages] ERROR --strict-report: no current weekly report "
              f"({manifest['reason']})", file=sys.stderr)
        return 1
    log(f"site ready at {site}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
