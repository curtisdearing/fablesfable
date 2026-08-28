#!/usr/bin/env python3
"""Self-scheduling wrapper: figures out the season/week/games itself so cron
or a Cowork scheduled task needs ZERO variables.

    python3 scripts/auto_weekly.py --job wed       # Wednesday full run + Discord
    python3 scripts/auto_weekly.py --job t90       # refresh games kicking off soon
    python3 scripts/auto_weekly.py --job tuesday   # grade + CLV + retrain the ML

Every job exits cleanly (code 0, one log line) in the offseason or when
there's nothing to do, so schedules can run year-round untouched. Kickoff
times are nflverse ET; comparisons use America/New_York.

Discord posts LIVE from here when config discord_enabled=true and a webhook
exists (env DISCORD_WEBHOOK_URL or config.local.json) — this wrapper is the
"hits my Discord every week" entry point. It never wagers; it informs.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ET = ZoneInfo("America/New_York")
T90_WINDOW_HOURS = 2.75      # refresh games kicking off within this window


def utc_stamp() -> str:
    return (dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
            .isoformat().replace("+00:00", "Z"))


def write_pipeline_heartbeat(status: str, detail: str, job: str) -> dict:
    """Make deployment freshness and missing integrations visible.

    ``generated_at`` remains the model-data timestamp.  A deployment/no-op
    must not make stale projections look newly generated.
    """
    from nflvalue import config as cfgmod
    from nflvalue.dashboard import write_dashboard
    from nflvalue.notify import resolve_webhook

    cfg = cfgmod.load_config()
    odds = "configured" if cfg.get("odds_api_key") else "missing"
    if not cfg.get("discord_enabled"):
        discord = "disabled"
    else:
        discord = "configured" if resolve_webhook() else "missing"
    effective_status = "degraded" if status == "active" and odds != "configured" else status
    if effective_status == "degraded":
        detail += " Live sportsbook pricing is unavailable until ODDS_API_KEY is configured."
    data = cfgmod.load_json(cfgmod.LATEST_PATH, {}) or {}
    data["pipeline"] = {
        "status": effective_status,
        "job": job,
        "last_checked_at": utc_stamp(),
        "detail": detail,
        "integrations": {"odds_api": odds, "discord": discord},
    }
    cfgmod.save_json(cfgmod.LATEST_PATH, data)
    write_dashboard(data)
    return data["pipeline"]


def schedule_status(slate, now: dt.datetime) -> str:
    cw = current_week(slate, now)
    if cw is None:
        return "offseason"
    first = slate[(slate.season == cw[0]) & (slate.week == cw[1])]["kickoff"].min()
    return "offseason" if first - now > dt.timedelta(days=8) else "active"


def ensure_dependencies() -> None:
    """Scheduled-task sessions can start with a fresh sandbox: self-heal by
    installing requirements when core imports are missing (evaluation catch —
    without this, every scheduled run in a new sandbox would die on import)."""
    try:
        import pandas, numpy, scipy, sklearn, pyarrow  # noqa: F401
    except ImportError:
        import subprocess
        root = str(Path(__file__).resolve().parents[1])
        print("[auto] bootstrapping python dependencies (fresh sandbox)…")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "--break-system-packages", "-r", f"{root}/requirements.txt"],
                       check=False, timeout=600)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "--break-system-packages", "nflreadpy"],
                       check=False, timeout=600)


def now_et() -> dt.datetime:
    return dt.datetime.now(ET)


def load_slate():
    from nflvalue.ingest import load_all_schedules
    s = load_all_schedules()
    s = s[s["game_type"] == "REG"].copy()
    s["kickoff"] = [
        dt.datetime.strptime(f"{d} {t or '13:00'}", "%Y-%m-%d %H:%M").replace(tzinfo=ET)
        for d, t in zip(s["gameday"], s["gametime"])
    ]
    return s


def current_week(slate, now: dt.datetime):
    """The REG week containing (or next after) now: earliest week whose LAST
    kickoff is still >= now - 12h. None in the offseason."""
    future = slate[slate["kickoff"] >= now - dt.timedelta(hours=12)]
    if future.empty:
        return None
    nxt = future.sort_values("kickoff").iloc[0]
    return int(nxt["season"]), int(nxt["week"])


def last_completed_week(slate, now: dt.datetime):
    done = slate[(slate["kickoff"] < now - dt.timedelta(hours=8)) & slate["result"].notna()]
    if done.empty:
        return None
    last = done.sort_values(["season", "week"]).iloc[-1]
    return int(last["season"]), int(last["week"])


def job_wed() -> int:
    from nflvalue import config as cfgmod, ingest
    import pipeline_weekly as pw
    r = ingest.refresh()
    print(f"[auto] ingest: season {r['season']} stale={r['stale']} errors={r['errors'] or 'none'}")
    slate = load_slate()
    cw = current_week(slate, now_et())
    if cw is None or (slate[(slate.season == cw[0]) & (slate.week == cw[1])]["kickoff"].min()
                      - now_et()) > dt.timedelta(days=8):
        print("[auto] no upcoming REG week within 8 days — offseason no-op")
        write_pipeline_heartbeat(
            "offseason", "Automation is healthy; no REG week starts within eight days.", "wed")
        return 0
    season, week = cw
    cfg = cfgmod.load_config()
    live_odds = bool(cfg.get("odds_api_key"))
    from nflvalue.notify import resolve_webhook
    post_live = bool(cfg.get("discord_enabled") and resolve_webhook())
    res = pw.run_week(season, week, mode="live", live_odds=live_odds,
                      discord=True, discord_dry_run=not post_live)
    print(f"[auto] wed run {season} wk{week}: {len(res['games'])} games, "
          f"publish={res['publish']}, odds={'live' if live_odds else 'no key -> no_market'}, "
          f"discord={res['discord']}")
    write_pipeline_heartbeat(
        "active", f"Wednesday model refresh completed for {season} week {week}.", "wed")
    return 0


def t90_line_snapshot(cfg, conn, soon, resnap=None, pull=None,
                      event_map_fn=None) -> dict:
    """Get the freshest REAL prop lines in front of the T-90 re-rank.

    Two distinct spends, both hard-stopped by the shared monthly credit
    budget, and both reported whether they happen or not:

    * **Resnap** (always, when a key exists): games that already carry an
      entry line get a second, pre-kickoff snapshot. Without it entry ==
      close and CLV could never resolve -- the kill-check would starve
      forever. It is also what lets ``run_t90`` price against a line that
      moved since Wednesday.
    * **First pull** (opt-in via ``odds_budget.t90_first_pull``): games with
      NO line at all. Wednesday's rotation and per-run cap skip several games
      every week and those publish ``no_market``; a first pull at T-90 is the
      most informative credit this pipeline can spend, because it is the read
      closest to kickoff. It is also a real change to the monthly credit
      profile, so it stays OFF until an operator turns it on -- and it goes
      through ``pull_week_props``, so the budget hard stop and the credit
      ledger apply exactly as they do on Wednesday.

    The two halves never touch the same game: a game either has a line
    (resnap) or it does not (first pull).

    Degrades, never aborts: the re-rank IS the product, so a dead odds call
    costs real prices and nothing else. The reason lands in ``note`` so a
    published ``no_market`` is never mistaken for "the model had nothing to
    say".
    """
    from nflvalue import db as dbmod
    out = {"resnapped": [], "first_pulled": [], "skipped_budget": [],
           "without_lines": [], "first_pull_enabled": False, "note": ""}
    notes = []
    if not cfg.get("odds_api_key"):
        out["note"] = ("no odds_api_key configured — T-90 runs on synthetic "
                       "reference lines (no_market), which the report labels")
        return out

    ob = cfg.get("odds_budget") or {}
    out["first_pull_enabled"] = bool(ob.get("t90_first_pull"))
    game_ids = [g.game_id for g in soon.itertuples(index=False)]
    have = set(dbmod.query_df(conn, "SELECT DISTINCT game_id FROM lines")["game_id"].tolist())
    with_lines = [g for g in game_ids if g in have]
    without_lines = [g for g in game_ids if g not in have]
    out["without_lines"] = without_lines

    if event_map_fn is None:
        import pipeline_weekly as pwmod
        event_map_fn = pwmod.build_event_map
    if resnap is None or pull is None:
        from nflvalue.sources import oddsapi_props as oap
        resnap = resnap or oap.resnap_lines
        pull = pull or oap.pull_week_props

    skipped = set()

    def _spend(label, fn, targets):
        if not targets:
            return []
        try:
            emap = event_map_fn(cfg, soon[soon.game_id.isin(targets)])
            if not emap:
                notes.append(f"{label}: no odds-api event matched {len(targets)} game(s)")
                return []
            res = fn(cfg, emap, conn=conn)
        except Exception as exc:  # noqa: BLE001 -- degrade to synthetic, loudly
            notes.append(f"{label} failed ({exc})")
            return []
        skipped.update(res.get("skipped_budget") or [])
        skipped.update(res.get("skipped_cap") or [])
        for e in res.get("skipped_error") or []:
            notes.append(f"{label} error on {e.get('game_id')}: {e.get('error')}")
        notes.append(f"{label}: {len(res.get('pulled') or [])} game(s), "
                     f"{res.get('rows_written', 0)} rows, "
                     f"{float(res.get('budget_remaining') or 0):.0f} credits left")
        return sorted(res.get("pulled") or [])

    out["resnapped"] = _spend("closing resnap", resnap, with_lines)
    if without_lines:
        if out["first_pull_enabled"]:
            out["first_pulled"] = _spend("T-90 first pull", pull, without_lines)
        else:
            notes.append(
                f"{len(without_lines)} game(s) still have no real line and will "
                f"publish no_market: {', '.join(without_lines)} "
                f"(enable odds_budget.t90_first_pull to price them at T-90)")
    out["skipped_budget"] = sorted(skipped)
    if skipped:
        notes.append(f"budget/cap stop skipped {len(skipped)} game(s): "
                     f"{', '.join(sorted(skipped))}")
    out["note"] = "; ".join(notes)
    return out


def job_t90() -> int:
    from nflvalue import config as cfgmod, db as dbmod
    import pipeline_weekly as pw
    slate = load_slate()
    now = now_et()
    soon = slate[(slate["kickoff"] > now)
                 & (slate["kickoff"] <= now + dt.timedelta(hours=T90_WINDOW_HOURS))]
    if soon.empty:
        print("[auto] no kickoffs within the T-90 window — no-op")
        write_pipeline_heartbeat(
            schedule_status(slate, now), "T-90 check completed; no kickoff is currently due.", "t90")
        return 0
    conn = dbmod.connect()
    done = set(dbmod.query_df(conn, "SELECT DISTINCT game_id FROM leans WHERE clock='t90'")
               ["game_id"].tolist())
    cfg = cfgmod.load_config()

    # LINE SNAPSHOT before the re-rank: the closing resnap that makes CLV
    # resolvable, plus (opt-in) a first pull for games Wednesday never
    # priced. run_t90 then re-ranks against whatever is freshest in `lines`.
    snap = t90_line_snapshot(cfg, conn, soon)
    if snap["note"]:
        print(f"[auto] t90 lines — {snap['note']}")
    conn.close()
    from nflvalue.notify import resolve_webhook
    post_live = bool(cfg.get("discord_enabled") and resolve_webhook())
    inputs = None
    failures = []
    for g in soon.itertuples(index=False):
        if g.game_id in done:
            continue
        try:
            if inputs is None:
                from nflvalue.candidates import build_week_inputs
                inputs = build_week_inputs()
            res = pw.run_t90(int(g.season), int(g.week), g.game_id, mode="live",
                             inputs=inputs, discord=True, discord_dry_run=not post_live)
            print(f"[auto] t90 {g.game_id}: {len(res['voided'])} voided")
        except Exception as exc:  # noqa: BLE001 -- one bad game must not skip the rest
            print(f"[auto] t90 {g.game_id} FAILED: {exc}")
            failures.append(g.game_id)
    if failures:
        print(f"[auto] T-90 failed for {len(failures)} game(s): {', '.join(failures)}")
        return 1
    write_pipeline_heartbeat(
        "active", f"T-90 refresh completed for {len(soon)} due game(s).", "t90")
    return 0


def job_tuesday() -> int:
    import subprocess
    import pipeline_weekly as pw
    from nflvalue import killcheck
    slate = load_slate()
    lw = last_completed_week(slate, now_et())
    if lw is None:
        print("[auto] no completed week — no-op")
        write_pipeline_heartbeat(
            schedule_status(slate, now_et()), "Tuesday grade check completed; no week is ready.",
            "tuesday")
        return 0
    season, week = lw
    graded = pw.run_grade(season, week)
    print(f"[auto] graded {season} wk{week}: {graded['graded']} leans, "
          f"hit {graded['hit_rate']}; misses: {graded['why'].get('recent_miss_reasons')}")
    clv = pw.resolve_clv(season, week)
    print(f"[auto] clv resolved {clv['resolved']}; kill-check {clv['killcheck']['verdict']}")
    # retrain the ML ranker on everything graded (frame append + refit)
    root = str(Path(__file__).resolve().parents[1])
    retrain_failed = False
    for cmd in ([sys.executable, "ml_test.py", "--stage", "frame",
                 "--seasons", str(season), "--append"],
                [sys.executable, "ml_test.py", "--stage", "fit"]):
        r = subprocess.run(cmd, cwd=root, capture_output=True, text=True, timeout=1800)
        print(f"[auto] {' '.join(cmd[1:])}: rc={r.returncode} {r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ''}")
        retrain_failed = retrain_failed or r.returncode != 0
    if retrain_failed:
        print("[auto] weekly ML retraining failed; refusing to publish partial production state")
        return 1
    write_pipeline_heartbeat(
        "active", f"Tuesday grading, CLV, and retraining completed for {season} week {week}.",
        "tuesday")
    return 0


def job_deploy() -> int:
    """Refresh public metadata without spending odds credits or notifying."""
    try:
        slate = load_slate()
        status = schedule_status(slate, now_et())
        detail = "Deployment completed without running the betting model."
    except Exception as exc:  # noqa: BLE001 - surface a readable status page
        status = "degraded"
        detail = f"Deployment completed, but schedule status could not be read: {exc}"
    write_pipeline_heartbeat(status, detail, "deploy")
    print(f"[auto] deploy-only dashboard refresh: {status}")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--job", choices=["deploy", "wed", "t90", "tuesday"], required=True)
    args = ap.parse_args()
    ensure_dependencies()
    raise SystemExit({"deploy": job_deploy, "wed": job_wed, "t90": job_t90,
                      "tuesday": job_tuesday}[args.job]())


if __name__ == "__main__":
    main()
