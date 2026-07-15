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

    # CLOSING SNAPSHOT (evaluation catch): without a second pre-kick line
    # pull, entry == close and CLV could never resolve — the kill-check
    # would starve forever. Resnap exactly the games that have entry lines.
    if cfg.get("odds_api_key"):
        try:
            from nflvalue.sources import oddsapi_props as oap
            import pipeline_weekly as pwmod
            have_lines = set(dbmod.query_df(
                conn, "SELECT DISTINCT game_id FROM lines")["game_id"].tolist())
            targets = [g.game_id for g in soon.itertuples(index=False)
                       if g.game_id in have_lines]
            if targets:
                emap = pwmod.build_event_map(cfg, soon[soon.game_id.isin(targets)])
                res = oap.resnap_lines(cfg, emap, conn=conn)
                print(f"[auto] closing resnap: {len(res['pulled'])} game(s), "
                      f"{res['rows_written']} rows, {res['budget_remaining']:.0f} credits left")
        except Exception as exc:  # noqa: BLE001
            print(f"[auto] closing resnap failed (CLV close may be stale): {exc}")
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
