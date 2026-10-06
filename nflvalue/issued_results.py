"""Results-only settlement of the issued-pick ledger: official finals -> durable grades -> public export.

This is the scheduled half of ``issued_grading``. It reads the ledger the production state holds
(``issued_picks`` + ``issued_pick_events``), reads ESPN's public scoreboard for each week that has
issued records, fetches the final box of each completed game, grades with
``issued_grading.grade`` and appends what it reached to ``issued_results`` (db migration 8). It
never fits, trains, prices, pulls odds or sends anything; ``issued_picks`` -- the frozen
predictions, lines, prices and clocks -- is only read.

Rules:

* Only an official final settles. A scheduled, live, postponed or canceled game -- or a game the
  scoreboard does not list -- leaves its records PENDING; nothing is zeroed or voided for it.
* Inside a final box a missing player or stat stays UNRESOLVED (``issued_grading``), never zero.
* Append-only and idempotent. A result's id covers (record, section, settlement, actual, hit), so
  re-reading the same final writes nothing. A stat correction is a NEW row naming the row it
  supersedes; the export shows the current value and every correction beside it.
* Every grade keeps the ledger's section (given before kickoff / recorded after kickoff / watch /
  generated-not-shown) and policy class: losses and policy-violating sent picks stay in the record.
* Bounded: weeks whose issued records were all recorded more than ``LOOKBACK_DAYS`` ago are not
  read; a final game is re-read at most every ``RECHECK_HOURS`` and only within
  ``CORRECTION_DAYS`` of kickoff; at most ``MAX_REQUESTS`` requests per run, each retried at
  most ``RETRIES`` times.
* Every capture (accepted or refused) is recorded in ``result_captures`` with its URL, clock and
  sha256. Box bytes are kept as run evidence, not in production state.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import tempfile
import time
import urllib.request
from typing import Callable, Dict, List, Optional

from . import issued_grading as ig
from . import issued_ledger as il

SITE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
LOOKBACK_DAYS = 9
RECHECK_HOURS = 20
CORRECTION_DAYS = 4
MAX_REQUESTS = 40
RETRIES = 2
SECTIONS = ("recommendations_given", "retrospective", "watch_published", "generated_not_shown")
PUBLIC_FIELDS = ("record_id", "revision", "section", "season", "week", "game_id", "kickoff", "slate_tag",
                 "player_name", "market", "side", "line", "tier", "pick_class", "card_status", "quote_book",
                 "quote_price", "quote_ts", "decision_ts", "model_p_side", "mean", "evidence_stage",
                 "first_seen_in_ledger", "delivery_evidence_kind", "policy_class", "settlement", "hit",
                 "actual", "detail", "espn_event", "actuals_url", "actuals_sha256", "actuals_captured_at",
                 "graded_at", "supersedes")
FINAL = "STATUS_FINAL"


def _iso(t: dt.datetime) -> str:
    return t.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


def fetch(url: str, timeout: float = 20.0, attempts: int = RETRIES + 1,
          sleep: Callable[[float], None] = time.sleep) -> bytes:
    """GET ``url`` -> bytes, retried a bounded number of times. Raises on the last failure."""
    from .sources._http import user_agent
    last = None
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": user_agent(),
                                                       "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001 -- retried, then reported
            last = exc
            if i + 1 < attempts:
                sleep(2.0 * (i + 1))
    raise RuntimeError(f"GET {url} failed after {attempts} attempt(s): {last}")


def scoreboard_url(season: int, week: int) -> str:
    return f"{SITE}/scoreboard?seasontype=2&week={int(week)}&dates={int(season)}"


def summary_url(event: str) -> str:
    return f"{SITE}/summary?event={event}"


def scoreboard_games(raw: bytes, season: int, week: int) -> Dict[str, Dict]:
    """{canonical game id: {event, kickoff, status, completed}} from one ESPN scoreboard."""
    out = {}
    for ev in json.loads(raw).get("events") or []:
        c = (ev.get("competitions") or [{}])[0]
        teams = {t.get("homeAway"): ig.ALIAS.get(t["team"]["abbreviation"], t["team"]["abbreviation"])
                 for t in c.get("competitors") or []}
        if set(teams) != {"home", "away"}:
            continue
        status = ((c.get("status") or ev.get("status") or {}).get("type") or {})
        out[f"{int(season)}_{int(week):02d}_{teams['away']}_{teams['home']}"] = {
            "event": str(ev.get("id") or c.get("id")), "kickoff": c.get("date") or ev.get("date"),
            "status": status.get("name"), "completed": bool(status.get("completed"))}
    return out


def _issued(records: List[Dict]) -> List[Dict]:
    return [r for r in records if r.get("pick_class") in il.ISSUED_CLASSES]


def due_weeks(records: List[Dict], now: dt.datetime) -> List[tuple]:
    """(season, week) pairs with an issued record recorded inside the lookback window."""
    floor = now - dt.timedelta(days=LOOKBACK_DAYS)
    latest: Dict[tuple, dt.datetime] = {}
    for r in _issued(records):
        t = ig._ts(r.get("recorded_at"))
        if t is None:
            continue
        k = (int(r["season"]), int(r["week"]))
        latest[k] = max(latest.get(k, t), t)
    return sorted(k for k, t in latest.items() if t >= floor)


def _last_accepted(conn, game_id: str) -> Optional[dt.datetime]:
    row = conn.execute("SELECT MAX(captured_at) FROM result_captures WHERE game_id=? AND accepted=1",
                       (game_id,)).fetchone()
    return ig._ts(row[0]) if row and row[0] else None


def _capture(conn, game_id, event, url, captured_at, raw, status_name, accepted, reason, game, recorded_at):
    sha = hashlib.sha256(raw).hexdigest() if raw is not None else None
    cur = conn.execute(
        "INSERT OR IGNORE INTO result_captures (capture_id, game_id, espn_event, url, captured_at, sha256, "
        "status_name, accepted, reason, away_score, home_score, recorded_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (_sha(game_id, url, sha, captured_at), game_id, event, url, captured_at, sha, status_name,
         int(accepted), reason, game and game.get("away_score"), game and game.get("home_score"), recorded_at))
    return cur.rowcount


def _persist(conn, res: Dict, url_of: Dict[str, str], graded_at: str) -> int:
    """Append each final-box grade; an unchanged grade writes nothing, a changed one supersedes."""
    written = 0
    for section in SECTIONS:
        for row in (res["sections"].get(section) or {}).get("rows") or []:
            if not row.get("actuals_sha256"):
                continue                                  # no verified final box: still pending
            row = {k: v for k, v in row.items() if k not in ("actuals_source", "exception")}
            row["actuals_url"] = url_of.get(row["game_id"])
            rid = _sha(row["record_id"], section, row["settlement"], row["actual"], row["hit"])
            prev = conn.execute("SELECT result_id FROM issued_results WHERE record_id=? AND section=? "
                                "ORDER BY graded_at DESC, rowid DESC LIMIT 1",
                                (row["record_id"], section)).fetchone()
            if prev and prev[0] == rid:
                continue
            if conn.execute("SELECT 1 FROM issued_results WHERE result_id=?", (rid,)).fetchone():
                # flipped back to an earlier value: record the reversion as its own row
                rid = _sha(row["record_id"], section, row["settlement"], row["actual"], row["hit"], graded_at)
            conn.execute(
                "INSERT INTO issued_results (result_id, record_id, section, season, week, game_id, settlement, "
                "hit, actual, actuals_url, actuals_sha256, actuals_captured_at, supersedes, row_json, graded_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (rid, row["record_id"], section, row["season"], row["week"], row["game_id"], row["settlement"],
                 row["hit"], row["actual"], row["actuals_url"], row["actuals_sha256"], row["actuals_captured_at"],
                 prev[0] if prev else None, json.dumps(row, sort_keys=True, default=str), graded_at))
            written += 1
    return written


def settle(conn, now: Optional[dt.datetime] = None, http: Optional[Callable[[str], bytes]] = None,
           evidence_dir: Optional[str] = None) -> Dict:
    """One bounded results-only pass. Returns a summary; raises only on a ledger/DB failure."""
    now = now or dt.datetime.now(dt.timezone.utc)
    http = http or fetch
    stamp = _iso(now)
    records = il.load(conn)
    weeks = due_weeks(records, now)
    out = {"checked_at": stamp, "weeks": [list(w) for w in weeks], "requests": 0, "captures": 0,
           "results_written": 0, "pending": [], "errors": [], "skipped_recent": [], "graded_games": []}
    if not weeks:
        out["written"] = 0
        return out
    tmp = tempfile.mkdtemp(prefix="results-boxes-")
    url_of, paths = {}, []

    def get(url):
        if out["requests"] >= MAX_REQUESTS:
            raise RuntimeError(f"request budget {MAX_REQUESTS} reached; {url} left for the next run")
        out["requests"] += 1
        return http(url)

    try:
        for season, week in weeks:
            mine = [r for r in _issued(records) if (int(r["season"]), int(r["week"])) == (season, week)]
            out["pending"].extend({"game_id": None, "record_id": r["record_id"], "reason": "record has no game id"}
                                  for r in mine if not r.get("game_id"))
            wanted = sorted({str(r["game_id"]) for r in mine if r.get("game_id")})
            try:
                board = scoreboard_games(get(scoreboard_url(season, week)), season, week)
            except Exception as exc:  # noqa: BLE001 -- pending, retried next run
                out["errors"].append(f"{season} wk{week} scoreboard: {exc}")
                out["pending"].extend({"game_id": g, "reason": "scoreboard unavailable"} for g in wanted)
                continue
            for gid in wanted:
                info = board.get(gid)
                if info is None:
                    out["pending"].append({"game_id": gid, "reason": "not on the official scoreboard"})
                    continue
                if not (info["completed"] and info["status"] == FINAL):
                    out["pending"].append({"game_id": gid, "reason": f"not final ({info['status']})"})
                    continue
                last, kick = _last_accepted(conn, gid), ig._ts(info["kickoff"])
                if last is not None and (now - last < dt.timedelta(hours=RECHECK_HOURS) or kick is None
                                         or now - kick > dt.timedelta(days=CORRECTION_DAYS)):
                    out["skipped_recent"].append(gid)
                    continue
                url = summary_url(info["event"])
                try:
                    raw = get(url)
                except Exception as exc:  # noqa: BLE001
                    out["errors"].append(f"{gid} summary: {exc}")
                    out["pending"].append({"game_id": gid, "reason": "final box fetch failed"})
                    continue
                path = os.path.join(tmp, f"{gid}.json")
                with open(path, "wb") as fh:
                    fh.write(raw)
                if evidence_dir:
                    os.makedirs(evidence_dir, exist_ok=True)
                    with open(os.path.join(evidence_dir, f"{gid}-{stamp.replace(':', '')}.json"), "wb") as fh:
                        fh.write(raw)
                url_of[gid], paths = url, paths + [path]
        boxes = ig.load_boxes(paths, stamp)
        by_path = {os.path.join(tmp, f"{g}.json"): g for g in url_of}
        for rej in boxes["rejected"]:
            gid = by_path.get(rej["source"])
            out["captures"] += _capture(conn, gid, None, url_of.get(gid), stamp,
                                        open(rej["source"], "rb").read(), None, False, rej["reason"], None, stamp)
            out["pending"].append({"game_id": gid, "reason": f"final box refused: {rej['reason']}"})
            rej["source"] = url_of.get(gid)
        for gid, g in boxes["games"].items():
            out["captures"] += _capture(conn, gid, g["espn_event"], url_of[gid], stamp,
                                        open(g["source"], "rb").read(), g.get("status_name"), True, None, g, stamp)
            out["graded_games"].append(gid)
        if boxes["games"]:
            graded = ig.grade([r for r in records if r["game_id"] in boxes["games"]], boxes)
            out["results_written"] = _persist(conn, graded, url_of, stamp)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        for p in paths:
            os.unlink(p)
        os.rmdir(tmp)
    out["written"] = out["results_written"] + out["captures"]
    return out


def current(conn) -> List[Dict]:
    """The latest result per (record, section), with the rows it superseded."""
    rows = conn.execute("SELECT result_id, record_id, section, row_json, graded_at, supersedes "
                        "FROM issued_results ORDER BY graded_at, rowid").fetchall()
    latest, by_id = {}, {}
    for rid, rec, sec, rj, ga, sup in rows:
        r = {**json.loads(rj), "section": sec, "graded_at": ga, "supersedes": sup, "result_id": rid}
        by_id[rid] = r
        latest[(rec, sec)] = r
    for r in latest.values():
        chain, sup = [], r["supersedes"]
        while sup and sup in by_id:
            p = by_id[sup]
            chain.append({"settlement": p["settlement"], "actual": p["actual"], "graded_at": p["graded_at"],
                          "actuals_captured_at": p.get("actuals_captured_at")})
            sup = p["supersedes"]
        r["corrections"] = chain
    return list(latest.values())


def export(conn, checked_at: Optional[str] = None) -> Dict:
    """Public, deterministic results document. Issued values are shown exactly as recorded."""
    try:
        cur = current(conn)
        pending_ids = {r["record_id"] for r in _issued(il.load(conn))} - {r["record_id"] for r in cur}
        pending = [r for r in _issued(il.load(conn)) if r["record_id"] in pending_ids]
    except Exception:  # noqa: BLE001 -- a database without the ledger/results tables
        cur, pending = [], []
    sections = {}
    for s in SECTIONS:
        rows = sorted((r for r in cur if r["section"] == s),
                      key=lambda r: (r["season"], r["week"], r["game_id"] or "", r.get("player_name") or "",
                                     r["market"] or "", r["record_id"]))
        counts = {k: sum(r["settlement"] == k for r in rows) for k in ("win", "loss", "push", "void", "unresolved")}
        sections[s] = {"counts": counts,
                       "rows": [{**{k: r.get(k) for k in PUBLIC_FIELDS}, "corrections": r["corrections"]}
                                for r in rows]}
    checked_at = checked_at or max((r["graded_at"] for r in cur), default=None)
    return {"schema": "fablesfable.issued_results.v1", "results_checked_at": checked_at,
            "universe": "issued-pick ledger only; sections are never pooled",
            "note": ("Lines, prices, probabilities and decision clocks are the issued values; nothing here "
                     "is a new forecast. Only official final box scores settle a pick; a missing player "
                     "is unresolved, never zero. Losses, pushes and policy-violating sent picks are kept."),
            "book_rules_unverified": list(ig.st.BOOK_RULES_UNVERIFIED),
            "sections": sections,
            "pending": sorted(({"record_id": r["record_id"], "season": r["season"], "week": r["week"],
                                "game_id": r["game_id"], "player_name": r.get("player_name"),
                                "market": r["market"], "side": r.get("side"), "line": r.get("line"),
                                "pick_class": r.get("pick_class"), "tier": r.get("tier")} for r in pending),
                              key=lambda r: (r["season"], r["week"], r["game_id"] or "", r["record_id"]))}


def export_readonly(db_path: str, checked_at: Optional[str] = None) -> Dict:
    import sqlite3
    if not os.path.isfile(db_path):
        return export_empty(checked_at)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='issued_results'").fetchone():
            return export_empty(checked_at)
        return export(conn, checked_at)
    finally:
        conn.close()


def export_empty(checked_at: Optional[str] = None) -> Dict:
    return {"schema": "fablesfable.issued_results.v1", "results_checked_at": checked_at,
            "universe": "issued-pick ledger only; sections are never pooled",
            "note": "No settled issued picks are recorded in this production state yet.",
            "book_rules_unverified": list(ig.st.BOOK_RULES_UNVERIFIED),
            "sections": {s: {"counts": {}, "rows": []} for s in SECTIONS}, "pending": []}
