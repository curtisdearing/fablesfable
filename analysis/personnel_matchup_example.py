"""Explicit acquisition adapter + worked example for nflvalue.personnel_matchup.

Separated from the pure builder on purpose: this script parses raw pages that
were ALREADY fetched (official NFL.com schedule/injury pages, official club
injury-report / depth-chart pages and club news articles, nflverse games.csv)
from a raw directory with a FETCH_LOG.txt, maps names to nflverse gsis ids via
a roster file (parquet cache or nflverse roster_YYYY.csv), and runs
``build_personnel_evidence``.  No network access here.

    python analysis/personnel_matchup_example.py RAW_DIR ROSTERS OUT_JSON \
        --game 2026_03_ATL_GB --as-of <UTC clock AFTER the last fetch> \
        [--report-article ATL:path] [--starter-article ATL:path]

Mode: LIVE CONTEXT RETRIEVAL.  Undated pages carry ``captured_at`` (the real
fetch clock), which bounds availability for this capture only; the script
refuses an ``--as-of`` earlier than any fetch it uses.  It is not a
historical as-of replay: a page captured today says nothing about what was
visible before a past kickoff.

Every parser returns ``(rows, meta)``; ``meta["complete"]`` is False when any
text in the parsed table could not be assigned to a row, and incomplete
sources are excluded from the builder rather than treated as full coverage.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import html
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nflvalue.personnel_matchup import build_personnel_evidence, norm_name  # noqa: E402

TEAM_SITE = {"ATL": "atlantafalcons_com", "GB": "packers_com"}
TEAM_NICK = {"ATL": "Falcons", "GB": "Packers"}
OFF_SLOTS = {"QB", "RB", "WR", "TE", "LT", "LG", "C", "RG", "RT"}
DEF_SLOTS = {"DE", "DL", "DT", "EDGE", "LB", "ILB", "OLB", "CB", "S", "NB"}
POS_CODES = OFF_SLOTS | DEF_SLOTS | {"FB", "T", "OT", "G", "OG", "OL", "NT", "MLB", "SS", "FS",
                                     "DB", "K", "P", "LS"}
PRACTICE = r"(DNP|LP|FP|\(-\))"
GAME_STATUS = r"(OUT|DOUBTFUL|QUESTIONABLE|UNSPECIFIED|\(-\))"
NFLCOM_PRACTICE = (r"(Did Not Participate In Practice|Limited Participation in Practice|"
                   r"Full Participation in Practice)")
TEAM_HDR = "Player Position Injury Wed Thu Fri Game Status"


def _text(path):
    s = open(path, errors="ignore").read()
    s = re.sub(r"<script.*?</script>|<style.*?</style>", "", s, flags=re.S)
    return html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", s)))


def _fetch_log(raw):
    log = {}
    for line in open(os.path.join(raw, "FETCH_LOG.txt")):
        parts = line.split()
        if len(parts) >= 5 and parts[1] == "200":
            # signed CDN redirects carry short-lived tokens: keep host+path only
            log[parts[-1]] = {"fetched_at": parts[0], "effective_url": parts[2].split("?")[0],
                              "sha256_16": parts[-2]}
    return log


def _raw(raw, token):
    hits = sorted(glob.glob(os.path.join(raw, f"*{token}*.raw")))
    return hits[0] if hits else None


def _src(log, needle):
    hits = [v for k, v in log.items() if needle in k]
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one fetch-log entry containing {needle!r}, "
                         f"found {len(hits)}")
    return hits[0]


def _split_name_pos(chunk):
    toks = chunk.split()
    idx = [n for n, x in enumerate(toks) if n > 0 and x in POS_CODES]
    if not idx:
        return None
    p = idx[0]
    return " ".join(toks[:p]), toks[p], " ".join(toks[p + 1:])


def parse_team_report(path):
    """First table on an official club injury-report page = the club's own
    players (Wed/Thu/Fri practice + game status)."""
    t = _text(path)
    i = t.find(TEAM_HDR)
    if i < 0:
        return [], {"complete": False, "unparsed": ["table header not found"]}
    seg = t[i + len(TEAM_HDR):]
    seg = re.split(r" (?:(?!(?:OUT|DOUBTFUL|QUESTIONABLE|UNSPECIFIED|DNP|LP|FP)\b)[A-Z][\w.]* ){0,3}"
                   r"[A-Z][\w.]* Table - Injury report| Legend ", seg)[0]
    pat = re.compile(r"\s*(?P<body>.+?) " + PRACTICE + " " + PRACTICE + " " + PRACTICE
                     + " " + GAME_STATUS + r"(?= |$)")
    rows, unparsed, pos = [], [], 0
    for m in pat.finditer(seg):
        gap = seg[pos:m.start()].strip()
        if gap:
            unparsed.append(gap)
        pos = m.end()
        np_ = _split_name_pos(m.group("body"))
        if np_ is None or re.search(r"\b(DNP|LP|FP|OUT|QUESTIONABLE|DOUBTFUL)\b", m.group("body")):
            unparsed.append(m.group(0).strip())
            continue
        rows.append({"name": np_[0], "position_reported": np_[1], "injury": np_[2],
                     "practice": [m.group(2), m.group(3), m.group(4)],
                     "report_status": m.group(5).title()})
    if seg[pos:].strip():
        unparsed.append(seg[pos:].strip())
    return rows, {"complete": not unparsed, "unparsed": unparsed}


def parse_nflcom_report(path, nickname):
    """NFL.com league page: the section headed by the club nickname.  Rows are
    delimited by the practice phrase, so an injury text can never swallow the
    next player's row."""
    t = _text(path)
    hdr = f"{nickname} Player Position Injuries Practice Status Game Status"
    k = t.find(hdr)
    if k < 0:
        return [], {"complete": False, "unparsed": ["section not found"]}
    seg = t[k + len(hdr):]
    seg = re.split(r" \d{1,2}:\d\d [AP]M | [A-Z][a-z]+ Player Position Injuries", seg)[0]
    parts = re.split(NFLCOM_PRACTICE + r"(?: (Out|Doubtful|Questionable)(?= |$))?", seg)
    rows, unparsed = [], []
    for n in range(0, len(parts) - 1, 3):
        np_ = _split_name_pos(parts[n].strip())
        if np_ is None:
            unparsed.append(parts[n].strip())
            continue
        rows.append({"name": np_[0], "position_reported": np_[1], "injury": np_[2],
                     "practice": [parts[n + 1]], "report_status": parts[n + 2] or "(-)"})
    if parts[-1].strip():
        unparsed.append(parts[-1].strip())
    return rows, {"complete": not unparsed, "unparsed": unparsed}


def _article_meta(s):
    for m in re.finditer(r'<script[^>]*ld\+json[^>]*>(.*?)</script>', s, flags=re.S):
        try:
            d = json.loads(m.group(1))
        except ValueError:
            continue
        for x in d if isinstance(d, list) else [d]:
            if x.get("@type") == "NewsArticle":
                return {"headline": x.get("headline"), "published_at": x.get("datePublished"),
                        "modified_at": x.get("dateModified"),
                        "author": (x.get("author") or {}).get("name")}
    return {}


def _article_body(s):
    b = re.sub(r"<script.*?</script>|<style.*?</style>", "", s, flags=re.S)
    t = html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", b)))
    i = t.find(" — ")
    return t[max(0, i - 40):] if i >= 0 else t


def parse_practice_article(path):
    """Club article listing an (estimated) practice report as
    ``Full Participation / Limited Participation / Did Not Participate``
    sections of ``POS Name (injury)`` items.  Practice only: no game status."""
    s = open(path, errors="ignore").read()
    meta = _article_meta(s)
    body = _article_body(s)
    heads = [(m.start(), m.group(0)) for m in re.finditer(
        r"Full Participation|Limited Participation|Did Not Participate", body)]
    rows = []
    for n, (start, label) in enumerate(heads):
        end = heads[n + 1][0] if n + 1 < len(heads) else len(body)
        seg = body[start + len(label):end]
        items = list(re.finditer(r"\b([A-Z]{1,4}) ([A-Z][\w.'’\- ]+?) \(([^)]+)\)", seg))
        for m in items:
            if m.group(1) in POS_CODES:
                rows.append({"name": m.group(2).strip(), "position_reported": m.group(1),
                             "injury": m.group(3), "practice": [label],
                             "report_status": "practice_only_no_game_status"})
        if n + 1 == len(heads) and items:
            body_tail = seg[items[-1].end():].strip()
            meta["tail_after_last_item"] = body_tail[:80]
    meta["sections"] = [h[1] for h in heads]
    meta["complete"] = len(heads) == 3 and bool(rows)
    return rows, meta


def parse_starter_article(path):
    """Club article whose headline AND body state that a named QB will start."""
    s = open(path, errors="ignore").read()
    meta = _article_meta(s)
    body = _article_body(s)
    m = re.match(r"(.+?) named starting QB", meta.get("headline") or "")
    if not m:
        return None, meta
    name = m.group(1).strip()
    sent = re.search(re.escape(name) + r"[^.]{0,80}named[^.]{0,40}starting quarterback[^.]*\.",
                     body)
    if not sent:
        return None, dict(meta, reason="headline not corroborated by body sentence")
    return {"name": name, "evidence_sentence": sent.group(0)}, meta


def parse_depth(path, roster_names):
    """Official club depth chart -> [{slot, rank, name}] (names split by
    jersey number when present, else by greedy match to roster names)."""
    t = _text(path)
    slots = OFF_SLOTS | DEF_SLOTS
    out, counter = [], {}
    # nav bars repeat "Offense/Defense/Special Teams": start at the table header
    m = re.search(r"Position (?:1st|Player)", t)
    begin = m.start() if m else 0
    stop = min([x for x in (t.find("Special Teams Special Teams", begin),
                            t.find("Special Teams Position", begin),
                            t.find("SPECIAL TEAMS", begin)) if x > 0] or [len(t)])
    body = t[begin:stop]
    parts = re.split(r" (%s) " % "|".join(sorted(slots, key=len, reverse=True)), body)
    for n in range(1, len(parts) - 1, 2):
        slot, seg = parts[n], parts[n + 1]
        if seg.startswith(("Player", "1st")) or len(seg) > 200:
            continue
        seg = re.sub(r"\b(Offense|Defense|DEFENSE|OFFENSE|Position|Player|1st|2nd|3rd|4th|5th|6th)\b",
                     " ", seg)
        if re.search(r"\d", seg):
            names = [x.strip() for x in re.split(r"\s*\d+\s+", seg) if x.strip()]
        else:
            names, words = [], seg.split()
            while words:
                for L in range(min(5, len(words)), 0, -1):
                    cand = " ".join(words[:L])
                    if norm_name(cand) in roster_names or L == 1:
                        break
                if norm_name(cand) not in roster_names:
                    cand = " ".join(words[:2])       # unresolved; fails safe on id lookup
                    L = min(2, len(words))
                names.append(cand)
                words = words[L:]
        counter[slot] = counter.get(slot, 0) + 1
        key = slot if counter[slot] == 1 else f"{slot}{counter[slot]}"
        for rank, nm in enumerate(names, 1):
            out.append({"slot": key, "rank": rank, "name": nm})
    return out


def _ts(s):
    return dt.datetime.fromisoformat(str(s).replace("Z", "+00:00"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("rosters")
    ap.add_argument("out")
    ap.add_argument("--game", required=True)
    ap.add_argument("--as-of", required=True)
    ap.add_argument("--report-article", action="append", default=[])
    ap.add_argument("--starter-article", action="append", default=[])
    a = ap.parse_args()
    import pandas as pd

    log = _fetch_log(a.raw)
    as_of = _ts(a.as_of)
    late = [(u, v["fetched_at"]) for u, v in log.items() if _ts(v["fetched_at"]) > as_of]
    if late:
        raise SystemExit(f"--as-of {a.as_of} precedes fetched inputs: {late}")
    games = list(csv.DictReader(open(_raw(a.raw, "games_csv"))))
    g = next(r for r in games if r["game_id"] == a.game)
    home, away = g["home_team"], g["away_team"]
    sched_path = _raw(a.raw, "nfl_com_schedules")
    starts = re.findall(r'startTime\\":\\"([0-9TZ:\-]+)\\"', open(sched_path, errors="ignore").read())
    # nflverse gametime is US/Eastern; September kickoffs are EDT (UTC-4)
    local = dt.datetime.fromisoformat(f"{g['gameday']}T{g['gametime']}:00")
    cand_utc = (local + dt.timedelta(hours=4)).strftime("%Y-%m-%dT%H:%M:%SZ")
    kickoff = cand_utc if cand_utc in starts else None
    sched_src = _src(log, "nfl.com/schedules")

    if not a.rosters.endswith(".parquet"):
        ros = pd.read_csv(a.rosters, dtype=str).rename(columns={"gsis_id": "player_id"})
        ros["position"] = ros["depth_chart_position"].fillna(ros["position"])
        ros["week"] = ros["week"].astype(int)
        ros["season"] = ros["season"].astype(int)
        roster_src = _src(log, "roster_")
    else:
        ros = pd.read_parquet(a.rosters)
        roster_src = {"fetched_at": None, "effective_url": a.rosters}
    ros = ros.dropna(subset=["player_id"])
    last = ros[ros["season"] == ros["season"].max()]
    last = last[last["week"] == last["week"].max()]
    roster, names_by_team, roster_status = [], {}, {}
    for r in last[last["team"].isin([home, away])].itertuples(index=False):
        roster.append({"team": r.team, "player_id": r.player_id, "name": r.full_name,
                       "position": r.position})
        names_by_team.setdefault(r.team, set()).add(norm_name(r.full_name))
        roster_status[r.player_id] = getattr(r, "status", None)
    by_name = {}
    for p in roster:
        by_name.setdefault((p["team"], norm_name(p["name"])), []).append(p["player_id"])

    def pid_of(team, name):
        ids = by_name.get((team, norm_name(name)), [])
        return ids[0] if len(ids) == 1 else None

    prior_game = {}
    for t in (home, away):
        pg = [r for r in games if r["season"] == g["season"]
              and int(r["week"]) == int(g["week"]) - 1 and t in (r["home_team"], r["away_team"])]
        prior_game[t] = pg[0]["game_id"] if pg else None

    depth, prior, parse_meta, report_rows = [], [], {}, {}
    league_path = _raw(a.raw, "nfl_com_injuries")
    lsrc = _src(log, "nfl.com/injuries")
    for t in (home, away):
        site = TEAM_SITE[t]
        dsrc = _src(log, f"{site.replace('_', '.')}/team/depth-chart")
        for d in parse_depth(_raw(a.raw, f"{site}_team_depth"), names_by_team.get(t, set())):
            depth.append({"team": t, "name": d["name"], "slot": d["slot"], "rank": d["rank"],
                          "source_id": f"{site}_depth_chart", "source_url": dsrc["effective_url"],
                          "captured_at": dsrc["fetched_at"], "published_at": None})
        rsrc = _src(log, f"{site.replace('_', '.')}/team/injury-report")
        team_rows, tmeta = parse_team_report(_raw(a.raw, f"{site}_team_injury"))
        league_rows, lmeta = parse_nflcom_report(league_path, TEAM_NICK[t])
        parse_meta[t] = {"team_site": tmeta, "nfl_com": lmeta}
        report_rows[t] = {"team_site": team_rows, "nfl_com": league_rows}
        for src_id, rows, src, meta in ((f"{site}_injury_report", team_rows, rsrc, tmeta),
                                        ("nflcom_injuries_prior_week", league_rows, lsrc, lmeta)):
            if not meta["complete"]:
                continue                  # incomplete extraction is not coverage
            for r in rows:
                # the prior-week final report: clock is capture only, so rows
                # from different sources can never be ordered as updates
                prior.append({"game_id": prior_game[t], "team": t, "name": r["name"],
                              "report_status": r["report_status"], "practice": r["practice"],
                              "source_id": src_id, "source_url": src["effective_url"],
                              "captured_at": src["fetched_at"], "published_at": None})

    availability, report_index, claims, article_meta = [], [], [], []
    for spec in a.report_article:
        team, path = spec.split(":", 1)
        rows, meta = parse_practice_article(path)
        src = next(v for k, v in log.items() if os.path.basename(path).startswith(
            re.sub(r"[^a-zA-Z0-9]", "_", k)[:60]))
        article_meta.append(dict(meta, team=team, source_url=src["effective_url"],
                                 captured_at=src["fetched_at"], kind="practice_report",
                                 rows=len(rows)))
        if not meta["complete"] or not meta.get("published_at"):
            continue
        report_index.append({"game_id": a.game, "team": team, "published_at": meta["published_at"],
                             "source_id": f"{TEAM_SITE[team]}_article_practice_report",
                             "source_url": src["effective_url"], "captured_at": src["fetched_at"]})
        for r in rows:
            availability.append({"game_id": a.game, "team": team, "name": r["name"],
                                 "report_status": r["report_status"], "practice": r["practice"],
                                 "published_at": meta["published_at"],
                                 "captured_at": src["fetched_at"],
                                 "source_id": f"{TEAM_SITE[team]}_article_practice_report",
                                 "source_url": src["effective_url"]})
    for spec in a.starter_article:
        team, path = spec.split(":", 1)
        claim, meta = parse_starter_article(path)
        src = next(v for k, v in log.items() if os.path.basename(path).startswith(
            re.sub(r"[^a-zA-Z0-9]", "_", k)[:60]))
        article_meta.append(dict(meta, team=team, source_url=src["effective_url"],
                                 captured_at=src["fetched_at"], kind="starter_claim",
                                 claim=claim))
        if claim and pid_of(team, claim["name"]):
            claims.append({"game_id": a.game, "team": team, "player_id": pid_of(team, claim["name"]),
                           "claim": "expected_starter",
                           "attribution": f"{meta.get('author')} ({TEAM_SITE[team]})",
                           "published_at": meta.get("published_at"),
                           "captured_at": src["fetched_at"], "source_url": src["effective_url"],
                           "source_id": f"{TEAM_SITE[team]}_article_starter"})

    depth_ok, depth_unresolved = [], []
    for d in depth:
        pid = pid_of(d["team"], d["name"])
        if pid:
            depth_ok.append(dict(d, player_id=pid))
        else:
            depth_unresolved.append({"team": d["team"], "name": d["name"], "slot": d["slot"]})
    offensive = []
    for t in (home, away):
        for slot in ("QB", "RB", "WR", "TE"):
            d = next((x for x in depth_ok if x["team"] == t and x["slot"] == slot
                      and x["rank"] == 1), None)
            if d:
                offensive.append({"player_id": d["player_id"], "team": t, "game_id": a.game,
                                  "position": slot})

    # prior-week source differences: both pages are undated captures, so a
    # difference may be a later update on one page, not a contradiction
    differences = []
    for t, rr in report_rows.items():
        ts_ = {norm_name(r["name"]): r for r in rr["team_site"]}
        for r in rr["nfl_com"]:
            o = ts_.get(norm_name(r["name"]))
            if o and o["report_status"].lower() != r["report_status"].lower():
                differences.append({"team": t, "name": r["name"], "team_site": o["report_status"],
                                    "nfl_com": r["report_status"],
                                    "ordering": "unknown: neither page states a publication time"})

    # source-clock audit: nothing used may postdate as_of
    clocks = []
    for kind, rows in (("depth", depth_ok), ("prior", prior), ("availability", availability),
                       ("report_index", report_index), ("starter_claim", claims)):
        for r in rows:
            for fld in ("published_at", "captured_at"):
                if r.get(fld):
                    clocks.append((kind, fld, r[fld]))
    bad = [c for c in clocks if _ts(c[2]) > as_of]
    if bad:
        raise SystemExit(f"future inputs relative to as_of: {bad[:5]}")

    out = build_personnel_evidence(
        games=[{"game_id": a.game, "season": int(g["season"]), "week": int(g["week"]),
                "home_team": home, "away_team": away, "kickoff_utc": kickoff,
                "source_id": "nflcom_schedule+nflverse_games_csv",
                "source_url": sched_src["effective_url"], "captured_at": sched_src["fetched_at"]}],
        roster=roster, depth=depth_ok, availability=availability, report_index=report_index,
        prior_availability=prior, starter_claims=claims, offensive_players=offensive,
        as_of=a.as_of)
    out["adapter"] = {
        "mode": "live_context_retrieval (not a historical as-of replay)",
        "schedule_check": {"nflverse_local": f"{g['gameday']} {g['gametime']}",
                           "derived_utc": cand_utc, "nflcom_startTime_present": kickoff is not None},
        "roster_source": {"url": roster_src["effective_url"], "captured_at": roster_src["fetched_at"],
                          "season_week": f"{int(last['season'].max())}-W{int(last['week'].max())}"},
        "prior_games": prior_game, "parse_meta": parse_meta, "articles": article_meta,
        "target_week_reports": {t: bool([r for r in report_index if r["team"] == t])
                                for t in (home, away)},
        "depth_rows": len(depth), "depth_resolved": len(depth_ok),
        "depth_unresolved": depth_unresolved,
        "roster_status_non_active_depth": sorted(
            {(d["team"], d["name"], d["slot"], roster_status.get(d["player_id"]))
             for d in depth_ok if roster_status.get(d["player_id"]) not in (None, "ACT")}),
        "prior_report_differences": differences,
        "nflverse_listed_qb_proxy": {
            "values": {home: g["home_qb_name"], away: g["away_qb_name"]},
            "measurement_kind": "proxy",
            "note": "unplayed-game value in nflverse games.csv; method/clock undocumented; "
                    "not used as a starter claim"},
        "source_clock_audit": {"as_of": a.as_of, "inputs_checked": len(clocks),
                               "max_input_clock": max((c[2] for c in clocks), default=None),
                               "max_fetch_clock": max(v["fetched_at"] for v in log.values()),
                               "future_inputs": 0},
        "fetch_log": log}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True, default=str)
    s = out["games"].get(a.game, {})
    print(json.dumps({"kickoff": kickoff, "records": len(out["records"]),
                      "identity_errors": len(out["identity_errors"]),
                      "excluded_rows": len(out["excluded_rows"]),
                      "parse_complete": {t: {k: v["complete"] for k, v in m.items()}
                                         for t, m in parse_meta.items()},
                      "differences": differences,
                      "qb": {t: (v["qb"]["state"], v["qb"]["expected_starter_name"])
                             for t, v in s.get("teams", {}).items()}}, indent=1))


if __name__ == "__main__":
    main()
