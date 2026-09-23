"""Explicit acquisition adapter + worked example for nflvalue.personnel_matchup.

Separated from the pure builder on purpose: this script parses raw pages that
were ALREADY fetched (official NFL.com schedule/injury pages, official team
injury-report and depth-chart pages, nflverse games.csv) from a raw directory
with a FETCH_LOG.txt, maps names to nflverse gsis ids via the cached weekly
roster (parquet cache or nflverse roster_YYYY.csv), and runs ``build_personnel_evidence``.  No network access here.

    python analysis/personnel_matchup_example.py RAW_DIR ROSTERS(.parquet|.csv) OUT_JSON \
        --game 2026_03_ATL_GB --as-of 2026-09-23T01:31:00Z

Any game whose two teams have official team-site pages in RAW_DIR works; the
team->site map below is the only team-specific table (URL routing, not logic).
"""

from __future__ import annotations

import argparse
import csv
import glob
import html
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nflvalue.personnel_matchup import build_personnel_evidence, norm_name  # noqa: E402

TEAM_SITE = {"ATL": "atlantafalcons_com", "GB": "packers_com"}
OFF_SLOTS = {"QB", "RB", "WR", "TE", "LT", "LG", "C", "RG", "RT"}
DEF_SLOTS = {"DE", "DL", "DT", "EDGE", "LB", "ILB", "OLB", "CB", "S", "NB"}
PRACTICE = r"(DNP|LP|FP|\(-\))"
GAME_STATUS = r"(OUT|DOUBTFUL|QUESTIONABLE|UNSPECIFIED|\(-\))"


def _text(path):
    s = open(path, errors="ignore").read()
    s = re.sub(r"<script.*?</script>", "", s, flags=re.S)
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


def parse_team_report(path, own_first=True):
    """First table on an official club injury-report page = the club's own
    players.  Returns rows with documented position/practice/game status."""
    t = _text(path)
    i = t.find("Player Position Injury Wed Thu Fri Game Status")
    j = t.find("Player Position Injury Wed Thu Fri Game Status", i + 10)
    seg = t[i + len("Player Position Injury Wed Thu Fri Game Status"):j if j > 0 else None]
    seg = seg.split(" Table - Injury report")[0]
    pat = re.compile(r"\s*(?P<name>[A-Z][\w.'’\- ]+?) (?P<pos>[A-Z]{1,4}) (?P<inj>.*?) "
                     + PRACTICE + " " + PRACTICE + " " + PRACTICE + " " + GAME_STATUS)
    rows = []
    for m in pat.finditer(seg):
        name = m.group("name").strip()
        # the opponent's table header ("Carolina Panthers") glues onto the
        # previous row only after the split above, so names are clean here
        rows.append({"name": name, "position_reported": m.group("pos"),
                     "injury": m.group("inj").strip(),
                     "practice": [m.group(4), m.group(5), m.group(6)],
                     "report_status": m.group(7).title().replace("(-)", "(-)")})
    return rows


def parse_nflcom_report(path, nickname):
    """NFL.com league page: the section headed by the club nickname."""
    t = _text(path)
    k = t.find(f"{nickname} Player Position Injuries Practice Status Game Status")
    if k < 0:
        return []
    seg = t[k + len(nickname) + 48:]
    seg = re.split(r" \d{1,2}:\d\d [AP]M | [A-Z]{2,3} [A-Z][a-z]+ \(\d", seg)[0]
    seg = re.split(r" [A-Z][a-z]+ Player Position Injuries", seg)[0]
    pr = r"(Did Not Participate In Practice|Limited Participation in Practice|Full Participation in Practice)"
    pat = re.compile(r"\s*(?P<name>[A-Z][\w.'’\- ]+?) (?P<pos>[A-Z]{1,4})(?P<inj>(?: [A-Z][\w, ]*?)?) "
                     + pr + r"(?: (?P<gs>Out|Doubtful|Questionable))?(?= [A-Z]|$)")
    return [{"name": m.group("name").strip(), "position_reported": m.group("pos"),
             "injury": m.group("inj").strip(), "practice": [m.group(4)],
             "report_status": m.group("gs") or "(-)"} for m in pat.finditer(seg)]


def parse_depth(path, roster_names):
    """Official club depth chart -> [{slot, rank, name}] (names split by
    jersey number when present, else by greedy match to roster names)."""
    t = _text(path)
    slots = OFF_SLOTS | DEF_SLOTS
    out, counter = [], {}
    # nav bars repeat "Offense/Defense/Special Teams": start at the table header
    m = re.search(r"Position (?:1st|Player)", t)
    begin = m.start() if m else 0
    stop = min([x for x in (t.find("Special Teams Position", begin),
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
                    # unknown token run: join two words as a best-effort name
                    cand = " ".join(words[:2])
                    L = min(2, len(words))
                names.append(cand)
                words = words[L:]
        counter[slot] = counter.get(slot, 0) + 1
        key = slot if counter[slot] == 1 else f"{slot}{counter[slot]}"
        for rank, nm in enumerate(names, 1):
            out.append({"slot": key, "rank": rank, "name": nm})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("rosters")
    ap.add_argument("out")
    ap.add_argument("--game", required=True)
    ap.add_argument("--as-of", required=True)
    a = ap.parse_args()
    import pandas as pd

    log = _fetch_log(a.raw)
    games = list(csv.DictReader(open(_raw(a.raw, "games_csv"))))
    g = next(r for r in games if r["game_id"] == a.game)
    home, away = g["home_team"], g["away_team"]
    sched_txt = _text(_raw(a.raw, "nfl_com_schedules"))
    sched_html = open(_raw(a.raw, "nfl_com_schedules"), errors="ignore").read()
    starts = re.findall(r'startTime\\":\\"([0-9TZ:\-]+)\\"', sched_html)
    notes = []
    kickoff = None
    # official schedule confirms identity + time: nflverse local time is ET
    et_hour = int(g["gametime"].split(":")[0])
    cand_utc = f"{g['gameday']}T{et_hour + 4:02d}:{g['gametime'].split(':')[1]}:00Z" \
        if et_hour + 4 < 24 else None
    if cand_utc is None:
        import datetime as dt
        d = dt.datetime.fromisoformat(g["gameday"]) + dt.timedelta(days=1)
        cand_utc = f"{d.date()}T{et_hour + 4 - 24:02d}:{g['gametime'].split(':')[1]}:00Z"
    if cand_utc in starts:
        kickoff = cand_utc
    notes.append({"schedule_check": {"nflverse_gameday_et": f"{g['gameday']} {g['gametime']}",
                                     "derived_utc_assuming_edt": cand_utc,
                                     "nflcom_startTime_present": cand_utc in starts,
                                     "nflcom_text_hit": bool(re.search(
                                         r"at [A-Z][a-z]+, \w+day, [A-Z][a-z]+ \d+\w*, \d", sched_txt))}})
    sched_src = next(v for k, v in log.items() if "schedules" in k)

    if not a.rosters.endswith(".parquet"):
        ros = pd.read_csv(a.rosters, dtype=str)
        ros = ros.rename(columns={"gsis_id": "player_id"})
        ros["position"] = ros["depth_chart_position"].fillna(ros["position"])
        ros["week"] = ros["week"].astype(int)
        ros["season"] = ros["season"].astype(int)
    else:
        ros = pd.read_parquet(a.rosters)
    ros = ros.dropna(subset=["player_id"])
    last = ros[ros["season"] == ros["season"].max()]
    last = last[last["week"] == last["week"].max()]
    roster, names_by_team, roster_status = [], {}, {}
    for r in last[last["team"].isin([home, away])].itertuples(index=False):
        roster.append({"team": r.team, "player_id": r.player_id, "name": r.full_name,
                       "position": r.position})
        names_by_team.setdefault(r.team, set()).add(norm_name(r.full_name))
        roster_status[r.player_id] = getattr(r, "status", None)
    roster_src = {"source_id": "nflverse_rosters_weekly_cache",
                  "season_week": f"{int(last['season'].max())}-W{int(last['week'].max())}"}

    prior_game = {}
    for t in (home, away):
        pg = [r for r in games if r["season"] == g["season"] and int(r["week"]) == int(g["week"]) - 1
              and t in (r["home_team"], r["away_team"])]
        prior_game[t] = pg[0]["game_id"] if pg else None

    depth, prior, report_rows = [], [], {}
    nick = {"ATL": "Falcons", "GB": "Packers"}
    for t in (home, away):
        site = TEAM_SITE[t]
        dpath = _raw(a.raw, f"{site}_team_depth")
        dsrc = next(v for k, v in log.items() if site.split("_")[0] in k and "depth" in k)
        for d in parse_depth(dpath, names_by_team.get(t, set())):
            depth.append({"team": t, "name": d["name"], "slot": d["slot"], "rank": d["rank"],
                          "source_id": f"{site}_depth_chart", "source_url": dsrc["effective_url"],
                          "fetched_at": dsrc["fetched_at"], "published_at": None})
        rpath = _raw(a.raw, f"{site}_team_injury")
        rsrc = next(v for k, v in log.items() if site.split("_")[0] in k and "injury" in k)
        team_rows = parse_team_report(rpath)
        league_rows = parse_nflcom_report(_raw(a.raw, "nfl_com_injuries"), nick[t])
        lsrc = next(v for k, v in log.items() if "nfl.com/injuries" in k)
        report_rows[t] = {"team_site": team_rows, "nfl_com": league_rows}
        for src_id, rows, src in ((f"{site}_injury_report", team_rows, rsrc),
                                  ("nflcom_injuries_prior_week", league_rows, lsrc)):
            for r in rows:
                prior.append({"game_id": prior_game[t], "team": t, "name": r["name"],
                              "report_status": r["report_status"], "practice": r["practice"],
                              "position_reported": r["position_reported"],
                              "source_id": src_id, "source_url": src["effective_url"],
                              "fetched_at": src["fetched_at"], "published_at": None})

    # map depth names -> ids (unique exact normalized match within team, else drop + log)
    by_team = {}
    for p in roster:
        by_team.setdefault((p["team"], norm_name(p["name"])), []).append(p["player_id"])
    depth_ok, depth_unresolved = [], []
    for d in depth:
        ids = by_team.get((d["team"], norm_name(d["name"])), [])
        if len(ids) == 1:
            depth_ok.append(dict(d, player_id=ids[0]))
        else:
            depth_unresolved.append({"team": d["team"], "name": d["name"], "slot": d["slot"],
                                     "reason": "ambiguous" if ids else "no_roster_match"})

    offensive = []
    for t in (home, away):
        for slot in ("QB", "RB", "WR", "TE"):
            d = next((x for x in depth_ok if x["team"] == t and x["slot"] == slot
                      and x["rank"] == 1), None)
            if d:
                offensive.append({"player_id": d["player_id"], "team": t, "game_id": a.game,
                                  "position": slot})

    # source disagreements on the prior-week report are surfaced, not resolved
    disagreements = []
    for t, rr in report_rows.items():
        ts = {norm_name(r["name"]): r for r in rr["team_site"]}
        for r in rr["nfl_com"]:
            o = ts.get(norm_name(r["name"]))
            if o and o["report_status"].lower() != r["report_status"].lower():
                disagreements.append({"team": t, "name": r["name"], "team_site": o["report_status"],
                                      "nfl_com": r["report_status"]})
            if o and o["position_reported"] != r["position_reported"]:
                disagreements.append({"team": t, "name": r["name"],
                                      "team_site_position": o["position_reported"],
                                      "nfl_com_position": r["position_reported"]})
    nflverse_qb = {home: g["home_qb_name"], away: g["away_qb_name"]}

    out = build_personnel_evidence(
        games=[{"game_id": a.game, "season": int(g["season"]), "week": int(g["week"]),
                "home_team": home, "away_team": away, "kickoff_utc": kickoff,
                "source_id": "nflcom_schedule+nflverse_games_csv",
                "source_url": sched_src["effective_url"], "fetched_at": sched_src["fetched_at"]}],
        roster=roster, depth=depth_ok, availability=[], report_index=[],
        prior_availability=prior, offensive_players=offensive, as_of=a.as_of)
    out["adapter"] = {
        "notes": notes, "roster_source": roster_src, "prior_games": prior_game,
        "target_week_report": ("not published at fetch: NFL.com reg%s URL redirected to the "
                               "prior week and club pages still show the prior opponent" % g["week"]),
        "depth_rows": len(depth), "depth_resolved": len(depth_ok),
        "depth_unresolved": depth_unresolved,
        "roster_status_non_active_depth": sorted(
            {(d["team"], d["name"], d["slot"], roster_status.get(d["player_id"]))
             for d in depth_ok if roster_status.get(d["player_id"]) not in (None, "ACT")}),
        "prior_report_rows": {t: {k: len(v) for k, v in rr.items()} for t, rr in report_rows.items()},
        "prior_report_disagreements": disagreements,
        "nflverse_listed_qb_proxy": {
            "values": nflverse_qb, "measurement_kind": "proxy",
            "note": ("nflverse games.csv qb_name for an unplayed game; method/clock undocumented "
                     "and overwritten with the actual starter after the game; not used as a "
                     "starter claim")},
        "fetch_log": log}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True, default=str)
    s = out["games"].get(a.game, {})
    print(json.dumps({"kickoff": kickoff, "records": len(out["records"]),
                      "identity_errors": len(out["identity_errors"]),
                      "depth_unresolved": len(depth_unresolved),
                      "disagreements": len(disagreements),
                      "qb": {t: (v["qb"]["state"], v["qb"]["expected_starter_name"])
                             for t, v in s.get("teams", {}).items()}}, indent=1))


if __name__ == "__main__":
    main()
