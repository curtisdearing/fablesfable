"""Per-game writeup: records, weather, injuries, and the story lines —
birthdays, revenge games, contract incentives — computed from fact tables.

Feeds the report's "Game notes" block and the Discord embed description.
Display-only by construction: notes are assembled AFTER ranking from the
same candidate frame the ranker used, so nothing here can move a score.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def build_records(schedules: pd.DataFrame) -> Dict[Tuple[int, int, str], str]:
    """{(season, week, team): 'W-L' entering that week} — walk-forward from
    completed results (ties rendered as W-L-T)."""
    if "result" not in schedules.columns:
        return {}
    s = schedules[(schedules["game_type"] == "REG") & schedules["result"].notna()]
    tally: Dict[Tuple[int, str], List[int]] = {}
    out: Dict[Tuple[int, int, str], str] = {}
    for g in s.sort_values(["season", "week"]).itertuples(index=False):
        season = int(g.season)
        for team in (g.home_team, g.away_team):
            w, l, t = tally.get((season, team), (0, 0, 0))
            out[(season, int(g.week), team)] = f"{w}-{l}" + (f"-{t}" if t else "")
        res = float(g.result)
        hw, hl, ht = tally.get((season, g.home_team), (0, 0, 0))
        aw, al, at = tally.get((season, g.away_team), (0, 0, 0))
        if res > 0:
            tally[(season, g.home_team)] = (hw + 1, hl, ht)
            tally[(season, g.away_team)] = (aw, al + 1, at)
        elif res < 0:
            tally[(season, g.home_team)] = (hw, hl + 1, ht)
            tally[(season, g.away_team)] = (aw + 1, al, at)
        else:
            tally[(season, g.home_team)] = (hw, hl, ht + 1)
            tally[(season, g.away_team)] = (aw, al, at + 1)
    # also expose "entering week N+1" for teams' next appearance
    for (season, team), (w, l, t) in tally.items():
        out[(season, 99, team)] = f"{w}-{l}" + (f"-{t}" if t else "")
    return out


def record_for(records: Dict, season: int, week: int, team: str) -> str:
    r = records.get((season, week, team))
    if r is None:
        # first game of the season (or unknown): 0-0 unless later info exists
        r = "0-0" if week <= 1 else records.get((season, 99, team), "0-0")
    return r


def _val(row: Dict, key: str):
    v = row.get(key)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return v


def build_game_notes(game_id: str, game_cands: List[Dict],
                     records: Dict, season: int, week: int,
                     home_team: str, away_team: str) -> List[str]:
    """The writeup lines for one game, scanned across ALL screened candidates
    (not just the leans) so a backup's birthday still makes the notes."""
    notes: List[str] = []
    notes.append(f"Records: {away_team} {record_for(records, season, week, away_team)} "
                 f"@ {home_team} {record_for(records, season, week, home_team)}")

    first = game_cands[0] if game_cands else {}
    wind = _val(first, "wind")
    temp = _val(first, "temp")
    if wind is not None and wind >= 12:
        wx = f"Weather: {int(wind)} mph wind" + (f", {int(temp)}F" if temp is not None else "")
        notes.append(wx + (" — passing suppressor" if wind >= 15 else ""))

    seen: Dict[str, set] = {"bday": set(), "rev": set(), "cy": set()}
    def_outs: Dict[str, Tuple[int, int]] = {}
    ol_outs: Dict[str, int] = {}
    qbc: Dict[str, float] = {}
    for c in game_cands:
        name, team = c.get("name"), c.get("team")
        if c.get("is_birthday_week"):
            seen["bday"].add(f"{name} ({team})")
        if c.get("revenge_game"):
            seen["rev"].add(f"{name} ({team} — ex-{c.get('defteam')})")
        if c.get("is_contract_year"):
            seen["cy"].add(f"{name} ({team})")
        d = _val(c, "def_out_total")
        if d:
            def_outs[c.get("defteam")] = (int(d), int(_val(c, "def_out_db") or 0))
        ol = _val(c, "oline_outs")
        if ol:
            ol_outs[team] = int(ol)
        q = _val(c, "qb_continuity")
        if q is not None:
            qbc[team] = float(q)

    if seen["bday"]:
        notes.append("Birthday week: " + ", ".join(sorted(seen["bday"])))
    if seen["rev"]:
        notes.append("Revenge game: " + ", ".join(sorted(seen["rev"])))
    if seen["cy"]:
        cy = sorted(seen["cy"])
        shown = ", ".join(cy[:6]) + (f" +{len(cy)-6} more" if len(cy) > 6 else "")
        notes.append(f"Contract year (incentive watch): {shown}")
    for team, (tot, dbs) in sorted(def_outs.items()):
        if tot >= 2:
            notes.append(f"{team} defense lists {tot} Out/Doubtful"
                         + (f" ({dbs} in the secondary)" if dbs else ""))
    for team, n in sorted(ol_outs.items()):
        if n >= 2:
            notes.append(f"{team} O-line lists {n} Out/Doubtful")
    for team, q in sorted(qbc.items()):
        if q < 0.5:
            notes.append(f"{team}: projected starting QB threw only {q:.0%} of trailing "
                         "attempts — receiver histories discounted")

    # mismatched matchups: extreme opponent-vs-position factors on this slate
    mismatches: Dict[str, Tuple[float, str]] = {}
    for c in game_cands:
        pc = c.get("proj_components") or c.get("components") or {}
        f = pc.get("opp_factor")
        if f in (None, 1.0) or (isinstance(f, float) and np.isnan(f)):
            continue
        key = f"{c.get('pos')} ({c.get('team')}) vs {c.get('defteam')} D"
        if abs(f - 1.0) > abs(mismatches.get(key, (1.0, ""))[0] - 1.0):
            mismatches[key] = (float(f), c.get("market", ""))
    hot = {k: v for k, v in mismatches.items() if v[0] >= 1.12 or v[0] <= 0.88}
    for key, (f, _) in sorted(hot.items(), key=lambda x: -abs(x[1][0] - 1.0))[:3]:
        direction = "soft" if f > 1.0 else "tough"
        notes.append(f"Mismatch: {key} is {direction} ({(f-1)*100:+.0f}% vs league avg)")
    return notes


def correlation_notes_for_game(leans: List[Dict], corr, as_of_season: Optional[int] = None) -> List[str]:
    """Display-only flags for pairs of SELECTED leans whose type the 7.5 audit
    measured as real correlation -- so a reader can see when a "N leans" slip
    is really fewer independent bets (or, for negative pairs, a hedge).
    Computed AFTER selection; never feeds back into scores. ``corr`` is a
    ``correlation.CorrelationStructure`` (or None -- returns [] then)."""
    if corr is None or not leans:
        return []
    notes: List[str] = []
    seen = set()
    for i in range(len(leans)):
        for j in range(i + 1, len(leans)):
            a, b = leans[i], leans[j]
            rho = corr.rho_for(a.get("pos"), a.get("market"), a.get("player_id"), a.get("team"),
                               b.get("pos"), b.get("market"), b.get("player_id"), b.get("team"),
                               as_of_season=as_of_season)
            if rho == 0.0:
                continue
            key = tuple(sorted([f"{a.get('player_id')}:{a.get('market')}",
                                f"{b.get('player_id')}:{b.get('market')}"]))
            if key in seen:
                continue
            seen.add(key)
            direction = "move together" if rho > 0 else "hedge against each other"
            strength = ("near-duplicate" if abs(rho) >= 0.5 else
                       "moderately correlated" if abs(rho) >= 0.2 else "mildly correlated")
            notes.append(
                f"Correlated legs: {a.get('name')} {str(a.get('market') or '').replace('_', ' ')} "
                f"and {b.get('name')} {str(b.get('market') or '').replace('_', ' ')} are {strength} "
                f"and {direction} (ρ≈{rho:+.2f}) — not fully independent edges.")
    return notes


def attach_notes(games: List[Dict], cands: pd.DataFrame, schedules: pd.DataFrame,
                 season: int, week: int, corr=None, as_of_season: Optional[int] = None) -> None:
    """Stamp g['notes'] onto each shortlist game dict, in place.

    ``corr`` (Phase 7.6, optional): a ``correlation.CorrelationStructure`` --
    when given, appends ``correlation_notes_for_game`` flags for this game's
    SELECTED leans onto the same display-only notes list."""
    records = build_records(schedules)
    slate = schedules[(schedules["season"] == season) & (schedules["week"] == week)
                      & (schedules["game_type"] == "REG")]
    home_away = {g.game_id: (g.home_team, g.away_team)
                 for g in slate.itertuples(index=False)}
    by_game = {gid: grp.to_dict("records") for gid, grp in cands.groupby("game_id")} \
        if not cands.empty else {}
    for g in games:
        home, away = home_away.get(g["game_id"], ("", ""))
        g["notes"] = build_game_notes(g["game_id"], by_game.get(g["game_id"], []),
                                      records, season, week, home, away)
        g["notes"].extend(correlation_notes_for_game(g.get("leans", []), corr, as_of_season))
