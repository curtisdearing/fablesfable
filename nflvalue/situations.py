"""Phase 6.6: situational context tags -- built as FACTS, tested before use.

Every tag here follows the strict gate: it may appear in the context panel /
ledger and in the historical significance study (scripts/study_situations.py),
but it carries ZERO weight anywhere -- not even as an ML feature -- until it
clears n>=100 + BH-q<0.05 AND a human promotes it in config
(context_learning.enabled_tags). That is the prompt's letter, stricter than
the birthday/revenge precedent (which pre-dates the rule and stays
grandfathered as ML features).

Tags:
  primetime           TNF / SNF / MNF (schedule weekday + kickoff)
  short_week          <= 5 days rest (schedule home_rest/away_rest)
  long_travel_2tz     >= 2 timezones crossed for the road team
  west_east_early     PT/MT-based road team kicking off <= 13:00 ET
  division_game       intra-division matchup (static map)
  revenge_trade / revenge_cut / revenge_fa
                      revenge (prior roster stint vs this opponent, the
                      existing >=3-week definition) SPLIT by how the player
                      LEFT: a trade row in nflverse trades (pfr-name matched),
                      else mid-contract departure ~ cut, else contract
                      expiry ~ free agency. The cut/fa line is an
                      APPROXIMATION from contracts (documented; trades are
                      exact).
  + stratifications of the existing birthday/revenge tags by home/away and
    opponent-quality tercile (opp_epa_factor).
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Set, Tuple

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIST = os.path.join(ROOT, "historical")

DIVISIONS: Dict[str, str] = {
    "BUF": "AFCE", "MIA": "AFCE", "NE": "AFCE", "NYJ": "AFCE",
    "BAL": "AFCN", "CIN": "AFCN", "CLE": "AFCN", "PIT": "AFCN",
    "HOU": "AFCS", "IND": "AFCS", "JAX": "AFCS", "TEN": "AFCS",
    "DEN": "AFCW", "KC": "AFCW", "LV": "AFCW", "LAC": "AFCW",
    "DAL": "NFCE", "NYG": "NFCE", "PHI": "NFCE", "WAS": "NFCE",
    "CHI": "NFCN", "DET": "NFCN", "GB": "NFCN", "MIN": "NFCN",
    "ATL": "NFCS", "CAR": "NFCS", "NO": "NFCS", "TB": "NFCS",
    "ARI": "NFCW", "LA": "NFCW", "SF": "NFCW", "SEA": "NFCW",
    # legacy abbrs seen in older seasons
    "OAK": "AFCW", "SD": "AFCW", "STL": "NFCW",
}

TEAM_TZ_OFFSET: Dict[str, int] = {  # hours behind ET (0=ET .. 3=PT)
    "BUF": 0, "MIA": 0, "NE": 0, "NYJ": 0, "BAL": 0, "CIN": 0, "CLE": 0,
    "PIT": 0, "HOU": 1, "IND": 0, "JAX": 0, "TEN": 1, "DEN": 2, "KC": 1,
    "LV": 3, "LAC": 3, "DAL": 1, "NYG": 0, "PHI": 0, "WAS": 0, "CHI": 1,
    "DET": 0, "GB": 1, "MIN": 1, "ATL": 0, "CAR": 0, "NO": 1, "TB": 0,
    "ARI": 2, "LA": 3, "SF": 3, "SEA": 3, "OAK": 3, "SD": 3, "STL": 1,
}


def _kick_hour_et(gametime: Optional[str]) -> Optional[float]:
    try:
        hh, mm = str(gametime).split(":")[:2]
        return int(hh) + int(mm) / 60.0
    except (ValueError, AttributeError, TypeError):
        return None


def game_tags(schedules: pd.DataFrame) -> pd.DataFrame:
    """One row per (game_id, team) with the game-level situational flags."""
    g = schedules[schedules["game_type"] == "REG"].drop_duplicates("game_id")
    rows = []
    for r in g.itertuples(index=False):
        hour = _kick_hour_et(getattr(r, "gametime", None))
        wd = str(getattr(r, "weekday", "") or "")
        primetime = int(wd == "Thursday" or wd == "Monday"
                        or (wd in ("Sunday", "Saturday") and hour is not None and hour >= 19.5))
        division = int(DIVISIONS.get(r.home_team) is not None
                       and DIVISIONS.get(r.home_team) == DIVISIONS.get(r.away_team))
        for team, opp, home in ((r.home_team, r.away_team, 1), (r.away_team, r.home_team, 0)):
            rest = getattr(r, "home_rest" if home else "away_rest", None)
            tz_from = TEAM_TZ_OFFSET.get(team)
            tz_to = TEAM_TZ_OFFSET.get(r.home_team)
            tz_cross = abs(tz_from - tz_to) if (tz_from is not None and tz_to is not None and not home) else 0
            rows.append({
                "game_id": r.game_id, "season": int(r.season), "week": int(r.week),
                "team": team, "home": home,
                "primetime": primetime, "division_game": division,
                "short_week": int(rest is not None and pd.notna(rest) and float(rest) <= 5),
                "long_travel_2tz": int(tz_cross >= 2),
                "west_east_early": int((not home) and tz_from is not None and tz_from >= 2
                                       and tz_to == 0 and hour is not None and hour <= 13.5),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Revenge subtypes
# --------------------------------------------------------------------------- #
def _normalize(name) -> str:
    from .sources.availability import normalize_name
    return normalize_name(name)


def load_trade_moves() -> Set[Tuple[str, int]]:
    """{(normalized player name, season)} for players moved BY TRADE."""
    path = os.path.join(HIST, "trades.parquet")
    if not os.path.exists(path):
        return set()
    tr = pd.read_parquet(path)
    if "pfr_name" not in tr.columns:
        return set()
    tr = tr.dropna(subset=["pfr_name"])
    return {(_normalize(n), int(s)) for n, s in zip(tr["pfr_name"], tr["season"])}


def contract_expiry_lookup() -> Dict[str, Set[int]]:
    """{gsis_id: {seasons in which a known deal ENDED}} -- expiry ~ free agency."""
    path = os.path.join(HIST, "contracts.parquet")
    if not os.path.exists(path):
        return {}
    con = pd.read_parquet(path).dropna(subset=["gsis_id", "year_signed", "years"])
    out: Dict[str, Set[int]] = {}
    for r in con.itertuples(index=False):
        try:
            out.setdefault(r.gsis_id, set()).add(int(r.year_signed) + int(r.years) - 1)
        except (TypeError, ValueError):
            continue
    return out


def revenge_subtype(player_id: str, player_name: str, season: int, week: int,
                    team: str, opponent: str, stints: Dict, trades: Set,
                    expiries: Dict) -> Optional[str]:
    """None if not a revenge spot; else 'revenge_trade' / 'revenge_fa' /
    'revenge_cut' (cut = mid-contract departure that wasn't a trade --
    an approximation, contracts data isn't a transaction log)."""
    entries = stints.get(player_id, [])
    dep_season = None
    on_opp = 0
    for (s, w, t) in entries:
        if (s, w) >= (season, week):
            break
        if t == opponent:
            on_opp += 1
            dep_season = s
    if on_opp < 3 or dep_season is None:
        return None
    nname = _normalize(player_name)
    if (nname, dep_season) in trades or (nname, dep_season + 1) in trades:
        return "revenge_trade"
    if dep_season in expiries.get(player_id, set()):
        return "revenge_fa"
    return "revenge_cut"
