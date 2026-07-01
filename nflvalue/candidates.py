"""Enumerate every eligible player-market prop candidate for a week's games.

For a given (season, week) this module walks the slate from
``historical/historical_lines.parquet`` (the nflverse schedules table, which
carries pre-game consensus spread/total -- legitimate prior information, not
leakage) and produces one candidate per (player, market) with the Phase-1
projection contract attached:

    candidate = projection.project(...) + {game_id, matchup, team, defteam,
                line, line_source, prices?, spread_line, total_line}

Guardrails baked in:
  * FEATURES ARE STRICTLY PRIOR-WEEK. Rows come from ``features.py``'s
    walk-forward tables; the roll_* values attached to (season, week) only
    aggregate weeks < week (tested by tests/test_leakage.py).
  * Candidate SET for a completed historical week follows the
    ``prop_backtest.py`` convention: players with a ``player_week`` row that
    week (i.e., who actually recorded usage). For LIVE weeks the pipeline
    instead passes ``roster_mode="carry_forward"`` which enumerates from the
    most recent prior week per team -- no week-W information at all -- and
    lets the availability resolver (Phase 1B) trim it.
  * Cold-start gate: only ``eligible_for_shortlist`` players (Phase 1B
    MIN_GAMES_ELIGIBLE) plus a configurable minimum-usage floor, so scrubs
    never reach the ranker.
  * SD comes from walk-forward residuals of the SAME projection engine over
    weeks strictly before the target week (per market) -- never from the
    target week's outcomes.
  * LINE: a real prop line (Phase 3, per-event Odds API pull) when supplied;
    otherwise a SYNTHETIC line (the player's own trailing mean, floor+0.5)
    tagged ``line_source="synthetic_trailing_mean"`` so no consumer can
    mistake it for a market price. Synthetic lines carry no prices, so the
    edge component stays ``no_market`` (PROP_SHORTLISTER_SPEC.md §3).
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import projection
from .features import build_opp_pos_def, build_player_week, build_team_week, load_pbp
from .projection import MARKETS, MIN_GAMES_ELIGIBLE, game_script_multipliers

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEDULES_PATH = os.path.join(ROOT, "historical", "historical_lines.parquet")

ACTUAL_COL = {
    "receiving_yards": "rec_yards", "receptions": "receptions",
    "rushing_yards": "rush_yards", "passing_yards": "pass_yards",
    "pass_attempts": "pass_attempts", "rush_attempts": "carries",
}

# minimum trailing usage so the candidate pool isn't scrubs (configurable via
# config.json "candidates" section; these are the defaults)
DEFAULT_MIN_USAGE = {
    "targets": 2.5,        # roll_targets    -- WR/TE receiving markets
    "carries": 5.0,        # roll_carries    -- RB rushing markets
    "pass_attempts": 12.0,  # roll_pass_attempts -- QB markets
}
MIN_SD_HISTORY = 30        # walk-forward residuals needed before trusting a market SD


# --------------------------------------------------------------------------- #
# Schedule / slate
# --------------------------------------------------------------------------- #
def load_schedules(path: Optional[str] = None) -> pd.DataFrame:
    return pd.read_parquet(path or SCHEDULES_PATH)


def games_for_week(season: int, week: int, schedules: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """The week's slate: game_id, home/away abbrs, pre-game spread/total.

    ``spread_line`` is the HOME team's expected margin (positive = home
    favored), per nflverse convention -- verified against 2023_01_DET_KC
    (KC home, spread_line=4.0, KC favored by ~4).
    """
    sched = schedules if schedules is not None else load_schedules()
    g = sched[(sched["season"] == season) & (sched["week"] == week)
              & (sched["game_type"] == "REG")].copy()
    keep = ["game_id", "season", "week", "gameday", "gametime", "home_team", "away_team",
            "spread_line", "total_line"]
    return g[keep].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Feature-table bundle (built once, reused across games/markets)
# --------------------------------------------------------------------------- #
class WeekInputs:
    """All walk-forward tables needed to project one week, built once."""

    def __init__(self, pw: pd.DataFrame, opd: pd.DataFrame, tw: pd.DataFrame,
                 schedules: pd.DataFrame):
        self.pw = pw
        self.opd = opd
        self.tw = tw
        self.schedules = schedules
        self.team_idx = {(r.season, r.week, r.team): r._asdict()
                         for r in tw.itertuples(index=False)}
        self.opp_idx = {(r.season, r.week, r.defteam, r.role): r._asdict()
                        for r in opd.itertuples(index=False)}


def build_week_inputs(pbp: Optional[pd.DataFrame] = None,
                      schedules: Optional[pd.DataFrame] = None) -> WeekInputs:
    if pbp is None:
        pbp = load_pbp()
    return WeekInputs(
        pw=build_player_week(pbp),
        opd=build_opp_pos_def(pbp),
        tw=build_team_week(pbp),
        schedules=schedules if schedules is not None else load_schedules(),
    )


# --------------------------------------------------------------------------- #
# Walk-forward per-market SD + synthetic line
# --------------------------------------------------------------------------- #
def market_residual_sd(inputs: WeekInputs, market: str, season: int, week: int) -> Optional[float]:
    """SD of (actual - projected mean) over all weeks STRICTLY BEFORE
    (season, week) -- the same quantity prop_backtest.py's expanding
    walk-forward SD converges to, evaluated at the target week's cutoff."""
    import prop_backtest  # root module; imported lazily to avoid import cycles

    pw = inputs.pw
    hist = pw[(pw["season"] < season) | ((pw["season"] == season) & (pw["week"] < week))]
    if hist.empty:
        return None
    preds = prop_backtest._predictions_for_market(hist, market, inputs.team_idx, inputs.opp_idx)
    if preds.empty or len(preds) < MIN_SD_HISTORY:
        return None
    resid = preds["actual"] - preds["mean_pred"]
    sd = float(resid.std(ddof=1))
    return sd if math.isfinite(sd) and sd > 0 else None


def synthetic_lines(inputs: WeekInputs, market: str) -> pd.Series:
    """Each player's trailing rolling mean of the actual stat (shift(1) before
    rolling -- leak-free), snapped to a half-point so it reads like a prop
    line and can never push. Indexed like ``inputs.pw``."""
    actual_col = ACTUAL_COL.get(market)
    pw = inputs.pw
    if actual_col is None:  # anytime_td: the "line" is always 0.5 (yes/no)
        return pd.Series(0.5, index=pw.index)
    g = pw.sort_values(["player_id", "season", "week"]).groupby("player_id")
    trail = g[actual_col].transform(lambda s: s.shift(1).rolling(8, min_periods=3).mean())
    return np.floor(trail) + 0.5


# --------------------------------------------------------------------------- #
# Candidate enumeration
# --------------------------------------------------------------------------- #
def _passes_usage_floor(row: Dict, spec: Dict, min_usage: Dict) -> bool:
    opp = spec["opportunity"]
    if opp is None:  # anytime_td: require some red-zone-relevant volume
        vol = float(row.get("roll_carries") or 0.0) + float(row.get("roll_targets") or 0.0)
        return vol >= min(min_usage.get("targets", 2.5), min_usage.get("carries", 5.0))
    col = {"targets": "roll_targets", "carries": "roll_carries",
           "pass_attempts": "roll_pass_attempts"}[opp]
    v = row.get(col)
    v = 0.0 if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)
    return v >= float(min_usage.get(opp, 0.0))


def enumerate_candidates(
    season: int,
    week: int,
    inputs: Optional[WeekInputs] = None,
    markets: Optional[List[str]] = None,
    min_usage: Optional[Dict[str, float]] = None,
    min_games: int = MIN_GAMES_ELIGIBLE,
    prop_lines: Optional[pd.DataFrame] = None,
    roster_mode: str = "as_played",
) -> pd.DataFrame:
    """All eligible (player, market) candidates for every game of (season, week).

    ``prop_lines`` (Phase 3): DataFrame [game_id, market, player_id, point,
    over_price, under_price, book] of REAL prop lines; where a row matches, it
    replaces the synthetic line and carries prices (enabling the edge
    component). Everything else stays synthetic + no_market.

    ``roster_mode``:
      * "as_played"    -- players with a player_week row AT (season, week)
                          (historical/backtest convention; features still
                          strictly prior-week).
      * "carry_forward" -- each team's players from their most recent week
                          < (season, week); zero week-W information (live mode;
                          availability resolver trims it downstream).
    """
    inputs = inputs or build_week_inputs()
    markets = markets or list(MARKETS)
    min_usage = {**DEFAULT_MIN_USAGE, **(min_usage or {})}

    slate = games_for_week(season, week, inputs.schedules)
    if slate.empty:
        raise ValueError(f"no REG games found for season={season} week={week}")
    team_to_game: Dict[str, Dict] = {}
    for g in slate.itertuples(index=False):
        # spread_line = home margin; away margin is its negation
        team_to_game[g.home_team] = {"game_id": g.game_id, "opp": g.away_team,
                                     "margin": float(g.spread_line) if pd.notna(g.spread_line) else None,
                                     "home": True, "spread_line": g.spread_line,
                                     "total_line": g.total_line}
        team_to_game[g.away_team] = {"game_id": g.game_id, "opp": g.home_team,
                                     "margin": -float(g.spread_line) if pd.notna(g.spread_line) else None,
                                     "home": False, "spread_line": g.spread_line,
                                     "total_line": g.total_line}

    pw = inputs.pw
    if roster_mode == "as_played":
        week_rows = pw[(pw["season"] == season) & (pw["week"] == week)].copy()
    elif roster_mode == "carry_forward":
        hist = pw[((pw["season"] < season) | ((pw["season"] == season) & (pw["week"] < week)))]
        hist = hist[hist["team"].isin(team_to_game)]
        latest = hist.sort_values(["season", "week"]).groupby("player_id").tail(1).copy()
        # roll features on a player's LAST PLAYED row exclude that game itself;
        # they are the freshest leak-free estimate available pre-slate. The
        # honest cost: a player's very latest game isn't in his features and
        # debuts/trades are invisible -- exactly what availability + Phase 3
        # live rosters correct.
        latest["season"], latest["week"] = season, week
        week_rows = latest
    else:
        raise ValueError(f"unknown roster_mode {roster_mode!r}")

    week_rows = week_rows[week_rows["team"].isin(team_to_game)]

    # index real prop lines if provided
    line_idx: Dict = {}
    if prop_lines is not None and not prop_lines.empty:
        for r in prop_lines.itertuples(index=False):
            line_idx[(r.game_id, r.market, r.player_id)] = r._asdict()

    synth = {m: synthetic_lines(inputs, m) for m in markets}
    sd_by_market = {m: market_residual_sd(inputs, m, season, week) for m in markets}

    out: List[Dict] = []
    for idx, row in week_rows.iterrows():
        player_row = row.to_dict()
        role = player_row.get("role")
        ginfo = team_to_game.get(player_row.get("team"))
        if ginfo is None or role not in ("QB", "RB", "WR", "TE"):
            continue
        gs = game_script_multipliers(ginfo["margin"])
        team_row = inputs.team_idx.get((season, week, player_row["team"]))
        for market in markets:
            spec = MARKETS[market]
            if role not in spec["role"]:
                continue
            if not _passes_usage_floor(player_row, spec, min_usage):
                continue
            opp_row = (inputs.opp_idx.get((season, week, ginfo["opp"], role))
                       if spec["use_opp_factor"] else None)

            real = line_idx.get((ginfo["game_id"], market, player_row["player_id"]))
            if real is not None:
                line, line_source = float(real["point"]), "odds_api"
                prices = {"over": real.get("over_price"), "under": real.get("under_price"),
                          "book": real.get("book")}
            else:
                sl = synth[market].get(idx) if roster_mode == "as_played" else None
                if sl is None or (isinstance(sl, float) and math.isnan(sl)):
                    sl = _carry_forward_synth(inputs, market, player_row["player_id"])
                    if sl is None:
                        continue  # no trailing history to hang a line on
                line, line_source = float(sl), "synthetic_trailing_mean"
                prices = None

            proj = projection.project(
                player_row, market, team_row=team_row, opp_row=opp_row,
                line=line, sd=sd_by_market.get(market), game_script=gs,
                min_games=min_games,
            )
            if not proj["eligible_for_shortlist"]:
                continue
            proj.update({
                "season": season, "week": week, "game_id": ginfo["game_id"],
                "team": player_row.get("team"), "defteam": ginfo["opp"],
                "home": ginfo["home"], "matchup": ginfo["game_id"].split("_", 2)[-1].replace("_", " @ "),
                "line_source": line_source, "prices": prices,
                "spread_line": ginfo["spread_line"], "total_line": ginfo["total_line"],
                "sd_source": ("walk_forward_residuals" if sd_by_market.get(market) else "default_fraction"),
            })
            out.append(proj)

    df = pd.DataFrame(out)
    if not df.empty:
        df = df.sort_values(["game_id", "player_id", "market"], kind="mergesort").reset_index(drop=True)
    return df


def _carry_forward_synth(inputs: WeekInputs, market: str, player_id: str) -> Optional[float]:
    """Synthetic line for a carry-forward row: trailing mean of the player's
    actuals over his own prior rows (leak-free by construction)."""
    actual_col = ACTUAL_COL.get(market)
    if actual_col is None:
        return 0.5
    hist = inputs.pw[inputs.pw["player_id"] == player_id].sort_values(["season", "week"])
    tail = hist[actual_col].tail(8)
    if len(tail) < 3:
        return None
    return float(np.floor(tail.mean()) + 0.5)
