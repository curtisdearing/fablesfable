"""Hands-off data ingest: keep the feature foundation current, automatically.

Called at the top of every live pipeline run (skippable with --no-refresh).
Keeps one parquet per season alongside the frozen 2019-2023 base file:

    historical/historical_pbp.parquet     2019-2023 (frozen -- the backtest
                                          baseline; never rewritten)
    historical/pbp_{season}.parquet       2024, 2025, 2026, ... (one per season;
                                          the CURRENT season's file is
                                          re-downloaded on refresh because
                                          nflverse updates it nightly in-season)
    historical/lines_extra.parquet        schedules 2024->now (spread/total/
                                          kickoffs; current season refreshed)
    historical/rosters_weekly.parquet     extended in place (rosters.py)

Loaders compose base + per-season files into one frame; the pipeline and
feature builders call these instead of assuming 2019-2023. Fail-loud rules:
a refresh that can't reach nflverse KEEPS the cached files and reports
``stale=True`` so the freshness gate (not silence) decides what to do.

The NFL data year: a season's files exist from ~September; January/February
games belong to the PREVIOUS season label.
"""

from __future__ import annotations

import datetime as dt
import os
from typing import Dict, List, Optional

import pandas as pd

from .features import PBP_COLUMNS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIST = os.path.join(ROOT, "historical")
BASE_PBP = os.path.join(HIST, "historical_pbp.parquet")
BASE_LINES = os.path.join(ROOT, "historical_lines.parquet")
LINES_EXTRA = os.path.join(HIST, "lines_extra.parquet")
BASE_SEASONS = {2019, 2020, 2021, 2022, 2023}

SCHED_COLS = ["game_id", "season", "game_type", "week", "gameday", "weekday", "gametime",
              "away_team", "away_score", "home_team", "home_score", "result", "total",
              "spread_line", "total_line", "away_moneyline", "home_moneyline",
              "roof", "surface", "stadium", "temp", "wind", "div_game", "referee",
              "overtime", "location", "away_rest", "home_rest"]


def current_season(today: Optional[dt.date] = None) -> int:
    today = today or dt.date.today()
    return today.year if today.month >= 3 else today.year - 1


def _season_pbp_path(season: int) -> str:
    return os.path.join(HIST, f"pbp_{season}.parquet")


def extra_seasons_on_disk() -> List[int]:
    out = []
    if os.path.isdir(HIST):
        for fn in os.listdir(HIST):
            if fn.startswith("pbp_") and fn.endswith(".parquet"):
                try:
                    out.append(int(fn[4:8]))
                except ValueError:
                    continue
    return sorted(s for s in out if s not in BASE_SEASONS)


# --------------------------------------------------------------------------- #
# Refresh (network; degrades loudly to cache)
# --------------------------------------------------------------------------- #
def refresh(season: Optional[int] = None, force: bool = False) -> Dict:
    """Bring the current season's pbp + schedules + rosters up to date.

    Returns {"season", "pbp_rows", "sched_rows", "stale", "errors"} --
    ``stale=True`` means a live pull failed and cached data (if any) is being
    served; the pipeline's freshness gate turns that into publish decisions.
    """
    season = season or current_season()
    errors: List[str] = []
    stale = False
    pbp_rows = sched_rows = 0

    try:
        import nflreadpy as nfl
    except ImportError:
        return {"season": season, "pbp_rows": 0, "sched_rows": 0, "stale": True,
                "errors": ["nflreadpy not installed -- run: pip install nflreadpy"]}

    # -- play-by-play: refresh current season, backfill any missing prior ---- #
    from .advanced_features import EXT_PBP_COLUMNS
    need = [s for s in range(2024, season + 1)
            if force or s == season or not os.path.exists(_season_pbp_path(s))]
    for s in need:
        try:
            pbp = nfl.load_pbp(seasons=[s]).to_pandas()
            if len(pbp):
                cols = [c for c in EXT_PBP_COLUMNS if c in pbp.columns]
                pbp[cols].to_parquet(_season_pbp_path(s), index=False)
                if s == season:
                    pbp_rows = int(len(pbp))
        except Exception as exc:  # noqa: BLE001
            msg = f"pbp {s}: {exc}"
            errors.append(msg)
            if s == season and not os.path.exists(_season_pbp_path(s)):
                stale = True
            print(f"[ingest] {msg}")

    # -- schedules 2024 -> now (kickoffs + pre-game lines for the slate) ----- #
    try:
        sched = nfl.load_schedules().to_pandas()
        extra = sched[sched["season"] >= 2024]
        keep = [c for c in SCHED_COLS if c in extra.columns]
        extra[keep].to_parquet(LINES_EXTRA, index=False)
        sched_rows = int(len(extra))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"schedules: {exc}")
        stale = stale or not os.path.exists(LINES_EXTRA)
        print(f"[ingest] schedules refresh failed: {exc}")

    # -- rosters (extends the shared cache in place) -------------------------- #
    try:
        from .sources import rosters as rostersmod
        rostersmod.fetch_rosters_weekly([season], force_refresh=force)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"rosters {season}: {exc}")
        print(f"[ingest] roster refresh failed: {exc}")

    # -- injury reports + player DOBs (context features) ---------------------- #
    try:
        from . import context_features as cf
        if force or not os.path.exists(cf.PLAYERS_META):
            cf.load_players_meta(refresh=True)
        cf.load_injury_history([season], refresh=True)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"context data {season}: {exc}")
        print(f"[ingest] context data refresh failed: {exc}")

    # -- NGS weekly tracking metrics + contracts (advanced features) ---------- #
    try:
        for st in ("receiving", "passing"):
            path = os.path.join(HIST, f"ngs_{st}.parquet")
            cached = pd.read_parquet(path) if os.path.exists(path) else pd.DataFrame()
            fresh = nfl.load_nextgen_stats(seasons=[season], stat_type=st).to_pandas()
            fresh = fresh[fresh["week"] > 0]
            if len(fresh):
                merged = pd.concat([cached[cached["season"] != season] if len(cached) else cached,
                                    fresh], ignore_index=True)
                merged.to_parquet(path, index=False)
        cpath = os.path.join(HIST, "contracts.parquet")
        if force or not os.path.exists(cpath):
            con = nfl.load_contracts().to_pandas()
            con[["gsis_id", "player", "position", "year_signed", "years",
                 "value", "apy", "is_active"]].to_parquet(cpath, index=False)
        from . import ftn_features
        if season >= ftn_features.FTN_SEASON_MIN:
            ftn_features.refresh(season)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"ngs/contracts {season}: {exc}")
        print(f"[ingest] NGS/contracts refresh failed: {exc}")

    return {"season": season, "pbp_rows": pbp_rows, "sched_rows": sched_rows,
            "stale": stale, "errors": errors}


# --------------------------------------------------------------------------- #
# Loaders (no network; compose whatever is on disk)
# --------------------------------------------------------------------------- #
def load_all_pbp() -> pd.DataFrame:
    """2019-2023 base + every per-season file on disk, REG only, one frame."""
    frames = [pd.read_parquet(BASE_PBP, columns=PBP_COLUMNS)]
    for s in extra_seasons_on_disk():
        frames.append(pd.read_parquet(_season_pbp_path(s), columns=PBP_COLUMNS))
    df = pd.concat(frames, ignore_index=True)
    return df[df["season_type"] == "REG"].reset_index(drop=True)


def load_all_schedules() -> pd.DataFrame:
    frames = [pd.read_parquet(BASE_LINES)]
    if os.path.exists(LINES_EXTRA):
        frames.append(pd.read_parquet(LINES_EXTRA))
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates(subset=["game_id"], keep="last").reset_index(drop=True)
