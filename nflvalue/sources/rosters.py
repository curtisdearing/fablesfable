"""Real player positions, free (nflreadpy `load_rosters_weekly`).

Phase 1 Sub-phase A inferred a coarse QB/RB/REC role bucket from play-by-play
participation because the free PBP has no position column. That's a real
limitation: it can't split WR from TE, and it's an approximation of "position"
built from performance data. `load_rosters_weekly` gives the real thing --
each player's actual listed position, per (season, week, team), no inference
needed. This module fetches and caches that table.

This is NOT load-bearing the way live weekly injuries/inactives are (Phase
1B Part 2) -- roster position for HISTORICAL weeks is stable, factual, and
doesn't change after the fact, so caching it to parquet (like
`historical_pbp.parquet`) is safe and avoids re-hitting the network on every
run. A small recorded fixture (real rows, not synthetic) is shipped for
offline/deterministic tests.
"""

from __future__ import annotations

import os
from typing import List, Optional

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HIST = os.path.join(ROOT, "historical")
CACHE_PATH = os.path.join(HIST, "rosters_weekly.parquet")
FIXTURE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "rosters_weekly_sample.parquet")
FIXTURE_CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "rosters_weekly_sample.csv")

# Positions we actually use downstream; everything else (OL/DL/LB/DB/K/P/LS/...)
# is dropped early since those players never show up with pass/rush/rec stats.
KEEP_POSITIONS = {"QB", "RB", "WR", "TE", "FB"}

COLUMNS = ["season", "week", "team", "position", "gsis_id", "full_name"]


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"gsis_id": "player_id"})
    df = df.dropna(subset=["player_id", "position"])
    df = df[df["position"].isin(KEEP_POSITIONS)].copy()
    # FB (fullback) behaves like a RB for our markets (carries + some targets)
    df["position"] = df["position"].replace({"FB": "RB"})
    df = df.drop_duplicates(subset=["season", "week", "player_id"], keep="last")
    return df[["season", "week", "team", "position", "player_id", "full_name"]].reset_index(drop=True)


def fetch_rosters_weekly(seasons: List[int], cache_path: str = CACHE_PATH,
                          force_refresh: bool = False) -> pd.DataFrame:
    """Return real weekly positions for ``seasons``, cached to parquet.

    Tries the on-disk cache first (and only fetches whatever seasons are
    missing from it), then falls back to a live `nflreadpy` pull. If neither
    is available (e.g. no network in this environment), raises rather than
    silently returning nothing -- per the project's fail-loud rule, a caller
    that needs real positions should know when it doesn't have them, not
    quietly get an empty table.
    """
    # A forced refresh replaces only requested seasons.  Always load the cache
    # first so refreshing 2026 cannot erase 2019-2025 roster history.
    cached = pd.read_parquet(cache_path) if os.path.exists(cache_path) else pd.DataFrame(columns=COLUMNS)

    have_seasons = set(cached["season"].unique()) if len(cached) else set()
    missing = list(seasons) if force_refresh else [s for s in seasons if s not in have_seasons]

    if missing:
        try:
            import nflreadpy as nfl
        except ImportError as exc:
            if have_seasons:
                print(f"[rosters] nflreadpy not installed; using cached seasons {sorted(have_seasons)} only "
                      f"(missing {missing})")
                return cached[cached["season"].isin(seasons)].reset_index(drop=True)
            raise RuntimeError(
                "nflreadpy is not installed and no cached roster data covers the requested "
                f"seasons {seasons}. Install nflreadpy or provide a cache at {cache_path}."
            ) from exc

        try:
            pulled = nfl.load_rosters_weekly(seasons=missing).to_pandas()
        except Exception as exc:  # noqa: BLE001
            if have_seasons:
                print(f"[rosters] live fetch failed ({exc}); using cached seasons {sorted(have_seasons)} only "
                      f"(missing {missing})")
                return cached[cached["season"].isin(seasons)].reset_index(drop=True)
            raise RuntimeError(f"Could not fetch roster data for seasons {missing} and no cache exists.") from exc

        pulled = _normalize(pulled[["season", "week", "team", "position", "gsis_id", "full_name"]])
        if len(cached):
            cached = cached[~cached["season"].isin(missing)]
            combined = pd.concat([cached, pulled], ignore_index=True)
        else:
            combined = pulled
        combined = combined.drop_duplicates(subset=["season", "week", "player_id"], keep="last")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        combined.to_parquet(cache_path, index=False)
        cached = combined

    return cached[cached["season"].isin(seasons)].reset_index(drop=True)


def load_fixture() -> pd.DataFrame:
    """A small, REAL (recorded, not synthetic) sample for offline tests."""
    if os.path.exists(FIXTURE_PATH):
        return pd.read_parquet(FIXTURE_PATH)
    if os.path.exists(FIXTURE_CSV_PATH):
        return pd.read_csv(FIXTURE_CSV_PATH, dtype={"player_id": str})
    raise FileNotFoundError("recorded roster fixture is missing (parquet and CSV)")


def build_fixture(seasons: Optional[List[int]] = None, n_per_position: int = 6) -> pd.DataFrame:
    """Record a small deterministic sample of real rows to ship as a fixture.

    Picks a few well-known players per position across a couple of weeks so
    the fixture is small, stable, and human-readable, not a giant blob.
    """
    seasons = seasons or [2023]
    df = fetch_rosters_weekly(seasons)
    picks = []
    for pos in ("QB", "RB", "WR", "TE"):
        sub = df[(df["position"] == pos) & (df["week"].isin([1, 2, 3]))]
        picks.append(sub.head(n_per_position))
    sample = pd.concat(picks, ignore_index=True)
    os.makedirs(os.path.dirname(FIXTURE_PATH), exist_ok=True)
    sample.to_parquet(FIXTURE_PATH, index=False)
    sample.to_csv(FIXTURE_CSV_PATH, index=False)
    return sample


if __name__ == "__main__":
    build_fixture()
    print(f"Wrote fixture: {FIXTURE_PATH}")
