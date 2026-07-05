"""Phase 6.4: weather ground truth + the fits that replace guessed constants.

Ground truth per game comes from the pbp ``weather`` string (94-98% of games,
all seasons): conditions text, temperature, humidity, wind DIRECTION and
speed -- e.g. "Cloudy Temp: 64° F, Humidity: 82%, Wind: S 7 mph". Schedule
temp/wind fill the gaps. Precipitation is a conditions-text flag (rain/snow
families); Open-Meteo's archive exists as a validation source but 1,700
per-kickoff-hour calls buy little over the recorded conditions text
(validated on a subsample; see decisions_p6.md).

Everything here is POST-GAME observational data used ONLY to fit constants
offline (scripts/fit_weather.py prints them); live games consume forecasts
(sources/weather.py). Nothing in this module runs in the live path.
"""

from __future__ import annotations

import os
import re
from typing import Dict, Optional

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIST = os.path.join(ROOT, "historical")

# Field long-axis bearings, degrees true (direction you face driving from one
# end zone toward the other; 0 = north, 90 = east). Entered from satellite
# imagery -- STATIC FACTS, approximate to ~±15°, which the cosine decomposition
# tolerates (cos error < 4% at 15°). Domes/fixed roofs omitted on purpose.
STADIUM_ORIENTATION_DEG: Dict[str, float] = {
    "BAL": 0, "BUF": 20, "CAR": 160, "CHI": 0, "CIN": 100, "CLE": 55,
    "DEN": 0, "GB": 0, "JAX": 0, "KC": 15, "LAC": 90, "LA": 90,  # SoFi fixed roof; kept for completeness
    "MIA": 340, "NE": 340, "NYG": 15, "NYJ": 15, "PHI": 10, "PIT": 120,
    "SF": 0, "SEA": 0, "TB": 0, "TEN": 0, "WAS": 80,
    # retractables (orientation matters on open-roof games)
    "ARI": 90, "ATL": 0, "DAL": 20, "HOU": 90, "IND": 0,
}

COMPASS = {"N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5, "E": 90, "ESE": 112.5,
           "SE": 135, "SSE": 157.5, "S": 180, "SSW": 202.5, "SW": 225,
           "WSW": 247.5, "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5}

_TEMP_RE = re.compile(r"Temp:\s*(-?\d+)\s*°")
_WIND_RE = re.compile(r"Wind:\s*([A-Za-z]{0,3})\s*\.?\s*(\d+)\s*mph", re.I)
_PRECIP_WORDS = ("rain", "shower", "drizzle", "snow", "flurr", "sleet",
                 "thunder", "storm", "wintry")


def parse_weather_string(s: Optional[str]) -> Dict:
    """{'temp_f','wind_mph','wind_dir_deg','precip_flag'} (NaN/None-safe)."""
    out = {"temp_f": np.nan, "wind_mph": np.nan, "wind_dir_deg": np.nan,
           "precip_flag": 0}
    if not s or not isinstance(s, str):
        return out
    m = _TEMP_RE.search(s)
    if m:
        out["temp_f"] = float(m.group(1))
    m = _WIND_RE.search(s)
    if m:
        out["wind_mph"] = float(m.group(2))
        d = m.group(1).upper().strip()
        if d in COMPASS:
            out["wind_dir_deg"] = COMPASS[d]
    head = s.split("Temp:")[0].lower()
    out["precip_flag"] = int(any(w in head for w in _PRECIP_WORDS))
    return out


def crosswind_headwind(wind_mph: float, wind_dir_deg: float,
                       field_axis_deg: float) -> Dict[str, float]:
    """Split wind into the along-field component (helps/hurts kicks & deep
    balls at one end, symmetric across a game) and the crosswind component.
    Both reported as magnitudes -- teams switch ends each quarter, so the
    signed direction washes out at game level."""
    theta = np.deg2rad(wind_dir_deg - field_axis_deg)
    return {"along_mph": abs(wind_mph * np.cos(theta)),
            "cross_mph": abs(wind_mph * np.sin(theta))}


def build_game_weather(pbp_frames: Optional[list] = None,
                       schedules: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """One row per game: parsed weather + roof + team + orientation.

    ``effective_outdoor`` = roof in (outdoors, open): weather physically
    touched the game. Retractable CLOSED games carry their outdoor forecast
    columns as NaN-neutralized (the roof did the neutralizing)."""
    if pbp_frames is None:
        cols = ["game_id", "season", "week", "home_team", "weather", "roof",
                "stadium", "season_type"]
        pbp_frames = [pd.read_parquet(os.path.join(HIST, "historical_pbp.parquet"), columns=cols)]
        for fn in sorted(os.listdir(HIST)):
            if fn.startswith("pbp_") and fn.endswith(".parquet"):
                pbp_frames.append(pd.read_parquet(os.path.join(HIST, fn), columns=cols))
    g = pd.concat(pbp_frames, ignore_index=True)
    g = g[g["season_type"] == "REG"].drop_duplicates("game_id").copy()

    parsed = g["weather"].apply(parse_weather_string).apply(pd.Series)
    g = pd.concat([g.reset_index(drop=True), parsed.reset_index(drop=True)], axis=1)

    if schedules is not None:
        sch = schedules.drop_duplicates("game_id")[["game_id", "temp", "wind"]]
        g = g.merge(sch, on="game_id", how="left")
        g["temp_f"] = g["temp_f"].fillna(g["temp"])
        g["wind_mph"] = g["wind_mph"].fillna(g["wind"])

    g["roof"] = g["roof"].fillna("outdoors").str.lower()
    g["effective_outdoor"] = g["roof"].isin(["outdoors", "open"])
    g["axis_deg"] = g["home_team"].map(STADIUM_ORIENTATION_DEG)
    ok = g["effective_outdoor"] & g["wind_mph"].notna() & g["wind_dir_deg"].notna() & g["axis_deg"].notna()
    comp = g.loc[ok].apply(lambda r: crosswind_headwind(
        r["wind_mph"], r["wind_dir_deg"], r["axis_deg"]), axis=1).apply(pd.Series)
    g.loc[ok, "along_mph"] = comp["along_mph"]
    g.loc[ok, "cross_mph"] = comp["cross_mph"]
    g["is_denver"] = (g["home_team"] == "DEN").astype(int)
    keep = ["game_id", "season", "week", "home_team", "roof", "effective_outdoor",
            "temp_f", "wind_mph", "wind_dir_deg", "axis_deg", "along_mph",
            "cross_mph", "precip_flag", "is_denver"]
    return g[keep]
