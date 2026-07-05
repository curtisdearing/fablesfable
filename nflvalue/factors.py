"""Situational factors that move a game away from the market price.

Each factor turns raw context (weather forecast, injury report, revenge spot,
matchup ratings) into a small signed number. The model multiplies those
numbers by learned weights and adds them to the market's fair log-odds.

The weights start small and conservative on purpose: the market is sharp, so
we only drift away from it a little, and the learning loop (learn.py) grows or
shrinks each weight based on whether it actually predicted winners.
"""

from __future__ import annotations

from typing import Dict, Optional


# --------------------------------------------------------------------------- #
# NFL stadiums: (lat, lon, is_dome). Weather is ignored for domes / fixed roofs.
#
# Phase 6.4 retractable-roof audit (ARI, ATL, DAL, HOU, IND -- all currently
# dome=True here): MEASURED over 292 retractable-stadium games 2019-2025,
# the roof was open in 12.7%, and ZERO open-roof games had pricable weather
# (all 53-90F, dry, mean wind 5.5 mph; scripts/fit_weather.py [roof]).
# Roof policy already neutralizes weather before kickoff, so the static
# dome=True treatment is empirically correct for pricing -- kept, now as a
# measured fact instead of an assumption. (LV Allegiant and LA SoFi are
# FIXED roofs; MIN/DET/NO fixed domes.)
# --------------------------------------------------------------------------- #
STADIUMS: Dict[str, Dict] = {
    "Arizona Cardinals":      {"lat": 33.5277, "lon": -112.2626, "dome": True},
    "Atlanta Falcons":        {"lat": 33.7554, "lon": -84.4008,  "dome": True},
    "Baltimore Ravens":       {"lat": 39.2780, "lon": -76.6227,  "dome": False},
    "Buffalo Bills":          {"lat": 42.7738, "lon": -78.7870,  "dome": False},
    "Carolina Panthers":      {"lat": 35.2258, "lon": -80.8528,  "dome": False},
    "Chicago Bears":          {"lat": 41.8623, "lon": -87.6167,  "dome": False},
    "Cincinnati Bengals":     {"lat": 39.0954, "lon": -84.5160,  "dome": False},
    "Cleveland Browns":       {"lat": 41.5061, "lon": -81.6995,  "dome": False},
    "Dallas Cowboys":         {"lat": 32.7473, "lon": -97.0945,  "dome": True},
    "Denver Broncos":         {"lat": 39.7439, "lon": -105.0201, "dome": False},
    "Detroit Lions":          {"lat": 42.3400, "lon": -83.0456,  "dome": True},
    "Green Bay Packers":      {"lat": 44.5013, "lon": -88.0622,  "dome": False},
    "Houston Texans":         {"lat": 29.6847, "lon": -95.4107,  "dome": True},
    "Indianapolis Colts":     {"lat": 39.7601, "lon": -86.1639,  "dome": True},
    "Jacksonville Jaguars":   {"lat": 30.3239, "lon": -81.6373,  "dome": False},
    "Kansas City Chiefs":     {"lat": 39.0489, "lon": -94.4839,  "dome": False},
    "Las Vegas Raiders":      {"lat": 36.0909, "lon": -115.1833, "dome": True},
    "Los Angeles Chargers":   {"lat": 33.9535, "lon": -118.3392, "dome": True},
    "Los Angeles Rams":       {"lat": 33.9535, "lon": -118.3392, "dome": True},
    "Miami Dolphins":         {"lat": 25.9580, "lon": -80.2389,  "dome": False},
    "Minnesota Vikings":      {"lat": 44.9737, "lon": -93.2581,  "dome": True},
    "New England Patriots":   {"lat": 42.0909, "lon": -71.2643,  "dome": False},
    "New Orleans Saints":     {"lat": 29.9511, "lon": -90.0812,  "dome": True},
    "New York Giants":        {"lat": 40.8135, "lon": -74.0745,  "dome": False},
    "New York Jets":          {"lat": 40.8135, "lon": -74.0745,  "dome": False},
    "Philadelphia Eagles":    {"lat": 39.9008, "lon": -75.1675,  "dome": False},
    "Pittsburgh Steelers":    {"lat": 40.4468, "lon": -80.0158,  "dome": False},
    "San Francisco 49ers":    {"lat": 37.4030, "lon": -121.9700, "dome": False},
    "Seattle Seahawks":       {"lat": 47.5952, "lon": -122.3316, "dome": False},
    "Tampa Bay Buccaneers":   {"lat": 27.9759, "lon": -82.5033,  "dome": False},
    "Tennessee Titans":       {"lat": 36.1665, "lon": -86.7713,  "dome": False},
    "Washington Commanders":  {"lat": 38.9078, "lon": -76.8645,  "dome": False},
}

# Positional importance weights for injury impact (QB dominates everything).
POSITION_WEIGHT = {
    "QB": 1.00, "RB": 0.30, "WR": 0.28, "TE": 0.18, "OT": 0.22, "OL": 0.18,
    "G": 0.15, "C": 0.15, "EDGE": 0.25, "DE": 0.22, "DT": 0.18, "LB": 0.18,
    "CB": 0.24, "S": 0.18, "K": 0.10, "DEF": 0.20,
}
STATUS_WEIGHT = {"out": 1.0, "doubtful": 0.75, "questionable": 0.35, "ir": 1.0}


# --------------------------------------------------------------------------- #
# Weather -- FITTED constants (Phase 6.4), not the old guessed thresholds.
#
# scripts/fit_weather.py, 2019-2023 outdoor-effective team-games (n=1,556),
# pass yards OLS controlling trailing team passing:
#   wind        -2.46 yds/mph up to 10 mph (t=-3.4), -1.43/mph above
#               (kink term itself t=+0.9 -- the old "30mph = max" shape was
#               wrong twice: the effect is linear-ish and already present at
#               single-digit wind)
#   precip flag -38.0 yds (t=-5.9)  -- the DOMINANT term; the old heuristic
#               gave precip less weight than wind
#   cold        -0.98 yds per degree below 32F, t=-1.3 -> FAILS the t>=2 bar,
#               deliberately NOT shipped
#   crosswind   nothing for passing (t=-0.1); real for FG% (-0.032/mph logit,
#               t=-2.9, n=4,042 attempts) -- kicking, not passing, cares
#               about direction
#   rushing     no term clears (wind t=+0.6, precip t=+1.9): the old
#               "bad weather boosts rushing" rush-severity bonus was a vibe
# Severity = fitted pass-yards deficit vs a TYPICAL outdoor day (8 mph, dry),
# normalized so ~20 mph + rain ~= 1.0.
# --------------------------------------------------------------------------- #
WX_PASS_WIND = -2.46        # yds per mph, 0-10 mph
WX_PASS_WIND_HI = -1.43     # yds per mph above 10 (=-2.46+1.03)
WX_PASS_PRECIP = -38.0      # yds, conditions-flag rain/snow
WX_TYPICAL_WIND = 8.0       # mph; the centering point (league outdoor mean)
WX_SEV_NORM = 60.0          # yds of deficit-vs-typical that reads as sev 1.0
WX_PRECIP_MM_FLAG = 0.5     # forecast mm/hr at/above this -> precip flag
LEAGUE_PASS_YDS = 243.4     # 2019-2023 mean team-game passing yards (fit sample)


def _pass_yds_delta(wind_mph: float, precip_flag: int) -> float:
    """Fitted expected team-pass-yards delta vs zero wind, dry."""
    w = max(float(wind_mph or 0.0), 0.0)
    d = WX_PASS_WIND * min(w, 10.0) + WX_PASS_WIND_HI * max(w - 10.0, 0.0)
    return d + (WX_PASS_PRECIP if precip_flag else 0.0)


def weather_severity(weather: Optional[Dict]) -> float:
    """0 (typical/dome) .. ~1 (brutal). Fitted, see module note.

    ``weather`` keys: wind_mph, precip_mm, temp_f (all optional; temp is
    accepted but unused -- cold failed the significance bar).
    """
    if not weather or weather.get("dome"):
        return 0.0
    wind = float(weather.get("wind_mph", 0) or 0)
    precip_flag = int(float(weather.get("precip_mm", 0) or 0) >= WX_PRECIP_MM_FLAG)
    deficit = _pass_yds_delta(WX_TYPICAL_WIND, 0) - _pass_yds_delta(wind, precip_flag)
    return round(min(max(deficit / WX_SEV_NORM, 0.0), 1.0), 4)


def weather_pass_multiplier(wind_mph: Optional[float], precip_mm: Optional[float],
                            effective_outdoor: bool) -> float:
    """Phase 6.4 prop-level multiplier for pass-family YARDS markets: the
    fitted team-passing delta vs a typical day, expressed multiplicatively.
    Centered at 1.0 on a typical outdoor day; calm days boost mildly, wind/
    rain dampen. Clipped [0.85, 1.06]. Domes/closed roofs -> exactly 1.0."""
    if not effective_outdoor or wind_mph is None:
        return 1.0
    precip_flag = int(float(precip_mm or 0.0) >= WX_PRECIP_MM_FLAG)
    delta = _pass_yds_delta(float(wind_mph), precip_flag) - _pass_yds_delta(WX_TYPICAL_WIND, 0)
    return round(min(max(1.0 + delta / LEAGUE_PASS_YDS, 0.85), 1.06), 4)


# --------------------------------------------------------------------------- #
# Injuries
# --------------------------------------------------------------------------- #
def injury_severity(injuries) -> float:
    """Sum a team's injury impact. ``injuries`` is a list of dicts with
    keys: position, status. Returns roughly 0 (healthy) .. ~2 (decimated)."""
    if not injuries:
        return 0.0
    total = 0.0
    for inj in injuries:
        pos = str(inj.get("position", "")).upper()
        status = str(inj.get("status", "")).lower()
        total += POSITION_WEIGHT.get(pos, 0.12) * STATUS_WEIGHT.get(status, 0.3)
    return round(total, 4)


# --------------------------------------------------------------------------- #
# Feature vectors
# --------------------------------------------------------------------------- #
def game_side_features(ctx: Dict, side: str) -> Dict[str, float]:
    """Features for backing ``side`` ('home' or 'away') on the moneyline/spread.

    ``ctx`` carries pre-computed numbers for the game (see model.build_*).
    Features are signed from the perspective of the backed side.
    """
    sign = 1.0 if side == "home" else -1.0
    inj_home = ctx.get("inj_home", 0.0)
    inj_away = ctx.get("inj_away", 0.0)
    # Positive when the OTHER team is more banged up than ours.
    inj_diff = sign * (inj_away - inj_home)

    revenge = ctx.get("revenge_home", 0.0) if side == "home" else ctx.get("revenge_away", 0.0)
    rest_diff = sign * (ctx.get("rest_home", 0) - ctx.get("rest_away", 0)) / 7.0
    matchup = sign * ctx.get("matchup_edge", 0.0)   # >0 favours home offense/defense net
    home_field = 1.0 if side == "home" else 0.0

    return {
        "g_injury_diff": round(inj_diff, 4),
        "g_revenge": round(revenge, 4),
        "g_rest_diff": round(rest_diff, 4),
        "g_matchup": round(matchup, 4),
        "g_home_field": home_field,
    }


def total_features(ctx: Dict, side: str) -> Dict[str, float]:
    """Features for totals. side = 'over' or 'under'. Bad weather favours under."""
    sign = 1.0 if side == "under" else -1.0
    sev = ctx.get("weather_sev", 0.0)
    inj_off = ctx.get("inj_home", 0.0) + ctx.get("inj_away", 0.0)  # injuries lower scoring
    pace = ctx.get("pace_edge", 0.0)  # >0 = faster/higher scoring expected
    return {
        "t_weather": round(sign * sev, 4),
        "t_injuries": round(sign * 0.5 * inj_off, 4),
        "t_pace": round(-sign * pace, 4),
    }


def prop_features(ctx: Dict, prop_type: str, side: str) -> Dict[str, float]:
    """Features for a player prop. side = 'over' or 'under'.

    ``ctx`` keys used: weather_sev, opp_def_rating (vs this prop type, >0 = tough
    defense), usage_trend (>0 = trending up), is_pass_prop / is_rush_prop bools.
    """
    sign = 1.0 if side == "over" else -1.0
    sev = ctx.get("weather_sev", 0.0)
    opp_def = ctx.get("opp_def_rating", 0.0)
    usage = ctx.get("usage_trend", 0.0)

    feats = {
        "p_usage_trend": round(sign * usage, 4),
        "p_opp_defense": round(-sign * opp_def, 4),
    }
    if prop_type.startswith("pass") or prop_type.startswith("recept") or prop_type.startswith("rec"):
        # Wind/rain suppress passing & receiving.
        feats["p_weather_pass"] = round(-sign * sev, 4)
    elif prop_type.startswith("rush"):
        # Bad weather slightly boosts rushing volume.
        feats["p_weather_rush"] = round(sign * 0.5 * sev, 4)
    return feats
