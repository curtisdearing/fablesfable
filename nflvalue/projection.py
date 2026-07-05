"""Deterministic per-player, per-market prop projection model.

Pure math, no I/O: given a player's rolling usage/efficiency (from
``features.build_player_week``), his team's rolling pass/rush volume (from
``features.build_team_week``), and the opponent's rolling defense-vs-role
factor (from ``features.build_opp_pos_def``), produce a projected
distribution for one market and read off P(over)/P(under) a line.

    expected volume     = team rolling volume x player's rolling usage share
                           x game-script multiplier (trailing teams pass more,
                           leading teams run more -- see ``game_script_multipliers``,
                           which wraps ``nflvalue.montecarlo.simulate``)
    expected efficiency = player's rolling efficiency x opponent-vs-role factor
    mean                = expected volume x expected efficiency
    sd                  = supplied by the caller (see note below) or a
                          reasonable default; the distribution is then read
                          off for p_over/p_under.

Everything here is deterministic and seeded where randomness is used (Monte
Carlo game-script only) -- same inputs always produce the same outputs, so
this can run inside a walk-forward backtest with the LLM layer completely
absent (PHASE1_HANDSOFF_DESIGN.md H6: the LLM never touches a number).

A note on SD: per-market residual dispersion is a property of HISTORICAL
ACCURACY (how far projections tend to miss by), not of a single player-week,
so it belongs to whoever is running many predictions and can measure
residuals walk-forward (``prop_backtest.py`` does this and passes ``sd``
in). A conservative default is used here only as a fallback so the function
never crashes with no dispersion info.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

try:
    from scipy import stats as _stats
except ImportError:  # pragma: no cover - scipy is in requirements, but degrade gracefully
    _stats = None

# --------------------------------------------------------------------------- #
# Market registry
# --------------------------------------------------------------------------- #
# role: which real position(s) (from nflvalue/sources/rosters.py) this market
#       applies to -- a tuple, since Phase 1B splits WR/TE (was a single
#       combined "REC" bucket in Phase 1A's role-inference version)
# opportunity: the usage-share column driving volume ("targets"/"carries"/"pass_attempts")
# efficiency: the rolling efficiency column multiplied onto volume to get the mean
# use_opp_factor: if True, look up the opponent's defense-vs-role factor using
#       the PLAYER'S OWN role (QB -> pass D, WR -> WR-specific D, TE -> TE-specific
#       D, RB -> rush D) -- one flag works for every market since the role
#       itself always determines which opp_pos_def row applies
# dist: distribution family used to read off p_over/p_under
# low_confidence: markets flagged as high-variance / weak signal (spec: TDs)
MARKETS: Dict[str, Dict] = {
    "receiving_yards": dict(role=("WR", "TE"), opportunity="targets", efficiency="roll_ypt",
                             use_opp_factor=True, dist="gamma", low_confidence=False),
    "receptions": dict(role=("WR", "TE"), opportunity="targets", efficiency="roll_catch_rate",
                        use_opp_factor=False, dist="negbinom", low_confidence=False),
    "rushing_yards": dict(role=("RB",), opportunity="carries", efficiency="roll_ypc",
                           use_opp_factor=True, dist="gamma", low_confidence=False),
    "passing_yards": dict(role=("QB",), opportunity="pass_attempts", efficiency="roll_ypa",
                           use_opp_factor=True, dist="normal", low_confidence=False),
    "pass_attempts": dict(role=("QB",), opportunity="pass_attempts", efficiency=None,
                           use_opp_factor=False, dist="negbinom", low_confidence=False),
    "rush_attempts": dict(role=("RB",), opportunity="carries", efficiency=None,
                           use_opp_factor=False, dist="negbinom", low_confidence=False),
    "anytime_td": dict(role=("RB", "WR", "TE"), opportunity=None, efficiency=None,
                        use_opp_factor=False, dist="poisson", low_confidence=True),
}

# player role -> the opp_pos_def factor column that applies to that role
_OPP_FACTOR_COL = {
    "QB": "roll_ypa_allowed_factor", "WR": "roll_ypt_allowed_factor",
    "TE": "roll_ypt_allowed_factor", "RB": "roll_ypc_allowed_factor",
}

# ---- Phase 6.1: depth/location shape tilt ---------------------------------- #
# Which markets take which tilt, and the player-profile column that maps the
# player onto the defense's shape. Receiving markets tilt on target depth AND
# field location; QB passing tilts on his own throw-depth profile only.
_TILT_SPEC = {
    "receiving_yards": {"depth": "roll_short_tgt_share", "location": "roll_mid_tgt_share"},
    "passing_yards": {"depth": "roll_short_pass_share"},
}
TILT_CLIP = 0.10  # each tilt bounded to [0.90, 1.10] -- a shape lean, not a new factor
# module-level switch so walk-forward ablations can flip tilts off without
# touching call sites; shipped default = both on (see decisions_p6.md)
TILTS_ENABLED = {"depth": True, "location": True}


def _one_tilt(profile: Optional[float], league_share: Optional[float],
              shape_a: Optional[float], shape_b: Optional[float]) -> Optional[float]:
    """Blend the defense's band-A/band-B shape by the PLAYER's band mix,
    normalized by the LEAGUE band mix, so profile==league -> exactly 1.0.
    Any missing ingredient -> None (neutral), never a guess."""
    vals = [profile, league_share, shape_a, shape_b]
    if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in vals):
        return None
    p, l = float(profile), float(league_share)
    num = p * float(shape_a) + (1.0 - p) * float(shape_b)
    den = l * float(shape_a) + (1.0 - l) * float(shape_b)
    if den <= 0:
        return None
    return max(1.0 - TILT_CLIP, min(1.0 + TILT_CLIP, num / den))


def shape_tilts(player_row: Dict, opp_row: Optional[Dict], market: str) -> Dict[str, float]:
    """{"depth": t, "location": t} for the market (missing data -> absent key)."""
    out: Dict[str, float] = {}
    spec = _TILT_SPEC.get(market)
    if not spec or opp_row is None:
        return out
    if TILTS_ENABLED.get("depth") and "depth" in spec:
        t = _one_tilt(player_row.get(spec["depth"]), opp_row.get("league_short_share"),
                      opp_row.get("roll_shape_short"), opp_row.get("roll_shape_deep"))
        if t is not None:
            out["depth"] = t
    if TILTS_ENABLED.get("location") and "location" in spec:
        t = _one_tilt(player_row.get(spec["location"]), opp_row.get("league_mid_share"),
                      opp_row.get("roll_shape_mid"), opp_row.get("roll_shape_out"))
        if t is not None:
            out["location"] = t
    return out

# Fallback SD used only when the caller doesn't supply a measured one.
DEFAULT_SD_FRACTION = 0.45  # sd ~= 45% of the mean, a generic count/yardage prior

# ---- Phase 6.2: anytime-TD red-zone path ----------------------------------- #
# TD_BLEND_W: weight on the RZ-share path vs the overall-rate path. MEASURED,
# not guessed: walk-forward log-loss grid over the frozen 2019-2023 base
# seasons, n=16,871 candidate-gated player-weeks (scripts/fit_td_blend.py):
# w=0.4 minimizes log-loss (.5398 -> .5341) and Brier (.1781 -> .1760).
# The same fit showed the opponent RZ-defense factor HURTS lambda at every w
# (factor-ON uniformly worse), so roll_rz_td_factor is deliberately NOT
# multiplied into the deterministic mean -- it rides along as an ML feature
# only (opp_rz_td_factor). Provenance: docs/decisions_p6.md.
TD_BLEND_W = 0.40
RZ_FACTOR_CLIP = (0.80, 1.25)  # bounds used when scoring the factor ablation


def _rz_lambda(player_row: Dict, team_row: Optional[Dict]) -> Optional[float]:
    """Expected TDs via the red-zone path; None when any ingredient is
    missing (cold starts, no team volume yet) -- caller falls back to the
    overall-rate path rather than inventing a number."""
    if team_row is None:
        return None
    need = {
        "tgt_share": player_row.get("roll_rz_tgt_share"),
        "car_share": player_row.get("roll_rz_carry_share"),
        "team_tgt": team_row.get("roll_team_rz_tgt"),
        "team_car": team_row.get("roll_team_rz_car"),
        "lg_tgt": team_row.get("league_rz_tgt_td_rate"),
        "lg_car": team_row.get("league_rz_car_td_rate"),
    }
    def _f(v):
        return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)
    vals = {k: _f(v) for k, v in need.items()}
    # a receiver with no RZ carry history is fine (that term is just 0);
    # what's REQUIRED is team volume + league rates, plus at least one share
    if vals["team_tgt"] is None or vals["team_car"] is None \
            or vals["lg_tgt"] is None or vals["lg_car"] is None:
        return None
    if vals["tgt_share"] is None and vals["car_share"] is None:
        return None
    lam = 0.0
    if vals["tgt_share"] is not None:
        lam += vals["team_tgt"] * vals["tgt_share"] * vals["lg_tgt"]
    if vals["car_share"] is not None:
        lam += vals["team_car"] * vals["car_share"] * vals["lg_car"]
    return lam

# Phase 1B cold-start gate (Checkpoint 1 finding: 0-2 trailing games project
# poorly, sometimes negatively correlated with the actual outcome -- see
# docs/phase1.md). Below this many trailing games, a player is marked
# ineligible for any downstream shortlist and forced low_confidence,
# regardless of the market's own low_confidence default.
MIN_GAMES_ELIGIBLE = 3


# --------------------------------------------------------------------------- #
# Volume: team pace x player share, with an optional game-script tilt
# --------------------------------------------------------------------------- #
def game_script_multipliers(projected_margin: Optional[float], sd: float = 13.0,
                             max_tilt: float = 0.12) -> Dict[str, float]:
    """Trailing teams pass more, leading teams run more.

    ``projected_margin`` is this TEAM's expected margin (positive = favored),
    e.g. from ``build_ratings`` ratings fed into ``nflvalue.montecarlo.simulate``.
    Returns ``{"pass_mult", "rush_mult"}`` centered at 1.0, tilted by how much
    the team is expected to be leading/trailing, capped at +/-``max_tilt``.
    """
    if projected_margin is None:
        return {"pass_mult": 1.0, "rush_mult": 1.0}
    # normalize margin into roughly [-1, 1] using the game's margin SD, then
    # scale into a small multiplicative tilt (a big favorite runs ~12% more).
    z = max(-1.0, min(1.0, -projected_margin / max(sd, 1e-6)))  # favored (>0 margin) -> negative z -> more run
    tilt = -z * max_tilt
    return {"pass_mult": round(1.0 - tilt, 4), "rush_mult": round(1.0 + tilt, 4)}


def expected_volume(player_row: Dict, team_row: Optional[Dict], market_spec: Dict,
                     game_script: Optional[Dict] = None) -> float:
    opp_key = market_spec["opportunity"]
    if opp_key is None:
        return float("nan")
    game_script = game_script or {"pass_mult": 1.0, "rush_mult": 1.0}

    if opp_key == "targets":
        team_pass = (team_row or {}).get("roll_team_pass_att")
        share = player_row.get("roll_target_share")
        if team_pass is None or share is None or (isinstance(share, float) and math.isnan(share)):
            return float(player_row.get("roll_targets") or 0.0)
        return float(team_pass) * float(share) * game_script["pass_mult"]

    if opp_key == "carries":
        team_rush = (team_row or {}).get("roll_team_rush_att")
        share = player_row.get("roll_carry_share")
        if team_rush is None or share is None or (isinstance(share, float) and math.isnan(share)):
            return float(player_row.get("roll_carries") or 0.0)
        return float(team_rush) * float(share) * game_script["rush_mult"]

    if opp_key == "pass_attempts":
        # a starting QB is ~all of a team's dropbacks; his own rolling rate is
        # already the right volume basis (no need to re-derive a team share).
        base = player_row.get("roll_pass_attempts")
        base = float(base) if base is not None and not (isinstance(base, float) and math.isnan(base)) else 0.0
        return base * game_script["pass_mult"]

    raise ValueError(f"unknown opportunity key {opp_key!r}")


# --------------------------------------------------------------------------- #
# Distribution helpers
# --------------------------------------------------------------------------- #
def _norm_sf(x, mean, sd):
    if _stats is not None:
        return float(_stats.norm.sf(x, loc=mean, scale=max(sd, 1e-6)))
    z = (x - mean) / max(sd, 1e-6)
    return 0.5 * math.erfc(z / math.sqrt(2))


def _gamma_sf(x, mean, sd):
    mean = max(mean, 1e-6)
    sd = max(sd, 1e-6)
    shape = (mean / sd) ** 2
    scale = (sd ** 2) / mean
    if _stats is not None:
        return float(_stats.gamma.sf(x, a=shape, scale=scale))
    # crude fallback: normal approx
    return _norm_sf(x, mean, sd)


def _negbinom_sf(x, mean, sd):
    mean = max(mean, 1e-6)
    var = max(sd ** 2, mean * 1.01)  # negbinom requires var > mean
    p = mean / var
    n = (mean ** 2) / (var - mean)
    if _stats is not None:
        # P(X > x) = 1 - CDF(floor(x)); props use half-integer lines so no push
        return float(_stats.nbinom.sf(math.floor(x), n, p))
    # Poisson fallback
    return _poisson_sf(x, mean)


def _poisson_sf(x, mean, sd=None):
    del sd  # poisson mean fixes the variance; sd is accepted only for a uniform call signature
    mean = max(mean, 1e-9)
    if _stats is not None:
        return float(_stats.poisson.sf(math.floor(x), mean))
    # manual survival via CDF sum for small means
    k = int(math.floor(x))
    cdf = sum(math.exp(-mean) * mean ** i / math.factorial(i) for i in range(0, max(k + 1, 1)))
    return max(0.0, 1.0 - cdf)


_SF = {"normal": _norm_sf, "gamma": _gamma_sf, "negbinom": _negbinom_sf, "poisson": _poisson_sf}


def p_over(mean: float, sd: float, line: float, dist: str) -> float:
    fn = _SF.get(dist, _norm_sf)
    return max(0.0, min(1.0, fn(line, mean, sd)))


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def project(player_row: Dict, market: str, team_row: Optional[Dict] = None,
            opp_row: Optional[Dict] = None, line: Optional[float] = None,
            sd: Optional[float] = None, game_script: Optional[Dict] = None,
            seed: Optional[int] = None, min_games: int = MIN_GAMES_ELIGIBLE) -> Dict:
    """Project one player's stat distribution for one market.

    ``player_row`` / ``team_row`` / ``opp_row`` are dict-like rows (or pandas
    Series) from ``player_week`` / ``build_team_week`` / ``opp_pos_def``
    respectively -- already containing only PRIOR-WEEKS-ONLY rolling values.
    ``sd`` should come from a walk-forward residual estimate (see
    ``prop_backtest.py``); if omitted, a generic fraction-of-mean default is
    used and the result should be treated as low-confidence.

    Cold-start gate (Phase 1B): if the player has fewer than ``min_games``
    trailing games (``player_row["roll_games"]``), the result is forced
    ``low_confidence=True`` and ``eligible_for_shortlist=False`` regardless
    of the market's own low_confidence default -- Checkpoint 1 showed 0-2
    game histories project poorly (sometimes negatively correlated with the
    outcome), so those rows should never surface in a ranked shortlist later
    (Phase 2), even though they're still returned here for visibility/backtesting.

    Returns the Phase-1 contract:
        {player_id, name, pos, market, mean, sd, dist, line, p_over, p_under,
         components: {volume, efficiency, opp_factor, game_script},
         low_confidence, eligible_for_shortlist, roll_games}

    Deterministic: no randomness is used for anytime_td/yards/counts math (all
    closed-form); ``seed`` is accepted for interface stability with any future
    simulation-based market and is unused today.
    """
    del seed  # reserved; current markets are all closed-form, not simulated
    if market not in MARKETS:
        raise ValueError(f"unknown market {market!r}; choices: {sorted(MARKETS)}")
    spec = MARKETS[market]

    if market == "anytime_td":
        carries = float(player_row.get("roll_carries") or 0.0)
        targets = float(player_row.get("roll_targets") or 0.0)
        rush_rate = float(player_row.get("roll_rush_td_rate") or 0.0)
        rec_rate = float(player_row.get("roll_rec_td_rate") or 0.0)
        lam_base = carries * rush_rate + targets * rec_rate

        # Phase 6.2: red-zone path -- expected team RZ opportunities x this
        # player's trailing RZ share x the league's PRIOR-weeks TD-per-RZ-
        # opportunity. Blended with the overall-rate path by TD_BLEND_W
        # (measured walk-forward on 2019-2023, see decisions_p6.md). Any
        # missing ingredient -> pure base path, never a guess. The opponent
        # RZ factor is deliberately absent here -- it FAILED the same fit
        # (see TD_BLEND_W note) and enters only as an ML feature.
        lam_rz = _rz_lambda(player_row, team_row)
        lam = ((1.0 - TD_BLEND_W) * lam_base + TD_BLEND_W * lam_rz
               if lam_rz is not None else lam_base)

        mean_, sd_, dist = lam, max(math.sqrt(max(lam, 1e-6)), 0.35), "poisson"
        components = {"volume": round(carries + targets, 3), "efficiency": round(rush_rate + rec_rate, 4),
                      "opp_factor": 1.0, "game_script": 1.0,
                      "lam_base": round(lam_base, 4),
                      "lam_rz": round(lam_rz, 4) if lam_rz is not None else None}
    else:
        volume = expected_volume(player_row, team_row, spec, game_script)
        eff_col = spec["efficiency"]
        raw_efficiency = player_row.get(eff_col) if eff_col else 1.0
        if raw_efficiency is None:
            efficiency = 0.0
        else:
            efficiency = float(raw_efficiency)
            if math.isnan(efficiency):
                efficiency = 0.0

        opp_factor = 1.0
        if spec["use_opp_factor"] and opp_row is not None:
            factor_col = _OPP_FACTOR_COL.get(player_row.get("role"))
            v = opp_row.get(factor_col) if factor_col else None
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                opp_factor = float(v)
            # Phase 6.1: multiply in the defense's depth/location SHAPE as seen
            # by THIS player's own target/throw profile (normalized so a
            # league-average profile or shapeless defense tilts nothing)
            tilts = shape_tilts(player_row, opp_row, market)
            for t in tilts.values():
                opp_factor *= t
        else:
            tilts = {}

        mean_ = volume * efficiency * opp_factor
        gs = game_script or {"pass_mult": 1.0, "rush_mult": 1.0}
        gs_component = gs.get("pass_mult") if spec["opportunity"] in ("targets", "pass_attempts") else gs.get("rush_mult")
        dist = spec["dist"]
        sd_ = sd if sd is not None else max(mean_ * DEFAULT_SD_FRACTION, 0.75)
        components = {"volume": round(float(volume), 3), "efficiency": round(float(efficiency), 4),
                      "opp_factor": round(float(opp_factor), 4), "game_script": round(float(gs_component or 1.0), 4)}
        if tilts:
            components["shape_tilts"] = {k: round(v, 4) for k, v in tilts.items()}

    mean_ = float(max(mean_, 0.0))
    sd_ = float(max(sd_, 1e-3))

    roll_games = player_row.get("roll_games")
    roll_games = 0.0 if roll_games is None or (isinstance(roll_games, float) and math.isnan(roll_games)) else float(roll_games)
    eligible = roll_games >= min_games

    out = {
        "player_id": player_row.get("player_id"),
        "name": player_row.get("player_name"),
        "pos": player_row.get("role"),
        "market": market,
        "mean": round(mean_, 3),
        "sd": round(sd_, 3),
        "dist": dist,
        "line": line,
        "p_over": None,
        "p_under": None,
        "components": components,
        "low_confidence": bool(spec["low_confidence"]) or not eligible,
        "eligible_for_shortlist": eligible,
        "roll_games": roll_games,
    }
    if line is not None:
        po = p_over(mean_, sd_, float(line), dist)
        out["p_over"] = round(po, 4)
        out["p_under"] = round(1.0 - po, 4)
    return out
