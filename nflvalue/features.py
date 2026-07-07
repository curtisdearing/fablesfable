"""Walk-forward player & opponent feature tables from play-by-play.

Builds two tables (see ``nflvalue/db.py`` for schema):

* ``player_week``   -- one row per (season, week, player) with this week's
                       ACTUALS plus rolling, PRIOR-WEEKS-ONLY usage/efficiency
                       features derived from that player's own history.
* ``opp_pos_def``   -- one row per (season, week, defteam, role) with rolling,
                       PRIOR-WEEKS-ONLY yards/EPA allowed to that role,
                       expressed as a factor relative to the league average
                       (1.0 = average defense).

LEAKAGE RULE (the #1 kill bug per PHASE1_HANDSOFF_DESIGN.md): every ``roll_*``
column is computed by sorting each group by (season, week), then calling
``.shift(1)`` BEFORE the rolling/expanding window, so the value attached to
row (season, week) only ever aggregates rows strictly earlier in that
player's/team's own sorted sequence. Season boundaries are not reset -- a
player's week-1 rolling features come from the END of the prior season,
which is intentional (real prior information, not leakage) and mirrors how
`build_ratings.py` carries ratings across season boundaries.

Position (Phase 1B update): real positions now come from `nflreadpy`'s
weekly rosters (`nflvalue/sources/rosters.py`) -- QB/RB/WR/TE per (season,
week, player), not inferred. This replaces Phase 1A's role-inference
heuristic (which could only bucket a coarse QB/RB/REC from play-by-play
participation and couldn't split WR from TE). The old heuristic is KEPT as a
fallback for the rare row where a player is missing from that week's roster
snapshot (e.g. a same-day practice-squad elevation); those rows are tagged
`position_source="inferred_fallback"` so they stay visible/flaggable rather
than silently passing as equally reliable.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

from .sources import rosters as rostersmod

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIST = os.path.join(ROOT, "historical")

ROLL_WINDOW = 8          # games of history used for the roll_games sample-size count
EWM_SPAN = 4              # games; recent-usage weighting for the rolling MEAN features (below)
SHRINK_K = 6.0            # "games" of role-mean prior weight in shrinkage
QB_ATTEMPT_THRESHOLD = 10  # cumulative pass attempts before we call someone a QB (fallback heuristic only)

# ---- Phase 6.1: depth/location + archetype constants ----------------------- #
SHORT_AIR_YARDS = 8.0      # air_yards < 8 = "short game"; >= 8 = downfield.
                           # ~60/40 league split, so both bands stay well-sampled
                           # per defense per rolling window.
WR_DEEP_ADOT = 11.0        # trailing aDOT >= 11 = downfield profile (league WR
                           # median sits just under this; fixed, documented, not
                           # refit per week -- an archetype label, not a weight).
RB_RECEIVING_MIX = 0.35    # trailing targets/(targets+carries) >= .35 = receiving back
ARCHETYPE_MIN_GAMES = 3    # below this trailing sample, archetype = generic
                           # (nothing stable to classify on; coarse role prior).

PBP_COLUMNS = [
    "season", "week", "game_id", "season_type", "posteam", "defteam", "epa",
    "pass_attempt", "rush_attempt", "complete_pass", "pass_touchdown", "rush_touchdown",
    "air_yards", "yards_after_catch", "passing_yards", "rushing_yards",
    "receiver_player_id", "receiver_player_name",
    "rusher_player_id", "rusher_player_name",
    "passer_player_id", "passer_player_name",
    # Phase 6.1: target depth/location splits + red-zone defense
    "pass_location", "yardline_100", "fixed_drive",
    # Phase 6.3: garbage-time filter + deterministic PROE/pace game script
    "down", "qtr", "score_differential", "wp", "pass_oe",
]

# ---- Phase 6.3: garbage time ------------------------------------------------ #
# Q4 with a 3-possession margin, or a Q4 win probability outside 5-95%.
# Same ingredient columns as the neutral-situation PROE/pace filter in
# advanced_features (down/qtr/score_differential/wp); this is its complement
# concept -- drop desperation/kneel-down noise from the CORE rolling stats.
#
# MEASURED AND REJECTED (Phase 6.3 ablation, shipped OFF): filtering shares/
# efficiencies cost ~1pt of composite hit rate in BOTH 2024 and 2025 replays
# and was a wash on line-free projection accuracy (2024-25 eligible-candidate
# MAE: receiving 24.77 vs 24.58 filtered-vs-not, receptions 1.670 vs 1.658,
# passing 72.76 vs 72.60, rushing 25.52 vs 25.70 -- only rushing improved).
# The machinery stays for re-testing (an additive variant -- filtered columns
# ALONGSIDE full-game ones for the GBDT -- is the natural next experiment);
# flip this switch or pass garbage_filter=True to rebuild the filtered world.
GARBAGE_Q4_MARGIN = 17
GARBAGE_WP_BAND = (0.05, 0.95)
GARBAGE_FILTER_ENABLED = False

# ---- Phase 8.3: FITTED recency weight + rest-game cleaning (shipped ON) ---- #
# scripts/fit_recency_weight.py + merge_recency_shards.py, walk-forward OOS
# next-game MAE over 2019-2025 (data/recency_weight_fit.json):
#   * EWM span 8 beats BOTH the flat-8 window production actually shipped and
#     the ewm-4 the 1B docstring intended, in ALL 7 markets, 6/6 seasons each
#     (pooled 5.379 -> 5.293; receiving alone -0.11 MAE every season).
#   * drop_rest (zero-weighting prior games the player's team entered with its
#     playoff fate settled -- game_context.meaningless_game_flags, a COARSE
#     labeled proxy) adds a further consistent sliver (pooled 5.298 -> 5.293)
#     and wins/ties per-market.
#   * drop_injury was measured WORSE (pooled 5.320) -- injury-shortened games
#     still carry role information; they are NOT dropped from the means (the
#     tag ships as a feature/ledger annotation instead, 8.4).
# Attempts markets marginally prefer span 6 (1.3295 vs 1.3324 at 8); one
# global span keeps the volume/efficiency decomposition coherent, documented
# as the accepted rounding. Flip "enabled" (or pass recency_fit=False) to
# reproduce the flat-8 world byte-for-byte.
RECENCY_FIT = {"enabled": True, "ewm_span": 8, "drop_rest": True}


def _garbage_mask(pbp: pd.DataFrame) -> pd.Series:
    """True = garbage-time play. NaN-tolerant: missing wp falls back to the
    score-margin rule alone; missing qtr/score never flags a play."""
    q4 = pbp["qtr"].fillna(0) >= 4
    blowout = pbp["score_differential"].abs().fillna(0) >= GARBAGE_Q4_MARGIN
    wp = pbp["wp"]
    wp_extreme = wp.notna() & ~wp.between(*GARBAGE_WP_BAND)
    return q4 & (blowout | wp_extreme)


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #
def load_pbp(path: Optional[str] = None) -> pd.DataFrame:
    path = path or os.path.join(HIST, "historical_pbp.parquet")
    df = pd.read_parquet(path, columns=PBP_COLUMNS)
    df = df[df["season_type"] == "REG"].copy()  # keep regular season only for consistency
    return df


# --------------------------------------------------------------------------- #
# Per-player-week actuals
# --------------------------------------------------------------------------- #
def _team_week(pbp: pd.DataFrame) -> pd.DataFrame:
    p = pbp.copy()
    rz = p["yardline_100"].notna() & (p["yardline_100"] <= 20)
    gl = p["yardline_100"].notna() & (p["yardline_100"] <= 5)
    p["_rz_tgt"] = ((p["pass_attempt"] == 1) & rz).astype(float)
    p["_rz_car"] = ((p["rush_attempt"] == 1) & rz).astype(float)
    p["_gl_car"] = ((p["rush_attempt"] == 1) & gl).astype(float)
    p["_rz_pass_td"] = ((p["pass_touchdown"] == 1) & rz).astype(float)
    p["_rz_rush_td"] = ((p["rush_touchdown"] == 1) & rz).astype(float)
    g = p.groupby(["season", "week", "posteam"])
    out = g.agg(
        team_pass_att=("pass_attempt", "sum"),
        team_rush_att=("rush_attempt", "sum"),
        team_rz_tgt=("_rz_tgt", "sum"),
        team_rz_car=("_rz_car", "sum"),
        team_gl_car=("_gl_car", "sum"),
        team_rz_pass_td=("_rz_pass_td", "sum"),
        team_rz_rush_td=("_rz_rush_td", "sum"),
    ).reset_index()
    out["team_plays"] = out["team_pass_att"] + out["team_rush_att"]
    out = out.rename(columns={"posteam": "team"})
    return out


def _with_depth_loc_flags(p: pd.DataFrame) -> pd.DataFrame:
    """Per-play depth/location indicator columns (NaN-aware: a play with no
    recorded air_yards / pass_location contributes to neither band, so a
    profile is share-of-KNOWN, never share-of-all)."""
    p = p.copy()
    ay_known = p["air_yards"].notna()
    p["_ay_known"] = ay_known.astype(float)
    p["_ay_short"] = (ay_known & (p["air_yards"] < SHORT_AIR_YARDS)).astype(float)
    loc_known = p["pass_location"].notna() if "pass_location" in p.columns else pd.Series(False, index=p.index)
    p["_loc_known"] = loc_known.astype(float)
    p["_loc_mid"] = (loc_known & (p["pass_location"] == "middle")).astype(float)
    return p


def _passer_week(pbp: pd.DataFrame) -> pd.DataFrame:
    p = pbp[pbp["pass_attempt"] == 1].dropna(subset=["passer_player_id"])
    p = _with_depth_loc_flags(p)
    g = p.groupby(["season", "week", "passer_player_id"])
    out = g.agg(
        pass_attempts=("pass_attempt", "sum"),
        completions=("complete_pass", "sum"),
        pass_yards=("passing_yards", lambda s: np.nansum(s.to_numpy())),
        pass_tds=("pass_touchdown", "sum"),
        pass_epa_sum=("epa", "sum"),
        short_att=("_ay_short", "sum"),
        known_ay_att=("_ay_known", "sum"),
        player_name=("passer_player_name", "first"),
        team=("posteam", "first"),
        defteam=("defteam", "first"),
    ).reset_index().rename(columns={"passer_player_id": "player_id"})
    return out


def _receiver_week(pbp: pd.DataFrame) -> pd.DataFrame:
    r = pbp[pbp["pass_attempt"] == 1].dropna(subset=["receiver_player_id"])
    r = _with_depth_loc_flags(r)
    r["_rz"] = (r["yardline_100"].notna() & (r["yardline_100"] <= 20)).astype(float)
    g = r.groupby(["season", "week", "receiver_player_id"])
    out = g.agg(
        targets=("pass_attempt", "sum"),
        receptions=("complete_pass", "sum"),
        rec_yards=("passing_yards", lambda s: np.nansum(s.to_numpy())),
        air_yards_sum=("air_yards", lambda s: np.nansum(s.to_numpy())),
        yac_sum=("yards_after_catch", lambda s: np.nansum(s.to_numpy())),
        rec_tds=("pass_touchdown", "sum"),
        rec_epa_sum=("epa", "sum"),
        short_tgt=("_ay_short", "sum"),
        known_ay_tgt=("_ay_known", "sum"),
        mid_tgt=("_loc_mid", "sum"),
        known_loc_tgt=("_loc_known", "sum"),
        rz_tgt=("_rz", "sum"),
        player_name=("receiver_player_name", "first"),
        team=("posteam", "first"),
        defteam=("defteam", "first"),
    ).reset_index().rename(columns={"receiver_player_id": "player_id"})
    return out


def _rusher_week(pbp: pd.DataFrame) -> pd.DataFrame:
    r = pbp[pbp["rush_attempt"] == 1].dropna(subset=["rusher_player_id"]).copy()
    r["_rz"] = (r["yardline_100"].notna() & (r["yardline_100"] <= 20)).astype(float)
    r["_gl"] = (r["yardline_100"].notna() & (r["yardline_100"] <= 5)).astype(float)
    g = r.groupby(["season", "week", "rusher_player_id"])
    out = g.agg(
        carries=("rush_attempt", "sum"),
        rush_yards=("rushing_yards", lambda s: np.nansum(s.to_numpy())),
        rush_tds=("rush_touchdown", "sum"),
        rush_epa_sum=("epa", "sum"),
        rz_car=("_rz", "sum"),
        gl_car=("_gl", "sum"),
        player_name=("rusher_player_name", "first"),
        team=("posteam", "first"),
        defteam=("defteam", "first"),
    ).reset_index().rename(columns={"rusher_player_id": "player_id"})
    return out


def _early_exit_week(pbp: pd.DataFrame) -> pd.DataFrame:
    """Phase 6.5 durability input: player had meaningful first-half usage
    (>=3 touches/targets/attempts in Q1-Q2) and ZERO in Q3-Q4 while his team
    ran second-half plays -- the pbp signature of leaving a game early.
    Returns one row per (season, week, player_id) with ``early_exit`` 0/1."""
    if "qtr" not in pbp.columns:
        return pd.DataFrame(columns=["season", "week", "player_id", "early_exit"])
    frames = []
    for id_col in ("receiver_player_id", "rusher_player_id", "passer_player_id"):
        d = pbp.dropna(subset=[id_col])
        frames.append(pd.DataFrame({
            "season": d["season"], "week": d["week"], "player_id": d[id_col],
            "posteam": d["posteam"], "h1": (d["qtr"] <= 2).astype(float),
            "h2": (d["qtr"] >= 3).astype(float)}))
    u = pd.concat(frames, ignore_index=True)
    per = (u.groupby(["season", "week", "player_id", "posteam"])[["h1", "h2"]]
           .sum().reset_index())
    team_h2 = (pbp[pbp["qtr"] >= 3].groupby(["season", "week", "posteam"])
               .size().rename("team_h2_plays").reset_index())
    per = per.merge(team_h2, on=["season", "week", "posteam"], how="left")
    per["early_exit"] = ((per["h1"] >= 3) & (per["h2"] == 0)
                         & (per["team_h2_plays"].fillna(0) >= 10)).astype(float)
    return per[["season", "week", "player_id", "early_exit"]].drop_duplicates(
        subset=["season", "week", "player_id"])


def _combine_player_week(pbp: pd.DataFrame) -> pd.DataFrame:
    """Outer-merge passer/receiver/rusher weekly stats into one row per player-week."""
    passer = _passer_week(pbp)
    receiver = _receiver_week(pbp)
    rusher = _rusher_week(pbp)

    keys = ["season", "week", "player_id"]
    merged = passer.merge(receiver, on=keys, how="outer", suffixes=("_p", "_r"))
    merged = merged.merge(rusher, on=keys, how="outer")

    # reconcile name/team/defteam columns that came from up to 3 sources: after
    # the two-stage merge, duplicate-named cols get _p/_r suffixes on the first
    # merge only; the second merge (rusher) keeps plain names if no clash.
    player_name_candidates = [c for c in ["player_name_p", "player_name_r", "player_name"] if c in merged.columns]
    team_candidates = [c for c in ["team_p", "team_r", "team"] if c in merged.columns]
    defteam_candidates = [c for c in ["defteam_p", "defteam_r", "defteam"] if c in merged.columns]

    merged["player_name"] = merged[player_name_candidates].bfill(axis=1).iloc[:, 0] if player_name_candidates else None
    merged["team"] = merged[team_candidates].bfill(axis=1).iloc[:, 0] if team_candidates else None
    merged["defteam"] = merged[defteam_candidates].bfill(axis=1).iloc[:, 0] if defteam_candidates else None

    drop_cols = [c for c in merged.columns if c in (
        "player_name_p", "player_name_r", "team_p", "team_r", "defteam_p", "defteam_r")]
    merged = merged.drop(columns=drop_cols)

    numeric_fill = [
        "pass_attempts", "completions", "pass_yards", "pass_tds", "pass_epa_sum",
        "targets", "receptions", "rec_yards", "air_yards_sum", "yac_sum", "rec_tds", "rec_epa_sum",
        "carries", "rush_yards", "rush_tds", "rush_epa_sum",
        "short_att", "known_ay_att", "short_tgt", "known_ay_tgt", "mid_tgt", "known_loc_tgt",
        "rz_tgt", "rz_car", "gl_car",
    ]
    for c in numeric_fill:
        if c in merged.columns:
            merged[c] = merged[c].fillna(0.0)
        else:
            merged[c] = 0.0

    return merged


# --------------------------------------------------------------------------- #
# Position: real roster data first, participation-based inference as fallback
# --------------------------------------------------------------------------- #
def _infer_role_fallback(df: pd.DataFrame) -> pd.Series:
    """Phase 1A's participation-based heuristic (QB/RB/REC only -- can't
    split WR from TE). Used ONLY where a real roster position is missing."""
    df = df.sort_values(["player_id", "season", "week"])
    prior_pass = df.groupby("player_id")["pass_attempts"].cumsum() - df["pass_attempts"]
    prior_carries = df.groupby("player_id")["carries"].cumsum() - df["carries"]
    prior_targets = df.groupby("player_id")["targets"].cumsum() - df["targets"]

    role = np.where(
        prior_pass >= QB_ATTEMPT_THRESHOLD, "QB",
        np.where(prior_carries >= prior_targets, "RB", "WR"),  # REC bucket defaults to WR (more common)
    )
    no_history = (prior_pass == 0) & (prior_carries == 0) & (prior_targets == 0)
    cold_role = np.where(
        df["pass_attempts"] >= QB_ATTEMPT_THRESHOLD, "QB",
        np.where(df["carries"] >= df["targets"], "RB", "WR"),
    )
    return pd.Series(np.where(no_history, cold_role, role), index=df.index)


def _assign_position(df: pd.DataFrame, rosters: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Attach real position (QB/RB/WR/TE) from weekly rosters; fall back to
    participation-based inference only for rows missing a roster match.

    Roster position for a HISTORICAL (season, week) is a factual snapshot,
    not something derived from that week's outcome, so joining it in isn't a
    leakage concern the way a rolling stat would be -- it's just accurate
    ground truth, and strictly more accurate than the old inference.
    """
    df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    if rosters is None:
        seasons = sorted(df["season"].unique().tolist())
        rosters = rostersmod.fetch_rosters_weekly(seasons)

    merged = df.merge(
        rosters[["season", "week", "player_id", "position"]],
        on=["season", "week", "player_id"], how="left",
    )
    fallback = _infer_role_fallback(df)
    merged["position_source"] = np.where(merged["position"].notna(), "roster", "inferred_fallback")
    merged["role"] = merged["position"].fillna(fallback)
    return merged.drop(columns=["position"])


# --------------------------------------------------------------------------- #
# Rolling player features (leakage-safe: shift(1) then rolling)
# --------------------------------------------------------------------------- #
def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    with np.errstate(divide="ignore", invalid="ignore"):
        r = num / den.replace(0, np.nan)
    return r


def _rolling_shifted(s: pd.Series, window: int = ROLL_WINDOW, how: str = "mean",
                     span: Optional[int] = None) -> pd.Series:
    """PRIOR-weeks-only feature from a player's/team's own history.

    ``how="mean"`` is a FLAT trailing average (window=``ROLL_WINDOW``); ``how="ewm"``
    rather than a flat rolling average. Phase 1B change: Checkpoint 1's
    calibration curve showed predicted P(over) barely tracking the actual
    over-rate, worst in the low-probability buckets -- consistent with a
    flat 8-game average LAGGING a player's real, recent usage change (e.g. a
    breakout game bumping his role) while the calibration line (his own
    rolling median) reacts faster. EWM weights the last 1-2 games far more
    than games 6-8 back, cutting that lag while still using the same
    leak-free shift(1)-before-aggregating pattern. ``how="count"`` keeps a
    flat windowed count -- it drives the cold-start sample-size gate
    (``roll_games`` / ``MIN_GAMES_ELIGIBLE``), which should stay a literal
    "how many games of history exist," not a decayed number.
    """
    shifted = s.shift(1)
    if how == "mean":
        return shifted.rolling(window, min_periods=1).mean()
    if how == "ewm":
        # NaN inputs (Phase 8.3 rest-masked games) keep their ABSOLUTE age in
        # the decay (pandas ignore_na=False) -- identical semantics to the
        # sweep's zero-weighted dropped games in scripts/fit_recency_weight.py
        return shifted.ewm(span=span or EWM_SPAN, min_periods=1).mean()
    if how == "count":
        return shifted.rolling(window, min_periods=1).count()
    raise ValueError(how)


def _league_prior_mean_by(df: pd.DataFrame, rate_col: str, group_cols: list,
                          fill: Optional[float] = 0.0) -> pd.Series:
    """Expanding, PRIOR-weeks-only league average of a per-week rate, by
    ``group_cols`` (e.g. ["role"] or ["role", "archetype"]).

    Computes one number per (*group_cols, season, week) -- the across-players
    average rate up through the PREVIOUS week only -- then broadcasts it back
    onto every matching player-week row. Used as the shrinkage target so a
    3-target rookie regresses toward his own kind's league mean, not a
    stranger's.

    ``fill``: the ONLY rows still NaN after the expanding mean are the very
    first (*group, season, week) in the dataset -- no prior data exists at
    all. Filling those with this dataframe's overall mean would leak future
    weeks into that first prediction, so the default is a fixed constant
    (0.0): not derived from the data, so it can never leak, at the cost of a
    deliberately weak (zero-information) estimate for that edge case.
    ``fill=None`` keeps them NaN so the caller can chain a coarser prior
    (Phase 6.1: archetype prior falls back to the role prior, not to 0).
    """
    keys = group_cols + ["season", "week"]
    weekly = (df.groupby(keys)[rate_col]
              .mean().reset_index().sort_values(keys))
    weekly["league_prior_mean"] = (
        weekly.groupby(group_cols)[rate_col]
        .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    )
    if fill is not None:
        weekly["league_prior_mean"] = weekly["league_prior_mean"].fillna(fill)
    return df.merge(weekly[keys + ["league_prior_mean"]],
                     on=keys, how="left")["league_prior_mean"]


def _league_role_prior_mean(df: pd.DataFrame, rate_col: str) -> pd.Series:
    """Coarse-role prior (the Phase 1 behavior; kept as the fallback tier)."""
    return _league_prior_mean_by(df, rate_col, ["role"], fill=0.0)


def _league_full_ng_ratio(pw: pd.DataFrame, full_rate: pd.Series,
                          ng_rate: pd.Series) -> pd.Series:
    """PRIOR-weeks-only league ratio of full-game rate to garbage-filtered
    rate (Phase 6.3). Multiplied onto the filtered rate so filtering can't
    shift the league level (projections are graded against full games).
    Expanding shift(1) means: never sees the current week; first-week rows
    fall back to 1.0 (no adjustment, never a leak)."""
    t = pd.DataFrame({"season": pw["season"], "week": pw["week"],
                      "f": full_rate, "n": ng_rate})
    weekly = (t.groupby(["season", "week"])[["f", "n"]].mean()
              .reset_index().sort_values(["season", "week"]))
    ef = weekly["f"].shift(1).expanding(min_periods=1).mean()
    en = weekly["n"].shift(1).expanding(min_periods=1).mean()
    weekly["_ratio"] = (ef / en.replace(0, np.nan)).clip(0.8, 1.25).fillna(1.0)
    return t.merge(weekly[["season", "week", "_ratio"]],
                   on=["season", "week"], how="left")["_ratio"].set_axis(pw.index)


def _assign_archetype(pw: pd.DataFrame) -> pd.Series:
    """Walk-forward archetype label from TRAILING usage only (Phase 6.1).

    Built exclusively from roll_* columns (shift-1-then-roll by construction),
    so the label at (season, week) can never see week W. Players under
    ARCHETYPE_MIN_GAMES trailing games are 'generic' -- there is nothing
    stable to classify on, and generic routes them to the coarse role prior.

      RB_receiving / RB_early_down   trailing targets/(targets+carries) mix
      WR_deep / WR_short             trailing aDOT split at WR_DEEP_ADOT
      generic                        QB, TE (no honest free split: inline-vs-
                                     move is alignment data, see DATA_SOURCES
                                     paywall boundary), and cold starts
    """
    tgt = pw["roll_targets"].fillna(0.0)
    car = pw["roll_carries"].fillna(0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        mix = tgt / (tgt + car).replace(0, np.nan)
    arch = np.select(
        [
            pw["role"].eq("RB") & (mix >= RB_RECEIVING_MIX),
            pw["role"].eq("RB"),
            pw["role"].eq("WR") & (pw["roll_adot"] >= WR_DEEP_ADOT),
            pw["role"].eq("WR") & pw["roll_adot"].notna(),
        ],
        ["RB_receiving", "RB_early_down", "WR_deep", "WR_short"],
        default="generic",
    )
    cold = pw["roll_games"].fillna(0.0) < ARCHETYPE_MIN_GAMES
    return pd.Series(np.where(cold, "generic", arch), index=pw.index)


def build_player_week(pbp: Optional[pd.DataFrame] = None, rosters: Optional[pd.DataFrame] = None,
                      garbage_filter: Optional[bool] = None,
                      schedules: Optional[pd.DataFrame] = None,
                      recency_fit: Optional[object] = None) -> pd.DataFrame:
    if pbp is None:
        pbp = load_pbp()
    team_week = _team_week(pbp)
    pw = _combine_player_week(pbp)
    pw = pw.merge(team_week, on=["season", "week", "team"], how="left")
    pw = _assign_position(pw, rosters=rosters)
    pw = pw.sort_values(["player_id", "season", "week"]).reset_index(drop=True)

    # ---- Phase 6.3: garbage-time filter for the RATE inputs ------------------ #
    # Blowout-inflated usage shares and garbage-time efficiency previously
    # leaked straight into mean = volume x efficiency (audit §1). Shares and
    # efficiencies are now computed from NON-GARBAGE plays; absolute volume
    # counts (targets/carries/attempts) stay full-game, because projections
    # are graded against full-game stats. Each filtered rate is recalibrated
    # by the PRIOR-WEEKS-ONLY league full/non-garbage ratio so the filter
    # sharpens cross-player signal without shifting the league level (a
    # derived walk-forward series, not a constant).
    garbage_filter = GARBAGE_FILTER_ENABLED if garbage_filter is None else garbage_filter
    can_filter = {"qtr", "score_differential"} <= set(pbp.columns)
    if garbage_filter and can_filter:
        ng_pw = _combine_player_week(pbp[~_garbage_mask(pbp)])
        ng_tw = _team_week(pbp[~_garbage_mask(pbp)])
        ng_cols = ["targets", "receptions", "rec_yards", "carries", "rush_yards",
                   "pass_attempts", "completions", "pass_yards",
                   "pass_tds", "rush_tds", "rec_tds"]
        ng = ng_pw[["season", "week", "player_id"] + ng_cols].rename(
            columns={c: f"{c}_ng" for c in ng_cols})
        pw = pw.merge(ng, on=["season", "week", "player_id"], how="left")
        ngt = ng_tw[["season", "week", "team", "team_pass_att", "team_rush_att"]].rename(
            columns={"team_pass_att": "team_pass_att_ng", "team_rush_att": "team_rush_att_ng"})
        pw = pw.merge(ngt, on=["season", "week", "team"], how="left")
        # a played week whose every snap was garbage -> 0 usage, not NaN
        # (the row exists, so this can't encode missingness)
        for c in [f"{c}_ng" for c in ng_cols] + ["team_pass_att_ng", "team_rush_att_ng"]:
            pw[c] = pw[c].fillna(0.0)
        S = "_ng"
    else:
        S = ""

    def _col(c: str) -> pd.Series:
        return pw[c + S]

    # ---- per-week raw ratios (this week's realized rate; NOT leaked yet -- ---
    # these are just intermediate columns used to build ROLLING features below)
    pw["_target_share"] = _safe_ratio(_col("targets"), _col("team_pass_att"))
    pw["_carry_share"] = _safe_ratio(_col("carries"), _col("team_rush_att"))
    pw["_adot"] = _safe_ratio(pw["air_yards_sum"], pw["targets"])
    pw["_ypt"] = _safe_ratio(_col("rec_yards"), _col("targets"))
    pw["_catch_rate"] = _safe_ratio(_col("receptions"), _col("targets"))
    pw["_ypc"] = _safe_ratio(_col("rush_yards"), _col("carries"))
    pw["_ypa"] = _safe_ratio(_col("pass_yards"), _col("pass_attempts"))
    # completion rate (completions / attempts) -- the trailing efficiency the
    # pass_completions market multiplies onto projected attempts. Shrunk toward
    # the QB league/archetype prior below exactly like _ypa/_catch_rate.
    pw["_comp_rate"] = _safe_ratio(_col("completions"), _col("pass_attempts"))
    pw["_pass_td_rate"] = _safe_ratio(_col("pass_tds"), _col("pass_attempts"))
    pw["_rush_td_rate"] = _safe_ratio(_col("rush_tds"), _col("carries"))
    pw["_rec_td_rate"] = _safe_ratio(_col("rec_tds"), _col("targets"))

    if S:  # recalibrate each filtered rate by the league full/filtered ratio
        rate_full = {
            "_target_share": ("targets", "team_pass_att"),
            "_carry_share": ("carries", "team_rush_att"),
            "_ypt": ("rec_yards", "targets"), "_catch_rate": ("receptions", "targets"),
            "_ypc": ("rush_yards", "carries"), "_ypa": ("pass_yards", "pass_attempts"),
            "_comp_rate": ("completions", "pass_attempts"),
            "_pass_td_rate": ("pass_tds", "pass_attempts"),
            "_rush_td_rate": ("rush_tds", "carries"), "_rec_td_rate": ("rec_tds", "targets"),
        }
        for rate_col, (num, den) in rate_full.items():
            full_rate = _safe_ratio(pw[num], pw[den])
            pw[rate_col] = pw[rate_col] * _league_full_ng_ratio(pw, full_rate, pw[rate_col])
    # Phase 6.1: depth/location profiles (share of KNOWN-depth/-location plays)
    pw["_short_tgt_share"] = _safe_ratio(pw["short_tgt"], pw["known_ay_tgt"])
    pw["_mid_tgt_share"] = _safe_ratio(pw["mid_tgt"], pw["known_loc_tgt"])
    pw["_short_pass_share"] = _safe_ratio(pw["short_att"], pw["known_ay_att"])
    # Phase 6.2: red-zone / goal-line usage shares (the anytime-TD drivers).
    # Every played week has a row (share NaN when the TEAM had no RZ play),
    # so missingness can't encode week-W info -- unlike the advanced_features
    # RZ table, whose rows only exist for RZ-active weeks (AsOf-consumed there).
    pw["_rz_tgt_share"] = _safe_ratio(pw["rz_tgt"], pw["team_rz_tgt"])
    pw["_rz_carry_share"] = _safe_ratio(pw["rz_car"], pw["team_rz_car"])
    pw["_gl_carry_share"] = _safe_ratio(pw["gl_car"], pw["team_gl_car"])

    # ---- Phase 8.3: FITTED recency weight + rest-game cleaning --------------- #
    # (see RECENCY_FIT provenance note above). When enabled: every rolling MEAN
    # below switches from the flat-8 window to EWM span-8, and prior games the
    # player's team entered with its playoff fate settled are ZERO-WEIGHTED in
    # those means (their raw inputs masked to NaN pre-roll; ACTUALS untouched,
    # so grading/synthetic lines never change). roll_games (the eligibility
    # sample count) deliberately stays a literal count of games played.
    rf = RECENCY_FIT if recency_fit is None else (
        dict(recency_fit) if isinstance(recency_fit, dict)
        else {**RECENCY_FIT, "enabled": bool(recency_fit)})
    rf_on = bool(rf.get("enabled"))
    rf_span = int(rf.get("ewm_span", 8))
    pw["game_meaningless"] = 0.0
    if rf_on and rf.get("drop_rest", True):
        try:
            if schedules is None:
                from . import ingest as _ingest      # lazy: avoids import cycle
                schedules = _ingest.load_all_schedules()
            from . import game_context as _gc
            mg = _gc.meaningless_game_flags(schedules)
            if len(mg):
                fmap = {(int(r.season), int(r.week), r.team): float(r.meaningless)
                        for r in mg.itertuples(index=False)}
                pw["game_meaningless"] = [
                    fmap.get((int(s), int(w), t), 0.0)
                    for s, w, t in zip(pw["season"], pw["week"], pw["team"])]
        except Exception as exc:  # noqa: BLE001 -- loud degrade: fit runs un-cleaned
            print(f"[features] rest-game flags unavailable ({exc}); "
                  "recency fit runs without drop_rest")
    _keep = pw["game_meaningless"].to_numpy() == 0.0
    _mask_rest = rf_on and rf.get("drop_rest", True) and not _keep.all()
    _ROLL_INPUTS = ["targets", "_target_share", "air_yards_sum", "_adot",
                    "carries", "_carry_share", "pass_attempts", "completions",
                    "_ypt", "_catch_rate", "_ypc", "_ypa", "_comp_rate",
                    "_pass_td_rate", "_rush_td_rate", "_rec_td_rate",
                    "_short_tgt_share", "_mid_tgt_share", "_short_pass_share",
                    "_rz_tgt_share", "_rz_carry_share", "_gl_carry_share"]
    if _mask_rest:
        for c in _ROLL_INPUTS:
            pw[f"_rf_{c}"] = pw[c].where(_keep)

    def _IN(col: str) -> str:
        """Name of the roll INPUT column (rest-masked copy when cleaning)."""
        return f"_rf_{col}" if _mask_rest else col

    _mean_roll = ((lambda s: _rolling_shifted(s, how="ewm", span=rf_span))
                  if rf_on else _rolling_shifted)

    g = pw.groupby("player_id")
    pw["roll_games"] = g["targets"].transform(lambda s: _rolling_shifted(s, how="count"))
    pw["roll_targets"] = g[_IN("targets")].transform(_mean_roll)
    pw["roll_target_share"] = g[_IN("_target_share")].transform(_mean_roll)
    pw["roll_air_yards"] = g[_IN("air_yards_sum")].transform(_mean_roll)
    pw["roll_adot"] = g[_IN("_adot")].transform(_mean_roll)
    pw["roll_carries"] = g[_IN("carries")].transform(_mean_roll)
    pw["roll_carry_share"] = g[_IN("_carry_share")].transform(_mean_roll)
    pw["roll_pass_attempts"] = g[_IN("pass_attempts")].transform(_mean_roll)
    pw["roll_completions"] = g[_IN("completions")].transform(_mean_roll)

    # Cold start (a player's very first row has no own history -> NaN above):
    # fall back to the role's PRIOR-weeks-only league average rather than
    # leaving these NaN, so a rookie's debut still gets a "replacement level"
    # volume estimate instead of an undefined one. Same leakage-safe pattern
    # as the efficiency shrinkage below (expanding, shift(1), by role).
    volume_fallbacks = {
        "roll_targets": "targets", "roll_target_share": "_target_share",
        "roll_air_yards": "air_yards_sum", "roll_adot": "_adot",
        "roll_carries": "carries", "roll_carry_share": "_carry_share",
        "roll_pass_attempts": "pass_attempts", "roll_completions": "completions",
    }
    for roll_col, raw_col in volume_fallbacks.items():
        league_mean = _league_role_prior_mean(pw, raw_col)
        pw[roll_col] = pw[roll_col].fillna(league_mean)

    raw_eff = {
        "roll_ypt": "_ypt",
        "roll_catch_rate": "_catch_rate",
        "roll_ypc": "_ypc",
        "roll_ypa": "_ypa",
        "roll_comp_rate": "_comp_rate",
        "roll_pass_td_rate": "_pass_td_rate",
        "roll_rush_td_rate": "_rush_td_rate",
        "roll_rec_td_rate": "_rec_td_rate",
        # Phase 6.1 depth/location profiles: shrunk like efficiencies so a
        # 3-game profile doesn't read as an extreme depth specialist
        "roll_short_tgt_share": "_short_tgt_share",
        "roll_mid_tgt_share": "_mid_tgt_share",
        "roll_short_pass_share": "_short_pass_share",
    }
    for out_col, raw_col in raw_eff.items():
        pw[f"_raw_{out_col}"] = g[_IN(raw_col)].transform(_mean_roll)

    # Phase 6.2: RZ/GL shares roll over a LONGER window (16) -- red-zone
    # events are ~10x sparser than targets, so an 8-game window is mostly
    # noise; matches the 16-game window advanced_features already uses.
    # Phase 8.3 note: the sweep fit the CORE window; RZ shares keep their own
    # sparse-event window (flat-16) but do take the rest-game mask.
    raw_eff_rz = {
        "roll_rz_tgt_share": "_rz_tgt_share",
        "roll_rz_carry_share": "_rz_carry_share",
        "roll_gl_carry_share": "_gl_carry_share",
    }
    for out_col, raw_col in raw_eff_rz.items():
        pw[f"_raw_{out_col}"] = g[_IN(raw_col)].transform(
            lambda s: _rolling_shifted(s, window=16))
    raw_eff = {**raw_eff, **raw_eff_rz}

    # Phase 6.5 durability: rolling share of recent games the player failed
    # to finish (first-half usage, none after halftime). 32-game window --
    # durability is a slow-moving trait, not week-to-week form.
    ee = _early_exit_week(pbp)
    if len(ee):
        pw = pw.merge(ee, on=["season", "week", "player_id"], how="left")
        pw["early_exit"] = pw["early_exit"].fillna(0.0)  # played but low H1 usage
    else:
        pw["early_exit"] = 0.0
    g = pw.groupby("player_id")  # re-group after the merge (fresh frame)
    pw["roll_early_exit_rate"] = g["early_exit"].transform(
        lambda s: _rolling_shifted(s, window=32)).fillna(0.0)  # no history = no exits observed
    # Phase 8.4: "was his LAST game injury-shortened?" -- strictly-prior by
    # construction (shift), pbp-only. Rides the pw frame + ML frame as a
    # RETRAIN-GATED feature (ml_ranker.RETRAIN_PENDING_FEATURES) and tags the
    # player-learning ledger so an availability-truncated game is never
    # attributed as model error.
    pw["prev_early_exit"] = g["early_exit"].transform(lambda s: s.shift(1)).fillna(0.0)

    # ---- archetype (Phase 6.1): assigned from trailing-only rolls, used as
    # the FIRST shrinkage tier; coarse role remains the fallback tier ---------- #
    pw["archetype"] = _assign_archetype(pw)

    # ---- shrink each rolling efficiency toward its archetype's prior league
    # mean where one exists (falling back to the coarse role prior) ------------ #
    for out_col, raw_col in raw_eff.items():
        role_mean = _league_role_prior_mean(pw, raw_col)
        arch_mean = _league_prior_mean_by(pw, raw_col, ["role", "archetype"], fill=None)
        league_mean = arch_mean.fillna(role_mean)
        n = pw["roll_games"].fillna(0.0)
        raw = pw[f"_raw_{out_col}"]
        pw[out_col] = np.where(
            raw.isna(),
            league_mean,
            (n * raw.fillna(0.0) + SHRINK_K * league_mean) / (n + SHRINK_K),
        )

    keep = [
        "season", "week", "player_id", "player_name", "team", "defteam", "role", "position_source",
        "archetype",
        "targets", "receptions", "rec_yards", "air_yards_sum", "yac_sum",
        "carries", "rush_yards", "pass_attempts", "completions", "pass_yards",
        "pass_tds", "rush_tds", "rec_tds",
        "rz_tgt", "rz_car", "gl_car",
        "team_pass_att", "team_rush_att", "team_plays",
        "team_rz_tgt", "team_rz_car", "team_gl_car",
        "roll_games", "roll_targets", "roll_target_share", "roll_air_yards", "roll_adot",
        "roll_carries", "roll_carry_share", "roll_pass_attempts", "roll_completions",
        "roll_ypt", "roll_catch_rate", "roll_ypc", "roll_ypa", "roll_comp_rate",
        "roll_pass_td_rate", "roll_rush_td_rate", "roll_rec_td_rate",
        "roll_short_tgt_share", "roll_mid_tgt_share", "roll_short_pass_share",
        "roll_rz_tgt_share", "roll_rz_carry_share", "roll_gl_carry_share",
        "roll_early_exit_rate",
        # Phase 8.3/8.4 observation-quality tags: game_meaningless is PRE-GAME
        # knowable (records strictly before the week); prev_early_exit is the
        # shift-1 of a realized flag -- both leak-safe by construction.
        # early_exit is the REALIZED-week truncation flag: exported for LABEL
        # context only (ledger attribution, calibration-frame cleaning) --
        # never a pre-game feature, exactly like the actual stat columns.
        "game_meaningless", "prev_early_exit", "early_exit",
    ]
    return pw[keep].reset_index(drop=True)


def _meaningless_keys(schedules: Optional[pd.DataFrame],
                      recency_fit: Optional[object]) -> set:
    """{(season, week, team)} rest/meaningless flags (Phase 8.3 proxy), shared
    by the player means, the opponent-defense factors and the team-pace basis
    (§8.4: one tagging lens, one switch). Empty set when the fit is off or
    schedules are unavailable (loud degrade)."""
    rf = RECENCY_FIT if recency_fit is None else (
        dict(recency_fit) if isinstance(recency_fit, dict)
        else {**RECENCY_FIT, "enabled": bool(recency_fit)})
    if not (rf.get("enabled") and rf.get("drop_rest", True)):
        return set()
    try:
        if schedules is None:
            from . import ingest as _ingest
            schedules = _ingest.load_all_schedules()
        from . import game_context as _gc
        mg = _gc.meaningless_game_flags(schedules)
        return {(int(r.season), int(r.week), r.team)
                for r in mg.itertuples(index=False) if r.meaningless > 0}
    except Exception as exc:  # noqa: BLE001
        print(f"[features] rest-game flags unavailable ({exc}); no rest cleaning")
        return set()


# --------------------------------------------------------------------------- #
# Opponent-vs-role defense table
# --------------------------------------------------------------------------- #
def build_opp_pos_def(pbp: Optional[pd.DataFrame] = None, rosters: Optional[pd.DataFrame] = None,
                      schedules: Optional[pd.DataFrame] = None,
                      recency_fit: Optional[object] = None) -> pd.DataFrame:
    """Rolling defense-vs-role factors. Phase 1B split: WR and TE are now
    tracked SEPARATELY (a defense can be tough on WRs but soft on TEs, or vice
    versa -- a real signal real positions unlock that the old combined REC
    bucket couldn't see), using each play's actual targeted receiver position.

    Phase 8.4 (same lens as the player means, one flag): defense-weeks spent
    facing an OFFENSE whose playoff fate was settled (rest/meaningless proxy)
    are zero-weighted in the rolling factors -- a defense that padded its
    numbers against three resting offenses looked better than it was. Rows
    still exist (values masked, never deleted), so missingness can't encode
    week-W information.
    """
    if pbp is None:
        pbp = load_pbp()
    if rosters is None:
        seasons = sorted(pbp["season"].unique().tolist())
        rosters = rostersmod.fetch_rosters_weekly(seasons)

    # (season, week, defteam) -> the offense faced that week; a defense-week is
    # masked when THAT offense was rest-flagged
    rest_keys = _meaningless_keys(schedules, recency_fit)
    faced: dict = {}
    if rest_keys:
        f = pbp.dropna(subset=["posteam", "defteam"]).groupby(
            ["season", "week", "defteam"])["posteam"].first()
        faced = {(int(s), int(w), d): p for (s, w, d), p in f.items()}

    def _rest_masked_def_keys() -> set:
        return {k for k, off in faced.items()
                if (k[0], k[1], off) in rest_keys}

    masked_def_weeks = _rest_masked_def_keys() if rest_keys else set()

    pass_plays = pbp[pbp["pass_attempt"] == 1].copy()
    recv_pos = rosters[["season", "week", "player_id", "position"]].rename(
        columns={"player_id": "receiver_player_id", "position": "receiver_position"})
    pass_plays = pass_plays.merge(recv_pos, on=["season", "week", "receiver_player_id"], how="left")
    # unmatched (rare -- practice-squad elevations etc.) default to WR, the
    # far more common target position, rather than being dropped.
    pass_plays["receiver_position"] = pass_plays["receiver_position"].fillna("WR")

    def _agg_pass(frame: pd.DataFrame) -> pd.DataFrame:
        return (frame.groupby(["season", "week", "defteam"])
                .agg(pass_yards_allowed=("passing_yards", lambda s: np.nansum(s.to_numpy())),
                     attempts_faced=("pass_attempt", "sum"),
                     epa_allowed_sum=("epa", "sum"))
                .reset_index())

    pass_def_all = _agg_pass(pass_plays)          # QB market: overall pass defense
    pass_def_wr = _agg_pass(pass_plays[pass_plays["receiver_position"] == "WR"])
    pass_def_te = _agg_pass(pass_plays[pass_plays["receiver_position"] == "TE"])
    rush_def = (pbp[pbp["rush_attempt"] == 1]
                .groupby(["season", "week", "defteam"])
                .agg(rush_yards_allowed=("rushing_yards", lambda s: np.nansum(s.to_numpy())),
                     carries_faced=("rush_attempt", "sum"),
                     epa_allowed_sum=("epa", "sum"))
                .reset_index())

    rows = []
    for role, src, yards_col, plays_col in (
        ("QB", pass_def_all, "pass_yards_allowed", "attempts_faced"),
        ("WR", pass_def_wr, "pass_yards_allowed", "attempts_faced"),
        ("TE", pass_def_te, "pass_yards_allowed", "attempts_faced"),
        ("RB", rush_def, "rush_yards_allowed", "carries_faced"),
    ):
        t = src.copy()
        t["role"] = role
        is_receiving = role in ("QB", "WR", "TE")
        t["targets_allowed"] = t[plays_col] if is_receiving else 0.0
        t["rec_yards_allowed"] = t[yards_col] if role in ("WR", "TE") else 0.0
        t["carries_allowed"] = t[plays_col] if role == "RB" else 0.0
        t["rush_yards_allowed"] = t[yards_col] if role == "RB" else 0.0
        t["pass_yards_allowed"] = t[yards_col] if is_receiving else 0.0
        t["plays_faced"] = t[plays_col]
        rows.append(t[["season", "week", "defteam", "role", "targets_allowed", "rec_yards_allowed",
                        "carries_allowed", "rush_yards_allowed", "pass_yards_allowed",
                        "epa_allowed_sum", "plays_faced"]])
    opp = pd.concat(rows, ignore_index=True)
    opp = opp.sort_values(["defteam", "role", "season", "week"]).reset_index(drop=True)

    opp["_ypp"] = _safe_ratio(
        opp["pass_yards_allowed"].where(opp["role"].isin(["QB", "WR", "TE"]), opp["rush_yards_allowed"]),
        opp["plays_faced"],
    )
    opp["_epa_pp"] = _safe_ratio(opp["epa_allowed_sum"], opp["plays_faced"])

    # Phase 8.4: zero-weight defense-weeks vs rest-flagged offenses in the
    # rolling factors (values masked to NaN pre-roll; ROWS always remain, so
    # missingness can't encode week-W info, and the league priors average the
    # same cleaned panel -- consistent level anchor)
    if masked_def_weeks:
        _dw_key = list(zip(opp["season"].astype(int), opp["week"].astype(int), opp["defteam"]))
        _dw_mask = pd.Series([k in masked_def_weeks for k in _dw_key], index=opp.index)
        opp["_ypp"] = opp["_ypp"].where(~_dw_mask)
        opp["_epa_pp"] = opp["_epa_pp"].where(~_dw_mask)

    g = opp.groupby(["defteam", "role"])
    opp["roll_games"] = g["plays_faced"].transform(lambda s: _rolling_shifted(s, how="count"))
    opp["_roll_ypp"] = g["_ypp"].transform(_rolling_shifted)
    opp["_roll_epa_pp"] = g["_epa_pp"].transform(_rolling_shifted)

    # league-average (prior-weeks-only) per role, to express each defense as a factor
    def _league_prior(df, col):
        weekly = (df.groupby(["role", "season", "week"])[col].mean()
                  .reset_index().sort_values(["role", "season", "week"]))
        weekly["lp"] = weekly.groupby("role")[col].transform(
            lambda s: s.shift(1).expanding(min_periods=1).mean())
        # only the very first (role, season, week) in the dataset has no prior
        # data at all; fill with a fixed constant (not this dataframe's overall
        # mean) so that one edge case can never leak future weeks -- see the
        # identical reasoning in _league_role_prior_mean above.
        weekly["lp"] = weekly["lp"].fillna(0.0)
        return df.merge(weekly[["role", "season", "week", "lp"]], on=["role", "season", "week"], how="left")["lp"]

    league_ypp = _league_prior(opp, "_ypp")
    league_epa = _league_prior(opp, "_epa_pp")

    # bounded to [0.6, 1.6] so a small early-season sample can't produce an
    # implausible multiplier (e.g. one huge play against a 1-game defense).
    ypp_factor = _safe_ratio(opp["_roll_ypp"].fillna(league_ypp), league_ypp).fillna(1.0).clip(0.6, 1.6)
    # epa factor: 1.0 = average; >1 = allows MORE epa/play than average (worse defense).
    # Additive-then-bounded (not a ratio) because league-mean epa/play sits near
    # zero, which would blow up a ratio; +/-0.15 EPA/play is a realistic spread
    # between the best and worst defenses, so the factor is capped to [0.85, 1.15].
    epa_diff = (opp["_roll_epa_pp"].fillna(league_epa) - league_epa).clip(-0.15, 0.15).fillna(0.0)
    epa_factor = 1.0 + epa_diff

    opp["roll_ypt_allowed_factor"] = np.where(opp["role"].isin(["QB", "WR", "TE"]), ypp_factor, np.nan)
    opp["roll_ypa_allowed_factor"] = np.where(opp["role"] == "QB", ypp_factor, np.nan)
    opp["roll_ypc_allowed_factor"] = np.where(opp["role"] == "RB", ypp_factor, np.nan)
    opp["roll_epa_allowed_factor"] = epa_factor

    # ---- Phase 6.1: defense SHAPE vs target depth/location + red-zone TD rate
    # (one value per defteam-week, merged onto every role row of that week).
    # Both are computed on the FULL defense-week grid -- a week where a
    # defense happened to face no red-zone trip still gets a row (carrying
    # its trailing value), so row-missingness can never encode current-week
    # information (the AsOfLookup lesson from the advanced-features build). -- #
    grid = opp[["season", "week", "defteam"]].drop_duplicates().reset_index(drop=True)
    shape = _build_def_shape(pbp, grid, masked_keys=masked_def_weeks)
    rz = _build_rz_def(pbp, grid, masked_keys=masked_def_weeks)
    opp = opp.merge(shape, on=["season", "week", "defteam"], how="left")
    opp = opp.merge(rz, on=["season", "week", "defteam"], how="left")

    keep = [
        "season", "week", "defteam", "role",
        "targets_allowed", "rec_yards_allowed", "carries_allowed", "rush_yards_allowed",
        "pass_yards_allowed", "epa_allowed_sum", "plays_faced", "roll_games",
        "roll_ypt_allowed_factor", "roll_ypc_allowed_factor", "roll_ypa_allowed_factor",
        "roll_epa_allowed_factor",
        "roll_shape_short", "roll_shape_deep", "roll_shape_mid", "roll_shape_out",
        "league_short_share", "league_mid_share",
        "roll_rz_td_factor",
    ]
    return opp[keep].reset_index(drop=True)


def _def_roll_factor(d: pd.DataFrame, num: str, den: str, out: str,
                     clip: tuple = (0.6, 1.6),
                     masked_keys: Optional[set] = None) -> pd.DataFrame:
    """Shared walk-forward defense-factor idiom: per-week rate -> shift(1)
    rolling mean per defteam -> ratio to the prior-weeks-only league mean,
    clipped. Returns d with column ``out`` added. ``masked_keys`` (Phase 8.4):
    {(season, week, defteam)} whose rate is zero-weighted (rest-flagged
    opposing offense) -- masked, never deleted."""
    d = d.sort_values(["defteam", "season", "week"]).reset_index(drop=True)
    d["_rate"] = _safe_ratio(d[num], d[den])
    if masked_keys:
        keys = list(zip(d["season"].astype(int), d["week"].astype(int), d["defteam"]))
        d["_rate"] = d["_rate"].where(pd.Series([k not in masked_keys for k in keys],
                                                index=d.index))
    d["_roll"] = d.groupby("defteam")["_rate"].transform(_rolling_shifted)
    weekly = (d.groupby(["season", "week"])["_rate"].mean()
              .reset_index().sort_values(["season", "week"]))
    weekly["_lp"] = weekly["_rate"].shift(1).expanding(min_periods=1).mean()
    weekly["_lp"] = weekly["_lp"].fillna(0.0)  # first week in dataset only; see _league_prior_mean_by
    d = d.merge(weekly[["season", "week", "_lp"]], on=["season", "week"], how="left")
    d[out] = _safe_ratio(d["_roll"].fillna(d["_lp"]), d["_lp"]).fillna(1.0).clip(*clip)
    return d.drop(columns=["_rate", "_roll", "_lp"])


def _build_def_shape(pbp: pd.DataFrame, grid: pd.DataFrame,
                     masked_keys: Optional[set] = None) -> pd.DataFrame:
    """Per-defense depth/location SHAPE factors (Phase 6.1).

    Free-data feasibility verdict (constraint: no fake matchup data): man/zone
    and slot/perimeter alignment are NOT honestly buildable free + live --
    they lived in NGS participation data (dead after 2023; already rejected
    for live features in the chemistry build) and the free FTN subset carries
    no coverage/alignment columns (verified at bootstrap, see decisions_p6).
    The live-safe substitute is target DEPTH (air_yards) x field LOCATION
    (pass_location), available every season from standard pbp.

    Shape = the defense's yards-per-target factor on that band vs the league,
    same clip/priors as the coarse factors. Consumers normalize by the
    league band mix, so an average-everywhere defense tilts nothing.
    """
    p = pbp[(pbp["pass_attempt"] == 1)].dropna(subset=["receiver_player_id"])
    p = _with_depth_loc_flags(p)
    yds = p["passing_yards"].fillna(0.0)
    p = p.assign(
        _y_short=yds * p["_ay_short"], _a_short=p["_ay_short"],
        _y_deep=yds * (p["_ay_known"] - p["_ay_short"]), _a_deep=p["_ay_known"] - p["_ay_short"],
        _y_mid=yds * p["_loc_mid"], _a_mid=p["_loc_mid"],
        _y_out=yds * (p["_loc_known"] - p["_loc_mid"]), _a_out=p["_loc_known"] - p["_loc_mid"],
    )
    d = (p.groupby(["season", "week", "defteam"])
         [["_y_short", "_a_short", "_y_deep", "_a_deep", "_y_mid", "_a_mid", "_y_out", "_a_out"]]
         .sum().reset_index())
    d = grid.merge(d, on=["season", "week", "defteam"], how="left")
    d[[c for c in d.columns if c.startswith("_")]] = d[[c for c in d.columns if c.startswith("_")]].fillna(0.0)
    for num, den, out in (("_y_short", "_a_short", "roll_shape_short"),
                          ("_y_deep", "_a_deep", "roll_shape_deep"),
                          ("_y_mid", "_a_mid", "roll_shape_mid"),
                          ("_y_out", "_a_out", "roll_shape_out")):
        d = _def_roll_factor(d, num, den, out, masked_keys=masked_keys)

    # league band mix (prior-weeks-only): what share of known-depth targets are
    # short / known-location targets are middle -- the tilt's neutral point
    mix = (d.groupby(["season", "week"])[["_a_short", "_a_deep", "_a_mid", "_a_out"]]
           .sum().reset_index().sort_values(["season", "week"]))
    mix["_short_share"] = mix["_a_short"] / (mix["_a_short"] + mix["_a_deep"]).replace(0, np.nan)
    mix["_mid_share"] = mix["_a_mid"] / (mix["_a_mid"] + mix["_a_out"]).replace(0, np.nan)
    mix["league_short_share"] = mix["_short_share"].shift(1).expanding(min_periods=1).mean()
    mix["league_mid_share"] = mix["_mid_share"].shift(1).expanding(min_periods=1).mean()
    d = d.merge(mix[["season", "week", "league_short_share", "league_mid_share"]],
                on=["season", "week"], how="left")
    return d[["season", "week", "defteam",
              "roll_shape_short", "roll_shape_deep", "roll_shape_mid", "roll_shape_out",
              "league_short_share", "league_mid_share"]]


def _build_rz_def(pbp: pd.DataFrame, grid: pd.DataFrame,
                  masked_keys: Optional[set] = None) -> pd.DataFrame:
    """Red-zone defense (Phase 6.1): TDs allowed per red-zone TRIP, as a
    walk-forward factor vs league (>1 = bleeds TDs once opponents reach the
    20). A trip = a distinct (game, drive) with at least one snap at
    yardline_100 <= 20; TDs counted are offensive skill TDs (pass/rush), the
    thing anytime_td prices."""
    rz = pbp[((pbp["pass_attempt"] == 1) | (pbp["rush_attempt"] == 1))
             & (pbp["yardline_100"] <= 20)].copy()
    rz["_td"] = ((rz["pass_touchdown"] == 1) | (rz["rush_touchdown"] == 1)).astype(float)
    trips = (rz.groupby(["season", "week", "defteam", "game_id", "fixed_drive"])["_td"]
             .max().reset_index())
    d = (trips.groupby(["season", "week", "defteam"])
         .agg(rz_tds_allowed=("_td", "sum"), rz_trips_faced=("_td", "size"))
         .reset_index())
    d = grid.merge(d, on=["season", "week", "defteam"], how="left")
    d[["rz_tds_allowed", "rz_trips_faced"]] = d[["rz_tds_allowed", "rz_trips_faced"]].fillna(0.0)
    d = _def_roll_factor(d, "rz_tds_allowed", "rz_trips_faced", "roll_rz_td_factor",
                         masked_keys=masked_keys)
    return d[["season", "week", "defteam", "roll_rz_td_factor"]]


def build_team_week(pbp: Optional[pd.DataFrame] = None,
                    schedules: Optional[pd.DataFrame] = None,
                    recency_fit: Optional[object] = None) -> pd.DataFrame:
    """Rolling, PRIOR-WEEKS-ONLY team pass/rush volume (for expected-volume math).

    This is the team-level analog of ``roll_pass_attempts``/``roll_carries`` on
    ``player_week``: how many pass/rush plays a team is expected to run THIS
    week, based on its own trailing games. ``projection.py`` multiplies this by
    a player's rolling target/carry SHARE to get expected targets/carries.

    Phase 8.4 (same lens/flag as the player means): a team RESTING starters
    tanks its own volume/pace/PROE basis, so its rest-flagged weeks are
    zero-weighted in the team rolls (inputs masked pre-roll; actuals and
    league fallbacks keep every week's ROW).
    """
    if pbp is None:
        pbp = load_pbp()
    tw = _team_week(pbp).sort_values(["team", "season", "week"]).reset_index(drop=True)
    tw["_plays"] = tw["team_pass_att"] + tw["team_rush_att"]
    # Phase 6.3: trailing neutral-situation PROE (same neutral filter as
    # advanced_features: 1st/2nd down, Q1-Q3, score within 7, wp 20-80%)
    if "pass_oe" in pbp.columns and "wp" in pbp.columns:
        neutral = (pbp["down"].isin([1, 2]) & (pbp["qtr"] <= 3)
                   & (pbp["score_differential"].abs() <= 7)
                   & pbp["wp"].between(0.20, 0.80) & pbp["pass_oe"].notna())
        proe = (pbp[neutral].groupby(["season", "week", "posteam"])["pass_oe"]
                .mean().rename("_proe").reset_index().rename(columns={"posteam": "team"}))
        tw = tw.merge(proe, on=["season", "week", "team"], how="left")
        tw = tw.sort_values(["team", "season", "week"]).reset_index(drop=True)
    else:
        tw["_proe"] = np.nan

    rest_keys = _meaningless_keys(schedules, recency_fit)
    if rest_keys:
        _tk = list(zip(tw["season"].astype(int), tw["week"].astype(int), tw["team"]))
        _t_keep = pd.Series([k not in rest_keys for k in _tk], index=tw.index)
        for c in ("team_pass_att", "team_rush_att", "_plays", "_proe",
                  "team_rz_tgt", "team_rz_car"):
            tw[f"_rf_{c}"] = tw[c].where(_t_keep)

    def _TIN(col: str) -> str:
        return f"_rf_{col}" if rest_keys else col

    # every team-grouped transform runs BEFORE any season/week merges so the
    # groupby view can never go stale against a reordered frame
    g = tw.groupby("team")
    tw["roll_team_pass_att"] = g[_TIN("team_pass_att")].transform(_rolling_shifted)
    tw["roll_team_rush_att"] = g[_TIN("team_rush_att")].transform(_rolling_shifted)
    tw["roll_team_plays"] = g[_TIN("_plays")].transform(_rolling_shifted)          # 6.3 pace basis
    tw["roll_team_neutral_proe"] = g[_TIN("_proe")].transform(_rolling_shifted)   # 6.3 intent
    # Phase 6.2: expected red-zone opportunity volume (anytime-TD's RZ path)
    tw["roll_team_rz_tgt"] = g[_TIN("team_rz_tgt")].transform(
        lambda s: _rolling_shifted(s, window=16))
    tw["roll_team_rz_car"] = g[_TIN("team_rz_car")].transform(
        lambda s: _rolling_shifted(s, window=16))

    # Phase 6.3: prior-weeks-only league mean plays (the neutral point the
    # opponent-pace multiplier tilts around)
    lgp = (tw.groupby(["season", "week"])["_plays"].mean()
           .reset_index().sort_values(["season", "week"]))
    lgp["league_plays_prior"] = lgp["_plays"].shift(1).expanding(min_periods=1).mean()
    tw = tw.merge(lgp[["season", "week", "league_plays_prior"]], on=["season", "week"], how="left")

    # league TD-per-RZ-opportunity, PRIOR-weeks-only expanding (a derived
    # walk-forward series, not a hand-picked constant): what fraction of
    # red-zone targets / carries league-wide became TDs, up through last week
    lg = (tw.groupby(["season", "week"])
          [["team_rz_tgt", "team_rz_car", "team_rz_pass_td", "team_rz_rush_td"]]
          .sum().reset_index().sort_values(["season", "week"]))
    csum = lg[["team_rz_tgt", "team_rz_car", "team_rz_pass_td", "team_rz_rush_td"]].shift(1).cumsum()
    lg["league_rz_tgt_td_rate"] = (csum["team_rz_pass_td"] / csum["team_rz_tgt"].replace(0, np.nan))
    lg["league_rz_car_td_rate"] = (csum["team_rz_rush_td"] / csum["team_rz_car"].replace(0, np.nan))
    tw = tw.merge(lg[["season", "week", "league_rz_tgt_td_rate", "league_rz_car_td_rate"]],
                  on=["season", "week"], how="left")

    # Cold start (a team's first game in the dataset): fall back to the
    # PRIOR-weeks-only cross-team league average for that same (season, week)
    # cutoff -- NOT this table's overall mean, which would leak every future
    # week into an early prediction. Only the very first (season, week) in
    # the whole dataset has no prior week at all; that last edge case uses a
    # fixed constant (0.0), never data pulled from the table itself.
    weekly_league = (tw.groupby(["season", "week"])[["team_pass_att", "team_rush_att"]]
                     .mean().reset_index().sort_values(["season", "week"]))
    weekly_league["lp_pass"] = weekly_league["team_pass_att"].shift(1).expanding(min_periods=1).mean()
    weekly_league["lp_rush"] = weekly_league["team_rush_att"].shift(1).expanding(min_periods=1).mean()
    weekly_league[["lp_pass", "lp_rush"]] = weekly_league[["lp_pass", "lp_rush"]].fillna(0.0)
    tw = tw.merge(weekly_league[["season", "week", "lp_pass", "lp_rush"]], on=["season", "week"], how="left")
    tw["roll_team_pass_att"] = tw["roll_team_pass_att"].fillna(tw["lp_pass"])
    tw["roll_team_rush_att"] = tw["roll_team_rush_att"].fillna(tw["lp_rush"])
    tw = tw.drop(columns=["lp_pass", "lp_rush"])
    return tw[["season", "week", "team", "roll_team_pass_att", "roll_team_rush_att",
               "roll_team_rz_tgt", "roll_team_rz_car",
               "league_rz_tgt_td_rate", "league_rz_car_td_rate",
               "roll_team_plays", "league_plays_prior", "roll_team_neutral_proe"]]


if __name__ == "__main__":
    pbp = load_pbp()
    print(f"Loaded {len(pbp):,} regular-season plays, seasons {sorted(pbp['season'].unique())}")
    pw = build_player_week(pbp)
    print(f"player_week: {len(pw):,} rows, {pw['player_id'].nunique():,} players")
    opd = build_opp_pos_def(pbp)
    print(f"opp_pos_def: {len(opd):,} rows")
