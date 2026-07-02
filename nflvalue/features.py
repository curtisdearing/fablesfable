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

PBP_COLUMNS = [
    "season", "week", "game_id", "season_type", "posteam", "defteam", "epa",
    "pass_attempt", "rush_attempt", "complete_pass", "pass_touchdown", "rush_touchdown",
    "air_yards", "yards_after_catch", "passing_yards", "rushing_yards",
    "receiver_player_id", "receiver_player_name",
    "rusher_player_id", "rusher_player_name",
    "passer_player_id", "passer_player_name",
]


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
    g = pbp.groupby(["season", "week", "posteam"])
    out = g.agg(
        team_pass_att=("pass_attempt", "sum"),
        team_rush_att=("rush_attempt", "sum"),
    ).reset_index()
    out["team_plays"] = out["team_pass_att"] + out["team_rush_att"]
    out = out.rename(columns={"posteam": "team"})
    return out


def _passer_week(pbp: pd.DataFrame) -> pd.DataFrame:
    p = pbp[pbp["pass_attempt"] == 1].dropna(subset=["passer_player_id"]).copy()
    g = p.groupby(["season", "week", "passer_player_id"])
    out = g.agg(
        pass_attempts=("pass_attempt", "sum"),
        completions=("complete_pass", "sum"),
        pass_yards=("passing_yards", lambda s: np.nansum(s.to_numpy())),
        pass_tds=("pass_touchdown", "sum"),
        pass_epa_sum=("epa", "sum"),
        player_name=("passer_player_name", "first"),
        team=("posteam", "first"),
        defteam=("defteam", "first"),
    ).reset_index().rename(columns={"passer_player_id": "player_id"})
    return out


def _receiver_week(pbp: pd.DataFrame) -> pd.DataFrame:
    r = pbp[pbp["pass_attempt"] == 1].dropna(subset=["receiver_player_id"]).copy()
    g = r.groupby(["season", "week", "receiver_player_id"])
    out = g.agg(
        targets=("pass_attempt", "sum"),
        receptions=("complete_pass", "sum"),
        rec_yards=("passing_yards", lambda s: np.nansum(s.to_numpy())),
        air_yards_sum=("air_yards", lambda s: np.nansum(s.to_numpy())),
        yac_sum=("yards_after_catch", lambda s: np.nansum(s.to_numpy())),
        rec_tds=("pass_touchdown", "sum"),
        rec_epa_sum=("epa", "sum"),
        player_name=("receiver_player_name", "first"),
        team=("posteam", "first"),
        defteam=("defteam", "first"),
    ).reset_index().rename(columns={"receiver_player_id": "player_id"})
    return out


def _rusher_week(pbp: pd.DataFrame) -> pd.DataFrame:
    r = pbp[pbp["rush_attempt"] == 1].dropna(subset=["rusher_player_id"]).copy()
    g = r.groupby(["season", "week", "rusher_player_id"])
    out = g.agg(
        carries=("rush_attempt", "sum"),
        rush_yards=("rushing_yards", lambda s: np.nansum(s.to_numpy())),
        rush_tds=("rush_touchdown", "sum"),
        rush_epa_sum=("epa", "sum"),
        player_name=("rusher_player_name", "first"),
        team=("posteam", "first"),
        defteam=("defteam", "first"),
    ).reset_index().rename(columns={"rusher_player_id": "player_id"})
    return out


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


def _rolling_shifted(s: pd.Series, window: int = ROLL_WINDOW, how: str = "mean") -> pd.Series:
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
        return shifted.ewm(span=EWM_SPAN, min_periods=1).mean()
    if how == "count":
        return shifted.rolling(window, min_periods=1).count()
    raise ValueError(how)


def _league_role_prior_mean(df: pd.DataFrame, rate_col: str) -> pd.Series:
    """Expanding, PRIOR-weeks-only league average of a per-week rate, by role.

    Computes one number per (role, season, week) -- the across-players average
    rate up through the PREVIOUS week only -- then broadcasts it back onto
    every player-week row of that role. Used as the shrinkage target so a
    3-target rookie regresses toward his role's league mean, not a stranger's.
    """
    weekly = (df.groupby(["role", "season", "week"])[rate_col]
              .mean().reset_index().sort_values(["role", "season", "week"]))
    weekly["league_prior_mean"] = (
        weekly.groupby("role")[rate_col]
        .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    )
    # The ONLY rows still NaN here are the very first (role, season, week) in
    # the whole dataset -- by definition there is no prior data to average at
    # all. Filling those with this dataframe's overall mean would leak future
    # weeks into that first prediction, so use a fixed constant (0.0) instead:
    # not derived from the data, so it can never leak, at the cost of a
    # deliberately weak (zero-information) estimate for that one edge case.
    weekly["league_prior_mean"] = weekly["league_prior_mean"].fillna(0.0)
    return df.merge(weekly[["role", "season", "week", "league_prior_mean"]],
                     on=["role", "season", "week"], how="left")["league_prior_mean"]


def build_player_week(pbp: Optional[pd.DataFrame] = None, rosters: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if pbp is None:
        pbp = load_pbp()
    team_week = _team_week(pbp)
    pw = _combine_player_week(pbp)
    pw = pw.merge(team_week, on=["season", "week", "team"], how="left")
    pw = _assign_position(pw, rosters=rosters)
    pw = pw.sort_values(["player_id", "season", "week"]).reset_index(drop=True)

    # ---- per-week raw ratios (this week's realized rate; NOT leaked yet -- ---
    # these are just intermediate columns used to build ROLLING features below)
    pw["_target_share"] = _safe_ratio(pw["targets"], pw["team_pass_att"])
    pw["_carry_share"] = _safe_ratio(pw["carries"], pw["team_rush_att"])
    pw["_adot"] = _safe_ratio(pw["air_yards_sum"], pw["targets"])
    pw["_ypt"] = _safe_ratio(pw["rec_yards"], pw["targets"])
    pw["_catch_rate"] = _safe_ratio(pw["receptions"], pw["targets"])
    pw["_ypc"] = _safe_ratio(pw["rush_yards"], pw["carries"])
    pw["_ypa"] = _safe_ratio(pw["pass_yards"], pw["pass_attempts"])
    pw["_pass_td_rate"] = _safe_ratio(pw["pass_tds"], pw["pass_attempts"])
    pw["_rush_td_rate"] = _safe_ratio(pw["rush_tds"], pw["carries"])
    pw["_rec_td_rate"] = _safe_ratio(pw["rec_tds"], pw["targets"])

    g = pw.groupby("player_id")
    pw["roll_games"] = g["targets"].transform(lambda s: _rolling_shifted(s, how="count"))
    pw["roll_targets"] = g["targets"].transform(_rolling_shifted)
    pw["roll_target_share"] = g["_target_share"].transform(_rolling_shifted)
    pw["roll_air_yards"] = g["air_yards_sum"].transform(_rolling_shifted)
    pw["roll_adot"] = g["_adot"].transform(_rolling_shifted)
    pw["roll_carries"] = g["carries"].transform(_rolling_shifted)
    pw["roll_carry_share"] = g["_carry_share"].transform(_rolling_shifted)
    pw["roll_pass_attempts"] = g["pass_attempts"].transform(_rolling_shifted)
    pw["roll_completions"] = g["completions"].transform(_rolling_shifted)

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
        "roll_pass_td_rate": "_pass_td_rate",
        "roll_rush_td_rate": "_rush_td_rate",
        "roll_rec_td_rate": "_rec_td_rate",
    }
    for out_col, raw_col in raw_eff.items():
        pw[f"_raw_{out_col}"] = g[raw_col].transform(_rolling_shifted)

    # ---- shrink each rolling efficiency toward its role's prior league mean -- #
    for out_col, raw_col in raw_eff.items():
        league_mean = _league_role_prior_mean(pw, raw_col)
        n = pw["roll_games"].fillna(0.0)
        raw = pw[f"_raw_{out_col}"]
        pw[out_col] = np.where(
            raw.isna(),
            league_mean,
            (n * raw.fillna(0.0) + SHRINK_K * league_mean) / (n + SHRINK_K),
        )

    keep = [
        "season", "week", "player_id", "player_name", "team", "defteam", "role", "position_source",
        "targets", "receptions", "rec_yards", "air_yards_sum", "yac_sum",
        "carries", "rush_yards", "pass_attempts", "completions", "pass_yards",
        "pass_tds", "rush_tds", "rec_tds",
        "team_pass_att", "team_rush_att", "team_plays",
        "roll_games", "roll_targets", "roll_target_share", "roll_air_yards", "roll_adot",
        "roll_carries", "roll_carry_share", "roll_pass_attempts", "roll_completions",
        "roll_ypt", "roll_catch_rate", "roll_ypc", "roll_ypa",
        "roll_pass_td_rate", "roll_rush_td_rate", "roll_rec_td_rate",
    ]
    return pw[keep].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Opponent-vs-role defense table
# --------------------------------------------------------------------------- #
def build_opp_pos_def(pbp: Optional[pd.DataFrame] = None, rosters: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Rolling defense-vs-role factors. Phase 1B split: WR and TE are now
    tracked SEPARATELY (a defense can be tough on WRs but soft on TEs, or vice
    versa -- a real signal real positions unlock that the old combined REC
    bucket couldn't see), using each play's actual targeted receiver position.
    """
    if pbp is None:
        pbp = load_pbp()
    if rosters is None:
        seasons = sorted(pbp["season"].unique().tolist())
        rosters = rostersmod.fetch_rosters_weekly(seasons)

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

    keep = [
        "season", "week", "defteam", "role",
        "targets_allowed", "rec_yards_allowed", "carries_allowed", "rush_yards_allowed",
        "pass_yards_allowed", "epa_allowed_sum", "plays_faced", "roll_games",
        "roll_ypt_allowed_factor", "roll_ypc_allowed_factor", "roll_ypa_allowed_factor",
        "roll_epa_allowed_factor",
    ]
    return opp[keep].reset_index(drop=True)


def build_team_week(pbp: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Rolling, PRIOR-WEEKS-ONLY team pass/rush volume (for expected-volume math).

    This is the team-level analog of ``roll_pass_attempts``/``roll_carries`` on
    ``player_week``: how many pass/rush plays a team is expected to run THIS
    week, based on its own trailing games. ``projection.py`` multiplies this by
    a player's rolling target/carry SHARE to get expected targets/carries.
    """
    if pbp is None:
        pbp = load_pbp()
    tw = _team_week(pbp).sort_values(["team", "season", "week"]).reset_index(drop=True)
    g = tw.groupby("team")
    tw["roll_team_pass_att"] = g["team_pass_att"].transform(_rolling_shifted)
    tw["roll_team_rush_att"] = g["team_rush_att"].transform(_rolling_shifted)

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
    return tw[["season", "week", "team", "roll_team_pass_att", "roll_team_rush_att"]]


if __name__ == "__main__":
    pbp = load_pbp()
    print(f"Loaded {len(pbp):,} regular-season plays, seasons {sorted(pbp['season'].unique())}")
    pw = build_player_week(pbp)
    print(f"player_week: {len(pw):,} rows, {pw['player_id'].nunique():,} players")
    opd = build_opp_pos_def(pbp)
    print(f"opp_pos_def: {len(opd):,} rows")
